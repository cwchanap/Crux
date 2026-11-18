"""Host-driver tests for the scratch TF 2.15 OaF worker (HPA-856 Tasks 3-4).

The driver runs on the Crux Python 3.12 environment and only launches the
Python 3.11 scratch worker. These tests use a fake process factory: they check
the launched command/environment, restore accounting, the two-same-worker plus
one-fresh-worker inference boundary and the derived result label. They never
run the frozen tensor inventory, a real checkpoint or a real inference.

Expected native events are derived from the existing smoke-oracle fixture at
test time; no oracle golden is copied into this module.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import pytest

from runtime.oaf_tf1.model import OafModelConfigError, load_model_config
from src.benchmark.backends.oaf import _decode_native_event
from src.benchmark.checkpoint_acquisition import CheckpointAcquisitionError
from src.benchmark.oaf_smoke_oracle import (
    SmokeOracleEvent,
    _event_from_native,
    read_smoke_oracle,
)
from src.benchmark.worker_process import WorkerProcessError

REPO_ROOT = Path(__file__).parents[2]
COMPARE_PATH = REPO_ROOT / "experiments" / "oaf_tf215" / "compare.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "oaf_tf1_smoke" / "canonical.wav"
ORACLE = REPO_ROOT / "tests" / "fixtures" / "oaf_tf1_smoke" / "smoke-oracle.json"
WORKER_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"
READY = {
    "type": "ready",
    "backend_id": WORKER_BACKEND_ID,
    "restored_tensor_count": 78,
}
SANITIZED_VARIABLES = ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE")
WORKER_FLAGS = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "PYTHONHASHSEED": "0",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "NUMBA_CACHE_DIR": "/tmp/numba-cache",
}


def _load_compare() -> ModuleType:
    spec = importlib.util.spec_from_file_location("crux_oaf_tf215_compare", COMPARE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {COMPARE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="compare")
def compare_fixture() -> ModuleType:
    return _load_compare()


def _binary64(value: str) -> float:
    return struct.unpack(">d", bytes.fromhex(value))[0]


def _worker_event(event: SmokeOracleEvent) -> dict[str, object]:
    """Render one oracle event exactly as the worker protocol reports it."""
    return {
        "time_sec": _binary64(event.time_sec_binary64),
        "native_class_id": event.native_class_id,
        "model_output_bin": event.model_output_bin,
        "native_midi_note": event.native_midi_note,
        "upstream_8hit_group_id": event.upstream_8hit_group_id,
        "confidence": (
            None if event.confidence_binary64 is None else _binary64(event.confidence_binary64)
        ),
        "velocity_midi": event.velocity_midi,
    }


ORACLE_EVENTS = tuple(read_smoke_oracle(ORACLE.read_bytes()).native_events)
EXACT_EVENTS = [_worker_event(event) for event in ORACLE_EVENTS]


def _response(events: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {"id": "fake-request", "events": EXACT_EVENTS if events is None else events}


def _converted(events: list[dict[str, object]]) -> list[SmokeOracleEvent]:
    """Convert worker payloads to oracle events through the real decoder path."""
    return [_event_from_native(_decode_native_event(event)) for event in events]


def _divergent_events() -> list[dict[str, object]]:
    events = [dict(event) for event in EXACT_EVENTS]
    events[0]["native_midi_note"] = 39
    return events


def _synthetic_events(count: int) -> list[dict[str, object]]:
    return [
        {
            "time_sec": (index + 1) / 10,
            "native_class_id": "midi_38",
            "model_output_bin": 17,
            "native_midi_note": 38,
            "upstream_8hit_group_id": "snare",
            "confidence": 0.5,
            "velocity_midi": 100,
        }
        for index in range(count)
    ]


class _FakeWorker:
    """One fake worker process with a shared boundary log."""

    def __init__(
        self,
        responses: list[object],
        *,
        ready: object = READY,
        log: list[str] | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.ready = ready
        self.requests: list[str] = []
        self.close_count = 0
        self.closed = False
        self.log = [] if log is None else log
        self.close_error = close_error
        self._responses = list(responses)

    def request(self, path: str, **_kwargs: object) -> dict[str, object]:
        if self.closed:
            raise AssertionError("request after close")
        self.requests.append(path)
        self.log.append("request")
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result  # type: ignore[return-value]

    def close(self) -> None:
        self.close_count += 1
        self.closed = True
        self.log.append("close")
        if self.close_error is not None:
            raise self.close_error


def _host_error() -> WorkerProcessError:
    """Build the error a real Popen failure produces: a WorkerProcessError from OSError."""
    try:
        raise WorkerProcessError("worker could not be started") from FileNotFoundError("python3.11")
    except WorkerProcessError as error:
        return error


class _FakeFactory:
    """Process factory: one worker per call, with shared ordering evidence."""

    def __init__(
        self,
        plans: list[object] | None = None,
        *,
        ready: object = READY,
        error: Exception | None = None,
    ) -> None:
        self._plans = list(plans or [])
        self._ready = ready
        self._error = error
        self.calls: list[dict[str, object]] = []
        self.workers: list[_FakeWorker] = []
        self.log: list[str] = []

    def __call__(self, command: list[str], **kwargs: object) -> _FakeWorker:
        self.calls.append({"command": list(command), **kwargs})
        self.log.append("start")
        if self._error is not None:
            raise self._error
        plan = self._plans.pop(0) if self._plans else [_response() for _ in range(3)]
        if isinstance(plan, Exception):
            raise plan
        if isinstance(plan, _FakeWorker):
            worker = plan
            worker.log = self.log
        else:
            worker = _FakeWorker(plan, ready=self._ready, log=self.log)  # type: ignore[arg-type]
        self.workers.append(worker)
        return worker


def _paths(tmp_path: Path, *, fixture: Path = FIXTURE) -> dict[str, Path]:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "worker.py").write_text("# scratch worker\n", encoding="utf-8")
    return {
        "python311": tmp_path / "python311",
        "runtime_root": runtime_root,
        "checkpoint_cache": tmp_path / "cache",
        "fixture": fixture,
        "oracle": ORACLE,
        "output": tmp_path / "compare.json",
    }


def _install_checkpoint(
    monkeypatch: pytest.MonkeyPatch, compare: ModuleType, checkpoint: Path
) -> None:
    def _prepare(_config: object, _cache_root: Path, *, download: bool) -> Path:
        assert download is False
        return checkpoint

    monkeypatch.setattr(compare, "prepare_oaf_checkpoint", _prepare)


def _run(
    compare: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
    factory: _FakeFactory,
) -> dict:
    checkpoint = paths["checkpoint_cache"] / "sha256" / "checkpoint"
    _install_checkpoint(monkeypatch, compare, checkpoint)
    return compare.run_comparison(
        python311=paths["python311"],
        runtime_root=paths["runtime_root"],
        checkpoint_cache=paths["checkpoint_cache"],
        fixture=paths["fixture"],
        oracle_path=paths["oracle"],
        process_factory=factory,
    )


def _cli(paths: dict[str, Path]) -> list[str]:
    return [
        "--python311",
        str(paths["python311"]),
        "--runtime-root",
        str(paths["runtime_root"]),
        "--checkpoint-cache",
        str(paths["checkpoint_cache"]),
        "--fixture",
        str(paths["fixture"]),
        "--oracle",
        str(paths["oracle"]),
        "--output",
        str(paths["output"]),
    ]


def test_driver_launches_the_scratch_worker_with_an_isolated_environment(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PATH", "/sentinel/bin")
    monkeypatch.setenv("HOME", "/sentinel/home")
    monkeypatch.setenv("TMPDIR", "/sentinel/tmp")
    for key in SANITIZED_VARIABLES:
        monkeypatch.setenv(key, f"/sentinel/{key.lower()}")
    paths = _paths(tmp_path)
    checkpoint = paths["checkpoint_cache"] / "sha256" / "checkpoint"
    factory = _FakeFactory()

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "smoke_exact"
    assert record["state"]["failed_stage"] is None
    assert [stage["name"] for stage in record["stages"]] == [
        "config",
        "checkpoint",
        "input",
        "worker_start",
        "restore",
        "inference",
    ]
    assert len(factory.calls) == 2
    assert factory.calls[0] == factory.calls[1]
    call = factory.calls[0]
    assert call["command"] == [
        str(paths["python311"]),
        "-s",
        str(paths["runtime_root"] / "worker.py"),
        "--checkpoint-dir",
        str(checkpoint),
    ]
    assert call["cwd"] == FIXTURE.parent
    environment = call["env"]
    assert isinstance(environment, dict)
    assert environment["PATH"] == "/sentinel/bin"
    assert environment["HOME"] == "/sentinel/home"
    assert environment["TMPDIR"] == "/sentinel/tmp"
    assert all(key not in environment for key in SANITIZED_VARIABLES)
    assert {key: environment[key] for key in WORKER_FLAGS} == WORKER_FLAGS
    assert factory.log == ["start", "request", "request", "close", "start", "request", "close"]
    assert [worker.requests for worker in factory.workers] == [[FIXTURE.name] * 2, [FIXTURE.name]]
    assert [worker.close_count for worker in factory.workers] == [1, 1]


def test_driver_writes_a_smoke_exact_record_and_exits_zero(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    factory = _FakeFactory()
    _install_checkpoint(monkeypatch, compare, paths["checkpoint_cache"] / "sha256" / "checkpoint")

    code = compare.main(_cli(paths), process_factory=factory)

    assert code == 0
    record = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert record["schema"] == "oaf-tf215-compare/v2"
    assert record["status"] == "smoke_exact"
    assert record["state"] == {
        "inference_completed": True,
        "repeatable": True,
        "oracle_exact": True,
        "failed_stage": None,
    }
    assert set(record["state"]) == {
        "inference_completed",
        "repeatable",
        "oracle_exact",
        "failed_stage",
    }
    restore = record["stages"][-2]
    assert restore["name"] == "restore"
    assert restore["details"]["restored_tensor_count"] == 78
    inference = record["stages"][-1]
    assert inference["name"] == "inference"
    assert inference["details"]["event_counts"] == [3, 3, 3]


def test_driver_retains_full_events_for_two_runs_and_a_fresh_worker(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)

    record = _run(compare, monkeypatch, paths, _FakeFactory())

    expected_events = [asdict(event) for event in ORACLE_EVENTS]
    inference = record["inference"]
    assert inference["relative_wav"] == FIXTURE.name
    assert inference["oracle_events"] == expected_events
    assert [(run["run"], run["worker"]) for run in inference["runs"]] == [
        (1, 1),
        (2, 1),
        (3, 2),
    ]
    assert [run["events"] for run in inference["runs"]] == [expected_events] * 3
    assert inference["fresh_worker"] == {
        "restored_tensor_count": 78,
        "backend_id": WORKER_BACKEND_ID,
    }
    assert inference["deltas"] == []


@pytest.mark.parametrize(
    "ready",
    [
        {"type": "ready", "restored_tensor_count": 77},
        {"type": "ready", "restored_tensor_count": 130},
        {"type": "ready"},
        {"restored_tensor_count": 78},
        {"type": "starting", "restored_tensor_count": 78},
        {"type": "ready", "restored_tensor_count": "78"},
        {"type": "ready", "restored_tensor_count": 78.0},
        {"type": "ready", "restored_tensor_count": True},
        {"type": "ready", "restored_tensor_count": None},
        None,
    ],
)
def test_driver_requires_a_real_restored_tensor_count(
    compare: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ready: object,
) -> None:
    paths = _paths(tmp_path)
    factory = _FakeFactory(ready=ready)

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "failed"
    assert record["state"]["failed_stage"] == "restore"
    assert [stage["name"] for stage in record["stages"]] == [
        "config",
        "checkpoint",
        "input",
        "worker_start",
        "restore",
    ]
    assert record["stages"][-1]["status"] == "failed"
    (worker,) = factory.workers
    assert worker.requests == []
    assert worker.close_count == 1


def test_driver_writes_the_failure_record_and_exits_nonzero(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    factory = _FakeFactory(ready={"type": "ready", "restored_tensor_count": 77})
    _install_checkpoint(monkeypatch, compare, paths["checkpoint_cache"] / "sha256" / "checkpoint")

    code = compare.main(_cli(paths), process_factory=factory)

    assert code == 1
    record = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["state"]["failed_stage"] == "restore"
    assert "restored tensor count" in record["stages"][-1]["error"]


def test_driver_reports_a_child_initialization_failure_as_failed(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    factory = _FakeFactory(
        error=WorkerProcessError("worker exited before ready: ImportError: no tensorflow")
    )

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "failed"
    assert record["state"]["failed_stage"] == "worker_start"
    assert [stage["name"] for stage in record["stages"]] == [
        "config",
        "checkpoint",
        "input",
        "worker_start",
    ]
    assert "ImportError" in record["stages"][-1]["error"]


def test_driver_reports_a_missing_interpreter_as_blocked(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    _install_checkpoint(monkeypatch, compare, paths["checkpoint_cache"] / "sha256" / "checkpoint")

    code = compare.main(_cli(paths))

    assert code == 1
    record = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert record["status"] == "blocked"
    assert record["state"]["failed_stage"] == "host"
    assert [stage["name"] for stage in record["stages"]] == [
        "config",
        "checkpoint",
        "input",
        "host",
    ]
    assert "WorkerProcessError: worker could not be started" in record["stages"][-1]["error"]


def test_driver_reports_a_real_worker_import_failure_as_failed(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    paths["python311"] = Path(sys.executable)
    (paths["runtime_root"] / "worker.py").write_text(
        "raise ImportError('tensorflow is unavailable')\n", encoding="utf-8"
    )
    _install_checkpoint(monkeypatch, compare, paths["checkpoint_cache"] / "sha256" / "checkpoint")

    code = compare.main(_cli(paths))

    assert code == 1
    record = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["state"]["failed_stage"] == "worker_start"
    assert [stage["name"] for stage in record["stages"]] == [
        "config",
        "checkpoint",
        "input",
        "worker_start",
    ]
    assert "worker exited before ready" in record["stages"][-1]["error"]


def test_driver_rejects_a_fixture_that_is_not_the_oracle_input(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = bytearray(FIXTURE.read_bytes())
    content[-1] ^= 0xFF
    mismatched = tmp_path / "canonical.wav"
    mismatched.write_bytes(bytes(content))
    paths = _paths(tmp_path, fixture=mismatched)
    factory = _FakeFactory()

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "failed"
    assert record["state"]["failed_stage"] == "input"
    assert "input_audio_sha256" in record["stages"][-1]["error"]
    assert factory.calls == []


def test_driver_reports_a_checkpoint_acquisition_failure_as_blocked(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)

    def _fail(_config: object, _cache_root: Path, *, download: bool) -> Path:
        assert download is False
        raise CheckpointAcquisitionError("checkpoint cache is missing")

    monkeypatch.setattr(compare, "prepare_oaf_checkpoint", _fail)
    factory = _FakeFactory()

    record = compare.run_comparison(
        python311=paths["python311"],
        runtime_root=paths["runtime_root"],
        checkpoint_cache=paths["checkpoint_cache"],
        fixture=paths["fixture"],
        oracle_path=paths["oracle"],
        process_factory=factory,
    )

    assert record["status"] == "blocked"
    assert record["state"]["failed_stage"] == "checkpoint"
    assert "checkpoint cache is missing" in record["stages"][-1]["error"]
    assert factory.calls == []


@pytest.mark.parametrize(
    ("reason", "status", "failed_stage", "stage_name"),
    [
        (
            "checkpoint component hash differs",
            "failed",
            "checkpoint_validation",
            "checkpoint_validation",
        ),
        (
            "checkpoint cache entries differ",
            "failed",
            "checkpoint_validation",
            "checkpoint_validation",
        ),
        (
            "checkpoint cache component is unavailable",
            "blocked",
            "checkpoint",
            "checkpoint",
        ),
        (
            "checkpoint cache destination is unavailable",
            "blocked",
            "checkpoint",
            "checkpoint",
        ),
    ],
)
def test_driver_classifies_a_populated_cache_failure_by_reason(
    compare: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reason: str,
    status: str,
    failed_stage: str,
    stage_name: str,
) -> None:
    paths = _paths(tmp_path)
    archive = load_model_config(compare.MODEL_CONFIG_PATH).checkpoint.archive_sha256
    (paths["checkpoint_cache"] / "sha256" / archive).mkdir(parents=True)

    def _fail(_config: object, _cache_root: Path, *, download: bool) -> Path:
        assert download is False
        raise CheckpointAcquisitionError(reason)

    monkeypatch.setattr(compare, "prepare_oaf_checkpoint", _fail)
    factory = _FakeFactory()

    record = compare.run_comparison(
        python311=paths["python311"],
        runtime_root=paths["runtime_root"],
        checkpoint_cache=paths["checkpoint_cache"],
        fixture=paths["fixture"],
        oracle_path=paths["oracle"],
        process_factory=factory,
    )

    assert record["status"] == status
    assert record["state"]["failed_stage"] == failed_stage
    assert [stage["name"] for stage in record["stages"]] == ["config", stage_name]
    assert reason in record["stages"][-1]["error"]
    assert factory.calls == []


def test_driver_records_a_model_configuration_failure(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)

    def _fail(_path: Path) -> object:
        raise OafModelConfigError("model configuration is invalid")

    monkeypatch.setattr(compare, "load_model_config", _fail)
    factory = _FakeFactory()

    code = compare.main(_cli(paths), process_factory=factory)

    assert code == 1
    record = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["state"]["failed_stage"] == "config"
    assert [stage["name"] for stage in record["stages"]] == ["config"]
    assert "model configuration is invalid" in record["stages"][-1]["error"]
    assert factory.calls == []


@pytest.mark.parametrize(
    ("inference_completed", "repeatable", "oracle_exact", "failed_stage", "expected"),
    [
        (None, None, None, "config", "failed"),
        (None, None, None, "checkpoint", "blocked"),
        (None, None, None, "checkpoint_validation", "failed"),
        (None, None, None, "host", "blocked"),
        (None, None, None, "worker_start", "failed"),
        (None, None, None, "restore", "failed"),
        (None, None, None, "input", "failed"),
        (None, None, None, "inference", "failed"),
        (None, None, None, "worker_close", "failed"),
        (None, None, None, None, "environment_only_passed"),
        (None, True, True, None, "failed"),
        (None, False, None, None, "failed"),
        (None, None, True, None, "failed"),
        (None, None, False, None, "failed"),
        (False, None, None, None, "failed"),
        (True, None, None, None, "failed"),
        (True, False, None, None, "non_repeatable"),
        (True, False, False, None, "non_repeatable"),
        (True, True, None, None, "failed"),
        (True, True, False, None, "oracle_divergent"),
        (True, True, True, None, "smoke_exact"),
        (True, True, True, "worker_close", "failed"),
    ],
)
def test_classify_result_labels(
    compare: ModuleType,
    inference_completed: bool | None,
    repeatable: bool | None,
    oracle_exact: bool | None,
    failed_stage: str | None,
    expected: str,
) -> None:
    assert (
        compare.classify_result(inference_completed, repeatable, oracle_exact, failed_stage)
        == expected
    )


def test_driver_reports_non_repeatable_runs(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    divergent = _divergent_events()
    plans = [
        _FakeWorker([_response(), _response()]),
        _FakeWorker([_response(divergent)]),
    ]
    _install_checkpoint(monkeypatch, compare, paths["checkpoint_cache"] / "sha256" / "checkpoint")

    code = compare.main(_cli(paths), process_factory=_FakeFactory(plans))

    assert code == 1
    record = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert record["status"] == "non_repeatable"
    assert record["state"] == {
        "inference_completed": True,
        "repeatable": False,
        "oracle_exact": None,
        "failed_stage": None,
    }
    assert record["inference"]["runs"][2]["events"] == [
        asdict(event) for event in _converted(divergent)
    ]


def test_driver_reports_a_repeatable_oracle_divergence(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    divergent = _divergent_events()
    plans = [
        _FakeWorker([_response(divergent), _response(divergent)]),
        _FakeWorker([_response(divergent)]),
    ]
    _install_checkpoint(monkeypatch, compare, paths["checkpoint_cache"] / "sha256" / "checkpoint")

    code = compare.main(_cli(paths), process_factory=_FakeFactory(plans))

    assert code == 1
    record = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert record["status"] == "oracle_divergent"
    assert record["state"] == {
        "inference_completed": True,
        "repeatable": True,
        "oracle_exact": False,
        "failed_stage": None,
    }
    deltas = record["inference"]["deltas"]
    assert deltas == [
        {
            "run": run,
            "index": 0,
            "fields": {"native_midi_note": {"actual": 39, "expected": 38}},
        }
        for run in (1, 2, 3)
    ]


def test_driver_caps_recorded_field_deltas(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    events = _synthetic_events(20)
    plans = [
        _FakeWorker([_response(events), _response(events)]),
        _FakeWorker([_response(events)]),
    ]

    record = _run(compare, monkeypatch, paths, _FakeFactory(plans))

    assert record["status"] == "oracle_divergent"
    deltas = record["inference"]["deltas"]
    assert len(deltas) == compare.MAX_RECORDED_DELTAS
    assert deltas[0] == {
        "run": 1,
        "event_count": {"actual": 20, "expected": 3},
    }


def test_driver_caps_a_count_delta_at_the_boundary_of_mixed_runs(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    plans = [
        _FakeWorker([_response(_synthetic_events(4)), _response(_synthetic_events(3))]),
        _FakeWorker([_response(_synthetic_events(4))]),
    ]

    record = _run(compare, monkeypatch, paths, _FakeFactory(plans))

    assert record["status"] == "non_repeatable"
    deltas = record["inference"]["deltas"]
    assert len(deltas) == compare.MAX_RECORDED_DELTAS == 8
    assert deltas[-1] == {"run": 3, "event_count": {"actual": 4, "expected": 3}}


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (WorkerProcessError("inference failed"), "inference failed"),
        (
            {
                "id": "fake-request",
                "error": {"code": "inference_failed", "message": "inference failed"},
            },
            "worker inference failed",
        ),
        ({"id": "fake-request", "events": [{"not": "an event"}]}, "worker event is invalid"),
    ],
)
def test_driver_records_an_inference_failure_and_closes_the_first_worker(
    compare: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: object,
    message: str,
) -> None:
    paths = _paths(tmp_path)
    plans = [[_response(), failure]]
    factory = _FakeFactory(plans)
    _install_checkpoint(monkeypatch, compare, paths["checkpoint_cache"] / "sha256" / "checkpoint")

    code = compare.main(_cli(paths), process_factory=factory)

    assert code == 1
    record = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["state"] == {
        "inference_completed": False,
        "repeatable": None,
        "oracle_exact": None,
        "failed_stage": "inference",
    }
    assert message in record["stages"][-1]["error"]
    (worker,) = factory.workers
    assert worker.close_count == 1
    assert len(record["inference"]["runs"]) == 1


def test_driver_reports_a_fresh_worker_start_failure_after_closing_the_first(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    plans = [
        _FakeWorker([_response(), _response()]),
        WorkerProcessError("worker exited before ready: ImportError: no tensorflow"),
    ]
    factory = _FakeFactory(plans)

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "failed"
    assert record["state"]["failed_stage"] == "inference"
    assert "worker exited before ready" in record["stages"][-1]["error"]
    assert len(factory.workers) == 1
    assert factory.workers[0].close_count == 1


def test_driver_reports_a_fresh_worker_host_failure_as_blocked(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    plans = [_FakeWorker([_response(), _response()]), _host_error()]
    factory = _FakeFactory(plans)

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "blocked"
    assert record["state"]["failed_stage"] == "host"
    assert record["stages"][-1]["name"] == "host"
    assert "worker could not be started" in record["stages"][-1]["error"]
    assert len(factory.workers) == 1
    assert factory.workers[0].close_count == 1
    assert len(record["inference"]["runs"]) == 2


def test_driver_keeps_a_worker_close_failure_over_a_later_host_blocker(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    plans = [
        _FakeWorker([_response(), _response()], close_error=WorkerProcessError("close failed")),
        _host_error(),
    ]
    factory = _FakeFactory(plans)

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "failed"
    assert record["state"]["failed_stage"] == "worker_close"
    assert [stage["name"] for stage in record["stages"][-2:]] == ["worker_close", "host"]
    assert "close failed" in record["stages"][-2]["error"]
    assert "worker could not be started" in record["stages"][-1]["error"]
    assert len(factory.workers) == 1
    assert factory.workers[0].close_count == 1
    assert len(record["inference"]["runs"]) == 2


def test_driver_retains_comparison_when_a_worker_close_fails(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    plans = [
        _FakeWorker([_response(), _response()], close_error=WorkerProcessError("close failed")),
        _FakeWorker([_response()]),
    ]
    factory = _FakeFactory(plans)

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "failed"
    assert record["state"] == {
        "inference_completed": True,
        "repeatable": True,
        "oracle_exact": True,
        "failed_stage": "worker_close",
    }
    assert len(record["inference"]["runs"]) == 3
    assert record["inference"]["deltas"] == []
    assert [stage["name"] for stage in record["stages"][-2:]] == ["worker_close", "inference"]
    assert "close failed" in record["stages"][-2]["error"]
    assert [worker.close_count for worker in factory.workers] == [1, 1]


def test_driver_retains_comparison_when_the_fresh_worker_close_fails(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    plans = [
        _FakeWorker([_response(), _response()]),
        _FakeWorker([_response()], close_error=WorkerProcessError("fresh close failed")),
    ]
    factory = _FakeFactory(plans)

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "failed"
    assert record["state"] == {
        "inference_completed": True,
        "repeatable": True,
        "oracle_exact": True,
        "failed_stage": "worker_close",
    }
    assert len(record["inference"]["runs"]) == 3
    assert record["inference"]["deltas"] == []
    assert [stage["name"] for stage in record["stages"][-2:]] == ["worker_close", "inference"]
    assert "fresh close failed" in record["stages"][-2]["error"]
    assert [worker.close_count for worker in factory.workers] == [1, 1]


def test_driver_preserves_the_primary_inference_failure_when_close_also_fails(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    plans = [
        _FakeWorker(
            [_response(), WorkerProcessError("inference failed")],
            close_error=WorkerProcessError("close failed"),
        )
    ]
    factory = _FakeFactory(plans)

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "failed"
    assert record["state"]["failed_stage"] == "inference"
    assert record["state"]["inference_completed"] is False
    inference_stage = record["stages"][-1]
    close_stage = record["stages"][-2]
    assert inference_stage["name"] == "inference"
    assert "inference failed" in inference_stage["error"]
    assert close_stage["name"] == "worker_close"
    assert "close failed" in close_stage["error"]
    assert len(factory.workers) == 1
    assert factory.workers[0].close_count == 1
    assert len(record["inference"]["runs"]) == 1


def test_driver_validates_the_fresh_worker_restore_count(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    plans = [
        _FakeWorker([_response(), _response()]),
        _FakeWorker([_response()], ready={"type": "ready", "restored_tensor_count": 77}),
    ]
    factory = _FakeFactory(plans)

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "failed"
    assert record["state"]["inference_completed"] is False
    assert record["state"]["failed_stage"] == "inference"
    assert "restored tensor count" in record["stages"][-1]["error"]
    assert [worker.close_count for worker in factory.workers] == [1, 1]


def test_driver_requires_absolute_input_paths(compare: ModuleType, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    arguments = _cli(paths)
    arguments[arguments.index("--fixture") + 1] = "tests/fixtures/oaf_tf1_smoke/canonical.wav"

    with pytest.raises(SystemExit) as error:
        compare.main(arguments)

    assert error.value.code == 2
