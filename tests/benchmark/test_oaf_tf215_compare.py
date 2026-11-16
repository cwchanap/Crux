"""Host-driver tests for the scratch TF 2.15 OaF worker (HPA-856 Task 3).

The driver runs on the Crux Python 3.12 environment and only launches the
Python 3.11 scratch worker. These tests use a fake process factory: they check
the launched command/environment and the restore accounting, never the frozen
tensor inventory or a real checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from src.benchmark.checkpoint_acquisition import CheckpointAcquisitionError
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


class _FakeWorker:
    def __init__(self, ready: object) -> None:
        self.ready = ready
        self.requests: list[str] = []
        self.close_count = 0

    def request(self, path: str, **_kwargs: object) -> dict[str, object]:
        self.requests.append(path)
        raise AssertionError("Task 3 must not issue an inference request")

    def close(self) -> None:
        self.close_count += 1


class _FakeFactory:
    def __init__(self, ready: object = READY, error: Exception | None = None) -> None:
        self._ready = ready
        self._error = error
        self.calls: list[dict[str, object]] = []
        self.worker: _FakeWorker | None = None

    def __call__(self, command: list[str], **kwargs: object) -> _FakeWorker:
        self.calls.append({"command": list(command), **kwargs})
        if self._error is not None:
            raise self._error
        self.worker = _FakeWorker(self._ready)
        return self.worker


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

    assert record["status"] == "passed"
    assert record["failed_stage"] is None
    assert [stage["name"] for stage in record["stages"]] == [
        "checkpoint",
        "input",
        "worker_start",
        "restore",
    ]
    (call,) = factory.calls
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
    assert factory.worker is not None
    assert factory.worker.requests == []
    assert factory.worker.close_count == 1


def test_driver_writes_a_passed_record_and_exits_zero(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    factory = _FakeFactory()
    _install_checkpoint(monkeypatch, compare, paths["checkpoint_cache"] / "sha256" / "checkpoint")

    code = compare.main(_cli(paths), process_factory=factory)

    assert code == 0
    record = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert record["schema"] == "oaf-tf215-compare/v1"
    assert record["status"] == "passed"
    restore = record["stages"][-1]
    assert restore["name"] == "restore"
    assert restore["details"]["restored_tensor_count"] == 78


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
    assert record["failed_stage"] == "restore"
    assert [stage["name"] for stage in record["stages"]] == [
        "checkpoint",
        "input",
        "worker_start",
        "restore",
    ]
    assert record["stages"][-1]["status"] == "failed"
    assert factory.worker is not None
    assert factory.worker.close_count == 1


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
    assert record["failed_stage"] == "restore"
    assert "restored tensor count" in record["stages"][-1]["error"]


def test_driver_reports_a_worker_start_failure(
    compare: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    factory = _FakeFactory(error=WorkerProcessError("worker exited before ready: import failed"))

    record = _run(compare, monkeypatch, paths, factory)

    assert record["status"] == "failed"
    assert record["failed_stage"] == "worker_start"
    assert [stage["name"] for stage in record["stages"]] == [
        "checkpoint",
        "input",
        "worker_start",
    ]
    assert "import failed" in record["stages"][-1]["error"]


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
    assert record["failed_stage"] == "input"
    assert "input_audio_sha256" in record["stages"][-1]["error"]
    assert factory.calls == []


def test_driver_reports_a_checkpoint_acquisition_failure(
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

    assert record["status"] == "failed"
    assert record["failed_stage"] == "checkpoint"
    assert "checkpoint cache is missing" in record["stages"][-1]["error"]
    assert factory.calls == []


def test_driver_requires_absolute_input_paths(compare: ModuleType, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    arguments = _cli(paths)
    arguments[arguments.index("--fixture") + 1] = "tests/fixtures/oaf_tf1_smoke/canonical.wav"

    with pytest.raises(SystemExit) as error:
        compare.main(arguments)

    assert error.value.code == 2
