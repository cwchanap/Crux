# ruff: noqa: E402 - the repository root is on sys.path before first-party imports.
"""Run three real inferences through the scratch TF 2.15 OaF worker.

This is an experimental host driver, not a production entry point. It runs under
the Crux Python 3.12 environment, reuses the frozen checkpoint, fixture, worker
protocol and oracle modules, and launches only the Python 3.11 scratch worker
created by ``build_runtime.py``. Two requests run through one loaded worker and
a third through a fresh worker, so run repeatability and identity with the
authoritative binary64 smoke oracle are measured, not assumed.

The checkpoint cache is verified here, never downloaded. Populate it with the
existing public command first::

    uv run crux benchmark prepare-backend --backend oaf --download

Usage::

    uv run experiments/oaf_tf215/compare.py \
        --python311 /path/to/python3.11 \
        --runtime-root /path/to/scratch/runtime \
        --checkpoint-cache /abs/path/artifacts/benchmark/model-cache \
        --fixture /abs/path/tests/fixtures/oaf_tf1_smoke/canonical.wav \
        --oracle /abs/path/tests/fixtures/oaf_tf1_smoke/smoke-oracle.json \
        --output artifacts/experiments/oaf-tf215/compare.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from runtime.oaf_tf1.model import REQUIRED_INFERENCE_COUNT, OafModelConfig, load_model_config
from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backends.idm import _isolated_worker_environment
from src.benchmark.backends.oaf import _decode_native_event
from src.benchmark.checkpoint_acquisition import (
    CheckpointAcquisitionError,
    prepare_oaf_checkpoint,
)
from src.benchmark.input_view import load_direct_audio
from src.benchmark.oaf_smoke_oracle import (
    SmokeOracle,
    SmokeOracleEvent,
    _event_from_native,
    read_smoke_oracle,
)
from src.benchmark.worker_process import WorkerProcess, WorkerProcessError

SCHEMA = "oaf-tf215-compare/v2"
MODEL_CONFIG_PATH = REPOSITORY_ROOT / "runtime" / "oaf_tf1" / "model.json"
SOURCE_AUDIO_ID = "oaf-smoke-canonical-v1"
# A cold TensorFlow import plus a real checkpoint restore can exceed one minute.
WORKER_START_TIMEOUT_SECONDS = 300.0
WORKER_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "PYTHONHASHSEED": "0",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "NUMBA_CACHE_DIR": "/tmp/numba-cache",
}
# Pre-inference failures mean the experiment could not reach a loaded model.
# ``checkpoint`` is cache access and ``host`` is interpreter/host availability;
# both are blocked (not proven incompatible), never a pass. A child that starts
# but fails to become ready is a real failure under ``worker_start``.
BLOCKED_STAGES = frozenset({"checkpoint", "host"})
# The frozen verifier's access-vs-integrity reasons for a populated cache. A
# component/entry mismatch is real evidence and fails; unavailable reads are
# blocked. These are the only reasons the verify-only path can raise.
CHECKPOINT_ACCESS_REASONS = frozenset(
    {
        "checkpoint cache is missing",
        "checkpoint cache destination is unavailable",
        "checkpoint cache component is unavailable",
    }
)
# Diagnostics stay small: only the first differing event fields are recorded.
MAX_RECORDED_DELTAS = 8
EVENT_FIELDS = tuple(field.name for field in fields(SmokeOracleEvent))

ProcessFactory = Callable[..., Any]


class CompareError(RuntimeError):
    """A driver stage produced an invalid or unexpected result."""


class StageLabel(RuntimeError):
    """Attribute a failure to a more specific stage than the caller's."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class StageFailure(RuntimeError):
    """A named driver stage failed."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def classify_result(
    inference_completed: bool | None,
    repeatable: bool | None,
    oracle_exact: bool | None,
    failed_stage: str | None,
) -> str:
    """Derive the one result label from the nullable tri-state record.

    ``blocked`` means external checkpoint or host access was unavailable;
    ``failed`` is any other experiment failure with its stage. Environment-only
    scope requires an entirely absent tri-state; contradictory claims are
    rejected as ``failed`` rather than reported as success.
    ``oracle_divergent`` is repeatable but differs from the frozen oracle, and
    ``non_repeatable`` means the candidate runs differ among themselves.
    """
    if failed_stage is not None:
        return "blocked" if failed_stage in BLOCKED_STAGES else "failed"
    if inference_completed is None and repeatable is None and oracle_exact is None:
        return "environment_only_passed"
    if inference_completed is not True:
        return "failed"
    if repeatable is False:
        return "non_repeatable"
    if repeatable is not True:
        return "failed"
    if oracle_exact is True:
        return "smoke_exact"
    return "oracle_divergent" if oracle_exact is False else "failed"


def _load_config(holder: dict[str, Any]) -> dict[str, Any]:
    config = load_model_config(MODEL_CONFIG_PATH)
    holder["config"] = config
    return {
        "backend_id": config.backend_id,
        "model_id": config.model_id,
        "archive_sha256": config.checkpoint.archive_sha256,
    }


def _prepare_checkpoint(config: OafModelConfig, checkpoint_cache: Path) -> dict[str, Any]:
    try:
        checkpoint = prepare_oaf_checkpoint(config, checkpoint_cache, download=False)
    except CheckpointAcquisitionError as error:
        # Classify the verifier's reason: unavailable reads stay blocked, while
        # component or entry mismatches are real validation failures.
        if str(error) in CHECKPOINT_ACCESS_REASONS:
            raise
        raise StageLabel("checkpoint_validation", str(error)) from error
    return {
        "cache_root": os.fspath(checkpoint_cache),
        "checkpoint": os.fspath(checkpoint),
        "archive_sha256": config.checkpoint.archive_sha256,
    }


def _validate_input(
    config: OafModelConfig,
    fixture: Path,
    oracle_path: Path,
    holder: dict[str, Any],
) -> dict[str, Any]:
    audio = load_direct_audio(
        fixture,
        source_audio_id=SOURCE_AUDIO_ID,
        input_view_id=SOURCE_AUDIO_ID,
        max_input_audio_frames=config.max_input_audio_frames,
    )
    oracle = read_smoke_oracle(read_regular_file_no_follow(oracle_path))
    if audio.input_audio_sha256 != oracle.input_audio_sha256:
        raise CompareError("fixture input_audio_sha256 differs from the smoke oracle")
    holder["oracle"] = oracle
    return {
        "fixture": os.fspath(fixture),
        "input_audio_sha256": audio.input_audio_sha256,
        "oracle_input_audio_sha256": oracle.input_audio_sha256,
        "audio_frame_count": audio.audio_frame_count,
        "sample_rate": audio.sample_rate,
        "oracle_native_event_count": len(oracle.native_events),
    }


def _worker_environment() -> dict[str, str]:
    environment = _isolated_worker_environment()
    environment.update(WORKER_ENVIRONMENT)
    return environment


def _start_worker(
    process_factory: ProcessFactory,
    python311: Path,
    runtime_root: Path,
    checkpoint: Path,
    fixture: Path,
    holder: dict[str, Any],
    *,
    timeout_seconds: float = WORKER_START_TIMEOUT_SECONDS,
    close_timeout_seconds: float | None = None,
    worker_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    worker = runtime_root / "worker.py"
    if worker.is_symlink() or not worker.is_file():
        raise CompareError(f"scratch worker is unavailable: {worker}")
    command = [
        os.fspath(python311),
        "-s",
        os.fspath(worker),
        "--checkpoint-dir",
        os.fspath(checkpoint),
    ]
    try:
        close_options = (
            {}
            if close_timeout_seconds is None
            else {"close_timeout_seconds": close_timeout_seconds}
        )
        holder["process"] = process_factory(
            command,
            timeout_seconds=timeout_seconds,
            cwd=fixture.parent,
            env=_worker_environment() if worker_environment is None else worker_environment,
            **close_options,
        )
    except WorkerProcessError as error:
        # Popen failed, so the interpreter or host itself is unavailable. A
        # child that executed and then failed is a real initialization failure.
        if isinstance(error.__cause__, OSError):
            raise StageLabel("host", str(error)) from error
        raise
    return {
        "command": command,
        "cwd": os.fspath(fixture.parent),
        "flags": dict(WORKER_ENVIRONMENT),
    }


def _validate_worker_ready(ready: object) -> dict[str, Any]:
    if not isinstance(ready, Mapping) or ready.get("type") != "ready":
        raise CompareError("worker ready response is invalid")
    count = ready.get("restored_tensor_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != REQUIRED_INFERENCE_COUNT:
        raise CompareError(f"worker restored tensor count is invalid: {count!r}")
    return {"restored_tensor_count": count, "backend_id": ready.get("backend_id")}


def _request_events(process: Any, relative_wav: str) -> tuple[SmokeOracleEvent, ...]:
    """Decode one worker response into the frozen oracle event representation."""
    response = process.request(relative_wav)
    if not isinstance(response, Mapping):
        raise CompareError("worker response is invalid")
    error = response.get("error")
    if isinstance(error, Mapping) and isinstance(error.get("message"), str):
        raise CompareError(f"worker inference failed: {error['message']}")
    if "error" in response:
        raise CompareError("worker response is invalid")
    raw_events = response.get("events")
    if not isinstance(raw_events, (list, tuple)):
        raise CompareError("worker response is invalid")
    return tuple(_event_from_native(_decode_native_event(value)) for value in raw_events)


def _oracle_deltas(
    attempts: Sequence[tuple[SmokeOracleEvent, ...]],
    oracle_events: Sequence[SmokeOracleEvent],
) -> list[dict[str, Any]]:
    """Record the first differing fields of every attempt, capped in size."""
    deltas: list[dict[str, Any]] = []
    for run, events in enumerate(attempts, start=1):
        if len(events) != len(oracle_events):
            deltas.append(
                {
                    "run": run,
                    "event_count": {"actual": len(events), "expected": len(oracle_events)},
                }
            )
            if len(deltas) >= MAX_RECORDED_DELTAS:
                return deltas
        for index, (actual, expected) in enumerate(zip(events, oracle_events, strict=False)):
            changed = {
                name: {"actual": getattr(actual, name), "expected": getattr(expected, name)}
                for name in EVENT_FIELDS
                if getattr(actual, name) != getattr(expected, name)
            }
            if changed:
                deltas.append({"run": run, "index": index, "fields": changed})
            if len(deltas) >= MAX_RECORDED_DELTAS:
                return deltas
    return deltas


def _run_inference(
    record: dict[str, Any],
    holder: dict[str, Any],
    process_factory: ProcessFactory,
    python311: Path,
    runtime_root: Path,
    checkpoint: Path,
    fixture: Path,
) -> dict[str, Any]:
    """Two runs through the loaded worker, then one through a fresh worker."""
    state = record["state"]
    oracle: SmokeOracle = holder["oracle"]
    oracle_events = tuple(oracle.native_events)
    relative_wav = fixture.name
    record["inference"]["relative_wav"] = relative_wav
    record["inference"]["oracle_events"] = [asdict(event) for event in oracle_events]
    state["inference_completed"] = False
    attempts: list[tuple[SmokeOracleEvent, ...]] = []
    runs = record["inference"]["runs"]

    first = holder.pop("process")
    try:
        for run in (1, 2):
            events = _request_events(first, relative_wav)
            attempts.append(events)
            runs.append(
                {
                    "run": run,
                    "worker": 1,
                    "event_count": len(events),
                    "events": [asdict(event) for event in events],
                }
            )
    finally:
        # Ownership is gone, and _close_worker never raises, so a cleanup
        # failure can neither mask the primary error nor double-close.
        _close_worker(first, record)

    _start_worker(process_factory, python311, runtime_root, checkpoint, fixture, holder)
    fresh = holder.pop("process")
    try:
        record["inference"]["fresh_worker"] = _validate_worker_ready(fresh.ready)
        events = _request_events(fresh, relative_wav)
        attempts.append(events)
        runs.append(
            {
                "run": 3,
                "worker": 2,
                "event_count": len(events),
                "events": [asdict(event) for event in events],
            }
        )
        # Retain completion and comparison before cleanup: a close failure
        # still makes the final result fail, but must not discard evidence.
        state["inference_completed"] = True
        repeatable = attempts[0] == attempts[1] == attempts[2]
        state["repeatable"] = repeatable
        state["oracle_exact"] = attempts[0] == oracle_events if repeatable else None
        record["inference"]["deltas"] = _oracle_deltas(attempts, oracle_events)
    finally:
        _close_worker(fresh, record)

    return {"event_counts": [run["event_count"] for run in runs]}


def run_stage(record: dict[str, Any], name: str, action: Callable[[], Any]) -> Any:
    """Run one named stage, recording duration, details and the failure."""
    started = time.monotonic()
    try:
        details = action()
    except StageLabel as error:
        cause = error.__cause__
        detail = (
            f"{type(cause).__name__}: {cause}"
            if cause is not None
            else f"{type(error).__name__}: {error}"
        )
        record["stages"].append(
            {
                "name": error.stage,
                "status": "failed",
                "duration_ms": round((time.monotonic() - started) * 1000),
                "error": detail,
            }
        )
        raise StageFailure(error.stage) from error
    except Exception as error:  # noqa: BLE001 - recorded as the failed stage
        record["stages"].append(
            {
                "name": name,
                "status": "failed",
                "duration_ms": round((time.monotonic() - started) * 1000),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        raise StageFailure(name) from error
    record["stages"].append(
        {
            "name": name,
            "status": "passed",
            "duration_ms": round((time.monotonic() - started) * 1000),
            "details": details,
        }
    )
    return details


def _close_worker(process: Any, record: dict[str, Any]) -> None:
    try:
        process.close()
    except Exception as error:  # noqa: BLE001 - recorded as the failed stage
        record["stages"].append(
            {
                "name": "worker_close",
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        if record["state"]["failed_stage"] is None:
            record["state"]["failed_stage"] = "worker_close"


def build_record(inputs: Mapping[str, str]) -> dict[str, Any]:
    """Create the result skeleton, including non-secret provenance only."""
    return {
        "schema": SCHEMA,
        "status": None,
        "state": {
            "inference_completed": None,
            "repeatable": None,
            "oracle_exact": None,
            "failed_stage": None,
        },
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "ci": {
            key: os.environ[key]
            for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "RUNNER_OS", "RUNNER_ARCH")
            if key in os.environ
        },
        "inputs": dict(inputs),
        "stages": [],
        "inference": {
            "relative_wav": None,
            "oracle_events": [],
            "runs": [],
            "fresh_worker": None,
            "deltas": [],
        },
    }


def finalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Derive the single result label from the recorded state."""
    record["status"] = classify_result(**record["state"])
    return record


def write_result(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_comparison(
    *,
    python311: Path,
    runtime_root: Path,
    checkpoint_cache: Path,
    fixture: Path,
    oracle_path: Path,
    process_factory: ProcessFactory | None = None,
) -> dict[str, Any]:
    """Verify checkpoint and fixture, then run three real inferences."""
    factory = WorkerProcess.start if process_factory is None else process_factory
    record = build_record(
        {
            "python311": os.fspath(python311),
            "runtime_root": os.fspath(runtime_root),
            "checkpoint_cache": os.fspath(checkpoint_cache),
            "fixture": os.fspath(fixture),
            "oracle": os.fspath(oracle_path),
        }
    )
    holder: dict[str, Any] = {}
    try:
        run_stage(record, "config", lambda: _load_config(holder))
        config = holder["config"]
        checkpoint = run_stage(
            record, "checkpoint", lambda: _prepare_checkpoint(config, checkpoint_cache)
        )
        run_stage(record, "input", lambda: _validate_input(config, fixture, oracle_path, holder))
        run_stage(
            record,
            "worker_start",
            lambda: _start_worker(
                factory,
                python311,
                runtime_root,
                Path(checkpoint["checkpoint"]),
                fixture,
                holder,
            ),
        )
        run_stage(record, "restore", lambda: _validate_worker_ready(holder["process"].ready))
        run_stage(
            record,
            "inference",
            lambda: _run_inference(
                record,
                holder,
                factory,
                python311,
                runtime_root,
                Path(checkpoint["checkpoint"]),
                fixture,
            ),
        )
    except StageFailure as failure:
        current = record["state"]["failed_stage"]
        # A later access blocker must not downgrade an actual failure (e.g. a
        # failed worker close followed by a fresh-worker Popen error). Both
        # stages stay recorded; only the label keeps the actual failure.
        if current is None or failure.stage not in BLOCKED_STAGES or current in BLOCKED_STAGES:
            record["state"]["failed_stage"] = failure.stage
    finally:
        process = holder.pop("process", None)
        if process is not None:
            _close_worker(process, record)
    return finalize_record(record)


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the public OaF checkpoint and fixture, launch the scratch "
            "Python 3.11 TF 2.15 worker and compare two same-worker plus one "
            "fresh-worker inference with the frozen binary64 smoke oracle."
        )
    )
    parser.add_argument(
        "--python311",
        required=True,
        type=_absolute_path,
        help="absolute path to the Python 3.11 interpreter",
    )
    parser.add_argument(
        "--runtime-root",
        required=True,
        type=_absolute_path,
        help="absolute scratch runtime directory containing worker.py",
    )
    parser.add_argument(
        "--checkpoint-cache",
        required=True,
        type=_absolute_path,
        help="absolute checkpoint cache root prepared by prepare-backend",
    )
    parser.add_argument(
        "--fixture",
        required=True,
        type=_absolute_path,
        help="absolute canonical WAV fixture path",
    )
    parser.add_argument(
        "--oracle",
        required=True,
        type=_absolute_path,
        help="absolute authoritative smoke-oracle JSON path",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="path for the experimental JSON result, normally "
        "artifacts/experiments/oaf-tf215/compare.json",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    process_factory: ProcessFactory | None = None,
) -> int:
    arguments = parse_args(argv)
    record = run_comparison(
        python311=arguments.python311,
        runtime_root=arguments.runtime_root,
        checkpoint_cache=arguments.checkpoint_cache,
        fixture=arguments.fixture,
        oracle_path=arguments.oracle,
        process_factory=process_factory,
    )
    write_result(arguments.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "smoke_exact" else 1


if __name__ == "__main__":
    sys.exit(main())
