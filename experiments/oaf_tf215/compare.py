# ruff: noqa: E402 - the repository root is on sys.path before first-party imports.
"""Launch the scratch TF 2.15 OaF worker and prove checkpoint restoration.

This is an experimental host driver, not a production entry point. It runs under
the Crux Python 3.12 environment, reuses the frozen checkpoint and fixture
validation modules, and launches only the Python 3.11 scratch worker created by
``build_runtime.py``. This task covers acquisition, fixture identity, worker
startup and the restored-tensor gate; inference and oracle comparison follow.

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
        --output artifacts/experiments/oaf-tf215/restore.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from runtime.oaf_tf1.model import REQUIRED_INFERENCE_COUNT, OafModelConfig, load_model_config
from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backends.idm import _isolated_worker_environment
from src.benchmark.checkpoint_acquisition import prepare_oaf_checkpoint
from src.benchmark.input_view import load_direct_audio
from src.benchmark.oaf_smoke_oracle import read_smoke_oracle
from src.benchmark.worker_process import WorkerProcess

SCHEMA = "oaf-tf215-compare/v1"
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

ProcessFactory = Callable[..., Any]


class CompareError(RuntimeError):
    """A driver stage produced an invalid or unexpected result."""


class StageFailure(RuntimeError):
    """A named driver stage failed."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def _prepare_checkpoint(config: OafModelConfig, checkpoint_cache: Path) -> dict[str, Any]:
    checkpoint = prepare_oaf_checkpoint(config, checkpoint_cache, download=False)
    return {
        "cache_root": os.fspath(checkpoint_cache),
        "checkpoint": os.fspath(checkpoint),
        "archive_sha256": config.checkpoint.archive_sha256,
    }


def _validate_input(
    config: OafModelConfig,
    fixture: Path,
    oracle_path: Path,
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
    holder["process"] = process_factory(
        command,
        timeout_seconds=WORKER_START_TIMEOUT_SECONDS,
        cwd=fixture.parent,
        env=_worker_environment(),
    )
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


def run_stage(record: dict[str, Any], name: str, action: Callable[[], Any]) -> Any:
    """Run one named stage, recording duration, details and the failure."""
    started = time.monotonic()
    try:
        details = action()
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
        if record["status"] == "passed":
            record["status"] = "failed"
            record["failed_stage"] = "worker_close"


def build_record(inputs: Mapping[str, str]) -> dict[str, Any]:
    """Create the result skeleton, including non-secret provenance only."""
    return {
        "schema": SCHEMA,
        "status": "failed",
        "failed_stage": None,
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
    }


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
    """Verify the checkpoint and fixture, then prove a real worker restore."""
    factory = WorkerProcess.start if process_factory is None else process_factory
    config = load_model_config(MODEL_CONFIG_PATH)
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
        checkpoint = run_stage(
            record, "checkpoint", lambda: _prepare_checkpoint(config, checkpoint_cache)
        )
        run_stage(record, "input", lambda: _validate_input(config, fixture, oracle_path))
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
    except StageFailure as failure:
        record["failed_stage"] = failure.stage
    else:
        record["status"] = "passed"
    finally:
        process = holder.pop("process", None)
        if process is not None:
            _close_worker(process, record)
    return record


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the public OaF checkpoint and fixture, then launch the "
            "scratch Python 3.11 TF 2.15 worker and prove its restore count."
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
        help="path for the compact redacted JSON result",
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
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
