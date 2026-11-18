# ruff: noqa: E402 - direct CLI execution needs the repository on sys.path.
"""Prepare verified TF1 inputs and measure bounded TF2.15 prediction drift."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from experiments.oaf_tf215 import compare
from experiments.oaf_tf215.runtime_identity import (
    build_candidate_descriptor,
    build_runtime_identity,
    collect_probe,
)
from experiments.oaf_tf215.selected_inputs import (
    SELECTED_SIMFILE_IDS,
    SelectionBlocked,
    SelectionError,
    prepare_fresh_selection,
    prepare_selection,
    selection_sha256,
    verify_selection,
)
from experiments.oaf_tf215.selected_metrics import (
    baseline_native_events,
    candidate_native_events,
    compare_events,
)
from runtime.oaf_tf1.model import load_model_config
from src.benchmark.artifact_io import publish_immutable_file
from src.benchmark.backend_identity import (
    canonical_json_bytes,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backends.base import CanonicalAudio
from src.benchmark.oaf_corpus_run import (
    OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
    OAF_WORKER_CLOSE_TIMEOUT_SECONDS,
)
from src.benchmark.oaf_smoke_oracle import SmokeOracleEvent
from src.benchmark.prediction_artifact import PredictionArtifact
from src.benchmark.worker_process import WorkerProcess

SCHEMA = "crux.oaf-tf215-selected-comparison/v1"
SMOKE_ROOT = REPOSITORY_ROOT / "tests/fixtures/oaf_tf1_smoke"
EXPERIMENT_MODULES = (
    "build_runtime.py",
    "compare.py",
    "runtime_identity.py",
    "selected_compare.py",
    "selected_inputs.py",
    "selected_metrics.py",
)


def _driver_source_files(root: Path) -> tuple[Path, ...]:
    experiment = Path("experiments/oaf_tf215")
    benchmark = root / "src/benchmark"
    return (
        *(experiment / filename for filename in EXPERIMENT_MODULES),
        *(path.relative_to(root) for path in sorted(benchmark.rglob("*.py"))),
        Path("runtime/oaf_tf1/model.py"),
    )


def _git_provenance() -> dict[str, Any]:
    def read(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = read("status", "--porcelain")
    return {
        "git_head": read("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def _set_failure(
    record: dict[str, Any], stage: str, error: Exception | str, *, blocked: bool = False
):
    record.update(
        status="blocked" if blocked else "failed",
        failed_stage=stage,
        error=str(error)[:1024],
    )


def _write_json(path: Path, value: Any) -> None:
    publish_immutable_file(path, canonical_json_bytes(value, trailing_newline=True))


def _run_song(
    row: dict[str, Any],
    audio: CanonicalAudio,
    baseline: PredictionArtifact,
    *,
    launch: Callable[[Path, dict[str, Any]], Any],
    stream: BinaryIO,
) -> None:
    attempts: list[tuple[SmokeOracleEvent, ...]] = []
    for worker_number, request_count in ((1, 2), (2, 1)):
        holder: dict[str, Any] = {}
        stage = "worker_start"
        try:
            launch(audio.path, holder)
            process = holder["process"]
            stage = "restore"
            ready = compare._validate_worker_ready(process.ready)
            row["worker_ready"].append(ready)
            stage = "inference"
            for _ in range(request_count):
                events = compare._request_events(process, audio.path.name)
                attempts.append(events)
                row["completed_attempts"] = len(attempts)
                stream.write(
                    canonical_json_bytes(
                        {
                            "simfile_id": row["simfile_id"],
                            "attempt": len(attempts),
                            "worker": worker_number,
                            "worker_ready": ready,
                            "events": [asdict(event) for event in events],
                        },
                        trailing_newline=True,
                    )
                )
                stream.flush()
        except (OSError, RuntimeError, ValueError) as error:
            blocked = isinstance(error, compare.StageLabel) and error.stage == "host"
            _set_failure(row, stage, error, blocked=blocked)
        finally:
            process = holder.pop("process", None)
            if process is not None:
                try:
                    process.close()
                except (OSError, RuntimeError, ValueError) as error:
                    row["close_error"] = str(error)[:1024]
                    if row["status"] != "failed":
                        _set_failure(row, "worker_close", error)
        if row["status"] in {"failed", "blocked"}:
            return

    row["repeatable"] = attempts[0] == attempts[1] == attempts[2]
    if not row["repeatable"]:
        row.update(status="non_repeatable", failed_stage="repeatability")
        return
    try:
        row["metrics"] = compare_events(
            baseline_native_events(baseline), candidate_native_events(attempts[0])
        )
    except (RuntimeError, ValueError) as error:
        _set_failure(row, "metrics", error)
    else:
        row["status"] = "complete"


def run_selected_comparison(
    *,
    selection_path: Path,
    expected_selection_sha256: str,
    baseline_run: Path,
    python311: Path,
    runtime_root: Path,
    checkpoint_cache: Path,
    input_root: Path,
    output_root: Path,
    process_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    execution_id = str(uuid4())
    directory = output_root / "preflight" / execution_id
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "execution_id": execution_id,
        "descriptor": None,
        "descriptor_sha256": None,
        "runtime": None,
        "runtime_sha256": None,
        "status": "unattempted",
        "failed_stage": None,
        "selection_sha256": expected_selection_sha256,
        "baseline": None,
        "smoke": None,
        "provenance": _git_provenance(),
        "measurement_policy": {
            "precision": "six_decimal_places",
            "strict_tolerance_sec": "0.001",
            "frame_shift_tolerance": "512 / 44100 + 0.000001 seconds",
            "offset_sec": 0,
            "matching_class": "native_class_id",
        },
        "items": [
            {
                "simfile_id": identifier,
                "status": "unattempted",
                "failed_stage": None,
                "repeatable": None,
                "completed_attempts": 0,
                "worker_ready": [],
            }
            for identifier in SELECTED_SIMFILE_IDS
        ],
    }
    stage = "selection"
    try:
        selection, bindings = verify_selection(
            selection_path=selection_path,
            expected_selection_sha256=expected_selection_sha256,
            baseline_run=baseline_run,
            input_root=input_root,
        )
        record["baseline"] = {key: value for key, value in selection.items() if key != "items"}
        stage = "runtime"
        config = load_model_config(compare.MODEL_CONFIG_PATH)
        descriptor = build_candidate_descriptor(config)
        record.update(descriptor=dict(descriptor.payload), descriptor_sha256=descriptor.sha256)
        environment = compare._worker_environment()
        probe = collect_probe(python311, worker_environment=environment)
        record["probe"] = probe
        if probe["status"] == "blocked":
            _set_failure(record, stage, "runtime probe is blocked", blocked=True)
            return record

        driver_files = _driver_source_files(REPOSITORY_ROOT)
        runtime = build_runtime_identity(
            REPOSITORY_ROOT, runtime_root, probe, driver_files=driver_files
        )
        runtime_sha = sha256_hex(canonical_json_bytes(runtime))
        record.update(runtime=runtime, runtime_sha256=runtime_sha)
        directory = output_root / descriptor.sha256 / runtime_sha / execution_id
        directory.mkdir(parents=True, exist_ok=False)
        _write_json(directory / "selection.json", selection)

        stage = "smoke"
        factory = WorkerProcess.start if process_factory is None else process_factory

        def start_with_probed_environment(command, **options):
            return factory(command, **{**options, "env": environment})

        smoke = compare.run_comparison(
            python311=python311,
            runtime_root=runtime_root,
            checkpoint_cache=checkpoint_cache,
            fixture=SMOKE_ROOT / "canonical.wav",
            oracle_path=SMOKE_ROOT / "smoke-oracle.json",
            process_factory=start_with_probed_environment,
        )
        compare.write_result(directory / "smoke.json", smoke)
        record["smoke"] = strict_json_loads(json.dumps(smoke).encode())
        smoke_state = smoke["state"]
        if (
            smoke["status"] not in {"smoke_exact", "oracle_divergent"}
            or smoke_state["inference_completed"] is not True
            or smoke_state["repeatable"] is not True
            or smoke_state["failed_stage"] is not None
        ):
            _set_failure(record, stage, smoke["status"], blocked=smoke["status"] == "blocked")
            return record

        stage = "checkpoint"
        checkpoint = Path(compare._prepare_checkpoint(config, checkpoint_cache)["checkpoint"])

        def launch(fixture: Path, holder: dict[str, Any]):
            return compare._start_worker(
                factory,
                python311,
                runtime_root,
                checkpoint,
                fixture,
                holder,
                timeout_seconds=OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
                close_timeout_seconds=OAF_WORKER_CLOSE_TIMEOUT_SECONDS,
                worker_environment=environment,
            )

        stage = "inference"
        with (directory / "native-events.jsonl").open("xb") as stream:
            for row, (audio, baseline) in zip(record["items"], bindings, strict=True):
                _run_song(row, audio, baseline, launch=launch, stream=stream)

        stage = "runtime_recheck"
        after_probe = collect_probe(python311, worker_environment=environment)
        after = build_runtime_identity(
            REPOSITORY_ROOT,
            runtime_root,
            after_probe,
            driver_files=_driver_source_files(REPOSITORY_ROOT),
        )
        after_sha = sha256_hex(canonical_json_bytes(after))
        record["runtime_after_sha256"] = after_sha
        if after_sha != runtime_sha:
            raise ValueError("runtime fingerprint changed during measurement")

        statuses = {row["status"] for row in record["items"]}
        record["status"] = next(
            (status for status in ("failed", "non_repeatable", "blocked") if status in statuses),
            "complete",
        )
        if record["status"] != "complete":
            record["failed_stage"] = "selected_items"
    except SelectionBlocked as error:
        _set_failure(record, stage, error, blocked=True)
    except compare.CheckpointAcquisitionError as error:
        _set_failure(record, stage, error, blocked=str(error) in compare.CHECKPOINT_ACCESS_REASONS)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        _set_failure(record, stage, error)
    finally:
        record["output_dir"] = os.fspath(directory)
        _write_json(directory / "comparison.json", record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="verify and seal the five archived inputs")
    prepare_fresh = subcommands.add_parser(
        "prepare-fresh", help="verify and seal a fresh, scoped September TF1 run"
    )
    prepare.add_argument("--baseline-evidence-dir", required=True, type=Path)
    for flag in (
        "baseline-run",
        "reference-manifest",
        "timing-manifest",
        "cache-dir",
        "input-root",
        "output",
    ):
        prepare.add_argument(f"--{flag}", required=True, type=Path)
        prepare_fresh.add_argument(f"--{flag}", required=True, type=Path)
    run = subcommands.add_parser("run", help="measure a verified, sealed five-song selection")
    for flag in (
        "selection-path",
        "baseline-run",
        "python311",
        "runtime-root",
        "checkpoint-cache",
        "input-root",
        "output-root",
    ):
        run.add_argument(f"--{flag}", required=True, type=Path)
    run.add_argument("--selection-sha256", dest="expected_selection_sha256", required=True)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    arguments = {
        key: value.absolute() if isinstance(value, Path) else value
        for key, value in arguments.items()
    }
    try:
        if command in {"prepare", "prepare-fresh"}:
            output = arguments.pop("output")
            if output.exists() or output.is_symlink():
                raise SelectionError("selection output already exists")
            prepare_inputs = (
                prepare_fresh_selection if command == "prepare-fresh" else prepare_selection
            )
            selection = prepare_inputs(**arguments)
            _write_json(output, selection)
            print(selection_sha256(selection))
            return 0
        result = run_selected_comparison(**arguments)
    except (OSError, SelectionError) as error:
        blocked = isinstance(error, SelectionBlocked)
        print(json.dumps({"status": "blocked" if blocked else "failed", "error": str(error)}))
        return 2 if blocked else 1

    print(json.dumps({"status": result["status"], "output_dir": result["output_dir"]}))
    return {"complete": 0, "blocked": 2}.get(result["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
