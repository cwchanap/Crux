# ruff: noqa: E402 - direct CLI execution needs the repository on sys.path.
"""Prepare verified TF1 inputs and measure bounded TF2.15 prediction drift."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from decimal import Decimal
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, BinaryIO
from uuid import uuid4

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from experiments.oaf_tf215 import compare, panel_tf1
from experiments.oaf_tf215.panel_metrics import aggregate_panel_scores, score_panel_song
from experiments.oaf_tf215.selected_inputs import (
    PANEL_SIMFILE_IDS,
    SELECTED_SIMFILE_IDS,
    SelectionBlocked,
    SelectionError,
    _validate_panel_timings,
    compute_selection_sha256,
    prepare_fresh_selection,
    prepare_panel_selection,
    prepare_selection,
    verify_selection,
)
from experiments.oaf_tf215.selected_metrics import (
    compare_events,
    decode_candidate_native_events,
    extract_baseline_native_events,
)
from runtime.oaf_tf1.model import load_model_config
from src.benchmark.artifact_io import publish_immutable_file, read_regular_file_no_follow
from src.benchmark.backend_identity import (
    SIX_PLACES,
    BackendDescriptor,
    canonical_json_bytes,
    quantize_six,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backends.base import CanonicalAudio
from src.benchmark.oaf_corpus_run import (
    OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
    OAF_WORKER_CLOSE_TIMEOUT_SECONDS,
    parse_oaf_corpus_run,
)
from src.benchmark.oaf_smoke_oracle import SmokeOracleEvent
from src.benchmark.oaf_tf215_runtime import (
    build_candidate_descriptor,
    build_runtime_identity,
    build_tf215_worker_environment,
    collect_probe,
)
from src.benchmark.prediction_artifact import PredictionArtifact
from src.benchmark.reference_set import ReferenceMappingResult
from src.benchmark.reference_set_manifest import (
    load_reference_set_manifest,
    preflight_reference_mappings,
)
from src.benchmark.reference_timing_manifest import load_reference_timing_manifest
from src.benchmark.worker_process import WorkerProcess

SCHEMA = "crux.oaf-tf215-selected-comparison/v1"
PANEL_SCHEMA = "crux.oaf-tf215-panel-comparison/v1"
PANEL_ANCHORS = (194, 384)
SMOKE_ROOT = panel_tf1.SMOKE_ROOT
EXPERIMENT_MODULES = (
    "build_runtime.py",
    "compare.py",
    "selected_compare.py",
    "selected_inputs.py",
    "selected_metrics.py",
)


def _driver_source_files(root: Path, *, panel: bool = False) -> tuple[Path, ...]:
    experiment = Path("experiments/oaf_tf215")
    benchmark = root / "src/benchmark"
    return (
        *(experiment / filename for filename in EXPERIMENT_MODULES),
        *(
            (experiment / "panel_metrics.py", experiment / "panel_tf1.py", experiment / "PANEL.md")
            if panel
            else ()
        ),
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


def _write_attempt(
    stream: BinaryIO,
    row: dict[str, Any],
    worker: int,
    ready: dict[str, Any],
    events: tuple[SmokeOracleEvent, ...],
) -> None:
    row["completed_attempts"] += 1
    stream.write(
        canonical_json_bytes(
            {
                "simfile_id": row["simfile_id"],
                "attempt": row["completed_attempts"],
                "worker": worker,
                "worker_ready": ready,
                "events": [asdict(event) for event in events],
            },
            trailing_newline=True,
        )
    )
    stream.flush()


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
                _write_attempt(stream, row, worker_number, ready, events)
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
            extract_baseline_native_events(baseline), decode_candidate_native_events(attempts[0])
        )
    except (RuntimeError, ValueError) as error:
        _set_failure(row, "metrics", error)
    else:
        row["status"] = "complete"


def _panel_references(
    manifests: tuple[Path, Path], baseline: dict[str, Any]
) -> dict[int, ReferenceMappingResult]:
    reference_path, timing_path = manifests
    reference = load_reference_set_manifest(reference_path)
    timing = load_reference_timing_manifest(timing_path)
    if (
        reference.manifest_sha256 != baseline["reference_manifest_sha256"]
        or timing.manifest_sha256 != baseline["reference_timing_manifest_sha256"]
    ):
        raise SelectionError("panel scoring manifests do not match the seal")
    scoped_rows = tuple(row for row in reference.rows if row.view.simfile_id in PANEL_SIMFILE_IDS)
    mappings = preflight_reference_mappings(
        replace(reference, rows=scoped_rows),
        timing,
        timing_output_root=timing_path.parent.parent,
    )
    if set(mappings) != set(PANEL_SIMFILE_IDS) or any(
        not isinstance(mapping, ReferenceMappingResult) for mapping in mappings.values()
    ):
        raise SelectionError("panel reference mapping is incomplete")
    return {song: mapping for song, mapping in mappings.items() if mapping is not None}


def _stage_wavs(
    bindings: tuple[tuple[CanonicalAudio, PredictionArtifact], ...], directory: Path
) -> tuple[tuple[CanonicalAudio, PredictionArtifact], ...]:
    staged = []
    for audio, baseline in bindings:
        destination = directory / audio.path.name
        with audio.path.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        with destination.open("rb") as target:
            if file_digest(target, "sha256").hexdigest() != audio.input_audio_sha256:
                raise SelectionError(f"canonical WAV changed after selection: {audio.path}")
        staged.append((replace(audio, path=destination), baseline))
    return tuple(staged)


def _run_panel(
    record: dict[str, Any],
    bindings: tuple[tuple[CanonicalAudio, PredictionArtifact], ...],
    references: dict[int, ReferenceMappingResult],
    baseline_wall_times: dict[int, int | Decimal],
    descriptor: BackendDescriptor,
    *,
    launch: Callable[[Path, dict[str, Any]], Any],
    stream: BinaryIO,
) -> None:
    scheduled = {
        row["simfile_id"]: (row, audio, artifact)
        for row, (audio, artifact) in zip(record["items"], bindings, strict=True)
    }
    primary: dict[int, tuple[SmokeOracleEvent, ...]] = {}
    replay: dict[int, list[tuple[SmokeOracleEvent, ...]]] = {song: [] for song in PANEL_ANCHORS}
    started_workers = []
    record["panel_workers"] = started_workers

    for worker, songs in ((1, sorted(scheduled)), (2, PANEL_ANCHORS)):
        holder: dict[str, Any] = {}
        started = perf_counter()
        worker_stage = "worker_start"
        try:
            launch(scheduled[songs[0]][1].path, holder)
            process = holder["process"]
            worker_stage = "restore"
            ready = compare._validate_worker_ready(process.ready)
            started_workers.append(
                {
                    "worker": worker,
                    "startup_wall_time_sec": quantize_six(perf_counter() - started),
                    **ready,
                }
            )
            worker_stage = "inference"
            requests = (*songs, *PANEL_ANCHORS) if worker == 1 else songs
            for song in requests:
                row, audio, artifact = scheduled[song]
                # Preserve a failed primary row and replay only successful anchors.
                if row["status"] == "failed" or (worker == 2 and song not in primary):
                    continue
                stage = "inference"
                try:
                    started = perf_counter()
                    events = compare._request_events(process, audio.path.name)
                    elapsed = perf_counter() - started
                    _write_attempt(stream, row, worker, ready, events)
                    if song in primary:
                        replay[song].append(events)
                        continue
                    stage = "metrics"
                    row["metrics"] = compare_events(
                        extract_baseline_native_events(artifact),
                        decode_candidate_native_events(events),
                    )
                    row["reference_scores"] = score_panel_song(
                        artifact, events, references[song], descriptor
                    )
                    row["duration_sec"] = quantize_six(audio.audio_frame_count / audio.sample_rate)
                    row["candidate_wall_time_sec"] = quantize_six(elapsed)
                    row["baseline_wall_time_sec"] = baseline_wall_times[song]
                    primary[song] = events
                    row["status"] = "complete"
                except (OSError, RuntimeError, ValueError) as error:
                    _set_failure(row, stage, error)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            _set_failure(record, worker_stage, error)
        finally:
            process = holder.get("process")
            if process is not None:
                try:
                    process.close()
                except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                    if record["status"] == "failed":
                        record["close_error"] = str(error)[:1024]
                    else:
                        _set_failure(record, "worker_close", error)
        if record["status"] == "failed":
            record["failed_worker"] = worker
            break

    for song, attempts in replay.items():
        row, _, _ = scheduled[song]
        if row["status"] != "complete":
            continue
        if any(events != primary[song] for events in attempts):
            row["repeatable"] = False
            _set_failure(row, "repeatability", "panel anchor events changed")
            row["status"] = "non_repeatable"
        elif len(attempts) == 2:
            row["repeatable"] = True
    record["panel_scores"] = aggregate_panel_scores(
        row["reference_scores"] for row in record["items"] if "reference_scores" in row
    )
    _panel_runtime_times(record)


def _panel_runtime_times(record: dict[str, Any]) -> None:
    measured = sorted(
        (row for row in record["items"] if "candidate_wall_time_sec" in row),
        key=lambda row: row["simfile_id"],
    )
    if not measured:
        return
    with_startup = measured[0]["simfile_id"] == min(PANEL_SIMFILE_IDS)
    totals = {}
    for label, rows in (
        ("all_matched", measured),
        (
            "after_first_scheduled_song",
            [row for row in measured if row["simfile_id"] != min(PANEL_SIMFILE_IDS)],
        ),
    ):
        if not rows:
            continue
        duration = sum(row["duration_sec"] for row in rows)
        baseline = sum(row["baseline_wall_time_sec"] for row in rows)
        candidate = sum(row["candidate_wall_time_sec"] for row in rows)
        totals[label] = {
            "songs": len(rows),
            "duration_sec": duration,
            "baseline_rtf": (baseline / duration).quantize(SIX_PLACES),
            "candidate_request_rtf": (candidate / duration).quantize(SIX_PLACES),
        }
    totals["baseline_total_includes_startup"] = with_startup
    if with_startup:
        totals["candidate_total_rtf_with_startup"] = (
            (
                sum(row["candidate_wall_time_sec"] for row in measured)
                + record["panel_workers"][0]["startup_wall_time_sec"]
            )
            / sum(row["duration_sec"] for row in measured)
        ).quantize(SIX_PLACES)
    record["panel_runtime"] = totals


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
    panel_manifests: tuple[Path, Path] | None = None,
    _tf1_measurement: panel_tf1.Tf1PanelMeasurement | None = None,
) -> dict[str, Any]:
    panel = panel_manifests is not None
    execution_id = str(uuid4())
    directory = output_root / "preflight" / execution_id
    record: dict[str, Any] = {
        "schema": PANEL_SCHEMA if panel else SCHEMA,
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
            for identifier in (PANEL_SIMFILE_IDS if panel else SELECTED_SIMFILE_IDS)
        ],
    }
    if panel:
        record["panel_gate_passed"] = False
        record["tf1_smoke"] = None
    stage = "selection"
    try:
        selection, bindings = verify_selection(
            selection_path=selection_path,
            expected_selection_sha256=expected_selection_sha256,
            baseline_run=baseline_run,
            input_root=input_root,
            panel=panel,
        )
        record["baseline"] = {key: value for key, value in selection.items() if key != "items"}
        if panel_manifests is not None:
            if _tf1_measurement is not None:
                if (
                    selection.get("tf1_measurement") != _tf1_measurement.binding()
                    or selection.get("baseline_run_sha256") != _tf1_measurement.run_sha256
                    or _tf1_measurement.launcher_sha256 != panel_tf1.TF1_LAUNCHER_SHA256
                ):
                    raise SelectionError("panel seal does not bind the measured TF1 run")
                record["tf1_smoke"] = {
                    "status": "exact",
                    "boot_id": _tf1_measurement.boot_id,
                    "prediction_sha256": _tf1_measurement.smoke_prediction_sha256,
                    "oracle_sha256": _tf1_measurement.smoke_oracle_sha256,
                }
            run = read_regular_file_no_follow(baseline_run)
            if sha256_hex(run) != selection["baseline_run_sha256"]:
                raise SelectionError("baseline run changed after the seal was checked")
            baseline_rows = parse_oaf_corpus_run(
                run, expected_run_id=selection["baseline_identity"]["run_id"]
            )["items"]
            baseline_wall_times = _validate_panel_timings(baseline_rows)
            stage = "references"
            references = _panel_references(panel_manifests, selection["baseline_identity"])
        stage = "runtime"
        config = load_model_config(compare.MODEL_CONFIG_PATH)
        descriptor = build_candidate_descriptor(config)
        record.update(descriptor=dict(descriptor.payload), descriptor_sha256=descriptor.sha256)
        environment = build_tf215_worker_environment()
        probe = collect_probe(python311, worker_environment=environment)
        record["probe"] = probe
        if probe["status"] == "blocked":
            _set_failure(record, stage, "runtime probe is blocked", blocked=True)
            return record

        driver_files = _driver_source_files(REPOSITORY_ROOT, panel=panel)
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
                host_unavailable_is_blocked=False,
            )

        stage = "sealed_wavs"
        with TemporaryDirectory(prefix="sealed-wavs-", dir=directory) as staging:
            snapshots = _stage_wavs(bindings, Path(staging))
            stage = "inference"
            with (directory / "native-events.jsonl").open("xb") as stream:
                if panel_manifests is not None:
                    _run_panel(
                        record,
                        snapshots,
                        references,
                        baseline_wall_times,
                        descriptor,
                        launch=launch,
                        stream=stream,
                    )
                else:
                    for row, (audio, baseline) in zip(record["items"], snapshots, strict=True):
                        _run_song(row, audio, baseline, launch=launch, stream=stream)

        if record["status"] == "failed":
            return record

        stage = "runtime_recheck"
        after_probe = collect_probe(python311, worker_environment=environment)
        after = build_runtime_identity(
            REPOSITORY_ROOT,
            runtime_root,
            after_probe,
            driver_files=_driver_source_files(REPOSITORY_ROOT, panel=panel),
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
        if panel and _tf1_measurement is not None:
            stage = "measurement_boot"
            record["candidate_boot_id"] = panel_tf1.read_boot_id()
            record["panel_gate_passed"] = (
                record["status"] == "complete"
                and record["candidate_boot_id"] == _tf1_measurement.boot_id
                and record["panel_scores"]["micro_f1_nonregression"] is True
            )
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


def measure_panel(
    *,
    launcher_path: Path,
    reference_manifest: Path,
    timing_manifest: Path,
    cache_dir: Path,
    python311: Path,
    runtime_root: Path,
    checkpoint_cache: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Measure the TF1 and TF2.15 panels on one boot and evaluate the gate."""
    directory = output_root / str(uuid4())
    directory.mkdir(parents=True, exist_ok=False)
    baseline = panel_tf1.measure_tf1_panel(
        launcher_path=launcher_path,
        reference_manifest=reference_manifest,
        timing_manifest=timing_manifest,
        cache_dir=cache_dir,
        checkpoint_cache=checkpoint_cache,
        directory=directory,
    )
    input_root = directory / "sealed-wavs"
    selection = prepare_panel_selection(
        baseline_run=baseline.run_path,
        reference_manifest=reference_manifest,
        timing_manifest=timing_manifest,
        cache_dir=cache_dir,
        input_root=input_root,
    )
    if selection["baseline_run_sha256"] != baseline.run_sha256:
        raise SelectionError("TF1 run changed before the panel was sealed")
    selection["tf1_measurement"] = baseline.binding()
    selection_path = directory / "selection.json"
    _write_json(selection_path, selection)
    return run_selected_comparison(
        selection_path=selection_path,
        expected_selection_sha256=compute_selection_sha256(selection),
        baseline_run=baseline.run_path,
        python311=python311,
        runtime_root=runtime_root,
        checkpoint_cache=checkpoint_cache,
        input_root=input_root,
        output_root=directory / "candidate",
        panel_manifests=(reference_manifest, timing_manifest),
        _tf1_measurement=baseline,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="verify and seal the five archived inputs")
    prepare_fresh = subcommands.add_parser(
        "prepare-fresh", help="verify and seal a fresh, scoped September TF1 run"
    )
    prepare_panel = subcommands.add_parser(
        "prepare-panel", help="verify and seal the predeclared twenty-song TF1 panel"
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
        for parser_ in (prepare, prepare_fresh, prepare_panel):
            parser_.add_argument(f"--{flag}", required=True, type=Path)
    run = subcommands.add_parser("run", help="measure a verified, sealed five-song selection")
    run_panel = subcommands.add_parser("run-panel", help="compare against a sealed TF1 panel")
    paired = subcommands.add_parser(
        "measure-panel", help="measure both backends on one VM and evaluate the panel gate"
    )
    for parser_ in (run_panel, paired):
        parser_.add_argument("--reference-manifest", required=True, type=Path)
        parser_.add_argument("--timing-manifest", required=True, type=Path)
    paired.add_argument("--cache-dir", required=True, type=Path)
    paired.add_argument("--tf1-rootfs-launcher", dest="launcher_path", required=True, type=Path)
    for flag in ("python311", "runtime-root", "checkpoint-cache", "output-root"):
        for parser_ in (run, run_panel, paired):
            parser_.add_argument(f"--{flag}", required=True, type=Path)
    for flag in ("selection-path", "baseline-run", "input-root"):
        for parser_ in (run, run_panel):
            parser_.add_argument(f"--{flag}", required=True, type=Path)
    for parser_ in (run, run_panel):
        parser_.add_argument("--selection-sha256", dest="expected_selection_sha256", required=True)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    arguments = {
        key: value.absolute() if isinstance(value, Path) else value
        for key, value in arguments.items()
    }
    try:
        preparations = {
            "prepare": prepare_selection,
            "prepare-fresh": prepare_fresh_selection,
            "prepare-panel": prepare_panel_selection,
        }
        if command in preparations:
            output = arguments.pop("output")
            if output.exists() or output.is_symlink():
                raise SelectionError("selection output already exists")
            selection = preparations[command](**arguments)
            input_root = arguments["input_root"]
            published = input_root.lstat()
            try:
                _write_json(output, selection)
            except (OSError, ValueError) as error:
                try:
                    if os.path.samestat(input_root.lstat(), published):
                        shutil.rmtree(input_root)
                except FileNotFoundError:
                    pass
                except OSError as cleanup_error:
                    error.add_note(f"WAV directory cleanup failed: {cleanup_error}")
                raise
            print(compute_selection_sha256(selection))
            return 0
        if command == "measure-panel":
            result = measure_panel(**arguments)
        else:
            if command == "run-panel":
                arguments["panel_manifests"] = (
                    arguments.pop("reference_manifest"),
                    arguments.pop("timing_manifest"),
                )
            result = run_selected_comparison(**arguments)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        blocked = isinstance(error, SelectionBlocked)
        print(json.dumps({"status": "blocked" if blocked else "failed", "error": str(error)}))
        return 2 if blocked else 1

    result_fields = {"status", "output_dir", "panel_gate_passed"}
    print(json.dumps({key: value for key, value in result.items() if key in result_fields}))
    return {"complete": 0, "blocked": 2}.get(result["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
