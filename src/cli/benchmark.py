from __future__ import annotations

# Commands keep optional and heavy implementation modules behind their Click boundary.
# pylint: disable=import-outside-toplevel
import json
import time
from dataclasses import asdict
from pathlib import Path

import click

from runtime.oaf_tf1.model import load_model_config
from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import canonical_json_bytes, quantize_six
from src.benchmark.backend_registry import default_backend_registry
from src.benchmark.backends.oaf import RESTORED_TENSOR_COUNT
from src.benchmark.input_view import load_direct_audio
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.oaf_smoke_oracle import (
    assert_smoke_oracle_matches,
    read_smoke_oracle,
)
from src.benchmark.prediction_artifact import (
    publish_prediction_artifact,
    read_prediction_artifact,
)
from src.benchmark.r2_corpus_models import MAX_SIMFILE_ID, SyncOutcome, SyncRequest
from src.benchmark.r2_corpus_sync import ProgressEvent, sync_r2_corpus
from src.cli.options import (
    charts_dir_option,
    output_dir_option,
    predictions_dir_option,
    raw_dir_option,
    resolve_benchmark_output_dir,
    run_name_option,
    song_dir_option,
)


@click.group()
def benchmark() -> None:
    """Benchmark drum transcription against DTX ground truth."""


@benchmark.command("prepare-backend")
@click.option("--backend", type=str, default="oaf", show_default=True)
@click.option("--download", is_flag=True)
@click.option(
    "--archive",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--cache-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/model-cache"),
    show_default=True,
)
@click.pass_context
def prepare_backend_command(
    ctx: click.Context,
    backend: str,
    download: bool,
    archive: Path | None,
    cache_root: Path,
) -> None:
    """Acquire or verify the immutable OaF checkpoint cache."""
    if download and archive is not None:
        raise click.UsageError("--download and --archive are mutually exclusive.")

    if backend != "oaf":
        click.echo("backend_selection_invalid", err=True)
        _emit_prepare_backend_summary(
            backend=backend, checkpoint_path=None, status="integrity_failed"
        )
        ctx.exit(2)

    from runtime.oaf_tf1.model import OafModelConfigError, load_model_config
    from src.benchmark.checkpoint_acquisition import (
        CheckpointAcquisitionError,
        prepare_oaf_checkpoint,
    )

    try:
        checkpoint_path = prepare_oaf_checkpoint(
            load_model_config(),
            cache_root,
            download=download,
            archive_path=archive,
        )
    except OafModelConfigError:
        _emit_prepare_backend_summary(
            backend=backend,
            checkpoint_path=None,
            status="integrity_failed",
        )
        ctx.exit(2)
    except CheckpointAcquisitionError:
        _emit_prepare_backend_summary(
            backend=backend,
            checkpoint_path=None,
            status="acquisition_failed",
        )
        ctx.exit(1)

    _emit_prepare_backend_summary(
        backend=backend,
        checkpoint_path=checkpoint_path,
        status="ready",
    )


def _emit_prepare_backend_summary(
    *,
    backend: str,
    checkpoint_path: Path | None,
    status: str,
) -> None:
    payload = {
        "backend": backend,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "status": status,
    }
    standard_output = click.get_binary_stream("stdout")
    standard_output.write(canonical_json_bytes(payload, trailing_newline=True))
    standard_output.flush()


def _validate_verification_backend_id(
    _ctx: click.Context,
    _parameter: click.Parameter,
    value: str,
) -> str:
    supported = {
        "oaf",
    }
    if value not in supported:
        raise click.BadParameter("must select a supported verification backend")
    return value


@benchmark.command("verify-backend")
@click.option(
    "--backend",
    type=str,
    default="oaf",
    show_default=True,
    callback=_validate_verification_backend_id,
)
@click.option(
    "--reports-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/backends"),
    show_default=True,
)
@click.option("--allow-emulated-diagnostics", is_flag=True)
@click.pass_context
def verify_backend_command(
    ctx: click.Context,
    backend: str,
    reports_root: Path,
    allow_emulated_diagnostics: bool,
) -> None:
    del ctx, backend, reports_root, allow_emulated_diagnostics
    raise click.ClickException("verify-backend was removed; use the Task E smoke command")


# Click owns this fixed external signature, and backend imports must remain lazy.
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals
@benchmark.command("transcribe-one")
@click.option("--backend", type=str, default=None)
@click.option(
    "--audio",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
)
@click.option("--source-audio-id", type=str, default=None)
@click.option("--input-view-id", type=str, default=None)
@click.option(
    "--input-view-manifest",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
)
@click.option(
    "--midi-output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--reports-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/backends"),
    show_default=True,
)
@click.pass_context
def transcribe_one(
    ctx: click.Context,
    backend: str | None,
    audio: Path,
    source_audio_id: str | None,
    input_view_id: str | None,
    input_view_manifest: Path | None,
    output: Path,
    midi_output: Path | None,
    reports_root: Path,
) -> None:
    del (
        ctx,
        backend,
        audio,
        source_audio_id,
        input_view_id,
        input_view_manifest,
        output,
        midi_output,
        reports_root,
    )
    raise click.ClickException("transcribe-one was removed; use the Task E smoke command")


# pylint: enable=too-many-arguments,too-many-positional-arguments
# pylint: enable=too-many-locals


@benchmark.command("smoke-backend")
@click.option(
    "--backend",
    type=str,
    default="oaf",
    show_default=True,
    callback=_validate_verification_backend_id,
)
@click.option(
    "--oracle",
    type=click.Path(exists=True, path_type=Path, dir_okay=False),
    default=None,
    help="Optional exact native-event oracle JSON path.",
)
def smoke_backend(backend: str, oracle: Path | None) -> None:
    """Run one OaF inference against the checked-in canonical smoke fixture."""
    fixture_path = Path("tests/fixtures/oaf_tf1_smoke/canonical.wav").resolve()
    prediction_path = Path("artifacts/benchmark/oaf-smoke/prediction.jsonl")
    config = load_model_config()
    checkpoint_dir = (
        Path("artifacts/benchmark/model-cache").resolve()
        / "sha256"
        / config.checkpoint.archive_sha256
    )
    try:
        audio = load_direct_audio(
            fixture_path,
            source_audio_id="oaf-smoke-canonical-v1",
            input_view_id="oaf-smoke-canonical-v1",
            max_input_audio_frames=config.max_input_audio_frames,
        )
        backend_instance = default_backend_registry().create(
            backend,
            checkpoint_dir=checkpoint_dir,
            input_root=fixture_path.parent,
            image="crux-oaf-tf1:local",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    try:
        started = time.perf_counter()
        native_prediction = backend_instance.transcribe(audio)
        inference_elapsed_seconds = time.perf_counter() - started

        mapped_prediction, diagnostics = map_oaf_prediction(native_prediction)
        mapped_event_count = sum(
            event.mapping_status == "mapped" for event in mapped_prediction.events
        )
        unmapped_event_count = sum(diagnostics.unmapped.values())
        if mapped_event_count == 0:
            raise click.ClickException("smoke inference produced no mapped events")
        published = publish_prediction_artifact(prediction_path, mapped_prediction)

        oracle_status = "not_checked"
        if oracle is not None:
            try:
                oracle_content = read_regular_file_no_follow(oracle)
                canonical_prediction = read_prediction_artifact(
                    read_regular_file_no_follow(prediction_path)
                ).prediction
                assert_smoke_oracle_matches(
                    canonical_prediction,
                    read_smoke_oracle(oracle_content),
                )
            except (AssertionError, OSError, TypeError, ValueError) as error:
                raise click.ClickException(str(error)) from error
            oracle_status = "matched"

        duration_seconds = audio.audio_frame_count / audio.sample_rate
        summary = {
            "backend_id": native_prediction.descriptor.payload["backend_id"],
            "checkpoint_archive_sha256": config.checkpoint.archive_sha256,
            "fixture_sha256": audio.input_audio_sha256,
            "inference_elapsed_seconds": quantize_six(inference_elapsed_seconds),
            "mapped_event_count": mapped_event_count,
            "oracle_status": oracle_status,
            "prediction_path": str(prediction_path),
            "prediction_sha256": published.sha256,
            "real_time_factor": quantize_six(inference_elapsed_seconds / duration_seconds),
            "restored_tensor_count": RESTORED_TENSOR_COUNT,
            "status": "ok",
            "unmapped_event_count": unmapped_event_count,
            "upstream_source_commit": native_prediction.descriptor.payload[
                "upstream_source_commit"
            ],
        }
        standard_output = click.get_binary_stream("stdout")
        standard_output.write(canonical_json_bytes(summary, trailing_newline=True))
        standard_output.flush()
    finally:
        backend_instance.close()


def _emit_progress(event: ProgressEvent) -> None:
    click.echo(event.message, err=True)


class _ProgressEmitter:
    def __init__(self) -> None:
        self.emitted_failed_final = False

    def __call__(self, event: ProgressEvent) -> None:
        if (
            event.phase == "failed"
            and event.completed == 1
            and event.total == 1
            and event.message == "failed synchronization outcome."
        ):
            self.emitted_failed_final = True
        _emit_progress(event)


def _emit_sync_summary(outcome: SyncOutcome) -> None:
    click.echo(
        json.dumps(
            {
                "corpus_version": (
                    outcome.manifest.corpus_version if outcome.manifest is not None else None
                ),
                "counts": asdict(outcome.counters),
                "exit_code": outcome.exit_code,
                "manifest_published": outcome.manifest is not None,
                "report_path": (
                    str(outcome.report_path) if outcome.report_path is not None else None
                ),
                "status": outcome.overall_status,
            },
            sort_keys=True,
        )
    )


def _emit_reference_chart_selection_summary(
    *,
    status: str,
    exit_code: int,
    manifest_path: Path | None,
    manifest_sha256: str | None,
    corpus_version: str | None,
    selected_count: int,
    quarantined_count: int,
) -> None:
    """Write the sole machine-readable result after reference-chart selection."""
    payload = {
        "corpus_version": corpus_version,
        "exit_code": exit_code,
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "quarantined_count": quarantined_count,
        "selected_count": selected_count,
        "status": status,
    }
    standard_output = click.get_binary_stream("stdout")
    standard_output.write(canonical_json_bytes(payload, trailing_newline=True))
    standard_output.flush()


def _emit_reference_timing_summary(
    *,
    status: str,
    exit_code: int,
    manifest_path: Path | None,
    manifest_sha256: str | None,
    corpus_version: str | None,
    ready_count: int,
    quarantined_count: int,
    upstream_quarantined_count: int,
    events_published: int,
) -> None:
    """Write the sole machine-readable result after a reference-timing build.

    Surfaces ``upstream_quarantined_count`` so an operator can distinguish
    timing-stage quarantines from HPA-322 upstream gaps even though both share
    the exit-``1`` partial outcome.
    """
    payload = {
        "corpus_version": corpus_version,
        "events_published": events_published,
        "exit_code": exit_code,
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "quarantined_count": quarantined_count,
        "ready_count": ready_count,
        "status": status,
        "upstream_quarantined_count": upstream_quarantined_count,
    }
    standard_output = click.get_binary_stream("stdout")
    standard_output.write(canonical_json_bytes(payload, trailing_newline=True))
    standard_output.flush()


def _emit_reference_set_summary(
    *,
    status: str,
    exit_code: int,
    manifest_path: Path | None,
    manifest_sha256: str | None,
    corpus_version: str | None,
    eligible_count: int,
    quarantined_count: int,
) -> None:
    """Write the sole machine-readable result after reference-set publication."""
    payload = {
        "corpus_version": corpus_version,
        "eligible_count": eligible_count,
        "exit_code": exit_code,
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "quarantined_count": quarantined_count,
        "status": status,
    }
    standard_output = click.get_binary_stream("stdout")
    standard_output.write(canonical_json_bytes(payload, trailing_newline=True))
    standard_output.flush()


@benchmark.command("select-reference-charts")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Required path to an immutable local HPA-321 source manifest (JSONL).",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Local HPA-321 cache root (default: <manifest parent>/../cache/).",
)
@click.option(
    "--overrides-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("config/benchmark-reference-chart-overrides.json"),
    show_default=True,
    help="Reference-chart override document mapping simfile IDs to authoritative charts.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/reference-charts"),
    show_default=True,
    help="Directory where the published reference-chart manifest and artifacts are written.",
)
@click.pass_context
def select_reference_charts_command(
    ctx: click.Context,
    manifest_path: Path,
    cache_dir: Path | None,
    overrides_file: Path,
    output_dir: Path,
) -> None:
    """Select authoritative charts from an immutable local HPA-321 manifest."""
    from src.benchmark.reference_chart_manifest import SelectionRequest, select_reference_manifest

    resolved_manifest_path = manifest_path.absolute()
    outcome = select_reference_manifest(
        SelectionRequest(
            manifest_path=resolved_manifest_path,
            cache_dir=(
                resolved_manifest_path.parent.parent / "cache" if cache_dir is None else cache_dir
            ),
            overrides_file=overrides_file,
            output_dir=output_dir,
            default_overrides_missing_ok=(
                ctx.get_parameter_source("overrides_file") is click.core.ParameterSource.DEFAULT
            ),
        )
    )
    published = outcome.manifest
    _emit_reference_chart_selection_summary(
        status=outcome.status,
        exit_code=outcome.exit_code,
        manifest_path=None if published is None else published.path,
        manifest_sha256=None if published is None else published.manifest_sha256,
        corpus_version=None if published is None else published.corpus_version,
        selected_count=outcome.selected_count,
        quarantined_count=outcome.quarantined_count,
    )
    if outcome.exit_code:
        ctx.exit(outcome.exit_code)


@benchmark.command("build-reference-timing")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Required path to an immutable local HPA-322 reference-chart manifest (JSONL).",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Required local cache root holding the verified chart and audio bodies.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/reference-timing"),
    show_default=True,
    help="Directory where the published reference-timing manifest and events are written.",
)
@click.pass_context
def build_reference_timing_command(
    ctx: click.Context,
    manifest_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Build and publish the reference-timing manifest and bounded events."""
    from src.benchmark.reference_timing_manifest import (
        ReferenceTimingRequest,
        run_reference_timing,
    )

    outcome = run_reference_timing(
        ReferenceTimingRequest(
            manifest_path=manifest_path.absolute(),
            cache_dir=cache_dir,
            output_dir=output_dir,
        )
    )
    published = outcome.manifest
    _emit_reference_timing_summary(
        status=outcome.status,
        exit_code=outcome.exit_code,
        manifest_path=None if published is None else published.path,
        manifest_sha256=None if published is None else published.manifest_sha256,
        corpus_version=None if published is None else published.corpus_version,
        ready_count=outcome.ready_count,
        quarantined_count=outcome.quarantined_count,
        upstream_quarantined_count=outcome.upstream_quarantined_count,
        events_published=outcome.events_published,
    )
    if outcome.exit_code:
        ctx.exit(outcome.exit_code)


@benchmark.command("build-reference-set")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Required path to an immutable HPA-323 reference-timing manifest (JSONL).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/reference-set"),
    show_default=True,
    help="Publication directory.",
)
@click.pass_context
def build_reference_set_command(
    ctx: click.Context,
    manifest_path: Path,
    output_dir: Path,
) -> None:
    """Build and publish model-independent reference eligibility."""
    from src.benchmark.reference_set_manifest import ReferenceSetRequest, run_reference_set

    outcome = run_reference_set(
        ReferenceSetRequest(
            manifest_path=manifest_path,
            output_dir=output_dir,
        )
    )
    published = outcome.manifest
    _emit_reference_set_summary(
        status=outcome.status,
        exit_code=outcome.exit_code,
        manifest_path=None if published is None else published.path,
        manifest_sha256=None if published is None else published.manifest_sha256,
        corpus_version=None if published is None else published.corpus_version,
        eligible_count=outcome.eligible_count,
        quarantined_count=outcome.quarantined_count,
    )
    if outcome.exit_code:
        ctx.exit(outcome.exit_code)


@benchmark.command("run-oaf-corpus")
@click.option(
    "--manifest",
    "reference_manifest_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Required path to an immutable local HPA-324 reference-set manifest (JSONL).",
)
@click.option(
    "--timing-manifest",
    "timing_manifest_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Required path to the source HPA-323 reference-timing manifest (JSONL).",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Required local HPA-321 corpus/audio cache root.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Directory where the OaF corpus run and reports are written.",
)
@click.option(
    "--include-simfile-id",
    "include_simfile_ids",
    type=click.IntRange(0, MAX_SIMFILE_ID),
    multiple=True,
)
@click.option(
    "--exclude-simfile-id",
    "exclude_simfile_ids",
    type=click.IntRange(0, MAX_SIMFILE_ID),
    multiple=True,
)
@click.option("--resume", is_flag=True)
@click.pass_context
def run_oaf_corpus_command(
    ctx: click.Context,
    reference_manifest_path: Path,
    timing_manifest_path: Path,
    cache_dir: Path,
    output_dir: Path,
    include_simfile_ids: tuple[int, ...],
    exclude_simfile_ids: tuple[int, ...],
    resume: bool,
) -> None:
    """Run the validated OaF backend over an HPA-324 corpus scope."""
    from src.benchmark.oaf_corpus_run import OafCorpusRunRequest, run_oaf_corpus

    outcome = run_oaf_corpus(
        OafCorpusRunRequest(
            reference_manifest_path=reference_manifest_path,
            timing_manifest_path=timing_manifest_path,
            cache_dir=cache_dir,
            output_dir=output_dir,
            include_simfile_ids=include_simfile_ids,
            exclude_simfile_ids=exclude_simfile_ids,
            resume=resume,
        )
    )
    payload = {
        "aggregate_rtf": (
            None if outcome.aggregate_rtf is None else quantize_six(outcome.aggregate_rtf)
        ),
        "exit_code": outcome.exit_code,
        "failed_count": outcome.failed_count,
        "projected_full_wall_time_sec": (
            None
            if outcome.projected_full_wall_time_sec is None
            else quantize_six(outcome.projected_full_wall_time_sec)
        ),
        "quarantined_count": outcome.quarantined_count,
        "reports_path": None if outcome.reports_path is None else str(outcome.reports_path),
        "run_id": outcome.run_id,
        "run_path": None if outcome.run_path is None else str(outcome.run_path),
        "skipped_count": outcome.skipped_count,
        "status": outcome.overall_status,
        "success_count": outcome.success_count,
    }
    standard_output = click.get_binary_stream("stdout")
    standard_output.write(canonical_json_bytes(payload, trailing_newline=True))
    standard_output.flush()
    if outcome.exit_code:
        ctx.exit(outcome.exit_code)


@benchmark.command("sync-r2-corpus")
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Local cache root (default: <output-dir>/cache/).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/r2-corpus"),
    show_default=True,
)
@click.option(
    "--include-simfile-id",
    "include_simfile_ids",
    type=click.IntRange(0, MAX_SIMFILE_ID),
    multiple=True,
)
@click.option(
    "--exclude-simfile-id",
    "exclude_simfile_ids",
    type=click.IntRange(0, MAX_SIMFILE_ID),
    multiple=True,
)
@click.option(
    "--provenance-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option("--dry-run", is_flag=True)
@click.pass_context
def sync_r2_corpus_command(
    ctx: click.Context,
    cache_dir: Path | None,
    output_dir: Path,
    include_simfile_ids: tuple[int, ...],
    exclude_simfile_ids: tuple[int, ...],
    provenance_file: Path | None,
    dry_run: bool,
) -> None:
    """Inventory selected R2 corpus metadata and cache chart definition files."""
    request = SyncRequest(
        cache_dir=output_dir / "cache" if cache_dir is None else cache_dir,
        output_dir=output_dir,
        include_simfile_ids=frozenset(include_simfile_ids),
        exclude_simfile_ids=frozenset(exclude_simfile_ids),
        provenance_file=provenance_file,
        dry_run=dry_run,
    )
    progress = _ProgressEmitter()
    outcome = sync_r2_corpus(request, progress=progress)
    if (
        outcome.overall_status == "failed"
        and outcome.report_path is None
        and not progress.emitted_failed_final
    ):
        click.echo("R2 synchronization failed before a report could be written.", err=True)
    _emit_sync_summary(outcome)
    if outcome.exit_code:
        ctx.exit(outcome.exit_code)


@benchmark.command("prepare-corpus")
@raw_dir_option
@run_name_option
@output_dir_option
def prepare_benchmark_corpus(raw_dir: Path, run_name: str | None, output_dir: Path | None) -> None:
    """Convert raw song folders into the parsed benchmark corpus format."""
    from src.benchmark.prepare import prepare_corpus

    result = prepare_corpus(raw_dir, resolve_benchmark_output_dir(output_dir, run_name, raw_dir))
    click.echo(
        f"Prepared {len(result.valid_items)} benchmark item(s); "
        f"skipped {len(result.invalid_items)} invalid folder(s)"
    )


@benchmark.command("validate-corpus")
@charts_dir_option
@predictions_dir_option
def validate_corpus(charts_dir: Path, predictions_dir: Path) -> None:
    """Check DTX charts and prediction MIDIs for missing files, stray files, and duplicate stems."""
    from src.benchmark.corpus import validate_score_midi_corpus

    result = validate_score_midi_corpus(charts_dir, predictions_dir)
    for error in result.errors:
        click.echo(error, err=True)
    if result.errors:
        raise click.ClickException(f"{len(result.errors)} validation error(s)")
    click.echo(f"Valid corpus: {len(result.valid_items)} chart(s)")


@benchmark.command("inspect-dtx")
@click.argument("dtx_path", type=click.Path(exists=True, path_type=Path))
def inspect_dtx(dtx_path: Path) -> None:
    """Print parsed DTX chart metadata: event count, BPM, measure changes, and lane list."""
    from src.benchmark.dtx_parser import parse_dtx_file

    chart = parse_dtx_file(dtx_path, chart_id=dtx_path.stem)
    lanes = sorted({event.lane_id for event in chart.events})
    click.echo(f"chart_id: {chart.chart_id}")
    click.echo(f"title: {chart.title}")
    click.echo(f"base_bpm: {chart.base_bpm}")
    click.echo(f"events: {len(chart.events)}")
    click.echo(f"bpm_events: {len(chart.bpm_events)}")
    click.echo(f"measure_length_changes: {len(chart.measure_lengths)}")
    click.echo(f"lanes: {','.join(lanes)}")


@benchmark.command("render-audio")
@song_dir_option
@click.option("--raw-dir", type=click.Path(path_type=Path), required=False)
@run_name_option
@output_dir_option
def render_audio(
    song_dir: Path | None,
    raw_dir: Path | None,
    run_name: str | None,
    output_dir: Path | None,
) -> None:
    """Render drum-only audio from a song folder or raw corpus."""
    from src.benchmark.render_audio import render_audio_corpus, render_audio_song

    if (song_dir is None) == (raw_dir is None):
        raise click.UsageError("Provide exactly one of --song-dir or --raw-dir")

    source_dir = song_dir or raw_dir
    assert source_dir is not None
    resolved_output_dir = resolve_benchmark_output_dir(output_dir, run_name, source_dir)

    if song_dir is not None:
        result = render_audio_song(song_dir, resolved_output_dir)
        if result.invalid_items and not result.valid_items:
            invalid = result.invalid_items[0]
            raise click.ClickException(
                f"{invalid.reason}: {invalid.details} (see {result.invalid_report_path})"
            )
        click.echo(f"Rendered {len(result.valid_items)} song(s) to {resolved_output_dir / 'audio'}")
        return

    result = render_audio_corpus(raw_dir, resolved_output_dir)
    if result.invalid_items:
        click.echo(f"Skipped {len(result.invalid_items)} invalid song folder(s)", err=True)
    click.echo(f"Rendered {len(result.valid_items)} song(s) to {resolved_output_dir / 'audio'}")
