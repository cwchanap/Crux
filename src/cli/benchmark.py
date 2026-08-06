from __future__ import annotations

# Commands keep optional and heavy implementation modules behind their Click boundary.
# pylint: disable=import-outside-toplevel
import json
from dataclasses import asdict
from pathlib import Path

import click

from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.r2_corpus_models import MAX_SIMFILE_ID, SyncOutcome, SyncRequest
from src.benchmark.r2_corpus_sync import ProgressEvent, sync_r2_corpus
from src.cli.options import (
    audio_dir_option,
    charts_dir_option,
    output_dir_option,
    predictions_dir_option,
    raw_dir_option,
    resolve_benchmark_output_dir,
    run_name_option,
    song_dir_option,
    tolerance_option,
)


@click.group()
def benchmark() -> None:
    """Benchmark drum transcription against DTX ground truth."""


def _emit_backend_summary(
    *,
    status: str,
    exit_code: int,
    report_path: Path | None,
    report_sha256: str | None,
) -> None:
    """Write the sole machine-readable result after a backend command has parsed."""
    payload = {
        "exit_code": exit_code,
        "report_path": None if report_path is None else str(report_path),
        "report_sha256": report_sha256,
        "status": status,
    }
    standard_output = click.get_binary_stream("stdout")
    standard_output.write(canonical_json_bytes(payload, trailing_newline=True))
    standard_output.flush()


@benchmark.command("prepare-backend")
@click.option("--backend", type=str, required=True)
@click.option("--download", is_flag=True)
@click.option(
    "--archive",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--cache-root",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
@click.option(
    "--acquisition-request",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--evidence-output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--backend-lock",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.pass_context
def prepare_backend_command(
    ctx: click.Context,
    backend: str,
    download: bool,
    archive: Path | None,
    cache_root: Path,
    acquisition_request: Path | None,
    evidence_output: Path | None,
    backend_lock: Path | None,
) -> None:
    """Acquire or verify the immutable frozen OaF checkpoint cache."""
    if download and archive is not None:
        raise click.UsageError("--download and --archive are mutually exclusive.")

    from src.benchmark import backend_prepare
    from src.benchmark.backend_lock import load_backend_lock
    from src.benchmark.backend_registry import OFFICIAL_BACKEND_ID

    if backend != OFFICIAL_BACKEND_ID:
        click.echo("backend_selection_invalid", err=True)
        outcome = backend_prepare.PrepareBackendOutcome(
            status="integrity_failed",
            exit_code=2,
            model_cache_path=None,
        )
    else:
        loaded_lock = None
        if backend_lock is not None:
            try:
                loaded_lock = load_backend_lock(backend_lock)
            except (OSError, ValueError):
                click.echo("backend_lock_invalid", err=True)
                outcome = backend_prepare.PrepareBackendOutcome(
                    status="integrity_failed",
                    exit_code=2,
                    model_cache_path=None,
                )
            else:
                outcome = backend_prepare.prepare_oaf_backend(
                    backend_prepare.PrepareBackendRequest(
                        backend_id=backend,
                        cache_root=cache_root,
                        archive_path=archive,
                        download=download,
                        acquisition_request_path=acquisition_request,
                        evidence_output_path=evidence_output,
                        backend_lock_path=backend_lock,
                    ),
                    backend_lock=loaded_lock,
                )
        else:
            outcome = backend_prepare.prepare_oaf_backend(
                backend_prepare.PrepareBackendRequest(
                    backend_id=backend,
                    cache_root=cache_root,
                    archive_path=archive,
                    download=download,
                    acquisition_request_path=acquisition_request,
                    evidence_output_path=evidence_output,
                    backend_lock_path=None,
                ),
            )

    _emit_backend_summary(
        status=outcome.status,
        exit_code=outcome.exit_code,
        report_path=(None if outcome.evidence_artifact is None else outcome.evidence_artifact.path),
        report_sha256=(
            None if outcome.evidence_artifact is None else outcome.evidence_artifact.sha256
        ),
    )
    if outcome.exit_code:
        ctx.exit(outcome.exit_code)


def _validate_verification_backend_id(
    _ctx: click.Context,
    _parameter: click.Parameter,
    value: str,
) -> str:
    supported = {
        "magenta-egmd-tf1-94529798-8hit-v1",
        "heuristic-onset-v1",
    }
    if value not in supported:
        raise click.BadParameter("must select a supported verification backend")
    return value


@benchmark.command("verify-backend")
@click.option(
    "--backend",
    type=str,
    default="magenta-egmd-tf1-94529798-8hit-v1",
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
    """Verify a frozen transcription backend and publish its sealed evidence."""
    from src.benchmark.backend_registry import default_backend_registry
    from src.benchmark.backend_reports import OperationalReportPublicationError
    from src.benchmark.transcription import VerifyBackendRequest, run_verify_backend

    try:
        outcome = run_verify_backend(
            VerifyBackendRequest(
                backend_id=backend,
                reports_root=reports_root,
                allow_emulated_diagnostics=allow_emulated_diagnostics,
            ),
            registry=default_backend_registry(),
        )
    except OperationalReportPublicationError:
        click.echo(
            "report_publication_failed: Operational report could not be published.",
            err=True,
        )
        ctx.exit(2)

    _emit_backend_summary(
        status=outcome.status,
        exit_code=outcome.exit_code,
        report_path=outcome.report_artifact.path,
        report_sha256=outcome.report_artifact.sha256,
    )
    if outcome.exit_code:
        ctx.exit(outcome.exit_code)


def _validate_transcribe_one_provenance(
    source_audio_id: str | None,
    input_view_id: str | None,
    input_view_manifest: Path | None,
) -> None:
    direct_mode = (
        source_audio_id is not None and input_view_id is not None and input_view_manifest is None
    )
    derived_mode = (
        source_audio_id is None and input_view_id is None and input_view_manifest is not None
    )
    if not (direct_mode or derived_mode):
        raise click.UsageError(
            "Provide exactly one provenance mode: direct IDs or an input-view manifest."
        )


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
    """Transcribe one canonical audio input to authoritative native JSONL."""
    _validate_transcribe_one_provenance(
        source_audio_id,
        input_view_id,
        input_view_manifest,
    )

    from src.benchmark.backend_registry import default_backend_registry
    from src.benchmark.backend_reports import OperationalReportPublicationError
    from src.benchmark.transcription import TranscribeOneRequest, run_transcribe_one

    request = TranscribeOneRequest(
        backend_id=backend,
        audio_path=audio,
        output_path=output,
        source_audio_id=source_audio_id,
        input_view_id=input_view_id,
        input_view_manifest=input_view_manifest,
        midi_output_path=midi_output,
        reports_root=reports_root,
    )
    try:
        outcome = run_transcribe_one(request, registry=default_backend_registry())
    except OperationalReportPublicationError:
        click.echo(
            "report_publication_failed: Operational report could not be published.",
            err=True,
        )
        ctx.exit(2)

    _emit_backend_summary(
        status=outcome.status,
        exit_code=outcome.exit_code,
        report_path=outcome.report_artifact.path,
        report_sha256=outcome.report_artifact.sha256,
    )
    if outcome.exit_code:
        ctx.exit(outcome.exit_code)


# pylint: enable=too-many-arguments,too-many-positional-arguments
# pylint: enable=too-many-locals


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


@benchmark.command("select-reference-charts")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
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
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/reference-charts"),
    show_default=True,
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

    outcome = select_reference_manifest(
        SelectionRequest(
            manifest_path=manifest_path,
            cache_dir=manifest_path.parent.parent / "cache" if cache_dir is None else cache_dir,
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


@benchmark.command("score-midi")
@charts_dir_option
@predictions_dir_option
@run_name_option
@output_dir_option
@tolerance_option
@click.option(
    "--align/--no-align",
    default=True,
    show_default=True,
    help="Compute and apply a global time-offset correction. Emits both raw and aligned report rows when enabled.",
)
@click.option(
    "--export-reference-midi/--no-export-reference-midi", default=False, show_default=True
)
def score_midi(
    charts_dir: Path,
    predictions_dir: Path,
    run_name: str | None,
    output_dir: Path | None,
    tolerance_ms: tuple[int, ...],
    align: bool,
    export_reference_midi: bool,
) -> None:
    """Score precomputed prediction MIDI files."""
    from src.benchmark.runner import run_score_midi

    resolved_output_dir = resolve_benchmark_output_dir(output_dir, run_name, charts_dir.parent)
    try:
        reports = run_score_midi(
            charts_dir=charts_dir,
            predictions_dir=predictions_dir,
            output_dir=resolved_output_dir,
            tolerance_ms=list(tolerance_ms),
            align=align,
            export_reference_midi=export_reference_midi,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Wrote benchmark reports for {len({report.chart_id for report in reports})} chart(s)"
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


@benchmark.command("export-reference-midi")
@charts_dir_option
@run_name_option
@output_dir_option
def export_reference_midi(charts_dir: Path, run_name: str | None, output_dir: Path | None) -> None:
    """Export MIDI files derived from DTX charts for manual inspection (not used for scoring)."""
    from src.benchmark.runner import export_reference_midis

    count = export_reference_midis(
        charts_dir, resolve_benchmark_output_dir(output_dir, run_name, charts_dir.parent)
    )
    click.echo(f"Wrote {count} reference MIDI file(s)")


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


@benchmark.command("transcribe-and-score")
@charts_dir_option
@audio_dir_option
@run_name_option
@output_dir_option
@tolerance_option
def transcribe_and_score(
    charts_dir: Path,
    audio_dir: Path,
    run_name: str | None,
    output_dir: Path | None,
    tolerance_ms: tuple[int, ...],
) -> None:
    """Run transcription and score generated MIDI."""
    from src.benchmark.runner import run_transcribe_and_score

    reports = run_transcribe_and_score(
        charts_dir,
        audio_dir,
        resolve_benchmark_output_dir(output_dir, run_name, charts_dir.parent),
        list(tolerance_ms),
    )
    click.echo(
        f"Wrote benchmark reports for {len({report.chart_id for report in reports})} chart(s)"
    )
