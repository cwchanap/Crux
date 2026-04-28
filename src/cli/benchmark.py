from __future__ import annotations

from pathlib import Path

import click

from src.benchmark.corpus import validate_score_midi_corpus
from src.benchmark.dtx_parser import parse_dtx_file
from src.benchmark.prepare import prepare_corpus
from src.benchmark.render_audio import render_audio_corpus, render_audio_song
from src.benchmark.runner import export_reference_midis, run_score_midi, run_transcribe_and_score
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


@benchmark.command("prepare-corpus")
@raw_dir_option
@run_name_option
@output_dir_option
def prepare_benchmark_corpus(raw_dir: Path, run_name: str | None, output_dir: Path | None) -> None:
    """Convert raw song folders into the parsed benchmark corpus format."""
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
    reports = run_transcribe_and_score(
        charts_dir,
        audio_dir,
        resolve_benchmark_output_dir(output_dir, run_name, charts_dir.parent),
        list(tolerance_ms),
    )
    click.echo(
        f"Wrote benchmark reports for {len({report.chart_id for report in reports})} chart(s)"
    )
