from __future__ import annotations

from pathlib import Path

import click

from src.benchmark.corpus import validate_score_midi_corpus
from src.benchmark.dtx_parser import parse_dtx_file
from src.benchmark.runner import export_reference_midis, run_score_midi, run_transcribe_and_score
from src.cli.options import (
    audio_dir_option,
    charts_dir_option,
    output_dir_option,
    predictions_dir_option,
    tolerance_option,
)


@click.group()
def benchmark() -> None:
    """Benchmark drum transcription against DTX ground truth."""


@benchmark.command("score-midi")
@charts_dir_option
@predictions_dir_option
@output_dir_option
@tolerance_option
@click.option("--align/--no-align", default=True, show_default=True)
@click.option(
    "--export-reference-midi/--no-export-reference-midi", default=False, show_default=True
)
def score_midi(
    charts_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    tolerance_ms: tuple[int, ...],
    align: bool,
    export_reference_midi: bool,
) -> None:
    """Score precomputed prediction MIDI files."""
    reports = run_score_midi(
        charts_dir=charts_dir,
        predictions_dir=predictions_dir,
        output_dir=output_dir,
        tolerance_ms=list(tolerance_ms),
        align=align,
        export_reference_midi=export_reference_midi,
    )
    click.echo(
        f"Wrote benchmark reports for {len({report.chart_id for report in reports})} chart(s)"
    )


@benchmark.command("validate-corpus")
@charts_dir_option
@predictions_dir_option
def validate_corpus(charts_dir: Path, predictions_dir: Path) -> None:
    """Validate benchmark corpus folder matching."""
    result = validate_score_midi_corpus(charts_dir, predictions_dir)
    for error in result.errors:
        click.echo(error, err=True)
    if result.errors:
        raise click.ClickException(f"{len(result.errors)} validation error(s)")
    click.echo(f"Valid corpus: {len(result.valid_items)} chart(s)")


@benchmark.command("inspect-dtx")
@click.argument("dtx_path", type=click.Path(exists=True, path_type=Path))
def inspect_dtx(dtx_path: Path) -> None:
    """Inspect parsed DTX timing and lane statistics."""
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
@output_dir_option
def export_reference_midi(charts_dir: Path, output_dir: Path) -> None:
    """Export benchmark-owned reference MIDI artifacts."""
    count = export_reference_midis(charts_dir, output_dir)
    click.echo(f"Wrote {count} reference MIDI file(s)")


@benchmark.command("transcribe-and-score")
@charts_dir_option
@audio_dir_option
@output_dir_option
@tolerance_option
def transcribe_and_score(
    charts_dir: Path,
    audio_dir: Path,
    output_dir: Path,
    tolerance_ms: tuple[int, ...],
) -> None:
    """Run transcription and score generated MIDI."""
    reports = run_transcribe_and_score(charts_dir, audio_dir, output_dir, list(tolerance_ms))
    click.echo(
        f"Wrote benchmark reports for {len({report.chart_id for report in reports})} chart(s)"
    )
