from __future__ import annotations

from pathlib import Path

import click

from src.benchmark.corpus import validate_score_midi_corpus
from src.benchmark.runner import run_score_midi
from src.cli.options import (
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
    raise click.ClickException("inspect-dtx is planned but not implemented yet")


@benchmark.command("export-reference-midi")
def export_reference_midi() -> None:
    """Export benchmark-owned reference MIDI artifacts."""
    raise click.ClickException("export-reference-midi is planned but not implemented yet")


@benchmark.command("transcribe-and-score")
def transcribe_and_score() -> None:
    """Run transcription and score generated MIDI."""
    raise click.ClickException("transcribe-and-score is planned but not implemented yet")
