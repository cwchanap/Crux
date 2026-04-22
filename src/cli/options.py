from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import click

F = TypeVar("F", bound=Callable)
BENCHMARK_ARTIFACT_ROOT = Path("artifacts") / "benchmark"


def charts_dir_option(function: F) -> F:
    return click.option("--charts-dir", type=click.Path(path_type=Path), required=True)(function)


def raw_dir_option(function: F) -> F:
    return click.option("--raw-dir", type=click.Path(path_type=Path), required=True)(function)


def song_dir_option(function: F) -> F:
    return click.option("--song-dir", type=click.Path(path_type=Path), required=False)(function)


def audio_dir_option(function: F) -> F:
    return click.option("--audio-dir", type=click.Path(path_type=Path), required=True)(function)


def predictions_dir_option(function: F) -> F:
    return click.option("--predictions-dir", type=click.Path(path_type=Path), required=True)(
        function
    )


def output_dir_option(function: F) -> F:
    return click.option(
        "--output-dir",
        type=click.Path(path_type=Path),
        required=False,
        help="Defaults to artifacts/benchmark/<run-name-or-input-dir-name>/",
    )(function)


def run_name_option(function: F) -> F:
    return click.option(
        "--run-name",
        type=str,
        required=False,
        help="Optional artifact directory name when --output-dir is omitted.",
    )(function)


def resolve_benchmark_output_dir(
    output_dir: Path | None, run_name: str | None, source_dir: Path
) -> Path:
    if output_dir is not None:
        return output_dir

    name = run_name or source_dir.name or "benchmark-run"
    return BENCHMARK_ARTIFACT_ROOT / name


def tolerance_option(function: F) -> F:
    return click.option(
        "--tolerance-ms",
        type=int,
        multiple=True,
        required=True,
        help="Onset tolerance window in milliseconds. Repeat for multiple windows.",
    )(function)
