from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import click

F = TypeVar("F", bound=Callable)


def charts_dir_option(function: F) -> F:
    return click.option("--charts-dir", type=click.Path(path_type=Path), required=True)(function)


def audio_dir_option(function: F) -> F:
    return click.option("--audio-dir", type=click.Path(path_type=Path), required=True)(function)


def predictions_dir_option(function: F) -> F:
    return click.option("--predictions-dir", type=click.Path(path_type=Path), required=True)(
        function
    )


def output_dir_option(function: F) -> F:
    return click.option("--output-dir", type=click.Path(path_type=Path), required=True)(function)


def tolerance_option(function: F) -> F:
    return click.option(
        "--tolerance-ms",
        type=int,
        multiple=True,
        required=True,
        help="Onset tolerance window in milliseconds. Repeat for multiple windows.",
    )(function)
