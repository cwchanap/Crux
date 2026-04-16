from __future__ import annotations

import click

from src.cli.benchmark import benchmark
from src.cli.convert import main as convert_checkpoint


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Crux command-line tools."""


@click.group()
def convert() -> None:
    """Model and asset conversion commands."""


convert.add_command(convert_checkpoint, "checkpoint")
main.add_command(benchmark)
main.add_command(convert)


if __name__ == "__main__":
    main()
