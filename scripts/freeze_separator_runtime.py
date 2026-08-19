#!/usr/bin/env python3
"""Freeze one already-working external separator runtime into a lock."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

from src.benchmark.separators import (
    HTDEMUCS_SEPARATOR_ID,
    SPLEETER_SEPARATOR_ID,
    SeparatorLock,
)


class FreezeError(RuntimeError):
    """The mechanical separator lock freeze could not complete."""


def freeze_separator_runtime(
    *,
    separator_id: str,
    interpreter: Path,
    model_files: Mapping[str, Path],
    repository_revision: str,
    output: Path,
) -> SeparatorLock:
    """Hash explicit model files and publish one lock without running inference."""
    raise FreezeError("v2 separator lock freezing is deferred until Task 4")


def _parse_model_file(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("model files must use NAME=PATH")
    return name, Path(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--separator-id",
        choices=(SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID),
        required=True,
    )
    parser.add_argument("--interpreter", "--python", dest="interpreter", type=Path, required=True)
    parser.add_argument(
        "--model-file",
        action="append",
        type=_parse_model_file,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--repository-revision",
        "--revision",
        dest="repository_revision",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    model_files: dict[str, Path] = {}
    for name, path in args.model_file:
        if name in model_files:
            print(f"freeze failed: duplicate model file name {name}", file=sys.stderr)
            return 1
        model_files[name] = path
    try:
        lock = freeze_separator_runtime(
            separator_id=args.separator_id,
            interpreter=args.interpreter,
            model_files=model_files,
            repository_revision=args.repository_revision,
            output=args.output,
        )
    except (FreezeError, OSError, ValueError) as error:
        print(f"freeze failed: {error}", file=sys.stderr)
        return 1
    print(lock.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
