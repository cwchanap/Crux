#!/usr/bin/env python3
"""Freeze one already-working external separator runtime into a lock."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.benchmark.separators import (
    HTDEMUCS_SEPARATOR_ID,
    SPLEETER_SEPARATOR_ID,
    SeparatorExecutionError,
    freeze_separator_runtime,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--separator-id",
        choices=(SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID),
        required=True,
    )
    parser.add_argument("--interpreter", "--python", dest="interpreter", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
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
    try:
        lock = freeze_separator_runtime(
            separator_id=args.separator_id,
            interpreter=args.interpreter,
            model_root=args.model_root,
            repository_revision=args.repository_revision,
            output=args.output,
        )
    except (SeparatorExecutionError, OSError, ValueError) as error:
        print(f"freeze failed: {error}", file=sys.stderr)
        return 1
    print(lock.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
