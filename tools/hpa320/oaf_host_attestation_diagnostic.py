#!/usr/bin/env python3
"""Run the native-host publisher only to expose a safe diagnostic cause chain."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.benchmark.backend_attestation import AttestationError
from src.benchmark.backend_identity import StrictJsonError
from src.benchmark.backend_publication import ArtifactPublicationError, DirectoryPublicationError
from tools.hpa320.oaf_host_attestation import (
    PHASE_WORKFLOWS,
    HostAttestationError,
    publish_github_host_attestation,
)

_CONTROLLED_MESSAGE_TYPES = (
    HostAttestationError,
    AttestationError,
    StrictJsonError,
    ArtifactPublicationError,
    DirectoryPublicationError,
)
_MAX_CAUSE_DEPTH = 6


def safe_exception_chain(error: BaseException) -> str:
    """Describe controlled causes while redacting unknown exception messages."""

    rows: list[str] = []
    current: BaseException | None = error
    for _ in range(_MAX_CAUSE_DEPTH):
        if current is None:
            break
        name = type(current).__name__
        if isinstance(current, _CONTROLLED_MESSAGE_TYPES):
            message = str(current)
            rows.append(f"{name}: {message}" if message else name)
        elif isinstance(current, OSError):
            detail = f"errno={current.errno}" if current.errno is not None else "errno=unknown"
            rows.append(f"{name}({detail})")
        else:
            rows.append(name)
        current = current.__cause__
    if current is not None:
        rows.append("cause-chain-truncated")
    return " <- ".join(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=tuple(PHASE_WORKFLOWS))
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the real publisher, report a safe cause, and always refuse authority."""

    arguments = _parser().parse_args(argv)
    try:
        publish_github_host_attestation(
            phase=arguments.phase,
            output_directory=arguments.output,
        )
    except HostAttestationError as error:
        print(safe_exception_chain(error), file=sys.stderr)
        return 2
    shutil.rmtree(arguments.output, ignore_errors=True)
    print(
        "native-host diagnostic unexpectedly succeeded; refusing to continue",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
