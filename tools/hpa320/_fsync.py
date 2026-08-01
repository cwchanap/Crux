"""Shared filesystem fsync helper for OaF tooling."""

from __future__ import annotations

import os
from pathlib import Path


def _fsync_directory(path: Path) -> None:
    """Open ``path`` read-only and fsync it, closing the descriptor.

    Used to durably flush directory metadata after creating or renaming
    entries within it.
    """
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
