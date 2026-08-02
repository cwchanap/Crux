"""Shared filesystem fsync helper for OaF tooling."""

from __future__ import annotations

import os
from pathlib import Path


def fsync_directory(path: Path) -> None:
    """Open ``path`` read-only and fsync it, closing the descriptor.

    Used to durably flush directory metadata after creating or renaming
    entries within it. The directory is opened with ``O_NOFOLLOW`` so a
    symlink substitution cannot redirect the fsync to a different inode, and
    ``O_DIRECTORY`` so a non-directory at the path is rejected rather than
    silently synced.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
