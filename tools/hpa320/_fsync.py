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
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OSError("directory durability support is unavailable")
    descriptor = os.open(path, os.O_RDONLY | no_follow | directory)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
