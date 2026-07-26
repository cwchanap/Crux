from __future__ import annotations

import os
import stat
from pathlib import Path


def fsync_directory(path: Path) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OSError("directory durability support is unavailable")
    descriptor = os.open(path, os.O_RDONLY | no_follow | directory)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory is unavailable")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_durable_directory(path: Path) -> None:
    missing: list[Path] = []
    candidate = path
    while True:
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            missing.append(candidate)
            parent = candidate.parent
            if parent == candidate:
                raise OSError("directory ancestor is unavailable")
            candidate = parent
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError("directory path is unavailable")
        break
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            metadata = directory.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise OSError("directory path is unavailable") from None
        fsync_directory(directory.parent)
