from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from uuid import uuid4

_logger = logging.getLogger(__name__)


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


def atomic_replace_bytes(path: Path, content: bytes) -> None:
    temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    temporary_exists = False
    completed = False
    cleanup_failed = False
    try:
        with temporary_path.open("xb") as temporary:
            temporary_exists = True
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_exists = False
        fsync_directory(path.parent)
        completed = True
    except Exception as original_error:
        completed = False
        _logger.debug("atomic replacement failed for %s", path, exc_info=original_error)
    finally:
        if temporary_exists:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception as cleanup_error:
                cleanup_failed = True
                _logger.debug(
                    "atomic replacement cleanup failed for %s",
                    temporary_path,
                    exc_info=cleanup_error,
                )
    if not completed or cleanup_failed:
        raise OSError("artifact publication failed")


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
