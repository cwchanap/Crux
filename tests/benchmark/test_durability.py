import os
from pathlib import Path

import pytest

from src.benchmark.durability import (
    ensure_durable_directory,
    fsync_directory,
)


def test_fsync_directory_raises_when_no_follow_unavailable(monkeypatch, tmp_path):
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    with pytest.raises(OSError, match="directory durability support is unavailable"):
        fsync_directory(tmp_path)


def test_fsync_directory_raises_when_o_directory_unavailable(monkeypatch, tmp_path):
    monkeypatch.delattr(os, "O_DIRECTORY", raising=False)
    with pytest.raises(OSError, match="directory durability support is unavailable"):
        fsync_directory(tmp_path)


def test_fsync_directory_raises_when_path_is_not_a_directory(monkeypatch, tmp_path):
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("data")

    real_open = os.open

    def open_non_dir(path, flags, *args, **kwargs):
        # Force opening the file as if it were the directory path.
        return real_open(file_path, os.O_RDONLY)

    monkeypatch.setattr(os, "open", open_non_dir)
    with pytest.raises(OSError, match="directory is unavailable"):
        fsync_directory(tmp_path)


def test_ensure_durable_directory_raises_when_ancestor_is_unavailable(monkeypatch, tmp_path):
    # Simulate walking all the way up to root without finding an existing dir.
    def lstat_always_missing(self):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(Path, "lstat", lstat_always_missing)
    with pytest.raises(OSError, match="directory ancestor is unavailable"):
        ensure_durable_directory(tmp_path / "deep" / "nested" / "path")


def test_ensure_durable_directory_raises_when_path_is_not_a_directory(monkeypatch, tmp_path):
    file_path = tmp_path / "a_file"
    file_path.write_text("data")

    def lstat_file(self):
        if self == file_path:
            return os.stat(file_path)
        return Path.lstat(self)

    monkeypatch.setattr(Path, "lstat", lstat_file)
    with pytest.raises(OSError, match="directory path is unavailable"):
        ensure_durable_directory(file_path)


def test_ensure_durable_directory_handles_mkdir_race_for_directory(monkeypatch, tmp_path):
    target = tmp_path / "concurrent_dir"
    calls = {"count": 0}

    real_mkdir = Path.mkdir

    def mkdir_race(self, *args, **kwargs):
        if self == target:
            calls["count"] += 1
            if calls["count"] == 1:
                # Another process created the directory concurrently.
                real_mkdir(target, *args, **kwargs)
                raise FileExistsError("race")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir_race)
    # fsync on parent must work on real filesystem.
    ensure_durable_directory(target)
    assert target.is_dir()
    assert calls["count"] == 1


def test_ensure_durable_directory_handles_mkdir_race_for_non_directory(monkeypatch, tmp_path):
    target = tmp_path / "concurrent_file"

    real_mkdir = Path.mkdir
    calls = {"count": 0}

    def mkdir_race(self, *args, **kwargs):
        if self == target:
            calls["count"] += 1
            # Another process created a regular file at this path concurrently.
            target.write_text("blocking")
            raise FileExistsError("race")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir_race)
    with pytest.raises(OSError, match="directory path is unavailable"):
        ensure_durable_directory(target)
    assert calls["count"] == 1
