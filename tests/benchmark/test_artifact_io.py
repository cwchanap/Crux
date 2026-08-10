from __future__ import annotations

import hashlib

import pytest

from src.benchmark.artifact_io import (
    ArtifactPublicationError,
    publish_immutable_file,
    read_regular_file_no_follow,
)


def test_read_regular_file_no_follow_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"abc")
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(OSError):
        read_regular_file_no_follow(link)


def test_publish_immutable_file_reuses_identical_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    content = b'{"ok":true}\n'

    first = publish_immutable_file(path, content)
    second = publish_immutable_file(path, content)

    assert first == second
    assert first.path == path
    assert first.sha256 == hashlib.sha256(content).hexdigest()


def test_publish_immutable_file_rejects_different_existing_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    publish_immutable_file(path, b"one")

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_file(path, b"two")
