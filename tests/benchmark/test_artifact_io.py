from __future__ import annotations

import hashlib
import os

import pytest

from src.benchmark.artifact_io import (
    ArtifactPublicationError,
    publish_immutable_file,
    publish_immutable_file_at,
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


def test_publish_immutable_file_at_reuses_identical_bytes(tmp_path):
    content = b'{"ok":true}\n'
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        first = publish_immutable_file_at(directory_fd, "artifact.json", content)
        second = publish_immutable_file_at(directory_fd, "artifact.json", content)
    finally:
        os.close(directory_fd)

    assert first == second == hashlib.sha256(content).hexdigest()
    assert (tmp_path / "artifact.json").read_bytes() == content
    # The exclusive temporary file is always cleaned up.
    assert list(tmp_path.iterdir()) == [tmp_path / "artifact.json"]


def test_publish_immutable_file_at_rejects_different_existing_bytes(tmp_path):
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        publish_immutable_file_at(directory_fd, "artifact.json", b"one")
        with pytest.raises(ArtifactPublicationError):
            publish_immutable_file_at(directory_fd, "artifact.json", b"two")
    finally:
        os.close(directory_fd)

    assert (tmp_path / "artifact.json").read_bytes() == b"one"


def test_publish_immutable_file_at_rejects_non_component_name(tmp_path):
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ArtifactPublicationError):
            publish_immutable_file_at(directory_fd, "nested/name.json", b"x")
    finally:
        os.close(directory_fd)

    assert not (tmp_path / "nested").exists()
