from __future__ import annotations

import errno
import hashlib
import os

import pytest

from src.benchmark import artifact_io
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


@pytest.mark.parametrize("failure", ["write", "fsync"])
def test_publish_immutable_file_at_cleans_temporary_file_on_fill_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
):
    """A write/fsync failure while filling the temporary file must not leak it
    and must surface as ArtifactPublicationError, not a raw OSError.

    ``_create_temporary_file_at`` runs before the caller's unlink finally is in
    scope, so without an internal cleanup a partial ``.<name>.<random>`` file
    would remain behind and the failure would escape as a raw OSError, breaking
    the helper's ArtifactPublicationError contract (OaF maps that to
    ``prediction_publish_failed`` rather than ``prediction_artifact_invalid``).
    """
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if failure == "write":

            def fail_write(_descriptor: int, _content: bytes) -> None:
                raise OSError("simulated write failure")

            monkeypatch.setattr(artifact_io, "_write_all", fail_write)
        else:

            def fail_fsync(_fd: int) -> None:
                raise OSError("simulated fsync failure")

            monkeypatch.setattr(os, "fsync", fail_fsync)

        with pytest.raises(ArtifactPublicationError):
            publish_immutable_file_at(directory_fd, "artifact.json", b"payload")
    finally:
        os.close(directory_fd)

    # No partial temporary file leaks behind the failed publication.
    assert list(tmp_path.iterdir()) == []


def test_publish_immutable_file_at_rejects_lost_race_with_different_bytes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Losing the os.link race to different bytes must fail loudly.

    A concurrent publisher can place different content at the target name
    between the existence check and the link; treating the FileExistsError as
    success would silently keep bytes the caller never wrote.
    """
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        target = tmp_path / "artifact.json"

        def win_race(*_args, **_kwargs):
            target.write_bytes(b"raced")
            raise FileExistsError("lost publication race")

        monkeypatch.setattr(os, "link", win_race)

        with pytest.raises(ArtifactPublicationError):
            publish_immutable_file_at(directory_fd, "artifact.json", b"payload")
    finally:
        os.close(directory_fd)

    assert target.read_bytes() == b"raced"
    # The loser's temporary file is still cleaned up.
    assert list(tmp_path.iterdir()) == [target]


def test_publish_immutable_file_at_wraps_unsupported_link(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    """NotImplementedError from os.link is not an OSError, so it must be
    wrapped explicitly or it escapes the helper's contract unwrapped."""
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:

        def unsupported_link(*_args, **_kwargs):
            raise NotImplementedError("os.link follow_symlinks unsupported")

        monkeypatch.setattr(os, "link", unsupported_link)

        with pytest.raises(ArtifactPublicationError):
            publish_immutable_file_at(directory_fd, "artifact.json", b"payload")
    finally:
        os.close(directory_fd)

    assert list(tmp_path.iterdir()) == []


def test_publish_immutable_file_at_wraps_temporary_open_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A failure creating the exclusive temporary file itself must surface as
    ArtifactPublicationError, not a raw OSError.

    ``_create_temporary_file_at`` is called before the caller's unlink finally
    and OSError-to-ArtifactPublicationError wrapper are in scope.  Only
    FileExistsError is a collision retry; other OSError failures from the
    exclusive ``os.open`` (ENOSPC, EACCES, EMFILE, ...) would otherwise escape
    unwrapped, breaking the helper's contract (OaF maps
    ArtifactPublicationError to prediction_publish_failed but a raw OSError to
    prediction_artifact_invalid, so the distinction is observable).
    """
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        real_open = os.open

        def fail_exclusive_open(path, flags, *args, **kwargs):
            if os.fspath(path).startswith(".artifact.json."):
                raise OSError(errno.ENOSPC, "simulated temporary open failure")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", fail_exclusive_open)

        with pytest.raises(ArtifactPublicationError):
            publish_immutable_file_at(directory_fd, "artifact.json", b"payload")
    finally:
        os.close(directory_fd)

    # No temporary file leaks behind the failed publication.
    assert list(tmp_path.iterdir()) == []
