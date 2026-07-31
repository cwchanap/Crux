from __future__ import annotations

# Exact schema/durability fixtures intentionally repeat production-facing field
# shapes, and the failing writer deliberately owns a manually controlled file.
# pylint: disable=duplicate-code,consider-using-with
import contextlib
import os
import stat
import tempfile
from pathlib import Path
from uuid import UUID

import pytest

from src.benchmark import backend_publication
from src.benchmark.backend_identity import sha256_hex
from src.benchmark.backend_publication import (
    ArtifactPublicationError,
    atomic_replace_bytes,
    open_directory_anchor,
    publish_immutable_bytes,
    read_regular_file_no_follow,
    rename_directory_no_replace,
)


def test_immutable_publication_uses_expected_bytes_hash_and_role(tmp_path: Path) -> None:
    content = b"immutable backend artifact\n"
    destination = tmp_path / "nested" / "artifact.json"

    published = publish_immutable_bytes(
        destination,
        content,
        sha256_hex(content),
        role="prediction",
    )

    assert published.role == "prediction"
    assert published.path == destination
    assert published.sha256 == sha256_hex(content)
    assert destination.read_bytes() == content
    assert stat.S_ISREG(destination.lstat().st_mode)


def test_rename_directory_no_replace_publishes_once(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "published"

    rename_directory_no_replace(source, destination)

    assert not source.exists()
    assert (destination / "evidence.json").read_bytes() == b"{}\n"


def test_rename_directory_no_replace_never_replaces_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "published"
    source.mkdir()
    destination.mkdir()
    (destination / "sentinel").write_text("keep", encoding="utf-8")

    with pytest.raises(ArtifactPublicationError):
        rename_directory_no_replace(source, destination)

    assert (destination / "sentinel").read_text(encoding="utf-8") == "keep"
    assert source.is_dir()


@pytest.mark.parametrize("raise_inside", [False, True])
def test_directory_anchor_closes_descriptor_on_every_exit(
    tmp_path: Path,
    raise_inside: bool,
) -> None:
    descriptor: int | None = None

    with pytest.raises(RuntimeError) if raise_inside else contextlib.nullcontext():
        with open_directory_anchor(tmp_path) as anchor:
            descriptor = anchor.descriptor
            assert stat.S_ISDIR(os.fstat(descriptor).st_mode)
            if raise_inside:
                raise RuntimeError("injected callback failure")

    assert descriptor is not None
    with pytest.raises(OSError):
        os.fstat(descriptor)


@pytest.mark.parametrize("raise_inside", [False, True])
def test_regular_file_anchor_closes_descriptor_on_every_exit(
    tmp_path: Path,
    raise_inside: bool,
) -> None:
    path = tmp_path / "input.wav"
    path.write_bytes(b"original bytes")
    descriptor: int | None = None

    with pytest.raises(RuntimeError) if raise_inside else contextlib.nullcontext():
        with backend_publication.open_regular_file_anchor(path) as anchor:
            descriptor = anchor.descriptor
            assert stat.S_ISREG(os.fstat(descriptor).st_mode)
            assert anchor.content == b"original bytes"
            if raise_inside:
                raise RuntimeError("injected callback failure")

    assert descriptor is not None
    with pytest.raises(OSError):
        os.fstat(descriptor)


@pytest.mark.parametrize("raise_inside", [False, True])
def test_private_file_snapshot_is_read_only_and_cleans_every_exit(
    raise_inside: bool,
) -> None:
    content = b"captured canonical audio"
    snapshot_path: Path | None = None
    file_descriptor: int | None = None
    directory_descriptor: int | None = None

    with pytest.raises(RuntimeError) if raise_inside else contextlib.nullcontext():
        with backend_publication.open_private_file_snapshot(
            content,
            sha256_hex(content),
            root=backend_publication.resolve_private_snapshot_root(),
        ) as snapshot:
            snapshot_path = snapshot.path
            file_descriptor = snapshot.descriptor
            directory_descriptor = snapshot.directory_descriptor
            snapshot.verify()
            assert snapshot.path.read_bytes() == content
            assert stat.S_IMODE(snapshot.path.stat().st_mode) == 0o400
            assert stat.S_IMODE(snapshot.path.parent.stat().st_mode) == 0o500
            if raise_inside:
                raise RuntimeError("injected backend failure")

    assert snapshot_path is not None
    assert not snapshot_path.exists()
    assert not snapshot_path.parent.exists()
    assert file_descriptor is not None
    assert directory_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(file_descriptor)
    with pytest.raises(OSError):
        os.fstat(directory_descriptor)


def test_private_file_snapshot_detects_path_replacement_and_removes_replacement() -> None:
    content = b"captured canonical audio"
    snapshot_path: Path | None = None

    with backend_publication.open_private_file_snapshot(
        content,
        sha256_hex(content),
        root=backend_publication.resolve_private_snapshot_root(),
    ) as snapshot:
        snapshot_path = snapshot.path
        snapshot.path.parent.chmod(0o700)
        snapshot.path.unlink()
        snapshot.path.write_bytes(b"attacker audio")
        with pytest.raises(OSError):
            snapshot.verify()

    assert snapshot_path is not None
    assert not snapshot_path.exists()
    assert not snapshot_path.parent.exists()


def test_private_snapshot_supports_symlinked_system_temp_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_root = tmp_path / "resolved-temp"
    resolved_root.mkdir()
    configured_alias = tmp_path / "configured-temp"
    configured_alias.symlink_to(resolved_root, target_is_directory=True)
    monkeypatch.setenv("TMPDIR", str(configured_alias))
    monkeypatch.setattr(tempfile, "tempdir", None)
    content = b"captured canonical audio"
    snapshot_path: Path | None = None

    assert Path(tempfile.gettempdir()) == configured_alias
    snapshot_root = backend_publication.resolve_private_snapshot_root()
    assert snapshot_root == resolved_root
    with backend_publication.open_private_file_snapshot(
        content,
        sha256_hex(content),
        root=snapshot_root,
    ) as snapshot:
        snapshot_path = snapshot.path
        assert snapshot.path.parent.parent == resolved_root
        assert snapshot.path.read_bytes() == content

    assert snapshot_path is not None
    assert not snapshot_path.exists()
    assert list(resolved_root.iterdir()) == []
    assert configured_alias.is_symlink()


@pytest.mark.parametrize("invalid_kind", ["file", "symlink"])
def test_private_snapshot_creation_rejects_unresolved_or_non_directory_root(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    invalid_root = tmp_path / "invalid-root"
    if invalid_kind == "file":
        invalid_root.write_bytes(b"not a directory")
    else:
        target = tmp_path / "target"
        target.mkdir()
        invalid_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(OSError):
        with backend_publication.open_private_file_snapshot(
            b"captured canonical audio",
            sha256_hex(b"captured canonical audio"),
            root=invalid_root,
        ):
            pytest.fail("invalid snapshot root was accepted")


@pytest.mark.parametrize("replacement_is_empty", [True, False])
def test_private_snapshot_cleanup_follows_renamed_directory_inode(
    replacement_is_empty: bool,
) -> None:
    content = b"captured canonical audio"
    original_path: Path | None = None
    renamed_path: Path | None = None
    marker_path: Path | None = None
    file_descriptor: int | None = None
    directory_descriptor: int | None = None
    parent_descriptor: int | None = None

    try:
        with pytest.raises(OSError, match="private snapshot integrity failed"):
            with backend_publication.open_private_file_snapshot(
                content,
                sha256_hex(content),
                root=backend_publication.resolve_private_snapshot_root(),
            ) as snapshot:
                original_path = snapshot.path.parent
                renamed_path = original_path.with_name(f"{original_path.name}-renamed")
                file_descriptor = snapshot.descriptor
                directory_descriptor = snapshot.directory_descriptor
                parent_descriptor = snapshot.parent_descriptor
                original_path.rename(renamed_path)
                original_path.mkdir()
                if not replacement_is_empty:
                    marker_path = original_path / "unrelated.txt"
                    marker_path.write_bytes(b"must survive")

        assert original_path is not None
        assert renamed_path is not None
        assert not renamed_path.exists()
        if replacement_is_empty:
            assert not original_path.exists()
        else:
            assert marker_path is not None
            assert marker_path.read_bytes() == b"must survive"
            assert original_path.is_dir()
        assert file_descriptor is not None
        assert directory_descriptor is not None
        assert parent_descriptor is not None
        with pytest.raises(OSError):
            os.fstat(file_descriptor)
        with pytest.raises(OSError):
            os.fstat(directory_descriptor)
        with pytest.raises(OSError):
            os.fstat(parent_descriptor)
    finally:
        if marker_path is not None and marker_path.exists():
            marker_path.unlink()
        if original_path is not None and original_path.exists():
            original_path.rmdir()
        if renamed_path is not None and renamed_path.exists():
            snapshot_file = renamed_path / "input.wav"
            if snapshot_file.exists():
                snapshot_file.unlink()
            renamed_path.rmdir()


def test_immutable_publication_rejects_wrong_expected_hash_before_visibility(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(destination, b"content", "0" * 64, role="prediction")

    assert not destination.exists()


def test_exact_existing_immutable_content_is_idempotent_and_keeps_inode(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"
    content = b"same bytes\n"
    destination.write_bytes(content)
    before = destination.stat()

    published = publish_immutable_bytes(
        destination,
        content,
        sha256_hex(content),
        role="report",
    )

    after = destination.stat()
    assert published.sha256 == sha256_hex(content)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def test_existing_immutable_hash_collision_rejects_different_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_bytes(b"prior valid bytes\n")

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            destination,
            b"new bytes\n",
            sha256_hex(b"new bytes\n"),
            role="report",
        )

    assert destination.read_bytes() == b"prior valid bytes\n"


@pytest.mark.parametrize("broken", [False, True])
def test_immutable_publication_rejects_symlink_destination(
    tmp_path: Path,
    broken: bool,
) -> None:
    target = tmp_path / "target.json"
    if not broken:
        target.write_bytes(b"target")
    destination = tmp_path / "artifact.json"
    destination.symlink_to(target if not broken else tmp_path / "missing.json")

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            destination,
            b"content",
            sha256_hex(b"content"),
            role="report",
        )

    assert destination.is_symlink()
    if not broken:
        assert target.read_bytes() == b"target"


def test_immutable_publication_rejects_non_regular_destination_without_blocking(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.fifo"
    os.mkfifo(destination)

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            destination,
            b"content",
            sha256_hex(b"content"),
            role="report",
        )

    assert stat.S_ISFIFO(destination.lstat().st_mode)


def test_immutable_temporary_creation_is_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = UUID("12345678-1234-4678-9234-567812345678")
    temporary = tmp_path / ".artifact.json.12345678123446789234567812345678.tmp"
    temporary.write_bytes(b"attacker bytes")
    monkeypatch.setattr(backend_publication, "uuid4", lambda: fixed)

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            tmp_path / "artifact.json",
            b"content",
            sha256_hex(b"content"),
            role="report",
        )

    assert temporary.read_bytes() == b"attacker bytes"
    assert not (tmp_path / "artifact.json").exists()


def test_fresh_immutable_publication_fsyncs_file_then_directory_after_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    events: list[str] = []
    real_link = os.link
    real_fsync = os.fsync

    def tracked_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        real_link(source, target, *args, **kwargs)  # type: ignore[arg-type]
        events.append("linked")

    def tracked_fsync(descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        if stat.S_ISREG(descriptor_stat.st_mode) and destination.exists():
            destination_stat = destination.stat(follow_symlinks=False)
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (
                destination_stat.st_dev,
                destination_stat.st_ino,
            ):
                events.append("file_fsync")
        if stat.S_ISDIR(descriptor_stat.st_mode):
            directory_stat = tmp_path.stat()
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (
                directory_stat.st_dev,
                directory_stat.st_ino,
            ):
                events.append("directory_fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(backend_publication.os, "link", tracked_link)
    monkeypatch.setattr(backend_publication.os, "fsync", tracked_fsync)

    publish_immutable_bytes(
        destination,
        b"content",
        sha256_hex(b"content"),
        role="report",
    )

    linked = events.index("linked")
    winner_fsync = events.index("file_fsync", linked)
    directory_fsync = events.index("directory_fsync", winner_fsync)
    assert linked < winner_fsync < directory_fsync


def test_immutable_publication_rejects_destination_inode_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    real_link = os.link

    def substituted_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        real_link(source, target, *args, **kwargs)  # type: ignore[arg-type]
        destination.unlink()
        destination.write_bytes(b"substituted")

    monkeypatch.setattr(backend_publication.os, "link", substituted_link)

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            destination,
            b"expected",
            sha256_hex(b"expected"),
            role="report",
        )

    assert destination.read_bytes() == b"substituted"


def test_immutable_publication_propagates_temporary_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    real_unlink = os.unlink

    def failed_temporary_unlink(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if Path(path).name.startswith(".artifact.json."):
            raise OSError("cleanup failed")
        real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(backend_publication.os, "unlink", failed_temporary_unlink)

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            destination,
            b"expected",
            sha256_hex(b"expected"),
            role="report",
        )

    assert destination.read_bytes() == b"expected"


def test_atomic_replace_publishes_exact_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "latest.json"
    destination.write_bytes(b"old")

    atomic_replace_bytes(destination, b"new\n")

    assert destination.read_bytes() == b"new\n"
    assert stat.S_ISREG(destination.lstat().st_mode)


@pytest.mark.parametrize("broken", [False, True])
def test_atomic_replace_rejects_symlink_destination(
    tmp_path: Path,
    broken: bool,
) -> None:
    target = tmp_path / "target.json"
    if not broken:
        target.write_bytes(b"target")
    destination = tmp_path / "latest.json"
    destination.symlink_to(target if not broken else tmp_path / "missing.json")

    with pytest.raises(ArtifactPublicationError):
        atomic_replace_bytes(destination, b"new")

    assert destination.is_symlink()
    if not broken:
        assert target.read_bytes() == b"target"


def test_atomic_replace_write_failure_keeps_previous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "latest.json"
    destination.write_bytes(b"old")

    def failed_write(_descriptor: int, _content: bytes) -> int:
        raise OSError("write failed")

    monkeypatch.setattr(backend_publication.os, "write", failed_write)

    with pytest.raises(ArtifactPublicationError):
        atomic_replace_bytes(destination, b"new")

    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob(".latest.json.*.tmp")) == []


def test_atomic_replace_fsync_failure_keeps_previous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "latest.json"
    destination.write_bytes(b"old")

    def failed_fsync(_descriptor: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(backend_publication.os, "fsync", failed_fsync)

    with pytest.raises(ArtifactPublicationError):
        atomic_replace_bytes(destination, b"new")

    assert destination.read_bytes() == b"old"


def test_atomic_replace_failure_keeps_previous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "latest.json"
    destination.write_bytes(b"old")

    def failed_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(backend_publication.os, "replace", failed_replace)

    with pytest.raises(ArtifactPublicationError):
        atomic_replace_bytes(destination, b"new")

    assert destination.read_bytes() == b"old"


def test_immutable_publication_rejects_ancestor_swap_without_writing_attacker_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted"
    original = tmp_path / "trusted-original"
    attacker = tmp_path / "attacker"
    destination = trusted / "reports" / "artifact.json"
    destination.parent.mkdir(parents=True)
    (attacker / "reports").mkdir(parents=True)
    real_link = os.link
    swapped = False

    def swap_before_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            trusted.rename(original)
            trusted.symlink_to(attacker, target_is_directory=True)
            source_name = Path(source).name
            real_link(
                original / "reports" / source_name,
                attacker / "reports" / source_name,
            )
        real_link(source, target, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(backend_publication.os, "link", swap_before_link)

    with pytest.raises(ArtifactPublicationError):
        publish_immutable_bytes(
            destination,
            b"trusted content",
            sha256_hex(b"trusted content"),
            role="report",
        )

    assert not (attacker / "reports" / "artifact.json").exists()


def test_atomic_replace_rejects_ancestor_swap_without_replacing_attacker_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted"
    original = tmp_path / "trusted-original"
    attacker = tmp_path / "attacker"
    destination = trusted / "latest.json"
    trusted.mkdir()
    destination.write_bytes(b"trusted old")
    attacker.mkdir()
    (attacker / "latest.json").write_bytes(b"attacker old")
    real_replace = os.replace
    real_link = os.link
    swapped = False

    def swap_before_replace(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            trusted.rename(original)
            trusted.symlink_to(attacker, target_is_directory=True)
            source_name = Path(source).name
            real_link(original / source_name, attacker / source_name)
        real_replace(source, target, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(backend_publication.os, "replace", swap_before_replace)

    with pytest.raises(ArtifactPublicationError):
        atomic_replace_bytes(destination, b"trusted new")

    assert (attacker / "latest.json").read_bytes() == b"attacker old"


def test_no_follow_read_rejects_ancestor_swap_instead_of_reading_attacker_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted"
    original = tmp_path / "trusted-original"
    attacker = tmp_path / "attacker"
    source = trusted / "artifact.json"
    trusted.mkdir()
    source.write_bytes(b"trusted")
    attacker.mkdir()
    (attacker / "artifact.json").write_bytes(b"attacker")
    real_open = os.open
    swapped = False

    def swap_before_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and (Path(path) == source or path == source.name):
            swapped = True
            trusted.rename(original)
            trusted.symlink_to(attacker, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(backend_publication.os, "open", swap_before_open)

    with pytest.raises(OSError):
        read_regular_file_no_follow(source)
