from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import struct
import threading
import zipfile
import zlib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

import pytest

import src.benchmark.backend_prepare as backend_prepare
from src.benchmark.backend_identity import BackendDescriptor
from src.benchmark.backend_lock import LoadedBackendLock
from src.benchmark.backend_prepare import (
    PrepareBackendRequest,
)
from src.benchmark.backend_prepare import (
    prepare_oaf_backend as _public_prepare_oaf_backend,
)
from src.benchmark.backend_registry import OFFICIAL_BACKEND_ID

CHECKPOINT_URL = (
    "https://storage.googleapis.com/magentadata/models/"
    "onsets_frames_transcription/e-gmd_checkpoint.zip"
)
COMPONENT_BYTES = {
    "model.ckpt-569400.data-00000-of-00001": b"frozen data",
    "model.ckpt-569400.index": b"frozen index",
    "model.ckpt-569400.meta": b"frozen meta",
}


def _artifact_set_sha256(component_rows: list[dict[str, object]]) -> str:
    content = json.dumps(
        sorted(component_rows, key=lambda row: str(row["name"])),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _zip_bytes(
    entries: list[tuple[str | zipfile.ZipInfo, bytes]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        for entry, content in entries:
            archive.writestr(entry, content)
    return stream.getvalue()


def _valid_archive_bytes() -> bytes:
    return _zip_bytes(list(COMPONENT_BYTES.items()))


class _NonSeekableBytesIO(io.BytesIO):
    def seekable(self) -> bool:
        return False

    def seek(self, *args: object, **kwargs: object) -> int:
        del args, kwargs
        raise io.UnsupportedOperation("not seekable")


def _hidden_trailing_archive_bytes(*, data_descriptor: bool) -> bytes:
    stream = _NonSeekableBytesIO() if data_descriptor else io.BytesIO()
    trailing = b"TRAILING"
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in COMPONENT_BYTES.items():
            archive.writestr(
                name,
                content + trailing if name.endswith(".index") else content,
            )
    content = bytearray(stream.getvalue())
    name = b"model.ckpt-569400.index"
    central_name_offset = content.rfind(name)
    central_offset = content.rfind(b"PK\x01\x02", 0, central_name_offset)
    local_name_offset = content.find(name)
    local_offset = content.rfind(b"PK\x03\x04", 0, local_name_offset)
    assert central_offset >= 0 and local_offset >= 0
    prefix = COMPONENT_BYTES[name.decode()]
    crc = zlib.crc32(prefix)
    struct.pack_into("<L", content, central_offset + 16, crc)
    struct.pack_into("<L", content, central_offset + 24, len(prefix))
    flags = struct.unpack_from("<H", content, local_offset + 6)[0]
    if data_descriptor:
        assert flags & 0x8
        name_length, extra_length = struct.unpack_from("<HH", content, local_offset + 26)
        descriptor_offset = (
            local_offset + 30 + name_length + extra_length + len(prefix) + len(trailing)
        )
        assert content[descriptor_offset : descriptor_offset + 4] == b"PK\x07\x08"
        struct.pack_into("<L", content, descriptor_offset + 4, crc)
        struct.pack_into("<L", content, descriptor_offset + 12, len(prefix))
    else:
        assert not flags & 0x8
        struct.pack_into("<L", content, local_offset + 14, crc)
        struct.pack_into("<L", content, local_offset + 22, len(prefix))
    return bytes(content)


def _archive_with_central_disk_start(disk_start: int) -> bytes:
    content = bytearray(_valid_archive_bytes())
    name = b"model.ckpt-569400.index"
    central_name_offset = content.rfind(name)
    central_offset = content.rfind(b"PK\x01\x02", 0, central_name_offset)
    assert central_offset >= 0
    struct.pack_into("<H", content, central_offset + 34, disk_start)
    return bytes(content)


def _loaded_lock(
    archive_bytes: bytes,
    *,
    component_bytes: dict[str, bytes] | None = None,
    checkpoint_url: str = CHECKPOINT_URL,
    archive_size: int | None = None,
    archive_sha256: str | None = None,
) -> LoadedBackendLock:
    locked_components = COMPONENT_BYTES if component_bytes is None else component_bytes
    rows = [
        {
            "name": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        for name, content in sorted(locked_components.items())
    ]
    artifact_set_sha256 = _artifact_set_sha256(rows)
    descriptor = BackendDescriptor(
        payload=MappingProxyType({"model_artifact_set_sha256": artifact_set_sha256}),
        sha256="d" * 64,
    )
    payload = MappingProxyType(
        {
            "backend_id": OFFICIAL_BACKEND_ID,
            "checkpoint_archive": MappingProxyType(
                {
                    "name": "e-gmd_checkpoint.zip",
                    "sha256": (
                        hashlib.sha256(archive_bytes).hexdigest()
                        if archive_sha256 is None
                        else archive_sha256
                    ),
                    "size": len(archive_bytes) if archive_size is None else archive_size,
                }
            ),
            "checkpoint_components": tuple(MappingProxyType(row) for row in rows),
            "checkpoint_url": checkpoint_url,
        }
    )
    return LoadedBackendLock(
        path=Path("backend-lock.json"),
        payload=payload,
        sha256="e" * 64,
        descriptor=descriptor,
        max_input_audio_frames=1,
    )


def prepare_oaf_backend(
    request: PrepareBackendRequest,
    *,
    backend_lock: LoadedBackendLock,
) -> backend_prepare.PrepareBackendOutcome:
    archive = backend_lock.payload["checkpoint_archive"]
    component_rows = backend_lock.payload["checkpoint_components"]
    assert isinstance(archive, MappingProxyType)
    assert isinstance(component_rows, tuple)
    contract = backend_prepare._LockContract(
        archive_size=int(archive["size"]),
        archive_sha256=str(archive["sha256"]),
        checkpoint_url=str(backend_lock.payload["checkpoint_url"]),
        components=tuple(
            backend_prepare._Component(
                name=str(row["name"]),
                size=int(row["size"]),
                sha256=str(row["sha256"]),
            )
            for row in component_rows
        ),
        model_artifact_set_sha256=str(backend_lock.descriptor.payload["model_artifact_set_sha256"]),
    )
    return backend_prepare._prepare_oaf_contract(request, contract)


def _request(
    tmp_path: Path,
    *,
    archive_path: Path | None = None,
    download: bool = False,
) -> PrepareBackendRequest:
    return PrepareBackendRequest(
        backend_id=OFFICIAL_BACKEND_ID,
        cache_root=tmp_path / "cache",
        archive_path=archive_path,
        download=download,
    )


def _write_archive(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "checkpoint.zip"
    path.write_bytes(content)
    return path


def _final_path(tmp_path: Path, backend_lock: LoadedBackendLock) -> Path:
    return (
        tmp_path
        / "cache"
        / "sha256"
        / str(backend_lock.descriptor.payload["model_artifact_set_sha256"])
    )


def _install_directory(path: Path, contents: dict[str, bytes]) -> None:
    path.mkdir(parents=True)
    for name, content in contents.items():
        (path / name).write_bytes(content)


def test_archive_installs_only_after_every_hash_matches(tmp_path: Path) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    archive_path = _write_archive(tmp_path, archive_bytes)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=archive_path),
        backend_lock=backend_lock,
    )

    expected = _final_path(tmp_path, backend_lock)
    assert outcome.status == "ready"
    assert outcome.exit_code == 0
    assert outcome.model_cache_path == expected
    assert {path.name: path.read_bytes() for path in expected.iterdir()} == COMPONENT_BYTES
    assert [
        path.name
        for path in expected.parent.iterdir()
        if path.name.startswith(f".{expected.name}.staging-")
    ] == []


def test_mutually_exclusive_acquisition_modes_fail_without_mutation(tmp_path: Path) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    archive_path = _write_archive(tmp_path, archive_bytes)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=archive_path, download=True),
        backend_lock=backend_lock,
    )

    assert (outcome.status, outcome.exit_code, outcome.model_cache_path) == (
        "integrity_failed",
        2,
        None,
    )
    assert not (tmp_path / "cache").exists()


def test_verify_only_missing_cache_never_opens_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_lock = _loaded_lock(_valid_archive_bytes())

    def unexpected_network(url: str) -> BinaryIO:
        raise AssertionError(f"verify-only opened network for {url}")

    monkeypatch.setattr(backend_prepare, "_open_download_url", unexpected_network)

    outcome = prepare_oaf_backend(_request(tmp_path), backend_lock=backend_lock)

    assert (outcome.status, outcome.exit_code, outcome.model_cache_path) == (
        "acquisition_failed",
        1,
        None,
    )


def test_verify_only_accepts_exact_cache_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_lock = _loaded_lock(_valid_archive_bytes())
    expected = _final_path(tmp_path, backend_lock)
    _install_directory(expected, COMPONENT_BYTES)
    monkeypatch.setattr(
        backend_prepare,
        "_open_download_url",
        lambda url: pytest.fail(f"unexpected network call: {url}"),
    )

    outcome = prepare_oaf_backend(_request(tmp_path), backend_lock=backend_lock)

    assert outcome.status == "ready"
    assert outcome.exit_code == 0
    assert outcome.model_cache_path == expected


@pytest.mark.parametrize(
    "conflicting_contents",
    [
        {**COMPONENT_BYTES, "model.ckpt-569400.index": b"wrong index"},
        {**COMPONENT_BYTES, "extra": b"unexpected"},
    ],
)
def test_verify_only_rejects_conflicting_cache_without_replacing_it(
    tmp_path: Path,
    conflicting_contents: dict[str, bytes],
) -> None:
    backend_lock = _loaded_lock(_valid_archive_bytes())
    expected = _final_path(tmp_path, backend_lock)
    _install_directory(expected, conflicting_contents)
    before = {path.name: path.read_bytes() for path in expected.iterdir()}

    outcome = prepare_oaf_backend(_request(tmp_path), backend_lock=backend_lock)

    assert (outcome.status, outcome.exit_code, outcome.model_cache_path) == (
        "integrity_failed",
        2,
        None,
    )
    assert {path.name: path.read_bytes() for path in expected.iterdir()} == before


@pytest.mark.parametrize(
    ("size_delta", "wrong_hash"),
    [(1, False), (0, True)],
)
def test_archive_identity_is_checked_before_zip_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size_delta: int,
    wrong_hash: bool,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(
        archive_bytes,
        archive_size=len(archive_bytes) + size_delta,
        archive_sha256="0" * 64 if wrong_hash else None,
    )
    archive_path = _write_archive(tmp_path, archive_bytes)
    monkeypatch.setattr(
        backend_prepare.zipfile,
        "ZipFile",
        lambda *args, **kwargs: pytest.fail("ZIP parsed before byte identity passed"),
    )

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=archive_path),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not _final_path(tmp_path, backend_lock).exists()


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../model.ckpt-569400.index",
        "/model.ckpt-569400.index",
        "nested/model.ckpt-569400.index",
        r"nested\model.ckpt-569400.index",
        r"C:\model.ckpt-569400.index",
        "//server/share/model.ckpt-569400.index",
        "model.ckpt-569400.index\x00hidden",
    ],
)
def test_archive_rejects_unsafe_member_paths(tmp_path: Path, unsafe_name: str) -> None:
    entries = [
        (unsafe_name if name.endswith(".index") else name, content)
        for name, content in COMPONENT_BYTES.items()
    ]
    archive_bytes = _zip_bytes(entries)
    if "\x00" in unsafe_name:
        safe_name = b"model.ckpt-569400.index"
        raw_unsafe_name = safe_name[:-2] + b"\x00x"
        assert len(raw_unsafe_name) == len(safe_name)
        archive_bytes = archive_bytes.replace(safe_name, raw_unsafe_name)
    backend_lock = _loaded_lock(archive_bytes)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not _final_path(tmp_path, backend_lock).exists()


@pytest.mark.parametrize(
    "entries",
    [
        [
            *COMPONENT_BYTES.items(),
            ("model.ckpt-569400.index", COMPONENT_BYTES["model.ckpt-569400.index"]),
        ],
        [*COMPONENT_BYTES.items(), ("extra", b"extra")],
        [
            (name, content)
            for name, content in COMPONENT_BYTES.items()
            if not name.endswith(".meta")
        ],
    ],
)
def test_archive_requires_exactly_one_of_each_component(
    tmp_path: Path,
    entries: list[tuple[str, bytes]],
) -> None:
    archive_bytes = _zip_bytes(entries)
    backend_lock = _loaded_lock(archive_bytes)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not _final_path(tmp_path, backend_lock).exists()


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo", "encrypted"])
def test_archive_rejects_non_regular_or_encrypted_members(tmp_path: Path, kind: str) -> None:
    target_name = "model.ckpt-569400.index"
    info = zipfile.ZipInfo(target_name + "/" if kind == "directory" else target_name)
    info.create_system = 3
    if kind == "directory":
        info.external_attr = (stat.S_IFDIR | 0o755) << 16
    elif kind == "symlink":
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
    elif kind == "fifo":
        info.external_attr = (stat.S_IFIFO | 0o600) << 16
    entries: list[tuple[str | zipfile.ZipInfo, bytes]] = [
        (info if name == target_name else name, content)
        for name, content in COMPONENT_BYTES.items()
    ]
    archive_bytes = _zip_bytes(entries)
    if kind == "encrypted":
        mutable = bytearray(archive_bytes)
        local = mutable.index(b"PK\x03\x04")
        central = mutable.index(b"PK\x01\x02")
        mutable[local + 6 : local + 8] = (
            int.from_bytes(mutable[local + 6 : local + 8], "little") | 1
        ).to_bytes(2, "little")
        mutable[central + 8 : central + 10] = (
            int.from_bytes(mutable[central + 8 : central + 10], "little") | 1
        ).to_bytes(2, "little")
        archive_bytes = bytes(mutable)
    backend_lock = _loaded_lock(archive_bytes)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not _final_path(tmp_path, backend_lock).exists()


def test_archive_rejects_unsupported_compression(tmp_path: Path) -> None:
    archive_bytes = _zip_bytes(list(COMPONENT_BYTES.items()), compression=zipfile.ZIP_BZIP2)
    backend_lock = _loaded_lock(archive_bytes)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2


@pytest.mark.parametrize(
    "wrong_component",
    [
        b"longer than frozen index",
        b"broken index",
    ],
)
def test_archive_rejects_component_size_or_hash_mismatch(
    tmp_path: Path,
    wrong_component: bytes,
) -> None:
    entries = [
        (name, wrong_component if name.endswith(".index") else content)
        for name, content in COMPONENT_BYTES.items()
    ]
    archive_bytes = _zip_bytes(entries)
    backend_lock = _loaded_lock(archive_bytes)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not _final_path(tmp_path, backend_lock).exists()


def test_interrupted_component_write_is_acquisition_failure_and_preserves_cache_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    marker = cache_root / "operator-owned"
    marker.write_bytes(b"keep")
    real_write = backend_prepare.os.write
    calls = 0

    def interrupted_write(fd: int, content: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interrupted")
        return real_write(fd, content)

    monkeypatch.setattr(backend_prepare.os, "write", interrupted_write)

    outcome = prepare_oaf_backend(
        replace(
            _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
            cache_root=cache_root,
        ),
        backend_lock=backend_lock,
    )

    assert outcome.status == "acquisition_failed"
    assert outcome.exit_code == 1
    assert marker.read_bytes() == b"keep"
    assert not _final_path(tmp_path, backend_lock).exists()


def test_staging_fsync_failure_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    monkeypatch.setattr(backend_prepare, "_fsync_staging_directory", lambda fd: False)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "acquisition_failed"
    assert outcome.exit_code == 1
    assert not _final_path(tmp_path, backend_lock).exists()


def test_cleanup_failure_is_not_reported_as_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    monkeypatch.setattr(
        backend_prepare,
        "_open_download_url",
        lambda url: _DownloadResponse(archive_bytes, url),
    )
    monkeypatch.setattr(backend_prepare, "_remove_download_archive", lambda *args: False)

    outcome = prepare_oaf_backend(
        _request(tmp_path, download=True),
        backend_lock=backend_lock,
    )

    assert outcome.status == "acquisition_failed"
    assert outcome.exit_code == 1
    assert not _final_path(tmp_path, backend_lock).exists()


def test_publication_failure_is_integrity_failure_and_leaves_no_final_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    monkeypatch.setattr(
        backend_prepare,
        "_rename_no_replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("publication failed")),
    )

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not _final_path(tmp_path, backend_lock).exists()


def test_concurrent_loser_verifies_winner_instead_of_replacing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    expected = _final_path(tmp_path, backend_lock)
    real_rename = backend_prepare._rename_no_replace

    def publish_winner_then_fail(
        source: str,
        target: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        _install_directory(expected, COMPONENT_BYTES)
        real_rename(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(backend_prepare, "_rename_no_replace", publish_winner_then_fail)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "ready"
    assert outcome.exit_code == 0
    assert outcome.model_cache_path == expected
    assert {path.name: path.read_bytes() for path in expected.iterdir()} == COMPONENT_BYTES


def test_final_reverification_rejects_symlink_swap_and_rolls_back_owned_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    real_rename = backend_prepare._rename_no_replace

    def swap_then_publish(
        source: str,
        target: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        os.unlink("model.ckpt-569400.index", dir_fd=src_dir_fd)
        os.symlink("model.ckpt-569400.meta", "model.ckpt-569400.index", dir_fd=src_dir_fd)
        real_rename(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(backend_prepare, "_rename_no_replace", swap_then_publish)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not _final_path(tmp_path, backend_lock).exists()


class _DownloadResponse(io.BytesIO):
    def __init__(self, content: bytes, effective_url: str):
        super().__init__(content)
        self._effective_url = effective_url

    def geturl(self) -> str:
        return self._effective_url

    def __enter__(self) -> _DownloadResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_download_uses_exact_lock_url_and_installs_verified_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    opened: list[str] = []

    def open_download(url: str) -> _DownloadResponse:
        opened.append(url)
        return _DownloadResponse(archive_bytes, url)

    monkeypatch.setattr(backend_prepare, "_open_download_url", open_download)

    outcome = prepare_oaf_backend(
        _request(tmp_path, download=True),
        backend_lock=backend_lock,
    )

    assert opened == [CHECKPOINT_URL]
    assert outcome.status == "ready"
    assert outcome.exit_code == 0


@pytest.mark.parametrize(
    "unsafe_url",
    [
        CHECKPOINT_URL.replace("https://", "http://"),
        CHECKPOINT_URL + "#fragment",
        CHECKPOINT_URL.replace("storage.googleapis.com", "user:secret@storage.googleapis.com"),
        CHECKPOINT_URL + "?mirror=1",
    ],
)
def test_download_rejects_non_allowlisted_lock_url_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_url: str,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes, checkpoint_url=unsafe_url)
    monkeypatch.setattr(
        backend_prepare,
        "_open_download_url",
        lambda url: pytest.fail(f"unsafe URL reached network: {url}"),
    )

    outcome = prepare_oaf_backend(
        _request(tmp_path, download=True),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2


def test_download_rejects_redirected_effective_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    monkeypatch.setattr(
        backend_prepare,
        "_open_download_url",
        lambda url: _DownloadResponse(archive_bytes, "https://example.invalid/checkpoint.zip"),
    )

    outcome = prepare_oaf_backend(
        _request(tmp_path, download=True),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not _final_path(tmp_path, backend_lock).exists()


def test_download_transport_failure_is_acquisition_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_lock = _loaded_lock(_valid_archive_bytes())
    monkeypatch.setattr(
        backend_prepare,
        "_open_download_url",
        lambda url: (_ for _ in ()).throw(OSError("network unavailable")),
    )

    outcome = prepare_oaf_backend(
        _request(tmp_path, download=True),
        backend_lock=backend_lock,
    )

    assert outcome.status == "acquisition_failed"
    assert outcome.exit_code == 1


def test_symlinked_cache_ancestor_is_rejected_without_touching_target(tmp_path: Path) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    target = tmp_path / "operator-directory"
    target.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.symlink_to(target, target_is_directory=True)

    outcome = prepare_oaf_backend(
        replace(
            _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
            cache_root=cache_root,
        ),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert list(target.iterdir()) == []


def test_two_concurrent_publishers_finish_with_one_exact_cache(tmp_path: Path) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    archive_path = _write_archive(tmp_path, archive_bytes)
    barrier = threading.Barrier(2)
    outcomes = []

    def prepare() -> None:
        barrier.wait()
        outcomes.append(
            prepare_oaf_backend(
                _request(tmp_path, archive_path=archive_path),
                backend_lock=backend_lock,
            )
        )

    threads = [threading.Thread(target=prepare), threading.Thread(target=prepare)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert [(outcome.status, outcome.exit_code) for outcome in outcomes] == [
        ("ready", 0),
        ("ready", 0),
    ]
    expected = _final_path(tmp_path, backend_lock)
    assert {path.name: path.read_bytes() for path in expected.iterdir()} == COMPONENT_BYTES


def test_public_prepare_rejects_forged_loaded_lock_before_filesystem_mutation(
    tmp_path: Path,
) -> None:
    archive_bytes = _valid_archive_bytes()

    outcome = _public_prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=_loaded_lock(archive_bytes),
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not (tmp_path / "cache").exists()


def test_verify_only_missing_cache_leaves_no_filesystem_state(tmp_path: Path) -> None:
    backend_lock = _loaded_lock(_valid_archive_bytes())

    outcome = prepare_oaf_backend(_request(tmp_path), backend_lock=backend_lock)

    assert outcome.status == "acquisition_failed"
    assert outcome.exit_code == 1
    assert not (tmp_path / "cache").exists()


def test_archive_rejects_dos_directory_attribute_on_basename(tmp_path: Path) -> None:
    target_name = "model.ckpt-569400.index"
    info = zipfile.ZipInfo(target_name)
    info.create_system = 0
    info.external_attr = 0x10
    archive_bytes = _zip_bytes(
        [
            (info if name == target_name else name, content)
            for name, content in COMPONENT_BYTES.items()
        ]
    )
    backend_lock = _loaded_lock(archive_bytes)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2


@pytest.mark.parametrize("data_descriptor", [False, True])
def test_archive_rejects_hidden_trailing_member_output(
    tmp_path: Path,
    data_descriptor: bool,
) -> None:
    archive_bytes = _hidden_trailing_archive_bytes(data_descriptor=data_descriptor)
    backend_lock = _loaded_lock(archive_bytes)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not _final_path(tmp_path, backend_lock).exists()


@pytest.mark.parametrize("disk_start", [1, 0xFFFF])
def test_archive_rejects_nonzero_central_member_disk_start(
    tmp_path: Path,
    disk_start: int,
) -> None:
    archive_bytes = _archive_with_central_disk_start(disk_start)
    backend_lock = _loaded_lock(archive_bytes)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not _final_path(tmp_path, backend_lock).exists()


def test_local_archive_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    archive_path = _write_archive(real_parent, archive_bytes)
    symlinked_parent = tmp_path / "linked"
    symlinked_parent.symlink_to(real_parent, target_is_directory=True)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=symlinked_parent / archive_path.name),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert not _final_path(tmp_path, backend_lock).exists()


def test_local_archive_is_parsed_from_private_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    archive_path = _write_archive(tmp_path, archive_bytes)
    original_inode = archive_path.stat().st_ino
    real_verify = backend_prepare._verify_archive_identity
    observed_inodes: list[int] = []

    def verify_private_snapshot(archive_fd: int, contract: object) -> None:
        observed_inodes.append(os.fstat(archive_fd).st_ino)
        assert observed_inodes[-1] != original_inode
        real_verify(archive_fd, contract)

    monkeypatch.setattr(backend_prepare, "_verify_archive_identity", verify_private_snapshot)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=archive_path),
        backend_lock=backend_lock,
    )

    assert outcome.status == "ready"
    assert len(observed_inodes) >= 2


def test_private_archive_snapshot_is_reverified_after_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    real_extract = backend_prepare._extract_archive

    def corrupt_after_extract(archive_fd: int, stage_fd: int, contract: object) -> None:
        real_extract(archive_fd, stage_fd, contract)
        byte = os.pread(archive_fd, 1, 10)
        os.pwrite(archive_fd, bytes([byte[0] ^ 1]), 10)

    monkeypatch.setattr(backend_prepare, "_extract_archive", corrupt_after_extract)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2


def test_atomic_publication_never_replaces_racing_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_bytes = _valid_archive_bytes()
    backend_lock = _loaded_lock(archive_bytes)
    expected = _final_path(tmp_path, backend_lock)
    real_rename = backend_prepare._rename_no_replace

    def race_then_rename(
        source: str,
        target: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        os.mkdir(target, dir_fd=dst_dir_fd)
        real_rename(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(backend_prepare, "_rename_no_replace", race_then_rename)

    outcome = prepare_oaf_backend(
        _request(tmp_path, archive_path=_write_archive(tmp_path, archive_bytes)),
        backend_lock=backend_lock,
    )

    assert outcome.status == "integrity_failed"
    assert outcome.exit_code == 2
    assert expected.is_dir()
    assert list(expected.iterdir()) == []
