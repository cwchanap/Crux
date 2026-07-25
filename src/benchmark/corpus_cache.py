from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from errno import EAGAIN, ENOSYS, ENOTSUP, EOPNOTSUPP, EWOULDBLOCK
from hashlib import sha256
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Iterator
from uuid import uuid4

from src.benchmark.r2_corpus_models import (
    CACHE_INDEX_SCHEMA,
    CacheAction,
    CacheSyncResult,
    ErrorCode,
    R2Config,
    RemoteObject,
    SimfileInventory,
    SyncError,
    format_manifest_timestamp,
)
from src.benchmark.r2_inventory import R2ObjectStore, R2StoreError

_INDEX_FILENAME = "index-v1.json"
_ENTRY_FIELDS = {
    "source_endpoint_sha256",
    "bucket",
    "key",
    "etag",
    "etag_is_weak",
    "size",
    "last_modified",
    "sha256",
    "cache_path",
}
_STRING_ENTRY_FIELDS = {
    "source_endpoint_sha256",
    "bucket",
    "key",
    "etag",
    "last_modified",
    "sha256",
    "cache_path",
}
_UNSUPPORTED_LOCK_ERRNOS = {ENOSYS, ENOTSUP, EOPNOTSUPP}
_HEAD_ERROR_CODES = frozenset({"object_head_failed", "object_metadata_invalid"})
_DOWNLOAD_ERROR_MESSAGES = {
    "artifact_write_failed": "A required cache artifact write failed.",
    "byte_count_mismatch": "Downloaded byte count does not match object metadata.",
    "cache_corrupt": "A referenced cache body is corrupt and could not be repaired.",
    "object_get_failed": "Object body read failed.",
    "object_metadata_invalid": "Object metadata is invalid.",
    "source_changed_during_sync": "Object metadata changed after inventory.",
    "weak_etag_unverifiable": "Weak ETag response metadata cannot verify the inventory.",
}
_body_install_lock = Lock()


@dataclass(frozen=True)
class CacheIndexEntry:
    source_endpoint_sha256: str
    bucket: str
    key: str
    etag: str
    etag_is_weak: bool
    size: int
    last_modified: str
    sha256: str
    cache_path: str


@dataclass(frozen=True)
class CacheValidation:
    state: str
    entry: CacheIndexEntry | None


@dataclass(frozen=True)
class _DownloadResult:
    object: RemoteObject
    action: CacheAction


@dataclass
class _OwnedTemporaryFile:
    name: str
    descriptor: int
    device: int
    inode: int
    closed: bool = False

    def close(self) -> None:
        if not self.closed:
            os.close(self.descriptor)
            self.closed = True


@dataclass(frozen=True)
class _PreparedDownload:
    remote: RemoteObject
    validation: CacheValidation
    temporary: _OwnedTemporaryFile
    sha256: str
    size: int


@dataclass(frozen=True)
class _PinnedContentDirectories:
    cache_dir: Path
    cache_fd: int
    sha256_fd: int
    incoming_fd: int


class _DownloadRejected(Exception):
    def __init__(self, error: SyncError):
        super().__init__(error.code)
        self.error = error


class CacheIndexStore:
    def __init__(
        self,
        cache_dir: Path,
        entries: dict[tuple[str, str, str], CacheIndexEntry],
    ) -> None:
        self.cache_dir = cache_dir
        self._entries = entries
        self._checkpoint_lock = Lock()

    @classmethod
    def load(cls, cache_dir: Path) -> CacheIndexStore:
        index_path = cache_dir / _INDEX_FILENAME
        try:
            index_metadata = os.stat(index_path, follow_symlinks=False)
        except FileNotFoundError:
            return cls(cache_dir, {})
        except OSError:
            raise ValueError("invalid cache index JSON") from None
        if not stat.S_ISREG(index_metadata.st_mode):
            raise ValueError("invalid cache index JSON")
        document = _load_index_document(index_path)
        entries = _validated_entries(document)
        return cls(cache_dir, entries)

    def get(self, endpoint_hash: str, bucket: str, key: str) -> CacheIndexEntry | None:
        with self._checkpoint_lock:
            return self._entries.get((endpoint_hash, bucket, key))

    def checkpoint(self, entry: CacheIndexEntry) -> None:
        entry = _validated_entry(asdict(entry))
        with self._checkpoint_lock:
            entries = dict(self._entries)
            entries[(entry.source_endpoint_sha256, entry.bucket, entry.key)] = entry
            self._publish_locked(entries)
            self._entries = entries

    def _publish_locked(self, entries: dict[tuple[str, str, str], CacheIndexEntry]) -> None:
        _ensure_durable_directory(self.cache_dir)
        payload = {
            "schema_version": CACHE_INDEX_SCHEMA,
            "entries": [
                asdict(entry)
                for entry in sorted(
                    entries.values(),
                    key=lambda item: (
                        item.source_endpoint_sha256,
                        item.bucket,
                        item.key,
                    ),
                )
            ],
        }
        content = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        index_path = self.cache_dir / _INDEX_FILENAME
        temporary_path = self.cache_dir / f".{_INDEX_FILENAME}.{uuid4().hex}.tmp"
        temporary_created = False
        try:
            with temporary_path.open("xb") as file:
                temporary_created = True
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, index_path)
            _fsync_directory(self.cache_dir)
        except Exception:
            if temporary_created:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise


def is_selected(key: str) -> bool:
    basename = key.rsplit("/", 1)[-1].lower()
    lowered = key.lower()
    return basename == "set.def" or lowered.endswith((".dtx", ".txt"))


def validate_cached_body(
    cache_dir: Path,
    entry: CacheIndexEntry | None,
) -> CacheValidation:
    if entry is None:
        return CacheValidation("remote_changed", None)
    shard_fd: int | None = None
    final_fd: int | None = None
    try:
        with _open_existing_content_root(cache_dir) as (cache_fd, sha256_fd):
            shard_name = entry.sha256[:2]
            shard_fd = _open_directory_at(sha256_fd, shard_name)
            _verify_directory_binding(sha256_fd, shard_name, shard_fd)
            final_fd = _open_regular_file_at(shard_fd, entry.sha256)
            digest, size = _hash_fd(final_fd)
            _verify_regular_file_binding(shard_fd, entry.sha256, final_fd)
            _verify_directory_binding(sha256_fd, shard_name, shard_fd)
            _verify_directory_binding(cache_fd, "sha256", sha256_fd)
    except FileNotFoundError:
        return CacheValidation("missing", entry)
    except OSError:
        return CacheValidation("sha256_mismatch", entry)
    finally:
        if final_fd is not None:
            os.close(final_fd)
        if shard_fd is not None:
            os.close(shard_fd)
    if size != entry.size:
        return CacheValidation("size_mismatch", entry)
    if digest != entry.sha256:
        return CacheValidation("sha256_mismatch", entry)
    return CacheValidation("verified", entry)


def sync_cache(
    simfiles: tuple[SimfileInventory, ...],
    store: R2ObjectStore,
    index: CacheIndexStore,
    config: R2Config,
    dry_run: bool,
    item_progress: Callable[[int, int, int], None] | None = None,
) -> CacheSyncResult:
    object_results: dict[tuple[int, int], RemoteObject] = {}
    actions: dict[tuple[int, int], CacheAction] = {}
    misses: list[tuple[int, int, RemoteObject, CacheValidation]] = []
    total_selected = sum(
        is_selected(remote.key) for simfile in simfiles for remote in simfile.objects
    )
    completed_selected = 0
    completed_bytes = 0
    progressed: set[tuple[int, int]] = set()

    def record_progress(identity: tuple[int, int], byte_count: int) -> None:
        nonlocal completed_selected, completed_bytes
        if identity in progressed:
            return
        progressed.add(identity)
        completed_selected += 1
        completed_bytes += byte_count
        if item_progress is not None:
            item_progress(completed_selected, total_selected, completed_bytes)

    for simfile_index, simfile in enumerate(simfiles):
        for object_index, remote in enumerate(simfile.objects):
            identity = (simfile_index, object_index)
            if not is_selected(remote.key):
                object_results[identity] = remote
                continue
            if any(error.code in _HEAD_ERROR_CODES for error in remote.errors):
                failed = replace(remote, cache_status="failed")
                object_results[identity] = failed
                actions[identity] = CacheAction(
                    remote.key,
                    "failed",
                    0,
                    errors=remote.errors,
                )
                record_progress(identity, 0)
                continue

            entry = index.get(config.source_endpoint_sha256, config.bucket, remote.key)
            matching_entry = entry if _remote_identity_matches(entry, remote, config) else None
            validation = validate_cached_body(index.cache_dir, matching_entry)
            if validation.state == "verified":
                assert validation.entry is not None
                verified = replace(
                    remote,
                    cache_status="verified",
                    sha256=validation.entry.sha256,
                    cache_path=validation.entry.cache_path,
                )
                object_results[identity] = verified
                actions[identity] = CacheAction(remote.key, "cache_hit", 0)
                record_progress(identity, 0)
                continue
            if dry_run:
                object_results[identity] = remote
                actions[identity] = CacheAction(
                    remote.key,
                    "planned",
                    remote.size,
                    validation.state,
                )
                record_progress(identity, remote.size)
                continue
            misses.append((simfile_index, object_index, remote, validation))

    if misses:
        ordered_misses = sorted(
            misses,
            key=lambda item: item[3].state not in {"size_mismatch", "sha256_mismatch"},
        )
        try:
            with _open_content_directories(index.cache_dir) as directories:
                with ThreadPoolExecutor(max_workers=config.download_concurrency) as executor:
                    for batch_start in range(
                        0,
                        len(ordered_misses),
                        config.download_concurrency,
                    ):
                        batch = ordered_misses[
                            batch_start : batch_start + config.download_concurrency
                        ]
                        prepared: list[tuple[tuple[int, int], _PreparedDownload]] = []
                        try:
                            futures = {
                                executor.submit(
                                    _prepare_download,
                                    store,
                                    directories,
                                    remote,
                                    validation,
                                ): (simfile_index, object_index)
                                for simfile_index, object_index, remote, validation in batch
                            }
                            for future in as_completed(futures):
                                identity = futures[future]
                                result = future.result()
                                if isinstance(result, _PreparedDownload):
                                    prepared.append((identity, result))
                                    record_progress(identity, result.size)
                                else:
                                    object_results[identity] = result.object
                                    actions[identity] = result.action
                                    record_progress(identity, result.action.bytes)
                            prepared.sort(key=lambda item: item[0])
                            for identity, result in _install_prepared_downloads(
                                prepared,
                                directories,
                                index,
                                config,
                            ).items():
                                object_results[identity] = result.object
                                actions[identity] = result.action
                        finally:
                            for _, download in prepared:
                                download.temporary.close()
        except (OSError, ValueError):
            error = _download_error("artifact_write_failed", None)
            for simfile_index, object_index, remote, validation in misses:
                identity = (simfile_index, object_index)
                if identity in object_results:
                    continue
                result = _failed_download(
                    remote,
                    validation,
                    (replace(error, object_key=remote.key),),
                )
                object_results[identity] = result.object
                actions[identity] = result.action
                record_progress(identity, result.action.bytes)

    rebuilt_simfiles = tuple(
        _rebuild_simfile(simfile, simfile_index, object_results)
        for simfile_index, simfile in enumerate(simfiles)
    )
    ordered_actions = tuple(action for _, action in sorted(actions.items()))
    return CacheSyncResult(rebuilt_simfiles, ordered_actions)


def _remote_identity_matches(
    entry: CacheIndexEntry | None,
    remote: RemoteObject,
    config: R2Config,
) -> bool:
    return entry is not None and (
        entry.source_endpoint_sha256,
        entry.bucket,
        entry.key,
        entry.etag,
        entry.etag_is_weak,
        entry.size,
        entry.last_modified,
    ) == (
        config.source_endpoint_sha256,
        config.bucket,
        remote.key,
        remote.etag,
        remote.etag_is_weak,
        remote.size,
        format_manifest_timestamp(remote.last_modified),
    )


def _prepare_download(
    store: R2ObjectStore,
    directories: _PinnedContentDirectories,
    remote: RemoteObject,
    validation: CacheValidation,
) -> _PreparedDownload | _DownloadResult:
    temporary_name = f"{uuid4().hex}.tmp"
    temporary: _OwnedTemporaryFile | None = None
    errors: list[SyncError] = []
    byte_count = 0
    digest = sha256()
    try:
        _verify_content_directories(directories)
        if_match = None if remote.etag_is_weak else remote.etag
        with store.open_object(remote.key, if_match) as download:
            weak_error = _weak_response_error(remote, download)
            if weak_error is not None:
                raise _DownloadRejected(weak_error)
            temporary = _create_temporary_file(directories.incoming_fd, temporary_name)
            writer_descriptor = os.dup(temporary.descriptor)
            try:
                incoming = os.fdopen(writer_descriptor, "wb")
            except OSError:
                os.close(writer_descriptor)
                raise
            with incoming:
                while chunk := download.body.read(1024 * 1024):
                    incoming.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                incoming.flush()
                os.fsync(temporary.descriptor)
                _verify_regular_file_binding(
                    directories.incoming_fd,
                    temporary_name,
                    temporary.descriptor,
                )
        _verify_content_directories(directories)
        if byte_count != remote.size:
            raise _DownloadRejected(_download_error("byte_count_mismatch", remote.key))
    except _DownloadRejected as error:
        errors.append(error.error)
    except R2StoreError as error:
        code: ErrorCode = (
            error.code
            if error.code
            in {
                "object_get_failed",
                "object_metadata_invalid",
                "source_changed_during_sync",
            }
            else "object_get_failed"
        )
        errors.append(_download_error(code, remote.key))
    except (OSError, ValueError):
        errors.append(_download_error("artifact_write_failed", remote.key))

    if errors:
        if temporary is not None:
            if not _unlink_owned_temporary(
                directories.incoming_fd,
                temporary,
                missing_ok=False,
            ):
                errors.append(_download_error("artifact_write_failed", remote.key))
            temporary.close()
        return _failed_download(remote, validation, tuple(errors))
    assert temporary is not None
    return _PreparedDownload(
        remote,
        validation,
        temporary,
        digest.hexdigest(),
        byte_count,
    )


def _install_prepared_downloads(
    prepared: list[tuple[tuple[int, int], _PreparedDownload]],
    directories: _PinnedContentDirectories,
    index: CacheIndexStore,
    config: R2Config,
) -> dict[tuple[int, int], _DownloadResult]:
    groups: dict[str, list[tuple[tuple[int, int], _PreparedDownload]]] = {}
    for identity, download in prepared:
        groups.setdefault(download.sha256, []).append((identity, download))

    results: dict[tuple[int, int], _DownloadResult] = {}
    for group in groups.values():
        results.update(_install_digest_group(group, directories, index, config))
    return results


def _install_digest_group(
    group: list[tuple[tuple[int, int], _PreparedDownload]],
    directories: _PinnedContentDirectories,
    index: CacheIndexStore,
    config: R2Config,
) -> dict[tuple[int, int], _DownloadResult]:
    content_sha256 = group[0][1].sha256
    shard_name = content_sha256[:2]
    cache_path = f"sha256/{shard_name}/{content_sha256}"
    shard_fd: int | None = None
    final_fd: int | None = None
    install_error: SyncError | None = None
    moved_temporary: _OwnedTemporaryFile | None = None
    cleanup_errors: set[tuple[int, int]] = set()
    results: dict[tuple[int, int], _DownloadResult] = {}
    try:
        with _body_install_lock:
            try:
                _verify_content_directories(directories)
                shard_fd = _open_or_create_directory_at(directories.sha256_fd, shard_name)
                _verify_directory_binding(directories.sha256_fd, shard_name, shard_fd)
                if os.fstat(directories.incoming_fd).st_dev != os.fstat(shard_fd).st_dev:
                    raise _DownloadRejected(
                        _download_error("artifact_write_failed", group[0][1].remote.key)
                    )

                try:
                    final_fd = _open_regular_file_at(shard_fd, content_sha256)
                except FileNotFoundError:
                    replacement = group[0][1]
                else:
                    existing_digest, existing_size = _hash_fd(final_fd)
                    if existing_digest == content_sha256 and existing_size == group[0][1].size:
                        replacement = None
                    else:
                        replacement = next(
                            (
                                download
                                for _, download in group
                                if _can_repair_content_address(download, cache_path)
                            ),
                            None,
                        )
                        if replacement is None:
                            raise _DownloadRejected(
                                _download_error(
                                    "artifact_write_failed",
                                    group[0][1].remote.key,
                                )
                            )
                        os.close(final_fd)
                        final_fd = None

                if replacement is not None:
                    _verify_owned_temporary_binding(
                        directories.incoming_fd,
                        replacement.temporary,
                    )
                    os.replace(
                        replacement.temporary.name,
                        content_sha256,
                        src_dir_fd=directories.incoming_fd,
                        dst_dir_fd=shard_fd,
                    )
                    moved_temporary = replacement.temporary
                    final_fd = _open_regular_file_at(shard_fd, content_sha256)
                    _verify_owned_descriptor(replacement.temporary, final_fd)
                    installed_digest, installed_size = _hash_fd(final_fd)
                    if installed_digest != content_sha256 or installed_size != replacement.size:
                        raise _DownloadRejected(
                            _download_error("artifact_write_failed", replacement.remote.key)
                        )

                assert final_fd is not None
                os.fsync(final_fd)
                _verify_regular_file_binding(shard_fd, content_sha256, final_fd)
                os.fsync(shard_fd)
                _verify_installed_content_path(
                    directories,
                    shard_name,
                    shard_fd,
                    content_sha256,
                    final_fd,
                )
            except (_DownloadRejected, OSError, ValueError) as error:
                install_error = (
                    error.error
                    if isinstance(error, _DownloadRejected)
                    else _download_error("artifact_write_failed", group[0][1].remote.key)
                )

            for identity, download in group:
                if not _unlink_owned_temporary(
                    directories.incoming_fd,
                    download.temporary,
                    missing_ok=download.temporary is moved_temporary,
                ):
                    cleanup_errors.add(identity)
                download.temporary.close()

            binding_error: SyncError | None = None
            if install_error is None:
                try:
                    assert shard_fd is not None
                    assert final_fd is not None
                    _verify_installed_content_path(
                        directories,
                        shard_name,
                        shard_fd,
                        content_sha256,
                        final_fd,
                    )
                except (OSError, ValueError):
                    binding_error = _download_error(
                        "artifact_write_failed",
                        group[0][1].remote.key,
                    )

            for identity, download in group:
                errors: list[SyncError] = []
                if install_error is not None:
                    errors.append(replace(install_error, object_key=download.remote.key))
                if identity in cleanup_errors:
                    errors.append(_download_error("artifact_write_failed", download.remote.key))
                if binding_error is not None:
                    errors.append(replace(binding_error, object_key=download.remote.key))
                if errors:
                    results[identity] = _failed_download(
                        download.remote,
                        download.validation,
                        tuple(errors),
                    )
                    continue
                try:
                    assert shard_fd is not None
                    assert final_fd is not None
                    _verify_installed_content_path(
                        directories,
                        shard_name,
                        shard_fd,
                        content_sha256,
                        final_fd,
                    )
                except (OSError, ValueError):
                    results[identity] = _failed_download(
                        download.remote,
                        download.validation,
                        (_download_error("artifact_write_failed", download.remote.key),),
                    )
                    continue
                results[identity] = _checkpoint_download(
                    download,
                    index,
                    config,
                    cache_path,
                )
    finally:
        for _, download in group:
            download.temporary.close()
        if final_fd is not None:
            os.close(final_fd)
        if shard_fd is not None:
            os.close(shard_fd)
    return results


def _checkpoint_download(
    download: _PreparedDownload,
    index: CacheIndexStore,
    config: R2Config,
    cache_path: str,
) -> _DownloadResult:
    remote = download.remote
    entry = CacheIndexEntry(
        source_endpoint_sha256=config.source_endpoint_sha256,
        bucket=config.bucket,
        key=remote.key,
        etag=remote.etag,
        etag_is_weak=remote.etag_is_weak,
        size=download.size,
        last_modified=format_manifest_timestamp(remote.last_modified),
        sha256=download.sha256,
        cache_path=cache_path,
    )
    try:
        index.checkpoint(entry)
    except (OSError, ValueError):
        return _failed_download(
            remote,
            download.validation,
            (_download_error("artifact_write_failed", remote.key),),
        )
    verified = replace(
        remote,
        cache_status="verified",
        sha256=download.sha256,
        cache_path=cache_path,
    )
    return _DownloadResult(
        verified,
        CacheAction(
            remote.key,
            "downloaded",
            download.size,
            download.validation.state,
        ),
    )


def _can_repair_content_address(
    download: _PreparedDownload,
    cache_path: str,
) -> bool:
    validation = download.validation
    return (
        validation.state in {"size_mismatch", "sha256_mismatch"}
        and validation.entry is not None
        and validation.entry.cache_path == cache_path
    )


def _weak_response_error(remote: RemoteObject, download: object) -> SyncError | None:
    if not remote.etag_is_weak:
        return None
    etag = getattr(download, "etag", None)
    etag_is_weak = getattr(download, "etag_is_weak", None)
    size = getattr(download, "size", None)
    last_modified = getattr(download, "last_modified", None)
    if None in (etag, etag_is_weak, size, last_modified):
        return _download_error("weak_etag_unverifiable", remote.key)
    if (
        etag != remote.etag
        or etag_is_weak is not True
        or size != remote.size
        or format_manifest_timestamp(last_modified)
        != format_manifest_timestamp(remote.last_modified)
    ):
        return _download_error("source_changed_during_sync", remote.key)
    return None


def _failed_download(
    remote: RemoteObject,
    validation: CacheValidation,
    operational_errors: tuple[SyncError, ...],
) -> _DownloadResult:
    errors = list(operational_errors)
    if validation.state in {"missing", "size_mismatch", "sha256_mismatch"}:
        errors.insert(0, _download_error("cache_corrupt", remote.key))
    unique_errors = {
        (error.scope, error.code, error.message, error.object_key): error for error in errors
    }
    ordered_errors = tuple(sorted(unique_errors.values(), key=_error_sort_key))
    failed = replace(remote, cache_status="failed", errors=remote.errors + ordered_errors)
    return _DownloadResult(
        failed,
        CacheAction(
            remote.key,
            "failed",
            0,
            validation.state,
            ordered_errors,
        ),
    )


def _download_error(code: ErrorCode, key: str | None) -> SyncError:
    return SyncError(
        "cache" if code in {"artifact_write_failed", "cache_corrupt"} else "object",
        code,
        _DOWNLOAD_ERROR_MESSAGES[code],
        key,
    )


def _rebuild_simfile(
    source: SimfileInventory,
    simfile_index: int,
    object_results: dict[tuple[int, int], RemoteObject],
) -> SimfileInventory:
    objects = tuple(
        object_results[(simfile_index, object_index)] for object_index in range(len(source.objects))
    )
    selected = tuple(remote for remote in objects if is_selected(remote.key))
    failed = tuple(remote for remote in selected if remote.cache_status == "failed")
    if source.sync_status == "empty":
        status = "empty"
    elif selected and len(failed) == len(selected):
        status = "failed"
    elif failed or source.sync_status in {"partial", "failed"}:
        status = "partial"
    else:
        status = source.sync_status
    errors = {
        (error.scope, error.code, error.message, error.object_key): error
        for error in source.sync_errors
    }
    for remote in objects:
        for error in remote.errors:
            errors[(error.scope, error.code, error.message, error.object_key)] = error
    return replace(
        source,
        objects=objects,
        sync_status=status,
        sync_errors=tuple(sorted(errors.values(), key=_error_sort_key)),
    )


@contextmanager
def _open_existing_content_root(cache_dir: Path) -> Iterator[tuple[int, int]]:
    cache_fd = _open_directory_path(cache_dir)
    sha256_fd: int | None = None
    try:
        _verify_cache_directory_binding(cache_dir, cache_fd)
        sha256_fd = _open_directory_at(cache_fd, "sha256")
        _verify_directory_binding(cache_fd, "sha256", sha256_fd)
        yield cache_fd, sha256_fd
        _verify_cache_directory_binding(cache_dir, cache_fd)
        _verify_directory_binding(cache_fd, "sha256", sha256_fd)
    finally:
        if sha256_fd is not None:
            os.close(sha256_fd)
        os.close(cache_fd)


@contextmanager
def _open_content_directories(
    cache_dir: Path,
) -> Iterator[_PinnedContentDirectories]:
    _ensure_durable_directory(cache_dir)
    cache_fd = _open_directory_path(cache_dir)
    sha256_fd: int | None = None
    incoming_fd: int | None = None
    try:
        sha256_fd = _open_or_create_directory_at(cache_fd, "sha256")
        incoming_fd = _open_or_create_directory_at(sha256_fd, ".incoming")
        directories = _PinnedContentDirectories(
            cache_dir,
            cache_fd,
            sha256_fd,
            incoming_fd,
        )
        _verify_content_directories(directories)
        yield directories
    finally:
        if incoming_fd is not None:
            os.close(incoming_fd)
        if sha256_fd is not None:
            os.close(sha256_fd)
        os.close(cache_fd)


def _open_directory_path(path: Path) -> int:
    return os.open(path, _directory_open_flags())


def _open_directory_at(parent_fd: int, name: str) -> int:
    descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise OSError("cache content directory is unavailable")
    return descriptor


def _open_or_create_directory_at(parent_fd: int, name: str) -> int:
    created = False
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    descriptor = _open_directory_at(parent_fd, name)
    try:
        _verify_directory_binding(parent_fd, name, descriptor)
        if created:
            os.fsync(parent_fd)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _directory_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OSError(ENOTSUP, "no-follow directory descriptors are unavailable")
    return os.O_RDONLY | no_follow | directory


def _verify_content_directories(directories: _PinnedContentDirectories) -> None:
    _verify_cache_directory_binding(directories.cache_dir, directories.cache_fd)
    _verify_directory_binding(
        directories.cache_fd,
        "sha256",
        directories.sha256_fd,
    )
    _verify_directory_binding(
        directories.sha256_fd,
        ".incoming",
        directories.incoming_fd,
    )


def _verify_cache_directory_binding(cache_dir: Path, descriptor: int) -> None:
    cache_stat = os.stat(cache_dir, follow_symlinks=False)
    if not stat.S_ISDIR(cache_stat.st_mode) or not _same_inode(cache_stat, os.fstat(descriptor)):
        raise OSError("cache directory binding changed")


def _verify_directory_binding(parent_fd: int, name: str, descriptor: int) -> None:
    path_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    descriptor_stat = os.fstat(descriptor)
    if not stat.S_ISDIR(path_stat.st_mode) or not _same_inode(path_stat, descriptor_stat):
        raise OSError("cache content directory binding changed")


def _verify_regular_file_binding(parent_fd: int, name: str, descriptor: int) -> None:
    path_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    descriptor_stat = os.fstat(descriptor)
    if not stat.S_ISREG(path_stat.st_mode) or not _same_inode(path_stat, descriptor_stat):
        raise OSError("cache content file binding changed")


def _verify_owned_descriptor(temporary: _OwnedTemporaryFile, descriptor: int) -> None:
    owned_stat = os.fstat(temporary.descriptor)
    descriptor_stat = os.fstat(descriptor)
    if (
        not stat.S_ISREG(owned_stat.st_mode)
        or (owned_stat.st_dev, owned_stat.st_ino) != (temporary.device, temporary.inode)
        or not _same_inode(owned_stat, descriptor_stat)
    ):
        raise OSError("cache temporary file ownership changed")


def _verify_owned_temporary_binding(
    parent_fd: int,
    temporary: _OwnedTemporaryFile,
) -> None:
    _verify_regular_file_binding(parent_fd, temporary.name, temporary.descriptor)
    descriptor_stat = os.fstat(temporary.descriptor)
    if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
        temporary.device,
        temporary.inode,
    ):
        raise OSError("cache temporary file ownership changed")


def _verify_installed_content_path(
    directories: _PinnedContentDirectories,
    shard_name: str,
    shard_fd: int,
    content_sha256: str,
    final_fd: int,
) -> None:
    _verify_content_directories(directories)
    _verify_directory_binding(directories.sha256_fd, shard_name, shard_fd)
    _verify_regular_file_binding(shard_fd, content_sha256, final_fd)


def _same_inode(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _open_regular_file_at(parent_fd: int, name: str) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise OSError(ENOTSUP, "no-follow file descriptors are unavailable")
    descriptor = os.open(name, os.O_RDONLY | no_follow, dir_fd=parent_fd)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise OSError("cache content body is not a regular file")
    return descriptor


def _create_temporary_file(parent_fd: int, name: str) -> _OwnedTemporaryFile:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise OSError(ENOTSUP, "no-follow file descriptors are unavailable")
    descriptor = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow,
        0o600,
        dir_fd=parent_fd,
    )
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise OSError("cache temporary file is not a regular file")
        return _OwnedTemporaryFile(
            name,
            descriptor,
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        )
    except Exception:
        os.close(descriptor)
        raise


def _unlink_owned_temporary(
    parent_fd: int,
    temporary: _OwnedTemporaryFile,
    *,
    missing_ok: bool,
) -> bool:
    try:
        _verify_owned_temporary_binding(parent_fd, temporary)
    except FileNotFoundError:
        return missing_ok
    except OSError:
        return False
    try:
        os.unlink(temporary.name, dir_fd=parent_fd)
    except OSError:
        return False
    return True


def _hash_fd(descriptor: int) -> tuple[str, int]:
    digest = sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _error_sort_key(error: SyncError) -> tuple[str, str, str, str]:
    return error.scope, error.code, error.object_key or "", error.message


@contextmanager
def cache_writer_lock(cache_dir: Path) -> Iterator[None]:
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("unsupported_platform") from exc

    _ensure_durable_directory(cache_dir)
    handle = _open_lock_file(cache_dir / ".index-v1.lock")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(_lock_error_message(exc)) from exc
        yield
    finally:
        handle.close()


def _load_index_document(index_path: Path) -> object:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("invalid cache index JSON")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            index_path,
            os.O_RDONLY | os.O_NONBLOCK | no_follow,
        )
        _verify_index_binding(index_path, descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        _verify_index_binding(index_path, descriptor)
        content = b"".join(chunks).decode("utf-8", errors="strict")
        return json.loads(content, object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, ValueError):
        raise ValueError("invalid cache index JSON") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_index_binding(index_path: Path, descriptor: int) -> None:
    path_metadata = os.stat(index_path, follow_symlinks=False)
    descriptor_metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(path_metadata.st_mode)
        or not stat.S_ISREG(descriptor_metadata.st_mode)
        or not _same_inode(path_metadata, descriptor_metadata)
    ):
        raise OSError("cache index binding changed")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise _DuplicateObjectField
        result[name] = value
    return result


def _validated_entries(document: object) -> dict[tuple[str, str, str], CacheIndexEntry]:
    if type(document) is not dict:
        raise ValueError("invalid cache index document")
    if set(document) != {"schema_version", "entries"}:
        raise ValueError("invalid cache index document")
    if document["schema_version"] != CACHE_INDEX_SCHEMA:
        raise ValueError("unsupported cache index schema_version")
    raw_entries = document["entries"]
    if type(raw_entries) is not list:
        raise ValueError("invalid cache index entries")

    entries: dict[tuple[str, str, str], CacheIndexEntry] = {}
    for raw_entry in raw_entries:
        entry = _validated_entry(raw_entry)
        identity = (entry.source_endpoint_sha256, entry.bucket, entry.key)
        if identity in entries:
            raise ValueError("duplicate cache index entry")
        entries[identity] = entry
    return entries


def _validated_entry(raw_entry: object) -> CacheIndexEntry:
    if type(raw_entry) is not dict or set(raw_entry) != _ENTRY_FIELDS:
        raise ValueError("invalid cache index entry")
    if any(type(raw_entry[name]) is not str for name in _STRING_ENTRY_FIELDS):
        raise ValueError("invalid cache index entry")
    if any(not _is_utf8_encodable(raw_entry[name]) for name in _STRING_ENTRY_FIELDS):
        raise ValueError("invalid cache index entry")
    if type(raw_entry["etag_is_weak"]) is not bool:
        raise ValueError("invalid cache index entry")
    if type(raw_entry["size"]) is not int or raw_entry["size"] < 0:
        raise ValueError("invalid cache index entry")

    source_endpoint_sha256 = raw_entry["source_endpoint_sha256"]
    sha256 = raw_entry["sha256"]
    if not _is_lower_hex_sha256(source_endpoint_sha256) or not _is_lower_hex_sha256(sha256):
        raise ValueError("invalid cache index entry")
    last_modified = raw_entry["last_modified"]
    if not _is_canonical_utc_timestamp(last_modified):
        raise ValueError("invalid cache index entry")
    cache_path = raw_entry["cache_path"]
    _validate_relative_cache_path(cache_path, sha256)

    return CacheIndexEntry(
        source_endpoint_sha256=source_endpoint_sha256,
        bucket=raw_entry["bucket"],
        key=raw_entry["key"],
        etag=raw_entry["etag"],
        etag_is_weak=raw_entry["etag_is_weak"],
        size=raw_entry["size"],
        last_modified=last_modified,
        sha256=sha256,
        cache_path=cache_path,
    )


def _is_lower_hex_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_utf8_encodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_canonical_utc_timestamp(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc and format_manifest_timestamp(parsed) == value


def _validate_relative_cache_path(cache_path: str, sha256: str) -> None:
    parts = cache_path.split("/")
    if cache_path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("invalid cache index entry")
    path = PurePosixPath(cache_path)
    expected = PurePosixPath("sha256", sha256[:2], sha256)
    if path.is_absolute() or path != expected:
        raise ValueError("invalid cache index entry")


def _ensure_durable_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError("cache directory is unavailable")
        return

    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise ValueError("cache directory is unavailable")
        current = parent
    if not current.is_dir():
        raise ValueError("cache directory is unavailable")
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        if not directory.is_dir():
            raise ValueError("cache directory is unavailable")
        _fsync_directory(directory.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_lock_file(lock_path: Path):
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError("unsupported_platform")
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_APPEND | os.O_CREAT | no_follow,
            0o600,
        )
    except OSError as exc:
        raise RuntimeError(_lock_open_error_message(exc)) from exc
    try:
        return os.fdopen(descriptor, "a+b")
    except OSError as exc:
        os.close(descriptor)
        raise RuntimeError(_lock_open_error_message(exc)) from exc


def _lock_error_message(exc: OSError) -> str:
    if exc.errno in {EAGAIN, EWOULDBLOCK}:
        return "cache_locked"
    if exc.errno in _UNSUPPORTED_LOCK_ERRNOS:
        return "unsupported_platform"
    return "cache_lock_failed"


def _lock_open_error_message(exc: OSError) -> str:
    if exc.errno in _UNSUPPORTED_LOCK_ERRNOS:
        return "unsupported_platform"
    return "cache_lock_failed"


class _DuplicateObjectField(ValueError):
    pass
