from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
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
        if not index_path.exists():
            return cls(cache_dir, {})
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
    if _path_has_symlink(cache_dir, PurePosixPath(entry.cache_path)):
        return CacheValidation("sha256_mismatch", entry)
    path = cache_dir / PurePosixPath(entry.cache_path)
    if not path.is_file():
        return CacheValidation("missing", entry)
    try:
        digest, size = _hash_file(path)
    except OSError:
        return CacheValidation("sha256_mismatch", entry)
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
) -> CacheSyncResult:
    object_results: dict[tuple[int, int], RemoteObject] = {}
    actions: dict[tuple[int, int], CacheAction] = {}
    misses: list[tuple[int, int, RemoteObject, CacheValidation]] = []

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
                continue
            if dry_run:
                object_results[identity] = remote
                actions[identity] = CacheAction(
                    remote.key,
                    "planned",
                    remote.size,
                    validation.state,
                )
                continue
            misses.append((simfile_index, object_index, remote, validation))

    if misses:
        with ThreadPoolExecutor(max_workers=config.download_concurrency) as executor:
            futures = [
                (
                    (simfile_index, object_index),
                    executor.submit(
                        _download_one,
                        store,
                        index,
                        config,
                        remote,
                        validation,
                    ),
                )
                for simfile_index, object_index, remote, validation in misses
            ]
            for identity, future in futures:
                result = future.result()
                object_results[identity] = result.object
                actions[identity] = result.action

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


def _download_one(
    store: R2ObjectStore,
    index: CacheIndexStore,
    config: R2Config,
    remote: RemoteObject,
    validation: CacheValidation,
) -> _DownloadResult:
    incoming_path: Path | None = None
    incoming_created = False
    errors: list[SyncError] = []
    byte_count = 0
    content_sha256 = ""
    cache_path = ""
    try:
        incoming_dir = index.cache_dir / "sha256" / ".incoming"
        _ensure_safe_content_directory(index.cache_dir, incoming_dir)
        incoming_path = incoming_dir / f"{uuid4().hex}.tmp"
        digest = sha256()
        if_match = None if remote.etag_is_weak else remote.etag
        with store.open_object(remote.key, if_match) as download:
            weak_error = _weak_response_error(remote, download)
            if weak_error is not None:
                raise _DownloadRejected(weak_error)
            incoming = incoming_path.open("xb")
            incoming_created = True
            with incoming:
                while chunk := download.body.read(1024 * 1024):
                    incoming.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                incoming.flush()
                os.fsync(incoming.fileno())

        if byte_count != remote.size:
            raise _DownloadRejected(_download_error("byte_count_mismatch", remote.key))

        content_sha256 = digest.hexdigest()
        cache_path = f"sha256/{content_sha256[:2]}/{content_sha256}"
        shard_dir = index.cache_dir / "sha256" / content_sha256[:2]
        _ensure_safe_content_directory(index.cache_dir, shard_dir)
        if incoming_dir.stat().st_dev != shard_dir.stat().st_dev:
            raise _DownloadRejected(_download_error("artifact_write_failed", remote.key))
        final_path = index.cache_dir / PurePosixPath(cache_path)
        with _body_install_lock:
            if _path_has_symlink(index.cache_dir, PurePosixPath(cache_path)):
                raise _DownloadRejected(_download_error("artifact_write_failed", remote.key))
            if final_path.exists():
                existing_digest, existing_size = _hash_file(final_path)
                if existing_digest != content_sha256 or existing_size != byte_count:
                    if (
                        validation.state not in {"size_mismatch", "sha256_mismatch"}
                        or validation.entry is None
                        or validation.entry.cache_path != cache_path
                    ):
                        raise _DownloadRejected(
                            _download_error("artifact_write_failed", remote.key)
                        )
                    os.replace(incoming_path, final_path)
            else:
                os.replace(incoming_path, final_path)
            _fsync_directory(shard_dir)
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

    if incoming_created and incoming_path is not None:
        try:
            incoming_path.unlink(missing_ok=True)
        except OSError:
            errors.append(_download_error("artifact_write_failed", remote.key))
    if errors:
        return _failed_download(remote, validation, tuple(errors))

    entry = CacheIndexEntry(
        source_endpoint_sha256=config.source_endpoint_sha256,
        bucket=config.bucket,
        key=remote.key,
        etag=remote.etag,
        etag_is_weak=remote.etag_is_weak,
        size=byte_count,
        last_modified=format_manifest_timestamp(remote.last_modified),
        sha256=content_sha256,
        cache_path=cache_path,
    )
    try:
        index.checkpoint(entry)
    except (OSError, ValueError):
        return _failed_download(
            remote,
            validation,
            (_download_error("artifact_write_failed", remote.key),),
        )
    verified = replace(
        remote,
        cache_status="verified",
        sha256=content_sha256,
        cache_path=cache_path,
    )
    return _DownloadResult(
        verified,
        CacheAction(
            remote.key,
            "downloaded",
            byte_count,
            validation.state,
        ),
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


def _download_error(code: ErrorCode, key: str) -> SyncError:
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


def _ensure_safe_content_directory(cache_dir: Path, path: Path) -> None:
    _ensure_durable_directory(cache_dir)
    try:
        relative = path.relative_to(cache_dir)
    except ValueError as exc:
        raise ValueError("cache directory is unavailable") from exc
    current = cache_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("cache directory is unavailable")
        if current.exists():
            if not current.is_dir():
                raise ValueError("cache directory is unavailable")
            continue
        current.mkdir(exist_ok=True)
        if current.is_symlink() or not current.is_dir():
            raise ValueError("cache directory is unavailable")
        _fsync_directory(current.parent)


def _path_has_symlink(cache_dir: Path, relative_path: PurePosixPath) -> bool:
    current = cache_dir
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _hash_file(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow)
    with os.fdopen(descriptor, "rb") as body:
        while chunk := body.read(1024 * 1024):
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
    try:
        content = index_path.read_text(encoding="utf-8")
        return json.loads(content, object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("invalid cache index JSON") from exc


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
