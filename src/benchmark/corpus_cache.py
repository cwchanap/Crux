from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from errno import EAGAIN, ENOSYS, ENOTSUP, EOPNOTSUPP, EWOULDBLOCK
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Iterator
from uuid import uuid4

from src.benchmark.r2_corpus_models import CACHE_INDEX_SCHEMA, format_manifest_timestamp

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
