from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args
from uuid import uuid4

from src.benchmark.corpus_provenance import provenance_for
from src.benchmark.durability import (
    atomic_replace_bytes,
    ensure_durable_directory,
    fsync_directory,
)
from src.benchmark.r2_corpus_models import (
    CACHE_PROFILE,
    MANIFEST_SCHEMA,
    MAX_SIMFILE_ID,
    CacheStatus,
    ErrorCode,
    ErrorScope,
    ProvenanceRecord,
    PublishedManifest,
    RemoteObject,
    RenderedManifest,
    SimfileInventory,
    SyncError,
    format_manifest_timestamp,
    parse_manifest_timestamp,
)

_LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")
_MANIFEST_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "cache_profile",
        "corpus_version",
        "simfile_id",
        "object_prefix",
        "source_endpoint_sha256",
        "source_bucket",
        "source_discovery_method",
        "objects",
        "sync_status",
        "sync_errors",
        "source_origin",
        "source_author_or_pack",
        "source_reference",
        "rights_status",
        "redistribution_allowed",
        "provenance_notes",
    }
)
_MANIFEST_OBJECT_FIELDS = frozenset(
    {
        "key",
        "size",
        "etag",
        "etag_is_weak",
        "version",
        "last_modified",
        "content_type",
        "cache_status",
        "sha256",
        "cache_path",
    }
)
_MANIFEST_ERROR_FIELDS = frozenset({"scope", "code", "object_key", "message"})
_CACHE_STATUSES = frozenset(get_args(CacheStatus))
_ERROR_CODES = frozenset(get_args(ErrorCode))
_ERROR_SCOPES = frozenset(get_args(ErrorScope))
_SIMFILE_STATUSES = frozenset({"complete", "partial", "failed", "empty"})
_SOURCE_DISCOVERY_METHOD = "r2_list_objects_v2"
_PUBLICATION_ERROR = SyncError(
    "artifact",
    "artifact_write_failed",
    "A required manifest artifact write failed.",
)
_logger = logging.getLogger(__name__)


class ManifestPublicationError(Exception):
    def __init__(self, error: SyncError):
        super().__init__(error.code)
        self.error = error


@dataclass(frozen=True)
class ManifestRowView:
    inventory: SimfileInventory
    provenance: ProvenanceRecord
    corpus_version: str
    cache_profile: str
    source_endpoint_sha256: str
    source_bucket: str
    source_discovery_method: str


def manifest_row_view_from_row(row: Mapping[str, object]) -> ManifestRowView:
    if not isinstance(row, Mapping) or set(row) != _MANIFEST_ROW_FIELDS:
        raise ValueError("invalid HPA-321 manifest row")

    corpus_version = _manifest_corpus_version(row["corpus_version"])
    cache_profile = row["cache_profile"]
    source_endpoint_sha256 = row["source_endpoint_sha256"]
    source_bucket = row["source_bucket"]
    source_discovery_method = row["source_discovery_method"]
    simfile_id = row["simfile_id"]
    object_prefix = row["object_prefix"]
    sync_status = row["sync_status"]

    if not all(
        (
            row["schema_version"] == MANIFEST_SCHEMA,
            cache_profile == CACHE_PROFILE,
            _is_lowercase_sha256(source_endpoint_sha256),
            _is_nonempty_string(source_bucket),
            source_discovery_method == _SOURCE_DISCOVERY_METHOD,
            _is_simfile_id(simfile_id),
            _is_object_prefix_for_simfile(object_prefix, simfile_id),
            _is_simfile_status(sync_status),
        )
    ):
        raise ValueError("invalid HPA-321 manifest row")

    objects = _manifest_objects_from_row(row["objects"], object_prefix)
    object_keys = frozenset(remote.key for remote in objects)
    sync_errors = _manifest_errors_from_row(row["sync_errors"], object_keys, object_prefix)
    provenance = _manifest_provenance_from_row(row)
    inventory = SimfileInventory(
        simfile_id=simfile_id,
        object_prefix=object_prefix,
        objects=objects,
        sync_status=sync_status,
        sync_errors=sync_errors,
    )
    return ManifestRowView(
        inventory=inventory,
        provenance=provenance,
        corpus_version=corpus_version,
        cache_profile=cache_profile,
        source_endpoint_sha256=source_endpoint_sha256,
        source_bucket=source_bucket,
        source_discovery_method=source_discovery_method,
    )


def inventory_from_manifest_row(row: Mapping[str, object]) -> SimfileInventory:
    return manifest_row_view_from_row(row).inventory


def _manifest_corpus_version(value: object) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError("invalid HPA-321 manifest row")
    if not _is_lowercase_sha256(value.removeprefix("sha256:")):
        raise ValueError("invalid HPA-321 manifest row")
    return value


def _manifest_objects_from_row(value: object, object_prefix: str) -> tuple[RemoteObject, ...]:
    if not isinstance(value, list):
        raise ValueError("invalid HPA-321 manifest row")

    objects: list[RemoteObject] = []
    object_keys: set[str] = set()
    for raw_object in value:
        if not isinstance(raw_object, Mapping) or set(raw_object) != _MANIFEST_OBJECT_FIELDS:
            raise ValueError("invalid HPA-321 manifest row")
        key = raw_object["key"]
        size = raw_object["size"]
        etag = raw_object["etag"]
        etag_is_weak = raw_object["etag_is_weak"]
        last_modified = raw_object["last_modified"]
        content_type = raw_object["content_type"]
        cache_status = raw_object["cache_status"]
        digest = raw_object["sha256"]
        cache_path = raw_object["cache_path"]
        if not all(
            (
                _is_nonempty_string(key),
                _is_object_key_in_prefix(key, object_prefix),
                key not in object_keys if isinstance(key, str) else False,
                _is_nonnegative_int(size),
                isinstance(etag, str),
                isinstance(etag_is_weak, bool),
                raw_object["version"] is None,
                content_type is None or isinstance(content_type, str),
                _is_cache_status(cache_status),
                digest is None or _is_lowercase_sha256(digest),
                cache_path is None or _is_nonempty_string(cache_path),
            )
        ):
            raise ValueError("invalid HPA-321 manifest row")
        try:
            parsed_last_modified = parse_manifest_timestamp(last_modified)
        except ValueError:
            raise ValueError("invalid HPA-321 manifest row") from None
        object_keys.add(key)
        objects.append(
            RemoteObject(
                key=key,
                size=size,
                etag=etag,
                etag_is_weak=etag_is_weak,
                last_modified=parsed_last_modified,
                content_type=content_type,
                cache_status=cache_status,
                sha256=digest,
                cache_path=cache_path,
            )
        )
    return tuple(objects)


def _manifest_errors_from_row(
    value: object,
    object_keys: frozenset[str],
    object_prefix: str,
) -> tuple[SyncError, ...]:
    if not isinstance(value, list):
        raise ValueError("invalid HPA-321 manifest row")

    errors: list[SyncError] = []
    for raw_error in value:
        if not isinstance(raw_error, Mapping) or set(raw_error) != _MANIFEST_ERROR_FIELDS:
            raise ValueError("invalid HPA-321 manifest row")
        scope = raw_error["scope"]
        code = raw_error["code"]
        object_key = raw_error["object_key"]
        message = raw_error["message"]
        if not all(
            (
                _is_error_scope(scope),
                _is_error_code(code),
                object_key is None or _is_nonempty_string(object_key),
                isinstance(message, str),
            )
        ):
            raise ValueError("invalid HPA-321 manifest row")
        if object_key is not None and object_key not in object_keys:
            if (scope, code, object_key) != ("simfile", "empty_prefix", object_prefix):
                raise ValueError("invalid HPA-321 manifest row")
        errors.append(SyncError(scope, code, message, object_key))
    return tuple(errors)


def _manifest_provenance_from_row(row: Mapping[str, object]) -> ProvenanceRecord:
    nullable_fields = (
        "source_origin",
        "source_author_or_pack",
        "source_reference",
        "provenance_notes",
    )
    if any(row[field] is not None and not isinstance(row[field], str) for field in nullable_fields):
        raise ValueError("invalid HPA-321 manifest row")
    rights_status = row["rights_status"]
    redistribution_allowed = row["redistribution_allowed"]
    if not _is_nonempty_string(rights_status) or (
        redistribution_allowed is not None and not isinstance(redistribution_allowed, bool)
    ):
        raise ValueError("invalid HPA-321 manifest row")
    return ProvenanceRecord(
        source_origin=row["source_origin"],
        source_author_or_pack=row["source_author_or_pack"],
        source_reference=row["source_reference"],
        rights_status=rights_status,
        redistribution_allowed=redistribution_allowed,
        provenance_notes=row["provenance_notes"],
    )


def _is_lowercase_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWERCASE_HEX_DIGITS for character in value)
    )


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_simfile_id(value: object) -> bool:
    return _is_nonnegative_int(value) and value <= MAX_SIMFILE_ID


def _is_object_prefix_for_simfile(object_prefix: object, simfile_id: object) -> bool:
    if not isinstance(object_prefix, str) or not _is_simfile_id(simfile_id):
        return False
    first_segment, separator, remainder = object_prefix.partition("/")
    return (
        separator == "/"
        and not remainder
        and first_segment.isascii()
        and first_segment.isdecimal()
        and int(first_segment) == simfile_id
    )


def _is_object_key_in_prefix(key: object, object_prefix: str) -> bool:
    if not isinstance(key, str):
        return False
    first_segment, separator, _ = key.partition("/")
    return separator == "/" and f"{first_segment}/" == object_prefix


def _is_simfile_status(value: object) -> bool:
    return isinstance(value, str) and value in _SIMFILE_STATUSES


def _is_cache_status(value: object) -> bool:
    return isinstance(value, str) and value in _CACHE_STATUSES


def _is_error_scope(value: object) -> bool:
    return isinstance(value, str) and value in _ERROR_SCOPES


def _is_error_code(value: object) -> bool:
    return isinstance(value, str) and value in _ERROR_CODES


def canonical_json_line(value: dict[str, object]) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8") + b"\n"


def build_manifest_rows(
    simfiles: tuple[SimfileInventory, ...],
    provenance: Mapping[int, ProvenanceRecord],
    source_endpoint_sha256: str,
    bucket: str,
) -> tuple[dict[str, object], ...]:
    if len(source_endpoint_sha256) != 64 or any(
        character not in _LOWERCASE_HEX_DIGITS for character in source_endpoint_sha256
    ):
        raise ValueError("source_endpoint_sha256 must be lowercase SHA-256")
    provenance_records = dict(provenance)
    return tuple(
        _build_row(
            simfile,
            provenance_for(provenance_records, simfile.simfile_id),
            source_endpoint_sha256,
            bucket,
        )
        for simfile in sorted(
            simfiles,
            key=lambda item: (item.simfile_id, item.object_prefix),
        )
    )


def render_manifest(rows: tuple[dict[str, object], ...]) -> RenderedManifest:
    if any("corpus_version" in row for row in rows):
        raise ValueError("normalized payload rows must omit corpus_version")
    identity_bytes = b"".join(canonical_json_line(row) for row in rows)
    corpus_version = f"sha256:{hashlib.sha256(identity_bytes).hexdigest()}"
    final_rows = tuple({"corpus_version": corpus_version, **row} for row in rows)
    content = b"".join(canonical_json_line(row) for row in final_rows)
    manifest_sha256 = hashlib.sha256(content).hexdigest()
    return RenderedManifest(final_rows, corpus_version, manifest_sha256, content)


def publish_manifest(
    output_dir: Path,
    rendered: RenderedManifest,
) -> PublishedManifest:
    try:
        ensure_durable_directory(output_dir)
        manifests_dir = output_dir / "manifests"
        ensure_durable_directory(manifests_dir)
        fsync_directory(manifests_dir)
        manifest_path = manifests_dir / f"{rendered.manifest_sha256}.jsonl"
        _publish_immutable(manifest_path, rendered.content, rendered.manifest_sha256)
        return PublishedManifest(
            corpus_version=rendered.corpus_version,
            manifest_sha256=rendered.manifest_sha256,
            relative_path=f"manifests/{rendered.manifest_sha256}.jsonl",
            path=manifest_path,
            latest_path=output_dir / "latest.json",
        )
    except ManifestPublicationError:
        raise
    except Exception:
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None


def publish_latest_manifest(
    output_dir: Path,
    published: PublishedManifest,
    overall_status: Literal["complete", "partial"],
    published_at: datetime,
) -> None:
    try:
        is_aware = published_at.tzinfo is not None and published_at.utcoffset() is not None
    except Exception:
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None
    if not is_aware:
        raise ValueError("published_at must be timezone-aware")
    try:
        latest = {
            "corpus_version": published.corpus_version,
            "manifest_sha256": published.manifest_sha256,
            "manifest_path": published.relative_path,
            "overall_status": overall_status,
            "published_at": format_manifest_timestamp(published_at),
        }
        _atomic_replace_json(output_dir / "latest.json", latest)
    except ManifestPublicationError:
        raise
    except Exception:
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None


def _build_row(
    simfile: SimfileInventory,
    provenance: ProvenanceRecord,
    source_endpoint_sha256: str,
    bucket: str,
) -> dict[str, object]:
    object_dicts = [
        _object_dict(remote) for remote in sorted(simfile.objects, key=_object_sort_key)
    ]
    error_dicts = [_error_dict(error) for error in sorted(simfile.sync_errors, key=_error_sort_key)]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "cache_profile": CACHE_PROFILE,
        "simfile_id": simfile.simfile_id,
        "object_prefix": simfile.object_prefix,
        "source_endpoint_sha256": source_endpoint_sha256,
        "source_bucket": bucket,
        "source_discovery_method": "r2_list_objects_v2",
        "objects": object_dicts,
        "sync_status": simfile.sync_status,
        "sync_errors": error_dicts,
        "source_origin": provenance.source_origin,
        "source_author_or_pack": provenance.source_author_or_pack,
        "source_reference": provenance.source_reference,
        "rights_status": provenance.rights_status,
        "redistribution_allowed": provenance.redistribution_allowed,
        "provenance_notes": provenance.provenance_notes,
    }


def _object_dict(remote: RemoteObject) -> dict[str, object]:
    return {
        "key": remote.key,
        "size": remote.size,
        "etag": remote.etag,
        "etag_is_weak": remote.etag_is_weak,
        "version": None,
        "last_modified": format_manifest_timestamp(remote.last_modified),
        "content_type": remote.content_type,
        "cache_status": remote.cache_status,
        "sha256": remote.sha256,
        "cache_path": remote.cache_path,
    }


def _error_dict(error: SyncError) -> dict[str, object]:
    return {
        "scope": error.scope,
        "code": error.code,
        "object_key": error.object_key,
        "message": error.message,
    }


def _object_sort_key(remote: RemoteObject) -> str:
    return remote.key


def _error_sort_key(error: SyncError) -> tuple[str, str, str, str]:
    object_key = error.object_key if error.object_key is not None else ""
    return (error.scope, error.code, object_key, error.message)


def _publish_immutable(path: Path, content: bytes, expected_sha256: str) -> None:
    temporary_path: Path | None = None
    temporary_exists = False
    completed = False
    cleanup_failed = False
    try:
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError("manifest content hash mismatch")
        if _path_exists_no_follow(path):
            _verify_existing_manifest(path, content)
            completed = True
        else:
            temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
            with temporary_path.open("xb") as temporary:
                temporary_exists = True
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                pass
            temporary_path.unlink()
            temporary_exists = False
            _verify_existing_manifest(path, content)
            completed = True
    except Exception as original_error:
        completed = False
        _logger.debug("immutable manifest publication failed for %s", path, exc_info=original_error)
    finally:
        if temporary_exists and temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception as cleanup_error:
                cleanup_failed = True
                _logger.debug(
                    "immutable manifest cleanup failed for %s",
                    temporary_path,
                    exc_info=cleanup_error,
                )
    if not completed or cleanup_failed:
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None


def publish_immutable_bytes(path: Path, content: bytes, expected_sha256: str) -> None:
    """Publish ``content`` at ``path`` immutably.

    Public entry point for cross-module callers.  Thin delegation to the
    internal :func:`_publish_immutable` publisher — the existing hash-checked,
    fsync + hardlink + verify implementation.  No durability or conflict-handling
    logic is duplicated.  Raises :class:`ManifestPublicationError` on failure.
    """
    _publish_immutable(path, content, expected_sha256)


def _verify_existing_manifest(path: Path, expected_content: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, _regular_file_open_flags())
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise OSError("manifest destination is not a regular file")
        actual_content = _read_descriptor(descriptor)
        _verify_path_binding(path, descriptor_stat)
        if actual_content != expected_content:
            raise OSError("manifest destination content differs")
        os.fsync(descriptor)
        _verify_path_binding(path, descriptor_stat)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    fsync_directory(path.parent)


def _atomic_replace_json(path: Path, payload: dict[str, object]) -> None:
    try:
        atomic_replace_bytes(path, canonical_json_line(payload))
    except OSError:
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None


def _path_exists_no_follow(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _regular_file_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    non_block = getattr(os, "O_NONBLOCK", None)
    if no_follow is None:
        raise OSError("no-follow file descriptors are unavailable")
    if non_block is None:
        raise OSError("non-blocking file descriptors are unavailable")
    return os.O_RDONLY | no_follow | non_block


def _read_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _verify_path_binding(path: Path, descriptor_stat: os.stat_result) -> None:
    path_stat = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(path_stat.st_mode) or (path_stat.st_dev, path_stat.st_ino) != (
        descriptor_stat.st_dev,
        descriptor_stat.st_ino,
    ):
        raise OSError("manifest destination binding changed")
