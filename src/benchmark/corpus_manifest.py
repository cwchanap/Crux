from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping
from uuid import uuid4

from src.benchmark.corpus_provenance import provenance_for
from src.benchmark.r2_corpus_models import (
    CACHE_PROFILE,
    MANIFEST_SCHEMA,
    ProvenanceRecord,
    PublishedManifest,
    RemoteObject,
    RenderedManifest,
    SimfileInventory,
    SyncError,
    format_manifest_timestamp,
)

_LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")
_PUBLICATION_ERROR = SyncError(
    "artifact",
    "artifact_write_failed",
    "A required manifest artifact write failed.",
)


class ManifestPublicationError(Exception):
    def __init__(self, error: SyncError):
        super().__init__(error.code)
        self.error = error


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
        _ensure_durable_directory(output_dir)
        manifests_dir = output_dir / "manifests"
        _ensure_durable_directory(manifests_dir)
        _fsync_directory(manifests_dir)
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
    except Exception:
        completed = False
    finally:
        if temporary_exists and temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                cleanup_failed = True
    if not completed or cleanup_failed:
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None


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
    _fsync_directory(path.parent)


def _atomic_replace_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path: Path | None = None
    temporary_exists = False
    completed = False
    cleanup_failed = False
    try:
        content = canonical_json_line(payload)
        temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        with temporary_path.open("xb") as temporary:
            temporary_exists = True
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_exists = False
        _fsync_directory(path.parent)
        completed = True
    except Exception:
        completed = False
    finally:
        if temporary_exists and temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                cleanup_failed = True
    if not completed or cleanup_failed:
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _directory_open_flags())
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("manifest directory is unavailable")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(path: Path) -> None:
    missing: list[Path] = []
    candidate = path
    while True:
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            missing.append(candidate)
            parent = candidate.parent
            if parent == candidate:
                raise OSError("manifest directory ancestor is unavailable")
            candidate = parent
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError("manifest directory path is not a directory")
        break

    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            metadata = directory.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise OSError("manifest directory path is not a directory") from None
        _fsync_directory(directory.parent)


def _path_exists_no_follow(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _regular_file_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise OSError("no-follow file descriptors are unavailable")
    return os.O_RDONLY | no_follow


def _directory_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OSError("no-follow directory descriptors are unavailable")
    return os.O_RDONLY | no_follow | directory


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
