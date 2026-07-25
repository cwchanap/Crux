from __future__ import annotations

import hashlib
import json
import os
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
    manifests_dir = output_dir / "manifests"
    try:
        manifests_dir.mkdir(parents=True, exist_ok=True)
        _fsync_directory(manifests_dir.parent)
        _fsync_directory(manifests_dir)
    except (OSError, ValueError):
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None
    manifest_path = manifests_dir / f"{rendered.manifest_sha256}.jsonl"
    _publish_immutable(manifest_path, rendered.content, rendered.manifest_sha256)
    return PublishedManifest(
        corpus_version=rendered.corpus_version,
        manifest_sha256=rendered.manifest_sha256,
        relative_path=f"manifests/{rendered.manifest_sha256}.jsonl",
        path=manifest_path,
        latest_path=output_dir / "latest.json",
    )


def publish_latest_manifest(
    output_dir: Path,
    published: PublishedManifest,
    overall_status: Literal["complete", "partial"],
    published_at: datetime,
) -> None:
    latest = {
        "corpus_version": published.corpus_version,
        "manifest_sha256": published.manifest_sha256,
        "manifest_path": published.relative_path,
        "overall_status": overall_status,
        "published_at": format_manifest_timestamp(published_at),
    }
    _atomic_replace_json(output_dir / "latest.json", latest)


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
    return (error.scope, error.code, error.object_key or "", error.message)


def _publish_immutable(path: Path, content: bytes, expected_sha256: str) -> None:
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ManifestPublicationError(_PUBLICATION_ERROR)
    if path.exists():
        _verify_existing_manifest(path, content)
        return

    temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    temporary_exists = False
    installed = False
    try:
        with temporary_path.open("xb") as temporary:
            temporary_exists = True
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
            installed = True
        except FileExistsError:
            _verify_existing_manifest(path, content)
        temporary_path.unlink()
        temporary_exists = False
        if installed:
            _fsync_directory(path.parent)
    except ManifestPublicationError:
        raise
    except (OSError, ValueError):
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None
    finally:
        if temporary_exists:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _verify_existing_manifest(path: Path, expected_content: bytes) -> None:
    try:
        matches = path.read_bytes() == expected_content
    except OSError:
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None
    if not matches:
        raise ManifestPublicationError(_PUBLICATION_ERROR)


def _atomic_replace_json(path: Path, payload: dict[str, object]) -> None:
    content = canonical_json_line(payload)
    temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    temporary_exists = False
    try:
        with temporary_path.open("xb") as temporary:
            temporary_exists = True
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_exists = False
        _fsync_directory(path.parent)
    except (OSError, UnicodeError, ValueError):
        raise ManifestPublicationError(_PUBLICATION_ERROR) from None
    finally:
        if temporary_exists:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
