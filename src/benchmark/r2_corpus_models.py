from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal, Mapping
from urllib.parse import urlsplit

MAX_SIMFILE_ID = (1 << 53) - 1
CACHE_PROFILE = "setdef_dtx_txt_v1"
CACHE_INDEX_SCHEMA = "crux.r2-cache-index/v1"
MANIFEST_SCHEMA = "crux.r2-corpus-manifest/v1"
PROVENANCE_SCHEMA = "crux.corpus-provenance/v1"
REPORT_SCHEMA = "crux.r2-corpus-sync-report/v1"

ErrorScope = Literal["configuration", "root", "simfile", "object", "cache", "artifact"]
ErrorCode = Literal[
    "invalid_config",
    "missing_optional_dependency",
    "missing_credentials",
    "auth_failed",
    "bucket_inaccessible",
    "root_list_failed",
    "cache_locked",
    "cache_lock_failed",
    "cache_index_invalid",
    "unsupported_platform",
    "provenance_invalid",
    "artifact_write_failed",
    "object_head_failed",
    "object_get_failed",
    "source_changed_during_sync",
    "weak_etag_unverifiable",
    "byte_count_mismatch",
    "cache_corrupt",
    "object_metadata_invalid",
    "ambiguous_simfile_prefix",
    "malformed_root_key",
    "empty_prefix",
    "internal_error",
]
CacheStatus = Literal["not_selected", "verified", "failed"]
SimfileStatus = Literal["complete", "partial", "failed", "empty"]
OverallStatus = Literal["complete", "partial", "failed", "dry_run_complete", "dry_run_partial"]
CacheActionName = Literal["planned", "cache_hit", "downloaded", "failed"]
CacheMissReason = Literal[
    "remote_changed", "missing", "size_mismatch", "sha256_mismatch", "unreadable"
]


@dataclass(frozen=True)
class SyncError:
    scope: ErrorScope
    code: ErrorCode
    message: str
    object_key: str | None = None

    def __post_init__(self) -> None:
        if self.object_key == "":
            raise ValueError("object_key must be non-empty or None")


@dataclass(frozen=True)
class R2Config:
    endpoint_url: str
    source_endpoint_sha256: str
    bucket: str
    head_concurrency: int = 8
    download_concurrency: int = 4
    connect_timeout_seconds: int = 10
    read_timeout_seconds: int = 60
    max_attempts: int = 5
    region_name: str = "auto"

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> R2Config:
        normalized = _normalized_https_origin(environ.get("CRUX_R2_ENDPOINT_URL", ""))
        bucket = environ.get("CRUX_R2_BUCKET", "simfile-dtx")
        if not bucket or "/" in bucket:
            raise ValueError("CRUX_R2_BUCKET must be a non-empty bucket name")
        return cls(
            endpoint_url=normalized,
            source_endpoint_sha256=sha256(normalized.encode("ascii")).hexdigest(),
            bucket=bucket,
            head_concurrency=_bounded_int(environ, "CRUX_R2_HEAD_CONCURRENCY", 8, 1, 32),
            download_concurrency=_bounded_int(environ, "CRUX_R2_DOWNLOAD_CONCURRENCY", 4, 1, 16),
            connect_timeout_seconds=_positive_int(environ, "CRUX_R2_CONNECT_TIMEOUT_SECONDS", 10),
            read_timeout_seconds=_positive_int(environ, "CRUX_R2_READ_TIMEOUT_SECONDS", 60),
            max_attempts=_positive_int(environ, "CRUX_R2_MAX_ATTEMPTS", 5),
        )


def _normalized_https_origin(raw_endpoint: str) -> str:
    try:
        parts = urlsplit(raw_endpoint)
        invalid = (
            parts.scheme.lower() != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.path not in ("", "/")
            or bool(parts.query)
            or bool(parts.fragment)
        )
        if invalid:
            raise ValueError
        hostname = parts.hostname.encode("idna").decode("ascii").lower()
        port = "" if parts.port in (None, 443) else f":{parts.port}"
    except (UnicodeError, ValueError):
        raise ValueError("CRUX_R2_ENDPOINT_URL must be an HTTPS origin") from None
    return f"https://{hostname}{port}"


def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bounded_int(
    environ: Mapping[str, str], name: str, default: int, minimum: int, maximum: int
) -> int:
    value = _positive_int(environ, name, default)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return value


def parse_etag(value: str) -> tuple[str, bool]:
    weak = value.startswith("W/")
    tag = value[2:] if weak else value
    opaque_tag = tag[1:-1]
    invalid_characters = any(
        character == '"' or ord(character) < 0x21 or ord(character) == 0x7F
        for character in opaque_tag
    )
    if len(tag) < 2 or tag[0] != '"' or tag[-1] != '"' or invalid_characters:
        raise ValueError("ETag must be a quoted HTTP entity tag")
    return opaque_tag, weak


def format_manifest_timestamp(value: datetime) -> str:
    try:
        is_aware = (
            isinstance(value, datetime)
            and value.tzinfo is not None
            and value.utcoffset() is not None
        )
    except Exception:
        is_aware = False
    if not is_aware:
        raise ValueError("timestamp must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    fraction = f"{normalized.microsecond:06d}".rstrip("0")
    suffix = f".{fraction}" if fraction else ""
    return normalized.strftime("%Y-%m-%dT%H:%M:%S") + suffix + "Z"


def parse_manifest_timestamp(value: object) -> datetime:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value) is None
    ):
        raise ValueError("timestamp must be a canonical UTC manifest timestamp")
    try:
        format_string = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in value else "%Y-%m-%dT%H:%M:%SZ"
        parsed = datetime.strptime(value, format_string).replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError("timestamp must be a canonical UTC manifest timestamp") from None
    if format_manifest_timestamp(parsed) != value:
        raise ValueError("timestamp must be a canonical UTC manifest timestamp")
    return parsed


def format_report_filename_timestamp(value: datetime) -> str:
    try:
        is_aware = (
            isinstance(value, datetime)
            and value.tzinfo is not None
            and value.utcoffset() is not None
        )
    except Exception:
        is_aware = False
    if not is_aware:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


@dataclass(frozen=True)
class ListedObject:
    key: str
    size: int
    etag: str
    etag_is_weak: bool
    last_modified: datetime
    errors: tuple[SyncError, ...] = ()


@dataclass(frozen=True)
class HeadMetadata:
    size: int | None
    etag: str | None
    etag_is_weak: bool | None
    last_modified: datetime | None
    content_type: str | None


@dataclass(frozen=True)
class RemoteObject:
    key: str
    size: int
    etag: str
    etag_is_weak: bool
    last_modified: datetime
    content_type: str | None
    cache_status: CacheStatus = "not_selected"
    sha256: str | None = None
    cache_path: str | None = None
    errors: tuple[SyncError, ...] = ()


@dataclass(frozen=True)
class SimfileInventory:
    simfile_id: int
    object_prefix: str
    objects: tuple[RemoteObject, ...]
    sync_status: SimfileStatus
    sync_errors: tuple[SyncError, ...] = ()


@dataclass(frozen=True)
class InventoryResult:
    simfiles: tuple[SimfileInventory, ...]
    malformed_root_keys: tuple[str, ...]
    ambiguous_prefixes: dict[int, tuple[str, ...]]
    root_errors: tuple[SyncError, ...]
    simfiles_discovered: int
    simfiles_excluded_by_filter: int
    objects_listed: int


@dataclass(frozen=True)
class ProvenanceRecord:
    source_origin: str | None = None
    source_author_or_pack: str | None = None
    source_reference: str | None = None
    rights_status: str = "unknown"
    redistribution_allowed: bool | None = None
    provenance_notes: str | None = None


@dataclass(frozen=True)
class CacheAction:
    object_key: str
    action: CacheActionName
    bytes: int
    miss_reason: CacheMissReason | None = None
    errors: tuple[SyncError, ...] = ()


@dataclass(frozen=True)
class CacheSyncResult:
    simfiles: tuple[SimfileInventory, ...]
    actions: tuple[CacheAction, ...]


@dataclass(frozen=True)
class RenderedManifest:
    rows: tuple[dict[str, object], ...]
    corpus_version: str
    manifest_sha256: str
    content: bytes


@dataclass(frozen=True)
class PublishedManifest:
    corpus_version: str
    manifest_sha256: str
    relative_path: str
    path: Path
    latest_path: Path


@dataclass(frozen=True)
class SyncCounters:
    simfiles_discovered: int = 0
    simfiles_included: int = 0
    simfiles_excluded_by_filter: int = 0
    simfiles_empty: int = 0
    objects_listed: int = 0
    objects_selected: int = 0
    cache_hits: int = 0
    downloads_planned: int = 0
    downloads_completed: int = 0
    downloads_failed: int = 0
    download_bytes_planned: int = 0
    download_bytes_completed: int = 0


@dataclass(frozen=True)
class SyncRequest:
    output_dir: Path
    cache_dir: Path
    provenance_file: Path | None
    include_simfile_ids: frozenset[int] = field(default_factory=frozenset)
    exclude_simfile_ids: frozenset[int] = field(default_factory=frozenset)
    dry_run: bool = False

    def __post_init__(self) -> None:
        simfile_ids = self.include_simfile_ids | self.exclude_simfile_ids
        if any(simfile_id < 0 or simfile_id > MAX_SIMFILE_ID for simfile_id in simfile_ids):
            raise ValueError(f"simfile IDs must be in 0..{MAX_SIMFILE_ID}")


@dataclass(frozen=True)
class SyncOutcome:
    overall_status: OverallStatus
    exit_code: Literal[0, 1, 2]
    report_path: Path | None
    manifest: PublishedManifest | None
    errors: tuple[SyncError, ...] = ()
    counters: SyncCounters = field(default_factory=SyncCounters)
