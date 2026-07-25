# HPA-321 R2 Corpus Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a secure, resumable R2 inventory command that selectively caches chart-definition files and publishes deterministic, immutable benchmark manifests.

**Architecture:** A lazy `boto3` adapter implements a narrow object-store protocol without making AWS packages part of the base runtime. Focused inventory, provenance, cache, manifest, and orchestration modules exchange immutable domain records; the Click command only maps options, progress, summaries, and explicit `0`/`1`/`2` outcomes.

**Tech Stack:** Python 3.12, Click 8, optional boto3/botocore, dataclasses, SHA-256, canonical JSON/JSONL, POSIX `fcntl.flock`, `ThreadPoolExecutor`, pytest

## Global Constraints

- R2 object contents are authoritative; D1 and GraphQL must not determine chart truth.
- Enumerate the complete bucket root with `ListObjectsV2` and no delimiter.
- Preserve exact Unicode object keys without URL decoding, case folding, or normalization.
- Accept simfile IDs only in `0..9007199254740991`.
- Quarantine ambiguous numeric aliases such as `1/` and `01/`, even for explicit includes.
- Treat every zero-byte key ending in `/` as a folder marker; a prefix containing only markers is `empty`.
- The fixed cache profile is `setdef_dtx_txt_v1`; there is no profile option in v1.
- Select case-insensitive `set.def`, `.dtx`, and `.txt` objects only.
- Treat ETags as remote change signals, never universal content hashes.
- Store ETag weakness separately and use `If-Match` only for strong ETags.
- Verify weak-ETag downloads against response ETag, content length, and last-modified metadata.
- Omit provider checksum fields and `ChecksumMode` from the v1 adapter and manifest.
- Verify every reused or downloaded selected body with local SHA-256 and byte count.
- Store cache bodies at `sha256/<first-two-hex>/<full-sha256>` relative to the cache root.
- Stage bodies only under `<cache-dir>/sha256/.incoming/`, never process-wide `TMPDIR`.
- Hold one non-blocking POSIX writer lock for the whole real transaction.
- Use an in-process `threading.Lock` around index mutation and checkpoint publication.
- Publish cache indexes, bodies, manifests, reports, and pointers with durable atomic replacement.
- Real manifests are canonical UTF-8 JSONL and immutable at `manifests/<sha256>.jsonl`.
- Every manifest field except `corpus_version`, including status and errors, participates in corpus identity.
- `latest.json` and `latest-report.json` are convenience pointers with last-completed-writer-wins semantics.
- Dry-run may write reports but must not GET bodies or mutate cache, index, manifests, or `latest.json`.
- Exit `0` for complete, `1` for partial, and `2` for fatal/no-manifest outcomes.
- Do not let an HPA-321 domain failure escape as a bare `click.ClickException`.
- Never store or log credentials, signed URLs, signed headers, raw endpoint URLs, or raw SDK exceptions.
- Pin `boto3`, `botocore`, and `urllib3` loggers to at least `WARNING`.
- Keep chart selection, DTX parsing, audio selection, inference, and scoring out of HPA-321.
- Keep Python lines at or below the repository's 100-character formatting limit.

---

## File Map

### New source files

- `src/benchmark/r2_corpus_models.py` — shared constants, immutable records, endpoint/ETag/timestamp normalization, and diagnostic vocabulary.
- `src/benchmark/r2_inventory.py` — lazy boto3 adapter, object-store protocol, root classification, filtering, and bounded HEAD enrichment.
- `src/benchmark/corpus_provenance.py` — strict optional provenance loader and unknown defaults.
- `src/benchmark/corpus_cache.py` — selection policy, index model, whole-run lock, local validation, concurrent downloads, durability, and checkpoints.
- `src/benchmark/corpus_manifest.py` — canonical row construction, two-pass identity, immutable manifest publication, and `latest.json`.
- `src/benchmark/r2_corpus_sync.py` — transaction orchestration, progress, report assembly/publication, fatal containment, and outcome mapping.

### New configuration and tests

- `config/corpus-provenance.json` — version-controlled empty v1 provenance mapping.
- `tests/benchmark/test_r2_corpus_models.py`
- `tests/benchmark/test_r2_inventory.py`
- `tests/benchmark/test_corpus_provenance.py`
- `tests/benchmark/test_corpus_cache.py`
- `tests/benchmark/test_corpus_manifest.py`
- `tests/benchmark/test_r2_corpus_sync.py`
- `tests/benchmark/test_r2_corpus_acceptance.py`

### Modified files

- `pyproject.toml` — add only the `r2` optional extra.
- `uv.lock` — lock boto3, botocore, jmespath, s3transfer, and urllib3.
- `src/cli/benchmark.py` — register `sync-r2-corpus` and map explicit outcomes through `ctx.exit`.
- `tests/test_cli_benchmark.py` — command help, options, output, redaction, and exit-code tests.
- `docs/drumery-dtx-midi-benchmarking-reference.md` — document the R2 inventory stage and artifact contract.

### Cross-task interfaces

Later tasks must use the exact public names defined below:

- Models: `MAX_SIMFILE_ID`, `CACHE_PROFILE`, `ErrorCode`, `SyncError`, `R2Config`,
  `ListedObject`, `HeadMetadata`, `RemoteObject`, `SimfileInventory`,
  `InventoryResult`, `ProvenanceRecord`, `CacheAction`, `CacheSyncResult`,
  `RenderedManifest`, `PublishedManifest`, `SyncCounters`, `SyncRequest`,
  `SyncOutcome`,
  `parse_etag`, `format_manifest_timestamp`, and
  `format_report_filename_timestamp`.
- Inventory: `R2ObjectStore`, `R2StoreError`, `ensure_r2_dependency`,
  `create_boto3_store`, and `build_inventory`.
- Provenance: `load_provenance` and `provenance_for`.
- Cache: `CacheIndexStore`, `cache_writer_lock`, and `sync_cache`.
- Manifest: `canonical_json_line`, `build_manifest_rows`, `render_manifest`,
  `publish_manifest`, and `publish_latest_manifest`.
- Orchestration: `ProgressEvent` and `sync_r2_corpus`.

Each task defines the complete signature and record fields before a later task consumes
the interface.

---

### Task 1: Add the optional dependency and shared domain primitives

**Files:**
- Create: `src/benchmark/r2_corpus_models.py`
- Create: `tests/benchmark/test_r2_corpus_models.py`
- Modify: `pyproject.toml:22-38`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: environment mappings and SDK-returned datetime/ETag values.
- Produces: all immutable records and constants listed in the file map.

- [ ] **Step 1: Write failing normalization and configuration tests**

Create `tests/benchmark/test_r2_corpus_models.py` with focused tests:

```python
from datetime import datetime, timezone
from hashlib import sha256

import pytest

from src.benchmark.r2_corpus_models import (
    CACHE_PROFILE,
    MAX_SIMFILE_ID,
    R2Config,
    format_manifest_timestamp,
    format_report_filename_timestamp,
    parse_etag,
)


def test_config_normalizes_endpoint_and_hashes_only_normalized_origin():
    config = R2Config.from_environ(
        {
            "CRUX_R2_ENDPOINT_URL": "HTTPS://ABC123.r2.cloudflarestorage.com/",
            "CRUX_R2_BUCKET": "simfile-dtx",
        }
    )
    normalized = "https://abc123.r2.cloudflarestorage.com"
    assert config.endpoint_url == normalized
    assert config.source_endpoint_sha256 == sha256(normalized.encode("ascii")).hexdigest()
    assert config.bucket == "simfile-dtx"
    assert config.head_concurrency == 8
    assert config.download_concurrency == 4
    assert config.connect_timeout_seconds == 10
    assert config.read_timeout_seconds == 60
    assert config.max_attempts == 5


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://abc.r2.cloudflarestorage.com",
        "https://user@example.com",
        "https://example.com/path",
        "https://example.com?token=value",
        "https://example.com#fragment",
    ],
)
def test_config_rejects_non_https_origin(endpoint):
    with pytest.raises(ValueError, match="HTTPS origin"):
        R2Config.from_environ({"CRUX_R2_ENDPOINT_URL": endpoint})


@pytest.mark.parametrize(
    "bucket",
    [
        "",
        "bad bucket",
        "Bucket",
        "b",
        "bu",
        "bucket?x",
        "bucket_underscore",
        "bucket.with.dots",
        "a--b",
        "-leading",
        "trailing-",
        "192.168.0.1",
        "x" * 64,
    ],
)
def test_config_rejects_invalid_bucket_name(bucket):
    with pytest.raises(ValueError, match="CRUX_R2_BUCKET"):
        R2Config.from_environ(
            {"CRUX_R2_ENDPOINT_URL": "https://abc.r2.cloudflarestorage.com", "CRUX_R2_BUCKET": bucket}
        )


@pytest.mark.parametrize(
    "bucket",
    ["simfile-dtx", "a-b", "abc123", "n" * 63, "0-9"],
)
def test_config_accepts_valid_bucket_name(bucket):
    config = R2Config.from_environ(
        {"CRUX_R2_ENDPOINT_URL": "https://abc.r2.cloudflarestorage.com", "CRUX_R2_BUCKET": bucket}
    )
    assert config.bucket == bucket


def test_etag_and_timestamp_normalization_are_canonical():
    assert parse_etag('"abc-2"') == ("abc-2", False)
    assert parse_etag('W/"weak-value"') == ("weak-value", True)
    with pytest.raises(ValueError, match="entity tag"):
        parse_etag("not-quoted")
    value = datetime(2026, 7, 25, 1, 2, 3, 120000, tzinfo=timezone.utc)
    assert format_manifest_timestamp(value) == "2026-07-25T01:02:03.12Z"
    assert format_report_filename_timestamp(value) == "20260725T010203.120000Z"


def test_fixed_contract_constants_are_stable():
    assert CACHE_PROFILE == "setdef_dtx_txt_v1"
    assert MAX_SIMFILE_ID == 9_007_199_254_740_991
```

Add parameterized cases for a missing endpoint, empty/invalid bucket, concurrency
outside `1..32` or `1..16`, and nonpositive timeout/attempt values. Assert messages
name only the relevant environment variable and never echo the supplied value.

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
rtk uv run pytest tests/benchmark/test_r2_corpus_models.py -q
```

Expected: collection fails with `ModuleNotFoundError: src.benchmark.r2_corpus_models`.

- [ ] **Step 3: Add boto3 only to the optional `r2` extra**

Run:

```bash
rtk uv add --optional r2 "boto3>=1.42,<2"
```

Verify `pyproject.toml` contains:

```toml
[project.optional-dependencies]
r2 = [
    "boto3>=1.42,<2",
]
```

Do not add boto3 to `[project.dependencies]`. Confirm `uv.lock` contains the resolved
boto3/botocore graph.

- [ ] **Step 4: Implement the shared records and canonical primitives**

Create `src/benchmark/r2_corpus_models.py` with:

```python
from __future__ import annotations

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
OverallStatus = Literal[
    "complete", "partial", "failed", "dry_run_complete", "dry_run_partial"
]
CacheActionName = Literal["planned", "cache_hit", "downloaded", "failed"]
CacheMissReason = Literal["remote_changed", "missing", "size_mismatch", "sha256_mismatch"]


@dataclass(frozen=True)
class SyncError:
    scope: ErrorScope
    code: ErrorCode
    message: str
    object_key: str | None = None


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
    def from_environ(cls, environ: Mapping[str, str]) -> "R2Config":
        raw_endpoint = environ.get("CRUX_R2_ENDPOINT_URL", "")
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
            raise ValueError("R2 endpoint must be an HTTPS origin")
        hostname = parts.hostname.encode("idna").decode("ascii").lower()
        port = "" if parts.port in (None, 443) else f":{parts.port}"
        normalized = f"https://{hostname}{port}"
        bucket = environ.get("CRUX_R2_BUCKET", "simfile-dtx")
        _validate_bucket_name(bucket)
        return cls(
            endpoint_url=normalized,
            source_endpoint_sha256=sha256(normalized.encode("ascii")).hexdigest(),
            bucket=bucket,
            head_concurrency=_bounded_int(environ, "CRUX_R2_HEAD_CONCURRENCY", 8, 1, 32),
            download_concurrency=_bounded_int(
                environ, "CRUX_R2_DOWNLOAD_CONCURRENCY", 4, 1, 16
            ),
            connect_timeout_seconds=_positive_int(
                environ, "CRUX_R2_CONNECT_TIMEOUT_SECONDS", 10
            ),
            read_timeout_seconds=_positive_int(
                environ, "CRUX_R2_READ_TIMEOUT_SECONDS", 60
            ),
            max_attempts=_positive_int(environ, "CRUX_R2_MAX_ATTEMPTS", 5),
        )
```

Add these integer helpers:

```python
def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bounded_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = _positive_int(environ, name, default)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return value


import re

_BUCKET_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,61}[a-z0-9])?$")


def _validate_bucket_name(bucket: str) -> None:
    """Apply the R2/S3 DNS-compatible bucket-name grammar.

    Rejects names that would otherwise reach the SDK and surface as remote-access
    or internal errors instead of the documented deterministic ``invalid_config``.
    Rules: 3-63 chars, lowercase letters/digits/hyphens, must start and end with
    an alphanumeric character, no consecutive hyphens, no ``.``-separated
    IP-address format, and not the reserved ``.`` or ``..`` names. The message
    never echoes the offending value.
    """
    if not bucket:
        raise ValueError("CRUX_R2_BUCKET must be a non-empty bucket name")
    if not _BUCKET_NAME_RE.match(bucket):
        raise ValueError("CRUX_R2_BUCKET must use lowercase letters, digits, and hyphens")
    if "--" in bucket:
        raise ValueError("CRUX_R2_BUCKET must not contain consecutive hyphens")
    if len(bucket) < 3 or len(bucket) > 63:
        raise ValueError("CRUX_R2_BUCKET must be 3-63 characters")
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", bucket):
        raise ValueError("CRUX_R2_BUCKET must not be an IP-address format")
```

Add `re` to the module's top-level imports (Ruff `I`-sorted) alongside the
existing `datetime`, `json`, `pathlib`, and `hashlib` imports; do not import it
inline.

Wrap URL parsing, IDNA encoding, and port extraction so malformed input becomes the
same deterministic HTTPS-origin error without echoing the value. Add these immutable
records:

```python
@dataclass(frozen=True)
class ListedObject:
    key: str
    size: int
    etag: str
    etag_is_weak: bool
    last_modified: datetime


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


@dataclass(frozen=True)
class SyncOutcome:
    overall_status: OverallStatus
    exit_code: Literal[0, 1, 2]
    report_path: Path | None
    manifest: PublishedManifest | None
    errors: tuple[SyncError, ...] = ()
    counters: SyncCounters = field(default_factory=SyncCounters)
```

Implement `parse_etag` as a strict quoted HTTP entity-tag parser that retains the
`W/` bit separately. Implement timestamps with UTC conversion, manifest fractional
zero trimming, and fixed six-digit report filename precision.

- [ ] **Step 5: Run focused tests and import-without-extra checks**

Run:

```bash
rtk uv run pytest tests/benchmark/test_r2_corpus_models.py -q
rtk uv run python -c "import src.benchmark.r2_corpus_models"
rtk uv run ruff check src/benchmark/r2_corpus_models.py tests/benchmark/test_r2_corpus_models.py
rtk uv run black --check src/benchmark/r2_corpus_models.py tests/benchmark/test_r2_corpus_models.py
```

Expected: all tests pass, the base import succeeds without importing boto3, and both
formatters report clean files.

- [ ] **Step 6: Commit the foundation**

```bash
rtk git add pyproject.toml uv.lock src/benchmark/r2_corpus_models.py \
  tests/benchmark/test_r2_corpus_models.py
rtk git commit -m "feat: add R2 corpus domain primitives"
```

---

### Task 2: Implement the lazy, sanitized R2 object-store adapter

**Files:**
- Create: `src/benchmark/r2_inventory.py`
- Create: `tests/benchmark/test_r2_inventory.py`

**Interfaces:**
- Consumes: `R2Config`, `ListedObject`, `HeadMetadata`, `parse_etag`.
- Produces: `R2ObjectStore`, `R2StoreError`, `ObjectDownload`, `ensure_r2_dependency`, and `create_boto3_store`.

- [ ] **Step 1: Write failing adapter tests with fake paginator and client objects**

Start `tests/benchmark/test_r2_inventory.py` with:

```python
from datetime import datetime, timezone
from io import BytesIO
import logging

import pytest

from src.benchmark.r2_corpus_models import R2Config
from src.benchmark.r2_inventory import (
    Boto3R2Store,
    R2StoreError,
    create_boto3_store,
)


class FakePaginator:
    def paginate(self, **kwargs):
        assert kwargs == {"Bucket": "simfile-dtx"}
        return [
            {
                "Contents": [
                    {
                        "Key": "42/曲/SET.DEF",
                        "Size": 5,
                        "ETag": '"etag-2"',
                        "LastModified": datetime(2026, 7, 25, tzinfo=timezone.utc),
                    }
                ]
            },
            {"Contents": []},
        ]


class FakeClient:
    def __init__(self):
        self.get_calls = []

    def head_bucket(self, **kwargs):
        assert kwargs == {"Bucket": "simfile-dtx"}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator()

    def head_object(self, **kwargs):
        assert "ChecksumMode" not in kwargs
        return {
            "ContentLength": 5,
            "ETag": '"etag-2"',
            "LastModified": datetime(2026, 7, 25, tzinfo=timezone.utc),
            "ContentType": "text/plain",
        }

    def get_object(self, **kwargs):
        self.get_calls.append(kwargs)
        return {
            "Body": BytesIO(b"chart"),
            "ContentLength": 5,
            "ETag": '"etag-2"',
            "LastModified": datetime(2026, 7, 25, tzinfo=timezone.utc),
        }


def test_adapter_lists_heads_and_opens_strong_and_weak_downloads():
    store = Boto3R2Store(FakeClient(), "simfile-dtx")
    listed = store.list_objects()
    assert listed[0].key == "42/曲/SET.DEF"
    assert store.head_object(listed[0].key).content_type == "text/plain"

    with store.open_object(listed[0].key, if_match='"etag-2"') as response:
        assert response.body.read() == b"chart"
    with store.open_object(listed[0].key, if_match=None):
        pass

    assert store.client.get_calls[0]["IfMatch"] == '"etag-2"'
    assert "IfMatch" not in store.client.get_calls[1]


def test_adapter_never_requests_provider_checksum_mode():
    store = Boto3R2Store(FakeClient(), "simfile-dtx")
    store.head_object("42/SET.DEF")


def test_sdk_logger_namespaces_are_pinned_to_warning():
    for name in ("boto3", "botocore", "urllib3"):
        logging.getLogger(name).setLevel(logging.DEBUG)
    Boto3R2Store(FakeClient(), "simfile-dtx")
    assert all(
        logging.getLogger(name).getEffectiveLevel() >= logging.WARNING
        for name in ("boto3", "botocore", "urllib3")
    )
```

Add tests that construct fake `ClientError`, `NoCredentialsError`, and timeout
exceptions and assert only closed codes and deterministic messages are exposed.
Include a test that monkeypatches the lazy import helper to raise `ImportError` and
asserts `missing_optional_dependency` plus the exact install hint
`uv pip install -e '.[r2]'`.

- [ ] **Step 2: Run the adapter tests and verify failure**

Run:

```bash
rtk uv run --extra r2 pytest tests/benchmark/test_r2_inventory.py -q
```

Expected: import fails because `src.benchmark.r2_inventory` does not exist.

- [ ] **Step 3: Define the protocol and download response without eager SDK imports**

Implement these public definitions in `src/benchmark/r2_inventory.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from logging import WARNING, getLogger
from typing import BinaryIO, ContextManager, Protocol

from src.benchmark.r2_corpus_models import (
    ErrorCode,
    HeadMetadata,
    ListedObject,
    R2Config,
    parse_etag,
)


@dataclass(frozen=True)
class ObjectDownload:
    body: BinaryIO
    size: int | None
    etag: str | None
    etag_is_weak: bool | None
    last_modified: datetime | None


class R2ObjectStore(Protocol):
    def validate_bucket(self) -> None:
        raise NotImplementedError

    def list_objects(self) -> tuple[ListedObject, ...]:
        raise NotImplementedError

    def head_object(self, key: str) -> HeadMetadata:
        raise NotImplementedError

    def open_object(
        self, key: str, if_match: str | None
    ) -> ContextManager[ObjectDownload]:
        raise NotImplementedError


class R2StoreError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str, object_key: str | None = None):
        super().__init__(message)
        self.code = code
        self.object_key = object_key
```

`ensure_r2_dependency()` must import `boto3`, `botocore.config.Config`, and botocore
exceptions inside the function. It raises `R2StoreError` with
`missing_optional_dependency` when imports fail. Module import itself must succeed in
the base environment. `Boto3R2Store.__init__` calls a small SDK-logger suppression
helper so direct adapter construction is protected as well as factory construction.

- [ ] **Step 4: Implement boto3 configuration and safe operation mapping**

`create_boto3_store(config)` must:

1. call `ensure_r2_dependency`;
2. set `boto3`, `botocore`, and `urllib3` logger levels to `WARNING`;
3. build `botocore.config.Config` with:

```python
Config(
    connect_timeout=config.connect_timeout_seconds,
    read_timeout=config.read_timeout_seconds,
    tcp_keepalive=True,
    retries={"mode": "standard", "total_max_attempts": config.max_attempts},
    max_pool_connections=max(
        16,
        config.head_concurrency + config.download_concurrency,
    ),
)
```

4. create an S3 client using the normalized endpoint, region `auto`, and the standard
credential provider chain, without passing access-key, secret-key, or session-token
arguments;
5. return `Boto3R2Store(client, config.bucket)`.

Implement paginated listing with no delimiter, strict required metadata validation,
post-HEAD response parsing, and a context-managed GET response that always closes the
SDK body. Listing identity fields are required. GET response ETag, content length, and
last-modified fields remain nullable in `ObjectDownload` so the weak-ETag verifier can
emit the specific `weak_etag_unverifiable` code instead of accepting incomplete
comparison metadata.

Add adapter assertions for all configured timeout/retry/pool values, a paginator with
more than 1,000 exact keys, and exhausted-retry errors. Confirm the adapter relies on
botocore's configured standard retries and has no outer general retry loop.

Map SDK outcomes to only:

| Operation | Stable codes |
| --- | --- |
| credential resolution | `missing_credentials` |
| auth rejection | `auth_failed` |
| bucket validation | `bucket_inaccessible` |
| root pagination | `root_list_failed` |
| object HEAD | `object_head_failed` |
| object GET | `object_get_failed` |
| conditional 412 | `source_changed_during_sync` |
| malformed SDK metadata | `object_metadata_invalid` |

Never include raw exception text, request IDs, response bodies, endpoint URLs, headers,
or SDK exception class names in `R2StoreError`.

- [ ] **Step 5: Run adapter tests, logging test, and base-import test**

Run:

```bash
rtk uv run --extra r2 pytest tests/benchmark/test_r2_inventory.py -q
rtk uv run python -c "import src.benchmark.r2_inventory"
rtk uv run ruff check src/benchmark/r2_inventory.py tests/benchmark/test_r2_inventory.py
rtk uv run black --check src/benchmark/r2_inventory.py tests/benchmark/test_r2_inventory.py
```

Expected: all adapter tests pass and the import succeeds without enabling the extra.

- [ ] **Step 6: Commit the adapter**

```bash
rtk git add src/benchmark/r2_inventory.py tests/benchmark/test_r2_inventory.py
rtk git commit -m "feat: add safe R2 object store adapter"
```

---

### Task 3: Implement complete root discovery and bounded metadata enrichment

**Files:**
- Modify: `src/benchmark/r2_inventory.py`
- Modify: `tests/benchmark/test_r2_inventory.py`

**Interfaces:**
- Consumes: `R2ObjectStore`, include/exclude sets, `MAX_SIMFILE_ID`, and configured HEAD concurrency.
- Produces: `build_inventory(store, include_ids, exclude_ids, head_concurrency) -> InventoryResult`.

- [ ] **Step 1: Add failing prefix, filter, and enrichment tests**

Append tests using a protocol fake that records every list and HEAD call:

```python
def listed(key: str, size: int = 5) -> ListedObject:
    return ListedObject(
        key=key,
        size=size,
        etag="listed-etag",
        etag_is_weak=False,
        last_modified=datetime(2026, 7, 25, tzinfo=timezone.utc),
    )


class FakeStore:
    def __init__(
        self,
        listed: list[ListedObject],
        heads: dict[str, HeadMetadata | R2StoreError] | None = None,
    ):
        self.listed = tuple(listed)
        self.heads = heads or {}
        self.list_calls = 0
        self.head_calls: list[str] = []

    def list_objects(self) -> tuple[ListedObject, ...]:
        self.list_calls += 1
        return self.listed

    def head_object(self, key: str) -> HeadMetadata:
        self.head_calls.append(key)
        result = self.heads.get(
            key,
            HeadMetadata(None, None, None, None, None),
        )
        if isinstance(result, R2StoreError):
            raise result
        return result


def test_inventory_discovers_before_filtering_and_preserves_exact_keys():
    store = FakeStore(
        listed=[
            listed("42/SET.DEF"),
            listed("42/assets/曲 name.ogg"),
            listed("99/chart.dtx"),
            listed("README"),
            listed("other/file.txt"),
        ]
    )
    result = build_inventory(
        store,
        include_ids=frozenset({42}),
        exclude_ids=frozenset(),
        head_concurrency=2,
    )
    assert store.list_calls == 1
    assert [item.simfile_id for item in result.simfiles] == [42]
    assert result.simfiles_discovered == 2
    assert result.simfiles_excluded_by_filter == 1
    assert result.objects_listed == 5
    assert result.malformed_root_keys == ("README", "other/file.txt")
    assert result.simfiles[0].objects[1].key == "42/assets/曲 name.ogg"
    assert sorted(store.head_calls) == ["42/SET.DEF", "42/assets/曲 name.ogg"]


def test_ambiguous_aliases_are_quarantined_even_when_included():
    store = FakeStore(listed=[listed("1/chart.dtx"), listed("01/chart.dtx")])
    result = build_inventory(store, frozenset({1}), frozenset(), 2)
    assert result.simfiles == ()
    assert result.ambiguous_prefixes == {1: ("01/", "1/")}
    assert result.root_errors[0].code == "ambiguous_simfile_prefix"


def test_nested_marker_only_prefix_is_empty_and_retains_markers():
    store = FakeStore(
        listed=[
            listed("42/", size=0),
            listed("42/assets/", size=0),
        ]
    )
    result = build_inventory(store, frozenset(), frozenset(), 2)
    row = result.simfiles[0]
    assert row.sync_status == "empty"
    assert [item.key for item in row.objects] == ["42/", "42/assets/"]
    assert all(item.cache_status == "not_selected" for item in row.objects)
    assert row.sync_errors[0].code == "empty_prefix"


def test_explicit_missing_id_creates_empty_row_with_no_objects():
    result = build_inventory(FakeStore(listed=[]), frozenset({42}), frozenset(), 2)
    assert result.simfiles[0].simfile_id == 42
    assert result.simfiles[0].objects == ()
    assert result.simfiles[0].sync_errors[0].code == "empty_prefix"
```

Also add tests for:

- include/exclude overlap where exclude wins;
- IDs `0` and `MAX_SIMFILE_ID`;
- `MAX_SIMFILE_ID + 1` becoming `malformed_root_key`;
- a zero-byte non-slash key counting as content, not a folder marker;
- a HEAD failure retaining listing metadata and making only that simfile partial;
- malformed HEAD metadata retaining valid listing identity and emitting
  `object_metadata_invalid`;
- HEAD values overriding listing size, ETag, mtime, and content type;
- no provider checksum fields in the resulting object;
- exact preservation of keys containing spaces, `+`, `%2F`, `#`, `?`, and
  non-ASCII code points without URL decoding;
- deterministic numeric/prefix/object ordering;
- a synchronization barrier fake proving active HEAD calls never exceed the supplied bound.

- [ ] **Step 2: Run the new inventory tests and verify failure**

Run:

```bash
rtk uv run --extra r2 pytest tests/benchmark/test_r2_inventory.py -q
```

Expected: tests fail because `build_inventory` and inventory helpers do not exist.

- [ ] **Step 3: Implement root classification before filtering**

Implement `_classify_root(listed: tuple[ListedObject, ...])` returning
`tuple[dict[str, list[ListedObject]], tuple[str, ...], dict[int, tuple[str, ...]]]`.
The three values are exact-prefix groups, malformed keys, and quarantined numeric
aliases.

Classification rules:

1. Require a slash and a digit-only first segment.
2. Parse the segment and require `0 <= value <= MAX_SIMFILE_ID`.
3. Preserve the exact segment plus `/` as `object_prefix`.
4. Collect every exact prefix before detecting numeric aliases.
5. Sort malformed keys by Unicode code point.
6. Sort conflicting prefixes by exact string.
7. Apply include/exclude only after classification and ambiguity detection.
8. Count valid, non-quarantined discovered simfiles removed by filters.
9. Record total root objects and valid, non-quarantined discovered simfiles before
   filters.
10. Derive requested-but-absent IDs as
    `include_ids - exclude_ids - discovered_ids - ambiguous_ids`, preserving exclude
    precedence over include, and add each as a canonical `<id>/` empty row. Excluded
    and quarantined (ambiguous) IDs never produce absent rows even when explicitly
    included.

- [ ] **Step 4: Implement bounded HEAD merge and row statuses**

Use `ThreadPoolExecutor(max_workers=head_concurrency)` only for filtered,
non-quarantined objects. Submit exact keys and reassemble results in deterministic key
order, independent of completion order.

For successful HEAD:

- overwrite listing size, ETag/weakness, and last-modified when returned;
- add nullable content type;
- never request or store checksum fields.

For failed HEAD:

- retain the listing identity and `content_type=None`;
- append the adapter's allowlisted `object_head_failed` or
  `object_metadata_invalid` error to object and row errors;
- set the row `partial`;
- continue unrelated objects and simfiles.

A discovered row is `empty` only when every object satisfies
`size == 0 and key.endswith("/")`; add one deterministic `empty_prefix` row error
while retaining every marker. Any other row with no errors is `complete`. Each
ambiguous-root error names the numeric ID and its sorted conflicting prefixes without
allowing an explicit include to restore a row.

- [ ] **Step 5: Run focused tests and format checks**

Run:

```bash
rtk uv run --extra r2 pytest tests/benchmark/test_r2_inventory.py -q
rtk uv run ruff check src/benchmark/r2_inventory.py tests/benchmark/test_r2_inventory.py
rtk uv run black --check src/benchmark/r2_inventory.py tests/benchmark/test_r2_inventory.py
```

Expected: all inventory and adapter tests pass.

- [ ] **Step 6: Commit root inventory**

```bash
rtk git add src/benchmark/r2_inventory.py tests/benchmark/test_r2_inventory.py
rtk git commit -m "feat: inventory R2 simfile prefixes"
```

---

### Task 4: Implement strict provenance loading

**Files:**
- Create: `src/benchmark/corpus_provenance.py`
- Create: `tests/benchmark/test_corpus_provenance.py`
- Create: `config/corpus-provenance.json`

**Interfaces:**
- Consumes: optional provenance path, `MAX_SIMFILE_ID`, `PROVENANCE_SCHEMA`.
- Produces: `load_provenance(path) -> dict[int, ProvenanceRecord]` and `provenance_for(mapping, simfile_id) -> ProvenanceRecord`.

- [ ] **Step 1: Write failing provenance validation tests**

Create `tests/benchmark/test_corpus_provenance.py`:

```python
import json
from pathlib import Path

import pytest

from src.benchmark.corpus_provenance import load_provenance, provenance_for


def write_mapping(path: Path, simfiles: dict[str, object], schema: str = "crux.corpus-provenance/v1"):
    path.write_text(
        json.dumps({"schema_version": schema, "simfiles": simfiles}),
        encoding="utf-8",
    )


def test_loads_known_record_and_supplies_explicit_unknown_default(tmp_path: Path):
    path = tmp_path / "provenance.json"
    write_mapping(
        path,
        {
            "42": {
                "source_origin": "personal",
                "source_author_or_pack": "Example Pack",
                "source_reference": "private archive",
                "rights_status": "privately_authorized",
                "redistribution_allowed": False,
                "provenance_notes": "Local benchmark use.",
            }
        },
    )
    records = load_provenance(path)
    assert records[42].rights_status == "privately_authorized"
    assert records[42].redistribution_allowed is False
    assert provenance_for(records, 99).rights_status == "unknown"
    assert provenance_for(records, 99).redistribution_allowed is None


@pytest.mark.parametrize(
    "schema",
    ["", "crux.corpus-provenance/v2", "other"],
)
def test_rejects_unknown_schema(schema, tmp_path: Path):
    path = tmp_path / "provenance.json"
    write_mapping(path, {}, schema=schema)
    with pytest.raises(ValueError, match="schema_version"):
        load_provenance(path)


def test_rejects_duplicate_ids_after_numeric_normalization(tmp_path: Path):
    path = tmp_path / "provenance.json"
    write_mapping(path, {"1": {}, "01": {}})
    with pytest.raises(ValueError, match="duplicate simfile ID"):
        load_provenance(path)
```

Add cases for malformed JSON, non-object `simfiles`, negative and above-safe-range
keys, unknown record fields, wrong string types, and non-boolean
`redistribution_allowed`. Assert error messages never echo field values. Add cases
for a nonexistent `--provenance-file` and an unreadable one (e.g. a path inside a
directory with `0o000` permissions, skipped on platforms where the test runner
cannot create it): both must raise `ValueError` with a `provenance file` message so
the command boundary maps them to `provenance_invalid` rather than
`internal_error`.

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run:

```bash
rtk uv run pytest tests/benchmark/test_corpus_provenance.py -q
```

Expected: collection fails because `src.benchmark.corpus_provenance` does not exist.

- [ ] **Step 3: Implement the strict loader and unknown default**

Implement:

```python
from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.r2_corpus_models import (
    MAX_SIMFILE_ID,
    PROVENANCE_SCHEMA,
    ProvenanceRecord,
)

ALLOWED_FIELDS = {
    "source_origin",
    "source_author_or_pack",
    "source_reference",
    "rights_status",
    "redistribution_allowed",
    "provenance_notes",
}


def load_provenance(path: Path | None) -> dict[int, ProvenanceRecord]:
    if path is None:
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"provenance file not found: {path}") from exc
    except OSError as exc:
        raise ValueError(f"provenance file unreadable: {path}") from exc
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != PROVENANCE_SCHEMA:
        raise ValueError("unsupported provenance schema_version")
    raw_simfiles = payload.get("simfiles")
    if not isinstance(raw_simfiles, dict):
        raise ValueError("provenance simfiles must be an object")

    records: dict[int, ProvenanceRecord] = {}
    for raw_id, raw_record in raw_simfiles.items():
        simfile_id = _parse_id(raw_id)
        if simfile_id in records:
            raise ValueError("duplicate simfile ID after numeric normalization")
        records[simfile_id] = _parse_record(raw_record)
    return records


def provenance_for(
    mapping: dict[int, ProvenanceRecord],
    simfile_id: int,
) -> ProvenanceRecord:
    return mapping.get(simfile_id, ProvenanceRecord())
```

`_parse_record` must reject unknown keys, require nullable string fields to be string
or null, require `rights_status` to be a non-empty string when present, and require
`redistribution_allowed` to be exactly bool or null. Free-text rights values remain
allowed; downstream publication is permitted only for an exact `True`.

- [ ] **Step 4: Add the checked-in empty mapping**

Create `config/corpus-provenance.json` with exact bytes:

```json
{
  "schema_version": "crux.corpus-provenance/v1",
  "simfiles": {}
}
```

- [ ] **Step 5: Run tests and style checks**

Run:

```bash
rtk uv run pytest tests/benchmark/test_corpus_provenance.py -q
rtk uv run ruff check src/benchmark/corpus_provenance.py tests/benchmark/test_corpus_provenance.py
rtk uv run black --check src/benchmark/corpus_provenance.py tests/benchmark/test_corpus_provenance.py
```

Expected: all provenance tests pass.

- [ ] **Step 6: Commit provenance support**

```bash
rtk git add config/corpus-provenance.json src/benchmark/corpus_provenance.py \
  tests/benchmark/test_corpus_provenance.py
rtk git commit -m "feat: add corpus provenance mapping"
```

---

### Task 5: Implement the canonical cache index and whole-run writer lock

**Files:**
- Create: `src/benchmark/corpus_cache.py`
- Create: `tests/benchmark/test_corpus_cache.py`

**Interfaces:**
- Consumes: `R2Config`, `RemoteObject`, exact cache root.
- Produces: `CacheIndexEntry`, `CacheValidation`, `CacheIndexStore`, and `cache_writer_lock`.

- [ ] **Step 1: Write failing cache-index round-trip and validation tests**

Create `tests/benchmark/test_corpus_cache.py` with:

```python
import json
from pathlib import Path

import pytest

from src.benchmark.corpus_cache import (
    CacheIndexEntry,
    CacheIndexStore,
    cache_writer_lock,
)


def entry(key: str = "42/SET.DEF") -> CacheIndexEntry:
    return CacheIndexEntry(
        source_endpoint_sha256="a" * 64,
        bucket="simfile-dtx",
        key=key,
        etag="etag",
        etag_is_weak=False,
        size=5,
        last_modified="2026-07-25T00:00:00Z",
        sha256="b" * 64,
        cache_path=f"sha256/bb/{'b' * 64}",
    )


def test_index_checkpoint_is_canonical_and_restart_readable(tmp_path: Path):
    store = CacheIndexStore.load(tmp_path)
    store.checkpoint(entry("42/z.dtx"))
    store.checkpoint(entry("42/a.dtx"))

    payload = json.loads((tmp_path / "index-v1.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "crux.r2-cache-index/v1"
    assert [item["key"] for item in payload["entries"]] == ["42/a.dtx", "42/z.dtx"]
    restarted = CacheIndexStore.load(tmp_path)
    assert restarted.get("a" * 64, "simfile-dtx", "42/a.dtx") is not None


def test_invalid_index_json_and_schema_fail_closed(tmp_path: Path):
    (tmp_path / "index-v1.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cache index"):
        CacheIndexStore.load(tmp_path)
    (tmp_path / "index-v1.json").write_text(
        '{"schema_version":"crux.r2-cache-index/v2","entries":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema"):
        CacheIndexStore.load(tmp_path)


def test_live_writer_lock_fails_fast(tmp_path: Path):
    with cache_writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match="cache_locked"):
            with cache_writer_lock(tmp_path):
                raise AssertionError("second writer must not enter")
```

Add tests that:

- reject absolute cache paths and any path containing `..`;
- reject duplicate `(endpoint_hash, bucket, key)` entries;
- sort entries by endpoint hash, bucket, and exact key;
- contain no cache-profile or provider-checksum fields;
- monkeypatch `os.replace`, file `flush`, `os.fsync`, and directory opens to assert
  file-before-directory durability ordering;
- retain one lock-file object for the full context lifetime;
- replace `fcntl` import with failure and assert `unsupported_platform`;
- start two threads calling `checkpoint` and prove writes never overlap.

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run:

```bash
rtk uv run pytest tests/benchmark/test_corpus_cache.py -q
```

Expected: collection fails because `src.benchmark.corpus_cache` does not exist.

- [ ] **Step 3: Implement cache index records and strict loading**

Create `src/benchmark/corpus_cache.py` with:

```python
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Iterator
from uuid import uuid4

from src.benchmark.r2_corpus_models import CACHE_INDEX_SCHEMA


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
    def __init__(self, cache_dir: Path, entries: dict[tuple[str, str, str], CacheIndexEntry]):
        self.cache_dir = cache_dir
        self._entries = entries
        self._checkpoint_lock = Lock()

    def get(self, endpoint_hash: str, bucket: str, key: str) -> CacheIndexEntry | None:
        return self._entries.get((endpoint_hash, bucket, key))

    def checkpoint(self, entry: CacheIndexEntry) -> None:
        with self._checkpoint_lock:
            self._entries[(entry.source_endpoint_sha256, entry.bucket, entry.key)] = entry
            self._publish_locked()
```

Add the classmethod
`load(cls, cache_dir: Path) -> CacheIndexStore` with this concrete JSON validation:

1. return an empty store if `index-v1.json` is absent;
2. parse UTF-8 JSON;
3. require only schema `crux.r2-cache-index/v1`;
4. require `entries` as a list;
5. construct every dataclass with exact field types;
6. reject booleans where integers are required, require `etag_is_weak` to be a
   boolean, require a nonnegative size, and require canonical UTC `last_modified`;
7. validate lower-case 64-hex SHA fields;
8. inspect raw slash-separated path parts before constructing `PurePosixPath`, and
   reject roots, empty parts, `.`, or `..`;
9. require `cache_path == f"sha256/{sha256[:2]}/{sha256}"`;
10. reject duplicate composite identities.

`_publish_locked` must encode:

```python
payload = {
    "schema_version": CACHE_INDEX_SCHEMA,
    "entries": [
        asdict(entry)
        for entry in sorted(
            self._entries.values(),
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
```

Write `content` to a unique sibling of `index-v1.json`, flush, `fsync`, replace, and
`fsync` the cache directory. Clean the temporary file if publication fails.

- [ ] **Step 4: Implement the POSIX lock with one descriptor**

Implement:

```python
@contextmanager
def cache_writer_lock(cache_dir: Path) -> Iterator[None]:
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("unsupported_platform") from exc

    _ensure_durable_directory(cache_dir)
    lock_path = cache_dir / ".index-v1.lock"
    handle = lock_path.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("cache_locked") from exc
        except OSError as exc:
            raise RuntimeError("unsupported_platform") from exc
        yield
    finally:
        handle.close()
```

Do not open the lock path anywhere else. The orchestration task will hold this context
across inventory, cache, manifest, report, and pointer publication.
`_ensure_durable_directory` creates each missing directory component and `fsync`s its
parent before continuing, and does nothing when the complete path already exists.

- [ ] **Step 5: Run cache-index and lock tests**

Run:

```bash
rtk uv run pytest tests/benchmark/test_corpus_cache.py -q
rtk uv run ruff check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
rtk uv run black --check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
```

Expected: canonical index, lock, durability, and thread-serialization tests pass.

- [ ] **Step 6: Commit the cache-index foundation**

```bash
rtk git add src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
rtk git commit -m "feat: add durable R2 cache index"
```

---

### Task 6: Implement selective cache validation, repair, and download

**Files:**
- Modify: `src/benchmark/corpus_cache.py`
- Modify: `tests/benchmark/test_corpus_cache.py`

**Interfaces:**
- Consumes: enriched `SimfileInventory`, `R2ObjectStore`, `CacheIndexStore`, `R2Config`, and dry-run flag.
- Produces: `sync_cache(...) -> CacheSyncResult` with reproducible rows and invocation-only actions.

- [ ] **Step 1: Add failing selection, hit, miss, and download tests**

Append cache-local constructors rather than importing test helpers across modules:

```python
FIXED_MTIME = datetime(2026, 7, 25, tzinfo=timezone.utc)


def remote_object(
    key: str = "42/chart.dtx",
    *,
    size: int = 5,
    etag: str = "etag",
    etag_is_weak: bool = False,
) -> RemoteObject:
    return RemoteObject(
        key=key,
        size=size,
        etag=etag,
        etag_is_weak=etag_is_weak,
        last_modified=FIXED_MTIME,
        content_type="text/plain",
    )


def simfile(*objects: RemoteObject) -> SimfileInventory:
    return SimfileInventory(42, "42/", tuple(objects), "complete")


def config() -> R2Config:
    return R2Config("https://example.invalid", "a" * 64, "simfile-dtx")


def empty_index(cache_dir: Path) -> CacheIndexStore:
    return CacheIndexStore.load(cache_dir)
```

Define a `FakeStore` implementing `open_object` with recorded `OpenCall` values and
an `ObjectDownload` backed by `BytesIO`; let each test inject the response ETag,
size, mtime, bytes, or `R2StoreError`. Define `seeded_index` by computing the local
file SHA-256 and checkpointing the exact remote identity. Define
`index_with_local_state` by starting from `seeded_index` and then deleting,
truncating, or replacing the local body for the named state.

Append tests that build one complete simfile with these keys:

```python
(
    remote_object("42/SET.DEF"),
    remote_object("42/charts/MAS.DTX"),
    remote_object("42/notes.TxT"),
    remote_object("42/audio/song.ogg"),
    remote_object("42/assets/", size=0),
)
```

Assert only the first three are selected. Add:

```python
def test_verified_hit_reads_no_remote_body(tmp_path: Path):
    cache_path = tmp_path / "sha256" / "2c" / sha256(b"chart").hexdigest()
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"chart")
    index = seeded_index(tmp_path, remote_object(), cache_path)
    store = FakeStore()

    result = sync_cache(
        simfiles=(simfile(remote_object()),),
        store=store,
        index=index,
        config=config(),
        dry_run=False,
    )

    assert store.open_calls == []
    assert result.simfiles[0].objects[0].cache_status == "verified"
    assert result.actions[0].action == "cache_hit"


@pytest.mark.parametrize("local_state", ["missing", "size_mismatch", "sha256_mismatch"])
def test_dry_run_reports_repair_as_planned_without_get(tmp_path: Path, local_state: str):
    index = index_with_local_state(tmp_path, local_state)
    store = FakeStore()
    result = sync_cache((simfile(remote_object()),), store, index, config(), dry_run=True)
    assert store.open_calls == []
    assert result.actions[0].action == "planned"
    assert result.actions[0].miss_reason == local_state


def test_strong_etag_uses_if_match_and_weak_etag_does_not(tmp_path: Path):
    strong_store = FakeStore(body=b"strong")
    sync_cache(
        (simfile(remote_object(etag_is_weak=False)),),
        strong_store,
        empty_index(tmp_path),
        config(),
        False,
    )
    assert strong_store.open_calls[0].if_match == '"etag"'

    weak_store = FakeStore(body=b"weak", response_etag='W/"etag"')
    weak_result = sync_cache(
        (simfile(remote_object(size=4, etag_is_weak=True)),),
        weak_store,
        empty_index(tmp_path),
        config(),
        False,
    )
    assert weak_store.open_calls[0].if_match is None
    assert weak_result.simfiles[0].objects[0].cache_status == "verified"
    assert weak_result.actions[0].action == "downloaded"
```

Add cases for:

- remote identity mismatch producing `remote_changed`;
- weak response missing ETag/size/mtime producing `weak_etag_unverifiable`;
- weak response metadata drift producing `source_changed_during_sync`;
- strong-ETag GET whose response `Last-Modified` or `Content-Length` differs
  from the HEAD identity (metadata-only rewrite preserving the content ETag)
  producing `source_changed_during_sync` with no accepted bytes and no
  checkpoint;
- strong 412 producing `source_changed_during_sync` with no SDK details;
- streamed byte-count mismatch;
- failed GET and failed local write;
- successful repair producing no `cache_corrupt`; the repaired entry's indexed
  `cache_path` is the digest-derived final path, so a size/SHA-mismatched indexed
  entry means that final path already holds corrupt bytes — repair must replace
  those bytes, not reject them;
- failed repair of a locally missing, size-mismatched, or SHA-mismatched entry
  producing `cache_corrupt` plus the operational error;
- a failed `remote_changed` download producing only its operational error;
- body temp paths always under `sha256/.incoming`;
- incoming/final `st_dev` mismatch failing before replace;
- file `fsync`, `os.replace`, shard-directory `fsync`, then index checkpoint order;
- deduplication when a final SHA path already contains identical bytes;
- replacement of an existing final SHA path whose bytes mismatch **only when that
  path is the indexed entry currently being repaired** (the corrupt-indexed case
  above);
- rejection (no `os.replace`, no checkpoint, sanitized `cache_corrupt` with a
  collision/`artifact_write_failed` operational error) when a final SHA path
  already contains mismatched bytes for a **fresh install or an entry whose
  indexed `cache_path` is not this path** — i.e. an unrelated collision or
  filesystem anomaly, never the indexed corrupt entry itself;
- failed temporary-file cleanup;
- independent download workers never exceeding `download_concurrency`;
- restart after checkpoint becoming a verified hit.

- [ ] **Step 2: Run the expanded cache tests and verify failure**

Run:

```bash
rtk uv run --extra r2 pytest tests/benchmark/test_corpus_cache.py -q
```

Expected: new tests fail because cache selection and synchronization are absent.

- [ ] **Step 3: Implement cache selection and local validation**

Add:

```python
def is_selected(key: str) -> bool:
    basename = key.rsplit("/", 1)[-1].lower()
    lowered = key.lower()
    return basename == "set.def" or lowered.endswith((".dtx", ".txt"))


def validate_cached_body(
    cache_dir: Path,
    entry: CacheIndexEntry | None,
) -> CacheValidation:
    if entry is None:
        return CacheValidation("missing", None)
    path = cache_dir / PurePosixPath(entry.cache_path)
    if not path.is_file():
        return CacheValidation("missing", entry)
    digest, size = _hash_file(path)
    if size != entry.size:
        return CacheValidation("size_mismatch", entry)
    if digest != entry.sha256:
        return CacheValidation("sha256_mismatch", entry)
    return CacheValidation("verified", entry)
```

Call this only after exact endpoint, bucket, key, ETag/weakness, remote size, and
canonical mtime comparison. A mismatched remote identity is `remote_changed`, not
local corruption.

- [ ] **Step 4: Implement concurrent real and dry-run synchronization**

Implement
`sync_cache(simfiles: tuple[SimfileInventory, ...], store: R2ObjectStore,
index: CacheIndexStore, config: R2Config, dry_run: bool) -> CacheSyncResult`
with this concrete flow:

1. Leave folder markers and non-profile objects `not_selected`.
2. Mark HEAD-failed selected objects `failed` without attempting GET.
3. Compare remote identity with the index, then hash matching local entries.
4. Emit `cache_hit` and `verified` only for byte-count and SHA matches.
5. In dry-run, emit `planned` for every miss and do not mutate rows, cache, or index.
6. In real mode, submit misses to `ThreadPoolExecutor(config.download_concurrency)`.
7. Rebuild objects and rows by original indices rather than future completion order.
8. Set a row `failed` only when all selected objects fail; otherwise any failed
   selected object makes it `partial`; preserve `empty` and never erase prior
   inventory/HEAD errors.

Implement `_download_one` to:

- create and durability-sync `<cache-dir>/sha256/.incoming`;
- reconstruct strong `If-Match` as a quoted ETag; `If-Match` only asserts the
  content ETag, so after a successful strong-ETag GET also verify the response's
  `Last-Modified` and `Content-Length` match the HEAD identity exactly — a
  metadata-only rewrite between HEAD and GET can preserve the strong ETag while
  changing `Last-Modified`, which would otherwise checkpoint a stale mtime into
  the cache and manifest. On mismatch, emit `source_changed_during_sync` (the
  same code as a 412) and do not accept bytes or checkpoint;
- omit `If-Match` for weak ETags and verify response identity (ETag weakness,
  size, mtime) before accepting bytes;
- stream in fixed-size chunks to a unique incoming file while hashing/counting;
- flush and `fsync` the incoming file;
- create and sync the two-hex shard directory;
- require incoming and shard directories to share `st_dev`;
- verify an existing final path has the expected bytes; if they match, dedup
  (no rewrite); if they mismatch, distinguish the two cases:
  - **indexed-corrupt repair**: the entry being downloaded has `cache_path`
    equal to this final path and was flagged `size_mismatch`/`sha256_mismatch`
    by `validate_cached_body` — `os.replace` the freshly downloaded bytes into
    the extensionless final path (the corrupt bytes are exactly what repair
    must overwrite);
  - **unrelated collision**: any other case (fresh install, or the path belongs
    to a different indexed entry) — do **not** `os.replace`; emit a sanitized
    `cache_corrupt` row with an `artifact_write_failed` operational error, do
    not checkpoint, and delete the incoming file;
  in both branches serialize same-digest installations under
  `_body_install_lock`;
- `fsync` the shard directory;
- call `index.checkpoint` only afterward;
- delete the invocation's incoming file on every failure path.

Use one module-local `_body_install_lock = Lock()` only around the existing-final
verification, `os.replace`, and shard-directory `fsync`. Do not hold it during
network reads or hashing. The separate `CacheIndexStore` checkpoint lock continues to
serialize index mutation/publication.

- [ ] **Step 5: Run cache tests and restart/idempotence coverage**

Run:

```bash
rtk uv run --extra r2 pytest tests/benchmark/test_corpus_cache.py -q
rtk uv run ruff check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
rtk uv run black --check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
```

Expected: all cache selection, durability, repair, concurrency, and restart tests pass.

- [ ] **Step 6: Commit cache synchronization**

```bash
rtk git add src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
rtk git commit -m "feat: synchronize selected R2 corpus files"
```

---

### Task 7: Build deterministic manifests and publish immutable versions

**Files:**
- Create: `src/benchmark/corpus_manifest.py`
- Create: `tests/benchmark/test_corpus_manifest.py`

**Interfaces:**
- Consumes: `CacheSyncResult`, provenance records, source endpoint identity, bucket, and a
  caller-supplied publication timestamp.
- Produces: canonical rows, `RenderedManifest`, immutable manifest bytes, and a
  separately replaceable `latest.json` pointer that orchestration can defer until
  the attempt's report is durable.

- [ ] **Step 1: Write failing canonical-row and identity tests**

Create `tests/benchmark/test_corpus_manifest.py`. Add compact constructors for
`RemoteObject`, `SimfileInventory`, and `CacheSyncResult` at the top of the file so
every test builds complete records without mocks hidden in another suite. The
`render_for_action` helper routes through `sync_cache`, so also import
`sync_cache`, `CacheIndexStore`, `CacheIndexEntry`, and `R2Config` from
`src.benchmark.corpus_cache` and `r2_corpus_models`, plus `ObjectDownload` from
`src.benchmark.r2_inventory` (where Task 2 defines it), plus `contextmanager`,
`BytesIO`, `sha256`, and `Path` for the in-file fakes. Define a local `OpenCall`
record for the fake store's call log rather than importing it from another test
suite.

```python
FIXED_TIME = datetime(2026, 7, 25, 1, 2, 3, tzinfo=timezone.utc)


def make_simfile(
    simfile_id: int = 2,
    *,
    key: str | None = None,
    status: SimfileStatus = "complete",
    error_code: ErrorCode | None = None,
) -> SimfileInventory:
    object_key = key or f"{simfile_id}/chart.dtx"
    errors = (
        ()
        if error_code is None
        else (SyncError("object", error_code, "Safe deterministic message.", object_key),)
    )
    remote = RemoteObject(
        key=object_key,
        size=5,
        etag="etag",
        etag_is_weak=False,
        last_modified=FIXED_TIME,
        content_type="text/plain",
        cache_status="verified",
        sha256="c" * 64,
        cache_path=f"sha256/cc/{'c' * 64}",
        errors=errors,
    )
    return SimfileInventory(
        simfile_id=simfile_id,
        object_prefix=f"{simfile_id}/",
        objects=(remote,),
        sync_status=status,
        sync_errors=errors,
    )


def render_fixture(
    *,
    key: str = "2/chart.dtx",
    status: SimfileStatus = "complete",
    error_code: ErrorCode | None = None,
    source_reference: str | None = None,
) -> RenderedManifest:
    provenance = {2: ProvenanceRecord(source_reference=source_reference)}
    rows = build_manifest_rows(
        (make_simfile(2, key=key, status=status, error_code=error_code),),
        provenance,
        "a" * 64,
        "simfile-dtx",
    )
    return render_manifest(rows)


def render_for_action(action: CacheActionName, tmp_path: Path) -> RenderedManifest:
    # Route through sync_cache so downloaded and cache_hit actions are translated
    # into identical durable (verified) state before rendering, rather than
    # bypassing the cache-synchronization path by handing build_manifest_rows
    # manually-constructed simfiles that already claim "verified".
    body = b"chart"
    store = _ManifestFakeStore(body=body)
    if action == "cache_hit":
        index = _seeded_manifest_index(tmp_path, body=body)
    else:
        index = CacheIndexStore.load(tmp_path)

    cache_result = sync_cache(
        (make_simfile(),),
        store,
        index,
        _manifest_config(),
        False,
    )
    assert cache_result.simfiles[0].objects[0].cache_status == "verified"
    assert cache_result.actions[0].action == action

    rows = build_manifest_rows(
        cache_result.simfiles,
        {},
        "a" * 64,
        "simfile-dtx",
    )
    return render_manifest(rows)


def _manifest_config() -> R2Config:
    return R2Config("https://example.invalid", "a" * 64, "simfile-dtx")


class _ManifestFakeStore:
    def __init__(self, *, body: bytes = b"chart"):
        self._body = body
        self.open_calls: list[OpenCall] = []

    @contextmanager
    def open_object(self, key: str, if_match: str | None):
        self.open_calls.append(OpenCall(key, if_match))
        yield ObjectDownload(
            body=BytesIO(self._body),
            size=len(self._body),
            etag='"etag"',
            etag_is_weak=False,
            last_modified=FIXED_TIME,
        )


@dataclass(frozen=True)
class OpenCall:
    key: str
    if_match: str | None


def _seeded_manifest_index(cache_dir: Path, *, body: bytes) -> CacheIndexStore:
    digest = sha256(body).hexdigest()
    cache_path = f"sha256/{digest[:2]}/{digest}"
    (cache_dir / cache_path).parent.mkdir(parents=True, exist_ok=True)
    (cache_dir / cache_path).write_bytes(body)
    entry = CacheIndexEntry(
        source_endpoint_sha256="a" * 64,
        bucket="simfile-dtx",
        key="2/chart.dtx",
        etag="etag",
        etag_is_weak=False,
        size=len(body),
        last_modified="2026-07-25T01:02:03Z",
        sha256=digest,
        cache_path=cache_path,
    )
    store = CacheIndexStore.load(cache_dir)
    store.checkpoint(entry)
    return store
```

Cover these cases:

```python
def test_render_manifest_is_order_independent_for_inputs():
    first = render_manifest(
        build_manifest_rows(
            simfiles=(make_simfile(10), make_simfile(2)),
            provenance={},
            source_endpoint_sha256="a" * 64,
            bucket="bucket",
        )
    )
    second = render_manifest(
        build_manifest_rows(
            simfiles=(make_simfile(2), make_simfile(10)),
            provenance={},
            source_endpoint_sha256="a" * 64,
            bucket="bucket",
        )
    )

    assert first.corpus_version == second.corpus_version
    assert first.content == second.content
    assert first.content.endswith(b"\n")
    assert [row["simfile_id"] for row in first.rows] == [2, 10]


def test_render_manifest_hashes_the_self_reference_free_bytes():
    rendered = render_manifest(
        build_manifest_rows(
            simfiles=(make_simfile(2),),
            provenance={},
            source_endpoint_sha256="b" * 64,
            bucket="bucket",
        )
    )
    identity_bytes = b"".join(
        canonical_json_line(
            {key: value for key, value in row.items() if key != "corpus_version"}
        )
        for row in rendered.rows
    )

    assert rendered.corpus_version == f"sha256:{sha256(identity_bytes).hexdigest()}"
    assert all(row["corpus_version"] == rendered.corpus_version for row in rendered.rows)


def test_cache_hit_and_downloaded_actions_have_identical_manifest_identity(tmp_path: Path):
    downloaded = render_for_action("downloaded", tmp_path)
    cache_hit = render_for_action("cache_hit", tmp_path)

    assert downloaded.content == cache_hit.content
    assert downloaded.corpus_version == cache_hit.corpus_version


def test_status_error_and_provenance_changes_rekey_the_manifest():
    baseline = render_fixture()

    assert render_fixture(status="partial").corpus_version != baseline.corpus_version
    assert (
        render_fixture(error_code="object_head_failed").corpus_version
        != baseline.corpus_version
    )
    assert (
        render_fixture(source_reference="private archive").corpus_version
        != baseline.corpus_version
    )


def test_manifest_preserves_non_ascii_keys_without_ascii_escaping():
    rendered = render_fixture(key="7/音楽/譜面.DTX")

    assert "音楽".encode() in rendered.content
    assert b"\\u97f3" not in rendered.content
```

Also assert:

- objects sort by exact key and errors sort by
  `(scope, code, object_key or "", message)`;
- each object contains the complete manifest contract and no cache action, miss
  reason, invocation ID, report path, provider checksum, or publication time;
- the reserved object `version` field is always `None`;
- each row includes `cache_profile="setdef_dtx_txt_v1"` and
  `source_discovery_method="r2_list_objects_v2"`;
- source identity is the normalized endpoint SHA-256, never the raw endpoint;
- manifest timestamps are UTC whole-second `Z` strings;
- empty and failed rows still have stable shapes;
- changing only object enumeration order does not change bytes;
- changing remote metadata or locally verified SHA-256 does change bytes.
- normalized payload rows omit `corpus_version`; only final rendered rows contain it.

Run the new tests once and confirm they fail because the module does not exist:

```bash
rtk uv run pytest tests/benchmark/test_corpus_manifest.py -q
```

- [ ] **Step 2: Implement canonical JSONL and the two-pass corpus identity**

Implement these exact public functions:

```python
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
    return tuple(
        _build_row(
            simfile,
            provenance_for(provenance, simfile.simfile_id),
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
    final_rows = tuple(
        {"corpus_version": corpus_version, **row} for row in rows
    )
    content = b"".join(canonical_json_line(row) for row in final_rows)
    manifest_sha256 = hashlib.sha256(content).hexdigest()
    return RenderedManifest(final_rows, corpus_version, manifest_sha256, content)
```

Use this row contract:

```python
{
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
```

Each object dictionary must contain:

```python
{
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
```

Serialize each `SyncError` exactly as:

```python
{
    "scope": error.scope,
    "code": error.code,
    "object_key": error.object_key,
    "message": error.message,
}
```

Keep `content_type` nullable and `cache_status` limited to durable corpus facts
(`verified`, `failed`, or `not_selected`). `sync_cache` has already translated both
download and cache-hit actions to `verified` on the immutable object records.

- [ ] **Step 3: Write failing immutable-publication tests**

Add:

```python
def test_publish_manifest_creates_version_file_and_latest_pointer(tmp_path):
    rendered = render_fixture()
    published_at = datetime(2026, 7, 25, 1, 2, 3, tzinfo=timezone.utc)

    published = publish_manifest(tmp_path, rendered)
    publish_latest_manifest(tmp_path, published, "complete", published_at)

    version_path = tmp_path / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert version_path.read_bytes() == rendered.content
    assert published.path == version_path
    assert latest == {
        "corpus_version": rendered.corpus_version,
        "manifest_sha256": rendered.manifest_sha256,
        "manifest_path": f"manifests/{rendered.manifest_sha256}.jsonl",
        "overall_status": "complete",
        "published_at": "2026-07-25T01:02:03Z",
    }


def test_publish_manifest_reuses_identical_existing_version(tmp_path):
    rendered = render_fixture()
    first = publish_manifest(tmp_path, rendered)
    inode = first.path.stat().st_ino

    second = publish_manifest(tmp_path, rendered)

    assert second.path.stat().st_ino == inode


def test_publish_manifest_rejects_conflicting_existing_version(tmp_path):
    rendered = render_fixture()
    version_path = tmp_path / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    version_path.parent.mkdir(parents=True)
    version_path.write_bytes(b"conflict\n")

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, rendered)

    assert raised.value.error.code == "artifact_write_failed"
```

Monkeypatch `os.replace` and directory `fsync` helpers to assert that
`latest.json` is replaced only after the immutable manifest is durable.

- [ ] **Step 4: Implement durable immutable publication**

Implement:

```python
def publish_manifest(
    output_dir: Path,
    rendered: RenderedManifest,
) -> PublishedManifest:
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    _fsync_directory(manifests_dir.parent)
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
```

Define `ManifestPublicationError` with one sanitized `SyncError` field.
`_publish_immutable(path, content, expected_sha256)` must:

1. recompute the content hash and reject a mismatch before publication;
2. compare exact bytes when the final path already exists;
3. return without rewriting when they match;
4. raise sanitized `artifact_write_failed` when they differ;
5. otherwise write a unique sibling temporary file, flush and `fsync` it;
6. install without overwriting an existing path, handling a concurrent winner by
   comparing its exact bytes;
7. `fsync` the manifest directory after installation;
8. unlink the temporary file on every failure path.

`_atomic_replace_json` must write canonical UTF-8 JSON plus one newline, flush and
`fsync` the temporary file, `os.replace` it, and `fsync` its parent directory.
`os.replace` is the commit point: once it succeeds the new pointer content is
visible at the target path, so a failure of the subsequent parent-directory
`fsync` must not produce a mismatch between the on-disk pointer, the published
report, and the command exit code. Handle that case as follows:

- treat `os.replace` as the irreversible commit; the pointer publication is
  considered successful once `os.replace` returns;
- if the parent-directory `fsync` raises `OSError` after `os.replace` succeeded,
  record a sanitized `artifact_write_failed` error but **do not** rewrite the
  already-published report, `latest-report.json`, or `latest.json` to a failed
  status, and **do not** change the outcome's `overall_status` or exit code;
  the durability gap is a recoverable filesystem concern, not a logical
  failure of the sync;
- if the caller is `publish_latest_manifest` and the report/`latest-report.json`
  have already been published with a `complete`/`partial` status, the
  `latest.json` replacement is the final commit and a post-replace `fsync`
  failure leaves `latest.json` pointing at the new manifest — which is the
  intended end state. Return normally in that case after logging the
  durability warning through the standard sanitized error channel.

Equivalently: never let a post-`os.replace` `fsync` failure cause the command
to return exit `2` after the report and pointers already record
`complete`/`partial`. If `os.replace` itself fails (the pointer was not
advanced), that remains a fatal `artifact_write_failed` with exit `2` and the
report/pointers keep their prior state.

Test explicitly that `publish_manifest` alone leaves an existing `latest.json`
unchanged; only `publish_latest_manifest` may move the convenience pointer.
Add a test that injects an `OSError` on the parent-directory `fsync` after a
successful `os.replace` in `publish_latest_manifest` and asserts: the new
`latest.json` is in place, the previously published report and
`latest-report.json` are unchanged, and the call returns without raising. Add a
second test that injects `OSError` on `os.replace` itself and asserts the
prior `latest.json`, report, and `latest-report.json` are all unchanged and the
call raises a sanitized `artifact_write_failed`.

- [ ] **Step 5: Run manifest tests and deterministic-byte checks**

Run:

```bash
rtk uv run pytest tests/benchmark/test_corpus_manifest.py -q
rtk uv run ruff check src/benchmark/corpus_manifest.py tests/benchmark/test_corpus_manifest.py
rtk uv run black --check src/benchmark/corpus_manifest.py tests/benchmark/test_corpus_manifest.py
```

Expected: canonicalization, two-pass identity, immutable publication, pointer
status, conflict handling, and non-ASCII tests pass.

- [ ] **Step 6: Commit deterministic manifest publication**

```bash
rtk git add src/benchmark/corpus_manifest.py tests/benchmark/test_corpus_manifest.py
rtk git commit -m "feat: publish versioned corpus manifests"
```

---

### Task 8: Orchestrate the sync transaction, reports, progress, and outcomes

**Files:**
- Create: `src/benchmark/r2_corpus_sync.py`
- Create: `tests/benchmark/test_r2_corpus_sync.py`

**Interfaces:**
- Consumes: a `SyncRequest`, an optional environment mapping, dependency/store
  factories, wall/monotonic clocks, a run-ID factory, and a progress callback.
- Produces: one `SyncOutcome`; all expected domain failures are represented by
  sanitized outcome/report data.

- [ ] **Step 1: Build a reusable fake object store and write failing workflow tests**

At the top of `tests/benchmark/test_r2_corpus_sync.py`, define a `FakeStore` that:

- stores exact `ListedObject`, `HeadMetadata`, and body bytes in dictionaries;
- records list, HEAD, and GET calls;
- can raise `R2StoreError` by operation and key;
- returns fixed-size body chunks from the `open_object` context manager;
- never imports boto3.

Add an `invoke_sync` helper that passes fixed clock/run-ID factories and a
capturing progress callback into `sync_r2_corpus`.

Write these tests first:

```python
def test_complete_real_run_publishes_cache_manifest_report_and_pointers(tmp_path):
    outcome, events = invoke_sync(tmp_path, complete_store())

    assert outcome.exit_code == 0
    assert outcome.overall_status == "complete"
    assert outcome.manifest is not None
    assert outcome.manifest.corpus_version.startswith("sha256:")
    assert outcome.manifest.path.is_file()
    assert outcome.report_path.is_file()
    assert (tmp_path / "output" / "latest.json").is_file()
    assert (tmp_path / "output" / "latest-report.json").is_file()
    assert events[0].phase == "configuration"
    assert events[-1].phase == "complete"


def test_partial_run_publishes_manifest_and_returns_one(tmp_path):
    outcome, _ = invoke_sync(tmp_path, store_with_one_get_failure())

    assert outcome.exit_code == 1
    assert outcome.overall_status == "partial"
    assert outcome.manifest is not None
    assert outcome.manifest.path.is_file()
    assert read_latest(tmp_path)["overall_status"] == "partial"


def test_fatal_configuration_error_returns_two_without_network(tmp_path):
    store = complete_store()

    outcome, _ = invoke_sync(tmp_path, store, endpoint_url="http://insecure.example")

    assert outcome.exit_code == 2
    assert outcome.overall_status == "failed"
    assert outcome.manifest is None
    assert store.calls == []
    assert read_report(outcome.report_path)["errors"][0]["code"] == "invalid_config"


def test_malformed_provenance_fails_before_network(tmp_path):
    path = tmp_path / "provenance.json"
    path.write_text("{")
    store = complete_store()

    outcome, _ = invoke_sync(tmp_path, store, provenance_file=path)

    assert outcome.exit_code == 2
    assert store.calls == []


def test_dry_run_lists_and_heads_but_does_not_get_or_mutate_corpus_state(tmp_path):
    store = complete_store()
    outcome, _ = invoke_sync(tmp_path, store, dry_run=True)

    assert outcome.exit_code == 0
    assert outcome.overall_status == "dry_run_complete"
    assert outcome.manifest is None
    assert not any(call[0] == "get" for call in store.calls)
    assert not (tmp_path / "cache" / "index-v1.json").exists()
    assert not (tmp_path / "output" / "manifests").exists()
    assert not (tmp_path / "output" / "latest.json").exists()
    assert outcome.report_path.is_file()
    assert (tmp_path / "output" / "latest-report.json").is_file()
```

Use the same `store` instance in the dry-run call and GET assertion. Also cover:

- missing optional dependency and missing/invalid R2 environment configuration;
- full-page pagination and HEAD failure aggregation;
- a non-blocking writer-lock conflict before any network request;
- empty includes and ambiguity quarantine;
- report-write failure returning `2` with a sanitized one-line fallback message;
- manifest-publication failure returning `2` and still attempting a fatal report;
- report failure after immutable installation leaving `latest.json` unchanged and
  omitting the unreferenced file from `SyncOutcome.manifest`;
- identical start timestamps with different UUID4 run IDs producing different report
  paths, and fixed six-digit timestamp names sorting chronologically;
- report counts for listed objects, selected objects, included/excluded simfiles,
  empty prefixes, ambiguous prefixes, cache hits, planned/completed/failed
  downloads, bytes, and each fixed cache miss reason;
- no credential, raw endpoint, raw SDK message, signed URL, or authorization header
  in outcome text, reports, progress events, manifests, or pointers;
- two real writers using distinct cache roots and one output root that complete in
  reverse start order leave `latest.json` pointing at the last completion;
- concurrent real and dry-run writers leave `latest.json` on the real manifest while
  `latest-report.json` points at whichever attempt completes last;
- repeated fixed-input runs produce byte-identical immutable manifest files.

- [ ] **Step 2: Define progress, report, and orchestration interfaces**

Add these local records:

```python
@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    completed: int
    total: int | None
    message: str


ProgressCallback = Callable[[ProgressEvent], None]
StoreFactory = Callable[[R2Config], R2ObjectStore]
Clock = Callable[[], datetime]
MonotonicClock = Callable[[], float]
RunIdFactory = Callable[[], str]
```

Implement the public entry point with explicit dependency injection:

```python
def sync_r2_corpus(
    request: SyncRequest,
    *,
    environ: Mapping[str, str] | None = None,
    dependency_check: Callable[[], None] = ensure_r2_dependency,
    store_factory: StoreFactory = create_boto3_store,
    clock: Clock = utc_now,
    monotonic: MonotonicClock = time.monotonic,
    run_id_factory: RunIdFactory = new_run_id,
    progress: ProgressCallback = ignore_progress,
) -> SyncOutcome:
    started_at = clock()
    run_id = run_id_factory()
    return _run_sync(
        request=request,
        environ=os.environ if environ is None else environ,
        dependency_check=dependency_check,
        store_factory=store_factory,
        clock=clock,
        run_id=run_id,
        started_at=started_at,
        progress=progress,
        monotonic=monotonic,
    )
```

The entry point must catch only the documented domain/configuration exceptions,
convert them to allowlisted `SyncError` records, and attempt a fatal report.
Unexpected programming errors must not be copied into user-visible text; record
`internal_error` with no raw exception detail, then return exit `2`.

Use this boundary mapping:

| Failure source | Serialized code |
| --- | --- |
| `R2Config` or request/path validation | `invalid_config` |
| optional dependency check | `missing_optional_dependency` |
| credential, auth, bucket, root-list, HEAD, or GET adapter error | preserve the adapter's allowlisted code |
| provenance loader validation | `provenance_invalid` |
| cache-index parse/schema validation | `cache_index_invalid` |
| writer-lock conflict | `cache_locked` |
| unavailable POSIX lock/durability support | `unsupported_platform` |
| cache, manifest, report, or pointer filesystem failure | `artifact_write_failed` |
| unforeseen exception at the command boundary | `internal_error` |

Never serialize an exception's string or class name as the message.

- [ ] **Step 3: Implement the real and dry-run phase order**

Use this exact phase order:

1. validate the optional dependency;
2. build `R2Config` and validate all local paths/options;
3. load and validate provenance;
4. for a real run, acquire `cache_writer_lock` before index reading, store creation,
   or network I/O;
5. load one `CacheIndexStore` snapshot;
6. create the store and validate bucket access;
7. build the complete inventory;
8. synchronize or plan the cache;
9. for a real run, build, render, and install the immutable manifest without moving
   `latest.json`;
10. assemble and publish the report;
11. for a real run, publish `latest-report.json` only after the report is durable;
12. publish `latest.json` only after the manifest, report, and report pointer are all
   durable; treat `os.replace` as the commit point so a failure of the
   subsequent parent-directory `fsync` is a recoverable durability warning
   (sanitized `artifact_write_failed` logged without changing exit code or
   already-published status) rather than a fatal pointer failure — only an
   `os.replace` failure leaves the manifest pointer un-advanced and is fatal;
13. release the whole-run writer lock;
14. return the explicit outcome.

Use a `nullcontext()` instead of the writer lock for dry-run. Do not call
`publish_manifest` during dry-run. The dry-run report must describe planned cache
actions and set manifest corpus version, hash, and path to `None`.
If report publication fails after immutable manifest installation, leave that
content-addressed file as an unreferenced, reusable artifact, do not update
`latest.json`, and return `failed` with no `PublishedManifest` in the outcome.

Compute overall status as:

- `complete` when a real manifest exists and no row or operation has a nonfatal
  error;
- `partial` when a real manifest exists and any row is non-complete, any malformed
  or ambiguous root entry exists, or any nonfatal operation failed;
- `failed` when no real manifest was produced;
- `dry_run_complete` when dry-run planning finds no partial condition;
- `dry_run_partial` when dry-run planning finds any partial condition.

Every `empty` row is partial. Dry-run repair plans for locally missing or corrupt
bodies are not partial unless another error occurs. Map `complete` and
`dry_run_complete` to `0`, `partial` and `dry_run_partial` to `1`, and `failed` to
`2`.

- [ ] **Step 4: Implement deterministic report content and collision-proof names**

Write reports under `<output-dir>/reports/` with:

```python
def report_filename(started_at: datetime, run_id: str) -> str:
    stamp = format_report_filename_timestamp(started_at)
    return f"{stamp}-{run_id}.json"
```

`format_report_filename_timestamp` supplies exactly six UTC fractional digits.
`new_run_id` returns `str(uuid.uuid4())`; validate injected IDs by parsing them as a
UUID, requiring version 4, and requiring the lowercase canonical string before using
them as a filename component.

The report must contain:

```python
{
    "schema_version": REPORT_SCHEMA,
    "run_id": run_id,
    "started_at": format_manifest_timestamp(started_at),
    "completed_at": format_manifest_timestamp(completed_at),
    "dry_run": request.dry_run,
    "overall_status": overall_status,
    "exit_code": exit_code,
    "source_endpoint_sha256": (
        config.source_endpoint_sha256 if config is not None else None
    ),
    "source_bucket": config.bucket if config is not None else None,
    "cache_profile": CACHE_PROFILE,
    "filters": {
        "include_simfile_ids": sorted(request.include_simfile_ids),
        "exclude_simfile_ids": sorted(request.exclude_simfile_ids),
    },
    "network": {
        "head_concurrency": config.head_concurrency if config is not None else None,
        "download_concurrency": (
            config.download_concurrency if config is not None else None
        ),
        "connect_timeout_seconds": (
            config.connect_timeout_seconds if config is not None else None
        ),
        "read_timeout_seconds": (
            config.read_timeout_seconds if config is not None else None
        ),
        "retry_mode": "standard" if config is not None else None,
        "max_attempts": config.max_attempts if config is not None else None,
    },
    "artifacts": {
        "corpus_version": corpus_version,
        "manifest_sha256": manifest_sha256,
        "manifest_path": relative_manifest_path,
    },
    "counters": asdict(counters),
    "cache_misses_by_reason": {
        "remote_changed": remote_changed_count,
        "missing": missing_count,
        "size_mismatch": size_mismatch_count,
        "sha256_mismatch": sha256_mismatch_count,
    },
    "malformed_root_keys": malformed_root_keys,
    "ambiguous_prefixes": ambiguous_prefixes,
    "simfiles": report_rows,
    "errors": top_level_errors,
}
```

Canonicalize reports and pointers with the same JSON settings as manifests. Report
rows may include cache action and cache miss reason, but never credentials or raw
SDK exceptions. Sort report rows numerically, object actions by exact key, error
lists by the manifest error sort key, and counter dictionaries by key through
canonical JSON.

Use these fixed counter keys: `simfiles_discovered`, `simfiles_included`,
`simfiles_excluded_by_filter`, `simfiles_empty`, `objects_listed`,
`objects_selected`, `cache_hits`, `downloads_planned`, `downloads_completed`,
`downloads_failed`, `download_bytes_planned`, and `download_bytes_completed`.
`simfiles_discovered` counts valid, non-ambiguous remote prefixes before filters;
`simfiles_included` counts final rows after exclude precedence, including explicitly
requested absent rows derived from `include_ids - exclude_ids - discovered_ids`;
`objects_listed` counts every root-list object, including malformed and quarantined
keys; the remaining fields derive from the filtered rows and `CacheAction` records.
When fatal configuration fails before source identity exists, serialize
`source_endpoint_sha256` and `source_bucket` as `None`; never substitute the raw
invalid value.

Publish the report with temp-file write, file `fsync`, `os.replace`, and parent
directory `fsync`. Then atomically publish `latest-report.json` containing report
schema, relative report path, overall status, exit code, corpus version, report
SHA-256, and completion time.

If report publication itself fails, do not expose the raw exception. Return a
`SyncOutcome` with exit `2`, no report path, and one `artifact_write_failed` error so
the CLI can print a single sanitized fallback line.

- [ ] **Step 5: Implement bounded operator progress**

Emit progress only through the callback, never directly from domain modules.
Emit:

- one event on entry to configuration, inventory, metadata, cache, manifest, and
  report phases;
- item progress every 100 completed objects, at the final item, or when at
  least five seconds elapsed since the prior item event;
- a final `complete`, `partial`, `failed`, `dry_run_complete`, or
  `dry_run_partial` phase.

Messages may contain phase names, counts, and byte totals only. They must not contain
simfile IDs, object keys, endpoint URLs, credentials, headers, signed query
parameters, error details, or raw exception text. Test the item throttle with an
injected monotonic clock instead of sleeping.

- [ ] **Step 6: Run orchestration tests and lint**

Run:

```bash
rtk uv run --extra r2 pytest tests/benchmark/test_r2_corpus_sync.py -q
rtk uv run ruff check src/benchmark/r2_corpus_sync.py tests/benchmark/test_r2_corpus_sync.py
rtk uv run black --check src/benchmark/r2_corpus_sync.py tests/benchmark/test_r2_corpus_sync.py
```

Expected: complete, partial, failed, dry-run, locking, reporting, progress,
redaction, and repeatability tests pass without live credentials.

- [ ] **Step 7: Commit the transaction orchestrator**

```bash
rtk git add src/benchmark/r2_corpus_sync.py tests/benchmark/test_r2_corpus_sync.py
rtk git commit -m "feat: orchestrate R2 corpus synchronization"
```

---

### Task 9: Expose the Click command and document the operator contract

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md:365`

**Interfaces:**
- Consumes: Click options and R2 environment variables.
- Produces: a `SyncRequest`, sanitized stderr progress, a concise stdout summary,
  and explicit Click exit codes.

- [ ] **Step 1: Write failing command registration and option tests**

Append these imports and tests to `tests/test_cli_benchmark.py`:

```python
import pytest

from src.benchmark.r2_corpus_models import (
    OverallStatus,
    PublishedManifest,
    SyncOutcome,
)
from src.cli import benchmark as benchmark_cli

runner = CliRunner()


def test_sync_r2_corpus_help_lists_local_options_without_endpoint_flag():
    result = runner.invoke(main, ["benchmark", "sync-r2-corpus", "--help"])

    assert result.exit_code == 0
    assert "--cache-dir" in result.output
    assert "--output-dir" in result.output
    assert "--include-simfile-id" in result.output
    assert "--exclude-simfile-id" in result.output
    assert "--provenance-file" in result.output
    assert "--dry-run" in result.output
    assert "--profile" not in result.output
    assert "--endpoint" not in result.output


def make_outcome(status: OverallStatus, tmp_path: Path) -> SyncOutcome:
    exit_code = {
        "complete": 0,
        "partial": 1,
        "failed": 2,
    }[status]
    report_path = tmp_path / "report.json"
    report_path.write_text("{}\n", encoding="utf-8")
    manifest = None
    if status != "failed":
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_text("{}\n", encoding="utf-8")
        manifest = PublishedManifest(
            corpus_version=f"sha256:{'a' * 64}",
            manifest_sha256="b" * 64,
            relative_path="manifests/test.jsonl",
            path=manifest_path,
            latest_path=tmp_path / "latest.json",
        )
    return SyncOutcome(status, exit_code, report_path, manifest)


@pytest.mark.parametrize(
    ("outcome_status", "expected_exit", "exception_type"),
    [("complete", 0, None), ("partial", 1, SystemExit), ("failed", 2, SystemExit)],
)
def test_sync_r2_corpus_maps_outcome_to_explicit_exit(
    monkeypatch, tmp_path, outcome_status, expected_exit, exception_type
):
    monkeypatch.setattr(
        benchmark_cli,
        "sync_r2_corpus",
        lambda request, progress: make_outcome(outcome_status, tmp_path),
    )

    result = runner.invoke(
        main,
        [
            "benchmark",
            "sync-r2-corpus",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == expected_exit
    if exception_type is None:
        assert result.exception is None
    else:
        assert type(result.exception) is exception_type
```

Also cover:

- repeatable include/exclude filters, exclude precedence, and the inclusive
  JavaScript-safe integer bounds;
- default output `artifacts/benchmark/r2-corpus/` and cache
  `<output-dir>/cache/` resolution;
- absent and explicit provenance-file handling;
- dependency/configuration failures exit `2`, not Click's default usage exit;
- progress uses stderr while the final summary uses stdout;
- summary includes status, corpus version when present, report path when present,
  and counts;
- raw endpoint and credential sentinel strings never appear in either stream;
- report-write failure prints exactly one sanitized stderr line;
- dry-run summary states that no manifest was published.

- [ ] **Step 2: Add the command with explicit context exits**

Import `sync_r2_corpus` and `ProgressEvent` from `r2_corpus_sync` at module scope.
That module has no eager boto3/botocore imports, so this keeps the base CLI safe and
makes the orchestrator directly monkeypatchable in CLI tests. Register:

```python
@benchmark.command("sync-r2-corpus")
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Local cache root (default: <output-dir>/cache/).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/r2-corpus"),
    show_default=True,
)
@click.option(
    "--include-simfile-id",
    "include_simfile_ids",
    type=click.IntRange(0, MAX_SIMFILE_ID),
    multiple=True,
)
@click.option(
    "--exclude-simfile-id",
    "exclude_simfile_ids",
    type=click.IntRange(0, MAX_SIMFILE_ID),
    multiple=True,
)
@click.option(
    "--provenance-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option("--dry-run", is_flag=True)
@click.pass_context
def sync_r2_corpus_command(
    ctx: click.Context,
    cache_dir: Path | None,
    output_dir: Path,
    include_simfile_ids: tuple[int, ...],
    exclude_simfile_ids: tuple[int, ...],
    provenance_file: Path | None,
    dry_run: bool,
) -> None:
    request = SyncRequest(
        cache_dir=output_dir / "cache" if cache_dir is None else cache_dir,
        output_dir=output_dir,
        include_simfile_ids=frozenset(include_simfile_ids),
        exclude_simfile_ids=frozenset(exclude_simfile_ids),
        provenance_file=provenance_file,
        dry_run=dry_run,
    )
    outcome = sync_r2_corpus(request, progress=_emit_progress)
    _emit_sync_summary(outcome)
    if outcome.exit_code:
        ctx.exit(outcome.exit_code)
```

Do not add endpoint, credential, profile, concurrency, timeout, retry, or maximum
simfile flags.

`_emit_progress` uses `click.echo(message, err=True)`. `_emit_sync_summary` uses
`click.echo` for the concise summary and reads counts only from
`outcome.counters`; it does not parse the report. Only a missing-report fatal outcome
emits one sanitized stderr fallback line. Do not raise `click.ClickException`.

- [ ] **Step 3: Run CLI tests before editing documentation**

Run:

```bash
rtk uv run --extra r2 pytest tests/test_cli_benchmark.py -q
rtk uv run ruff check src/cli/benchmark.py tests/test_cli_benchmark.py
rtk uv run black --check src/cli/benchmark.py tests/test_cli_benchmark.py
```

Expected: existing benchmark commands and new `sync-r2-corpus` tests pass.

- [ ] **Step 4: Document installation, configuration, and artifacts**

Add an “R2 corpus inventory” subsection to
`docs/drumery-dtx-midi-benchmarking-reference.md`. Include:

```bash
rtk uv pip install -e '.[r2]'

export CRUX_R2_ENDPOINT_URL="https://<account-id>.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="<access-key-id>"
export AWS_SECRET_ACCESS_KEY="<secret-access-key>"
# Export AWS_SESSION_TOKEN as well when the credential provider requires it.
export CRUX_R2_BUCKET="<bucket>"

rtk uv run --extra r2 crux benchmark sync-r2-corpus \
  --provenance-file config/corpus-provenance.json \
  --dry-run
rtk uv run --extra r2 crux benchmark sync-r2-corpus \
  --provenance-file config/corpus-provenance.json
```

Document:

- R2 is the sole content authority for this stage;
- environment-only endpoint and credential configuration;
- the optional extra's boto3, botocore, s3transfer, jmespath, and urllib3
  installation footprint, which remains outside the base runtime;
- optional timeout, retry, and concurrency environment variables and their
  defaults from `R2Config`;
- exact local default paths;
- root discovery, ambiguity quarantine, marker-only `empty`, and include/exclude
  precedence;
- the fixed `setdef_dtx_txt_v1` profile;
- SHA-256 cache layout and resume-by-rerun behavior;
- the cache index as local non-authoritative state;
- dry-run mutation boundaries;
- immutable manifest, `latest.json`, timestamped report, and
  `latest-report.json` roles;
- the requirement to check `latest.json.overall_status == "complete"` and then pin
  the immutable manifest path rather than treating the pointer itself as input;
- exits `0`, `1`, and `2`;
- provenance edits intentionally re-key the corpus version;
- `version` is reserved and always null in v1;
- provider checksums are intentionally absent;
- chart selection, DTX parsing, audio selection, inference, and scoring remain
  HPA-322 responsibilities;
- a real credentialed smoke run is required before acceptance but credentials
  must never be committed or pasted into reports.

- [ ] **Step 5: Verify the rendered help and documentation text**

Run:

```bash
rtk uv run --extra r2 crux benchmark sync-r2-corpus --help
rtk rg -n "sync-r2-corpus|latest-report|overall_status|setdef_dtx_txt_v1|HPA-322" \
  docs/drumery-dtx-midi-benchmarking-reference.md
```

Expected: help exposes only the documented local options and the reference
contains every artifact/status boundary.

- [ ] **Step 6: Commit the CLI and operator documentation**

```bash
rtk git add src/cli/benchmark.py tests/test_cli_benchmark.py \
  docs/drumery-dtx-midi-benchmarking-reference.md
rtk git commit -m "feat: expose R2 corpus sync command"
```

---

### Task 10: Prove end-to-end repeatability and run the full quality gate

**Files:**
- Create: `tests/benchmark/test_r2_corpus_acceptance.py`

**Interfaces:**
- Consumes: only the public `sync_r2_corpus` entry point and an in-memory fake
  object store.
- Produces: cross-module acceptance proof and the final local verification record.

- [ ] **Step 1: Write a cross-module deterministic acceptance test**

Create a fake corpus containing:

- `1/set.def`, `1/main.DTX`, and `1/音源/readme.TXT`;
- nested zero-byte folder markers;
- `2/chart.dtx`;
- `3/` with markers only;
- ambiguous `4/` and `04/`;
- a non-selected audio object;
- paginated listing boundaries.

Run the public orchestrator three times:

1. a cold real run into empty cache/output directories;
2. a warm real run with the identical remote fixture;
3. a real run after changing one chart body and matching remote metadata.

Assert:

```python
assert first.exit_code == 1
assert second.exit_code == 1
assert first.manifest is not None
assert second.manifest is not None
assert first.manifest.corpus_version == second.manifest.corpus_version
assert first.manifest.path.read_bytes() == second.manifest.path.read_bytes()
assert second_report["counters"]["cache_hits"] == 4
assert third.manifest is not None
assert third.manifest.corpus_version != second.manifest.corpus_version
assert third.manifest.path != second.manifest.path
assert read_latest(output_dir)["corpus_version"] == third.manifest.corpus_version
```

The fixture is partial because aliases `4/` and `04/` are quarantined. Also assert
that simfile `3` is `empty`, exact non-ASCII keys survive, bodies are extensionless
and content-addressed, the index is canonical JSON, provenance and error changes
alter identity, and neither reports nor manifests contain the secret sentinel or
raw endpoint. After the third run, assert the second manifest and the prior changed
object's cache body still exist with their original bytes.

- [ ] **Step 2: Add a base-install import boundary test**

Run this subprocess test without requesting the `r2` extra:

```python
def test_base_cli_import_does_not_import_boto3():
    code = (
        "import sys; import src.cli.main; "
        "assert 'boto3' not in sys.modules; "
        "assert 'botocore' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
```

This verifies lazy imports even when the developer environment happens to have
boto3 installed.

- [ ] **Step 3: Run the focused HPA-321 suite**

Run:

```bash
rtk uv run --extra r2 pytest \
  tests/benchmark/test_r2_corpus_models.py \
  tests/benchmark/test_r2_inventory.py \
  tests/benchmark/test_corpus_provenance.py \
  tests/benchmark/test_corpus_cache.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_r2_corpus_sync.py \
  tests/benchmark/test_r2_corpus_acceptance.py \
  tests/test_cli_benchmark.py -q
```

Expected: the complete local HPA-321 suite passes without network access or live
credentials.

- [ ] **Step 4: Run the repository quality gate**

Run:

```bash
rtk uv run --extra r2 pytest
rtk uv run ruff check src tests
rtk uv run black --check src tests
rtk uv run --extra r2 pylint src/app src/cli src/benchmark
```

Fix only HPA-321 regressions. If an unrelated pre-existing failure appears, record
the exact command and failure in the handoff without broadening this implementation.

- [ ] **Step 5: Perform the credentialed acceptance smoke check**

With operator-provided R2 environment variables present only in the shell, set
`CRUX_R2_SMOKE_ID` to a numeric simfile ID already known to contain nested and
non-ASCII keys:

```bash
rtk uv run --extra r2 crux benchmark sync-r2-corpus \
  --include-simfile-id "$CRUX_R2_SMOKE_ID" \
  --provenance-file config/corpus-provenance.json \
  --dry-run
rtk uv run --extra r2 crux benchmark sync-r2-corpus \
  --include-simfile-id "$CRUX_R2_SMOKE_ID" \
  --provenance-file config/corpus-provenance.json
rtk uv run --extra r2 crux benchmark sync-r2-corpus \
  --include-simfile-id "$CRUX_R2_SMOKE_ID" \
  --provenance-file config/corpus-provenance.json
rtk uv run --extra r2 crux benchmark sync-r2-corpus \
  --provenance-file config/corpus-provenance.json \
  --dry-run
```

Verify:

- the small dry-run performs listing/HEAD but no GET or corpus-state mutation;
- the first real run publishes cache, manifest, report, and both pointers;
- every selected cached body recomputes to its manifest SHA-256 and byte count;
- the second real run reports cache hits and reuses the exact immutable manifest;
- exit code and `latest.json.overall_status` agree;
- no credential or raw endpoint appears in terminal output or published JSON;
- the final command completes a full-corpus dry-run before the first full production
  synchronization;
- the observed real R2 ETag forms are recorded in the acceptance notes, and a real
  weak-ETag object exercises the metadata-verified path when available.

Do not fabricate this result when credentials are unavailable. Report the local
fake-store proof separately from the pending live smoke requirement.

- [ ] **Step 6: Commit the acceptance coverage**

```bash
rtk git add tests/benchmark/test_r2_corpus_acceptance.py
rtk git commit -m "test: cover R2 corpus sync acceptance"
```
