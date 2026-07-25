# HPA-321 R2 Corpus Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a secure, resumable R2 inventory command that selectively caches chart-definition files and publishes deterministic, immutable benchmark manifests.

**Architecture:** A `boto3` adapter exposes a narrow object-store protocol. Separate inventory, provenance, cache, and manifest modules operate on typed domain records, while one orchestration module owns transactional flow and machine-readable reports. A Click command provides the operator interface without accepting credential values.

**Tech Stack:** Python 3.12, Click, boto3/botocore, dataclasses, SHA-256, JSONL, pytest

## Global Constraints

- R2 object contents are authoritative; do not use D1 or GraphQL to determine corpus truth.
- Enumerate the complete bucket root with `ListObjectsV2` and no delimiter so nested keys remain visible.
- Preserve exact Unicode object keys; do not URL-decode, case-fold, or normalize stored keys.
- The initial cache profile selects case-insensitive `set.def`, `.dtx`, and `.txt` files only.
- Treat ETags as remote change signals, never as universal content hashes.
- Verify every reused or downloaded cache body with locally computed SHA-256 and byte count.
- Store cache bodies at `sha256/<first-two-hex>/<full-sha256>` relative to the cache root.
- Never store or log credentials, signed URLs, request headers, raw endpoint URLs, or raw SDK exceptions.
- Real manifests are canonical UTF-8 JSONL and immutable at `manifests/<manifest-sha256>.jsonl`.
- `latest.json` is a convenience pointer only; benchmark consumers must use the concrete hash path.
- Dry-run may write reports but must not call `GetObject` or mutate cache, cache index, manifests, or `latest.json`.
- Partial object or simfile failures remain visible, publish a partial manifest, and return a nonzero result.
- Published reports and manifests contain no source audio bodies.
- Keep Python lines at or below the repository's 100-character formatting limit.

---

## File Map

### New source files

- `src/benchmark/r2_corpus_models.py` — immutable domain records, endpoint identity, ETag and timestamp normalization.
- `src/benchmark/r2_store.py` — narrow R2 protocol, safe SDK errors, and lazy `boto3` adapter.
- `src/benchmark/r2_inventory.py` — root-key classification, include/exclude filtering, bounded HEAD enrichment.
- `src/benchmark/corpus_provenance.py` — strict optional provenance mapping loader.
- `src/benchmark/corpus_cache.py` — cache selection, index, SHA-256 verification, conditional download, atomic install.
- `src/benchmark/corpus_manifest.py` — canonical JSONL, corpus identity, immutable publication, latest pointer.
- `src/benchmark/r2_corpus_sync.py` — end-to-end orchestration, reports, partial/fatal outcomes.
- `src/cli/r2_corpus.py` — `sync-r2-corpus` Click command.

### New configuration and tests

- `config/corpus-provenance.json` — version-controlled empty v1 provenance mapping.
- `tests/benchmark/test_r2_corpus_models.py`
- `tests/benchmark/test_r2_store.py`
- `tests/benchmark/test_r2_inventory.py`
- `tests/benchmark/test_corpus_provenance.py`
- `tests/benchmark/test_corpus_cache.py`
- `tests/benchmark/test_corpus_manifest.py`
- `tests/benchmark/test_r2_corpus_sync.py`

### Modified files

- `pyproject.toml` — add the `boto3` runtime dependency.
- `uv.lock` — lock the resolved boto3/botocore dependency graph.
- `src/cli/benchmark.py` — register the new command.
- `tests/test_cli_benchmark.py` — CLI configuration, filtering, dry-run, output, and failure tests.
- `docs/drumery-dtx-midi-benchmarking-reference.md` — document R2 inventory workflow and artifact contract.

---

### Task 1: Add the dependency, configuration model, and canonical primitives

**Files:**
- Create: `src/benchmark/r2_corpus_models.py`
- Create: `tests/benchmark/test_r2_corpus_models.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `R2Config.from_values(endpoint_url: str, bucket: str) -> R2Config`
- Produces: `normalize_etag(value: str | None) -> str | None`
- Produces: `format_r2_datetime(value: datetime) -> str`
- Produces: `SyncError`, `ListedObject`, `HeadMetadata`, `RemoteObject`, and `SimfileInventory`
- Consumes: no HPA-321 source interfaces

- [ ] **Step 1: Add failing canonicalization tests**

Create `tests/benchmark/test_r2_corpus_models.py`:

```python
from datetime import datetime, timezone
from hashlib import sha256

import pytest

from src.benchmark.r2_corpus_models import R2Config, format_r2_datetime, normalize_etag


def test_r2_config_normalizes_endpoint_and_hashes_source_identity():
    config = R2Config.from_values(
        "HTTPS://ABC123.r2.cloudflarestorage.com/",
        "simfile-dtx",
    )

    normalized = "https://abc123.r2.cloudflarestorage.com"
    assert config.endpoint_url == normalized
    assert config.source_endpoint_sha256 == sha256(normalized.encode("ascii")).hexdigest()
    assert config.bucket == "simfile-dtx"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://abc123.r2.cloudflarestorage.com",
        "https://user@example.com",
        "https://example.com/path",
        "https://example.com?token=secret",
        "https://example.com#fragment",
    ],
)
def test_r2_config_rejects_non_origin_or_non_https_endpoint(endpoint):
    with pytest.raises(ValueError, match="HTTPS origin"):
        R2Config.from_values(endpoint, "simfile-dtx")


def test_etag_and_datetime_normalization():
    assert normalize_etag('"abc-2"') == "abc-2"
    assert normalize_etag(None) is None
    assert format_r2_datetime(
        datetime(2026, 7, 24, 12, 34, 56, 120000, tzinfo=timezone.utc)
    ) == "2026-07-24T12:34:56.12Z"
```

- [ ] **Step 2: Run the new tests and verify the missing module failure**

Run:

```bash
rtk uv run pytest tests/benchmark/test_r2_corpus_models.py -q
```

Expected: collection fails with `ModuleNotFoundError: src.benchmark.r2_corpus_models`.

- [ ] **Step 3: Add boto3 through uv**

Run:

```bash
rtk uv add "boto3>=1.35.37"
```

Expected: `pyproject.toml` adds boto3 and `uv.lock` records boto3, botocore, jmespath,
s3transfer, and their resolved transitive dependencies.

- [ ] **Step 4: Implement the typed model module**

Create `src/benchmark/r2_corpus_models.py` with these public definitions:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Literal
from urllib.parse import urlsplit

ErrorScope = Literal["root", "simfile", "object", "cache", "configuration"]
SimfileStatus = Literal["complete", "partial", "failed", "empty"]


@dataclass(frozen=True)
class R2Config:
    endpoint_url: str
    bucket: str
    source_endpoint_sha256: str
    region_name: str = "auto"

    @classmethod
    def from_values(cls, endpoint_url: str, bucket: str) -> "R2Config":
        parts = urlsplit(endpoint_url)
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
            raise ValueError("R2 endpoint must be an HTTPS origin without credentials or a path")
        if not bucket or "/" in bucket:
            raise ValueError("R2 bucket must be a non-empty bucket name")
        hostname = parts.hostname.lower()
        port = f":{parts.port}" if parts.port not in (None, 443) else ""
        normalized = f"https://{hostname}{port}"
        source_hash = sha256(normalized.encode("ascii")).hexdigest()
        return cls(normalized, bucket, source_hash)


def normalize_etag(value: str | None) -> str | None:
    if value is None:
        return None
    return value[1:-1] if len(value) >= 2 and value.startswith('"') and value.endswith('"') else value


def format_r2_datetime(value: datetime) -> str:
    utc_value = value.astimezone(timezone.utc)
    text = utc_value.strftime("%Y-%m-%dT%H:%M:%S.%f").rstrip("0").rstrip(".")
    return f"{text}Z"


@dataclass(frozen=True)
class SyncError:
    scope: ErrorScope
    code: str
    message: str
    object_key: str | None = None


@dataclass(frozen=True)
class ListedObject:
    key: str
    size: int
    etag: str | None
    last_modified: datetime


@dataclass(frozen=True)
class HeadMetadata:
    content_type: str | None
    version: str | None
    remote_checksums: dict[str, str]


@dataclass(frozen=True)
class RemoteObject:
    key: str
    size: int
    etag: str | None
    last_modified: datetime
    content_type: str | None = None
    version: str | None = None
    remote_checksums: dict[str, str] = field(default_factory=dict)
    errors: tuple[SyncError, ...] = ()


@dataclass(frozen=True)
class SimfileInventory:
    simfile_id: int
    object_prefix: str
    objects: tuple[RemoteObject, ...]
    status: SimfileStatus
    errors: tuple[SyncError, ...] = ()
```

- [ ] **Step 5: Run model tests and formatting**

Run:

```bash
rtk uv run pytest tests/benchmark/test_r2_corpus_models.py -q
rtk uv run black src/benchmark/r2_corpus_models.py tests/benchmark/test_r2_corpus_models.py
rtk uv run ruff check src/benchmark/r2_corpus_models.py tests/benchmark/test_r2_corpus_models.py
```

Expected: all model tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit the foundation**

Run:

```bash
rtk git add pyproject.toml uv.lock src/benchmark/r2_corpus_models.py \
  tests/benchmark/test_r2_corpus_models.py
rtk git commit -m "feat: add R2 corpus configuration models"
```

---

### Task 2: Implement the safe boto3 object-store adapter

**Files:**
- Create: `src/benchmark/r2_store.py`
- Create: `tests/benchmark/test_r2_store.py`

**Interfaces:**
- Consumes: `R2Config`, `ListedObject`, `HeadMetadata`, and `normalize_etag`
- Produces: `R2ObjectStore` protocol
- Produces: `Boto3R2Store.from_config(config: R2Config) -> Boto3R2Store`
- Produces: `validate_bucket`, `list_objects`, `head_object`, and `get_object`
- Produces: `R2OperationError` with allowlisted `code`, `operation`, `status`, and `object_key`

- [ ] **Step 1: Write failing adapter tests**

Create `tests/benchmark/test_r2_store.py` with fake paginator/client objects and these
assertions:

```python
from datetime import datetime, timezone
from io import BytesIO

import pytest

from src.benchmark.r2_corpus_models import R2Config
from src.benchmark.r2_store import Boto3R2Store, R2OperationError


class FakePaginator:
    def paginate(self, **kwargs):
        assert kwargs == {"Bucket": "simfile-dtx"}
        return [
            {
                "Contents": [
                    {
                        "Key": "42/曲/SET.DEF",
                        "Size": 12,
                        "ETag": '"etag-2"',
                        "LastModified": datetime(2026, 7, 24, tzinfo=timezone.utc),
                    }
                ]
            },
            {"Contents": []},
        ]


class FakeBody(BytesIO):
    closed_by_adapter = False

    def close(self):
        self.closed_by_adapter = True
        super().close()


class FakeClient:
    def __init__(self):
        self.get_calls = []
        self.body = FakeBody(b"chart")

    def head_bucket(self, **kwargs):
        assert kwargs == {"Bucket": "simfile-dtx"}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator()

    def head_object(self, **kwargs):
        assert kwargs == {
            "Bucket": "simfile-dtx",
            "Key": "42/曲/SET.DEF",
            "ChecksumMode": "ENABLED",
        }
        return {
            "ContentType": "text/plain",
            "VersionId": "version-1",
            "ChecksumSHA256": "base64-value",
        }

    def get_object(self, **kwargs):
        self.get_calls.append(kwargs)
        return {"Body": self.body, "ContentLength": 5}


def test_store_paginates_heads_and_conditionally_gets_exact_key():
    client = FakeClient()
    store = Boto3R2Store(client)

    listed = store.list_objects("simfile-dtx")
    metadata = store.head_object("simfile-dtx", listed[0].key)
    response = store.get_object("simfile-dtx", listed[0].key, "etag-2")

    assert listed[0].key == "42/曲/SET.DEF"
    assert listed[0].etag == "etag-2"
    assert metadata.remote_checksums == {"sha256": "base64-value"}
    assert response.content_length == 5
    assert client.get_calls == [
        {
            "Bucket": "simfile-dtx",
            "Key": "42/曲/SET.DEF",
            "IfMatch": '"etag-2"',
        }
    ]
    response.close()
    assert client.body.closed_by_adapter is True


def test_operation_error_does_not_include_raw_sdk_message():
    error = R2OperationError("access_denied", "list", 403)
    assert str(error) == "R2 list failed with access_denied (HTTP 403)"
    assert "credential" not in str(error).lower()
```

- [ ] **Step 2: Run the adapter tests and verify failure**

Run:

```bash
rtk uv run pytest tests/benchmark/test_r2_store.py -q
```

Expected: collection fails because `src.benchmark.r2_store` does not exist.

- [ ] **Step 3: Implement the protocol and download response**

Create `src/benchmark/r2_store.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Protocol

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.benchmark.r2_corpus_models import HeadMetadata, ListedObject, R2Config, normalize_etag

CHECKSUM_FIELDS = {
    "ChecksumCRC32": "crc32",
    "ChecksumCRC32C": "crc32c",
    "ChecksumSHA1": "sha1",
    "ChecksumSHA256": "sha256",
    "ChecksumCRC64NVME": "crc64nvme",
}


class R2OperationError(RuntimeError):
    def __init__(
        self,
        code: str,
        operation: str,
        status: int | None = None,
        object_key: str | None = None,
    ) -> None:
        self.code = code
        self.operation = operation
        self.status = status
        self.object_key = object_key
        suffix = f" (HTTP {status})" if status is not None else ""
        super().__init__(f"R2 {operation} failed with {code}{suffix}")


@dataclass
class ObjectDownload:
    body: BinaryIO
    content_length: int | None

    def close(self) -> None:
        self.body.close()


class R2ObjectStore(Protocol):
    def validate_bucket(self, bucket: str) -> None: ...
    def list_objects(self, bucket: str) -> list[ListedObject]: ...
    def head_object(self, bucket: str, key: str) -> HeadMetadata: ...
    def get_object(self, bucket: str, key: str, etag: str | None) -> ObjectDownload: ...
```

Implement `Boto3R2Store` so:

- `from_config` calls `boto3.client("s3", endpoint_url=config.endpoint_url,
  region_name=config.region_name)` without explicit credential values;
- `validate_bucket` uses `head_bucket(Bucket=bucket)`;
- `list_objects` uses `client.get_paginator("list_objects_v2").paginate(Bucket=bucket)`;
- missing `Contents` is treated as an empty list;
- `head_object` requests `ChecksumMode="ENABLED"` and allowlists the five checksum fields;
- when checksum mode alone returns SDK code `NotImplemented`, `InvalidArgument`, or
  `InvalidRequest`, retry that HEAD once without checksum mode so content type remains
  available; do not retry authentication, authorization, or other failures;
- `get_object` adds `IfMatch=f'"{etag}"'` only when ETag is present;
- every SDK exception is converted by a private `_operation_error` that reads only
  `Error.Code` and `ResponseMetadata.HTTPStatusCode`;
- raw `str(exc)`, request IDs, response headers, endpoint URLs, and request dictionaries
  never enter the raised message.

Normalize SDK codes through an allowlist:

```python
SAFE_ERROR_CODES = {
    "AccessDenied": "access_denied",
    "InvalidAccessKeyId": "invalid_access_key",
    "SignatureDoesNotMatch": "signature_mismatch",
    "NoSuchBucket": "bucket_not_found",
    "NoSuchKey": "object_not_found",
    "PreconditionFailed": "precondition_failed",
    "SlowDown": "rate_limited",
    "NotImplemented": "checksum_mode_unsupported",
    "InvalidArgument": "invalid_request",
    "InvalidRequest": "invalid_request",
}


def _operation_error(error: Exception, operation: str, key: str | None = None):
    if isinstance(error, ClientError):
        response = error.response if isinstance(error.response, dict) else {}
        error_data = response.get("Error", {})
        metadata = response.get("ResponseMetadata", {})
        raw_code = error_data.get("Code") if isinstance(error_data, dict) else None
        status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
        safe_code = SAFE_ERROR_CODES.get(raw_code, "r2_request_failed")
        return R2OperationError(safe_code, operation, status, key)
    if isinstance(error, BotoCoreError):
        return R2OperationError("r2_client_error", operation, object_key=key)
    return R2OperationError("r2_request_failed", operation, object_key=key)
```

Call this helper with `raise _operation_error(error, operation, key) from None` so
tracebacks presented by Click cannot reveal the SDK exception chain.

Add a test whose first `head_object` call raises `NotImplemented`, whose second call
without `ChecksumMode` returns `ContentType`, and assert exactly two calls occur with
no checksum values. Add a separate `AccessDenied` HEAD test and assert it is not
retried.

- [ ] **Step 4: Add SDK-error redaction cases**

Extend `tests/benchmark/test_r2_store.py` with a `ClientError` whose raw message contains
an endpoint and signature:

```python
from botocore.exceptions import ClientError


def test_client_error_is_reduced_to_allowlisted_fields():
    client = FakeClient()

    def fail(**_kwargs):
        raise ClientError(
            {
                "Error": {
                    "Code": "AccessDenied",
                    "Message": "signed request https://account.r2.example?X-Amz-Signature=secret",
                },
                "ResponseMetadata": {"HTTPStatusCode": 403, "RequestId": "private-request-id"},
            },
            "HeadBucket",
        )

    client.head_bucket = fail
    store = Boto3R2Store(client)

    with pytest.raises(R2OperationError) as raised:
        store.validate_bucket("simfile-dtx")

    assert raised.value.code == "access_denied"
    assert "http" not in str(raised.value).lower()
    assert "signature" not in str(raised.value).lower()
    assert "request-id" not in str(raised.value).lower()
```

- [ ] **Step 5: Run adapter tests and checks**

Run:

```bash
rtk uv run pytest tests/benchmark/test_r2_store.py -q
rtk uv run black src/benchmark/r2_store.py tests/benchmark/test_r2_store.py
rtk uv run ruff check src/benchmark/r2_store.py tests/benchmark/test_r2_store.py
```

Expected: all adapter tests pass and formatting checks succeed.

- [ ] **Step 6: Commit the adapter**

Run:

```bash
rtk git add src/benchmark/r2_store.py tests/benchmark/test_r2_store.py
rtk git commit -m "feat: add safe R2 object store adapter"
```

---

### Task 3: Build complete inventory and prefix classification

**Files:**
- Create: `src/benchmark/r2_inventory.py`
- Create: `tests/benchmark/test_r2_inventory.py`

**Interfaces:**
- Consumes: `R2ObjectStore`, `R2Config`, `ListedObject`, `RemoteObject`, `SimfileInventory`
- Produces: `InventorySnapshot`
- Produces: `inventory_r2_corpus(store, config, include_ids, exclude_ids, head_workers=8)`
- Guarantees: deterministic simfile/object order and complete root discovery

- [ ] **Step 1: Write failing root-classification tests**

Create `tests/benchmark/test_r2_inventory.py` with a fake store returning:

```python
listed = [
    ListedObject("42/", 0, "folder", NOW),
    ListedObject("42/SET.DEF", 10, "set", NOW),
    ListedObject("42/assets/曲 snare.ogg", 20, "audio", NOW),
    ListedObject("7/mas.dtx", 30, "chart", NOW),
    ListedObject("01/chart.dtx", 40, "alias-a", NOW),
    ListedObject("1/chart.dtx", 50, "alias-b", NOW),
    ListedObject("derived/report.json", 60, "derived", NOW),
    ListedObject("root-object", 70, "root", NOW),
]
```

Add assertions that:

- simfiles are ordered `7`, then `42`;
- `42/assets/曲 snare.ogg` remains exact and nested;
- the `42/` marker is retained in inventory;
- `1/` and `01/` produce one `ambiguous_simfile_prefix` root error and no simfile row;
- `derived/report.json` and `root-object` produce `malformed_root_key` errors;
- include `{42, 99}` yields rows `42` and empty `99/`;
- exclude `{42}` removes 42 even when included;
- HEAD failure on one object preserves listing metadata and marks simfile 42 `partial`.

Use this fake protocol:

```python
class FakeStore:
    def __init__(self, listed, head_errors=None):
        self.listed = listed
        self.head_errors = head_errors or {}

    def validate_bucket(self, bucket):
        assert bucket == "simfile-dtx"

    def list_objects(self, bucket):
        assert bucket == "simfile-dtx"
        return self.listed

    def head_object(self, bucket, key):
        if key in self.head_errors:
            raise self.head_errors[key]
        return HeadMetadata("application/octet-stream", None, {})
```

- [ ] **Step 2: Run inventory tests and verify failure**

Run:

```bash
rtk uv run pytest tests/benchmark/test_r2_inventory.py -q
```

Expected: collection fails because `src.benchmark.r2_inventory` is missing.

- [ ] **Step 3: Implement the inventory result and root classifier**

Create `src/benchmark/r2_inventory.py` with:

```python
@dataclass(frozen=True)
class InventorySnapshot:
    simfiles: tuple[SimfileInventory, ...]
    root_errors: tuple[SyncError, ...]
    source_discovery_method: str = "r2_list_objects_v2"

    @property
    def is_partial(self) -> bool:
        return bool(self.root_errors) or any(item.status != "complete" for item in self.simfiles)


def inventory_r2_corpus(
    store: R2ObjectStore,
    config: R2Config,
    include_ids: frozenset[int] = frozenset(),
    exclude_ids: frozenset[int] = frozenset(),
    head_workers: int = 8,
) -> InventorySnapshot:
    store.validate_bucket(config.bucket)
    listed_objects = store.list_objects(config.bucket)
    grouped, root_errors = _classify_root_objects(listed_objects)
    selected = _filter_and_add_empty_prefixes(grouped, include_ids, exclude_ids)
    simfiles = _head_enrich_simfiles(
        store,
        config.bucket,
        selected,
        max_workers=head_workers,
    )
    return InventorySnapshot(
        simfiles=tuple(sorted(simfiles, key=lambda item: (item.simfile_id, item.object_prefix))),
        root_errors=tuple(root_errors),
    )
```

Implementation rules:

- call `validate_bucket` before listing;
- list the root once with no prefix or delimiter;
- classify the first path segment only when a slash exists and the segment is digits;
- map digit segments to `int` and quarantine all prefixes involved in duplicate numeric IDs;
- apply include and exclude after classification, with exclude winning;
- synthesize `<id>/` only for explicitly included missing IDs;
- sort simfiles by `(simfile_id, object_prefix)` and objects by exact key;
- enrich selected objects with `ThreadPoolExecutor(max_workers=head_workers)`;
- convert `R2OperationError` to `SyncError(scope="object", code=error.code,
  message=safe_message, object_key=key)`;
- do not use raw exception messages.

Define the private helpers used above with these exact signatures:

```python
def _classify_root_objects(
    objects: list[ListedObject],
) -> tuple[dict[int, tuple[str, list[ListedObject]]], list[SyncError]]:
    candidates: dict[int, dict[str, list[ListedObject]]] = {}
    errors: list[SyncError] = []
    for item in objects:
        segment, separator, _remainder = item.key.partition("/")
        if not separator or not segment.isdigit():
            errors.append(
                SyncError("root", "malformed_root_key", "Object has no numeric prefix", item.key)
            )
            continue
        candidates.setdefault(int(segment), {}).setdefault(f"{segment}/", []).append(item)

    grouped: dict[int, tuple[str, list[ListedObject]]] = {}
    for simfile_id, prefixes in candidates.items():
        if len(prefixes) != 1:
            errors.append(
                SyncError(
                    "root",
                    "ambiguous_simfile_prefix",
                    f"Numeric simfile ID maps to prefixes {sorted(prefixes)}",
                )
            )
            continue
        prefix, prefix_objects = next(iter(prefixes.items()))
        grouped[simfile_id] = (prefix, sorted(prefix_objects, key=lambda item: item.key))
    return grouped, errors


def _filter_and_add_empty_prefixes(
    grouped: dict[int, tuple[str, list[ListedObject]]],
    include_ids: frozenset[int],
    exclude_ids: frozenset[int],
) -> dict[int, tuple[str, list[ListedObject]]]:
    selected_ids = set(include_ids) if include_ids else set(grouped)
    selected_ids.difference_update(exclude_ids)
    return {
        simfile_id: grouped.get(simfile_id, (f"{simfile_id}/", []))
        for simfile_id in sorted(selected_ids)
    }


def _head_enrich_simfiles(
    store: R2ObjectStore,
    bucket: str,
    grouped: dict[int, tuple[str, list[ListedObject]]],
    max_workers: int,
) -> list[SimfileInventory]:
    enriched_by_key: dict[str, RemoteObject] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(store.head_object, bucket, item.key): item
            for _prefix, items in grouped.values()
            for item in items
        }
        for future, item in futures.items():
            try:
                head = future.result()
                enriched_by_key[item.key] = RemoteObject(
                    item.key,
                    item.size,
                    item.etag,
                    item.last_modified,
                    head.content_type,
                    head.version,
                    head.remote_checksums,
                )
            except R2OperationError as error:
                sync_error = SyncError(
                    "object",
                    error.code,
                    str(error),
                    item.key,
                )
                enriched_by_key[item.key] = RemoteObject(
                    item.key,
                    item.size,
                    item.etag,
                    item.last_modified,
                    errors=(sync_error,),
                )

    simfiles: list[SimfileInventory] = []
    for simfile_id, (prefix, listed) in grouped.items():
        objects = tuple(enriched_by_key[item.key] for item in listed)
        errors = tuple(error for item in objects for error in item.errors)
        only_marker = not objects or all(
            item.key == prefix and item.size == 0 for item in objects
        )
        status: SimfileStatus = "empty" if only_marker else "partial" if errors else "complete"
        simfiles.append(SimfileInventory(simfile_id, prefix, objects, status, errors))
    return simfiles
```

The helpers are private implementation units, not future extension points. A prefix
with no object beyond its exact zero-byte folder marker has status `empty`; a prefix
with objects and no errors has status `complete`.

- [ ] **Step 4: Add a deterministic >1,000-object fixture**

Add a test that gives the inventory service 1,005 objects already returned by the
adapter, with nested Unicode keys, and asserts all objects survive sorted under the
single simfile. Pagination itself remains covered by Task 2; this test proves the
inventory does not truncate adapter output.

- [ ] **Step 5: Run inventory tests and checks**

Run:

```bash
rtk uv run pytest tests/benchmark/test_r2_inventory.py -q
rtk uv run black src/benchmark/r2_inventory.py tests/benchmark/test_r2_inventory.py
rtk uv run ruff check src/benchmark/r2_inventory.py tests/benchmark/test_r2_inventory.py
```

Expected: all inventory tests pass.

- [ ] **Step 6: Commit inventory behavior**

Run:

```bash
rtk git add src/benchmark/r2_inventory.py tests/benchmark/test_r2_inventory.py
rtk git commit -m "feat: inventory R2 simfile prefixes"
```

---

### Task 4: Add strict provenance and rights metadata

**Files:**
- Create: `src/benchmark/corpus_provenance.py`
- Create: `tests/benchmark/test_corpus_provenance.py`
- Create: `config/corpus-provenance.json`

**Interfaces:**
- Produces: `ProvenanceRecord`
- Produces: `load_provenance(path: Path | None) -> dict[int, ProvenanceRecord]`
- Produces: `unknown_provenance() -> ProvenanceRecord`
- Consumes: no network or R2 interfaces

- [ ] **Step 1: Write failing provenance tests**

Create `tests/benchmark/test_corpus_provenance.py`:

```python
import json

import pytest

from src.benchmark.corpus_provenance import load_provenance, unknown_provenance


def test_missing_mapping_entry_uses_explicit_unknown_values():
    assert unknown_provenance().to_dict() == {
        "source_origin": None,
        "source_author_or_pack": None,
        "source_reference": None,
        "rights_status": "unknown",
        "redistribution_allowed": None,
        "provenance_notes": None,
    }


def test_loads_versioned_mapping(tmp_path):
    path = tmp_path / "provenance.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "crux.corpus-provenance/v1",
                "simfiles": {
                    "42": {
                        "source_origin": "personal",
                        "source_author_or_pack": "Example Pack",
                        "source_reference": "private archive",
                        "rights_status": "privately_authorized",
                        "redistribution_allowed": False,
                        "provenance_notes": "Local benchmark use.",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    records = load_provenance(path)

    assert records[42].rights_status == "privately_authorized"
    assert records[42].redistribution_allowed is False


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": "wrong", "simfiles": {}},
        {"schema_version": "crux.corpus-provenance/v1", "simfiles": {"01": {}, "1": {}}},
        {
            "schema_version": "crux.corpus-provenance/v1",
            "simfiles": {"42": {"redistribution_allowed": "no"}},
        },
    ],
)
def test_rejects_invalid_schema_aliases_and_field_types(tmp_path, payload):
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_provenance(path)
```

- [ ] **Step 2: Run provenance tests and verify failure**

Run:

```bash
rtk uv run pytest tests/benchmark/test_corpus_provenance.py -q
```

Expected: collection fails because `src.benchmark.corpus_provenance` is missing.

- [ ] **Step 3: Implement strict parsing**

Create `src/benchmark/corpus_provenance.py` with a frozen `ProvenanceRecord` containing
the six approved fields and `to_dict()`. `load_provenance` must:

- return `{}` for `None`;
- require schema `crux.corpus-provenance/v1`;
- require a JSON object named `simfiles`;
- require canonical decimal keys (`str(int(key)) == key`, non-negative);
- reject numeric aliases and unknown provenance field names;
- accept nullable string fields;
- require a non-empty string for an explicitly provided `rights_status`;
- accept only `bool` or `null` for `redistribution_allowed`;
- raise `ValueError` with a safe local validation message.

Use this record and parsing structure:

```python
PROVENANCE_FIELDS = {
    "source_origin",
    "source_author_or_pack",
    "source_reference",
    "rights_status",
    "redistribution_allowed",
    "provenance_notes",
}


@dataclass(frozen=True)
class ProvenanceRecord:
    source_origin: str | None = None
    source_author_or_pack: str | None = None
    source_reference: str | None = None
    rights_status: str = "unknown"
    redistribution_allowed: bool | None = None
    provenance_notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def unknown_provenance() -> ProvenanceRecord:
    return ProvenanceRecord()


def load_provenance(path: Path | None) -> dict[int, ProvenanceRecord]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Corpus provenance root must be an object")
    if payload.get("schema_version") != "crux.corpus-provenance/v1":
        raise ValueError("Unsupported corpus provenance schema")
    simfiles = payload.get("simfiles")
    if not isinstance(simfiles, dict):
        raise ValueError("Corpus provenance simfiles must be an object")

    records: dict[int, ProvenanceRecord] = {}
    for raw_id, raw_record in simfiles.items():
        if not isinstance(raw_id, str) or not raw_id.isdigit() or str(int(raw_id)) != raw_id:
            raise ValueError("Corpus provenance IDs must be canonical non-negative decimals")
        simfile_id = int(raw_id)
        if simfile_id in records:
            raise ValueError("Corpus provenance contains a duplicate simfile ID")
        if not isinstance(raw_record, dict) or set(raw_record) - PROVENANCE_FIELDS:
            raise ValueError(f"Corpus provenance record {raw_id} has unsupported fields")
        records[simfile_id] = _parse_record(raw_id, raw_record)
    return records


def _parse_record(raw_id: str, values: dict[str, object]) -> ProvenanceRecord:
    string_fields = (
        "source_origin",
        "source_author_or_pack",
        "source_reference",
        "provenance_notes",
    )
    for field_name in string_fields:
        value = values.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Corpus provenance {raw_id}.{field_name} must be text or null")

    rights_status = values.get("rights_status", "unknown")
    if not isinstance(rights_status, str) or not rights_status:
        raise ValueError(f"Corpus provenance {raw_id}.rights_status must be non-empty text")
    redistribution = values.get("redistribution_allowed")
    if redistribution is not None and not isinstance(redistribution, bool):
        raise ValueError(
            f"Corpus provenance {raw_id}.redistribution_allowed must be boolean or null"
        )
    return ProvenanceRecord(
        source_origin=values.get("source_origin"),
        source_author_or_pack=values.get("source_author_or_pack"),
        source_reference=values.get("source_reference"),
        rights_status=rights_status,
        redistribution_allowed=redistribution,
        provenance_notes=values.get("provenance_notes"),
    )
```

Do not coerce strings, integers, or truthy values.

- [ ] **Step 4: Add the version-controlled empty mapping**

Create `config/corpus-provenance.json`:

```json
{
  "schema_version": "crux.corpus-provenance/v1",
  "simfiles": {}
}
```

- [ ] **Step 5: Run provenance tests and checks**

Run:

```bash
rtk uv run pytest tests/benchmark/test_corpus_provenance.py -q
rtk uv run black src/benchmark/corpus_provenance.py \
  tests/benchmark/test_corpus_provenance.py
rtk uv run ruff check src/benchmark/corpus_provenance.py \
  tests/benchmark/test_corpus_provenance.py
```

Expected: all provenance tests pass.

- [ ] **Step 6: Commit provenance support**

Run:

```bash
rtk git add config/corpus-provenance.json src/benchmark/corpus_provenance.py \
  tests/benchmark/test_corpus_provenance.py
rtk git commit -m "feat: add corpus provenance mapping"
```

---

### Task 5: Implement content-addressed cache synchronization

**Files:**
- Create: `src/benchmark/corpus_cache.py`
- Create: `tests/benchmark/test_corpus_cache.py`

**Interfaces:**
- Consumes: `R2ObjectStore`, `R2Config`, and `RemoteObject`
- Produces: `is_chart_definition_key(key: str) -> bool`
- Produces: `CacheIdentity`, `CacheIndexEntry`, `CacheObjectResult`, and `CacheSyncResult`
- Produces: `sync_corpus_cache(store, config, objects, cache_root, dry_run=False)`
- Guarantees: no body read or cache/index mutation in dry-run

- [ ] **Step 1: Write failing selection and first-download tests**

Create `tests/benchmark/test_corpus_cache.py` with:

```python
from datetime import datetime, timezone
from io import BytesIO

from src.benchmark.corpus_cache import is_chart_definition_key, sync_corpus_cache
from src.benchmark.r2_corpus_models import R2Config, RemoteObject
from src.benchmark.r2_store import ObjectDownload

NOW = datetime(2026, 7, 24, tzinfo=timezone.utc)


class FakeStore:
    def __init__(self, bodies):
        self.bodies = bodies
        self.get_calls = []

    def get_object(self, bucket, key, etag):
        self.get_calls.append((bucket, key, etag))
        body = self.bodies[key]
        return ObjectDownload(BytesIO(body), len(body))


def object_for(key, body, etag="etag-1"):
    return RemoteObject(key, len(body), etag, NOW, "text/plain")


def test_initial_profile_is_case_insensitive_and_excludes_audio():
    assert is_chart_definition_key("42/SET.DEF")
    assert is_chart_definition_key("42/MAS.DTX")
    assert is_chart_definition_key("42/readme.Txt")
    assert not is_chart_definition_key("42/bgm.ogg")
    assert not is_chart_definition_key("42/assets/snare.wav")


def test_downloads_selected_body_and_installs_by_sha256(tmp_path):
    body = b"#TITLE: chart\n"
    remote = object_for("42/SET.DEF", body)
    store = FakeStore({remote.key: body})
    config = R2Config.from_values(
        "https://abc.r2.cloudflarestorage.com",
        "simfile-dtx",
    )

    result = sync_corpus_cache(store, config, [remote], tmp_path)

    item = result.objects[remote.key]
    assert item.cache_status == "verified"
    assert item.action == "downloaded"
    assert item.sha256 is not None
    assert item.cache_path == f"sha256/{item.sha256[:2]}/{item.sha256}"
    assert (tmp_path / item.cache_path).read_bytes() == body
    assert store.get_calls == [("simfile-dtx", remote.key, "etag-1")]
```

- [ ] **Step 2: Run cache tests and verify failure**

Run:

```bash
rtk uv run pytest tests/benchmark/test_corpus_cache.py -q
```

Expected: collection fails because `src.benchmark.corpus_cache` is missing.

- [ ] **Step 3: Implement cache records and strict index loading**

Create `src/benchmark/corpus_cache.py` with:

```python
CacheStatus = Literal["not_selected", "planned", "verified", "failed"]
CacheAction = Literal["none", "planned", "cache_hit", "downloaded", "failed"]


@dataclass(frozen=True)
class CacheIdentity:
    source_endpoint_sha256: str
    bucket: str
    key: str


@dataclass(frozen=True)
class CacheIndexEntry:
    identity: CacheIdentity
    etag: str | None
    size: int
    last_modified: str
    sha256: str
    cache_path: str


@dataclass(frozen=True)
class CacheObjectResult:
    cache_status: CacheStatus
    action: CacheAction
    sha256: str | None = None
    cache_path: str | None = None
    error: SyncError | None = None


@dataclass(frozen=True)
class CacheSyncResult:
    objects: dict[str, CacheObjectResult]
    index_entries: tuple[CacheIndexEntry, ...]
    bytes_planned: int
    bytes_downloaded: int
```

Use an index document shaped as:

```json
{
  "schema_version": "crux.r2-cache-index/v1",
  "entries": [
    {
      "source_endpoint_sha256": "…",
      "bucket": "simfile-dtx",
      "key": "42/SET.DEF",
      "etag": "etag-1",
      "size": 14,
      "last_modified": "2026-07-24T00:00:00Z",
      "sha256": "…",
      "cache_path": "sha256/ab/ab…"
    }
  ]
}
```

Reject duplicate identities, absolute cache paths, and paths containing `..`.

- [ ] **Step 4: Implement verification, conditional download, and atomic writes**

`sync_corpus_cache` must:

- return `not_selected`/`none` without touching bodies for unselected keys;
- compare endpoint hash, bucket, exact key, ETag, size, and normalized modification time;
- hash and count a matching local cache body before returning `verified`/`cache_hit`;
- download mismatches through `store.get_object(bucket, key, etag)`;
- read in 1 MiB chunks, hash while writing to `<cache-root>/tmp/`;
- close the body in `finally`;
- verify both listed size and response content length when present;
- use `os.replace` for index publication;
- install the content path atomically and never overwrite different bytes;
- remove temporary files after every failure;
- convert 412/precondition errors to `source_changed_during_sync`;
- use only safe `R2OperationError` fields in `SyncError`.

For `dry_run=True`, it may hash an existing candidate cache file but must return
`planned` for misses without creating directories, temp files, bodies, or the index.

Use this streaming core; the surrounding function converts its raised safe errors to
`CacheObjectResult`:

```python
def _download_and_install(
    store: R2ObjectStore,
    config: R2Config,
    remote: RemoteObject,
    cache_root: Path,
) -> tuple[str, str, int]:
    temp_dir = cache_root / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    response: ObjectDownload | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            digest = hashlib.sha256()
            byte_count = 0
            response = store.get_object(config.bucket, remote.key, remote.etag)
            while chunk := response.body.read(1024 * 1024):
                digest.update(chunk)
                temp_file.write(chunk)
                byte_count += len(chunk)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        if byte_count != remote.size:
            raise CacheVerificationError("Downloaded byte count differs from inventory")
        if response.content_length is not None and byte_count != response.content_length:
            raise CacheVerificationError("Downloaded byte count differs from response metadata")

        sha256_hex = digest.hexdigest()
        relative_path = Path("sha256") / sha256_hex[:2] / sha256_hex
        destination = cache_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        _install_content_once(temp_path, destination, sha256_hex)
        temp_path = None
        return sha256_hex, relative_path.as_posix(), byte_count
    finally:
        if response is not None:
            response.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _install_content_once(temp_path: Path, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        if _sha256_file(destination) != expected_sha256:
            raise CacheVerificationError("Existing content-addressed cache file is corrupt")
        temp_path.unlink()
        return
    try:
        os.link(temp_path, destination)
    except FileExistsError:
        if _sha256_file(destination) != expected_sha256:
            raise CacheVerificationError("Concurrent cache publication produced different bytes")
    finally:
        temp_path.unlink(missing_ok=True)
```

Define `_sha256_file(path: Path) -> str` using the same 1 MiB chunk size. Write the
cache index through a temporary sibling, `flush`, `os.fsync`, and `os.replace`.

- [ ] **Step 5: Add cache-hit, corruption, mismatch, and dry-run tests**

Add exact tests asserting:

- an unchanged rerun returns `verified`/`cache_hit` and makes zero new `get_object` calls;
- replacing cached bytes with `b"corrupt"` causes a new download and repairs the file;
- a response content length or final byte count mismatch returns `failed` and leaves no
  temp file or index entry;
- an `R2OperationError("precondition_failed", "get", 412, key)` becomes
  `source_changed_during_sync`;
- dry-run returns `planned`, reports bytes, performs zero GETs, and leaves the cache root
  absent;
- two keys with the same body share one SHA-256 content file but retain separate index
  entries.

- [ ] **Step 6: Run cache tests and checks**

Run:

```bash
rtk uv run pytest tests/benchmark/test_corpus_cache.py -q
rtk uv run black src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
rtk uv run ruff check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
```

Expected: all cache tests pass.

- [ ] **Step 7: Commit cache synchronization**

Run:

```bash
rtk git add src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
rtk git commit -m "feat: add verified R2 corpus cache"
```

---

### Task 6: Canonicalize and publish immutable JSONL manifests

**Files:**
- Create: `src/benchmark/corpus_manifest.py`
- Create: `tests/benchmark/test_corpus_manifest.py`

**Interfaces:**
- Consumes: `InventorySnapshot`, cache results, provenance records, `R2Config`
- Produces: `build_manifest_rows(snapshot: InventorySnapshot, cache: CacheSyncResult,
  provenance: Mapping[int, ProvenanceRecord], config: R2Config) ->
  list[dict[str, object]]`
- Produces: `seal_manifest(rows, output_dir, published_at) -> ManifestPublication`
- Produces: `verify_manifest_cache(manifest_path: Path, cache_root: Path) ->
  list[SyncError]`
- Produces: `ManifestPublication(corpus_version, manifest_sha256, manifest_path)`

- [ ] **Step 1: Write failing determinism and immutability tests**

Create `tests/benchmark/test_corpus_manifest.py` with fixtures for one complete simfile
and assert:

```python
from datetime import datetime, timezone

PUBLISHED_AT = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
LATER_PUBLISHED_AT = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)


def make_rows(etag="etag-1"):
    return [
        {
            "schema_version": "crux.r2-corpus-manifest/v1",
            "simfile_id": 42,
            "object_prefix": "42/",
            "source_endpoint_sha256": "a" * 64,
            "source_bucket": "simfile-dtx",
            "source_discovery_method": "r2_list_objects_v2",
            "objects": [
                {
                    "key": "42/SET.DEF",
                    "size": 3,
                    "etag": etag,
                    "version": None,
                    "last_modified": "2026-07-24T00:00:00Z",
                    "content_type": "text/plain",
                    "remote_checksums": {},
                    "cache_status": "verified",
                    "sha256": "b" * 64,
                    "cache_path": f"sha256/bb/{'b' * 64}",
                }
            ],
            "sync_status": "complete",
            "sync_errors": [],
            "source_origin": None,
            "source_author_or_pack": None,
            "source_reference": None,
            "rights_status": "unknown",
            "redistribution_allowed": None,
            "provenance_notes": None,
        }
    ]


def test_manifest_is_identical_for_unchanged_rows_across_publication_times(tmp_path):
    rows = make_rows()

    first = seal_manifest(rows, tmp_path, PUBLISHED_AT)
    second = seal_manifest(rows, tmp_path, LATER_PUBLISHED_AT)

    assert first.corpus_version == second.corpus_version
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()


def test_changed_object_metadata_creates_new_manifest_without_overwriting_old(tmp_path):
    first = seal_manifest(make_rows(etag="etag-1"), tmp_path, PUBLISHED_AT)
    second = seal_manifest(make_rows(etag="etag-2"), tmp_path, LATER_PUBLISHED_AT)

    assert first.manifest_sha256 != second.manifest_sha256
    assert first.manifest_path.exists()
    assert second.manifest_path.exists()
```

Also assert:

- `build_manifest_rows` returns identical rows when two otherwise identical
  `CacheSyncResult` fixtures differ only by `CacheObjectResult.action` values
  `downloaded` and `cache_hit`, and serialized bytes contain no `cache_action`;
- simfiles sort numerically and objects sort by exact Unicode key;
- JSON is compact, `ensure_ascii=False`, and ends with one newline per row;
- absolute cache roots and invocation timestamps are absent;
- changing provenance changes corpus and manifest identities;
- a pre-existing mismatched file at the computed hash raises `ManifestCollisionError`;
- `latest.json` contains the concrete relative hash path and updates atomically.
- `verify_manifest_cache` returns no errors for valid references, returns stable errors
  for missing/size/SHA mismatches, and rejects absolute or `..` cache paths without
  reading outside the cache root.

- [ ] **Step 2: Run manifest tests and verify failure**

Run:

```bash
rtk uv run pytest tests/benchmark/test_corpus_manifest.py -q
```

Expected: collection fails because `src.benchmark.corpus_manifest` is missing.

- [ ] **Step 3: Implement row construction**

Implement `build_manifest_rows` so every row contains the exact v1 fields approved in
the design:

```python
{
    "schema_version": "crux.r2-corpus-manifest/v1",
    "simfile_id": simfile.simfile_id,
    "object_prefix": simfile.object_prefix,
    "source_endpoint_sha256": config.source_endpoint_sha256,
    "source_bucket": config.bucket,
    "source_discovery_method": snapshot.source_discovery_method,
    "objects": object_rows,
    "sync_status": simfile.status,
    "sync_errors": error_rows,
    **provenance.to_dict(),
}
```

Object rows use `cache_status` but never `cache_action`. Normalize ETags, timestamps,
checksum key order, nullable values, safe errors, and logical cache paths here.

- [ ] **Step 4: Implement two-stage identity and atomic publication**

Use:

```python
def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
```

Then:

1. Sort base rows and their object arrays.
2. Hash canonical JSONL payload bytes without `corpus_version`.
3. Insert `corpus_version=f"sha256:{payload_hash}"` into every row.
4. Serialize final JSONL with one trailing newline per row.
5. Hash the final bytes.
6. Write and fsync a temporary file beneath `manifests/`.
7. If the hash destination exists, compare exact bytes; reuse only when identical.
8. Atomically install a new destination otherwise.
9. Atomically write `latest.json` containing corpus version, manifest hash, relative
   path, and the supplied publication timestamp.

Implement `verify_manifest_cache` by reading each JSONL row, considering only objects
with `cache_status == "verified"`, validating the logical relative path, then comparing
the local byte count and SHA-256. Return `SyncError(scope="cache", ...)` values instead
of raising for per-object verification failures.

- [ ] **Step 5: Run manifest tests and checks**

Run:

```bash
rtk uv run pytest tests/benchmark/test_corpus_manifest.py -q
rtk uv run black src/benchmark/corpus_manifest.py tests/benchmark/test_corpus_manifest.py
rtk uv run ruff check src/benchmark/corpus_manifest.py tests/benchmark/test_corpus_manifest.py
```

Expected: all manifest tests pass.

- [ ] **Step 6: Commit immutable manifest publication**

Run:

```bash
rtk git add src/benchmark/corpus_manifest.py tests/benchmark/test_corpus_manifest.py
rtk git commit -m "feat: publish immutable corpus manifests"
```

---

### Task 7: Orchestrate sync attempts and machine-readable reports

**Files:**
- Create: `src/benchmark/r2_corpus_sync.py`
- Create: `tests/benchmark/test_r2_corpus_sync.py`

**Interfaces:**
- Consumes: every public interface from Tasks 1–6
- Produces: `SyncOptions`, `SyncSummary`, `SyncOutcome`, and `sync_r2_corpus`
- Produces: timestamped report and `latest-report.json`
- Guarantees: partial isolation, fatal handling, and dry-run mutation boundary

- [ ] **Step 1: Write failing end-to-end service tests**

Create `tests/benchmark/test_r2_corpus_sync.py` around an injected fake store:

```python
from datetime import datetime, timezone
from io import BytesIO

FIXED_NOW = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


class SuccessfulStore:
    def __init__(self, bodies):
        self.bodies = bodies
        self.get_calls = []

    def validate_bucket(self, bucket):
        assert bucket == "simfile-dtx"

    def list_objects(self, bucket):
        return [
            ListedObject(key, len(body), f"etag-{index}", FIXED_NOW)
            for index, (key, body) in enumerate(sorted(self.bodies.items()))
        ]

    def head_object(self, bucket, key):
        return HeadMetadata("text/plain", None, {})

    def get_object(self, bucket, key, etag):
        self.get_calls.append((bucket, key, etag))
        body = self.bodies[key]
        return ObjectDownload(BytesIO(body), len(body))


def successful_store(bodies):
    return SuccessfulStore(bodies)


def make_options(tmp_path, dry_run=False):
    return SyncOptions(
        config=R2Config.from_values(
            "https://abc.r2.cloudflarestorage.com",
            "simfile-dtx",
        ),
        output_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        dry_run=dry_run,
    )


def test_real_sync_publishes_manifest_report_and_latest_pointer(tmp_path):
    store = successful_store({"42/SET.DEF": b"set", "42/mas.dtx": b"chart"})
    outcome = sync_r2_corpus(make_options(tmp_path), store=store, now=FIXED_NOW)

    assert outcome.success is True
    assert outcome.partial is False
    assert outcome.publication is not None
    assert outcome.publication.manifest_path.exists()
    assert (tmp_path / "latest.json").exists()
    report = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "complete"
    assert report["downloads"]["completed"] == 2
    assert "endpoint_url" not in report


def test_dry_run_writes_report_only(tmp_path):
    store = successful_store({"42/SET.DEF": b"set"})
    outcome = sync_r2_corpus(
        make_options(tmp_path, dry_run=True),
        store=store,
        now=FIXED_NOW,
    )

    assert outcome.success is True
    assert outcome.publication is None
    assert outcome.report_path.exists()
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "manifests").exists()
    assert not (tmp_path / "latest.json").exists()
    assert store.get_calls == []
```

Add tests proving:

- an object HEAD failure publishes a partial manifest and returns `success=False`;
- a selected GET failure affects only its simfile and unrelated simfiles complete;
- malformed root keys produce overall `partial`, appear in the report, and cause a
  nonzero outcome;
- invalid provenance fails before `validate_bucket` or list calls;
- auth/root-list failure writes a fatal report and publishes no manifest;
- an unwritable report root raises `SyncReportWriteError` with a sanitized message;
- a second unchanged real run reports cache hits, zero downloaded bytes, and the same
  manifest hash.

Add this exact cache-reference test for the final verification command:

```python
def test_manifest_cache_references_verify(tmp_path):
    store = successful_store({"42/SET.DEF": b"set", "42/mas.dtx": b"chart"})
    outcome = sync_r2_corpus(make_options(tmp_path), store=store, now=FIXED_NOW)
    assert outcome.publication is not None

    for line in outcome.publication.manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for item in record["objects"]:
            if item["cache_status"] != "verified":
                continue
            cache_path = tmp_path / "cache" / item["cache_path"]
            body = cache_path.read_bytes()
            assert len(body) == item["size"]
            assert hashlib.sha256(body).hexdigest() == item["sha256"]
```

- [ ] **Step 2: Run orchestration tests and verify failure**

Run:

```bash
rtk uv run pytest tests/benchmark/test_r2_corpus_sync.py -q
```

Expected: collection fails because `src.benchmark.r2_corpus_sync` is missing.

- [ ] **Step 3: Implement options and outcome records**

Create:

```python
@dataclass(frozen=True)
class SyncOptions:
    config: R2Config
    output_dir: Path
    cache_dir: Path
    provenance_file: Path | None = None
    include_ids: frozenset[int] = frozenset()
    exclude_ids: frozenset[int] = frozenset()
    dry_run: bool = False
    head_workers: int = 8


@dataclass(frozen=True)
class SyncSummary:
    simfiles: int
    objects: int
    planned_downloads: int
    completed_downloads: int
    cache_hits: int
    bytes_planned: int
    bytes_downloaded: int


@dataclass(frozen=True)
class SyncOutcome:
    success: bool
    partial: bool
    report_path: Path
    publication: ManifestPublication | None
    summary: SyncSummary
```

Validate non-negative filters and `head_workers >= 1` before provenance parsing or
network access.

- [ ] **Step 4: Implement transactional orchestration**

`sync_r2_corpus(options, store=None, now=None)` must:

1. validate options;
2. load provenance;
3. construct `Boto3R2Store.from_config` only when no store is injected;
4. inventory and HEAD-enrich the selected simfiles;
5. flatten objects for cache planning/synchronization;
6. merge per-object cache failures back into simfile status and errors;
7. build normalized manifest rows;
8. skip manifest publication in dry-run;
9. publish real complete or partial manifests;
10. compute operational counters from cache actions;
11. write a timestamped report and atomically update `latest-report.json`;
12. return `success=False` for partial/fatal results without discarding publication.

Use report schema `crux.r2-corpus-sync-report/v1`. Include the endpoint hash and bucket,
but not the raw endpoint. Use allowlisted errors throughout.

Use this control-flow shape so fatal results return an outcome after writing their
report, while only report-write failures raise:

```python
def sync_r2_corpus(
    options: SyncOptions,
    store: R2ObjectStore | None = None,
    now: datetime | None = None,
) -> SyncOutcome:
    started_at = now or datetime.now(timezone.utc)
    try:
        _validate_options(options)
        provenance = load_provenance(options.provenance_file)
        resolved_store = store or Boto3R2Store.from_config(options.config)
        snapshot = inventory_r2_corpus(
            resolved_store,
            options.config,
            options.include_ids,
            options.exclude_ids,
            options.head_workers,
        )
        cache = sync_corpus_cache(
            resolved_store,
            options.config,
            [obj for item in snapshot.simfiles for obj in item.objects],
            options.cache_dir,
            options.dry_run,
        )
        merged_snapshot = _merge_cache_failures(snapshot, cache)
        rows = build_manifest_rows(merged_snapshot, cache, provenance, options.config)
        publication = (
            None
            if options.dry_run
            else seal_manifest(rows, options.output_dir, started_at)
        )
        partial = merged_snapshot.is_partial
        report = _build_report(
            options,
            merged_snapshot,
            cache,
            publication,
            started_at,
            "partial" if partial else "complete",
        )
    except (ValueError, R2OperationError, OSError) as error:
        report = _build_fatal_report(options, started_at, _safe_fatal_error(error))
        report_path = _write_report(options.output_dir, report)
        return SyncOutcome(False, False, report_path, None, _empty_summary())

    report_path = _write_report(options.output_dir, report)
    return SyncOutcome(
        not partial,
        partial,
        report_path,
        publication,
        _summary_from(merged_snapshot, cache),
    )
```

`_merge_cache_failures` uses `dataclasses.replace` to append cache errors and change
only affected simfiles to `partial` or `failed`. `_safe_fatal_error` maps known local
and SDK exceptions to stable codes/messages and does not use `repr(error)`.
`_empty_summary()` returns seven zero values; `_summary_from` derives its values only
from the inventory and cache action records used for the report.

- [ ] **Step 5: Implement fatal report fallback**

Wrap configuration/provenance/store/inventory failures. When the report directory is
usable, write a fatal report with no manifest fields. When it is not usable, raise
`SyncReportWriteError("Unable to write R2 corpus sync report")` without chaining a raw
filesystem or SDK message into user output.

Implement report writes through:

```python
def _write_report(output_dir: Path, report: dict[str, object]) -> Path:
    try:
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_bytes = canonical_report_json(report)
        report_path = reports_dir / _report_filename(report["started_at"])
        _atomic_write(report_path, report_bytes)
        _atomic_write(
            output_dir / "latest-report.json",
            canonical_report_json({"report_path": str(report_path.relative_to(output_dir))}),
        )
        return report_path
    except OSError:
        raise SyncReportWriteError("Unable to write R2 corpus sync report") from None
```

Define `canonical_report_json` as sorted UTF-8 JSON plus one final newline.
`_report_filename` converts the already normalized `started_at` value to
`sync-YYYYMMDDTHHMMSSffffffZ.json`. `_atomic_write` writes a temporary sibling,
flushes and fsyncs it, then calls `os.replace`.

- [ ] **Step 6: Run orchestration tests and checks**

Run:

```bash
rtk uv run pytest tests/benchmark/test_r2_corpus_sync.py -q
rtk uv run black src/benchmark/r2_corpus_sync.py \
  tests/benchmark/test_r2_corpus_sync.py
rtk uv run ruff check src/benchmark/r2_corpus_sync.py \
  tests/benchmark/test_r2_corpus_sync.py
```

Expected: all orchestration tests pass.

- [ ] **Step 7: Commit orchestration and reports**

Run:

```bash
rtk git add src/benchmark/r2_corpus_sync.py tests/benchmark/test_r2_corpus_sync.py
rtk git commit -m "feat: orchestrate R2 corpus synchronization"
```

---

### Task 8: Expose the CLI and document the workflow

**Files:**
- Create: `src/cli/r2_corpus.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md`

**Interfaces:**
- Consumes: `R2Config`, `SyncOptions`, `sync_r2_corpus`
- Produces: Click command `sync-r2-corpus`
- Produces: documented operator commands and artifact layout

- [ ] **Step 1: Write failing CLI help and validation tests**

Add to `tests/test_cli_benchmark.py`:

```python
def test_sync_r2_corpus_help_lists_safe_configuration_options():
    result = CliRunner().invoke(main, ["benchmark", "sync-r2-corpus", "--help"])

    assert result.exit_code == 0
    assert "--include-simfile-id" in result.output
    assert "--exclude-simfile-id" in result.output
    assert "--provenance-file" in result.output
    assert "--dry-run" in result.output
    assert "secret" not in result.output.lower()
    assert "access-key" not in result.output.lower()


def test_sync_r2_corpus_requires_endpoint_without_printing_values(monkeypatch):
    monkeypatch.delenv("CRUX_R2_ENDPOINT_URL", raising=False)
    result = CliRunner().invoke(main, ["benchmark", "sync-r2-corpus", "--dry-run"])

    assert result.exit_code != 0
    assert "--endpoint-url" in result.output
    assert "AWS_SECRET_ACCESS_KEY" not in result.output
```

Add injected-service CLI tests by monkeypatching
`src.cli.r2_corpus.sync_r2_corpus` and asserting:

- repeated include/exclude IDs become `frozenset[int]`;
- exclude wins in the service result;
- default output is `artifacts/benchmark/r2-corpus`;
- default cache is `<output>/cache`;
- `CRUX_R2_BUCKET` defaults to `simfile-dtx`;
- complete output exits 0 and prints manifest hash/path;
- partial output prints the report path and raises `click.ClickException`;
- dry-run prints planned download/object counts and no manifest claim.

- [ ] **Step 2: Run CLI tests and verify missing command failure**

Run:

```bash
rtk uv run pytest tests/test_cli_benchmark.py -q
```

Expected: the new help test fails with `No such command 'sync-r2-corpus'`.

- [ ] **Step 3: Implement the Click command**

Create `src/cli/r2_corpus.py`:

```python
DEFAULT_OUTPUT_DIR = Path("artifacts") / "benchmark" / "r2-corpus"


@click.command("sync-r2-corpus")
@click.option(
    "--endpoint-url",
    envvar="CRUX_R2_ENDPOINT_URL",
    required=True,
    show_envvar=True,
    help="Cloudflare R2 S3 endpoint; prefer CRUX_R2_ENDPOINT_URL.",
)
@click.option(
    "--bucket",
    envvar="CRUX_R2_BUCKET",
    default="simfile-dtx",
    show_default=True,
)
@click.option("--include-simfile-id", type=click.IntRange(min=0), multiple=True)
@click.option("--exclude-simfile-id", type=click.IntRange(min=0), multiple=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR)
@click.option("--cache-dir", type=click.Path(path_type=Path), required=False)
@click.option("--provenance-file", type=click.Path(path_type=Path), required=False)
@click.option("--dry-run", is_flag=True)
def sync_r2_corpus_command(
    endpoint_url: str,
    bucket: str,
    include_simfile_id: tuple[int, ...],
    exclude_simfile_id: tuple[int, ...],
    output_dir: Path,
    cache_dir: Path | None,
    provenance_file: Path | None,
    dry_run: bool,
) -> None:
    """Inventory R2 simfiles and publish a verified, immutable base manifest."""
    try:
        config = R2Config.from_values(endpoint_url, bucket)
    except ValueError as error:
        raise click.ClickException(str(error)) from None
    resolved_cache_dir = cache_dir or output_dir / "cache"
    outcome = sync_r2_corpus(
        SyncOptions(
            config=config,
            output_dir=output_dir,
            cache_dir=resolved_cache_dir,
            provenance_file=provenance_file,
            include_ids=frozenset(include_simfile_id),
            exclude_ids=frozenset(exclude_simfile_id),
            dry_run=dry_run,
        )
    )
    summary = outcome.summary
    click.echo(
        f"Inventoried {summary.simfiles} simfile(s), {summary.objects} object(s); "
        f"cache hits {summary.cache_hits}, downloads {summary.completed_downloads}, "
        f"planned {summary.planned_downloads}"
    )
    if outcome.publication is not None:
        click.echo(
            f"Manifest {outcome.publication.manifest_sha256} "
            f"at {outcome.publication.manifest_path}"
        )
    click.echo(f"Sync report: {outcome.report_path}")
    if not outcome.success:
        raise click.ClickException("R2 corpus synchronization completed with errors")
```

The endpoint option exists for controlled automation but its value must never be echoed.
There are no access-key, secret-key, session-token, signed-URL, or credential-file
options.

Construct `R2Config`, derive the default cache directory, invoke the service, print a
concise safe summary, and raise `click.ClickException` after printing the report path
when `outcome.success` is false.

Register it in `src/cli/benchmark.py`:

```python
from src.cli.r2_corpus import sync_r2_corpus_command

benchmark.add_command(sync_r2_corpus_command)
```

- [ ] **Step 4: Document environment, commands, artifacts, and rights**

Add an "Authoritative R2 corpus inventory" section before the existing local
`prepare-corpus` workflow in `docs/drumery-dtx-midi-benchmarking-reference.md`. Include:

```bash
export CRUX_R2_ENDPOINT_URL=https://ACCOUNT_ID.r2.cloudflarestorage.com
export CRUX_R2_BUCKET=simfile-dtx
export AWS_ACCESS_KEY_ID=REDACTED
export AWS_SECRET_ACCESS_KEY=REDACTED

uv run crux benchmark sync-r2-corpus \
  --provenance-file config/corpus-provenance.json \
  --dry-run

uv run crux benchmark sync-r2-corpus \
  --provenance-file config/corpus-provenance.json
```

The documentation must state:

- use a bucket-scoped read-only token;
- shell examples use `REDACTED`, never real values;
- the initial profile caches chart-definition text only;
- `latest.json` is not an immutable benchmark input;
- benchmark runs must record `manifests/<sha256>.jsonl`;
- unknown rights metadata permits private inventory but not public redistribution;
- expected artifact layout and sync report locations.

- [ ] **Step 5: Run CLI and documentation checks**

Run:

```bash
rtk uv run pytest tests/test_cli_benchmark.py -q
rtk uv run crux benchmark sync-r2-corpus --help
rtk uv run black src/cli/r2_corpus.py src/cli/benchmark.py tests/test_cli_benchmark.py
rtk uv run ruff check src/cli/r2_corpus.py src/cli/benchmark.py tests/test_cli_benchmark.py
```

Expected: CLI tests pass, help contains no credential-valued options, and formatting
checks succeed.

- [ ] **Step 6: Commit CLI and documentation**

Run:

```bash
rtk git add src/cli/r2_corpus.py src/cli/benchmark.py tests/test_cli_benchmark.py \
  docs/drumery-dtx-midi-benchmarking-reference.md
rtk git commit -m "feat: expose R2 corpus sync command"
```

---

### Task 9: Verify the full implementation and production R2 behavior

**Files:**
- Modify only if a verification failure exposes a scoped HPA-321 defect.

**Interfaces:**
- Consumes: complete HPA-321 CLI and artifact contract
- Produces: verified local test stack and credential-gated production evidence

- [ ] **Step 1: Run the complete HPA-321 test slice**

Run:

```bash
rtk uv run pytest \
  tests/benchmark/test_r2_corpus_models.py \
  tests/benchmark/test_r2_store.py \
  tests/benchmark/test_r2_inventory.py \
  tests/benchmark/test_corpus_provenance.py \
  tests/benchmark/test_corpus_cache.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_r2_corpus_sync.py \
  tests/test_cli_benchmark.py -q
```

Expected: all HPA-321 and benchmark CLI tests pass.

- [ ] **Step 2: Run the repository verification stack**

Run:

```bash
rtk uv run pytest
rtk uv run ruff check src tests
rtk uv run black --check src tests
rtk uv run pylint src/app src/cli src/benchmark
```

Expected: every command exits 0. If Pylint exposes pre-existing unrelated failures,
record the exact command and baseline evidence; do not broaden HPA-321 into unrelated
cleanup.

- [ ] **Step 3: Inspect the diff for secret and path leaks**

Run:

```bash
rtk rg -n -P \
  "AWS_(ACCESS_KEY_ID|SECRET_ACCESS_KEY)=(?!REDACTED(?:$|\\s))|X-Amz-Signature=" \
  src tests config docs/drumery-dtx-midi-benchmarking-reference.md
rtk rg -n \
  "https://[a-f0-9]{32}\\.r2\\.cloudflarestorage\\.com" \
  src tests config docs/drumery-dtx-midi-benchmarking-reference.md
rtk git diff --check origin/main...HEAD
```

Expected: no real credential, signature, or account-bearing endpoint is committed; the
documentation endpoint uses `ACCOUNT_ID` and credential values use `REDACTED`.

- [ ] **Step 4: Run a credential-gated include dry-run**

With `CRUX_R2_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` configured
outside Git, run:

```bash
rtk uv run crux benchmark sync-r2-corpus \
  --provenance-file config/corpus-provenance.json \
  --include-simfile-id 116 \
  --dry-run
```

Expected: the report contains exact nested keys or an explicit `empty` result for
simfile 116, planned chart-definition downloads, zero GET/download counts, and no
manifest.

- [ ] **Step 5: Run a full production dry-run**

Run:

```bash
rtk uv run crux benchmark sync-r2-corpus \
  --provenance-file config/corpus-provenance.json \
  --dry-run
```

Expected: every valid numeric prefix is inventoried; malformed, ambiguous, inaccessible,
and empty cases are explicit; nested and non-ASCII keys remain exact; no manifest or
cache mutation occurs.

- [ ] **Step 6: Perform the first real chart-definition sync**

Run:

```bash
rtk uv run crux benchmark sync-r2-corpus \
  --provenance-file config/corpus-provenance.json
```

Expected: selected `set.def`, `.dtx`, and `.txt` bodies are SHA-256 verified; the
immutable JSONL manifest, cache index, sync report, `latest.json`, and
`latest-report.json` exist; any partial failure is explicit and returns nonzero.

- [ ] **Step 7: Repeat the real sync to prove idempotence**

Run the same command again:

```bash
rtk uv run crux benchmark sync-r2-corpus \
  --provenance-file config/corpus-provenance.json
```

Expected: zero body downloads, verified cache hits for every previously synchronized
selected object, and the same corpus version, manifest SHA-256, and immutable manifest
bytes as Step 6.

- [ ] **Step 8: Verify every manifest cache reference**

Run the manifest verifier against the produced default artifacts:

```bash
rtk uv run python -c 'import json; from pathlib import Path; from src.benchmark.corpus_manifest import verify_manifest_cache; root = Path("artifacts/benchmark/r2-corpus"); latest = json.loads((root / "latest.json").read_text(encoding="utf-8")); errors = verify_manifest_cache(root / latest["manifest_path"], root / "cache"); assert not errors, [(error.code, error.object_key) for error in errors]; print("verified manifest cache references")'
```

Expected: every `cache_status="verified"` row resolves beneath
`artifacts/benchmark/r2-corpus/cache/`, has the recorded byte count, and hashes to the
recorded SHA-256.

- [ ] **Step 9: Commit only scoped fixes discovered by final verification**

If Steps 1–8 required an HPA-321 correction, run:

```bash
rtk git add pyproject.toml uv.lock \
  src/benchmark/r2_corpus_models.py \
  src/benchmark/r2_store.py \
  src/benchmark/r2_inventory.py \
  src/benchmark/corpus_provenance.py \
  src/benchmark/corpus_cache.py \
  src/benchmark/corpus_manifest.py \
  src/benchmark/r2_corpus_sync.py \
  src/cli/r2_corpus.py \
  src/cli/benchmark.py \
  tests/benchmark/test_r2_corpus_models.py \
  tests/benchmark/test_r2_store.py \
  tests/benchmark/test_r2_inventory.py \
  tests/benchmark/test_corpus_provenance.py \
  tests/benchmark/test_corpus_cache.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_r2_corpus_sync.py \
  tests/test_cli_benchmark.py \
  config/corpus-provenance.json \
  docs/drumery-dtx-midi-benchmarking-reference.md
rtk git commit -m "fix: address R2 corpus verification findings"
```

If no correction was necessary, leave the branch unchanged and record the successful
commands in the implementation handoff.

---

## Reference Documentation

- Cloudflare R2 boto3 setup: <https://developers.cloudflare.com/r2/examples/aws/boto3/>
- Cloudflare R2 S3 compatibility: <https://developers.cloudflare.com/r2/api/s3/api/>
- Cloudflare multipart ETag behavior: <https://developers.cloudflare.com/r2/objects/upload-objects/>
- Boto3 ListObjectsV2 paginator: <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/paginator/ListObjectsV2.html>
- Boto3 HeadObject: <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/head_object.html>
- Boto3 GetObject and `IfMatch`: <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/get_object.html>
- Boto3 credential provider chain: <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html>
