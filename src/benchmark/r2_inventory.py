from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from logging import WARNING, getLogger
from typing import Any, BinaryIO, ContextManager, Protocol

from src.benchmark.r2_corpus_models import (
    MAX_SIMFILE_ID,
    ErrorCode,
    HeadMetadata,
    InventoryResult,
    ListedObject,
    R2Config,
    RemoteObject,
    SimfileInventory,
    SyncError,
    parse_etag,
)

_MISSING_DEPENDENCY_MESSAGE = (
    "R2 support is not installed. Install it with uv pip install -e '.[r2]'."
)
_MESSAGES: dict[ErrorCode, str] = {
    "missing_optional_dependency": _MISSING_DEPENDENCY_MESSAGE,
    "missing_credentials": "No usable R2 credentials were resolved.",
    "auth_failed": "R2 rejected the resolved credentials.",
    "bucket_inaccessible": "The configured R2 bucket is inaccessible.",
    "root_list_failed": "R2 root listing did not complete.",
    "object_head_failed": "Object metadata inspection failed.",
    "object_get_failed": "Object body read failed.",
    "source_changed_during_sync": "Object metadata changed after inventory.",
    "object_metadata_invalid": "Object metadata is invalid.",
}
_AUTH_ERROR_CODES = frozenset(
    {
        "AuthorizationHeaderMalformed",
        "ExpiredToken",
        "InvalidAccessKeyId",
        "InvalidToken",
        "SignatureDoesNotMatch",
        "TokenRefreshRequired",
    }
)
_AMBIGUOUS_PREFIX_MESSAGE = "Multiple exact prefixes normalize to one numeric ID."
_EMPTY_PREFIX_MESSAGE = "A requested prefix has no objects or only folder markers."
_MALFORMED_ROOT_KEY_MESSAGE = "A root key cannot be assigned to a valid simfile prefix."
_HEAD_ERROR_CODES = frozenset({"object_head_failed", "object_metadata_invalid"})


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

    def open_object(self, key: str, if_match: str | None) -> ContextManager[ObjectDownload]:
        raise NotImplementedError


class R2StoreError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str, object_key: str | None = None):
        super().__init__(message)
        self.code = code
        self.object_key = object_key


def build_inventory(
    store: R2ObjectStore,
    include_ids: frozenset[int],
    exclude_ids: frozenset[int],
    head_concurrency: int,
    item_progress: Callable[[int, int, int], None] | None = None,
) -> InventoryResult:
    listed = store.list_objects()
    groups, malformed_keys, ambiguous_prefixes = _classify_root(listed)
    ambiguous_ids = frozenset(ambiguous_prefixes)
    root_errors = _root_errors(malformed_keys, ambiguous_prefixes)
    selected: list[tuple[int, str, list[ListedObject]]] = []
    discovered_ids: set[int] = set()
    excluded_by_filter = 0

    for prefix in sorted(groups):
        simfile_id = int(prefix[:-1])
        if simfile_id in ambiguous_ids:
            continue
        discovered_ids.add(simfile_id)
        if simfile_id in exclude_ids or (include_ids and simfile_id not in include_ids):
            excluded_by_filter += 1
            continue
        selected.append((simfile_id, prefix, groups[prefix]))

    selected.extend(
        (simfile_id, f"{simfile_id}/", [])
        for simfile_id in sorted(include_ids - exclude_ids - discovered_ids - ambiguous_ids)
    )
    ordered_selected = sorted(selected, key=lambda row: (row[0], row[1]))
    total_heads = sum(len(objects) for _, _, objects in ordered_selected)
    completed_heads = 0
    completed_bytes = 0

    def record_head(size: int) -> None:
        nonlocal completed_heads, completed_bytes
        completed_heads += 1
        completed_bytes += size
        if item_progress is not None:
            item_progress(completed_heads, total_heads, completed_bytes)

    simfiles = tuple(
        _build_simfile_inventory(
            store,
            simfile_id,
            prefix,
            objects,
            head_concurrency,
            record_head,
        )
        for simfile_id, prefix, objects in ordered_selected
    )
    return InventoryResult(
        simfiles=simfiles,
        malformed_root_keys=malformed_keys,
        ambiguous_prefixes=ambiguous_prefixes,
        root_errors=root_errors,
        simfiles_discovered=len(groups)
        - sum(len(prefixes) for prefixes in ambiguous_prefixes.values()),
        simfiles_excluded_by_filter=excluded_by_filter,
        objects_listed=len(listed),
    )


def _classify_root(
    listed: tuple[ListedObject, ...],
) -> tuple[dict[str, list[ListedObject]], tuple[str, ...], dict[int, tuple[str, ...]]]:
    groups: dict[str, list[ListedObject]] = {}
    malformed_keys: list[str] = []

    for item in listed:
        first_segment, separator, _ = item.key.partition("/")
        if not separator or not first_segment.isascii() or not first_segment.isdigit():
            malformed_keys.append(item.key)
            continue
        simfile_id = int(first_segment)
        if simfile_id > MAX_SIMFILE_ID:
            malformed_keys.append(item.key)
            continue
        groups.setdefault(f"{first_segment}/", []).append(item)

    prefixes_by_id: dict[int, list[str]] = {}
    for prefix in groups:
        prefixes_by_id.setdefault(int(prefix[:-1]), []).append(prefix)
    ambiguous_prefixes = {
        simfile_id: tuple(sorted(prefixes))
        for simfile_id, prefixes in sorted(prefixes_by_id.items())
        if len(prefixes) > 1
    }
    return groups, tuple(sorted(malformed_keys)), ambiguous_prefixes


def _build_simfile_inventory(
    store: R2ObjectStore,
    simfile_id: int,
    object_prefix: str,
    listed: list[ListedObject],
    head_concurrency: int,
    item_progress: Callable[[int], None],
) -> SimfileInventory:
    ordered_listed = sorted(listed, key=lambda item: item.key)
    metadata = _head_metadata(store, ordered_listed, head_concurrency, item_progress)
    objects: list[RemoteObject] = []
    row_errors: list[SyncError] = []

    for item, result in zip(ordered_listed, metadata, strict=True):
        if isinstance(result, R2StoreError):
            error = _head_error(item.key, result)
            objects.append(
                RemoteObject(
                    key=item.key,
                    size=item.size,
                    etag=item.etag,
                    etag_is_weak=item.etag_is_weak,
                    last_modified=item.last_modified,
                    content_type=None,
                    errors=(*item.errors, error),
                )
            )
            row_errors.append(error)
            row_errors.extend(item.errors)
            continue
        objects.append(_merge_metadata(item, result))
        row_errors.extend(item.errors)

    has_object_errors = bool(row_errors)
    is_empty = not objects or all(item.size == 0 and item.key.endswith("/") for item in objects)
    if is_empty:
        row_errors.append(
            SyncError("simfile", "empty_prefix", _EMPTY_PREFIX_MESSAGE, object_prefix)
        )
    if has_object_errors:
        status = "partial"
    elif is_empty:
        status = "empty"
    else:
        status = "complete"
    return SimfileInventory(
        simfile_id=simfile_id,
        object_prefix=object_prefix,
        objects=tuple(objects),
        sync_status=status,
        sync_errors=tuple(sorted(row_errors, key=_error_sort_key)),
    )


def _head_metadata(
    store: R2ObjectStore,
    listed: list[ListedObject],
    head_concurrency: int,
    item_progress: Callable[[int], None],
) -> list[HeadMetadata | R2StoreError]:
    if not listed:
        return []
    with ThreadPoolExecutor(max_workers=head_concurrency) as executor:
        futures = {
            executor.submit(_head_one, store, item.key): (index, item.size)
            for index, item in enumerate(listed)
        }
        results: list[HeadMetadata | R2StoreError | None] = [None] * len(listed)
        for future in as_completed(futures):
            index, size = futures[future]
            results[index] = future.result()
            item_progress(size)
    assert all(result is not None for result in results)
    return [result for result in results if result is not None]


def _head_one(store: R2ObjectStore, key: str) -> HeadMetadata | R2StoreError:
    try:
        return store.head_object(key)
    except R2StoreError as error:
        return error


def _merge_metadata(item: ListedObject, metadata: HeadMetadata) -> RemoteObject:
    return RemoteObject(
        key=item.key,
        size=metadata.size if metadata.size is not None else item.size,
        etag=metadata.etag if metadata.etag is not None else item.etag,
        etag_is_weak=(
            metadata.etag_is_weak if metadata.etag_is_weak is not None else item.etag_is_weak
        ),
        last_modified=metadata.last_modified or item.last_modified,
        content_type=metadata.content_type,
        errors=item.errors,
    )


def _root_errors(
    malformed_keys: tuple[str, ...], ambiguous_prefixes: dict[int, tuple[str, ...]]
) -> tuple[SyncError, ...]:
    errors = [
        SyncError("root", "malformed_root_key", _MALFORMED_ROOT_KEY_MESSAGE, key)
        for key in malformed_keys
    ]
    errors.extend(
        SyncError(
            "root",
            "ambiguous_simfile_prefix",
            f"{_AMBIGUOUS_PREFIX_MESSAGE} ID {simfile_id}: {', '.join(prefixes)}",
            None,
        )
        for simfile_id, prefixes in ambiguous_prefixes.items()
    )
    return tuple(sorted(errors, key=_error_sort_key))


def _head_error(key: str, error: R2StoreError) -> SyncError:
    code: ErrorCode = error.code if error.code in _HEAD_ERROR_CODES else "object_head_failed"
    return SyncError("object", code, _MESSAGES[code], key)


def _error_sort_key(error: SyncError) -> tuple[str, str, str, str]:
    return error.scope, error.code, error.object_key or "", error.message


@dataclass(frozen=True)
class _R2Dependencies:
    boto3: Any
    config: Any
    client_error: type[Exception]
    no_credentials_error: type[Exception]
    partial_credentials_error: type[Exception]


def _import_optional_dependency(name: str) -> Any:
    return import_module(name)


def ensure_r2_dependency() -> _R2Dependencies:
    try:
        boto3 = _import_optional_dependency("boto3")
        config_module = _import_optional_dependency("botocore.config")
        exceptions_module = _import_optional_dependency("botocore.exceptions")
    except ImportError:
        raise R2StoreError(
            "missing_optional_dependency", _MESSAGES["missing_optional_dependency"]
        ) from None
    return _R2Dependencies(
        boto3=boto3,
        config=config_module.Config,
        client_error=exceptions_module.ClientError,
        no_credentials_error=exceptions_module.NoCredentialsError,
        partial_credentials_error=exceptions_module.PartialCredentialsError,
    )


def _suppress_sdk_loggers() -> None:
    for name in ("boto3", "botocore", "urllib3"):
        getLogger(name).setLevel(WARNING)


class Boto3R2Store:
    def __init__(self, client: Any, bucket: str):
        _suppress_sdk_loggers()
        self.client = client
        self.bucket = bucket

    def validate_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception as error:
            raise self._map_sdk_error(error, "bucket_inaccessible") from None

    def list_objects(self) -> tuple[ListedObject, ...]:
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            objects: list[ListedObject] = []
            for page in paginator.paginate(Bucket=self.bucket):
                if not isinstance(page, dict):
                    raise self._metadata_error()
                contents = page.get("Contents", [])
                if not isinstance(contents, list):
                    raise self._metadata_error()
                for item in contents:
                    try:
                        objects.append(self._parse_listed_object(item))
                    except R2StoreError:
                        continue
            return tuple(objects)
        except R2StoreError:
            raise
        except Exception as error:
            raise self._map_sdk_error(error, "root_list_failed") from None

    def head_object(self, key: str) -> HeadMetadata:
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as error:
            raise self._map_sdk_error(error, "object_head_failed", key) from None
        try:
            return self._parse_head_metadata(response)
        except R2StoreError:
            raise
        except (KeyError, TypeError, ValueError):
            raise self._metadata_error(key) from None

    @contextmanager
    def open_object(self, key: str, if_match: str | None) -> Iterator[ObjectDownload]:
        request = {"Bucket": self.bucket, "Key": key}
        if if_match is not None:
            request["IfMatch"] = f'"{if_match}"'
        try:
            response = self.client.get_object(**request)
        except Exception as error:
            raise self._map_sdk_error(error, "object_get_failed", key) from None
        try:
            body, download = self._parse_download(response, key)
        except R2StoreError as error:
            _close_response_body(response)
            if error.code == "object_metadata_invalid" and error.object_key is None:
                raise self._metadata_error(key) from None
            raise
        except (KeyError, TypeError, ValueError):
            _close_response_body(response)
            raise self._metadata_error(key) from None
        try:
            yield download
        except BaseException:
            try:
                body.close()
            except Exception:
                pass
            raise
        else:
            try:
                body.close()
            except Exception as error:
                raise self._map_sdk_error(error, "object_get_failed", key) from None

    def _parse_listed_object(self, response: Any) -> ListedObject:
        if not isinstance(response, dict):
            raise self._metadata_error()
        key = response.get("Key")
        if not isinstance(key, str) or not key:
            raise self._metadata_error(key if isinstance(key, str) else None) from None
        errors: list[SyncError] = []
        size: int = 0
        try:
            raw_size = response["Size"]
            if not _is_nonnegative_int(raw_size):
                raise ValueError
            size = raw_size
        except (KeyError, TypeError, ValueError):
            errors.append(
                SyncError(
                    "object", "object_metadata_invalid", _MESSAGES["object_metadata_invalid"], key
                )
            )
        raw_etag: str = ""
        etag: str = ""
        etag_is_weak: bool = False
        try:
            raw_etag = response["ETag"]
            if not isinstance(raw_etag, str):
                raise ValueError
            etag, etag_is_weak = parse_etag(raw_etag)
        except (KeyError, TypeError, ValueError):
            errors.append(
                SyncError(
                    "object", "object_metadata_invalid", _MESSAGES["object_metadata_invalid"], key
                )
            )
        last_modified: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
        try:
            raw_last_modified = response["LastModified"]
            if not _is_aware_datetime(raw_last_modified):
                raise ValueError
            last_modified = raw_last_modified
        except (KeyError, TypeError, ValueError):
            errors.append(
                SyncError(
                    "object", "object_metadata_invalid", _MESSAGES["object_metadata_invalid"], key
                )
            )
        return ListedObject(
            key=key,
            size=size,
            etag=etag,
            etag_is_weak=etag_is_weak,
            last_modified=last_modified,
            errors=tuple(errors),
        )

    def _parse_head_metadata(self, response: Any) -> HeadMetadata:
        if not isinstance(response, dict):
            raise self._metadata_error()
        etag, etag_is_weak = _nullable_etag(response.get("ETag"))
        return HeadMetadata(
            size=_nullable_size(response.get("ContentLength")),
            etag=etag,
            etag_is_weak=etag_is_weak,
            last_modified=_nullable_datetime(response.get("LastModified")),
            content_type=_nullable_string(response.get("ContentType")),
        )

    def _parse_download(self, response: Any, key: str) -> tuple[BinaryIO, ObjectDownload]:
        if not isinstance(response, dict):
            raise self._metadata_error()
        raw_body = response.get("Body")
        if not callable(getattr(raw_body, "read", None)) or not callable(
            getattr(raw_body, "close", None)
        ):
            raise self._metadata_error()
        body = _SanitizedObjectBody(raw_body, self, key)
        etag, etag_is_weak = _nullable_etag(response.get("ETag"))
        return body, ObjectDownload(
            body=body,
            size=_nullable_size(response.get("ContentLength")),
            etag=etag,
            etag_is_weak=etag_is_weak,
            last_modified=_nullable_datetime(response.get("LastModified")),
        )

    def _map_sdk_error(
        self, error: Exception, operation_code: ErrorCode, object_key: str | None = None
    ) -> R2StoreError:
        dependencies = ensure_r2_dependency()
        if isinstance(
            error, (dependencies.no_credentials_error, dependencies.partial_credentials_error)
        ):
            return R2StoreError("missing_credentials", _MESSAGES["missing_credentials"], object_key)
        if isinstance(error, dependencies.client_error):
            error_code, status = _client_error_details(error)
            if error_code in _AUTH_ERROR_CODES:
                return R2StoreError("auth_failed", _MESSAGES["auth_failed"], object_key)
            if status == 412:
                return R2StoreError(
                    "source_changed_during_sync",
                    _MESSAGES["source_changed_during_sync"],
                    object_key,
                )
        return R2StoreError(operation_code, _MESSAGES[operation_code], object_key)

    @staticmethod
    def _metadata_error(object_key: str | None = None) -> R2StoreError:
        return R2StoreError(
            "object_metadata_invalid", _MESSAGES["object_metadata_invalid"], object_key
        )


class _SanitizedObjectBody:
    def __init__(self, body: BinaryIO, store: Boto3R2Store, object_key: str):
        self._body = body
        self._store = store
        self._object_key = object_key

    def read(self, size: int = -1) -> bytes:
        try:
            return self._body.read(size)
        except Exception as error:
            raise self._store._map_sdk_error(error, "object_get_failed", self._object_key) from None

    def close(self) -> None:
        try:
            self._body.close()
        except Exception as error:
            raise self._store._map_sdk_error(error, "object_get_failed", self._object_key) from None


def _close_response_body(response: Any) -> None:
    if not isinstance(response, dict):
        return
    body = response.get("Body")
    close = getattr(body, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        pass


def create_boto3_store(config: R2Config) -> Boto3R2Store:
    dependencies = ensure_r2_dependency()
    _suppress_sdk_loggers()
    client_config = dependencies.config(
        connect_timeout=config.connect_timeout_seconds,
        read_timeout=config.read_timeout_seconds,
        tcp_keepalive=True,
        retries={"mode": "standard", "total_max_attempts": config.max_attempts},
        max_pool_connections=max(16, config.head_concurrency + config.download_concurrency),
    )
    try:
        client = dependencies.boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
            config=client_config,
        )
    except (dependencies.no_credentials_error, dependencies.partial_credentials_error):
        raise R2StoreError("missing_credentials", _MESSAGES["missing_credentials"]) from None
    return Boto3R2Store(client, config.bucket)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nullable_size(value: Any) -> int | None:
    if value is None:
        return None
    if not _is_nonnegative_int(value):
        raise ValueError
    return value


def _nullable_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if not _is_aware_datetime(value):
        raise ValueError
    return value


def _is_aware_datetime(value: Any) -> bool:
    if not isinstance(value, datetime) or value.tzinfo is None:
        return False
    try:
        return value.utcoffset() is not None
    except Exception:
        return False


def _nullable_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError
    return value


def _nullable_etag(value: Any) -> tuple[str | None, bool | None]:
    if value is None:
        return None, None
    if not isinstance(value, str):
        raise ValueError
    etag, weak = parse_etag(value)
    return etag, weak


def _client_error_details(error: Exception) -> tuple[str | None, int | None]:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None, None
    error_data = response.get("Error")
    metadata = response.get("ResponseMetadata")
    error_code = error_data.get("Code") if isinstance(error_data, dict) else None
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    return error_code if isinstance(error_code, str) else None, (
        status if isinstance(status, int) else None
    )
