from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from logging import WARNING, getLogger
from typing import Any, BinaryIO, ContextManager, Protocol

from src.benchmark.r2_corpus_models import (
    ErrorCode,
    HeadMetadata,
    ListedObject,
    R2Config,
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
            objects = []
            for page in paginator.paginate(Bucket=self.bucket):
                if not isinstance(page, dict):
                    raise self._metadata_error()
                contents = page.get("Contents", [])
                if not isinstance(contents, list):
                    raise self._metadata_error()
                objects.extend(self._parse_listed_object(item) for item in contents)
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
            request["IfMatch"] = if_match
        try:
            response = self.client.get_object(**request)
        except Exception as error:
            raise self._map_sdk_error(error, "object_get_failed", key) from None
        try:
            body, download = self._parse_download(response, key)
        except R2StoreError as error:
            if error.code == "object_metadata_invalid" and error.object_key is None:
                raise self._metadata_error(key) from None
            raise
        except (KeyError, TypeError, ValueError):
            raise self._metadata_error(key) from None
        try:
            yield download
        finally:
            try:
                body.close()
            except Exception as error:
                raise self._map_sdk_error(error, "object_get_failed", key) from None

    def _parse_listed_object(self, response: Any) -> ListedObject:
        try:
            if not isinstance(response, dict):
                raise ValueError
            key = response["Key"]
            size = response["Size"]
            raw_etag = response["ETag"]
            last_modified = response["LastModified"]
            if (
                not isinstance(key, str)
                or not key
                or not _is_nonnegative_int(size)
                or not isinstance(raw_etag, str)
                or not isinstance(last_modified, datetime)
            ):
                raise ValueError
            _, etag_is_weak = parse_etag(raw_etag)
            return ListedObject(
                key=key,
                size=size,
                etag=raw_etag,
                etag_is_weak=etag_is_weak,
                last_modified=last_modified,
            )
        except (KeyError, TypeError, ValueError):
            object_key = response.get("Key") if isinstance(response, dict) else None
            raise self._metadata_error(
                object_key if isinstance(object_key, str) else None
            ) from None

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
    client = dependencies.boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        region_name="auto",
        config=client_config,
    )
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
    if not isinstance(value, datetime):
        raise ValueError
    return value


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
    _, weak = parse_etag(value)
    return value, weak


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
