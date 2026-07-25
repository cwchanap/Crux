import logging
from datetime import datetime, timezone
from io import BytesIO

import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError, NoCredentialsError

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


def client_error(code: str, operation: str, status: int | None = None) -> ClientError:
    response = {"Error": {"Code": code, "Message": "private R2 detail"}}
    if status is not None:
        response["ResponseMetadata"] = {"HTTPStatusCode": status, "RequestId": "secret"}
    return ClientError(response, operation)


def assert_store_error(action, code: str, message: str, object_key: str | None = None) -> None:
    with pytest.raises(R2StoreError) as caught:
        action()
    assert caught.value.code == code
    assert str(caught.value) == message
    assert caught.value.object_key == object_key
    assert "private" not in str(caught.value)
    assert "secret" not in str(caught.value)


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


def test_factory_uses_configured_standard_retry_client_without_explicit_credentials(monkeypatch):
    import boto3

    captured = {}

    def create_client(service_name, **kwargs):
        captured["service_name"] = service_name
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(boto3, "client", create_client)
    config = R2Config(
        endpoint_url="https://account.r2.cloudflarestorage.com",
        source_endpoint_sha256="x",
        bucket="simfile-dtx",
        head_concurrency=12,
        download_concurrency=8,
        connect_timeout_seconds=7,
        read_timeout_seconds=31,
        max_attempts=6,
    )

    store = create_boto3_store(config)

    assert isinstance(store, Boto3R2Store)
    assert captured["service_name"] == "s3"
    assert captured["endpoint_url"] == "https://account.r2.cloudflarestorage.com"
    assert captured["region_name"] == "auto"
    assert set(captured) == {"service_name", "endpoint_url", "region_name", "config"}
    assert captured["config"].connect_timeout == 7
    assert captured["config"].read_timeout == 31
    assert captured["config"].tcp_keepalive is True
    assert captured["config"].retries == {"mode": "standard", "total_max_attempts": 6}
    assert captured["config"].max_pool_connections == 20


def test_adapter_preserves_more_than_one_thousand_exact_keys():
    keys = tuple(f"{index}/SET.DEF" for index in range(1001))

    class ManyObjectsPaginator:
        def paginate(self, **kwargs):
            assert kwargs == {"Bucket": "simfile-dtx"}
            return [
                {
                    "Contents": [
                        {
                            "Key": key,
                            "Size": index,
                            "ETag": f'"etag-{index}"',
                            "LastModified": datetime(2026, 7, 25, tzinfo=timezone.utc),
                        }
                        for index, key in enumerate(keys)
                    ]
                }
            ]

    class ManyObjectsClient(FakeClient):
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return ManyObjectsPaginator()

    listed = Boto3R2Store(ManyObjectsClient(), "simfile-dtx").list_objects()
    assert tuple(item.key for item in listed) == keys


def test_adapter_maps_sdk_errors_to_closed_messages_without_sdk_details():
    class ErrorClient(FakeClient):
        def head_bucket(self, **kwargs):
            raise client_error("AccessDenied", "HeadBucket", 403)

        def get_paginator(self, name):
            raise ConnectTimeoutError(endpoint_url="https://private.example")

        def head_object(self, **kwargs):
            raise client_error("InvalidAccessKeyId", "HeadObject", 403)

        def get_object(self, **kwargs):
            raise client_error("InternalError", "GetObject", 500)

    store = Boto3R2Store(ErrorClient(), "simfile-dtx")
    assert_store_error(
        store.validate_bucket,
        "bucket_inaccessible",
        "The configured R2 bucket is inaccessible.",
    )
    assert_store_error(
        store.list_objects,
        "root_list_failed",
        "R2 root listing did not complete.",
    )
    assert_store_error(
        lambda: store.head_object("42/SET.DEF"),
        "auth_failed",
        "R2 rejected the resolved credentials.",
        "42/SET.DEF",
    )
    assert_store_error(
        lambda: store.open_object("42/SET.DEF", None).__enter__(),
        "object_get_failed",
        "Object body read failed.",
        "42/SET.DEF",
    )


def test_adapter_maps_missing_credentials_and_conditional_change():
    class CredentialsClient(FakeClient):
        def head_bucket(self, **kwargs):
            raise NoCredentialsError()

    assert_store_error(
        Boto3R2Store(CredentialsClient(), "simfile-dtx").validate_bucket,
        "missing_credentials",
        "No usable R2 credentials were resolved.",
    )

    class ChangedClient(FakeClient):
        def get_object(self, **kwargs):
            raise client_error("PreconditionFailed", "GetObject", 412)

    assert_store_error(
        lambda: Boto3R2Store(ChangedClient(), "simfile-dtx")
        .open_object("42/SET.DEF", '"stale"')
        .__enter__(),
        "source_changed_during_sync",
        "Object metadata changed after inventory.",
        "42/SET.DEF",
    )


def test_adapter_rejects_malformed_sdk_metadata_with_closed_error():
    class MalformedClient(FakeClient):
        def get_paginator(self, name):
            return type(
                "MalformedPaginator",
                (),
                {"paginate": lambda self, **kwargs: [{"Contents": [{"Key": "42/SET.DEF"}]}]},
            )()

    assert_store_error(
        Boto3R2Store(MalformedClient(), "simfile-dtx").list_objects,
        "object_metadata_invalid",
        "Object metadata is invalid.",
        "42/SET.DEF",
    )


def test_adapter_keeps_the_object_key_separate_for_malformed_object_metadata():
    class MalformedHeadClient(FakeClient):
        def head_object(self, **kwargs):
            return {"ETag": 42}

    assert_store_error(
        lambda: Boto3R2Store(MalformedHeadClient(), "simfile-dtx").head_object("42/SET.DEF"),
        "object_metadata_invalid",
        "Object metadata is invalid.",
        "42/SET.DEF",
    )

    class MalformedGetClient(FakeClient):
        def get_object(self, **kwargs):
            return {"ContentLength": 5}

    assert_store_error(
        lambda: Boto3R2Store(MalformedGetClient(), "simfile-dtx")
        .open_object("42/SET.DEF", None)
        .__enter__(),
        "object_metadata_invalid",
        "Object metadata is invalid.",
        "42/SET.DEF",
    )


def test_adapter_does_not_retry_exhausted_sdk_failures():
    class ExhaustedClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def get_object(self, **kwargs):
            self.attempts += 1
            raise ConnectTimeoutError(endpoint_url="https://private.example")

    client = ExhaustedClient()
    assert_store_error(
        lambda: Boto3R2Store(client, "simfile-dtx").open_object("42/SET.DEF", None).__enter__(),
        "object_get_failed",
        "Object body read failed.",
        "42/SET.DEF",
    )
    assert client.attempts == 1


def test_adapter_redacts_body_close_failures():
    class FailingBody(BytesIO):
        def close(self):
            raise RuntimeError("private stream failure")

    class CloseFailureClient(FakeClient):
        def get_object(self, **kwargs):
            return {"Body": FailingBody(b"chart")}

    store = Boto3R2Store(CloseFailureClient(), "simfile-dtx")
    assert_store_error(
        lambda: _consume_download(store),
        "object_get_failed",
        "Object body read failed.",
        "42/SET.DEF",
    )


def test_adapter_redacts_body_read_failures():
    class FailingBody(BytesIO):
        def read(self, size=-1):
            raise ConnectTimeoutError(endpoint_url="https://private.example")

    class ReadFailureClient(FakeClient):
        def get_object(self, **kwargs):
            return {"Body": FailingBody(b"chart")}

    store = Boto3R2Store(ReadFailureClient(), "simfile-dtx")
    assert_store_error(
        lambda: _read_download(store),
        "object_get_failed",
        "Object body read failed.",
        "42/SET.DEF",
    )


def _consume_download(store: Boto3R2Store) -> None:
    with store.open_object("42/SET.DEF", None):
        pass


def _read_download(store: Boto3R2Store) -> None:
    with store.open_object("42/SET.DEF", None) as download:
        download.body.read()


def test_missing_r2_dependency_has_closed_install_hint(monkeypatch):
    import src.benchmark.r2_inventory as inventory

    def missing_dependency(_name):
        raise ImportError("private missing package detail")

    monkeypatch.setattr(inventory, "_import_optional_dependency", missing_dependency)

    assert_store_error(
        inventory.ensure_r2_dependency,
        "missing_optional_dependency",
        "R2 support is not installed. Install it with uv pip install -e '.[r2]'.",
    )
