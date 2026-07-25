import logging
from datetime import datetime, timezone
from io import BytesIO
from threading import Condition, Event, Thread

import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError, NoCredentialsError

from src.benchmark.r2_corpus_models import (
    MAX_SIMFILE_ID,
    HeadMetadata,
    ListedObject,
    R2Config,
)
from src.benchmark.r2_inventory import (
    Boto3R2Store,
    R2StoreError,
    build_inventory,
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
    assert (listed[0].etag, listed[0].etag_is_weak) == ("etag-2", False)
    head = store.head_object(listed[0].key)
    assert (head.etag, head.etag_is_weak, head.content_type) == ("etag-2", False, "text/plain")

    with store.open_object(listed[0].key, if_match=listed[0].etag) as response:
        assert response.body.read() == b"chart"
        assert (response.etag, response.etag_is_weak) == ("etag-2", False)
    with store.open_object(listed[0].key, if_match=None):
        pass

    assert store.client.get_calls[0]["IfMatch"] == '"etag-2"'
    assert "IfMatch" not in store.client.get_calls[1]


def test_adapter_normalizes_weak_sdk_etags_at_every_metadata_boundary():
    class WeakEtagsPaginator:
        def paginate(self, **kwargs):
            assert kwargs == {"Bucket": "simfile-dtx"}
            return [
                {
                    "Contents": [
                        {
                            "Key": "42/SET.DEF",
                            "Size": 5,
                            "ETag": 'W/"listed"',
                            "LastModified": datetime(2026, 7, 25, tzinfo=timezone.utc),
                        }
                    ]
                }
            ]

    class WeakEtagsClient(FakeClient):
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return WeakEtagsPaginator()

        def head_object(self, **kwargs):
            return {"ETag": 'W/"head"'}

        def get_object(self, **kwargs):
            self.get_calls.append(kwargs)
            return {"Body": BytesIO(b"chart"), "ETag": 'W/"download"'}

    store = Boto3R2Store(WeakEtagsClient(), "simfile-dtx")
    listed = store.list_objects()[0]
    head = store.head_object(listed.key)
    with store.open_object(listed.key, if_match="download") as download:
        assert (download.etag, download.etag_is_weak) == ("download", True)

    assert (listed.etag, listed.etag_is_weak) == ("listed", True)
    assert (head.etag, head.etag_is_weak) == ("head", True)
    assert store.client.get_calls[0]["IfMatch"] == '"download"'


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
        objects: list[ListedObject],
        heads: dict[str, HeadMetadata | R2StoreError] | None = None,
    ):
        self.listed = tuple(objects)
        self.heads = heads or {}
        self.list_calls = 0
        self.head_calls: list[str] = []

    def list_objects(self) -> tuple[ListedObject, ...]:
        self.list_calls += 1
        return self.listed

    def head_object(self, key: str) -> HeadMetadata:
        self.head_calls.append(key)
        result = self.heads.get(key, HeadMetadata(None, None, None, None, None))
        if isinstance(result, R2StoreError):
            raise result
        return result


def test_inventory_discovers_before_filtering_and_preserves_exact_keys():
    store = FakeStore(
        objects=[
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
    store = FakeStore(objects=[listed("1/chart.dtx"), listed("01/chart.dtx")])

    result = build_inventory(store, frozenset({1}), frozenset(), 2)

    assert result.simfiles == ()
    assert result.ambiguous_prefixes == {1: ("01/", "1/")}
    assert result.root_errors[0].code == "ambiguous_simfile_prefix"


def test_nested_marker_only_prefix_is_empty_and_retains_markers():
    store = FakeStore(objects=[listed("42/", size=0), listed("42/assets/", size=0)])

    result = build_inventory(store, frozenset(), frozenset(), 2)

    row = result.simfiles[0]
    assert row.sync_status == "empty"
    assert [item.key for item in row.objects] == ["42/", "42/assets/"]
    assert all(item.cache_status == "not_selected" for item in row.objects)
    assert row.sync_errors[0].code == "empty_prefix"


def test_explicit_missing_id_creates_empty_row_with_no_objects():
    result = build_inventory(FakeStore(objects=[]), frozenset({42}), frozenset(), 2)

    assert result.simfiles[0].simfile_id == 42
    assert result.simfiles[0].objects == ()
    assert result.simfiles[0].sync_errors[0].code == "empty_prefix"


def test_exclude_wins_when_id_is_also_included():
    store = FakeStore(objects=[listed("42/SET.DEF")])

    result = build_inventory(store, frozenset({42}), frozenset({42}), 2)

    assert result.simfiles == ()
    assert result.simfiles_discovered == 1
    assert result.simfiles_excluded_by_filter == 1
    assert store.head_calls == []


def test_boundary_ids_are_valid_and_out_of_range_root_is_malformed():
    store = FakeStore(
        objects=[
            listed("0/SET.DEF"),
            listed(f"{MAX_SIMFILE_ID}/SET.DEF"),
            listed(f"{MAX_SIMFILE_ID + 1}/SET.DEF"),
        ]
    )

    result = build_inventory(store, frozenset(), frozenset(), 2)

    assert [row.simfile_id for row in result.simfiles] == [0, MAX_SIMFILE_ID]
    assert result.malformed_root_keys == (f"{MAX_SIMFILE_ID + 1}/SET.DEF",)


def test_zero_byte_non_marker_is_content_and_complete():
    result = build_inventory(
        FakeStore(objects=[listed("42/empty.dtx", size=0)]), frozenset(), frozenset(), 2
    )

    assert result.simfiles[0].sync_status == "complete"
    assert result.simfiles[0].sync_errors == ()


def test_head_failure_retains_listing_identity_and_only_its_simfile_is_partial():
    failed_key = "42/SET.DEF"
    store = FakeStore(
        objects=[listed(failed_key), listed("43/SET.DEF")],
        heads={failed_key: R2StoreError("object_head_failed", "closed", failed_key)},
    )

    result = build_inventory(store, frozenset(), frozenset(), 2)

    failed, complete = result.simfiles
    assert failed.sync_status == "partial"
    assert failed.objects[0].size == 5
    assert failed.objects[0].etag == "listed-etag"
    assert failed.objects[0].content_type is None
    assert failed.objects[0].errors[0].code == "object_head_failed"
    assert complete.sync_status == "complete"


def test_invalid_head_metadata_retains_listing_identity_and_records_error():
    key = "42/SET.DEF"
    store = FakeStore(
        objects=[listed(key)],
        heads={key: R2StoreError("object_metadata_invalid", "closed", key)},
    )

    result = build_inventory(store, frozenset(), frozenset(), 2)

    obj = result.simfiles[0].objects[0]
    assert obj.size == 5
    assert obj.etag == "listed-etag"
    assert obj.errors[0].code == "object_metadata_invalid"
    assert result.simfiles[0].sync_status == "partial"


def test_head_metadata_overrides_listing_identity_without_checksum_fields():
    key = "42/SET.DEF"
    updated = datetime(2026, 7, 26, tzinfo=timezone.utc)
    store = FakeStore(
        objects=[listed(key)],
        heads={key: HeadMetadata(7, "head-etag", True, updated, "text/plain")},
    )

    result = build_inventory(store, frozenset(), frozenset(), 2)

    obj = result.simfiles[0].objects[0]
    assert (obj.size, obj.etag, obj.etag_is_weak, obj.last_modified, obj.content_type) == (
        7,
        "head-etag",
        True,
        updated,
        "text/plain",
    )
    assert not {"checksum_algorithm", "checksum_type", "checksum_value"} & set(
        obj.__dataclass_fields__
    )


def test_marker_only_prefix_with_head_failure_is_partial_and_keeps_both_errors():
    key = "42/"
    result = build_inventory(
        FakeStore(
            objects=[listed(key, size=0)],
            heads={key: R2StoreError("object_head_failed", "closed", key)},
        ),
        frozenset(),
        frozenset(),
        2,
    )

    row = result.simfiles[0]
    assert row.sync_status == "partial"
    assert [error.code for error in row.sync_errors] == ["object_head_failed", "empty_prefix"]


def test_inventory_preserves_encoded_and_unicode_keys_exactly():
    keys = [
        "42/a space.ogg",
        "42/a+plus.ogg",
        "42/a%2Fencoded.ogg",
        "42/a#fragment.ogg",
        "42/a?query.ogg",
        "42/曲.ogg",
    ]

    result = build_inventory(
        FakeStore(objects=[listed(key) for key in reversed(keys)]), frozenset(), frozenset(), 2
    )

    assert [obj.key for obj in result.simfiles[0].objects] == sorted(keys)


def test_inventory_orders_simfiles_prefixes_and_objects_deterministically():
    store = FakeStore(
        objects=[
            listed("10/z"),
            listed("2/z"),
            listed("2/a"),
            listed("0/z"),
        ]
    )

    result = build_inventory(store, frozenset(), frozenset(), 2)

    assert [(row.simfile_id, row.object_prefix) for row in result.simfiles] == [
        (0, "0/"),
        (2, "2/"),
        (10, "10/"),
    ]
    assert [obj.key for obj in result.simfiles[1].objects] == ["2/a", "2/z"]


def test_head_requests_respect_the_configured_concurrency_bound():
    class BlockingStore(FakeStore):
        def __init__(self):
            super().__init__([listed(f"42/{index}.dtx") for index in range(4)])
            self.condition = Condition()
            self.started = Event()
            self.release = False
            self.active = 0
            self.max_active = 0

        def head_object(self, key: str) -> HeadMetadata:
            with self.condition:
                self.head_calls.append(key)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                if self.active == 2:
                    self.started.set()
                self.condition.wait_for(lambda: self.release, timeout=5)
                self.active -= 1
            return HeadMetadata(None, None, None, None, None)

    store = BlockingStore()
    result: list[object] = []
    thread = Thread(
        target=lambda: result.append(build_inventory(store, frozenset(), frozenset(), 2)),
        daemon=True,
    )
    thread.start()
    assert store.started.wait(timeout=2)
    with store.condition:
        assert store.max_active == 2
        store.release = True
        store.condition.notify_all()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result
    assert store.max_active <= 2
