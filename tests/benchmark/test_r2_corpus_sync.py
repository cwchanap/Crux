from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Event, Lock, Thread

import pytest

import src.benchmark.r2_corpus_sync as r2_corpus_sync
from src.benchmark.corpus_cache import CacheIndexEntry, CacheIndexStore, cache_writer_lock
from src.benchmark.corpus_manifest import ManifestPublicationError
from src.benchmark.r2_corpus_models import (
    HeadMetadata,
    ListedObject,
    SyncRequest,
    format_manifest_timestamp,
)
from src.benchmark.r2_corpus_sync import report_filename, sync_r2_corpus
from src.benchmark.r2_inventory import ObjectDownload, R2StoreError

STARTED_AT = datetime(2026, 7, 25, 1, 2, 3, 120000, tzinfo=timezone.utc)
COMPLETED_AT = STARTED_AT + timedelta(seconds=2)
RUN_ID = "12345678-1234-4234-9234-123456789abc"
SECOND_RUN_ID = "abcdefab-cdef-4abc-8def-abcdefabcdef"
ENDPOINT = "https://account.example.invalid"
DEFAULT_ENVIRON = {
    "CRUX_R2_ENDPOINT_URL": ENDPOINT,
    "CRUX_R2_BUCKET": "simfile-dtx",
    "AWS_ACCESS_KEY_ID": "test-access-key",
    "AWS_SECRET_ACCESS_KEY": "test-secret-key",
}


class ChunkedBody(BytesIO):
    def __init__(self, content: bytes, chunk_size: int) -> None:
        super().__init__(content)
        self.chunk_size = chunk_size

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > self.chunk_size:
            size = self.chunk_size
        return super().read(size)


class FakeStore:
    def __init__(
        self,
        listed: tuple[ListedObject, ...],
        heads: dict[str, HeadMetadata],
        bodies: dict[str, bytes],
        *,
        errors: dict[tuple[str, str | None], R2StoreError] | None = None,
        chunk_size: int = 2,
    ) -> None:
        self.listed = listed
        self.heads = heads
        self.bodies = bodies
        self.errors = errors or {}
        self.chunk_size = chunk_size
        self.calls: list[tuple[str, str | None, str | None]] = []
        self._calls_lock = Lock()

    def _record(self, operation: str, key: str | None = None, detail: str | None = None) -> None:
        with self._calls_lock:
            self.calls.append((operation, key, detail))

    def _raise_configured_error(self, operation: str, key: str | None = None) -> None:
        error = self.errors.get((operation, key)) or self.errors.get((operation, None))
        if error is not None:
            raise error

    def validate_bucket(self) -> None:
        self._record("validate")
        self._raise_configured_error("validate")

    def list_objects(self) -> tuple[ListedObject, ...]:
        self._record("list")
        self._raise_configured_error("list")
        return self.listed

    def head_object(self, key: str) -> HeadMetadata:
        self._record("head", key)
        self._raise_configured_error("head", key)
        return self.heads[key]

    @contextmanager
    def open_object(self, key: str, if_match: str | None) -> Iterator[ObjectDownload]:
        self._record("get", key, if_match)
        self._raise_configured_error("get", key)
        metadata = self.heads[key]
        yield ObjectDownload(
            body=ChunkedBody(self.bodies[key], self.chunk_size),
            size=metadata.size,
            etag=metadata.etag,
            etag_is_weak=metadata.etag_is_weak,
            last_modified=metadata.last_modified,
        )


def listed_object(key: str, body: bytes, *, etag: str | None = None) -> ListedObject:
    return ListedObject(
        key=key,
        size=len(body),
        etag=etag or f"etag-{key}",
        etag_is_weak=False,
        last_modified=STARTED_AT,
    )


def head_metadata(item: ListedObject) -> HeadMetadata:
    return HeadMetadata(
        size=item.size,
        etag=item.etag,
        etag_is_weak=item.etag_is_weak,
        last_modified=item.last_modified,
        content_type="text/plain",
    )


def store_from_bodies(
    bodies: dict[str, bytes],
    *,
    errors: dict[tuple[str, str | None], R2StoreError] | None = None,
) -> FakeStore:
    listed = tuple(listed_object(key, body) for key, body in bodies.items())
    return FakeStore(
        listed=listed,
        heads={item.key: head_metadata(item) for item in listed},
        bodies=bodies,
        errors=errors,
    )


def complete_store() -> FakeStore:
    return store_from_bodies(
        {
            "2/chart.dtx": b"chart",
            "2/cover.png": b"png",
            "10/SET.DEF": b"set",
        }
    )


def store_with_one_get_failure() -> FakeStore:
    return store_from_bodies(
        {
            "2/chart.dtx": b"chart",
            "10/SET.DEF": b"set",
        },
        errors={
            ("get", "2/chart.dtx"): R2StoreError(
                "object_get_failed",
                "SECRET raw SDK message https://signed.invalid/?token=SECRET",
                "2/chart.dtx",
            )
        },
    )


def invoke_sync(
    tmp_path: Path,
    store: FakeStore,
    *,
    dry_run: bool = False,
    endpoint_url: str = ENDPOINT,
    provenance_file: Path | None = None,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    include_simfile_ids: frozenset[int] = frozenset(),
    exclude_simfile_ids: frozenset[int] = frozenset(),
    run_id: str = RUN_ID,
    monotonic=lambda: 0.0,
):
    events = []
    times = iter((STARTED_AT, COMPLETED_AT))
    request = SyncRequest(
        output_dir=output_dir or tmp_path / "output",
        cache_dir=cache_dir or tmp_path / "cache",
        provenance_file=provenance_file,
        include_simfile_ids=include_simfile_ids,
        exclude_simfile_ids=exclude_simfile_ids,
        dry_run=dry_run,
    )
    environ = {**DEFAULT_ENVIRON, "CRUX_R2_ENDPOINT_URL": endpoint_url}
    outcome = sync_r2_corpus(
        request,
        environ=environ,
        dependency_check=lambda: None,
        store_factory=lambda _config: store,
        clock=lambda: next(times, COMPLETED_AT),
        monotonic=monotonic,
        run_id_factory=lambda: run_id,
        progress=events.append,
    )
    return outcome, events


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_latest(tmp_path: Path) -> dict[str, object]:
    return read_json(tmp_path / "output" / "latest.json")


def read_report(path: Path | None) -> dict[str, object]:
    assert path is not None
    return read_json(path)


def test_complete_real_run_publishes_cache_manifest_report_and_pointers(tmp_path):
    outcome, events = invoke_sync(tmp_path, complete_store())

    assert outcome.exit_code == 0
    assert outcome.overall_status == "complete"
    assert outcome.manifest is not None
    assert outcome.manifest.corpus_version.startswith("sha256:")
    assert outcome.manifest.path.is_file()
    assert outcome.report_path is not None
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
    path.write_text("{", encoding="utf-8")
    store = complete_store()

    outcome, _ = invoke_sync(tmp_path, store, provenance_file=path)

    assert outcome.exit_code == 2
    assert store.calls == []
    assert read_report(outcome.report_path)["errors"][0]["code"] == "provenance_invalid"


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
    assert outcome.report_path is not None
    assert outcome.report_path.is_file()
    assert (tmp_path / "output" / "latest-report.json").is_file()
    assert read_report(outcome.report_path)["artifacts"] == {
        "corpus_version": None,
        "manifest_path": None,
        "manifest_sha256": None,
    }


def test_missing_optional_dependency_is_fatal_before_configuration_or_network(tmp_path):
    store = complete_store()
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)

    outcome = sync_r2_corpus(
        request,
        environ={},
        dependency_check=lambda: (_ for _ in ()).throw(
            R2StoreError("missing_optional_dependency", "unsafe dependency detail")
        ),
        store_factory=lambda _config: store,
        clock=lambda: STARTED_AT,
        run_id_factory=lambda: RUN_ID,
    )

    assert outcome.exit_code == 2
    assert store.calls == []
    report = read_report(outcome.report_path)
    assert report["source_endpoint_sha256"] is None
    assert report["source_bucket"] is None
    assert report["errors"][0]["code"] == "missing_optional_dependency"
    assert "unsafe dependency detail" not in json.dumps(report)


def test_invalid_bucket_configuration_fails_before_network(tmp_path):
    store = complete_store()
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)

    outcome = sync_r2_corpus(
        request,
        environ={**DEFAULT_ENVIRON, "CRUX_R2_BUCKET": "invalid/bucket"},
        dependency_check=lambda: None,
        store_factory=lambda _config: store,
        clock=lambda: STARTED_AT,
        run_id_factory=lambda: RUN_ID,
    )

    assert outcome.exit_code == 2
    assert store.calls == []
    assert read_report(outcome.report_path)["errors"][0]["code"] == "invalid_config"


def test_missing_credentials_from_store_factory_preserves_allowlisted_code(tmp_path):
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)

    outcome = sync_r2_corpus(
        request,
        environ=DEFAULT_ENVIRON,
        dependency_check=lambda: None,
        store_factory=lambda _config: (_ for _ in ()).throw(
            R2StoreError("missing_credentials", "unsafe credential provider detail")
        ),
        clock=lambda: STARTED_AT,
        run_id_factory=lambda: RUN_ID,
    )

    report = read_report(outcome.report_path)
    assert outcome.exit_code == 2
    assert report["errors"][0]["code"] == "missing_credentials"
    assert "unsafe credential provider detail" not in json.dumps(report)


def test_head_failures_are_aggregated_into_a_partial_manifest(tmp_path):
    store = complete_store()
    store.errors[("head", "2/chart.dtx")] = R2StoreError(
        "object_head_failed", "unsafe HEAD detail", "2/chart.dtx"
    )

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.overall_status == "partial"
    assert outcome.exit_code == 1
    report = read_report(outcome.report_path)
    row = next(row for row in report["simfiles"] if row["simfile_id"] == 2)
    assert [error["code"] for error in row["errors"]] == ["object_head_failed"]
    assert "unsafe HEAD detail" not in json.dumps(report)


def test_head_failure_is_not_counted_as_a_failed_or_planned_download(tmp_path):
    store = complete_store()
    store.errors[("head", "2/chart.dtx")] = R2StoreError(
        "object_head_failed", "unsafe HEAD detail", "2/chart.dtx"
    )

    outcome, _ = invoke_sync(tmp_path, store)

    report = read_report(outcome.report_path)
    assert report["counters"]["downloads_planned"] == 1
    assert report["counters"]["downloads_completed"] == 1
    assert report["counters"]["downloads_failed"] == 0
    assert report["counters"]["download_bytes_planned"] == 3


def test_writer_lock_conflict_fails_before_store_creation_or_network(tmp_path):
    store = complete_store()

    with cache_writer_lock(tmp_path / "cache"):
        outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert store.calls == []
    assert read_report(outcome.report_path)["errors"][0]["code"] == "cache_locked"


def test_empty_include_and_ambiguous_prefixes_publish_partial_manifest(tmp_path):
    store = store_from_bodies(
        {
            "1/chart.dtx": b"one",
            "01/chart.dtx": b"alias",
        }
    )

    outcome, _ = invoke_sync(
        tmp_path,
        store,
        include_simfile_ids=frozenset({1, 3}),
    )

    assert outcome.exit_code == 1
    assert outcome.manifest is not None
    rows = [
        json.loads(line) for line in outcome.manifest.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["simfile_id"], row["sync_status"]) for row in rows] == [(3, "empty")]
    report = read_report(outcome.report_path)
    assert report["ambiguous_prefixes"] == {"1": ["01/", "1/"]}


def test_report_write_failure_returns_sanitized_fallback_outcome(tmp_path, monkeypatch):
    store = complete_store()

    def fail_report(*_args, **_kwargs):
        raise OSError("SECRET report path detail")

    monkeypatch.setattr(r2_corpus_sync, "_publish_report_file", fail_report)

    outcome, events = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.overall_status == "failed"
    assert outcome.manifest is None
    assert outcome.report_path is None
    assert [(error.code, error.message) for error in outcome.errors] == [
        ("artifact_write_failed", "A required synchronization report write failed.")
    ]
    assert events[-1].phase == "failed"
    assert "SECRET" not in repr(outcome)
    assert not (tmp_path / "output" / "latest.json").exists()


def test_manifest_publication_failure_attempts_a_fatal_report(tmp_path, monkeypatch):
    store = complete_store()

    def fail_manifest(*_args, **_kwargs):
        raise ManifestPublicationError(
            r2_corpus_sync.SyncError(
                "artifact",
                "artifact_write_failed",
                "A required manifest artifact write failed.",
            )
        )

    monkeypatch.setattr(r2_corpus_sync, "publish_manifest", fail_manifest)

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.manifest is None
    assert outcome.report_path is not None
    assert read_report(outcome.report_path)["errors"][0]["code"] == "artifact_write_failed"
    assert not (tmp_path / "output" / "latest.json").exists()


def test_report_failure_after_immutable_install_leaves_manifest_unreferenced(tmp_path, monkeypatch):
    store = complete_store()
    installed_paths: list[Path] = []
    real_publish_manifest = r2_corpus_sync.publish_manifest

    def record_manifest(*args, **kwargs):
        published = real_publish_manifest(*args, **kwargs)
        installed_paths.append(published.path)
        return published

    monkeypatch.setattr(r2_corpus_sync, "publish_manifest", record_manifest)
    monkeypatch.setattr(
        r2_corpus_sync,
        "_publish_report_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unsafe report detail")),
    )

    outcome, _ = invoke_sync(tmp_path, store)

    assert installed_paths and installed_paths[0].is_file()
    assert outcome.exit_code == 2
    assert outcome.manifest is None
    assert not (tmp_path / "output" / "latest.json").exists()


def test_latest_report_failure_restores_previous_latest_manifest_pointer(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    previous_latest = b'{"previous":"manifest"}\n'
    (output_dir / "latest.json").write_bytes(previous_latest)
    store = complete_store()

    monkeypatch.setattr(
        r2_corpus_sync,
        "_publish_latest_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unsafe pointer detail")),
    )

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.manifest is None
    assert outcome.report_path is None
    assert (output_dir / "latest.json").read_bytes() == previous_latest


def test_latest_manifest_failure_restores_previous_pointer_before_fatal_report(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    previous_latest = b'{"previous":"manifest"}\n'
    (output_dir / "latest.json").write_bytes(previous_latest)
    store = complete_store()

    def fail_after_replacement(output, *_args, **_kwargs):
        (output / "latest.json").write_text('{"failed":"replacement"}\n', encoding="utf-8")
        raise ManifestPublicationError(
            r2_corpus_sync.SyncError(
                "artifact",
                "artifact_write_failed",
                "A required manifest artifact write failed.",
            )
        )

    monkeypatch.setattr(r2_corpus_sync, "publish_latest_manifest", fail_after_replacement)

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.manifest is None
    assert outcome.report_path is not None
    assert (output_dir / "latest.json").read_bytes() == previous_latest
    assert read_report(outcome.report_path)["errors"][0]["code"] == "artifact_write_failed"


def test_report_names_include_fixed_microseconds_and_canonical_uuid4():
    first = report_filename(STARTED_AT, RUN_ID)
    second = report_filename(STARTED_AT, SECOND_RUN_ID)
    later = report_filename(STARTED_AT + timedelta(microseconds=1), RUN_ID)

    assert first == f"20260725T010203.120000Z-{RUN_ID}.json"
    assert first != second
    assert first < later


def test_report_counters_and_cache_miss_reasons_are_exact(tmp_path):
    outcome, _ = invoke_sync(tmp_path, complete_store())

    report = read_report(outcome.report_path)
    assert report["counters"] == {
        "cache_hits": 0,
        "download_bytes_completed": 8,
        "download_bytes_planned": 8,
        "downloads_completed": 2,
        "downloads_failed": 0,
        "downloads_planned": 2,
        "objects_listed": 3,
        "objects_selected": 2,
        "simfiles_discovered": 2,
        "simfiles_empty": 0,
        "simfiles_excluded_by_filter": 0,
        "simfiles_included": 2,
    }
    assert report["cache_misses_by_reason"] == {
        "missing": 0,
        "remote_changed": 2,
        "sha256_mismatch": 0,
        "size_mismatch": 0,
    }


@pytest.mark.parametrize(
    ("local_state", "expected_reason"),
    [
        ("missing", "missing"),
        ("size_mismatch", "size_mismatch"),
        ("sha256_mismatch", "sha256_mismatch"),
    ],
)
def test_report_counts_each_local_cache_miss_reason(tmp_path, local_state, expected_reason):
    body = b"chart"
    store = store_from_bodies({"2/chart.dtx": body})
    remote = store.listed[0]
    digest = sha256(body).hexdigest()
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "sha256" / digest[:2] / digest
    if local_state != "missing":
        cache_path.parent.mkdir(parents=True)
        cache_path.write_bytes(b"xx" if local_state == "size_mismatch" else b"xxxxx")
    index = CacheIndexStore.load(cache_dir)
    index.checkpoint(
        CacheIndexEntry(
            source_endpoint_sha256=sha256(ENDPOINT.encode("ascii")).hexdigest(),
            bucket="simfile-dtx",
            key=remote.key,
            etag=remote.etag,
            etag_is_weak=False,
            size=len(body),
            last_modified=format_manifest_timestamp(remote.last_modified),
            sha256=digest,
            cache_path=f"sha256/{digest[:2]}/{digest}",
        )
    )

    outcome, _ = invoke_sync(tmp_path, store, dry_run=True)

    report = read_report(outcome.report_path)
    assert report["cache_misses_by_reason"][expected_reason] == 1
    assert report["counters"]["downloads_planned"] == 1
    assert report["counters"]["download_bytes_planned"] == len(body)


def test_secrets_urls_headers_and_raw_adapter_messages_are_not_serialized(tmp_path):
    secret = "DO-NOT-LEAK-9c79"
    store = complete_store()
    store.errors[("list", None)] = R2StoreError(
        "root_list_failed",
        f"{secret} Authorization: bearer signed=https://evil.invalid/?token={secret}",
    )
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)
    events = []

    outcome = sync_r2_corpus(
        request,
        environ={
            **DEFAULT_ENVIRON,
            "CRUX_R2_ENDPOINT_URL": f"https://{secret.lower()}.example.invalid",
            "AWS_ACCESS_KEY_ID": secret,
            "AWS_SECRET_ACCESS_KEY": secret,
            "AWS_SESSION_TOKEN": secret,
        },
        dependency_check=lambda: None,
        store_factory=lambda _config: store,
        clock=lambda: STARTED_AT,
        run_id_factory=lambda: RUN_ID,
        progress=events.append,
    )

    serialized = (
        repr(outcome)
        + json.dumps(read_report(outcome.report_path))
        + repr(events)
        + "".join(
            path.read_text(encoding="utf-8")
            for path in (tmp_path / "output").rglob("*")
            if path.is_file()
        )
    )
    assert secret not in serialized
    assert "Authorization:" not in serialized
    assert "https://evil.invalid" not in serialized


def test_progress_emits_phase_entries_and_count_throttled_metadata_updates(tmp_path):
    store = store_from_bodies({f"2/file-{index:03}.bin": b"x" for index in range(201)})

    outcome, events = invoke_sync(tmp_path, store, dry_run=True)

    assert outcome.exit_code == 0
    phases = [event.phase for event in events]
    assert phases[0] == "configuration"
    assert phases[-1] == "dry_run_complete"
    for phase in ("inventory", "metadata", "cache", "report"):
        assert phase in phases
    metadata_completed = [
        event.completed for event in events if event.phase == "metadata" and event.completed > 0
    ]
    assert metadata_completed == [100, 200, 201]
    assert all("file-" not in event.message and "https://" not in event.message for event in events)


def test_progress_emits_item_update_after_five_monotonic_seconds(tmp_path):
    store = store_from_bodies({f"2/file-{index}.bin": b"x" for index in range(3)})
    values = iter((0.0, 1.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0))

    outcome, events = invoke_sync(
        tmp_path,
        store,
        dry_run=True,
        monotonic=lambda: next(values, 7.0),
    )

    assert outcome.exit_code == 0
    metadata_completed = [
        event.completed for event in events if event.phase == "metadata" and event.completed > 0
    ]
    assert metadata_completed == [2, 3]


def test_real_writer_lock_is_held_through_latest_report_publication(tmp_path, monkeypatch):
    store = complete_store()
    observed_conflict = False
    real_publish_latest_report = r2_corpus_sync._publish_latest_report

    def probe_lock(*args, **kwargs):
        nonlocal observed_conflict
        try:
            with cache_writer_lock(tmp_path / "cache"):
                pass
        except RuntimeError as error:
            observed_conflict = error.args == ("cache_locked",)
        return real_publish_latest_report(*args, **kwargs)

    monkeypatch.setattr(r2_corpus_sync, "_publish_latest_report", probe_lock)

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 0
    assert observed_conflict


def test_fixed_input_rerun_reuses_byte_identical_immutable_manifest(tmp_path):
    first, _ = invoke_sync(tmp_path, complete_store(), run_id=RUN_ID)
    second_store = complete_store()
    second, _ = invoke_sync(tmp_path, second_store, run_id=SECOND_RUN_ID)

    assert first.manifest is not None
    assert second.manifest is not None
    assert first.manifest.path == second.manifest.path
    assert first.manifest.path.read_bytes() == second.manifest.path.read_bytes()
    assert not any(call[0] == "get" for call in second_store.calls)


def test_invalid_cache_index_fails_before_store_creation_or_network(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "index-v1.json").write_text("{", encoding="utf-8")
    store = complete_store()

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.manifest is None
    assert store.calls == []
    assert read_report(outcome.report_path)["errors"][0]["code"] == "cache_index_invalid"


def test_full_inventory_and_head_failure_counts_remain_complete(tmp_path):
    bodies = {f"2/file-{index:04}.bin": b"x" for index in range(1001)}
    store = store_from_bodies(bodies)
    store.errors[("head", "2/file-1000.bin")] = R2StoreError(
        "object_head_failed", "unsafe page-tail failure", "2/file-1000.bin"
    )

    outcome, _ = invoke_sync(tmp_path, store, dry_run=True)

    report = read_report(outcome.report_path)
    assert outcome.overall_status == "dry_run_partial"
    assert report["counters"]["objects_listed"] == 1001
    assert sum(call[0] == "head" for call in store.calls) == 1001
    row = report["simfiles"][0]
    assert [error["code"] for error in row["errors"]] == ["object_head_failed"]


def test_unexpected_failure_is_sanitized_as_internal_error(tmp_path):
    secret = "SECRET-runtime-class-and-detail"
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)

    outcome = sync_r2_corpus(
        request,
        environ=DEFAULT_ENVIRON,
        dependency_check=lambda: None,
        store_factory=lambda _config: (_ for _ in ()).throw(RuntimeError(secret)),
        clock=lambda: STARTED_AT,
        run_id_factory=lambda: RUN_ID,
    )

    report = read_report(outcome.report_path)
    assert outcome.exit_code == 2
    assert report["errors"][0]["code"] == "internal_error"
    assert secret not in repr(outcome)
    assert secret not in json.dumps(report)


@pytest.mark.parametrize(
    "run_id",
    [
        "not-a-uuid",
        "12345678-1234-1234-9234-123456789abc",
        RUN_ID.upper(),
        RUN_ID.replace("-", ""),
    ],
)
def test_report_filename_rejects_noncanonical_or_non_uuid4_ids(run_id):
    with pytest.raises(ValueError, match="canonical UUID4"):
        report_filename(STARTED_AT, run_id)


def test_report_and_latest_report_hashes_match_durable_canonical_bytes(tmp_path):
    outcome, _ = invoke_sync(tmp_path, complete_store())

    assert outcome.report_path is not None
    report_content = outcome.report_path.read_bytes()
    pointer = read_json(tmp_path / "output" / "latest-report.json")
    assert report_content.endswith(b"\n")
    assert pointer["report_path"] == f"reports/{outcome.report_path.name}"
    assert pointer["report_sha256"] == sha256(report_content).hexdigest()
    assert not list((tmp_path / "output").rglob("*.tmp"))


def test_report_publication_fsyncs_file_before_replace_and_directory_after(tmp_path, monkeypatch):
    events: list[tuple[str, str]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def record_fsync(descriptor):
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    def record_replace(source, destination):
        events.append(("replace", Path(destination).name))
        real_replace(source, destination)

    monkeypatch.setattr(r2_corpus_sync.os, "fsync", record_fsync)
    monkeypatch.setattr(r2_corpus_sync.os, "replace", record_replace)

    report_path, _, _ = r2_corpus_sync._publish_report_file(
        tmp_path / "output",
        STARTED_AT,
        RUN_ID,
        {"schema_version": "test"},
    )

    replace_index = events.index(("replace", report_path.name))
    assert events[replace_index - 1] == ("fsync", "file")
    assert events[replace_index + 1] == ("fsync", "directory")


class BlockingGetStore(FakeStore):
    def __init__(self, store: FakeStore, entered: Event, release: Event) -> None:
        super().__init__(store.listed, store.heads, store.bodies, errors=store.errors)
        self.entered = entered
        self.release = release

    @contextmanager
    def open_object(self, key: str, if_match: str | None) -> Iterator[ObjectDownload]:
        self.entered.set()
        assert self.release.wait(timeout=5)
        with super().open_object(key, if_match) as download:
            yield download


class BlockingListStore(FakeStore):
    def __init__(self, store: FakeStore, entered: Event, release: Event) -> None:
        super().__init__(store.listed, store.heads, store.bodies, errors=store.errors)
        self.entered = entered
        self.release = release

    def list_objects(self) -> tuple[ListedObject, ...]:
        self.entered.set()
        assert self.release.wait(timeout=5)
        return super().list_objects()


def threaded_sync(
    *,
    output_dir: Path,
    cache_dir: Path,
    store: FakeStore,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    dry_run: bool = False,
):
    times = iter((started_at, completed_at))
    return sync_r2_corpus(
        SyncRequest(output_dir, cache_dir, None, dry_run=dry_run),
        environ=DEFAULT_ENVIRON,
        dependency_check=lambda: None,
        store_factory=lambda _config: store,
        clock=lambda: next(times, completed_at),
        run_id_factory=lambda: run_id,
    )


def test_reverse_start_order_real_writers_leave_latest_on_last_completion(tmp_path):
    output_dir = tmp_path / "output"
    first_entered = Event()
    release_first = Event()
    first_store = BlockingGetStore(
        store_from_bodies({"2/chart.dtx": b"first"}),
        first_entered,
        release_first,
    )
    second_store = store_from_bodies({"2/chart.dtx": b"second"})
    outcomes = {}

    first_thread = Thread(
        target=lambda: outcomes.setdefault(
            "first",
            threaded_sync(
                output_dir=output_dir,
                cache_dir=tmp_path / "cache-first",
                store=first_store,
                run_id=RUN_ID,
                started_at=STARTED_AT,
                completed_at=STARTED_AT + timedelta(seconds=4),
            ),
        )
    )
    first_thread.start()
    assert first_entered.wait(timeout=5)

    outcomes["second"] = threaded_sync(
        output_dir=output_dir,
        cache_dir=tmp_path / "cache-second",
        store=second_store,
        run_id=SECOND_RUN_ID,
        started_at=STARTED_AT + timedelta(seconds=1),
        completed_at=STARTED_AT + timedelta(seconds=2),
    )
    release_first.set()
    first_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert outcomes["first"].manifest is not None
    assert outcomes["second"].manifest is not None
    latest = read_json(output_dir / "latest.json")
    assert latest["manifest_sha256"] == outcomes["first"].manifest.manifest_sha256
    latest_report = read_json(output_dir / "latest-report.json")
    assert latest_report["report_path"].endswith(f"{RUN_ID}.json")


def test_dry_run_finishing_after_real_run_moves_only_latest_report(tmp_path):
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    dry_entered = Event()
    release_dry = Event()
    dry_store = BlockingListStore(complete_store(), dry_entered, release_dry)
    outcomes = {}

    dry_thread = Thread(
        target=lambda: outcomes.setdefault(
            "dry",
            threaded_sync(
                output_dir=output_dir,
                cache_dir=cache_dir,
                store=dry_store,
                run_id=RUN_ID,
                started_at=STARTED_AT,
                completed_at=STARTED_AT + timedelta(seconds=4),
                dry_run=True,
            ),
        )
    )
    dry_thread.start()
    assert dry_entered.wait(timeout=5)

    outcomes["real"] = threaded_sync(
        output_dir=output_dir,
        cache_dir=cache_dir,
        store=complete_store(),
        run_id=SECOND_RUN_ID,
        started_at=STARTED_AT + timedelta(seconds=1),
        completed_at=STARTED_AT + timedelta(seconds=2),
    )
    release_dry.set()
    dry_thread.join(timeout=5)

    assert not dry_thread.is_alive()
    assert outcomes["real"].manifest is not None
    assert outcomes["dry"].manifest is None
    latest = read_json(output_dir / "latest.json")
    assert latest["manifest_sha256"] == outcomes["real"].manifest.manifest_sha256
    latest_report = read_json(output_dir / "latest-report.json")
    assert latest_report["report_path"].endswith(f"{RUN_ID}.json")
    assert latest_report["overall_status"] == "dry_run_complete"
