from __future__ import annotations

import json
import multiprocessing
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Event, Lock, Thread

import pytest

import src.benchmark.durability as durability
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


class ManualMonotonic:
    def __init__(self) -> None:
        self.value = 0.0
        self._lock = Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.value

    def set(self, value: float) -> None:
        with self._lock:
            self.value = value


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
    store_factory=None,
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
        store_factory=store_factory or (lambda _config: store),
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


def assert_only_failed_attempt_report(output_dir: Path, report_path: Path | None) -> None:
    assert report_path is not None
    reports = sorted((output_dir / "reports").glob("*.json"))
    assert reports == [report_path]
    report = read_json(report_path)
    assert report["overall_status"] == "failed"
    assert report["exit_code"] == 2
    assert report["artifacts"] == {
        "corpus_version": None,
        "manifest_path": None,
        "manifest_sha256": None,
    }
    assert [error["code"] for error in report["errors"]] == ["artifact_write_failed"]


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


@pytest.mark.parametrize("failure", ["surrogate", "missing"])
def test_invalid_provenance_fails_before_store_creation_or_cache_mutation(tmp_path, failure):
    path = tmp_path / "SECRET-provenance.json"
    if failure == "surrogate":
        path.write_text(
            '{"schema_version":"crux.corpus-provenance/v1","simfiles":{"42":'
            '{"source_origin":"\\ud800"}}}',
            encoding="utf-8",
        )
    store = complete_store()
    factory_calls = 0

    def store_factory(_config):
        nonlocal factory_calls
        factory_calls += 1
        return store

    outcome, _ = invoke_sync(
        tmp_path,
        store,
        provenance_file=path,
        store_factory=store_factory,
    )

    assert outcome.exit_code == 2
    assert factory_calls == 0
    assert store.calls == []
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "output" / "manifests").exists()
    assert not (tmp_path / "output" / "latest.json").exists()
    report = read_report(outcome.report_path)
    assert report["errors"] == [
        {
            "scope": "configuration",
            "code": "provenance_invalid",
            "object_key": None,
            "message": "The provenance document is malformed or unsupported.",
        }
    ]


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


def test_symlinked_cache_ancestor_is_invalid_before_network_or_local_writes(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    cache_link = tmp_path / "cache-link"
    cache_link.symlink_to(external, target_is_directory=True)
    output_dir = tmp_path / "output"
    store = complete_store()

    outcome, _ = invoke_sync(
        tmp_path,
        store,
        output_dir=output_dir,
        cache_dir=cache_link / "cache",
    )

    assert outcome.exit_code == 2
    assert outcome.errors[0].code == "invalid_config"
    assert outcome.report_path is None
    assert store.calls == []
    assert not output_dir.exists()
    assert list(external.iterdir()) == []


def test_broken_cache_symlink_is_invalid_without_writing_a_fatal_report(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.symlink_to(tmp_path / "missing-cache", target_is_directory=True)
    output_dir = tmp_path / "output"
    store = complete_store()

    outcome, _ = invoke_sync(
        tmp_path,
        store,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )

    assert outcome.errors[0].code == "invalid_config"
    assert outcome.report_path is None
    assert store.calls == []
    assert cache_dir.is_symlink()
    assert not output_dir.exists()


def test_nondirectory_cache_ancestor_is_invalid_before_any_output_write(tmp_path):
    ancestor = tmp_path / "not-a-directory"
    ancestor.write_text("sentinel", encoding="utf-8")
    output_dir = tmp_path / "output"
    store = complete_store()

    outcome, _ = invoke_sync(
        tmp_path,
        store,
        output_dir=output_dir,
        cache_dir=ancestor / "cache",
    )

    assert outcome.errors[0].code == "invalid_config"
    assert outcome.report_path is None
    assert store.calls == []
    assert ancestor.read_text(encoding="utf-8") == "sentinel"
    assert not output_dir.exists()


def test_symlinked_output_ancestor_is_invalid_before_cache_or_network_mutation(tmp_path):
    external = tmp_path / "external-output"
    external.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(external, target_is_directory=True)
    cache_dir = tmp_path / "cache"
    store = complete_store()

    outcome, _ = invoke_sync(
        tmp_path,
        store,
        output_dir=output_link / "output",
        cache_dir=cache_dir,
    )

    assert outcome.errors[0].code == "invalid_config"
    assert outcome.report_path is None
    assert store.calls == []
    assert not cache_dir.exists()
    assert list(external.iterdir()) == []


def test_existing_output_file_remains_invalid_config_without_cache_mutation(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.write_text("sentinel", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    store = complete_store()

    outcome, _ = invoke_sync(
        tmp_path,
        store,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )

    assert outcome.errors[0].code == "invalid_config"
    assert outcome.report_path is None
    assert store.calls == []
    assert output_dir.read_text(encoding="utf-8") == "sentinel"
    assert not cache_dir.exists()


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


def test_writer_lock_unexpected_failure_surfaces_cache_lock_failed(tmp_path, monkeypatch):
    import errno

    import src.benchmark.corpus_cache as cache

    store = complete_store()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    real_os_open = cache.os.open

    def fail_only_lock_file(path, *args, **kwargs):
        if str(path).endswith(".index-v1.lock"):
            raise OSError(errno.EACCES, "private lock detail")
        return real_os_open(path, *args, **kwargs)

    monkeypatch.setattr(cache.os, "open", fail_only_lock_file)

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert store.calls == []
    assert [error.code for error in outcome.errors] == ["cache_lock_failed"]


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
    previous_latest_report = b'{"previous":"report"}\n'
    (output_dir / "latest.json").write_bytes(previous_latest)
    (output_dir / "latest-report.json").write_bytes(previous_latest_report)
    store = complete_store()
    real_publish_latest_report = r2_corpus_sync._publish_latest_report

    def fail_after_replacement(*args, **kwargs):
        real_publish_latest_report(*args, **kwargs)
        raise OSError("unsafe pointer detail")

    monkeypatch.setattr(r2_corpus_sync, "_publish_latest_report", fail_after_replacement)

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.manifest is None
    assert_only_failed_attempt_report(output_dir, outcome.report_path)
    assert (output_dir / "latest.json").read_bytes() == previous_latest
    latest_report = read_json(output_dir / "latest-report.json")
    assert latest_report["overall_status"] == "failed"
    assert latest_report["exit_code"] == 2


def test_latest_manifest_failure_restores_previous_pointer_before_fatal_report(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    previous_latest = b'{"previous":"manifest"}\n'
    previous_latest_report = b'{"previous":"report"}\n'
    (output_dir / "latest.json").write_bytes(previous_latest)
    (output_dir / "latest-report.json").write_bytes(previous_latest_report)
    store = complete_store()
    real_publish_latest_manifest = r2_corpus_sync.publish_latest_manifest

    def fail_after_replacement(*args, **kwargs):
        real_publish_latest_manifest(*args, **kwargs)
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
    assert_only_failed_attempt_report(output_dir, outcome.report_path)
    assert (output_dir / "latest.json").read_bytes() == previous_latest
    latest_report = read_json(output_dir / "latest-report.json")
    assert latest_report["overall_status"] == "failed"
    assert latest_report["exit_code"] == 2


def test_latest_report_directory_fsync_failure_restores_manifest_pointer_and_updates_report_pointer(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    previous_latest = b'{"previous":"manifest"}\n'
    previous_latest_report = b'{"previous":"report"}\n'
    (output_dir / "latest.json").write_bytes(previous_latest)
    (output_dir / "latest-report.json").write_bytes(previous_latest_report)
    real_replace = r2_corpus_sync.os.replace
    real_fsync_directory = durability.fsync_directory
    latest_report_replaced = False
    failed_once = False

    def record_replace(source, destination, *args, **kwargs):
        nonlocal latest_report_replaced
        real_replace(source, destination, *args, **kwargs)
        if Path(destination).name == "latest-report.json":
            latest_report_replaced = True

    def fail_after_latest_report_replace(path):
        nonlocal failed_once
        if latest_report_replaced and not failed_once:
            failed_once = True
            raise OSError("unsafe directory fsync detail")
        real_fsync_directory(path)

    monkeypatch.setattr(r2_corpus_sync.os, "replace", record_replace)
    monkeypatch.setattr(durability, "fsync_directory", fail_after_latest_report_replace)

    outcome, _ = invoke_sync(tmp_path, complete_store())

    assert failed_once
    assert outcome.exit_code == 2
    assert_only_failed_attempt_report(output_dir, outcome.report_path)
    assert outcome.manifest is None
    assert (output_dir / "latest.json").read_bytes() == previous_latest
    latest_report = read_json(output_dir / "latest-report.json")
    assert latest_report["overall_status"] == "failed"
    assert latest_report["exit_code"] == 2


def test_failed_fatal_rewrite_removes_stale_success_report_and_restores_pointers(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    previous_latest = b'{"previous":"manifest"}\n'
    previous_latest_report = b'{"previous":"report"}\n'
    (output_dir / "latest.json").write_bytes(previous_latest)
    (output_dir / "latest-report.json").write_bytes(previous_latest_report)
    real_publish_latest_report = r2_corpus_sync._publish_latest_report
    real_publish_report_file = r2_corpus_sync._publish_report_file

    def fail_pointer_after_replacement(*args, **kwargs):
        real_publish_latest_report(*args, **kwargs)
        raise OSError("unsafe pointer detail")

    def fail_fatal_rewrite(*args, **kwargs):
        report = args[3]
        if report["overall_status"] == "failed":
            raise OSError("unsafe report detail")
        return real_publish_report_file(*args, **kwargs)

    monkeypatch.setattr(r2_corpus_sync, "_publish_latest_report", fail_pointer_after_replacement)
    monkeypatch.setattr(r2_corpus_sync, "_publish_report_file", fail_fatal_rewrite)

    outcome, _ = invoke_sync(tmp_path, complete_store())

    assert outcome.exit_code == 2
    assert outcome.report_path is None
    assert list((output_dir / "reports").glob("*.json")) == []
    assert (output_dir / "latest.json").read_bytes() == previous_latest
    assert (output_dir / "latest-report.json").read_bytes() == previous_latest_report


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
        "unreadable": 0,
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


def test_unkeyable_listing_entry_is_reported_as_root_scope(tmp_path):
    store = complete_store()
    store.errors[("list", None)] = R2StoreError(
        "object_metadata_invalid", "Object metadata is invalid.", None
    )

    outcome, _ = invoke_sync(tmp_path, store, dry_run=True)

    assert outcome.exit_code == 2
    report = read_report(outcome.report_path)
    assert report["errors"] == [
        {
            "scope": "root",
            "code": "object_metadata_invalid",
            "object_key": None,
            "message": "Object metadata is invalid.",
        }
    ]


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


def test_metadata_progress_is_emitted_live_before_a_blocked_head_returns(tmp_path):
    monotonic = ManualMonotonic()
    blocked_started = Event()
    release_blocked = Event()
    progress_seen = Event()
    base = store_from_bodies(
        {
            "2/blocked.dtx": b"chart",
            "2/fast.dtx": b"chart",
        }
    )

    class LiveProgressStore(FakeStore):
        def head_object(self, key: str) -> HeadMetadata:
            self._record("head", key)
            if key == "2/blocked.dtx":
                blocked_started.set()
                assert release_blocked.wait(timeout=5)
            else:
                monotonic.set(6.0)
            return self.heads[key]

    store = LiveProgressStore(base.listed, base.heads, base.bodies)
    events = []
    outcome_holder = []
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None, dry_run=True)
    times = iter((STARTED_AT, COMPLETED_AT))

    def capture_progress(event):
        events.append(event)
        if event.phase == "metadata" and event.completed == 1:
            progress_seen.set()

    thread = Thread(
        target=lambda: outcome_holder.append(
            sync_r2_corpus(
                request,
                environ=DEFAULT_ENVIRON,
                dependency_check=lambda: None,
                store_factory=lambda _config: store,
                clock=lambda: next(times, COMPLETED_AT),
                monotonic=monotonic,
                run_id_factory=lambda: RUN_ID,
                progress=capture_progress,
            )
        ),
        daemon=True,
    )
    thread.start()
    try:
        assert blocked_started.wait(timeout=2)
        assert progress_seen.wait(timeout=2)
        assert thread.is_alive()
    finally:
        release_blocked.set()
        thread.join(timeout=5)

    assert outcome_holder[0].exit_code == 0
    metadata = [
        (event.completed, event.total)
        for event in events
        if event.phase == "metadata" and event.completed > 0
    ]
    assert metadata == [(1, 2), (2, 2)]


def test_phase_progress_callback_failure_is_disabled_without_changing_success(tmp_path):
    secret = "SECRET phase callback detail"
    callback_phases: list[str] = []
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)
    times = iter((STARTED_AT, COMPLETED_AT))

    def fail_first_callback(event):
        callback_phases.append(event.phase)
        raise RuntimeError(secret)

    outcome = sync_r2_corpus(
        request,
        environ=DEFAULT_ENVIRON,
        dependency_check=lambda: None,
        store_factory=lambda _config: complete_store(),
        clock=lambda: next(times, COMPLETED_AT),
        run_id_factory=lambda: RUN_ID,
        progress=fail_first_callback,
    )

    assert outcome.overall_status == "complete"
    assert outcome.exit_code == 0
    assert callback_phases == ["configuration"]
    assert read_report(outcome.report_path)["overall_status"] == "complete"
    assert secret not in repr(outcome)


def test_live_item_progress_callback_failure_does_not_abort_completed_work(tmp_path):
    callback_events = []
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None, dry_run=True)
    times = iter((STARTED_AT, COMPLETED_AT))

    def fail_on_metadata_item(event):
        callback_events.append((event.phase, event.completed))
        if event.phase == "metadata" and event.completed > 0:
            raise RuntimeError("SECRET live item callback detail")

    outcome = sync_r2_corpus(
        request,
        environ=DEFAULT_ENVIRON,
        dependency_check=lambda: None,
        store_factory=lambda _config: complete_store(),
        clock=lambda: next(times, COMPLETED_AT),
        run_id_factory=lambda: RUN_ID,
        progress=fail_on_metadata_item,
    )

    assert outcome.overall_status == "dry_run_complete"
    assert outcome.exit_code == 0
    assert callback_events[-1][0] == "metadata"
    assert callback_events[-1][1] > 0
    assert read_report(outcome.report_path)["overall_status"] == "dry_run_complete"


def test_fatal_path_progress_callback_failure_preserves_primary_failure(tmp_path):
    secret = "SECRET fatal callback detail"
    callback_phases: list[str] = []
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)
    environ = {
        "CRUX_R2_ENDPOINT_URL": ENDPOINT,
        "CRUX_R2_BUCKET": "simfile-dtx",
    }

    def fail_on_report_phase(event):
        callback_phases.append(event.phase)
        if event.phase == "report":
            raise RuntimeError(secret)

    outcome = sync_r2_corpus(
        request,
        environ=environ,
        dependency_check=lambda: None,
        store_factory=lambda _config: (_ for _ in ()).throw(
            R2StoreError("missing_credentials", "unsafe credential detail")
        ),
        clock=lambda: STARTED_AT,
        run_id_factory=lambda: RUN_ID,
        progress=fail_on_report_phase,
    )

    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2
    assert [error.code for error in outcome.errors] == ["missing_credentials"]
    assert callback_phases == ["configuration", "report"]
    report = read_report(outcome.report_path)
    assert [error["code"] for error in report["errors"]] == ["missing_credentials"]
    assert secret not in repr(outcome) + json.dumps(report)


def test_fatal_finish_progress_callback_failure_does_not_escape(tmp_path):
    callback_phases: list[str] = []
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)

    def fail_on_fatal_finish(event):
        callback_phases.append(event.phase)
        if event.phase == "failed":
            raise RuntimeError("SECRET fatal finish callback detail")

    outcome = sync_r2_corpus(
        request,
        environ=DEFAULT_ENVIRON,
        dependency_check=lambda: None,
        store_factory=lambda _config: (_ for _ in ()).throw(
            R2StoreError("missing_credentials", "unsafe credential detail")
        ),
        clock=lambda: STARTED_AT,
        run_id_factory=lambda: RUN_ID,
        progress=fail_on_fatal_finish,
    )

    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2
    assert [error.code for error in outcome.errors] == ["missing_credentials"]
    assert callback_phases == ["configuration", "report", "failed"]
    assert read_report(outcome.report_path)["overall_status"] == "failed"


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


def test_output_publication_lock_rejects_symlink_without_touching_target(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    external_target = tmp_path / "external-lock-target"
    external_target.write_bytes(b"sentinel")
    (output_dir / ".r2-corpus-publication.lock").symlink_to(external_target)

    outcome, _ = invoke_sync(tmp_path, complete_store())

    assert outcome.exit_code == 2
    assert outcome.report_path is None
    assert external_target.read_bytes() == b"sentinel"
    assert not (output_dir / "latest.json").exists()
    assert not (output_dir / "latest-report.json").exists()
    assert [error.code for error in outcome.errors] == ["artifact_write_failed"]


@pytest.mark.parametrize("missing_flag", ["O_NOFOLLOW", "O_DIRECTORY"])
def test_missing_posix_flag_fails_preflight_without_network_or_local_mutation(
    tmp_path, monkeypatch, missing_flag
):
    store = complete_store()
    monkeypatch.delattr(r2_corpus_sync.os, missing_flag)

    outcome, _ = invoke_sync(tmp_path, store, dry_run=True)

    assert outcome.exit_code == 2
    assert outcome.report_path is None
    assert [error.code for error in outcome.errors] == ["unsupported_platform"]
    assert all(missing_flag not in error.message for error in outcome.errors)
    assert store.calls == []
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "cache").exists()


def test_incompatible_fcntl_fails_preflight_without_network_or_local_mutation(
    tmp_path, monkeypatch
):
    store = complete_store()
    monkeypatch.setattr(r2_corpus_sync, "_load_fcntl", lambda: object(), raising=False)

    outcome, _ = invoke_sync(tmp_path, store, dry_run=True)

    assert outcome.exit_code == 2
    assert outcome.report_path is None
    assert [error.code for error in outcome.errors] == ["unsupported_platform"]
    assert store.calls == []
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "cache").exists()


def test_fcntl_without_nonblocking_lock_fails_preflight_without_mutation(tmp_path, monkeypatch):
    class PartialFcntl:
        LOCK_EX = 2
        LOCK_UN = 8

        @staticmethod
        def flock(_descriptor, _operation):
            return None

    store = complete_store()
    monkeypatch.setattr(r2_corpus_sync, "_load_fcntl", lambda: PartialFcntl)

    outcome, _ = invoke_sync(tmp_path, store, dry_run=True)

    assert outcome.exit_code == 2
    assert outcome.report_path is None
    assert [error.code for error in outcome.errors] == ["unsupported_platform"]
    assert store.calls == []
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "cache").exists()


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


def test_invalid_injected_run_id_uses_trusted_fallback_and_reports_internal_error(tmp_path):
    invalid_run_id = "not-a-uuid"
    store = complete_store()

    outcome, _ = invoke_sync(tmp_path, store, run_id=invalid_run_id)

    assert outcome.exit_code == 2
    assert outcome.errors[0].code == "internal_error"
    report = read_report(outcome.report_path)
    assert report["errors"][0]["code"] == "internal_error"
    assert report["run_id"] != invalid_run_id
    assert invalid_run_id not in outcome.report_path.name
    assert store.calls == []


def test_naive_injected_start_clock_uses_trusted_fallback_for_fatal_report(tmp_path):
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)
    naive = STARTED_AT.replace(tzinfo=None)
    store = complete_store()

    outcome = sync_r2_corpus(
        request,
        environ=DEFAULT_ENVIRON,
        dependency_check=lambda: None,
        store_factory=lambda _config: store,
        clock=lambda: naive,
        run_id_factory=lambda: RUN_ID,
    )

    assert outcome.errors[0].code == "internal_error"
    report = read_report(outcome.report_path)
    assert report["errors"][0]["code"] == "internal_error"
    assert report["started_at"] != "2026-07-25T01:02:03.12Z"
    assert report["started_at"].endswith("Z")
    assert store.calls == []


def test_raising_clock_factory_retains_internal_error_when_report_fallback_is_needed(tmp_path):
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)

    outcome = sync_r2_corpus(
        request,
        environ=DEFAULT_ENVIRON,
        dependency_check=lambda: None,
        store_factory=lambda _config: complete_store(),
        clock=lambda: (_ for _ in ()).throw(RuntimeError("unsafe clock detail")),
        run_id_factory=lambda: RUN_ID,
    )

    assert outcome.exit_code == 2
    assert outcome.errors[0].code == "internal_error"
    report = read_report(outcome.report_path)
    assert report["errors"][0]["code"] == "internal_error"
    assert "unsafe clock detail" not in repr(outcome)
    assert "unsafe clock detail" not in json.dumps(report)


def test_raising_run_id_factory_uses_trusted_fallback_without_reusing_factory(tmp_path):
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)
    calls = 0

    def raise_run_id():
        nonlocal calls
        calls += 1
        raise RuntimeError("unsafe run ID detail")

    outcome = sync_r2_corpus(
        request,
        environ=DEFAULT_ENVIRON,
        dependency_check=lambda: None,
        store_factory=lambda _config: complete_store(),
        clock=lambda: STARTED_AT,
        run_id_factory=raise_run_id,
    )

    assert calls == 1
    assert outcome.errors[0].code == "internal_error"
    report = read_report(outcome.report_path)
    assert report["run_id"] != RUN_ID
    assert report["errors"][0]["code"] == "internal_error"
    assert "unsafe run ID detail" not in json.dumps(report)


def test_later_clock_failure_uses_trusted_completion_time_and_keeps_internal_error(tmp_path):
    values = iter((STARTED_AT,))

    def clock():
        try:
            return next(values)
        except StopIteration:
            raise RuntimeError("unsafe completion clock detail") from None

    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)
    outcome = sync_r2_corpus(
        request,
        environ=DEFAULT_ENVIRON,
        dependency_check=lambda: None,
        store_factory=lambda _config: complete_store(),
        clock=clock,
        run_id_factory=lambda: RUN_ID,
    )

    assert outcome.exit_code == 2
    assert outcome.errors[0].code == "internal_error"
    report = read_report(outcome.report_path)
    assert report["errors"][0]["code"] == "internal_error"
    assert report["completed_at"].endswith("Z")
    assert "unsafe completion clock detail" not in json.dumps(report)


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


def process_sync_worker(
    output_dir: Path,
    cache_dir: Path,
    body: bytes,
    run_id: str,
    dry_run: bool,
    block_before_latest_report: bool,
    reached_publication,
    release_publication,
    started,
    done,
    results,
) -> None:
    started.set()
    if block_before_latest_report:
        real_publish_latest_report = r2_corpus_sync._publish_latest_report

        def block_then_publish(*args, **kwargs):
            reached_publication.set()
            assert release_publication.wait(timeout=10)
            return real_publish_latest_report(*args, **kwargs)

        r2_corpus_sync._publish_latest_report = block_then_publish
    try:
        outcome = threaded_sync(
            output_dir=output_dir,
            cache_dir=cache_dir,
            store=store_from_bodies({"2/chart.dtx": body}),
            run_id=run_id,
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
            dry_run=dry_run,
        )
        results.put(
            {
                "run_id": run_id,
                "status": outcome.overall_status,
                "manifest_sha256": (
                    outcome.manifest.manifest_sha256 if outcome.manifest is not None else None
                ),
            }
        )
    except BaseException as error:
        results.put({"run_id": run_id, "error": type(error).__name__})
    finally:
        done.set()


def test_two_process_real_runs_publish_as_one_serialized_unit(tmp_path):
    context = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
    output_dir = tmp_path / "output"
    reached_publication = context.Event()
    release_publication = context.Event()
    first_started = context.Event()
    first_done = context.Event()
    second_started = context.Event()
    second_done = context.Event()
    results = context.Queue()
    first = context.Process(
        target=process_sync_worker,
        args=(
            output_dir,
            tmp_path / "cache-first",
            b"first",
            RUN_ID,
            False,
            True,
            reached_publication,
            release_publication,
            first_started,
            first_done,
            results,
        ),
    )
    second = context.Process(
        target=process_sync_worker,
        args=(
            output_dir,
            tmp_path / "cache-second",
            b"second",
            SECOND_RUN_ID,
            False,
            False,
            context.Event(),
            context.Event(),
            second_started,
            second_done,
            results,
        ),
    )

    first.start()
    try:
        assert first_started.wait(timeout=5)
        assert reached_publication.wait(timeout=10)
        second.start()
        assert second_started.wait(timeout=5)
        assert not second_done.wait(timeout=0.5)
    finally:
        release_publication.set()
        first.join(timeout=10)
        if second.pid is not None:
            second.join(timeout=10)

    assert first.exitcode == 0
    assert second.exitcode == 0
    outcomes = {result["run_id"]: result for result in (results.get(), results.get())}
    assert "error" not in outcomes[RUN_ID]
    assert "error" not in outcomes[SECOND_RUN_ID]
    assert (
        read_json(output_dir / "latest.json")["manifest_sha256"]
        == outcomes[SECOND_RUN_ID]["manifest_sha256"]
    )
    assert read_json(output_dir / "latest-report.json")["report_path"].endswith(
        f"{SECOND_RUN_ID}.json"
    )


def test_two_process_dry_run_waits_for_real_publication_then_moves_only_report(tmp_path):
    context = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
    output_dir = tmp_path / "output"
    reached_publication = context.Event()
    release_publication = context.Event()
    real_started = context.Event()
    real_done = context.Event()
    dry_started = context.Event()
    dry_done = context.Event()
    results = context.Queue()
    real_process = context.Process(
        target=process_sync_worker,
        args=(
            output_dir,
            tmp_path / "cache-real",
            b"real",
            RUN_ID,
            False,
            True,
            reached_publication,
            release_publication,
            real_started,
            real_done,
            results,
        ),
    )
    dry_process = context.Process(
        target=process_sync_worker,
        args=(
            output_dir,
            tmp_path / "cache-dry",
            b"dry",
            SECOND_RUN_ID,
            True,
            False,
            context.Event(),
            context.Event(),
            dry_started,
            dry_done,
            results,
        ),
    )

    real_process.start()
    try:
        assert real_started.wait(timeout=5)
        assert reached_publication.wait(timeout=10)
        dry_process.start()
        assert dry_started.wait(timeout=5)
        assert not dry_done.wait(timeout=0.5)
    finally:
        release_publication.set()
        real_process.join(timeout=10)
        if dry_process.pid is not None:
            dry_process.join(timeout=10)

    assert real_process.exitcode == 0
    assert dry_process.exitcode == 0
    outcomes = {result["run_id"]: result for result in (results.get(), results.get())}
    assert "error" not in outcomes[RUN_ID]
    assert "error" not in outcomes[SECOND_RUN_ID]
    assert (
        read_json(output_dir / "latest.json")["manifest_sha256"]
        == outcomes[RUN_ID]["manifest_sha256"]
    )
    latest_report = read_json(output_dir / "latest-report.json")
    assert latest_report["report_path"].endswith(f"{SECOND_RUN_ID}.json")
    assert latest_report["overall_status"] == "dry_run_complete"


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


# ---------------------------------------------------------------------------
# Coverage gap closures for r2_corpus_sync error paths
# ---------------------------------------------------------------------------


def test_utc_now_and_new_run_id_helpers_return_trusted_values() -> None:
    now = r2_corpus_sync.utc_now()
    assert now.tzinfo is not None
    run_id = r2_corpus_sync.new_run_id()
    # Should be a canonical UUID4 string.
    from uuid import UUID

    parsed = UUID(run_id)
    assert parsed.version == 4
    assert str(parsed) == run_id


def test_run_sync_internal_error_path_returns_internal_error(tmp_path) -> None:
    """Cover line 318-321: catch-all Exception in _run_sync configuration phase."""
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", None)

    def raising_dependency_check() -> None:
        raise RuntimeError("SECRET dependency detail")

    outcome = sync_r2_corpus(
        request,
        environ=DEFAULT_ENVIRON,
        dependency_check=raising_dependency_check,
        store_factory=lambda _config: complete_store(),
        clock=lambda: STARTED_AT,
        run_id_factory=lambda: RUN_ID,
    )

    assert outcome.exit_code == 2
    assert outcome.errors[0].code == "internal_error"
    assert "SECRET" not in repr(outcome)


def test_run_sync_oserror_from_lock_returns_artifact_write_failed(tmp_path, monkeypatch) -> None:
    """Cover lines 364-373: OSError from lock context manager (outside with lock)."""
    store = complete_store()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    @contextmanager
    def raising_lock(_path: Path):
        raise OSError("SECRET lock detail")
        yield  # pragma: no cover

    monkeypatch.setattr(r2_corpus_sync, "cache_writer_lock", raising_lock)

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.errors[0].code == "artifact_write_failed"
    assert "SECRET" not in repr(outcome)


def test_run_sync_unexpected_exception_from_lock_returns_internal_error(
    tmp_path, monkeypatch
) -> None:
    """Cover lines 374-377: catch-all Exception from lock context manager."""
    store = complete_store()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    @contextmanager
    def raising_lock(_path: Path):
        raise KeyError("SECRET lock detail")
        yield  # pragma: no cover

    monkeypatch.setattr(r2_corpus_sync, "cache_writer_lock", raising_lock)

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.errors[0].code == "internal_error"
    assert "SECRET" not in repr(outcome)


def test_snapshot_pointers_failure_raises_report_write_failure(tmp_path, monkeypatch) -> None:
    """Cover lines 457-458: _snapshot_pointers failure in _run_transaction."""
    store = complete_store()

    def raising_snapshot(_output_dir: Path):
        raise OSError("SECRET snapshot detail")

    monkeypatch.setattr(r2_corpus_sync, "_snapshot_pointers", raising_snapshot)

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.report_path is None
    assert outcome.errors[0].code == "artifact_write_failed"
    assert "SECRET" not in repr(outcome)


def test_manifest_publication_unexpected_exception_restores_and_raises(
    tmp_path, monkeypatch
) -> None:
    """Cover lines 473-475: non-ManifestPublicationError during publish_manifest."""
    store = complete_store()

    def raising_manifest(*_args, **_kwargs):
        raise RuntimeError("SECRET manifest detail")

    monkeypatch.setattr(r2_corpus_sync, "publish_manifest", raising_manifest)

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.report_path is None
    assert outcome.errors[0].code == "artifact_write_failed"
    assert "SECRET" not in repr(outcome)


def test_report_publication_restore_failure_removes_attempt_report(tmp_path, monkeypatch) -> None:
    """Cover lines 531-533: _restore_pointers failure during publication error."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    store = complete_store()

    # Make _publish_latest_report fail to trigger the restore path.
    def fail_latest_report(*_args, **_kwargs):
        raise OSError("SECRET report detail")

    # Make _restore_pointers fail to trigger _remove_attempt_report path.
    def fail_restore(_output_dir: Path, _snapshots) -> None:
        raise OSError("SECRET restore detail")

    monkeypatch.setattr(r2_corpus_sync, "_publish_latest_report", fail_latest_report)
    monkeypatch.setattr(r2_corpus_sync, "_restore_pointers", fail_restore)

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.report_path is None
    assert outcome.errors[0].code == "artifact_write_failed"
    assert "SECRET" not in repr(outcome)


def test_fatal_outcome_publication_restore_failure_returns_fallback(tmp_path, monkeypatch) -> None:
    """Cover lines 643-645: _restore_pointers_or_report_failure in _fatal_outcome."""
    store = complete_store()

    # Trigger a fatal outcome by making the cache index invalid.
    def raising_load(_path: Path):
        raise ValueError("SECRET invalid cache")

    monkeypatch.setattr(r2_corpus_sync.CacheIndexStore, "load", raising_load)

    # Make the report publication fail in _fatal_outcome to trigger restore.
    def fail_publish_report(*_args, **_kwargs):
        raise OSError("SECRET fatal report detail")

    monkeypatch.setattr(r2_corpus_sync, "_publish_report_file", fail_publish_report)

    outcome, _ = invoke_sync(tmp_path, store)

    assert outcome.exit_code == 2
    assert outcome.report_path is None
    assert outcome.errors[0].code == "cache_index_invalid"
    assert "SECRET" not in repr(outcome)


def test_report_fallback_outcome_with_primary_error_includes_both_errors() -> None:
    """Cover line 665: primary_error + artifact_error in _report_fallback_outcome."""
    tracker = r2_corpus_sync._Progress(lambda _event: None, lambda: 0.0)
    primary = r2_corpus_sync.SyncError(
        "configuration", "invalid_config", "Configuration is invalid."
    )
    outcome = r2_corpus_sync._report_fallback_outcome(tracker, primary)
    assert outcome.exit_code == 2
    assert [e.code for e in outcome.errors] == ["invalid_config", "artifact_write_failed"]


def test_output_publication_lock_swallows_flock_unlock_oserror(tmp_path, monkeypatch) -> None:
    """Cover lines 876-877: flock unlock OSError is swallowed."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    import fcntl

    real_flock = fcntl.flock
    unlock_calls: list[int] = []

    def failing_flock(fd, operation):
        if operation == fcntl.LOCK_UN:
            unlock_calls.append(fd)
            raise OSError("SECRET unlock detail")
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", failing_flock)

    outcome, _ = invoke_sync(tmp_path, complete_store())

    assert outcome.exit_code == 0
    assert unlock_calls, "flock unlock should have been attempted"


def test_output_publication_lock_swallows_close_oserror(tmp_path, monkeypatch) -> None:
    """Cover lines 880-881: os.close OSError is swallowed."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    real_close = r2_corpus_sync.os.close
    real_fstat = r2_corpus_sync.os.fstat
    lock_path = output_dir / ".r2-corpus-publication.lock"
    lock_inodes: set[int] = set()
    real_open = r2_corpus_sync.os.open

    def tracking_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if str(path) == str(lock_path):
            lock_inodes.add(real_fstat(fd).st_ino)
        return fd

    def failing_close(fd):
        try:
            stat = real_fstat(fd)
        except OSError:
            return real_close(fd)
        if stat.st_ino in lock_inodes:
            raise OSError("SECRET close detail")
        return real_close(fd)

    monkeypatch.setattr(r2_corpus_sync.os, "open", tracking_open)
    monkeypatch.setattr(r2_corpus_sync.os, "close", failing_close)

    outcome, _ = invoke_sync(tmp_path, complete_store())

    assert outcome.exit_code == 0


def test_validate_platform_support_rejects_missing_fcntl_attribute(tmp_path, monkeypatch) -> None:
    """Cover lines 895-896: ImportError/AttributeError in _load_fcntl."""
    real_import = __import__

    def import_without_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_fcntl)

    with pytest.raises(r2_corpus_sync._SyncFailure) as caught:
        r2_corpus_sync._validate_platform_support()
    assert caught.value.error.code == "unsupported_platform"


def test_acquire_output_lock_rejects_non_regular_lock_file(tmp_path, monkeypatch) -> None:
    """Cover line 927: non-regular lock file raises OSError."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    lock_path = output_dir / ".r2-corpus-publication.lock"
    # Create a FIFO instead of a regular file.
    if hasattr(os, "mkfifo"):
        os.mkfifo(lock_path)
    else:
        pytest.skip("mkfifo unavailable")

    with pytest.raises(r2_corpus_sync._SyncFailure) as caught:
        r2_corpus_sync._acquire_output_lock(output_dir)
    assert caught.value.error.code == "artifact_write_failed"


def test_acquire_output_lock_rejects_flock_unsupported_errno(tmp_path, monkeypatch) -> None:
    """Cover lines 931-938: flock raises unsupported errno."""
    import errno
    import fcntl

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def unsupported_flock(_fd, _operation):
        raise OSError(errno.ENOTSUP, "private")

    monkeypatch.setattr(fcntl, "flock", unsupported_flock)

    with pytest.raises(r2_corpus_sync._SyncFailure) as caught:
        r2_corpus_sync._acquire_output_lock(output_dir)
    assert caught.value.error.code == "unsupported_platform"


def test_acquire_output_lock_closes_descriptor_on_oserror(tmp_path, monkeypatch) -> None:
    """Cover lines 941-947: descriptor cleanup on OSError."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    real_open = r2_corpus_sync.os.open

    def fail_open(path, flags, *args, **kwargs):
        if str(path).endswith(".r2-corpus-publication.lock"):
            fd = real_open(path, flags, *args, **kwargs)
            # Return a valid fd, but make fstat fail to trigger the OSError path.
            return fd
        return real_open(path, flags, *args, **kwargs)

    def fail_fstat(_fd):
        raise OSError("SECRET fstat detail")

    monkeypatch.setattr(r2_corpus_sync.os, "open", fail_open)
    monkeypatch.setattr(r2_corpus_sync.os, "fstat", fail_fstat)

    with pytest.raises(r2_corpus_sync._SyncFailure) as caught:
        r2_corpus_sync._acquire_output_lock(output_dir)
    assert caught.value.error.code == "artifact_write_failed"
    assert "SECRET" not in str(caught.value)


def test_snapshot_pointer_rejects_non_regular_file(tmp_path) -> None:
    """Cover line 965: non-regular file in _snapshot_pointer."""
    path = tmp_path / "latest.json"
    if hasattr(os, "mkfifo"):
        os.mkfifo(path)
    else:
        pytest.skip("mkfifo unavailable")

    with pytest.raises(OSError, match="artifact pointer is unavailable"):
        r2_corpus_sync._snapshot_pointer(path)


def test_snapshot_pointer_rejects_binding_change(tmp_path, monkeypatch) -> None:
    """Cover line 976: descriptor/path binding mismatch in _snapshot_pointer."""
    path = tmp_path / "latest.json"
    path.write_bytes(b'{"data":"latest"}\n')

    real_fstat = r2_corpus_sync.os.fstat

    def shifted_fstat(fd):
        result = real_fstat(fd)
        return r2_corpus_sync.os.stat_result(
            (result.st_mode, result.st_ino + 1, result.st_dev, 0, 0, 0, 0, 0, 0, 0)
        )

    monkeypatch.setattr(r2_corpus_sync.os, "fstat", shifted_fstat)

    with pytest.raises(OSError, match="artifact pointer is unavailable"):
        r2_corpus_sync._snapshot_pointer(path)


def test_restore_pointers_aggregates_failures(tmp_path, monkeypatch) -> None:
    """Cover lines 993-996: _restore_pointers collects failures and raises."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def fail_restore(_path, _content):
        raise OSError("SECRET restore detail")

    monkeypatch.setattr(r2_corpus_sync, "_restore_pointer", fail_restore)

    snapshots = r2_corpus_sync._PointerSnapshots(latest_manifest=b"x", latest_report=b"y")
    with pytest.raises(OSError, match="artifact pointer rollback failed"):
        r2_corpus_sync._restore_pointers(output_dir, snapshots)


def test_restore_pointers_or_report_failure_wraps_as_report_write_failure(
    tmp_path, monkeypatch
) -> None:
    """Cover lines 1006-1007: _restore_pointers_or_report_failure wraps exception."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def fail_restore(_output_dir, _snapshots) -> None:
        raise OSError("SECRET detail")

    monkeypatch.setattr(r2_corpus_sync, "_restore_pointers", fail_restore)

    primary = r2_corpus_sync.SyncError("artifact", "artifact_write_failed", "primary failure")
    with pytest.raises(r2_corpus_sync._ReportWriteFailure) as caught:
        r2_corpus_sync._restore_pointers_or_report_failure(output_dir, None, primary)
    assert caught.value.primary_error is primary


def test_restore_pointer_rejects_non_regular_for_removal(tmp_path) -> None:
    """Cover lines 1018-1019: non-regular file in _restore_pointer removal path."""
    path = tmp_path / "latest.json"
    if hasattr(os, "mkfifo"):
        os.mkfifo(path)
    else:
        pytest.skip("mkfifo unavailable")

    with pytest.raises(OSError, match="artifact pointer is unavailable"):
        r2_corpus_sync._restore_pointer(path, None)


def test_restore_pointer_removes_existing_regular_file(tmp_path) -> None:
    """Cover lines 1020-1021: regular file is unlinked and directory fsynced."""
    path = tmp_path / "latest.json"
    path.write_bytes(b'{"old":"data"}\n')

    r2_corpus_sync._restore_pointer(path, None)

    assert not path.exists()


def test_remove_attempt_report_skips_non_regular_file(tmp_path) -> None:
    """Cover lines 1030-1031: non-regular file is skipped in _remove_attempt_report."""
    reports_dir = tmp_path / "output" / "reports"
    reports_dir.mkdir(parents=True)
    report_path = reports_dir / report_filename(STARTED_AT, RUN_ID)
    if hasattr(os, "mkfifo"):
        os.mkfifo(report_path)
    else:
        pytest.skip("mkfifo unavailable")

    # Should not raise; non-regular file is skipped.
    r2_corpus_sync._remove_attempt_report(tmp_path / "output", STARTED_AT, RUN_ID)
    assert report_path.exists()


def test_remove_attempt_report_swallows_unlink_oserror(tmp_path, monkeypatch) -> None:
    """Cover lines 1035-1036: OSError during unlink is swallowed."""
    reports_dir = tmp_path / "output" / "reports"
    reports_dir.mkdir(parents=True)
    report_path = reports_dir / report_filename(STARTED_AT, RUN_ID)
    report_path.write_text("{}", encoding="utf-8")

    def fail_unlink(_path, *_args, **_kwargs):
        raise OSError("SECRET unlink detail")

    monkeypatch.setattr(r2_corpus_sync.Path, "unlink", fail_unlink)

    # Should not raise; OSError is swallowed.
    r2_corpus_sync._remove_attempt_report(tmp_path / "output", STARTED_AT, RUN_ID)


def test_atomic_replace_bytes_cleanup_failure_sets_cleanup_flag(tmp_path, monkeypatch) -> None:
    """Cover durability.atomic_replace_bytes: temporary cleanup failure sets cleanup_failed."""
    path = tmp_path / "output" / "target.json"
    path.parent.mkdir(parents=True)

    def fail_unlink(_self, missing_ok=False):
        raise OSError("SECRET unlink detail")

    # Make os.replace fail so the temporary file needs cleanup.
    monkeypatch.setattr(
        durability.os,
        "replace",
        lambda *_: (_ for _ in ()).throw(OSError("SECRET replace detail")),
    )
    monkeypatch.setattr(durability.Path, "unlink", fail_unlink)

    with pytest.raises(OSError, match="artifact publication failed"):
        durability.atomic_replace_bytes(path, b'{"data":"new"}\n')


def test_adapter_error_assigns_object_scope_for_keyed_metadata() -> None:
    """Cover line 1077: object scope for keyed object_metadata_invalid."""
    error = r2_corpus_sync.R2StoreError("object_metadata_invalid", "private detail", "42/chart.dtx")
    sync_error = r2_corpus_sync._adapter_error(error)
    assert sync_error.scope == "object"
    assert sync_error.code == "object_metadata_invalid"


def test_adapter_error_assigns_root_scope_for_unkeyable_listing() -> None:
    """Cover line 1087: root scope for root_list_failed."""
    error = r2_corpus_sync.R2StoreError("root_list_failed", "private detail", None)
    sync_error = r2_corpus_sync._adapter_error(error)
    assert sync_error.scope == "root"


def test_lock_error_code_returns_unsupported_platform() -> None:
    """Cover line 1097: unsupported_platform lock error code."""
    error = RuntimeError("unsupported_platform")
    assert r2_corpus_sync._lock_error_code(error) == "unsupported_platform"


def test_lock_error_code_returns_cache_lock_failed() -> None:
    """Cover line 1099-1100: cache_lock_failed and fallback to artifact_write_failed."""
    error = RuntimeError("cache_lock_failed")
    assert r2_corpus_sync._lock_error_code(error) == "cache_lock_failed"
    unknown = RuntimeError("unknown detail")
    assert r2_corpus_sync._lock_error_code(unknown) == "artifact_write_failed"


def test_validate_request_paths_rejects_non_path_output_dir(tmp_path) -> None:
    """Cover line 1119: non-Path output_dir raises ValueError."""
    request = SyncRequest("not-a-path", tmp_path / "cache", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pathlib paths"):
        r2_corpus_sync._validate_request_paths(request)


def test_validate_request_paths_rejects_non_path_provenance(tmp_path) -> None:
    """Cover line 1122: non-Path provenance_file raises ValueError."""
    request = SyncRequest(tmp_path / "output", tmp_path / "cache", "not-a-path")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pathlib path"):
        r2_corpus_sync._validate_request_paths(request)


def test_validate_directory_components_rejects_parent_traversal() -> None:
    """Cover line 1127: parent traversal in path raises ValueError."""
    path = Path("/safe/../escape")
    with pytest.raises(ValueError, match="parent traversal"):
        r2_corpus_sync._validate_directory_components(path)


def test_validate_directory_components_rejects_oserror_on_lstat(tmp_path, monkeypatch) -> None:
    """Cover lines 1136-1137: OSError during lstat raises ValueError."""
    path = tmp_path / "subdir"
    real_lstat = r2_corpus_sync.Path.lstat

    def fail_lstat(self):
        if str(self) == str(path):
            raise OSError("SECRET lstat detail")
        return real_lstat(self)

    monkeypatch.setattr(r2_corpus_sync.Path, "lstat", fail_lstat)

    with pytest.raises(ValueError, match="artifact path is unavailable"):
        r2_corpus_sync._validate_directory_components(path)


def test_validate_run_id_rejects_non_string() -> None:
    """Cover line 1144: non-string run_id raises ValueError."""
    with pytest.raises(ValueError, match="canonical UUID4"):
        r2_corpus_sync._validate_run_id(12345)  # type: ignore[arg-type]


def test_acquire_output_lock_reraises_non_unsupported_flock_oserror(tmp_path, monkeypatch) -> None:
    """Cover line 939: non-unsupported flock errno is re-raised as _SyncFailure."""
    import errno
    import fcntl

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def flock_with_eacces(_fd, _operation):
        raise OSError(errno.EACCES, "private flock detail")

    monkeypatch.setattr(fcntl, "flock", flock_with_eacces)

    with pytest.raises(r2_corpus_sync._SyncFailure) as caught:
        r2_corpus_sync._acquire_output_lock(output_dir)
    assert caught.value.error.code == "artifact_write_failed"
    assert "private" not in str(caught.value)


def test_safe_error_assigns_root_scope_for_root_list_failed() -> None:
    """Cover line 1087: root scope for root_list_failed in _safe_error."""
    error = r2_corpus_sync._safe_error("root_list_failed")
    assert error.scope == "root"
    assert error.code == "root_list_failed"
