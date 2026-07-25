import builtins
import errno
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Event, Lock, Thread
from time import sleep
from typing import Iterator

import pytest

from src.benchmark.corpus_cache import (
    CacheIndexEntry,
    CacheIndexStore,
    cache_writer_lock,
    sync_cache,
)
from src.benchmark.r2_corpus_models import (
    R2Config,
    RemoteObject,
    SimfileInventory,
    SyncError,
    format_manifest_timestamp,
)
from src.benchmark.r2_inventory import ObjectDownload, R2StoreError

FIXED_MTIME = datetime(2026, 7, 25, tzinfo=timezone.utc)


@dataclass(frozen=True)
class OpenCall:
    key: str
    if_match: str | None


class FakeStore:
    def __init__(
        self,
        *,
        body: bytes = b"chart",
        response_etag: str | None = "etag",
        response_etag_is_weak: bool | None = False,
        response_size: int | None = 5,
        response_mtime: datetime | None = FIXED_MTIME,
        error: R2StoreError | None = None,
    ) -> None:
        self.body = body
        self.response_etag = response_etag
        self.response_etag_is_weak = response_etag_is_weak
        self.response_size = response_size
        self.response_mtime = response_mtime
        self.error = error
        self.open_calls: list[OpenCall] = []

    @contextmanager
    def open_object(self, key: str, if_match: str | None) -> Iterator[ObjectDownload]:
        self.open_calls.append(OpenCall(key, if_match))
        if self.error is not None:
            raise self.error
        yield ObjectDownload(
            body=BytesIO(self.body),
            size=self.response_size,
            etag=self.response_etag,
            etag_is_weak=self.response_etag_is_weak,
            last_modified=self.response_mtime,
        )


class ConcurrentStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self.active = 0
        self.maximum_active = 0
        self.open_calls: list[OpenCall] = []

    @contextmanager
    def open_object(self, key: str, if_match: str | None) -> Iterator[ObjectDownload]:
        with self._lock:
            self.open_calls.append(OpenCall(key, if_match))
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            sleep(0.03)
            body = key.encode("utf-8")
            yield ObjectDownload(
                body=BytesIO(body),
                size=len(body),
                etag="etag",
                etag_is_weak=False,
                last_modified=FIXED_MTIME,
            )
        finally:
            with self._lock:
                self.active -= 1


def remote_object(
    key: str = "42/chart.dtx",
    *,
    size: int = 5,
    etag: str = "etag",
    etag_is_weak: bool = False,
    errors: tuple[SyncError, ...] = (),
) -> RemoteObject:
    return RemoteObject(
        key=key,
        size=size,
        etag=etag,
        etag_is_weak=etag_is_weak,
        last_modified=FIXED_MTIME,
        content_type="text/plain",
        errors=errors,
    )


def simfile(*objects: RemoteObject) -> SimfileInventory:
    return SimfileInventory(42, "42/", tuple(objects), "complete")


def config(*, download_concurrency: int = 4) -> R2Config:
    return R2Config(
        "https://example.invalid",
        "a" * 64,
        "simfile-dtx",
        download_concurrency=download_concurrency,
    )


def empty_index(cache_dir: Path) -> CacheIndexStore:
    return CacheIndexStore.load(cache_dir)


def seeded_index(
    cache_dir: Path,
    remote: RemoteObject,
    cache_path: Path,
    *,
    body: bytes = b"chart",
) -> CacheIndexStore:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(body)
    digest = sha256(body).hexdigest()
    store = CacheIndexStore.load(cache_dir)
    store.checkpoint(
        CacheIndexEntry(
            source_endpoint_sha256="a" * 64,
            bucket="simfile-dtx",
            key=remote.key,
            etag=remote.etag,
            etag_is_weak=remote.etag_is_weak,
            size=len(body),
            last_modified=format_manifest_timestamp(remote.last_modified),
            sha256=digest,
            cache_path=f"sha256/{digest[:2]}/{digest}",
        )
    )
    return store


def index_with_local_state(cache_dir: Path, local_state: str) -> CacheIndexStore:
    body = b"chart"
    digest = sha256(body).hexdigest()
    cache_path = cache_dir / "sha256" / digest[:2] / digest
    store = seeded_index(cache_dir, remote_object(), cache_path)
    if local_state == "missing":
        cache_path.unlink()
    elif local_state == "size_mismatch":
        cache_path.write_bytes(b"x")
    elif local_state == "sha256_mismatch":
        cache_path.write_bytes(b"wrong")
    else:
        raise AssertionError(f"unknown state: {local_state}")
    return store


def entry(
    key: str = "42/SET.DEF",
    *,
    endpoint_hash: str = "a" * 64,
    bucket: str = "simfile-dtx",
    sha256: str = "b" * 64,
) -> CacheIndexEntry:
    return CacheIndexEntry(
        source_endpoint_sha256=endpoint_hash,
        bucket=bucket,
        key=key,
        etag="etag",
        etag_is_weak=False,
        size=5,
        last_modified="2026-07-25T00:00:00Z",
        sha256=sha256,
        cache_path=f"sha256/{sha256[:2]}/{sha256}",
    )


def write_index(tmp_path: Path, entries: list[dict[str, object]]) -> None:
    (tmp_path / "index-v1.json").write_text(
        json.dumps({"schema_version": "crux.r2-cache-index/v1", "entries": entries}),
        encoding="utf-8",
    )


def test_index_checkpoint_is_canonical_and_restart_readable(tmp_path: Path) -> None:
    store = CacheIndexStore.load(tmp_path)
    store.checkpoint(entry("42/z.dtx"))
    store.checkpoint(entry("42/a.dtx"))

    payload = json.loads((tmp_path / "index-v1.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "crux.r2-cache-index/v1"
    assert [item["key"] for item in payload["entries"]] == ["42/a.dtx", "42/z.dtx"]
    restarted = CacheIndexStore.load(tmp_path)
    assert restarted.get("a" * 64, "simfile-dtx", "42/a.dtx") is not None


def test_checkpoint_sorts_by_endpoint_bucket_and_exact_key(tmp_path: Path) -> None:
    store = CacheIndexStore.load(tmp_path)
    store.checkpoint(entry("42/a", endpoint_hash="b" * 64))
    store.checkpoint(entry("42/z", bucket="z-bucket"))
    store.checkpoint(entry("42/z"))
    store.checkpoint(entry("42/a"))

    payload = json.loads((tmp_path / "index-v1.json").read_text(encoding="utf-8"))
    assert [
        (item["source_endpoint_sha256"], item["bucket"], item["key"]) for item in payload["entries"]
    ] == [
        ("a" * 64, "simfile-dtx", "42/a"),
        ("a" * 64, "simfile-dtx", "42/z"),
        ("a" * 64, "z-bucket", "42/z"),
        ("b" * 64, "simfile-dtx", "42/a"),
    ]


def test_checkpoint_contains_only_canonical_index_entry_fields(tmp_path: Path) -> None:
    store = CacheIndexStore.load(tmp_path)
    store.checkpoint(entry())

    payload = json.loads((tmp_path / "index-v1.json").read_text(encoding="utf-8"))
    assert set(payload["entries"][0]) == {
        "source_endpoint_sha256",
        "bucket",
        "key",
        "etag",
        "etag_is_weak",
        "size",
        "last_modified",
        "sha256",
        "cache_path",
    }


def test_invalid_index_json_and_schema_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "index-v1.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cache index"):
        CacheIndexStore.load(tmp_path)
    (tmp_path / "index-v1.json").write_text(
        '{"schema_version":"crux.r2-cache-index/v2","entries":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema"):
        CacheIndexStore.load(tmp_path)


@pytest.mark.parametrize(
    "cache_path",
    [
        "/sha256/bb/" + "b" * 64,
        "sha256/../" + "b" * 64,
        "sha256//" + "b" * 64,
        "sha256/./" + "b" * 64,
    ],
)
def test_load_rejects_unsafe_cache_paths(tmp_path: Path, cache_path: str) -> None:
    raw_entry = entry().__dict__ | {"cache_path": cache_path}
    write_index(tmp_path, [raw_entry])

    with pytest.raises(ValueError, match="cache index"):
        CacheIndexStore.load(tmp_path)


def test_checkpoint_rejects_an_entry_with_an_unsafe_cache_path(tmp_path: Path) -> None:
    unsafe_entry = replace(entry(), cache_path="sha256/../outside")

    with pytest.raises(ValueError, match="cache index"):
        CacheIndexStore.load(tmp_path).checkpoint(unsafe_entry)
    assert not (tmp_path / "index-v1.json").exists()


def test_load_rejects_duplicate_composite_identity(tmp_path: Path) -> None:
    raw_entry = entry().__dict__
    write_index(tmp_path, [raw_entry, raw_entry])

    with pytest.raises(ValueError, match="duplicate"):
        CacheIndexStore.load(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_endpoint_sha256", "A" * 64),
        ("sha256", "b" * 63),
        ("size", True),
        ("size", -1),
        ("etag_is_weak", 0),
        ("last_modified", "2026-07-25T00:00:00+00:00"),
        ("key", 42),
    ],
)
def test_load_rejects_noncanonical_or_wrongly_typed_entry_values(
    tmp_path: Path, field: str, value: object
) -> None:
    raw_entry = entry().__dict__ | {field: value}
    write_index(tmp_path, [raw_entry])

    with pytest.raises(ValueError, match="cache index"):
        CacheIndexStore.load(tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "source_endpoint_sha256",
        "bucket",
        "key",
        "etag",
        "last_modified",
        "sha256",
        "cache_path",
    ],
)
def test_load_rejects_non_utf8_serializable_entry_strings(tmp_path: Path, field: str) -> None:
    raw_entry = entry().__dict__ | {field: "\ud800"}
    write_index(tmp_path, [raw_entry])

    with pytest.raises(ValueError, match="invalid cache index entry"):
        CacheIndexStore.load(tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "source_endpoint_sha256",
        "bucket",
        "key",
        "etag",
        "last_modified",
        "sha256",
        "cache_path",
    ],
)
def test_checkpoint_rejects_non_utf8_serializable_entry_strings(tmp_path: Path, field: str) -> None:
    poisoned_entry = replace(entry(), **{field: "\ud800"})
    store = CacheIndexStore.load(tmp_path)

    with pytest.raises(ValueError, match="invalid cache index entry"):
        store.checkpoint(poisoned_entry)
    assert store.get("a" * 64, "simfile-dtx", "42/SET.DEF") is None
    assert not (tmp_path / "index-v1.json").exists()


def test_load_normalizes_huge_json_integer_errors_without_echoing_input(tmp_path: Path) -> None:
    huge_integer = "9" * 5_000
    (tmp_path / "index-v1.json").write_text(
        '{"schema_version":"crux.r2-cache-index/v1","entries":[{"size":' + huge_integer + "}]}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as caught:
        CacheIndexStore.load(tmp_path)
    assert str(caught.value) == "invalid cache index JSON"
    assert huge_integer not in str(caught.value)


def test_checkpoint_flushes_file_before_replacing_and_syncing_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    events: list[str] = []
    writer_fds: set[int] = set()
    directory_fds: set[int] = set()
    real_path_open = Path.open
    real_os_open = os.open
    real_fsync = os.fsync
    real_replace = os.replace

    class TrackingWriter:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            writer_fds.add(wrapped.fileno())

        def __enter__(self):
            self._wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self._wrapped.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def flush(self):
            events.append("file_flush")
            return self._wrapped.flush()

    def path_open(self: Path, *args, **kwargs):
        opened = real_path_open(self, *args, **kwargs)
        if self.name.startswith(".index-v1.json."):
            return TrackingWriter(opened)
        return opened

    def directory_open(path, flags, *args):
        if Path(path) == tmp_path:
            events.append("directory_open")
        descriptor = real_os_open(path, flags, *args)
        if Path(path) == tmp_path:
            directory_fds.add(descriptor)
        return descriptor

    def fsync(file_descriptor: int) -> None:
        if file_descriptor in directory_fds:
            events.append("directory_fsync")
        elif file_descriptor in writer_fds:
            events.append("file_fsync")
        real_fsync(file_descriptor)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(cache.os, "open", directory_open)
    monkeypatch.setattr(cache.os, "fsync", fsync)
    monkeypatch.setattr(
        cache.os,
        "replace",
        lambda source, target: events.append("replace") or real_replace(source, target),
    )

    CacheIndexStore.load(tmp_path).checkpoint(entry())

    assert events.index("file_flush") < events.index("file_fsync") < events.index("replace")
    assert (
        events.index("replace") < events.index("directory_open") < events.index("directory_fsync")
    )


def test_checkpoint_keeps_a_preexisting_temp_sibling_when_creation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    temporary_path = tmp_path / ".index-v1.json.collision.tmp"
    temporary_path.write_bytes(b"not created by this writer")
    monkeypatch.setattr(cache, "uuid4", lambda: type("Uuid", (), {"hex": "collision"})())

    with pytest.raises(FileExistsError):
        CacheIndexStore.load(tmp_path).checkpoint(entry())

    assert temporary_path.read_bytes() == b"not created by this writer"


def test_checkpoint_keeps_last_durable_snapshot_when_publication_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    store = CacheIndexStore.load(tmp_path)
    durable_entry = entry("42/durable.dtx")
    failed_entry = entry("42/failed.dtx")
    store.checkpoint(durable_entry)
    monkeypatch.setattr(
        cache.os,
        "replace",
        lambda *_: (_ for _ in ()).throw(OSError("private")),
    )

    with pytest.raises(OSError):
        store.checkpoint(failed_entry)

    assert store.get("a" * 64, "simfile-dtx", "42/durable.dtx") == durable_entry
    assert store.get("a" * 64, "simfile-dtx", "42/failed.dtx") is None
    assert [
        item["key"] for item in json.loads((tmp_path / "index-v1.json").read_text())["entries"]
    ] == ["42/durable.dtx"]


def test_get_waits_for_a_checkpoint_to_be_durable_before_exposing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    store = CacheIndexStore.load(tmp_path)
    replacement_started = Event()
    allow_replacement = Event()
    reader_started = Event()
    get_finished = Event()
    observed: list[CacheIndexEntry | None] = []
    real_replace = cache.os.replace

    def blocking_replace(source, target) -> None:
        replacement_started.set()
        assert allow_replacement.wait(1)
        real_replace(source, target)

    monkeypatch.setattr(cache.os, "replace", blocking_replace)
    checkpoint_thread = Thread(target=store.checkpoint, args=(entry("42/pending.dtx"),))
    checkpoint_thread.start()
    assert replacement_started.wait(1)

    def read_entry() -> None:
        reader_started.set()
        observed.append(store.get("a" * 64, "simfile-dtx", "42/pending.dtx"))
        get_finished.set()

    reader_thread = Thread(target=read_entry)
    reader_thread.start()
    assert reader_started.wait(1)
    assert not get_finished.wait(0.05)
    allow_replacement.set()
    checkpoint_thread.join()
    reader_thread.join()

    assert observed == [entry("42/pending.dtx")]


def test_live_writer_lock_fails_fast(tmp_path: Path) -> None:
    with cache_writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match="cache_locked"):
            with cache_writer_lock(tmp_path):
                raise AssertionError("second writer must not enter")


def test_writer_lock_retains_one_open_lock_file_for_entire_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    real_fdopen = os.fdopen
    opened = []

    def fdopen(descriptor: int, *args, **kwargs):
        handle = real_fdopen(descriptor, *args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(cache.os, "fdopen", fdopen)

    with cache_writer_lock(tmp_path):
        assert len(opened) == 1
        assert not opened[0].closed
    assert len(opened) == 1
    assert opened[0].closed


@pytest.mark.parametrize("error_number", [errno.EAGAIN, errno.EWOULDBLOCK])
def test_writer_lock_maps_busy_flock_errnos_to_cache_locked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_number: int
) -> None:
    import fcntl

    def busy(*_) -> None:
        raise OSError(error_number, "private lock detail")

    monkeypatch.setattr(fcntl, "flock", busy)

    with pytest.raises(RuntimeError) as caught:
        with cache_writer_lock(tmp_path):
            raise AssertionError("busy writer must not enter")
    assert str(caught.value) == "cache_locked"


@pytest.mark.parametrize(
    "error_number",
    sorted({errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}),
)
def test_writer_lock_maps_unsupported_flock_errnos_to_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_number: int
) -> None:
    import fcntl

    monkeypatch.setattr(
        fcntl,
        "flock",
        lambda *_: (_ for _ in ()).throw(OSError(error_number, "private lock detail")),
    )

    with pytest.raises(RuntimeError) as caught:
        with cache_writer_lock(tmp_path):
            raise AssertionError("unsupported platform must not enter")
    assert str(caught.value) == "unsupported_platform"


def test_writer_lock_sanitizes_unexpected_open_and_flock_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import fcntl

    import src.benchmark.corpus_cache as cache

    real_os_open = cache.os.open
    monkeypatch.setattr(
        cache.os,
        "open",
        lambda *_: (_ for _ in ()).throw(OSError(errno.EACCES, "private lock path")),
    )
    with pytest.raises(RuntimeError) as caught:
        with cache_writer_lock(tmp_path):
            raise AssertionError("failed lock open must not enter")
    assert str(caught.value) == "cache_lock_failed"
    assert "private" not in str(caught.value)

    monkeypatch.setattr(cache.os, "open", real_os_open)
    monkeypatch.setattr(
        fcntl,
        "flock",
        lambda *_: (_ for _ in ()).throw(OSError(errno.EACCES, "private flock detail")),
    )
    with pytest.raises(RuntimeError) as caught:
        with cache_writer_lock(tmp_path):
            raise AssertionError("failed flock must not enter")
    assert str(caught.value) == "cache_lock_failed"
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("error_number", [errno.EAGAIN, errno.EWOULDBLOCK])
def test_writer_lock_does_not_treat_busy_open_as_a_live_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_number: int
) -> None:
    import src.benchmark.corpus_cache as cache

    monkeypatch.setattr(
        cache.os,
        "open",
        lambda *_: (_ for _ in ()).throw(OSError(error_number, "private lock path")),
    )

    with pytest.raises(RuntimeError) as caught:
        with cache_writer_lock(tmp_path):
            raise AssertionError("failed lock open must not enter")
    assert str(caught.value) == "cache_lock_failed"
    assert "private" not in str(caught.value)


def test_writer_lock_rejects_a_preexisting_symlink(tmp_path: Path) -> None:
    target = tmp_path / "private-lock-target"
    target.write_bytes(b"do not follow")
    (tmp_path / ".index-v1.lock").symlink_to(target)

    with pytest.raises(RuntimeError) as caught:
        with cache_writer_lock(tmp_path):
            raise AssertionError("symlinked lock must not enter")
    assert str(caught.value) == "cache_lock_failed"
    assert target.read_bytes() == b"do not follow"


def test_writer_lock_reports_unsupported_platform_without_mutating_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_import = builtins.__import__

    def import_without_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("not available")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_fcntl)

    with pytest.raises(RuntimeError, match="unsupported_platform"):
        with cache_writer_lock(tmp_path):
            raise AssertionError("unsupported platform must not enter")
    assert not (tmp_path / ".index-v1.lock").exists()


def test_checkpoint_serializes_concurrent_publications(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = CacheIndexStore.load(tmp_path)
    original_publish = store._publish_locked
    active_writers = 0
    maximum_active_writers = 0

    def publish(entries: dict[tuple[str, str, str], CacheIndexEntry]) -> None:
        nonlocal active_writers, maximum_active_writers
        active_writers += 1
        maximum_active_writers = max(maximum_active_writers, active_writers)
        sleep(0.02)
        try:
            original_publish(entries)
        finally:
            active_writers -= 1

    monkeypatch.setattr(store, "_publish_locked", publish)
    threads = [
        Thread(target=store.checkpoint, args=(entry(f"42/{suffix}"),))
        for suffix in ("one.dtx", "two.dtx")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert maximum_active_writers == 1
    assert [item.key for item in CacheIndexStore.load(tmp_path)._entries.values()] == [
        "42/one.dtx",
        "42/two.dtx",
    ]


def test_dry_run_selects_only_setdef_dtx_and_txt_without_mutating_cache(
    tmp_path: Path,
) -> None:
    objects = (
        remote_object("42/SET.DEF"),
        remote_object("42/charts/MAS.DTX"),
        remote_object("42/notes.TxT"),
        remote_object("42/audio/song.ogg"),
        remote_object("42/assets/", size=0),
    )
    store = FakeStore()

    result = sync_cache((simfile(*objects),), store, empty_index(tmp_path), config(), True)

    assert [action.object_key for action in result.actions] == [
        "42/SET.DEF",
        "42/charts/MAS.DTX",
        "42/notes.TxT",
    ]
    assert [action.action for action in result.actions] == ["planned", "planned", "planned"]
    assert result.simfiles == (simfile(*objects),)
    assert store.open_calls == []
    assert list(tmp_path.iterdir()) == []


def test_verified_hit_reads_no_remote_body(tmp_path: Path) -> None:
    body = b"chart"
    digest = sha256(body).hexdigest()
    cache_path = tmp_path / "sha256" / digest[:2] / digest
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
    assert result.simfiles[0].objects[0].sha256 == digest
    assert result.simfiles[0].objects[0].cache_path == f"sha256/{digest[:2]}/{digest}"
    assert result.actions[0].action == "cache_hit"
    assert result.actions[0].miss_reason is None


@pytest.mark.parametrize("local_state", ["missing", "size_mismatch", "sha256_mismatch"])
def test_dry_run_reports_repair_as_planned_without_get(tmp_path: Path, local_state: str) -> None:
    index = index_with_local_state(tmp_path, local_state)
    store = FakeStore()

    result = sync_cache((simfile(remote_object()),), store, index, config(), True)

    assert store.open_calls == []
    assert result.actions[0].action == "planned"
    assert result.actions[0].miss_reason == local_state
    assert result.simfiles == (simfile(remote_object()),)


def test_remote_identity_mismatch_is_not_local_corruption(tmp_path: Path) -> None:
    body = b"chart"
    digest = sha256(body).hexdigest()
    indexed_remote = remote_object(etag="old")
    cache_path = tmp_path / "sha256" / digest[:2] / digest
    index = seeded_index(tmp_path, indexed_remote, cache_path)

    result = sync_cache(
        (simfile(remote_object(etag="new")),),
        FakeStore(),
        index,
        config(),
        True,
    )

    assert result.actions[0].action == "planned"
    assert result.actions[0].miss_reason == "remote_changed"
    assert result.actions[0].errors == ()


def test_strong_etag_passes_opaque_if_match_and_weak_etag_passes_none(
    tmp_path: Path,
) -> None:
    strong_store = FakeStore()
    sync_cache(
        (simfile(remote_object(etag_is_weak=False)),),
        strong_store,
        empty_index(tmp_path / "strong"),
        config(),
        False,
    )
    assert strong_store.open_calls == [OpenCall("42/chart.dtx", "etag")]

    weak_store = FakeStore(response_etag_is_weak=True)
    sync_cache(
        (simfile(remote_object(etag_is_weak=True)),),
        weak_store,
        empty_index(tmp_path / "weak"),
        config(),
        False,
    )
    assert weak_store.open_calls == [OpenCall("42/chart.dtx", None)]


def test_selected_object_with_head_error_fails_without_get(tmp_path: Path) -> None:
    head_error = SyncError(
        "object",
        "object_head_failed",
        "Object metadata inspection failed.",
        "42/chart.dtx",
    )
    source = remote_object(errors=(head_error,))
    store = FakeStore()

    result = sync_cache((simfile(source),), store, empty_index(tmp_path), config(), False)

    assert store.open_calls == []
    assert result.simfiles[0].objects[0].cache_status == "failed"
    assert result.simfiles[0].objects[0].errors == (head_error,)
    assert result.simfiles[0].sync_status == "failed"
    assert result.simfiles[0].sync_errors == (head_error,)
    assert result.actions[0].action == "failed"
    assert result.actions[0].errors == (head_error,)


@pytest.mark.parametrize(
    "missing_field",
    ["etag", "etag_is_weak", "size", "last_modified"],
)
def test_weak_response_missing_identity_metadata_fails_closed(
    tmp_path: Path, missing_field: str
) -> None:
    arguments = {
        "response_etag": "etag",
        "response_etag_is_weak": True,
        "response_size": 5,
        "response_mtime": FIXED_MTIME,
    }
    arguments[
        {
            "etag": "response_etag",
            "etag_is_weak": "response_etag_is_weak",
            "size": "response_size",
            "last_modified": "response_mtime",
        }[missing_field]
    ] = None

    result = sync_cache(
        (simfile(remote_object(etag_is_weak=True)),),
        FakeStore(**arguments),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert result.simfiles[0].objects[0].cache_status == "failed"
    assert [error.code for error in result.actions[0].errors] == ["weak_etag_unverifiable"]
    assert "example.invalid" not in result.actions[0].errors[0].message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("response_etag", "changed"),
        ("response_etag_is_weak", False),
        ("response_size", 6),
        ("response_mtime", datetime(2026, 7, 26, tzinfo=timezone.utc)),
    ],
)
def test_weak_response_identity_drift_fails_as_source_changed(
    tmp_path: Path, field: str, value: object
) -> None:
    arguments = {
        "response_etag": "etag",
        "response_etag_is_weak": True,
        "response_size": 5,
        "response_mtime": FIXED_MTIME,
    }
    arguments[field] = value

    result = sync_cache(
        (simfile(remote_object(etag_is_weak=True)),),
        FakeStore(**arguments),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert [error.code for error in result.actions[0].errors] == ["source_changed_during_sync"]


def test_strong_conditional_failure_uses_sanitized_domain_error(tmp_path: Path) -> None:
    store = FakeStore(
        error=R2StoreError(
            "source_changed_during_sync",
            "private SDK response and request ID",
            "42/chart.dtx",
        )
    )

    result = sync_cache(
        (simfile(remote_object()),),
        store,
        empty_index(tmp_path),
        config(),
        False,
    )

    error = result.actions[0].errors[0]
    assert error.code == "source_changed_during_sync"
    assert error.message == "Object metadata changed after inventory."
    assert "private" not in error.message


def test_streamed_byte_count_mismatch_rejects_download(tmp_path: Path) -> None:
    result = sync_cache(
        (simfile(remote_object(size=5)),),
        FakeStore(body=b"x", response_size=5),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert result.actions[0].action == "failed"
    assert [error.code for error in result.actions[0].errors] == ["byte_count_mismatch"]
    assert list((tmp_path / "sha256" / ".incoming").iterdir()) == []
    assert not (tmp_path / "index-v1.json").exists()


def test_failed_get_is_sanitized_and_does_not_checkpoint(tmp_path: Path) -> None:
    store = FakeStore(error=R2StoreError("object_get_failed", "private signed URL", "42/chart.dtx"))

    result = sync_cache(
        (simfile(remote_object()),),
        store,
        empty_index(tmp_path),
        config(),
        False,
    )

    assert [error.code for error in result.actions[0].errors] == ["object_get_failed"]
    assert result.actions[0].errors[0].message == "Object body read failed."
    assert not (tmp_path / "index-v1.json").exists()


def test_failed_local_body_write_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_path_open = Path.open

    def fail_incoming_write(path: Path, *args, **kwargs):
        if path.parent.name == ".incoming":
            raise OSError("private local path")
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_incoming_write)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert [error.code for error in result.actions[0].errors] == ["artifact_write_failed"]
    assert "private" not in result.actions[0].errors[0].message


@pytest.mark.parametrize("local_state", ["missing", "size_mismatch", "sha256_mismatch"])
def test_successful_repair_does_not_report_cache_corrupt(tmp_path: Path, local_state: str) -> None:
    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        index_with_local_state(tmp_path, local_state),
        config(),
        False,
    )

    assert result.actions[0].action == "downloaded"
    assert result.actions[0].miss_reason == local_state
    assert result.actions[0].errors == ()
    assert result.simfiles[0].objects[0].cache_status == "verified"
    assert result.simfiles[0].objects[0].errors == ()


@pytest.mark.parametrize("local_state", ["missing", "size_mismatch", "sha256_mismatch"])
def test_failed_repair_reports_cache_corrupt_and_operational_error(
    tmp_path: Path, local_state: str
) -> None:
    store = FakeStore(
        error=R2StoreError("object_get_failed", "private SDK details", "42/chart.dtx")
    )

    result = sync_cache(
        (simfile(remote_object()),),
        store,
        index_with_local_state(tmp_path, local_state),
        config(),
        False,
    )

    assert {error.code for error in result.actions[0].errors} == {
        "cache_corrupt",
        "object_get_failed",
    }
    assert result.simfiles[0].sync_status == "failed"


def test_failed_remote_changed_download_has_no_cache_corrupt_error(
    tmp_path: Path,
) -> None:
    store = FakeStore(
        error=R2StoreError("object_get_failed", "private SDK details", "42/chart.dtx")
    )

    result = sync_cache(
        (simfile(remote_object()),),
        store,
        empty_index(tmp_path),
        config(),
        False,
    )

    assert [error.code for error in result.actions[0].errors] == ["object_get_failed"]


def test_body_temporary_file_is_created_only_under_cache_incoming(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened_temporary_paths: list[Path] = []
    real_path_open = Path.open

    def track_open(path: Path, *args, **kwargs):
        if args and args[0] == "xb" and path.parent.name == ".incoming":
            opened_temporary_paths.append(path)
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)

    sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert len(opened_temporary_paths) == 1
    assert opened_temporary_paths[0].parent == tmp_path / "sha256" / ".incoming"


def test_cross_device_incoming_and_shard_fail_before_body_replace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    real_stat = Path.stat
    real_replace = os.replace
    body_replacements: list[tuple[Path, Path]] = []

    class StatWithDifferentDevice:
        def __init__(self, wrapped, st_dev: int):
            self._wrapped = wrapped
            self.st_dev = st_dev

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def differing_stat(path: Path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path.parent.name == "sha256" and path.name != ".incoming":
            return StatWithDifferentDevice(result, result.st_dev + 1)
        return result

    def track_replace(source, target) -> None:
        if Path(source).parent.name == ".incoming":
            body_replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(Path, "stat", differing_stat)
    monkeypatch.setattr(cache.os, "replace", track_replace)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert [error.code for error in result.actions[0].errors] == ["artifact_write_failed"]
    assert body_replacements == []
    assert not (tmp_path / "index-v1.json").exists()


def test_body_is_synced_replaced_and_directory_synced_before_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    events: list[str] = []
    body_fds: set[int] = set()
    shard_fds: set[int] = set()
    real_path_open = Path.open
    real_os_open = os.open
    real_fsync = os.fsync
    real_replace = os.replace
    index = empty_index(tmp_path)
    real_checkpoint = index.checkpoint

    class TrackingBody:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            body_fds.add(wrapped.fileno())

        def __enter__(self):
            self._wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self._wrapped.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def path_open(path: Path, *args, **kwargs):
        opened = real_path_open(path, *args, **kwargs)
        if path.parent.name == ".incoming":
            return TrackingBody(opened)
        return opened

    def directory_open(path, flags, *args):
        descriptor = real_os_open(path, flags, *args)
        candidate = Path(path)
        if candidate.parent.name == "sha256" and candidate.name != ".incoming":
            shard_fds.add(descriptor)
        return descriptor

    def fsync(descriptor: int) -> None:
        if descriptor in shard_fds:
            events.append("shard_fsync")
        elif descriptor in body_fds:
            events.append("body_fsync")
        real_fsync(descriptor)

    def replace_body(source, target) -> None:
        if Path(source).parent.name == ".incoming":
            events.append("body_replace")
        real_replace(source, target)

    def checkpoint(index_entry: CacheIndexEntry) -> None:
        events.append("index_checkpoint")
        real_checkpoint(index_entry)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(cache.os, "open", directory_open)
    monkeypatch.setattr(cache.os, "fsync", fsync)
    monkeypatch.setattr(cache.os, "replace", replace_body)
    monkeypatch.setattr(index, "checkpoint", checkpoint)

    sync_cache((simfile(remote_object()),), FakeStore(), index, config(), False)

    assert events.index("body_fsync") < events.index("body_replace")
    assert events.index("body_replace") < events.index("shard_fsync")
    assert events.index("shard_fsync") < events.index("index_checkpoint")


def test_existing_identical_content_address_is_deduplicated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    body = b"chart"
    digest = sha256(body).hexdigest()
    final_path = tmp_path / "sha256" / digest[:2] / digest
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(body)
    real_replace = os.replace
    body_replacements: list[tuple[Path, Path]] = []

    def track_replace(source, target) -> None:
        if Path(source).parent.name == ".incoming":
            body_replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(cache.os, "replace", track_replace)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(body=body),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert result.actions[0].action == "downloaded"
    assert body_replacements == []
    assert final_path.read_bytes() == body


def test_existing_mismatched_content_address_is_rejected(tmp_path: Path) -> None:
    body = b"chart"
    digest = sha256(body).hexdigest()
    final_path = tmp_path / "sha256" / digest[:2] / digest
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"wrong")

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(body=body),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert [error.code for error in result.actions[0].errors] == ["artifact_write_failed"]
    assert final_path.read_bytes() == b"wrong"
    assert not (tmp_path / "index-v1.json").exists()


def test_preexisting_incoming_collision_is_never_deleted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    incoming_dir = tmp_path / "sha256" / ".incoming"
    incoming_dir.mkdir(parents=True)
    collision = incoming_dir / "collision.tmp"
    collision.write_bytes(b"owned by another invocation")
    monkeypatch.setattr(cache, "uuid4", lambda: type("Uuid", (), {"hex": "collision"})())

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert result.actions[0].action == "failed"
    assert collision.read_bytes() == b"owned by another invocation"


def test_failed_owned_temporary_cleanup_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_unlink = Path.unlink

    def fail_owned_cleanup(path: Path, *args, **kwargs):
        if path.parent.name == ".incoming":
            raise OSError("private cleanup path")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_owned_cleanup)

    result = sync_cache(
        (simfile(remote_object(size=5)),),
        FakeStore(body=b"x", response_size=5),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert {error.code for error in result.actions[0].errors} == {
        "artifact_write_failed",
        "byte_count_mismatch",
    }
    assert all("private" not in error.message for error in result.actions[0].errors)


def test_download_workers_are_bounded_and_results_remain_input_ordered(
    tmp_path: Path,
) -> None:
    objects = tuple(
        remote_object(key, size=len(key.encode("utf-8")))
        for key in (
            "42/one.dtx",
            "42/two.dtx",
            "42/three.dtx",
            "42/four.dtx",
            "42/five.dtx",
        )
    )
    store = ConcurrentStore()

    result = sync_cache(
        (simfile(*objects),),
        store,
        empty_index(tmp_path),
        config(download_concurrency=2),
        False,
    )

    assert store.maximum_active == 2
    assert [action.object_key for action in result.actions] == [remote.key for remote in objects]
    assert all(action.action == "downloaded" for action in result.actions)


def test_restart_after_checkpoint_becomes_verified_hit(tmp_path: Path) -> None:
    first_store = FakeStore()
    first = sync_cache(
        (simfile(remote_object()),),
        first_store,
        empty_index(tmp_path),
        config(),
        False,
    )
    restarted_index = CacheIndexStore.load(tmp_path)
    second_store = FakeStore(
        error=R2StoreError("object_get_failed", "must not be read", "42/chart.dtx")
    )

    second = sync_cache(
        (simfile(remote_object()),),
        second_store,
        restarted_index,
        config(),
        False,
    )

    assert first.actions[0].action == "downloaded"
    assert second.actions[0].action == "cache_hit"
    assert second_store.open_calls == []
    assert first.simfiles == second.simfiles


def test_indexed_symlinked_body_is_not_a_verified_hit(tmp_path: Path) -> None:
    body = b"chart"
    digest = sha256(body).hexdigest()
    external = tmp_path / "external-body"
    external.write_bytes(body)
    cache_path = tmp_path / "sha256" / digest[:2] / digest
    index = seeded_index(tmp_path, remote_object(), cache_path)
    cache_path.unlink()
    cache_path.symlink_to(external)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        index,
        config(),
        True,
    )

    assert result.actions[0].action == "planned"
    assert result.actions[0].miss_reason == "sha256_mismatch"
    assert external.read_bytes() == body


@pytest.mark.parametrize("symlink_component", ["sha256", "incoming", "shard", "final"])
def test_symlinked_cache_content_components_fail_closed_without_external_mutation(
    tmp_path: Path, symlink_component: str
) -> None:
    body = b"chart"
    digest = sha256(body).hexdigest()
    external = tmp_path / "external"
    external.mkdir()
    sha_dir = tmp_path / "sha256"

    if symlink_component == "sha256":
        sha_dir.symlink_to(external, target_is_directory=True)
    else:
        sha_dir.mkdir()
        if symlink_component == "incoming":
            (sha_dir / ".incoming").symlink_to(external, target_is_directory=True)
        elif symlink_component == "shard":
            (sha_dir / digest[:2]).symlink_to(external, target_is_directory=True)
        else:
            shard = sha_dir / digest[:2]
            shard.mkdir()
            target = external / "body"
            target.write_bytes(body)
            (shard / digest).symlink_to(target)
    before = sorted(path.relative_to(external) for path in external.rglob("*"))

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(body=body),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert result.actions[0].action == "failed"
    assert [error.code for error in result.actions[0].errors] == ["artifact_write_failed"]
    assert sorted(path.relative_to(external) for path in external.rglob("*")) == before
    assert not (tmp_path / "index-v1.json").exists()


def test_one_failed_selected_object_makes_otherwise_complete_row_partial(
    tmp_path: Path,
) -> None:
    head_error = SyncError(
        "object",
        "object_head_failed",
        "Object metadata inspection failed.",
        "42/bad.dtx",
    )
    failed = remote_object("42/bad.dtx", errors=(head_error,))
    verified = remote_object("42/good.dtx")
    body = b"chart"
    digest = sha256(body).hexdigest()
    index = seeded_index(
        tmp_path,
        verified,
        tmp_path / "sha256" / digest[:2] / digest,
    )

    result = sync_cache(
        (simfile(failed, verified),),
        FakeStore(),
        index,
        config(),
        False,
    )

    assert result.simfiles[0].sync_status == "partial"
    assert [remote.cache_status for remote in result.simfiles[0].objects] == [
        "failed",
        "verified",
    ]
    assert result.simfiles[0].sync_errors == (head_error,)


def test_empty_row_and_prior_inventory_errors_are_preserved(tmp_path: Path) -> None:
    empty_error = SyncError(
        "simfile",
        "empty_prefix",
        "A requested prefix has no objects or only folder markers.",
        "42/",
    )
    source = SimfileInventory(
        42,
        "42/",
        (remote_object("42/assets/", size=0),),
        "empty",
        (empty_error,),
    )

    result = sync_cache((source,), FakeStore(), empty_index(tmp_path), config(), False)

    assert result.simfiles == (source,)
    assert result.actions == ()


def test_checkpoint_failure_never_indexes_body_and_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = empty_index(tmp_path)
    monkeypatch.setattr(
        index,
        "checkpoint",
        lambda _: (_ for _ in ()).throw(OSError("private index path")),
    )
    body = b"chart"
    digest = sha256(body).hexdigest()
    final_path = tmp_path / "sha256" / digest[:2] / digest

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(body=body),
        index,
        config(),
        False,
    )

    assert [error.code for error in result.actions[0].errors] == ["artifact_write_failed"]
    assert "private" not in result.actions[0].errors[0].message
    assert final_path.read_bytes() == body
    assert index.get("a" * 64, "simfile-dtx", "42/chart.dtx") is None
    assert not (tmp_path / "index-v1.json").exists()


def test_downloaded_body_uses_extensionless_content_address(tmp_path: Path) -> None:
    body = b"chart"
    digest = sha256(body).hexdigest()

    result = sync_cache(
        (simfile(remote_object("42/unsafe name.TxT")),),
        FakeStore(body=body),
        empty_index(tmp_path),
        config(),
        False,
    )

    remote = result.simfiles[0].objects[0]
    assert remote.cache_path == f"sha256/{digest[:2]}/{digest}"
    assert (tmp_path / remote.cache_path).read_bytes() == body
    assert (tmp_path / remote.cache_path).suffix == ""


def test_unreadable_indexed_body_is_a_repairable_validation_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    body = b"chart"
    digest = sha256(body).hexdigest()
    cache_path = tmp_path / "sha256" / digest[:2] / digest
    index = seeded_index(tmp_path, remote_object(), cache_path)
    monkeypatch.setattr(
        cache,
        "_hash_file",
        lambda _: (_ for _ in ()).throw(OSError("private local path")),
    )

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        index,
        config(),
        True,
    )

    assert result.actions[0].action == "planned"
    assert result.actions[0].miss_reason == "sha256_mismatch"
    assert result.actions[0].errors == ()


def test_repeated_local_failure_code_is_reported_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    real_replace = os.replace
    real_unlink = Path.unlink

    def fail_body_replace(source, target) -> None:
        if Path(source).parent.name == ".incoming":
            raise OSError("private replace path")
        real_replace(source, target)

    def fail_owned_cleanup(path: Path, *args, **kwargs):
        if path.parent.name == ".incoming":
            raise OSError("private cleanup path")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(cache.os, "replace", fail_body_replace)
    monkeypatch.setattr(Path, "unlink", fail_owned_cleanup)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert [error.code for error in result.actions[0].errors] == ["artifact_write_failed"]
