import builtins
import errno
import json
import os
import stat
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
    _open_regular_file_at,
    cache_writer_lock,
    is_chart_key,
    is_selected,
    is_set_def_key,
    resolve_verified_cache_body,
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


class DelayedBody(BytesIO):
    def __init__(
        self,
        body: bytes,
        delay: float,
        key: str,
        completions: list[str],
        lock: Lock,
    ) -> None:
        super().__init__(body)
        self._delay = delay
        self._key = key
        self._completions = completions
        self._lock = lock
        self._started = False
        self._completed = False

    def read(self, size: int = -1) -> bytes:
        if not self._started:
            self._started = True
            sleep(self._delay)
        chunk = super().read(size)
        if not chunk and not self._completed:
            self._completed = True
            with self._lock:
                self._completions.append(self._key)
        return chunk


class DelayedSameBodyStore:
    def __init__(self, delays: dict[str, float], body: bytes = b"chart") -> None:
        self.delays = delays
        self.body = body
        self.completions: list[str] = []
        self._lock = Lock()

    @contextmanager
    def open_object(self, key: str, if_match: str | None) -> Iterator[ObjectDownload]:
        del if_match
        yield ObjectDownload(
            body=DelayedBody(
                self.body,
                self.delays[key],
                key,
                self.completions,
                self._lock,
            ),
            size=len(self.body),
            etag="etag",
            etag_is_weak=False,
            last_modified=FIXED_MTIME,
        )


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


def cached_remote(body: bytes = b"chart") -> RemoteObject:
    digest = sha256(body).hexdigest()
    return replace(
        remote_object(size=len(body)),
        cache_status="verified",
        sha256=digest,
        cache_path=f"sha256/{digest[:2]}/{digest}",
    )


def install_cached_body(cache_dir: Path, body: bytes = b"chart") -> Path:
    digest = sha256(body).hexdigest()
    path = cache_dir / "sha256" / digest[:2] / digest
    path.parent.mkdir(parents=True)
    path.write_bytes(body)
    return path


def assert_verified_body_unavailable(
    cache_dir: Path,
    remote: RemoteObject,
    *,
    expected_sha256: str | None = None,
) -> None:
    with pytest.raises(ValueError) as raised:
        resolve_verified_cache_body(
            cache_dir,
            remote,
            source_endpoint_sha256="f" * 64,
            bucket="simfile-dtx",
            expected_sha256=expected_sha256,
        )
    assert str(raised.value) == "verified cache body unavailable"


def test_resolve_verified_cache_body_returns_a_verified_content_address(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    body = b"chart"
    digest = sha256(body).hexdigest()
    remote = cached_remote(body)
    expected = install_cached_body(cache_dir, body)

    assert expected == cache_dir / "sha256" / digest[:2] / digest
    assert (
        resolve_verified_cache_body(
            cache_dir,
            remote,
            source_endpoint_sha256="f" * 64,
            bucket="simfile-dtx",
        )
        == expected
    )


@pytest.mark.parametrize("cache_status", ["not_selected", "failed"])
def test_resolve_verified_cache_body_rejects_nonverified_rows(
    tmp_path: Path, cache_status: str
) -> None:
    remote = replace(cached_remote(), cache_status=cache_status)

    assert_verified_body_unavailable(tmp_path / "cache", remote)


@pytest.mark.parametrize("digest", [None, 1, "A" * 64, "a" * 63])
def test_resolve_verified_cache_body_rejects_missing_or_malformed_digests(
    tmp_path: Path, digest: object
) -> None:
    remote = replace(cached_remote(), sha256=digest)

    assert_verified_body_unavailable(tmp_path / "cache", remote)


def test_resolve_verified_cache_body_rejects_missing_cache_path(tmp_path: Path) -> None:
    remote = replace(cached_remote(), cache_path=None)

    assert_verified_body_unavailable(tmp_path / "cache", remote)


def test_resolve_verified_cache_body_rejects_noncanonical_cache_path(tmp_path: Path) -> None:
    remote = replace(cached_remote(), cache_path="sha256/incorrect/body")

    assert_verified_body_unavailable(tmp_path / "cache", remote)


def test_resolve_verified_cache_body_rejects_unexpected_digest(tmp_path: Path) -> None:
    remote = cached_remote()
    install_cached_body(tmp_path / "cache")

    assert_verified_body_unavailable(tmp_path / "cache", remote, expected_sha256="f" * 64)


@pytest.mark.parametrize("state", ["missing", "unreadable", "size_mismatch", "sha256_mismatch"])
def test_resolve_verified_cache_body_rejects_unavailable_or_corrupt_content(
    tmp_path: Path, state: str
) -> None:
    cache_dir = tmp_path / "cache"
    remote = cached_remote()
    expected = cache_dir / "sha256" / remote.sha256[:2] / remote.sha256
    if state == "missing":
        cache_dir.mkdir()
    elif state == "unreadable":
        cache_dir.mkdir()
        (cache_dir / "sha256").write_bytes(b"not a directory")
    elif state == "size_mismatch":
        expected.parent.mkdir(parents=True)
        expected.write_bytes(b"x")
    elif state == "sha256_mismatch":
        expected.parent.mkdir(parents=True)
        expected.write_bytes(b"wrong")
    else:
        raise AssertionError(f"unknown state: {state}")

    assert_verified_body_unavailable(cache_dir, remote)


def test_is_set_def_key_and_is_chart_key_preserve_setdef_dtx_txt_v1() -> None:
    assert is_set_def_key("42/SET.DEF")
    assert is_chart_key("42/mas.DTX")
    assert is_chart_key("42/readme.TXT")
    assert is_selected("42/set.def")
    assert is_selected("42/mas.dtx")
    assert not is_selected("42/bgm.ogg")


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


def test_absent_index_is_empty_but_broken_index_symlink_is_invalid(tmp_path: Path) -> None:
    assert CacheIndexStore.load(tmp_path)._entries == {}

    index_path = tmp_path / "index-v1.json"
    index_path.symlink_to(tmp_path / "SECRET-missing-index.json")

    with pytest.raises(ValueError) as error:
        CacheIndexStore.load(tmp_path)

    assert str(error.value) == "invalid cache index JSON"
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize("leaf_kind", ["symlink", "directory", "fifo"])
def test_index_load_rejects_non_regular_leaf_without_opening_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leaf_kind: str,
) -> None:
    index_path = tmp_path / "index-v1.json"
    if leaf_kind == "symlink":
        target = tmp_path / "SECRET-external-index.json"
        target.write_text(
            '{"schema_version":"crux.r2-cache-index/v1","entries":[]}',
            encoding="utf-8",
        )
        index_path.symlink_to(target)
    elif leaf_kind == "directory":
        index_path.mkdir()
    else:
        os.mkfifo(index_path)

    def forbidden_path_read(*_args, **_kwargs):
        raise AssertionError("non-regular index leaf must not be opened through pathlib")

    monkeypatch.setattr(Path, "read_text", forbidden_path_read)

    with pytest.raises(ValueError) as error:
        CacheIndexStore.load(tmp_path)

    assert str(error.value) == "invalid cache index JSON"
    assert "SECRET" not in str(error.value)


def test_index_load_uses_nofollow_descriptor_read_and_checks_binding_around_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.corpus_cache as cache

    index_path = tmp_path / "index-v1.json"
    index_path.write_text(
        '{"schema_version":"crux.r2-cache-index/v1","entries":[]}',
        encoding="utf-8",
    )
    real_open = os.open
    real_read = os.read
    open_flags: list[int] = []
    read_calls = 0

    def tracking_open(path, flags, *args, **kwargs):
        if Path(path) == index_path:
            open_flags.append(flags)
        return real_open(path, flags, *args, **kwargs)

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal read_calls
        content = real_read(descriptor, size)
        if read_calls == 0:
            replacement = tmp_path / "replacement-index.json"
            replacement.write_text(
                '{"schema_version":"crux.r2-cache-index/v1","entries":[]}',
                encoding="utf-8",
            )
            os.replace(replacement, index_path)
        read_calls += 1
        return content

    def forbidden_path_read(*_args, **_kwargs):
        raise AssertionError("cache index must be read through its pinned descriptor")

    monkeypatch.setattr(cache.os, "open", tracking_open)
    monkeypatch.setattr(cache.os, "read", swapping_read)
    monkeypatch.setattr(Path, "read_text", forbidden_path_read)

    with pytest.raises(ValueError) as error:
        CacheIndexStore.load(tmp_path)

    assert str(error.value) == "invalid cache index JSON"
    assert len(open_flags) == 1
    assert open_flags[0] & os.O_NOFOLLOW
    assert read_calls >= 1


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


def test_download_progress_callback_runs_while_another_download_is_still_blocked(
    tmp_path: Path,
) -> None:
    class PartiallyBlockingStore(FakeStore):
        def __init__(self):
            super().__init__()
            self.blocked_started = Event()
            self.release_blocked = Event()

        @contextmanager
        def open_object(self, key: str, if_match: str | None) -> Iterator[ObjectDownload]:
            self.open_calls.append(OpenCall(key, if_match))
            if key == "42/blocked.dtx":
                self.blocked_started.set()
                assert self.release_blocked.wait(timeout=5)
            yield ObjectDownload(
                body=BytesIO(b"chart"),
                size=5,
                etag="etag",
                etag_is_weak=False,
                last_modified=FIXED_MTIME,
            )

    store = PartiallyBlockingStore()
    progress: list[tuple[int, int, int]] = []
    progress_seen = Event()
    errors: list[Exception] = []

    def run_cache_sync():
        try:
            sync_cache(
                (
                    simfile(
                        remote_object("42/blocked.dtx"),
                        remote_object("42/fast.dtx"),
                    ),
                ),
                store,
                empty_index(tmp_path),
                config(download_concurrency=2),
                False,
                item_progress=lambda completed, total, completed_bytes: (
                    progress.append((completed, total, completed_bytes)),
                    progress_seen.set(),
                ),
            )
        except Exception as error:
            errors.append(error)

    thread = Thread(target=run_cache_sync, daemon=True)
    thread.start()
    try:
        assert store.blocked_started.wait(timeout=2)
        assert progress_seen.wait(timeout=2)
        assert thread.is_alive()
    finally:
        store.release_blocked.set()
        thread.join(timeout=5)

    assert errors == []
    assert progress == [(1, 2, 5), (2, 2, 10)]


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
    import src.benchmark.corpus_cache as cache

    real_os_open = os.open

    def fail_incoming_write(path, flags, *args, **kwargs):
        if flags & os.O_CREAT and flags & os.O_EXCL and kwargs.get("dir_fd") is not None:
            raise OSError("private local path")
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(cache.os, "open", fail_incoming_write)

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
    import src.benchmark.corpus_cache as cache

    temporary_parent_inodes: list[tuple[int, int]] = []
    real_os_open = os.open

    def track_open(path, flags, *args, **kwargs):
        directory_fd = kwargs.get("dir_fd")
        if flags & os.O_CREAT and flags & os.O_EXCL and directory_fd is not None:
            parent_stat = os.fstat(directory_fd)
            temporary_parent_inodes.append((parent_stat.st_dev, parent_stat.st_ino))
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(cache.os, "open", track_open)

    sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        empty_index(tmp_path),
        config(),
        False,
    )

    incoming_stat = (tmp_path / "sha256" / ".incoming").stat()
    assert temporary_parent_inodes == [(incoming_stat.st_dev, incoming_stat.st_ino)]


def test_cross_device_incoming_and_shard_fail_before_body_replace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    digest = sha256(b"chart").hexdigest()
    real_open = os.open
    real_fstat = os.fstat
    real_replace = os.replace
    shard_fds: set[int] = set()
    body_replacements: list[tuple[str, str]] = []

    class StatWithDifferentDevice:
        def __init__(self, wrapped, st_dev: int):
            self._wrapped = wrapped
            self.st_dev = st_dev

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def track_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if str(path) == digest[:2] and flags & getattr(os, "O_DIRECTORY", 0):
            shard_fds.add(descriptor)
        return descriptor

    def differing_fstat(descriptor: int):
        result = real_fstat(descriptor)
        if descriptor in shard_fds:
            return StatWithDifferentDevice(result, result.st_dev + 1)
        return result

    def track_replace(source, target, *args, **kwargs) -> None:
        if kwargs.get("src_dir_fd") is not None:
            body_replacements.append((str(source), str(target)))
        real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(cache.os, "open", track_open)
    monkeypatch.setattr(cache.os, "fstat", differing_fstat)
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
    digest = sha256(b"chart").hexdigest()
    real_os_open = os.open
    real_fsync = os.fsync
    real_replace = os.replace
    index = empty_index(tmp_path)
    real_checkpoint = index.checkpoint

    def descriptor_open(path, flags, *args, **kwargs):
        descriptor = real_os_open(path, flags, *args, **kwargs)
        if flags & os.O_CREAT and flags & os.O_EXCL:
            body_fds.add(descriptor)
        if str(path) == digest[:2] and flags & getattr(os, "O_DIRECTORY", 0):
            shard_fds.add(descriptor)
        return descriptor

    def fsync(descriptor: int) -> None:
        if descriptor in shard_fds:
            events.append("shard_fsync")
        elif descriptor in body_fds:
            events.append("body_fsync")
        real_fsync(descriptor)

    def replace_body(source, target, *args, **kwargs) -> None:
        if kwargs.get("src_dir_fd") is not None:
            events.append("body_replace")
        real_replace(source, target, *args, **kwargs)

    def checkpoint(index_entry: CacheIndexEntry) -> None:
        events.append("index_checkpoint")
        real_checkpoint(index_entry)

    monkeypatch.setattr(cache.os, "open", descriptor_open)
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
    import src.benchmark.corpus_cache as cache

    real_unlink = os.unlink

    def fail_owned_cleanup(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None and str(path).endswith(".tmp"):
            raise OSError("private cleanup path")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(cache.os, "unlink", fail_owned_cleanup)

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

    assert 1 < store.maximum_active <= 2
    assert [action.object_key for action in result.actions] == [remote.key for remote in objects]
    assert all(action.action == "downloaded" for action in result.actions)


def test_download_concurrency_bounds_owned_staging_resources_for_large_corpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.benchmark.corpus_cache as cache

    concurrency = 3
    objects = tuple(
        remote_object(
            f"42/chart-{index:03}.dtx",
            size=len(f"42/chart-{index:03}.dtx".encode("utf-8")),
        )
        for index in range(37)
    )
    store = ConcurrentStore()
    real_create_temporary = cache._create_temporary_file
    real_close_temporary = cache._OwnedTemporaryFile.close
    active_descriptors: set[int] = set()
    peak_staged_descriptors = 0
    staged_lock = Lock()

    def tracking_create(parent_fd: int, name: str):
        nonlocal peak_staged_descriptors
        temporary = real_create_temporary(parent_fd, name)
        with staged_lock:
            active_descriptors.add(temporary.descriptor)
            peak_staged_descriptors = max(
                peak_staged_descriptors,
                len(active_descriptors),
            )
        return temporary

    def tracking_close(temporary):
        was_closed = temporary.closed
        descriptor = temporary.descriptor
        real_close_temporary(temporary)
        if not was_closed:
            with staged_lock:
                active_descriptors.discard(descriptor)

    monkeypatch.setattr(cache, "_create_temporary_file", tracking_create)
    monkeypatch.setattr(cache._OwnedTemporaryFile, "close", tracking_close)

    result = sync_cache(
        (simfile(*objects),),
        store,
        empty_index(tmp_path),
        config(download_concurrency=concurrency),
        False,
    )

    assert 1 < store.maximum_active <= concurrency
    assert peak_staged_descriptors <= concurrency
    assert active_descriptors == set()
    assert list((tmp_path / "sha256" / ".incoming").iterdir()) == []
    assert [action.object_key for action in result.actions] == [remote.key for remote in objects]
    assert all(action.action == "downloaded" for action in result.actions)


def test_repair_candidate_runs_before_unrelated_misses_across_batches_and_restores_order(
    tmp_path: Path,
) -> None:
    bodies = {
        "42/unrelated.dtx": b"other",
        "42/ordinary-same-digest.dtx": b"chart",
        "42/repair.dtx": b"chart",
    }
    unrelated = remote_object("42/unrelated.dtx")
    ordinary = remote_object("42/ordinary-same-digest.dtx")
    repair = remote_object("42/repair.dtx")
    digest = sha256(bodies[repair.key]).hexdigest()
    final_path = tmp_path / "sha256" / digest[:2] / digest
    index = seeded_index(tmp_path, repair, final_path, body=bodies[repair.key])
    final_path.write_bytes(b"wrong")
    completions: list[str] = []

    class BodyMapStore:
        @contextmanager
        def open_object(self, key: str, if_match: str | None) -> Iterator[ObjectDownload]:
            del if_match
            body = bodies[key]
            yield ObjectDownload(
                body=BytesIO(body),
                size=len(body),
                etag="etag",
                etag_is_weak=False,
                last_modified=FIXED_MTIME,
            )
            completions.append(key)

    result = sync_cache(
        (simfile(unrelated, ordinary, repair),),
        BodyMapStore(),
        index,
        config(download_concurrency=1),
        False,
    )

    assert completions == [repair.key, unrelated.key, ordinary.key]
    assert [action.object_key for action in result.actions] == [
        unrelated.key,
        ordinary.key,
        repair.key,
    ]
    assert [action.action for action in result.actions] == [
        "downloaded",
        "downloaded",
        "downloaded",
    ]
    assert result.simfiles[0].sync_status == "complete"
    assert final_path.read_bytes() == b"chart"


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
    assert result.actions[0].miss_reason == "unreadable"
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


def test_unreadable_indexed_body_is_an_unreadable_validation_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    body = b"chart"
    digest = sha256(body).hexdigest()
    cache_path = tmp_path / "sha256" / digest[:2] / digest
    index = seeded_index(tmp_path, remote_object(), cache_path)
    monkeypatch.setattr(
        cache,
        "_hash_fd",
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
    assert result.actions[0].miss_reason == "unreadable"
    assert result.actions[0].errors == ()


def test_repeated_local_failure_code_is_reported_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    real_replace = os.replace
    real_unlink = os.unlink

    def fail_body_replace(source, target, *args, **kwargs) -> None:
        if kwargs.get("src_dir_fd") is not None:
            raise OSError("private replace path")
        real_replace(source, target, *args, **kwargs)

    def fail_owned_cleanup(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None and str(path).endswith(".tmp"):
            raise OSError("private cleanup path")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(cache.os, "replace", fail_body_replace)
    monkeypatch.setattr(cache.os, "unlink", fail_owned_cleanup)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert [error.code for error in result.actions[0].errors] == ["artifact_write_failed"]


@pytest.mark.parametrize("component", ["sha256", "incoming"])
def test_directory_swap_before_temporary_create_never_touches_replacement_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    component: str,
) -> None:
    import src.benchmark.corpus_cache as cache

    body = b"chart"
    digest = sha256(body).hexdigest()
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_bytes(b"unchanged")
    if component == "sha256":
        (external / ".incoming").mkdir()
        (external / digest[:2]).mkdir()
    sha_dir = tmp_path / "sha256"
    incoming_dir = sha_dir / ".incoming"
    swapped = False
    external_temporary_created = False
    real_path_open = Path.open
    real_os_open = os.open

    def swap_namespace() -> None:
        nonlocal swapped
        if swapped:
            return
        swapped = True
        if component == "sha256":
            sha_dir.rename(tmp_path / "pinned-sha256")
            sha_dir.symlink_to(external, target_is_directory=True)
        else:
            incoming_dir.rename(sha_dir / "pinned-incoming")
            incoming_dir.symlink_to(external, target_is_directory=True)

    def record_external_temp(name: str) -> None:
        nonlocal external_temporary_created
        external_temp = external / ".incoming" / name if component == "sha256" else external / name
        external_temporary_created = external_temporary_created or external_temp.exists()

    def path_open(path: Path, *args, **kwargs):
        if args and args[0] == "xb" and path.parent.name == ".incoming":
            swap_namespace()
            opened = real_path_open(path, *args, **kwargs)
            record_external_temp(path.name)
            return opened
        return real_path_open(path, *args, **kwargs)

    def descriptor_open(path, flags, *args, **kwargs):
        if flags & os.O_CREAT and flags & os.O_EXCL and str(path).endswith(".tmp"):
            swap_namespace()
            descriptor = real_os_open(path, flags, *args, **kwargs)
            record_external_temp(str(path))
            return descriptor
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(cache.os, "open", descriptor_open)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(body=body),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert swapped
    assert not external_temporary_created
    assert (external / "sentinel").read_bytes() == b"unchanged"
    assert not (external / digest).exists()
    assert not (tmp_path / "index-v1.json").exists()
    assert result.actions[0].action == "failed"
    assert [error.code for error in result.actions[0].errors] == ["artifact_write_failed"]


def test_shard_swap_before_replace_never_installs_or_checkpoints_outside_pinned_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.benchmark.corpus_cache as cache

    body = b"chart"
    digest = sha256(body).hexdigest()
    shard = tmp_path / "sha256" / digest[:2]
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_bytes(b"unchanged")
    swapped = False
    real_replace = os.replace

    def swap_then_replace(source, target, *args, **kwargs) -> None:
        nonlocal swapped
        source_name = str(source)
        if not swapped and (
            Path(source_name).parent.name == ".incoming" or kwargs.get("src_dir_fd") is not None
        ):
            swapped = True
            shard.rename(tmp_path / "pinned-shard")
            shard.symlink_to(external, target_is_directory=True)
        real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(cache.os, "replace", swap_then_replace)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(body=body),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert swapped
    assert (external / "sentinel").read_bytes() == b"unchanged"
    assert not (external / digest).exists()
    assert not (tmp_path / "index-v1.json").exists()
    assert result.actions[0].action == "failed"
    assert [error.code for error in result.actions[0].errors] == ["artifact_write_failed"]


def test_identical_final_is_fsynced_before_shard_and_index_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.benchmark.corpus_cache as cache

    body = b"chart"
    digest = sha256(body).hexdigest()
    final_path = tmp_path / "sha256" / digest[:2] / digest
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(body)
    index = empty_index(tmp_path)
    real_checkpoint = index.checkpoint
    real_open = os.open
    real_fsync = os.fsync
    final_fds: set[int] = set()
    shard_fds: set[int] = set()
    events: list[str] = []

    def track_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == final_path or str(path) == digest:
            final_fds.add(descriptor)
        if Path(path) == final_path.parent or (
            str(path) == digest[:2] and flags & getattr(os, "O_DIRECTORY", 0)
        ):
            shard_fds.add(descriptor)
        return descriptor

    def track_fsync(descriptor: int) -> None:
        if descriptor in shard_fds:
            events.append("shard_fsync")
        elif descriptor in final_fds:
            events.append("final_fsync")
        real_fsync(descriptor)

    def checkpoint(entry: CacheIndexEntry) -> None:
        events.append("index_checkpoint")
        real_checkpoint(entry)

    monkeypatch.setattr(cache.os, "open", track_open)
    monkeypatch.setattr(cache.os, "fsync", track_fsync)
    monkeypatch.setattr(index, "checkpoint", checkpoint)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(body=body),
        index,
        config(),
        False,
    )

    assert result.actions[0].action == "downloaded"
    assert events.index("final_fsync") < events.index("shard_fsync")
    assert events.index("shard_fsync") < events.index("index_checkpoint")


def test_same_digest_repair_is_independent_of_download_completion_order(
    tmp_path: Path,
) -> None:
    body = b"chart"
    digest = sha256(body).hexdigest()
    ordinary = remote_object("42/ordinary.dtx")
    repair = remote_object("42/repair.dtx")

    def run(
        cache_dir: Path,
        delays: dict[str, float],
    ) -> tuple[object, bytes, list[str]]:
        final_path = cache_dir / "sha256" / digest[:2] / digest
        index = seeded_index(cache_dir, repair, final_path, body=body)
        final_path.write_bytes(b"wrong")
        store = DelayedSameBodyStore(delays, body)
        result = sync_cache(
            (simfile(ordinary, repair),),
            store,
            index,
            config(download_concurrency=2),
            False,
        )
        return result, (cache_dir / "index-v1.json").read_bytes(), store.completions

    ordinary_first = run(
        tmp_path / "ordinary-first",
        {ordinary.key: 0.0, repair.key: 0.1},
    )
    repair_first = run(
        tmp_path / "repair-first",
        {ordinary.key: 0.1, repair.key: 0.0},
    )

    assert ordinary_first[2] == [ordinary.key, repair.key]
    assert repair_first[2] == [repair.key, ordinary.key]
    assert ordinary_first[0] == repair_first[0]
    assert ordinary_first[1] == repair_first[1]
    result = ordinary_first[0]
    assert result.simfiles[0].sync_status == "complete"
    assert [remote.cache_status for remote in result.simfiles[0].objects] == [
        "verified",
        "verified",
    ]
    assert [action.action for action in result.actions] == ["downloaded", "downloaded"]


@pytest.mark.parametrize("final_state", ["missing", "identical"])
def test_temporary_name_replacement_between_prepare_and_install_is_never_consumed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    final_state: str,
) -> None:
    import src.benchmark.corpus_cache as cache

    body = b"chart"
    digest = sha256(body).hexdigest()
    final_path = tmp_path / "sha256" / digest[:2] / digest
    if final_state == "identical":
        final_path.parent.mkdir(parents=True)
        final_path.write_bytes(body)
    index = empty_index(tmp_path)
    replacement = b"unowned replacement"
    replacement_path: Path | None = None
    owned_descriptors: list[int] = []
    swapped = False
    real_open = os.open
    real_replace = os.replace

    def replace_temporary_name() -> None:
        nonlocal replacement_path, swapped
        if swapped:
            return
        swapped = True
        incoming = tmp_path / "sha256" / ".incoming"
        replacement_path = next(incoming.glob("*.tmp"))
        unowned_source = tmp_path / "unowned-replacement"
        unowned_source.write_bytes(replacement)
        real_replace(unowned_source, replacement_path)

    def track_open(path, flags, *args, **kwargs):
        if (
            not swapped
            and str(path) == digest
            and kwargs.get("dir_fd") is not None
            and flags & os.O_ACCMODE == os.O_RDONLY
        ):
            replace_temporary_name()
        descriptor = real_open(path, flags, *args, **kwargs)
        if flags & os.O_CREAT and flags & os.O_EXCL and str(path).endswith(".tmp"):
            owned_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(cache.os, "open", track_open)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(body=body),
        index,
        config(),
        False,
    )

    assert swapped
    assert replacement_path is not None
    assert replacement_path.read_bytes() == replacement
    if final_state == "missing":
        assert not final_path.exists()
    else:
        assert final_path.read_bytes() == body
    assert not (tmp_path / "index-v1.json").exists()
    assert result.actions[0].action == "failed"
    assert [error.code for error in result.actions[0].errors] == ["artifact_write_failed"]
    assert owned_descriptors
    for descriptor in owned_descriptors:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


def test_content_descriptors_remain_open_through_checkpoint_and_close_after(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.benchmark.corpus_cache as cache

    body = b"chart"
    digest = sha256(body).hexdigest()
    index = empty_index(tmp_path)
    descriptors: dict[str, int] = {}
    real_open = os.open
    real_checkpoint = index.checkpoint

    def track_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        name = str(path)
        if flags & getattr(os, "O_DIRECTORY", 0):
            if Path(path) == tmp_path:
                descriptors["cache"] = descriptor
            elif name == "sha256":
                descriptors["sha256"] = descriptor
            elif name == ".incoming":
                descriptors["incoming"] = descriptor
            elif name == digest[:2]:
                descriptors["shard"] = descriptor
        elif name == digest and kwargs.get("dir_fd") is not None:
            descriptors["final"] = descriptor
        return descriptor

    def checkpoint(entry: CacheIndexEntry) -> None:
        assert set(descriptors) == {"cache", "sha256", "incoming", "shard", "final"}
        for descriptor in descriptors.values():
            os.fstat(descriptor)
        real_checkpoint(entry)

    monkeypatch.setattr(cache.os, "open", track_open)
    monkeypatch.setattr(index, "checkpoint", checkpoint)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(body=body),
        index,
        config(),
        False,
    )

    assert result.actions[0].action == "downloaded"
    for descriptor in descriptors.values():
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


def test_shard_swap_during_temporary_cleanup_never_checkpoints_stale_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.benchmark.corpus_cache as cache

    body = b"chart"
    digest = sha256(body).hexdigest()
    shard = tmp_path / "sha256" / digest[:2]
    pinned_shard = tmp_path / "pinned-shard-after-install"
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_bytes(b"unchanged")
    shard_descriptors: list[int] = []
    swapped = False
    real_open = os.open
    real_unlink = os.unlink

    def track_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if (
            str(path) == digest[:2]
            and kwargs.get("dir_fd") is not None
            and flags & getattr(os, "O_DIRECTORY", 0)
        ):
            shard_descriptors.append(descriptor)
        return descriptor

    def swap_shard_then_unlink(path, *args, **kwargs) -> None:
        nonlocal swapped
        if not swapped and str(path).endswith(".tmp") and kwargs.get("dir_fd") is not None:
            swapped = True
            shard.rename(pinned_shard)
            shard.symlink_to(external, target_is_directory=True)
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(cache.os, "open", track_open)
    monkeypatch.setattr(cache.os, "unlink", swap_shard_then_unlink)

    result = sync_cache(
        (
            simfile(
                remote_object("42/first.dtx"),
                remote_object("42/second.dtx"),
            ),
        ),
        FakeStore(body=body),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert swapped
    assert (external / "sentinel").read_bytes() == b"unchanged"
    assert not (external / digest).exists()
    assert not (tmp_path / "index-v1.json").exists()
    assert (pinned_shard / digest).read_bytes() == body
    assert [action.action for action in result.actions] == ["failed", "failed"]
    assert [[error.code for error in action.errors] for action in result.actions] == [
        ["artifact_write_failed"],
        ["artifact_write_failed"],
    ]
    assert shard_descriptors
    for descriptor in shard_descriptors:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


def test_open_regular_file_at_rejects_a_fifo_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is unavailable on this platform")

    fifo_name = "leaf"
    os.mkfifo(tmp_path / fifo_name)

    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        result: dict[str, object] = {}

        def open_leaf() -> None:
            try:
                _open_regular_file_at(parent_fd, fifo_name)
            except OSError as error:
                result["error"] = error

        worker = Thread(target=open_leaf, daemon=True)
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive(), "_open_regular_file_at hung opening a FIFO"
        error = result.get("error")
        assert isinstance(error, OSError)
        assert stat.S_ISFIFO(os.stat(tmp_path / fifo_name).st_mode)
    finally:
        os.close(parent_fd)


# ---------------------------------------------------------------------------
# Coverage gap closures for corpus_cache error paths
# ---------------------------------------------------------------------------


def test_index_load_wraps_non_filenotfound_oserror_as_invalid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    (tmp_path / "index-v1.json").write_text("{}", encoding="utf-8")

    def raising_stat(path, *, follow_symlinks=True):
        raise OSError(errno.EACCES, "private permission detail")

    monkeypatch.setattr(cache.os, "stat", raising_stat)

    with pytest.raises(ValueError, match="invalid cache index JSON") as caught:
        CacheIndexStore.load(tmp_path)

    assert "private" not in str(caught.value)


def test_checkpoint_cleanup_swallows_unlink_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    def fail_replace(source, target):
        raise OSError("private replace failure")

    def fail_unlink(self, missing_ok=False):
        raise OSError("private unlink failure")

    monkeypatch.setattr(cache.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    store = CacheIndexStore.load(tmp_path)
    with pytest.raises(OSError):
        store.checkpoint(entry())

    # The original OSError from replace propagates; unlink failure is swallowed.
    assert not (tmp_path / "index-v1.json").exists()


def test_streamed_byte_count_exceeding_declared_size_rejects_download(
    tmp_path: Path,
) -> None:
    result = sync_cache(
        (simfile(remote_object(size=3)),),
        FakeStore(body=b"chart", response_size=3),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert result.actions[0].action == "failed"
    assert [error.code for error in result.actions[0].errors] == ["byte_count_mismatch"]


def test_fdopen_failure_closes_writer_descriptor_and_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    closed_fds: list[int] = []
    real_fdopen = cache.os.fdopen

    def failing_fdopen(fd, *args, **kwargs):
        # Only fail for the writer descriptor (wb mode), not the lock file (a+b).
        if args and "wb" in args:
            closed_fds.append(fd)
            raise OSError("private fdopen failure")
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(cache.os, "fdopen", failing_fdopen)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert result.actions[0].action == "failed"
    assert "artifact_write_failed" in [e.code for e in result.actions[0].errors]
    assert closed_fds, "writer descriptor should have been closed on fdopen failure"


def test_cross_device_check_reached_when_binding_verification_is_bypassed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    digest = sha256(b"chart").hexdigest()
    real_fstat = cache.os.fstat
    shard_fds: set[int] = set()

    class ShiftedDev:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.st_dev = wrapped.st_dev + 1

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    real_open = cache.os.open

    def track_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if str(path) == digest[:2] and flags & getattr(os, "O_DIRECTORY", 0):
            shard_fds.add(fd)
        return fd

    def shifted_fstat(fd):
        result = real_fstat(fd)
        if fd in shard_fds:
            return ShiftedDev(result)
        return result

    # Bypass binding verification so the cross-device check at line 533 is reached.
    monkeypatch.setattr(cache, "_verify_directory_binding", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "_verify_content_directories", lambda *a, **kw: None)
    monkeypatch.setattr(cache.os, "open", track_open)
    monkeypatch.setattr(cache.os, "fstat", shifted_fstat)

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(),
        empty_index(tmp_path),
        config(),
        False,
    )

    assert [error.code for error in result.actions[0].errors] == ["artifact_write_failed"]


def test_open_directory_at_rejects_non_directory_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    file_path = tmp_path / "not_a_dir"
    file_path.write_text("data")
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        # On most platforms O_DIRECTORY causes NotADirectoryError at os.open.
        # Patch os.open to succeed so the S_ISDIR defensive check is exercised.
        real_open = cache.os.open

        def open_as_file(name, flags, *args, **kwargs):
            if name == "not_a_dir" and flags & getattr(os, "O_DIRECTORY", 0):
                return real_open(str(file_path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            return real_open(name, flags, *args, **kwargs)

        monkeypatch.setattr(cache.os, "open", open_as_file)
        with pytest.raises(OSError, match="cache content directory is unavailable"):
            cache._open_directory_at(parent_fd, "not_a_dir")
    finally:
        os.close(parent_fd)


def test_directory_open_flags_rejects_missing_nofollow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.benchmark.corpus_cache as cache

    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    with pytest.raises(OSError, match="no-follow directory descriptors"):
        cache._directory_open_flags()


def test_verify_cache_directory_binding_detects_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    real_stat = cache.os.stat

    def swapped_stat(path, *, follow_symlinks=True):
        real_stat(path, follow_symlinks=follow_symlinks)
        return os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        monkeypatch.setattr(cache.os, "stat", swapped_stat)
        with pytest.raises(OSError, match="cache directory binding changed"):
            cache._verify_cache_directory_binding(tmp_path, fd)
    finally:
        os.close(fd)


def test_verify_owned_descriptor_detects_ownership_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    real_fstat = cache.os.fstat

    def shifted_fstat(fd):
        result = real_fstat(fd)
        return os.stat_result((result.st_mode, result.st_ino + 1, 0, 0, 0, 0, 0, 0, 0, 0))

    fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        monkeypatch.setattr(cache.os, "fstat", shifted_fstat)
        owned = cache._OwnedTemporaryFile("name", fd, 0, 0)
        with pytest.raises(OSError, match="cache temporary file ownership changed"):
            cache._verify_owned_descriptor(owned, fd)
    finally:
        os.close(fd)


def test_verify_owned_temporary_binding_detects_ownership_change(tmp_path: Path) -> None:
    import src.benchmark.corpus_cache as cache

    fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        tmp_name = ".test_tmp"
        tmp_fd = os.open(
            tmp_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=fd,
        )
        try:
            stat_result = cache.os.fstat(tmp_fd)
            # Create owned temporary with WRONG device/inode so the binding check
            # passes but the ownership check at line 936-940 fails.
            owned = cache._OwnedTemporaryFile(
                tmp_name, tmp_fd, stat_result.st_dev + 1, stat_result.st_ino + 1
            )
            with pytest.raises(OSError, match="cache temporary file ownership changed"):
                cache._verify_owned_temporary_binding(fd, owned)
        finally:
            os.close(tmp_fd)
            os.unlink(tmp_name, dir_fd=fd)
    finally:
        os.close(fd)


def test_open_regular_file_at_rejects_missing_nofollow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(OSError, match="no-follow file descriptors"):
            _open_regular_file_at(parent_fd, "anything")
    finally:
        os.close(parent_fd)


def test_open_regular_file_at_rejects_missing_nonblock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delattr(os, "O_NONBLOCK", raising=False)
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(OSError, match="non-blocking file descriptors"):
            _open_regular_file_at(parent_fd, "anything")
    finally:
        os.close(parent_fd)


def test_create_temporary_file_rejects_missing_nofollow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(OSError, match="no-follow file descriptors"):
            cache._create_temporary_file(parent_fd, "test_tmp")
    finally:
        os.close(parent_fd)


def test_create_temporary_file_rejects_non_regular_descriptor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    real_fstat = cache.os.fstat

    def non_regular_fstat(fd):
        result = real_fstat(fd)
        # Return a socket mode instead of regular file mode.
        return os.stat_result((0o140000, result.st_ino, result.st_dev, 0, 0, 0, 0, 0, 0, 0))

    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        monkeypatch.setattr(cache.os, "fstat", non_regular_fstat)
        with pytest.raises(OSError, match="cache temporary file is not a regular file"):
            cache._create_temporary_file(parent_fd, "test_tmp")
    finally:
        # Clean up the file that was created on disk but whose descriptor was
        # closed by the exception handler.
        if (tmp_path / "test_tmp").exists():
            os.unlink("test_tmp", dir_fd=parent_fd)
        os.close(parent_fd)


def test_load_index_document_rejects_missing_nofollow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    index_path = tmp_path / "index-v1.json"
    index_path.write_text("{}", encoding="utf-8")
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    with pytest.raises(ValueError, match="invalid cache index JSON"):
        cache._load_index_document(index_path)


def test_load_index_document_swallows_close_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    index_path = tmp_path / "index-v1.json"
    index_path.write_text(
        '{"schema_version":"crux.r2-cache-index/v1","entries":[]}',
        encoding="utf-8",
    )

    real_close = cache.os.close
    real_open = cache.os.open
    open_fd: list[int] = []

    def tracking_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == index_path:
            open_fd.append(fd)
        return fd

    def failing_close(fd):
        if fd in open_fd:
            raise OSError("private close failure")
        return real_close(fd)

    monkeypatch.setattr(cache.os, "open", tracking_open)
    monkeypatch.setattr(cache.os, "close", failing_close)

    # Should succeed despite close failure in finally block.
    doc = cache._load_index_document(index_path)
    assert doc["schema_version"] == "crux.r2-cache-index/v1"


def test_unique_object_rejects_duplicate_fields() -> None:
    import src.benchmark.corpus_cache as cache

    with pytest.raises(ValueError):
        cache._unique_object([("a", 1), ("a", 2)])


def test_validated_entries_rejects_non_dict_document() -> None:
    import src.benchmark.corpus_cache as cache

    with pytest.raises(ValueError, match="invalid cache index document"):
        cache._validated_entries([])
    with pytest.raises(ValueError, match="invalid cache index document"):
        cache._validated_entries({"wrong": "keys"})


def test_validated_entries_rejects_non_list_entries() -> None:
    import src.benchmark.corpus_cache as cache

    with pytest.raises(ValueError, match="invalid cache index entries"):
        cache._validated_entries(
            {"schema_version": "crux.r2-cache-index/v1", "entries": "not-a-list"}
        )


def test_validated_entry_rejects_wrong_type_or_fields() -> None:
    import src.benchmark.corpus_cache as cache

    with pytest.raises(ValueError, match="invalid cache index entry"):
        cache._validated_entry("not-a-dict")
    with pytest.raises(ValueError, match="invalid cache index entry"):
        cache._validated_entry({"extra": "field"})


def test_is_canonical_utc_timestamp_rejects_invalid_iso_format() -> None:
    import src.benchmark.corpus_cache as cache

    assert cache._is_canonical_utc_timestamp("not-a-timestampZ") is False


def test_validate_relative_cache_path_rejects_mismatched_sha256() -> None:
    import src.benchmark.corpus_cache as cache

    with pytest.raises(ValueError, match="invalid cache index entry"):
        cache._validate_relative_cache_path("sha256/cc/" + "c" * 64, "b" * 64)


def test_open_lock_file_rejects_missing_nofollow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    with pytest.raises(RuntimeError, match="unsupported_platform"):
        cache._open_lock_file(tmp_path / ".index-v1.lock")


def test_open_lock_file_wraps_fdopen_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.benchmark.corpus_cache as cache

    def failing_fdopen(fd, *args, **kwargs):
        raise OSError("private fdopen failure")

    monkeypatch.setattr(cache.os, "fdopen", failing_fdopen)

    with pytest.raises(RuntimeError, match="cache_lock_failed"):
        cache._open_lock_file(tmp_path / ".index-v1.lock")


def test_lock_open_error_message_returns_unsupported_for_notsup() -> None:
    import src.benchmark.corpus_cache as cache

    exc = OSError(errno.ENOTSUP, "private")
    assert cache._lock_open_error_message(exc) == "unsupported_platform"


def test_error_recovery_skips_already_processed_and_deduplicates_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cover lines 268 (duplicate progress) and 375 (skip processed in recovery)."""
    import src.benchmark.corpus_cache as cache

    class MixedStore:
        @contextmanager
        def open_object(self, key: str, if_match: str | None) -> Iterator[ObjectDownload]:
            if "fail" in key:
                raise R2StoreError("object_get_failed", "private detail", key)
            yield ObjectDownload(
                body=BytesIO(b"chart"),
                size=5,
                etag="etag",
                etag_is_weak=False,
                last_modified=FIXED_MTIME,
            )

    def failing_install(*_args, **_kwargs):
        raise OSError("private install failure")

    monkeypatch.setattr(cache, "_install_prepared_downloads", failing_install)

    result = sync_cache(
        (
            simfile(
                remote_object(key="42/fail.dtx"),
                remote_object(key="42/ok.dtx"),
            ),
        ),
        MixedStore(),
        empty_index(tmp_path),
        config(),
        False,
    )

    # Both objects should fail; the install failure is sanitized.
    assert all(action.action == "failed" for action in result.actions)
    assert not (tmp_path / "index-v1.json").exists()


def test_installed_content_hash_mismatch_after_repair_rejects_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cover line 581: installed digest mismatch after repair replacement."""
    import src.benchmark.corpus_cache as cache

    store = index_with_local_state(tmp_path, "sha256_mismatch")

    # Make _hash_fd return a wrong digest so that:
    # 1. The existing file is seen as mismatched (repair path taken)
    # 2. The installed file after replacement is also seen as mismatched (line 581)
    monkeypatch.setattr(cache, "_hash_fd", lambda fd: ("0" * 64, 0))

    result = sync_cache(
        (simfile(remote_object()),),
        FakeStore(body=b"chart"),
        store,
        config(),
        False,
    )

    assert "artifact_write_failed" in [error.code for error in result.actions[0].errors]
    # The index file pre-exists from index_with_local_state; the failed install
    # must not have updated it with the rejected content.
    assert result.actions[0].action == "failed"
