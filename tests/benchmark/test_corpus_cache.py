import builtins
import json
import os
from dataclasses import replace
from pathlib import Path
from threading import Thread
from time import sleep

import pytest

from src.benchmark.corpus_cache import (
    CacheIndexEntry,
    CacheIndexStore,
    cache_writer_lock,
)


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


def test_live_writer_lock_fails_fast(tmp_path: Path) -> None:
    with cache_writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match="cache_locked"):
            with cache_writer_lock(tmp_path):
                raise AssertionError("second writer must not enter")


def test_writer_lock_retains_one_open_lock_file_for_entire_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_path_open = Path.open
    opened = []

    def path_open(self: Path, *args, **kwargs):
        handle = real_path_open(self, *args, **kwargs)
        if self == tmp_path / ".index-v1.lock":
            opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", path_open)

    with cache_writer_lock(tmp_path):
        assert len(opened) == 1
        assert not opened[0].closed
    assert len(opened) == 1
    assert opened[0].closed


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

    def publish() -> None:
        nonlocal active_writers, maximum_active_writers
        active_writers += 1
        maximum_active_writers = max(maximum_active_writers, active_writers)
        sleep(0.02)
        try:
            original_publish()
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
