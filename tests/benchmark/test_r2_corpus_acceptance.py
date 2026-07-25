from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Lock

from src.benchmark.r2_corpus_models import HeadMetadata, ListedObject, SyncRequest
from src.benchmark.r2_corpus_sync import sync_r2_corpus
from src.benchmark.r2_inventory import ObjectDownload, R2StoreError

FIXED_MTIME = datetime(2026, 7, 25, 1, 2, 3, tzinfo=timezone.utc)
COMPLETED_AT = FIXED_MTIME + timedelta(seconds=2)
RAW_ENDPOINT = "https://private-account.r2.cloudflarestorage.com"
SECRET_SENTINEL = "HPA321-DO-NOT-PUBLISH-CREDENTIAL"
RUN_IDS = (
    "10000000-0000-4000-8000-000000000001",
    "10000000-0000-4000-8000-000000000002",
    "10000000-0000-4000-8000-000000000003",
    "10000000-0000-4000-8000-000000000004",
    "10000000-0000-4000-8000-000000000005",
    "10000000-0000-4000-8000-000000000006",
)
EXPECTED_SELECTED_FIXTURE = {
    "1/set.def": (
        b"set-definition",
        14,
        "14e9b6a8adecf03c85c74d46bc38230bd9cde66cc9d95099d9bb355aa16a47d5",
    ),
    "1/main.DTX": (
        b"main-chart",
        10,
        "46829163852a40de1473ca08982687285211684ad6b69c49026394761e56f8c2",
    ),
    "1/音源/readme.TXT": (
        b"\xe8\xaa\xac\xe6\x98\x8e",
        6,
        "9683551a843c71d8b84ba930126021ef40831ee11bd5e69b06a3ea589148bba6",
    ),
    "2/chart.dtx": (
        b"chart-v1",
        8,
        "0bdc777f11e9ed6f56d418040e9716d177d16ec2ec45ab7468b6350a1a01379a",
    ),
}
EXPECTED_CHANGED_CHART = (
    b"chart-v2-updated",
    16,
    "0b97e9e3214d72b38920ef3138b376e5158acab0eef2b16290eb1501d8fa0adc",
)
EXPECTED_ROW_OBJECTS = {
    1: (
        ("1/assets/", 0, "not_selected"),
        ("1/assets/音源/", 0, "not_selected"),
        ("1/audio/song.ogg", 18, "not_selected"),
        ("1/main.DTX", 10, "verified"),
        ("1/set.def", 14, "verified"),
        ("1/音源/readme.TXT", 6, "verified"),
    ),
    2: (("2/chart.dtx", 8, "verified"),),
    3: (
        ("3/", 0, "not_selected"),
        ("3/nested/", 0, "not_selected"),
    ),
}
EXPECTED_ROW_STATUSES = {1: "complete", 2: "complete", 3: "empty"}


@dataclass(frozen=True)
class FixtureObject:
    key: str
    body: bytes
    etag: str
    last_modified: datetime = FIXED_MTIME
    content_type: str = "application/octet-stream"


class PaginatedMemoryStore:
    def __init__(
        self,
        pages: tuple[tuple[FixtureObject, ...], ...],
        *,
        head_errors: dict[str, R2StoreError] | None = None,
    ) -> None:
        self.pages = pages
        self.objects = {item.key: item for page in pages for item in page}
        self.head_errors = head_errors or {}
        self.page_visits: list[int] = []
        self.head_calls: list[str] = []
        self.get_calls: list[tuple[str, str | None]] = []
        self._calls_lock = Lock()

    def validate_bucket(self) -> None:
        return None

    def list_objects(self) -> tuple[ListedObject, ...]:
        listed = []
        for page_number, page in enumerate(self.pages):
            self.page_visits.append(page_number)
            listed.extend(
                ListedObject(
                    key=item.key,
                    size=len(item.body),
                    etag=item.etag,
                    etag_is_weak=False,
                    last_modified=item.last_modified,
                )
                for item in page
            )
        return tuple(listed)

    def head_object(self, key: str) -> HeadMetadata:
        with self._calls_lock:
            self.head_calls.append(key)
        if key in self.head_errors:
            raise self.head_errors[key]
        item = self.objects[key]
        return HeadMetadata(
            size=len(item.body),
            etag=item.etag,
            etag_is_weak=False,
            last_modified=item.last_modified,
            content_type=item.content_type,
        )

    @contextmanager
    def open_object(self, key: str, if_match: str | None) -> Iterator[ObjectDownload]:
        with self._calls_lock:
            self.get_calls.append((key, if_match))
        item = self.objects[key]
        assert if_match == item.etag
        body = BytesIO(item.body)
        try:
            yield ObjectDownload(
                body=body,
                size=len(item.body),
                etag=item.etag,
                etag_is_weak=False,
                last_modified=item.last_modified,
            )
        finally:
            body.close()


def fixture_pages() -> tuple[tuple[FixtureObject, ...], ...]:
    return (
        (
            FixtureObject("1/set.def", b"set-definition", "set-v1", content_type="text/plain"),
            FixtureObject("1/assets/", b"", "marker-assets"),
            FixtureObject("1/audio/song.ogg", b"not-selected-audio", "audio-v1"),
        ),
        (
            FixtureObject("1/main.DTX", b"main-chart", "main-v1", content_type="text/plain"),
            FixtureObject("1/assets/音源/", b"", "marker-nested"),
            FixtureObject(
                "1/音源/readme.TXT",
                "説明".encode(),
                "readme-v1",
                content_type="text/plain",
            ),
            FixtureObject("2/chart.dtx", b"chart-v1", "chart-v1", content_type="text/plain"),
        ),
        (
            FixtureObject("3/", b"", "marker-root"),
            FixtureObject("3/nested/", b"", "marker-empty-nested"),
            FixtureObject("4/chart.dtx", b"alias-four", "alias-four"),
            FixtureObject("04/chart.dtx", b"alias-zero-four", "alias-zero-four"),
        ),
    )


def with_changed_chart(
    pages: tuple[tuple[FixtureObject, ...], ...],
) -> tuple[tuple[FixtureObject, ...], ...]:
    return tuple(
        tuple(
            (
                replace(
                    item,
                    body=b"chart-v2-updated",
                    etag="chart-v2",
                    last_modified=FIXED_MTIME + timedelta(seconds=1),
                )
                if item.key == "2/chart.dtx"
                else item
            )
            for item in page
        )
        for page in pages
    )


def run_sync(
    root: Path,
    store: PaginatedMemoryStore,
    run_id: str,
    *,
    provenance_file: Path | None = None,
):
    times = iter((FIXED_MTIME, COMPLETED_AT))
    return sync_r2_corpus(
        SyncRequest(
            output_dir=root / "output",
            cache_dir=root / "cache",
            provenance_file=provenance_file,
        ),
        environ={
            "CRUX_R2_ENDPOINT_URL": RAW_ENDPOINT,
            "CRUX_R2_BUCKET": "simfile-dtx",
            "AWS_ACCESS_KEY_ID": SECRET_SENTINEL,
            "AWS_SECRET_ACCESS_KEY": SECRET_SENTINEL,
        },
        dependency_check=lambda: None,
        store_factory=lambda _config: store,
        clock=lambda: next(times, COMPLETED_AT),
        monotonic=lambda: 0.0,
        run_id_factory=lambda: run_id,
    )


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(path: Path) -> tuple[dict[str, object], ...]:
    content = path.read_bytes()
    assert content.endswith(b"\n")
    return tuple(json.loads(line) for line in content.splitlines())


def independently_verify_manifest(path: Path, corpus_version: str, manifest_sha256: str) -> None:
    content = path.read_bytes()
    rows = read_manifest(path)
    identity_bytes = b""
    for row in rows:
        normalized = dict(row)
        assert normalized.pop("corpus_version") == corpus_version
        identity_bytes += (
            json.dumps(
                normalized,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )

    expected_corpus_version = f"sha256:{sha256(identity_bytes).hexdigest()}"
    expected_manifest_sha256 = sha256(content).hexdigest()
    assert corpus_version == expected_corpus_version
    assert manifest_sha256 == expected_manifest_sha256
    assert path.name == f"{expected_manifest_sha256}.jsonl"


def assert_canonical_index(cache_dir: Path) -> dict[str, object]:
    index_path = cache_dir / "index-v1.json"
    content = index_path.read_bytes()
    document = json.loads(content)
    expected = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert content == expected
    entries = document["entries"]
    assert [entry["key"] for entry in entries] == sorted(entry["key"] for entry in entries)
    return document


def assert_selected_fixture_contract(
    rows_by_id: dict[int, dict[str, object]],
    cache_dir: Path,
    index: dict[str, object],
) -> None:
    manifest_objects = {
        item["key"]: item
        for row in rows_by_id.values()
        for item in row["objects"]
        if item["cache_status"] == "verified"
    }
    index_entries = {entry["key"]: entry for entry in index["entries"]}
    assert set(manifest_objects) == set(EXPECTED_SELECTED_FIXTURE)
    assert set(index_entries) == set(EXPECTED_SELECTED_FIXTURE)

    for key, (expected_body, expected_size, expected_sha256) in EXPECTED_SELECTED_FIXTURE.items():
        expected_cache_path = f"sha256/{expected_sha256[:2]}/{expected_sha256}"
        manifest_object = manifest_objects[key]
        assert manifest_object["size"] == expected_size
        assert manifest_object["sha256"] == expected_sha256
        assert manifest_object["cache_path"] == expected_cache_path
        assert manifest_object["cache_status"] == "verified"

        index_entry = index_entries[key]
        assert index_entry["size"] == expected_size
        assert index_entry["sha256"] == expected_sha256
        assert index_entry["cache_path"] == expected_cache_path

        cache_body = (cache_dir / expected_cache_path).read_bytes()
        assert cache_body == expected_body
        assert len(cache_body) == expected_size
        assert sha256(cache_body).hexdigest() == expected_sha256


def test_sync_is_repeatable_and_preserves_changed_history(tmp_path: Path):
    pages = fixture_pages()
    first_store = PaginatedMemoryStore(pages)
    first = run_sync(tmp_path, first_store, RUN_IDS[0])
    second_store = PaginatedMemoryStore(pages)
    second = run_sync(tmp_path, second_store, RUN_IDS[1])

    assert first.exit_code == 1
    assert second.exit_code == 1
    assert first.manifest is not None
    assert second.manifest is not None
    assert first.manifest.corpus_version == second.manifest.corpus_version
    assert first.manifest.path == second.manifest.path
    second_manifest_bytes = second.manifest.path.read_bytes()
    assert first.manifest.path.read_bytes() == second_manifest_bytes
    assert first_store.page_visits == [0, 1, 2]
    assert second_store.page_visits == [0, 1, 2]
    assert sorted(key for key, _ in first_store.get_calls) == [
        "1/main.DTX",
        "1/set.def",
        "1/音源/readme.TXT",
        "2/chart.dtx",
    ]
    assert second_store.get_calls == []

    first_report = read_json(first.report_path)
    second_report = read_json(second.report_path)
    assert first_report["counters"]["downloads_completed"] == 4
    assert second_report["counters"]["cache_hits"] == 4
    assert first_report["ambiguous_prefixes"] == {"4": ["04/", "4/"]}

    second_rows = read_manifest(second.manifest.path)
    rows_by_id = {row["simfile_id"]: row for row in second_rows}
    assert {row["simfile_id"] for row in second_rows} == {1, 2, 3}
    assert {
        simfile_id: row["sync_status"] for simfile_id, row in rows_by_id.items()
    } == EXPECTED_ROW_STATUSES
    assert {
        simfile_id: tuple(
            (item["key"], item["size"], item["cache_status"]) for item in row["objects"]
        )
        for simfile_id, row in rows_by_id.items()
    } == EXPECTED_ROW_OBJECTS
    for row in rows_by_id.values():
        for item in row["objects"]:
            if item["cache_status"] == "not_selected":
                assert item["sha256"] is None
                assert item["cache_path"] is None

    index = assert_canonical_index(tmp_path / "cache")
    assert_selected_fixture_contract(rows_by_id, tmp_path / "cache", index)
    old_chart_body, old_chart_size, old_chart_sha256 = EXPECTED_SELECTED_FIXTURE["2/chart.dtx"]
    old_chart_path = tmp_path / "cache" / f"sha256/{old_chart_sha256[:2]}/{old_chart_sha256}"
    assert old_chart_path.read_bytes() == old_chart_body
    assert old_chart_path.stat().st_size == old_chart_size
    independently_verify_manifest(
        second.manifest.path,
        second.manifest.corpus_version,
        second.manifest.manifest_sha256,
    )

    third_store = PaginatedMemoryStore(with_changed_chart(pages))
    third = run_sync(tmp_path, third_store, RUN_IDS[2])

    assert third.exit_code == 1
    assert third.manifest is not None
    assert third.manifest.corpus_version != second.manifest.corpus_version
    assert third.manifest.path != second.manifest.path
    assert third_store.get_calls == [("2/chart.dtx", "chart-v2")]
    assert second.manifest.path.read_bytes() == second_manifest_bytes
    assert old_chart_path.read_bytes() == old_chart_body
    third_rows = {row["simfile_id"]: row for row in read_manifest(third.manifest.path)}
    changed_chart = third_rows[2]["objects"][0]
    changed_body, changed_size, changed_sha256 = EXPECTED_CHANGED_CHART
    changed_cache_path = f"sha256/{changed_sha256[:2]}/{changed_sha256}"
    assert changed_chart["key"] == "2/chart.dtx"
    assert changed_chart["size"] == changed_size
    assert changed_chart["sha256"] == changed_sha256
    assert changed_chart["cache_path"] == changed_cache_path
    assert (tmp_path / "cache" / changed_cache_path).read_bytes() == changed_body
    latest = read_json(tmp_path / "output" / "latest.json")
    assert latest["corpus_version"] == third.manifest.corpus_version
    assert latest["manifest_sha256"] == third.manifest.manifest_sha256
    assert latest["manifest_path"] == third.manifest.relative_path
    assert latest["overall_status"] == "partial"
    third_report = read_json(third.report_path)
    assert third_report["overall_status"] == "partial"
    assert third_report["exit_code"] == third.exit_code
    assert third_report["counters"]["cache_hits"] == 3
    assert third_report["counters"]["downloads_completed"] == 1
    latest_report = read_json(tmp_path / "output" / "latest-report.json")
    assert (
        latest_report["report_path"]
        == third.report_path.relative_to(tmp_path / "output").as_posix()
    )
    assert latest_report["overall_status"] == third.overall_status
    assert latest_report["exit_code"] == third.exit_code
    independently_verify_manifest(
        third.manifest.path,
        third.manifest.corpus_version,
        third.manifest.manifest_sha256,
    )
    assert len(assert_canonical_index(tmp_path / "cache")["entries"]) == 4

    for artifact in (
        *sorted((tmp_path / "output" / "manifests").glob("*.jsonl")),
        *sorted((tmp_path / "output" / "reports").glob("*.json")),
    ):
        content = artifact.read_bytes()
        assert SECRET_SENTINEL.encode() not in content
        assert RAW_ENDPOINT.encode() not in content


def test_provenance_and_error_changes_rekey_the_public_manifest(tmp_path: Path):
    pages = fixture_pages()
    baseline = run_sync(tmp_path, PaginatedMemoryStore(pages), RUN_IDS[3])
    assert baseline.manifest is not None

    provenance_file = tmp_path / "provenance.json"
    provenance_file.write_bytes(
        b'{"schema_version":"crux.corpus-provenance/v1","simfiles":'
        b'{"1":{"source_reference":"private archive"}}}\n'
    )
    with_provenance = run_sync(
        tmp_path,
        PaginatedMemoryStore(pages),
        RUN_IDS[4],
        provenance_file=provenance_file,
    )
    assert with_provenance.manifest is not None
    assert with_provenance.manifest.corpus_version != baseline.manifest.corpus_version

    raw_error = R2StoreError(
        "object_head_failed",
        f"{SECRET_SENTINEL} {RAW_ENDPOINT}",
        "2/chart.dtx",
    )
    with_error = run_sync(
        tmp_path,
        PaginatedMemoryStore(pages, head_errors={"2/chart.dtx": raw_error}),
        RUN_IDS[5],
        provenance_file=provenance_file,
    )
    assert with_error.manifest is not None
    assert with_error.manifest.corpus_version != with_provenance.manifest.corpus_version
    row = next(item for item in read_manifest(with_error.manifest.path) if item["simfile_id"] == 2)
    assert row["sync_errors"] == [
        {
            "code": "object_head_failed",
            "message": "Object metadata inspection failed.",
            "object_key": "2/chart.dtx",
            "scope": "object",
        }
    ]
    assert SECRET_SENTINEL.encode() not in with_error.manifest.path.read_bytes()
    assert RAW_ENDPOINT.encode() not in with_error.manifest.path.read_bytes()
    assert with_error.report_path is not None
    assert SECRET_SENTINEL.encode() not in with_error.report_path.read_bytes()
    assert RAW_ENDPOINT.encode() not in with_error.report_path.read_bytes()


def test_base_cli_import_does_not_import_boto3():
    code = (
        "import sys; import src.cli.main; "
        "assert 'boto3' not in sys.modules; "
        "assert 'botocore' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
