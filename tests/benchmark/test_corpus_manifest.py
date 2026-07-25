import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pytest

import src.benchmark.corpus_manifest as corpus_manifest
from src.benchmark.corpus_manifest import (
    ManifestPublicationError,
    build_manifest_rows,
    canonical_json_line,
    publish_latest_manifest,
    publish_manifest,
    render_manifest,
)
from src.benchmark.r2_corpus_models import (
    CacheAction,
    CacheActionName,
    CacheSyncResult,
    ErrorCode,
    ProvenanceRecord,
    RemoteObject,
    RenderedManifest,
    SimfileInventory,
    SimfileStatus,
    SyncError,
)

FIXED_TIME = datetime(2026, 7, 25, 1, 2, 3, tzinfo=timezone.utc)


def make_simfile(
    simfile_id: int = 2,
    *,
    key: str | None = None,
    status: SimfileStatus = "complete",
    error_code: ErrorCode | None = None,
    objects: tuple[RemoteObject, ...] | None = None,
) -> SimfileInventory:
    object_key = key or f"{simfile_id}/chart.dtx"
    errors = (
        ()
        if error_code is None
        else (SyncError("object", error_code, "Safe deterministic message.", object_key),)
    )
    remote = RemoteObject(
        key=object_key,
        size=5,
        etag="etag",
        etag_is_weak=False,
        last_modified=FIXED_TIME,
        content_type="text/plain",
        cache_status="verified",
        sha256="c" * 64,
        cache_path=f"sha256/cc/{'c' * 64}",
        errors=errors,
    )
    return SimfileInventory(
        simfile_id=simfile_id,
        object_prefix=f"{simfile_id}/",
        objects=(remote,) if objects is None else objects,
        sync_status=status,
        sync_errors=errors,
    )


def render_fixture(
    *,
    key: str = "2/chart.dtx",
    status: SimfileStatus = "complete",
    error_code: ErrorCode | None = None,
    source_reference: str | None = None,
) -> RenderedManifest:
    provenance = {2: ProvenanceRecord(source_reference=source_reference)}
    rows = build_manifest_rows(
        (make_simfile(2, key=key, status=status, error_code=error_code),),
        provenance,
        "a" * 64,
        "simfile-dtx",
    )
    return render_manifest(rows)


def render_for_action(action: CacheActionName) -> RenderedManifest:
    simfiles = (make_simfile(),)
    cache_result = CacheSyncResult(
        simfiles,
        (CacheAction("2/chart.dtx", action, 5),),
    )
    rows = build_manifest_rows(
        cache_result.simfiles,
        {},
        "a" * 64,
        "simfile-dtx",
    )
    return render_manifest(rows)


def test_canonical_json_line_uses_sorted_compact_utf8_with_one_newline() -> None:
    content = canonical_json_line({"z": "音楽", "a": None})

    assert content == '{"a":null,"z":"音楽"}\n'.encode()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_line_rejects_non_json_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json_line({"value": value})


def test_canonical_json_line_rejects_strings_that_are_not_utf8_encodable() -> None:
    with pytest.raises(UnicodeEncodeError):
        canonical_json_line({"unsafe": "\ud800"})


def test_render_manifest_is_order_independent_for_inputs() -> None:
    first = render_manifest(
        build_manifest_rows(
            simfiles=(make_simfile(10), make_simfile(2)),
            provenance={},
            source_endpoint_sha256="a" * 64,
            bucket="bucket",
        )
    )
    second = render_manifest(
        build_manifest_rows(
            simfiles=(make_simfile(2), make_simfile(10)),
            provenance={},
            source_endpoint_sha256="a" * 64,
            bucket="bucket",
        )
    )

    assert first.corpus_version == second.corpus_version
    assert first.content == second.content
    assert first.content.endswith(b"\n")
    assert [row["simfile_id"] for row in first.rows] == [2, 10]


def test_render_manifest_hashes_the_self_reference_free_bytes() -> None:
    rendered = render_manifest(
        build_manifest_rows(
            simfiles=(make_simfile(2),),
            provenance={},
            source_endpoint_sha256="b" * 64,
            bucket="bucket",
        )
    )
    identity_bytes = b"".join(
        canonical_json_line({key: value for key, value in row.items() if key != "corpus_version"})
        for row in rendered.rows
    )

    assert rendered.corpus_version == f"sha256:{sha256(identity_bytes).hexdigest()}"
    assert all(row["corpus_version"] == rendered.corpus_version for row in rendered.rows)
    assert rendered.manifest_sha256 == sha256(rendered.content).hexdigest()


def test_render_manifest_rejects_rows_that_already_contain_corpus_version() -> None:
    with pytest.raises(ValueError, match="normalized payload rows must omit corpus_version"):
        render_manifest(({"corpus_version": "sha256:unsafe"},))


def test_cache_hit_and_downloaded_actions_have_identical_manifest_identity() -> None:
    downloaded = render_for_action("downloaded")
    cache_hit = render_for_action("cache_hit")

    assert downloaded.content == cache_hit.content
    assert downloaded.corpus_version == cache_hit.corpus_version


def test_status_error_and_provenance_changes_rekey_the_manifest() -> None:
    baseline = render_fixture()

    assert render_fixture(status="partial").corpus_version != baseline.corpus_version
    assert render_fixture(error_code="object_head_failed").corpus_version != baseline.corpus_version
    assert (
        render_fixture(source_reference="private archive").corpus_version != baseline.corpus_version
    )


def test_manifest_preserves_non_ascii_keys_without_ascii_escaping() -> None:
    rendered = render_fixture(key="7/音楽/譜面.DTX")

    assert "音楽".encode() in rendered.content
    assert b"\\u97f3" not in rendered.content


def test_build_manifest_rows_sorts_objects_and_errors_by_the_contract() -> None:
    remote_z = make_simfile(key="2/z.dtx").objects[0]
    remote_a = replace(
        remote_z,
        key="2/A.dtx",
        errors=(
            SyncError("object", "object_get_failed", "z-message", None),
            SyncError("cache", "artifact_write_failed", "a-message", "2/A.dtx"),
        ),
    )
    errors = (
        SyncError("object", "object_get_failed", "z-message", None),
        SyncError("object", "object_get_failed", "a-message", None),
        SyncError("cache", "artifact_write_failed", "a-message", "2/A.dtx"),
    )
    simfile = replace(
        make_simfile(),
        objects=(remote_z, remote_a),
        sync_status="partial",
        sync_errors=errors,
    )

    (row,) = build_manifest_rows((simfile,), {}, "a" * 64, "bucket")

    assert [item["key"] for item in row["objects"]] == ["2/A.dtx", "2/z.dtx"]
    assert row["sync_errors"] == [
        {
            "scope": "cache",
            "code": "artifact_write_failed",
            "object_key": "2/A.dtx",
            "message": "a-message",
        },
        {
            "scope": "object",
            "code": "object_get_failed",
            "object_key": None,
            "message": "a-message",
        },
        {
            "scope": "object",
            "code": "object_get_failed",
            "object_key": None,
            "message": "z-message",
        },
    ]


def test_manifest_row_and_object_have_only_durable_contract_fields() -> None:
    (row,) = build_manifest_rows(
        (make_simfile(),),
        {},
        "a" * 64,
        "simfile-dtx",
    )

    assert set(row) == {
        "schema_version",
        "cache_profile",
        "simfile_id",
        "object_prefix",
        "source_endpoint_sha256",
        "source_bucket",
        "source_discovery_method",
        "objects",
        "sync_status",
        "sync_errors",
        "source_origin",
        "source_author_or_pack",
        "source_reference",
        "rights_status",
        "redistribution_allowed",
        "provenance_notes",
    }
    assert row["schema_version"] == "crux.r2-corpus-manifest/v1"
    assert row["cache_profile"] == "setdef_dtx_txt_v1"
    assert row["source_discovery_method"] == "r2_list_objects_v2"
    assert row["source_endpoint_sha256"] == "a" * 64
    assert "https://" not in json.dumps(row)
    assert row["objects"] == [
        {
            "key": "2/chart.dtx",
            "size": 5,
            "etag": "etag",
            "etag_is_weak": False,
            "version": None,
            "last_modified": "2026-07-25T01:02:03Z",
            "content_type": "text/plain",
            "cache_status": "verified",
            "sha256": "c" * 64,
            "cache_path": f"sha256/cc/{'c' * 64}",
        }
    ]
    forbidden = {
        "action",
        "cache_action",
        "miss_reason",
        "invocation_id",
        "report_path",
        "provider_checksum",
        "publication_time",
        "published_at",
    }
    assert forbidden.isdisjoint(row)
    assert forbidden.isdisjoint(row["objects"][0])
    assert "corpus_version" not in row


def test_build_manifest_rows_rejects_a_raw_endpoint_in_place_of_its_sha256() -> None:
    with pytest.raises(ValueError, match="source_endpoint_sha256 must be lowercase SHA-256"):
        build_manifest_rows(
            (make_simfile(),),
            {},
            "https://account-id.r2.cloudflarestorage.com",
            "simfile-dtx",
        )


def test_manifest_timestamps_are_normalized_to_utc_z_form() -> None:
    non_utc = timezone(timedelta(hours=9))
    remote = replace(
        make_simfile().objects[0],
        last_modified=datetime(2026, 7, 25, 10, 2, 3, 120_000, tzinfo=non_utc),
    )
    (row,) = build_manifest_rows(
        (make_simfile(objects=(remote,)),),
        {},
        "a" * 64,
        "bucket",
    )

    assert row["objects"][0]["last_modified"] == "2026-07-25T01:02:03.12Z"


@pytest.mark.parametrize(
    ("status", "errors"),
    [
        ("empty", ()),
        (
            "failed",
            (
                SyncError(
                    "object",
                    "object_get_failed",
                    "Object body read failed.",
                    "2/chart.dtx",
                ),
            ),
        ),
    ],
)
def test_empty_and_failed_rows_keep_the_same_top_level_shape(
    status: SimfileStatus,
    errors: tuple[SyncError, ...],
) -> None:
    baseline = build_manifest_rows((make_simfile(),), {}, "a" * 64, "bucket")[0]
    simfile = SimfileInventory(2, "2/", (), status, errors)

    (row,) = build_manifest_rows((simfile,), {}, "a" * 64, "bucket")

    assert set(row) == set(baseline)
    assert row["objects"] == []
    assert row["sync_status"] == status


def test_changing_only_object_enumeration_order_does_not_change_manifest_bytes() -> None:
    first_object = make_simfile(key="2/a.dtx").objects[0]
    second_object = replace(first_object, key="2/b.dtx", etag="other")
    first = replace(make_simfile(), objects=(second_object, first_object))
    second = replace(make_simfile(), objects=(first_object, second_object))

    first_rendered = render_manifest(build_manifest_rows((first,), {}, "a" * 64, "bucket"))
    second_rendered = render_manifest(build_manifest_rows((second,), {}, "a" * 64, "bucket"))

    assert first_rendered.content == second_rendered.content


@pytest.mark.parametrize(
    "changed",
    [
        {"size": 6},
        {"etag": "different"},
        {"etag_is_weak": True},
        {"last_modified": FIXED_TIME + timedelta(seconds=1)},
        {"content_type": None},
        {"cache_status": "failed"},
        {"sha256": "d" * 64},
        {"cache_path": f"sha256/dd/{'d' * 64}"},
    ],
)
def test_remote_metadata_and_verified_content_changes_rekey_manifest(
    changed: dict[str, object],
) -> None:
    baseline_simfile = make_simfile()
    changed_remote = replace(baseline_simfile.objects[0], **changed)
    changed_simfile = replace(baseline_simfile, objects=(changed_remote,))

    baseline = render_manifest(build_manifest_rows((baseline_simfile,), {}, "a" * 64, "bucket"))
    updated = render_manifest(build_manifest_rows((changed_simfile,), {}, "a" * 64, "bucket"))

    assert updated.corpus_version != baseline.corpus_version


def test_publish_manifest_creates_version_file_and_latest_pointer(tmp_path: Path) -> None:
    rendered = render_fixture()
    published_at = datetime(2026, 7, 25, 1, 2, 3, tzinfo=timezone.utc)

    published = publish_manifest(tmp_path, rendered)
    publish_latest_manifest(tmp_path, published, "complete", published_at)

    version_path = tmp_path / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    latest_path = tmp_path / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert version_path.read_bytes() == rendered.content
    assert published.path == version_path
    assert published.latest_path == latest_path
    assert latest_path.read_bytes().endswith(b"\n")
    assert latest == {
        "corpus_version": rendered.corpus_version,
        "manifest_sha256": rendered.manifest_sha256,
        "manifest_path": f"manifests/{rendered.manifest_sha256}.jsonl",
        "overall_status": "complete",
        "published_at": "2026-07-25T01:02:03Z",
    }


@pytest.mark.parametrize("status", ["complete", "partial"])
def test_publish_latest_manifest_preserves_pointer_status(
    status: str,
    tmp_path: Path,
) -> None:
    published = publish_manifest(tmp_path, render_fixture())

    publish_latest_manifest(tmp_path, published, status, FIXED_TIME)  # type: ignore[arg-type]

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["overall_status"] == status


def test_publish_manifest_reuses_identical_existing_version(tmp_path: Path) -> None:
    rendered = render_fixture()
    first = publish_manifest(tmp_path, rendered)
    inode = first.path.stat().st_ino

    second = publish_manifest(tmp_path, rendered)

    assert second.path.stat().st_ino == inode


def test_publish_manifest_rejects_conflicting_existing_version(tmp_path: Path) -> None:
    rendered = render_fixture()
    version_path = tmp_path / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    version_path.parent.mkdir(parents=True)
    version_path.write_bytes(b"conflict\n")

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, rendered)

    assert raised.value.error == SyncError(
        "artifact",
        "artifact_write_failed",
        "A required manifest artifact write failed.",
    )
    assert str(version_path) not in str(raised.value)


def test_publish_manifest_rejects_content_that_does_not_match_rendered_hash(
    tmp_path: Path,
) -> None:
    rendered = replace(render_fixture(), content=b"tampered\n")

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, rendered)

    assert raised.value.error.code == "artifact_write_failed"
    assert not list((tmp_path / "manifests").glob(".*.tmp"))


def test_publish_manifest_alone_never_moves_latest_pointer(tmp_path: Path) -> None:
    latest_path = tmp_path / "latest.json"
    tmp_path.mkdir(parents=True, exist_ok=True)
    latest_path.write_bytes(b'{"existing":true}\n')

    publish_manifest(tmp_path, render_fixture())

    assert latest_path.read_bytes() == b'{"existing":true}\n'


def test_publish_manifest_accepts_an_identical_concurrent_winner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rendered = render_fixture()
    real_link = os.link
    link_calls = 0

    def concurrent_link(source: Path, destination: Path) -> None:
        nonlocal link_calls
        link_calls += 1
        destination.write_bytes(rendered.content)
        raise FileExistsError

    monkeypatch.setattr(corpus_manifest.os, "link", concurrent_link)

    published = publish_manifest(tmp_path, rendered)

    assert published.path.read_bytes() == rendered.content
    assert link_calls == 1
    assert not list(published.path.parent.glob(".*.tmp"))
    monkeypatch.setattr(corpus_manifest.os, "link", real_link)


def test_publish_manifest_rejects_a_conflicting_concurrent_winner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rendered = render_fixture()

    def concurrent_link(source: Path, destination: Path) -> None:
        del source
        destination.write_bytes(b"conflict\n")
        raise FileExistsError

    monkeypatch.setattr(corpus_manifest.os, "link", concurrent_link)

    with pytest.raises(ManifestPublicationError):
        publish_manifest(tmp_path, rendered)

    manifests_dir = tmp_path / "manifests"
    assert not list(manifests_dir.glob(".*.tmp"))
    assert list(manifests_dir.glob("*.jsonl"))[0].read_bytes() == b"conflict\n"


def test_latest_pointer_replacement_happens_after_manifest_directory_is_durable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, Path]] = []
    real_replace = os.replace
    real_fsync_directory = corpus_manifest._fsync_directory

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append(("replace", Path(destination)))
        real_replace(source, destination)

    def tracked_fsync_directory(path: Path) -> None:
        events.append(("fsync", path))
        real_fsync_directory(path)

    monkeypatch.setattr(corpus_manifest.os, "replace", tracked_replace)
    monkeypatch.setattr(corpus_manifest, "_fsync_directory", tracked_fsync_directory)
    published = publish_manifest(tmp_path, render_fixture())
    publish_latest_manifest(tmp_path, published, "complete", FIXED_TIME)

    manifest_fsync = events.index(("fsync", tmp_path / "manifests"))
    latest_replace = events.index(("replace", tmp_path / "latest.json"))
    latest_parent_fsync = len(events) - 1
    assert manifest_fsync < latest_replace < latest_parent_fsync
    assert events[latest_parent_fsync] == ("fsync", tmp_path)


def test_failed_latest_replace_cleans_temporary_file_and_keeps_old_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    published = publish_manifest(tmp_path, render_fixture())
    latest_path = tmp_path / "latest.json"
    latest_path.write_bytes(b'{"existing":true}\n')

    def failed_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("unsafe raw endpoint https://account-id.example")

    monkeypatch.setattr(corpus_manifest.os, "replace", failed_replace)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_latest_manifest(tmp_path, published, "complete", FIXED_TIME)

    assert raised.value.error.code == "artifact_write_failed"
    assert "unsafe raw endpoint" not in str(raised.value)
    assert latest_path.read_bytes() == b'{"existing":true}\n'
    assert not list(tmp_path.glob(".latest.json.*.tmp"))
