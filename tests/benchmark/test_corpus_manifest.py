import json
import os
import stat
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from threading import Thread

import pytest

import src.benchmark.corpus_manifest as corpus_manifest
import src.benchmark.durability as durability
from src.benchmark.corpus_manifest import (
    ManifestPublicationError,
    build_manifest_rows,
    canonical_json_line,
    inventory_from_manifest_row,
    manifest_row_view_from_row,
    publish_immutable_bytes,
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


def hpa_321_row() -> tuple[dict[str, object], SimfileInventory, ProvenanceRecord, str]:
    source = make_simfile(status="partial", error_code="object_head_failed")
    provenance = ProvenanceRecord(
        source_origin="Example archive",
        source_author_or_pack="Author",
        source_reference="https://example.invalid/source",
        rights_status="verified",
        redistribution_allowed=True,
        provenance_notes="Verified against the archive index.",
    )
    endpoint_hash = "f" * 64
    (base_row,) = build_manifest_rows(
        (source,),
        {source.simfile_id: provenance},
        endpoint_hash,
        "simfile-dtx",
    )
    (row,) = render_manifest((base_row,)).rows
    return row, source, provenance, endpoint_hash


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


def test_publish_immutable_bytes_is_the_public_publisher_entrypoint(tmp_path: Path) -> None:
    # Cross-module callers (e.g. reference_timing.publish_immutable_content) go
    # through this public name rather than the leading-underscore internals.  It
    # must expose the same hash-checked, durable publication as _publish_immutable.
    path = tmp_path / "payload.jsonl"
    content = b'{"a":1}\n'

    publish_immutable_bytes(path, content, sha256(content).hexdigest())

    assert path.read_bytes() == content

    # A hash mismatch surfaces as ManifestPublicationError — the full durability
    # / conflict path of _publish_immutable is delegated, not duplicated.
    with pytest.raises(ManifestPublicationError):
        publish_immutable_bytes(path, content, "0" * 64)


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


def test_manifest_row_view_reconstructs_the_hpa_321_contract() -> None:
    row, source, provenance, endpoint_hash = hpa_321_row()

    view = manifest_row_view_from_row(row)

    assert view.inventory.simfile_id == source.simfile_id
    assert view.inventory.object_prefix == source.object_prefix
    assert view.inventory.sync_status == source.sync_status
    assert view.inventory.sync_errors == source.sync_errors
    assert all(remote.errors == () for remote in view.inventory.objects)
    assert view.provenance == provenance
    assert view.corpus_version == row["corpus_version"]
    assert view.cache_profile == "setdef_dtx_txt_v1"
    assert view.source_endpoint_sha256 == endpoint_hash
    assert view.source_bucket == "simfile-dtx"
    assert view.source_discovery_method == "r2_list_objects_v2"
    assert inventory_from_manifest_row(row) == view.inventory

    rebuilt = build_manifest_rows(
        (view.inventory,),
        {view.inventory.simfile_id: view.provenance},
        view.source_endpoint_sha256,
        view.source_bucket,
    )[0]
    expected = {key: value for key, value in row.items() if key != "corpus_version"}

    assert rebuilt == expected


@pytest.mark.parametrize("extra", [False, True])
def test_manifest_row_view_rejects_nonexact_row_field_set(extra: bool) -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    if extra:
        invalid["unexpected"] = "field"
    else:
        invalid.pop("cache_profile")

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


@pytest.mark.parametrize("extra", [False, True])
def test_manifest_row_view_rejects_nonexact_object_field_set(extra: bool) -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    objects = invalid["objects"]
    assert isinstance(objects, list)
    first = objects[0]
    assert isinstance(first, dict)
    if extra:
        first["unexpected"] = "field"
    else:
        first.pop("cache_path")

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


def test_manifest_row_view_rejects_duplicate_object_keys() -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    objects = invalid["objects"]
    assert isinstance(objects, list)
    objects.append(deepcopy(objects[0]))

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


@pytest.mark.parametrize(
    ("location", "value"),
    [
        ("schema_version", "crux.r2-corpus-manifest/v2"),
        ("cache_profile", "different_profile"),
        ("source_discovery_method", "listing"),
        ("sync_status", "unknown"),
        ("object.cache_status", "cached"),
        ("error.scope", "unknown"),
        ("error.code", "unknown"),
    ],
)
def test_manifest_row_view_rejects_invalid_serialized_enums(location: str, value: str) -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    if location.startswith("object."):
        objects = invalid["objects"]
        assert isinstance(objects, list)
        object_field = location.removeprefix("object.")
        assert isinstance(objects[0], dict)
        objects[0][object_field] = value
    elif location.startswith("error."):
        errors = invalid["sync_errors"]
        assert isinstance(errors, list)
        error_field = location.removeprefix("error.")
        assert isinstance(errors[0], dict)
        errors[0][error_field] = value
    else:
        invalid[location] = value

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


@pytest.mark.parametrize("value", [42.0, True, "42"])
def test_manifest_row_view_rejects_noninteger_simfile_ids(value: object) -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    invalid["simfile_id"] = value

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


@pytest.mark.parametrize(
    ("object_prefix", "object_key"),
    [
        ("2", "20/chart.dtx"),
        ("3/", "3/chart.dtx"),
        ("2/nested/", "2/nested/chart.dtx"),
    ],
)
def test_manifest_row_view_rejects_prefixes_outside_the_simfile_top_level_contract(
    object_prefix: str,
    object_key: str,
) -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    invalid["object_prefix"] = object_prefix
    invalid["sync_status"] = "complete"
    invalid["sync_errors"] = []
    objects = invalid["objects"]
    assert isinstance(objects, list)
    assert isinstance(objects[0], dict)
    objects[0]["key"] = object_key

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


def test_manifest_row_view_accepts_zero_padded_numeric_top_level_prefix() -> None:
    row, _, _, _ = hpa_321_row()
    valid = deepcopy(row)
    valid["object_prefix"] = "02/"
    valid["sync_status"] = "complete"
    valid["sync_errors"] = []
    objects = valid["objects"]
    assert isinstance(objects, list)
    assert isinstance(objects[0], dict)
    objects[0]["key"] = "02/chart.dtx"

    view = manifest_row_view_from_row(valid)

    assert view.inventory.simfile_id == 2
    assert view.inventory.object_prefix == "02/"
    assert view.inventory.objects[0].key == "02/chart.dtx"


@pytest.mark.parametrize(
    ("location", "value"),
    [
        ("source_endpoint_sha256", "F" * 64),
        ("object.sha256", "c" * 63),
        ("corpus_version", "sha256:" + "C" * 64),
    ],
)
def test_manifest_row_view_rejects_invalid_hashes(location: str, value: str) -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    if location == "object.sha256":
        objects = invalid["objects"]
        assert isinstance(objects, list)
        assert isinstance(objects[0], dict)
        objects[0]["sha256"] = value
    else:
        invalid[location] = value

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


def test_manifest_row_view_rejects_malformed_object_timestamps() -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    objects = invalid["objects"]
    assert isinstance(objects, list)
    assert isinstance(objects[0], dict)
    objects[0]["last_modified"] = "2026-02-30T01:02:03Z"

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


def test_manifest_row_view_rejects_error_references_to_unknown_objects() -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    errors = invalid["sync_errors"]
    assert isinstance(errors, list)
    assert isinstance(errors[0], dict)
    errors[0]["object_key"] = "2/missing.dtx"

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


@pytest.mark.parametrize("value", [42, "not-sha256-prefixed"])
def test_manifest_row_view_rejects_invalid_corpus_version_format(value: object) -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    invalid["corpus_version"] = value

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


def test_manifest_row_view_rejects_non_list_objects() -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    invalid["objects"] = "not-a-list"

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


def test_manifest_row_view_rejects_non_list_sync_errors() -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    invalid["sync_errors"] = "not-a-list"

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


def test_manifest_row_view_rejects_error_entry_with_wrong_keys() -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    errors = invalid["sync_errors"]
    assert isinstance(errors, list)
    assert isinstance(errors[0], dict)
    errors[0]["unexpected"] = "field"
    errors[0].pop("message")

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


@pytest.mark.parametrize("field", ["source_origin", "source_author_or_pack"])
def test_manifest_row_view_rejects_non_string_nullable_provenance(field: str) -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    invalid[field] = 42

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rights_status", ""),
        ("redistribution_allowed", "yes"),
    ],
)
def test_manifest_row_view_rejects_invalid_rights_fields(field: str, value: object) -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    invalid[field] = value

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


def test_manifest_row_view_rejects_non_string_object_key() -> None:
    row, _, _, _ = hpa_321_row()
    invalid = deepcopy(row)
    objects = invalid["objects"]
    assert isinstance(objects, list)
    assert isinstance(objects[0], dict)
    objects[0]["key"] = 42

    with pytest.raises(ValueError):
        manifest_row_view_from_row(invalid)


def test_manifest_row_view_preserves_hpa_321_empty_prefix_errors() -> None:
    source = SimfileInventory(
        2,
        "2/",
        (),
        "empty",
        (SyncError("simfile", "empty_prefix", "No objects were listed.", "2/"),),
    )
    (base_row,) = build_manifest_rows((source,), {}, "a" * 64, "simfile-dtx")
    (row,) = render_manifest((base_row,)).rows

    view = manifest_row_view_from_row(row)

    assert view.inventory == source


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


def test_publish_manifest_rejects_a_matching_symlink_destination(tmp_path: Path) -> None:
    rendered = render_fixture()
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(rendered.content)
    version_path = manifests_dir / f"{rendered.manifest_sha256}.jsonl"
    version_path.symlink_to(outside)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, rendered)

    assert raised.value.error.code == "artifact_write_failed"
    assert version_path.is_symlink()
    assert outside.read_bytes() == rendered.content


def test_publish_manifest_rejects_a_non_regular_destination(tmp_path: Path) -> None:
    rendered = render_fixture()
    version_path = tmp_path / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    version_path.mkdir(parents=True)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, rendered)

    assert raised.value.error.code == "artifact_write_failed"
    assert version_path.is_dir()


def test_publish_manifest_rejects_a_fifo_destination_without_blocking(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is unavailable on this platform")

    rendered = render_fixture()
    version_path = tmp_path / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    version_path.parent.mkdir(parents=True)
    os.mkfifo(version_path)

    result: dict[str, object] = {}

    def publish() -> None:
        try:
            publish_manifest(tmp_path, rendered)
        except ManifestPublicationError as error:
            result["error"] = error

    worker = Thread(target=publish, daemon=True)
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "publish_manifest hung opening a FIFO destination"
    error = result.get("error")
    assert isinstance(error, ManifestPublicationError)
    assert error.error.code == "artifact_write_failed"
    assert stat.S_ISFIFO(os.stat(version_path, follow_symlinks=False).st_mode)


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


def test_publish_manifest_rejects_a_concurrent_matching_symlink_winner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rendered = render_fixture()
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(rendered.content)

    def concurrent_link(source: Path, destination: Path) -> None:
        del source
        destination.symlink_to(outside)
        raise FileExistsError

    monkeypatch.setattr(corpus_manifest.os, "link", concurrent_link)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, rendered)

    assert raised.value.error.code == "artifact_write_failed"
    assert outside.read_bytes() == rendered.content
    assert not list((tmp_path / "manifests").glob(".*.tmp"))


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


def test_concurrent_identical_winner_is_fsynced_before_its_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rendered = render_fixture()
    version_path = tmp_path / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    events: list[str] = []
    real_fsync = os.fsync

    def concurrent_link(source: Path, destination: Path) -> None:
        del source
        destination.write_bytes(rendered.content)
        events.append("winner_visible")
        raise FileExistsError

    def tracked_fsync(descriptor: int) -> None:
        descriptor_stat = os.fstat(descriptor)
        if version_path.exists():
            winner_stat = os.stat(version_path, follow_symlinks=False)
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (
                winner_stat.st_dev,
                winner_stat.st_ino,
            ):
                events.append("winner_fsync")
        if stat.S_ISDIR(descriptor_stat.st_mode) and version_path.parent.exists():
            directory_stat = os.stat(version_path.parent, follow_symlinks=False)
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (
                directory_stat.st_dev,
                directory_stat.st_ino,
            ):
                events.append("manifests_fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(corpus_manifest.os, "link", concurrent_link)
    monkeypatch.setattr(corpus_manifest.os, "fsync", tracked_fsync)

    publish_manifest(tmp_path, rendered)

    visible = events.index("winner_visible")
    winner_fsync = events.index("winner_fsync")
    following_directory_fsync = events.index("manifests_fsync", winner_fsync)
    assert visible < winner_fsync < following_directory_fsync


def test_fresh_multilevel_output_tree_fsyncs_every_created_child_attachment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "level-one" / "level-two" / "output"
    fsynced: list[Path] = []
    real_fsync_directory = durability.fsync_directory

    def tracked_fsync_directory(path: Path) -> None:
        fsynced.append(path)
        real_fsync_directory(path)

    monkeypatch.setattr(durability, "fsync_directory", tracked_fsync_directory)
    monkeypatch.setattr(corpus_manifest, "fsync_directory", tracked_fsync_directory)

    publish_manifest(output_dir, render_fixture())

    assert fsynced == [
        tmp_path,
        tmp_path / "level-one",
        tmp_path / "level-one" / "level-two",
        output_dir,
        output_dir / "manifests",
        output_dir / "manifests",
    ]


def test_latest_pointer_replacement_happens_after_manifest_directory_is_durable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, Path]] = []
    real_replace = os.replace
    real_fsync_directory = durability.fsync_directory

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append(("replace", Path(destination)))
        real_replace(source, destination)

    def tracked_fsync_directory(path: Path) -> None:
        events.append(("fsync", path))
        real_fsync_directory(path)

    monkeypatch.setattr(corpus_manifest.os, "replace", tracked_replace)
    monkeypatch.setattr(durability, "fsync_directory", tracked_fsync_directory)
    monkeypatch.setattr(corpus_manifest, "fsync_directory", tracked_fsync_directory)
    published = publish_manifest(tmp_path, render_fixture())
    publish_latest_manifest(tmp_path, published, "complete", FIXED_TIME)

    manifest_fsync = events.index(("fsync", tmp_path / "manifests"))
    latest_replace = events.index(("replace", tmp_path / "latest.json"))
    latest_parent_fsync = len(events) - 1
    assert manifest_fsync < latest_replace < latest_parent_fsync
    assert events[latest_parent_fsync] == ("fsync", tmp_path)


def test_publish_manifest_sanitizes_uuid_generation_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unsafe = "unsafe raw endpoint https://account-id.example"

    def failed_uuid() -> object:
        raise RuntimeError(unsafe)

    monkeypatch.setattr(corpus_manifest, "uuid4", failed_uuid)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, render_fixture())

    assert str(raised.value) == "artifact_write_failed"
    assert unsafe not in str(raised.value)


def test_publish_latest_sanitizes_uuid_generation_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    published = publish_manifest(tmp_path, render_fixture())
    unsafe = "unsafe signed query X-Amz-Signature=secret"

    def failed_uuid() -> object:
        raise RuntimeError(unsafe)

    monkeypatch.setattr(corpus_manifest, "uuid4", failed_uuid)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_latest_manifest(tmp_path, published, "complete", FIXED_TIME)

    assert str(raised.value) == "artifact_write_failed"
    assert unsafe not in str(raised.value)


def test_publish_latest_sanitizes_non_utf8_pointer_fields(tmp_path: Path) -> None:
    published = publish_manifest(tmp_path, render_fixture())
    unsafe = replace(published, relative_path="\ud800")

    with pytest.raises(ManifestPublicationError) as raised:
        publish_latest_manifest(tmp_path, unsafe, "complete", FIXED_TIME)

    assert str(raised.value) == "artifact_write_failed"
    assert not (tmp_path / "latest.json").exists()


def test_publish_latest_sanitizes_canonical_serialization_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    published = publish_manifest(tmp_path, render_fixture())
    unsafe = "unsafe credential AKIA-not-a-real-key"

    def failed_dumps(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise RuntimeError(unsafe)

    monkeypatch.setattr(corpus_manifest.json, "dumps", failed_dumps)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_latest_manifest(tmp_path, published, "complete", FIXED_TIME)

    assert str(raised.value) == "artifact_write_failed"
    assert unsafe not in str(raised.value)


def test_publish_manifest_sanitizes_unexpected_path_operation_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unsafe = "unsafe header Authorization=secret"

    def failed_link(source: Path, destination: Path) -> None:
        del source, destination
        raise RuntimeError(unsafe)

    monkeypatch.setattr(corpus_manifest.os, "link", failed_link)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, render_fixture())

    assert str(raised.value) == "artifact_write_failed"
    assert unsafe not in str(raised.value)
    assert not list((tmp_path / "manifests").glob(".*.tmp"))


def test_publish_manifest_sanitizes_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unsafe = "unsafe cleanup endpoint https://account-id.example"

    def failed_link(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("install failed")

    def failed_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        del path, missing_ok
        raise RuntimeError(unsafe)

    monkeypatch.setattr(corpus_manifest.os, "link", failed_link)
    monkeypatch.setattr(Path, "unlink", failed_unlink)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, render_fixture())

    assert str(raised.value) == "artifact_write_failed"
    assert unsafe not in str(raised.value)


def test_publish_latest_rejects_naive_publication_time_without_moving_pointer(
    tmp_path: Path,
) -> None:
    published = publish_manifest(tmp_path, render_fixture())
    latest_path = tmp_path / "latest.json"
    latest_path.write_bytes(b'{"existing":true}\n')

    with pytest.raises(ValueError, match="published_at must be timezone-aware"):
        publish_latest_manifest(
            tmp_path,
            published,
            "complete",
            datetime(2026, 7, 25, 1, 2, 3),
        )

    assert latest_path.read_bytes() == b'{"existing":true}\n'


def test_publish_latest_normalizes_aware_publication_time_to_utc(tmp_path: Path) -> None:
    published = publish_manifest(tmp_path, render_fixture())
    pacific_time = datetime(
        2026,
        7,
        24,
        18,
        2,
        3,
        tzinfo=timezone(timedelta(hours=-7)),
    )

    publish_latest_manifest(tmp_path, published, "complete", pacific_time)

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["published_at"] == "2026-07-25T01:02:03Z"


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


def test_reversed_equivalent_error_inputs_render_byte_identically() -> None:
    errors = (
        SyncError("object", "object_get_failed", "Object body read failed.", "2/a.dtx"),
        SyncError("object", "object_get_failed", "Object body read failed.", None),
    )
    first = replace(make_simfile(), sync_status="partial", sync_errors=errors)
    second = replace(make_simfile(), sync_status="partial", sync_errors=tuple(reversed(errors)))

    first_rendered = render_manifest(build_manifest_rows((first,), {}, "a" * 64, "bucket"))
    second_rendered = render_manifest(build_manifest_rows((second,), {}, "a" * 64, "bucket"))

    assert first_rendered.content == second_rendered.content
    assert first_rendered.corpus_version == second_rendered.corpus_version


def test_publish_manifest_wraps_durable_directory_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unsafe = "unsafe raw endpoint https://account-id.example"

    def failed_ensure(path: Path) -> None:
        del path
        raise RuntimeError(unsafe)

    monkeypatch.setattr(corpus_manifest, "ensure_durable_directory", failed_ensure)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, render_fixture())

    assert raised.value.error.code == "artifact_write_failed"
    assert unsafe not in str(raised.value)


class _BrokenTzDatetime(datetime):
    """Datetime subclass whose utcoffset() raises to exercise the except branch."""

    def utcoffset(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_publish_latest_wraps_utcoffset_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    published = publish_manifest(tmp_path, render_fixture())
    latest_path = tmp_path / "latest.json"
    latest_path.write_bytes(b'{"existing":true}\n')

    broken = _BrokenTzDatetime(2026, 7, 25, 1, 2, 3, tzinfo=timezone.utc)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_latest_manifest(tmp_path, published, "complete", broken)

    assert raised.value.error.code == "artifact_write_failed"
    assert latest_path.read_bytes() == b'{"existing":true}\n'


def test_publish_latest_wraps_format_timestamp_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    published = publish_manifest(tmp_path, render_fixture())
    latest_path = tmp_path / "latest.json"
    latest_path.write_bytes(b'{"existing":true}\n')

    unsafe = "unsafe signed query X-Amz-Signature=secret"

    def failed_format(value: datetime) -> str:
        del value
        raise RuntimeError(unsafe)

    monkeypatch.setattr(corpus_manifest, "format_manifest_timestamp", failed_format)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_latest_manifest(tmp_path, published, "complete", FIXED_TIME)

    assert raised.value.error.code == "artifact_write_failed"
    assert unsafe not in str(raised.value)
    assert latest_path.read_bytes() == b'{"existing":true}\n'


def test_publish_latest_cleanup_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    published = publish_manifest(tmp_path, render_fixture())
    latest_path = tmp_path / "latest.json"
    latest_path.write_bytes(b'{"existing":true}\n')

    def failed_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("replace failed")

    def failed_unlink(self: Path, missing_ok: bool = False) -> None:
        del self, missing_ok
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(corpus_manifest.os, "replace", failed_replace)
    monkeypatch.setattr(Path, "unlink", failed_unlink)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_latest_manifest(tmp_path, published, "complete", FIXED_TIME)

    assert raised.value.error.code == "artifact_write_failed"
    assert latest_path.read_bytes() == b'{"existing":true}\n'


def test_publish_manifest_rejects_destination_when_no_follow_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rendered = render_fixture()
    version_path = tmp_path / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    version_path.parent.mkdir(parents=True)
    version_path.write_bytes(rendered.content)

    # Bypass the durability helpers (which also rely on O_NOFOLLOW) so the
    # missing flag is first encountered inside _regular_file_open_flags.
    monkeypatch.setattr(corpus_manifest, "ensure_durable_directory", lambda _p: None)
    monkeypatch.setattr(corpus_manifest, "fsync_directory", lambda _p: None)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, rendered)

    assert raised.value.error.code == "artifact_write_failed"


def test_publish_manifest_rejects_destination_when_nonblock_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rendered = render_fixture()
    version_path = tmp_path / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    version_path.parent.mkdir(parents=True)
    version_path.write_bytes(rendered.content)

    monkeypatch.setattr(corpus_manifest, "ensure_durable_directory", lambda _p: None)
    monkeypatch.setattr(corpus_manifest, "fsync_directory", lambda _p: None)
    monkeypatch.delattr(os, "O_NONBLOCK", raising=False)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, rendered)

    assert raised.value.error.code == "artifact_write_failed"


def test_publish_manifest_rejects_destination_whose_binding_changed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rendered = render_fixture()
    version_path = tmp_path / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    version_path.parent.mkdir(parents=True)
    version_path.write_bytes(rendered.content)

    real_stat = Path.stat

    def shifted_stat(self: Path, follow_symlinks: bool = True) -> os.stat_result:
        if not follow_symlinks and self == version_path:
            return os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", shifted_stat)

    with pytest.raises(ManifestPublicationError) as raised:
        publish_manifest(tmp_path, rendered)

    assert raised.value.error.code == "artifact_write_failed"
