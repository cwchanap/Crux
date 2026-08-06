from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

import src.benchmark.reference_chart_manifest as reference_chart_manifest
from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.corpus_manifest import (
    ManifestPublicationError,
    build_manifest_rows,
    render_manifest,
)
from src.benchmark.r2_corpus_models import RemoteObject, SimfileInventory, SyncError
from src.benchmark.reference_chart_manifest import (
    REFERENCE_CHART_MANIFEST_SCHEMA,
    SelectionOutcome,
    SelectionRequest,
    _build_selection_row,
    _load_source_manifest,
    select_reference_manifest,
    validate_schema_golden,
)
from src.benchmark.reference_chart_selection import (
    ChartSelection,
    load_selection_overrides,
    select_reference_chart,
)

_FIXED_TIME = datetime(2026, 8, 5, tzinfo=timezone.utc)
_SOURCE_CORPUS_VERSION = "sha256:" + "c" * 64
_CHART_BODY = b"#TITLE: Example Song\n#ARTIST: Example Artist\n#DLEVEL: 99\n#00011: 01\n"
_HPA_321_ROW_KEYS = frozenset(
    {
        "schema_version",
        "corpus_version",
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
)
_SELECTION_ROW_KEYS = frozenset(
    {
        "source_manifest_sha256",
        "source_corpus_version",
        "selection_status",
        "selection_method",
        "selection_reason_codes",
        "selection_warnings",
        "set_def_key",
        "set_def_content_hash",
        "selected_chart_key",
        "selected_chart_content_hash",
        "selected_chart_cache_path",
        "selected_level_slot",
        "selected_level_label",
        "dlevel_raw",
        "dlevel_normalized",
        "title",
        "artist",
        "override_document_sha256",
        "selection_override",
    }
)
_SCHEMA_GOLDEN_PATH = (
    Path(__file__).parent / "schema_goldens/crux.reference-chart-manifest-v1.jsonl"
)


def _remote(simfile_id: int, key: str, body: bytes) -> RemoteObject:
    digest = sha256(body).hexdigest()
    return RemoteObject(
        key=f"{simfile_id}/{key}",
        size=len(body),
        etag=f"etag-{simfile_id}",
        etag_is_weak=False,
        last_modified=_FIXED_TIME,
        content_type="text/plain",
        cache_status="verified",
        sha256=digest,
        cache_path=f"sha256/{digest[:2]}/{digest}",
    )


def _source_row(
    simfile_id: int = 42,
    *,
    empty: bool = False,
    endpoint_sha256: str = "f" * 64,
    bucket: str = "simfile-dtx",
    objects: tuple[RemoteObject, ...] | None = None,
) -> dict[str, object]:
    source_objects = () if empty else objects or (_remote(simfile_id, "real.dtx", _CHART_BODY),)
    inventory = SimfileInventory(
        simfile_id=simfile_id,
        object_prefix=f"{simfile_id}/",
        objects=source_objects,
        sync_status="empty" if empty else "complete",
    )
    (base_row,) = build_manifest_rows((inventory,), {}, endpoint_sha256, bucket)
    (row,) = render_manifest((base_row,)).rows
    return row


def _source_manifest_bytes(rows: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(canonical_json_bytes(row, trailing_newline=True) for row in rows)


def _write_source_manifest(
    tmp_path: Path,
    rows: tuple[dict[str, object], ...],
) -> tuple[Path, bytes]:
    content = _source_manifest_bytes(rows)
    path = tmp_path / "source.jsonl"
    path.write_bytes(content)
    return path, content


def _source_rows_for_inventories(
    inventories: tuple[SimfileInventory, ...],
) -> tuple[dict[str, object], ...]:
    return render_manifest(build_manifest_rows(inventories, {}, "f" * 64, "simfile-dtx")).rows


def _install_cached_bodies(
    cache_dir: Path,
    fixtures: tuple[tuple[RemoteObject, bytes], ...],
) -> None:
    for remote, body in fixtures:
        assert remote.sha256 is not None
        cache_path = cache_dir / "sha256" / remote.sha256[:2] / remote.sha256
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(body)


def _render_selection_row(manifest_path: Path, cache_dir: Path) -> dict[str, object]:
    loaded = _load_source_manifest(manifest_path)
    (validated,) = loaded.rows
    overrides = load_selection_overrides(None, missing_ok=True)
    selection = select_reference_chart(validated.view, cache_dir=cache_dir, overrides=overrides)
    (row,) = render_manifest(
        (
            _build_selection_row(
                validated,
                source_manifest_sha256=loaded.source_manifest_sha256,
                override_document_sha256=overrides.document_sha256,
                selection=selection,
            ),
        )
    ).rows
    return row


def _selection_request(
    manifest_path: Path,
    cache_dir: Path,
    output_dir: Path,
    overrides_file: Path | None = None,
) -> SelectionRequest:
    return SelectionRequest(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        overrides_file=overrides_file,
        output_dir=output_dir,
        default_overrides_missing_ok=True,
    )


def _published_rows(outcome: SelectionOutcome) -> tuple[dict[str, object], ...]:
    assert outcome.manifest is not None
    return tuple(
        strict_json_loads(line[:-1], require_canonical=True)
        for line in outcome.manifest.path.read_bytes().splitlines(keepends=True)
    )  # type: ignore[return-value]


def _override_content(reason: str) -> bytes:
    return canonical_json_bytes(
        {
            "overrides": {"42": {"chart_key": "42/real.dtx", "reason": reason}},
            "schema_version": "crux.reference-chart-overrides/v1",
        },
        trailing_newline=True,
    )


def test_input_loader_reads_source_bytes_once_and_retains_the_exact_validated_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_row = _source_row(empty=True)
    manifest_path, content = _write_source_manifest(tmp_path, (source_row,))
    original_read_bytes = Path.read_bytes
    reads = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal reads
        if path == manifest_path:
            reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    loaded = _load_source_manifest(manifest_path)

    assert reads == 1
    assert loaded.source_manifest_sha256 == sha256(content).hexdigest()
    assert loaded.rows[0].source_row == source_row
    assert loaded.rows[0].view.inventory.simfile_id == 42


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"{}",
        b"{}\n\n",
        b"[]\n",
        b'{"cache_profile":"setdef_dtx_txt_v1"}\n',
    ],
    ids=["zero-records", "partial-line", "blank-line", "non-object", "malformed-row"],
)
def test_input_loader_rejects_empty_partial_blank_nonobject_and_malformed_records(
    tmp_path: Path,
    content: bytes,
) -> None:
    manifest_path = tmp_path / "source.jsonl"
    manifest_path.write_bytes(content)

    with pytest.raises(ValueError):
        _load_source_manifest(manifest_path)


def test_input_loader_rejects_noncanonical_record_bytes(tmp_path: Path) -> None:
    source_row = _source_row(empty=True)
    content = b"{ " + canonical_json_bytes(source_row, trailing_newline=False)[1:]
    manifest_path = tmp_path / "source.jsonl"
    manifest_path.write_bytes(content + b"\n")

    with pytest.raises(ValueError):
        _load_source_manifest(manifest_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "crux.unsupported/v1"),
        ("cache_profile", "unsupported-profile"),
        ("source_discovery_method", "unsupported-discovery"),
        ("simfile_id", "42"),
    ],
    ids=["wrong-schema", "wrong-cache-profile", "wrong-discovery", "malformed-row-view"],
)
def test_input_loader_rejects_invalid_hpa_321_row_contract(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source_row = _source_row(empty=True)
    source_row[field] = value
    manifest_path, _ = _write_source_manifest(tmp_path, (source_row,))

    with pytest.raises(ValueError):
        _load_source_manifest(manifest_path)


@pytest.mark.parametrize(
    "field,first_value,second_value",
    [
        ("corpus_version", "sha256:" + "a" * 64, "sha256:" + "b" * 64),
        ("source_endpoint_sha256", "a" * 64, "b" * 64),
        ("source_bucket", "first-bucket", "second-bucket"),
    ],
    ids=["corpus-version", "endpoint", "bucket"],
)
def test_input_loader_rejects_mixed_source_identity(
    tmp_path: Path,
    field: str,
    first_value: str,
    second_value: str,
) -> None:
    first = _source_row(42, empty=True)
    second = _source_row(43, empty=True)
    first["corpus_version"] = _SOURCE_CORPUS_VERSION
    second["corpus_version"] = _SOURCE_CORPUS_VERSION
    first[field] = first_value
    second[field] = second_value
    manifest_path, _ = _write_source_manifest(tmp_path, (first, second))

    with pytest.raises(ValueError):
        _load_source_manifest(manifest_path)


def test_input_loader_rejects_duplicate_simfile_ids(tmp_path: Path) -> None:
    first = _source_row(42, empty=True)
    second = _source_row(42, empty=True)
    first["corpus_version"] = _SOURCE_CORPUS_VERSION
    second["corpus_version"] = _SOURCE_CORPUS_VERSION
    manifest_path, _ = _write_source_manifest(tmp_path, (first, second))

    with pytest.raises(ValueError):
        _load_source_manifest(manifest_path)


def test_row_construction_passes_the_validated_hpa_321_mapping_through_verbatim(
    tmp_path: Path,
) -> None:
    remote = _remote(42, "real.dtx", _CHART_BODY)
    source_row = _source_row(objects=(remote,))
    manifest_path, source_content = _write_source_manifest(tmp_path, (source_row,))
    cache_dir = tmp_path / "cache"
    _install_cached_bodies(cache_dir, ((remote, _CHART_BODY),))

    row = _render_selection_row(manifest_path, cache_dir)
    assert set(row) == _HPA_321_ROW_KEYS | _SELECTION_ROW_KEYS
    assert row["schema_version"] == REFERENCE_CHART_MANIFEST_SCHEMA
    assert row["source_corpus_version"] == source_row["corpus_version"]
    assert row["source_manifest_sha256"] == sha256(source_content).hexdigest()
    assert row["selection_status"] == "selected"
    assert row["selection_method"] == "single_candidate_fallback"
    assert row["selection_reason_codes"] == []
    assert row["selected_chart_key"] == remote.key
    assert row["selected_chart_content_hash"] == remote.sha256
    assert row["selected_chart_cache_path"] == remote.cache_path
    assert row["selected_level_slot"] is None
    assert row["selected_level_label"] is None
    assert row["dlevel_raw"] == "99"
    assert row["dlevel_normalized"] == 99
    assert row["title"] == "Example Song"
    assert row["artist"] == "Example Artist"
    assert row["set_def_key"] is None
    assert row["set_def_content_hash"] is None
    assert row["selection_override"] is None
    assert {
        key: value
        for key, value in row.items()
        if key not in _SELECTION_ROW_KEYS | {"corpus_version"}
    } == {
        **{key: value for key, value in source_row.items() if key != "corpus_version"},
        "schema_version": REFERENCE_CHART_MANIFEST_SCHEMA,
    }


def test_row_construction_proves_the_typed_reader_without_reserializing_the_source(
    tmp_path: Path,
) -> None:
    source_row = _source_row(empty=True)
    manifest_path, _ = _write_source_manifest(tmp_path, (source_row,))

    loaded = _load_source_manifest(manifest_path)
    validated = loaded.rows[0]
    (rebuilt,) = build_manifest_rows(
        (validated.view.inventory,),
        {validated.view.inventory.simfile_id: validated.view.provenance},
        validated.view.source_endpoint_sha256,
        validated.view.source_bucket,
    )

    assert {
        key: value for key, value in validated.source_row.items() if key != "corpus_version"
    } == (rebuilt)


def test_row_construction_quarantines_empty_source_inventory_with_null_chart_fields(
    tmp_path: Path,
) -> None:
    source_row = _source_row(empty=True)
    manifest_path, _ = _write_source_manifest(tmp_path, (source_row,))

    row = _render_selection_row(manifest_path, tmp_path / "cache")
    assert row["selection_status"] == "quarantined"
    assert row["selection_method"] is None
    assert row["selection_reason_codes"] == ["source_inventory_unusable"]
    assert all(
        row[field] is None
        for field in (
            "selected_chart_key",
            "selected_chart_content_hash",
            "selected_chart_cache_path",
            "selected_level_slot",
            "selected_level_label",
            "dlevel_raw",
            "dlevel_normalized",
            "title",
            "artist",
        )
    )


def test_row_construction_keeps_set_def_identity_after_selected_chart_parse_failure(
    tmp_path: Path,
) -> None:
    set_def_body = b"#L5FILE: broken.dtx\n"
    broken_chart_body = b"\xff"
    set_def = _remote(42, "set.def", set_def_body)
    broken_chart = _remote(42, "broken.dtx", broken_chart_body)
    source_row = _source_row(objects=(set_def, broken_chart))
    manifest_path, _ = _write_source_manifest(tmp_path, (source_row,))
    cache_dir = tmp_path / "cache"
    _install_cached_bodies(cache_dir, ((set_def, set_def_body), (broken_chart, broken_chart_body)))

    row = _render_selection_row(manifest_path, cache_dir)
    assert row["selection_status"] == "quarantined"
    assert row["selection_reason_codes"] == ["selected_chart_parse_failed"]
    assert row["set_def_key"] == set_def.key
    assert row["set_def_content_hash"] == set_def.sha256
    assert row["selected_chart_key"] is None


@pytest.mark.parametrize(
    "selection",
    [
        ChartSelection(
            status="quarantined",
            method=None,
            reason_codes=("source_inventory_unusable",),
            warnings=(),
            set_def=None,
            selected_chart=None,
            selected_level_slot=None,
            selected_level_label=None,
            dlevel_raw="99",
            dlevel_normalized=None,
            title=None,
            artist=None,
            override=None,
        ),
        ChartSelection(
            status="selected",
            method="override",
            reason_codes=(),
            warnings=(),
            set_def=None,
            selected_chart=RemoteObject(
                key="42/real.dtx",
                size=1,
                etag="etag",
                etag_is_weak=False,
                last_modified=_FIXED_TIME,
                content_type="text/plain",
                cache_status="verified",
                sha256=None,
                cache_path=None,
            ),
            selected_level_slot=None,
            selected_level_label=None,
            dlevel_raw=None,
            dlevel_normalized=None,
            title="Example Song",
            artist="Example Artist",
            override=None,
        ),
    ],
    ids=["quarantined-metadata", "selected-missing-identities"],
)
def test_row_construction_rejects_an_invalid_closed_selection_shape(
    tmp_path: Path,
    selection: ChartSelection,
) -> None:
    source_row = _source_row(empty=True)
    manifest_path, _ = _write_source_manifest(tmp_path, (source_row,))
    loaded = _load_source_manifest(manifest_path)
    (validated,) = loaded.rows

    with pytest.raises(ValueError):
        _build_selection_row(
            validated,
            source_manifest_sha256=loaded.source_manifest_sha256,
            override_document_sha256="e" * 64,
            selection=selection,
        )


def _schema_golden_rows() -> list[dict[str, object]]:
    return [
        strict_json_loads(line[:-1], require_canonical=True)
        for line in _SCHEMA_GOLDEN_PATH.read_bytes().splitlines(keepends=True)
    ]  # type: ignore[return-value]


def _canonical_jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(row, trailing_newline=True) for row in rows)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows[1].__setitem__("selection_reason_codes", []),
        lambda rows: rows[1].__setitem__("selected_chart_key", "43/real.dtx"),
        lambda rows: rows[0].__setitem__("selected_chart_cache_path", "../escape"),
        lambda rows: rows[0].__setitem__("selection_override", {"chart_key": "42/real.dtx"}),
        lambda rows: rows[1].__setitem__("source_endpoint_sha256", "F" * 64),
        lambda rows: rows[1].__setitem__("corpus_version", "sha256:" + "a" * 64),
    ],
    ids=[
        "row-count",
        "empty-quarantine-reasons",
        "quarantine-chart-identity",
        "chart-cache-path",
        "override-shape",
        "source-identity",
        "derived-version",
    ],
)
def test_schema_golden_validator_rejects_closed_contract_drift(
    mutation,
) -> None:
    rows = _schema_golden_rows()
    mutation(rows)

    with pytest.raises(ValueError):
        validate_schema_golden(REFERENCE_CHART_MANIFEST_SCHEMA, _canonical_jsonl(rows))


def test_publication_is_byte_identical_for_identical_normalized_selection_rows(
    tmp_path: Path,
) -> None:
    remote = _remote(42, "real.dtx", _CHART_BODY)
    source_row = _source_row(objects=(remote,))
    manifest_path, _ = _write_source_manifest(tmp_path, (source_row,))
    cache_dir = tmp_path / "cache"
    _install_cached_bodies(cache_dir, ((remote, _CHART_BODY),))

    first = select_reference_manifest(
        _selection_request(manifest_path, cache_dir, tmp_path / "first-output")
    )
    second = select_reference_manifest(
        _selection_request(manifest_path, cache_dir, tmp_path / "second-output")
    )

    assert first.status == "complete"
    assert first.exit_code == 0
    assert first.selected_count == 1
    assert first.quarantined_count == 0
    assert first.manifest is not None
    assert second.manifest is not None
    assert first.manifest.path.read_bytes() == second.manifest.path.read_bytes()
    assert first.manifest.corpus_version == second.manifest.corpus_version
    assert first.manifest.manifest_sha256 == second.manifest.manifest_sha256


def test_publication_preserves_verbatim_source_object_array_order(tmp_path: Path) -> None:
    chart = _remote(42, "real.dtx", _CHART_BODY)
    unrelated = _remote(42, "notes.txt", b"audit note")
    source_row = _source_row(objects=(chart, unrelated))
    source_row["objects"] = list(reversed(source_row["objects"]))
    expected_objects = source_row["objects"]
    manifest_path, _ = _write_source_manifest(tmp_path, (source_row,))
    cache_dir = tmp_path / "cache"
    _install_cached_bodies(cache_dir, ((chart, _CHART_BODY),))

    outcome = select_reference_manifest(
        _selection_request(manifest_path, cache_dir, tmp_path / "output")
    )

    (published_row,) = _published_rows(outcome)
    assert published_row["objects"] == expected_objects
    assert [remote["key"] for remote in published_row["objects"]] == [
        "42/real.dtx",
        "42/notes.txt",
    ]


def test_publication_keeps_selected_and_quarantined_rows_and_marks_partial(
    tmp_path: Path,
) -> None:
    remote = _remote(42, "real.dtx", _CHART_BODY)
    selected = SimfileInventory(42, "42/", (remote,), "complete")
    quarantined = SimfileInventory(43, "43/", (), "empty")
    rows = _source_rows_for_inventories((selected, quarantined))
    manifest_path, _ = _write_source_manifest(tmp_path, rows)
    cache_dir = tmp_path / "cache"
    _install_cached_bodies(cache_dir, ((remote, _CHART_BODY),))

    outcome = select_reference_manifest(
        _selection_request(manifest_path, cache_dir, tmp_path / "output")
    )

    published_rows = _published_rows(outcome)
    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert outcome.selected_count == 1
    assert outcome.quarantined_count == 1
    assert [row["simfile_id"] for row in published_rows] == [42, 43]
    assert published_rows[0]["selection_status"] == "selected"
    assert published_rows[1]["selection_status"] == "quarantined"
    assert published_rows[1]["selection_reason_codes"] == ["source_inventory_unusable"]


def test_publication_publishes_an_all_quarantined_manifest(tmp_path: Path) -> None:
    rows = _source_rows_for_inventories(
        (
            SimfileInventory(42, "42/", (), "empty"),
            SimfileInventory(43, "43/", (), "empty"),
        )
    )
    manifest_path, _ = _write_source_manifest(tmp_path, rows)

    outcome = select_reference_manifest(
        _selection_request(manifest_path, tmp_path / "cache", tmp_path / "output")
    )

    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert outcome.manifest is not None
    assert outcome.selected_count == 0
    assert outcome.quarantined_count == 2
    assert all(row["selection_status"] == "quarantined" for row in _published_rows(outcome))


def test_publication_hashes_the_exact_override_document_bytes(tmp_path: Path) -> None:
    remote = _remote(42, "real.dtx", _CHART_BODY)
    source_row = _source_row(objects=(remote,))
    manifest_path, _ = _write_source_manifest(tmp_path, (source_row,))
    cache_dir = tmp_path / "cache"
    _install_cached_bodies(cache_dir, ((remote, _CHART_BODY),))
    content = _override_content("first audit")
    overrides_file = tmp_path / "overrides.json"
    overrides_file.write_bytes(content)

    outcome = select_reference_manifest(
        _selection_request(manifest_path, cache_dir, tmp_path / "output", overrides_file)
    )

    (row,) = _published_rows(outcome)
    assert row["selection_method"] == "override"
    assert row["override_document_sha256"] == sha256(content).hexdigest()
    assert row["selection_override"] == {
        "chart_key": "42/real.dtx",
        "reason": "first audit",
    }


def test_publication_rekeys_when_only_override_bytes_change(tmp_path: Path) -> None:
    remote = _remote(42, "real.dtx", _CHART_BODY)
    source_row = _source_row(objects=(remote,))
    manifest_path, _ = _write_source_manifest(tmp_path, (source_row,))
    cache_dir = tmp_path / "cache"
    _install_cached_bodies(cache_dir, ((remote, _CHART_BODY),))
    first_overrides = tmp_path / "first-overrides.json"
    second_overrides = tmp_path / "second-overrides.json"
    first_overrides.write_bytes(_override_content("first audit"))
    second_overrides.write_bytes(_override_content("second audit"))

    first = select_reference_manifest(
        _selection_request(manifest_path, cache_dir, tmp_path / "first-output", first_overrides)
    )
    second = select_reference_manifest(
        _selection_request(manifest_path, cache_dir, tmp_path / "second-output", second_overrides)
    )

    assert first.manifest is not None
    assert second.manifest is not None
    assert first.manifest.path.read_bytes() != second.manifest.path.read_bytes()
    assert first.manifest.corpus_version != second.manifest.corpus_version
    assert first.manifest.manifest_sha256 != second.manifest.manifest_sha256


def test_publication_returns_fatal_without_a_manifest_for_invalid_source_input(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "source.jsonl"
    manifest_path.write_bytes(b"")

    outcome = select_reference_manifest(
        _selection_request(manifest_path, tmp_path / "cache", tmp_path / "output")
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert outcome.manifest is None


def test_publication_returns_fatal_without_a_manifest_when_immutable_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _remote(42, "real.dtx", _CHART_BODY)
    source_row = _source_row(objects=(remote,))
    manifest_path, _ = _write_source_manifest(tmp_path, (source_row,))
    cache_dir = tmp_path / "cache"
    _install_cached_bodies(cache_dir, ((remote, _CHART_BODY),))

    def fail_publication(*_args: object, **_kwargs: object) -> None:
        raise ManifestPublicationError(
            SyncError("artifact", "artifact_write_failed", "publication failed")
        )

    monkeypatch.setattr(reference_chart_manifest, "publish_manifest", fail_publication)

    outcome = select_reference_manifest(
        _selection_request(manifest_path, cache_dir, tmp_path / "output")
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert outcome.manifest is None
