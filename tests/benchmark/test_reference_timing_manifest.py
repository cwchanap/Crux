from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pytest

import src.benchmark.corpus_manifest as corpus_manifest
import src.benchmark.reference_chart_manifest as reference_chart_manifest
import src.benchmark.reference_timing_manifest as reference_timing_manifest
from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.r2_corpus_models import PublishedManifest, SimfileInventory
from src.benchmark.reference_chart_manifest import (
    REFERENCE_CHART_MANIFEST_SCHEMA,
    select_reference_manifest,
)
from src.benchmark.reference_timing_manifest import (
    REFERENCE_TIMING_MANIFEST_SCHEMA,
    TIMING_SEMANTICS_VERSION,
    TimingRowResolution,
    build_reference_timing_outcome,
    build_timing_row,
    failed_reference_timing_outcome,
    load_reference_chart_manifest,
    upstream_chart_unavailable_resolution,
    validate_schema_golden,
)

# Reuse the HPA-322 fixture builders from the sibling test module by file path
# (no sys.path pollution, isort-friendly).  The brief requires reuse of
# ``_source_row`` / ``_source_manifest_bytes`` / ``_source_rows_for_inventories``
# / ``_render_selection_row`` / ``_published_rows`` rather than hand-rolling
# HPA-322 row machinery.


def _load_hpa322_fixtures() -> object:
    spec = importlib.util.spec_from_file_location(
        "_hpa322_reference_chart_fixtures",
        Path(__file__).parent / "test_reference_chart_manifest.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_hpa322 = _load_hpa322_fixtures()
_FIXED_TIME = _hpa322._FIXED_TIME
_CHART_BODY = _hpa322._CHART_BODY
_remote = _hpa322._remote
_source_rows_for_inventories = _hpa322._source_rows_for_inventories
_write_source_manifest = _hpa322._write_source_manifest
_install_cached_bodies = _hpa322._install_cached_bodies
_selection_request = _hpa322._selection_request
_published_rows = _hpa322._published_rows
_source_manifest_bytes = _hpa322._source_manifest_bytes

_AUDIO_HASH = "c" * 64
_AUDIO_KEY = "42/bgm.wav"
_EVENTS_CACHE_PATH = f"sha256/cc/{_AUDIO_HASH}"
_TIMING_GOLDEN_PATH = (
    Path(__file__).parent / "schema_goldens/crux.reference-timing-manifest-v1.jsonl"
)


def _ready_resolution(
    *,
    reason_codes: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
) -> TimingRowResolution:
    return TimingRowResolution(
        status="ready",
        reason_codes=reason_codes,  # type: ignore[arg-type]
        warnings=warnings,
        source_audio_key=_AUDIO_KEY,
        source_audio_content_hash=_AUDIO_HASH,
        reference_events_cache_path=_EVENTS_CACHE_PATH,
    )


def _hpa323_quarantine_resolution(
    reason_codes: tuple[str, ...],
    *,
    warnings: tuple[str, ...] = (),
) -> TimingRowResolution:
    return TimingRowResolution(
        status="quarantined",
        reason_codes=reason_codes,  # type: ignore[arg-type]
        warnings=warnings,
        source_audio_key=None,
        source_audio_content_hash=None,
        reference_events_cache_path=None,
    )


@dataclass(frozen=True)
class _Hpa322Fixture:
    outcome: object
    rows: tuple[dict[str, object], ...]
    manifest_path: Path
    manifest_content: bytes


def _published_hpa322(
    tmp_path: Path,
    *,
    inventories: tuple[SimfileInventory, ...] | None = None,
    cache_fixtures: tuple[tuple[object, bytes], ...] = (),
) -> _Hpa322Fixture:
    if inventories is None:
        chart = _remote(42, "real.dtx", _CHART_BODY)
        inventories = (
            SimfileInventory(42, "42/", (chart,), "complete"),
            SimfileInventory(43, "43/", (), "empty"),
        )
        cache_fixtures = ((chart, _CHART_BODY),)
    source_rows = _source_rows_for_inventories(inventories)
    manifest_path, _ = _write_source_manifest(tmp_path, source_rows)
    cache_dir = tmp_path / "cache"
    _install_cached_bodies(cache_dir, cache_fixtures)
    outcome = select_reference_manifest(
        _selection_request(manifest_path, cache_dir, tmp_path / "output")
    )
    assert outcome.manifest is not None  # type: ignore[attr-defined]
    rows = _published_rows(outcome)
    manifest_path = outcome.manifest.path  # type: ignore[attr-defined]
    return _Hpa322Fixture(
        outcome=outcome,
        rows=rows,
        manifest_path=manifest_path,
        manifest_content=manifest_path.read_bytes(),
    )


def _render_hpa322_bytes(rows: tuple[dict[str, object], ...]) -> bytes:
    return _source_manifest_bytes(rows)


# ---------------------------------------------------------------------------
# Step 1: canonical HPA-322 source-loading
# ---------------------------------------------------------------------------


def test_loader_reads_reference_chart_manifest_bytes_once_and_records_exact_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _published_hpa322(tmp_path)
    original_read_bytes = Path.read_bytes
    reads = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal reads
        if path == fixture.manifest_path:
            reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    loaded = load_reference_chart_manifest(fixture.manifest_path)

    assert reads == 1
    assert (
        loaded.source_reference_chart_manifest_sha256
        == sha256(fixture.manifest_content).hexdigest()
    )
    assert loaded.source_reference_chart_version == fixture.rows[0]["corpus_version"]
    assert [v.source_row for v in loaded.rows] == list(fixture.rows)
    assert loaded.rows[0].view.selection_status == "selected"
    assert loaded.rows[1].view.selection_status == "quarantined"


def test_loader_routes_rows_through_reference_chart_row_view_from_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader must validate via reference_chart_row_view_from_row (the merged
    HPA-322 validator), never via manifest_row_view_from_row on the HPA-322 row.
    """
    fixture = _published_hpa322(tmp_path)
    real = reference_timing_manifest.reference_chart_row_view_from_row
    original_manifest_view = reference_chart_manifest.manifest_row_view_from_row
    view_calls: list[object] = []
    manifest_view_schemas: list[object] = []

    def view_spy(row: Mapping[str, object]) -> object:
        view_calls.append(row)
        return real(row)

    def manifest_view_spy(row: Mapping[str, object]) -> object:
        if isinstance(row, Mapping):
            manifest_view_schemas.append(row.get("schema_version"))
        return original_manifest_view(row)

    monkeypatch.setattr(reference_timing_manifest, "reference_chart_row_view_from_row", view_spy)
    # The merged validator calls manifest_row_view_from_row on the reconstructed
    # HPA-321 *payload*; patch it everywhere it is bound so we can prove it is
    # never handed the HPA-322 reference-chart row itself.
    monkeypatch.setattr(corpus_manifest, "manifest_row_view_from_row", manifest_view_spy)
    monkeypatch.setattr(reference_chart_manifest, "manifest_row_view_from_row", manifest_view_spy)

    loaded = load_reference_chart_manifest(fixture.manifest_path)

    assert len(view_calls) == len(loaded.rows)
    # manifest_row_view_from_row only ever saw the HPA-321 schema, never the
    # HPA-322 reference-chart schema — i.e. it was never called on the row.
    assert REFERENCE_CHART_MANIFEST_SCHEMA not in manifest_view_schemas


def test_loader_reproduces_input_corpus_version_after_dropping_it(tmp_path: Path) -> None:
    fixture = _published_hpa322(tmp_path)
    rows = fixture.rows
    normalized = tuple({k: v for k, v in row.items() if k != "corpus_version"} for row in rows)
    rendered = render_manifest(normalized)
    assert rendered.content == fixture.manifest_content
    assert rendered.corpus_version == rows[0]["corpus_version"]


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"{}",
        b"{}\n\n",
        b"[]\n",
        b'{"schema_version":"crux.reference-chart-manifest/v1"}\n',
    ],
    ids=["zero-records", "partial-line", "blank-line", "non-object", "malformed-row"],
)
def test_loader_rejects_noncanonical_or_malformed_records(tmp_path: Path, content: bytes) -> None:
    manifest_path = tmp_path / "hpa322.jsonl"
    manifest_path.write_bytes(content)
    with pytest.raises(ValueError):
        load_reference_chart_manifest(manifest_path)


def test_loader_rejects_noncanonical_record_bytes(tmp_path: Path) -> None:
    fixture = _published_hpa322(tmp_path)
    first = canonical_json_bytes(fixture.rows[0])
    non_canonical = first.replace(b'":', b'" :', 1)
    content = non_canonical + canonical_json_bytes(fixture.rows[1])
    manifest_path = tmp_path / "hpa322.jsonl"
    manifest_path.write_bytes(content)
    with pytest.raises(ValueError):
        load_reference_chart_manifest(manifest_path)


def test_loader_rejects_unsupported_schema(tmp_path: Path) -> None:
    fixture = _published_hpa322(tmp_path)
    rows = list(fixture.rows)
    rows[0] = {**rows[0], "schema_version": "crux.unsupported/v1"}
    manifest_path = tmp_path / "hpa322.jsonl"
    manifest_path.write_bytes(_render_hpa322_bytes(tuple(rows)))
    with pytest.raises(ValueError):
        load_reference_chart_manifest(manifest_path)


def test_loader_rejects_duplicate_simfile_ids(tmp_path: Path) -> None:
    fixture = _published_hpa322(tmp_path)
    rows = list(fixture.rows)
    rows[1] = {**rows[1], "simfile_id": rows[0]["simfile_id"]}
    manifest_path = tmp_path / "hpa322.jsonl"
    manifest_path.write_bytes(_render_hpa322_bytes(tuple(rows)))
    with pytest.raises(ValueError):
        load_reference_chart_manifest(manifest_path)


@pytest.mark.parametrize(
    "field",
    ["source_endpoint_sha256", "source_bucket", "cache_profile"],
)
def test_loader_rejects_mixed_source_identity(tmp_path: Path, field: str) -> None:
    fixture = _published_hpa322(tmp_path)
    rows = list(fixture.rows)
    rows[1] = {**rows[1], field: "different-identity-value"}
    manifest_path = tmp_path / "hpa322.jsonl"
    manifest_path.write_bytes(_render_hpa322_bytes(tuple(rows)))
    with pytest.raises(ValueError):
        load_reference_chart_manifest(manifest_path)


def test_loader_rejects_invalid_selected_null_shape(tmp_path: Path) -> None:
    fixture = _published_hpa322(tmp_path)
    rows = list(fixture.rows)
    # The selected row must carry non-null chart identity; nulling it violates
    # the merged reference-chart validator (selected/null shape contract).
    rows[0] = {**rows[0], "selected_chart_key": None}
    manifest_path = tmp_path / "hpa322.jsonl"
    manifest_path.write_bytes(_render_hpa322_bytes(tuple(rows)))
    with pytest.raises(ValueError):
        load_reference_chart_manifest(manifest_path)


def test_loader_rejects_inconsistent_derived_corpus_version(tmp_path: Path) -> None:
    fixture = _published_hpa322(tmp_path)
    rows = list(fixture.rows)
    rows[0] = {**rows[0], "provenance_notes": "mutated after corpus rendering"}
    manifest_path = tmp_path / "hpa322.jsonl"
    manifest_path.write_bytes(_render_hpa322_bytes(tuple(rows)))
    with pytest.raises(ValueError, match="corpus"):
        load_reference_chart_manifest(manifest_path)


def test_loader_rejects_unreadable_manifest_path(tmp_path: Path) -> None:
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    with pytest.raises(ValueError, match="unavailable"):
        load_reference_chart_manifest(directory)


# ---------------------------------------------------------------------------
# Step 2: lineage preservation and field naming
# ---------------------------------------------------------------------------


def test_build_timing_row_preserves_lineage_and_removes_corpus_version(tmp_path: Path) -> None:
    fixture = _published_hpa322(tmp_path)
    loaded = load_reference_chart_manifest(fixture.manifest_path)

    (ready_row,) = render_manifest(
        (
            build_timing_row(
                loaded.rows[0],
                source_reference_chart_manifest_sha256=loaded.source_reference_chart_manifest_sha256,
                source_reference_chart_version=loaded.source_reference_chart_version,
                timing=_ready_resolution(),
            ),
        )
    ).rows

    assert ready_row["schema_version"] == REFERENCE_TIMING_MANIFEST_SCHEMA
    assert ready_row["timing_semantics_version"] == TIMING_SEMANTICS_VERSION
    # HPA-322 lineage preserved verbatim.
    assert ready_row["source_manifest_sha256"] == fixture.rows[0]["source_manifest_sha256"]
    assert ready_row["source_corpus_version"] == fixture.rows[0]["source_corpus_version"]
    # HPA-322 manifest identity recorded.
    assert (
        ready_row["source_reference_chart_manifest_sha256"]
        == sha256(fixture.manifest_content).hexdigest()
    )
    assert ready_row["source_reference_chart_version"] == fixture.rows[0]["corpus_version"]
    # corpus_version is re-derived by render_manifest, not carried from HPA-322.
    assert ready_row["corpus_version"] != fixture.rows[0]["corpus_version"]


# ---------------------------------------------------------------------------
# Step 3: pure derived-row rendering
# ---------------------------------------------------------------------------


def test_build_timing_row_ready_row_carries_complete_source_audio_identity(tmp_path: Path) -> None:
    fixture = _published_hpa322(tmp_path)
    loaded = load_reference_chart_manifest(fixture.manifest_path)

    row = build_timing_row(
        loaded.rows[0],
        source_reference_chart_manifest_sha256=loaded.source_reference_chart_manifest_sha256,
        source_reference_chart_version=loaded.source_reference_chart_version,
        timing=_ready_resolution(),
    )

    assert row["timing_status"] == "ready"
    assert row["timing_reason_codes"] == []
    assert row["timing_warnings"] == []
    assert row["source_audio_key"] == _AUDIO_KEY
    assert row["source_audio_content_hash"] == _AUDIO_HASH
    assert row["reference_events_cache_path"] == _EVENTS_CACHE_PATH


def test_build_timing_row_upstream_quarantine_uses_upstream_chart_selection_unavailable(
    tmp_path: Path,
) -> None:
    fixture = _published_hpa322(tmp_path)
    loaded = load_reference_chart_manifest(fixture.manifest_path)

    row = build_timing_row(
        loaded.rows[1],
        source_reference_chart_manifest_sha256=loaded.source_reference_chart_manifest_sha256,
        source_reference_chart_version=loaded.source_reference_chart_version,
        timing=upstream_chart_unavailable_resolution(),
    )

    assert row["timing_status"] == "quarantined"
    assert row["timing_reason_codes"] == ["upstream_chart_selection_unavailable"]
    assert row["source_audio_key"] is None
    assert row["source_audio_content_hash"] is None
    assert row["reference_events_cache_path"] is None


def test_build_timing_row_hpa323_quarantine_nulls_source_event_fields_and_preserves_selected_chart(
    tmp_path: Path,
) -> None:
    fixture = _published_hpa322(tmp_path)
    loaded = load_reference_chart_manifest(fixture.manifest_path)

    row = build_timing_row(
        loaded.rows[0],
        source_reference_chart_manifest_sha256=loaded.source_reference_chart_manifest_sha256,
        source_reference_chart_version=loaded.source_reference_chart_version,
        timing=_hpa323_quarantine_resolution(("timing_map_invalid",)),
    )

    # An HPA-322 *selected* row may still be timing *quarantined*; the upstream
    # chart identity is preserved while timing source/event identity is null.
    assert row["timing_status"] == "quarantined"
    assert row["timing_reason_codes"] == ["timing_map_invalid"]
    assert row["source_audio_key"] is None
    assert row["source_audio_content_hash"] is None
    assert row["reference_events_cache_path"] is None
    assert row["selection_status"] == "selected"
    assert row["selected_chart_key"] == fixture.rows[0]["selected_chart_key"]


def test_build_timing_row_passes_through_every_other_hpa322_field_unchanged(
    tmp_path: Path,
) -> None:
    fixture = _published_hpa322(tmp_path)
    loaded = load_reference_chart_manifest(fixture.manifest_path)

    hpa322_row = fixture.rows[0]
    timing_row = build_timing_row(
        loaded.rows[0],
        source_reference_chart_manifest_sha256=loaded.source_reference_chart_manifest_sha256,
        source_reference_chart_version=loaded.source_reference_chart_version,
        timing=_ready_resolution(),
    )

    introduced = {
        "schema_version",
        "source_reference_chart_manifest_sha256",
        "source_reference_chart_version",
        "timing_semantics_version",
        "timing_status",
        "timing_reason_codes",
        "timing_warnings",
        "source_audio_key",
        "source_audio_content_hash",
        "reference_events_cache_path",
    }
    passthrough = {
        key: value
        for key, value in timing_row.items()
        if key not in introduced and key != "corpus_version"
    }
    expected = {
        key: value
        for key, value in hpa322_row.items()
        if key != "corpus_version" and key != "schema_version"
    }
    assert passthrough == expected


def test_build_timing_row_sorts_reason_codes_and_warnings_deterministically(
    tmp_path: Path,
) -> None:
    fixture = _published_hpa322(tmp_path)
    loaded = load_reference_chart_manifest(fixture.manifest_path)

    row = build_timing_row(
        loaded.rows[0],
        source_reference_chart_manifest_sha256=loaded.source_reference_chart_manifest_sha256,
        source_reference_chart_version=loaded.source_reference_chart_version,
        timing=_hpa323_quarantine_resolution(
            ("source_audio_missing", "ambiguous_bgm_start"),
            warnings=("z-warn", "a-warn"),
        ),
    )

    assert row["timing_reason_codes"] == ["ambiguous_bgm_start", "source_audio_missing"]
    assert row["timing_warnings"] == ["a-warn", "z-warn"]


def test_build_timing_row_rejects_unknown_timing_reason_code(tmp_path: Path) -> None:
    fixture = _published_hpa322(tmp_path)
    loaded = load_reference_chart_manifest(fixture.manifest_path)

    with pytest.raises(ValueError):
        build_timing_row(
            loaded.rows[0],
            source_reference_chart_manifest_sha256=loaded.source_reference_chart_manifest_sha256,
            source_reference_chart_version=loaded.source_reference_chart_version,
            timing=_hpa323_quarantine_resolution(("totally_unknown_reason_code",)),
        )


# ---------------------------------------------------------------------------
# Step 4: outcome accounting
# ---------------------------------------------------------------------------


def _published_manifest(path: Path) -> PublishedManifest:
    return PublishedManifest(
        corpus_version="sha256:" + "a" * 64,
        manifest_sha256="b" * 64,
        relative_path=f"manifests/{'b' * 64}.jsonl",
        path=path,
        latest_path=path.parent / "latest.json",
    )


def test_outcome_complete_when_no_quarantines(tmp_path: Path) -> None:
    manifest = _published_manifest(tmp_path / "manifest.jsonl")
    outcome = build_reference_timing_outcome(
        manifest=manifest,
        total_input_rows=3,
        ready_count=3,
        quarantined_count=0,
        upstream_quarantined_count=0,
        events_published=3,
    )
    assert outcome.status == "complete"
    assert outcome.exit_code == 0
    assert outcome.manifest is manifest


def test_outcome_partial_when_any_quarantine_with_published_manifest(tmp_path: Path) -> None:
    manifest = _published_manifest(tmp_path / "manifest.jsonl")
    outcome = build_reference_timing_outcome(
        manifest=manifest,
        total_input_rows=7,
        ready_count=0,
        quarantined_count=7,
        upstream_quarantined_count=5,
        events_published=0,
    )
    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    # quarantined=7 with upstream=5 leaves 2 HPA-323-stage quarantines.
    assert outcome.quarantined_count - outcome.upstream_quarantined_count == 2


def test_failed_outcome_is_exit_two_without_a_manifest() -> None:
    outcome = failed_reference_timing_outcome()
    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert outcome.manifest is None
    assert outcome.events_published == 0


@pytest.mark.parametrize(
    ("ready_count", "quarantined_count", "total_input_rows"),
    [(2, 1, 4), (0, 2, 3)],
    ids=["too-few", "too-many"],
)
def test_outcome_rejects_unbalanced_rows(
    tmp_path: Path,
    ready_count: int,
    quarantined_count: int,
    total_input_rows: int,
) -> None:
    manifest = _published_manifest(tmp_path / "manifest.jsonl")
    with pytest.raises(ValueError, match="balance"):
        build_reference_timing_outcome(
            manifest=manifest,
            total_input_rows=total_input_rows,
            ready_count=ready_count,
            quarantined_count=quarantined_count,
            upstream_quarantined_count=0,
            events_published=ready_count,
        )


def test_outcome_rejects_upstream_exceeding_total_quarantine(tmp_path: Path) -> None:
    manifest = _published_manifest(tmp_path / "manifest.jsonl")
    with pytest.raises(ValueError, match="upstream"):
        build_reference_timing_outcome(
            manifest=manifest,
            total_input_rows=2,
            ready_count=1,
            quarantined_count=1,
            upstream_quarantined_count=2,
            events_published=1,
        )


def test_outcome_rejects_events_published_not_equal_ready(tmp_path: Path) -> None:
    manifest = _published_manifest(tmp_path / "manifest.jsonl")
    with pytest.raises(ValueError, match="events published"):
        build_reference_timing_outcome(
            manifest=manifest,
            total_input_rows=3,
            ready_count=2,
            quarantined_count=1,
            upstream_quarantined_count=1,
            events_published=3,
        )


# ---------------------------------------------------------------------------
# Step 5: schema golden (registry + validator focus)
# ---------------------------------------------------------------------------


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).parents[2]


def test_reference_timing_manifest_golden_is_registered_and_valid(
    repository_root: Path,
) -> None:
    from tests.benchmark.test_schema_goldens import (
        load_schema_golden_manifest,
        validate_schema_golden_entry,
    )

    entry = next(
        item
        for item in load_schema_golden_manifest(repository_root)
        if item.schema == REFERENCE_TIMING_MANIFEST_SCHEMA
    )
    validate_schema_golden_entry(entry, repository_root)


def _golden_rows() -> list[dict[str, object]]:
    return [
        strict_json_loads(line[:-1], require_canonical=True)
        for line in _TIMING_GOLDEN_PATH.read_bytes().splitlines(keepends=True)
    ]  # type: ignore[return-value]


def test_golden_validator_rejects_unknown_reason_string() -> None:
    rows = _golden_rows()
    rows[1]["timing_reason_codes"] = ["totally_unknown_reason_code"]
    content = b"".join(canonical_json_bytes(row, trailing_newline=True) for row in rows)
    with pytest.raises(ValueError):
        validate_schema_golden(REFERENCE_TIMING_MANIFEST_SCHEMA, content)


def test_golden_validator_rejects_non_canonical_json_line() -> None:
    rows = _golden_rows()
    first = canonical_json_bytes(rows[0])
    non_canonical = first.replace(b'":', b'" :', 1)
    content = non_canonical + canonical_json_bytes(rows[1])
    with pytest.raises(ValueError, match="canonical JSONL"):
        validate_schema_golden(REFERENCE_TIMING_MANIFEST_SCHEMA, content)


def test_golden_validator_rejects_extra_trailing_newline() -> None:
    content = _TIMING_GOLDEN_PATH.read_bytes() + b"\n"
    with pytest.raises(ValueError, match="canonical JSONL"):
        validate_schema_golden(REFERENCE_TIMING_MANIFEST_SCHEMA, content)


def test_golden_validator_rejects_duplicate_ready_rows() -> None:
    rows = _golden_rows()
    rows[1] = deepcopy(rows[0])
    content = b"".join(canonical_json_bytes(row, trailing_newline=True) for row in rows)
    with pytest.raises(ValueError, match="one ready and one quarantined"):
        validate_schema_golden(REFERENCE_TIMING_MANIFEST_SCHEMA, content)


def test_golden_validator_rejects_unsupported_schema_name() -> None:
    with pytest.raises(ValueError, match="unsupported schema golden"):
        validate_schema_golden("crux.wrong/v1", _TIMING_GOLDEN_PATH.read_bytes())
