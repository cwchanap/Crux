"""Reviewed-subset manifest contract tests (HPA-327 Task 3)."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from src.benchmark.backend_identity import strict_json_loads
from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.reference_set_manifest import (
    load_reference_set_manifest,
    preflight_reference_mappings,
)
from src.benchmark.reference_timing_manifest import load_reference_timing_manifest
from src.benchmark.reviewed_subset import (
    REVIEW_POLICY_VERSION,
    REVIEWED_REFERENCE_SUBSET_SCHEMA,
    LoadedReviewedSubsetManifest,
    load_reviewed_subset_manifest,
    validate_schema_golden,
)
from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_reference_fixture

_GOLDEN = Path(__file__).parent / "schema_goldens/crux.reviewed-reference-subset-v1.jsonl"
_LEDGER = "a" * 64
_REFERENCE_MANIFEST = "b" * 64
_REFERENCE_VERSION = "sha256:" + "c" * 64
_TIMING_MANIFEST = "d" * 64
_TIMING_VERSION = "sha256:" + "e" * 64


def _golden_rows(count: int) -> tuple[dict[str, object], ...]:
    """Expand the one-row golden into ``count`` canonical rows."""
    source = strict_json_loads(_GOLDEN.read_bytes()[:-1], require_canonical=True)
    assert isinstance(source, dict)
    rows: list[dict[str, object]] = []
    for offset in range(count):
        row = dict(source)
        row.pop("corpus_version", None)
        row["simfile_id"] = 100 + offset
        row["candidate_rank"] = offset + 1
        rows.append(row)
    return tuple(rows)


def _write_subset(
    tmp_path: Path,
    rows: tuple[dict[str, object], ...],
    *,
    name: str = "subset.jsonl",
) -> Path:
    rendered = render_manifest(rows)
    path = tmp_path / name
    path.write_bytes(rendered.content)
    return path


def test_reviewed_subset_schema_golden_is_valid() -> None:
    content = (
        Path(__file__).parent / "schema_goldens" / "crux.reviewed-reference-subset-v1.jsonl"
    ).read_bytes()
    validate_schema_golden(REVIEWED_REFERENCE_SUBSET_SCHEMA, content)


def test_reviewed_subset_loader_accepts_real_population(tmp_path: Path) -> None:
    golden = (
        Path(__file__).parent / "schema_goldens" / "crux.reviewed-reference-subset-v1.jsonl"
    ).read_bytes()
    source = strict_json_loads(golden[:-1], require_canonical=True)
    assert isinstance(source, dict)

    rows: list[dict[str, object]] = []
    for offset in range(20):
        row = dict(source)
        row.pop("corpus_version", None)
        row["simfile_id"] = 100 + offset
        row["candidate_rank"] = offset + 1
        rows.append(row)

    rendered = render_manifest(tuple(rows))
    path = tmp_path / "subset.jsonl"
    path.write_bytes(rendered.content)
    loaded = load_reviewed_subset_manifest(path)

    assert loaded.manifest_sha256 == sha256(rendered.content).hexdigest()
    assert len(loaded.rows) == 20


def test_reviewed_subset_loader_exposes_manifest_identity_contract(
    tmp_path: Path,
) -> None:
    path = _write_subset(tmp_path, _golden_rows(20))

    loaded = load_reviewed_subset_manifest(path)

    assert isinstance(loaded, LoadedReviewedSubsetManifest)
    assert loaded.corpus_version.startswith("sha256:")
    assert loaded.review_policy_version == REVIEW_POLICY_VERSION
    assert loaded.review_ledger_sha256 == _LEDGER
    assert loaded.prior_review_ledger_sha256 is None
    assert loaded.source_reference_manifest_sha256 == _REFERENCE_MANIFEST
    assert loaded.source_reference_manifest_version == _REFERENCE_VERSION
    assert loaded.source_timing_manifest_sha256 == _TIMING_MANIFEST
    assert loaded.source_timing_manifest_version == _TIMING_VERSION
    assert [row.view.simfile_id for row in loaded.rows] == [100 + offset for offset in range(20)]
    assert [row.view.candidate_rank for row in loaded.rows] == list(range(1, 21))
    assert loaded.rows[0].view.density_band == "medium"
    assert loaded.rows[0].view.class_richness_band == "low"
    assert loaded.rows[0].view.reason_codes == ("chart_simplification",)
    assert loaded.rows[0].view.reference_event_span_sec == 10.0


def test_reviewed_subset_loader_rejects_duplicate_simfile_ids(tmp_path: Path) -> None:
    rows = list(_golden_rows(20))
    rows[1]["simfile_id"] = rows[0]["simfile_id"]
    path = _write_subset(tmp_path, tuple(rows))

    with pytest.raises(ValueError, match="duplicate simfile IDs"):
        load_reviewed_subset_manifest(path)


def test_reviewed_subset_loader_rejects_duplicate_candidate_ranks(
    tmp_path: Path,
) -> None:
    rows = list(_golden_rows(20))
    rows[1]["candidate_rank"] = rows[0]["candidate_rank"]
    path = _write_subset(tmp_path, tuple(rows))

    with pytest.raises(ValueError, match="duplicate candidate ranks"):
        load_reviewed_subset_manifest(path)


@pytest.mark.parametrize(
    "field",
    [
        "source_reference_manifest_sha256",
        "source_reference_manifest_version",
        "source_timing_manifest_sha256",
        "source_timing_manifest_version",
    ],
)
def test_reviewed_subset_loader_rejects_mixed_source_identity(
    tmp_path: Path,
    field: str,
) -> None:
    rows = list(_golden_rows(20))
    rows[1][field] = "sha256:" + "9" * 64 if "version" in field else "9" * 64
    path = _write_subset(tmp_path, tuple(rows))

    with pytest.raises(ValueError, match="mixed source identity"):
        load_reviewed_subset_manifest(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("review_ledger_sha256", "9" * 64),
        ("prior_review_ledger_sha256", "9" * 64),
    ],
)
def test_reviewed_subset_loader_rejects_mixed_review_ledger_hash(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    rows = list(_golden_rows(20))
    rows[1][field] = value
    path = _write_subset(tmp_path, tuple(rows))

    with pytest.raises(ValueError, match="review ledger"):
        load_reviewed_subset_manifest(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("density_band", "extreme"),
        ("class_richness_band", "very_high"),
        ("musical_fidelity", "perfect"),
        ("drum_character", "robotic"),
        ("reason_codes", ["not_a_reason"]),
    ],
)
def test_reviewed_subset_loader_rejects_invalid_enums(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    rows = list(_golden_rows(20))
    rows[0][field] = value
    path = _write_subset(tmp_path, tuple(rows))

    with pytest.raises(ValueError, match="invalid"):
        load_reviewed_subset_manifest(path)


def test_reviewed_subset_loader_rejects_fewer_than_min_rows(tmp_path: Path) -> None:
    path = _write_subset(tmp_path, _golden_rows(19))

    with pytest.raises(ValueError, match="20 to 30"):
        load_reviewed_subset_manifest(path)


def test_reviewed_subset_loader_rejects_more_than_max_rows(tmp_path: Path) -> None:
    path = _write_subset(tmp_path, _golden_rows(31))

    with pytest.raises(ValueError, match="20 to 30"):
        load_reviewed_subset_manifest(path)


@pytest.mark.parametrize(
    ("content", "match"),
    [(b"{}\n", "unsupported"), (b"{}\n\n", "canonical")],
    ids=["non-schema-row", "blank-line"],
)
def test_reviewed_subset_loader_rejects_noncanonical_jsonl(
    tmp_path: Path,
    content: bytes,
    match: str,
) -> None:
    path = tmp_path / "noncanonical-subset.jsonl"
    path.write_bytes(content)

    with pytest.raises(ValueError, match=match):
        load_reviewed_subset_manifest(path)


def test_reviewed_subset_loader_rejects_noncanonical_record_bytes(
    tmp_path: Path,
) -> None:
    rendered = render_manifest(_golden_rows(20))
    content = rendered.content.replace(b'":', b'" :', 1)
    path = tmp_path / "noncanonical-subset.jsonl"
    path.write_bytes(content)

    with pytest.raises(ValueError, match="canonical"):
        load_reviewed_subset_manifest(path)


def test_reference_fixture_builds_eligible_population_and_supports_reverse(
    tmp_path: Path,
) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    reference = load_reference_set_manifest(fixture.reference_manifest_path)
    timing = load_reference_timing_manifest(fixture.timing_manifest_path)
    expected_ids = tuple(100 + offset for offset in range(36))
    assert tuple(row.view.simfile_id for row in reference.rows) == expected_ids
    assert tuple(row.view.simfile_id for row in timing.rows) == expected_ids
    assert all(row.view.eligibility_status == "eligible" for row in reference.rows)
    mappings = preflight_reference_mappings(
        reference,
        timing,
        timing_output_root=fixture.timing_output_root,
    )
    assert set(mappings) == set(expected_ids)
    assert all(mappings[simfile_id] is not None for simfile_id in expected_ids)

    reversed_fixture = build_reviewed_subset_reference_fixture(
        tmp_path / "reversed",
        eligible_count=36,
        reverse_rows=True,
    )
    reversed_reference = load_reference_set_manifest(reversed_fixture.reference_manifest_path)
    assert tuple(row.view.simfile_id for row in reversed_reference.rows) == tuple(
        reversed(expected_ids)
    )
