"""Reviewed-subset manifest contract and candidate preparation tests (HPA-327)."""

from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.reference_set_manifest import (
    load_reference_set_manifest,
    preflight_reference_mappings,
)
from src.benchmark.reference_timing_manifest import load_reference_timing_manifest
from src.benchmark.reviewed_subset import (
    REVIEW_CSV_FIELDS,
    REVIEW_MANUAL_FIELDS,
    REVIEW_POLICY_VERSION,
    REVIEW_SELECTION_SEED,
    REVIEW_TARGET_COUNT,
    REVIEWED_REFERENCE_SUBSET_SCHEMA,
    FinalizeReviewedSubsetOutcome,
    FinalizeReviewedSubsetRequest,
    LoadedReviewedSubsetManifest,
    PrepareReviewedSubsetRequest,
    ReviewCandidate,
    ScoreReviewedSubsetOutcome,
    ScoreReviewedSubsetRequest,
    build_candidate_stream,
    canonical_stratum_key,
    finalize_reviewed_subset,
    load_reviewed_subset_manifest,
    prepare_reviewed_subset,
    score_oaf_reviewed_subset,
    validate_schema_golden,
)
from tests.benchmark.reviewed_subset_fixtures import (
    ReviewedSubsetOafFixture,
    ReviewedSubsetRowSpec,
    build_reviewed_subset_oaf_fixture,
    build_reviewed_subset_reference_fixture,
)

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


def test_reviewed_subset_module_has_no_import_time_oaf_dependency() -> None:
    """Pin the no-import-time OaF execution dependency constraint.

    Must run in a subprocess: in-process ``sys.modules`` checks are unreliable
    because other tests import ``src.benchmark.oaf_corpus_run`` first.
    """
    probe = (
        "import sys, src.benchmark.reviewed_subset; "
        "sys.exit(1 if 'src.benchmark.oaf_corpus_run' in sys.modules else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8")


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
        ("reference_event_span_sec", "10.5000000"),
        ("reference_event_span_sec", 10),
        ("common_event_density_per_sec", "1.2000000"),
        ("common_event_density_per_sec", 1.2),
    ],
    ids=[
        "density-band",
        "richness-band",
        "fidelity",
        "character",
        "reason",
        "span-trailing-fraction-digits",
        "span-int",
        "density-trailing-fraction-digits",
        "density-float",
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


def test_reviewed_subset_loader_accepts_fractional_metric_tokens(tmp_path: Path) -> None:
    rows = list(_golden_rows(20))
    rows[0]["reference_event_span_sec"] = "10.5"
    rows[0]["common_event_density_per_sec"] = "1.25"
    path = _write_subset(tmp_path, tuple(rows))

    loaded = load_reviewed_subset_manifest(path)

    assert loaded.rows[0].view.reference_event_span_sec == Decimal("10.5")
    assert loaded.rows[0].view.common_event_density_per_sec == Decimal("1.25")


def test_reviewed_subset_loader_requires_notes_for_other_reason(tmp_path: Path) -> None:
    rows = list(_golden_rows(20))
    rows[0]["reason_codes"] = ["other"]
    path = _write_subset(tmp_path, tuple(rows))

    with pytest.raises(ValueError, match="requires notes"):
        load_reviewed_subset_manifest(path)


def test_reviewed_subset_loader_accepts_other_reason_with_notes(tmp_path: Path) -> None:
    rows = list(_golden_rows(20))
    rows[0]["reason_codes"] = ["other"]
    rows[0]["notes"] = "hand-verified against the source pack"
    path = _write_subset(tmp_path, tuple(rows))

    loaded = load_reviewed_subset_manifest(path)

    assert loaded.rows[0].view.reason_codes == ("other",)


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


def _candidate_stream(
    fixture: object,
    *,
    reference_manifest_path: Path | None = None,
) -> tuple[ReviewCandidate, ...]:
    reference = load_reference_set_manifest(
        reference_manifest_path or fixture.reference_manifest_path  # type: ignore[attr-defined]
    )
    timing = load_reference_timing_manifest(fixture.timing_manifest_path)  # type: ignore[attr-defined]
    mappings = preflight_reference_mappings(
        reference,
        timing,
        timing_output_root=fixture.timing_output_root,  # type: ignore[attr-defined]
    )
    return build_candidate_stream(reference, timing, mappings=mappings)


def _load_manifest_rows(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line in path.read_bytes().splitlines(keepends=True):
        value = strict_json_loads(line[:-1], require_canonical=True)
        assert isinstance(value, dict)
        rows.append({key: item for key, item in value.items() if key != "corpus_version"})
    return tuple(rows)


def _fill_manual_fields(rows: list[dict[str, str]], *, include_ids: set[int]) -> None:
    for row in rows:
        if int(row["simfile_id"]) in include_ids:
            row.update(
                {
                    "reviewer": "auditor-1",
                    "reviewed_at": "2026-08-15T00:00:00Z",
                    "chart_selection_confirmed": "true",
                    "audio_revision_confirmed": "true",
                    "bgm_alignment_confirmed": "true",
                    "technical_mapping_confirmed": "true",
                    "musical_fidelity": "close",
                    "drum_character": "acoustic",
                    "known_limitations": "",
                    "decision": "include",
                    "reason_codes": "",
                    "notes": "",
                }
            )
        else:
            row.update(
                {
                    "reviewer": "auditor-1",
                    "reviewed_at": "2026-08-15T00:00:00Z",
                    "chart_selection_confirmed": "false",
                    "audio_revision_confirmed": "true",
                    "bgm_alignment_confirmed": "true",
                    "technical_mapping_confirmed": "true",
                    "musical_fidelity": "not_representative",
                    "drum_character": "unknown",
                    "known_limitations": "",
                    "decision": "exclude",
                    "reason_codes": "chart_selection_mismatch",
                    "notes": "",
                }
            )


def _write_prior_ledger(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    name: str = "prior.csv",
) -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _prepare_request(
    fixture: object,
    output: Path,
    *,
    prior_ledger_path: Path | None = None,
    reference_manifest_path: Path | None = None,
) -> PrepareReviewedSubsetRequest:
    return PrepareReviewedSubsetRequest(
        reference_manifest_path=reference_manifest_path or fixture.reference_manifest_path,  # type: ignore[attr-defined]
        timing_manifest_path=fixture.timing_manifest_path,  # type: ignore[attr-defined]
        output_file=output,
        prior_ledger_path=prior_ledger_path,
    )


def _stratum_row_specs() -> tuple[ReviewedSubsetRowSpec, ...]:
    """36 recipes whose (density, richness, warning, chart) strata are distinct.

    Row ``offset`` targets richness band ``offset % 3`` (1/2/3 common classes)
    and density band ``(offset // 3) % 3`` (0.5/1.0/2.0 events per second); the
    second half carries timing warnings and odd rows select ``mas.dtx``.
    """
    events_by_band: dict[tuple[int, int], tuple[tuple[str, float], ...]] = {
        (1, 0): (("13", 0.0), ("13", 4.0)),
        (1, 1): (("13", 0.5),),
        (1, 2): (("13", 0.0), ("13", 1.0)),
        (2, 0): (("13", 0.0), ("12", 4.0)),
        (2, 1): (("13", 0.0), ("12", 2.0)),
        (2, 2): (("13", 0.0), ("12", 1.0)),
        (3, 0): (("13", 0.0), ("12", 3.0), ("16", 6.0)),
        (3, 1): (("13", 0.0), ("12", 1.5), ("16", 3.0)),
        (3, 2): (("13", 0.0), ("12", 0.75), ("16", 1.5)),
    }
    specs: list[ReviewedSubsetRowSpec] = []
    for offset in range(36):
        specs.append(
            ReviewedSubsetRowSpec(
                timing_warnings=("sync_shift",) if offset >= 18 else (),
                chart_basename="mas.dtx" if offset % 2 else "real.dtx",
                events=events_by_band[(1 + offset % 3, (offset // 3) % 3)],
            )
        )
    return tuple(specs)


def test_prepare_selects_exactly_30_without_model_inputs(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    output = tmp_path / "review.csv"
    request = PrepareReviewedSubsetRequest(
        reference_manifest_path=fixture.reference_manifest_path,
        timing_manifest_path=fixture.timing_manifest_path,
        output_file=output,
    )

    assert set(request.__dataclass_fields__) == {  # pylint: disable=no-member
        "reference_manifest_path",
        "timing_manifest_path",
        "output_file",
        "prior_ledger_path",
    }

    outcome = prepare_reviewed_subset(request)
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert outcome.exit_code == 0
    assert len(rows) == 30
    assert [int(row["candidate_rank"]) for row in rows] == list(range(1, 31))


def test_prepare_selects_all_eligible_between_20_and_29(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=24)
    output = tmp_path / "review.csv"

    outcome = prepare_reviewed_subset(_prepare_request(fixture, output))
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))

    assert outcome.exit_code == 0
    assert outcome.output_file == output
    assert outcome.candidate_count == 24
    assert len(rows) == 24
    assert [int(row["candidate_rank"]) for row in rows] == list(range(1, 25))


def test_prepare_exits_2_when_population_below_minimum(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=18)
    output = tmp_path / "review.csv"

    outcome = prepare_reviewed_subset(_prepare_request(fixture, output))

    assert outcome.exit_code == 2
    assert outcome.output_file is None
    assert outcome.candidate_count == 0


def test_candidate_stream_matches_published_accounting_and_cache_paths(
    tmp_path: Path,
) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    reference = load_reference_set_manifest(fixture.reference_manifest_path)
    timing = load_reference_timing_manifest(fixture.timing_manifest_path)
    mappings = preflight_reference_mappings(
        reference,
        timing,
        timing_output_root=fixture.timing_output_root,
    )

    stream = build_candidate_stream(reference, timing, mappings=mappings)

    assert len(stream) == 36
    by_id = {candidate.simfile_id: candidate for candidate in stream}
    for loaded in reference.rows:
        candidate = by_id[loaded.view.simfile_id]
        assert candidate.common_event_count == loaded.view.common_scored_event_count
        assert candidate.density_band in {"low", "medium", "high"}
        assert candidate.class_richness_band in {"low", "medium", "high"}
        assert candidate.source_audio_cache_path == (
            f"sha256/{candidate.source_audio_content_hash[:2]}/{candidate.source_audio_content_hash}"
        )
        assert candidate.reference_event_span_sec == 0.0
        assert candidate.common_event_density_per_sec == 1.0
        assert candidate.common_class_count == 1
        assert candidate.has_timing_warning is False
        assert candidate.selects_real_or_full_chart is True


def test_candidate_timing_warning_and_chart_basename_features(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(
        tmp_path,
        eligible_count=5,
        row_specs=(
            ReviewedSubsetRowSpec(),
            ReviewedSubsetRowSpec(timing_warnings=("sync_shift",)),
            ReviewedSubsetRowSpec(chart_basename="full.dtx"),
            ReviewedSubsetRowSpec(chart_basename="mas.dtx"),
            ReviewedSubsetRowSpec(chart_basename="REAL.DTX"),
        ),
    )

    stream = _candidate_stream(fixture)
    by_id = {candidate.simfile_id: candidate for candidate in stream}

    assert by_id[100].has_timing_warning is False
    assert by_id[101].has_timing_warning is True
    assert by_id[102].has_timing_warning is False
    assert by_id[100].selects_real_or_full_chart is True
    assert by_id[102].selects_real_or_full_chart is True
    assert by_id[103].selects_real_or_full_chart is False
    assert by_id[104].selects_real_or_full_chart is True


def test_prepare_rejects_common_event_count_mismatch(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=24)
    source_rows = list(_load_manifest_rows(fixture.reference_manifest_path))
    for row in source_rows:
        row["mapped_event_count"] = 2
        row["common_scored_event_count"] = 2
    mutated_reference = tmp_path / "inflated-reference.jsonl"
    mutated_reference.write_bytes(render_manifest(tuple(source_rows)).content)

    outcome = prepare_reviewed_subset(
        _prepare_request(
            fixture, tmp_path / "review.csv", reference_manifest_path=mutated_reference
        )
    )

    assert outcome.exit_code == 2
    assert outcome.output_file is None


def test_prepare_writes_exact_csv_boundary(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=24)
    reference = load_reference_set_manifest(fixture.reference_manifest_path)
    timing = load_reference_timing_manifest(fixture.timing_manifest_path)
    output = tmp_path / "review.csv"

    outcome = prepare_reviewed_subset(_prepare_request(fixture, output))
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))

    assert outcome.exit_code == 0
    assert len(rows) == 24
    assert list(rows[0]) == list(REVIEW_CSV_FIELDS)
    assert len(REVIEW_CSV_FIELDS) == 36
    row = rows[0]
    assert row["review_policy_version"] == REVIEW_POLICY_VERSION
    assert row["selection_seed"] == REVIEW_SELECTION_SEED
    assert row["prior_review_ledger_sha256"] == ""
    assert row["source_reference_manifest_sha256"] == reference.manifest_sha256
    assert row["source_reference_manifest_version"] == reference.corpus_version
    assert row["source_timing_manifest_sha256"] == timing.manifest_sha256
    assert row["source_timing_manifest_version"] == timing.corpus_version
    assert row["common_event_count"] == "1"
    assert row["reference_event_span_sec"] == "0"
    assert row["common_event_density_per_sec"] == "1"
    assert row["common_class_count"] == "1"
    assert row["density_band"] in {"low", "medium", "high"}
    assert row["class_richness_band"] in {"low", "medium", "high"}
    assert row["has_timing_warning"] == "false"
    assert row["selects_real_or_full_chart"] == "true"
    assert row["source_audio_cache_path"] == (
        f"sha256/{row['source_audio_content_hash'][:2]}/{row['source_audio_content_hash']}"
    )
    assert all(row[field] == "" for field in REVIEW_MANUAL_FIELDS)
    assert output.read_bytes().endswith(b"\n")
    assert b"\r" not in output.read_bytes()


def test_seeded_stratum_order_with_more_than_30_strata(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(
        tmp_path,
        eligible_count=36,
        row_specs=_stratum_row_specs(),
    )

    stream = _candidate_stream(fixture)

    assert len(stream) == 36
    strata = {canonical_stratum_key(candidate) for candidate in stream}
    assert len(strata) == 36
    nonempty_strata: list[ReviewCandidate] = []
    seen: set[str] = set()
    for candidate in stream:
        key = canonical_stratum_key(candidate)
        if key not in seen:
            seen.add(key)
            nonempty_strata.append(candidate)
    expected_order = sorted(
        nonempty_strata,
        key=lambda key: sha256(
            f"{REVIEW_SELECTION_SEED}:{canonical_stratum_key(key)}".encode()
        ).hexdigest(),
    )
    selected_first_round_strata = tuple(stream[: len(nonempty_strata)])
    assert selected_first_round_strata == tuple(expected_order[: len(selected_first_round_strata)])
    assert len({(c.density_band, c.class_richness_band) for c in stream[:30]}) == 9


def test_seeded_stratum_selection_is_input_order_independent(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    reference = load_reference_set_manifest(fixture.reference_manifest_path)
    timing = load_reference_timing_manifest(fixture.timing_manifest_path)
    mappings = preflight_reference_mappings(
        reference,
        timing,
        timing_output_root=fixture.timing_output_root,
    )

    forward = build_candidate_stream(reference, timing, mappings=mappings)
    reversed_reference = replace(reference, rows=tuple(reversed(reference.rows)))
    reversed_timing = replace(timing, rows=tuple(reversed(timing.rows)))
    backward = build_candidate_stream(reversed_reference, reversed_timing, mappings=mappings)

    assert [candidate.simfile_id for candidate in forward] == [
        candidate.simfile_id for candidate in backward
    ]


def test_prepare_continuation_carries_valid_includes_and_replaces_excludes(
    tmp_path: Path,
) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    output = tmp_path / "review.csv"
    assert prepare_reviewed_subset(_prepare_request(fixture, output)).exit_code == 0
    initial_rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    carried_rows = initial_rows[:24]
    excluded_rows = initial_rows[24:]
    carried_ids = {int(row["simfile_id"]) for row in carried_rows}
    excluded_ids = {int(row["simfile_id"]) for row in excluded_rows}
    _fill_manual_fields(initial_rows, include_ids=carried_ids)
    prior_path = _write_prior_ledger(tmp_path, initial_rows)

    continued_output = tmp_path / "continued.csv"
    continued = prepare_reviewed_subset(
        _prepare_request(fixture, continued_output, prior_ledger_path=prior_path)
    )
    current = list(csv.DictReader(continued_output.open(encoding="utf-8", newline="")))

    assert continued.exit_code == 0
    assert continued.output_file == continued_output
    assert continued.candidate_count == REVIEW_TARGET_COUNT
    assert continued.carried_include_count == 24
    assert continued.replacement_count == 6
    current_ids = {int(row["simfile_id"]) for row in current}
    assert not excluded_ids & current_ids
    assert carried_ids <= current_ids
    assert [int(row["candidate_rank"]) for row in current] == list(range(1, 31))
    assert [int(row["simfile_id"]) for row in current[:24]] == [
        int(row["simfile_id"]) for row in carried_rows
    ]
    stream = _candidate_stream(fixture)
    assert [int(row["simfile_id"]) for row in current[24:]] == [
        candidate.simfile_id for candidate in stream[30:36]
    ]
    carried_cell = next(
        row for row in current if row["simfile_id"] == carried_rows[0]["simfile_id"]
    )
    assert carried_cell["reviewer"] == "auditor-1"
    assert carried_cell["decision"] == "include"
    assert carried_cell["reason_codes"] == ""
    prior_hash = sha256(prior_path.read_bytes()).hexdigest()
    assert all(row["prior_review_ledger_sha256"] == prior_hash for row in current)


def test_prepare_continuation_drops_review_when_source_hash_changes(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    output = tmp_path / "review.csv"
    assert prepare_reviewed_subset(_prepare_request(fixture, output)).exit_code == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    _fill_manual_fields(rows, include_ids={int(row["simfile_id"]) for row in rows})
    prior_path = _write_prior_ledger(tmp_path, rows)

    source_rows = list(_load_manifest_rows(fixture.reference_manifest_path))
    target = next(row for row in source_rows if row["simfile_id"] == 100)
    target["timing_warnings"] = ["sync_shift"]
    mutated_reference = tmp_path / "mutated-reference.jsonl"
    mutated_reference.write_bytes(render_manifest(tuple(source_rows)).content)

    continued = prepare_reviewed_subset(
        _prepare_request(
            fixture,
            tmp_path / "continued.csv",
            prior_ledger_path=prior_path,
            reference_manifest_path=mutated_reference,
        )
    )
    current = list(csv.DictReader((tmp_path / "continued.csv").open(encoding="utf-8", newline="")))

    assert continued.exit_code == 0
    assert continued.carried_include_count == 29
    assert continued.replacement_count == 1
    carried_ids = {int(row["simfile_id"]) for row in rows} - {100}
    stream = _candidate_stream(fixture, reference_manifest_path=mutated_reference)
    expected_replacement = [
        candidate.simfile_id for candidate in stream if candidate.simfile_id not in carried_ids
    ][:1]
    assert [int(row["simfile_id"]) for row in current[29:]] == expected_replacement
    changed_cell = next((row for row in current if row["simfile_id"] == "100"), None)
    if changed_cell is not None:
        assert changed_cell["reviewer"] == ""
        assert changed_cell["decision"] == ""


def test_prepare_rejects_malformed_completed_prior_manual_fields(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    output = tmp_path / "review.csv"
    assert prepare_reviewed_subset(_prepare_request(fixture, output)).exit_code == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    _fill_manual_fields(rows, include_ids={int(row["simfile_id"]) for row in rows})
    rows[0]["reviewer"] = ""
    prior_path = _write_prior_ledger(tmp_path, rows)

    outcome = prepare_reviewed_subset(
        _prepare_request(fixture, tmp_path / "continued.csv", prior_ledger_path=prior_path)
    )

    assert outcome.exit_code == 2
    assert outcome.output_file is None


def test_prepare_rejects_truncated_prior_row(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    output = tmp_path / "review.csv"
    assert prepare_reviewed_subset(_prepare_request(fixture, output)).exit_code == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    _fill_manual_fields(rows, include_ids={int(row["simfile_id"]) for row in rows})
    prior_path = tmp_path / "prior.csv"
    with prior_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(rows[0])
        handle.write("v1,crux-hpa327-v1,sha256,1,123,abc\n")  # trailing cells cut off

    outcome = prepare_reviewed_subset(
        _prepare_request(fixture, tmp_path / "continued.csv", prior_ledger_path=prior_path)
    )

    assert outcome.exit_code == 2
    assert outcome.output_file is None


def test_prepare_rejects_prior_row_missing_simfile_id_cell(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    output = tmp_path / "review.csv"
    assert prepare_reviewed_subset(_prepare_request(fixture, output)).exit_code == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    _fill_manual_fields(rows, include_ids={int(row["simfile_id"]) for row in rows})
    prior_path = tmp_path / "prior.csv"
    with prior_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(rows[0])
        handle.write("v1,crux-hpa327-v1,sha256,1\n")  # cut before the simfile_id column

    outcome = prepare_reviewed_subset(
        _prepare_request(fixture, tmp_path / "continued.csv", prior_ledger_path=prior_path)
    )

    assert outcome.exit_code == 2
    assert outcome.output_file is None


def test_prepare_rejects_leading_zero_prior_simfile_id(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    output = tmp_path / "review.csv"
    assert prepare_reviewed_subset(_prepare_request(fixture, output)).exit_code == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    _fill_manual_fields(rows, include_ids={int(row["simfile_id"]) for row in rows})
    rows[0]["simfile_id"] = "0" + rows[0]["simfile_id"]
    prior_path = _write_prior_ledger(tmp_path, rows)

    outcome = prepare_reviewed_subset(
        _prepare_request(fixture, tmp_path / "continued.csv", prior_ledger_path=prior_path)
    )

    assert outcome.exit_code == 2
    assert outcome.output_file is None


def test_prepare_rejects_duplicate_prior_simfile_ids(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    output = tmp_path / "review.csv"
    assert prepare_reviewed_subset(_prepare_request(fixture, output)).exit_code == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    _fill_manual_fields(rows, include_ids={int(row["simfile_id"]) for row in rows})
    rows[1]["simfile_id"] = rows[0]["simfile_id"]
    prior_path = _write_prior_ledger(tmp_path, rows)

    outcome = prepare_reviewed_subset(
        _prepare_request(fixture, tmp_path / "continued.csv", prior_ledger_path=prior_path)
    )

    assert outcome.exit_code == 2
    assert outcome.output_file is None


def _finalize_request(
    fixture: object,
    review_path: Path,
    output_dir: Path,
    *,
    prior_ledger_path: Path | None = None,
) -> FinalizeReviewedSubsetRequest:
    return FinalizeReviewedSubsetRequest(
        reference_manifest_path=fixture.reference_manifest_path,  # type: ignore[attr-defined]
        timing_manifest_path=fixture.timing_manifest_path,  # type: ignore[attr-defined]
        review_file=review_path,
        output_dir=output_dir,
        prior_ledger_path=prior_ledger_path,
    )


def _completed_review_file(
    tmp_path: Path,
    fixture: object,
    *,
    include_count: int = 20,
    mutate: object = None,
    prior_ledger_path: Path | None = None,
) -> Path:
    prepared = tmp_path / "prepared.csv"
    outcome = prepare_reviewed_subset(
        _prepare_request(fixture, prepared, prior_ledger_path=prior_ledger_path)
    )
    assert outcome.exit_code == 0
    rows = list(csv.DictReader(prepared.open(encoding="utf-8", newline="")))
    include_ids = {int(row["simfile_id"]) for row in rows[:include_count]}
    _fill_manual_fields(rows, include_ids=include_ids)
    if mutate is not None:
        mutate(rows)
    return _write_prior_ledger(tmp_path, rows, name="review.csv")


def test_finalize_request_and_outcome_shapes() -> None:
    assert set(FinalizeReviewedSubsetRequest.__dataclass_fields__) == {  # pylint: disable=no-member
        "reference_manifest_path",
        "timing_manifest_path",
        "review_file",
        "output_dir",
        "prior_ledger_path",
    }
    assert set(FinalizeReviewedSubsetOutcome.__dataclass_fields__) == {  # pylint: disable=no-member
        "exit_code",
        "manifest",
        "review_ledger_path",
        "included_count",
        "excluded_count",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[0].update(reviewer=""),
        lambda rows: rows[0].update(reviewed_at="2026-08-15 00:00:00"),
        lambda rows: rows[0].update(reviewed_at="2026-08-15T00:00:00+01:00"),
        lambda rows: rows[0].update(chart_selection_confirmed="True"),
        lambda rows: rows[0].update(chart_selection_confirmed="false"),
        lambda rows: rows[0].update(musical_fidelity="not_representative"),
        lambda rows: rows[0].update(decision="exclude", reason_codes=""),
        lambda rows: rows[0].update(reason_codes="other", notes=""),
        lambda rows: rows[0].update(reason_codes="not_a_reason"),
    ],
    ids=[
        "blank-reviewer",
        "non-rfc3339-timestamp",
        "non-utc-offset",
        "noncanonical-confirmation",
        "include-false-confirmation",
        "include-not-representative",
        "exclude-no-reasons",
        "other-no-notes",
        "unknown-reason",
    ],
)
def test_finalize_rejects_closed_review_violations(
    tmp_path: Path,
    mutate: object,
) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    review_path = _completed_review_file(tmp_path, fixture, mutate=mutate)

    outcome = finalize_reviewed_subset(_finalize_request(fixture, review_path, tmp_path / "out"))

    assert outcome.exit_code == 2
    assert outcome.manifest is None
    assert outcome.review_ledger_path is None


def test_finalize_publishes_valid_20_includes(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    review_path = _completed_review_file(tmp_path, fixture)

    outcome = finalize_reviewed_subset(_finalize_request(fixture, review_path, tmp_path / "out"))

    assert outcome.exit_code == 0
    assert outcome.manifest is not None
    assert outcome.review_ledger_path is not None
    assert outcome.included_count == 20
    assert outcome.excluded_count == 10


def test_finalize_accepts_utc_zero_offset_timestamps_and_canonicalizes(
    tmp_path: Path,
) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)

    def use_utc_offset(rows: list[dict[str, str]]) -> None:
        rows[0]["reviewed_at"] = "2026-08-15T00:00:00+00:00"

    review_path = _completed_review_file(tmp_path, fixture, mutate=use_utc_offset)

    outcome = finalize_reviewed_subset(_finalize_request(fixture, review_path, tmp_path / "out"))

    assert outcome.exit_code == 0
    ledger_rows = list(
        csv.DictReader(outcome.review_ledger_path.open(encoding="utf-8", newline=""))  # type: ignore[union-attr]
    )
    assert ledger_rows[0]["reviewed_at"] == "2026-08-15T00:00:00Z"


def test_finalize_rejects_fewer_than_20_includes(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    review_path = _completed_review_file(tmp_path, fixture, include_count=19)

    outcome = finalize_reviewed_subset(_finalize_request(fixture, review_path, tmp_path / "out"))

    assert outcome.exit_code == 2
    assert outcome.manifest is None
    assert outcome.review_ledger_path is None


def test_finalize_ignores_rewritten_generated_cells(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)

    def rewrite_generated(rows: list[dict[str, str]]) -> None:
        for row in rows:
            row["candidate_rank"] = "999"
            row["density_band"] = "nonsense"
            row["class_richness_band"] = "nonsense"
            row["source_row_sha256"] = "0" * 64
            row["selected_chart_content_hash"] = "f" * 64
            row["has_timing_warning"] = "maybe"
            row["selects_real_or_full_chart"] = "yes"
            row["reference_event_span_sec"] = "-12.5"
            row["common_event_density_per_sec"] = "NaN"
            row["common_event_count"] = "-3"
            row["common_class_count"] = "0x10"
            row["selected_chart_key"] = "42/other.dtx"
            row["source_audio_cache_path"] = "C:\\Users\\reviewer\\Desktop\\audio.wav"
            row["prior_review_ledger_sha256"] = "z" * 64

    review_path = _completed_review_file(tmp_path, fixture, mutate=rewrite_generated)

    outcome = finalize_reviewed_subset(_finalize_request(fixture, review_path, tmp_path / "out"))

    assert outcome.exit_code == 0
    assert outcome.included_count == 20


def test_finalize_rejects_stale_simfile_membership(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    stale_id = _candidate_stream(fixture)[30].simfile_id

    def swap_to_stale(rows: list[dict[str, str]]) -> None:
        rows[0]["simfile_id"] = str(stale_id)

    review_path = _completed_review_file(tmp_path, fixture, mutate=swap_to_stale)

    outcome = finalize_reviewed_subset(_finalize_request(fixture, review_path, tmp_path / "out"))

    assert outcome.exit_code == 2
    assert outcome.manifest is None


def test_finalize_continuation_requires_the_same_prior_ledger(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    initial = tmp_path / "initial.csv"
    assert prepare_reviewed_subset(_prepare_request(fixture, initial)).exit_code == 0
    initial_rows = list(csv.DictReader(initial.open(encoding="utf-8", newline="")))
    _fill_manual_fields(
        initial_rows,
        include_ids={int(row["simfile_id"]) for row in initial_rows[:20]},
    )
    prior_path = _write_prior_ledger(tmp_path, initial_rows)

    review_path = _completed_review_file(
        tmp_path,
        fixture,
        prior_ledger_path=prior_path,
    )
    request = _finalize_request(
        fixture,
        review_path,
        tmp_path / "out",
        prior_ledger_path=prior_path,
    )

    without_prior = finalize_reviewed_subset(replace(request, prior_ledger_path=None))
    assert without_prior.exit_code == 2

    with_prior = finalize_reviewed_subset(request)
    assert with_prior.exit_code == 0
    assert with_prior.included_count == 20
    assert with_prior.excluded_count == 6


def _failed_simfile_ids(fixture: ReviewedSubsetOafFixture) -> set[int]:
    from src.benchmark.oaf_corpus_run import parse_oaf_corpus_run

    snapshot = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    return {
        int(row["simfile_id"])
        for row in snapshot["items"]
        if row["execution_disposition"] == "failed"
    }


def _prepare_slate_rows(fixture: ReviewedSubsetOafFixture) -> list[dict[str, str]]:
    prepared = Path(fixture.oaf_output_dir).parent / "prepared.csv"
    outcome = prepare_reviewed_subset(
        PrepareReviewedSubsetRequest(
            reference_manifest_path=fixture.reference_manifest_path,
            timing_manifest_path=fixture.timing_manifest_path,
            output_file=prepared,
        )
    )
    assert outcome.exit_code == 0
    return list(csv.DictReader(prepared.open(encoding="utf-8", newline="")))


def _all_success_includes(fixture: ReviewedSubsetOafFixture) -> set[int]:
    """The first 20 slate rows, skipping persisted non-success rows."""
    failed = _failed_simfile_ids(fixture)
    include_ids: list[int] = []
    for row in _prepare_slate_rows(fixture):
        simfile_id = int(row["simfile_id"])
        if simfile_id in failed:
            continue
        include_ids.append(simfile_id)
        if len(include_ids) == 20:
            break
    return set(include_ids)


def _finalize_subset(
    tmp_path: Path,
    fixture: ReviewedSubsetOafFixture,
    *,
    include_ids: set[int],
) -> Path:
    """Prepare, review, and finalize a subset whose includes are exactly ``include_ids``."""
    rows = _prepare_slate_rows(fixture)
    _fill_manual_fields(rows, include_ids=include_ids)
    review = tmp_path / "review.csv"
    _write_prior_ledger(tmp_path, rows, name="review.csv")
    outcome = finalize_reviewed_subset(
        FinalizeReviewedSubsetRequest(
            reference_manifest_path=fixture.reference_manifest_path,
            timing_manifest_path=fixture.timing_manifest_path,
            review_file=review,
            output_dir=tmp_path / "subset",
        )
    )
    assert outcome.exit_code == 0
    assert outcome.manifest is not None
    return outcome.manifest.path


def _score_request(
    fixture: ReviewedSubsetOafFixture,
    subset_path: Path,
    output_dir: Path,
) -> ScoreReviewedSubsetRequest:
    return ScoreReviewedSubsetRequest(
        run_path=fixture.run_path,
        reference_manifest_path=fixture.reference_manifest_path,
        timing_manifest_path=fixture.timing_manifest_path,
        subset_manifest_path=subset_path,
        output_dir=output_dir,
    )


def _parent_reports(fixture: ReviewedSubsetOafFixture, reports_dir: Path) -> None:
    """Reconstruct the parent-run reports exactly as the broad finalizer would."""
    from src.benchmark.cohort_scoring import score_cohort
    from src.benchmark.oaf_corpus_run import build_oaf_cohort_from_snapshot, parse_oaf_corpus_run
    from src.benchmark.reports import write_cohort_reports

    snapshot = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    reference = load_reference_set_manifest(fixture.reference_manifest_path)
    timing = load_reference_timing_manifest(fixture.timing_manifest_path)
    mappings = preflight_reference_mappings(
        reference,
        timing,
        timing_output_root=fixture.timing_output_root,
    )
    identity, items = build_oaf_cohort_from_snapshot(
        snapshot,
        mappings=mappings,
        output_dir=fixture.oaf_output_dir,
    )
    write_cohort_reports(score_cohort(identity, items, diagnostics_for=()), reports_dir)


_REPORT_NAMES = (
    "summary.json",
    "items.csv",
    "per_song.csv",
    "per_class.csv",
    "event_diagnostics.jsonl",
    "summary.md",
)


def _hash_reports(reports_dir: Path) -> tuple[bytes, ...]:
    return tuple((reports_dir / name).read_bytes() for name in _REPORT_NAMES)


def test_score_request_and_outcome_shapes() -> None:
    assert set(ScoreReviewedSubsetRequest.__dataclass_fields__) == {  # pylint: disable=no-member
        "run_path",
        "reference_manifest_path",
        "timing_manifest_path",
        "subset_manifest_path",
        "output_dir",
    }
    assert set(ScoreReviewedSubsetOutcome.__dataclass_fields__) == {  # pylint: disable=no-member
        "exit_code",
        "cohort_id",
        "reports_path",
        "success_count",
        "failed_count",
        "skipped_count",
        "quarantined_count",
    }


def test_score_reviewed_subset_never_constructs_backend_and_preserves_parent_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    subset_path = _finalize_subset(
        tmp_path,
        fixture,
        include_ids=_all_success_includes(fixture),
    )
    parent_reports = fixture.run_path.parent / "reports"
    _parent_reports(fixture, parent_reports)
    before = _hash_reports(parent_reports)

    def fail_backend(*args: object, **kwargs: object) -> object:
        raise AssertionError("reviewed subset scoring must not construct OafBackend")

    monkeypatch.setattr("src.benchmark.oaf_corpus_run.create_backend", fail_backend)
    outcome = score_oaf_reviewed_subset(
        _score_request(fixture, subset_path, tmp_path / "subset-reports")
    )

    assert outcome.exit_code in {0, 1}
    assert _hash_reports(parent_reports) == before


def test_score_reviewed_subset_exits_0_with_derived_cohort_id_and_reports(
    tmp_path: Path,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    include_ids = _all_success_includes(fixture)
    subset_path = _finalize_subset(tmp_path, fixture, include_ids=include_ids)
    from src.benchmark.oaf_corpus_run import parse_oaf_corpus_run

    run_id = str(parse_oaf_corpus_run(fixture.run_path.read_bytes())["run_id"])
    expected_cohort_id = sha256(
        canonical_json_bytes(
            {
                "parent_run_id": run_id,
                "reviewed_subset_manifest_sha256": sha256(subset_path.read_bytes()).hexdigest(),
            }
        )
    ).hexdigest()

    outcome = score_oaf_reviewed_subset(
        _score_request(fixture, subset_path, tmp_path / "subset-reports")
    )

    assert outcome.exit_code == 0
    assert outcome.cohort_id == expected_cohort_id
    assert outcome.cohort_id != run_id
    assert outcome.reports_path == tmp_path / "subset-reports"
    assert outcome.success_count == 20
    assert outcome.failed_count == 0
    assert outcome.skipped_count == 0
    assert outcome.quarantined_count == 0
    items = list(csv.DictReader((tmp_path / "subset-reports" / "items.csv").open(encoding="utf-8")))
    assert {row["simfile_id"] for row in items} == {str(simfile_id) for simfile_id in include_ids}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda snapshot: snapshot.update(reference_manifest_sha256="0" * 64),
        lambda snapshot: snapshot.update(reference_timing_manifest_sha256="0" * 64),
        lambda snapshot: snapshot.update(reference_timing_version="sha256:" + "0" * 64),
    ],
    ids=["run-reference-sha", "run-timing-sha", "run-timing-version"],
)
def test_score_reviewed_subset_exits_2_on_run_lineage_mismatch(
    tmp_path: Path,
    mutate: object,
) -> None:
    from src.benchmark.oaf_corpus_run import parse_oaf_corpus_run, write_oaf_corpus_run

    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    subset_path = _finalize_subset(tmp_path, fixture, include_ids=_all_success_includes(fixture))
    snapshot = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    mutate(snapshot)
    write_oaf_corpus_run(fixture.run_path, snapshot)

    outcome = score_oaf_reviewed_subset(
        _score_request(fixture, subset_path, tmp_path / "subset-reports")
    )

    assert outcome.exit_code == 2
    assert outcome.cohort_id is None
    assert outcome.reports_path is None
    assert outcome.success_count == 0


@pytest.mark.parametrize(
    "field",
    [
        "source_reference_manifest_sha256",
        "source_reference_manifest_version",
        "source_timing_manifest_sha256",
        "source_timing_manifest_version",
    ],
)
def test_score_reviewed_subset_exits_2_on_subset_lineage_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    subset_path = _finalize_subset(tmp_path, fixture, include_ids=_all_success_includes(fixture))
    rows = list(_load_manifest_rows(subset_path))
    for row in rows:
        row[field] = "sha256:" + "9" * 64 if "version" in field else "9" * 64
    mutated = tmp_path / "mutated-subset.jsonl"
    mutated.write_bytes(render_manifest(tuple(rows)).content)

    outcome = score_oaf_reviewed_subset(
        _score_request(fixture, mutated, tmp_path / "subset-reports")
    )

    assert outcome.exit_code == 2
    assert outcome.cohort_id is None
    assert outcome.reports_path is None


def test_score_reviewed_subset_exits_2_when_member_absent_from_parent_population(
    tmp_path: Path,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    subset_path = _finalize_subset(tmp_path, fixture, include_ids=_all_success_includes(fixture))
    rows = list(_load_manifest_rows(subset_path))
    rows[0]["simfile_id"] = 999
    foreign = tmp_path / "foreign-subset.jsonl"
    foreign.write_bytes(render_manifest(tuple(rows)).content)

    outcome = score_oaf_reviewed_subset(
        _score_request(fixture, foreign, tmp_path / "subset-reports")
    )

    assert outcome.exit_code == 2
    assert outcome.cohort_id is None
    assert outcome.reports_path is None


@pytest.mark.parametrize("mode", ["garbage", "missing"], ids=["noncanonical", "unreadable"])
def test_score_reviewed_subset_exits_2_on_noncanonical_or_unreadable_run_snapshot(
    tmp_path: Path,
    mode: str,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    subset_path = _finalize_subset(tmp_path, fixture, include_ids=_all_success_includes(fixture))
    if mode == "garbage":
        fixture.run_path.write_bytes(b"{not canonical run bytes}\n")
    else:
        fixture.run_path.unlink()

    outcome = score_oaf_reviewed_subset(
        _score_request(fixture, subset_path, tmp_path / "subset-reports")
    )

    assert outcome.exit_code == 2
    assert outcome.cohort_id is None
    assert outcome.reports_path is None


def test_score_reviewed_subset_exits_2_on_report_publication_failure(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    subset_path = _finalize_subset(tmp_path, fixture, include_ids=_all_success_includes(fixture))
    blocked = tmp_path / "blocked-reports"
    blocked.write_bytes(b"occupied")

    outcome = score_oaf_reviewed_subset(_score_request(fixture, subset_path, blocked))

    assert outcome.exit_code == 2
    assert outcome.cohort_id is None
    assert outcome.reports_path is None


@pytest.mark.parametrize("corrupt", [False, True], ids=["missing", "corrupt"])
def test_score_reviewed_subset_missing_or_corrupt_prediction_is_item_failure(
    tmp_path: Path,
    corrupt: bool,
) -> None:
    from src.benchmark.oaf_corpus_run import parse_oaf_corpus_run

    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    snapshot = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    target = next(
        row for row in snapshot["items"] if row["execution_disposition"] in {"inferred", "resumed"}
    )
    artifact_path = fixture.oaf_output_dir / str(target["prediction_path"])
    assert artifact_path.exists()
    if corrupt:
        artifact_path.write_bytes(b"corrupted prediction bytes\n")
    else:
        artifact_path.unlink()
    include_ids = _all_success_includes(fixture) | {int(target["simfile_id"])}
    subset_path = _finalize_subset(tmp_path, fixture, include_ids=include_ids)

    outcome = score_oaf_reviewed_subset(
        _score_request(fixture, subset_path, tmp_path / "subset-reports")
    )

    assert outcome.exit_code == 1
    assert outcome.failed_count >= 1
    items = list(csv.DictReader((tmp_path / "subset-reports" / "items.csv").open(encoding="utf-8")))
    failed_rows = [row for row in items if row["status"] == "failed"]
    assert failed_rows
    assert all(
        row["failure_reason"]
        == ("prediction_artifact_invalid" if corrupt else "prediction_missing")
        for row in failed_rows
    )


def test_score_reviewed_subset_diagnostics_only_for_successful_selected_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    include_ids = _all_success_includes(fixture)
    subset_path = _finalize_subset(tmp_path, fixture, include_ids=include_ids)
    import src.benchmark.reviewed_subset as reviewed_module

    real_score = reviewed_module.score_cohort
    captured: dict[str, object] = {}

    def spy(identity, items, *, diagnostics_for=()):
        captured["diagnostics_for"] = diagnostics_for
        captured["selected_items"] = items
        return real_score(identity, items, diagnostics_for=diagnostics_for)

    monkeypatch.setattr(reviewed_module, "score_cohort", spy)
    outcome = score_oaf_reviewed_subset(
        _score_request(fixture, subset_path, tmp_path / "subset-reports")
    )

    assert outcome.exit_code == 0
    selected_items = captured["selected_items"]
    assert isinstance(selected_items, tuple)
    assert captured["diagnostics_for"] == tuple(
        sorted(item.simfile_id for item in selected_items if item.status == "success")
    )
    assert captured["diagnostics_for"]
    diagnostic_lines = (
        (tmp_path / "subset-reports" / "event_diagnostics.jsonl").read_bytes().splitlines()
    )
    assert diagnostic_lines
