"""Prepare → finalize reviewed-subset acceptance chain (HPA-327 Task 5)."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from click.testing import CliRunner

from src.benchmark.backend_identity import strict_json_loads
from src.benchmark.reviewed_subset import (
    FinalizeReviewedSubsetRequest,
    PrepareReviewedSubsetRequest,
    ScoreReviewedSubsetRequest,
    finalize_reviewed_subset,
    load_reviewed_subset_manifest,
    prepare_reviewed_subset,
    score_oaf_reviewed_subset,
    score_reviewed_subset_cohort,
)
from tests.benchmark.reviewed_subset_fixtures import (
    ReviewedSubsetOafFixture,
    ReviewedSubsetReferenceFixture,
    build_reviewed_subset_oaf_fixture,
    build_reviewed_subset_reference_fixture,
)

_INCLUDED_MANUAL_FIELDS: dict[str, str] = {
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
_EXCLUDED_MANUAL_FIELDS: dict[str, str] = {
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


def _apply_manual_review(
    rows: list[dict[str, str]],
    *,
    include: Callable[[int, dict[str, str]], bool],
) -> None:
    """Stamp the shared include/exclude manual payload using ``include``.

    ``include`` receives the zero-based row index and the row itself so the
    two public wrappers can express their distinct inclusion rules without
    duplicating the payload dictionaries.
    """
    for index, row in enumerate(rows):
        row.update(_INCLUDED_MANUAL_FIELDS if include(index, row) else _EXCLUDED_MANUAL_FIELDS)


def _fill_manual_fields(rows: list[dict[str, str]], *, include_count: int) -> None:
    _apply_manual_review(rows, include=lambda index, _row: index < include_count)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(
    path: Path,
    *,
    encoding: str = "utf-8",
    newline: str | None = "",
) -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline=newline) as handle:
        return list(csv.DictReader(handle))


def _completed_review_file(
    tmp_path: Path,
    fixture: ReviewedSubsetReferenceFixture,
    *,
    include_count: int = 20,
    prior_ledger_path: Path | None = None,
) -> Path:
    prepared = tmp_path / "prepared.csv"
    outcome = prepare_reviewed_subset(
        PrepareReviewedSubsetRequest(
            reference_manifest_path=fixture.reference_manifest_path,
            timing_manifest_path=fixture.timing_manifest_path,
            output_file=prepared,
            prior_ledger_path=prior_ledger_path,
        )
    )
    assert outcome.exit_code == 0
    rows = _read_csv_rows(prepared)
    _fill_manual_fields(rows, include_count=include_count)
    review = tmp_path / "review.csv"
    _write_csv(review, rows)
    return review


def test_prepare_to_finalize_publishes_initial_subset(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    review = _completed_review_file(tmp_path, fixture)

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
    assert outcome.review_ledger_path is not None
    assert outcome.included_count == 20
    assert outcome.excluded_count == 10
    loaded = load_reviewed_subset_manifest(outcome.manifest.path)
    assert 20 <= len(loaded.rows) <= 30
    assert (
        loaded.review_ledger_sha256 == sha256(outcome.review_ledger_path.read_bytes()).hexdigest()
    )
    assert loaded.prior_review_ledger_sha256 is None
    ledger_rows = _read_csv_rows(outcome.review_ledger_path)
    assert {row.view.simfile_id for row in loaded.rows} <= {
        int(row["simfile_id"]) for row in ledger_rows
    }
    assert sum(row["decision"] == "include" for row in ledger_rows) == len(loaded.rows)


def test_prepare_to_finalize_continuation_binds_prior_ledger_hash(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=36)
    initial = tmp_path / "initial.csv"
    outcome = prepare_reviewed_subset(
        PrepareReviewedSubsetRequest(
            reference_manifest_path=fixture.reference_manifest_path,
            timing_manifest_path=fixture.timing_manifest_path,
            output_file=initial,
        )
    )
    assert outcome.exit_code == 0
    initial_rows = _read_csv_rows(initial)
    _fill_manual_fields(initial_rows, include_count=20)
    prior_path = tmp_path / "prior.csv"
    _write_csv(prior_path, initial_rows)

    review = _completed_review_file(tmp_path, fixture, prior_ledger_path=prior_path)
    finalized = finalize_reviewed_subset(
        FinalizeReviewedSubsetRequest(
            reference_manifest_path=fixture.reference_manifest_path,
            timing_manifest_path=fixture.timing_manifest_path,
            review_file=review,
            output_dir=tmp_path / "subset",
            prior_ledger_path=prior_path,
        )
    )

    assert finalized.exit_code == 0
    assert finalized.manifest is not None
    assert finalized.review_ledger_path is not None
    loaded = load_reviewed_subset_manifest(finalized.manifest.path)
    assert (
        loaded.review_ledger_sha256 == sha256(finalized.review_ledger_path.read_bytes()).hexdigest()
    )
    assert loaded.prior_review_ledger_sha256 == sha256(prior_path.read_bytes()).hexdigest()
    assert all(
        row.source_row["prior_review_ledger_sha256"] == loaded.prior_review_ledger_sha256
        for row in loaded.rows
    )
    assert all(
        row.source_row["review_ledger_sha256"] == loaded.review_ledger_sha256 for row in loaded.rows
    )


def _oaf_failed_ids(fixture: ReviewedSubsetOafFixture) -> set[int]:
    from src.benchmark.oaf_corpus_run import parse_oaf_corpus_run

    snapshot = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    return {
        int(row["simfile_id"])
        for row in snapshot["items"]
        if row["execution_disposition"] == "failed"
    }


def _oaf_slate_rows(fixture: ReviewedSubsetOafFixture) -> list[dict[str, str]]:
    prepared = Path(fixture.oaf_output_dir).parent / "prepared.csv"
    outcome = prepare_reviewed_subset(
        PrepareReviewedSubsetRequest(
            reference_manifest_path=fixture.reference_manifest_path,
            timing_manifest_path=fixture.timing_manifest_path,
            output_file=prepared,
        )
    )
    assert outcome.exit_code == 0
    return _read_csv_rows(prepared)


def _fill_oaf_manual_fields(rows: list[dict[str, str]], *, include_ids: set[int]) -> None:
    _apply_manual_review(rows, include=lambda _index, row: int(row["simfile_id"]) in include_ids)


def _oaf_review_file(
    tmp_path: Path,
    fixture: ReviewedSubsetOafFixture,
    *,
    include_ids: set[int],
) -> Path:
    rows = _oaf_slate_rows(fixture)
    _fill_oaf_manual_fields(rows, include_ids=include_ids)
    review = tmp_path / "review.csv"
    _write_csv(review, rows)
    return review


def _oaf_subset_manifest(
    tmp_path: Path,
    fixture: ReviewedSubsetOafFixture,
    *,
    include_ids: set[int],
) -> Path:
    review = _oaf_review_file(tmp_path, fixture, include_ids=include_ids)
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


def _oaf_all_success_includes(fixture: ReviewedSubsetOafFixture) -> set[int]:
    failed = _oaf_failed_ids(fixture)
    include_ids: list[int] = []
    for row in _oaf_slate_rows(fixture):
        simfile_id = int(row["simfile_id"])
        if simfile_id in failed:
            continue
        include_ids.append(simfile_id)
        if len(include_ids) == 20:
            break
    return set(include_ids)


def test_cli_chain_matches_domain_chain_bytes(tmp_path: Path) -> None:
    """Run prepare → finalize → score through CliRunner and compare bytes.

    Path-valued CLI output (``output_file``/``manifest_path``/``reports_path``)
    is excluded from comparison; every durable artifact must be byte-identical
    to the direct domain-function chain.
    """
    from src.cli.main import main

    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    include_ids = _oaf_all_success_includes(fixture)

    domain_review = _oaf_review_file(tmp_path, fixture, include_ids=include_ids)
    domain_finalized = finalize_reviewed_subset(
        FinalizeReviewedSubsetRequest(
            reference_manifest_path=fixture.reference_manifest_path,
            timing_manifest_path=fixture.timing_manifest_path,
            review_file=domain_review,
            output_dir=tmp_path / "domain-subset",
        )
    )
    assert domain_finalized.exit_code == 0
    assert domain_finalized.manifest is not None
    assert domain_finalized.review_ledger_path is not None
    domain_reports_dir = tmp_path / "domain-reports"
    domain_scored = score_oaf_reviewed_subset(
        ScoreReviewedSubsetRequest(
            run_path=fixture.run_path,
            reference_manifest_path=fixture.reference_manifest_path,
            timing_manifest_path=fixture.timing_manifest_path,
            subset_manifest_path=domain_finalized.manifest.path,
            output_dir=domain_reports_dir,
        )
    )
    assert domain_scored.exit_code == 0
    assert domain_scored.reports_path is not None

    runner = CliRunner()
    cli_prepared = tmp_path / "cli-prepared.csv"
    prepared = runner.invoke(
        main,
        [
            "benchmark",
            "prepare-reviewed-subset",
            "--manifest",
            str(fixture.reference_manifest_path),
            "--timing-manifest",
            str(fixture.timing_manifest_path),
            "--output-file",
            str(cli_prepared),
        ],
        catch_exceptions=False,
    )
    assert prepared.exit_code == 0
    assert json.loads(prepared.output)["exit_code"] == 0
    domain_prepared = Path(fixture.oaf_output_dir).parent / "prepared.csv"
    assert cli_prepared.read_bytes() == domain_prepared.read_bytes()

    cli_rows = _read_csv_rows(cli_prepared)
    _fill_oaf_manual_fields(cli_rows, include_ids=include_ids)
    cli_review = tmp_path / "cli-review.csv"
    _write_csv(cli_review, cli_rows)

    cli_subset_dir = tmp_path / "cli-subset"
    finalized = runner.invoke(
        main,
        [
            "benchmark",
            "finalize-reviewed-subset",
            "--manifest",
            str(fixture.reference_manifest_path),
            "--timing-manifest",
            str(fixture.timing_manifest_path),
            "--review-file",
            str(cli_review),
            "--output-dir",
            str(cli_subset_dir),
        ],
        catch_exceptions=False,
    )
    assert finalized.exit_code == 0
    finalize_summary = json.loads(finalized.output)
    assert finalize_summary["exit_code"] == 0
    cli_manifest_path = Path(finalize_summary["manifest_path"])

    cli_reports_dir = tmp_path / "cli-reports"
    scored = runner.invoke(
        main,
        [
            "benchmark",
            "score-oaf-reviewed-subset",
            "--run",
            str(fixture.run_path),
            "--manifest",
            str(fixture.reference_manifest_path),
            "--timing-manifest",
            str(fixture.timing_manifest_path),
            "--subset-manifest",
            str(cli_manifest_path),
            "--output-dir",
            str(cli_reports_dir),
        ],
        catch_exceptions=False,
    )
    assert scored.exit_code == 0
    score_summary = json.loads(scored.output)
    assert score_summary["exit_code"] == 0

    assert cli_manifest_path.read_bytes() == domain_finalized.manifest.path.read_bytes()
    assert (cli_subset_dir / "review-ledger.csv").read_bytes() == (
        domain_finalized.review_ledger_path.read_bytes()
    )
    assert finalize_summary["included_count"] == domain_finalized.included_count
    assert finalize_summary["excluded_count"] == domain_finalized.excluded_count
    assert score_summary["cohort_id"] == domain_scored.cohort_id
    assert score_summary["success_count"] == domain_scored.success_count
    for name in (
        "summary.json",
        "items.csv",
        "per_song.csv",
        "per_class.csv",
        "event_diagnostics.jsonl",
        "summary.md",
    ):
        assert (cli_reports_dir / name).read_bytes() == (domain_reports_dir / name).read_bytes()


def test_prepare_finalize_score_chain_rescores_exact_membership(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    include_ids = _oaf_all_success_includes(fixture)
    subset_path = _oaf_subset_manifest(tmp_path, fixture, include_ids=include_ids)
    from src.benchmark.cohort_scoring import score_cohort
    from src.benchmark.oaf_corpus_run import build_oaf_cohort_from_snapshot, parse_oaf_corpus_run
    from src.benchmark.reference_set_manifest import (
        load_reference_set_manifest,
        preflight_reference_mappings,
    )
    from src.benchmark.reference_timing_manifest import load_reference_timing_manifest
    from src.benchmark.reports import write_cohort_reports

    snapshot = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    reference = load_reference_set_manifest(fixture.reference_manifest_path)
    timing = load_reference_timing_manifest(fixture.timing_manifest_path)
    mappings = preflight_reference_mappings(
        reference,
        timing,
        timing_output_root=fixture.timing_output_root,
    )
    parent_identity, parent_items = build_oaf_cohort_from_snapshot(
        snapshot,
        mappings=mappings,
        output_dir=fixture.oaf_output_dir,
    )
    parent_reports = fixture.run_path.parent / "reports"
    write_cohort_reports(
        score_cohort(parent_identity, parent_items, diagnostics_for=()), parent_reports
    )
    before = {
        name: (parent_reports / name).read_bytes()
        for name in (
            "summary.json",
            "items.csv",
            "per_song.csv",
            "per_class.csv",
            "event_diagnostics.jsonl",
            "summary.md",
        )
    }

    outcome = score_oaf_reviewed_subset(
        ScoreReviewedSubsetRequest(
            run_path=fixture.run_path,
            reference_manifest_path=fixture.reference_manifest_path,
            timing_manifest_path=fixture.timing_manifest_path,
            subset_manifest_path=subset_path,
            output_dir=tmp_path / "subset-reports",
        )
    )

    assert outcome.exit_code == 0
    assert outcome.cohort_id != parent_identity.cohort_id
    assert outcome.reports_path == tmp_path / "subset-reports"
    assert outcome.success_count == len(include_ids)
    assert outcome.failed_count == 0
    assert outcome.skipped_count == 0
    assert outcome.quarantined_count == 0
    for name, content in before.items():
        assert (parent_reports / name).read_bytes() == content
    summary = strict_json_loads(
        (tmp_path / "subset-reports" / "summary.json").read_bytes(),
        require_canonical=True,
    )
    assert summary["identity"]["cohort_id"] == outcome.cohort_id
    assert summary["identity"]["reference_manifest_sha256"] == reference.manifest_sha256
    assert summary["population"] == {
        "total_count": 20,
        "success_count": 20,
        "failed_count": 0,
        "skipped_count": 0,
        "quarantined_count": 0,
        "reason_counts": {},
    }
    items = _read_csv_rows(tmp_path / "subset-reports" / "items.csv", newline=None)
    assert {row["simfile_id"] for row in items} == {str(simfile_id) for simfile_id in include_ids}
    diagnostic_lines = (
        (tmp_path / "subset-reports" / "event_diagnostics.jsonl").read_bytes().splitlines()
    )
    assert diagnostic_lines
    for line in diagnostic_lines:
        diagnostic = strict_json_loads(line, require_canonical=True)
        assert diagnostic["cohort_id"] == outcome.cohort_id
        assert diagnostic["simfile_id"] in {str(simfile_id) for simfile_id in include_ids}


def test_core_score_chain_publishes_subset_reports(tmp_path: Path) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    include_ids = _oaf_all_success_includes(fixture)
    subset_path = _oaf_subset_manifest(tmp_path, fixture, include_ids=include_ids)

    from src.benchmark.oaf_corpus_run import build_oaf_cohort_from_snapshot, parse_oaf_corpus_run
    from src.benchmark.reference_set_manifest import (
        load_reference_set_manifest,
        preflight_reference_mappings,
    )
    from src.benchmark.reference_timing_manifest import load_reference_timing_manifest

    reference = load_reference_set_manifest(fixture.reference_manifest_path)
    timing = load_reference_timing_manifest(fixture.timing_manifest_path)
    mappings = preflight_reference_mappings(
        reference,
        timing,
        timing_output_root=fixture.timing_output_root,
    )
    parent_identity, parent_items = build_oaf_cohort_from_snapshot(
        parse_oaf_corpus_run(fixture.run_path.read_bytes()),
        mappings=mappings,
        output_dir=fixture.oaf_output_dir,
    )
    outcome = score_reviewed_subset_cohort(
        parent_identity,
        parent_items,
        reference,
        timing,
        load_reviewed_subset_manifest(subset_path),
        output_dir=tmp_path / "subset-reports",
    )

    assert outcome.exit_code == 0
    assert outcome.success_count == len(include_ids)
    assert outcome.failed_count == 0
    assert (tmp_path / "subset-reports" / "summary.json").exists()


def test_prepare_finalize_score_chain_exits_1_with_non_success_member(
    tmp_path: Path,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path)
    failed_ids = _oaf_failed_ids(fixture)
    assert failed_ids
    include_ids = _oaf_all_success_includes(fixture) | failed_ids
    subset_path = _oaf_subset_manifest(tmp_path, fixture, include_ids=include_ids)

    outcome = score_oaf_reviewed_subset(
        ScoreReviewedSubsetRequest(
            run_path=fixture.run_path,
            reference_manifest_path=fixture.reference_manifest_path,
            timing_manifest_path=fixture.timing_manifest_path,
            subset_manifest_path=subset_path,
            output_dir=tmp_path / "subset-reports",
        )
    )

    assert outcome.exit_code == 1
    assert outcome.success_count == len(include_ids) - len(failed_ids)
    assert outcome.failed_count == len(failed_ids)
    items = _read_csv_rows(tmp_path / "subset-reports" / "items.csv", newline=None)
    failed_rows = {row["simfile_id"]: row for row in items if row["status"] == "failed"}
    assert set(failed_rows) == {str(simfile_id) for simfile_id in failed_ids}
    assert all(row["failure_reason"] == "inference_failed" for row in failed_rows.values())
