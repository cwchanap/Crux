"""Prepare → finalize reviewed-subset acceptance chain (HPA-327 Task 5)."""

from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path

from src.benchmark.reviewed_subset import (
    FinalizeReviewedSubsetRequest,
    PrepareReviewedSubsetRequest,
    finalize_reviewed_subset,
    load_reviewed_subset_manifest,
    prepare_reviewed_subset,
)
from tests.benchmark.reviewed_subset_fixtures import (
    ReviewedSubsetReferenceFixture,
    build_reviewed_subset_reference_fixture,
)


def _fill_manual_fields(rows: list[dict[str, str]], *, include_count: int) -> None:
    for index, row in enumerate(rows):
        if index < include_count:
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


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
    rows = list(csv.DictReader(prepared.open(encoding="utf-8", newline="")))
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
    ledger_rows = list(
        csv.DictReader(outcome.review_ledger_path.open(encoding="utf-8", newline=""))
    )
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
    initial_rows = list(csv.DictReader(initial.open(encoding="utf-8", newline="")))
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
