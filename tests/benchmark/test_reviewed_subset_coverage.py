from __future__ import annotations

from pathlib import Path

from src.benchmark.reviewed_subset import (
    ScoreReviewedSubsetRequest,
    score_muscriptor_reviewed_subset,
)


def test_score_muscriptor_reviewed_subset_exits_2_on_shallow_run_path(tmp_path: Path) -> None:
    """A run path with fewer than 3 parents is a fatal lineage error (exit 2)."""
    request = ScoreReviewedSubsetRequest(
        run_path=Path("run.json"),
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "timing.jsonl",
        subset_manifest_path=tmp_path / "subset.jsonl",
        output_dir=tmp_path / "subset-reports",
    )

    outcome = score_muscriptor_reviewed_subset(request)

    assert outcome.exit_code == 2
    assert outcome.cohort_id is None
    assert outcome.reports_path is None
    assert outcome.success_count == 0
    assert outcome.failed_count == 0
    assert outcome.skipped_count == 0
    assert outcome.quarantined_count == 0
