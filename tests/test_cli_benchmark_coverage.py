from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import src.cli.benchmark as benchmark_module
from src.benchmark.reviewed_subset import ScoreReviewedSubsetOutcome
from src.cli.main import main


def test_run_muscriptor_corpus_overlap_scope_emits_failed_outcome_and_exits_2(
    tmp_path: Path,
) -> None:
    """Overlapping include/exclude simfile IDs make request construction fail (exit 2)."""
    manifest = tmp_path / "hpa324.jsonl"
    timing_manifest = tmp_path / "hpa323.jsonl"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "run-muscriptor-corpus",
            "--manifest",
            str(manifest),
            "--timing-manifest",
            str(timing_manifest),
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--include-simfile-id",
            "1",
            "--exclude-simfile-id",
            "1",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    summary = json.loads(result.output)
    assert summary["status"] == "failed"
    assert summary["exit_code"] == 2
    assert summary["run_id"] is None
    assert summary["run_path"] is None
    assert summary["success_count"] == 0


def test_score_muscriptor_reviewed_subset_command_exits_nonzero_on_fatal_outcome(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A fatal scoring outcome (exit 2) makes the CLI raise click.exceptions.Exit."""
    import src.benchmark.reviewed_subset as reviewed_module

    run_path = tmp_path / "run.json"
    reference_manifest_path = tmp_path / "hpa324.jsonl"
    timing_manifest_path = tmp_path / "hpa323.jsonl"
    subset_manifest_path = tmp_path / "subset.jsonl"
    output_dir = tmp_path / "reports"
    for path in (
        run_path,
        reference_manifest_path,
        timing_manifest_path,
        subset_manifest_path,
    ):
        path.write_bytes(b"x")

    def fatal_score(request: object) -> ScoreReviewedSubsetOutcome:
        return ScoreReviewedSubsetOutcome(
            exit_code=2,
            cohort_id=None,
            reports_path=None,
            success_count=0,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
        )

    monkeypatch.setattr(reviewed_module, "score_muscriptor_reviewed_subset", fatal_score)
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "score-muscriptor-reviewed-subset",
            "--run",
            str(run_path),
            "--manifest",
            str(reference_manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--subset-manifest",
            str(subset_manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    lines = [line for line in result.output.splitlines() if line]
    assert "score-muscriptor-reviewed-subset: fatal scoring error" in lines[0]
    summary = json.loads(lines[-1])
    assert summary["exit_code"] == 2
    assert summary["cohort_id"] is None
    assert summary["success_count"] == 0


def test_score_muscriptor_reviewed_subset_command_emits_fatal_diagnostic_on_invalid_input(
    tmp_path: Path,
) -> None:
    """Invalid persisted run input triggers a fatal stderr diagnostic and exit 2."""
    run_path = tmp_path / "run.json"
    reference_manifest_path = tmp_path / "hpa324.jsonl"
    timing_manifest_path = tmp_path / "hpa323.jsonl"
    subset_manifest_path = tmp_path / "subset.jsonl"
    output_dir = tmp_path / "reports"
    for path in (
        run_path,
        reference_manifest_path,
        timing_manifest_path,
        subset_manifest_path,
    ):
        path.write_bytes(b"garbage\n")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "score-muscriptor-reviewed-subset",
            "--run",
            str(run_path),
            "--manifest",
            str(reference_manifest_path),
            "--timing-manifest",
            str(timing_manifest_path),
            "--subset-manifest",
            str(subset_manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    lines = [line for line in result.output.splitlines() if line]
    assert "score-muscriptor-reviewed-subset: fatal scoring error" in lines[0]
    summary = json.loads(lines[-1])
    assert summary["exit_code"] == 2
    assert summary["cohort_id"] is None
    assert summary["success_count"] == 0


def test_compare_oaf_muscriptor_command_emits_error_payload_and_exits_2(
    tmp_path: Path,
) -> None:
    """Garbage run/manifest inputs trigger ComparisonIntegrityError (exit 2)."""
    oaf_run = tmp_path / "oaf" / "run.json"
    muscriptor_run = tmp_path / "muscriptor" / "run.json"
    manifest = tmp_path / "hpa324.jsonl"
    timing_manifest = tmp_path / "hpa323.jsonl"
    output_dir = tmp_path / "comparison"
    for path in (oaf_run, muscriptor_run, manifest, timing_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"garbage\n")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "compare-oaf-muscriptor",
            "--oaf-run",
            str(oaf_run),
            "--muscriptor-run",
            str(muscriptor_run),
            "--manifest",
            str(manifest),
            "--timing-manifest",
            str(timing_manifest),
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    lines = [line for line in result.output.splitlines() if line]
    summary = json.loads(lines[-1])
    assert summary["exit_code"] == 2
    assert summary["error"] == "ComparisonIntegrityError"
    assert summary["paired_class_count"] == 0
    assert summary["paired_song_count"] == 0
    assert summary["pairable_success_count"] == 0


def test_current_crux_commit_raises_on_subprocess_error() -> None:
    """_current_crux_commit wraps CalledProcessError as ValueError."""
    with patch.object(
        subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(1, ["git"]),
    ):
        with pytest.raises(ValueError, match="current Crux commit is unavailable"):
            benchmark_module._current_crux_commit()


def test_current_crux_commit_rejects_non_canonical_commit() -> None:
    """_current_crux_commit rejects a short or non-hex commit."""
    fake_result = subprocess.CompletedProcess(
        args=["git", "rev-parse", "--verify", "HEAD"],
        returncode=0,
        stdout="short\n",
    )
    with patch.object(subprocess, "run", return_value=fake_result):
        with pytest.raises(ValueError, match="not canonical"):
            benchmark_module._current_crux_commit()


def test_run_oaf_separation_pilot_command_emits_failed_outcome_on_commit_error(
    tmp_path,
    monkeypatch,
) -> None:
    """When _current_crux_commit raises ValueError the CLI emits a failed outcome."""
    monkeypatch.setattr(
        benchmark_module,
        "_current_crux_commit",
        lambda: (_ for _ in ()).throw(ValueError("no commit")),
    )
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "run-oaf-separation-pilot",
            "--manifest",
            str(tmp_path / "m.json"),
            "--timing-manifest",
            str(tmp_path / "t.json"),
            "--subset-manifest",
            str(tmp_path / "s.json"),
            "--oaf-run",
            str(tmp_path / "run.json"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
            "--spleeter-python",
            str(tmp_path / "sp"),
            "--demucs-python",
            str(tmp_path / "dm"),
            "--spleeter-model-root",
            str(tmp_path / "smr"),
            "--demucs-model-root",
            str(tmp_path / "dmr"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    summary = json.loads(result.output)
    assert summary["status"] == "failed"
    assert summary["exit_code"] == 2
    assert summary["success_count"] == 0


def test_finalize_oaf_separation_pilot_command_emits_failed_outcome_on_invalid_decision(
    tmp_path,
) -> None:
    """An invalid decision value triggers ValueError in request construction."""
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "finalize-oaf-separation-pilot",
            "--run",
            str(tmp_path / "run.json"),
            "--subset-manifest",
            str(tmp_path / "subset.json"),
            "--output-manifest",
            str(tmp_path / "out.json"),
            "--decision",
            "bogus_decision",
            "--rationale",
            "test rationale",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    summary = json.loads(result.output)
    assert summary["exit_code"] == 2
    assert summary["manifest_path"] is None
    assert "decision is invalid" in summary["failure_reason"]
