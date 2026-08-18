"""Focused HPA-396 handoff finalizer and consumer tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.benchmark.backend_identity import strict_json_loads
from src.benchmark.separation_handoff import (
    SEPARATION_PILOT_SCHEMA,
    FinalizeSeparationPilotRequest,
    finalize_separation_pilot,
    load_separation_pilot_manifest,
    validate_schema_golden,
)

_GOLDEN = Path(__file__).parent / "schema_goldens/oaf-separation-pilot-v1.jsonl"
_COMPARISON_FILES = (
    "summary.json",
    "summary.md",
    "spleeter/paired_per_song.csv",
    "spleeter/paired_per_class.csv",
    "htdemucs/paired_per_song.csv",
    "htdemucs/paired_per_class.csv",
)


def _publish_comparison_fixtures(output_dir: Path) -> None:
    for relative in _COMPARISON_FILES:
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"comparison:{relative}\n".encode("utf-8"))


def _successful_pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[object, Path, Path]:
    """Run the existing synthetic pilot seams and retain its immutable evidence."""
    from src.benchmark.separation_pilot import run_oaf_separation_pilot
    from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_oaf_fixture
    from tests.benchmark.test_separation_pilot import _request, _subset_path, _task6_seams

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20, failed_count=0)
    subset_path = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset_path)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)

    import src.benchmark.separation_pilot as pilot

    monkeypatch.setattr(pilot, "_comparison_reports_ready", lambda _run_dir: True)
    monkeypatch.setattr(
        pilot,
        "compare_oaf_separation",
        lambda comparison_request: _publish_comparison_fixtures(comparison_request.output_dir),
    )
    ticks = iter(float(index) for index in range(400))
    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
        perf_counter=lambda: next(ticks),
    )
    assert outcome.exit_code == 0
    assert outcome.run_path is not None

    # The synthetic control points at the parent run's output root.  Copy only
    # those immutable prediction bytes into the separation output root so the
    # finalizer can resolve every retained prediction from the published run.
    parent_prediction_root = fixture.oaf_output_dir
    separation_prediction_root = request.output_dir
    snapshot = json.loads(outcome.run_path.read_text(encoding="utf-8"))
    for item in snapshot["items"]:
        prediction = item["full_mix"]["prediction"]
        if prediction is None:
            continue
        source = parent_prediction_root / prediction["path"]
        destination = separation_prediction_root / prediction["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return request, subset_path, outcome.run_path


def test_oaf_separation_pilot_schema_golden_round_trips() -> None:
    validate_schema_golden(SEPARATION_PILOT_SCHEMA, _GOLDEN.read_bytes())


def test_finalize_publishes_stable_htdemucs_evidence_without_run_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    output_manifest = tmp_path / "handoff" / "manifest.jsonl"

    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=output_manifest,
            decision="use_htdemucs",
            rationale="HTDemucs is the reviewed pilot choice.",
        )
    )

    assert outcome.exit_code == 0
    assert outcome.manifest is not None
    loaded = load_separation_pilot_manifest(outcome.manifest.path)
    assert len(loaded.rows) == 20
    row = loaded.rows[0]
    assert row["simfile_id"] == 100
    assert row["source_row_sha256"]
    assert row["source_audio_sha256"]
    htdemucs = row["htdemucs"]
    assert htdemucs["status"] in {"success", "resumed"}
    assert htdemucs["stem"]["path"]
    assert len(htdemucs["stem"]["sha256"]) == 64
    assert len(htdemucs["input"]["input_audio_sha256"]) == 64
    assert len(htdemucs["prediction"]["artifact_sha256"]) == 64
    assert row["comparison_artifacts"]["htdemucs/paired_per_song.csv"]["sha256"]

    # The handoff remains consumable after the mutable HPA-328 run snapshot is
    # gone; all HPA-396 evidence is in the immutable manifest row.
    run_path.unlink()
    loaded_again = load_separation_pilot_manifest(outcome.manifest.path)
    assert loaded_again.rows[0] == row
    assert request.output_dir.exists()


@pytest.mark.parametrize("evidence", ["stem", "prediction"])
def test_finalize_rejects_edited_htdemucs_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence: str,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    target = snapshot["items"][0]["htdemucs"][evidence]
    if evidence == "stem":
        path = request.cache_dir / target["path"]
    else:
        path = request.output_dir / target["path"]
    path.write_bytes(b"edited retained evidence\n")

    output_manifest = tmp_path / "edited-handoff" / "manifest.jsonl"
    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=output_manifest,
            decision="gather_more_evidence",
            rationale="Retained evidence must remain immutable.",
        )
    )

    assert outcome.exit_code == 2
    assert outcome.manifest is None
    assert not output_manifest.exists()


def test_loader_enforces_success_evidence_nullability() -> None:
    rows = [
        strict_json_loads(line, require_canonical=True)
        for line in _GOLDEN.read_bytes().splitlines()
    ]
    assert isinstance(rows[0], dict)
    broken = dict(rows[0])
    broken_htdemucs = dict(broken["htdemucs"])
    broken_htdemucs["prediction"] = None
    broken["htdemucs"] = broken_htdemucs
    from src.benchmark.corpus_manifest import render_manifest

    content = render_manifest(
        ({key: value for key, value in broken.items() if key != "corpus_version"},)
    ).content
    with pytest.raises(ValueError, match="prediction"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, content)


def test_handoff_schema_has_no_cost_fields() -> None:
    content = _GOLDEN.read_text(encoding="utf-8").lower()
    assert "cost" not in content
    assert "dollar" not in content
    assert "$" not in content
