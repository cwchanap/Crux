"""HPA-328 fixed-subset preflight and snapshot tests."""

# Fixtures intentionally import the production seam inside tests so the
# baseline collection remains free of the optional runtime modules.
# pylint: disable=import-outside-toplevel,too-many-locals,duplicate-code

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.oaf_corpus_run import parse_oaf_corpus_run
from src.benchmark.reference_set_manifest import load_reference_set_manifest
from src.benchmark.reference_timing_manifest import load_reference_timing_manifest
from src.benchmark.reviewed_subset import (
    ScoreReviewedSubsetOutcome,
    load_reviewed_subset_manifest,
)
from src.benchmark.separators import HTDEMUCS_SEPARATOR_ID, SPLEETER_SEPARATOR_ID
from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_oaf_fixture

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "separators"
_GOLDEN = Path(__file__).parent / "schema_goldens/crux.reviewed-reference-subset-v1.jsonl"


def _subset_path(tmp_path: Path, fixture: object) -> Path:
    source = strict_json_loads(_GOLDEN.read_bytes()[:-1], require_canonical=True)
    assert isinstance(source, dict)
    reference = load_reference_set_manifest(fixture.reference_manifest_path)
    timing = load_reference_timing_manifest(fixture.timing_manifest_path)
    rows: list[dict[str, object]] = []
    for rank, loaded in enumerate(reference.rows, start=1):
        row = dict(source)
        row.pop("corpus_version", None)
        row.update(
            {
                "candidate_rank": rank,
                "simfile_id": loaded.view.simfile_id,
                "source_reference_manifest_sha256": reference.manifest_sha256,
                "source_reference_manifest_version": reference.corpus_version,
                "source_timing_manifest_sha256": timing.manifest_sha256,
                "source_timing_manifest_version": timing.corpus_version,
                "source_row_sha256": hashlib.sha256(
                    canonical_json_bytes({"simfile_id": loaded.view.simfile_id})
                ).hexdigest(),
                "selected_chart_key": loaded.source_row["selected_chart_key"],
                "selected_chart_content_hash": loaded.source_row["selected_chart_content_hash"],
                "source_audio_key": loaded.source_row["source_audio_key"],
                "source_audio_content_hash": loaded.source_row["source_audio_content_hash"],
                "common_event_count": loaded.view.common_scored_event_count,
                "review_ledger_sha256": "a" * 64,
            }
        )
        rows.append(row)
    path = tmp_path / "subset.jsonl"
    path.write_bytes(render_manifest(tuple(rows)).content)
    return path


def _request(tmp_path: Path, fixture: object, subset_path: Path):
    from src.benchmark.separation_pilot import OafSeparationPilotRequest

    return OafSeparationPilotRequest(
        reference_manifest_path=fixture.reference_manifest_path,
        timing_manifest_path=fixture.timing_manifest_path,
        subset_manifest_path=subset_path,
        oaf_run_path=fixture.run_path,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "separation-output",
        spleeter_python=Path("/isolated/spleeter/python"),
        demucs_python=Path("/isolated/demucs/python"),
        crux_commit="c" * 40,
    )


def _install_fixture_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.benchmark.separation_pilot as pilot

    monkeypatch.setattr(
        pilot,
        "SEPARATOR_LOCK_PATHS",
        {
            SPLEETER_SEPARATOR_ID: FIXTURE_ROOT / "spleeter-model.json",
            HTDEMUCS_SEPARATOR_ID: FIXTURE_ROOT / "htdemucs-model.json",
        },
    )


def test_request_exposes_only_the_fixed_pilot_controls() -> None:
    from src.benchmark.separation_pilot import OafSeparationPilotRequest

    assert {field.name for field in fields(OafSeparationPilotRequest)} == {
        "reference_manifest_path",
        "timing_manifest_path",
        "subset_manifest_path",
        "oaf_run_path",
        "cache_dir",
        "output_dir",
        "spleeter_python",
        "demucs_python",
        "resume",
        "crux_commit",
    }


def test_lineage_preflight_fails_before_full_mix_control_or_runtime_touch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    calls: list[str] = []

    def forbidden_score(_request: object) -> object:
        calls.append("score")
        raise AssertionError("full-mix control must not run after fatal preflight")

    monkeypatch.setattr("src.benchmark.separation_pilot.score_oaf_reviewed_subset", forbidden_score)
    broken_rows = list(
        strict_json_loads(line[:-1], require_canonical=True)
        for line in subset.read_bytes().splitlines(keepends=True)
    )
    assert isinstance(broken_rows[0], dict)
    broken_rows[0]["source_reference_manifest_sha256"] = "d" * 64
    for row in broken_rows:
        assert isinstance(row, dict)
        row.pop("corpus_version", None)
    broken = tmp_path / "broken-subset.jsonl"
    broken.write_bytes(render_manifest(tuple(broken_rows)).content)

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(replace(request, subset_manifest_path=broken))

    assert outcome.exit_code == 2
    assert outcome.run_path is None
    assert not calls


def test_other_fatal_preflight_cases_never_call_public_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls: list[str] = []

    def forbidden_score(_request: object) -> object:
        calls.append("score")
        raise AssertionError("fatal preflight must stop before the public control")

    monkeypatch.setattr("src.benchmark.separation_pilot.score_oaf_reviewed_subset", forbidden_score)

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    _install_fixture_locks(monkeypatch)
    missing_member_rows = [
        json.loads(line[:-1]) for line in subset.read_bytes().splitlines(keepends=True)
    ]
    for row in missing_member_rows:
        assert isinstance(row, dict)
        row.pop("corpus_version", None)
    assert isinstance(missing_member_rows[0], dict)
    missing_member_rows[0]["simfile_id"] = 9999
    missing_member = tmp_path / "missing-member.jsonl"
    missing_member.write_bytes(render_manifest(tuple(missing_member_rows)).content)
    assert (
        run_oaf_separation_pilot(replace(request, subset_manifest_path=missing_member)).exit_code
        == 2
    )

    _install_fixture_locks(monkeypatch)
    monkeypatch.setattr(
        "src.benchmark.separation_pilot.SEPARATOR_LOCK_PATHS",
        {
            SPLEETER_SEPARATOR_ID: tmp_path / "missing-spleeter-lock.json",
            HTDEMUCS_SEPARATOR_ID: FIXTURE_ROOT / "htdemucs-model.json",
        },
    )
    assert run_oaf_separation_pilot(request).exit_code == 2

    _install_fixture_locks(monkeypatch)
    aliased_output = fixture.run_path.parent / "reports"
    assert run_oaf_separation_pilot(replace(request, output_dir=aliased_output)).exit_code == 2
    assert not calls


def test_exact_subset_is_sorted_identity_bound_and_full_mix_uses_public_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    captured: list[object] = []

    def fake_score(score_request: object) -> ScoreReviewedSubsetOutcome:
        captured.append(score_request)
        return ScoreReviewedSubsetOutcome(
            exit_code=0,
            cohort_id="subset-cohort",
            reports_path=score_request.output_dir,  # type: ignore[attr-defined]
            success_count=20,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
        )

    monkeypatch.setattr("src.benchmark.separation_pilot.score_oaf_reviewed_subset", fake_score)

    from src.benchmark.separation_pilot import (
        SEPARATION_RUN_SCHEMA,
        run_oaf_separation_pilot,
    )

    outcome = run_oaf_separation_pilot(request)

    assert outcome.exit_code == 0
    assert outcome.run_id is not None
    assert outcome.run_path is not None
    assert len(captured) == 1
    score_request = captured[0]
    assert score_request.run_path == request.oaf_run_path  # type: ignore[attr-defined]
    assert score_request.output_dir == (  # type: ignore[attr-defined]
        outcome.run_path.parent / "views" / "full_mix" / "reports"
    )

    snapshot = json.loads(outcome.run_path.read_text(encoding="utf-8"))
    subset_loaded = load_reviewed_subset_manifest(subset)
    expected_ids = sorted(row.view.simfile_id for row in subset_loaded.rows)
    assert snapshot["schema"] == SEPARATION_RUN_SCHEMA
    assert [row["simfile_id"] for row in snapshot["items"]] == expected_ids
    assert len(snapshot["items"]) == len(expected_ids)
    assert "sample_size" not in snapshot
    assert "seed" not in snapshot
    assert "include_simfile_ids" not in snapshot
    assert all("source_row_sha256" in row for row in snapshot["items"])
    assert all("full_mix" in row for row in snapshot["items"])
    assert all("spleeter" in row and "htdemucs" in row for row in snapshot["items"])

    parent = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    identity = {
        "schema": SEPARATION_RUN_SCHEMA,
        "reviewed_subset_manifest_sha256": subset_loaded.manifest_sha256,
        "reference_manifest_sha256": load_reference_set_manifest(
            fixture.reference_manifest_path
        ).manifest_sha256,
        "reference_timing_manifest_sha256": load_reference_timing_manifest(
            fixture.timing_manifest_path
        ).manifest_sha256,
        "parent_oaf_run_id": parent["run_id"],
        "oaf_backend_descriptor_sha256": parent["backend_descriptor_sha256"],
        "oaf_model_lock_sha256": parent["model_lock_sha256"],
        "oaf_checkpoint_archive_sha256": parent["checkpoint_archive_sha256"],
        "spleeter_lock_sha256": snapshot["spleeter_lock_sha256"],
        "htdemucs_lock_sha256": snapshot["htdemucs_lock_sha256"],
        "spleeter_input_view_id": snapshot["spleeter_input_view_id"],
        "htdemucs_input_view_id": snapshot["htdemucs_input_view_id"],
        "scoring_version": snapshot["scoring_version"],
        "crux_commit": request.crux_commit,
    }
    expected_run_id = (
        "oaf-separation-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:16]
    )
    assert outcome.run_id == expected_run_id


@pytest.mark.parametrize("field", ["input_view_id", "model_lock_sha256"])
def test_parent_mixed_oaf_identity_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    parent = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    parent[field] = "e" * 64 if field == "model_lock_sha256" else "mixed-input-view"
    from src.benchmark.oaf_corpus_run import write_oaf_corpus_run

    write_oaf_corpus_run(fixture.run_path, parent)
    monkeypatch.setattr(
        "src.benchmark.separation_pilot.score_oaf_reviewed_subset",
        lambda _request: pytest.fail("mixed parent identity must fail before scoring"),
    )

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    assert run_oaf_separation_pilot(request).exit_code == 2
