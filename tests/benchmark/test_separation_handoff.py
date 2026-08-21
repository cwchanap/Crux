"""Focused HPA-396 handoff finalizer and consumer tests."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import src.benchmark.separation_handoff as separation_handoff
from src.benchmark.backend_identity import strict_json_loads
from src.benchmark.corpus_manifest import canonical_json_line, render_manifest
from src.benchmark.oaf_corpus_run import OAF_FULL_MIX_INPUT_VIEW_ID
from src.benchmark.separation_handoff import (
    SEPARATION_PILOT_SCHEMA,
    FinalizeSeparationPilotRequest,
    SeparationHandoffError,
    _commit,
    _direct_manifest_alias,
    _hash,
    _manifest_duration,
    _manifest_json_value,
    _owner_root,
    _path,
    _positive_duration,
    _read_matching_artifact,
    _recorded_artifact_path,
    _text,
    _validate_comparison_artifacts,
    _validate_input,
    _validate_prediction,
    _validate_row,
    _validate_stem,
    _validate_view,
    _version,
    finalize_separation_pilot,
    load_separation_pilot_manifest,
    validate_schema_golden,
)
from src.benchmark.separation_pilot import (
    HTDEMUCS_INPUT_VIEW_ID,
    SPLEETER_INPUT_VIEW_ID,
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


def _successful_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cache_dir: Path | None = None,
    actual_comparison: bool = False,
    runtime_inputs: tuple[Path, Path, Path, Path] | None = None,
) -> tuple[object, Path, Path]:
    """Run the existing synthetic pilot seams and retain its immutable evidence."""
    from src.benchmark.separation_pilot import run_oaf_separation_pilot
    from tests.benchmark.reviewed_subset_fixtures import build_reviewed_subset_oaf_fixture
    from tests.benchmark.test_separation_pilot import _request, _subset_path, _task6_seams

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20, failed_count=0)
    subset_path = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset_path)
    if runtime_inputs is not None:
        request = replace(
            request,
            spleeter_python=runtime_inputs[0],
            demucs_python=runtime_inputs[1],
            spleeter_model_root=runtime_inputs[2],
            demucs_model_root=runtime_inputs[3],
        )
    import src.benchmark.separation_pilot as pilot

    real_score_oaf_reviewed_subset = pilot.score_oaf_reviewed_subset
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    backend_factory = calls["factory"][0]

    if actual_comparison:
        monkeypatch.setattr(pilot, "score_oaf_reviewed_subset", real_score_oaf_reviewed_subset)

        original_backend_factory = backend_factory

        def canonical_comparison_backend_factory(**kwargs: object) -> object:
            backend = original_backend_factory(**kwargs)  # type: ignore[operator]
            original_transcribe = backend.transcribe  # type: ignore[attr-defined]

            def transcribe(audio: object) -> object:
                native = original_transcribe(audio)
                return replace(
                    native,
                    events=tuple(replace(event, time_sec=0.375) for event in native.events),
                )

            backend.transcribe = transcribe  # type: ignore[attr-defined]
            return backend

        backend_factory = canonical_comparison_backend_factory
    else:
        monkeypatch.setattr(pilot, "_comparison_reports_ready", lambda _run_dir: True)
        monkeypatch.setattr(
            pilot,
            "compare_oaf_separation",
            lambda comparison_request: _publish_comparison_fixtures(comparison_request.output_dir),
        )
    ticks = iter(float(index) for index in range(400))
    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=backend_factory,  # type: ignore[arg-type]
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
    if cache_dir is not None:
        # The shared Task 6 seam writes its stems under tmp_path/cache. Move
        # that exact cache to the caller-supplied owner and update only the
        # mutable producer snapshot's owner metadata.
        shutil.move(str(request.cache_dir), str(cache_dir))
        for item in snapshot["items"]:
            for view_name in ("spleeter", "htdemucs"):
                item[view_name]["stem"]["owner_root"] = str(cache_dir.resolve())
        from src.benchmark.separation_pilot import write_oaf_separation_run

        write_oaf_separation_run(outcome.run_path, snapshot)
        request = replace(request, cache_dir=cache_dir)
    return request, subset_path, outcome.run_path


def test_oaf_separation_pilot_schema_golden_round_trips() -> None:
    validate_schema_golden(SEPARATION_PILOT_SCHEMA, _GOLDEN.read_bytes())


def test_finalize_failure_reason_hides_filesystem_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)

    def failing_publish(*args: object, **kwargs: object) -> object:
        raise OSError(f"cannot publish under {tmp_path}")

    monkeypatch.setattr(separation_handoff, "publish_manifest", failing_publish)

    with caplog.at_level(logging.ERROR, logger="src.benchmark.separation_handoff"):
        outcome = finalize_separation_pilot(
            FinalizeSeparationPilotRequest(
                run_path=run_path,
                subset_manifest_path=subset_path,
                output_manifest=tmp_path / "unwritable-handoff" / "manifest.jsonl",
                decision="keep_full_mix",
                rationale="A raw publication error must not leak paths.",
            )
        )

    assert outcome.exit_code == 2
    assert outcome.manifest is None
    assert outcome.failure_reason == "separation handoff finalization failed"
    assert str(tmp_path) not in outcome.failure_reason
    assert str(tmp_path) in caplog.text


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


def test_finalize_rehashes_full_mix_prediction_at_parent_owner_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    relative = snapshot["items"][0]["full_mix"]["prediction"]["path"]
    parent_path = tmp_path / "oaf-output" / relative
    separation_decoy = request.output_dir / relative
    assert parent_path.exists()
    separation_decoy.unlink()

    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "parent-owned-handoff" / "manifest.jsonl",
            decision="keep_full_mix",
            rationale="The inherited full-mix prediction remains parent-owned.",
        )
    )

    assert outcome.exit_code == 0
    assert outcome.manifest is not None


def test_finalize_rejects_full_mix_decoy_when_parent_prediction_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    relative = snapshot["items"][0]["full_mix"]["prediction"]["path"]
    parent_path = tmp_path / "oaf-output" / relative
    separation_decoy = request.output_dir / relative
    assert parent_path.exists() and separation_decoy.exists()
    parent_path.unlink()

    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "parent-decoy-handoff" / "manifest.jsonl",
            decision="keep_full_mix",
            rationale="A separation-root decoy cannot replace parent evidence.",
        )
    )

    assert outcome.exit_code == 2
    assert outcome.manifest is None


def test_finalize_rejects_copied_parent_snapshot_and_prediction_decoy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    relative = snapshot["items"][0]["full_mix"]["prediction"]["path"]
    parent_prediction = request.oaf_run_path.parent.parent.parent / relative
    separation_prediction = request.output_dir / relative
    assert parent_prediction.exists() and separation_prediction.exists()
    parent_prediction.unlink()

    copied_parent = request.output_dir / "runs" / snapshot["parent_oaf_run_id"] / "run.json"
    copied_parent.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(request.oaf_run_path, copied_parent)

    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "copied-parent-handoff" / "manifest.jsonl",
            decision="keep_full_mix",
            rationale="A copied parent snapshot cannot move parent-owned evidence.",
        )
    )

    assert outcome.exit_code == 2
    assert outcome.manifest is None


def test_finalize_uses_caller_supplied_cache_owner_for_retained_stems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(
        tmp_path,
        monkeypatch,
        cache_dir=tmp_path / "caller-cache",
    )

    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "caller-cache-handoff" / "manifest.jsonl",
            decision="use_htdemucs",
            rationale="Retained stems stay under the supplied cache owner.",
        )
    )

    assert outcome.exit_code == 0
    assert outcome.manifest is not None


@pytest.mark.parametrize("evidence", ["stem", "prediction"])
def test_finalize_rejects_missing_recorded_evidence_even_with_decoy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence: str,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    target = snapshot["items"][0]["htdemucs"][evidence]
    if evidence == "stem":
        recorded = request.cache_dir / target["path"]
    else:
        recorded = request.output_dir / target["path"]
    decoy = tmp_path / target["path"]
    decoy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(recorded, decoy)
    recorded.unlink()

    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "missing-recorded-handoff" / "manifest.jsonl",
            decision="gather_more_evidence",
            rationale="A missing recorded artifact cannot be replaced by a decoy.",
        )
    )

    assert outcome.exit_code == 2
    assert outcome.manifest is None


def test_finalize_rejects_snapshot_lineage_stale_against_subset_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    import src.benchmark.separation_handoff as handoff

    loaded_subset = handoff.load_reviewed_subset_manifest(subset_path)
    stale_subset = replace(
        loaded_subset,
        source_reference_manifest_sha256="0" * 64,
    )
    monkeypatch.setattr(handoff, "load_reviewed_subset_manifest", lambda _path: stale_subset)

    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "stale-lineage-handoff" / "manifest.jsonl",
            decision="keep_full_mix",
            rationale="The subset source lineage must be exact.",
        )
    )

    assert outcome.exit_code == 2
    assert outcome.manifest is None


def test_finalize_accepts_partial_full_mix_input_with_successful_htdemucs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    full_mix = dict(snapshot["items"][0]["full_mix"])
    full_mix["status"] = "failed"
    full_mix["failure_code"] = "inference_failed"
    full_mix["prediction"] = None
    snapshot["items"][0]["full_mix"] = full_mix
    snapshot["overall_status"] = "partial"
    from src.benchmark.separation_pilot import write_oaf_separation_run

    write_oaf_separation_run(run_path, snapshot)

    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "partial-handoff" / "manifest.jsonl",
            decision="use_htdemucs",
            rationale="A successful HTDemucs row remains usable in a partial pilot.",
        )
    )

    assert outcome.exit_code == 0
    assert outcome.manifest is not None
    row = load_separation_pilot_manifest(outcome.manifest.path).rows[0]
    assert row["full_mix"]["status"] == "failed"
    assert row["full_mix"]["failure_code"] == "inference_failed"
    assert row["full_mix"]["input"] is not None
    assert row["full_mix"]["prediction"] is None
    assert row["htdemucs"]["status"] in {"success", "resumed"}


def test_loader_rejects_successful_view_failure_code() -> None:
    rows = [
        strict_json_loads(line, require_canonical=True)
        for line in _GOLDEN.read_bytes().splitlines()
    ]
    broken = dict(rows[0])
    broken_htdemucs = dict(broken["htdemucs"])
    broken_htdemucs["failure_code"] = "stale_failure"
    broken["htdemucs"] = broken_htdemucs
    content = render_manifest(
        ({key: value for key, value in broken.items() if key != "corpus_version"},)
    ).content
    with pytest.raises(ValueError, match="failure_code"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, content)


def test_loader_rejects_contradictory_parent_or_comparison_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "identity-handoff" / "manifest.jsonl",
            decision="use_htdemucs",
            rationale="The canonical handoff keeps one parent identity.",
        )
    )
    assert outcome.manifest is not None
    rows = [
        strict_json_loads(line, require_canonical=True)
        for line in outcome.manifest.path.read_bytes().splitlines()
    ]
    assert len(rows) >= 2

    for field in ("oaf_model_id", "oaf_inference_config_sha256", "comparison_artifacts"):
        broken_rows = [dict(row) for row in rows]
        broken = broken_rows[1]
        if field == "comparison_artifacts":
            comparison = dict(broken[field])
            artifact = dict(comparison["summary.json"])
            artifact["sha256"] = "f" * 64
            comparison["summary.json"] = artifact
            broken[field] = comparison
        elif field.endswith("sha256"):
            broken[field] = "e" * 64
        else:
            broken[field] = "different-oaf-model"
        content = render_manifest(
            tuple(
                {key: value for key, value in row.items() if key != "corpus_version"}
                for row in broken_rows
            )
        ).content
        with pytest.raises(ValueError, match="mixed run or decision identity"):
            validate_schema_golden(SEPARATION_PILOT_SCHEMA, content)


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


# ---------------------------------------------------------------------------
# Helpers and constants for the validator-focused coverage tests.
# ---------------------------------------------------------------------------

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_FULL_MIX_ID = OAF_FULL_MIX_INPUT_VIEW_ID
_SPLEETER_ID = SPLEETER_INPUT_VIEW_ID
_HTEMUCS_ID = HTDEMUCS_INPUT_VIEW_ID


def _golden_row() -> dict:
    rows = [
        strict_json_loads(line, require_canonical=True)
        for line in _GOLDEN.read_bytes().splitlines()
    ]
    return dict(rows[0])


def _render_rows(*rows: object) -> bytes:
    return render_manifest(
        tuple(
            {k: v for k, v in row.items() if k != "corpus_version"}  # type: ignore[union-attr]
            for row in rows
        )
    ).content


def _valid_stem(
    *,
    source_audio_sha256: str = _SHA_A,
    separator_lock_sha256: str = _SHA_B,
) -> dict[str, object]:
    return {
        "path": "derived/stems/x/drums.wav",
        "sha256": _SHA_A,
        "source_audio_sha256": source_audio_sha256,
        "separator_lock_sha256": separator_lock_sha256,
    }


def _valid_input(
    field: str,
    *,
    view_id: str,
    source_audio_id: str = "100/bgm.wav",
    source_audio_sha256: str = _SHA_A,
    input_audio_sha256: str = _SHA_B,
) -> dict[str, object]:
    return {
        "path": None if field == "full_mix" else "inputs/100/x.wav",
        "input_view_id": view_id,
        "input_audio_sha256": input_audio_sha256,
        "source_audio_id": source_audio_id,
        "source_audio_sha256": source_audio_sha256,
    }


def _valid_prediction(
    field: str,
    *,
    view_id: str,
    source_audio_id: str = "100/bgm.wav",
    source_audio_sha256: str = _SHA_A,
    input_audio_sha256: str = _SHA_B,
) -> dict[str, object]:
    return {
        "path": "predictions/x.json",
        "artifact_sha256": _SHA_A,
        "source_audio_id": source_audio_id,
        "source_audio_sha256": source_audio_sha256,
        "input_view_id": view_id,
        "input_audio_sha256": input_audio_sha256,
    }


def _valid_view(field: str, *, status: str = "success") -> dict[str, object]:
    view_id = {"full_mix": _FULL_MIX_ID, "spleeter": _SPLEETER_ID, "htdemucs": _HTEMUCS_ID}[field]
    if field == "full_mix" and status == "success":
        status = "inferred"
    successful = status in {"inferred", "success", "resumed"}
    if field == "full_mix":
        return {
            "status": status,
            "failure_code": None if successful else "failure",
            "separator_lock_sha256": None,
            "input_view_id": view_id,
            "stem": None,
            "input": _valid_input(field, view_id=view_id) if successful else None,
            "prediction": _valid_prediction(field, view_id=view_id) if successful else None,
        }
    return {
        "status": status,
        "failure_code": None if successful else "failure",
        "separator_lock_sha256": _SHA_B,
        "input_view_id": view_id,
        "stem": _valid_stem() if successful else None,
        "input": _valid_input(field, view_id=view_id) if successful else None,
        "prediction": _valid_prediction(field, view_id=view_id) if successful else None,
    }


def _finalize_with_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot: object,
    subset_path: Path,
    run_path: Path,
    *,
    output_name: str = "custom-handoff",
    decision: str = "use_htdemucs",
) -> object:
    import src.benchmark.separation_handoff as handoff

    monkeypatch.setattr(handoff, "parse_oaf_separation_run", lambda _content: snapshot)
    return finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / output_name / "manifest.jsonl",
            decision=decision,  # type: ignore[arg-type]
            rationale="Custom rationale for coverage.",
        )
    )


# ---------------------------------------------------------------------------
# FinalizeSeparationPilotRequest.__post_init__ validation (lines 159-163).
# ---------------------------------------------------------------------------


def test_request_rejects_non_path_field() -> None:
    with pytest.raises(TypeError, match="run_path must be a Path"):
        FinalizeSeparationPilotRequest(
            run_path="not-a-path",  # type: ignore[arg-type]
            subset_manifest_path=Path("subset.jsonl"),
            output_manifest=Path("out.jsonl"),
            decision="keep_full_mix",
            rationale="x",
        )


def test_request_rejects_invalid_decision() -> None:
    with pytest.raises(ValueError, match="decision is invalid"):
        FinalizeSeparationPilotRequest(
            run_path=Path("run.json"),
            subset_manifest_path=Path("subset.jsonl"),
            output_manifest=Path("out.jsonl"),
            decision="bogus",  # type: ignore[arg-type]
            rationale="x",
        )


@pytest.mark.parametrize("rationale", ["", "   "])
def test_request_rejects_empty_rationale(rationale: str) -> None:
    with pytest.raises(ValueError, match="rationale must be nonempty"):
        FinalizeSeparationPilotRequest(
            run_path=Path("run.json"),
            subset_manifest_path=Path("subset.jsonl"),
            output_manifest=Path("out.jsonl"),
            decision="keep_full_mix",
            rationale=rationale,
        )


# ---------------------------------------------------------------------------
# Scalar validator unit tests (lines 186-258).
# ---------------------------------------------------------------------------


def test_text_rejects_non_string_and_empty() -> None:
    with pytest.raises(ValueError, match="must be a nonempty string"):
        _text(123, "field")
    with pytest.raises(ValueError, match="must be a nonempty string"):
        _text("", "field")


def test_hash_rejects_non_string_and_invalid_sha() -> None:
    with pytest.raises(ValueError, match="must be a lowercase SHA-256"):
        _hash(123, "field")
    with pytest.raises(ValueError, match="must be a lowercase SHA-256"):
        _hash("not-a-hash", "field")


def test_version_rejects_invalid_corpus_version() -> None:
    with pytest.raises(ValueError, match="must be a corpus version"):
        _version("not-a-version", "field")


def test_commit_rejects_invalid_commit() -> None:
    with pytest.raises(ValueError, match="crux_commit must be a lowercase 40-character commit"):
        _commit("short")


def test_path_rejects_invalid_paths() -> None:
    with pytest.raises(ValueError, match="path is invalid"):
        _path("", "field")
    with pytest.raises(ValueError, match="path is invalid"):
        _path("/absolute/path", "field")
    with pytest.raises(ValueError, match="path is invalid"):
        _path("../escape", "field")
    with pytest.raises(ValueError, match="path is invalid"):
        _path("a/../b", "field")


def test_owner_root_rejects_invalid_roots() -> None:
    with pytest.raises(ValueError, match="owner root is invalid"):
        _owner_root("", "field")
    with pytest.raises(ValueError, match="owner root is invalid"):
        _owner_root("relative/path", "field")


def test_positive_duration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="is invalid"):
        _positive_duration(True, "field")
    with pytest.raises(ValueError, match="is invalid"):
        _positive_duration("not-numeric", "field")
    with pytest.raises(ValueError, match="is invalid"):
        _positive_duration(0, "field")
    with pytest.raises(ValueError, match="is invalid"):
        _positive_duration(-1, "field")
    with pytest.raises(ValueError, match="is invalid"):
        _positive_duration(float("nan"), "field")
    with pytest.raises(ValueError, match="is invalid"):
        _positive_duration(float("inf"), "field")


def test_manifest_duration_returns_float_for_fractional() -> None:
    assert _manifest_duration(2, "field") == 2
    assert isinstance(_manifest_duration(2, "field"), int)
    result = _manifest_duration(1.5, "field")
    assert result == 1.5
    assert isinstance(result, float)


def test_manifest_json_value_normalizes_containers() -> None:
    from decimal import Decimal

    assert _manifest_json_value(Decimal("3")) == 3
    assert isinstance(_manifest_json_value(Decimal("3")), int)
    assert _manifest_json_value(Decimal("3.5")) == 3.5
    assert isinstance(_manifest_json_value(Decimal("3.5")), float)
    assert _manifest_json_value({"a": Decimal("1")}) == {"a": 1}
    assert _manifest_json_value([Decimal("1"), Decimal("2.5")]) == [1, 2.5]
    assert _manifest_json_value("str") == "str"


# ---------------------------------------------------------------------------
# Composite validator unit tests (lines 269-428).
# ---------------------------------------------------------------------------


def test_validate_stem_rejects_invalid_shape_and_identity() -> None:
    with pytest.raises(ValueError, match="stem evidence is invalid"):
        _validate_stem(
            "not-a-mapping",
            field="spleeter",
            source_audio_sha256=_SHA_A,
            separator_lock_sha256=_SHA_B,
        )
    bad_keys = dict(_valid_stem())
    bad_keys["extra"] = "x"
    with pytest.raises(ValueError, match="stem evidence is invalid"):
        _validate_stem(
            bad_keys,
            field="spleeter",
            source_audio_sha256=_SHA_A,
            separator_lock_sha256=_SHA_B,
        )
    mismatched_source = _valid_stem(source_audio_sha256=_SHA_C)
    with pytest.raises(ValueError, match="stem source identity does not match"):
        _validate_stem(
            mismatched_source,
            field="spleeter",
            source_audio_sha256=_SHA_A,
            separator_lock_sha256=_SHA_B,
        )
    mismatched_lock = _valid_stem(separator_lock_sha256=_SHA_C)
    with pytest.raises(ValueError, match="stem separator identity does not match"):
        _validate_stem(
            mismatched_lock,
            field="spleeter",
            source_audio_sha256=_SHA_A,
            separator_lock_sha256=_SHA_B,
        )


def test_validate_input_rejects_invalid_shape_and_identity() -> None:
    with pytest.raises(ValueError, match="input evidence is invalid"):
        _validate_input(
            "bad",
            field="spleeter",
            input_view_id=_SPLEETER_ID,
            source_audio_id="100/bgm.wav",
            source_audio_sha256=_SHA_A,
        )
    none_path = _valid_input("spleeter", view_id=_SPLEETER_ID)
    none_path["path"] = None
    with pytest.raises(ValueError, match="path is invalid"):
        _validate_input(
            none_path,
            field="spleeter",
            input_view_id=_SPLEETER_ID,
            source_audio_id="100/bgm.wav",
            source_audio_sha256=_SHA_A,
        )
    wrong_view = _valid_input("spleeter", view_id=_FULL_MIX_ID)
    with pytest.raises(ValueError, match="input view identity does not match"):
        _validate_input(
            wrong_view,
            field="spleeter",
            input_view_id=_SPLEETER_ID,
            source_audio_id="100/bgm.wav",
            source_audio_sha256=_SHA_A,
        )
    wrong_id = _valid_input("spleeter", view_id=_SPLEETER_ID, source_audio_id="999/bgm.wav")
    with pytest.raises(ValueError, match="input source ID does not match"):
        _validate_input(
            wrong_id,
            field="spleeter",
            input_view_id=_SPLEETER_ID,
            source_audio_id="100/bgm.wav",
            source_audio_sha256=_SHA_A,
        )
    wrong_sha = _valid_input("spleeter", view_id=_SPLEETER_ID, source_audio_sha256=_SHA_C)
    with pytest.raises(ValueError, match="input source hash does not match"):
        _validate_input(
            wrong_sha,
            field="spleeter",
            input_view_id=_SPLEETER_ID,
            source_audio_id="100/bgm.wav",
            source_audio_sha256=_SHA_A,
        )


def test_validate_prediction_rejects_invalid_shape_and_identity() -> None:
    with pytest.raises(ValueError, match="prediction evidence is invalid"):
        _validate_prediction(
            "bad",
            field="spleeter",
            input_view_id=_SPLEETER_ID,
            source_audio_id="100/bgm.wav",
            source_audio_sha256=_SHA_A,
            input_audio_sha256=_SHA_B,
        )
    wrong_view = _valid_prediction("spleeter", view_id=_FULL_MIX_ID)
    with pytest.raises(ValueError, match="prediction view identity does not match"):
        _validate_prediction(
            wrong_view,
            field="spleeter",
            input_view_id=_SPLEETER_ID,
            source_audio_id="100/bgm.wav",
            source_audio_sha256=_SHA_A,
            input_audio_sha256=_SHA_B,
        )
    wrong_id = _valid_prediction("spleeter", view_id=_SPLEETER_ID, source_audio_id="999/bgm.wav")
    with pytest.raises(ValueError, match="prediction source ID does not match"):
        _validate_prediction(
            wrong_id,
            field="spleeter",
            input_view_id=_SPLEETER_ID,
            source_audio_id="100/bgm.wav",
            source_audio_sha256=_SHA_A,
            input_audio_sha256=_SHA_B,
        )
    wrong_sha = _valid_prediction("spleeter", view_id=_SPLEETER_ID, source_audio_sha256=_SHA_C)
    with pytest.raises(ValueError, match="prediction source hash does not match"):
        _validate_prediction(
            wrong_sha,
            field="spleeter",
            input_view_id=_SPLEETER_ID,
            source_audio_id="100/bgm.wav",
            source_audio_sha256=_SHA_A,
            input_audio_sha256=_SHA_B,
        )
    wrong_input = _valid_prediction("spleeter", view_id=_SPLEETER_ID, input_audio_sha256=_SHA_C)
    with pytest.raises(ValueError, match="prediction input hash does not match"):
        _validate_prediction(
            wrong_input,
            field="spleeter",
            input_view_id=_SPLEETER_ID,
            source_audio_id="100/bgm.wav",
            source_audio_sha256=_SHA_A,
            input_audio_sha256=_SHA_B,
        )


def test_validate_view_rejects_invalid_key_set() -> None:
    bad = dict(_valid_view("spleeter"))
    del bad["status"]
    with pytest.raises(ValueError, match="view evidence is invalid"):
        _validate_view(
            bad, field="spleeter", source_audio_id="100/bgm.wav", source_audio_sha256=_SHA_A
        )


def test_validate_view_rejects_invalid_status() -> None:
    bad = dict(_valid_view("spleeter"))
    bad["status"] = "bogus"
    with pytest.raises(ValueError, match="status is invalid"):
        _validate_view(
            bad, field="spleeter", source_audio_id="100/bgm.wav", source_audio_sha256=_SHA_A
        )


def test_validate_view_rejects_wrong_input_view_id() -> None:
    bad = dict(_valid_view("spleeter"))
    bad["input_view_id"] = _FULL_MIX_ID
    with pytest.raises(ValueError, match="input view identity is invalid"):
        _validate_view(
            bad, field="spleeter", source_audio_id="100/bgm.wav", source_audio_sha256=_SHA_A
        )


def test_validate_view_rejects_full_mix_separator_lock() -> None:
    bad = dict(_valid_view("full_mix"))
    bad["separator_lock_sha256"] = _SHA_B
    with pytest.raises(ValueError, match="full_mix separator lock must be null"):
        _validate_view(
            bad, field="full_mix", source_audio_id="100/bgm.wav", source_audio_sha256=_SHA_A
        )


def test_validate_view_rejects_successful_full_mix_invalid_nullability() -> None:
    bad = dict(_valid_view("full_mix"))
    bad["stem"] = _valid_stem()
    with pytest.raises(ValueError, match="successful full_mix evidence has invalid nullability"):
        _validate_view(
            bad, field="full_mix", source_audio_id="100/bgm.wav", source_audio_sha256=_SHA_A
        )


def test_validate_view_rejects_non_successful_full_mix_with_evidence() -> None:
    bad = dict(_valid_view("full_mix", status="failed"))
    bad["prediction"] = _valid_prediction("full_mix", view_id=_FULL_MIX_ID)
    with pytest.raises(
        ValueError, match="non-successful full_mix evidence has invalid nullability"
    ):
        _validate_view(
            bad, field="full_mix", source_audio_id="100/bgm.wav", source_audio_sha256=_SHA_A
        )


def test_validate_view_rejects_successful_derived_missing_evidence() -> None:
    no_stem = dict(_valid_view("spleeter"))
    no_stem["stem"] = None
    with pytest.raises(
        ValueError, match="successful spleeter stem evidence has invalid nullability"
    ):
        _validate_view(
            no_stem, field="spleeter", source_audio_id="100/bgm.wav", source_audio_sha256=_SHA_A
        )
    no_input = dict(_valid_view("spleeter"))
    no_input["input"] = None
    with pytest.raises(
        ValueError, match="successful spleeter input evidence has invalid nullability"
    ):
        _validate_view(
            no_input, field="spleeter", source_audio_id="100/bgm.wav", source_audio_sha256=_SHA_A
        )


def test_validate_view_rejects_separation_failed_with_evidence() -> None:
    bad = dict(_valid_view("spleeter", status="separation_failed"))
    bad["stem"] = _valid_stem()
    with pytest.raises(
        ValueError, match="separation_failed spleeter evidence has invalid nullability"
    ):
        _validate_view(
            bad, field="spleeter", source_audio_id="100/bgm.wav", source_audio_sha256=_SHA_A
        )


def test_validate_view_rejects_inference_failed_with_prediction() -> None:
    bad = dict(_valid_view("spleeter", status="inference_failed"))
    bad["prediction"] = _valid_prediction("spleeter", view_id=_SPLEETER_ID)
    with pytest.raises(ValueError, match="failed spleeter prediction evidence must be null"):
        _validate_view(
            bad, field="spleeter", source_audio_id="100/bgm.wav", source_audio_sha256=_SHA_A
        )


def test_validate_comparison_artifacts_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="comparison_artifacts has an invalid key set"):
        _validate_comparison_artifacts("bad")
    incomplete = {"summary.json": {"path": "comparison/summary.json", "sha256": _SHA_A}}
    with pytest.raises(ValueError, match="comparison_artifacts has an invalid key set"):
        _validate_comparison_artifacts(incomplete)
    full = {
        key: {"path": f"comparison/{key}", "sha256": _SHA_A}
        for key in (
            "summary.json",
            "summary.md",
            "spleeter/paired_per_song.csv",
            "spleeter/paired_per_class.csv",
            "htdemucs/paired_per_song.csv",
            "htdemucs/paired_per_class.csv",
        )
    }
    bad_artifact = dict(full)
    bad_artifact["summary.json"] = {
        "path": "comparison/summary.json",
        "sha256": _SHA_A,
        "extra": "x",
    }
    with pytest.raises(ValueError, match="comparison_artifacts.summary.json is invalid"):
        _validate_comparison_artifacts(bad_artifact)
    bad_path = dict(full)
    bad_path["summary.json"] = {"path": "wrong/path.json", "sha256": _SHA_A}
    with pytest.raises(ValueError, match="comparison_artifacts.summary.json path is invalid"):
        _validate_comparison_artifacts(bad_path)


# ---------------------------------------------------------------------------
# _validate_row and _parse_manifest_content via validate_schema_golden
# (lines 451-540, 577).
# ---------------------------------------------------------------------------


def test_validate_row_rejects_invalid_key_set_with_corpus_version() -> None:
    row = _golden_row()
    row["extra_key"] = "bad"
    with pytest.raises(ValueError, match="invalid key set"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, _render_rows(row))


def test_validate_row_rejects_invalid_key_set_without_corpus_version() -> None:
    row = {k: v for k, v in _golden_row().items() if k != "corpus_version"}
    del row["decision"]
    with pytest.raises(ValueError, match="invalid key set"):
        _validate_row(row, allow_corpus_version=False)


def test_validate_row_rejects_unsupported_schema() -> None:
    row = _golden_row()
    row["schema_version"] = "crux.wrong/v1"
    with pytest.raises(ValueError, match="unsupported schema"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, _render_rows(row))


def test_validate_row_rejects_invalid_run_id() -> None:
    row = _golden_row()
    row["separation_run_id"] = "not-a-run-id"
    with pytest.raises(ValueError, match="separation_run_id is invalid"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, _render_rows(row))


@pytest.mark.parametrize("simfile_id", [0, -1, True])
def test_validate_row_rejects_invalid_simfile_id(simfile_id: object) -> None:
    row = _golden_row()
    row["simfile_id"] = simfile_id
    with pytest.raises(ValueError, match="simfile_id is invalid"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, _render_rows(row))


def test_validate_row_rejects_invalid_decision() -> None:
    row = _golden_row()
    row["decision"] = "bogus"
    with pytest.raises(ValueError, match="decision is invalid"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, _render_rows(row))


def test_validate_row_rejects_empty_rationale() -> None:
    row = _golden_row()
    row["rationale"] = "  "
    with pytest.raises(ValueError, match="rationale must be nonempty"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, _render_rows(row))


def test_parse_manifest_content_rejects_missing_trailing_newline() -> None:
    content = _render_rows(_golden_row()).rstrip(b"\n")
    with pytest.raises(ValueError, match="canonical JSONL records"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, content)


def test_parse_manifest_content_rejects_blank_lines() -> None:
    content = _render_rows(_golden_row()) + b"\n"
    with pytest.raises(ValueError, match="canonical JSONL records"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, content)


def test_parse_manifest_content_rejects_non_object_row() -> None:
    content = b"42\n"
    with pytest.raises(ValueError, match="rows must be objects"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, content)


def test_parse_manifest_content_rejects_mixed_corpus_versions() -> None:
    row_a = dict(_golden_row())
    row_a["corpus_version"] = "sha256:" + "0" * 64
    row_b = dict(_golden_row())
    row_b["corpus_version"] = "sha256:" + "1" * 64
    content = canonical_json_line(row_a) + canonical_json_line(row_b)
    with pytest.raises(ValueError, match="mixed corpus versions"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, content)


def test_parse_manifest_content_rejects_invalid_derived_corpus_version() -> None:
    row = dict(_golden_row())
    row["corpus_version"] = "sha256:" + "0" * 64
    content = canonical_json_line(row)
    with pytest.raises(ValueError, match="invalid derived corpus version"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, content)


def test_validate_schema_golden_rejects_unsupported_schema() -> None:
    with pytest.raises(ValueError, match="unsupported schema golden"):
        validate_schema_golden("crux.wrong/v1", _render_rows(_golden_row()))


# ---------------------------------------------------------------------------
# load_separation_pilot_manifest error paths (lines 584, 587-588).
# ---------------------------------------------------------------------------


def test_load_separation_pilot_manifest_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="path must be a Path"):
        load_separation_pilot_manifest("not-a-path")  # type: ignore[arg-type]


def test_load_separation_pilot_manifest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SeparationHandoffError, match="unavailable"):
        load_separation_pilot_manifest(tmp_path / "does-not-exist.jsonl")


# ---------------------------------------------------------------------------
# _recorded_artifact_path and _read_matching_artifact (lines 618-628, 653-654).
# ---------------------------------------------------------------------------


def test_recorded_artifact_path_rejects_missing_owner_root() -> None:
    with pytest.raises(ValueError, match="owner root is unavailable"):
        _recorded_artifact_path(
            "preds/x.json",
            field="full_mix.prediction",
            run_path=Path("/a/b/c/run.json"),
            owner_root=None,
        )


def test_recorded_artifact_path_rejects_short_prediction_run_path() -> None:
    with pytest.raises(ValueError, match="retained prediction owner root is unavailable"):
        _recorded_artifact_path(
            "preds/x.json", field="spleeter.prediction", run_path=Path("a/run.json")
        )


def test_recorded_artifact_path_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="has no retained artifact owner"):
        _recorded_artifact_path("x.wav", field="full_mix.stem", run_path=Path("/a/b/c/run.json"))


def test_read_matching_artifact_rejects_non_regular_file(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text("{}", encoding="utf-8")
    comparison_dir = tmp_path / "comparison" / "summary.json"
    comparison_dir.mkdir(parents=True)
    with pytest.raises(SeparationHandoffError, match="is unreadable"):
        _read_matching_artifact(
            "comparison/summary.json",
            _SHA_A,
            field="comparison_artifacts.summary.json",
            run_path=run_path,
            subset_path=tmp_path / "subset.jsonl",
        )


# ---------------------------------------------------------------------------
# finalize_separation_pilot top-level error paths (lines 1190, 1195-1196).
# ---------------------------------------------------------------------------


def test_finalize_rejects_non_request_type() -> None:
    outcome = finalize_separation_pilot("not-a-request")  # type: ignore[arg-type]
    assert outcome.exit_code == 2
    assert outcome.manifest is None


def test_finalize_rejects_corrupt_run_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    run_path.write_bytes(b"not json\n")
    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "corrupt-handoff" / "manifest.jsonl",
            decision="keep_full_mix",
            rationale="Corrupt run snapshot.",
        )
    )
    assert outcome.exit_code == 2
    assert outcome.manifest is None


# ---------------------------------------------------------------------------
# _build_rows error paths via finalize (lines 957-977, 1014-1046).
# ---------------------------------------------------------------------------


def test_build_rows_rejects_invalid_snapshot_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["schema"] = "crux.wrong/v1"
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_build_rows_rejects_unclosed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["overall_status"] = "running"
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_build_rows_rejects_invalid_subset_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    import src.benchmark.separation_handoff as handoff

    monkeypatch.setattr(handoff, "load_reviewed_subset_manifest", lambda _p: object())
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_build_rows_rejects_non_list_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["items"] = "not-a-list"
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_build_rows_rejects_membership_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["items"][0]["simfile_id"] = 99999
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_build_rows_rejects_subset_manifest_sha_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["reviewed_subset_manifest_sha256"] = "0" * 64
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_build_rows_rejects_invalid_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["run_id"] = "not-a-valid-run-id"
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_build_rows_rejects_non_mapping_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["items"][0] = "not-a-mapping"
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_build_rows_rejects_invalid_item_simfile_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["items"][0]["simfile_id"] = "not-an-int"
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_build_rows_rejects_source_row_sha_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["items"][0]["source_row_sha256"] = "0" * 64
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_build_rows_rejects_inconsistent_cache_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["items"][1]["htdemucs"]["stem"]["owner_root"] = str(tmp_path / "different-cache")
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


# ---------------------------------------------------------------------------
# _oaf_parent_identity error paths via finalize (lines 868-871, 875-880).
# ---------------------------------------------------------------------------


def test_oaf_parent_identity_rejects_missing_parent_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    parent_path = Path(snapshot["parent_oaf_run_path"])
    parent_path.unlink()
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_oaf_parent_identity_rejects_run_id_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["parent_oaf_run_id"] = "oaf-different1234567"
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_oaf_parent_identity_handles_non_mapping_inference_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    parent_path = Path(snapshot["parent_oaf_run_path"])
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    # prediction_map_version lives only inside inference_config; hoist it to
    # the top level so the non-Mapping inference_config fallback still resolves.
    parent["prediction_map_version"] = parent["inference_config"]["prediction_map_version"]
    parent["inference_config"] = "not-a-mapping"
    import src.benchmark.separation_handoff as handoff

    original_read = handoff.read_regular_file_no_follow

    def selective_read(path: Path) -> bytes:
        if path == parent_path:
            return canonical_json_line(parent).rstrip(b"\n")
        return original_read(path)

    monkeypatch.setattr(handoff, "read_regular_file_no_follow", selective_read)
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 0


def test_oaf_parent_identity_rejects_shallow_parent_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    parent_path = Path(snapshot["parent_oaf_run_path"])
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    short_parent = {"run_id": parent["run_id"], "schema": parent.get("schema", "s")}
    snapshot["parent_oaf_run_path"] = "/p/run.json"
    import src.benchmark.separation_handoff as handoff

    original_read = handoff.read_regular_file_no_follow

    def selective_read(path: Path) -> bytes:
        if str(path) == "/p/run.json":
            return canonical_json_line(short_parent)
        return original_read(path)

    monkeypatch.setattr(handoff, "read_regular_file_no_follow", selective_read)
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


# ---------------------------------------------------------------------------
# _validate_view_oaf_identity error paths via finalize (lines 928-941).
# ---------------------------------------------------------------------------


def test_validate_view_oaf_identity_rejects_non_mapping_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["items"][0]["spleeter"] = "not-a-mapping"
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_validate_view_oaf_identity_rejects_non_mapping_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["items"][0]["spleeter"]["inference_config"] = "not-a-mapping"
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_validate_view_oaf_identity_rejects_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["items"][0]["spleeter"]["inference_config"]["backend_descriptor_sha256"] = "e" * 64
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


# ---------------------------------------------------------------------------
# _view_payload error paths via finalize (lines 793, 808, 829).
# ---------------------------------------------------------------------------


def test_view_payload_rejects_prediction_without_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["items"][0]["spleeter"]["input"] = None
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_view_payload_rejects_full_mix_with_stem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    htdemucs_stem = dict(snapshot["items"][0]["htdemucs"]["stem"])
    snapshot["items"][0]["full_mix"]["stem"] = htdemucs_stem
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


def test_view_payload_rejects_pending_derived_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    snapshot["items"][0]["spleeter"]["status"] = "pending"
    outcome = _finalize_with_snapshot(tmp_path, monkeypatch, snapshot, subset_path, run_path)
    assert outcome.exit_code == 2


# ---------------------------------------------------------------------------
# _comparison_payload error path via finalize (lines 849-850).
# ---------------------------------------------------------------------------


def test_comparison_payload_rejects_missing_comparison_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    comparison_file = run_path.parent / "comparison" / "summary.json"
    comparison_file.unlink()
    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "missing-comparison-handoff" / "manifest.jsonl",
            decision="use_htdemucs",
            rationale="Missing comparison artifact.",
        )
    )
    assert outcome.exit_code == 2


# ---------------------------------------------------------------------------
# _prediction_payload error paths via finalize (lines 684-685, 693).
# ---------------------------------------------------------------------------


def test_prediction_payload_rejects_invalid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    relative = snapshot["items"][0]["htdemucs"]["prediction"]["path"]
    prediction_path = request.output_dir / relative
    prediction_path.write_bytes(b"not a prediction artifact\n")
    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "bad-prediction-handoff" / "manifest.jsonl",
            decision="use_htdemucs",
            rationale="Invalid prediction artifact.",
        )
    )
    assert outcome.exit_code == 2


def test_prediction_payload_rejects_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    rel_a = snapshot["items"][0]["htdemucs"]["prediction"]["path"]
    rel_b = snapshot["items"][1]["htdemucs"]["prediction"]["path"]
    path_a = request.output_dir / rel_a
    path_b = request.output_dir / rel_b
    content_a = path_a.read_bytes()
    content_b = path_b.read_bytes()
    path_a.write_bytes(content_b)
    path_b.write_bytes(content_a)
    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "identity-mismatch-handoff" / "manifest.jsonl",
            decision="use_htdemucs",
            rationale="Swapped prediction identity.",
        )
    )
    assert outcome.exit_code == 2


# ---------------------------------------------------------------------------
# _rehash_rows continue path for non-success derived view (line 1120).
# ---------------------------------------------------------------------------


def test_rehash_skips_non_success_derived_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    spleeter = dict(snapshot["items"][0]["spleeter"])
    spleeter["status"] = "separation_failed"
    spleeter["failure_code"] = "separation_failed"
    spleeter["stem"] = None
    spleeter["input"] = None
    spleeter["prediction"] = None
    snapshot["items"][0]["spleeter"] = spleeter
    from src.benchmark.separation_pilot import write_oaf_separation_run

    write_oaf_separation_run(run_path, snapshot)
    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "rehash-skip-handoff" / "manifest.jsonl",
            decision="use_htdemucs",
            rationale="Non-success spleeter is skipped in rehash.",
        )
    )
    assert outcome.exit_code == 0
    assert outcome.manifest is not None


# ---------------------------------------------------------------------------
# _direct_manifest_alias paths (lines 1169-1181).
# ---------------------------------------------------------------------------


def test_direct_manifest_alias_returns_when_paths_match(tmp_path: Path) -> None:
    from types import SimpleNamespace

    output = tmp_path / "out.jsonl"
    published = SimpleNamespace(path=output)
    _direct_manifest_alias(output, published, b"content\n")
    assert output == published.path


def test_direct_manifest_alias_rejects_existing_different_bytes(tmp_path: Path) -> None:
    from types import SimpleNamespace

    output = tmp_path / "alias.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"different\n")
    published = SimpleNamespace(path=tmp_path / "published.jsonl")
    with pytest.raises(SeparationHandoffError, match="different bytes"):
        _direct_manifest_alias(output, published, b"content\n")


def test_direct_manifest_alias_rejects_unreadable_existing(tmp_path: Path) -> None:
    from types import SimpleNamespace

    output = tmp_path / "alias_dir"
    output.mkdir(parents=True)
    published = SimpleNamespace(path=tmp_path / "published.jsonl")
    with pytest.raises(SeparationHandoffError, match="unavailable"):
        _direct_manifest_alias(output, published, b"content\n")


def test_direct_manifest_alias_reuses_matching_existing_bytes(tmp_path: Path) -> None:
    from types import SimpleNamespace

    output = tmp_path / "alias.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    content = b"matching\n"
    output.write_bytes(content)
    published = SimpleNamespace(path=tmp_path / "published.jsonl")
    _direct_manifest_alias(output, published, content)
    assert output.read_bytes() == content


def test_direct_manifest_alias_rejects_link_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import src.benchmark.separation_handoff as handoff

    output = tmp_path / "alias.jsonl"
    published = SimpleNamespace(path=tmp_path / "published.jsonl")
    (tmp_path / "published.jsonl").write_bytes(b"content\n")
    monkeypatch.setattr(handoff.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(SeparationHandoffError, match="alias could not be published"):
        _direct_manifest_alias(output, published, b"content\n")


def test_finalize_reuses_existing_output_manifest_with_matching_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    output_manifest = tmp_path / "reuse-handoff" / "manifest.jsonl"
    first = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=output_manifest,
            decision="use_htdemucs",
            rationale="Reuse finalization.",
        )
    )
    assert first.exit_code == 0
    second = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=output_manifest,
            decision="use_htdemucs",
            rationale="Reuse finalization.",
        )
    )
    assert second.exit_code == 0


# ---------------------------------------------------------------------------
# Additional _parse_manifest_content and load_separation_pilot_manifest paths
# (lines 523, 537, 540).
# ---------------------------------------------------------------------------


def test_parse_manifest_content_rejects_non_canonical_json() -> None:
    content = b'{"b": 1, "a": 2}\n'
    with pytest.raises(ValueError, match="not canonical JSONL"):
        validate_schema_golden(SEPARATION_PILOT_SCHEMA, content)


def test_load_rejects_manifest_with_wrong_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "pop-handoff" / "manifest.jsonl",
            decision="use_htdemucs",
            rationale="Population check.",
        )
    )
    assert outcome.exit_code == 0
    rows = [
        strict_json_loads(line, require_canonical=True)
        for line in outcome.manifest.path.read_bytes().splitlines()
    ]
    trimmed = tuple({k: v for k, v in row.items() if k != "corpus_version"} for row in rows[:19])
    small_manifest = tmp_path / "small.jsonl"
    small_manifest.write_bytes(render_manifest(trimmed).content)
    with pytest.raises(SeparationHandoffError, match="exact HPA-327 population"):
        load_separation_pilot_manifest(small_manifest)


def test_load_rejects_manifest_with_unsorted_simfile_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, subset_path, run_path = _successful_pilot(tmp_path, monkeypatch)
    outcome = finalize_separation_pilot(
        FinalizeSeparationPilotRequest(
            run_path=run_path,
            subset_manifest_path=subset_path,
            output_manifest=tmp_path / "unsorted-handoff" / "manifest.jsonl",
            decision="use_htdemucs",
            rationale="Unsorted check.",
        )
    )
    assert outcome.exit_code == 0
    rows = [
        strict_json_loads(line, require_canonical=True)
        for line in outcome.manifest.path.read_bytes().splitlines()
    ]
    rows[0], rows[1] = rows[1], rows[0]
    swapped = tuple({k: v for k, v in row.items() if k != "corpus_version"} for row in rows)
    unsorted_manifest = tmp_path / "unsorted.jsonl"
    unsorted_manifest.write_bytes(render_manifest(swapped).content)
    with pytest.raises(SeparationHandoffError, match="unique and sorted"):
        load_separation_pilot_manifest(unsorted_manifest)
