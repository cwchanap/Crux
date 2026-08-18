"""HPA-328 fixed-subset preflight and snapshot tests."""

# Fixtures intentionally import the production seam inside tests so the
# baseline collection remains free of the optional runtime modules.
# pylint: disable=import-outside-toplevel,too-many-locals,duplicate-code,too-many-arguments
# pylint: disable=too-many-lines

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path

import pytest

from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.oaf_corpus_run import OAF_FULL_MIX_INPUT_VIEW_ID, parse_oaf_corpus_run
from src.benchmark.reference_set_manifest import load_reference_set_manifest
from src.benchmark.reference_timing_manifest import load_reference_timing_manifest
from src.benchmark.reviewed_subset import (
    ScoreReviewedSubsetOutcome,
    _source_row_sha256,
    load_reviewed_subset_manifest,
)
from src.benchmark.separation_pilot import HTDEMUCS_INPUT_VIEW_ID, SPLEETER_INPUT_VIEW_ID
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
                "source_row_sha256": _source_row_sha256(loaded.source_row),
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


def _runtime_sentinels() -> dict[str, object]:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fatal preflight touched an execution seam")

    return {
        "backend_factory": explode,
        "spleeter_runner": explode,
        "htdemucs_runner": explode,
        "perf_counter": explode,
    }


def _run_with_runtime_sentinels(run: object, request: object) -> object:
    return run(request, **_runtime_sentinels())  # type: ignore[operator]


def _task6_seams(
    tmp_path: Path,
    fixture: object,
    monkeypatch: pytest.MonkeyPatch,
    *,
    spleeter_error: Exception | None = None,
) -> dict[str, list[object]]:
    """Install fake source, separator, materializer, scorer, and OaF seams."""
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.backend_identity import BackendDescriptor
    from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
    from src.benchmark.corpus_cache import ResolvedSourceAudio
    from src.benchmark.separators import SeparatedStem, StemQc

    parent = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    descriptor = BackendDescriptor(
        payload=parent["backend_descriptor"],  # type: ignore[arg-type]
        sha256=parent["backend_descriptor_sha256"],  # type: ignore[arg-type]
    )
    calls: dict[str, list[object]] = {
        "resolve": [],
        "separate": [],
        "materialize": [],
        "backend": [],
        "close": [],
        "transcribe": [],
        "score": [],
    }

    def resolve(source: object, *_args: object, **kwargs: object) -> ResolvedSourceAudio:
        calls["resolve"].append(kwargs.get("load_body"))
        assert isinstance(source, Mapping)
        source_id = source["source_audio_key"]
        source_sha = source["source_audio_content_hash"]
        return ResolvedSourceAudio(
            path=tmp_path / "authoritative-source.wav",
            source_audio_id=source_id,
            source_audio_sha256=source_sha,
            duration_sec=1.0,
        )

    def stem(separator_id: str, source_sha: str) -> SeparatedStem:
        stem_path = (
            tmp_path
            / "cache"
            / "derived"
            / "stems"
            / separator_id
            / source_sha
            / ("spleeter-lock" if separator_id == SPLEETER_SEPARATOR_ID else "htdemucs-lock")
            / "drums.wav"
        )
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        stem_bytes = f"native-{separator_id}-{source_sha}".encode()
        stem_path.write_bytes(stem_bytes)
        return SeparatedStem(
            separator_id=separator_id,
            source_audio_sha256=source_sha,
            separator_lock_sha256=(
                "84b478def61f9c78320e76a7a49afeb341a22dc5e02dd1b0c666bb8cc3667a95"
                if separator_id == SPLEETER_SEPARATOR_ID
                else "7db5a9bfb64d7ec4b2f55e134ddd720cf6e64b1c97343a50fd7dce1f0b81b5e0"
            ),
            path=stem_path,
            sha256=hashlib.sha256(stem_bytes).hexdigest(),
            qc=StemQc(
                sample_rate=44100,
                frame_count=44100,
                channel_count=1,
                duration_sec=1.0,
                rms_dbfs=-20.0,
                peak_abs=0.2,
                clipping_detected=False,
            ),
            cache_hit=False,
        )

    def spleeter(source: Path, **kwargs: object) -> SeparatedStem:
        del source
        calls["separate"].append(SPLEETER_INPUT_VIEW_ID)
        if spleeter_error is not None:
            raise spleeter_error
        return stem(SPLEETER_SEPARATOR_ID, kwargs["source_audio_sha256"])  # type: ignore[arg-type]

    def htdemucs(source: Path, **kwargs: object) -> SeparatedStem:
        del source
        calls["separate"].append(HTDEMUCS_INPUT_VIEW_ID)
        return stem(HTDEMUCS_SEPARATOR_ID, kwargs["source_audio_sha256"])  # type: ignore[arg-type]

    def materialize(
        source: ResolvedSourceAudio,
        native_stem: Path,
        output_path: Path,
        *,
        input_root: Path,
        input_view_id: str,
        max_input_audio_frames: int | None,
    ) -> CanonicalAudio:
        del input_root, native_stem, max_input_audio_frames
        calls["materialize"].append(input_view_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"temporary-canonical-input")
        input_sha = "c" * 64 if input_view_id == SPLEETER_INPUT_VIEW_ID else "d" * 64
        return CanonicalAudio(
            path=output_path,
            source_audio_id=source.source_audio_id,
            source_audio_sha256=source.source_audio_sha256,
            input_view_id=input_view_id,
            input_audio_sha256=input_sha,
            byte_length=88244,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=44100,
        )

    def fake_score(score_request: object) -> ScoreReviewedSubsetOutcome:
        calls["score"].append(score_request)
        score_request.output_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        return ScoreReviewedSubsetOutcome(
            exit_code=0,
            cohort_id="subset-cohort",
            reports_path=score_request.output_dir,  # type: ignore[attr-defined]
            success_count=20,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
        )

    class FakeBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            calls["transcribe"].append(audio.input_view_id)
            return NativePrediction(
                audio=audio,
                descriptor=descriptor,
                events=(
                    NativeEvent(
                        time_sec=0.25,
                        native_class_id="midi_36",
                        model_output_bin=15,
                        native_midi_note=36,
                        native_metadata={"upstream_8hit_group_id": "kick"},
                        confidence=0.9,
                        velocity_midi=100,
                    ),
                ),
            )

        def close(self) -> None:
            calls["close"].append(True)

    def backend_factory(**_kwargs: object) -> FakeBackend:
        calls["backend"].append(True)
        return FakeBackend()

    monkeypatch.setattr(pilot, "resolve_source_audio", resolve, raising=False)
    monkeypatch.setattr(pilot, "run_spleeter_drums", spleeter, raising=False)
    monkeypatch.setattr(pilot, "run_htdemucs_drums", htdemucs, raising=False)
    monkeypatch.setattr(pilot, "materialize_derived_audio", materialize, raising=False)
    monkeypatch.setattr(pilot, "score_oaf_reviewed_subset", fake_score)
    monkeypatch.setattr(
        pilot,
        "SEPARATOR_LOCK_PATHS",
        {
            SPLEETER_SEPARATOR_ID: FIXTURE_ROOT / "spleeter-model.json",
            HTDEMUCS_SEPARATOR_ID: FIXTURE_ROOT / "htdemucs-model.json",
        },
    )
    calls["factory"] = [backend_factory]
    return calls


def test_task6_infers_only_the_two_derived_views_after_resolving_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    ticks = iter(float(index) for index in range(200))
    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
        perf_counter=lambda: next(ticks),
    )

    assert outcome.exit_code == 0
    assert len(calls["resolve"]) == 20
    assert set(calls["resolve"]) == {False}
    assert set(calls["separate"]) == {SPLEETER_INPUT_VIEW_ID, HTDEMUCS_INPUT_VIEW_ID}
    assert set(calls["materialize"]) == {SPLEETER_INPUT_VIEW_ID, HTDEMUCS_INPUT_VIEW_ID}
    assert set(calls["transcribe"]) == {SPLEETER_INPUT_VIEW_ID, HTDEMUCS_INPUT_VIEW_ID}
    assert all(view_id != OAF_FULL_MIX_INPUT_VIEW_ID for view_id in calls["transcribe"])
    assert len(calls["score"]) == 1

    assert outcome.run_path is not None
    snapshot = json.loads(outcome.run_path.read_text(encoding="utf-8"))
    assert all(item["spleeter"]["status"] == "success" for item in snapshot["items"])
    assert all(item["htdemucs"]["status"] == "success" for item in snapshot["items"])
    parent = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    full_config = parent["inference_config"]
    for item in snapshot["items"]:
        for view_id, view_name in (
            (SPLEETER_INPUT_VIEW_ID, "spleeter"),
            (HTDEMUCS_INPUT_VIEW_ID, "htdemucs"),
        ):
            view = item[view_name]
            assert view["input_view_id"] == view_id
            assert view["input"]["input_view_id"] == view_id
            assert view["input"]["input_audio_sha256"] in {"c" * 64, "d" * 64}
            assert view["inference_config"] == {**full_config, "input_view_id": view_id}


def test_task6_default_backend_factory_is_bound_to_parent_frozen_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    import src.benchmark.separation_pilot as pilot

    parent = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    captured: list[dict[str, object]] = []
    fake_default_factory = calls["factory"][0]

    def default_factory(**kwargs: object) -> object:
        captured.append(kwargs)
        return fake_default_factory(**kwargs)  # type: ignore[operator]

    monkeypatch.setattr(pilot, "create_backend", default_factory)

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(request)

    assert outcome.exit_code == 0
    assert captured
    backend_kwargs = captured[0]
    checkpoint_root = Path(
        os.environ.get("CRUX_OAF_CHECKPOINT_CACHE", "artifacts/benchmark/model-cache")
    )
    assert backend_kwargs["checkpoint_dir"] == (
        checkpoint_root / "sha256" / parent["checkpoint_archive_sha256"]
    )
    assert (
        getattr(backend_kwargs["descriptor"], "sha256", None) == parent["backend_descriptor_sha256"]
    )


def test_task6_default_backend_factory_refuses_drifted_parent_model_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    import src.benchmark.separation_pilot as pilot

    drifted_lock = tmp_path / "drifted-model.json"
    drifted_lock.write_bytes(b"drifted model lock")
    monkeypatch.setattr(pilot, "_model_lock_path", lambda: drifted_lock, raising=False)
    monkeypatch.setattr(pilot, "create_backend", calls["factory"][0])

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(request)

    assert outcome.exit_code == 2
    assert not calls["transcribe"]


def test_task6_resume_exact_stem_and_prediction_skips_separator_and_oaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    first = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )
    assert first.exit_code == 0
    calls["separate"].clear()
    calls["transcribe"].clear()
    second = run_oaf_separation_pilot(
        replace(request, resume=True),
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert second.exit_code == 0
    assert not calls["separate"]
    assert not calls["transcribe"]
    assert second.run_path is not None
    snapshot = json.loads(second.run_path.read_text(encoding="utf-8"))
    assert {item["spleeter"]["status"] for item in snapshot["items"]} == {"resumed"}
    assert {item["htdemucs"]["status"] for item in snapshot["items"]} == {"resumed"}


def test_task6_resume_recovers_ledger_after_crash_between_snapshot_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    first = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )
    assert first.exit_code == 0
    assert first.run_path is not None

    def crash_after_initial_checkpoint(_request: object) -> object:
        raise RuntimeError("crash between durable boundaries")

    monkeypatch.setattr(pilot, "score_oaf_reviewed_subset", crash_after_initial_checkpoint)
    crashed = run_oaf_separation_pilot(
        replace(request, resume=True),
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )
    assert crashed.exit_code == 2
    interrupted = json.loads(first.run_path.read_text(encoding="utf-8"))
    assert all(item["spleeter"]["prediction"] is not None for item in interrupted["items"])
    assert all(item["htdemucs"]["prediction"] is not None for item in interrupted["items"])

    def resume_score(score_request: object) -> ScoreReviewedSubsetOutcome:
        score_request.output_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        return ScoreReviewedSubsetOutcome(
            exit_code=0,
            cohort_id="subset-cohort",
            reports_path=score_request.output_dir,  # type: ignore[attr-defined]
            success_count=20,
            failed_count=0,
            skipped_count=0,
            quarantined_count=0,
        )

    monkeypatch.setattr(pilot, "score_oaf_reviewed_subset", resume_score)
    calls["separate"].clear()
    calls["transcribe"].clear()
    resumed = run_oaf_separation_pilot(
        replace(request, resume=True),
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert resumed.exit_code == 0
    assert not calls["separate"]
    assert not calls["transcribe"]
    assert resumed.run_path is not None
    recovered = json.loads(resumed.run_path.read_text(encoding="utf-8"))
    assert {item["spleeter"]["status"] for item in recovered["items"]} == {"resumed"}
    assert {item["htdemucs"]["status"] for item in recovered["items"]} == {"resumed"}


def test_task6_resume_valid_stem_without_prediction_reinfers_only_missing_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    from src.benchmark.separation_pilot import run_oaf_separation_pilot, write_oaf_separation_run

    first = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )
    assert first.run_path is not None
    snapshot = json.loads(first.run_path.read_text(encoding="utf-8"))
    spleeter_view = snapshot["items"][0]["spleeter"]
    spleeter_view["runtime"]["separator_wall_time_sec"] = 12.345
    spleeter_view["runtime"]["separator_rtf"] = 12.345
    write_oaf_separation_run(first.run_path, snapshot)
    target = spleeter_view["prediction"]["path"]
    (request.output_dir / target).unlink()
    calls["separate"].clear()
    calls["transcribe"].clear()

    second = run_oaf_separation_pilot(
        replace(request, resume=True),
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert second.exit_code == 0
    assert not calls["separate"]
    assert calls["transcribe"] == [SPLEETER_INPUT_VIEW_ID]
    assert second.run_path is not None
    resumed = json.loads(second.run_path.read_text(encoding="utf-8"))
    assert resumed["items"][0]["spleeter"]["status"] == "success"
    assert resumed["items"][0]["spleeter"]["runtime"]["separator_wall_time_sec"] == 12.345
    assert resumed["items"][0]["spleeter"]["runtime"]["separator_rtf"] == 12.345
    assert all(item["htdemucs"]["status"] == "resumed" for item in resumed["items"])


def test_task6_closes_backend_after_scoring_raises_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    events: list[str] = []

    import src.benchmark.separation_pilot as pilot

    def fail_scoring(*_args: object, **_kwargs: object) -> None:
        events.append("score")
        raise RuntimeError("scoring failed unexpectedly")

    monkeypatch.setattr(pilot, "_score_derived_cohort", fail_scoring)
    original_close = calls["close"]

    def record_close() -> None:
        events.append("close")
        original_close.append(True)

    # Replace the fixture factory only to observe close ordering without
    # changing the backend behavior under test.
    base_factory = calls["factory"][0]

    def backend_factory(**kwargs: object) -> object:
        backend = base_factory(**kwargs)  # type: ignore[operator]
        backend.close = record_close  # type: ignore[attr-defined]
        return backend

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(request, backend_factory=backend_factory)

    assert outcome.exit_code == 2
    assert events == ["score", "close"]


def test_task6_does_not_retry_backend_close_after_unexpected_close_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    events: list[str] = []

    import src.benchmark.backends.oaf as oaf_backend

    base_factory = calls["factory"][0]

    def backend_factory(**kwargs: object) -> object:
        backend = base_factory(**kwargs)  # type: ignore[operator]

        def fail_transcribe(_audio: object) -> None:
            raise oaf_backend.OafBackendError("worker failed", code="worker_error")

        def fail_close() -> None:
            events.append("close")
            raise LookupError("unexpected close failure")

        backend.transcribe = fail_transcribe  # type: ignore[attr-defined]
        backend.close = fail_close  # type: ignore[attr-defined]
        return backend

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    with pytest.raises(LookupError, match="unexpected close failure"):
        run_oaf_separation_pilot(request, backend_factory=backend_factory)

    assert events == ["close"]


@pytest.mark.parametrize(
    ("mutation", "expected_status", "expected_code"),
    [
        ("stem_hash", "stem_invalid", "stem_identity_invalid"),
        ("run_row_identity", "prediction_invalid", "prediction_output_conflict"),
    ],
)
def test_task6_resume_rejects_immutable_identity_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_status: str,
    expected_code: str,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    from src.benchmark.separation_pilot import run_oaf_separation_pilot, write_oaf_separation_run

    first = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )
    assert first.run_path is not None
    snapshot = json.loads(first.run_path.read_text(encoding="utf-8"))
    view = snapshot["items"][0]["spleeter"]
    if mutation == "stem_hash":
        view["stem"]["sha256"] = "e" * 64
    else:
        view["input"]["input_audio_sha256"] = "e" * 64
    write_oaf_separation_run(first.run_path, snapshot)

    second = run_oaf_separation_pilot(
        replace(request, resume=True),
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert second.exit_code == 1
    assert second.run_path is not None
    resumed = json.loads(second.run_path.read_text(encoding="utf-8"))
    assert resumed["items"][0]["spleeter"]["status"] == expected_status
    assert resumed["items"][0]["spleeter"]["failure_code"] == expected_code


def test_task6_resume_rejects_current_audio_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    from src.benchmark.prediction_artifact import (
        read_prediction_artifact,
        render_prediction_artifact,
    )
    from src.benchmark.separation_pilot import (
        run_oaf_separation_pilot,
        write_oaf_separation_run,
    )

    first = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )
    assert first.run_path is not None
    snapshot = json.loads(first.run_path.read_text(encoding="utf-8"))
    view = snapshot["items"][0]["spleeter"]
    target = request.output_dir / view["prediction"]["path"]
    artifact = read_prediction_artifact(target.read_bytes())
    mismatched_audio = replace(artifact.prediction.audio, input_audio_sha256="e" * 64)
    mismatched_prediction = replace(artifact.prediction, audio=mismatched_audio)
    content = render_prediction_artifact(mismatched_prediction)
    target.write_bytes(content)
    view["input"]["input_audio_sha256"] = "e" * 64
    view["prediction"]["artifact_sha256"] = hashlib.sha256(content).hexdigest()
    write_oaf_separation_run(first.run_path, snapshot)

    second = run_oaf_separation_pilot(
        replace(request, resume=True),
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert second.exit_code == 1
    assert second.run_path is not None
    resumed = json.loads(second.run_path.read_text(encoding="utf-8"))
    assert resumed["items"][0]["spleeter"]["status"] == "prediction_invalid"
    assert resumed["items"][0]["spleeter"]["failure_code"] == "prediction_artifact_invalid"


def test_task6_resume_detects_immutable_prediction_conflict_before_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    first = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )
    assert first.run_path is not None
    snapshot = json.loads(first.run_path.read_text(encoding="utf-8"))
    target = request.output_dir / snapshot["items"][0]["spleeter"]["prediction"]["path"]
    target.write_bytes(b"different immutable prediction bytes\n")

    second = run_oaf_separation_pilot(
        replace(request, resume=True),
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert second.exit_code == 1
    assert second.run_path is not None
    resumed = json.loads(second.run_path.read_text(encoding="utf-8"))
    assert resumed["items"][0]["spleeter"]["status"] == "prediction_invalid"
    assert resumed["items"][0]["spleeter"]["failure_code"] == "prediction_output_conflict"


def test_task6_source_resolution_failure_is_fatal_before_control_or_runtimes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    import src.benchmark.separation_pilot as pilot

    def fail_source(*_args: object, **_kwargs: object) -> object:
        raise ValueError("authoritative source unavailable")

    monkeypatch.setattr(pilot, "resolve_source_audio", fail_source)
    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert outcome.exit_code == 2
    assert not calls["score"]
    assert not calls["separate"]
    assert not calls["transcribe"]


@pytest.mark.parametrize(
    ("backend_code", "expected_exit_code"),
    (("descriptor_invalid", 2), ("worker_error", 1)),
)
def test_task6_honors_fatal_and_poison_backend_dispositions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_code: str,
    expected_exit_code: int,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.backend_identity import BackendDescriptor
    from src.benchmark.backends.oaf import OafBackendError

    parent = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    descriptor = BackendDescriptor(
        payload=parent["backend_descriptor"],  # type: ignore[arg-type]
        sha256=parent["backend_descriptor_sha256"],  # type: ignore[arg-type]
    )

    class ErrorBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: object) -> object:
            calls["transcribe"].append(audio.input_view_id)  # type: ignore[attr-defined]
            raise OafBackendError("injected backend disposition", code=backend_code)

        def close(self) -> None:
            return None

    def error_factory(**_kwargs: object) -> ErrorBackend:
        calls["backend"].append(True)
        return ErrorBackend()

    monkeypatch.setattr(pilot, "create_backend", error_factory)

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = run_oaf_separation_pilot(request)

    assert outcome.exit_code == expected_exit_code
    assert calls["transcribe"] == [SPLEETER_INPUT_VIEW_ID]
    assert calls["separate"] == [SPLEETER_INPUT_VIEW_ID]
    if backend_code == "worker_error":
        assert outcome.run_path is not None
        snapshot = json.loads(outcome.run_path.read_text(encoding="utf-8"))
        for item in snapshot["items"]:
            for view_name in ("spleeter", "htdemucs"):
                view = item[view_name]
                assert view["status"] != "pending"
                if (
                    view_name == "htdemucs"
                    or item["simfile_id"] != snapshot["items"][0]["simfile_id"]
                ):
                    assert view["failure_code"] == "worker_protocol_failed"


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

    outcome = _run_with_runtime_sentinels(
        run_oaf_separation_pilot,
        replace(request, subset_manifest_path=broken),
    )

    assert outcome.exit_code == 2  # type: ignore[attr-defined]
    assert outcome.run_path is None  # type: ignore[attr-defined]
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
        _run_with_runtime_sentinels(
            run_oaf_separation_pilot,
            replace(request, subset_manifest_path=missing_member),
        ).exit_code
        == 2  # type: ignore[attr-defined]
    )

    _install_fixture_locks(monkeypatch)
    monkeypatch.setattr(
        "src.benchmark.separation_pilot.SEPARATOR_LOCK_PATHS",
        {
            SPLEETER_SEPARATOR_ID: tmp_path / "missing-spleeter-lock.json",
            HTDEMUCS_SEPARATOR_ID: FIXTURE_ROOT / "htdemucs-model.json",
        },
    )
    outcome = _run_with_runtime_sentinels(run_oaf_separation_pilot, request)
    assert outcome.exit_code == 2  # type: ignore[attr-defined]

    _install_fixture_locks(monkeypatch)
    aliased_output = fixture.run_path.parent / "reports"
    assert (
        _run_with_runtime_sentinels(
            run_oaf_separation_pilot,
            replace(request, output_dir=aliased_output),
        ).exit_code
        == 2  # type: ignore[attr-defined]
    )
    assert not calls


def test_exact_subset_is_sorted_identity_bound_and_full_mix_uses_public_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
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

    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

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
        "parent_oaf_run_path": str(request.oaf_run_path.resolve()),
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

    outcome = _run_with_runtime_sentinels(run_oaf_separation_pilot, request)
    assert outcome.exit_code == 2  # type: ignore[attr-defined]


def test_forged_parent_run_id_is_fatal_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    parent = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    parent["run_id"] = "oaf-forged-parent-id"
    from src.benchmark.oaf_corpus_run import write_oaf_corpus_run

    write_oaf_corpus_run(fixture.run_path, parent)
    monkeypatch.setattr(
        "src.benchmark.separation_pilot.score_oaf_reviewed_subset",
        lambda _request: pytest.fail("forged parent identity must fail before scoring"),
    )

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = _run_with_runtime_sentinels(run_oaf_separation_pilot, request)
    assert outcome.exit_code == 2  # type: ignore[attr-defined]


@pytest.mark.parametrize("lineage", ["subset_source_row", "parent_source_audio"])
def test_source_lineage_mismatch_is_fatal_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage: str,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    monkeypatch.setattr(
        "src.benchmark.separation_pilot.score_oaf_reviewed_subset",
        lambda _request: pytest.fail("source lineage mismatch must fail before scoring"),
    )

    if lineage == "subset_source_row":
        rows = [
            strict_json_loads(line[:-1], require_canonical=True)
            for line in subset.read_bytes().splitlines(keepends=True)
        ]
        assert isinstance(rows[0], dict)
        rows[0]["source_row_sha256"] = "e" * 64
        for row in rows:
            assert isinstance(row, dict)
            row.pop("corpus_version", None)
        foreign = tmp_path / "foreign-source-row.jsonl"
        foreign.write_bytes(render_manifest(tuple(rows)).content)
        request = replace(request, subset_manifest_path=foreign)
    else:
        parent = parse_oaf_corpus_run(fixture.run_path.read_bytes())
        assert isinstance(parent["items"], list)
        assert isinstance(parent["items"][0], dict)
        parent["items"][0]["source_audio_sha256"] = "e" * 64
        from src.benchmark.oaf_corpus_run import write_oaf_corpus_run

        write_oaf_corpus_run(fixture.run_path, parent)

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    outcome = _run_with_runtime_sentinels(run_oaf_separation_pilot, request)
    assert outcome.exit_code == 2  # type: ignore[attr-defined]


def test_missing_crux_commit_is_fatal_before_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    _install_fixture_locks(monkeypatch)
    monkeypatch.setattr(
        "src.benchmark.separation_pilot.score_oaf_reviewed_subset",
        lambda _request: pytest.fail("missing Crux provenance must fail before scoring"),
    )

    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    assert (
        _run_with_runtime_sentinels(
            run_oaf_separation_pilot,
            replace(request, crux_commit=None),
        ).exit_code
        == 2  # type: ignore[attr-defined]
    )
