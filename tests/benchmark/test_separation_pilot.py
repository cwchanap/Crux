"""HPA-328 fixed-subset preflight and snapshot tests."""

# Fixtures intentionally import the production seam inside tests so the
# baseline collection remains free of the optional runtime modules.
# pylint: disable=import-outside-toplevel,too-many-locals,duplicate-code,too-many-arguments
# pylint: disable=too-many-lines

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path

import pytest

from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.corpus_cache import _remote_from_source_mapping
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
from tests.benchmark.reviewed_subset_fixtures import (
    build_reviewed_subset_oaf_fixture,
    build_reviewed_subset_reference_fixture,
)

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
        spleeter_model_root=tmp_path / "spleeter-model-root",
        demucs_model_root=tmp_path / "demucs-model-root",
        crux_commit="c" * 40,
    )


def _install_fixture_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.benchmark.separation_pilot as pilot

    monkeypatch.setattr(
        pilot,
        "SEPARATOR_LOCK_PATHS",
        {
            SPLEETER_SEPARATOR_ID: FIXTURE_ROOT / "spleeter" / "model.json",
            HTDEMUCS_SEPARATOR_ID: FIXTURE_ROOT / "htdemucs" / "model.json",
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


def test_reviewed_subset_reference_fixture_contains_resolvable_source_audio_remote(
    tmp_path: Path,
) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=20)
    reference = load_reference_set_manifest(fixture.reference_manifest_path)
    source_row = reference.rows[0].source_row
    source_audio_key = source_row["source_audio_key"]
    assert isinstance(source_audio_key, str)

    remote = _remote_from_source_mapping(
        source_row,
        source_audio_key=source_audio_key,
    )

    assert remote.key == source_audio_key
    assert remote.sha256 == source_row["source_audio_content_hash"]


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
    from src.benchmark.separators import (
        AttestedSeparatorRuntime,
        SeparatedStem,
        StemQc,
        load_separator_environment_manifest,
        load_separator_lock,
    )

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
        "attest": [],
        "revalidate": [],
        "events": [],
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

    def stem(separator_id: str, source_sha: str, separator_lock_sha256: str) -> SeparatedStem:
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
            separator_lock_sha256=separator_lock_sha256,
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

    def fake_attest(
        lock_path: Path,
        interpreter: Path,
        model_root: Path,
    ) -> AttestedSeparatorRuntime:
        lock = load_separator_lock(lock_path)
        environment = load_separator_environment_manifest(lock_path, lock)
        calls["attest"].append(lock.separator_id)
        calls["events"].append(f"attest:{lock.separator_id}")
        return AttestedSeparatorRuntime(
            interpreter=interpreter,
            lock=lock,
            model_root=model_root,
            model_files=lock.model_files,
            environment=environment,
            launch_environment={},
        )

    def fake_revalidate(runtime: AttestedSeparatorRuntime) -> None:
        calls["revalidate"].append(runtime.lock.separator_id)

    def spleeter(
        source: Path,
        *,
        source_audio_sha256: str,
        source_duration_sec: float,
        runtime: AttestedSeparatorRuntime,
        cache_root: Path,
    ) -> SeparatedStem:
        del source
        del source_duration_sec, cache_root
        calls["separate"].append(SPLEETER_INPUT_VIEW_ID)
        if spleeter_error is not None:
            raise spleeter_error
        assert runtime.lock.separator_id == SPLEETER_SEPARATOR_ID
        return stem(SPLEETER_SEPARATOR_ID, source_audio_sha256, runtime.lock.sha256)

    def htdemucs(
        source: Path,
        *,
        source_audio_sha256: str,
        source_duration_sec: float,
        runtime: AttestedSeparatorRuntime,
        cache_root: Path,
    ) -> SeparatedStem:
        del source
        del source_duration_sec, cache_root
        calls["separate"].append(HTDEMUCS_INPUT_VIEW_ID)
        assert runtime.lock.separator_id == HTDEMUCS_SEPARATOR_ID
        return stem(HTDEMUCS_SEPARATOR_ID, source_audio_sha256, runtime.lock.sha256)

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
    monkeypatch.setattr(pilot, "attest_separator_runtime", fake_attest, raising=False)
    monkeypatch.setattr(pilot, "revalidate_separator_model_root", fake_revalidate, raising=False)
    monkeypatch.setattr(pilot, "materialize_derived_audio", materialize, raising=False)
    monkeypatch.setattr(pilot, "score_oaf_reviewed_subset", fake_score)
    monkeypatch.setattr(
        pilot,
        "SEPARATOR_LOCK_PATHS",
        {
            SPLEETER_SEPARATOR_ID: FIXTURE_ROOT / "spleeter" / "model.json",
            HTDEMUCS_SEPARATOR_ID: FIXTURE_ROOT / "htdemucs" / "model.json",
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


def test_task6_postflight_root_drift_restores_pending_views_without_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.separation_pilot import run_oaf_separation_pilot
    from src.benchmark.separators import SeparatorExecutionError

    original_spleeter = pilot.run_spleeter_drums

    def dirty_spleeter(*args: object, **kwargs: object) -> object:
        runtime = kwargs["runtime"]
        result = original_spleeter(*args, **kwargs)
        runtime.model_root.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        (runtime.model_root / "extra.bin").write_bytes(b"postflight drift")  # type: ignore[attr-defined]
        return result

    def revalidate(runtime: object) -> None:
        calls["revalidate"].append(runtime.lock.separator_id)  # type: ignore[attr-defined]
        if (runtime.model_root / "extra.bin").is_file():  # type: ignore[attr-defined]
            raise SeparatorExecutionError("separator_model_root_invalid")

    derived_score_calls: list[object] = []
    comparison_calls: list[object] = []
    monkeypatch.setattr(pilot, "run_spleeter_drums", dirty_spleeter)
    monkeypatch.setattr(pilot, "revalidate_separator_model_root", revalidate)
    monkeypatch.setattr(
        pilot,
        "_score_derived_cohort",
        lambda *args: derived_score_calls.append(args),
    )
    monkeypatch.setattr(pilot, "_comparison_reports_ready", lambda _run_dir: True)
    monkeypatch.setattr(
        pilot, "compare_oaf_separation", lambda request: comparison_calls.append(request)
    )

    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert outcome.exit_code == 2
    assert outcome.failure_code == "separator_model_root_invalid"
    assert not derived_score_calls
    assert not comparison_calls
    run_paths = list((request.output_dir / "runs").glob("*/run.json"))
    assert len(run_paths) == 1
    snapshot = json.loads(run_paths[0].read_text(encoding="utf-8"))
    assert snapshot["overall_status"] == "failed"
    for item in snapshot["items"]:
        for view_name in ("spleeter", "htdemucs"):
            view = item[view_name]
            assert view["status"] == "pending"
            assert view["failure_code"] is None
            assert view["stem"] is None
            assert view["input"] is None
            assert view["prediction"] is None
            assert view["runtime"] is None
            assert "inference_config" not in view
            assert "inference_config_sha256" not in view


def test_task6_postflight_root_drift_restores_resume_preimages_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.separation_pilot import run_oaf_separation_pilot
    from src.benchmark.separators import SeparatorExecutionError

    original_spleeter = pilot.run_spleeter_drums

    def dirty_spleeter(*args: object, **kwargs: object) -> object:
        runtime = kwargs["runtime"]
        result = original_spleeter(*args, **kwargs)
        runtime.model_root.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        (runtime.model_root / "extra.bin").write_bytes(b"postflight drift")  # type: ignore[attr-defined]
        return result

    monkeypatch.setattr(pilot, "run_spleeter_drums", dirty_spleeter)

    first = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )
    assert first.exit_code == 0
    assert first.run_path is not None
    prior_snapshot = json.loads(first.run_path.read_text(encoding="utf-8"))
    removed_prediction = prior_snapshot["items"][0]["spleeter"]["prediction"]["path"]
    (request.output_dir / removed_prediction).unlink()
    calls["separate"].clear()
    calls["transcribe"].clear()

    def revalidate(runtime: object) -> None:
        calls["revalidate"].append(runtime.lock.separator_id)  # type: ignore[attr-defined]
        if (runtime.model_root / "extra.bin").is_file():  # type: ignore[attr-defined]
            raise SeparatorExecutionError("separator_model_root_invalid")

    derived_score_calls: list[object] = []
    comparison_calls: list[object] = []
    monkeypatch.setattr(pilot, "revalidate_separator_model_root", revalidate)
    monkeypatch.setattr(
        pilot,
        "_score_derived_cohort",
        lambda *args: derived_score_calls.append(args),
    )
    monkeypatch.setattr(pilot, "_comparison_reports_ready", lambda _run_dir: True)
    monkeypatch.setattr(
        pilot, "compare_oaf_separation", lambda request: comparison_calls.append(request)
    )

    resumed = run_oaf_separation_pilot(
        replace(request, resume=True),
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert resumed.exit_code == 2
    assert resumed.failure_code == "separator_model_root_invalid"
    assert not derived_score_calls
    assert not comparison_calls
    assert resumed.run_path is None
    recovered = json.loads(first.run_path.read_text(encoding="utf-8"))
    assert recovered["overall_status"] == "failed"
    for prior_item, recovered_item in zip(prior_snapshot["items"], recovered["items"], strict=True):
        assert recovered_item["spleeter"] == prior_item["spleeter"]
        assert recovered_item["htdemucs"] == prior_item["htdemucs"]
    assert not calls["separate"]
    assert calls["transcribe"] == [SPLEETER_INPUT_VIEW_ID]


def test_task6_cleanup_revalidates_after_started_separator_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    def interrupting_spleeter(*_args: object, **_kwargs: object) -> object:
        calls["separate"].append(SPLEETER_INPUT_VIEW_ID)
        raise KeyboardInterrupt("separator interrupted")

    monkeypatch.setattr(pilot, "run_spleeter_drums", interrupting_spleeter)

    with pytest.raises(KeyboardInterrupt, match="separator interrupted"):
        run_oaf_separation_pilot(
            request,
            backend_factory=calls["factory"][0],  # type: ignore[arg-type]
        )

    assert calls["revalidate"] == [SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID]


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
        "spleeter_model_root",
        "demucs_model_root",
        "resume",
        "crux_commit",
    }


def test_pilot_attests_each_separator_once_before_snapshot_or_rtf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    real_writer = pilot.write_oaf_separation_run

    def recording_writer(run_path: Path, snapshot: Mapping[str, object]) -> None:
        calls["events"].append("write")
        real_writer(run_path, snapshot)

    monkeypatch.setattr(pilot, "write_oaf_separation_run", recording_writer)
    ticks = iter(float(index) for index in range(1000))

    def recording_perf_counter() -> float:
        calls["events"].append("perf")
        return next(ticks)

    first = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
        perf_counter=recording_perf_counter,
    )

    assert first.exit_code == 0
    assert calls["attest"] == [SPLEETER_SEPARATOR_ID, HTDEMUCS_SEPARATOR_ID]
    assert calls["events"].index("attest:" + HTDEMUCS_SEPARATOR_ID) < calls["events"].index("write")
    assert calls["events"].index("attest:" + HTDEMUCS_SEPARATOR_ID) < calls["events"].index("perf")

    second = run_oaf_separation_pilot(
        replace(request, resume=True),
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
        perf_counter=recording_perf_counter,
    )

    assert second.exit_code == 0
    assert calls["attest"] == [
        SPLEETER_SEPARATOR_ID,
        HTDEMUCS_SEPARATOR_ID,
        SPLEETER_SEPARATOR_ID,
        HTDEMUCS_SEPARATOR_ID,
    ]


def test_pilot_attestation_failure_is_fatal_before_any_mutable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.separation_pilot import run_oaf_separation_pilot
    from src.benchmark.separators import SeparatorExecutionError

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)

    def failing_attest(*_args: object, **_kwargs: object) -> object:
        calls["attest"].append(SPLEETER_SEPARATOR_ID)
        raise SeparatorExecutionError("separator_environment_mismatch")

    monkeypatch.setattr(pilot, "attest_separator_runtime", failing_attest)
    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert outcome.exit_code == 2
    assert outcome.failure_code == "separator_environment_mismatch"
    assert calls["attest"] == [SPLEETER_SEPARATOR_ID]
    assert not calls["score"]
    assert not calls["separate"]
    assert not request.output_dir.exists()


def test_pilot_rejects_lock_replacement_between_identity_and_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.benchmark.separation_pilot as pilot

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)

    replacement_directory = tmp_path / "replacement-spleeter-lock"
    shutil.copytree(FIXTURE_ROOT / "spleeter", replacement_directory)
    replacement_path = replacement_directory / "model.json"
    replacement_payload = json.loads(replacement_path.read_text(encoding="utf-8"))
    replacement_payload["repository_revision"] = "b" * 40
    replacement_path.write_bytes(canonical_json_bytes(replacement_payload, trailing_newline=True))

    original_load = pilot._load_separator_locks

    def load_then_replace() -> tuple[object, object]:
        locks = original_load()
        pilot.SEPARATOR_LOCK_PATHS = {
            **pilot.SEPARATOR_LOCK_PATHS,
            SPLEETER_SEPARATOR_ID: replacement_path,
        }
        return locks

    monkeypatch.setattr(pilot, "_load_separator_locks", load_then_replace)
    outcome = pilot.run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    assert outcome.exit_code == 2
    assert outcome.failure_code == "separator_lock_companion_mismatch"
    assert not request.output_dir.exists()
    assert calls["separate"] == []
    assert calls["score"] == []


@pytest.mark.parametrize("model_root_field", ("spleeter_model_root", "demucs_model_root"))
def test_output_nested_under_separator_model_root_fails_before_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_root_field: str,
) -> None:
    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    calls = _task6_seams(tmp_path, fixture, monkeypatch)
    model_root = getattr(request, model_root_field)
    nested_output = model_root / "nested" / ".." / "pilot-output"
    request = replace(request, output_dir=nested_output)

    outcome = run_oaf_separation_pilot(
        request,
        backend_factory=calls["factory"][0],  # type: ignore[arg-type]
    )

    normalized_output = nested_output.absolute()
    assert outcome.exit_code == 2
    assert outcome.run_path is None
    assert calls["attest"] == []
    assert calls["score"] == []
    assert calls["separate"] == []
    assert not normalized_output.exists()
    assert not (normalized_output / "runs").exists()


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
            HTDEMUCS_SEPARATOR_ID: FIXTURE_ROOT / "htdemucs" / "model.json",
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


# ---------------------------------------------------------------------------
# Direct unit coverage for the pure helper / validation functions.
# ---------------------------------------------------------------------------


def _fake_source():
    from src.benchmark.corpus_cache import ResolvedSourceAudio

    return ResolvedSourceAudio(
        path=Path("/fake/source.wav"),
        source_audio_id="100/bgm.wav",
        source_audio_sha256="a" * 64,
        duration_sec=1.0,
    )


def _fake_lock():
    from src.benchmark.separators import load_separator_lock

    return load_separator_lock(FIXTURE_ROOT / "spleeter" / "model.json")


def test_request_post_init_rejects_non_path_field(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import OafSeparationPilotRequest

    with pytest.raises(TypeError, match="cache_dir must be a Path"):
        OafSeparationPilotRequest(
            reference_manifest_path=tmp_path / "ref.jsonl",
            timing_manifest_path=tmp_path / "timing.jsonl",
            subset_manifest_path=tmp_path / "subset.jsonl",
            oaf_run_path=tmp_path / "run.json",
            cache_dir=str(tmp_path / "cache"),
            output_dir=tmp_path / "out",
            spleeter_python=Path("/p"),
            demucs_python=Path("/p"),
            spleeter_model_root=tmp_path / "s",
            demucs_model_root=tmp_path / "d",
            crux_commit="c" * 40,
        )


def test_request_post_init_rejects_non_bool_resume(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import OafSeparationPilotRequest

    with pytest.raises(TypeError, match="resume must be a bool"):
        OafSeparationPilotRequest(
            reference_manifest_path=tmp_path / "ref.jsonl",
            timing_manifest_path=tmp_path / "timing.jsonl",
            subset_manifest_path=tmp_path / "subset.jsonl",
            oaf_run_path=tmp_path / "run.json",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "out",
            spleeter_python=Path("/p"),
            demucs_python=Path("/p"),
            spleeter_model_root=tmp_path / "s",
            demucs_model_root=tmp_path / "d",
            resume="yes",  # type: ignore[arg-type]
            crux_commit="c" * 40,
        )


def test_request_post_init_rejects_malformed_crux_commit(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import OafSeparationPilotRequest

    with pytest.raises(ValueError, match="crux_commit must be a lowercase 40-character commit"):
        OafSeparationPilotRequest(
            reference_manifest_path=tmp_path / "ref.jsonl",
            timing_manifest_path=tmp_path / "timing.jsonl",
            subset_manifest_path=tmp_path / "subset.jsonl",
            oaf_run_path=tmp_path / "run.json",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "out",
            spleeter_python=Path("/p"),
            demucs_python=Path("/p"),
            spleeter_model_root=tmp_path / "s",
            demucs_model_root=tmp_path / "d",
            crux_commit="not-a-commit",
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"overall_status": "bogus"}, "overall_status is invalid"),
        ({"exit_code": 5}, "exit_code is invalid"),
        ({"run_id": 123}, "run_id must be a string or None"),
        ({"run_path": "not-a-path"}, "run_path must be a Path or None"),
        ({"success_count": -1}, "success_count must be a nonnegative integer"),
        ({"success_count": True}, "success_count must be a nonnegative integer"),
        ({"failure_code": "not_a_code"}, "failure_code is invalid"),
        ({"failure_code": 123}, "failure_code is invalid"),
    ],
)
def test_outcome_post_init_rejects_invalid_fields(overrides: dict[str, object], match: str) -> None:
    from src.benchmark.separation_pilot import OafSeparationPilotOutcome

    base: dict[str, object] = {
        "overall_status": "complete",
        "exit_code": 0,
        "run_id": "oaf-separation-abcdef0123456789",
        "run_path": None,
        "reports_path": None,
        "full_mix_reports_path": None,
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "quarantined_count": 0,
        "failure_code": None,
    }
    base.update(overrides)
    with pytest.raises((ValueError, TypeError), match=match):
        OafSeparationPilotOutcome(**base)  # type: ignore[arg-type]


def test_require_hash_rejects_non_string() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _require_hash

    with pytest.raises(SeparationRunError, match="must be a lowercase SHA-256"):
        _require_hash(123, "field")


def test_require_hash_rejects_invalid_hex() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _require_hash

    with pytest.raises(SeparationRunError, match="must be a lowercase SHA-256"):
        _require_hash("nothex", "field")


def test_require_nonempty_string_rejects_non_string() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _require_nonempty_string

    with pytest.raises(SeparationRunError, match="must be a nonempty string"):
        _require_nonempty_string(123, "field")


def test_require_nonempty_string_rejects_empty() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _require_nonempty_string

    with pytest.raises(SeparationRunError, match="must be a nonempty string"):
        _require_nonempty_string("", "field")


def test_require_absolute_path_rejects_non_string() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _require_absolute_path

    with pytest.raises(SeparationRunError, match="must be an absolute path"):
        _require_absolute_path(123, "field")


def test_require_absolute_path_rejects_empty() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _require_absolute_path

    with pytest.raises(SeparationRunError, match="must be an absolute path"):
        _require_absolute_path("", "field")


def test_require_absolute_path_rejects_relative(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _require_absolute_path

    with pytest.raises(SeparationRunError, match="must be an absolute path"):
        _require_absolute_path("relative/path", "field")


def test_require_crux_commit_rejects_invalid() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _require_crux_commit

    with pytest.raises(SeparationRunError, match="crux_commit"):
        _require_crux_commit("short")


def test_normalize_snapshot_value_rejects_non_string_key() -> None:
    from src.benchmark.backend_identity import StrictJsonError
    from src.benchmark.separation_pilot import _normalize_snapshot_value

    with pytest.raises(StrictJsonError, match="object keys must be strings"):
        _normalize_snapshot_value({1: "value"})


def test_normalize_snapshot_value_rejects_unsupported_type() -> None:
    from src.benchmark.backend_identity import StrictJsonError
    from src.benchmark.separation_pilot import _normalize_snapshot_value

    with pytest.raises(StrictJsonError, match="unsupported separation snapshot value"):
        _normalize_snapshot_value(object())


def test_normalize_snapshot_value_quantizes_float() -> None:
    from decimal import Decimal

    from src.benchmark.separation_pilot import _normalize_snapshot_value

    result = _normalize_snapshot_value(1.0000001)
    # Floats are quantized to a fixed six-place Decimal, not a string.
    assert isinstance(result, Decimal)


def test_validate_evidence_rejects_non_mapping() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_evidence

    with pytest.raises(SeparationRunError, match="evidence must be an object or null"):
        _validate_evidence("not-a-mapping", "field")


def test_validate_view_row_rejects_non_mapping() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_view_row

    with pytest.raises(SeparationRunError, match="view evidence must be an object"):
        _validate_view_row("not-a-mapping", field="spleeter", derived=True)


def test_validate_view_row_rejects_invalid_status() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_view_row

    with pytest.raises(SeparationRunError, match="view status is invalid"):
        _validate_view_row({"status": "bogus"}, field="spleeter", derived=True)


def test_validate_view_row_rejects_invalid_failure_code() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_view_row

    with pytest.raises(SeparationRunError, match="failure_code is invalid"):
        _validate_view_row(
            {"status": "success", "failure_code": 123},
            field="spleeter",
            derived=True,
        )


def test_validate_view_row_rejects_invalid_evidence() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_view_row

    with pytest.raises(SeparationRunError, match="stem evidence must be an object or null"):
        _validate_view_row(
            {"status": "success", "stem": "not-a-mapping"},
            field="spleeter",
            derived=True,
        )


def test_validate_view_row_derived_rejects_missing_separator_lock() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_view_row

    with pytest.raises(SeparationRunError, match="separator_lock_sha256"):
        _validate_view_row({"status": "pending"}, field="spleeter", derived=True)


def test_validate_view_row_derived_rejects_wrong_input_view_id() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_view_row

    with pytest.raises(SeparationRunError, match="input view identity is invalid"):
        _validate_view_row(
            {
                "status": "pending",
                "separator_lock_sha256": "a" * 64,
                "input_view_id": "wrong-view",
            },
            field="spleeter",
            derived=True,
        )


def test_validate_view_row_parent_rejects_wrong_full_mix_input_view_id() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_view_row

    with pytest.raises(SeparationRunError, match="full_mix input view identity is invalid"):
        _validate_view_row(
            {"status": "inferred", "input_view_id": "wrong-view"},
            field="full_mix",
            derived=False,
        )


def test_validate_view_row_rejects_input_evidence_view_mismatch() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_view_row

    with pytest.raises(SeparationRunError, match="input evidence view identity is invalid"):
        _validate_view_row(
            {
                "status": "pending",
                "separator_lock_sha256": "a" * 64,
                "input_view_id": SPLEETER_INPUT_VIEW_ID,
                "input": {"input_view_id": "wrong"},
            },
            field="spleeter",
            derived=True,
        )


def test_validate_view_row_rejects_invalid_input_audio_sha() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_view_row

    with pytest.raises(SeparationRunError, match="input_audio_sha256"):
        _validate_view_row(
            {
                "status": "pending",
                "separator_lock_sha256": "a" * 64,
                "input_view_id": SPLEETER_INPUT_VIEW_ID,
                "input": {
                    "input_view_id": SPLEETER_INPUT_VIEW_ID,
                    "input_audio_sha256": "nothex",
                },
            },
            field="spleeter",
            derived=True,
        )


def test_validate_view_row_rejects_empty_prediction_path() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_view_row

    with pytest.raises(SeparationRunError, match="prediction path is invalid"):
        _validate_view_row(
            {
                "status": "pending",
                "separator_lock_sha256": "a" * 64,
                "input_view_id": SPLEETER_INPUT_VIEW_ID,
                "prediction": {"path": ""},
            },
            field="spleeter",
            derived=True,
        )


def test_validate_view_row_rejects_invalid_prediction_sha() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _validate_view_row

    with pytest.raises(SeparationRunError, match="prediction.artifact_sha256"):
        _validate_view_row(
            {
                "status": "pending",
                "separator_lock_sha256": "a" * 64,
                "input_view_id": SPLEETER_INPUT_VIEW_ID,
                "prediction": {"path": "rel/path", "artifact_sha256": "nothex"},
            },
            field="spleeter",
            derived=True,
        )


def test_render_oaf_separation_run_rejects_non_mapping() -> None:
    from src.benchmark.separation_pilot import render_oaf_separation_run

    with pytest.raises(TypeError, match="run snapshot must be a mapping"):
        render_oaf_separation_run("not-a-mapping")  # type: ignore[arg-type]


def test_parse_oaf_separation_run_rejects_non_canonical() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, parse_oaf_separation_run

    with pytest.raises((SeparationRunError, ValueError)):
        parse_oaf_separation_run(b'{"schema":"wrong"}')


def test_write_oaf_separation_run_rejects_non_path() -> None:
    from src.benchmark.separation_pilot import write_oaf_separation_run

    with pytest.raises(TypeError, match="run_path must be a Path"):
        write_oaf_separation_run("not-a-path", {})  # type: ignore[arg-type]


def test_load_separator_locks_rejects_missing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.separation_pilot import SeparationRunError

    monkeypatch.setattr(
        pilot,
        "SEPARATOR_LOCK_PATHS",
        {
            SPLEETER_SEPARATOR_ID: tmp_path / "missing-spleeter.json",
            HTDEMUCS_SEPARATOR_ID: FIXTURE_ROOT / "htdemucs" / "model.json",
        },
    )
    with pytest.raises(SeparationRunError, match="separator lock is invalid"):
        pilot._load_separator_locks()


def test_load_separator_locks_rejects_wrong_separator_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.separation_pilot import SeparationRunError

    swapped = tmp_path / "swapped"
    shutil.copytree(FIXTURE_ROOT / "spleeter", swapped / "spleeter")
    shutil.copytree(FIXTURE_ROOT / "htdemucs", swapped / "htdemucs")
    monkeypatch.setattr(
        pilot,
        "SEPARATOR_LOCK_PATHS",
        {
            SPLEETER_SEPARATOR_ID: swapped / "htdemucs" / "model.json",
            HTDEMUCS_SEPARATOR_ID: swapped / "spleeter" / "model.json",
        },
    )
    with pytest.raises(SeparationRunError, match="Spleeter separator lock identity is invalid"):
        pilot._load_separator_locks()


def test_load_separator_locks_rejects_identical_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.separation_pilot import SeparationRunError
    from src.benchmark.separators import SeparatorLock, load_separator_lock

    spleeter_lock = load_separator_lock(FIXTURE_ROOT / "spleeter" / "model.json")
    htdemucs_lock = load_separator_lock(FIXTURE_ROOT / "htdemucs" / "model.json")
    # Force identical sha256 while preserving each lock's correct separator_id.
    forged_htdemucs = replace(htdemucs_lock, sha256=spleeter_lock.sha256)
    forged_spleeter = replace(spleeter_lock, sha256=spleeter_lock.sha256)

    def fake_load(path: Path) -> SeparatorLock:
        if path == FIXTURE_ROOT / "spleeter" / "model.json":
            return forged_spleeter
        return forged_htdemucs

    monkeypatch.setattr(pilot, "load_separator_lock", fake_load)
    monkeypatch.setattr(
        pilot,
        "SEPARATOR_LOCK_PATHS",
        {
            SPLEETER_SEPARATOR_ID: FIXTURE_ROOT / "spleeter" / "model.json",
            HTDEMUCS_SEPARATOR_ID: FIXTURE_ROOT / "htdemucs" / "model.json",
        },
    )
    with pytest.raises(SeparationRunError, match="separator locks must be distinct"):
        pilot._load_separator_locks()


def test_read_parent_run_rejects_invalid_file(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _read_parent_run

    bad = tmp_path / "bad-run.json"
    bad.write_bytes(b"not json at all")
    with pytest.raises(SeparationRunError, match="parent OaF run snapshot is invalid"):
        _read_parent_run(bad)


def test_separator_failure_status_maps_stem_codes() -> None:
    from src.benchmark.separation_pilot import _separator_failure_status

    assert _separator_failure_status("stem_decode_failed") == "stem_invalid"
    assert _separator_failure_status("stem_channel_count") == "stem_invalid"
    assert _separator_failure_status("stem_nonfinite") == "stem_invalid"
    assert _separator_failure_status("stem_duration_invalid") == "stem_invalid"
    assert _separator_failure_status("stem_duration_mismatch") == "stem_invalid"
    assert _separator_failure_status("stem_near_silent") == "stem_invalid"
    assert _separator_failure_status("stem_identity_invalid") == "stem_invalid"
    assert _separator_failure_status("separator_timeout") == "separation_failed"
    assert _separator_failure_status("unknown_code") == "separation_failed"


def test_derived_failure_reason_returns_code_reason() -> None:
    from src.benchmark.separation_pilot import _derived_failure_reason

    assert _derived_failure_reason("separation_failed", "stem_invalid") == "inference_failed"
    assert _derived_failure_reason("prediction_invalid", "prediction_publish_failed") == (
        "prediction_artifact_invalid"
    )


def test_derived_failure_reason_falls_back_to_status() -> None:
    from src.benchmark.separation_pilot import _derived_failure_reason

    assert _derived_failure_reason("pending", None) == "inference_failed"
    assert _derived_failure_reason("separation_failed", None) == "inference_failed"
    assert _derived_failure_reason("stem_invalid", None) == "inference_failed"
    assert _derived_failure_reason("inference_failed", None) == "inference_failed"
    assert _derived_failure_reason("prediction_invalid", None) == "prediction_artifact_invalid"


def test_derived_failure_reason_rejects_unsupported() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _derived_failure_reason

    with pytest.raises(SeparationRunError, match="no supported HPA-325 failure reason"):
        _derived_failure_reason("unknown_status", "unknown_code")


def test_close_backend_handles_none() -> None:
    from src.benchmark.separation_pilot import _close_backend

    _close_backend(None)


def test_close_backend_swallows_errors() -> None:
    from src.benchmark.separation_pilot import _close_backend

    class BadBackend:
        def close(self) -> None:
            raise OSError("boom")

    _close_backend(BadBackend())


def test_close_backend_closes_clean_backend() -> None:
    from src.benchmark.separation_pilot import _close_backend

    closed: list[bool] = []

    class GoodBackend:
        def close(self) -> None:
            closed.append(True)

    _close_backend(GoodBackend())
    assert closed == [True]


def test_relative_artifact_path_returns_relative(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _relative_artifact_path

    root = tmp_path / "root"
    child = root / "child" / "file.wav"
    child.parent.mkdir(parents=True)
    child.write_bytes(b"x")
    assert _relative_artifact_path(child, root) == "child/file.wav"


def test_relative_artifact_path_falls_back_to_absolute(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _relative_artifact_path

    root = tmp_path / "root"
    other = tmp_path / "other" / "file.wav"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"x")
    result = _relative_artifact_path(other, root)
    assert result == str(other)


def test_view_inference_config_rejects_missing_config() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _view_inference_config

    with pytest.raises(SeparationRunError, match="parent inference config is unavailable"):
        _view_inference_config({}, SPLEETER_INPUT_VIEW_ID)


def _full_mix_inference_config() -> dict[str, str]:
    return {
        "schema": "crux.oaf-inference-config/v1",
        "backend_descriptor_sha256": "a" * 64,
        "model_lock_sha256": "b" * 64,
        "checkpoint_archive_sha256": "c" * 64,
        "adapter_revision": "0.0.0",
        "prediction_map_version": "1",
        "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
        "canonicalization_revision": "0.0.0",
    }


def test_view_inference_config_rejects_empty_view_id() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _view_inference_config

    parent = {"inference_config": _full_mix_inference_config()}
    with pytest.raises(SeparationRunError, match="derived input view identity is invalid"):
        _view_inference_config(parent, "")


def test_view_inference_config_swaps_view_id() -> None:
    from src.benchmark.separation_pilot import _view_inference_config

    parent = {"inference_config": _full_mix_inference_config()}
    config, sha = _view_inference_config(parent, SPLEETER_INPUT_VIEW_ID)
    assert config["input_view_id"] == SPLEETER_INPUT_VIEW_ID
    assert isinstance(sha, str)


def test_resolve_retained_stem_rejects_missing_path(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _resolve_retained_stem
    from src.benchmark.separators import SeparatorExecutionError

    with pytest.raises(SeparatorExecutionError, match="retained stem path is missing"):
        _resolve_retained_stem(
            {},
            cache_root=tmp_path,
            source=_fake_source(),
            lock=_fake_lock(),
        )


def test_resolve_retained_stem_rejects_missing_hash(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _resolve_retained_stem
    from src.benchmark.separators import SeparatorExecutionError

    with pytest.raises(SeparatorExecutionError, match="retained stem hash is missing"):
        _resolve_retained_stem(
            {"path": str(tmp_path / "stem.wav")},
            cache_root=tmp_path,
            source=_fake_source(),
            lock=_fake_lock(),
        )


def test_resolve_retained_stem_rejects_outside_cache(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _resolve_retained_stem
    from src.benchmark.separators import SeparatorExecutionError

    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"x")
    with pytest.raises(SeparatorExecutionError, match="outside the native cache"):
        _resolve_retained_stem(
            {"path": str(outside), "sha256": hashlib.sha256(b"x").hexdigest()},
            cache_root=tmp_path / "cache",
            source=_fake_source(),
            lock=_fake_lock(),
        )


def test_resolve_retained_stem_rejects_hash_mismatch(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _resolve_retained_stem
    from src.benchmark.separators import SeparatorExecutionError

    cache_root = tmp_path / "cache"
    stem_path = cache_root / "stems" / "drums.wav"
    stem_path.parent.mkdir(parents=True)
    stem_path.write_bytes(b"real-bytes")
    with pytest.raises(SeparatorExecutionError, match="bytes do not match"):
        _resolve_retained_stem(
            {"path": str(stem_path), "sha256": "e" * 64},
            cache_root=cache_root,
            source=_fake_source(),
            lock=_fake_lock(),
        )


def test_resolve_retained_stem_rejects_source_mismatch(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _resolve_retained_stem
    from src.benchmark.separators import SeparatorExecutionError

    cache_root = tmp_path / "cache"
    stem_path = cache_root / "stems" / "drums.wav"
    stem_path.parent.mkdir(parents=True)
    content = b"real-bytes"
    stem_path.write_bytes(content)
    with pytest.raises(SeparatorExecutionError, match="source identity does not match"):
        _resolve_retained_stem(
            {
                "path": str(stem_path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "source_audio_sha256": "wrong",
            },
            cache_root=cache_root,
            source=_fake_source(),
            lock=_fake_lock(),
        )


def test_resolve_retained_stem_rejects_lock_mismatch(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _resolve_retained_stem
    from src.benchmark.separators import SeparatorExecutionError

    cache_root = tmp_path / "cache"
    stem_path = cache_root / "stems" / "drums.wav"
    stem_path.parent.mkdir(parents=True)
    content = b"real-bytes"
    stem_path.write_bytes(content)
    source = _fake_source()
    with pytest.raises(SeparatorExecutionError, match="separator identity does not match"):
        _resolve_retained_stem(
            {
                "path": str(stem_path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "source_audio_sha256": source.source_audio_sha256,
                "separator_lock_sha256": "wrong-lock",
            },
            cache_root=cache_root,
            source=source,
            lock=_fake_lock(),
        )


def test_resolve_retained_stem_succeeds_for_valid_stem(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _resolve_retained_stem

    cache_root = tmp_path / "cache"
    stem_path = cache_root / "stems" / "drums.wav"
    stem_path.parent.mkdir(parents=True)
    content = b"real-bytes"
    stem_path.write_bytes(content)
    source = _fake_source()
    lock = _fake_lock()
    resolved = _resolve_retained_stem(
        {
            "path": str(stem_path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "source_audio_sha256": source.source_audio_sha256,
            "separator_lock_sha256": lock.sha256,
        },
        cache_root=cache_root,
        source=source,
        lock=lock,
    )
    assert resolved == stem_path.resolve()


def test_stem_evidence_rejects_non_stem() -> None:
    from src.benchmark.separation_pilot import SeparationRunError, _stem_evidence

    with pytest.raises(SeparationRunError, match="separator returned an invalid stem"):
        _stem_evidence("not-a-stem", Path("/cache"))  # type: ignore[arg-type]


def test_mark_outstanding_derived_views_marks_pending(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _mark_outstanding_derived_views

    snapshot: dict[str, object] = {
        "items": [
            {
                "simfile_id": 1,
                "spleeter": {"status": "pending", "runtime": None},
                "htdemucs": {"status": "success"},
            },
        ]
    }
    _mark_outstanding_derived_views(snapshot)
    view = snapshot["items"][0]["spleeter"]  # type: ignore[index]
    assert view["status"] == "inference_failed"
    assert view["failure_code"] == "worker_protocol_failed"
    assert snapshot["items"][0]["htdemucs"]["status"] == "success"  # type: ignore[index]


def test_mark_outstanding_derived_views_rejects_non_list() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _mark_outstanding_derived_views,
    )

    with pytest.raises(SeparationRunError, match="items are unavailable"):
        _mark_outstanding_derived_views({"items": "not-a-list"})


def test_capture_derived_view_preimages_rejects_non_list() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _capture_derived_view_preimages,
    )

    with pytest.raises(SeparationRunError, match="items are unavailable"):
        _capture_derived_view_preimages({"items": "not-a-list"})


def test_capture_derived_view_preimages_rejects_invalid_item() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _capture_derived_view_preimages,
    )

    with pytest.raises(SeparationRunError, match="run snapshot item is invalid"):
        _capture_derived_view_preimages({"items": ["not-a-mapping"]})


def test_capture_derived_view_preimages_rejects_invalid_simfile_id() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _capture_derived_view_preimages,
    )

    with pytest.raises(SeparationRunError, match="simfile_id is invalid"):
        _capture_derived_view_preimages(
            {"items": [{"simfile_id": "not-int", "spleeter": {}, "htdemucs": {}}]}
        )


def test_capture_derived_view_preimages_rejects_invalid_view() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _capture_derived_view_preimages,
    )

    with pytest.raises(SeparationRunError, match="derived view evidence is invalid"):
        _capture_derived_view_preimages(
            {"items": [{"simfile_id": 1, "spleeter": "not-a-mapping", "htdemucs": {}}]}
        )


def test_restore_derived_view_preimages_rejects_non_list() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _restore_derived_view_preimages,
    )

    with pytest.raises(SeparationRunError, match="items are unavailable"):
        _restore_derived_view_preimages({"items": "not-a-list"}, {})


def test_restore_derived_view_preimages_rejects_invalid_preimage_name() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _restore_derived_view_preimages,
    )

    snapshot: dict[str, object] = {"items": [{"simfile_id": 1}]}
    with pytest.raises(SeparationRunError, match="preimage name is invalid"):
        _restore_derived_view_preimages(snapshot, {(1, "bogus"): {}})


def test_recover_prior_derived_evidence_rejects_non_list_items() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _recover_prior_derived_evidence,
    )

    with pytest.raises(SeparationRunError, match="items are unavailable"):
        _recover_prior_derived_evidence({"items": "not-a-list"}, {"items": []})


def test_recover_prior_derived_evidence_rejects_invalid_prior_item() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _recover_prior_derived_evidence,
    )

    with pytest.raises(SeparationRunError, match="prior run snapshot item is invalid"):
        _recover_prior_derived_evidence(
            {"items": [{"simfile_id": 1}]},
            {"items": [{"simfile_id": "not-int"}]},
        )


def test_recover_prior_derived_evidence_rejects_missing_membership() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _recover_prior_derived_evidence,
    )

    with pytest.raises(SeparationRunError, match="membership is incomplete"):
        _recover_prior_derived_evidence(
            {"items": [{"simfile_id": 1}]},
            {"items": [{"simfile_id": 2}]},
        )


def test_recover_prior_derived_evidence_rejects_source_mismatch() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _recover_prior_derived_evidence,
    )

    with pytest.raises(SeparationRunError, match="source identity does not match"):
        _recover_prior_derived_evidence(
            {"items": [{"simfile_id": 1, "source_row_sha256": "a" * 64}]},
            {"items": [{"simfile_id": 1, "source_row_sha256": "b" * 64}]},
        )


def test_recover_prior_derived_evidence_rejects_invalid_prior_view() -> None:
    from src.benchmark.separation_pilot import (
        SeparationRunError,
        _recover_prior_derived_evidence,
    )

    row = {
        "simfile_id": 1,
        "source_row_sha256": "a" * 64,
        "source_audio_id": "id",
        "source_audio_sha256": "b" * 64,
    }
    with pytest.raises(SeparationRunError, match="prior derived view evidence is invalid"):
        _recover_prior_derived_evidence(
            {"items": [dict(row)]},
            {"items": [dict(row, spleeter="not-a-mapping")]},
        )


def test_revalidate_separator_runtimes_raises_first_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.benchmark.separation_pilot import _revalidate_separator_runtimes
    from src.benchmark.separators import SeparatorExecutionError

    def fail(_runtime: object) -> None:
        raise SeparatorExecutionError("separator_model_root_invalid")

    monkeypatch.setattr("src.benchmark.separation_pilot.revalidate_separator_model_root", fail)

    class _StubRuntime:
        class lock:
            separator_id = "stub"

    with pytest.raises(SeparatorExecutionError, match="separator_model_root_invalid"):
        _revalidate_separator_runtimes({"stub": _StubRuntime()})


def test_validate_frozen_oaf_binding_rejects_missing_identity() -> None:
    from src.benchmark.backends.oaf import OafBackendError
    from src.benchmark.separation_pilot import _validate_frozen_oaf_binding

    with pytest.raises(OafBackendError, match="parent OaF model identity is unavailable"):
        _validate_frozen_oaf_binding({})


def test_validate_frozen_oaf_binding_rejects_drifted_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.benchmark.separation_pilot as pilot
    from src.benchmark.backends.oaf import OafBackendError

    drifted = tmp_path / "drifted-model.json"
    drifted.write_bytes(b"drifted")
    monkeypatch.setattr(pilot, "_model_lock_path", lambda: drifted, raising=False)
    with pytest.raises(OafBackendError, match="model lock cannot be verified"):
        pilot._validate_frozen_oaf_binding(
            {"model_lock_sha256": "a" * 64, "checkpoint_archive_sha256": "b" * 64}
        )


def test_comparison_reports_ready_returns_false_for_missing(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _comparison_reports_ready

    assert _comparison_reports_ready(tmp_path) is False


def test_comparison_reports_ready_returns_true_when_complete(tmp_path: Path) -> None:
    from src.benchmark.separation_pilot import _COMPARISON_REPORT_NAMES, _comparison_reports_ready

    for view_name in ("full_mix", "spleeter", "htdemucs"):
        report_dir = tmp_path / "views" / view_name / "reports"
        report_dir.mkdir(parents=True)
        for report_name in _COMPARISON_REPORT_NAMES:
            (report_dir / report_name).write_bytes(b"x")
    assert _comparison_reports_ready(tmp_path) is True


def test_fatal_outcome_defaults_to_no_failure_code() -> None:
    from src.benchmark.separation_pilot import _fatal_outcome

    outcome = _fatal_outcome()
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2
    assert outcome.failure_code is None


def test_fatal_outcome_carries_failure_code() -> None:
    from src.benchmark.separation_pilot import _fatal_outcome

    outcome = _fatal_outcome("separator_model_root_invalid")
    assert outcome.failure_code == "separator_model_root_invalid"


def test_pilot_rejects_non_callable_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    fixture = build_reviewed_subset_oaf_fixture(tmp_path, eligible_count=20)
    subset = _subset_path(tmp_path, fixture)
    request = _request(tmp_path, fixture, subset)
    with pytest.raises(TypeError, match="execution seams must be callable"):
        run_oaf_separation_pilot(request, perf_counter="not-callable")  # type: ignore[arg-type]


def test_pilot_rejects_non_request_type() -> None:
    from src.benchmark.separation_pilot import run_oaf_separation_pilot

    with pytest.raises(TypeError, match="request must be OafSeparationPilotRequest"):
        run_oaf_separation_pilot("not-a-request")  # type: ignore[arg-type]
