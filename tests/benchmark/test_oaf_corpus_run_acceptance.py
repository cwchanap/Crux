from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from runtime.oaf_tf1.model import load_model_config
from src.benchmark.backend_identity import BackendDescriptor
from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.backends.oaf import OafBackendError
from src.benchmark.oaf_corpus_run import (
    OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
    OAF_FULL_MIX_INPUT_VIEW_ID,
    OafCorpusRunRequest,
    ResolvedSourceAudio,
    _expected_oaf_descriptor,
    run_oaf_corpus,
)
from src.benchmark.prediction_artifact import read_prediction_artifact, render_prediction_artifact
from src.benchmark.reference_set import (
    ReferenceMappingResult,
    map_reference_events,
)
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    ReferenceSetRowView,
)
from src.benchmark.reference_timing import NativeReferenceEvent
from src.benchmark.reference_timing_manifest import LoadedReferenceTimingManifest

SHA_A = "a" * 64
SHA_B = "b" * 64


def _fake_manifests() -> tuple[LoadedReferenceSetManifest, LoadedReferenceTimingManifest]:
    rows = []
    for simfile_id in (10, 20, 30):
        rows.append(
            LoadedReferenceSetRow(
                source_row={
                    "selected_chart_key": f"{simfile_id}/chart.dtx",
                    "selected_chart_content_hash": SHA_A,
                    "source_audio_key": f"{simfile_id}/audio.wav",
                    "source_audio_content_hash": SHA_A,
                    "source_endpoint_sha256": SHA_A,
                    "source_bucket": "simfile-dtx",
                },
                view=ReferenceSetRowView(
                    simfile_id=simfile_id,
                    eligibility_status="eligible",
                    eligibility_reason_codes=(),
                    eligibility_warnings=(),
                    mapped_event_count=0,
                    common_scored_event_count=0,
                    ignored_event_count=0,
                    unmapped_event_count=0,
                    duplicate_common_event_count=0,
                ),
            )
        )
    return (
        LoadedReferenceSetManifest(
            manifest_sha256=SHA_A,
            corpus_version="sha256:" + SHA_B,
            source_reference_timing_manifest_sha256=SHA_B,
            source_reference_timing_version="sha256:" + SHA_A,
            rows=tuple(rows),
        ),
        LoadedReferenceTimingManifest(
            manifest_sha256=SHA_B,
            corpus_version="sha256:" + SHA_A,
            rows=(),
        ),
    )


def _reference_mapping(simfile_id: int) -> ReferenceMappingResult:
    return map_reference_events(
        (
            NativeReferenceEvent(
                simfile_id=simfile_id,
                selected_chart_key=f"{simfile_id}/chart.dtx",
                selected_chart_content_hash=SHA_A,
                source_audio_key=f"{simfile_id}/audio.wav",
                source_audio_content_hash=SHA_A,
                source_order=0,
                measure=1,
                position=0.0,
                lane_id="13",
                note_id="kick-0",
                chart_time_sec=0.25,
                audio_time_sec=0.25,
            ),
        )
    )


def _install_fake_run_seams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[LoadedReferenceSetManifest, BackendDescriptor, list[int]]:
    """Install deterministic manifest/source/materializer seams for lifecycle tests."""
    import src.benchmark.oaf_corpus_run as run_module

    reference_manifest, timing_manifest = _fake_manifests()
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference_manifest)
    monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing_manifest)
    empty_mappings = {
        row.view.simfile_id: _reference_mapping(row.view.simfile_id)
        for row in reference_manifest.rows
    }
    monkeypatch.setattr(
        run_module, "_preflight_reference_mappings", lambda *_args, **_kwargs: empty_mappings
    )
    monkeypatch.setattr(
        run_module,
        "_resolve_source_audio",
        lambda source, *_args, **_kwargs: ResolvedSourceAudio(
            path=tmp_path / "source.wav",
            source_audio_id=source["source_audio_key"],
            source_audio_sha256=SHA_A,
            duration_sec=1.0,
        ),
    )
    config = load_model_config()
    descriptor = _expected_oaf_descriptor(config)
    materialize_calls: list[int] = []

    def materialize(source, output_path, *, input_root, config):
        del input_root, config
        materialize_calls.append(int(source.source_audio_id.split("/")[0]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"temporary canonical wav")
        return CanonicalAudio(
            path=output_path,
            source_audio_id=source.source_audio_id,
            source_audio_sha256=source.source_audio_sha256,
            input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
            input_audio_sha256=SHA_B,
            byte_length=88244,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=44100,
        )

    monkeypatch.setattr(run_module, "_materialize_oaf_full_mix", materialize)
    return reference_manifest, descriptor, materialize_calls


def _request(tmp_path: Path, *, resume: bool = False) -> OafCorpusRunRequest:
    return OafCorpusRunRequest(
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "timing.jsonl",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        resume=resume,
    )


def test_run_oaf_corpus_public_entry_point_is_available(tmp_path: Path) -> None:
    request = OafCorpusRunRequest(
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "timing.jsonl",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
    )

    assert request.output_dir == tmp_path / "output"
    assert callable(run_oaf_corpus)


@pytest.mark.parametrize("close_failure", [False, True])
def test_run_oaf_corpus_uses_one_persistent_backend_and_request_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, close_failure: bool
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    reference_manifest, timing_manifest = _fake_manifests()
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference_manifest)
    monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing_manifest)
    monkeypatch.setattr(
        run_module,
        "_preflight_reference_mappings",
        lambda *_args, **_kwargs: {
            row.view.simfile_id: _reference_mapping(row.view.simfile_id)
            for row in reference_manifest.rows
        },
    )
    monkeypatch.setattr(
        run_module,
        "_resolve_source_audio",
        lambda source, *_args, **_kwargs: ResolvedSourceAudio(
            path=tmp_path / "source.wav",
            source_audio_id=source["source_audio_key"],
            source_audio_sha256=SHA_A,
            duration_sec=1.0,
        ),
    )

    config = load_model_config()
    descriptor = _expected_oaf_descriptor(config)
    calls: list[dict[str, object]] = []
    transcribe_calls: list[int] = []
    close_calls: list[bool] = []

    def materialize(source, output_path, *, input_root, config):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"temporary canonical wav")
        return CanonicalAudio(
            path=output_path,
            source_audio_id=source.source_audio_id,
            source_audio_sha256=source.source_audio_sha256,
            input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
            input_audio_sha256=SHA_B,
            byte_length=88244,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=44100,
        )

    monkeypatch.setattr(run_module, "_materialize_oaf_full_mix", materialize)

    class FakeBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            transcribe_calls.append(int(audio.source_audio_id.split("/")[0]))
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
            close_calls.append(True)
            if close_failure:
                raise OafBackendError("worker close failed", code="worker_close_failed")

    def factory(**kwargs: object) -> FakeBackend:
        calls.append(kwargs)
        return FakeBackend()

    perf_values = iter((0.0, 0.25, 1.0, 1.5, 2.0, 2.25))
    outcome = run_oaf_corpus(
        OafCorpusRunRequest(
            reference_manifest_path=tmp_path / "reference.jsonl",
            timing_manifest_path=tmp_path / "timing.jsonl",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
        ),
        backend_factory=factory,
        perf_counter=lambda: next(perf_values),
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert calls == [
        {
            "input_root": tmp_path / "output" / "runs" / outcome.run_id / "inputs",
            "timeout_seconds": OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
        }
    ]
    assert transcribe_calls == [10, 20, 30]
    assert close_calls == [True]
    assert outcome.overall_status == ("partial" if close_failure else "complete")
    assert outcome.exit_code == (1 if close_failure else 0)
    assert outcome.run_path is not None and outcome.run_path.exists()
    assert outcome.reports_path is not None and outcome.reports_path.is_dir()
    snapshot = run_module.parse_oaf_corpus_run(outcome.run_path.read_bytes())
    assert snapshot["success_count"] == 3
    assert snapshot["failed_count"] == 0
    for row in snapshot["items"]:
        artifact_path = tmp_path / "output" / row["prediction_path"]
        artifact = read_prediction_artifact(artifact_path.read_bytes())
        assert artifact.prediction.audio.source_audio_id == row["source_audio_id"]
        assert artifact.prediction.audio.input_audio_sha256 == row["input_audio_sha256"]
    if close_failure:
        assert snapshot["close_error"] == {
            "code": "worker_close_failed",
            "message": "worker close failed",
        }


def test_run_oaf_corpus_finalizes_through_existing_scorer_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)
    score_calls: list[tuple[object, tuple[object, ...], tuple[str, ...]]] = []
    report_calls: list[tuple[object, Path]] = []

    def fake_score(identity, items, *, diagnostics_for=()):
        score_calls.append((identity, items, diagnostics_for))
        return "scored"

    def fake_reports(result, output_dir):
        report_calls.append((result, output_dir))
        return "reports"

    monkeypatch.setattr(run_module, "score_cohort", fake_score, raising=False)
    monkeypatch.setattr(run_module, "write_cohort_reports", fake_reports, raising=False)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert len(score_calls) == 1
    identity, items, diagnostics_for = score_calls[0]
    assert identity.cohort_id == outcome.run_id
    assert tuple(item.simfile_id for item in items) == ("10", "20", "30")
    assert diagnostics_for == ()
    assert report_calls == [("scored", tmp_path / "output" / "runs" / outcome.run_id / "reports")]


def test_finalization_reports_directory_failure_returns_canonical_fatal_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)
    real_mkdir = run_module.Path.mkdir

    def fail_reports_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == "reports":
            raise OSError("reports directory unavailable")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(run_module.Path, "mkdir", fail_reports_mkdir)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2
    assert outcome.run_id is None
    assert outcome.run_path is None
    assert outcome.reports_path is None
    assert outcome.success_count == 0
    assert outcome.failed_count == 0
    assert outcome.skipped_count == 0
    assert outcome.quarantined_count == 0


@pytest.mark.parametrize("failing_stage", ["score", "reports"])
def test_finalization_scoring_or_report_failure_returns_canonical_fatal_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_stage: str
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(f"{failing_stage} failed")

    monkeypatch.setattr(
        run_module,
        "score_cohort" if failing_stage == "score" else "write_cohort_reports",
        fail,
    )

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2
    assert outcome.run_id is None
    assert outcome.run_path is None
    assert outcome.reports_path is None
    assert outcome.success_count == 0
    assert outcome.failed_count == 0
    assert outcome.skipped_count == 0
    assert outcome.quarantined_count == 0


def test_scorer_rejects_raw_artifact_mismatch_before_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    first = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.run_path is not None
    snapshot = run_module.parse_oaf_corpus_run(first.run_path.read_bytes())
    row = next(row for row in snapshot["items"] if row["simfile_id"] == 10)
    artifact_path = tmp_path / "output" / row["prediction_path"]
    artifact = read_prediction_artifact(artifact_path.read_bytes())

    wrong_sha_row = dict(row)
    wrong_sha_row["prediction_artifact_sha256"] = SHA_B
    identity = run_module._cohort_identity_from_snapshot(snapshot)
    sha_mismatch_item = run_module._cohort_item_from_run_row(
        identity,
        wrong_sha_row,
        _reference_mapping(10),
        output_dir=tmp_path / "output",
    )
    assert sha_mismatch_item.status == "failed"
    assert sha_mismatch_item.failure_reason == "prediction_artifact_invalid"

    tampered_audio = replace(
        artifact.prediction.audio,
        source_audio_id="10/tampered.wav",
    )
    tampered_prediction = replace(artifact.prediction, audio=tampered_audio)
    artifact_path.write_bytes(render_prediction_artifact(tampered_prediction))

    item = run_module._cohort_item_from_run_row(
        identity,
        row,
        _reference_mapping(10),
        output_dir=tmp_path / "output",
    )

    assert item.status == "failed"
    assert item.failure_reason == "prediction_artifact_invalid"


def test_run_oaf_corpus_resume_regenerates_reports_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    first = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0

    class ResumeGuardBackend(HealthyBackend):
        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            raise AssertionError("resume scoring must not invoke inference")

    resumed = run_oaf_corpus(
        _request(tmp_path, resume=True),
        backend_factory=lambda **_: ResumeGuardBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert resumed.exit_code == 0
    assert resumed.run_id == first.run_id
    assert resumed.reports_path is not None
    assert resumed.reports_path.is_dir()


def test_run_oaf_corpus_resume_retains_projection_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    first = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=iter((0.0, 0.25, 1.0, 1.5, 2.0, 2.25)).__next__,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0
    assert first.aggregate_rtf == pytest.approx(1.0 / 3.0)
    assert first.projected_full_wall_time_sec == pytest.approx(1.0)

    class ResumeGuardBackend(HealthyBackend):
        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            raise AssertionError("resume projection must not invoke inference")

    resumed = run_oaf_corpus(
        _request(tmp_path, resume=True),
        backend_factory=lambda **_: ResumeGuardBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert resumed.exit_code == 0
    assert resumed.aggregate_rtf == pytest.approx(1.0 / 3.0)
    assert resumed.projected_full_wall_time_sec == pytest.approx(1.0)


def test_resume_recovers_prediction_published_before_checkpoint_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)
    request = replace(_request(tmp_path), include_simfile_ids=(10,))

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    real_checkpoint = run_module._write_snapshot_checkpoint
    checkpoint_calls = 0

    def crash_after_publication(*args: object, **kwargs: object) -> None:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if checkpoint_calls >= 3:
            raise KeyboardInterrupt("crash after prediction publication")
        real_checkpoint(*args, **kwargs)

    monkeypatch.setattr(run_module, "_write_snapshot_checkpoint", crash_after_publication)
    with pytest.raises(KeyboardInterrupt, match="crash after prediction publication"):
        run_oaf_corpus(
            request,
            backend_factory=lambda **_: HealthyBackend(),
            perf_counter=lambda: 0.0,
            clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
        )

    run_paths = list((tmp_path / "output" / "runs").glob("*/run.json"))
    assert len(run_paths) == 1
    prior_snapshot = run_module.parse_oaf_corpus_run(run_paths[0].read_bytes())
    assert "execution_disposition" not in prior_snapshot["items"][0]

    monkeypatch.setattr(run_module, "_write_snapshot_checkpoint", real_checkpoint)
    transcribe_calls: list[int] = []

    class ResumeBackend(HealthyBackend):
        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            transcribe_calls.append(int(audio.source_audio_id.split("/")[0]))
            raise AssertionError("published prediction must be recovered without inference")

    resumed = run_oaf_corpus(
        replace(request, resume=True),
        backend_factory=lambda **_: ResumeBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert resumed.exit_code == 0
    assert resumed.success_count == 1
    assert transcribe_calls == []
    assert resumed.run_path is not None
    resumed_snapshot = run_module.parse_oaf_corpus_run(resumed.run_path.read_bytes())
    assert resumed_snapshot["items"][0]["execution_disposition"] == "resumed"


@pytest.mark.parametrize(
    "include_ids, exclude_ids",
    [((999,), ())],
)
def test_scope_rejection_precedes_backend_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_ids: tuple[int, ...],
    exclude_ids: tuple[int, ...],
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    reference_manifest, timing_manifest = _fake_manifests()
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference_manifest)
    monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing_manifest)
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object):
        calls.append(kwargs)
        raise AssertionError("scope rejection must happen before backend construction")

    outcome = run_oaf_corpus(
        OafCorpusRunRequest(
            reference_manifest_path=tmp_path / "reference.jsonl",
            timing_manifest_path=tmp_path / "timing.jsonl",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            include_simfile_ids=include_ids,
            exclude_simfile_ids=exclude_ids,
        ),
        backend_factory=factory,
    )

    assert outcome.exit_code == 2
    assert calls == []


def test_malformed_native_descriptor_is_poison_and_stops_later_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)
    malformed_descriptor = replace(
        descriptor,
        payload={**descriptor.payload, "backend_id": "malformed-native-backend"},
    )
    transcribe_calls: list[int] = []
    close_calls: list[bool] = []

    class MalformedBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            simfile_id = int(audio.source_audio_id.split("/")[0])
            transcribe_calls.append(simfile_id)
            prediction_descriptor = malformed_descriptor if simfile_id == 20 else descriptor
            return NativePrediction(audio=audio, descriptor=prediction_descriptor, events=())

        def close(self) -> None:
            close_calls.append(True)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: MalformedBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert outcome.exit_code == 1
    assert transcribe_calls == [10, 20]
    assert close_calls == [True]
    assert outcome.run_path is not None
    snapshot = run_module.parse_oaf_corpus_run(outcome.run_path.read_bytes())
    rows = {row["simfile_id"]: row for row in snapshot["items"]}
    assert rows[20]["execution_disposition"] == "failed"
    assert rows[20]["runner_failure_code"] == "worker_protocol_failed"
    assert rows[30]["execution_disposition"] == "failed"
    assert rows[30]["runner_failure_code"] == "worker_protocol_failed"


ITEM_LOCAL_BACKEND_ERRORS = (
    ("inference_failed", "inference_failed"),
    ("invalid_request", "inference_failed"),
    ("input_path_invalid", "canonical_input_failed"),
    ("native_event_invalid", "inference_failed"),
)
POISON_BACKEND_ERRORS = (
    "worker_error",
    "worker_start_failed",
    "worker_ready_invalid",
    "worker_identity_invalid",
    "worker_response_invalid",
    "backend_closed",
)


@pytest.mark.parametrize("error_code,runner_code", ITEM_LOCAL_BACKEND_ERRORS)
def test_named_item_local_backend_errors_attempt_later_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
    runner_code: str,
) -> None:
    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)
    transcribe_calls: list[int] = []
    close_calls: list[bool] = []

    class ItemLocalBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            simfile_id = int(audio.source_audio_id.split("/")[0])
            transcribe_calls.append(simfile_id)
            if simfile_id == 20:
                raise OafBackendError("item-local failure", code=error_code)
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            close_calls.append(True)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: ItemLocalBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert outcome.exit_code == 1
    assert transcribe_calls == [10, 20, 30]
    assert close_calls == [True]
    assert outcome.run_path is not None
    import src.benchmark.oaf_corpus_run as run_module

    snapshot = run_module.parse_oaf_corpus_run(outcome.run_path.read_bytes())
    rows = {row["simfile_id"]: row for row in snapshot["items"]}
    assert rows[20]["execution_disposition"] == "failed"
    assert rows[20]["runner_failure_code"] == runner_code
    assert rows[30]["execution_disposition"] == "inferred"


@pytest.mark.parametrize(
    "error_code,runner_code",
    tuple(
        (
            code,
            (
                "backend_unavailable"
                if code
                in {
                    "worker_start_failed",
                    "worker_ready_invalid",
                    "worker_identity_invalid",
                }
                else "worker_protocol_failed"
            ),
        )
        for code in POISON_BACKEND_ERRORS
    )
    + (("future_worker_code", "worker_protocol_failed"),),
)
def test_named_and_unknown_poison_backend_errors_stop_later_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
    runner_code: str,
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)
    transcribe_calls: list[int] = []
    close_calls: list[bool] = []

    class PoisonBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            simfile_id = int(audio.source_audio_id.split("/")[0])
            transcribe_calls.append(simfile_id)
            if simfile_id == 20:
                raise OafBackendError("poison failure", code=error_code)
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            close_calls.append(True)

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: PoisonBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert outcome.exit_code == 1
    assert transcribe_calls == [10, 20]
    assert close_calls == [True]
    assert outcome.run_path is not None
    snapshot = run_module.parse_oaf_corpus_run(outcome.run_path.read_bytes())
    rows = {row["simfile_id"]: row for row in snapshot["items"]}
    assert rows[20]["execution_disposition"] == "failed"
    assert rows[20]["runner_failure_code"] == runner_code
    assert rows[30]["execution_disposition"] == "failed"
    assert rows[30]["runner_failure_code"] == "worker_protocol_failed"


def test_backend_closes_and_final_snapshot_survives_base_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)
    factory_input_root: list[Path] = []
    close_calls: list[bool] = []

    class InterruptBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            del audio
            raise KeyboardInterrupt("operator interrupted")

        def close(self) -> None:
            close_calls.append(True)

    def factory(**kwargs: object) -> InterruptBackend:
        factory_input_root.append(kwargs["input_root"])  # type: ignore[arg-type]
        return InterruptBackend()

    with pytest.raises(KeyboardInterrupt, match="operator interrupted"):
        run_oaf_corpus(
            _request(tmp_path),
            backend_factory=factory,
            perf_counter=lambda: 0.0,
            clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
        )

    assert close_calls == [True]
    assert factory_input_root
    run_path = factory_input_root[0].parent / "run.json"
    assert run_path.exists()
    snapshot = run_module.parse_oaf_corpus_run(run_path.read_bytes())
    assert snapshot["overall_status"] == "failed"
    assert "execution_disposition" not in snapshot["items"][0]
    assert not (factory_input_root[0] / "10" / "full-mix.wav").exists()


def test_resume_rejects_current_input_mismatch_and_nonresume_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    _, descriptor, materialize_calls = _install_fake_run_seams(monkeypatch, tmp_path)
    input_hash = {"value": SHA_B}

    def materialize(source, output_path, *, input_root, config):
        del input_root, config
        materialize_calls.append(int(source.source_audio_id.split("/")[0]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"temporary canonical wav")
        return CanonicalAudio(
            path=output_path,
            source_audio_id=source.source_audio_id,
            source_audio_sha256=source.source_audio_sha256,
            input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
            input_audio_sha256=input_hash["value"],
            byte_length=88244,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=44100,
        )

    monkeypatch.setattr(run_module, "_materialize_oaf_full_mix", materialize)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    first = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0
    assert first.run_path is not None
    first_snapshot = run_module.parse_oaf_corpus_run(first.run_path.read_bytes())
    original_paths = {
        row["simfile_id"]: tmp_path / "output" / row["prediction_path"]
        for row in first_snapshot["items"]
    }
    original_bytes = {simfile_id: path.read_bytes() for simfile_id, path in original_paths.items()}

    input_hash["value"] = "c" * 64
    materialize_calls.clear()
    factory_calls: list[bool] = []

    def should_not_construct(**_: object) -> HealthyBackend:
        factory_calls.append(True)
        raise AssertionError("mismatched resume artifacts must not transcribe")

    resumed = run_oaf_corpus(
        _request(tmp_path, resume=True),
        backend_factory=should_not_construct,
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert resumed.exit_code == 1
    assert materialize_calls == [10, 20, 30]
    assert factory_calls == []
    assert resumed.run_path is not None
    resumed_snapshot = run_module.parse_oaf_corpus_run(resumed.run_path.read_bytes())
    resumed_rows = {row["simfile_id"]: row for row in resumed_snapshot["items"]}
    assert all(
        row["runner_failure_code"] == "prediction_artifact_invalid" for row in resumed_rows.values()
    )
    for simfile_id, content in original_bytes.items():
        assert original_paths[simfile_id].read_bytes() == content

    conflict = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=should_not_construct,
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert conflict.exit_code == 1
    assert factory_calls == []
    assert conflict.run_path is not None
    conflict_snapshot = run_module.parse_oaf_corpus_run(conflict.run_path.read_bytes())
    conflict_rows = {row["simfile_id"]: row for row in conflict_snapshot["items"]}
    assert all(
        row["runner_failure_code"] == "prediction_output_conflict" for row in conflict_rows.values()
    )
    for simfile_id, content in original_bytes.items():
        assert original_paths[simfile_id].read_bytes() == content


def test_poison_stops_the_worker_and_resume_reuses_exact_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    reference_manifest, timing_manifest = _fake_manifests()
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference_manifest)
    monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing_manifest)
    empty_mappings = {
        row.view.simfile_id: _reference_mapping(row.view.simfile_id)
        for row in reference_manifest.rows
    }
    monkeypatch.setattr(
        run_module, "_preflight_reference_mappings", lambda *_args, **_kwargs: empty_mappings
    )
    monkeypatch.setattr(
        run_module,
        "_resolve_source_audio",
        lambda source, *_args, **_kwargs: ResolvedSourceAudio(
            path=tmp_path / "source.wav",
            source_audio_id=source["source_audio_key"],
            source_audio_sha256=SHA_A,
            duration_sec=1.0,
        ),
    )
    config = load_model_config()
    descriptor = _expected_oaf_descriptor(config)
    materialize_calls: list[int] = []

    def materialize(source, output_path, *, input_root, config):
        materialize_calls.append(int(source.source_audio_id.split("/")[0]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"temporary canonical wav")
        return CanonicalAudio(
            path=output_path,
            source_audio_id=source.source_audio_id,
            source_audio_sha256=source.source_audio_sha256,
            input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
            input_audio_sha256=SHA_B,
            byte_length=88244,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=44100,
        )

    monkeypatch.setattr(run_module, "_materialize_oaf_full_mix", materialize)
    first_calls: list[int] = []

    class PoisonBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            simfile_id = int(audio.source_audio_id.split("/")[0])
            first_calls.append(simfile_id)
            if simfile_id == 20:
                raise OafBackendError("worker failed", code="worker_error")
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    first = run_oaf_corpus(
        OafCorpusRunRequest(
            reference_manifest_path=tmp_path / "reference.jsonl",
            timing_manifest_path=tmp_path / "timing.jsonl",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
        ),
        backend_factory=lambda **_: PoisonBackend(),
        perf_counter=iter((0.0, 0.1, 0.2, 0.3)).__next__,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert first.exit_code == 1
    assert first_calls == [10, 20]
    assert materialize_calls == [10, 20]
    assert first.run_path is not None
    first_snapshot = run_module.parse_oaf_corpus_run(first.run_path.read_bytes())
    first_rows = {row["simfile_id"]: row for row in first_snapshot["items"]}
    assert first_rows[20]["execution_disposition"] == "failed"
    assert first_rows[30]["execution_disposition"] == "failed"
    assert first_rows[30]["runner_failure_code"] == "worker_protocol_failed"
    assert first.success_count == 1

    first_artifact_path = (
        tmp_path
        / "output"
        / "predictions"
        / "10"
        / SHA_A
        / descriptor.sha256
        / f"{run_module.inference_config_sha256(run_module.build_inference_config(config, descriptor, run_module.compute_model_lock_sha256(run_module._model_lock_path())))}.jsonl"
    )
    first_artifact = first_artifact_path.read_bytes()
    second_calls: list[int] = []

    class RecoveryBackend(PoisonBackend):
        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            second_calls.append(int(audio.source_audio_id.split("/")[0]))
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

    materialize_calls.clear()
    second = run_oaf_corpus(
        OafCorpusRunRequest(
            reference_manifest_path=tmp_path / "reference.jsonl",
            timing_manifest_path=tmp_path / "timing.jsonl",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            resume=True,
        ),
        backend_factory=lambda **_: RecoveryBackend(),
        perf_counter=iter((0.0, 0.1, 0.2, 0.3)).__next__,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert second.exit_code == 0
    assert second_calls == [20, 30]
    assert materialize_calls == [10, 20, 30]
    assert first_artifact_path.read_bytes() == first_artifact


def test_poison_accounts_for_outstanding_rows_in_snapshot_scorer_and_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    _, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)

    class PoisonBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            simfile_id = int(audio.source_audio_id.split("/")[0])
            if simfile_id == 20:
                raise OafBackendError("worker failed", code="worker_error")
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    outcome = run_oaf_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: PoisonBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert outcome.exit_code == 1
    assert outcome.success_count == 1
    assert outcome.failed_count == 2
    assert outcome.skipped_count == 0
    assert outcome.quarantined_count == 0
    assert outcome.run_path is not None
    snapshot = run_module.parse_oaf_corpus_run(outcome.run_path.read_bytes())
    rows = {row["simfile_id"]: row for row in snapshot["items"]}
    assert snapshot["success_count"] == 1
    assert snapshot["failed_count"] == 2
    assert all(row["execution_disposition"] for row in rows.values())
    assert rows[30]["runner_failure_code"] == "worker_protocol_failed"
    assert outcome.reports_path is not None
    report_summary = json.loads((outcome.reports_path / "summary.json").read_bytes())
    assert report_summary["population"] == {
        "failed_count": 2,
        "quarantined_count": 0,
        "reason_counts": {"backend_unavailable": 2},
        "skipped_count": 0,
        "success_count": 1,
        "total_count": 3,
    }


def test_expected_skips_and_quarantines_do_not_make_healthy_run_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    reference_manifest, descriptor, _ = _install_fake_run_seams(monkeypatch, tmp_path)
    quarantined_view = replace(
        reference_manifest.rows[2].view,
        eligibility_status="quarantined",
        eligibility_reason_codes=("unclassified_reference_lane",),
    )
    mixed_manifest = replace(
        reference_manifest,
        rows=(
            reference_manifest.rows[0],
            reference_manifest.rows[1],
            replace(reference_manifest.rows[2], view=quarantined_view),
        ),
    )
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: mixed_manifest)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return NativePrediction(audio=audio, descriptor=descriptor, events=())

        def close(self) -> None:
            return None

    outcome = run_oaf_corpus(
        replace(_request(tmp_path), exclude_simfile_ids=(20,)),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )

    assert outcome.overall_status == "complete"
    assert outcome.exit_code == 0
    assert outcome.success_count == 1
    assert outcome.failed_count == 0
    assert outcome.skipped_count == 1
    assert outcome.quarantined_count == 1
