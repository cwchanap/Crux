from __future__ import annotations

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
from src.benchmark.reference_set import ReferenceMappingDiagnostics, ReferenceMappingResult
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    ReferenceSetRowView,
)
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
            row.view.simfile_id: ReferenceMappingResult(
                mapped_events=(),
                common_events=(),
                diagnostics=ReferenceMappingDiagnostics({}, {}, 0),
            )
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
    if close_failure:
        assert snapshot["close_error"] == {
            "code": "worker_close_failed",
            "message": "worker close failed",
        }


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


def test_poison_stops_the_worker_and_resume_reuses_exact_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    reference_manifest, timing_manifest = _fake_manifests()
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference_manifest)
    monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing_manifest)
    empty_mappings = {
        row.view.simfile_id: ReferenceMappingResult(
            mapped_events=(),
            common_events=(),
            diagnostics=ReferenceMappingDiagnostics({}, {}, 0),
        )
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
    assert first.run_path is not None
    first_snapshot = run_module.parse_oaf_corpus_run(first.run_path.read_bytes())
    first_rows = {row["simfile_id"]: row for row in first_snapshot["items"]}
    assert first_rows[20]["execution_disposition"] == "failed"
    assert 30 not in first_rows or "execution_disposition" not in first_rows[30]
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
    assert first_artifact_path.read_bytes() == first_artifact
