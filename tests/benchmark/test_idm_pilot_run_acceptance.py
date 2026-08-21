from __future__ import annotations

from pathlib import Path

from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.idm_model import load_idm_model_lock
from src.benchmark.idm_pilot_run import IdmPilotRunRequest, run_idm_pilot
from src.benchmark.reference_set_manifest import LoadedReferenceSetManifest
from src.benchmark.separation_handoff import LoadedSeparationPilotManifest
from src.benchmark.taxonomy import OAF_PREDICTION_MAP_ID
from tests.benchmark.idm_pilot_fixtures import (
    CRUX_COMMIT,
    SHA_B,
    SHA_C,
    canonical_wav,
    loaded_reference_manifests,
    oaf_artifact,
    oaf_descriptor,
    sha256,
)


def test_idm_pilot_acceptance_entry_point_is_available(tmp_path: Path) -> None:
    request = IdmPilotRunRequest(
        separation_handoff_path=tmp_path / "handoff.jsonl",
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "timing.jsonl",
        separation_artifact_root=tmp_path / "separation",
        stem_cache_root=tmp_path / "stems",
        output_dir=tmp_path / "output",
        model_lock_path=tmp_path / "model.json",
        model_root=tmp_path / "model",
        runtime_python=tmp_path / "python",
    )
    assert callable(run_idm_pilot)
    assert request.output_dir == tmp_path / "output"


def _synthetic_handoff(
    tmp_path: Path,
) -> tuple[LoadedSeparationPilotManifest, LoadedReferenceSetManifest, object, dict[int, object]]:
    ids = tuple(range(20, 40))
    reference_sha = "1" * 64
    reference_version = "sha256:" + "2" * 64
    timing_sha = "3" * 64
    timing_version = "sha256:" + "4" * 64
    reference, timing, mappings = loaded_reference_manifests(
        ids=ids,
        reference_sha256=reference_sha,
        reference_version=reference_version,
        timing_sha256=timing_sha,
        timing_version=timing_version,
    )
    separation_root = tmp_path / "separation"
    stem_root = tmp_path / "stems"
    wav = canonical_wav()
    input_sha = sha256(wav)
    descriptor = oaf_descriptor()
    rows: list[dict[str, object]] = []
    for simfile_id in ids:
        source_id = f"{simfile_id}/audio.wav"
        input_relative = f"inputs/{simfile_id}/htdemucs.wav"
        stem_relative = f"derived/stems/{simfile_id}/drums.wav"
        input_path = separation_root / input_relative
        stem_path = stem_root / stem_relative
        input_path.parent.mkdir(parents=True, exist_ok=True)
        stem_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(wav)
        stem_path.write_bytes(wav)
        audio = CanonicalAudio(
            input_path,
            source_id,
            SHA_B,
            "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
            input_sha,
            len(wav),
            44100,
            1,
            2,
            32,
        )
        oaf_relative = f"predictions/{simfile_id}.jsonl"
        oaf_path = separation_root / oaf_relative
        oaf_content = oaf_artifact(audio)
        oaf_path.parent.mkdir(parents=True, exist_ok=True)
        oaf_path.write_bytes(oaf_content)
        successful = simfile_id < 38
        htdemucs: dict[str, object] = {
            "status": "resumed" if successful else "separation_failed",
            "failure_code": None if successful else "separator_timeout",
            "separator_lock_sha256": SHA_C,
            "input_view_id": "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
            "stem": (
                {
                    "path": stem_relative,
                    "sha256": sha256(wav),
                    "source_audio_sha256": SHA_B,
                    "separator_lock_sha256": SHA_C,
                }
                if successful
                else None
            ),
            "input": (
                {
                    "path": input_relative,
                    "input_view_id": "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
                    "input_audio_sha256": input_sha,
                    "source_audio_id": source_id,
                    "source_audio_sha256": SHA_B,
                }
                if successful
                else None
            ),
            "prediction": (
                {
                    "path": oaf_relative,
                    "artifact_sha256": sha256(oaf_content),
                    "source_audio_id": source_id,
                    "source_audio_sha256": SHA_B,
                    "input_view_id": "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
                    "input_audio_sha256": input_sha,
                }
                if successful
                else None
            ),
        }
        rows.append(
            {
                "schema_version": "crux.oaf-separation-pilot/v1",
                "separation_run_id": "oaf-separation-1234567890abcdef",
                "simfile_id": simfile_id,
                "source_row_sha256": SHA_C,
                "reference_manifest_sha256": reference_sha,
                "reference_manifest_version": reference_version,
                "reference_timing_manifest_sha256": timing_sha,
                "reference_timing_version": timing_version,
                "source_audio_id": source_id,
                "source_audio_sha256": SHA_B,
                "source_duration_sec": 1,
                "oaf_model_id": descriptor.payload["model_id"],
                "oaf_backend_descriptor_sha256": descriptor.sha256,
                "oaf_model_lock_sha256": "5" * 64,
                "oaf_checkpoint_archive_sha256": "6" * 64,
                "oaf_adapter_revision": "oaf-adapter-v1",
                "oaf_canonicalization_revision": "oaf-canonicalization-v1",
                "oaf_inference_config_sha256": "7" * 64,
                "oaf_prediction_map_version": OAF_PREDICTION_MAP_ID,
                "crux_commit": CRUX_COMMIT,
                "full_mix": {},
                "spleeter": {},
                "htdemucs": htdemucs,
                "comparison_artifacts": {},
                "decision": "use_htdemucs",
                "rationale": "synthetic immutable fixture",
            }
        )
    handoff = LoadedSeparationPilotManifest(
        manifest_sha256="8" * 64,
        corpus_version="sha256:" + "9" * 64,
        rows=tuple(rows),
    )
    return handoff, reference, timing, mappings


def test_synthetic_immutable_handoff_runs_both_complete_populations_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    import src.benchmark.idm_pilot_run as runner

    handoff, reference, timing, mappings = _synthetic_handoff(tmp_path)
    monkeypatch.setattr(runner, "load_separation_pilot_manifest", lambda _: handoff)
    monkeypatch.setattr(runner, "load_reference_set_manifest", lambda _: reference)
    monkeypatch.setattr(runner, "load_reference_timing_manifest", lambda _: timing)
    monkeypatch.setattr(runner, "preflight_reference_mappings", lambda *_args, **_kwargs: mappings)
    lock = load_idm_model_lock(Path("runtime/idm/model.json"))
    descriptor = runner.descriptor_for_lock(lock)
    calls: list[int] = []

    class FakeBackend:
        def descriptor(self):
            return descriptor

        def transcribe(self, audio):
            calls.append(int(audio.source_audio_id.split("/")[0]))
            return NativePrediction(
                audio,
                descriptor,
                (
                    NativeEvent(
                        0.25,
                        "KD",
                        4,
                        None,
                        {"frame_index": "43", "native_velocity": "1"},
                        0.9,
                        64,
                    ),
                ),
            )

        def close(self):
            return None

    request = IdmPilotRunRequest(
        separation_handoff_path=tmp_path / "handoff.jsonl",
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "manifests" / "timing.jsonl",
        separation_artifact_root=tmp_path / "separation",
        stem_cache_root=tmp_path / "stems",
        output_dir=tmp_path / "output",
        model_lock_path=Path("runtime/idm/model.json"),
        model_root=tmp_path / "model",
        runtime_python=tmp_path / "python",
        crux_commit=CRUX_COMMIT,
    )
    outcome = run_idm_pilot(
        request, backend_factory=lambda **_: FakeBackend(), perf_counter=lambda: 1.0
    )
    assert outcome.overall_status == "partial"
    assert outcome.success_count == 18
    assert outcome.failed_count == 2
    assert len(calls) == 18
    assert outcome.run_path is not None and outcome.run_path.exists()
    assert outcome.reports_path is not None
    assert (outcome.reports_path / "oaf" / "summary.json").exists()
    assert (outcome.reports_path / "idm" / "summary.json").exists()

    calls.clear()
    resumed = run_idm_pilot(
        request.__class__(**{**request.__dict__, "resume": True}),
        backend_factory=lambda **_: FakeBackend(),
        perf_counter=lambda: 1.0,
    )
    assert resumed.success_count == 18
    assert resumed.failed_count == 2
    assert calls == []
