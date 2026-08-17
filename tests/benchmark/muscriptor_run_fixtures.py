"""Shared MuScriptor corpus-run test helpers.

Factors the lock/manifest/mapping/request seams out of
``test_muscriptor_corpus_run.py`` so both that module and the acceptance test
import them from a non-test module instead of from a sibling test file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.benchmark.backend_identity import BackendDescriptor, canonical_json_bytes
from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.corpus_cache import ResolvedSourceAudio
from src.benchmark.muscriptor_corpus_run import (
    MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
    MuscriptorCorpusRunRequest,
    _expected_muscriptor_descriptor,
)
from src.benchmark.muscriptor_model import (
    MUSCRIPTOR_RELEASE_COMMIT,
    MuscriptorModelLock,
    model_lock_payload,
)
from src.benchmark.reference_set import ReferenceMappingResult, map_reference_events
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    ReferenceSetRowView,
)
from src.benchmark.reference_timing import NativeReferenceEvent
from src.benchmark.reference_timing_manifest import LoadedReferenceTimingManifest

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CHECKPOINT_REVISION = "d" * 40
COMMIT = "e" * 40
WEIGHTS = b"fake safetensors"
CONFIG = b'{"model":"fake"}'


def _lock() -> MuscriptorModelLock:
    checkpoint_sha = hashlib.sha256(WEIGHTS).hexdigest()
    config_sha = hashlib.sha256(CONFIG).hexdigest()
    return MuscriptorModelLock(
        package_name="muscriptor",
        package_version="0.3.0",
        upstream_source_commit=MUSCRIPTOR_RELEASE_COMMIT,
        code_license="MIT",
        weight_license="CC BY-NC 4.0",
        checkpoint_variant="medium",
        checkpoint_repo_id="MuScriptor/muscriptor-medium",
        checkpoint_revision=CHECKPOINT_REVISION,
        checkpoint_filename="model.safetensors",
        checkpoint_sha256=checkpoint_sha,
        checkpoint_byte_length=len(WEIGHTS),
        checkpoint_config_filename="config.json",
        checkpoint_config_sha256=config_sha,
        checkpoint_config_byte_length=len(CONFIG),
        model_id=f"muscriptor-medium-{CHECKPOINT_REVISION[:12]}-{checkpoint_sha[:12]}",
        device="cpu",
        dtype="float32",
        input_sample_rate_hz=16000,
        chunk_duration_sec=5.0,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=1.0,
        instruments=("drums",),
        batch_size=1,
        no_eos_is_ok=True,
        beam_size=1,
        prelude_forcing=True,
        native_output_space_id="muscriptor-drums-midi128-v1",
        native_metadata_schema_id="muscriptor-note-start-metadata-v1",
        training_data_map_id="muscriptor-training-data-v0.3.0",
    )


def _write_lock(tmp_path: Path) -> tuple[Path, Path, MuscriptorModelLock]:
    lock = _lock()
    lock_path = tmp_path / "model.json"
    lock_path.write_bytes(canonical_json_bytes(model_lock_payload(lock), trailing_newline=True))
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / lock.checkpoint_filename).write_bytes(WEIGHTS)
    (checkpoint_dir / lock.checkpoint_config_filename).write_bytes(CONFIG)
    return lock_path, checkpoint_dir, lock


def _manifests() -> tuple[LoadedReferenceSetManifest, LoadedReferenceTimingManifest]:
    rows = tuple(
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
                mapped_event_count=1,
                common_scored_event_count=1,
                ignored_event_count=0,
                unmapped_event_count=0,
                duplicate_common_event_count=0,
            ),
        )
        for simfile_id in (10, 20, 30)
    )
    return (
        LoadedReferenceSetManifest(
            manifest_sha256=SHA_A,
            corpus_version="reference-v1",
            source_reference_timing_manifest_sha256=SHA_B,
            source_reference_timing_version="timing-v1",
            rows=rows,
        ),
        LoadedReferenceTimingManifest(
            manifest_sha256=SHA_B,
            corpus_version="timing-v1",
            rows=(),
        ),
    )


def _mapping(simfile_id: int) -> ReferenceMappingResult:
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


def _request(tmp_path: Path, *, resume: bool = False) -> MuscriptorCorpusRunRequest:
    return MuscriptorCorpusRunRequest(
        reference_manifest_path=tmp_path / "reference.jsonl",
        timing_manifest_path=tmp_path / "timing.jsonl",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        resume=resume,
        crux_commit=COMMIT,
    )


def _install_seams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    input_hash: str = SHA_B,
) -> tuple[MuscriptorModelLock, BackendDescriptor, list[tuple[int, bool]], list[int]]:
    import src.benchmark.muscriptor_corpus_run as run_module

    lock_path, checkpoint_dir, lock = _write_lock(tmp_path)
    reference_manifest, timing_manifest = _manifests()
    descriptor = _expected_muscriptor_descriptor(lock)
    monkeypatch.setattr(run_module, "_model_lock_path", lambda: lock_path)
    monkeypatch.setattr(run_module, "_checkpoint_dir", lambda: checkpoint_dir)
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference_manifest)
    monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing_manifest)
    monkeypatch.setattr(
        run_module,
        "preflight_reference_mappings",
        lambda *_args, **_kwargs: {simfile_id: _mapping(simfile_id) for simfile_id in (10, 20, 30)},
    )
    resolve_calls: list[tuple[int, bool]] = []

    def resolve(source, *_args, **kwargs):
        simfile_id = int(source["source_audio_key"].split("/", 1)[0])
        resolve_calls.append((simfile_id, kwargs["load_body"]))
        return ResolvedSourceAudio(
            path=tmp_path / f"{simfile_id}.wav",
            source_audio_id=source["source_audio_key"],
            source_audio_sha256=SHA_A,
            duration_sec=1.0,
            content=b"source bytes" if kwargs["load_body"] else None,
        )

    monkeypatch.setattr(run_module, "resolve_source_audio", resolve)
    materialize_calls: list[int] = []

    def materialize(source, output_path, *, input_root, input_view_id, max_input_audio_frames):
        del input_root, input_view_id, max_input_audio_frames
        simfile_id = int(source.source_audio_id.split("/", 1)[0])
        materialize_calls.append(simfile_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"canonical wav")
        return CanonicalAudio(
            path=output_path,
            source_audio_id=source.source_audio_id,
            source_audio_sha256=source.source_audio_sha256,
            input_view_id=MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
            input_audio_sha256=input_hash,
            byte_length=46,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=1,
        )

    monkeypatch.setattr(run_module, "materialize_full_mix_audio", materialize)
    return lock, descriptor, resolve_calls, materialize_calls


def _prediction(audio: CanonicalAudio, descriptor: BackendDescriptor) -> NativePrediction:
    return NativePrediction(
        audio=audio,
        descriptor=descriptor,
        events=(
            NativeEvent(
                time_sec=0.25,
                native_class_id="drums:midi_36",
                model_output_bin=None,
                native_midi_note=36,
                native_metadata={"instrument_group": "drums"},
                confidence=None,
                velocity_midi=None,
            ),
        ),
    )
