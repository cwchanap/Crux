from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.benchmark.backend_identity import BackendDescriptor
from src.benchmark.backends import (
    BackendError,
    BackendVerification,
    CanonicalAudio,
    MidiDerivative,
    NativeEvent,
    NativePrediction,
    PublishedArtifact,
    SmokeCheck,
    TensorCoverageCheck,
)


def test_backend_records_preserve_complete_domain_data() -> None:
    error = BackendError(code="backend_failed", message="Backend failed verification.")
    prediction_artifact = PublishedArtifact(
        role="prediction",
        path=Path("predictions/result.json"),
        sha256="a" * 64,
    )
    sidecar_artifact = PublishedArtifact(
        role="sidecar",
        path=Path("predictions/result.sidecar.json"),
        sha256="b" * 64,
    )
    audio = CanonicalAudio(
        path=Path("audio/input.wav"),
        source_audio_id="song-1",
        source_audio_sha256="c" * 64,
        input_view_id="direct",
        input_audio_sha256="d" * 64,
        byte_length=96044,
        sample_rate=48000,
        channel_count=1,
        sample_width_bytes=2,
        audio_frame_count=48000,
    )
    descriptor = BackendDescriptor(
        payload={
            "descriptor_schema": "crux.test-backend-descriptor/v1",
            "backend_id": "test-backend-v1",
        },
        sha256="e" * 64,
    )
    event = NativeEvent(
        time_sec=0.125,
        native_class_id="kick",
        model_output_bin=2,
        native_midi_note=36,
        native_metadata={"source": "model", "optional": None},
        confidence=0.75,
        velocity_midi=100,
    )
    prediction = NativePrediction(
        audio=audio,
        descriptor=descriptor,
        events=(event,),
        backend_lock_sha256="f" * 64,
        runtime_lock_sha256=None,
        parameter_lock_sha256=None,
        model_artifact_set_sha256="1" * 64,
        upstream_source_commit="94529798",
        training_data_map_id="egmd-8hit",
    )
    tensor_coverage = TensorCoverageCheck(
        status="passed",
        required_count=10,
        restored_count=10,
        non_inference_count=2,
        required_inventory_sha256="2" * 64,
        non_inference_inventory_sha256="3" * 64,
        report=sidecar_artifact,
    )
    smoke = SmokeCheck(
        status="passed",
        audio_sha256="4" * 64,
        oracle_sha256="5" * 64,
        prediction=prediction_artifact,
    )
    verification = BackendVerification(
        status="verified",
        descriptor=descriptor,
        max_input_audio_frames=96000,
        backend_lock_sha256="f" * 64,
        runtime_lock_sha256=None,
        parameter_lock_sha256=None,
        seal_evidence_sha256="6" * 64,
        execution_attestation=sidecar_artifact,
        tensor_coverage=tensor_coverage,
        smoke=smoke,
        errors=(),
    )

    assert MidiDerivative(prediction_artifact, sidecar_artifact).midi.role == "prediction"
    assert prediction.events == (event,)
    assert verification.tensor_coverage.required_count == 10
    assert verification.smoke.prediction == prediction_artifact
    for record, field in [
        (error, "code"),
        (prediction_artifact, "role"),
        (MidiDerivative(prediction_artifact, sidecar_artifact), "midi"),
        (audio, "path"),
        (descriptor, "sha256"),
        (event, "time_sec"),
        (prediction, "audio"),
        (tensor_coverage, "status"),
        (smoke, "status"),
        (verification, "status"),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, "changed")
