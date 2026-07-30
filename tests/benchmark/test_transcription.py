from __future__ import annotations

# End-to-end fixtures intentionally construct the complete frozen report identities.
# pylint: disable=duplicate-code,too-many-lines
import os
import struct
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Callable, cast
from uuid import UUID

import pytest

from src.benchmark import transcription as transcription_module
from src.benchmark.backend_identity import (
    BackendDescriptor,
    JsonValue,
    canonical_json_bytes,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backend_registry import (
    HEURISTIC_BACKEND_ID,
    LEGACY_TF2_BACKEND_ID,
    OFFICIAL_BACKEND_ID,
    BackendLockUnavailable,
    BackendRegistry,
    default_backend_registry,
)
from src.benchmark.backend_reports import (
    UNAVAILABLE_BACKEND_REPORT_ID,
    OperationalReportPublicationError,
)
from src.benchmark.backends import (
    BackendError,
    BackendFatalFailure,
    BackendItemFailure,
    BackendVerification,
    CanonicalAudio,
    MidiDerivative,
    NativeEvent,
    NativePrediction,
    PublishedArtifact,
    SmokeCheck,
    TensorCoverageCheck,
)
from src.benchmark.input_view import load_direct_audio
from src.benchmark.prediction_artifact import (
    PredictionArtifact,
    PredictionArtifactError,
    publish_prediction_artifact,
)
from src.benchmark.transcription import (
    TranscribeOneOutcome,
    TranscribeOneRequest,
    run_transcribe_one,
)

FIXED_UTC = datetime(2026, 7, 27, 1, 2, 3, 456789, tzinfo=UTC)
FIXED_UUID = UUID("12345678-1234-4678-9234-567812345678")
HEURISTIC_DESCRIPTOR_PAYLOAD = {
    "adapter_source_manifest_sha256": "7" * 64,
    "architecture_id": "librosa-onset-centroid-zcr-v1",
    "backend_id": HEURISTIC_BACKEND_ID,
    "descriptor_schema": "crux.heuristic-backend-descriptor/v1",
    "model_id": "crux-heuristic-onset-nonmodel-v1",
    "native_metadata_schema_id": "crux-empty-native-metadata-v1",
    "native_output_space_id": "crux-heuristic-midi7-v1",
    "parameter_lock_sha256": "6" * 64,
    "prediction_schema": "crux.drum-prediction-events/v1",
}
HEURISTIC_DESCRIPTOR = BackendDescriptor(
    payload=HEURISTIC_DESCRIPTOR_PAYLOAD,
    sha256=sha256_hex(canonical_json_bytes(HEURISTIC_DESCRIPTOR_PAYLOAD)),
)
OAF_DESCRIPTOR_PAYLOAD = {
    "architecture_id": "magenta-oaf-model-tpu-drums-v1",
    "backend_id": OFFICIAL_BACKEND_ID,
    "backend_lock_sha256": "b" * 64,
    "descriptor_schema": "crux.transcription-backend-descriptor/v1",
    "model_artifact_set_sha256": "7" * 64,
    "model_id": "magenta-egmd-ckpt-569400-v1",
    "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
    "native_output_space_id": "magenta-oaf-midi88-a0-v1",
    "prediction_schema": "crux.drum-prediction-events/v1",
    "protocol_schema": "crux.transcription-runner/v1",
    "runtime_image_manifest_digest": f"sha256:{'8' * 64}",
    "runtime_lock_sha256": "c" * 64,
    "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
    "upstream_source_commit": "9" * 40,
}
OAF_DESCRIPTOR = BackendDescriptor(
    payload=OAF_DESCRIPTOR_PAYLOAD,
    sha256=sha256_hex(canonical_json_bytes(OAF_DESCRIPTOR_PAYLOAD)),
)


def canonical_wav(frame_count: int = 2) -> bytes:
    pcm = b"\x00\x00" * frame_count
    return (
        struct.pack("<4sI4s", b"RIFF", 36 + len(pcm), b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16)
        + struct.pack("<4sI", b"data", len(pcm))
        + pcm
    )


def direct_request(
    root: Path,
    *,
    backend_id: str | None = HEURISTIC_BACKEND_ID,
    output_path: Path | None = None,
    midi_output_path: Path | None = None,
    frame_count: int = 2,
) -> TranscribeOneRequest:
    audio_path = root / "audio" / "input.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(canonical_wav(frame_count))
    return TranscribeOneRequest(
        backend_id=backend_id,
        audio_path=audio_path,
        output_path=root / "predictions" / "result.jsonl" if output_path is None else output_path,
        source_audio_id="song-one",
        input_view_id="direct",
        input_view_manifest=None,
        midi_output_path=midi_output_path,
        reports_root=root / "artifacts" / "benchmark" / "backends",
    )


def derived_request(
    root: Path,
    *,
    source_sha256: str | None = None,
    input_sha256: str | None = None,
) -> TranscribeOneRequest:
    view_root = root / "derived"
    view_root.mkdir(parents=True, exist_ok=True)
    source = view_root / "source.raw"
    canonical = view_root / "input.wav"
    manifest = view_root / "manifest.json"
    source.write_bytes(b"source-audio")
    canonical.write_bytes(canonical_wav())
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "input_audio_path": "input.wav",
                "input_audio_sha256": (
                    sha256_hex(canonical.read_bytes()) if input_sha256 is None else input_sha256
                ),
                "input_view_id": "canonical-44k1-mono",
                "schema": "crux.input-view-manifest/v1",
                "source_audio_id": "song-derived",
                "source_audio_sha256": (
                    sha256_hex(source.read_bytes()) if source_sha256 is None else source_sha256
                ),
                "source_path": "source.raw",
            },
            trailing_newline=True,
        )
    )
    return TranscribeOneRequest(
        backend_id=HEURISTIC_BACKEND_ID,
        audio_path=canonical,
        output_path=root / "predictions" / "derived.jsonl",
        source_audio_id=None,
        input_view_id=None,
        input_view_manifest=manifest,
        midi_output_path=None,
        reports_root=root / "artifacts" / "benchmark" / "backends",
    )


def heuristic_verification(
    *,
    status: str = "verified",
    errors: tuple[BackendError, ...] = (),
) -> BackendVerification:
    return BackendVerification(
        status=cast(object, status),  # type: ignore[arg-type]
        descriptor=HEURISTIC_DESCRIPTOR,
        max_input_audio_frames=None,
        backend_lock_sha256=None,
        runtime_lock_sha256=None,
        parameter_lock_sha256="6" * 64,
        seal_evidence_sha256=None,
        execution_attestation=PublishedArtifact(
            role="execution_attestation",
            path=Path("artifacts/benchmark/backends/heuristic/attestation.json"),
            sha256="e" * 64,
        ),
        tensor_coverage=TensorCoverageCheck(
            status="not_applicable",
            required_count=0,
            restored_count=0,
            non_inference_count=0,
            required_inventory_sha256=None,
            non_inference_inventory_sha256=None,
            report=None,
        ),
        smoke=SmokeCheck(
            status="not_applicable",
            audio_sha256=None,
            oracle_sha256=None,
            prediction=None,
        ),
        errors=errors,
    )


def oaf_verification(
    *,
    max_input_audio_frames: int | None = 2,
    status: str = "verified",
    errors: tuple[BackendError, ...] = (),
) -> BackendVerification:
    return BackendVerification(
        status=cast(object, status),  # type: ignore[arg-type]
        descriptor=OAF_DESCRIPTOR,
        max_input_audio_frames=max_input_audio_frames,
        backend_lock_sha256="b" * 64,
        runtime_lock_sha256="c" * 64,
        parameter_lock_sha256=None,
        seal_evidence_sha256="d" * 64,
        execution_attestation=PublishedArtifact(
            role="execution_attestation",
            path=Path("artifacts/benchmark/backends/oaf/attestation.json"),
            sha256="e" * 64,
        ),
        tensor_coverage=TensorCoverageCheck(
            status="passed",
            required_count=78,
            restored_count=78,
            non_inference_count=52,
            required_inventory_sha256="f" * 64,
            non_inference_inventory_sha256="1" * 64,
            report=PublishedArtifact(
                role="tensor_coverage",
                path=Path("artifacts/benchmark/backends/oaf/tensor.json"),
                sha256="2" * 64,
            ),
        ),
        smoke=SmokeCheck(
            status="passed",
            audio_sha256="3" * 64,
            oracle_sha256="4" * 64,
            prediction=PublishedArtifact(
                role="prediction",
                path=Path("artifacts/benchmark/backends/oaf/smoke.jsonl"),
                sha256="5" * 64,
            ),
        ),
        errors=errors,
    )


def heuristic_prediction(
    audio: CanonicalAudio,
    *,
    event_time: float = 0.5,
    confidence: float | None = None,
) -> NativePrediction:
    return NativePrediction(
        audio=audio,
        descriptor=HEURISTIC_DESCRIPTOR,
        events=(
            NativeEvent(
                time_sec=event_time,
                native_class_id="midi_36",
                model_output_bin=None,
                native_midi_note=36,
                native_metadata={},
                confidence=confidence,
                velocity_midi=80,
            ),
        ),
        backend_lock_sha256=None,
        runtime_lock_sha256=None,
        parameter_lock_sha256="6" * 64,
        model_artifact_set_sha256=None,
        upstream_source_commit=None,
        training_data_map_id=None,
    )


def oaf_prediction(audio: CanonicalAudio) -> NativePrediction:
    return NativePrediction(
        audio=audio,
        descriptor=OAF_DESCRIPTOR,
        events=(
            NativeEvent(
                time_sec=0.5,
                native_class_id="midi_36",
                model_output_bin=15,
                native_midi_note=36,
                native_metadata={"upstream_8hit_group_id": "kick"},
                confidence=0.75,
                velocity_midi=100,
            ),
        ),
        backend_lock_sha256="b" * 64,
        runtime_lock_sha256="c" * 64,
        parameter_lock_sha256=None,
        model_artifact_set_sha256="7" * 64,
        upstream_source_commit="9" * 40,
        training_data_map_id="magenta-egmd-data-8hit-94529798-v1",
    )


def write_valid_midi_derivative(
    prediction: PredictionArtifact,
    destination: Path,
) -> MidiDerivative:
    midi_bytes = b"MThd\x00\x00\x00\x06"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(midi_bytes)
    midi_sha256 = sha256_hex(midi_bytes)
    sidecar_path = destination.with_suffix(".sidecar.json")
    sidecar_bytes = canonical_json_bytes(
        {
            "channel": 9,
            "midi_sha256": midi_sha256,
            "note_duration_seconds": Decimal("0.05"),
            "prediction_sha256": prediction.artifact_sha256,
            "program": 0,
            "schema": "crux.prediction-midi-derivative/v1",
            "tempo_qpm": Decimal("120"),
            "ticks_per_quarter": 220,
            "writer_distribution": {
                "filename": "mido-1.3.3-py3-none-any.whl",
                "name": "mido",
                "sha256": ("01033c9b10b049e4436fca2762194ca839b09a4334091dd3c34e7f4ae674fd8a"),
                "version": "1.3.3",
            },
            "writer_lock_sha256": "b" * 64,
        },
        trailing_newline=True,
    )
    sidecar_path.write_bytes(sidecar_bytes)
    return MidiDerivative(
        midi=PublishedArtifact(
            role="midi",
            path=destination,
            sha256=midi_sha256,
        ),
        sidecar=PublishedArtifact(
            role="prediction_midi_sidecar",
            path=sidecar_path,
            sha256=sha256_hex(sidecar_bytes),
        ),
    )


class FakeBackend:
    def __init__(
        self,
        verification: object | BaseException,
        *,
        prediction_factory: Callable[[CanonicalAudio], object] = heuristic_prediction,
        transcription_error: BaseException | None = None,
        close_error: BaseException | None = None,
        verify_hook: Callable[[], None] | None = None,
        phases: list[str] | None = None,
    ) -> None:
        self.verification = verification
        self.prediction_factory = prediction_factory
        self.transcription_error = transcription_error
        self.close_error = close_error
        self.verify_hook = verify_hook
        self.phases = [] if phases is None else phases
        self.transcribe_calls = 0
        self.close_calls = 0

    def descriptor(self) -> BackendDescriptor:
        if isinstance(self.verification, BackendVerification):
            assert self.verification.descriptor is not None
            return self.verification.descriptor
        return HEURISTIC_DESCRIPTOR

    def verify(self) -> BackendVerification:
        self.phases.append("verify")
        if self.verify_hook is not None:
            self.verify_hook()
        if isinstance(self.verification, BaseException):
            raise self.verification
        return cast(BackendVerification, self.verification)

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        self.phases.append("transcribe")
        self.transcribe_calls += 1
        if self.transcription_error is not None:
            raise self.transcription_error
        return cast(NativePrediction, self.prediction_factory(audio))

    def close(self) -> None:
        self.phases.append("close")
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def registry_for(
    backend: FakeBackend,
    *,
    backend_id: str = HEURISTIC_BACKEND_ID,
    phases: list[str] | None = None,
) -> BackendRegistry:
    def factory() -> FakeBackend:
        if phases is not None:
            phases.append("create")
        return backend

    return BackendRegistry(default_backend_id=backend_id, factories={backend_id: factory})


def report_payload(outcome: TranscribeOneOutcome) -> dict[str, JsonValue]:
    content = outcome.report_artifact.path.read_bytes()
    assert content.endswith(b"\n")
    payload = strict_json_loads(content[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    return payload


def test_transcribe_one_uses_real_common_components_in_exact_phase_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    phases: list[str] = []
    backend = FakeBackend(heuristic_verification(), phases=phases)
    request = direct_request(tmp_path)
    real_load = transcription_module.load_direct_audio
    real_render = transcription_module.render_prediction_artifact
    real_read = transcription_module.read_prediction_artifact
    real_publish_prediction = transcription_module.publish_prediction_artifact
    real_publish_report = transcription_module.publish_operational_report

    def record_load(*args: object, **kwargs: object) -> CanonicalAudio:
        phases.append("load")
        return real_load(*args, **kwargs)  # type: ignore[arg-type]

    def record_render(prediction: NativePrediction) -> bytes:
        phases.append("render")
        return real_render(prediction)

    def record_read(content: bytes) -> PredictionArtifact:
        phases.append("strict_read")
        return real_read(content)

    def record_publish_prediction(
        path: Path,
        prediction: NativePrediction,
    ) -> PublishedArtifact:
        phases.append("publish_prediction")
        return real_publish_prediction(path, prediction)

    def record_publish_report(*args: object, **kwargs: object) -> PublishedArtifact:
        phases.append("publish_report")
        return real_publish_report(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(transcription_module, "load_direct_audio", record_load)
    monkeypatch.setattr(transcription_module, "render_prediction_artifact", record_render)
    monkeypatch.setattr(transcription_module, "read_prediction_artifact", record_read)
    monkeypatch.setattr(
        transcription_module,
        "publish_prediction_artifact",
        record_publish_prediction,
    )
    monkeypatch.setattr(
        transcription_module,
        "publish_operational_report",
        record_publish_report,
    )

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend, phases=phases),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "complete"
    assert outcome.exit_code == 0
    assert phases == [
        "create",
        "verify",
        "load",
        "transcribe",
        "render",
        "strict_read",
        "publish_prediction",
        "publish_report",
        "close",
    ]
    payload = report_payload(outcome)
    assert payload["verification_report"] is None
    assert payload["items"][0]["prediction"]["path"] == "predictions/result.jsonl"  # type: ignore[index]


@pytest.mark.parametrize(
    ("verification", "expected_status", "expected_exit"),
    [
        (
            heuristic_verification(
                status="environment_unsupported",
                errors=(
                    BackendError(
                        code="environment_unsupported",
                        message="This environment is unsupported.",
                    ),
                ),
            ),
            "environment_unsupported",
            1,
        ),
        (
            heuristic_verification(
                status="failed",
                errors=(
                    BackendError(
                        code="verification_failed",
                        message="Backend verification failed.",
                    ),
                ),
            ),
            "failed",
            2,
        ),
    ],
)
def test_verification_finishes_before_prediction_or_midi_output_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    verification: BackendVerification,
    expected_status: str,
    expected_exit: int,
) -> None:
    monkeypatch.chdir(tmp_path)
    midi_path = tmp_path / "midi" / "result.mid"
    request = direct_request(tmp_path, midi_output_path=midi_path)
    backend = FakeBackend(verification)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == expected_status
    assert outcome.exit_code == expected_exit
    assert not request.output_path.exists()
    assert not midi_path.exists()
    assert report_payload(outcome)["status"] == expected_status
    assert backend.transcribe_calls == 0
    assert backend.close_calls == 1


def test_environment_exception_during_verification_is_backend_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    error = BackendFatalFailure(
        BackendError(
            code="runner_start_failed",
            message="The backend runner did not start.",
        )
    )
    backend = FakeBackend(error)
    request = direct_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert report_payload(outcome)["errors"] == [
        {
            "code": "runner_start_failed",
            "message": "The backend runner did not start.",
        }
    ]
    assert backend.close_calls == 1


def test_unexpected_verification_exception_fails_closed_and_closes_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(RuntimeError("/private/verification/details"))
    request = direct_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert report_payload(outcome)["errors"] == [
        {
            "code": "backend_failure",
            "message": "The backend failed unexpectedly.",
        }
    ]
    assert "/private/verification/details" not in outcome.report_artifact.path.read_text()
    assert backend.close_calls == 1


def test_verified_backend_without_established_identity_fails_with_typed_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    verification = replace(
        heuristic_verification(),
        descriptor=None,
        execution_attestation=None,
    )
    backend = FakeBackend(verification)
    request = direct_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    payload = report_payload(outcome)
    assert payload["descriptor"] is None
    assert payload["errors"][0]["code"] == "invalid_backend_verification"  # type: ignore[index]
    assert backend.transcribe_calls == 0
    assert backend.close_calls == 1


@pytest.mark.parametrize("frame_bound", [None, 0, -1, False])
def test_locked_backend_requires_positive_integer_audio_frame_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frame_bound: int | None,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(
        oaf_verification(max_input_audio_frames=frame_bound),
        prediction_factory=oaf_prediction,
    )
    request = direct_request(tmp_path, backend_id=OFFICIAL_BACKEND_ID)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend, backend_id=OFFICIAL_BACKEND_ID),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert report_payload(outcome)["errors"][0]["code"] == "invalid_backend_verification"  # type: ignore[index]
    assert backend.transcribe_calls == 0
    assert backend.close_calls == 1


def test_locked_backend_applies_positive_audio_frame_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(
        oaf_verification(max_input_audio_frames=2),
        prediction_factory=oaf_prediction,
    )
    request = direct_request(
        tmp_path,
        backend_id=OFFICIAL_BACKEND_ID,
        frame_count=3,
    )

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend, backend_id=OFFICIAL_BACKEND_ID),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert not request.output_path.exists()
    assert backend.transcribe_calls == 0
    assert backend.close_calls == 1


def test_locked_backend_accepts_audio_at_exact_positive_frame_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(
        oaf_verification(max_input_audio_frames=2),
        prediction_factory=oaf_prediction,
    )
    request = direct_request(
        tmp_path,
        backend_id=OFFICIAL_BACKEND_ID,
        frame_count=2,
    )

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend, backend_id=OFFICIAL_BACKEND_ID),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "complete"
    assert outcome.exit_code == 0
    assert request.output_path.exists()
    assert backend.close_calls == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"source_audio_id": None},
        {"input_view_id": None},
        {"source_audio_id": None, "input_view_id": None},
        {"input_view_manifest": Path("manifest.json")},
        {"source_audio_id": None, "input_view_manifest": Path("manifest.json")},
    ],
)
def test_direct_and_derived_input_modes_are_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(heuristic_verification())
    request = replace(direct_request(tmp_path), **changes)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert not request.output_path.exists()
    assert report_payload(outcome)["errors"][0]["code"] == "input_contract_failed"  # type: ignore[index]
    assert backend.transcribe_calls == 0
    assert backend.close_calls == 1


def test_derived_input_loads_and_rehashes_both_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(heuristic_verification())
    request = derived_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    item = report_payload(outcome)["items"][0]  # type: ignore[index]
    assert outcome.status == "complete"
    assert item["source_audio_id"] == "song-derived"
    assert item["input_view_id"] == "canonical-44k1-mono"
    assert backend.close_calls == 1


@pytest.mark.parametrize(
    ("source_sha256", "input_sha256"),
    [
        ("0" * 64, None),
        (None, "0" * 64),
    ],
)
def test_derived_input_hash_failures_are_item_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_sha256: str | None,
    input_sha256: str | None,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(heuristic_verification())
    request = derived_request(
        tmp_path,
        source_sha256=source_sha256,
        input_sha256=input_sha256,
    )

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert report_payload(outcome)["errors"][0]["code"] == "input_contract_failed"  # type: ignore[index]
    assert backend.transcribe_calls == 0
    assert backend.close_calls == 1


def test_typed_item_inference_failure_publishes_failed_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(
        heuristic_verification(),
        transcription_error=BackendItemFailure(
            BackendError(
                code="inference_failed",
                message="Inference failed for this item.",
            )
        ),
    )
    request = direct_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    item = report_payload(outcome)["items"][0]  # type: ignore[index]
    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert item["status"] == "failed"
    assert item["prediction"] is None
    assert item["errors"] == [
        {
            "code": "inference_failed",
            "message": "Inference failed for this item.",
        }
    ]
    assert backend.close_calls == 1


def test_typed_backend_process_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(
        heuristic_verification(),
        transcription_error=BackendFatalFailure(
            BackendError(
                code="runner_process_died",
                message="The backend runner stopped.",
            )
        ),
    )
    request = direct_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert report_payload(outcome)["errors"] == [
        {
            "code": "runner_process_died",
            "message": "The backend runner stopped.",
        }
    ]
    assert not request.output_path.exists()
    assert backend.close_calls == 1


def test_unexpected_backend_exception_fails_closed_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(
        heuristic_verification(),
        transcription_error=RuntimeError("/private/model/path and traceback"),
    )
    request = direct_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    payload = report_payload(outcome)
    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert payload["errors"] == [
        {
            "code": "backend_failure",
            "message": "The backend failed unexpectedly.",
        }
    ]
    assert "/private/model/path" not in outcome.report_artifact.path.read_text()
    assert backend.close_calls == 1


def test_invalid_prediction_is_item_artifact_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(
        heuristic_verification(),
        prediction_factory=lambda audio: heuristic_prediction(audio, confidence=float("nan")),
    )
    request = direct_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert report_payload(outcome)["items"][0]["errors"][0]["code"] == (  # type: ignore[index]
        "prediction_artifact_failed"
    )
    assert not request.output_path.exists()
    assert backend.close_calls == 1


def test_strict_round_trip_failure_occurs_before_prediction_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(heuristic_verification())
    request = direct_request(tmp_path)

    def reject_round_trip(content: bytes) -> PredictionArtifact:
        del content
        raise PredictionArtifactError("round trip rejected")

    monkeypatch.setattr(
        transcription_module,
        "read_prediction_artifact",
        reject_round_trip,
    )

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "partial"
    assert not request.output_path.exists()
    assert backend.close_calls == 1


def test_immutable_prediction_publication_failure_preserves_prior_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = direct_request(tmp_path)
    audio = load_direct_audio(
        request.audio_path,
        source_audio_id="song-one",
        input_view_id="direct",
        max_input_audio_frames=None,
    )
    publish_prediction_artifact(
        request.output_path,
        heuristic_prediction(audio, event_time=0.25),
    )
    prior = request.output_path.read_bytes()
    backend = FakeBackend(heuristic_verification())

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert request.output_path.read_bytes() == prior
    assert report_payload(outcome)["items"][0]["errors"][0]["code"] == (  # type: ignore[index]
        "prediction_publication_failed"
    )
    assert backend.close_calls == 1


def test_complete_jsonl_only_execution_reports_null_midi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(heuristic_verification())
    request = direct_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    item = report_payload(outcome)["items"][0]  # type: ignore[index]
    assert outcome.status == "complete"
    assert outcome.exit_code == 0
    assert item["status"] == "complete"
    assert item["midi"] is None
    assert backend.close_calls == 1


def test_requested_midi_without_writer_retains_jsonl_as_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(heuristic_verification())
    request = direct_request(
        tmp_path,
        midi_output_path=tmp_path / "predictions" / "result.mid",
    )

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    item = report_payload(outcome)["items"][0]  # type: ignore[index]
    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert request.output_path.exists()
    assert item["status"] == "incomplete"
    assert item["prediction"]["path"] == "predictions/result.jsonl"
    assert item["midi"] is None
    assert item["errors"] == [
        {
            "code": "midi_derivation_failed",
            "message": "MIDI was not produced.",
        }
    ]
    assert backend.close_calls == 1


def test_midi_writer_reports_midi_while_sidecar_remains_prediction_hash_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(heuristic_verification())
    midi_path = tmp_path / "predictions" / "result.mid"
    request = direct_request(tmp_path, midi_output_path=midi_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
        midi_writer=write_valid_midi_derivative,
    )

    item = report_payload(outcome)["items"][0]  # type: ignore[index]
    prediction_sha256 = item["prediction"]["sha256"]
    sidecar = strict_json_loads(midi_path.with_suffix(".sidecar.json").read_bytes())
    assert outcome.status == "complete"
    assert item["midi"]["path"] == "predictions/result.mid"
    assert isinstance(sidecar, dict)
    assert sidecar["prediction_sha256"] == prediction_sha256
    assert backend.close_calls == 1


def test_midi_writer_failure_retains_prediction_and_reports_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(heuristic_verification())
    request = direct_request(
        tmp_path,
        midi_output_path=tmp_path / "predictions" / "result.mid",
    )

    def fail_writer(
        prediction: PredictionArtifact,
        destination: Path,
    ) -> MidiDerivative:
        del prediction, destination
        raise ValueError("writer internals")

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
        midi_writer=fail_writer,
    )

    item = report_payload(outcome)["items"][0]  # type: ignore[index]
    assert outcome.status == "partial"
    assert request.output_path.exists()
    assert item["status"] == "incomplete"
    assert item["errors"][0]["code"] == "midi_derivation_failed"
    assert "writer internals" not in outcome.report_artifact.path.read_text()
    assert backend.close_calls == 1


def test_absolute_and_relative_outputs_under_repository_root_are_reported_relatively(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    for index, output_path in enumerate(
        [
            tmp_path / "absolute" / "prediction.jsonl",
            Path("relative/prediction.jsonl"),
        ]
    ):
        backend = FakeBackend(heuristic_verification())
        request = direct_request(tmp_path, output_path=output_path)
        outcome = run_transcribe_one(
            request,
            registry=registry_for(backend),
            now=FIXED_UTC,
            run_id=UUID(f"12345678-1234-4678-9234-56781234567{index + 8}"),
        )
        prediction = report_payload(outcome)["items"][0]["prediction"]  # type: ignore[index]
        expected = (
            "absolute/prediction.jsonl"
            if output_path.is_absolute()
            else "relative/prediction.jsonl"
        )
        assert prediction["path"] == expected
        assert (tmp_path / expected).exists()
        assert backend.close_calls == 1


@pytest.mark.parametrize("outside_field", ["output_path", "midi_output_path"])
def test_output_paths_outside_repository_root_are_rejected_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outside_field: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-{outside_field}"
    backend = FakeBackend(heuristic_verification())
    request = direct_request(tmp_path)
    request = replace(request, **{outside_field: outside})

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert not request.output_path.exists()
    assert not outside.exists()
    assert report_payload(outcome)["errors"][0]["code"] == "output_path_invalid"  # type: ignore[index]
    assert backend.transcribe_calls == 0
    assert backend.close_calls == 1


def test_unknown_backend_publishes_typed_failure_in_report_only_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = direct_request(tmp_path, backend_id="../../unknown")
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        factories={},
    )

    outcome = run_transcribe_one(
        request,
        registry=registry,
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    payload = report_payload(outcome)
    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert outcome.report_artifact.path.parent.parent.name == UNAVAILABLE_BACKEND_REPORT_ID
    assert payload["descriptor"] is None
    assert payload["items"] == []
    assert payload["errors"] == [
        {
            "code": "backend_unavailable",
            "message": "Backend is unavailable.",
        }
    ]
    assert "../../unknown" not in outcome.report_artifact.path.as_posix()


def test_registered_legacy_backend_rejection_publishes_known_unavailable_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = direct_request(tmp_path, backend_id=LEGACY_TF2_BACKEND_ID)

    outcome = run_transcribe_one(
        request,
        registry=default_backend_registry(),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert outcome.report_artifact.path.parent.parent.name == LEGACY_TF2_BACKEND_ID
    assert report_payload(outcome)["errors"][0]["code"] == "backend_unavailable"  # type: ignore[index]


def test_unexpected_registered_factory_exception_fails_closed_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    fallback_calls = 0

    def unexpected() -> FakeBackend:
        raise RuntimeError("/private/factory/details")

    def fallback() -> FakeBackend:
        nonlocal fallback_calls
        fallback_calls += 1
        return FakeBackend(heuristic_verification())

    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        factories={
            OFFICIAL_BACKEND_ID: unexpected,
            HEURISTIC_BACKEND_ID: fallback,
        },
    )
    request = direct_request(tmp_path, backend_id=None)

    outcome = run_transcribe_one(
        request,
        registry=registry,
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert report_payload(outcome)["errors"] == [
        {
            "code": "backend_failure",
            "message": "The backend failed unexpectedly.",
        }
    ]
    assert "/private/factory/details" not in outcome.report_artifact.path.read_text()
    assert fallback_calls == 0


@pytest.mark.parametrize(
    "factory_error",
    [ImportError("missing"), BackendLockUnavailable("lock")],
)
def test_known_unavailable_factory_publishes_failed_report_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    factory_error: BaseException,
) -> None:
    monkeypatch.chdir(tmp_path)
    fallback_calls = 0

    def unavailable() -> FakeBackend:
        raise factory_error

    def fallback() -> FakeBackend:
        nonlocal fallback_calls
        fallback_calls += 1
        return FakeBackend(heuristic_verification())

    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        factories={
            OFFICIAL_BACKEND_ID: unavailable,
            HEURISTIC_BACKEND_ID: fallback,
        },
    )
    request = direct_request(tmp_path, backend_id=None)

    outcome = run_transcribe_one(
        request,
        registry=registry,
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert outcome.report_artifact.path.parent.parent.name == OFFICIAL_BACKEND_ID
    assert report_payload(outcome)["errors"][0]["code"] == "backend_unavailable"  # type: ignore[index]
    assert fallback_calls == 0


def test_execution_report_publication_failure_propagates_without_fake_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(heuristic_verification())
    request = direct_request(tmp_path)

    def fail_report(*args: object, **kwargs: object) -> PublishedArtifact:
        del args, kwargs
        raise OperationalReportPublicationError("operational_report_publication_failed")

    monkeypatch.setattr(
        transcription_module,
        "publish_operational_report",
        fail_report,
    )

    with pytest.raises(
        OperationalReportPublicationError,
        match="operational_report_publication_failed",
    ):
        run_transcribe_one(
            request,
            registry=registry_for(backend),
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert request.output_path.exists()
    assert backend.close_calls == 1


@pytest.mark.parametrize(
    "case",
    [
        "unexpected_type",
        "verified_with_errors",
        "failed_without_errors",
        "descriptor_schema",
        "descriptor_digest",
        "descriptor_backend_id",
        "attestation_role",
        "heuristic_lock",
        "heuristic_tensor",
        "heuristic_smoke",
        "heuristic_frame_bound",
        "oaf_tensor_counts",
        "oaf_noninference_count",
        "oaf_tensor_artifact",
        "oaf_smoke_relationship",
    ],
)
def test_malformed_verification_fails_closed_before_input_or_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    selected_backend_id = HEURISTIC_BACKEND_ID
    verification: object = heuristic_verification()
    if case == "unexpected_type":
        verification = None
    elif case == "verified_with_errors":
        verification = replace(
            cast(BackendVerification, verification),
            errors=(BackendError(code="verification_failed", message="Verification failed."),),
        )
    elif case == "failed_without_errors":
        verification = heuristic_verification(status="failed")
    elif case in {"descriptor_schema", "descriptor_digest", "descriptor_backend_id"}:
        descriptor_payload = dict(HEURISTIC_DESCRIPTOR_PAYLOAD)
        if case == "descriptor_schema":
            descriptor_payload["descriptor_schema"] = "crux.unknown/v1"
        elif case == "descriptor_backend_id":
            descriptor_payload["backend_id"] = OFFICIAL_BACKEND_ID
        descriptor = BackendDescriptor(
            payload=descriptor_payload,
            sha256=(
                "0" * 64
                if case == "descriptor_digest"
                else sha256_hex(canonical_json_bytes(descriptor_payload))
            ),
        )
        verification = replace(cast(BackendVerification, verification), descriptor=descriptor)
    elif case == "attestation_role":
        verification = replace(
            cast(BackendVerification, verification),
            execution_attestation=replace(
                cast(BackendVerification, verification).execution_attestation,
                role="wrong_role",
            ),
        )
    elif case == "heuristic_lock":
        verification = replace(
            cast(BackendVerification, verification), runtime_lock_sha256="0" * 64
        )
    elif case == "heuristic_tensor":
        tensor = replace(
            cast(BackendVerification, verification).tensor_coverage,
            required_count=1,
        )
        verification = replace(cast(BackendVerification, verification), tensor_coverage=tensor)
    elif case == "heuristic_smoke":
        smoke = replace(
            cast(BackendVerification, verification).smoke,
            status="passed",
            audio_sha256="0" * 64,
            oracle_sha256="1" * 64,
            prediction=PublishedArtifact(
                role="prediction",
                path=Path("artifacts/smoke.jsonl"),
                sha256="2" * 64,
            ),
        )
        verification = replace(cast(BackendVerification, verification), smoke=smoke)
    elif case == "heuristic_frame_bound":
        verification = replace(
            cast(BackendVerification, verification),
            max_input_audio_frames=1,
        )
    else:
        selected_backend_id = OFFICIAL_BACKEND_ID
        verification = oaf_verification()
        if case == "oaf_tensor_counts":
            tensor = replace(verification.tensor_coverage, restored_count=77)
            verification = replace(verification, tensor_coverage=tensor)
        elif case == "oaf_noninference_count":
            tensor = replace(verification.tensor_coverage, non_inference_count=79)
            verification = replace(verification, tensor_coverage=tensor)
        elif case == "oaf_tensor_artifact":
            tensor = replace(verification.tensor_coverage, report=None)
            verification = replace(verification, tensor_coverage=tensor)
        elif case == "oaf_smoke_relationship":
            smoke = replace(verification.smoke, oracle_sha256=None)
            verification = replace(verification, smoke=smoke)

    backend = FakeBackend(verification)
    request = direct_request(tmp_path, backend_id=selected_backend_id)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend, backend_id=selected_backend_id),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert not request.output_path.exists()
    payload = report_payload(outcome)
    assert payload["descriptor"] is None
    assert payload["errors"] == [
        {
            "code": "invalid_backend_verification",
            "message": "Backend verification did not satisfy the contract.",
        }
    ]
    assert backend.transcribe_calls == 0
    assert backend.close_calls == 1


@pytest.mark.parametrize(
    "case",
    [
        "none",
        "unexpected_type",
        "audio",
        "descriptor",
        "parameter_lock",
        "model_artifact",
        "official_runtime_lock",
        "official_upstream",
        "official_training_map",
    ],
)
def test_prediction_must_match_loaded_audio_and_verified_backend_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    official = case.startswith("official_")
    backend_id = OFFICIAL_BACKEND_ID if official else HEURISTIC_BACKEND_ID
    verification = oaf_verification() if official else heuristic_verification()

    def prediction_factory(audio: CanonicalAudio) -> object:
        prediction = oaf_prediction(audio) if official else heuristic_prediction(audio)
        if case == "none":
            return None
        if case == "unexpected_type":
            return {"events": []}
        if case == "audio":
            return replace(
                prediction,
                audio=replace(audio, input_audio_sha256="0" * 64),
            )
        if case == "descriptor":
            return replace(prediction, descriptor=OAF_DESCRIPTOR)
        if case == "parameter_lock":
            return replace(prediction, parameter_lock_sha256="0" * 64)
        if case == "model_artifact":
            return replace(prediction, model_artifact_set_sha256="0" * 64)
        if case == "official_runtime_lock":
            return replace(prediction, runtime_lock_sha256="0" * 64)
        if case == "official_upstream":
            return replace(prediction, upstream_source_commit="0" * 40)
        if case == "official_training_map":
            return replace(prediction, training_data_map_id="wrong-map")
        raise AssertionError(case)

    backend = FakeBackend(verification, prediction_factory=prediction_factory)
    request = direct_request(tmp_path, backend_id=backend_id)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend, backend_id=backend_id),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert not request.output_path.exists()
    assert report_payload(outcome)["errors"] == [
        {
            "code": "backend_protocol_failed",
            "message": "Backend prediction did not match its verified identity.",
        }
    ]
    assert backend.close_calls == 1


@pytest.mark.parametrize(
    "case",
    ["event_object", "events_list", "metadata_object"],
)
def test_prediction_nested_shape_must_be_valid_before_rendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    monkeypatch.chdir(tmp_path)

    def malformed_prediction(audio: CanonicalAudio) -> NativePrediction:
        prediction = heuristic_prediction(audio)
        if case == "event_object":
            return replace(
                prediction,
                events=cast(tuple[NativeEvent, ...], (object(),)),
            )
        if case == "events_list":
            object.__setattr__(prediction, "events", list(prediction.events))
            return prediction
        event = replace(
            prediction.events[0],
            native_metadata=cast(dict[str, str | None], object()),
        )
        return replace(prediction, events=(event,))

    backend = FakeBackend(
        heuristic_verification(),
        prediction_factory=malformed_prediction,
    )
    request = direct_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert outcome.exit_code == 2
    assert not request.output_path.exists()
    assert report_payload(outcome)["errors"][0]["code"] == "backend_protocol_failed"  # type: ignore[index]
    assert backend.close_calls == 1


@pytest.mark.parametrize("mutation_phase", ["factory", "verify", "midi"])
def test_repository_root_is_captured_before_callbacks_can_change_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation_phase: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    other = tmp_path / "callback-cwd"
    other.mkdir()
    request = replace(
        direct_request(
            tmp_path,
            midi_output_path=tmp_path / "predictions" / "result.mid",
        ),
        audio_path=Path("audio/input.wav"),
        output_path=Path("predictions/result.jsonl"),
        midi_output_path=Path("predictions/result.mid"),
        reports_root=Path("artifacts/benchmark/backends"),
    )

    def change_cwd() -> None:
        os.chdir(other)

    backend = FakeBackend(
        heuristic_verification(),
        verify_hook=change_cwd if mutation_phase == "verify" else None,
    )

    def factory() -> FakeBackend:
        if mutation_phase == "factory":
            change_cwd()
        return backend

    def midi_writer(
        prediction: PredictionArtifact,
        destination: Path,
    ) -> MidiDerivative:
        derivative = write_valid_midi_derivative(prediction, destination)
        if mutation_phase == "midi":
            change_cwd()
        return derivative

    try:
        outcome = run_transcribe_one(
            request,
            registry=BackendRegistry(
                default_backend_id=HEURISTIC_BACKEND_ID,
                factories={HEURISTIC_BACKEND_ID: factory},
            ),
            now=FIXED_UTC,
            run_id=FIXED_UUID,
            midi_writer=midi_writer,
        )
    finally:
        os.chdir(tmp_path)

    assert outcome.status == "complete"
    assert (tmp_path / "predictions" / "result.jsonl").exists()
    assert (tmp_path / "predictions" / "result.mid").exists()
    assert outcome.report_artifact.path.is_relative_to(tmp_path)
    assert backend.close_calls == 1


def test_invalid_initial_path_type_is_typed_operational_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = replace(
        direct_request(tmp_path),
        audio_path=cast(Path, object()),
    )

    with pytest.raises(
        OperationalReportPublicationError,
        match="operational_report_publication_failed",
    ):
        run_transcribe_one(
            request,
            registry=registry_for(FakeBackend(heuristic_verification())),
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert not request.output_path.exists()
    assert not request.reports_root.exists()


def test_initial_report_root_resolution_failure_is_typed_without_reroot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = direct_request(tmp_path)
    real_resolve = Path.resolve

    def fail_report_root_resolution(path: Path, strict: bool = False) -> Path:
        if path == request.reports_root:
            raise OSError("resolution details")
        return real_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", fail_report_root_resolution)

    with pytest.raises(
        OperationalReportPublicationError,
        match="operational_report_publication_failed",
    ):
        run_transcribe_one(
            request,
            registry=registry_for(FakeBackend(heuristic_verification())),
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert not request.output_path.exists()
    assert not request.reports_root.exists()


def test_close_failure_after_success_does_not_replace_durable_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(
        heuristic_verification(),
        close_error=RuntimeError("close failed"),
    )
    request = direct_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "complete"
    assert request.output_path.exists()
    assert outcome.report_artifact.path.exists()
    assert backend.close_calls == 1


def test_close_base_exception_after_success_does_not_replace_durable_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    class CloseFailure(BaseException):
        pass

    backend = FakeBackend(
        heuristic_verification(),
        close_error=CloseFailure("close failed"),
    )
    request = direct_request(tmp_path)

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "complete"
    assert request.output_path.exists()
    assert outcome.report_artifact.path.exists()
    assert backend.close_calls == 1


def test_close_failure_does_not_mask_report_publication_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(
        heuristic_verification(),
        close_error=RuntimeError("close failed"),
    )
    request = direct_request(tmp_path)

    def fail_report(*args: object, **kwargs: object) -> PublishedArtifact:
        del args, kwargs
        raise OperationalReportPublicationError("operational_report_publication_failed")

    monkeypatch.setattr(transcription_module, "publish_operational_report", fail_report)

    with pytest.raises(
        OperationalReportPublicationError,
        match="operational_report_publication_failed",
    ):
        run_transcribe_one(
            request,
            registry=registry_for(backend),
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert backend.close_calls == 1


def test_close_base_exception_does_not_mask_report_publication_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    class CloseFailure(BaseException):
        pass

    backend = FakeBackend(
        heuristic_verification(),
        close_error=CloseFailure("close failed"),
    )
    request = direct_request(tmp_path)

    def fail_report(*args: object, **kwargs: object) -> PublishedArtifact:
        del args, kwargs
        raise OperationalReportPublicationError("operational_report_publication_failed")

    monkeypatch.setattr(transcription_module, "publish_operational_report", fail_report)

    with pytest.raises(
        OperationalReportPublicationError,
        match="operational_report_publication_failed",
    ):
        run_transcribe_one(
            request,
            registry=registry_for(backend),
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert backend.close_calls == 1


@pytest.mark.parametrize(
    "case",
    [
        "unexpected_type",
        "midi_path",
        "midi_missing",
        "midi_hash",
        "sidecar_missing",
        "sidecar_hash",
        "sidecar_not_adjacent",
        "sidecar_schema",
        "sidecar_ticks",
        "sidecar_tempo",
        "sidecar_channel",
        "sidecar_program",
        "sidecar_duration",
        "distribution_name",
        "distribution_version",
        "distribution_filename",
        "distribution_sha256",
        "sidecar_prediction_hash",
        "sidecar_midi_hash",
        "sidecar_collides_with_prediction",
    ],
)
def test_midi_derivative_is_reread_and_strictly_validated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = direct_request(
        tmp_path,
        midi_output_path=tmp_path / "predictions" / "result.mid",
    )
    backend = FakeBackend(heuristic_verification())

    def adversarial_writer(
        prediction: PredictionArtifact,
        destination: Path,
    ) -> MidiDerivative:
        derivative = write_valid_midi_derivative(prediction, destination)
        if case == "unexpected_type":
            return cast(MidiDerivative, None)
        if case == "midi_path":
            other = destination.with_name("other.mid")
            other.write_bytes(destination.read_bytes())
            return replace(derivative, midi=replace(derivative.midi, path=other))
        if case == "midi_missing":
            derivative.midi.path.unlink()
        elif case == "midi_hash":
            derivative = replace(
                derivative,
                midi=replace(derivative.midi, sha256="0" * 64),
            )
        elif case == "sidecar_missing":
            derivative.sidecar.path.unlink()
        elif case == "sidecar_hash":
            derivative = replace(
                derivative,
                sidecar=replace(derivative.sidecar, sha256="0" * 64),
            )
        elif case == "sidecar_not_adjacent":
            other = tmp_path / "elsewhere" / "result.sidecar.json"
            other.parent.mkdir()
            derivative.sidecar.path.replace(other)
            derivative = replace(
                derivative,
                sidecar=replace(derivative.sidecar, path=other),
            )
        elif case in {
            "sidecar_schema",
            "sidecar_ticks",
            "sidecar_tempo",
            "sidecar_channel",
            "sidecar_program",
            "sidecar_duration",
            "distribution_name",
            "distribution_version",
            "distribution_filename",
            "distribution_sha256",
            "sidecar_prediction_hash",
            "sidecar_midi_hash",
        }:
            sidecar_payload = strict_json_loads(
                derivative.sidecar.path.read_bytes()[:-1],
                require_canonical=True,
            )
            assert isinstance(sidecar_payload, dict)
            if case == "sidecar_schema":
                sidecar_payload["schema"] = "crux.unknown/v1"
            elif case == "sidecar_ticks":
                sidecar_payload["ticks_per_quarter"] = 221
            elif case == "sidecar_tempo":
                sidecar_payload["tempo_qpm"] = Decimal("120.000001")
            elif case == "sidecar_channel":
                sidecar_payload["channel"] = 8
            elif case == "sidecar_program":
                sidecar_payload["program"] = 1
            elif case == "sidecar_duration":
                sidecar_payload["note_duration_seconds"] = Decimal("0.050001")
            elif case.startswith("distribution_"):
                distribution = sidecar_payload["writer_distribution"]
                assert isinstance(distribution, dict)
                field = case.removeprefix("distribution_")
                distribution[field] = "0" * 64 if field == "sha256" else f"wrong-{field}"
            elif case == "sidecar_prediction_hash":
                sidecar_payload["prediction_sha256"] = "0" * 64
            else:
                sidecar_payload["midi_sha256"] = "0" * 64
            content = canonical_json_bytes(sidecar_payload, trailing_newline=True)
            derivative.sidecar.path.write_bytes(content)
            derivative = replace(
                derivative,
                sidecar=replace(derivative.sidecar, sha256=sha256_hex(content)),
            )
        elif case == "sidecar_collides_with_prediction":
            derivative = replace(
                derivative,
                sidecar=PublishedArtifact(
                    role="prediction_midi_sidecar",
                    path=request.output_path,
                    sha256=prediction.artifact_sha256,
                ),
            )
        return derivative

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
        midi_writer=adversarial_writer,
    )

    item = report_payload(outcome)["items"][0]  # type: ignore[index]
    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert request.output_path.exists()
    assert item["status"] == "incomplete"
    assert item["prediction"]["path"] == "predictions/result.jsonl"
    assert item["midi"] is None
    assert item["errors"][0]["code"] == "midi_derivation_failed"
    assert backend.close_calls == 1


def test_prediction_and_midi_destinations_must_be_distinct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    shared = tmp_path / "predictions" / "result.jsonl"
    request = direct_request(tmp_path, output_path=shared, midi_output_path=shared)
    backend = FakeBackend(heuristic_verification())

    outcome = run_transcribe_one(
        request,
        registry=registry_for(backend),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
        midi_writer=write_valid_midi_derivative,
    )

    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert not shared.exists()
    assert backend.transcribe_calls == 0


def test_unclassified_factory_file_not_found_is_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    def missing_unrelated_file() -> FakeBackend:
        raise FileNotFoundError("/private/unrelated/input")

    request = direct_request(tmp_path, backend_id=OFFICIAL_BACKEND_ID)
    outcome = run_transcribe_one(
        request,
        registry=BackendRegistry(
            default_backend_id=OFFICIAL_BACKEND_ID,
            factories={OFFICIAL_BACKEND_ID: missing_unrelated_file},
        ),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.status == "failed"
    assert report_payload(outcome)["errors"][0]["code"] == "backend_failure"  # type: ignore[index]
    assert "/private/unrelated/input" not in outcome.report_artifact.path.read_text()


def test_invalid_run_uuid_is_operational_report_publication_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(heuristic_verification())
    request = direct_request(tmp_path)

    with pytest.raises(
        OperationalReportPublicationError,
        match="operational_report_publication_failed",
    ):
        run_transcribe_one(
            request,
            registry=registry_for(backend),
            now=FIXED_UTC,
            run_id=UUID("12345678-1234-1678-9234-567812345678"),
        )

    assert backend.close_calls == 1


def test_execution_reference_normalization_is_operational_publication_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    invalid_verification = replace(
        heuristic_verification(status="failed"),
        execution_attestation=PublishedArtifact(
            role="execution_attestation",
            path=Path("artifacts/attestation.json"),
            sha256="not-a-hash",
        ),
        errors=(BackendError(code="verification_failed", message="Verification failed."),),
    )
    request = direct_request(tmp_path)

    with pytest.raises(
        OperationalReportPublicationError,
        match="operational_report_publication_failed",
    ):
        transcription_module._publish_outcome(  # pylint: disable=protected-access
            request,
            backend_id=HEURISTIC_BACKEND_ID,
            verification=invalid_verification,
            status="failed",
            exit_code=2,
            items=(),
            errors=invalid_verification.errors,
            now=FIXED_UTC,
            run_id=FIXED_UUID,
            repository_root=tmp_path,
        )
