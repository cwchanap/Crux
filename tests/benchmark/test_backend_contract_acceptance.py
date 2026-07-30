from __future__ import annotations

import hashlib
import struct
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

import pytest

from src.benchmark.backend_identity import (
    BackendDescriptor,
    JsonValue,
    build_descriptor,
    strict_json_loads,
)
from src.benchmark.backend_publication import (
    publish_immutable_bytes,
    read_regular_file_no_follow,
)
from src.benchmark.backend_registry import (
    HEURISTIC_BACKEND_ID,
    OFFICIAL_BACKEND_ID,
    BackendRegistry,
)
from src.benchmark.backends import (
    BackendVerification,
    CanonicalAudio,
    NativeEvent,
    NativePrediction,
    PublishedArtifact,
    SmokeCheck,
    TensorCoverageCheck,
)
from src.benchmark.prediction_artifact import (
    PredictionArtifact,
    read_prediction_artifact,
    render_prediction_artifact,
)
from src.benchmark.transcription import (
    TranscribeOneOutcome,
    TranscribeOneRequest,
    run_transcribe_one,
)

FIRST_UUID = UUID("12345678-1234-4678-9234-567812345678")
SECOND_UUID = UUID("87654321-4321-4876-9234-567812345678")
OAF_UUID = UUID("abcdef12-3456-4789-abcd-ef1234567890")
FIRST_TIME = datetime(2026, 7, 27, 1, 2, 3, 456789, tzinfo=UTC)
SECOND_TIME = datetime(2026, 7, 27, 4, 5, 6, 789012, tzinfo=UTC)
OAF_TIME = datetime(2026, 7, 27, 7, 8, 9, 123456, tzinfo=UTC)
SOURCE_AUDIO_ID = "曲・🥁-é"
INPUT_VIEW_ID = "正規化-44k1"

PCM_BYTES = b"\x00\x00" * 4
WAV_BYTES = (
    struct.pack("<4sI4s", b"RIFF", 36 + len(PCM_BYTES), b"WAVE")
    + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16)
    + struct.pack("<4sI", b"data", len(PCM_BYTES))
    + PCM_BYTES
)
WAV_SHA256 = "51fccb29df62eb03616b5de4af3383d3b5382593ebb0a81e13368a2c82f3d1b9"

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
HEURISTIC_DESCRIPTOR_KEYS = frozenset(
    {
        "adapter_source_manifest_sha256",
        "architecture_id",
        "backend_id",
        "descriptor_schema",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "parameter_lock_sha256",
        "prediction_schema",
    }
)
HEURISTIC_DESCRIPTOR = build_descriptor(
    HEURISTIC_DESCRIPTOR_PAYLOAD,
    HEURISTIC_DESCRIPTOR_KEYS,
    "crux.heuristic-backend-descriptor/v1",
)
EXPECTED_HEURISTIC_DESCRIPTOR_SHA256 = (
    "a0b5d777338fe75c0452cc3501c880a3081d2eaf11de42246a7ac28db2c84c6e"
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
OAF_DESCRIPTOR_KEYS = frozenset(
    {
        "architecture_id",
        "backend_id",
        "backend_lock_sha256",
        "descriptor_schema",
        "model_artifact_set_sha256",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "prediction_schema",
        "protocol_schema",
        "runtime_image_manifest_digest",
        "runtime_lock_sha256",
        "training_data_map_id",
        "upstream_source_commit",
    }
)
OAF_DESCRIPTOR = build_descriptor(
    OAF_DESCRIPTOR_PAYLOAD,
    OAF_DESCRIPTOR_KEYS,
    "crux.transcription-backend-descriptor/v1",
)
EXPECTED_OAF_DESCRIPTOR_SHA256 = "b212dd4a29ba31bc74491868f2c37091e3f0a1fdc702074f30ee43f11b88c93c"

HEURISTIC_EVENTS = (
    NativeEvent(
        time_sec=0.5,
        native_class_id="heuristic_kick",
        model_output_bin=None,
        native_midi_note=None,
        native_metadata={},
        confidence=None,
        velocity_midi=None,
    ),
    NativeEvent(
        time_sec=0.5,
        native_class_id="heuristic_snare",
        model_output_bin=None,
        native_midi_note=38,
        native_metadata={},
        confidence=0.875,
        velocity_midi=96,
    ),
)
OAF_EVENTS = (
    NativeEvent(
        time_sec=0.75,
        native_class_id="midi_36",
        model_output_bin=15,
        native_midi_note=36,
        native_metadata={"upstream_8hit_group_id": "kick"},
        confidence=0.625,
        velocity_midi=100,
    ),
)

EXPECTED_HEURISTIC_PREDICTION = (
    '{"architecture_id":"librosa-onset-centroid-zcr-v1","artifact_role":"native",'
    '"audio_frame_count":4,"backend_descriptor":{"adapter_source_manifest_sha256":'
    '"7777777777777777777777777777777777777777777777777777777777777777",'
    '"architecture_id":"librosa-onset-centroid-zcr-v1","backend_id":'
    '"heuristic-onset-v1","descriptor_schema":"crux.heuristic-backend-descriptor/v1",'
    '"model_id":"crux-heuristic-onset-nonmodel-v1","native_metadata_schema_id":'
    '"crux-empty-native-metadata-v1","native_output_space_id":'
    '"crux-heuristic-midi7-v1","parameter_lock_sha256":'
    '"6666666666666666666666666666666666666666666666666666666666666666",'
    '"prediction_schema":"crux.drum-prediction-events/v1"},'
    '"backend_descriptor_sha256":'
    '"a0b5d777338fe75c0452cc3501c880a3081d2eaf11de42246a7ac28db2c84c6e",'
    '"backend_lock_sha256":null,"byte_length":52,"channel_count":1,'
    '"input_audio_sha256":'
    '"51fccb29df62eb03616b5de4af3383d3b5382593ebb0a81e13368a2c82f3d1b9",'
    '"input_view_id":"正規化-44k1","model_artifact_set_sha256":null,'
    '"model_id":"crux-heuristic-onset-nonmodel-v1","native_metadata_schema_id":'
    '"crux-empty-native-metadata-v1","native_output_space_id":'
    '"crux-heuristic-midi7-v1","parameter_lock_sha256":'
    '"6666666666666666666666666666666666666666666666666666666666666666",'
    '"record_type":"header","runtime_lock_sha256":null,"sample_rate":44100,'
    '"sample_width_bytes":2,"schema":"crux.drum-prediction-events/v1",'
    '"source_audio_id":"曲・🥁-é","source_audio_sha256":'
    '"51fccb29df62eb03616b5de4af3383d3b5382593ebb0a81e13368a2c82f3d1b9",'
    '"training_data_map_id":null,"upstream_source_commit":null}\n'
    '{"canonical_class":null,"confidence":null,"event_index":0,"mapping_status":'
    '"not_applied","model_output_bin":null,"native_class_id":"heuristic_kick",'
    '"native_metadata":{},"native_midi_note":null,"prediction_map_version":null,'
    '"record_type":"event","time_sec":0.5,"velocity_midi":null}\n'
    '{"canonical_class":null,"confidence":0.875,"event_index":1,"mapping_status":'
    '"not_applied","model_output_bin":null,"native_class_id":"heuristic_snare",'
    '"native_metadata":{},"native_midi_note":38,"prediction_map_version":null,'
    '"record_type":"event","time_sec":0.5,"velocity_midi":96}\n'
    '{"event_count":2,"prefix_sha256":'
    '"12263f8ac816a147e0ad0a56bdfb044f6753956ece14195ab4f308db748373d5",'
    '"record_type":"terminal"}\n'
).encode("utf-8")
EXPECTED_HEURISTIC_PREDICTION_SHA256 = (
    "433dd9f595193c36bab4d27a69c775a8fb73f180cbc2a381a6287aeda0645179"
)
EXPECTED_ITEM_ID = "sha256:f24ea06c2454b747bcc7ac130205aac015b8884f24a1577e26722fa46a1ddfa0"

DERIVED_MANIFEST_BYTES = (
    '{"input_audio_path":"canonical.wav","input_audio_sha256":'
    f'"{WAV_SHA256}","input_view_id":"{INPUT_VIEW_ID}",'
    '"schema":"crux.input-view-manifest/v1",'
    f'"source_audio_id":"{SOURCE_AUDIO_ID}","source_audio_sha256":"{WAV_SHA256}",'
    '"source_path":"source.wav"}\n'
).encode("utf-8")


@dataclass(frozen=True)
class SupportArtifacts:
    attestation: PublishedArtifact
    tensor_report: PublishedArtifact | None
    smoke_prediction: PublishedArtifact | None


@dataclass(frozen=True)
class FakeRun:
    outcome: TranscribeOneOutcome
    request: TranscribeOneRequest
    backend: DeterministicFakeBackend
    prediction_bytes: bytes
    prediction: PredictionArtifact
    item_id: str
    report_bytes: bytes
    report: dict[str, JsonValue]


class DeterministicFakeBackend:
    def __init__(
        self,
        descriptor: BackendDescriptor,
        support: SupportArtifacts,
    ) -> None:
        self._descriptor = descriptor
        self._support = support
        self.received_audio: CanonicalAudio | None = None
        self.closed = False

    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def verify(self) -> BackendVerification:
        if self._descriptor.payload["backend_id"] == HEURISTIC_BACKEND_ID:
            return BackendVerification(
                status="verified",
                descriptor=self._descriptor,
                max_input_audio_frames=None,
                backend_lock_sha256=None,
                runtime_lock_sha256=None,
                parameter_lock_sha256="6" * 64,
                seal_evidence_sha256=None,
                execution_attestation=self._support.attestation,
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
                errors=(),
            )

        assert self._support.tensor_report is not None
        assert self._support.smoke_prediction is not None
        return BackendVerification(
            status="verified",
            descriptor=self._descriptor,
            max_input_audio_frames=4,
            backend_lock_sha256="b" * 64,
            runtime_lock_sha256="c" * 64,
            parameter_lock_sha256=None,
            seal_evidence_sha256="d" * 64,
            execution_attestation=self._support.attestation,
            tensor_coverage=TensorCoverageCheck(
                status="passed",
                required_count=2,
                restored_count=2,
                non_inference_count=1,
                required_inventory_sha256="e" * 64,
                non_inference_inventory_sha256="f" * 64,
                report=self._support.tensor_report,
            ),
            smoke=SmokeCheck(
                status="passed",
                audio_sha256=WAV_SHA256,
                oracle_sha256="1" * 64,
                prediction=self._support.smoke_prediction,
            ),
            errors=(),
        )

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        self.received_audio = audio
        if self._descriptor.payload["backend_id"] == HEURISTIC_BACKEND_ID:
            return NativePrediction(
                audio=audio,
                descriptor=self._descriptor,
                events=HEURISTIC_EVENTS,
                backend_lock_sha256=None,
                runtime_lock_sha256=None,
                parameter_lock_sha256="6" * 64,
                model_artifact_set_sha256=None,
                upstream_source_commit=None,
                training_data_map_id=None,
            )
        return NativePrediction(
            audio=audio,
            descriptor=self._descriptor,
            events=OAF_EVENTS,
            backend_lock_sha256="b" * 64,
            runtime_lock_sha256="c" * 64,
            parameter_lock_sha256=None,
            model_artifact_set_sha256="7" * 64,
            upstream_source_commit="9" * 40,
            training_data_map_id="magenta-egmd-data-8hit-94529798-v1",
        )

    def close(self) -> None:
        self.closed = True


def _publish_support_artifact(
    root: Path,
    name: str,
    role: str,
    content: bytes,
) -> PublishedArtifact:
    return publish_immutable_bytes(
        root / "support" / name,
        content,
        hashlib.sha256(content).hexdigest(),
        role=role,
    )


def _support_artifacts(root: Path, backend_id: str) -> SupportArtifacts:
    attestation = _publish_support_artifact(
        root,
        "execution-attestation.json",
        "execution_attestation",
        b'{"backend":"deterministic-fake","schema":"crux.test-attestation/v1"}\n',
    )
    if backend_id == HEURISTIC_BACKEND_ID:
        return SupportArtifacts(attestation, None, None)
    return SupportArtifacts(
        attestation,
        _publish_support_artifact(
            root,
            "tensor-coverage.json",
            "tensor_coverage",
            b'{"required":2,"restored":2,"schema":"crux.test-tensor-coverage/v1"}\n',
        ),
        _publish_support_artifact(
            root,
            "smoke-prediction.jsonl",
            "prediction",
            b'{"schema":"crux.test-smoke-prediction/v1"}\n',
        ),
    )


def _request(
    root: Path,
    provenance: Literal["direct", "derived"],
) -> TranscribeOneRequest:
    input_root = root / "input"
    input_root.mkdir(parents=True)
    if provenance == "direct":
        audio_path = input_root / "direct.wav"
        audio_path.write_bytes(WAV_BYTES)
        source_audio_id: str | None = SOURCE_AUDIO_ID
        input_view_id: str | None = INPUT_VIEW_ID
        manifest_path: Path | None = None
    else:
        source_path = input_root / "source.wav"
        audio_path = input_root / "canonical.wav"
        manifest_path = input_root / "manifest.json"
        source_path.write_bytes(WAV_BYTES)
        audio_path.write_bytes(WAV_BYTES)
        manifest_path.write_bytes(DERIVED_MANIFEST_BYTES)
        source_audio_id = None
        input_view_id = None
    return TranscribeOneRequest(
        backend_id=None,
        audio_path=audio_path,
        output_path=root / "published" / "prediction.jsonl",
        source_audio_id=source_audio_id,
        input_view_id=input_view_id,
        input_view_manifest=manifest_path,
        midi_output_path=None,
        reports_root=root / "reports",
    )


def _run_fake_transcription(
    root: Path,
    *,
    descriptor: BackendDescriptor,
    provenance: Literal["direct", "derived"],
    now: datetime,
    run_id: UUID,
) -> FakeRun:
    backend_id = descriptor.payload["backend_id"]
    support = _support_artifacts(root, backend_id)
    created: list[DeterministicFakeBackend] = []

    def create_backend() -> DeterministicFakeBackend:
        backend = DeterministicFakeBackend(descriptor, support)
        created.append(backend)
        return backend

    registry = BackendRegistry(
        default_backend_id=backend_id,
        factories={backend_id: create_backend},
    )
    request = _request(root, provenance)
    outcome = run_transcribe_one(
        request,
        registry=registry,
        now=now,
        run_id=run_id,
    )

    assert outcome.status == "complete"
    assert outcome.exit_code == 0
    assert len(created) == 1
    backend = created[0]
    assert backend.closed
    assert backend.received_audio is not None

    prediction_bytes = read_regular_file_no_follow(request.output_path)
    prediction = read_prediction_artifact(prediction_bytes)
    assert prediction.content == prediction_bytes
    assert render_prediction_artifact(prediction.prediction) == prediction_bytes

    report_bytes = read_regular_file_no_follow(outcome.report_artifact.path)
    assert report_bytes.endswith(b"\n")
    report_value = strict_json_loads(report_bytes[:-1], require_canonical=True)
    assert isinstance(report_value, dict)
    report = cast(dict[str, JsonValue], report_value)
    items = report["items"]
    assert isinstance(items, list) and len(items) == 1
    item = items[0]
    assert isinstance(item, dict)
    item_id = item["item_id"]
    assert isinstance(item_id, str)
    return FakeRun(
        outcome=outcome,
        request=request,
        backend=backend,
        prediction_bytes=prediction_bytes,
        prediction=prediction,
        item_id=item_id,
        report_bytes=report_bytes,
        report=report,
    )


def _normalized_report(
    report: dict[str, JsonValue],
    *,
    run_id: UUID,
    timestamp: str,
    attestation_path: str,
    prediction_path: str,
) -> dict[str, JsonValue]:
    normalized = deepcopy(report)
    assert normalized["run_id"] == str(run_id)
    assert normalized["started_at"] == timestamp
    assert normalized["finished_at"] == timestamp
    normalized["run_id"] = "<run-id>"
    normalized["started_at"] = "<timestamp>"
    normalized["finished_at"] = "<timestamp>"

    attestation = normalized["execution_attestation"]
    assert isinstance(attestation, dict)
    assert attestation["path"] == attestation_path
    attestation["path"] = "<execution-attestation-path>"

    items = normalized["items"]
    assert isinstance(items, list) and len(items) == 1
    item = items[0]
    assert isinstance(item, dict)
    prediction = item["prediction"]
    assert isinstance(prediction, dict)
    assert prediction["path"] == prediction_path
    prediction["path"] = "<prediction-path>"
    return normalized


def test_fake_backend_repeats_byte_identical_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch run/path/provenance data leaking into native prediction identity."""
    monkeypatch.chdir(tmp_path)

    first = _run_fake_transcription(
        tmp_path / "first",
        descriptor=HEURISTIC_DESCRIPTOR,
        provenance="direct",
        now=FIRST_TIME,
        run_id=FIRST_UUID,
    )
    second = _run_fake_transcription(
        tmp_path / "second",
        descriptor=HEURISTIC_DESCRIPTOR,
        provenance="derived",
        now=SECOND_TIME,
        run_id=SECOND_UUID,
    )
    oaf = _run_fake_transcription(
        tmp_path / "oaf",
        descriptor=OAF_DESCRIPTOR,
        provenance="direct",
        now=OAF_TIME,
        run_id=OAF_UUID,
    )

    assert hashlib.sha256(WAV_BYTES).hexdigest() == WAV_SHA256
    assert HEURISTIC_DESCRIPTOR.sha256 == EXPECTED_HEURISTIC_DESCRIPTOR_SHA256
    assert OAF_DESCRIPTOR.sha256 == EXPECTED_OAF_DESCRIPTOR_SHA256
    assert first.backend.descriptor() == second.backend.descriptor() == HEURISTIC_DESCRIPTOR
    assert (
        first.report["descriptor_sha256"]
        == second.report["descriptor_sha256"]
        == (EXPECTED_HEURISTIC_DESCRIPTOR_SHA256)
    )

    assert first.request.input_view_manifest is None
    assert second.request.input_view_manifest is not None
    assert first.backend.received_audio is not None
    assert second.backend.received_audio is not None
    assert first.backend.received_audio.source_audio_id == SOURCE_AUDIO_ID
    assert second.backend.received_audio.source_audio_id == SOURCE_AUDIO_ID
    assert first.backend.received_audio.source_audio_sha256 == WAV_SHA256
    assert second.backend.received_audio.source_audio_sha256 == WAV_SHA256
    assert first.backend.received_audio.input_view_id == INPUT_VIEW_ID
    assert second.backend.received_audio.input_view_id == INPUT_VIEW_ID
    assert first.backend.received_audio.input_audio_sha256 == WAV_SHA256
    assert second.backend.received_audio.input_audio_sha256 == WAV_SHA256

    assert first.prediction_bytes == EXPECTED_HEURISTIC_PREDICTION
    assert second.prediction_bytes == EXPECTED_HEURISTIC_PREDICTION
    assert first.prediction.artifact_sha256 == EXPECTED_HEURISTIC_PREDICTION_SHA256
    assert second.prediction.artifact_sha256 == EXPECTED_HEURISTIC_PREDICTION_SHA256
    assert first.item_id == second.item_id == EXPECTED_ITEM_ID

    heuristic_prediction = first.prediction.prediction
    assert heuristic_prediction.events == HEURISTIC_EVENTS
    assert [event.time_sec for event in heuristic_prediction.events] == [0.5, 0.5]
    assert heuristic_prediction.events[0].native_class_id != (
        heuristic_prediction.events[1].native_class_id
    )
    assert heuristic_prediction.events[0].model_output_bin is None
    assert heuristic_prediction.events[0].native_midi_note is None
    assert heuristic_prediction.events[0].confidence is None
    assert heuristic_prediction.events[0].velocity_midi is None
    assert heuristic_prediction.backend_lock_sha256 is None
    assert heuristic_prediction.runtime_lock_sha256 is None
    assert heuristic_prediction.model_artifact_set_sha256 is None
    assert heuristic_prediction.upstream_source_commit is None
    assert heuristic_prediction.training_data_map_id is None

    assert oaf.backend.descriptor() == OAF_DESCRIPTOR
    assert oaf.report["descriptor_sha256"] == EXPECTED_OAF_DESCRIPTOR_SHA256
    assert oaf.prediction.prediction.events == OAF_EVENTS
    assert oaf.prediction.prediction.events[0].native_metadata == {"upstream_8hit_group_id": "kick"}

    assert first.report_bytes != second.report_bytes
    assert _normalized_report(
        first.report,
        run_id=FIRST_UUID,
        timestamp="2026-07-27T01:02:03.456789Z",
        attestation_path="first/support/execution-attestation.json",
        prediction_path="first/published/prediction.jsonl",
    ) == _normalized_report(
        second.report,
        run_id=SECOND_UUID,
        timestamp="2026-07-27T04:05:06.789012Z",
        attestation_path="second/support/execution-attestation.json",
        prediction_path="second/published/prediction.jsonl",
    )
    assert first.outcome.report_artifact.path == (
        tmp_path
        / "first"
        / "reports"
        / HEURISTIC_BACKEND_ID
        / "reports"
        / f"20260727T010203456789Z-{FIRST_UUID}.json"
    )
    assert second.outcome.report_artifact.path == (
        tmp_path
        / "second"
        / "reports"
        / HEURISTIC_BACKEND_ID
        / "reports"
        / f"20260727T040506789012Z-{SECOND_UUID}.json"
    )
