from __future__ import annotations

import struct
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import src.benchmark.backends.oaf_tf1 as oaf_tf1
from src.benchmark.backend_identity import (
    BackendDescriptor,
    canonical_json_bytes,
    sha256_hex,
)
from src.benchmark.backend_lock import (
    REQUIRED_ENVIRONMENT,
    LoadedBackendLock,
    LoadedConversionAudit,
    LoadedRuntimeLock,
    LoadedSealEvidence,
)
from src.benchmark.backend_process import (
    NativeHostEvidence,
    RunnerLaunchProfile,
    RunnerResponse,
)
from src.benchmark.backend_publication import PrivateSnapshotIntegrityError
from src.benchmark.backends import (
    BackendError,
    BackendFatalFailure,
    CanonicalAudio,
    NativeEvent,
    NativePrediction,
    PublishedArtifact,
)
from src.benchmark.input_view import load_direct_audio
from src.benchmark.prediction_artifact import render_prediction_artifact

BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"
BACKEND_SHA256 = "b" * 64
RUNTIME_SHA256 = "c" * 64
SEAL_SHA256 = "d" * 64
AUDIT_SHA256 = "e" * 64
Inventory = list[dict[str, object]]


def _canonical_file(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))  # type: ignore[arg-type]
    return path


def _wav_bytes(sample_frames: int = 4) -> bytes:
    pcm = b"\x00\x00" * sample_frames
    return (
        struct.pack("<4sI4s", b"RIFF", 36 + len(pcm), b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16)
        + struct.pack("<4sI", b"data", len(pcm))
        + pcm
    )


def _host_evidence() -> NativeHostEvidence:
    payload = {
        "api_record_sha256": "a" * 64,
        "approved_labels": ["Linux", "X64"],
        "job_id": 123,
        "run_url": "https://github.com/acme/crux/actions/runs/456/job/123",
        "runner_arch": "X64",
        "runner_os": "Linux",
        "workflow_commit": "f" * 40,
        "host_numeric_fingerprint": {
            "architecture": "x86_64",
            "cpu_vendor_id": "GenuineIntel",
            "cpu_family": "6",
            "cpu_model": "143",
            "cpu_stepping": "8",
        },
    }
    return NativeHostEvidence(
        kind="github_hosted",
        payload=payload,
        sha256=sha256_hex(canonical_json_bytes(payload)),
        official_execution_allowed=True,
    )


def _descriptor() -> BackendDescriptor:
    payload = {
        "architecture_id": "magenta-oaf-model-tpu-drums-v1",
        "backend_id": BACKEND_ID,
        "backend_lock_sha256": BACKEND_SHA256,
        "descriptor_schema": "crux.transcription-backend-descriptor/v1",
        "model_artifact_set_sha256": "7" * 64,
        "model_id": "magenta-egmd-ckpt-569400-v1",
        "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
        "native_output_space_id": "magenta-oaf-midi88-a0-v1",
        "prediction_schema": "crux.drum-prediction-events/v1",
        "protocol_schema": "crux.transcription-runner/v1",
        "runtime_image_manifest_digest": f"sha256:{'4' * 64}",
        "runtime_lock_sha256": RUNTIME_SHA256,
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
        "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
    }
    return BackendDescriptor(payload=payload, sha256=sha256_hex(canonical_json_bytes(payload)))


def _inventories() -> tuple[Inventory, Inventory, Inventory]:
    checkpoint = [
        {"dtype": "float32", "name": f"tensor_{index:03d}", "shape": [index + 1]}
        for index in range(130)
    ]
    required = checkpoint[:78]
    non_inference = [{**entry, "reason": "optimizer_state"} for entry in checkpoint[78:]]
    return checkpoint, required, non_inference


def _runner_event() -> dict[str, object]:
    frame_index = 64
    return {
        "confidence_raw": Decimal("0.75"),
        "frame_index": frame_index,
        "model_output_bin": 15,
        "native_class_id": "midi_36",
        "native_midi_note": 36,
        "time_sec_raw": Decimal(frame_index) * Decimal(512) / Decimal(44100),
        "upstream_group_id": "kick",
        "velocity": 100,
    }


def _native_event() -> NativeEvent:
    event = _runner_event()
    return NativeEvent(
        time_sec=float(event["time_sec_raw"]),  # type: ignore[arg-type]
        native_class_id="midi_36",
        model_output_bin=15,
        native_midi_note=36,
        native_metadata={"upstream_8hit_group_id": "kick"},
        confidence=0.75,
        velocity_midi=100,
    )


@dataclass
class FakeProcess:
    handshake: dict[str, object]
    request_error: str | None = None
    response_changes: dict[str, object] | None = None
    after_smoke_changes: dict[str, object] | None = None
    event: dict[str, object] | None = None
    after_smoke_event: dict[str, object] | None = None
    request_hook: Callable[[dict[str, object], int], None] | None = None
    request_count: int = 0
    close_count: int = 0
    requests: list[dict[str, object]] | None = None

    def request(
        self,
        payload: dict[str, object],
        *,
        deadline_seconds: int | None = None,
    ) -> RunnerResponse:
        del deadline_seconds
        self.request_count += 1
        request = dict(payload)
        if self.requests is None:
            self.requests = []
        self.requests.append(request)
        if self.request_hook is not None:
            self.request_hook(request, self.request_count)
        if self.request_error is not None and self.request_count > 1:
            raise BackendFatalFailure(
                BackendError(
                    code=self.request_error,
                    message="The fake runner failed.",
                )
            )
        response: dict[str, object] = {
            "audio_sha256": payload["audio_sha256"],
            "backend_descriptor_sha256": payload["backend_descriptor_sha256"],
            "native_events": [self.event or _runner_event()],
            "type": "transcription_result",
        }
        if self.response_changes:
            response.update(self.response_changes)
        if self.request_count > 1 and self.after_smoke_changes:
            if self.after_smoke_changes.get("type") == "transcription_error":
                response = dict(self.after_smoke_changes)
            else:
                response.update(self.after_smoke_changes)
        if self.request_count > 1 and self.after_smoke_event:
            response["native_events"] = [self.after_smoke_event]
        envelope = {
            "payload": response,
            "request_id": "fake-request",
            "type": "response",
        }
        assert set(envelope) == {"payload", "request_id", "type"}
        assert "request_id" not in response
        return RunnerResponse(
            request_id=envelope["request_id"],
            payload=envelope["payload"],  # type: ignore[arg-type]
        )

    def close(self) -> None:
        self.close_count += 1


@dataclass
class Harness:
    backend: oaf_tf1.OafTf1Backend
    config: oaf_tf1.OafBackendConfig
    process: FakeProcess
    descriptor: BackendDescriptor
    smoke_audio: CanonicalAudio
    expected_smoke_sha256: str
    captured_profiles: list[RunnerLaunchProfile]
    captured_attestations: list[dict[str, object]]


def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    handshake_changes: dict[str, object] | None = None,
    request_error: str | None = None,
    response_changes: dict[str, object] | None = None,
    event: dict[str, object] | None = None,
    after_smoke_changes: dict[str, object] | None = None,
    after_smoke_event: dict[str, object] | None = None,
    native_system: str = "Linux",
    allow_emulated_diagnostics: bool = False,
    strict_checkout: bool = True,
    changed_files: tuple[dict[str, object], ...] = (),
    native_evidence_matches: bool = True,
) -> Harness:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    config_root = repository / "config" / "benchmark" / "backends"
    runtime_root = repository / "runtime" / "oaf_tf1"
    input_root = repository / "artifacts" / "benchmark" / "inputs"
    model_root = repository / "artifacts" / "benchmark" / "model-cache" / "sha256" / ("7" * 64)
    model_root.mkdir(parents=True)

    host_manifest = _canonical_file(
        runtime_root / "host-adapter-source-manifest.json",
        {
            "covered_roots": ["src/benchmark"],
            "files": [
                {
                    "path": "src/benchmark/backends/oaf_tf1.py",
                    "sha256": "1" * 64,
                }
            ],
            "schema": "crux.oaf-host-adapter-source-manifest/v1",
        },
    )
    runner_manifest = _canonical_file(
        runtime_root / "runner-source-manifest.json",
        {
            "covered_roots": ["runtime/oaf_tf1"],
            "files": [
                {
                    "path": "runtime/oaf_tf1/oaf_backend.py",
                    "sha256": "2" * 64,
                }
            ],
            "schema": "crux.oaf-runner-source-manifest/v1",
        },
    )
    upstream_manifest = _canonical_file(
        runtime_root / "source-manifest.json",
        {
            "covered_roots": ["magenta"],
            "files": [],
            "schema": "crux.oaf-upstream-source-manifest/v1",
            "upstream_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
            "upstream_repository": "https://github.com/magenta/magenta.git",
        },
    )
    tensor_report = _canonical_file(
        repository / "docs" / "superpowers" / "evidence" / "hpa-320" / "oaf-tensor-coverage.json",
        {"schema": "crux.oaf-tensor-coverage/v1"},
    )

    smoke_path = input_root / "smoke" / "canonical.wav"
    smoke_path.parent.mkdir(parents=True)
    smoke_path.write_bytes(_wav_bytes())
    smoke_sha256 = sha256_hex(smoke_path.read_bytes())
    oracle_payload = {
        "input_audio_frame_count": 4,
        "input_audio_sha256": smoke_sha256,
        "input_view_id": "oaf-smoke-canonical-v1",
        "native_events": [_runner_event()],
        "schema": "crux.oaf-smoke-oracle/v1",
        "source_audio_id": "oaf-smoke-source-v1",
        "source_audio_sha256": smoke_sha256,
    }
    oracle_path = _canonical_file(input_root / "smoke" / "smoke-oracle.json", oracle_payload)

    evidence = _host_evidence()
    descriptor = _descriptor()
    checkpoint, required, non_inference = _inventories()
    backend_payload = {
        "checkpoint_inventory": checkpoint,
        "host_adapter_source_manifest_sha256": sha256_hex(host_manifest.read_bytes()),
        "max_input_audio_frames": 441000,
        "non_inference_inventory": non_inference,
        "required_inference_inventory": required,
        "smoke_audio_sha256": smoke_sha256,
        "smoke_oracle_sha256": sha256_hex(oracle_path.read_bytes()),
        "training_groups": [
            {"group_id": "kick"},
            {"group_id": "snare"},
            {"group_id": "toms"},
            {"group_id": "hihat"},
            {"group_id": "ride"},
            {"group_id": "ride_bell"},
            {"group_id": "crash"},
            {"group_id": "sticks"},
        ],
    }
    runtime_payload = {
        "environment": dict(REQUIRED_ENVIRONMENT),
        "python_version": "3.7.17",
        "runner_source_manifest_sha256": sha256_hex(runner_manifest.read_bytes()),
        "stderr_max_line_bytes": 8192,
        "stderr_read_chunk_bytes": 4096,
        "stderr_ring_buffer_bytes": 65536,
        "stdout_max_line_bytes": 262144,
        "tensorflow_abi": "cp37-cp37m-manylinux2010_x86_64",
        "tensorflow_build": "v1.15.5-0-g590d6eef7e",
        "upstream_source_manifest_sha256": sha256_hex(upstream_manifest.read_bytes()),
    }
    seal_payload = {
        "cpu_limit_millis": 2500,
        "memory_limit_bytes": 1073741824,
        "native_host_evidence": {
            "form": "github_hosted_linux_x64",
            "sha256": evidence.sha256 if native_evidence_matches else "0" * 64,
        },
        "pid_limit": 64,
        "request_deadline_seconds": 60,
        "runtime_gid": 10002,
        "runtime_uid": 10001,
        "shm_bytes": 67108864,
        "smoke_prediction_sha256": "",
        "startup_deadline_seconds": 120,
        "tensor_coverage_sha256": sha256_hex(tensor_report.read_bytes()),
        "tmp_bytes": 134217728,
    }
    backend_lock = LoadedBackendLock(
        path=_canonical_file(config_root / f"{BACKEND_ID}.backend-lock.json", {}),
        payload=backend_payload,
        sha256=BACKEND_SHA256,
        descriptor=descriptor,
        max_input_audio_frames=441000,
    )
    runtime_lock = LoadedRuntimeLock(
        path=_canonical_file(config_root / f"{BACKEND_ID}.runtime-lock.json", {}),
        payload=runtime_payload,
        sha256=RUNTIME_SHA256,
    )
    seal = LoadedSealEvidence(
        path=_canonical_file(config_root / f"{BACKEND_ID}.seal-evidence.json", {}),
        payload=seal_payload,
        sha256=SEAL_SHA256,
    )
    audit = LoadedConversionAudit(
        path=_canonical_file(
            repository
            / "docs"
            / "superpowers"
            / "evidence"
            / "hpa-320"
            / "legacy-tf2-conversion-coverage.json",
            {},
        ),
        payload={},
        sha256=AUDIT_SHA256,
    )

    smoke_audio = load_direct_audio(
        smoke_path,
        source_audio_id=oracle_payload["source_audio_id"],  # type: ignore[arg-type]
        input_view_id=oracle_payload["input_view_id"],  # type: ignore[arg-type]
        max_input_audio_frames=441000,
    )
    smoke_prediction = NativePrediction(
        audio=smoke_audio,
        descriptor=descriptor,
        events=(_native_event(),),
        backend_lock_sha256=BACKEND_SHA256,
        runtime_lock_sha256=RUNTIME_SHA256,
        parameter_lock_sha256=None,
        model_artifact_set_sha256="7" * 64,
        upstream_source_commit="94529798dfbbb14c27ddfd76f23027dc8e2ce185",
        training_data_map_id="magenta-egmd-data-8hit-94529798-v1",
    )
    expected_smoke_sha256 = sha256_hex(render_prediction_artifact(smoke_prediction))
    seal_payload["smoke_prediction_sha256"] = expected_smoke_sha256

    handshake: dict[str, object] = {
        "backend_descriptor": dict(descriptor.payload),
        "backend_descriptor_sha256": descriptor.sha256,
        "backend_lock_sha256": BACKEND_SHA256,
        "checkpoint_inventory_sha256": sha256_hex(canonical_json_bytes(checkpoint)),
        "non_inference_count": 52,
        "non_inference_inventory_sha256": sha256_hex(canonical_json_bytes(non_inference)),
        "protocol_schema": "crux.transcription-runner/v1",
        "python_version": "3.7.17",
        "required_inference_count": 78,
        "required_inference_inventory_sha256": sha256_hex(canonical_json_bytes(required)),
        "restored_inference_count": 78,
        "runner_source_manifest_sha256": runtime_payload["runner_source_manifest_sha256"],
        "runtime_lock_sha256": RUNTIME_SHA256,
        "smoke_audio_sha256": smoke_sha256,
        "smoke_oracle_sha256": backend_payload["smoke_oracle_sha256"],
        "smoke_prediction_sha256": expected_smoke_sha256,
        "smoke_status": "exact_match",
        "tensorflow_abi": runtime_payload["tensorflow_abi"],
        "tensorflow_build": runtime_payload["tensorflow_build"],
        "type": "ready",
        "upstream_source_manifest_sha256": runtime_payload["upstream_source_manifest_sha256"],
    }
    if handshake_changes:
        handshake.update(handshake_changes)
    process = FakeProcess(
        handshake=handshake,
        request_error=request_error,
        response_changes=response_changes,
        after_smoke_changes=after_smoke_changes,
        event=event,
        after_smoke_event=after_smoke_event,
    )
    captured_profiles: list[RunnerLaunchProfile] = []

    def process_factory(profile: RunnerLaunchProfile) -> FakeProcess:
        captured_profiles.append(profile)
        return process

    lock_by_path = {
        backend_lock.path: backend_lock,
        runtime_lock.path: runtime_lock,
        seal.path: seal,
        audit.path: audit,
    }
    monkeypatch.setattr(
        oaf_tf1,
        "load_backend_lock",
        lambda path: lock_by_path[path],
    )
    monkeypatch.setattr(
        oaf_tf1,
        "load_runtime_lock",
        lambda path: lock_by_path[path],
    )
    monkeypatch.setattr(
        oaf_tf1,
        "load_seal_evidence",
        lambda path: lock_by_path[path],
    )
    monkeypatch.setattr(
        oaf_tf1,
        "load_conversion_audit",
        lambda path: lock_by_path[path],
    )
    monkeypatch.setattr(oaf_tf1, "validate_oaf_lock_set", lambda *_args: None)
    monkeypatch.setattr(oaf_tf1, "_verify_model_cache", lambda *_args: None)
    monkeypatch.setattr(oaf_tf1.platform, "system", lambda: native_system)
    monkeypatch.setattr(oaf_tf1.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        oaf_tf1,
        "build_changed_file_manifest",
        lambda *_args: changed_files,
    )
    captured_attestations: list[dict[str, object]] = []

    def publish_attestation(
        repository_root: Path,
        backend_root: Path,
        **kwargs: Any,
    ) -> PublishedArtifact:
        captured_attestations.append(dict(kwargs))
        content = canonical_json_bytes(
            {
                "backend_id": BACKEND_ID,
                "changed_files_manifest": None,
                "checkout_dirty": False,
                "cpu_limit": kwargs["conditions"].cpu_limit,
                "descriptor_sha256": descriptor.sha256,
                "git_commit": "f" * 40,
                "host_numeric_fingerprint": kwargs["expected_host_numeric_fingerprint"].as_json(),
                "memory_bytes": kwargs["conditions"].memory_bytes,
                "pid_limit": kwargs["conditions"].pid_limit,
                "request_deadline_seconds": kwargs["conditions"].request_deadline_seconds,
                "schema": "crux.backend-execution-attestation/v1",
                "shm_bytes": kwargs["conditions"].shm_bytes,
                "startup_deadline_seconds": kwargs["conditions"].startup_deadline_seconds,
                "strict_mode": kwargs["strict_mode"],
                "tmp_bytes": kwargs["conditions"].tmp_bytes,
            },
            trailing_newline=True,
        )
        path = backend_root / "attestations" / "fake.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return PublishedArtifact(
            role="execution_attestation",
            path=path,
            sha256=sha256_hex(content),
        )

    monkeypatch.setattr(oaf_tf1, "publish_execution_attestation", publish_attestation)

    config = oaf_tf1.OafBackendConfig(
        backend_lock_path=backend_lock.path,
        runtime_lock_path=runtime_lock.path,
        seal_evidence_path=seal.path,
        conversion_audit_path=audit.path,
        host_adapter_source_manifest_path=host_manifest,
        model_cache_root=model_root,
        input_root=input_root,
        native_host_evidence=evidence,
        allow_emulated_diagnostics=allow_emulated_diagnostics,
        strict_checkout=strict_checkout,
    )
    return Harness(
        backend=oaf_tf1.OafTf1Backend(config, process_factory=process_factory),
        config=config,
        process=process,
        descriptor=descriptor,
        smoke_audio=smoke_audio,
        expected_smoke_sha256=expected_smoke_sha256,
        captured_profiles=captured_profiles,
        captured_attestations=captured_attestations,
    )


def test_verify_model_cache_passes_the_final_lock_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "config" / f"{BACKEND_ID}.backend-lock.json"
    backend_lock = LoadedBackendLock(
        path=lock_path,
        payload={},
        sha256=BACKEND_SHA256,
        descriptor=_descriptor(),
        max_input_audio_frames=441000,
    )
    cache_root = tmp_path / "model-cache" / "sha256" / ("7" * 64)
    captured: list[object] = []

    def prepare(request: object, *, backend_lock: object) -> object:
        captured.extend((request, backend_lock))
        return type("Outcome", (), {"status": "ready", "model_cache_path": cache_root})()

    monkeypatch.setattr(oaf_tf1, "prepare_oaf_backend", prepare)

    oaf_tf1._verify_model_cache(backend_lock, cache_root, lock_path)

    request = captured[0]
    assert getattr(request, "backend_lock_path") == lock_path


def test_verify_backend_accepts_only_matching_handshake_and_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)

    verification = harness.backend.verify()

    assert verification.status == "verified"
    assert verification.max_input_audio_frames == 441000
    assert (
        verification.host_numeric_fingerprint
        == harness.config.native_host_evidence.host_numeric_fingerprint
    )
    assert (
        harness.captured_attestations[0]["expected_host_numeric_fingerprint"]
        == verification.host_numeric_fingerprint
    )
    assert verification.smoke.prediction is not None
    assert verification.smoke.prediction.sha256 == harness.expected_smoke_sha256
    assert verification.tensor_coverage.required_count == 78
    assert verification.tensor_coverage.restored_count == 78
    assert verification.tensor_coverage.non_inference_count == 52
    assert harness.process.request_count == 1
    assert harness.captured_profiles[0].seal_evidence_path == harness.config.seal_evidence_path


def _item_audio(harness: Harness, *, beneath_root: bool = True) -> CanonicalAudio:
    root = harness.config.input_root if beneath_root else harness.config.input_root.parent
    audio_path = root / "items" / "audio.wav"
    audio_path.parent.mkdir(exist_ok=True)
    audio_path.write_bytes(_wav_bytes())
    return load_direct_audio(
        audio_path,
        source_audio_id="source-v1",
        input_view_id="input-v1",
        max_input_audio_frames=441000,
    )


def test_verify_backend_transcribe_never_falls_back_after_runner_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        request_error="backend_process_died",
    )
    verification = harness.backend.verify()
    assert verification.status == "verified"

    with pytest.raises(oaf_tf1.OafBackendFatal, match="backend_process_died"):
        harness.backend.transcribe(_item_audio(harness))


READY_IDENTITY_FIELDS = (
    "backend_descriptor_sha256",
    "backend_lock_sha256",
    "checkpoint_inventory_sha256",
    "non_inference_count",
    "non_inference_inventory_sha256",
    "protocol_schema",
    "python_version",
    "required_inference_count",
    "required_inference_inventory_sha256",
    "restored_inference_count",
    "runner_source_manifest_sha256",
    "runtime_lock_sha256",
    "smoke_audio_sha256",
    "smoke_oracle_sha256",
    "smoke_prediction_sha256",
    "smoke_status",
    "tensorflow_abi",
    "tensorflow_build",
    "type",
    "upstream_source_manifest_sha256",
)


@pytest.mark.parametrize("field", READY_IDENTITY_FIELDS)
def test_verify_backend_rejects_every_ready_identity_field_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch, handshake_changes={field: "wrong"})

    verification = harness.backend.verify()

    assert verification.status == "failed"
    assert verification.errors[0].code == "backend_handshake_mismatch"
    assert harness.process.request_count == 0


@pytest.mark.parametrize("field", tuple(_descriptor().payload))
def test_verify_backend_rejects_every_descriptor_field_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    descriptor = dict(_descriptor().payload)
    descriptor[field] = "wrong"
    harness = _harness(
        tmp_path,
        monkeypatch,
        handshake_changes={"backend_descriptor": descriptor},
    )

    verification = harness.backend.verify()

    assert verification.status == "failed"
    assert verification.errors[0].code == "backend_handshake_mismatch"
    assert harness.process.request_count == 0


@pytest.mark.parametrize(
    "handshake_changes",
    (
        {"unexpected": "field"},
        {"backend_lock_sha256": None},
    ),
)
def test_verify_backend_rejects_extra_or_missing_ready_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handshake_changes: dict[str, object],
) -> None:
    harness = _harness(tmp_path, monkeypatch, handshake_changes=handshake_changes)
    if "backend_lock_sha256" in handshake_changes:
        del harness.process.handshake["backend_lock_sha256"]

    verification = harness.backend.verify()

    assert verification.status == "failed"
    assert verification.errors[0].code == "backend_handshake_mismatch"


def test_verify_backend_rejects_changed_smoke_bytes_before_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    (harness.config.input_root / "smoke" / "canonical.wav").write_bytes(_wav_bytes(6))

    verification = harness.backend.verify()

    assert verification.status == "failed"
    assert verification.errors[0].code == "smoke_audio_mismatch"
    assert harness.process.request_count == 0


def test_verify_backend_rejects_post_ready_smoke_artifact_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        response_changes={"native_events": []},
    )

    verification = harness.backend.verify()

    assert verification.status == "failed"
    assert verification.errors[0].code == "smoke_prediction_mismatch"
    assert harness.process.request_count == 1


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    (
        ({"audio_sha256": "0" * 64}, "backend_response_identity_mismatch"),
        ({"backend_descriptor_sha256": "0" * 64}, "backend_response_identity_mismatch"),
    ),
)
def test_verify_backend_rejects_response_identity_mismatch_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    expected_code: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch, after_smoke_changes=changes)
    assert harness.backend.verify().status == "verified"

    with pytest.raises(oaf_tf1.OafBackendFatal) as captured:
        harness.backend.transcribe(_item_audio(harness))

    assert captured.value.error.code == expected_code
    assert harness.process.request_count == 2


def test_verify_backend_preserves_typed_item_error_from_real_inner_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        after_smoke_changes={
            "code": "input_wav_invalid",
            "message": "The canonical input does not match the locked WAV contract.",
            "type": "transcription_error",
        },
    )
    assert harness.backend.verify().status == "verified"

    with pytest.raises(oaf_tf1.OafBackendItem) as captured:
        harness.backend.transcribe(_item_audio(harness, beneath_root=False))

    assert captured.value.error.code == "input_wav_invalid"
    assert harness.process.request_count == 2


def test_verify_backend_snapshots_private_host_input_beneath_mounted_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    assert harness.backend.verify().status == "verified"
    audio = _item_audio(harness, beneath_root=False)
    original_content = audio.path.read_bytes()
    captured_snapshot: list[Path] = []

    def inspect_request(payload: dict[str, object], request_count: int) -> None:
        if request_count == 1:
            return
        relative_path = Path(payload["audio_path"])  # type: ignore[arg-type]
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        snapshot_path = harness.config.input_root / relative_path
        assert snapshot_path.read_bytes() == original_content
        captured_snapshot.append(snapshot_path)

    harness.process.request_hook = inspect_request
    prediction = harness.backend.transcribe(audio)

    assert prediction.audio == audio
    assert harness.process.request_count == 2
    assert len(captured_snapshot) == 1
    assert not captured_snapshot[0].exists()
    assert not captured_snapshot[0].parent.exists()


def test_verify_backend_rejects_private_host_input_hash_drift_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    assert harness.backend.verify().status == "verified"
    audio = _item_audio(harness, beneath_root=False)
    audio.path.write_bytes(_wav_bytes(5))

    with pytest.raises(oaf_tf1.OafBackendItem) as captured:
        harness.backend.transcribe(audio)

    assert captured.value.error.code == "input_hash_mismatch"
    assert harness.process.request_count == 1


def test_verify_backend_rejects_private_host_input_symlink_swap_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    assert harness.backend.verify().status == "verified"
    audio = _item_audio(harness, beneath_root=False)
    replacement = audio.path.with_name("replacement.wav")
    replacement.write_bytes(audio.path.read_bytes())
    audio.path.unlink()
    audio.path.symlink_to(replacement)

    with pytest.raises(oaf_tf1.OafBackendItem) as captured:
        harness.backend.transcribe(audio)

    assert captured.value.error.code == "input_path_invalid"
    assert harness.process.request_count == 1


def test_verify_backend_rejects_generated_snapshot_path_escape_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    assert harness.backend.verify().status == "verified"
    audio = _item_audio(harness, beneath_root=False)
    escaped_path = harness.config.input_root.parent / "escaped.wav"
    escaped_path.write_bytes(audio.path.read_bytes())

    class EscapedSnapshot:
        path = escaped_path

        @staticmethod
        def verify() -> None:
            return None

    @contextmanager
    def escaped_snapshot(*_args: object, **_kwargs: object):
        yield EscapedSnapshot()

    monkeypatch.setattr(oaf_tf1, "open_private_file_snapshot", escaped_snapshot)

    with pytest.raises(oaf_tf1.OafBackendFatal) as captured:
        harness.backend.transcribe(audio)

    assert captured.value.error.code == "backend_input_snapshot_invalid"
    assert harness.process.request_count == 1


def test_verify_backend_detects_request_snapshot_swap_and_cleans_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    assert harness.backend.verify().status == "verified"
    audio = _item_audio(harness, beneath_root=False)
    captured_snapshot: list[Path] = []

    def replace_snapshot(payload: dict[str, object], request_count: int) -> None:
        if request_count == 1:
            return
        snapshot_path = harness.config.input_root / str(payload["audio_path"])
        captured_snapshot.append(snapshot_path)
        snapshot_path.parent.chmod(0o700)
        snapshot_path.unlink()
        snapshot_path.write_bytes(b"swapped")

    harness.process.request_hook = replace_snapshot

    with pytest.raises(oaf_tf1.OafBackendFatal) as captured:
        harness.backend.transcribe(audio)

    assert captured.value.error.code == "backend_input_snapshot_changed"
    assert harness.process.request_count == 2
    assert len(captured_snapshot) == 1
    assert not captured_snapshot[0].exists()
    assert not captured_snapshot[0].parent.exists()


def test_verify_backend_treats_request_snapshot_cleanup_failure_as_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    assert harness.backend.verify().status == "verified"
    audio = _item_audio(harness, beneath_root=False)
    real_snapshot = oaf_tf1.open_private_file_snapshot

    @contextmanager
    def cleanup_failure(*args: object, **kwargs: object):
        with real_snapshot(*args, **kwargs) as snapshot:
            yield snapshot
        raise PrivateSnapshotIntegrityError("injected cleanup failure")

    monkeypatch.setattr(oaf_tf1, "open_private_file_snapshot", cleanup_failure)

    with pytest.raises(oaf_tf1.OafBackendFatal) as captured:
        harness.backend.transcribe(audio)

    assert captured.value.error.code == "backend_input_snapshot_changed"
    assert harness.process.request_count == 2


@pytest.mark.parametrize(
    "event",
    (
        {**_runner_event(), "velocity": float("inf")},
        {**_runner_event(), "native_midi_note": 37},
        {**_runner_event(), "unexpected": "field"},
    ),
)
def test_verify_backend_rejects_malformed_or_nonfinite_native_event_as_item_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event: dict[str, object],
) -> None:
    harness = _harness(tmp_path, monkeypatch, after_smoke_event=event)
    assert harness.backend.verify().status == "verified"

    with pytest.raises(oaf_tf1.OafBackendItem) as captured:
        harness.backend.transcribe(_item_audio(harness))

    assert captured.value.error.code == "native_event_invalid"
    assert harness.process.request_count == 2


@pytest.mark.parametrize(
    "failure_code",
    ("backend_request_timeout", "backend_process_died", "backend_oom_killed"),
)
def test_verify_backend_timeout_death_and_oom_are_fatal_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch, request_error=failure_code)
    assert harness.backend.verify().status == "verified"

    with pytest.raises(oaf_tf1.OafBackendFatal) as captured:
        harness.backend.transcribe(_item_audio(harness))

    assert captured.value.error.code == failure_code
    assert harness.process.request_count == 2


def test_verify_backend_emulated_environment_does_not_launch_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, native_system="Darwin")

    verification = harness.backend.verify()

    assert verification.status == "environment_unsupported"
    assert harness.captured_profiles == []
    assert harness.process.request_count == 0


def test_verify_backend_rejects_native_evidence_mismatch_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, native_evidence_matches=False)

    verification = harness.backend.verify()

    assert verification.status == "failed"
    assert verification.errors[0].code == "native_host_evidence_mismatch"
    assert harness.captured_profiles == []


def test_verify_backend_emulated_diagnostics_runs_strongest_check_without_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        native_system="Darwin",
        allow_emulated_diagnostics=True,
    )

    verification = harness.backend.verify()

    assert verification.status == "environment_unsupported"
    assert verification.tensor_coverage.status == "passed"
    assert verification.smoke.status == "passed"
    assert verification.smoke.prediction is None
    assert harness.process.request_count == 1
    assert harness.process.close_count == 1


def test_verify_backend_strict_checkout_rejects_inference_relevant_change_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        changed_files=({"path": "src/benchmark/backends/oaf_tf1.py"},),
    )

    verification = harness.backend.verify()

    assert verification.status == "failed"
    assert verification.errors[0].code == "inference_source_dirty"
    assert harness.captured_profiles == []


def test_verify_backend_unrelated_dirty_documentation_does_not_enter_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, changed_files=())

    verification = harness.backend.verify()

    assert verification.status == "verified"
    assert len(harness.captured_attestations[0]["source_manifests"]) == 2  # type: ignore[arg-type]


def test_verify_backend_launch_and_attestation_use_exact_sealed_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)

    assert harness.backend.verify().status == "verified"

    profile = harness.captured_profiles[0]
    assert profile.image_manifest_digest == f"sha256:{'4' * 64}"
    assert profile.uid == 10001
    assert profile.gid == 10002
    assert profile.cpu_limit == "2.5"
    assert profile.memory_bytes == 1073741824
    assert profile.pid_limit == 64
    assert profile.tmp_bytes == 134217728
    assert profile.shm_bytes == 67108864
    assert profile.startup_deadline_seconds == 120
    assert profile.request_deadline_seconds == 60
    assert profile.seal_evidence_path == harness.config.seal_evidence_path
    conditions = harness.captured_attestations[0]["conditions"]
    assert conditions.cpu_limit == profile.cpu_limit  # type: ignore[union-attr]
    assert conditions.memory_bytes == profile.memory_bytes  # type: ignore[union-attr]
    assert conditions.pid_limit == profile.pid_limit  # type: ignore[union-attr]


def test_verify_backend_close_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    assert harness.backend.verify().status == "verified"

    harness.backend.close()
    harness.backend.close()

    assert harness.process.close_count == 1
