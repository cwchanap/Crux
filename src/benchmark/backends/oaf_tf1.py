# The frozen adapter validates an intentionally broad explicit protocol surface.
# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-locals
# pylint: disable=unidiomatic-typecheck,broad-exception-caught,try-except-raise
# pylint: disable=duplicate-code,import-outside-toplevel
from __future__ import annotations

import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Protocol, cast

from src.benchmark.backend_attestation import (
    ExecutionConditions,
    build_changed_file_manifest,
    publish_execution_attestation,
)
from src.benchmark.backend_identity import (
    OAF_BACKEND_ID as OFFICIAL_BACKEND_ID,
)
from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backend_lock import (
    REQUIRED_ENVIRONMENT,
    BackendLockError,
    LoadedBackendLock,
    LoadedConversionAudit,
    LoadedRuntimeLock,
    LoadedSealEvidence,
    load_backend_lock,
    load_conversion_audit,
    load_runtime_lock,
    load_seal_evidence,
    validate_oaf_lock_set,
)
from src.benchmark.backend_prepare import (
    PrepareBackendRequest,
    prepare_oaf_backend,
)
from src.benchmark.backend_process import (
    NativeHostEvidence,
    RunnerLaunchProfile,
    RunnerProcess,
    RunnerResponse,
)
from src.benchmark.backend_publication import (
    PrivateFileSnapshot,
    PrivateSnapshotIntegrityError,
    open_private_file_snapshot,
    read_regular_file_no_follow,
)
from src.benchmark.backends import (
    BackendError,
    BackendFatalFailure,
    BackendItemFailure,
    BackendVerification,
    CanonicalAudio,
    NativeEvent,
    NativePrediction,
    PublishedArtifact,
    SmokeCheck,
    TensorCoverageCheck,
)
from src.benchmark.input_view import load_direct_audio
from src.benchmark.prediction_artifact import (
    PredictionArtifactError,
    publish_prediction_artifact,
    render_prediction_artifact,
)

OafBackendFatal = BackendFatalFailure
OafBackendItem = BackendItemFailure

_PROTOCOL_SCHEMA = "crux.transcription-runner/v1"
_PROTOCOL_RESPONSE_GOLDEN_SCHEMA = "crux.transcription-runner-response/v1"
_SMOKE_ORACLE_SCHEMA = "crux.oaf-smoke-oracle/v1"
_SMOKE_ORACLE_KEYS = frozenset(
    {
        "input_audio_frame_count",
        "input_audio_sha256",
        "input_view_id",
        "native_events",
        "schema",
        "source_audio_id",
        "source_audio_sha256",
    }
)
_READY_KEYS = frozenset(
    {
        "backend_descriptor",
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
    }
)
_RESULT_KEYS = frozenset(
    {
        "audio_sha256",
        "backend_descriptor_sha256",
        "native_events",
        "type",
    }
)
_ERROR_KEYS = frozenset({"code", "message", "type"})


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Validate canonical request and response fixtures at the host boundary."""
    if schema not in {_PROTOCOL_SCHEMA, _PROTOCOL_RESPONSE_GOLDEN_SCHEMA}:
        raise ValueError("host runner schema golden is unsupported")
    from runtime.oaf_tf1.protocol import (
        ProtocolFailure,
        validate_transcribe_request,
        validate_transcription_response,
    )

    try:
        if not content.endswith(b"\n") or content.endswith(b"\n\n"):
            raise ValueError("host runner schema golden newline is invalid")
        payload = strict_json_loads(content[:-1], require_canonical=True)
        if schema == _PROTOCOL_SCHEMA:
            validate_transcribe_request(payload, expected_descriptor_sha256="a" * 64)
            return
        validate_transcription_response(payload)
    except (ImportError, StrictJsonError, ProtocolFailure, TypeError, ValueError):
        raise ValueError("host runner schema golden is invalid") from None


@dataclass(frozen=True)
class OafBackendConfig:
    backend_lock_path: Path
    runtime_lock_path: Path
    seal_evidence_path: Path
    conversion_audit_path: Path
    host_adapter_source_manifest_path: Path
    model_cache_root: Path
    input_root: Path
    native_host_evidence: NativeHostEvidence
    allow_emulated_diagnostics: bool
    strict_checkout: bool

    def __post_init__(self) -> None:
        for field in (
            "backend_lock_path",
            "runtime_lock_path",
            "seal_evidence_path",
            "conversion_audit_path",
            "host_adapter_source_manifest_path",
            "model_cache_root",
            "input_root",
        ):
            value = getattr(self, field)
            if not isinstance(value, Path):
                raise TypeError(f"{field} must be a Path")
        if not isinstance(self.native_host_evidence, NativeHostEvidence):
            raise TypeError("native_host_evidence must be accepted native evidence")
        for field in ("allow_emulated_diagnostics", "strict_checkout"):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be boolean")


class _Runner(Protocol):
    @property
    def handshake(self) -> Mapping[str, JsonValue]: ...

    def request(
        self,
        payload: Mapping[str, JsonValue],
        *,
        deadline_seconds: int | None = None,
    ) -> RunnerResponse: ...

    def close(self) -> None: ...


ProcessFactory = Callable[[RunnerLaunchProfile], _Runner]


@dataclass(frozen=True)
class _Locks:
    backend: LoadedBackendLock
    runtime: LoadedRuntimeLock
    seal: LoadedSealEvidence
    audit: LoadedConversionAudit


@dataclass(frozen=True)
class SmokeVerificationArtifacts:
    """The three post-ready artifacts compared during one verification run."""

    persistent_first: PublishedArtifact
    persistent_second: PublishedArtifact
    fresh_first: PublishedArtifact

    @property
    def artifacts(self) -> tuple[PublishedArtifact, PublishedArtifact, PublishedArtifact]:
        return self.persistent_first, self.persistent_second, self.fresh_first


class OafTf1Backend:
    def __init__(
        self,
        config: OafBackendConfig,
        *,
        process_factory: ProcessFactory = RunnerProcess.start,
    ) -> None:
        if not isinstance(config, OafBackendConfig):
            raise TypeError("OaF backend config is required")
        self._config = config
        self._process_factory = process_factory
        self._locks: _Locks | None = None
        self._process: _Runner | None = None
        self._verification: BackendVerification | None = None
        self._smoke_verification_artifacts: SmokeVerificationArtifacts | None = None
        self._closed = False

    @property
    def smoke_verification_artifacts(self) -> SmokeVerificationArtifacts | None:
        """Return only the three post-ready smoke artifacts, never startup handshakes."""
        return self._smoke_verification_artifacts

    def descriptor(self):
        try:
            return self._load_locks().backend.descriptor
        except BackendFatalFailure:
            raise
        except (OSError, TypeError, ValueError):
            raise _fatal(
                "backend_integrity_failed",
                "The frozen backend identity could not be loaded.",
            ) from None

    def verify(self) -> BackendVerification:
        if self._verification is not None:
            return self._verification
        if self._closed:
            raise _fatal("backend_process_closed", "The frozen backend is closed.")
        try:
            verification = self._verify_once()
        except BackendFatalFailure as failure:
            self._close_process()
            verification = self._failed_verification(failure.error)
        except (OSError, RuntimeError, TypeError, ValueError):
            self._close_process()
            verification = self._failed_verification(
                BackendError(
                    code="backend_integrity_failed",
                    message="The frozen backend failed an integrity check.",
                )
            )
        self._verification = verification
        return verification

    def _verify_once(self) -> BackendVerification:
        locks = self._load_locks()
        _validate_runtime_environment(locks.runtime)
        repository_root = _repository_root(self._config.host_adapter_source_manifest_path)
        runner_manifest = repository_root / "runtime" / "oaf_tf1" / "runner-source-manifest.json"
        upstream_manifest = repository_root / "runtime" / "oaf_tf1" / "source-manifest.json"
        tensor_report = (
            repository_root
            / "docs"
            / "superpowers"
            / "evidence"
            / "hpa-320"
            / "oaf-tensor-coverage.json"
        )

        _require_artifact_hash(
            self._config.host_adapter_source_manifest_path,
            locks.backend.payload["host_adapter_source_manifest_sha256"],
            "host adapter source manifest",
        )
        _require_artifact_hash(
            runner_manifest,
            locks.runtime.payload["runner_source_manifest_sha256"],
            "runner source manifest",
        )
        _require_artifact_hash(
            upstream_manifest,
            locks.runtime.payload["upstream_source_manifest_sha256"],
            "upstream source manifest",
        )
        tensor_artifact = _tensor_artifact(
            tensor_report,
            locks.seal.payload["tensor_coverage_sha256"],
        )
        _validate_native_evidence(locks.seal, self._config.native_host_evidence)

        native_environment = _is_native_environment()
        if not native_environment:
            return self._unsupported_verification()

        source_manifests = (
            self._config.host_adapter_source_manifest_path,
            runner_manifest,
        )
        changed_files = build_changed_file_manifest(repository_root, source_manifests)
        if self._config.strict_checkout and changed_files:
            raise _fatal(
                "inference_source_dirty",
                "Strict verification rejects inference-relevant source changes.",
            )

        _verify_model_cache(
            locks.backend,
            self._config.model_cache_root,
            self._config.backend_lock_path,
        )
        profile = _launch_profile(self._config, locks)
        smoke_audio = _load_smoke_audio(self._config, locks)
        persistent = self._start_verified_process(profile, locks)
        self._process = persistent
        fresh: _Runner | None = None
        try:
            persistent_first = self._request_smoke_prediction(smoke_audio, persistent)
            persistent_second = self._request_smoke_prediction(smoke_audio, persistent)
            fresh = self._start_verified_process(profile, locks)
            fresh_first = self._request_smoke_prediction(smoke_audio, fresh)
        finally:
            if fresh is not None:
                try:
                    fresh.close()
                except BaseException:
                    pass

        expected_smoke = _expected_smoke_prediction(self._config, locks, smoke_audio)
        _verify_smoke_prediction_bytes(
            expected_smoke,
            (persistent_first, persistent_second, fresh_first),
        )

        backend_root = (
            repository_root / "artifacts" / "benchmark" / "backends" / OFFICIAL_BACKEND_ID
        )
        smoke_artifacts = _publish_smoke_verification_artifacts(
            backend_root,
            (persistent_first, persistent_second, fresh_first),
        )
        self._smoke_verification_artifacts = smoke_artifacts

        conditions = _execution_conditions(locks.seal)
        attestation = publish_execution_attestation(
            repository_root,
            backend_root,
            backend_id=OFFICIAL_BACKEND_ID,
            descriptor_sha256=locks.backend.descriptor.sha256,
            source_manifests=source_manifests,
            strict_mode=self._config.strict_checkout,
            conditions=conditions,
            expected_host_numeric_fingerprint=(
                self._config.native_host_evidence.host_numeric_fingerprint
            ),
        )
        return BackendVerification(
            status="verified",
            descriptor=locks.backend.descriptor,
            max_input_audio_frames=locks.backend.max_input_audio_frames,
            backend_lock_sha256=locks.backend.sha256,
            runtime_lock_sha256=locks.runtime.sha256,
            parameter_lock_sha256=None,
            seal_evidence_sha256=locks.seal.sha256,
            execution_attestation=attestation,
            tensor_coverage=_tensor_check(locks, tensor_artifact),
            smoke=SmokeCheck(
                status="passed",
                audio_sha256=smoke_audio.input_audio_sha256,
                oracle_sha256=cast(str, locks.backend.payload["smoke_oracle_sha256"]),
                prediction=smoke_artifacts.persistent_first,
            ),
            errors=(),
            host_numeric_fingerprint=self._config.native_host_evidence.host_numeric_fingerprint,
        )

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        if self._closed:
            raise _fatal("backend_process_closed", "The frozen backend is closed.")
        verification = self.verify()
        if verification.status != "verified":
            error = (
                verification.errors[0]
                if verification.errors
                else BackendError(
                    code="backend_not_verified",
                    message="The frozen backend is not verified.",
                )
            )
            raise BackendFatalFailure(error)
        return self._request_prediction(audio)

    def _request_prediction(
        self,
        audio: CanonicalAudio,
        *,
        process: _Runner | None = None,
    ) -> NativePrediction:
        locks = self._load_locks()
        active_process = self._process if process is None else process
        if active_process is None:
            raise _fatal("backend_not_verified", "The frozen backend is not verified.")
        response = self._request_runner(audio, locks, active_process)
        payload = dict(response.payload)
        if payload.get("type") == "transcription_error":
            if set(payload) != _ERROR_KEYS:
                raise _fatal(
                    "backend_protocol_invalid",
                    "The backend runner emitted a malformed item error.",
                )
            code = payload.get("code")
            if not isinstance(code, str):
                raise _fatal(
                    "backend_protocol_invalid",
                    "The backend runner emitted an invalid item error code.",
                )
            try:
                error = BackendError(
                    code=code,
                    message="The frozen backend rejected the canonical input.",
                )
            except ValueError:
                raise _fatal(
                    "backend_protocol_invalid",
                    "The backend runner emitted an invalid item error code.",
                ) from None
            raise BackendItemFailure(error)
        if set(payload) != _RESULT_KEYS or payload.get("type") != "transcription_result":
            raise _fatal(
                "backend_protocol_invalid",
                "The backend runner emitted a malformed transcription response.",
            )
        if (
            payload.get("audio_sha256") != audio.input_audio_sha256
            or payload.get("backend_descriptor_sha256") != locks.backend.descriptor.sha256
        ):
            raise _fatal(
                "backend_response_identity_mismatch",
                "The backend runner response identity did not match the request.",
            )
        raw_events = payload.get("native_events")
        if not isinstance(raw_events, (list, tuple)):
            raise _item("native_event_invalid", "The runner event list is malformed.")
        events = tuple(_native_event(event, locks.backend) for event in raw_events)
        prediction = NativePrediction(
            audio=audio,
            descriptor=locks.backend.descriptor,
            events=events,
            backend_lock_sha256=locks.backend.sha256,
            runtime_lock_sha256=locks.runtime.sha256,
            parameter_lock_sha256=None,
            model_artifact_set_sha256=cast(
                str,
                locks.backend.descriptor.payload["model_artifact_set_sha256"],
            ),
            upstream_source_commit=cast(
                str,
                locks.backend.descriptor.payload["upstream_source_commit"],
            ),
            training_data_map_id=cast(
                str,
                locks.backend.descriptor.payload["training_data_map_id"],
            ),
        )
        try:
            render_prediction_artifact(prediction)
        except PredictionArtifactError as error:
            code = (
                "duplicate_native_event"
                if str(error) == "duplicate_native_event"
                else "native_event_invalid"
            )
            raise _item(code, "The runner event list is malformed.") from None
        return prediction

    def _start_verified_process(self, profile: RunnerLaunchProfile, locks: _Locks) -> _Runner:
        try:
            process = self._process_factory(profile)
        except BackendFatalFailure:
            raise
        except (OSError, RuntimeError, TypeError, ValueError):
            raise _fatal(
                "backend_launch_failed",
                "The frozen backend runner could not be launched.",
            ) from None
        try:
            _validate_handshake(process.handshake, locks)
        except BaseException:
            try:
                process.close()
            except BaseException:
                pass
            raise
        return process

    def _request_smoke_prediction(
        self,
        audio: CanonicalAudio,
        process: _Runner,
    ) -> NativePrediction:
        try:
            return self._request_prediction(audio, process=process)
        except BackendItemFailure:
            raise _fatal(
                "smoke_mismatch",
                "The post-ready smoke prediction did not match the frozen oracle.",
            ) from None

    def _request_runner(
        self,
        audio: CanonicalAudio,
        locks: _Locks,
        process: _Runner,
    ) -> RunnerResponse:
        content = _read_authenticated_input(audio)
        try:
            with open_private_file_snapshot(
                content,
                audio.input_audio_sha256,
                root=self._config.input_root,
            ) as snapshot:
                _verify_request_snapshot(snapshot)
                try:
                    relative_path = _relative_input_path(
                        snapshot.path,
                        self._config.input_root,
                    )
                except BackendItemFailure:
                    raise _fatal(
                        "backend_input_snapshot_invalid",
                        "The private runner input escaped the mounted input root.",
                    ) from None
                try:
                    response = process.request(
                        {
                            "audio_path": relative_path,
                            "audio_sha256": audio.input_audio_sha256,
                            "backend_descriptor_sha256": locks.backend.descriptor.sha256,
                            "type": "transcribe",
                        },
                        deadline_seconds=cast(
                            int,
                            locks.seal.payload["request_deadline_seconds"],
                        ),
                    )
                except BaseException:
                    _verify_request_snapshot(snapshot)
                    raise
                _verify_request_snapshot(snapshot)
        except BackendFatalFailure:
            raise
        except PrivateSnapshotIntegrityError:
            raise _fatal(
                "backend_input_snapshot_changed",
                "The private runner input changed during the request.",
            ) from None
        except (OSError, RuntimeError, TypeError, ValueError):
            raise _fatal(
                "backend_input_snapshot_invalid",
                "The private runner input could not be staged safely.",
            ) from None
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_process()

    def _close_process(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            try:
                process.close()
            except BaseException:
                pass

    def _load_locks(self) -> _Locks:
        if self._locks is not None:
            return self._locks
        try:
            locks = _Locks(
                backend=load_backend_lock(self._config.backend_lock_path),
                runtime=load_runtime_lock(self._config.runtime_lock_path),
                seal=load_seal_evidence(self._config.seal_evidence_path),
                audit=load_conversion_audit(self._config.conversion_audit_path),
            )
            validate_oaf_lock_set(
                locks.backend,
                locks.runtime,
                locks.seal,
                locks.audit,
            )
        except (BackendLockError, OSError, TypeError, ValueError):
            raise _fatal(
                "backend_lock_invalid",
                "The frozen backend lock set is invalid.",
            ) from None
        self._locks = locks
        return locks

    def _failed_verification(self, error: BackendError) -> BackendVerification:
        locks = self._locks
        return BackendVerification(
            status="failed",
            descriptor=None if locks is None else locks.backend.descriptor,
            max_input_audio_frames=(
                None if locks is None else locks.backend.max_input_audio_frames
            ),
            backend_lock_sha256=None if locks is None else locks.backend.sha256,
            runtime_lock_sha256=None if locks is None else locks.runtime.sha256,
            parameter_lock_sha256=None,
            seal_evidence_sha256=None if locks is None else locks.seal.sha256,
            execution_attestation=None,
            tensor_coverage=_empty_tensor_check(),
            smoke=_empty_smoke_check(),
            errors=(error,),
            host_numeric_fingerprint=None,
        )

    def _unsupported_verification(
        self,
        *,
        tensor: PublishedArtifact | None = None,
        smoke_audio_sha256: str | None = None,
        smoke_oracle_sha256: str | None = None,
    ) -> BackendVerification:
        locks = self._load_locks()
        tensor_check = (
            _empty_tensor_check()
            if tensor is None
            else _tensor_check(locks, tensor, status="passed")
        )
        smoke = SmokeCheck(
            status="not_run" if smoke_audio_sha256 is None else "passed",
            audio_sha256=smoke_audio_sha256,
            oracle_sha256=smoke_oracle_sha256,
            prediction=None,
        )
        return BackendVerification(
            status="environment_unsupported",
            descriptor=locks.backend.descriptor,
            max_input_audio_frames=locks.backend.max_input_audio_frames,
            backend_lock_sha256=locks.backend.sha256,
            runtime_lock_sha256=locks.runtime.sha256,
            parameter_lock_sha256=None,
            seal_evidence_sha256=locks.seal.sha256,
            execution_attestation=None,
            tensor_coverage=tensor_check,
            smoke=smoke,
            errors=(
                BackendError(
                    code="environment_unsupported",
                    message="Official OaF execution requires a native Linux AMD64 worker.",
                ),
            ),
            host_numeric_fingerprint=self._config.native_host_evidence.host_numeric_fingerprint,
        )


def _repository_root(manifest_path: Path) -> Path:
    candidate = Path(os.path.abspath(manifest_path))
    for parent in (candidate.parent, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    raise _fatal(
        "repository_identity_invalid",
        "The host adapter manifest is not inside a repository.",
    )


def _require_artifact_hash(path: Path, expected: object, label: str) -> bytes:
    digest = _require_hash_value(expected, f"{label} SHA-256")
    try:
        content = read_regular_file_no_follow(path)
    except OSError:
        raise _fatal(
            "backend_evidence_invalid",
            f"The {label} is unavailable.",
        ) from None
    if sha256_hex(content) != digest:
        raise _fatal(
            "backend_evidence_invalid",
            f"The {label} hash does not match its lock.",
        )
    return content


def _tensor_artifact(path: Path, expected: object) -> PublishedArtifact:
    content = _require_artifact_hash(path, expected, "tensor coverage report")
    try:
        if not content.endswith(b"\n"):
            raise StrictJsonError("missing newline")
        payload = strict_json_loads(content[:-1], require_canonical=True)
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("schema"), str)
            or not payload["schema"]
        ):
            raise StrictJsonError("missing schema")
    except (StrictJsonError, TypeError, ValueError):
        raise _fatal(
            "backend_evidence_invalid",
            "The tensor coverage report is malformed.",
        ) from None
    return PublishedArtifact(
        role="tensor_coverage",
        path=path,
        sha256=sha256_hex(content),
    )


def _validate_native_evidence(
    seal: LoadedSealEvidence,
    evidence: NativeHostEvidence,
) -> None:
    record = seal.payload.get("native_host_evidence")
    if (
        not isinstance(record, Mapping)
        or set(record) != {"kind", "official_execution_allowed", "payload", "sha256"}
        or record.get("kind") != evidence.kind
        or record.get("official_execution_allowed") != evidence.official_execution_allowed
        or record.get("sha256") != evidence.sha256
        or sha256_hex(canonical_json_bytes(record.get("payload"))) != evidence.sha256
    ):
        raise _fatal(
            "native_host_evidence_mismatch",
            "Native host evidence does not match the accepted seal.",
        )


def _is_native_environment() -> bool:
    return platform.system() == "Linux" and platform.machine().lower() in {
        "amd64",
        "x86_64",
    }


def _verify_model_cache(
    backend_lock: LoadedBackendLock,
    model_cache_root: Path,
    backend_lock_path: Path,
) -> None:
    model_identity = backend_lock.descriptor.payload["model_artifact_set_sha256"]
    if model_cache_root.name != model_identity or model_cache_root.parent.name != "sha256":
        raise _fatal(
            "backend_model_invalid",
            "The checkpoint cache path does not match the frozen model identity.",
        )
    cache_root = model_cache_root.parent.parent
    outcome = prepare_oaf_backend(
        PrepareBackendRequest(
            backend_id=OFFICIAL_BACKEND_ID,
            cache_root=cache_root,
            archive_path=None,
            download=False,
            backend_lock_path=backend_lock_path,
        ),
        backend_lock=backend_lock,
    )
    if outcome.status != "ready" or outcome.model_cache_path != model_cache_root:
        raise _fatal(
            "backend_model_invalid",
            "The checkpoint cache does not match the frozen model identity.",
        )


def _validate_runtime_environment(runtime: LoadedRuntimeLock) -> None:
    if dict(runtime.payload["environment"]) != dict(REQUIRED_ENVIRONMENT):
        raise _fatal(
            "runtime_environment_mismatch",
            "Runtime environment identity does not match the image.",
        )


def _launch_profile(config: OafBackendConfig, locks: _Locks) -> RunnerLaunchProfile:
    seal = locks.seal.payload
    runtime = locks.runtime.payload
    return RunnerLaunchProfile(
        image_config_digest=cast(
            str,
            runtime["runtime_image_config_digest"],
        ),
        backend_lock_path=config.backend_lock_path.resolve(strict=True),
        runtime_lock_path=config.runtime_lock_path.resolve(strict=True),
        seal_evidence_path=config.seal_evidence_path.resolve(strict=True),
        model_cache_path=config.model_cache_root.resolve(strict=True),
        input_root=config.input_root.resolve(strict=True),
        environment=cast(Mapping[str, str], runtime["environment"]),
        uid=_positive_integer(seal["runtime_uid"], "runtime UID"),
        gid=_positive_integer(seal["runtime_gid"], "runtime GID"),
        cpu_limit=_cpu_limit(seal["cpu_limit_millis"]),
        memory_bytes=_positive_integer(seal["memory_limit_bytes"], "memory limit"),
        pid_limit=_positive_integer(seal["pid_limit"], "PID limit"),
        tmp_bytes=_positive_integer(seal["tmp_bytes"], "tmpfs limit"),
        shm_bytes=_positive_integer(seal["shm_bytes"], "shm limit"),
        startup_deadline_seconds=_positive_integer(
            seal["startup_deadline_seconds"],
            "startup deadline",
        ),
        request_deadline_seconds=_positive_integer(
            seal["request_deadline_seconds"],
            "request deadline",
        ),
        stdout_max_line_bytes=_positive_integer(
            runtime["stdout_max_line_bytes"],
            "stdout line limit",
        ),
        stderr_read_chunk_bytes=_positive_integer(
            runtime["stderr_read_chunk_bytes"],
            "stderr chunk limit",
        ),
        stderr_max_line_bytes=_positive_integer(
            runtime["stderr_max_line_bytes"],
            "stderr line limit",
        ),
        stderr_ring_buffer_bytes=_positive_integer(
            runtime["stderr_ring_buffer_bytes"],
            "stderr ring limit",
        ),
    )


def _execution_conditions(seal: LoadedSealEvidence) -> ExecutionConditions:
    payload = seal.payload
    return ExecutionConditions(
        cpu_limit=_cpu_limit(payload["cpu_limit_millis"]),
        memory_bytes=_positive_integer(payload["memory_limit_bytes"], "memory limit"),
        pid_limit=_positive_integer(payload["pid_limit"], "PID limit"),
        tmp_bytes=_positive_integer(payload["tmp_bytes"], "tmpfs limit"),
        shm_bytes=_positive_integer(payload["shm_bytes"], "shm limit"),
        startup_deadline_seconds=_positive_integer(
            payload["startup_deadline_seconds"],
            "startup deadline",
        ),
        request_deadline_seconds=_positive_integer(
            payload["request_deadline_seconds"],
            "request deadline",
        ),
    )


def _cpu_limit(value: object) -> str:
    millis = _positive_integer(value, "CPU limit millis")
    rendered = format(Decimal(millis) / Decimal(1000), "f").rstrip("0").rstrip(".")
    return rendered or "0"


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or cast(int, value) <= 0:
        raise _fatal("backend_profile_invalid", f"The {label} is invalid.")
    return cast(int, value)


def _validate_handshake(handshake: Mapping[str, JsonValue], locks: _Locks) -> None:
    backend = locks.backend
    runtime = locks.runtime
    seal = locks.seal
    checkpoint = backend.payload["checkpoint_inventory"]
    required = backend.payload["required_inference_inventory"]
    non_inference = backend.payload["non_inference_inventory"]
    expected: dict[str, object] = {
        "backend_descriptor": dict(backend.descriptor.payload),
        "backend_descriptor_sha256": backend.descriptor.sha256,
        "backend_lock_sha256": backend.sha256,
        "checkpoint_inventory_sha256": sha256_hex(canonical_json_bytes(checkpoint)),
        "non_inference_count": 52,
        "non_inference_inventory_sha256": sha256_hex(canonical_json_bytes(non_inference)),
        "protocol_schema": _PROTOCOL_SCHEMA,
        "python_version": runtime.payload["python_version"],
        "required_inference_count": 78,
        "required_inference_inventory_sha256": sha256_hex(canonical_json_bytes(required)),
        "restored_inference_count": 78,
        "runner_source_manifest_sha256": runtime.payload["runner_source_manifest_sha256"],
        "runtime_lock_sha256": runtime.sha256,
        "smoke_audio_sha256": backend.payload["smoke_audio_sha256"],
        "smoke_oracle_sha256": backend.payload["smoke_oracle_sha256"],
        "smoke_prediction_sha256": seal.payload["smoke_prediction_sha256"],
        "smoke_status": "exact_match",
        "tensorflow_abi": runtime.payload["tensorflow_abi"],
        "tensorflow_build": runtime.payload["tensorflow_build"],
        "type": "ready",
        "upstream_source_manifest_sha256": runtime.payload["upstream_source_manifest_sha256"],
    }
    if set(handshake) != _READY_KEYS or dict(handshake) != expected:
        raise _fatal(
            "backend_handshake_mismatch",
            "The backend runner handshake did not match the frozen identity.",
        )


def _load_smoke_audio(config: OafBackendConfig, locks: _Locks) -> CanonicalAudio:
    oracle_path = config.input_root / "smoke" / "smoke-oracle.json"
    content = _require_artifact_hash(
        oracle_path,
        locks.backend.payload["smoke_oracle_sha256"],
        "smoke oracle",
    )
    try:
        if not content.endswith(b"\n"):
            raise StrictJsonError("missing newline")
        oracle = strict_json_loads(content[:-1], require_canonical=True)
        if (
            not isinstance(oracle, dict)
            or set(oracle) != _SMOKE_ORACLE_KEYS
            or oracle.get("schema") != _SMOKE_ORACLE_SCHEMA
        ):
            raise StrictJsonError("smoke oracle schema mismatch")
        input_sha256 = _require_hash_value(
            oracle["input_audio_sha256"],
            "smoke input SHA-256",
        )
        source_sha256 = _require_hash_value(
            oracle["source_audio_sha256"],
            "smoke source SHA-256",
        )
        source_audio_id = _nonempty_string(oracle["source_audio_id"], "smoke source ID")
        input_view_id = _nonempty_string(oracle["input_view_id"], "smoke input view ID")
        frame_count = _positive_integer(
            oracle["input_audio_frame_count"],
            "smoke audio frame count",
        )
        if not isinstance(oracle["native_events"], list) or not oracle["native_events"]:
            raise ValueError("smoke events are invalid")
    except (KeyError, StrictJsonError, TypeError, ValueError):
        raise _fatal(
            "smoke_oracle_invalid",
            "The smoke oracle is malformed.",
        ) from None
    smoke = load_direct_audio(
        config.input_root / "smoke" / "canonical.wav",
        source_audio_id=source_audio_id,
        input_view_id=input_view_id,
        max_input_audio_frames=locks.backend.max_input_audio_frames,
    )
    if (
        smoke.input_audio_sha256 != input_sha256
        or smoke.input_audio_sha256 != locks.backend.payload["smoke_audio_sha256"]
        or smoke.audio_frame_count != frame_count
    ):
        raise _fatal(
            "smoke_audio_mismatch",
            "The smoke audio bytes do not match the frozen identity.",
        )
    return replace(smoke, source_audio_sha256=source_sha256)


def _expected_smoke_prediction(
    config: OafBackendConfig,
    locks: _Locks,
    smoke_audio: CanonicalAudio,
) -> NativePrediction:
    """Build the only expected JSONL source from the mounted oracle and locks."""
    oracle_path = config.input_root / "smoke" / "smoke-oracle.json"
    content = _require_artifact_hash(
        oracle_path,
        locks.backend.payload["smoke_oracle_sha256"],
        "smoke oracle",
    )
    try:
        oracle = strict_json_loads(content[:-1], require_canonical=True)
        if not isinstance(oracle, dict) or set(oracle) != _SMOKE_ORACLE_KEYS:
            raise ValueError("smoke oracle schema mismatch")
        raw_events = oracle["native_events"]
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError("smoke oracle events are invalid")
        events = tuple(_native_event(event, locks.backend) for event in raw_events)
    except (BackendItemFailure, KeyError, StrictJsonError, TypeError, ValueError):
        raise _fatal("smoke_oracle_invalid", "The smoke oracle is malformed.") from None
    return NativePrediction(
        audio=smoke_audio,
        descriptor=locks.backend.descriptor,
        events=events,
        backend_lock_sha256=locks.backend.sha256,
        runtime_lock_sha256=locks.runtime.sha256,
        parameter_lock_sha256=None,
        model_artifact_set_sha256=cast(
            str,
            locks.backend.descriptor.payload["model_artifact_set_sha256"],
        ),
        upstream_source_commit=cast(
            str,
            locks.backend.descriptor.payload["upstream_source_commit"],
        ),
        training_data_map_id=cast(
            str,
            locks.backend.descriptor.payload["training_data_map_id"],
        ),
    )


def _verify_smoke_prediction_bytes(
    expected: NativePrediction,
    predictions: tuple[NativePrediction, NativePrediction, NativePrediction],
) -> bytes:
    try:
        expected_content = render_prediction_artifact(expected)
        contents = tuple(render_prediction_artifact(prediction) for prediction in predictions)
    except PredictionArtifactError:
        raise _fatal(
            "smoke_mismatch",
            "The post-ready smoke prediction did not match the frozen oracle.",
        ) from None
    if any(content != expected_content for content in contents):
        raise _fatal(
            "smoke_mismatch",
            "The post-ready smoke prediction did not match the frozen oracle.",
        )
    return expected_content


def _publish_smoke_verification_artifacts(
    backend_root: Path,
    predictions: tuple[NativePrediction, NativePrediction, NativePrediction],
) -> SmokeVerificationArtifacts:
    labels = ("persistent-first", "persistent-second", "fresh-first")
    artifacts: list[PublishedArtifact] = []
    for label, prediction in zip(labels, predictions, strict=True):
        content = render_prediction_artifact(prediction)
        digest = sha256_hex(content)
        artifacts.append(
            publish_prediction_artifact(
                backend_root / "verification" / "smoke" / label / "sha256" / f"{digest}.jsonl",
                prediction,
            )
        )
    return SmokeVerificationArtifacts(*artifacts)


def _read_authenticated_input(audio: CanonicalAudio) -> bytes:
    try:
        content = read_regular_file_no_follow(audio.path)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise _item(
            "input_path_invalid",
            "The canonical input path is unavailable or unsafe.",
        ) from None
    if len(content) != audio.byte_length or sha256_hex(content) != audio.input_audio_sha256:
        raise _item(
            "input_hash_mismatch",
            "The canonical input hash changed before the runner request.",
        )
    return content


def _verify_request_snapshot(snapshot: PrivateFileSnapshot) -> None:
    try:
        snapshot.verify()
    except (OSError, RuntimeError, TypeError, ValueError):
        raise _fatal(
            "backend_input_snapshot_changed",
            "The private runner input changed during the request.",
        ) from None


def _relative_input_path(audio_path: Path, input_root: Path) -> str:
    try:
        root = input_root.resolve(strict=True)
        resolved = audio_path.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (OSError, RuntimeError, ValueError):
        raise _item(
            "input_path_invalid",
            "The canonical input must remain beneath the mounted input root.",
        ) from None
    if not relative or relative.startswith("../"):
        raise _item(
            "input_path_invalid",
            "The canonical input must remain beneath the mounted input root.",
        )
    return relative


def _native_event(value: object, backend: LoadedBackendLock) -> NativeEvent:
    from runtime.oaf_tf1.protocol import ProtocolFailure, decode_native_event

    try:
        protocol_event = decode_native_event(value)
    except ProtocolFailure:
        raise _item("native_event_invalid", "The runner event is malformed.") from None
    group = protocol_event.upstream_8hit_group_id
    known_groups = {
        row.get("group_id")
        for row in cast(list[Mapping[str, object]], backend.payload["training_groups"])
        if isinstance(row, Mapping)
    }
    if group is not None and (not isinstance(group, str) or group not in known_groups):
        raise _item("native_event_invalid", "The runner event training group is invalid.")
    return NativeEvent(
        time_sec=protocol_event.time_sec,
        native_class_id=protocol_event.native_class_id,
        model_output_bin=protocol_event.model_output_bin,
        native_midi_note=protocol_event.native_midi_note,
        native_metadata={"upstream_8hit_group_id": cast(str | None, group)},
        confidence=protocol_event.confidence,
        velocity_midi=protocol_event.velocity_midi,
    )


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty")
    return value


def _require_hash_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise _fatal("backend_identity_invalid", f"The {label} is invalid.")
    try:
        return require_sha256(value, label)
    except StrictJsonError:
        raise _fatal("backend_identity_invalid", f"The {label} is invalid.") from None


def _inventory_hash(value: object) -> str:
    return sha256_hex(canonical_json_bytes(cast(JsonValue, value)))


def _tensor_check(
    locks: _Locks,
    artifact: PublishedArtifact,
    *,
    status: str = "passed",
) -> TensorCoverageCheck:
    required = locks.backend.payload["required_inference_inventory"]
    non_inference = locks.backend.payload["non_inference_inventory"]
    return TensorCoverageCheck(
        status=cast(str, status),  # type: ignore[arg-type]
        required_count=78,
        restored_count=78,
        non_inference_count=52,
        required_inventory_sha256=_inventory_hash(required),
        non_inference_inventory_sha256=_inventory_hash(non_inference),
        report=artifact,
    )


def _empty_tensor_check() -> TensorCoverageCheck:
    return TensorCoverageCheck(
        status="not_run",
        required_count=0,
        restored_count=0,
        non_inference_count=0,
        required_inventory_sha256=None,
        non_inference_inventory_sha256=None,
        report=None,
    )


def _empty_smoke_check() -> SmokeCheck:
    return SmokeCheck(
        status="not_run",
        audio_sha256=None,
        oracle_sha256=None,
        prediction=None,
    )


def _fatal(code: str, message: str) -> BackendFatalFailure:
    return BackendFatalFailure(BackendError(code=code, message=message))


def _item(code: str, message: str) -> BackendItemFailure:
    return BackendItemFailure(BackendError(code=code, message=message))


def create_backend(
    config: OafBackendConfig | None = None,
    *,
    process_factory: ProcessFactory = RunnerProcess.start,
    allow_emulated_diagnostics: bool = False,
) -> OafTf1Backend:
    from src.benchmark.backend_registry import BackendLockUnavailable

    if config is not None:
        if allow_emulated_diagnostics and not config.allow_emulated_diagnostics:
            config = replace(config, allow_emulated_diagnostics=True)
        return OafTf1Backend(config, process_factory=process_factory)

    repository_root = Path(__file__).resolve().parents[3]
    base = repository_root / "config" / "benchmark" / "backends"
    required_paths = (
        base / f"{OFFICIAL_BACKEND_ID}.backend-lock.json",
        base / f"{OFFICIAL_BACKEND_ID}.runtime-lock.json",
        base / f"{OFFICIAL_BACKEND_ID}.seal-evidence.json",
        repository_root
        / "docs"
        / "superpowers"
        / "evidence"
        / "hpa-320"
        / "legacy-tf2-conversion-coverage.json",
        repository_root / "runtime" / "oaf_tf1" / "host-adapter-source-manifest.json",
    )
    if any(not path.is_file() for path in required_paths):
        raise BackendLockUnavailable("frozen OaF seal outputs are unavailable")
    raise BackendLockUnavailable("accepted native host evidence is unavailable")


__all__ = [
    "OafBackendConfig",
    "OafBackendFatal",
    "OafBackendItem",
    "OafTf1Backend",
    "create_backend",
]
