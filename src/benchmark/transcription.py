from __future__ import annotations

# The orchestration is intentionally branch-oriented: each phase has a distinct
# typed failure mapping and publication boundary.
# pylint: disable=too-many-arguments,too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-return-statements,broad-exception-caught
# pylint: disable=too-many-lines,unidiomatic-typecheck
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast
from uuid import UUID, uuid4

from src.benchmark.backend_identity import require_sha256, sha256_hex, strict_json_loads
from src.benchmark.backend_publication import read_regular_file_no_follow
from src.benchmark.backend_registry import (
    HEURISTIC_BACKEND_ID,
    OFFICIAL_BACKEND_ID,
    BackendRegistry,
    BackendUnavailable,
)
from src.benchmark.backend_reports import (
    ExecutionItem,
    ExecutionReport,
    OperationalReportPublicationError,
    ReportValidationError,
    VerificationReport,
    derive_item_id,
    publish_operational_report,
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
)
from src.benchmark.input_view import load_derived_audio, load_direct_audio
from src.benchmark.prediction_artifact import (
    PredictionArtifact,
    publish_prediction_artifact,
    read_prediction_artifact,
    render_prediction_artifact,
)

_BACKEND_UNAVAILABLE = BackendError(
    code="backend_unavailable",
    message="Backend is unavailable.",
)
_BACKEND_FAILURE = BackendError(
    code="backend_failure",
    message="The backend failed unexpectedly.",
)
_INVALID_VERIFICATION = BackendError(
    code="invalid_backend_verification",
    message="Backend verification did not satisfy the contract.",
)
_BACKEND_PROTOCOL_FAILED = BackendError(
    code="backend_protocol_failed",
    message="Backend prediction did not match its verified identity.",
)
_INPUT_CONTRACT_FAILED = BackendError(
    code="input_contract_failed",
    message="Input audio did not satisfy the canonical contract.",
)
_OUTPUT_PATH_INVALID = BackendError(
    code="output_path_invalid",
    message="Output path must remain beneath the repository root.",
)
_PREDICTION_ARTIFACT_FAILED = BackendError(
    code="prediction_artifact_failed",
    message="Prediction artifact validation failed.",
)
_PREDICTION_PUBLICATION_FAILED = BackendError(
    code="prediction_publication_failed",
    message="Prediction artifact publication failed.",
)
_MIDI_DERIVATION_FAILED = BackendError(
    code="midi_derivation_failed",
    message="MIDI was not produced.",
)
_MIDI_SIDECAR_SCHEMA = "crux.prediction-midi-derivative/v1"
_MIDI_WRITER_DISTRIBUTION = {
    "name": "mido",
    "version": "1.3.3",
    "filename": "mido-1.3.3-py3-none-any.whl",
    "sha256": "01033c9b10b049e4436fca2762194ca839b09a4334091dd3c34e7f4ae674fd8a",
}


@dataclass(frozen=True)
class TranscribeOneRequest:
    backend_id: str | None
    audio_path: Path
    output_path: Path
    source_audio_id: str | None
    input_view_id: str | None
    input_view_manifest: Path | None
    midi_output_path: Path | None
    reports_root: Path


@dataclass(frozen=True)
class TranscribeOneOutcome:
    status: Literal["complete", "partial", "failed", "environment_unsupported"]
    exit_code: Literal[0, 1, 2]
    report_artifact: PublishedArtifact


def run_transcribe_one(
    request: TranscribeOneRequest,
    *,
    registry: BackendRegistry,
    now: datetime | None = None,
    run_id: UUID | None = None,
    midi_writer: Callable[[PredictionArtifact, Path], MidiDerivative] | None = None,
) -> TranscribeOneOutcome:
    try:
        repository_root = Path.cwd().resolve(strict=True)
        anchored_request = _anchor_request(request, repository_root)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise OperationalReportPublicationError("operational_report_publication_failed") from None

    effective_now = datetime.now(UTC) if now is None else now
    effective_run_id = uuid4() if run_id is None else run_id
    selected_backend_id = (
        registry.default_backend_id if request.backend_id is None else request.backend_id
    )
    try:
        backend = registry.create(request.backend_id)
    except BackendUnavailable as error:
        return _publish_outcome(
            anchored_request,
            backend_id=error.report_backend_id,
            verification=None,
            status="failed",
            exit_code=2,
            items=(),
            errors=(_BACKEND_UNAVAILABLE,),
            now=effective_now,
            run_id=effective_run_id,
            repository_root=repository_root,
        )
    except Exception:
        return _publish_outcome(
            anchored_request,
            backend_id=selected_backend_id,
            verification=None,
            status="failed",
            exit_code=2,
            items=(),
            errors=(_BACKEND_FAILURE,),
            now=effective_now,
            run_id=effective_run_id,
            repository_root=repository_root,
        )

    try:
        outcome = _run_with_backend(
            anchored_request,
            backend_id=selected_backend_id,
            backend=backend,
            now=effective_now,
            run_id=effective_run_id,
            midi_writer=midi_writer,
            repository_root=repository_root,
        )
    except BaseException:
        try:
            backend.close()
        except BaseException:
            pass
        raise
    try:
        backend.close()
    except BaseException:
        pass
    return outcome


def _run_with_backend(
    request: TranscribeOneRequest,
    *,
    backend_id: str,
    backend: object,
    now: datetime,
    run_id: UUID,
    midi_writer: Callable[[PredictionArtifact, Path], MidiDerivative] | None,
    repository_root: Path,
) -> TranscribeOneOutcome:
    try:
        verification = backend.verify()  # type: ignore[attr-defined]
    except BackendFatalFailure as failure:
        return _publish_outcome(
            request,
            backend_id=backend_id,
            verification=None,
            status="failed",
            exit_code=2,
            items=(),
            errors=(failure.error,),
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )
    except Exception:
        return _publish_outcome(
            request,
            backend_id=backend_id,
            verification=None,
            status="failed",
            exit_code=2,
            items=(),
            errors=(_BACKEND_FAILURE,),
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )

    try:
        _validate_verification(
            verification,
            backend_id=backend_id,
            repository_root=repository_root,
            now=now,
            run_id=run_id,
        )
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return _publish_outcome(
            request,
            backend_id=backend_id,
            verification=None,
            status="failed",
            exit_code=2,
            items=(),
            errors=(_INVALID_VERIFICATION,),
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )

    if verification.status == "environment_unsupported":
        return _publish_outcome(
            request,
            backend_id=backend_id,
            verification=verification,
            status="environment_unsupported",
            exit_code=1,
            items=(),
            errors=verification.errors,
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )
    if verification.status != "verified":
        return _publish_outcome(
            request,
            backend_id=backend_id,
            verification=verification,
            status="failed",
            exit_code=2,
            items=(),
            errors=verification.errors,
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )

    try:
        prediction_path = _resolve_output_path(request.output_path, repository_root)
        midi_path = (
            None
            if request.midi_output_path is None
            else _resolve_output_path(request.midi_output_path, repository_root)
        )
        if midi_path is not None and midi_path == prediction_path:
            raise ValueError("prediction and MIDI destinations must be distinct")
    except (OSError, RuntimeError, ValueError):
        return _publish_outcome(
            request,
            backend_id=backend_id,
            verification=verification,
            status="partial",
            exit_code=1,
            items=(),
            errors=(_OUTPUT_PATH_INVALID,),
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )

    try:
        input_request = replace(
            request,
            audio_path=_resolve_input_path(request.audio_path, repository_root),
            input_view_manifest=(
                None
                if request.input_view_manifest is None
                else _resolve_input_path(request.input_view_manifest, repository_root)
            ),
        )
        audio = _load_audio(input_request, verification.max_input_audio_frames)
    except (OSError, RuntimeError, TypeError, ValueError):
        return _publish_outcome(
            request,
            backend_id=backend_id,
            verification=verification,
            status="partial",
            exit_code=1,
            items=(),
            errors=(_INPUT_CONTRACT_FAILED,),
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )

    try:
        prediction = backend.transcribe(audio)  # type: ignore[attr-defined]
    except BackendItemFailure as failure:
        return _publish_item_failure(
            request,
            backend_id=backend_id,
            verification=verification,
            audio=audio,
            error=failure.error,
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )
    except BackendFatalFailure as failure:
        return _publish_outcome(
            request,
            backend_id=backend_id,
            verification=verification,
            status="failed",
            exit_code=2,
            items=(),
            errors=(failure.error,),
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )
    except Exception:
        return _publish_outcome(
            request,
            backend_id=backend_id,
            verification=verification,
            status="failed",
            exit_code=2,
            items=(),
            errors=(_BACKEND_FAILURE,),
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )

    if not _valid_prediction(prediction, audio=audio, verification=verification):
        return _publish_outcome(
            request,
            backend_id=backend_id,
            verification=verification,
            status="failed",
            exit_code=2,
            items=(),
            errors=(_BACKEND_PROTOCOL_FAILED,),
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )

    try:
        rendered = render_prediction_artifact(prediction)
        prediction_artifact = read_prediction_artifact(rendered)
    except ValueError:
        return _publish_item_failure(
            request,
            backend_id=backend_id,
            verification=verification,
            audio=audio,
            error=_PREDICTION_ARTIFACT_FAILED,
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )

    try:
        published_prediction = publish_prediction_artifact(prediction_path, prediction)
    except (OSError, ValueError):
        return _publish_item_failure(
            request,
            backend_id=backend_id,
            verification=verification,
            audio=audio,
            error=_PREDICTION_PUBLICATION_FAILED,
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )
    try:
        report_prediction = _relative_artifact(published_prediction, repository_root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return _publish_item_failure(
            request,
            backend_id=backend_id,
            verification=verification,
            audio=audio,
            error=_PREDICTION_PUBLICATION_FAILED,
            now=now,
            run_id=run_id,
            repository_root=repository_root,
        )

    if midi_path is not None:
        if midi_writer is None:
            return _publish_midi_failure(
                request,
                backend_id=backend_id,
                verification=verification,
                audio=audio,
                prediction=report_prediction,
                now=now,
                run_id=run_id,
                repository_root=repository_root,
            )
        try:
            derivative = midi_writer(prediction_artifact, midi_path)
            report_midi = _validate_midi_derivative(
                derivative,
                requested_midi_path=midi_path,
                prediction_path=prediction_path,
                prediction=prediction_artifact,
                repository_root=repository_root,
            )
        except Exception:
            return _publish_midi_failure(
                request,
                backend_id=backend_id,
                verification=verification,
                audio=audio,
                prediction=report_prediction,
                now=now,
                run_id=run_id,
                repository_root=repository_root,
            )
    else:
        report_midi = None

    item = _execution_item(
        audio,
        status="complete",
        prediction=report_prediction,
        midi=report_midi,
        errors=(),
    )
    return _publish_outcome(
        request,
        backend_id=backend_id,
        verification=verification,
        status="complete",
        exit_code=0,
        items=(item,),
        errors=(),
        now=now,
        run_id=run_id,
        repository_root=repository_root,
    )


def _anchor_request(
    request: TranscribeOneRequest,
    repository_root: Path,
) -> TranscribeOneRequest:
    def anchor(path: Path) -> Path:
        if not isinstance(path, Path):
            raise TypeError("request paths must be Path values")
        return path if path.is_absolute() else repository_root / path

    reports_root = _resolve_output_path(anchor(request.reports_root), repository_root)
    return replace(
        request,
        audio_path=anchor(request.audio_path),
        output_path=anchor(request.output_path),
        input_view_manifest=(
            None if request.input_view_manifest is None else anchor(request.input_view_manifest)
        ),
        midi_output_path=(
            None if request.midi_output_path is None else anchor(request.midi_output_path)
        ),
        reports_root=reports_root,
    )


def _validate_verification(
    verification: object,
    *,
    backend_id: str,
    repository_root: Path,
    now: datetime,
    run_id: UUID,
) -> None:
    if not isinstance(verification, BackendVerification):
        raise TypeError("verify must return BackendVerification")
    if verification.status not in {"verified", "failed", "environment_unsupported"}:
        raise ValueError("verification status is invalid")
    if not isinstance(verification.errors, tuple) or any(
        not isinstance(error, BackendError) for error in verification.errors
    ):
        raise TypeError("verification errors must be typed")
    if (verification.status == "verified") != (not verification.errors):
        raise ValueError("verification status and errors disagree")

    descriptor = verification.descriptor
    if descriptor is not None and descriptor.payload.get("backend_id") != backend_id:
        raise ValueError("verification descriptor does not match selected backend")
    if verification.status == "verified" and descriptor is None:
        raise ValueError("verified backend requires descriptor")
    if descriptor is None and any(
        value is not None
        for value in (
            verification.backend_lock_sha256,
            verification.runtime_lock_sha256,
            verification.parameter_lock_sha256,
            verification.seal_evidence_sha256,
            verification.execution_attestation,
        )
    ):
        raise ValueError("identity fields require descriptor")

    if backend_id == HEURISTIC_BACKEND_ID:
        if verification.max_input_audio_frames is not None:
            raise ValueError("heuristic frame bound must be null")
        if any(
            value is not None
            for value in (
                verification.backend_lock_sha256,
                verification.runtime_lock_sha256,
                verification.seal_evidence_sha256,
            )
        ):
            raise ValueError("heuristic lock fields must be null")
    elif backend_id == OFFICIAL_BACKEND_ID:
        bound = verification.max_input_audio_frames
        if verification.status == "verified" and (type(bound) is not int or bound <= 0):
            raise ValueError("official frame bound must be positive")
        if bound is not None and (type(bound) is not int or bound <= 0):
            raise ValueError("official frame bound is invalid")
        if verification.parameter_lock_sha256 is not None:
            raise ValueError("official parameter lock must be null")

    tensor = verification.tensor_coverage
    smoke = verification.smoke
    if verification.status == "verified" and backend_id == OFFICIAL_BACKEND_ID:
        if (
            tensor.required_count <= 0
            or tensor.restored_count != tensor.required_count
            or tensor.non_inference_count < 0
            or tensor.non_inference_count > tensor.required_count
        ):
            raise ValueError("official tensor counts are inconsistent")

    tensor_report = _optional_relative_artifact(
        tensor.report,
        repository_root,
        expected_role="tensor_coverage",
    )
    smoke_prediction = _optional_relative_artifact(
        smoke.prediction,
        repository_root,
        expected_role="prediction",
    )
    attestation = _optional_relative_artifact(
        verification.execution_attestation,
        repository_root,
        expected_role="execution_attestation",
    )
    artifacts = tuple(
        artifact
        for artifact in (attestation, tensor_report, smoke_prediction)
        if artifact is not None
    )
    VerificationReport(
        {
            "schema": "crux.backend-verification-report/v1",
            "report_type": "verification",
            "run_id": str(run_id),
            "started_at": _report_timestamp(now),
            "finished_at": _report_timestamp(now),
            "status": verification.status,
            "exit_code": {
                "verified": 0,
                "failed": 2,
                "environment_unsupported": 1,
            }[verification.status],
            "descriptor": None if descriptor is None else descriptor.payload,
            "descriptor_sha256": None if descriptor is None else descriptor.sha256,
            "backend_lock_sha256": verification.backend_lock_sha256,
            "runtime_lock_sha256": verification.runtime_lock_sha256,
            "parameter_lock_sha256": verification.parameter_lock_sha256,
            "seal_evidence_sha256": verification.seal_evidence_sha256,
            "execution_attestation": attestation,
            "tensor_coverage": {
                "status": tensor.status,
                "required_count": tensor.required_count,
                "restored_count": tensor.restored_count,
                "non_inference_count": tensor.non_inference_count,
                "required_inventory_sha256": tensor.required_inventory_sha256,
                "non_inference_inventory_sha256": tensor.non_inference_inventory_sha256,
                "report_path": None if tensor_report is None else tensor_report.path.as_posix(),
                "report_sha256": None if tensor_report is None else tensor_report.sha256,
            },
            "smoke": {
                "status": smoke.status,
                "audio_sha256": smoke.audio_sha256,
                "oracle_sha256": smoke.oracle_sha256,
                "prediction_path": (
                    None if smoke_prediction is None else smoke_prediction.path.as_posix()
                ),
                "prediction_sha256": (
                    None if smoke_prediction is None else smoke_prediction.sha256
                ),
            },
            "artifacts": artifacts,
            "errors": verification.errors,
        }
    )


def _optional_relative_artifact(
    artifact: PublishedArtifact | None,
    repository_root: Path,
    *,
    expected_role: str,
) -> PublishedArtifact | None:
    if artifact is None:
        return None
    if not isinstance(artifact, PublishedArtifact):
        raise TypeError("artifact must be PublishedArtifact")
    if artifact.role != expected_role:
        raise ValueError("artifact role is invalid")
    return _relative_artifact(artifact, repository_root)


def _valid_prediction(
    prediction: object,
    *,
    audio: CanonicalAudio,
    verification: BackendVerification,
) -> bool:
    if not isinstance(prediction, NativePrediction):
        return False
    descriptor = verification.descriptor
    if descriptor is None:
        return False
    payload = descriptor.payload
    return (
        prediction.audio == audio
        and prediction.descriptor == descriptor
        and _valid_prediction_events(prediction.events)
        and prediction.backend_lock_sha256 == verification.backend_lock_sha256
        and prediction.runtime_lock_sha256 == verification.runtime_lock_sha256
        and prediction.parameter_lock_sha256 == verification.parameter_lock_sha256
        and prediction.model_artifact_set_sha256 == payload.get("model_artifact_set_sha256")
        and prediction.upstream_source_commit == payload.get("upstream_source_commit")
        and prediction.training_data_map_id == payload.get("training_data_map_id")
    )


def _valid_prediction_events(events: object) -> bool:
    if type(events) is not tuple:
        return False
    try:
        return all(
            isinstance(event, NativeEvent) and isinstance(event.native_metadata, Mapping)
            for event in events
        )
    except (AttributeError, TypeError):
        return False


def _resolve_output_path(path: Path, repository_root: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("artifact path must be a Path")
    candidate = path if path.is_absolute() else repository_root / path
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(repository_root) or resolved == repository_root:
        raise ValueError("output path escapes repository root")
    return resolved


def _resolve_input_path(path: Path, repository_root: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("input path must be a Path")
    candidate = path if path.is_absolute() else repository_root / path
    return candidate.resolve(strict=True)


def _validate_midi_derivative(
    derivative: object,
    *,
    requested_midi_path: Path,
    prediction_path: Path,
    prediction: PredictionArtifact,
    repository_root: Path,
) -> PublishedArtifact:
    if not isinstance(derivative, MidiDerivative):
        raise TypeError("MIDI writer must return MidiDerivative")
    if not isinstance(derivative.midi, PublishedArtifact) or not isinstance(
        derivative.sidecar,
        PublishedArtifact,
    ):
        raise TypeError("MIDI derivative artifacts are invalid")
    if derivative.midi.role != "midi" or derivative.sidecar.role != "prediction_midi_sidecar":
        raise ValueError("MIDI derivative roles are invalid")

    midi_path = _resolve_output_path(derivative.midi.path, repository_root)
    sidecar_path = _resolve_output_path(derivative.sidecar.path, repository_root)
    if midi_path != requested_midi_path:
        raise ValueError("MIDI artifact does not match requested destination")
    if sidecar_path.parent != midi_path.parent:
        raise ValueError("MIDI sidecar must be adjacent")
    if len({prediction_path, midi_path, sidecar_path}) != 3:
        raise ValueError("prediction, MIDI, and sidecar paths must be distinct")

    midi_content = read_regular_file_no_follow(midi_path)
    sidecar_content = read_regular_file_no_follow(sidecar_path)
    require_sha256(derivative.midi.sha256, "MIDI artifact sha256")
    require_sha256(derivative.sidecar.sha256, "MIDI sidecar sha256")
    midi_sha256 = sha256_hex(midi_content)
    if midi_sha256 != derivative.midi.sha256:
        raise ValueError("MIDI artifact hash mismatch")
    if sha256_hex(sidecar_content) != derivative.sidecar.sha256:
        raise ValueError("MIDI sidecar hash mismatch")
    if not sidecar_content.endswith(b"\n"):
        raise ValueError("MIDI sidecar must end with newline")
    sidecar = strict_json_loads(sidecar_content[:-1], require_canonical=True)
    _validate_midi_sidecar(
        sidecar,
        prediction_sha256=prediction.artifact_sha256,
        midi_sha256=midi_sha256,
    )
    return _relative_artifact(derivative.midi, repository_root)


def _validate_midi_sidecar(
    sidecar: object,
    *,
    prediction_sha256: str,
    midi_sha256: str,
) -> None:
    expected_keys = {
        "schema",
        "prediction_sha256",
        "midi_sha256",
        "writer_lock_sha256",
        "writer_distribution",
        "ticks_per_quarter",
        "tempo_qpm",
        "channel",
        "program",
        "note_duration_seconds",
    }
    if not isinstance(sidecar, Mapping) or set(sidecar) != expected_keys:
        raise ValueError("MIDI sidecar keys are invalid")
    if sidecar["schema"] != _MIDI_SIDECAR_SCHEMA:
        raise ValueError("MIDI sidecar schema is invalid")
    if sidecar["prediction_sha256"] != prediction_sha256:
        raise ValueError("MIDI sidecar prediction hash mismatch")
    if sidecar["midi_sha256"] != midi_sha256:
        raise ValueError("MIDI sidecar MIDI hash mismatch")
    require_sha256(cast(str, sidecar["writer_lock_sha256"]), "writer_lock_sha256")

    distribution = sidecar["writer_distribution"]
    if not isinstance(distribution, Mapping) or dict(distribution) != _MIDI_WRITER_DISTRIBUTION:
        raise ValueError("MIDI writer distribution is invalid")

    ticks = sidecar["ticks_per_quarter"]
    channel = sidecar["channel"]
    program = sidecar["program"]
    tempo = sidecar["tempo_qpm"]
    duration = sidecar["note_duration_seconds"]
    if type(ticks) is not int or ticks != 220:
        raise ValueError("ticks_per_quarter is invalid")
    if type(channel) is not int or channel != 9:
        raise ValueError("channel is invalid")
    if type(program) is not int or program != 0:
        raise ValueError("program is invalid")
    if not _is_exact_json_decimal(tempo, Decimal("120.0")):
        raise ValueError("tempo_qpm is invalid")
    if not _is_exact_json_decimal(duration, Decimal("0.05")):
        raise ValueError("note_duration_seconds is invalid")


def _is_exact_json_decimal(value: object, expected: Decimal) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, Decimal))
        and Decimal(value) == expected
    )


def _load_audio(
    request: TranscribeOneRequest,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    direct_mode = (
        request.source_audio_id is not None
        and request.input_view_id is not None
        and request.input_view_manifest is None
    )
    derived_mode = (
        request.source_audio_id is None
        and request.input_view_id is None
        and request.input_view_manifest is not None
    )
    if direct_mode == derived_mode:
        raise ValueError("exactly one input mode is required")
    if direct_mode:
        return load_direct_audio(
            request.audio_path,
            source_audio_id=cast(str, request.source_audio_id),
            input_view_id=cast(str, request.input_view_id),
            max_input_audio_frames=max_input_audio_frames,
        )
    return load_derived_audio(
        request.audio_path,
        cast(Path, request.input_view_manifest),
        max_input_audio_frames=max_input_audio_frames,
    )


def _publish_item_failure(
    request: TranscribeOneRequest,
    *,
    backend_id: str,
    verification: BackendVerification,
    audio: CanonicalAudio,
    error: BackendError,
    now: datetime,
    run_id: UUID,
    repository_root: Path,
) -> TranscribeOneOutcome:
    item = _execution_item(
        audio,
        status="failed",
        prediction=None,
        midi=None,
        errors=(error,),
    )
    return _publish_outcome(
        request,
        backend_id=backend_id,
        verification=verification,
        status="partial",
        exit_code=1,
        items=(item,),
        errors=(),
        now=now,
        run_id=run_id,
        repository_root=repository_root,
    )


def _publish_midi_failure(
    request: TranscribeOneRequest,
    *,
    backend_id: str,
    verification: BackendVerification,
    audio: CanonicalAudio,
    prediction: PublishedArtifact,
    now: datetime,
    run_id: UUID,
    repository_root: Path,
) -> TranscribeOneOutcome:
    item = _execution_item(
        audio,
        status="incomplete",
        prediction=prediction,
        midi=None,
        errors=(_MIDI_DERIVATION_FAILED,),
    )
    return _publish_outcome(
        request,
        backend_id=backend_id,
        verification=verification,
        status="partial",
        exit_code=1,
        items=(item,),
        errors=(),
        now=now,
        run_id=run_id,
        repository_root=repository_root,
    )


def _execution_item(
    audio: CanonicalAudio,
    *,
    status: Literal["complete", "incomplete", "failed"],
    prediction: PublishedArtifact | None,
    midi: PublishedArtifact | None,
    errors: tuple[BackendError, ...],
) -> ExecutionItem:
    return ExecutionItem(
        item_id=derive_item_id(
            source_audio_id=audio.source_audio_id,
            source_audio_sha256=audio.source_audio_sha256,
            input_view_id=audio.input_view_id,
            input_audio_sha256=audio.input_audio_sha256,
        ),
        source_audio_id=audio.source_audio_id,
        source_audio_sha256=audio.source_audio_sha256,
        input_view_id=audio.input_view_id,
        input_audio_sha256=audio.input_audio_sha256,
        status=status,
        prediction=prediction,
        midi=midi,
        errors=errors,
    )


def _relative_artifact(
    artifact: PublishedArtifact,
    repository_root: Path,
) -> PublishedArtifact:
    resolved = _resolve_output_path(artifact.path, repository_root)
    return PublishedArtifact(
        role=artifact.role,
        path=Path(resolved.relative_to(repository_root).as_posix()),
        sha256=artifact.sha256,
    )


def _publish_outcome(
    request: TranscribeOneRequest,
    *,
    backend_id: str,
    verification: BackendVerification | None,
    status: Literal["complete", "partial", "failed", "environment_unsupported"],
    exit_code: Literal[0, 1, 2],
    items: tuple[ExecutionItem, ...],
    errors: tuple[BackendError, ...],
    now: datetime,
    run_id: UUID,
    repository_root: Path,
) -> TranscribeOneOutcome:
    try:
        descriptor = None if verification is None else verification.descriptor
        execution_attestation = None if verification is None else verification.execution_attestation
        if execution_attestation is not None:
            execution_attestation = _relative_artifact(
                execution_attestation,
                repository_root,
            )
        report = ExecutionReport(
            {
                "schema": "crux.backend-execution-report/v1",
                "report_type": "execution",
                "run_id": str(run_id),
                "started_at": _report_timestamp(now),
                "finished_at": _report_timestamp(now),
                "status": status,
                "exit_code": exit_code,
                "descriptor": None if descriptor is None else descriptor.payload,
                "descriptor_sha256": None if descriptor is None else descriptor.sha256,
                "backend_lock_sha256": (
                    None if verification is None else verification.backend_lock_sha256
                ),
                "runtime_lock_sha256": (
                    None if verification is None else verification.runtime_lock_sha256
                ),
                "parameter_lock_sha256": (
                    None if verification is None else verification.parameter_lock_sha256
                ),
                "seal_evidence_sha256": (
                    None if verification is None else verification.seal_evidence_sha256
                ),
                "execution_attestation": execution_attestation,
                "verification_report": None,
                "items": items,
                "errors": errors,
            }
        )
        artifact = publish_operational_report(
            request.reports_root,
            backend_id=backend_id,
            report=report,
            now=now,
            run_id=run_id,
        )
    except OperationalReportPublicationError:
        raise
    except (OSError, ReportValidationError, RuntimeError, TypeError, ValueError):
        raise OperationalReportPublicationError("operational_report_publication_failed") from None
    return TranscribeOneOutcome(
        status=status,
        exit_code=exit_code,
        report_artifact=artifact,
    )


def _report_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
