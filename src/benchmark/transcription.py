from __future__ import annotations

# The orchestration is intentionally branch-oriented: each phase has a distinct
# typed failure mapping and publication boundary.
# pylint: disable=too-many-arguments,too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-return-statements,broad-exception-caught
# pylint: disable=too-many-lines,unidiomatic-typecheck
import os
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast
from uuid import UUID, uuid4

from src.benchmark.backend_attestation import (
    validate_changed_file_manifest,
    validate_execution_attestation,
)
from src.benchmark.backend_identity import require_sha256, sha256_hex, strict_json_loads
from src.benchmark.backend_publication import (
    DirectoryAnchor,
    PrivateSnapshotIntegrityError,
    RegularFileAnchor,
    open_directory_anchor,
    open_private_file_snapshot,
    open_regular_file_anchor,
    read_regular_file_no_follow,
    resolve_private_snapshot_root,
)
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
from src.benchmark.input_view import (
    InputViewManifest,
    input_view_artifact_paths,
    load_derived_audio_bytes,
    load_direct_audio_bytes,
    parse_input_view_manifest,
)
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


@dataclass(frozen=True)
class _CapturedInputAnchors:
    input_file: RegularFileAnchor
    manifest: InputViewManifest | None
    manifest_file: RegularFileAnchor | None
    source_file: RegularFileAnchor | None


@dataclass(frozen=True)
class _LoadedInput:
    audio: CanonicalAudio
    input_content: bytes


class _InputSnapshotChanged(OSError):
    pass


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
    except (OSError, RuntimeError, TypeError, ValueError):
        raise OperationalReportPublicationError("operational_report_publication_failed") from None
    try:
        with open_directory_anchor(repository_root) as repository_anchor:
            return _run_transcribe_one_anchored(
                request,
                registry=registry,
                now=now,
                run_id=run_id,
                midi_writer=midi_writer,
                repository_root=repository_root,
                repository_anchor=repository_anchor,
            )
    except OperationalReportPublicationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise OperationalReportPublicationError("operational_report_publication_failed") from None


def _run_transcribe_one_anchored(
    request: TranscribeOneRequest,
    *,
    registry: BackendRegistry,
    now: datetime | None,
    run_id: UUID | None,
    midi_writer: Callable[[PredictionArtifact, Path], MidiDerivative] | None,
    repository_root: Path,
    repository_anchor: DirectoryAnchor,
) -> TranscribeOneOutcome:
    anchored_request = _anchor_request(request, repository_root)
    with ExitStack() as input_stack:
        try:
            captured_input = _capture_input(
                anchored_request,
                repository_anchor,
                input_stack,
            )
            input_capture_failed = False
        except (OSError, RuntimeError, TypeError, ValueError):
            captured_input = None
            input_capture_failed = True
        return _run_transcribe_one_prepared(
            anchored_request,
            registry=registry,
            now=now,
            run_id=run_id,
            midi_writer=midi_writer,
            repository_root=repository_root,
            repository_anchor=repository_anchor,
            captured_input=captured_input,
            input_capture_failed=input_capture_failed,
        )


def _run_transcribe_one_prepared(
    request: TranscribeOneRequest,
    *,
    registry: BackendRegistry,
    now: datetime | None,
    run_id: UUID | None,
    midi_writer: Callable[[PredictionArtifact, Path], MidiDerivative] | None,
    repository_root: Path,
    repository_anchor: DirectoryAnchor,
    captured_input: _CapturedInputAnchors | None,
    input_capture_failed: bool,
) -> TranscribeOneOutcome:
    effective_now = datetime.now(UTC) if now is None else now
    effective_run_id = uuid4() if run_id is None else run_id
    selected_backend_id = (
        registry.default_backend_id if request.backend_id is None else request.backend_id
    )
    try:
        backend = registry.create(request.backend_id)
    except BackendUnavailable as error:
        return _publish_outcome(
            request,
            backend_id=error.report_backend_id,
            verification=None,
            status="failed",
            exit_code=2,
            items=(),
            errors=(_BACKEND_UNAVAILABLE,),
            now=effective_now,
            run_id=effective_run_id,
            repository_root=repository_root,
            repository_anchor=repository_anchor,
        )
    except Exception:
        return _publish_outcome(
            request,
            backend_id=selected_backend_id,
            verification=None,
            status="failed",
            exit_code=2,
            items=(),
            errors=(_BACKEND_FAILURE,),
            now=effective_now,
            run_id=effective_run_id,
            repository_root=repository_root,
            repository_anchor=repository_anchor,
        )

    try:
        outcome = _run_with_backend(
            request,
            backend_id=selected_backend_id,
            backend=backend,
            now=effective_now,
            run_id=effective_run_id,
            midi_writer=midi_writer,
            repository_root=repository_root,
            repository_anchor=repository_anchor,
            captured_input=captured_input,
            input_capture_failed=input_capture_failed,
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
    repository_anchor: DirectoryAnchor,
    captured_input: _CapturedInputAnchors | None,
    input_capture_failed: bool,
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
            repository_anchor=repository_anchor,
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
            repository_anchor=repository_anchor,
        )

    try:
        _validate_verification(
            verification,
            backend_id=backend_id,
            repository_root=repository_root,
            repository_anchor=repository_anchor,
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
            repository_anchor=repository_anchor,
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
            repository_anchor=repository_anchor,
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
            repository_anchor=repository_anchor,
        )

    try:
        prediction_path = _resolve_output_path(
            request.output_path,
            repository_root,
            repository_anchor,
        )
        midi_path = (
            None
            if request.midi_output_path is None
            else _resolve_output_path(
                request.midi_output_path,
                repository_root,
                repository_anchor,
            )
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
            repository_anchor=repository_anchor,
        )

    try:
        loaded_input = _load_request_audio(
            request,
            max_input_audio_frames=verification.max_input_audio_frames,
            captured_input=captured_input,
            input_capture_failed=input_capture_failed,
        )
        audio = loaded_input.audio
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
            repository_anchor=repository_anchor,
        )

    try:
        prediction, audio = _transcribe_private_snapshot(
            backend,
            loaded_input,
        )
    except _InputSnapshotChanged:
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
            repository_anchor=repository_anchor,
        )
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
            repository_anchor=repository_anchor,
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
            repository_anchor=repository_anchor,
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
            repository_anchor=repository_anchor,
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
            repository_anchor=repository_anchor,
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
            repository_anchor=repository_anchor,
        )

    try:
        published_prediction = publish_prediction_artifact(
            prediction_path,
            prediction,
            anchor=repository_anchor,
        )
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
            repository_anchor=repository_anchor,
        )
    try:
        report_prediction = _relative_artifact(
            published_prediction,
            repository_root,
            repository_anchor,
        )
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
            repository_anchor=repository_anchor,
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
                repository_anchor=repository_anchor,
            )
        try:
            derivative = midi_writer(prediction_artifact, midi_path)
            report_midi = _validate_midi_derivative(
                derivative,
                requested_midi_path=midi_path,
                prediction_path=prediction_path,
                prediction=prediction_artifact,
                repository_root=repository_root,
                repository_anchor=repository_anchor,
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
                repository_anchor=repository_anchor,
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
        repository_anchor=repository_anchor,
    )


def _anchor_request(
    request: TranscribeOneRequest,
    repository_root: Path,
) -> TranscribeOneRequest:
    def anchor(path: Path) -> Path:
        if not isinstance(path, Path):
            raise TypeError("request paths must be Path values")
        return path if path.is_absolute() else repository_root / path

    def normalize_input(path: Path) -> Path:
        candidate = anchor(path)
        return Path(os.path.abspath(candidate))

    reports_root = _resolve_output_path(anchor(request.reports_root), repository_root)
    return replace(
        request,
        audio_path=normalize_input(request.audio_path),
        output_path=anchor(request.output_path),
        input_view_manifest=(
            None
            if request.input_view_manifest is None
            else normalize_input(request.input_view_manifest)
        ),
        midi_output_path=(
            None if request.midi_output_path is None else anchor(request.midi_output_path)
        ),
        reports_root=reports_root,
    )


def _capture_input(
    request: TranscribeOneRequest,
    repository_anchor: DirectoryAnchor,
    stack: ExitStack,
) -> _CapturedInputAnchors:
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

    audio_is_repository_owned = _path_is_repository_owned(
        request.audio_path,
        repository_anchor,
    )
    if direct_mode:
        input_file = stack.enter_context(
            open_regular_file_anchor(
                request.audio_path,
                anchor=repository_anchor if audio_is_repository_owned else None,
            )
        )
        return _CapturedInputAnchors(
            input_file=input_file,
            manifest=None,
            manifest_file=None,
            source_file=None,
        )

    manifest_path = cast(Path, request.input_view_manifest)
    manifest_is_repository_owned = _path_is_repository_owned(
        manifest_path,
        repository_anchor,
    )
    if manifest_is_repository_owned != audio_is_repository_owned:
        raise ValueError("manifest and input ownership must match")

    if manifest_is_repository_owned:
        manifest_file = stack.enter_context(
            open_regular_file_anchor(
                manifest_path,
                anchor=repository_anchor,
            )
        )
        artifact_anchor = repository_anchor
    else:
        manifest_directory = stack.enter_context(open_directory_anchor(manifest_path.parent))
        manifest_file = stack.enter_context(
            open_regular_file_anchor(
                manifest_path,
                anchor=manifest_directory,
            )
        )
        artifact_anchor = manifest_directory

    manifest = parse_input_view_manifest(manifest_file.content)
    source_path, input_path = input_view_artifact_paths(manifest_path, manifest)
    if request.audio_path != input_path:
        raise ValueError("audio_path does not match manifest input_audio_path")
    source_file = stack.enter_context(
        open_regular_file_anchor(
            source_path,
            anchor=artifact_anchor,
        )
    )
    input_file = stack.enter_context(
        open_regular_file_anchor(
            input_path,
            anchor=artifact_anchor,
        )
    )
    return _CapturedInputAnchors(
        input_file=input_file,
        manifest=manifest,
        manifest_file=manifest_file,
        source_file=source_file,
    )


def _path_is_repository_owned(
    path: Path,
    repository_anchor: DirectoryAnchor,
) -> bool:
    try:
        repository_anchor.relative_path(path)
    except OSError:
        return False
    return True


def _validate_verification(
    verification: object,
    *,
    backend_id: str,
    repository_root: Path,
    repository_anchor: DirectoryAnchor,
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

    if verification.status == "verified":
        _validate_supporting_artifact(
            verification.execution_attestation,
            repository_root,
            expected_role="execution_attestation",
            descriptor_sha256=descriptor.sha256,
            backend_id=backend_id,
            repository_anchor=repository_anchor,
        )
        _validate_supporting_artifact(
            tensor.report,
            repository_root,
            expected_role="tensor_coverage",
            repository_anchor=repository_anchor,
        )
        _validate_supporting_artifact(
            smoke.prediction,
            repository_root,
            expected_role="prediction",
            repository_anchor=repository_anchor,
        )

    tensor_report = _optional_relative_artifact(
        tensor.report,
        repository_root,
        expected_role="tensor_coverage",
        repository_anchor=repository_anchor,
    )
    smoke_prediction = _optional_relative_artifact(
        smoke.prediction,
        repository_root,
        expected_role="prediction",
        repository_anchor=repository_anchor,
    )
    attestation = _optional_relative_artifact(
        verification.execution_attestation,
        repository_root,
        expected_role="execution_attestation",
        repository_anchor=repository_anchor,
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
    repository_anchor: DirectoryAnchor,
) -> PublishedArtifact | None:
    if artifact is None:
        return None
    if not isinstance(artifact, PublishedArtifact):
        raise TypeError("artifact must be PublishedArtifact")
    if artifact.role != expected_role:
        raise ValueError("artifact role is invalid")
    return _relative_artifact(artifact, repository_root, repository_anchor)


def _validate_supporting_artifact(
    artifact: PublishedArtifact | None,
    repository_root: Path,
    *,
    expected_role: str,
    descriptor_sha256: str | None = None,
    backend_id: str | None = None,
    repository_anchor: DirectoryAnchor,
) -> None:
    if artifact is None:
        return
    if not isinstance(artifact, PublishedArtifact) or artifact.role != expected_role:
        raise ValueError("supporting artifact role is invalid")
    require_sha256(artifact.sha256, "supporting artifact sha256")
    resolved = _resolve_output_path(
        artifact.path,
        repository_root,
        repository_anchor,
    )
    content = read_regular_file_no_follow(resolved, anchor=repository_anchor)
    if sha256_hex(content) != artifact.sha256:
        raise ValueError("supporting artifact hash mismatch")
    if expected_role == "execution_attestation":
        if descriptor_sha256 is None or backend_id is None:
            raise ValueError("attestation identity is unavailable")
        attestation = validate_execution_attestation(
            content,
            expected_backend_id=backend_id,
            expected_descriptor_sha256=descriptor_sha256,
        )
        changed_manifest = attestation.changed_files_manifest
        if changed_manifest is not None:
            changed_path = _resolve_output_path(
                changed_manifest.path,
                repository_root,
                repository_anchor,
            )
            changed_content = read_regular_file_no_follow(
                changed_path,
                anchor=repository_anchor,
            )
            if sha256_hex(changed_content) != changed_manifest.sha256:
                raise ValueError("changed-file manifest hash mismatch")
            validate_changed_file_manifest(changed_content)
    elif expected_role == "tensor_coverage":
        _validate_schema_bearing_json_artifact(content)
    elif expected_role == "prediction":
        read_prediction_artifact(content)


def _validate_schema_bearing_json_artifact(content: bytes) -> None:
    if not content.endswith(b"\n"):
        raise ValueError("supporting JSON artifact must end with newline")
    payload = strict_json_loads(content[:-1], require_canonical=True)
    if not isinstance(payload, dict):
        raise ValueError("supporting JSON artifact must be an object")
    schema = payload.get("schema")
    if not isinstance(schema, str) or not schema:
        raise ValueError("supporting JSON artifact schema is required")


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


def _resolve_output_path(
    path: Path,
    repository_root: Path,
    repository_anchor: DirectoryAnchor | None = None,
) -> Path:
    if not isinstance(path, Path):
        raise TypeError("artifact path must be a Path")
    candidate = path if path.is_absolute() else repository_root / path
    if repository_anchor is not None:
        repository_anchor.relative_path(candidate)
        return candidate
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(repository_root) or resolved == repository_root:
        raise ValueError("output path escapes repository root")
    return resolved


def _validate_midi_derivative(
    derivative: object,
    *,
    requested_midi_path: Path,
    prediction_path: Path,
    prediction: PredictionArtifact,
    repository_root: Path,
    repository_anchor: DirectoryAnchor,
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

    midi_path = _resolve_output_path(
        derivative.midi.path,
        repository_root,
        repository_anchor,
    )
    sidecar_path = _resolve_output_path(
        derivative.sidecar.path,
        repository_root,
        repository_anchor,
    )
    if midi_path != requested_midi_path:
        raise ValueError("MIDI artifact does not match requested destination")
    if sidecar_path.parent != midi_path.parent:
        raise ValueError("MIDI sidecar must be adjacent")
    if len({prediction_path, midi_path, sidecar_path}) != 3:
        raise ValueError("prediction, MIDI, and sidecar paths must be distinct")

    midi_content = read_regular_file_no_follow(midi_path, anchor=repository_anchor)
    sidecar_content = read_regular_file_no_follow(sidecar_path, anchor=repository_anchor)
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
    return _relative_artifact(derivative.midi, repository_root, repository_anchor)


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
    captured_input: _CapturedInputAnchors,
) -> _LoadedInput:
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
        if (
            captured_input.manifest is not None
            or captured_input.manifest_file is not None
            or captured_input.source_file is not None
            or captured_input.input_file.path != request.audio_path
        ):
            raise ValueError("captured direct input anchors are invalid")
        captured_input.input_file.verify()
        audio = load_direct_audio_bytes(
            request.audio_path,
            captured_input.input_file.content,
            source_audio_id=cast(str, request.source_audio_id),
            input_view_id=cast(str, request.input_view_id),
            max_input_audio_frames=max_input_audio_frames,
        )
    else:
        manifest_path = cast(Path, request.input_view_manifest)
        if (
            captured_input.manifest is None
            or captured_input.manifest_file is None
            or captured_input.source_file is None
            or captured_input.manifest_file.path != manifest_path
            or captured_input.input_file.path != request.audio_path
        ):
            raise ValueError("captured derived input anchors are invalid")
        captured_input.manifest_file.verify()
        captured_input.source_file.verify()
        captured_input.input_file.verify()
        audio = load_derived_audio_bytes(
            request.audio_path,
            captured_input.manifest,
            source_content=captured_input.source_file.content,
            input_content=captured_input.input_file.content,
            max_input_audio_frames=max_input_audio_frames,
        )
    return _LoadedInput(
        audio=audio,
        input_content=captured_input.input_file.content,
    )


def _load_request_audio(
    request: TranscribeOneRequest,
    *,
    max_input_audio_frames: int | None,
    captured_input: _CapturedInputAnchors | None,
    input_capture_failed: bool,
) -> _LoadedInput:
    if input_capture_failed or captured_input is None:
        raise ValueError("input capture failed")
    return _load_audio(
        request,
        max_input_audio_frames,
        captured_input,
    )


def _transcribe_private_snapshot(
    backend: object,
    loaded_input: _LoadedInput,
) -> tuple[object, CanonicalAudio]:
    snapshot_root = resolve_private_snapshot_root()
    try:
        with open_private_file_snapshot(
            loaded_input.input_content,
            loaded_input.audio.input_audio_sha256,
            root=snapshot_root,
        ) as snapshot:
            snapshot.verify()
            backend_audio = replace(loaded_input.audio, path=snapshot.path)
            try:
                prediction = backend.transcribe(backend_audio)  # type: ignore[attr-defined]
            except BaseException:
                try:
                    snapshot.verify()
                except (OSError, RuntimeError, TypeError, ValueError):
                    raise _InputSnapshotChanged("backend input snapshot changed") from None
                raise
            try:
                snapshot.verify()
            except (OSError, RuntimeError, TypeError, ValueError):
                raise _InputSnapshotChanged("backend input snapshot changed") from None
    except PrivateSnapshotIntegrityError:
        raise _InputSnapshotChanged("backend input snapshot changed") from None
    return prediction, backend_audio


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
    repository_anchor: DirectoryAnchor,
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
        repository_anchor=repository_anchor,
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
    repository_anchor: DirectoryAnchor,
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
        repository_anchor=repository_anchor,
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
    repository_anchor: DirectoryAnchor | None = None,
) -> PublishedArtifact:
    resolved = _resolve_output_path(
        artifact.path,
        repository_root,
        repository_anchor,
    )
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
    repository_anchor: DirectoryAnchor | None = None,
) -> TranscribeOneOutcome:
    try:
        descriptor = None if verification is None else verification.descriptor
        execution_attestation = None if verification is None else verification.execution_attestation
        if execution_attestation is not None:
            execution_attestation = _relative_artifact(
                execution_attestation,
                repository_root,
                repository_anchor,
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
            anchor=repository_anchor,
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
