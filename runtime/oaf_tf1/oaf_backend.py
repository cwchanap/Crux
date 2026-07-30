"""Frozen TensorFlow 1 OaF graph, coverage, and native event adapter."""

# Vendored imports are intentionally lazy and unavailable to host-side Pylint.
# Integrity checks keep their exact-schema branches local for fail-closed review.
# pylint: disable=import-error,import-outside-toplevel
# pylint: disable=too-many-arguments,too-many-boolean-expressions,too-many-branches
# pylint: disable=too-many-instance-attributes,too-many-lines,too-many-locals

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import stat
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

try:
    from .protocol import (
        AuthenticatedObject,
        ProtocolFailure,
        VerifiedWav,
        canonical_json_bytes,
        encode_binary64,
        load_authenticated_object,
        read_verified_canonical_wav,
    )
except (ImportError, ValueError):
    from protocol import (  # type: ignore
        AuthenticatedObject,
        ProtocolFailure,
        VerifiedWav,
        canonical_json_bytes,
        encode_binary64,
        load_authenticated_object,
        read_verified_canonical_wav,
    )

CHECKPOINT_COUNT = 130
REQUIRED_INFERENCE_COUNT = 78
NON_INFERENCE_COUNT = 52
MIN_MIDI_PITCH = 21
MODEL_SCRATCH_DIRECTORY = "/tmp/crux-oaf-model"
PRIVATE_CHECKPOINT_DIRECTORY = Path("/tmp/crux-oaf-checkpoint")
UNINSTRUMENTED_SEQUENCES_PATH = Path("/opt/crux/upstream/magenta/music/sequences_lib.py")
BACKEND_LOCK_PATH = Path("/run/crux/backend-lock.json")
RUNTIME_LOCK_PATH = Path("/run/crux/runtime-lock.json")
SEAL_EVIDENCE_PATH = Path("/run/crux/seal-evidence.json")
MODEL_CACHE_ROOT = Path("/model")
INPUT_ROOT = Path("/input")
RUNNER_SOURCE_MANIFEST_PATH = Path("/opt/crux/runtime/runner-source-manifest.json")
UPSTREAM_SOURCE_MANIFEST_PATH = Path("/opt/crux/vendor/source-manifest.json")
SMOKE_WAV_PATH = "smoke/canonical.wav"
SMOKE_ORACLE_PATH = Path("/input/smoke/smoke-oracle.json")

BACKEND_LOCK_KEYS = frozenset(
    {
        "architecture_id",
        "backend_id",
        "checkpoint_acquisition_evidence_sha256",
        "checkpoint_acquisition_request_sha256",
        "checkpoint_archive",
        "checkpoint_components",
        "checkpoint_inventory",
        "checkpoint_url",
        "descriptor_schema",
        "drum_prediction_map",
        "execution_report_schema",
        "host_adapter_source_manifest_sha256",
        "hparams",
        "hparams_source",
        "legacy_conversion_coverage_sha256",
        "legacy_score_report_schema",
        "max_input_audio_frames",
        "model_id",
        "native_metadata_fields",
        "native_metadata_schema_id",
        "native_output_bins",
        "native_output_space_id",
        "non_inference_inventory",
        "prediction_schema",
        "protocol_schema",
        "required_inference_inventory",
        "runtime_image_manifest_digest",
        "runtime_lock_sha256",
        "schema",
        "seal_evidence_sha256",
        "serialization",
        "smoke_audio_sha256",
        "smoke_oracle_sha256",
        "training_data_map_id",
        "training_groups",
        "upstream_repository",
        "upstream_source_commit",
        "upstream_source_manifest_sha256",
        "verification_report_schema",
    }
)
RUNTIME_LOCK_KEYS = frozenset(
    {
        "additional_system_packages",
        "base_image",
        "base_image_archive_keyring_sha256",
        "base_image_manifest_digest",
        "base_system_package_evidence_sha256",
        "base_system_package_inventory",
        "base_system_package_inventory_sha256",
        "base_system_package_request_sha256",
        "distribution_build_manifest_sha256",
        "environment",
        "oci_layout_manifest_sha256",
        "platform",
        "python_distributions",
        "python_version",
        "runner_source_manifest_sha256",
        "runtime_image_manifest_digest",
        "schema",
        "seal_evidence_sha256",
        "stdout_max_line_bytes",
        "stderr_max_line_bytes",
        "stderr_read_chunk_bytes",
        "stderr_ring_buffer_bytes",
        "tensorflow_abi",
        "tensorflow_build",
        "upstream_source_manifest_sha256",
    }
)
SEAL_EVIDENCE_KEYS = frozenset(
    {
        "additional_system_packages",
        "advisory_snapshot_sha256",
        "base_image_archive_keyring_sha256",
        "base_image_manifest_digest",
        "base_system_package_evidence_sha256",
        "base_system_package_inventory",
        "base_system_package_inventory_sha256",
        "base_system_package_request_sha256",
        "calibration_measurement_evidence_sha256",
        "calibration_measurement_request_sha256",
        "checkpoint_acquisition_evidence_sha256",
        "checkpoint_acquisition_request_sha256",
        "checkpoint_archive",
        "checkpoint_components",
        "checkpoint_inventory",
        "cpu_limit_millis",
        "distribution_build_manifest_sha256",
        "host_adapter_source_manifest_sha256",
        "instrumentation_patch_sha256",
        "legacy_conversion_coverage_sha256",
        "max_input_audio_frames",
        "measurements",
        "memory_limit_bytes",
        "native_host_evidence",
        "non_inference_inventory",
        "oci_layout_archive",
        "oci_layout_manifest_sha256",
        "pid_limit",
        "python_distributions",
        "reference_host_numeric_fingerprint",
        "request_deadline_seconds",
        "required_inference_inventory",
        "runner_source_manifest_sha256",
        "runtime_gid",
        "runtime_image_config_digest",
        "runtime_image_layer_digests",
        "runtime_image_manifest_digest",
        "runtime_uid",
        "schema",
        "seal_candidate_sha256",
        "seal_profile_request_sha256",
        "security_scan_sha256",
        "shm_bytes",
        "smoke_audio_sha256",
        "smoke_oracle_sha256",
        "smoke_prediction_sha256",
        "startup_deadline_seconds",
        "stdout_max_line_bytes",
        "stderr_max_line_bytes",
        "stderr_read_chunk_bytes",
        "stderr_ring_buffer_bytes",
        "tensor_coverage_sha256",
        "tensorflow_abi",
        "tensorflow_build",
        "tmp_bytes",
        "upstream_source_manifest_sha256",
    }
)
UPSTREAM_MANIFEST_KEYS = frozenset(
    {
        "covered_roots",
        "files",
        "schema",
        "upstream_commit",
        "upstream_repository",
    }
)
RUNNER_MANIFEST_KEYS = frozenset({"covered_roots", "files", "schema"})
SMOKE_ORACLE_KEYS = frozenset(
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
DESCRIPTOR_FIELDS = (
    "architecture_id",
    "backend_id",
    "descriptor_schema",
    "model_id",
    "native_metadata_schema_id",
    "native_output_space_id",
    "prediction_schema",
    "protocol_schema",
    "runtime_image_manifest_digest",
    "runtime_lock_sha256",
    "training_data_map_id",
    "upstream_source_commit",
)
EXPECTED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
}


def _safe_relative_source_path(value):
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return False
    parts = value.split("/")
    return not value.startswith("/") and all(part not in {"", ".", ".."} for part in parts)


def _source_manifest_path_is_covered(path, roots):
    """Accept selected roots, top-level support files, and required package ancestors."""
    if "/" not in path:
        return True
    if any(path == root or path.startswith(root + "/") for root in roots):
        return True
    parent = path.rsplit("/", 1)[0]
    return any(root.startswith(parent + "/") for root in roots)


def _validate_mounted_source_manifest(payload, source_root):
    roots = payload.get("covered_roots")
    files = payload.get("files")
    if not isinstance(roots, list) or not roots or not isinstance(files, list) or not files:
        raise ProtocolFailure(
            "mounted_identity_invalid", "Mounted source manifest is invalid.", fatal=True
        )
    if len(set(roots)) != len(roots) or not all(_safe_relative_source_path(root) for root in roots):
        raise ProtocolFailure(
            "mounted_identity_invalid", "Mounted source roots are invalid.", fatal=True
        )
    for root in roots:
        try:
            metadata = (Path(source_root) / root).lstat()
        except OSError:
            raise ProtocolFailure(
                "mounted_identity_invalid", "Mounted source root is missing.", fatal=True
            ) from None
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ProtocolFailure(
                "mounted_identity_invalid", "Mounted source root is invalid.", fatal=True
            )
    paths = []
    for row in files:
        if not isinstance(row, Mapping) or not {"path", "sha256"}.issubset(row):
            raise ProtocolFailure(
                "mounted_identity_invalid", "Mounted source row is invalid.", fatal=True
            )
        path = row["path"]
        digest = row["sha256"]
        if (
            not _safe_relative_source_path(path)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not _source_manifest_path_is_covered(path, roots)
        ):
            raise ProtocolFailure(
                "mounted_identity_invalid", "Mounted source row is invalid.", fatal=True
            )
        try:
            content = (Path(source_root) / path).read_bytes()
        except OSError:
            raise ProtocolFailure(
                "mounted_identity_invalid", "Mounted source file is missing.", fatal=True
            ) from None
        if hashlib.sha256(content).hexdigest() != digest:
            raise ProtocolFailure(
                "mounted_identity_invalid", "Mounted source hash differs.", fatal=True
            )
        paths.append(path)
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ProtocolFailure(
            "mounted_identity_invalid", "Mounted source rows are invalid.", fatal=True
        )


def validate_schema_golden(schema, content):
    """Validate runner fixtures with the shared production lock authority."""
    if schema in {
        "crux.transcription-backend-lock/v1",
        "crux.transcription-runtime-lock/v1",
        "crux.backend-seal-evidence/v1",
        "crux.legacy-tf2-conversion-coverage/v1",
    }:
        try:
            from src.benchmark.backend_lock import validate_schema_golden as validate_host_golden

            validate_host_golden(schema, content)
            return
        except (ImportError, ValueError) as error:
            raise ValueError("runner schema golden is invalid") from error

    """Check runner-native exact schemas for isolated drift fixtures."""
    if schema in {"crux.oaf-smoke-oracle/v1", "crux.oaf-tensor-coverage/v1"}:
        try:
            from tools.hpa320 import seal_oaf_backend as seal

            value = json.loads(content[:-1].decode("utf-8"))
            if schema == "crux.oaf-tensor-coverage/v1":
                if set(value) != {
                    "active_predict_dropout",
                    "checkpoint_inventory",
                    "non_inference_inventory",
                    "note_sequence_byte_parity",
                    "required_inference_inventory",
                    "schema",
                    "uninitialized_required",
                }:
                    raise ValueError("schema golden tensor keys are invalid")
                if value["schema"] != "crux.oaf-tensor-coverage/v1":
                    raise ValueError("schema golden tensor schema is invalid")
                evidence = seal.LoadedSealEvidence(
                    Path("golden-seal.json"),
                    {
                        field: value[field]
                        for field in (
                            "checkpoint_inventory",
                            "required_inference_inventory",
                            "non_inference_inventory",
                        )
                    },
                    "0" * 64,
                )
                seal._validate_tensor_coverage(value, evidence)
                return
            if set(value) != SMOKE_ORACLE_KEYS:
                raise ValueError("schema golden smoke keys are invalid")
            if value["input_view_id"] != "fixture":
                raise ValueError("schema golden input view differs")
            audio = b"schema-golden-audio\n"
            prediction = b"schema-golden-prediction\n"
            if value["input_audio_sha256"] != hashlib.sha256(audio).hexdigest():
                raise ValueError("schema golden smoke audio hash is invalid")
            if (
                not isinstance(value["source_audio_sha256"], str)
                or len(value["source_audio_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value["source_audio_sha256"]
                )
            ):
                raise ValueError("schema golden source audio hash is invalid")
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                root = Path(directory)
                (root / seal.SMOKE_AUDIO_NAME).write_bytes(audio)
                (root / seal.SMOKE_PREDICTION_NAME).write_bytes(prediction)
                evidence = seal.LoadedSealEvidence(
                    root / "seal.json",
                    {
                        "smoke_audio_sha256": value["input_audio_sha256"],
                        "smoke_prediction_sha256": hashlib.sha256(prediction).hexdigest(),
                    },
                    "0" * 64,
                )
                seal._validate_smoke(value, root, evidence)
            return
        except (ImportError, OSError, ValueError) as error:
            raise ValueError("runner schema golden is invalid") from error
    expected_by_schema = {
        "crux.oaf-upstream-source-manifest/v1": UPSTREAM_MANIFEST_KEYS,
        "crux.oaf-runner-source-manifest/v1": RUNNER_MANIFEST_KEYS,
        "crux.oaf-smoke-oracle/v1": SMOKE_ORACLE_KEYS,
        "crux.oaf-tensor-coverage/v1": {
            "active_predict_dropout",
            "checkpoint_inventory",
            "non_inference_inventory",
            "note_sequence_byte_parity",
            "required_inference_inventory",
            "schema",
            "uninitialized_required",
        },
    }
    try:
        expected = expected_by_schema[schema]
        if not content.endswith(b"\n") or content.endswith(b"\n\n"):
            raise ValueError("schema golden newline is invalid")
        value = json.loads(content[:-1].decode("utf-8"))
        if not isinstance(value, dict) or set(value) != expected or value.get("schema") != schema:
            raise ValueError("schema golden keys are invalid")
        if canonical_json_bytes(value, trailing_newline=True) != content:
            raise ValueError("schema golden is not canonical")
        if "additional_system_packages" in expected and not isinstance(
            value["additional_system_packages"], list
        ):
            raise ValueError("schema golden additional packages are invalid")
        if "architecture_id" in expected and not isinstance(value["architecture_id"], str):
            raise ValueError("schema golden architecture is invalid")
        if "covered_roots" in expected and not isinstance(value["covered_roots"], list):
            raise ValueError("schema golden roots are invalid")
        if "covered_roots" in expected:
            roots = value["covered_roots"]
            files = value["files"]
            if (
                not roots
                or not all(_safe_relative_source_path(root) for root in roots)
                or roots != sorted(roots)
                or len(set(roots)) != len(roots)
                or not isinstance(files, list)
                or any(
                    not isinstance(row, dict)
                    or set(row) != {"path", "sha256"}
                    or not _safe_relative_source_path(row["path"])
                    or not isinstance(row["sha256"], str)
                    or len(row["sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in row["sha256"])
                    for row in files
                )
                or [row["path"] for row in files] != sorted(row["path"] for row in files)
                or len({row["path"] for row in files}) != len(files)
                or any(not _source_manifest_path_is_covered(row["path"], roots) for row in files)
            ):
                raise ValueError("schema golden source manifest is invalid")
            fixture_root = Path(__file__).parents[2] / "tests/benchmark/schema_goldens"
            for row in files:
                relative = Path(row["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("schema golden source path is unsafe")
                source = fixture_root / relative
                if (
                    not source.is_file()
                    or hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]
                ):
                    raise ValueError("schema golden source hash differs")
        if "upstream_commit" in expected and value["upstream_commit"] != "a" * 40:
            raise ValueError("schema golden upstream authority differs")
        if "input_audio_frame_count" in expected and (
            not isinstance(value["input_audio_frame_count"], int)
            or isinstance(value["input_audio_frame_count"], bool)
            or value["input_audio_frame_count"] <= 0
        ):
            raise ValueError("schema golden frame count is invalid")
        if "input_audio_frame_count" in expected and (
            not isinstance(value["input_audio_sha256"], str)
            or not isinstance(value["source_audio_sha256"], str)
            or any(
                len(value[field]) != 64
                or any(character not in "0123456789abcdef" for character in value[field])
                for field in ("input_audio_sha256", "source_audio_sha256")
            )
            or not all(
                isinstance(value[field], str) and value[field]
                for field in ("input_view_id", "source_audio_id")
            )
            or not isinstance(value["native_events"], list)
            or not value["native_events"]
        ):
            raise ValueError("schema golden smoke oracle is invalid")
        if "active_predict_dropout" in expected and value["active_predict_dropout"] is not False:
            raise ValueError("schema golden tensor coverage is invalid")
        if "active_predict_dropout" in expected and (
            value["uninitialized_required"] != []
            or value["note_sequence_byte_parity"] is not True
            or any(
                not isinstance(value[field], list)
                for field in (
                    "checkpoint_inventory",
                    "required_inference_inventory",
                    "non_inference_inventory",
                )
            )
        ):
            raise ValueError("schema golden tensor coverage inventories are invalid")
    except (UnicodeDecodeError, ValueError, TypeError):
        raise ValueError("runner schema golden is invalid") from None


_STOCHASTIC_OPERATION_TYPES = frozenset(
    {
        "Multinomial",
        "ParameterizedTruncatedNormal",
        "RandomGamma",
        "RandomPoisson",
        "RandomShuffle",
        "RandomStandardNormal",
        "RandomUniform",
        "RandomUniformInt",
        "StatelessMultinomial",
        "StatelessRandomGetKeyCounter",
        "StatelessRandomNormal",
        "StatelessRandomUniform",
        "StatelessRandomUniformInt",
        "StatelessTruncatedNormal",
        "TruncatedNormal",
    }
)
_CHECKPOINT_COMPONENT_NAMES = (
    "model.ckpt-569400.data-00000-of-00001",
    "model.ckpt-569400.index",
    "model.ckpt-569400.meta",
)


class AdapterItemFailure(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ModelIntegrityFailure(ValueError):
    pass


@dataclass(frozen=True)
class TensorCoverage:
    checkpoint_count: int
    required_count: int
    restored_count: int
    non_inference_count: int
    checkpoint_inventory_sha256: str
    required_inventory_sha256: str
    non_inference_inventory_sha256: str


@dataclass(frozen=True)
class ModelHandle:
    estimator: Any
    hparams: Any
    checkpoint_prefix: str
    coverage: TensorCoverage
    training_groups: Sequence[Mapping[str, Any]]
    uninstrumented_sequences_module: Any


@dataclass(frozen=True)
class StartupState:
    backend_lock: AuthenticatedObject
    runtime_lock: AuthenticatedObject
    seal_evidence: AuthenticatedObject
    runner_source_manifest: AuthenticatedObject
    upstream_source_manifest: AuthenticatedObject
    smoke_oracle: AuthenticatedObject
    model: ModelHandle
    input_root: Path
    descriptor: Mapping[str, str]
    descriptor_sha256: str
    max_input_audio_frames: int
    stdout_max_line_bytes: int
    ready_payload: Mapping[str, Any]


def frame_time_seconds(frame_index: int) -> float:
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise AdapterItemFailure("native_event_invalid", "The emitted frame index is invalid.")
    frames_per_second = 44100 / 512
    frame_length_seconds = 1.0 / frames_per_second
    return frame_index * frame_length_seconds


def velocity_to_midi(raw_velocity: float) -> int:
    value = float(raw_velocity)
    if not math.isfinite(value):
        raise AdapterItemFailure("nonfinite_velocity", "The emitted raw velocity is not finite.")
    clamped = max(min(value, 1.0), 0.0)
    unscaled = clamped * 127 + 0
    return int(unscaled)


def _group_for_pitch(
    native_midi_note: int, training_groups: Sequence[Mapping[str, Any]]
) -> Optional[str]:
    matches = []
    for group in training_groups:
        members = group.get("member_pitches")
        if isinstance(members, (list, tuple)) and native_midi_note in members:
            group_id = group.get("group_id")
            if not isinstance(group_id, str) or not group_id:
                raise AdapterItemFailure(
                    "native_event_invalid", "The locked training group is invalid."
                )
            matches.append(group_id)
    if len(matches) > 1:
        raise AdapterItemFailure(
            "native_event_invalid", "The emitted pitch has ambiguous training groups."
        )
    return matches[0] if matches else None


def native_event_from_capture(
    *,
    start_frame: int,
    native_midi_note: int,
    raw_velocity: float,
    raw_confidence: float,
    training_groups: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if (
        isinstance(native_midi_note, bool)
        or not isinstance(native_midi_note, int)
        or not MIN_MIDI_PITCH <= native_midi_note <= 108
    ):
        raise AdapterItemFailure(
            "native_event_invalid", "The emitted native MIDI pitch is outside the output space."
        )
    confidence = float(raw_confidence)
    if not math.isfinite(confidence):
        raise AdapterItemFailure(
            "nonfinite_confidence", "The emitted raw confidence is not finite."
        )
    output_bin = native_midi_note - MIN_MIDI_PITCH
    return {
        "confidence_binary64": encode_binary64(confidence),
        "frame_index": start_frame,
        "model_output_bin": output_bin,
        "native_class_id": "midi_" + str(native_midi_note),
        "native_midi_note": native_midi_note,
        "time_sec_binary64": encode_binary64(frame_time_seconds(start_frame)),
        "upstream_8hit_group_id": _group_for_pitch(native_midi_note, training_groups),
        "velocity_midi": velocity_to_midi(raw_velocity),
    }


def _canonical_smoke_number(value: Any) -> str:
    value_type = value.__class__
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite smoke number")
        numeric_value = Decimal(repr(value))
    elif value_type is Decimal:
        if not value.is_finite():
            raise ValueError("nonfinite smoke number")
        numeric_value = value
    else:
        raise ValueError("unsupported smoke number")
    if numeric_value.is_zero():
        return "0"
    sign, digits, exponent = numeric_value.as_tuple()
    canonical_digits = list(digits)
    while len(canonical_digits) > 1 and canonical_digits[-1] == 0:
        canonical_digits.pop()
        exponent += 1
    return (
        ("-" if sign else "")
        + "".join(str(digit) for digit in canonical_digits)
        + "e"
        + str(exponent)
    )


def _canonical_smoke_value(value: Any):
    value_type = value.__class__
    if value is None:
        normalized = ["null"]
    elif value_type is bool:
        normalized = ["boolean", value]
    elif value_type is int:
        normalized = ["integer", str(value)]
    elif value_type in {float, Decimal}:
        normalized = ["number", _canonical_smoke_number(value)]
    elif value_type is str:
        normalized = ["string", value]
    elif value_type is list:
        normalized = ["list", [_canonical_smoke_value(item) for item in value]]
    elif value_type is dict:
        if any(key.__class__ is not str for key in value):
            raise ValueError("smoke object key is not a string")
        normalized = [
            "object",
            [[key, _canonical_smoke_value(value[key])] for key in sorted(value)],
        ]
    else:
        raise ValueError("unsupported smoke value")
    return normalized


def smoke_events_match(
    observed_events: Sequence[Mapping[str, Any]],
    oracle_events: Sequence[Mapping[str, Any]],
) -> bool:
    """Compare inference and strict-JSON oracle values canonically."""
    try:
        observed = _canonical_smoke_value(observed_events)
        oracle = _canonical_smoke_value(oracle_events)
    except ValueError:
        return False
    return canonical_json_bytes(observed, trailing_newline=False) == canonical_json_bytes(
        oracle, trailing_newline=False
    )


def _normalize_inventory(
    values: Sequence[Mapping[str, Any]], *, include_reason: bool, label: str
) -> Tuple[Mapping[str, Any], ...]:
    normalized = []
    names = set()
    expected_keys = {"dtype", "name", "shape"}
    if include_reason:
        expected_keys.add("reason")
    for value in values:
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ModelIntegrityFailure(label + " entry does not match the exact schema")
        name = value.get("name")
        dtype = value.get("dtype")
        shape = value.get("shape")
        if not isinstance(name, str) or not name or name in names:
            raise ModelIntegrityFailure(label + " names are invalid or duplicate")
        if not isinstance(dtype, str) or not dtype:
            raise ModelIntegrityFailure(label + " dtype is invalid")
        if not isinstance(shape, (list, tuple)) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in shape
        ):
            raise ModelIntegrityFailure(label + " shape is invalid")
        entry = {"dtype": dtype, "name": name, "shape": list(shape)}
        if include_reason:
            reason = value.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ModelIntegrityFailure(label + " reason is invalid")
            entry["reason"] = reason
        normalized.append(entry)
        names.add(name)
    if [entry["name"] for entry in normalized] != sorted(
        names, key=lambda item: item.encode("utf-8")
    ):
        raise ModelIntegrityFailure(label + " entries are not bytewise sorted")
    return tuple(normalized)


def _inventory_sha256(values: Sequence[Mapping[str, Any]]) -> str:
    content = json.dumps(
        list(values),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def validate_tensor_coverage(
    *,
    checkpoint_inventory: Sequence[Mapping[str, Any]],
    required_inventory: Sequence[Mapping[str, Any]],
    non_inference_inventory: Sequence[Mapping[str, Any]],
    graph_inventory: Sequence[Mapping[str, Any]],
    uninitialized_required: Sequence[str],
) -> TensorCoverage:
    if len(checkpoint_inventory) != CHECKPOINT_COUNT:
        raise ModelIntegrityFailure("checkpoint count must be exactly 130")
    if len(required_inventory) != REQUIRED_INFERENCE_COUNT:
        raise ModelIntegrityFailure("required count must be exactly 78")
    if len(non_inference_inventory) != NON_INFERENCE_COUNT:
        raise ModelIntegrityFailure("non-inference count must be exactly 52")
    checkpoint = _normalize_inventory(
        checkpoint_inventory, include_reason=False, label="checkpoint inventory"
    )
    required = _normalize_inventory(
        required_inventory, include_reason=False, label="required inventory"
    )
    non_inference = _normalize_inventory(
        non_inference_inventory, include_reason=True, label="non-inference inventory"
    )
    graph = _normalize_inventory(graph_inventory, include_reason=False, label="graph inventory")
    if graph != required:
        raise ModelIntegrityFailure("graph inventory does not match required inventory")
    classified_non_inference = tuple(
        {key: value for key, value in entry.items() if key != "reason"} for entry in non_inference
    )
    classified = tuple(
        sorted(required + classified_non_inference, key=lambda entry: entry["name"].encode("utf-8"))
    )
    if checkpoint != classified:
        raise ModelIntegrityFailure(
            "checkpoint inventory does not match the locked required/non-inference partition"
        )
    if uninitialized_required:
        raise ModelIntegrityFailure("required inference variable remains uninitialized")
    return TensorCoverage(
        checkpoint_count=len(checkpoint),
        required_count=len(required),
        restored_count=len(required),
        non_inference_count=len(non_inference),
        checkpoint_inventory_sha256=_inventory_sha256(checkpoint),
        required_inventory_sha256=_inventory_sha256(required),
        non_inference_inventory_sha256=_inventory_sha256(non_inference),
    )


def assert_no_reachable_stochastic_ops(fetch_operations: Iterable[Any]) -> None:
    pending = list(fetch_operations)
    visited = set()
    while pending:
        operation = pending.pop()
        identity = id(operation)
        if identity in visited:
            continue
        visited.add(identity)
        operation_type = getattr(operation, "type", None)
        if operation_type in _STOCHASTIC_OPERATION_TYPES:
            raise ModelIntegrityFailure(
                "PREDICT output has a reachable stochastic TensorFlow operation"
            )
        for tensor in getattr(operation, "inputs", ()):
            input_operation = getattr(tensor, "op", None)
            if input_operation is not None:
                pending.append(input_operation)
        pending.extend(getattr(operation, "control_inputs", ()))


def _plain_inventory(values: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    return tuple(
        {
            "dtype": str(value["dtype"]),
            "name": str(value["name"]),
            "shape": list(value["shape"]),
        }
        for value in values
    )


def _tensorflow_checkpoint_inventory(tf: Any, checkpoint_prefix: str):
    reader = tf.train.NewCheckpointReader(checkpoint_prefix)
    dtype_map = reader.get_variable_to_dtype_map()
    inventory = []
    for name, shape in tf.train.list_variables(checkpoint_prefix):
        dtype = dtype_map[name]
        dtype_name = getattr(dtype, "name", str(dtype))
        inventory.append({"dtype": dtype_name, "name": name, "shape": list(shape)})
    return tuple(sorted(inventory, key=lambda entry: entry["name"].encode("utf-8")))


def _flatten_prediction_operations(value: Any) -> Sequence[Any]:
    operations = []
    if isinstance(value, Mapping):
        for item in value.values():
            operations.extend(_flatten_prediction_operations(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            operations.extend(_flatten_prediction_operations(item))
    else:
        operation = getattr(value, "op", None)
        if operation is not None:
            operations.append(operation)
    return operations


def _build_coverage_graph(tf: Any, config: Any, hparams: Any):
    from magenta.models.onsets_frames_transcription import data

    graph = tf.Graph()
    with graph.as_default():
        examples = tf.placeholder(tf.string, [None], name="canonical_examples")
        dataset = data.provide_batch(
            examples=examples,
            preprocess_examples=True,
            params=hparams,
            is_training=False,
            shuffle_examples=False,
            skip_n_initial_records=0,
        )
        iterator = dataset.make_initializable_iterator()
        features, labels = iterator.get_next()
        estimator_spec = config.model_fn(
            features,
            labels,
            tf.estimator.ModeKeys.PREDICT,
            hparams,
            None,
        )
        variables = tuple(tf.global_variables())
        predictions = estimator_spec.predictions
    return graph, variables, predictions


def _tensorflow_graph_inventory(variables: Sequence[Any]):
    values = []
    for variable in variables:
        shape = variable.shape.as_list()
        values.append(
            {
                "dtype": variable.dtype.base_dtype.name,
                "name": variable.op.name,
                "shape": shape,
            }
        )
    return tuple(sorted(values, key=lambda entry: entry["name"].encode("utf-8")))


def _validated_checkpoint_components(components: Sequence[Mapping[str, Any]]):
    locked_components = []
    for component in components:
        if not isinstance(component, Mapping) or set(component) != {"name", "sha256", "size"}:
            raise ModelIntegrityFailure("checkpoint component lock is invalid")
        name = component["name"]
        digest = component["sha256"]
        size = component["size"]
        if (
            not isinstance(name, str)
            or "/" in name
            or "\\" in name
            or name in {"", ".", ".."}
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ModelIntegrityFailure("checkpoint component lock is invalid")
        locked_components.append((name, digest, size))
    names = tuple(name for name, _, _ in locked_components)
    if names != _CHECKPOINT_COMPONENT_NAMES:
        raise ModelIntegrityFailure("checkpoint component set is invalid")
    return tuple(locked_components)


def _copy_authenticated_checkpoint_component(
    source_path: Path,
    destination_path: Path,
    digest: str,
    size: int,
) -> None:
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(str(source_path), source_flags)
    try:
        source_status = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_status.st_mode) or source_status.st_size != size:
            raise OSError("checkpoint component shape differs")
        output_descriptor = os.open(
            str(destination_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
        )
        with os.fdopen(output_descriptor, "wb") as output:
            content_hash = hashlib.sha256()
            copied_size = 0
            while True:
                chunk = os.read(source_descriptor, 65536)
                if not chunk:
                    break
                content_hash.update(chunk)
                copied_size += len(chunk)
                output.write(chunk)
            if copied_size != size or content_hash.hexdigest() != digest:
                raise OSError("checkpoint component digest differs")
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(source_descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        str(path),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_checkpoint_stage(stage: Path) -> None:
    for name in _CHECKPOINT_COMPONENT_NAMES:
        try:
            (stage / name).unlink()
        except FileNotFoundError:
            pass
    try:
        stage.rmdir()
    except FileNotFoundError:
        pass


def materialize_authenticated_checkpoint(
    model_cache_root: Path,
    components: Sequence[Mapping[str, Any]],
    *,
    private_root: Path = PRIVATE_CHECKPOINT_DIRECTORY,
) -> str:
    """Authenticate mounted bytes once and atomically publish a private checkpoint."""
    locked_components = _validated_checkpoint_components(components)

    model_cache_root = Path(model_cache_root)
    private_root = Path(private_root)
    stage = None
    try:
        entries = tuple(entry.name for entry in os.scandir(str(model_cache_root)))
        if set(entries) != set(_CHECKPOINT_COMPONENT_NAMES) or len(entries) != len(
            _CHECKPOINT_COMPONENT_NAMES
        ):
            raise OSError("checkpoint cache entry set differs")
        private_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            private_root.lstat()
        except FileNotFoundError:
            pass
        else:
            raise OSError("private checkpoint destination already exists")
        stage = Path(
            tempfile.mkdtemp(
                prefix=".crux-oaf-checkpoint-stage-",
                dir=str(private_root.parent),
            )
        )
        os.chmod(str(stage), 0o700)

        for name, digest, size in locked_components:
            _copy_authenticated_checkpoint_component(
                model_cache_root / name,
                stage / name,
                digest,
                size,
            )

        _fsync_directory(stage)
        os.rename(str(stage), str(private_root))
        stage = None
        _fsync_directory(private_root.parent)
    except OSError:
        if stage is not None:
            _remove_checkpoint_stage(stage)
        raise ModelIntegrityFailure("checkpoint component identity mismatch") from None
    return str(private_root / "model.ckpt-569400")


def configure_prediction_estimator_session(estimator: Any, tf: Any):
    """Bind the Estimator prediction path to the locked thread configuration."""
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
    )
    configured = estimator.config.replace(session_config=session_config)
    setattr(estimator, "_config", configured)
    setattr(estimator, "_session_config", configured.session_config)
    return estimator


def load_uninstrumented_sequences_module(
    source_path: Path = UNINSTRUMENTED_SEQUENCES_PATH,
):
    """Load the unmodified converter from the separately retained source tree."""
    source = Path(source_path)
    try:
        status = source.lstat()
        if not stat.S_ISREG(status.st_mode):
            raise OSError("unmodified converter is not a regular file")
        spec = importlib.util.spec_from_file_location(
            "_crux_uninstrumented_sequences_lib",
            str(source),
        )
        if spec is None or spec.loader is None:
            raise OSError("unmodified converter has no source loader")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (OSError, ImportError, ValueError) as error:
        raise ModelIntegrityFailure("unmodified sequence converter is unavailable") from error
    return module


def build_and_restore_model(
    backend_lock: Mapping[str, Any], model_cache_root: Path = MODEL_CACHE_ROOT
) -> ModelHandle:
    import tensorflow.compat.v1 as tf
    from magenta.models.onsets_frames_transcription import configs, train_util

    config = configs.CONFIG_MAP["drums"]
    hparams = copy.deepcopy(config.hparams)
    hparams.batch_size = 1
    hparams.truncated_length_secs = 0
    checkpoint_prefix = materialize_authenticated_checkpoint(
        Path(model_cache_root), backend_lock["checkpoint_components"]
    )
    graph, variables, predictions = _build_coverage_graph(tf, config, hparams)
    graph_inventory = _tensorflow_graph_inventory(variables)
    checkpoint_inventory = _tensorflow_checkpoint_inventory(tf, checkpoint_prefix)
    required_inventory = _plain_inventory(backend_lock["required_inference_inventory"])
    non_inference_inventory = backend_lock["non_inference_inventory"]
    required_names = {entry["name"] for entry in required_inventory}
    required_variables = {
        variable.op.name: variable for variable in variables if variable.op.name in required_names
    }
    if set(required_variables) != required_names:
        raise ModelIntegrityFailure("required graph variable set is incomplete")
    with graph.as_default():
        required_saver = tf.train.Saver(var_list=required_variables)
        required_uninitialized = tf.report_uninitialized_variables(
            var_list=list(required_variables.values())
        )
    with tf.Session(
        graph=graph,
        config=tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
        ),
    ) as session:
        required_saver.restore(session, checkpoint_prefix)
        uninitialized_raw = session.run(required_uninitialized)
        uninitialized = tuple(
            value.decode("utf-8", errors="strict") if isinstance(value, bytes) else str(value)
            for value in uninitialized_raw
        )
        assert_no_reachable_stochastic_ops(_flatten_prediction_operations(predictions))
    coverage = validate_tensor_coverage(
        checkpoint_inventory=checkpoint_inventory,
        required_inventory=required_inventory,
        non_inference_inventory=non_inference_inventory,
        graph_inventory=graph_inventory,
        uninitialized_required=uninitialized,
    )
    estimator = configure_prediction_estimator_session(
        train_util.create_estimator(
            config.model_fn,
            MODEL_SCRATCH_DIRECTORY,
            hparams,
        ),
        tf,
    )
    return ModelHandle(
        estimator=estimator,
        hparams=hparams,
        checkpoint_prefix=checkpoint_prefix,
        coverage=coverage,
        training_groups=backend_lock["training_groups"],
        uninstrumented_sequences_module=load_uninstrumented_sequences_module(),
    )


def _serialized_example(verified_wav: VerifiedWav) -> bytes:
    import six
    from magenta.models.onsets_frames_transcription import audio_label_data_utils
    from magenta.music.protobuf import music_pb2

    examples = list(
        audio_label_data_utils.process_record(
            wav_data=verified_wav.content,
            sample_rate=44100,
            ns=music_pb2.NoteSequence(),
            example_id=six.ensure_text(verified_wav.relative_path, "utf-8"),
            min_length=0,
            max_length=-1,
            allow_empty_notesequence=True,
            load_audio_with_librosa=False,
        )
    )
    if len(examples) != 1:
        raise AdapterItemFailure(
            "example_count_invalid", "Canonical preprocessing did not return one example."
        )
    return examples[0].SerializeToString()


def transcribe_canonical_wav(model: ModelHandle, verified_wav: VerifiedWav):
    import tensorflow.compat.v1 as tf
    from magenta.models.onsets_frames_transcription import data, infer_util

    serialized = _serialized_example(verified_wav)

    def transcription_data(params):
        return data.provide_batch(
            examples=tf.constant([serialized], dtype=tf.string),
            preprocess_examples=True,
            params=params,
            is_training=False,
            shuffle_examples=False,
            skip_n_initial_records=0,
        )

    input_fn = infer_util.labels_to_features_wrapper(transcription_data)
    predictions = list(
        model.estimator.predict(
            input_fn,
            checkpoint_path=model.checkpoint_prefix,
            yield_single_examples=False,
        )
    )
    if len(predictions) != 1:
        raise AdapterItemFailure(
            "prediction_batch_count_invalid",
            "Frozen inference did not return exactly one prediction batch.",
        )
    prediction = predictions[0]
    sequence, captured = infer_util.predict_sequence(
        frame_probs=prediction["frame_probs"][0],
        onset_probs=prediction["onset_probs"][0],
        frame_predictions=prediction["frame_predictions"][0],
        onset_predictions=prediction["onset_predictions"][0],
        offset_predictions=prediction["offset_predictions"][0],
        velocity_values=prediction["velocity_values"][0],
        min_pitch=MIN_MIDI_PITCH,
        hparams=model.hparams,
        onsets_only=True,
        capture_emitted_frames=True,
    )
    upstream_sequence = model.uninstrumented_sequences_module.pianoroll_onsets_to_note_sequence(
        onsets=prediction["onset_predictions"][0],
        frames_per_second=data.hparams_frames_per_second(model.hparams),
        note_duration_seconds=0.05,
        min_midi_pitch=MIN_MIDI_PITCH,
        velocity_values=prediction["velocity_values"][0],
        velocity_scale=model.hparams.velocity_scale,
        velocity_bias=model.hparams.velocity_bias,
    )
    if sequence.SerializeToString() != upstream_sequence.SerializeToString():
        raise ModelIntegrityFailure("instrumented sequence differs from upstream prediction")
    paired = infer_util.pair_emitted_frame_confidence(
        captured, prediction["onset_probs"][0], min_pitch=MIN_MIDI_PITCH
    )
    return [
        native_event_from_capture(
            start_frame=int(start_frame),
            native_midi_note=int(native_midi_note),
            raw_velocity=float(raw_velocity),
            raw_confidence=float(raw_confidence),
            training_groups=model.training_groups,
        )
        for start_frame, native_midi_note, raw_velocity, raw_confidence in paired
    ]


def _descriptor(backend: AuthenticatedObject) -> Tuple[Mapping[str, str], str]:
    payload = backend.payload
    component_identity = hashlib.sha256(
        canonical_json_bytes(payload["checkpoint_components"], trailing_newline=False)
    ).hexdigest()
    descriptor = {field: payload[field] for field in DESCRIPTOR_FIELDS}
    descriptor.update(
        {
            "backend_lock_sha256": backend.sha256,
            "model_artifact_set_sha256": component_identity,
        }
    )
    if any(not isinstance(value, str) for value in descriptor.values()):
        raise ProtocolFailure(
            "mounted_identity_invalid",
            "The mounted backend descriptor is invalid.",
            fatal=True,
        )
    descriptor_bytes = canonical_json_bytes(descriptor, trailing_newline=False)
    return descriptor, hashlib.sha256(descriptor_bytes).hexdigest()


def _require_same(left: Mapping[str, Any], right: Mapping[str, Any], field: str) -> None:
    if left.get(field) != right.get(field):
        raise ProtocolFailure(
            "mounted_identity_invalid",
            "Mounted runner identities do not agree.",
            fatal=True,
        )


def validate_runtime_environment(environment: Any) -> None:
    """Reject a mounted runtime lock that differs from the sealed image environment."""
    if not isinstance(environment, Mapping) or dict(environment) != EXPECTED_ENVIRONMENT:
        raise ProtocolFailure(
            "runtime_environment_mismatch",
            "Mounted runtime environment does not match the sealed image.",
            fatal=True,
        )


def authenticate_runtime_environment(
    *,
    backend_lock_path: Path = BACKEND_LOCK_PATH,
    runtime_lock_path: Path = RUNTIME_LOCK_PATH,
) -> AuthenticatedObject:
    """Authenticate runtime environment identity before numeric imports."""
    backend = load_authenticated_object(
        backend_lock_path,
        label="backend_lock",
        exact_keys=BACKEND_LOCK_KEYS,
        expected_schema="crux.transcription-backend-lock/v1",
    )
    runtime = load_authenticated_object(
        runtime_lock_path,
        label="runtime_lock",
        exact_keys=RUNTIME_LOCK_KEYS,
        expected_schema="crux.transcription-runtime-lock/v1",
        expected_sha256=backend.payload["runtime_lock_sha256"],
    )
    validate_runtime_environment(runtime.payload["environment"])
    return runtime


def authenticate_startup(
    *,
    backend_lock_path: Path = BACKEND_LOCK_PATH,
    runtime_lock_path: Path = RUNTIME_LOCK_PATH,
    seal_evidence_path: Path = SEAL_EVIDENCE_PATH,
    runner_source_manifest_path: Path = RUNNER_SOURCE_MANIFEST_PATH,
    upstream_source_manifest_path: Path = UPSTREAM_SOURCE_MANIFEST_PATH,
    model_cache_root: Path = MODEL_CACHE_ROOT,
    input_root: Path = INPUT_ROOT,
    smoke_oracle_path: Path = SMOKE_ORACLE_PATH,
) -> StartupState:
    backend = load_authenticated_object(
        backend_lock_path,
        label="backend_lock",
        exact_keys=BACKEND_LOCK_KEYS,
        expected_schema="crux.transcription-backend-lock/v1",
    )
    runtime = load_authenticated_object(
        runtime_lock_path,
        label="runtime_lock",
        exact_keys=RUNTIME_LOCK_KEYS,
        expected_schema="crux.transcription-runtime-lock/v1",
        expected_sha256=backend.payload["runtime_lock_sha256"],
    )
    validate_runtime_environment(runtime.payload["environment"])
    seal = load_authenticated_object(
        seal_evidence_path,
        label="seal_evidence",
        exact_keys=SEAL_EVIDENCE_KEYS,
        expected_schema="crux.backend-seal-evidence/v1",
        expected_sha256=backend.payload["seal_evidence_sha256"],
    )
    runner_manifest = load_authenticated_object(
        runner_source_manifest_path,
        label="runner_source_manifest",
        exact_keys=RUNNER_MANIFEST_KEYS,
        expected_schema="crux.oaf-runner-source-manifest/v1",
        expected_sha256=runtime.payload["runner_source_manifest_sha256"],
    )
    upstream_manifest = load_authenticated_object(
        upstream_source_manifest_path,
        label="upstream_source_manifest",
        exact_keys=UPSTREAM_MANIFEST_KEYS,
        expected_schema="crux.oaf-upstream-source-manifest/v1",
        expected_sha256=runtime.payload["upstream_source_manifest_sha256"],
    )
    runner_source_root = Path(runner_source_manifest_path).parents[1]
    upstream_source_root = Path(upstream_source_manifest_path).parents[1] / "upstream"
    _validate_mounted_source_manifest(runner_manifest.payload, runner_source_root)
    _validate_mounted_source_manifest(upstream_manifest.payload, upstream_source_root)
    smoke_oracle = load_authenticated_object(
        smoke_oracle_path,
        label="smoke_oracle",
        exact_keys=SMOKE_ORACLE_KEYS,
        expected_schema="crux.oaf-smoke-oracle/v1",
        expected_sha256=backend.payload["smoke_oracle_sha256"],
    )
    for field in (
        "runtime_image_manifest_digest",
        "upstream_source_manifest_sha256",
    ):
        _require_same(backend.payload, runtime.payload, field)
    for field in (
        "checkpoint_inventory",
        "checkpoint_components",
        "required_inference_inventory",
        "non_inference_inventory",
        "smoke_audio_sha256",
        "smoke_oracle_sha256",
    ):
        _require_same(backend.payload, seal.payload, field)
    _require_same(runtime.payload, seal.payload, "runner_source_manifest_sha256")
    descriptor, descriptor_sha256 = _descriptor(backend)
    model = build_and_restore_model(backend.payload, model_cache_root)
    maximum_frames = backend.payload["max_input_audio_frames"]
    smoke = read_verified_canonical_wav(
        input_root,
        SMOKE_WAV_PATH,
        backend.payload["smoke_audio_sha256"],
        maximum_frames,
    )
    smoke_events = transcribe_canonical_wav(model, smoke)
    if not smoke_events_match(smoke_events, smoke_oracle.payload["native_events"]):
        raise ProtocolFailure(
            "smoke_oracle_mismatch",
            "The frozen smoke prediction does not match its exact oracle.",
            fatal=True,
        )
    coverage = model.coverage
    ready = {
        "backend_descriptor": descriptor,
        "backend_descriptor_sha256": descriptor_sha256,
        "backend_lock_sha256": backend.sha256,
        "checkpoint_inventory_sha256": coverage.checkpoint_inventory_sha256,
        "non_inference_count": coverage.non_inference_count,
        "non_inference_inventory_sha256": coverage.non_inference_inventory_sha256,
        "protocol_schema": "crux.transcription-runner/v1",
        "python_version": platform.python_version(),
        "required_inference_count": coverage.required_count,
        "required_inference_inventory_sha256": coverage.required_inventory_sha256,
        "restored_inference_count": coverage.restored_count,
        "runner_source_manifest_sha256": runner_manifest.sha256,
        "runtime_lock_sha256": runtime.sha256,
        "smoke_audio_sha256": smoke.sha256,
        "smoke_oracle_sha256": smoke_oracle.sha256,
        "smoke_prediction_sha256": seal.payload["smoke_prediction_sha256"],
        "smoke_status": "exact_match",
        "tensorflow_abi": runtime.payload["tensorflow_abi"],
        "tensorflow_build": runtime.payload["tensorflow_build"],
        "type": "ready",
        "upstream_source_manifest_sha256": upstream_manifest.sha256,
    }
    return StartupState(
        backend_lock=backend,
        runtime_lock=runtime,
        seal_evidence=seal,
        runner_source_manifest=runner_manifest,
        upstream_source_manifest=upstream_manifest,
        smoke_oracle=smoke_oracle,
        model=model,
        input_root=Path(input_root),
        descriptor=descriptor,
        descriptor_sha256=descriptor_sha256,
        max_input_audio_frames=maximum_frames,
        stdout_max_line_bytes=runtime.payload["stdout_max_line_bytes"],
        ready_payload=ready,
    )


class FrozenOafBackend:
    def __init__(self, model: ModelHandle) -> None:
        self._model = model

    @classmethod
    def from_startup(cls, startup: StartupState) -> "FrozenOafBackend":
        return cls(startup.model)

    def transcribe(self, verified_wav: VerifiedWav):
        try:
            return transcribe_canonical_wav(self._model, verified_wav)
        except AdapterItemFailure as error:
            raise ProtocolFailure(error.code, str(error), fatal=False) from None
        except ModelIntegrityFailure:
            raise ProtocolFailure(
                "model_integrity_failure",
                "The frozen model failed an inference integrity check.",
                fatal=True,
            ) from None
        except BaseException:
            raise ProtocolFailure(
                "inference_failed",
                "The frozen model could not transcribe this canonical input.",
                fatal=False,
            ) from None
