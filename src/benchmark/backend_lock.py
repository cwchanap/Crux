# Immutable schema tables intentionally keep this single-purpose module self-contained.
# pylint: disable=too-many-lines
from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import cast

from src.benchmark.backend_identity import (
    OAF_DESCRIPTOR_KEYS,
    BackendDescriptor,
    JsonValue,
    StrictJsonError,
    build_descriptor,
    canonical_json_bytes,
    normalize_known_backend_descriptor,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)

BACKEND_LOCK_SCHEMA = "crux.transcription-backend-lock/v1"
RUNTIME_LOCK_SCHEMA = "crux.transcription-runtime-lock/v1"
SEAL_EVIDENCE_SCHEMA = "crux.backend-seal-evidence/v1"
CONVERSION_AUDIT_SCHEMA = "crux.legacy-tf2-conversion-coverage/v1"

_BACKEND_IDENTITIES = MappingProxyType(
    {
        "architecture_id": "magenta-oaf-model-tpu-drums-v1",
        "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
        "descriptor_schema": "crux.transcription-backend-descriptor/v1",
        "execution_report_schema": "crux.backend-execution-report/v1",
        "legacy_score_report_schema": "crux.legacy-score-report/v1",
        "model_id": "magenta-egmd-ckpt-569400-v1",
        "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
        "native_output_space_id": "magenta-oaf-midi88-a0-v1",
        "prediction_schema": "crux.drum-prediction-events/v1",
        "protocol_schema": "crux.transcription-runner/v1",
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
        "upstream_repository": "https://github.com/magenta/magenta",
        "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
        "verification_report_schema": "crux.backend-verification-report/v1",
    }
)
_CHECKPOINT_ARCHIVE = MappingProxyType(
    {
        "name": "e-gmd_checkpoint.zip",
        "sha256": "09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0",
    }
)
_CHECKPOINT_URL = (
    "https://storage.googleapis.com/magentadata/models/"
    "onsets_frames_transcription/e-gmd_checkpoint.zip"
)
_HPARAMS_SOURCE = "magenta/models/onsets_frames_transcription/configs.py:drums"
_OBSERVED_HDF5_SHA256 = "d36ced8b2ee241bc37ad6fbb918ba38e95d666350dd4888bca59a1243bf4d10e"
_DEBIAN_SNAPSHOT_PATTERN = re.compile(
    r"https://snapshot\.debian\.org/archive/debian/(?P<timestamp>[0-9]{8}T[0-9]{6}Z)/?"
)
_DEBIAN_SNAPSHOT_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
CHECKPOINT_COMPONENT_HASHES = MappingProxyType(
    {
        "model.ckpt-569400.data-00000-of-00001": (
            "6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5"
        ),
        "model.ckpt-569400.index": (
            "475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a"
        ),
        "model.ckpt-569400.meta": (
            "e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422"
        ),
    }
)
REQUIRED_ENVIRONMENT = MappingProxyType(
    {
        "CUDA_VISIBLE_DEVICES": "-1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "TF_NUM_INTEROP_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
    }
)
REQUIRED_HPARAMS = MappingProxyType(
    {
        "acoustic_rnn_dropout_keep_prob": Decimal("0.5"),
        "bidirectional": True,
        "combined_lstm_dropout_keep_prob": 1,
        "combined_lstm_units": 256,
        "conv_dropout_keep_amts": (1, Decimal("0.25"), Decimal("0.25")),
        "conv_filters": (16, 16, 32),
        "drum_data_map": "8-hit",
        "drum_note_duration": Decimal("0.05"),
        "drum_prediction_map": "",
        "drums_only": True,
        "fc_dropout_keep_amt": Decimal("0.5"),
        "fc_size": 256,
        "frame_lstm_units": 0,
        "frame_threshold": Decimal("0.5"),
        "hop_length": 441,
        "log_amplitude": True,
        "min_gap": None,
        "num_pitches": 88,
        "offset_lstm_units": 256,
        "offset_network": True,
        "offset_threshold": 0,
        "onset_length": 0,
        "onset_lstm_units": 64,
        "onset_threshold": Decimal("0.5"),
        "peak_picking": False,
        "sample_rate": 44100,
        "share_conv_features": False,
        "spec_fmin": 30,
        "spec_hop_length": 512,
        "spec_htk": True,
        "spec_mel_bins": 250,
        "spec_type": "mel",
        "transform_audio": False,
        "use_librosa": False,
        "velocity_bias": 0,
        "velocity_lstm_units": 0,
        "velocity_scale": 127,
        "viterbi_decoding": False,
    }
)
NATIVE_OUTPUT_BINS = tuple(
    MappingProxyType(
        {
            "model_output_bin": output_bin,
            "native_class_id": f"midi_{output_bin + 21}",
            "native_midi_note": output_bin + 21,
        }
    )
    for output_bin in range(88)
)
TRAINING_GROUPS = (
    MappingProxyType(
        {"base_midi": 36, "group_id": "kick", "member_pitches": (36,), "output_bin": 15}
    ),
    MappingProxyType(
        {
            "base_midi": 38,
            "group_id": "snare",
            "member_pitches": (38, 40, 37, 39),
            "output_bin": 17,
        }
    ),
    MappingProxyType(
        {
            "base_midi": 48,
            "group_id": "toms",
            "member_pitches": (48, 50, 45, 47, 43, 58, 64),
            "output_bin": 27,
        }
    ),
    MappingProxyType(
        {
            "base_midi": 46,
            "group_id": "hihat",
            "member_pitches": (46, 26, 42, 22, 44, 54, 70),
            "output_bin": 25,
        }
    ),
    MappingProxyType(
        {
            "base_midi": 51,
            "group_id": "ride",
            "member_pitches": (51, 59),
            "output_bin": 30,
        }
    ),
    MappingProxyType(
        {
            "base_midi": 53,
            "group_id": "ride_bell",
            "member_pitches": (53, 56),
            "output_bin": 32,
        }
    ),
    MappingProxyType(
        {
            "base_midi": 49,
            "group_id": "crash",
            "member_pitches": (49, 55, 57, 52),
            "output_bin": 28,
        }
    ),
    MappingProxyType(
        {"base_midi": 75, "group_id": "sticks", "member_pitches": (75,), "output_bin": 54}
    ),
)
_NATIVE_METADATA_FIELDS = (
    MappingProxyType({"name": "frame_index", "nullable": False, "type": "integer"}),
    MappingProxyType({"name": "upstream_group_id", "nullable": True, "type": "string"}),
)
_SERIALIZATION_RULES = MappingProxyType(
    {
        "encoding": "utf-8",
        "final_newline": True,
        "key_order": "lexicographic",
        "whitespace": "none",
    }
)

BACKEND_LOCK_KEYS = frozenset(
    {
        "architecture_id",
        "backend_id",
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
        "base_image",
        "base_image_manifest_digest",
        "debian_release_sha256",
        "debian_snapshot_repository",
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
        "system_packages",
        "tensorflow_abi",
        "tensorflow_build",
        "upstream_source_manifest_sha256",
    }
)
SEAL_EVIDENCE_KEYS = frozenset(
    {
        "advisory_snapshot_sha256",
        "base_image_manifest_digest",
        "checkpoint_archive",
        "checkpoint_components",
        "checkpoint_inventory",
        "cpu_limit_millis",
        "debian_release_sha256",
        "debian_snapshot_repository",
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
        "request_deadline_seconds",
        "required_inference_inventory",
        "runner_source_manifest_sha256",
        "runtime_gid",
        "runtime_image_config_digest",
        "runtime_image_layer_digests",
        "runtime_image_manifest_digest",
        "runtime_uid",
        "schema",
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
        "system_packages",
        "tensor_coverage_sha256",
        "tensorflow_abi",
        "tensorflow_build",
        "tmp_bytes",
        "upstream_source_manifest_sha256",
    }
)
CONVERSION_AUDIT_KEYS = frozenset(
    {
        "candidate_matches",
        "converter_source_manifest_sha256",
        "matching_algorithm",
        "matching_algorithm_version",
        "model_artifact_set_sha256",
        "observed_hdf5_sha256",
        "required_inference_inventory_sha256",
        "restored_required",
        "restored_required_count",
        "schema",
        "tf2_model_source_manifest_sha256",
        "unmatched_required",
    }
)

_COMPONENT_KEYS = frozenset({"name", "sha256", "size"})
_VARIABLE_KEYS = frozenset({"dtype", "name", "shape"})
_NON_INFERENCE_KEYS = frozenset({"dtype", "name", "reason", "shape"})
_PACKAGE_KEYS = frozenset({"filename", "name", "sha256", "version"})
_MEASUREMENT_KEYS = frozenset(
    {
        "peak_cpu_millis",
        "peak_pid_count",
        "peak_rss_bytes",
        "peak_shm_bytes",
        "peak_tmp_bytes",
        "request_millis",
        "startup_millis",
    }
)
_RESOURCE_MEASUREMENT_RELATIONS = (
    ("memory_limit_bytes", "peak_rss_bytes", 1),
    ("pid_limit", "peak_pid_count", 1),
    ("tmp_bytes", "peak_tmp_bytes", 1),
    ("shm_bytes", "peak_shm_bytes", 1),
    ("startup_deadline_seconds", "startup_millis", 1000),
    ("request_deadline_seconds", "request_millis", 1000),
)
_CANDIDATE_MATCH_KEYS = frozenset(
    {
        "assigned",
        "candidate_name",
        "dtype_compatible",
        "match_kind",
        "required_name",
        "shape_compatible",
    }
)
_CANDIDATE_MATCH_KINDS = frozenset(
    {
        "dense_transpose",
        "exact_name",
        "loose_substring",
    }
)
_FINAL_LOCK_HASH_KEYS = frozenset({"backend_lock_sha256", "runtime_lock_sha256"})


class BackendLockError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedBackendLock:
    path: Path
    payload: Mapping[str, JsonValue]
    sha256: str
    descriptor: BackendDescriptor
    max_input_audio_frames: int


@dataclass(frozen=True)
class LoadedRuntimeLock:
    path: Path
    payload: Mapping[str, JsonValue]
    sha256: str


@dataclass(frozen=True)
class LoadedSealEvidence:
    path: Path
    payload: Mapping[str, JsonValue]
    sha256: str


@dataclass(frozen=True)
class LoadedConversionAudit:
    path: Path
    payload: Mapping[str, JsonValue]
    sha256: str


CandidateMatchSortKey = tuple[bytes, bytes, bytes]


def candidate_match_sort_key(
    match: Mapping[str, JsonValue],
) -> CandidateMatchSortKey:
    """Return the UTF-8 bytewise `(required_name, candidate_name, match_kind)` key."""
    fields = ("required_name", "candidate_name", "match_kind")
    values = tuple(
        _require_nonempty_string(match[field], f"candidate match {field}").encode("utf-8")
        for field in fields
    )
    return cast(CandidateMatchSortKey, values)


def load_backend_lock(path: Path) -> LoadedBackendLock:
    payload, digest = _load_canonical_object(path, "backend lock")
    _validate_backend_lock(payload)
    descriptor = _build_oaf_descriptor(payload, digest)
    return LoadedBackendLock(
        path=Path(path),
        payload=_immutable_payload(payload),
        sha256=digest,
        descriptor=descriptor,
        max_input_audio_frames=cast(int, payload["max_input_audio_frames"]),
    )


def revalidate_loaded_backend_lock(backend_lock: LoadedBackendLock) -> LoadedBackendLock:
    """Reproduce a loaded backend lock from its immutable in-memory payload."""
    if not isinstance(backend_lock, LoadedBackendLock):
        raise BackendLockError("backend lock was not strictly loaded")
    lock_fields_valid = (
        isinstance(backend_lock.path, Path)
        and isinstance(backend_lock.payload, Mapping)
        and isinstance(backend_lock.sha256, str)
    )
    descriptor_fields_valid = isinstance(backend_lock.descriptor, BackendDescriptor) and (
        isinstance(backend_lock.descriptor.payload, Mapping)
        and isinstance(backend_lock.descriptor.sha256, str)
    )
    frame_bound_valid = isinstance(backend_lock.max_input_audio_frames, int) and not isinstance(
        backend_lock.max_input_audio_frames, bool
    )
    if not (lock_fields_valid and descriptor_fields_valid and frame_bound_valid):
        raise BackendLockError("loaded backend lock field types are invalid")
    try:
        payload = _mutable_payload(backend_lock.payload)
        _validate_backend_lock(payload)
        _reproduce_loaded_hash(payload, backend_lock.sha256, "backend lock")
        descriptor = _build_oaf_descriptor(payload, backend_lock.sha256)
        if (
            dict(descriptor.payload) != dict(backend_lock.descriptor.payload)
            or descriptor.sha256 != backend_lock.descriptor.sha256
        ):
            raise BackendLockError("backend descriptor reproduction mismatch")
        max_input_audio_frames = cast(int, payload["max_input_audio_frames"])
        if backend_lock.max_input_audio_frames != max_input_audio_frames:
            raise BackendLockError("backend audio frame bound reproduction mismatch")
    except BackendLockError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError):
        raise BackendLockError("loaded backend lock shape is invalid") from None
    return LoadedBackendLock(
        path=backend_lock.path,
        payload=_immutable_payload(payload),
        sha256=backend_lock.sha256,
        descriptor=descriptor,
        max_input_audio_frames=max_input_audio_frames,
    )


def load_runtime_lock(path: Path) -> LoadedRuntimeLock:
    payload, digest = _load_canonical_object(path, "runtime lock")
    _validate_runtime_lock(payload)
    return LoadedRuntimeLock(
        path=Path(path),
        payload=_immutable_payload(payload),
        sha256=digest,
    )


def load_seal_evidence(path: Path) -> LoadedSealEvidence:
    payload, digest = _load_canonical_object(path, "seal evidence")
    _validate_seal_evidence(payload)
    return LoadedSealEvidence(
        path=Path(path),
        payload=_immutable_payload(payload),
        sha256=digest,
    )


def load_conversion_audit(path: Path) -> LoadedConversionAudit:
    payload, digest = _load_canonical_object(path, "conversion audit")
    _validate_conversion_audit(payload)
    return LoadedConversionAudit(
        path=Path(path),
        payload=_immutable_payload(payload),
        sha256=digest,
    )


# The cross-record gate spells out each independent identity comparison.
# pylint: disable-next=too-many-branches
def validate_oaf_lock_set(
    backend: LoadedBackendLock,
    runtime: LoadedRuntimeLock,
    seal: LoadedSealEvidence,
    audit: LoadedConversionAudit,
) -> None:
    backend_payload = _mutable_payload(backend.payload)
    runtime_payload = _mutable_payload(runtime.payload)
    seal_payload = _mutable_payload(seal.payload)
    audit_payload = _mutable_payload(audit.payload)
    _reproduce_loaded_hash(backend_payload, backend.sha256, "backend lock")
    _reproduce_loaded_hash(runtime_payload, runtime.sha256, "runtime lock")
    _reproduce_loaded_hash(seal_payload, seal.sha256, "seal evidence")
    _reproduce_loaded_hash(audit_payload, audit.sha256, "conversion audit")

    _require_same(
        backend_payload,
        seal_payload,
        "checkpoint_archive",
        "checkpoint archive mismatch",
    )
    _require_same(
        backend_payload,
        seal_payload,
        "checkpoint_components",
        "checkpoint components mismatch",
    )
    for field in (
        "checkpoint_inventory",
        "required_inference_inventory",
        "non_inference_inventory",
    ):
        _require_same(backend_payload, seal_payload, field, "tensor inventories mismatch")
    for field in ("smoke_audio_sha256", "smoke_oracle_sha256"):
        _require_same(backend_payload, seal_payload, field, "smoke evidence mismatch")

    _require_same(
        backend_payload,
        runtime_payload,
        "runtime_image_manifest_digest",
        "runtime image manifest mismatch",
    )
    _require_same(
        runtime_payload,
        seal_payload,
        "runtime_image_manifest_digest",
        "runtime image manifest mismatch",
    )
    _require_same(
        runtime_payload,
        seal_payload,
        "base_image_manifest_digest",
        "base image manifest mismatch",
    )
    for field in ("tensorflow_abi", "tensorflow_build"):
        _require_same(
            runtime_payload,
            seal_payload,
            field,
            "TensorFlow runtime identity mismatch",
        )
    _require_same(
        runtime_payload,
        seal_payload,
        "oci_layout_manifest_sha256",
        "OCI layout manifest evidence mismatch",
    )
    _require_same(
        runtime_payload,
        seal_payload,
        "distribution_build_manifest_sha256",
        "distribution build manifest evidence mismatch",
    )
    _require_same(
        backend_payload,
        runtime_payload,
        "upstream_source_manifest_sha256",
        "upstream source manifest mismatch",
    )
    _require_same(
        runtime_payload,
        seal_payload,
        "upstream_source_manifest_sha256",
        "upstream source manifest mismatch",
    )
    _require_same(
        backend_payload,
        seal_payload,
        "host_adapter_source_manifest_sha256",
        "host adapter source manifest mismatch",
    )
    for field in (
        "runner_source_manifest_sha256",
        "stdout_max_line_bytes",
        "debian_snapshot_repository",
        "debian_release_sha256",
        "python_distributions",
        "system_packages",
        "stderr_read_chunk_bytes",
        "stderr_max_line_bytes",
        "stderr_ring_buffer_bytes",
    ):
        _require_same(runtime_payload, seal_payload, field, "runtime evidence mismatch")
    _require_same(
        backend_payload,
        seal_payload,
        "max_input_audio_frames",
        "maximum input audio frames mismatch",
    )

    artifact_set_sha256 = _inventory_identity(backend_payload["checkpoint_components"])
    if audit_payload["model_artifact_set_sha256"] != artifact_set_sha256:
        raise BackendLockError("model artifact set audit mismatch")
    required_sha256 = _inventory_identity(backend_payload["required_inference_inventory"])
    if audit_payload["required_inference_inventory_sha256"] != required_sha256:
        raise BackendLockError("required inventory audit mismatch")
    required_names = [
        cast(dict[str, JsonValue], row)["name"]
        for row in cast(list[JsonValue], backend_payload["required_inference_inventory"])
    ]
    if audit_payload["restored_required"] != []:
        raise BackendLockError("conversion audit must prove zero restored tensors")
    if audit_payload["unmatched_required"] != required_names:
        raise BackendLockError("conversion audit unmatched inventory mismatch")

    if backend_payload["runtime_lock_sha256"] != runtime.sha256:
        raise BackendLockError("runtime lock SHA-256 mismatch")
    if backend_payload["seal_evidence_sha256"] != seal.sha256:
        raise BackendLockError("seal evidence SHA-256 mismatch")
    if runtime_payload["seal_evidence_sha256"] != seal.sha256:
        raise BackendLockError("runtime seal evidence SHA-256 mismatch")
    if backend_payload["legacy_conversion_coverage_sha256"] != audit.sha256:
        raise BackendLockError("conversion audit SHA-256 mismatch")
    if seal_payload["legacy_conversion_coverage_sha256"] != audit.sha256:
        raise BackendLockError("seal conversion audit SHA-256 mismatch")

    reproduced = _build_oaf_descriptor(backend_payload, backend.sha256)
    if (
        dict(reproduced.payload) != dict(backend.descriptor.payload)
        or reproduced.sha256 != backend.descriptor.sha256
    ):
        raise BackendLockError("backend descriptor reproduction mismatch")


def _load_canonical_object(path: Path, label: str) -> tuple[dict[str, JsonValue], str]:
    try:
        content = _read_regular_file_once(Path(path), label)
        if not content.endswith(b"\n") or content.endswith(b"\n\n"):
            raise BackendLockError(f"{label} must have exactly one final newline")
        value = strict_json_loads(content[:-1], require_canonical=True)
        if not isinstance(value, dict):
            raise BackendLockError(f"{label} must be a JSON object")
        return value, sha256_hex(content)
    except (OSError, StrictJsonError) as error:
        message = str(error)
        if isinstance(error, OSError):
            message = f"{label} must be a no-follow regular file"
        raise BackendLockError(message) from None


def _read_regular_file_once(path: Path, label: str) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    non_block = getattr(os, "O_NONBLOCK", None)
    if no_follow is None or non_block is None:
        raise OSError(f"{label} no-follow reads are unavailable")
    descriptor = os.open(path, os.O_RDONLY | no_follow | non_block)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError(f"{label} is not a regular file")
        content = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
        if len(content) != before.st_size or after.st_size != before.st_size:
            raise OSError(f"{label} changed while being read")
        return content
    finally:
        os.close(descriptor)


def _validate_backend_lock(payload: dict[str, JsonValue]) -> None:
    _require_fields(payload, BACKEND_LOCK_KEYS, "backend lock")
    _require_exact_string(payload, "schema", BACKEND_LOCK_SCHEMA, "backend lock schema")
    for field, expected in _BACKEND_IDENTITIES.items():
        _require_exact_string(payload, field, expected, f"backend lock {field}")
    _require_exact_string(payload, "checkpoint_url", _CHECKPOINT_URL, "checkpoint URL")
    _require_exact_string(
        payload,
        "hparams_source",
        _HPARAMS_SOURCE,
        "backend lock hparams source",
    )
    _require_positive_integer(payload["max_input_audio_frames"], "max_input_audio_frames")
    for field in (
        "host_adapter_source_manifest_sha256",
        "legacy_conversion_coverage_sha256",
        "runtime_lock_sha256",
        "seal_evidence_sha256",
        "smoke_audio_sha256",
        "smoke_oracle_sha256",
        "upstream_source_manifest_sha256",
    ):
        _require_hash(payload[field], field)
    _require_digest(payload["runtime_image_manifest_digest"], "runtime image manifest")
    _validate_archive(payload["checkpoint_archive"])
    _validate_components(payload["checkpoint_components"])
    _validate_inventories(payload)
    hparams = _require_object(payload["hparams"], "backend lock resolved hparams")
    if payload["drum_prediction_map"] != "" or (
        "drum_prediction_map" in hparams and hparams["drum_prediction_map"] != ""
    ):
        raise BackendLockError("backend lock prediction map must remain disabled")
    if hparams != _thaw(REQUIRED_HPARAMS):
        raise BackendLockError("backend lock resolved hparams do not match frozen drums")
    if payload["native_output_bins"] != _thaw(NATIVE_OUTPUT_BINS):
        raise BackendLockError("backend lock 88-bin arithmetic is invalid")
    if payload["training_groups"] != _thaw(TRAINING_GROUPS):
        raise BackendLockError("backend lock 8-hit groups are invalid")
    if payload["native_metadata_fields"] != _thaw(_NATIVE_METADATA_FIELDS):
        raise BackendLockError("backend lock native metadata schema is invalid")
    if payload["serialization"] != _thaw(_SERIALIZATION_RULES):
        raise BackendLockError("backend lock serialization rules are invalid")


def _validate_runtime_lock(payload: dict[str, JsonValue]) -> None:
    _require_fields(payload, RUNTIME_LOCK_KEYS, "runtime lock")
    _require_exact_string(payload, "schema", RUNTIME_LOCK_SCHEMA, "runtime lock schema")
    for field, expected in (
        ("platform", "linux/amd64"),
        ("base_image", "python:3.7.17-slim-bullseye"),
        ("python_version", "3.7.17"),
        (
            "base_image_manifest_digest",
            "sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673",
        ),
    ):
        _require_exact_string(payload, field, expected, f"runtime {field}")
    for field in (
        "debian_release_sha256",
        "distribution_build_manifest_sha256",
        "oci_layout_manifest_sha256",
        "runner_source_manifest_sha256",
        "seal_evidence_sha256",
        "upstream_source_manifest_sha256",
    ):
        _require_hash(payload[field], field)
    _require_digest(payload["runtime_image_manifest_digest"], "runtime image manifest")
    _require_debian_snapshot(payload["debian_snapshot_repository"])
    _require_nonempty_string(payload["tensorflow_abi"], "TensorFlow ABI")
    _require_nonempty_string(payload["tensorflow_build"], "TensorFlow build")
    if payload["environment"] != dict(REQUIRED_ENVIRONMENT):
        raise BackendLockError("runtime environment does not match the frozen allowlist")
    for field in (
        "stdout_max_line_bytes",
        "stderr_max_line_bytes",
        "stderr_read_chunk_bytes",
        "stderr_ring_buffer_bytes",
    ):
        _require_positive_integer(payload[field], field)
    distributions = _validate_packages(payload["python_distributions"], "Python distributions")
    _validate_packages(payload["system_packages"], "system packages")
    tensorflow = next(
        (distribution for distribution in distributions if distribution["name"] == "tensorflow"),
        None,
    )
    expected_tensorflow = {
        "filename": "tensorflow-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl",
        "name": "tensorflow",
        "sha256": "29831dda98d668067de75403b2fca0d06a2f026ef6f217fa2ca873c20b4ee4d3",
        "version": "1.15.5",
    }
    if tensorflow != expected_tensorflow:
        raise BackendLockError("runtime TensorFlow distribution does not match the frozen wheel")


def _validate_seal_evidence(payload: dict[str, JsonValue]) -> None:
    if _contains_forbidden_lock_hash(payload):
        raise BackendLockError("seal evidence must exclude final lock hashes")
    _require_fields(payload, SEAL_EVIDENCE_KEYS, "seal evidence")
    _require_exact_string(payload, "schema", SEAL_EVIDENCE_SCHEMA, "seal evidence schema")
    for field in (
        "advisory_snapshot_sha256",
        "debian_release_sha256",
        "distribution_build_manifest_sha256",
        "host_adapter_source_manifest_sha256",
        "instrumentation_patch_sha256",
        "legacy_conversion_coverage_sha256",
        "oci_layout_manifest_sha256",
        "runner_source_manifest_sha256",
        "security_scan_sha256",
        "smoke_audio_sha256",
        "smoke_oracle_sha256",
        "smoke_prediction_sha256",
        "tensor_coverage_sha256",
        "upstream_source_manifest_sha256",
    ):
        _require_hash(payload[field], field)
    for field in (
        "base_image_manifest_digest",
        "runtime_image_config_digest",
        "runtime_image_manifest_digest",
    ):
        _require_digest(payload[field], field)
    _require_debian_snapshot(payload["debian_snapshot_repository"])
    _require_nonempty_string(payload["tensorflow_abi"], "TensorFlow ABI")
    _require_nonempty_string(payload["tensorflow_build"], "TensorFlow build")
    for field in (
        "cpu_limit_millis",
        "max_input_audio_frames",
        "memory_limit_bytes",
        "pid_limit",
        "request_deadline_seconds",
        "runtime_gid",
        "runtime_uid",
        "shm_bytes",
        "startup_deadline_seconds",
        "stdout_max_line_bytes",
        "stderr_max_line_bytes",
        "stderr_read_chunk_bytes",
        "stderr_ring_buffer_bytes",
        "tmp_bytes",
    ):
        _require_positive_integer(payload[field], field)
    measurements = _require_object(payload["measurements"], "seal measurements")
    _require_fields(measurements, _MEASUREMENT_KEYS, "seal measurements")
    for field in _MEASUREMENT_KEYS:
        _require_positive_integer(measurements[field], f"measurements {field}")
    for limit_field, measurement_field, scale in _RESOURCE_MEASUREMENT_RELATIONS:
        limit = cast(int, payload[limit_field])
        measured = cast(int, measurements[measurement_field])
        if limit * scale < measured:
            raise BackendLockError(f"{limit_field} must cover {measurement_field}")
    host = _require_object(payload["native_host_evidence"], "native host evidence")
    _require_fields(host, {"form", "sha256"}, "native host evidence")
    if host["form"] not in {
        "github_hosted_linux_x64",
        "native_seal_host",
        "orchestrator_signed",
    }:
        raise BackendLockError("native host evidence form is unsupported")
    _require_hash(host["sha256"], "native host evidence sha256")
    _validate_archive(payload["checkpoint_archive"])
    _validate_components(payload["checkpoint_components"])
    _validate_inventories(payload)
    _validate_packages(payload["python_distributions"], "Python distributions")
    _validate_packages(payload["system_packages"], "system packages")
    _validate_archive(payload["oci_layout_archive"], fixed_identity=False)
    layers = _require_list(payload["runtime_image_layer_digests"], "runtime image layers")
    if not layers:
        raise BackendLockError("runtime image layers must not be empty")
    normalized_layers = [_require_digest(layer, "runtime image layer") for layer in layers]
    if len(set(normalized_layers)) != len(normalized_layers):
        raise BackendLockError("runtime image layer digests must be unique")


def _validate_conversion_audit(payload: dict[str, JsonValue]) -> None:
    _require_fields(payload, CONVERSION_AUDIT_KEYS, "conversion audit")
    _require_exact_string(
        payload,
        "schema",
        CONVERSION_AUDIT_SCHEMA,
        "conversion audit schema",
    )
    for field in (
        "converter_source_manifest_sha256",
        "model_artifact_set_sha256",
        "required_inference_inventory_sha256",
        "tf2_model_source_manifest_sha256",
    ):
        _require_hash(payload[field], field)
    _require_exact_string(
        payload,
        "observed_hdf5_sha256",
        _OBSERVED_HDF5_SHA256,
        "conversion audit observed HDF5 SHA-256",
    )
    _require_exact_string(
        payload,
        "matching_algorithm",
        "exact_assignment_trace",
        "conversion audit matching algorithm",
    )
    _require_exact_string(
        payload,
        "matching_algorithm_version",
        "v1",
        "conversion audit matching algorithm version",
    )
    restored = _validate_name_list(payload["restored_required"], "restored required")
    unmatched = _validate_name_list(payload["unmatched_required"], "unmatched required")
    if payload["restored_required_count"] != 0 or restored:
        raise BackendLockError("conversion audit must prove zero restored tensors")
    if len(unmatched) != 78:
        raise BackendLockError("conversion audit must enumerate 78 unmatched tensors")
    matches = _require_list(payload["candidate_matches"], "candidate matches")
    required_inventory = set(restored) | set(unmatched)
    assigned_required: set[str] = set()
    match_keys: list[CandidateMatchSortKey] = []
    for value in matches:
        required_name, assigned, sort_key = _validate_candidate_match(
            value,
            required_inventory,
        )
        if assigned:
            assigned_required.add(required_name)
        match_keys.append(sort_key)
    if len(set(match_keys)) != len(match_keys):
        raise BackendLockError("conversion audit candidate match relations must be unique")
    if match_keys != sorted(match_keys):
        raise BackendLockError("conversion audit candidate matches must follow semantic key order")
    if assigned_required != set(restored):
        raise BackendLockError("conversion audit candidate assignments contradict zero restored")


def _validate_candidate_match(
    value: JsonValue,
    required_inventory: set[str],
) -> tuple[str, bool, CandidateMatchSortKey]:
    match = _require_object(value, "candidate match")
    _require_fields(match, _CANDIDATE_MATCH_KEYS, "candidate match")
    sort_key = candidate_match_sort_key(match)
    required_name = cast(str, match["required_name"])
    match_kind = cast(str, match["match_kind"])
    if match_kind not in _CANDIDATE_MATCH_KINDS:
        raise BackendLockError("conversion audit candidate match kind is unsupported")
    if required_name not in required_inventory:
        raise BackendLockError(
            "conversion audit candidate required_name is outside locked required inventory"
        )
    for field in ("assigned", "dtype_compatible", "shape_compatible"):
        if not isinstance(match[field], bool):
            raise BackendLockError(f"candidate match {field} must be boolean")
    assigned = cast(bool, match["assigned"])
    dtype_compatible = cast(bool, match["dtype_compatible"])
    shape_compatible = cast(bool, match["shape_compatible"])
    if assigned and not (dtype_compatible and shape_compatible):
        raise BackendLockError("assigned candidate must be dtype and shape compatible")
    return required_name, assigned, sort_key


def _validate_inventories(payload: Mapping[str, JsonValue]) -> None:
    checkpoint = _validate_variable_inventory(
        payload["checkpoint_inventory"],
        "checkpoint inventory",
        expected_count=130,
    )
    required = _validate_variable_inventory(
        payload["required_inference_inventory"],
        "required inference inventory",
        expected_count=78,
    )
    non_inference = _validate_variable_inventory(
        payload["non_inference_inventory"],
        "non-inference inventory",
        expected_count=52,
        with_reason=True,
    )
    checkpoint_by_name = {cast(str, row["name"]): row for row in checkpoint}
    required_names = {cast(str, row["name"]) for row in required}
    non_inference_names = {cast(str, row["name"]) for row in non_inference}
    if required_names & non_inference_names:
        raise BackendLockError("required and non-inference tensor inventory overlap")
    if required_names | non_inference_names != set(checkpoint_by_name):
        raise BackendLockError("tensor inventories do not cover the checkpoint exactly")
    for row in required:
        if checkpoint_by_name[cast(str, row["name"])] != row:
            raise BackendLockError("required tensor inventory does not match checkpoint")
    for row in non_inference:
        comparable = {key: value for key, value in row.items() if key != "reason"}
        if checkpoint_by_name[cast(str, row["name"])] != comparable:
            raise BackendLockError("non-inference inventory does not match checkpoint")


def _validate_variable_inventory(
    value: JsonValue,
    label: str,
    *,
    expected_count: int,
    with_reason: bool = False,
) -> list[dict[str, JsonValue]]:
    values = _require_list(value, label)
    if len(values) != expected_count:
        raise BackendLockError(f"{label} must contain exactly {expected_count} entries")
    rows: list[dict[str, JsonValue]] = []
    for value_row in values:
        row = _require_object(value_row, f"{label} row")
        _require_fields(
            row,
            _NON_INFERENCE_KEYS if with_reason else _VARIABLE_KEYS,
            f"{label} row",
        )
        _require_nonempty_string(row["name"], f"{label} name")
        _require_nonempty_string(row["dtype"], f"{label} dtype")
        if with_reason:
            _require_nonempty_string(row["reason"], f"{label} reason")
        shape = _require_list(row["shape"], f"{label} shape")
        for dimension in shape:
            _require_positive_integer(dimension, f"{label} shape dimension")
        rows.append(row)
    names = [cast(str, row["name"]) for row in rows]
    if len(set(names)) != len(names):
        raise BackendLockError(f"{label} names must be unique")
    if names != sorted(names):
        raise BackendLockError(f"{label} names must be lexically ordered")
    return rows


def _validate_archive(value: JsonValue, *, fixed_identity: bool = True) -> None:
    archive = _require_object(value, "archive")
    _require_fields(archive, _COMPONENT_KEYS, "archive")
    _require_nonempty_string(archive["name"], "archive name")
    _require_positive_integer(archive["size"], "archive size")
    _require_hash(archive["sha256"], "archive sha256")
    if fixed_identity and (
        archive["name"] != _CHECKPOINT_ARCHIVE["name"]
        or archive["sha256"] != _CHECKPOINT_ARCHIVE["sha256"]
    ):
        raise BackendLockError("checkpoint archive identity does not match the release")


def _validate_components(value: JsonValue) -> None:
    values = _require_list(value, "checkpoint components")
    if len(values) != 3:
        raise BackendLockError("checkpoint components must contain exactly three files")
    components: list[dict[str, JsonValue]] = []
    for value_row in values:
        row = _require_object(value_row, "checkpoint component")
        _require_fields(row, _COMPONENT_KEYS, "checkpoint component")
        _require_nonempty_string(row["name"], "checkpoint component name")
        _require_positive_integer(row["size"], "checkpoint component size")
        _require_hash(row["sha256"], "checkpoint component sha256")
        components.append(row)
    names = [cast(str, row["name"]) for row in components]
    if names != sorted(CHECKPOINT_COMPONENT_HASHES):
        raise BackendLockError("checkpoint component names do not match the release")
    if any(
        row["sha256"] != CHECKPOINT_COMPONENT_HASHES[cast(str, row["name"])] for row in components
    ):
        raise BackendLockError("checkpoint component identity does not match the release")


def _validate_packages(value: JsonValue, label: str) -> list[dict[str, JsonValue]]:
    values = _require_list(value, label)
    if not values:
        raise BackendLockError(f"{label} must not be empty")
    packages: list[dict[str, JsonValue]] = []
    for package_value in values:
        package = _require_object(package_value, f"{label} entry")
        _require_fields(package, _PACKAGE_KEYS, f"{label} entry")
        for field in ("filename", "name", "version"):
            _require_nonempty_string(package[field], f"{label} {field}")
        _require_hash(package["sha256"], f"{label} sha256")
        packages.append(package)
    names = [cast(str, package["name"]) for package in packages]
    filenames = [cast(str, package["filename"]) for package in packages]
    if len(set(names)) != len(names) or len(set(filenames)) != len(filenames):
        raise BackendLockError(f"{label} names and filenames must be unique")
    return packages


def _build_oaf_descriptor(
    backend_payload: Mapping[str, JsonValue],
    backend_sha256: str,
) -> BackendDescriptor:
    artifact_set_sha256 = _inventory_identity(backend_payload["checkpoint_components"])
    descriptor_payload: dict[str, object] = {
        "architecture_id": backend_payload["architecture_id"],
        "backend_id": backend_payload["backend_id"],
        "backend_lock_sha256": backend_sha256,
        "descriptor_schema": backend_payload["descriptor_schema"],
        "model_artifact_set_sha256": artifact_set_sha256,
        "model_id": backend_payload["model_id"],
        "native_metadata_schema_id": backend_payload["native_metadata_schema_id"],
        "native_output_space_id": backend_payload["native_output_space_id"],
        "prediction_schema": backend_payload["prediction_schema"],
        "protocol_schema": backend_payload["protocol_schema"],
        "runtime_image_manifest_digest": backend_payload["runtime_image_manifest_digest"],
        "runtime_lock_sha256": backend_payload["runtime_lock_sha256"],
        "training_data_map_id": backend_payload["training_data_map_id"],
        "upstream_source_commit": backend_payload["upstream_source_commit"],
    }
    try:
        normalized = normalize_known_backend_descriptor(descriptor_payload)
        return build_descriptor(
            normalized,
            allowed_keys=OAF_DESCRIPTOR_KEYS,
            schema="crux.transcription-backend-descriptor/v1",
        )
    except StrictJsonError as error:
        raise BackendLockError(str(error)) from None


def _inventory_identity(value: JsonValue) -> str:
    return sha256_hex(canonical_json_bytes(cast(JsonValue, value)))


def _reproduce_loaded_hash(
    payload: dict[str, JsonValue],
    expected: str,
    label: str,
) -> None:
    actual = sha256_hex(canonical_json_bytes(payload, trailing_newline=True))
    if actual != expected:
        raise BackendLockError(f"{label} content SHA-256 reproduction mismatch")


def _require_same(
    left: Mapping[str, JsonValue],
    right: Mapping[str, JsonValue],
    field: str,
    message: str,
) -> None:
    if left[field] != right[field]:
        raise BackendLockError(message)


def _require_fields(
    payload: Mapping[str, JsonValue],
    expected: Sequence[str] | set[str] | frozenset[str],
    label: str,
) -> None:
    if set(payload) != set(expected):
        raise BackendLockError(f"{label} fields must match the exact schema")


def _require_object(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise BackendLockError(f"{label} must be an object")
    return value


def _require_list(value: JsonValue, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise BackendLockError(f"{label} must be an array")
    return value


def _require_nonempty_string(value: JsonValue, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BackendLockError(f"{label} must be a nonempty string")
    return value


def _require_positive_integer(value: JsonValue, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BackendLockError(f"{label} must be a positive integer")
    return value


def _require_hash(value: JsonValue, label: str) -> str:
    if not isinstance(value, str):
        raise BackendLockError(f"{label} must be lowercase SHA-256")
    try:
        return require_sha256(value, label)
    except StrictJsonError as error:
        raise BackendLockError(str(error)) from None


def _require_digest(value: JsonValue, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise BackendLockError(f"{label} must be a sha256 digest")
    _require_hash(value[7:], label)
    return value


def _require_debian_snapshot(value: JsonValue) -> str:
    """Require a calendar-valid snapshot URL, allowing only an optional trailing slash."""
    repository = _require_nonempty_string(value, "Debian snapshot")
    match = _DEBIAN_SNAPSHOT_PATTERN.fullmatch(repository)
    if match is None:
        raise BackendLockError("Debian repository must be snapshot-addressed")
    timestamp = match.group("timestamp")
    try:
        parsed = datetime.strptime(timestamp, _DEBIAN_SNAPSHOT_TIMESTAMP_FORMAT)
    except ValueError:
        raise BackendLockError("Debian repository must be snapshot-addressed") from None
    if parsed.strftime(_DEBIAN_SNAPSHOT_TIMESTAMP_FORMAT) != timestamp:
        raise BackendLockError("Debian repository must be snapshot-addressed")
    return repository


def _require_exact_string(
    payload: Mapping[str, JsonValue],
    field: str,
    expected: str,
    label: str,
) -> None:
    if payload[field] != expected:
        raise BackendLockError(f"{label} must be {expected}")


def _validate_name_list(value: JsonValue, label: str) -> list[str]:
    values = _require_list(value, label)
    names = [_require_nonempty_string(item, label) for item in values]
    if len(set(names)) != len(names) or names != sorted(names):
        raise BackendLockError(f"{label} must be unique and lexically ordered")
    return names


def _contains_forbidden_lock_hash(value: JsonValue) -> bool:
    if isinstance(value, dict):
        return bool(set(value) & _FINAL_LOCK_HASH_KEYS) or any(
            _contains_forbidden_lock_hash(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_lock_hash(child) for child in value)
    return False


def _immutable_payload(payload: dict[str, JsonValue]) -> Mapping[str, JsonValue]:
    return cast(Mapping[str, JsonValue], _freeze(payload))


def _mutable_payload(payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], _thaw(payload))


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value
