"""Strict MuScriptor model identity and checkpoint verification."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, fields
from decimal import Decimal
from pathlib import Path
from typing import Literal

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    StrictJsonError,
    canonical_json_bytes,
    require_sha256,
    strict_json_loads,
)

MUSCRIPTOR_MODEL_SCHEMA = "crux.muscriptor-model/v1"
MUSCRIPTOR_RELEASE_COMMIT = "d73147e75e5b9b0c0a79ebe154587db4fd603e0c"
MUSCRIPTOR_MODEL_ID_RE = re.compile(r"muscriptor-(medium|small)-[0-9a-f]{12}-[0-9a-f]{12}\Z")
MUSCRIPTOR_WEIGHT_LICENSE = "CC BY-NC 4.0"
MUSCRIPTOR_NATIVE_OUTPUT_SPACE_ID = "muscriptor-drums-midi128-v1"
MUSCRIPTOR_NATIVE_METADATA_SCHEMA_ID = "muscriptor-note-start-metadata-v1"
MUSCRIPTOR_TRAINING_DATA_MAP_ID = "muscriptor-training-data-v0.3.0"

_CHECKPOINT_VARIANTS = frozenset({"medium", "small"})
_DEVICE_RE = re.compile(r"(?:cpu|mps|cuda(?::[0-9]+)?)\Z")
_DTYPES = frozenset({"float16", "float32", "bfloat16"})


class MuscriptorModelLockError(ValueError):
    """The MuScriptor model lock is malformed or does not describe safe bytes."""


@dataclass(frozen=True)
class MuscriptorModelLock:
    package_name: str
    package_version: str
    upstream_source_commit: str
    code_license: str
    weight_license: str
    checkpoint_variant: Literal["medium", "small"]
    checkpoint_repo_id: str
    checkpoint_revision: str
    checkpoint_filename: str
    checkpoint_sha256: str
    checkpoint_byte_length: int
    checkpoint_config_filename: str
    checkpoint_config_sha256: str
    checkpoint_config_byte_length: int
    model_id: str
    device: str
    dtype: str
    input_sample_rate_hz: int
    chunk_duration_sec: float
    use_sampling: bool
    temperature: float
    cfg_coef: float
    instruments: tuple[str, ...]
    batch_size: int
    no_eos_is_ok: bool
    beam_size: int
    prelude_forcing: bool
    native_output_space_id: str
    native_metadata_schema_id: str
    training_data_map_id: str

    def __post_init__(self) -> None:
        _validate_lock(self)
        object.__setattr__(self, "chunk_duration_sec", float(self.chunk_duration_sec))
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "cfg_coef", float(self.cfg_coef))


_MODEL_KEYS = frozenset(field.name for field in fields(MuscriptorModelLock)) | {"schema"}


def derive_muscriptor_model_id(lock: MuscriptorModelLock) -> str:
    """Derive the immutable model ID from the selected revision and weights."""
    if not isinstance(lock, MuscriptorModelLock):
        raise TypeError("lock must be a MuscriptorModelLock")
    _require_variant(lock.checkpoint_variant)
    _require_hex(lock.checkpoint_revision, 40, "checkpoint_revision")
    require_sha256(lock.checkpoint_sha256, "checkpoint_sha256")
    return (
        f"muscriptor-{lock.checkpoint_variant}-"
        f"{lock.checkpoint_revision[:12]}-{lock.checkpoint_sha256[:12]}"
    )


def load_muscriptor_model_lock(path: Path) -> MuscriptorModelLock:
    """Load one canonical, closed MuScriptor model lock."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    try:
        content = read_regular_file_no_follow(path)
    except (OSError, TypeError):
        raise MuscriptorModelLockError("model lock is unavailable") from None
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise MuscriptorModelLockError("model lock must have one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise MuscriptorModelLockError(str(error)) from None
    if not isinstance(value, dict) or set(value) != _MODEL_KEYS:
        raise MuscriptorModelLockError("model lock must contain the exact key set")
    if value.get("schema") != MUSCRIPTOR_MODEL_SCHEMA:
        raise MuscriptorModelLockError("model lock schema is invalid")

    try:
        instruments = value["instruments"]
        if isinstance(instruments, list):
            instruments = tuple(instruments)
        chunk_duration_sec = _numeric_float(value["chunk_duration_sec"], "chunk_duration_sec")
        temperature = _numeric_float(value["temperature"], "temperature")
        cfg_coef = _numeric_float(value["cfg_coef"], "cfg_coef")
        return MuscriptorModelLock(
            package_name=value["package_name"],
            package_version=value["package_version"],
            upstream_source_commit=value["upstream_source_commit"],
            code_license=value["code_license"],
            weight_license=value["weight_license"],
            checkpoint_variant=value["checkpoint_variant"],
            checkpoint_repo_id=value["checkpoint_repo_id"],
            checkpoint_revision=value["checkpoint_revision"],
            checkpoint_filename=value["checkpoint_filename"],
            checkpoint_sha256=value["checkpoint_sha256"],
            checkpoint_byte_length=value["checkpoint_byte_length"],
            checkpoint_config_filename=value["checkpoint_config_filename"],
            checkpoint_config_sha256=value["checkpoint_config_sha256"],
            checkpoint_config_byte_length=value["checkpoint_config_byte_length"],
            model_id=value["model_id"],
            device=value["device"],
            dtype=value["dtype"],
            input_sample_rate_hz=value["input_sample_rate_hz"],
            chunk_duration_sec=chunk_duration_sec,
            use_sampling=value["use_sampling"],
            temperature=temperature,
            cfg_coef=cfg_coef,
            instruments=instruments,
            batch_size=value["batch_size"],
            no_eos_is_ok=value["no_eos_is_ok"],
            beam_size=value["beam_size"],
            prelude_forcing=value["prelude_forcing"],
            native_output_space_id=value["native_output_space_id"],
            native_metadata_schema_id=value["native_metadata_schema_id"],
            training_data_map_id=value["training_data_map_id"],
        )
    except MuscriptorModelLockError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError):
        raise MuscriptorModelLockError("model lock fields are invalid") from None


def verify_muscriptor_checkpoint(lock: MuscriptorModelLock, checkpoint_dir: Path) -> Path:
    """Verify the locked safetensors/config bytes and return the weights path."""
    if not isinstance(lock, MuscriptorModelLock):
        raise TypeError("lock must be a MuscriptorModelLock")
    if not isinstance(checkpoint_dir, Path):
        raise TypeError("checkpoint_dir must be a Path")
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise MuscriptorModelLockError("checkpoint directory is unavailable")

    checkpoint_path = checkpoint_dir / lock.checkpoint_filename
    config_path = checkpoint_dir / lock.checkpoint_config_filename
    _verify_file(
        checkpoint_path,
        expected_sha256=lock.checkpoint_sha256,
        expected_length=lock.checkpoint_byte_length,
        field="checkpoint",
    )
    _verify_file(
        config_path,
        expected_sha256=lock.checkpoint_config_sha256,
        expected_length=lock.checkpoint_config_byte_length,
        field="checkpoint config",
    )
    return checkpoint_path


def model_lock_payload(lock: MuscriptorModelLock) -> dict[str, object]:
    """Return the canonical JSON-compatible payload used by the freeze script."""
    if not isinstance(lock, MuscriptorModelLock):
        raise TypeError("lock must be a MuscriptorModelLock")
    payload: dict[str, object] = {"schema": MUSCRIPTOR_MODEL_SCHEMA}
    for field in fields(lock):
        value = getattr(lock, field.name)
        if isinstance(value, float):
            value = Decimal(str(value))
        elif isinstance(value, tuple):
            value = list(value)
        payload[field.name] = value
    # Exercise the same renderer used for publication before returning the value.
    canonical_json_bytes(payload)
    return payload


def _validate_lock(lock: MuscriptorModelLock) -> None:
    _require_exact_string(lock.package_name, "package_name", "muscriptor")
    _require_exact_string(lock.package_version, "package_version", "0.3.0")
    _require_exact_string(
        lock.upstream_source_commit,
        "upstream_source_commit",
        MUSCRIPTOR_RELEASE_COMMIT,
    )
    _require_exact_string(lock.code_license, "code_license", "MIT")
    _require_exact_string(lock.weight_license, "weight_license", MUSCRIPTOR_WEIGHT_LICENSE)
    _require_variant(lock.checkpoint_variant)
    expected_repo = f"MuScriptor/muscriptor-{lock.checkpoint_variant}"
    _require_exact_string(lock.checkpoint_repo_id, "checkpoint_repo_id", expected_repo)
    _require_hex(lock.checkpoint_revision, 40, "checkpoint_revision")
    _require_exact_string(lock.checkpoint_filename, "checkpoint_filename", "model.safetensors")
    _require_hash(lock.checkpoint_sha256, "checkpoint_sha256")
    _require_positive_int(lock.checkpoint_byte_length, "checkpoint_byte_length")
    _require_exact_string(
        lock.checkpoint_config_filename, "checkpoint_config_filename", "config.json"
    )
    _require_hash(lock.checkpoint_config_sha256, "checkpoint_config_sha256")
    _require_positive_int(lock.checkpoint_config_byte_length, "checkpoint_config_byte_length")
    if (
        not isinstance(lock.model_id, str)
        or MUSCRIPTOR_MODEL_ID_RE.fullmatch(lock.model_id) is None
    ):
        raise MuscriptorModelLockError("model_id is invalid")
    if lock.model_id != _derive_model_id(lock):
        raise MuscriptorModelLockError("model_id does not match the locked digest fragments")
    if not isinstance(lock.device, str) or _DEVICE_RE.fullmatch(lock.device) is None:
        raise MuscriptorModelLockError("device is invalid")
    if not isinstance(lock.dtype, str) or lock.dtype not in _DTYPES:
        raise MuscriptorModelLockError("dtype is invalid")
    _require_exact_int(lock.input_sample_rate_hz, "input_sample_rate_hz", 16000)
    _require_exact_number(lock.chunk_duration_sec, "chunk_duration_sec", 5.0)
    _require_exact_bool(lock.use_sampling, "use_sampling", False)
    _require_exact_number(lock.temperature, "temperature", 1.0)
    _require_exact_number(lock.cfg_coef, "cfg_coef", 1.0)
    if lock.instruments != ("drums",):
        raise MuscriptorModelLockError("instruments must be exactly ('drums',)")
    _require_exact_int(lock.batch_size, "batch_size", 1)
    _require_exact_bool(lock.no_eos_is_ok, "no_eos_is_ok", True)
    _require_exact_int(lock.beam_size, "beam_size", 1)
    _require_exact_bool(lock.prelude_forcing, "prelude_forcing", True)
    _require_exact_string(
        lock.native_output_space_id,
        "native_output_space_id",
        MUSCRIPTOR_NATIVE_OUTPUT_SPACE_ID,
    )
    _require_exact_string(
        lock.native_metadata_schema_id,
        "native_metadata_schema_id",
        MUSCRIPTOR_NATIVE_METADATA_SCHEMA_ID,
    )
    if not isinstance(lock.training_data_map_id, str) or not lock.training_data_map_id:
        raise MuscriptorModelLockError("training_data_map_id is invalid")


def _derive_model_id(lock: MuscriptorModelLock) -> str:
    return (
        f"muscriptor-{lock.checkpoint_variant}-"
        f"{lock.checkpoint_revision[:12]}-{lock.checkpoint_sha256[:12]}"
    )


def _numeric_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise MuscriptorModelLockError(f"{field} is invalid")
    converted = float(value)
    if not math.isfinite(converted):
        raise MuscriptorModelLockError(f"{field} is invalid")
    return converted


def _verify_file(path: Path, *, expected_sha256: str, expected_length: int, field: str) -> None:
    try:
        content = read_regular_file_no_follow(path)
    except (OSError, TypeError):
        raise MuscriptorModelLockError(f"{field} file is unavailable") from None
    if len(content) != expected_length:
        raise MuscriptorModelLockError(f"{field} file byte length differs")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise MuscriptorModelLockError(f"{field} file SHA-256 differs")


def _require_hash(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise MuscriptorModelLockError(f"{field} is invalid")
    try:
        require_sha256(value, field)
    except StrictJsonError:
        raise MuscriptorModelLockError(f"{field} is invalid") from None


def _require_hex(value: object, length: int, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MuscriptorModelLockError(f"{field} is invalid")


def _require_variant(value: object) -> None:
    if value not in _CHECKPOINT_VARIANTS:
        raise MuscriptorModelLockError("checkpoint_variant is invalid")


def _require_exact_string(value: object, field: str, expected: str) -> None:
    if value != expected:
        raise MuscriptorModelLockError(f"{field} is invalid")


def _require_positive_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MuscriptorModelLockError(f"{field} is invalid")


def _require_exact_int(value: object, field: str, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise MuscriptorModelLockError(f"{field} is invalid")


def _require_exact_bool(value: object, field: str, expected: bool) -> None:
    if not isinstance(value, bool) or value is not expected:
        raise MuscriptorModelLockError(f"{field} is invalid")


def _require_exact_number(value: object, field: str, expected: float) -> None:
    if _numeric_float(value, field) != expected:
        raise MuscriptorModelLockError(f"{field} is invalid")


__all__ = [
    "MUSCRIPTOR_MODEL_ID_RE",
    "MUSCRIPTOR_MODEL_SCHEMA",
    "MUSCRIPTOR_NATIVE_METADATA_SCHEMA_ID",
    "MUSCRIPTOR_NATIVE_OUTPUT_SPACE_ID",
    "MUSCRIPTOR_RELEASE_COMMIT",
    "MUSCRIPTOR_TRAINING_DATA_MAP_ID",
    "MUSCRIPTOR_WEIGHT_LICENSE",
    "MuscriptorModelLock",
    "MuscriptorModelLockError",
    "derive_muscriptor_model_id",
    "load_muscriptor_model_lock",
    "model_lock_payload",
    "verify_muscriptor_checkpoint",
]
