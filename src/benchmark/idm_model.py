"""Strict Inverse Drum Machine model and inference-config identity."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from decimal import Decimal
from pathlib import Path

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    IDM_MODEL_ID_RE,
    IDM_NATIVE_METADATA_SCHEMA_ID,
    IDM_NATIVE_OUTPUT_SPACE_ID,
    IDM_RELEASE_COMMIT,
    IDM_TRAINING_DATA_MAP_ID,
    StrictJsonError,
    canonical_json_bytes,
    require_sha256,
    strict_json_loads,
)

IDM_MODEL_SCHEMA = "crux.idm-model/v1"
IDM_REQUEST_TIMEOUT_SECONDS = 1800.0
IDM_WORKER_CLOSE_TIMEOUT_SECONDS = 30.0
IDM_AUDIO_LOADER_REVISION = "soundfile-preserve-wav/v1"
IDM_VELOCITY_TO_MIDI_REVISION = "clamp-half-round-midi127/v1"
IDM_NATIVE_VELOCITY_PERSISTENCE_REVISION = "quantize-six-canonical-string/v1"
IDM_TRAIN_CLASSES = (
    "CY_CR",
    "CY_RD",
    "HH_CHH",
    "HH_OHH",
    "KD",
    "SD",
    "TT_HFT",
    "TT_HMT",
    "TT_LMT",
)

IDM_REPOSITORY_URL = "https://github.com/bernardo-torres/inverse-drum-machine"
IDM_PACKAGE_NAME = "inverse-drum-machine"
IDM_PACKAGE_VERSION = "0.1.0"
IDM_CODE_LICENSE = "Apache-2.0"
IDM_WEIGHT_LICENSE = "Apache-2.0"
IDM_WEIGHT_LICENSE_BASIS = "repository-license-no-separate-weight-notice/v1"
IDM_MODEL_NAME = "idm-44-train-kits"
IDM_MODEL_CONFIG_RELATIVE_PATH = "pretrained/idm-44-train-kits/checkpoints/model.yaml"
IDM_CHECKPOINT_RELATIVE_PATH = (
    "pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt"
)
IDM_VELOCITY_ACTIVATION = "exp_sigmoid"
IDM_VELOCITY_EXPONENT = 10.0
IDM_VELOCITY_THRESHOLD = 1e-7

IDM_INFERENCE_CONFIG_SCHEMA = "crux.idm-inference-config/v1"
IDM_ADAPTER_REVISION = "crux.idm-adapter/v1"
IDM_PREDICTION_MAP_VERSION = "crux.prediction-map/idm-44-train-kits-v1"
IDM_DEFAULT_INPUT_VIEW_ID = "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"

_PYTHON_RE = re.compile(r"3\.11\.[0-9]+\Z")


class IdmModelLockError(ValueError):
    """The IDM model lock is malformed or does not describe safe bytes."""


@dataclass(frozen=True)
class IdmModelLock:
    repository_url: str
    repository_revision: str
    package_name: str
    package_version: str
    code_license: str
    weight_license: str
    weight_license_basis: str
    runtime_lock_sha256: str
    python_version: str
    model_name: str
    model_config_relative_path: str
    model_config_sha256: str
    model_config_byte_length: int
    checkpoint_relative_path: str
    checkpoint_sha256: str
    checkpoint_byte_length: int
    model_id: str
    device: str
    dtype: str
    sample_rate_hz: int
    input_channel_count: int
    input_container: str
    input_subtype: str
    audio_loader_revision: str
    resampling: str
    mixdown: str
    mel_n_fft: int
    mel_hop_length: int
    mel_n_mels: int
    activation_rate_hz: float
    train_classes: tuple[str, ...]
    onset_activation: str
    peak_pick_div_max: int
    peak_pick_div_avg: int
    peak_pick_div_wait: int
    peak_pick_div_threshold: int
    peak_pick_normalize: bool
    velocity_activation: str
    velocity_exponent: float
    velocity_max_value: float
    velocity_threshold: float
    velocity_to_midi_revision: str
    native_velocity_persistence_revision: str
    manual_onset_override: bool
    reconstructed_stems: bool
    masking: str
    chunking_mode: str
    native_output_space_id: str
    native_metadata_schema_id: str
    training_data_map_id: str

    def __post_init__(self) -> None:
        _validate_lock(self)
        object.__setattr__(self, "activation_rate_hz", float(self.activation_rate_hz))
        object.__setattr__(self, "velocity_exponent", float(self.velocity_exponent))
        object.__setattr__(self, "velocity_max_value", float(self.velocity_max_value))
        object.__setattr__(self, "velocity_threshold", float(self.velocity_threshold))


_MODEL_KEYS = frozenset(field.name for field in fields(IdmModelLock)) | {"schema"}
_INFERENCE_CONFIG_KEYS = frozenset(
    {
        "schema",
        "backend_descriptor_sha256",
        "model_lock_sha256",
        "adapter_revision",
        "prediction_map_version",
        "input_view_id",
        "request_timeout_seconds",
    }
)


def derive_idm_model_id(lock: IdmModelLock) -> str:
    """Derive the immutable model ID from the upstream revision and checkpoint."""
    if not isinstance(lock, IdmModelLock):
        raise TypeError("lock must be an IdmModelLock")
    _require_commit(lock.repository_revision, "repository_revision")
    _require_hash(lock.checkpoint_sha256, "checkpoint_sha256")
    return f"idm-44-train-kits-{lock.repository_revision[:12]}-{lock.checkpoint_sha256[:12]}"


def load_idm_model_lock(path: Path) -> IdmModelLock:
    """Load one canonical, closed IDM model lock."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    try:
        content = read_regular_file_no_follow(path)
    except (OSError, TypeError):
        raise IdmModelLockError("model lock is unavailable") from None
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise IdmModelLockError("model lock must have one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise IdmModelLockError(str(error)) from None
    if not isinstance(value, dict) or set(value) != _MODEL_KEYS:
        raise IdmModelLockError("model lock must contain the exact key set")
    if value.get("schema") != IDM_MODEL_SCHEMA:
        raise IdmModelLockError("model lock schema is invalid")

    try:
        train_classes = value["train_classes"]
        if isinstance(train_classes, list):
            train_classes = tuple(train_classes)
        activation_rate_hz = _numeric_float(value["activation_rate_hz"], "activation_rate_hz")
        velocity_exponent = _numeric_float(value["velocity_exponent"], "velocity_exponent")
        velocity_max_value = _numeric_float(value["velocity_max_value"], "velocity_max_value")
        velocity_threshold = _numeric_float(value["velocity_threshold"], "velocity_threshold")
        return IdmModelLock(
            repository_url=value["repository_url"],
            repository_revision=value["repository_revision"],
            package_name=value["package_name"],
            package_version=value["package_version"],
            code_license=value["code_license"],
            weight_license=value["weight_license"],
            weight_license_basis=value["weight_license_basis"],
            runtime_lock_sha256=value["runtime_lock_sha256"],
            python_version=value["python_version"],
            model_name=value["model_name"],
            model_config_relative_path=value["model_config_relative_path"],
            model_config_sha256=value["model_config_sha256"],
            model_config_byte_length=value["model_config_byte_length"],
            checkpoint_relative_path=value["checkpoint_relative_path"],
            checkpoint_sha256=value["checkpoint_sha256"],
            checkpoint_byte_length=value["checkpoint_byte_length"],
            model_id=value["model_id"],
            device=value["device"],
            dtype=value["dtype"],
            sample_rate_hz=value["sample_rate_hz"],
            input_channel_count=value["input_channel_count"],
            input_container=value["input_container"],
            input_subtype=value["input_subtype"],
            audio_loader_revision=value["audio_loader_revision"],
            resampling=value["resampling"],
            mixdown=value["mixdown"],
            mel_n_fft=value["mel_n_fft"],
            mel_hop_length=value["mel_hop_length"],
            mel_n_mels=value["mel_n_mels"],
            activation_rate_hz=activation_rate_hz,
            train_classes=train_classes,
            onset_activation=value["onset_activation"],
            peak_pick_div_max=value["peak_pick_div_max"],
            peak_pick_div_avg=value["peak_pick_div_avg"],
            peak_pick_div_wait=value["peak_pick_div_wait"],
            peak_pick_div_threshold=value["peak_pick_div_threshold"],
            peak_pick_normalize=value["peak_pick_normalize"],
            velocity_activation=value["velocity_activation"],
            velocity_exponent=velocity_exponent,
            velocity_max_value=velocity_max_value,
            velocity_threshold=velocity_threshold,
            velocity_to_midi_revision=value["velocity_to_midi_revision"],
            native_velocity_persistence_revision=value["native_velocity_persistence_revision"],
            manual_onset_override=value["manual_onset_override"],
            reconstructed_stems=value["reconstructed_stems"],
            masking=value["masking"],
            chunking_mode=value["chunking_mode"],
            native_output_space_id=value["native_output_space_id"],
            native_metadata_schema_id=value["native_metadata_schema_id"],
            training_data_map_id=value["training_data_map_id"],
        )
    except IdmModelLockError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError):
        raise IdmModelLockError("model lock fields are invalid") from None


def verify_idm_model_files(lock: IdmModelLock, model_root: Path) -> tuple[Path, Path]:
    """Verify the locked model YAML/checkpoint bytes and return both paths."""
    if not isinstance(lock, IdmModelLock):
        raise TypeError("lock must be an IdmModelLock")
    if not isinstance(model_root, Path):
        raise TypeError("model_root must be a Path")
    if model_root.is_symlink() or not model_root.is_dir():
        raise IdmModelLockError("model root is unavailable")

    config_path = model_root / lock.model_config_relative_path
    checkpoint_path = model_root / lock.checkpoint_relative_path
    _verify_file(
        config_path,
        expected_sha256=lock.model_config_sha256,
        expected_length=lock.model_config_byte_length,
        field="model config",
    )
    _verify_file(
        checkpoint_path,
        expected_sha256=lock.checkpoint_sha256,
        expected_length=lock.checkpoint_byte_length,
        field="checkpoint",
    )
    return config_path, checkpoint_path


def verify_idm_runtime_lock(lock: IdmModelLock, runtime_lock_path: Path) -> None:
    """Verify the exact isolated-runtime lock bytes bound by ``lock``."""
    if not isinstance(lock, IdmModelLock):
        raise TypeError("lock must be an IdmModelLock")
    _verify_file(
        runtime_lock_path,
        expected_sha256=lock.runtime_lock_sha256,
        expected_length=None,
        field="runtime lock",
    )


def model_lock_payload(lock: IdmModelLock) -> dict[str, object]:
    """Return the canonical JSON-compatible payload used by the freeze script."""
    if not isinstance(lock, IdmModelLock):
        raise TypeError("lock must be an IdmModelLock")
    payload: dict[str, object] = {"schema": IDM_MODEL_SCHEMA}
    for field in fields(lock):
        value = getattr(lock, field.name)
        if isinstance(value, float):
            value = Decimal(str(value))
        elif isinstance(value, tuple):
            value = list(value)
        payload[field.name] = value
    canonical_json_bytes(payload)
    return payload


def idm_inference_config(
    lock_sha256: str,
    descriptor_sha256: str,
    input_view_id: str,
) -> dict[str, object]:
    """Build the thin, identity-bearing IDM inference policy."""
    _require_config_hash(lock_sha256, "model_lock_sha256")
    _require_config_hash(descriptor_sha256, "backend_descriptor_sha256")
    _require_config_nonempty_string(input_view_id, "input_view_id")
    return {
        "schema": IDM_INFERENCE_CONFIG_SCHEMA,
        "backend_descriptor_sha256": descriptor_sha256,
        "model_lock_sha256": lock_sha256,
        "adapter_revision": IDM_ADAPTER_REVISION,
        "prediction_map_version": IDM_PREDICTION_MAP_VERSION,
        "input_view_id": input_view_id,
        "request_timeout_seconds": int(IDM_REQUEST_TIMEOUT_SECONDS),
    }


def idm_inference_config_sha256(config: Mapping[str, object]) -> str:
    """Hash one exact canonical IDM inference-config payload."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a Mapping")
    if set(config) != _INFERENCE_CONFIG_KEYS:
        raise StrictJsonError("inference config must contain the exact key set")
    if config.get("schema") != IDM_INFERENCE_CONFIG_SCHEMA:
        raise StrictJsonError("inference config schema is invalid")
    _require_config_hash(config["backend_descriptor_sha256"], "backend_descriptor_sha256")
    _require_config_hash(config["model_lock_sha256"], "model_lock_sha256")
    _require_config_exact_string(
        config["adapter_revision"], "adapter_revision", IDM_ADAPTER_REVISION
    )
    _require_config_exact_string(
        config["prediction_map_version"],
        "prediction_map_version",
        IDM_PREDICTION_MAP_VERSION,
    )
    _require_config_nonempty_string(config["input_view_id"], "input_view_id")
    _require_config_exact_int(
        config["request_timeout_seconds"],
        "request_timeout_seconds",
        int(IDM_REQUEST_TIMEOUT_SECONDS),
    )
    try:
        content = canonical_json_bytes(dict(config))
    except StrictJsonError:
        raise
    return hashlib.sha256(content).hexdigest()


def _validate_lock(lock: IdmModelLock) -> None:
    _require_exact_string(lock.repository_url, "repository_url", IDM_REPOSITORY_URL)
    _require_exact_string(lock.repository_revision, "repository_revision", IDM_RELEASE_COMMIT)
    _require_exact_string(lock.package_name, "package_name", IDM_PACKAGE_NAME)
    _require_exact_string(lock.package_version, "package_version", IDM_PACKAGE_VERSION)
    _require_exact_string(lock.code_license, "code_license", IDM_CODE_LICENSE)
    _require_exact_string(lock.weight_license, "weight_license", IDM_WEIGHT_LICENSE)
    _require_exact_string(
        lock.weight_license_basis, "weight_license_basis", IDM_WEIGHT_LICENSE_BASIS
    )
    _require_hash(lock.runtime_lock_sha256, "runtime_lock_sha256")
    if (
        not isinstance(lock.python_version, str)
        or _PYTHON_RE.fullmatch(lock.python_version) is None
    ):
        raise IdmModelLockError("python_version is invalid")
    _require_exact_string(lock.model_name, "model_name", IDM_MODEL_NAME)
    _require_exact_string(
        lock.model_config_relative_path,
        "model_config_relative_path",
        IDM_MODEL_CONFIG_RELATIVE_PATH,
    )
    _require_hash(lock.model_config_sha256, "model_config_sha256")
    _require_positive_int(lock.model_config_byte_length, "model_config_byte_length")
    _require_exact_string(
        lock.checkpoint_relative_path,
        "checkpoint_relative_path",
        IDM_CHECKPOINT_RELATIVE_PATH,
    )
    _require_hash(lock.checkpoint_sha256, "checkpoint_sha256")
    _require_positive_int(lock.checkpoint_byte_length, "checkpoint_byte_length")
    if not isinstance(lock.model_id, str) or IDM_MODEL_ID_RE.fullmatch(lock.model_id) is None:
        raise IdmModelLockError("model_id is invalid")
    if lock.model_id != _derive_model_id(lock):
        raise IdmModelLockError("model_id does not match the locked digest fragments")
    _require_exact_string(lock.device, "device", "cpu")
    _require_exact_string(lock.dtype, "dtype", "float32")
    _require_exact_int(lock.sample_rate_hz, "sample_rate_hz", 44100)
    _require_exact_int(lock.input_channel_count, "input_channel_count", 1)
    _require_exact_string(lock.input_container, "input_container", "WAV")
    _require_exact_string(lock.input_subtype, "input_subtype", "PCM_16")
    _require_exact_string(
        lock.audio_loader_revision,
        "audio_loader_revision",
        IDM_AUDIO_LOADER_REVISION,
    )
    _require_exact_string(lock.resampling, "resampling", "forbidden")
    _require_exact_string(lock.mixdown, "mixdown", "forbidden")
    _require_exact_int(lock.mel_n_fft, "mel_n_fft", 1024)
    _require_exact_int(lock.mel_hop_length, "mel_hop_length", 256)
    _require_exact_int(lock.mel_n_mels, "mel_n_mels", 128)
    _require_exact_number(lock.activation_rate_hz, "activation_rate_hz", 44100 / 256)
    if lock.train_classes != IDM_TRAIN_CLASSES:
        raise IdmModelLockError("train_classes do not match the frozen ordering")
    _require_exact_string(
        lock.onset_activation,
        "onset_activation",
        "sigmoid-over-native-logits-before-upstream-peak-picking",
    )
    _require_exact_int(lock.peak_pick_div_max, "peak_pick_div_max", 20)
    _require_exact_int(lock.peak_pick_div_avg, "peak_pick_div_avg", 10)
    _require_exact_int(lock.peak_pick_div_wait, "peak_pick_div_wait", 16)
    _require_exact_int(lock.peak_pick_div_threshold, "peak_pick_div_threshold", 5)
    _require_exact_bool(lock.peak_pick_normalize, "peak_pick_normalize", False)
    _require_exact_string(lock.velocity_activation, "velocity_activation", IDM_VELOCITY_ACTIVATION)
    _require_exact_number(lock.velocity_exponent, "velocity_exponent", IDM_VELOCITY_EXPONENT)
    _require_exact_number(lock.velocity_max_value, "velocity_max_value", 2.0)
    _require_exact_number(lock.velocity_threshold, "velocity_threshold", IDM_VELOCITY_THRESHOLD)
    _require_exact_string(
        lock.velocity_to_midi_revision,
        "velocity_to_midi_revision",
        IDM_VELOCITY_TO_MIDI_REVISION,
    )
    _require_exact_string(
        lock.native_velocity_persistence_revision,
        "native_velocity_persistence_revision",
        IDM_NATIVE_VELOCITY_PERSISTENCE_REVISION,
    )
    _require_exact_bool(lock.manual_onset_override, "manual_onset_override", False)
    _require_exact_bool(lock.reconstructed_stems, "reconstructed_stems", False)
    _require_exact_string(lock.masking, "masking", "none")
    _require_exact_string(lock.chunking_mode, "chunking_mode", "none")
    _require_exact_string(
        lock.native_output_space_id,
        "native_output_space_id",
        IDM_NATIVE_OUTPUT_SPACE_ID,
    )
    _require_exact_string(
        lock.native_metadata_schema_id,
        "native_metadata_schema_id",
        IDM_NATIVE_METADATA_SCHEMA_ID,
    )
    _require_exact_string(
        lock.training_data_map_id,
        "training_data_map_id",
        IDM_TRAINING_DATA_MAP_ID,
    )


def _derive_model_id(lock: IdmModelLock) -> str:
    return derive_idm_model_id(lock)


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_length: int | None,
    field: str,
) -> None:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    try:
        content = read_regular_file_no_follow(path)
    except (OSError, TypeError):
        raise IdmModelLockError(f"{field} file is unavailable") from None
    if expected_length is not None and len(content) != expected_length:
        raise IdmModelLockError(f"{field} file byte length differs")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise IdmModelLockError(f"{field} file SHA-256 differs")


def _numeric_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise IdmModelLockError(f"{field} is invalid")
    converted = float(value)
    if not math.isfinite(converted):
        raise IdmModelLockError(f"{field} is invalid")
    return converted


def _require_hash(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise IdmModelLockError(f"{field} is invalid")
    try:
        require_sha256(value, field)
    except StrictJsonError:
        raise IdmModelLockError(f"{field} is invalid") from None


def _require_commit(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IdmModelLockError(f"{field} is invalid")


def _require_exact_string(value: object, field: str, expected: str) -> None:
    if value != expected:
        raise IdmModelLockError(f"{field} is invalid")


def _require_config_hash(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise StrictJsonError(f"{field} must be lowercase SHA-256")
    try:
        return require_sha256(value, field)
    except StrictJsonError:
        raise StrictJsonError(f"{field} must be lowercase SHA-256") from None


def _require_config_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise StrictJsonError(f"{field} must be a nonempty string")
    return value


def _require_config_exact_string(value: object, field: str, expected: str) -> str:
    if value != expected:
        raise StrictJsonError(f"{field} is invalid")
    return expected


def _require_config_exact_int(value: object, field: str, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise StrictJsonError(f"{field} is invalid")
    return value


def _require_positive_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IdmModelLockError(f"{field} is invalid")


def _require_exact_int(value: object, field: str, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise IdmModelLockError(f"{field} is invalid")


def _require_exact_bool(value: object, field: str, expected: bool) -> None:
    if not isinstance(value, bool) or value is not expected:
        raise IdmModelLockError(f"{field} is invalid")


def _require_exact_number(value: object, field: str, expected: float) -> None:
    if _numeric_float(value, field) != expected:
        raise IdmModelLockError(f"{field} is invalid")


__all__ = [
    "IDM_ADAPTER_REVISION",
    "IDM_AUDIO_LOADER_REVISION",
    "IDM_CHECKPOINT_RELATIVE_PATH",
    "IDM_CODE_LICENSE",
    "IDM_DEFAULT_INPUT_VIEW_ID",
    "IDM_INFERENCE_CONFIG_SCHEMA",
    "IDM_MODEL_CONFIG_RELATIVE_PATH",
    "IDM_MODEL_ID_RE",
    "IDM_MODEL_NAME",
    "IDM_MODEL_SCHEMA",
    "IDM_NATIVE_METADATA_SCHEMA_ID",
    "IDM_NATIVE_OUTPUT_SPACE_ID",
    "IDM_NATIVE_VELOCITY_PERSISTENCE_REVISION",
    "IDM_PACKAGE_NAME",
    "IDM_PACKAGE_VERSION",
    "IDM_PREDICTION_MAP_VERSION",
    "IDM_RELEASE_COMMIT",
    "IDM_REQUEST_TIMEOUT_SECONDS",
    "IDM_REPOSITORY_URL",
    "IDM_TRAINING_DATA_MAP_ID",
    "IDM_TRAIN_CLASSES",
    "IDM_VELOCITY_ACTIVATION",
    "IDM_VELOCITY_EXPONENT",
    "IDM_VELOCITY_THRESHOLD",
    "IDM_VELOCITY_TO_MIDI_REVISION",
    "IDM_WEIGHT_LICENSE",
    "IDM_WEIGHT_LICENSE_BASIS",
    "IdmModelLock",
    "IdmModelLockError",
    "derive_idm_model_id",
    "idm_inference_config",
    "idm_inference_config_sha256",
    "load_idm_model_lock",
    "model_lock_payload",
    "verify_idm_model_files",
    "verify_idm_runtime_lock",
]
