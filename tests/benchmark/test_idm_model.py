from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from src.benchmark.backend_identity import StrictJsonError, canonical_json_bytes
from src.benchmark.idm_model import (
    IDM_AUDIO_LOADER_REVISION,
    IDM_MODEL_SCHEMA,
    IDM_NATIVE_VELOCITY_PERSISTENCE_REVISION,
    IDM_RELEASE_COMMIT,
    IDM_TRAIN_CLASSES,
    IDM_VELOCITY_ACTIVATION,
    IDM_VELOCITY_EXPONENT,
    IDM_VELOCITY_THRESHOLD,
    IDM_VELOCITY_TO_MIDI_REVISION,
    IDM_WEIGHT_LICENSE_BASIS,
    IdmModelLockError,
    derive_idm_model_id,
    idm_inference_config,
    idm_inference_config_sha256,
    load_idm_model_lock,
    model_lock_payload,
    verify_idm_model_files,
    verify_idm_runtime_lock,
)

MODEL_CONFIG = b"sample_rate: 44100\n"
CHECKPOINT = b"synthetic-idm-checkpoint"
RUNTIME_LOCK = b"synthetic-idm-runtime-lock"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _lock_payload(
    *,
    config: bytes = MODEL_CONFIG,
    checkpoint: bytes = CHECKPOINT,
    **overrides: object,
) -> dict[str, object]:
    checkpoint_sha = hashlib.sha256(checkpoint).hexdigest()
    config_sha = hashlib.sha256(config).hexdigest()
    payload: dict[str, object] = {
        "schema": IDM_MODEL_SCHEMA,
        "repository_url": "https://github.com/bernardo-torres/inverse-drum-machine",
        "repository_revision": IDM_RELEASE_COMMIT,
        "package_name": "inverse-drum-machine",
        "package_version": "0.1.0",
        "code_license": "Apache-2.0",
        "weight_license": "Apache-2.0",
        "weight_license_basis": IDM_WEIGHT_LICENSE_BASIS,
        "runtime_lock_sha256": hashlib.sha256(RUNTIME_LOCK).hexdigest(),
        "python_version": "3.11.12",
        "model_name": "idm-44-train-kits",
        "model_config_relative_path": "pretrained/idm-44-train-kits/checkpoints/model.yaml",
        "model_config_sha256": config_sha,
        "model_config_byte_length": len(config),
        "checkpoint_relative_path": (
            "pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt"
        ),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_byte_length": len(checkpoint),
        "model_id": f"idm-44-train-kits-{IDM_RELEASE_COMMIT[:12]}-{checkpoint_sha[:12]}",
        "device": "cpu",
        "dtype": "float32",
        "sample_rate_hz": 44100,
        "input_channel_count": 1,
        "input_container": "WAV",
        "input_subtype": "PCM_16",
        "audio_loader_revision": IDM_AUDIO_LOADER_REVISION,
        "resampling": "forbidden",
        "mixdown": "forbidden",
        "mel_n_fft": 1024,
        "mel_hop_length": 256,
        "mel_n_mels": 128,
        "activation_rate_hz": Decimal("172.265625"),
        "train_classes": list(IDM_TRAIN_CLASSES),
        "onset_activation": "sigmoid-over-native-logits-before-upstream-peak-picking",
        "peak_pick_div_max": 20,
        "peak_pick_div_avg": 10,
        "peak_pick_div_wait": 16,
        "peak_pick_div_threshold": 5,
        "peak_pick_normalize": False,
        "velocity_activation": IDM_VELOCITY_ACTIVATION,
        "velocity_exponent": Decimal("10"),
        "velocity_max_value": Decimal("2.0"),
        "velocity_threshold": Decimal("0.0000001"),
        "velocity_to_midi_revision": IDM_VELOCITY_TO_MIDI_REVISION,
        "native_velocity_persistence_revision": IDM_NATIVE_VELOCITY_PERSISTENCE_REVISION,
        "manual_onset_override": False,
        "reconstructed_stems": False,
        "masking": "none",
        "chunking_mode": "none",
        "native_output_space_id": "idm-44-train-kits-9class-v1",
        "native_metadata_schema_id": "idm-peak-event-metadata-v1",
        "training_data_map_id": "idm-training-contract-44-train-kits-v1",
    }
    payload.update(overrides)
    return payload


def _write_lock(path: Path, **overrides: object) -> tuple[Path, Path, Path]:
    config = overrides.pop("config", MODEL_CONFIG)
    checkpoint = overrides.pop("checkpoint", CHECKPOINT)
    assert isinstance(config, bytes)
    assert isinstance(checkpoint, bytes)
    model_root = path.parent / "model-root"
    config_path = model_root / "pretrained/idm-44-train-kits/checkpoints/model.yaml"
    checkpoint_path = (
        model_root / "pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(config)
    checkpoint_path.write_bytes(checkpoint)
    (path.parent / "runtime.lock").write_bytes(RUNTIME_LOCK)
    lock = path
    lock.write_bytes(
        canonical_json_bytes(
            _lock_payload(config=config, checkpoint=checkpoint, **overrides), trailing_newline=True
        )
    )
    return lock, config_path, checkpoint_path


def test_loads_the_frozen_identity_and_verifies_exact_model_files(tmp_path: Path) -> None:
    lock_path, config_path, checkpoint_path = _write_lock(tmp_path / "model.json")

    lock = load_idm_model_lock(lock_path)

    assert lock.repository_revision == IDM_RELEASE_COMMIT
    assert lock.python_version.startswith("3.11.")
    assert lock.model_name == "idm-44-train-kits"
    assert lock.code_license == "Apache-2.0"
    assert lock.weight_license == "Apache-2.0"
    assert lock.weight_license_basis == IDM_WEIGHT_LICENSE_BASIS
    assert lock.device == "cpu"
    assert lock.dtype == "float32"
    assert lock.sample_rate_hz == 44100
    assert lock.input_channel_count == 1
    assert lock.input_container == "WAV"
    assert lock.input_subtype == "PCM_16"
    assert lock.audio_loader_revision == IDM_AUDIO_LOADER_REVISION
    assert lock.resampling == "forbidden"
    assert lock.mixdown == "forbidden"
    assert lock.train_classes == IDM_TRAIN_CLASSES
    assert lock.peak_pick_div_max == 20
    assert lock.peak_pick_div_avg == 10
    assert lock.peak_pick_div_wait == 16
    assert lock.peak_pick_div_threshold == 5
    assert lock.peak_pick_normalize is False
    assert lock.velocity_activation == IDM_VELOCITY_ACTIVATION == "exp_sigmoid"
    assert lock.velocity_exponent == IDM_VELOCITY_EXPONENT == 10
    assert lock.velocity_max_value == 2
    assert lock.velocity_threshold == IDM_VELOCITY_THRESHOLD == 1e-7
    assert lock.velocity_to_midi_revision == IDM_VELOCITY_TO_MIDI_REVISION
    assert lock.native_velocity_persistence_revision == IDM_NATIVE_VELOCITY_PERSISTENCE_REVISION
    assert lock.manual_onset_override is False
    assert lock.reconstructed_stems is False
    assert lock.masking == "none"
    assert lock.chunking_mode == "none"
    assert lock.model_id == derive_idm_model_id(lock)
    assert verify_idm_model_files(lock, tmp_path / "model-root") == (config_path, checkpoint_path)
    verify_idm_runtime_lock(lock, tmp_path / "runtime.lock")


@pytest.mark.parametrize(
    "field,value",
    [
        ("code_license", "MIT"),
        ("weight_license", "unknown"),
        ("weight_license_basis", "repository-license-separate-notice/v1"),
        ("audio_loader_revision", "librosa-default"),
        ("resampling", "allowed"),
        ("mixdown", "allowed"),
        ("peak_pick_div_max", 19),
        ("peak_pick_div_avg", 11),
        ("peak_pick_div_wait", 15),
        ("peak_pick_div_threshold", 6),
        ("peak_pick_normalize", True),
        ("device", "tpu"),
        ("device", "mps"),
        ("dtype", "float64"),
        ("dtype", "float16"),
        ("velocity_activation", "sigmoid(exponent=10)"),
        ("velocity_exponent", Decimal("11")),
        ("velocity_threshold", Decimal("0.000001")),
    ],
)
def test_rejects_edited_identity_facts(tmp_path: Path, field: str, value: object) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json", **{field: value})

    with pytest.raises(IdmModelLockError):
        load_idm_model_lock(lock_path)


def test_rejects_reordered_classes(tmp_path: Path) -> None:
    classes = list(IDM_TRAIN_CLASSES)
    classes.reverse()
    lock_path, _, _ = _write_lock(tmp_path / "model.json", train_classes=classes)

    with pytest.raises(IdmModelLockError):
        load_idm_model_lock(lock_path)


def test_rejects_edited_or_missing_model_bytes(tmp_path: Path) -> None:
    lock_path, config_path, checkpoint_path = _write_lock(tmp_path / "model.json")
    lock = load_idm_model_lock(lock_path)
    runtime_path = tmp_path / "runtime.lock"
    runtime_path.write_bytes(b"edited")
    with pytest.raises(IdmModelLockError):
        verify_idm_runtime_lock(lock, runtime_path)

    checkpoint_path.write_bytes(b"edited")
    with pytest.raises(IdmModelLockError):
        verify_idm_model_files(lock, tmp_path / "model-root")

    checkpoint_path.unlink()
    with pytest.raises(IdmModelLockError):
        verify_idm_model_files(lock, tmp_path / "model-root")

    config_path.unlink()
    with pytest.raises(IdmModelLockError):
        verify_idm_model_files(lock, tmp_path / "model-root")


def test_rejects_noncanonical_lock_json(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json")
    value = json.loads(lock_path.read_text())
    lock_path.write_text(json.dumps(value, indent=2) + "\n")

    with pytest.raises(IdmModelLockError, match="canonical"):
        load_idm_model_lock(lock_path)


def test_optimized_freeze_validation_rejects_invalid_model_fact() -> None:
    script = Path(__file__).parents[2] / "src" / "cli" / "freeze_idm_model.py"
    code = """
from src.cli.freeze_idm_model import FreezeError, _verify_model_config

content = b'''
sampling_rate: 22050
encoder:
  sampling_rate: 44100
  transform:
    sample_rate: 44100
    n_fft: 1024
    hop_length: 256
    n_mels: 128
  transcription_head:
    onset_activation: none
    velocity_activation: exp_sigmoid
decoder:
  sampling_rate: 44100
train_classes: [CY_CR, CY_RD, HH_CHH, HH_OHH, KD, SD, TT_HFT, TT_HMT, TT_LMT]
'''

try:
    _verify_model_config(content)
except FreezeError as error:
    if str(error) != "model config facts differ from the frozen IDM contract":
        raise SystemExit(3)
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=script.parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


def test_inference_config_is_thin_and_hashes_canonical_bytes() -> None:
    config = idm_inference_config(
        SHA_A,
        SHA_B,
        "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
    )

    assert set(config) == {
        "schema",
        "backend_descriptor_sha256",
        "model_lock_sha256",
        "adapter_revision",
        "prediction_map_version",
        "input_view_id",
        "request_timeout_seconds",
    }
    assert config["request_timeout_seconds"] == 1800
    assert idm_inference_config_sha256(config) == idm_inference_config_sha256(dict(config))
    assert (
        idm_inference_config_sha256(config)
        == hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_timeout_seconds", 0),
        ("request_timeout_seconds", 1800.0),
        ("request_timeout_seconds", 1801),
        ("model_lock_sha256", "not-a-hash"),
        ("backend_descriptor_sha256", "not-a-hash"),
        ("input_view_id", ""),
        ("adapter_revision", "wrong/v1"),
    ],
)
def test_inference_config_hash_rejects_invalid_payload(field: str, value: object) -> None:
    config = idm_inference_config(SHA_A, SHA_B, "crux.input/v1")
    config[field] = value

    with pytest.raises(ValueError):
        idm_inference_config_sha256(config)


def test_derive_idm_model_id_rejects_non_lock() -> None:
    with pytest.raises(TypeError, match="lock must be an IdmModelLock"):
        derive_idm_model_id("not-a-lock")  # type: ignore[arg-type]


def test_load_idm_model_lock_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="path must be a Path"):
        load_idm_model_lock("not-a-path")  # type: ignore[arg-type]


def test_load_idm_model_lock_rejects_unavailable_file(tmp_path: Path) -> None:
    with pytest.raises(IdmModelLockError, match="model lock is unavailable"):
        load_idm_model_lock(tmp_path / "missing.json")


@pytest.mark.parametrize("content", [b'{"schema": "x"}', b'{"schema": "x"}\n\n'])
def test_load_idm_model_lock_rejects_non_canonical_newlines(tmp_path: Path, content: bytes) -> None:
    lock_path = tmp_path / "model.json"
    lock_path.write_bytes(content)

    with pytest.raises(IdmModelLockError, match="one final newline"):
        load_idm_model_lock(lock_path)


def test_load_idm_model_lock_rejects_wrong_key_set(tmp_path: Path) -> None:
    payload = _lock_payload()
    del payload["device"]
    lock_path = tmp_path / "model.json"
    lock_path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))

    with pytest.raises(IdmModelLockError, match="exact key set"):
        load_idm_model_lock(lock_path)


def test_load_idm_model_lock_rejects_wrong_schema(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json", schema="crux.wrong/v1")

    with pytest.raises(IdmModelLockError, match="schema is invalid"):
        load_idm_model_lock(lock_path)


def test_load_idm_model_lock_wraps_field_overflow_as_lock_error(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json", velocity_exponent=10**999)

    with pytest.raises(IdmModelLockError, match="model lock fields are invalid"):
        load_idm_model_lock(lock_path)


def test_verify_idm_model_files_rejects_non_lock(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="lock must be an IdmModelLock"):
        verify_idm_model_files("not-a-lock", tmp_path)  # type: ignore[arg-type]


def test_verify_idm_model_files_rejects_non_path_root(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json")
    lock = load_idm_model_lock(lock_path)

    with pytest.raises(TypeError, match="model_root must be a Path"):
        verify_idm_model_files(lock, "not-a-path")  # type: ignore[arg-type]


def test_verify_idm_model_files_rejects_missing_model_root(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json")
    lock = load_idm_model_lock(lock_path)

    with pytest.raises(IdmModelLockError, match="model root is unavailable"):
        verify_idm_model_files(lock, tmp_path / "does-not-exist")


def test_verify_idm_runtime_lock_rejects_non_lock(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="lock must be an IdmModelLock"):
        verify_idm_runtime_lock("not-a-lock", tmp_path)  # type: ignore[arg-type]


def test_verify_idm_runtime_lock_rejects_non_path(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json")
    lock = load_idm_model_lock(lock_path)

    with pytest.raises(TypeError, match="path must be a Path"):
        verify_idm_runtime_lock(lock, "not-a-path")  # type: ignore[arg-type]


def test_model_lock_payload_round_trips_canonical_fields(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json")
    lock = load_idm_model_lock(lock_path)

    payload = model_lock_payload(lock)

    assert payload["schema"] == IDM_MODEL_SCHEMA
    assert payload["train_classes"] == list(IDM_TRAIN_CLASSES)
    assert payload["velocity_exponent"] == Decimal("10")
    # The payload must re-serialize canonically.
    canonical_json_bytes(payload)


def test_model_lock_payload_rejects_non_lock() -> None:
    with pytest.raises(TypeError, match="lock must be an IdmModelLock"):
        model_lock_payload("not-a-lock")  # type: ignore[arg-type]


def test_inference_config_hash_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="config must be a Mapping"):
        idm_inference_config_sha256(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_inference_config_hash_rejects_wrong_key_set() -> None:
    config = idm_inference_config(SHA_A, SHA_B, "crux.input/v1")
    del config["input_view_id"]

    with pytest.raises(StrictJsonError, match="exact key set"):
        idm_inference_config_sha256(config)


def test_inference_config_hash_rejects_wrong_schema() -> None:
    config = idm_inference_config(SHA_A, SHA_B, "crux.input/v1")
    config["schema"] = "crux.wrong/v1"

    with pytest.raises(StrictJsonError, match="schema is invalid"):
        idm_inference_config_sha256(config)


def test_inference_config_hash_rejects_non_string_descriptor_hash() -> None:
    config = idm_inference_config(SHA_A, SHA_B, "crux.input/v1")
    config["backend_descriptor_sha256"] = 123

    with pytest.raises(StrictJsonError, match="lowercase SHA-256"):
        idm_inference_config_sha256(config)


def test_inference_config_hash_reraises_strict_json_on_unencodable_input_view(
    tmp_path: Path,
) -> None:
    config = idm_inference_config(SHA_A, SHA_B, "\ud800")

    with pytest.raises(StrictJsonError):
        idm_inference_config_sha256(config)


def test_load_rejects_invalid_python_version(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json", python_version="3.10.5")

    with pytest.raises(IdmModelLockError, match="python_version is invalid"):
        load_idm_model_lock(lock_path)


def test_load_rejects_malformed_model_id(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json", model_id="idm-bad-format")

    with pytest.raises(IdmModelLockError, match="model_id is invalid"):
        load_idm_model_lock(lock_path)


def test_load_rejects_model_id_not_matching_digest_fragments(tmp_path: Path) -> None:
    mismatched = "idm-44-train-kits-aaaaaaaaaaaa-bbbbbbbbbbbb"
    lock_path, _, _ = _write_lock(tmp_path / "model.json", model_id=mismatched)

    with pytest.raises(IdmModelLockError, match="locked digest fragments"):
        load_idm_model_lock(lock_path)


def test_load_rejects_non_numeric_float_field(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json", activation_rate_hz="not-a-number")

    with pytest.raises(IdmModelLockError, match="activation_rate_hz is invalid"):
        load_idm_model_lock(lock_path)


def test_load_rejects_nonfinite_float_field(tmp_path: Path) -> None:
    from src.benchmark.idm_model import _numeric_float

    with pytest.raises(IdmModelLockError, match="activation_rate_hz is invalid"):
        _numeric_float(float("inf"), "activation_rate_hz")


def test_load_rejects_non_string_hash_field(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json", runtime_lock_sha256=123)

    with pytest.raises(IdmModelLockError, match="runtime_lock_sha256 is invalid"):
        load_idm_model_lock(lock_path)


def test_load_rejects_malformed_hash_field(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json", runtime_lock_sha256="not-a-hash")

    with pytest.raises(IdmModelLockError, match="runtime_lock_sha256 is invalid"):
        load_idm_model_lock(lock_path)


def test_load_rejects_non_positive_byte_length(tmp_path: Path) -> None:
    lock_path, _, _ = _write_lock(tmp_path / "model.json", model_config_byte_length=0)

    with pytest.raises(IdmModelLockError, match="model_config_byte_length is invalid"):
        load_idm_model_lock(lock_path)


def test_require_commit_rejects_invalid_revision() -> None:
    from src.benchmark.idm_model import _require_commit

    with pytest.raises(IdmModelLockError, match="repository_revision is invalid"):
        _require_commit("not-a-commit", "repository_revision")
