from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from src.benchmark.backend_identity import canonical_json_bytes
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
