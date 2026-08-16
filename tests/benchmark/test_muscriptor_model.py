from __future__ import annotations

import hashlib
import json
import re
from dataclasses import fields
from decimal import Decimal
from pathlib import Path

import pytest

from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.muscriptor_model import (
    MUSCRIPTOR_MODEL_ID_RE,
    MUSCRIPTOR_MODEL_SCHEMA,
    MUSCRIPTOR_RELEASE_COMMIT,
    MuscriptorModelLock,
    derive_muscriptor_model_id,
    load_muscriptor_model_lock,
    verify_muscriptor_checkpoint,
)

WEIGHTS = b"frozen safetensors bytes"
CONFIG = b'{"model_type":"muscriptor"}'
WEIGHTS_SHA256 = hashlib.sha256(WEIGHTS).hexdigest()
CONFIG_SHA256 = hashlib.sha256(CONFIG).hexdigest()


def _payload(**overrides: object) -> dict[str, object]:
    revision = "a" * 40
    payload: dict[str, object] = {
        "schema": MUSCRIPTOR_MODEL_SCHEMA,
        "package_name": "muscriptor",
        "package_version": "0.3.0",
        "upstream_source_commit": MUSCRIPTOR_RELEASE_COMMIT,
        "code_license": "MIT",
        "weight_license": "CC BY-NC 4.0",
        "checkpoint_variant": "medium",
        "checkpoint_repo_id": "MuScriptor/muscriptor-medium",
        "checkpoint_revision": revision,
        "checkpoint_filename": "model.safetensors",
        "checkpoint_sha256": WEIGHTS_SHA256,
        "checkpoint_byte_length": len(WEIGHTS),
        "checkpoint_config_filename": "config.json",
        "checkpoint_config_sha256": CONFIG_SHA256,
        "checkpoint_config_byte_length": len(CONFIG),
        "model_id": f"muscriptor-medium-{revision[:12]}-{WEIGHTS_SHA256[:12]}",
        "device": "cpu",
        "dtype": "float32",
        "input_sample_rate_hz": 16000,
        "chunk_duration_sec": Decimal("5.0"),
        "use_sampling": False,
        "temperature": Decimal("1.0"),
        "cfg_coef": Decimal("1.0"),
        "instruments": ["drums"],
        "batch_size": 1,
        "no_eos_is_ok": True,
        "beam_size": 1,
        "prelude_forcing": True,
        "native_output_space_id": "muscriptor-drums-midi128-v1",
        "native_metadata_schema_id": "muscriptor-note-start-metadata-v1",
        "training_data_map_id": "muscriptor-training-data-v0.3.0",
    }
    payload.update(overrides)
    return payload


def _write_lock(tmp_path: Path, **overrides: object) -> Path:
    path = tmp_path / "model.json"
    path.write_bytes(canonical_json_bytes(_payload(**overrides), trailing_newline=True))
    return path


def test_loads_closed_lock_and_freezes_deterministic_settings(tmp_path: Path) -> None:
    path = _write_lock(tmp_path)

    lock = load_muscriptor_model_lock(path)

    assert lock.package_name == "muscriptor"
    assert lock.package_version == "0.3.0"
    assert lock.upstream_source_commit == MUSCRIPTOR_RELEASE_COMMIT
    assert lock.checkpoint_variant in {"medium", "small"}
    assert re.fullmatch(r"[0-9a-f]{40}", lock.checkpoint_revision)
    assert derive_muscriptor_model_id(lock) == (
        f"muscriptor-{lock.checkpoint_variant}-"
        f"{lock.checkpoint_revision[:12]}-{lock.checkpoint_sha256[:12]}"
    )
    assert lock.model_id == derive_muscriptor_model_id(lock)
    assert MUSCRIPTOR_MODEL_ID_RE.fullmatch(lock.model_id)
    assert lock.instruments == ("drums",)
    assert lock.use_sampling is False
    assert lock.temperature == 1.0
    assert lock.cfg_coef == 1.0
    assert lock.batch_size == 1
    assert lock.no_eos_is_ok is True
    assert lock.beam_size == 1
    assert lock.prelude_forcing is True
    assert lock.input_sample_rate_hz == 16000
    assert lock.chunk_duration_sec == 5.0

    assert set(_payload()) == {"schema", *(field.name for field in fields(MuscriptorModelLock))}


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": True},
        {"package_version": None},
    ],
)
def test_lock_rejects_unknown_or_missing_key_set(tmp_path: Path, change: dict[str, object]) -> None:
    payload = _payload()
    if "unexpected" in change:
        payload.update(change)
    else:
        payload.pop("package_version")
    path = tmp_path / "model.json"
    path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))

    with pytest.raises(ValueError):
        load_muscriptor_model_lock(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("package_name", "other"),
        ("package_version", "0.2.0"),
        ("upstream_source_commit", "b" * 40),
        ("code_license", "Apache-2.0"),
        ("weight_license", "MIT"),
        ("checkpoint_revision", "A" * 40),
        ("checkpoint_sha256", "not-a-hash"),
        ("checkpoint_byte_length", 0),
        ("checkpoint_config_sha256", "g" * 64),
        ("checkpoint_config_byte_length", -1),
        ("device", "auto"),
        ("dtype", "int8"),
        ("input_sample_rate_hz", 44100),
        ("chunk_duration_sec", Decimal("4.0")),
        ("use_sampling", True),
        ("temperature", Decimal("0.7")),
        ("cfg_coef", Decimal("0.0")),
        ("instruments", ["piano"]),
        ("batch_size", 2),
        ("no_eos_is_ok", False),
        ("beam_size", 2),
        ("prelude_forcing", False),
    ],
)
def test_lock_rejects_invalid_frozen_fields(tmp_path: Path, field: str, value: object) -> None:
    path = _write_lock(tmp_path, **{field: value})

    with pytest.raises(ValueError, match=field):
        load_muscriptor_model_lock(path)


def test_lock_rejects_model_id_digest_mismatch(tmp_path: Path) -> None:
    path = _write_lock(tmp_path, model_id="muscriptor-medium-aaaaaaaaaaaa-bbbbbbbbbbbb")

    with pytest.raises(ValueError, match="model_id"):
        load_muscriptor_model_lock(path)


def test_checkpoint_verifier_matches_both_immutable_files(tmp_path: Path) -> None:
    path = _write_lock(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(WEIGHTS)
    (checkpoint_dir / "config.json").write_bytes(CONFIG)
    lock = load_muscriptor_model_lock(path)

    assert verify_muscriptor_checkpoint(lock, checkpoint_dir) == (
        checkpoint_dir / "model.safetensors"
    )

    (checkpoint_dir / "config.json").write_bytes(CONFIG + b"edited")
    with pytest.raises(ValueError, match="config"):
        verify_muscriptor_checkpoint(lock, checkpoint_dir)


def test_checkpoint_verifier_rejects_missing_or_edited_weights(tmp_path: Path) -> None:
    path = _write_lock(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_bytes(CONFIG)
    lock = load_muscriptor_model_lock(path)

    with pytest.raises(ValueError, match="checkpoint"):
        verify_muscriptor_checkpoint(lock, checkpoint_dir)

    (checkpoint_dir / "model.safetensors").write_bytes(WEIGHTS + b"edited")
    with pytest.raises(ValueError, match="checkpoint"):
        verify_muscriptor_checkpoint(lock, checkpoint_dir)


def test_muscriptor_smoke_manifest_has_five_pre_model_cases() -> None:
    path = Path(__file__).parents[2] / "runtime/muscriptor/smoke.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert set(manifest) == {"schema", "cases"}
    assert manifest["schema"] == "crux.muscriptor-smoke/v1"
    cases = manifest["cases"]
    assert len(cases) == 5
    assert {case["reason"] for case in cases} == {
        "short",
        "long",
        "dense",
        "sparse",
        "non_drum_heavy",
    }
    ids = [case["simfile_id"] for case in cases]
    assert all(
        isinstance(simfile_id, int) and not isinstance(simfile_id, bool) for simfile_id in ids
    )
    assert all(simfile_id > 0 for simfile_id in ids)
    assert len(ids) == len(set(ids))
    assert all(set(case) == {"simfile_id", "reason"} for case in cases)
