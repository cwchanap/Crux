#!/usr/bin/env python3
"""Freeze the authenticated IDM runtime, model config, and checkpoint identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.artifact_io import (  # noqa: E402
    publish_immutable_file,
    read_regular_file_no_follow,
)
from src.benchmark.backend_identity import canonical_json_bytes  # noqa: E402
from src.benchmark.idm_model import (  # noqa: E402
    IDM_CHECKPOINT_RELATIVE_PATH,
    IDM_CODE_LICENSE,
    IDM_MODEL_CONFIG_RELATIVE_PATH,
    IDM_MODEL_NAME,
    IDM_NATIVE_METADATA_SCHEMA_ID,
    IDM_NATIVE_OUTPUT_SPACE_ID,
    IDM_PACKAGE_NAME,
    IDM_PACKAGE_VERSION,
    IDM_RELEASE_COMMIT,
    IDM_REPOSITORY_URL,
    IDM_TRAIN_CLASSES,
    IDM_TRAINING_DATA_MAP_ID,
    IDM_WEIGHT_LICENSE,
    IdmModelLock,
    IdmModelLockError,
    derive_idm_model_id,
    load_idm_model_lock,
    model_lock_payload,
    verify_idm_model_files,
    verify_idm_runtime_lock,
)

_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_DEVICE_RE = re.compile(r"(?:cpu|mps|cuda(?::[0-9]+)?)\Z")
_DTYPE_VALUES = frozenset({"float16", "float32", "bfloat16"})
_LICENSE_NOTE = (
    "No separate checkpoint notice exists in the pinned repository; this records repository-level "
    "provenance and is not an independent legal conclusion."
)


class FreezeError(RuntimeError):
    """The authenticated IDM model freeze could not produce a valid lock."""


def freeze_model(
    *,
    source_root: Path,
    runtime_lock_path: Path,
    output: Path,
    model_root: Path | None = None,
    runtime_python: Path | None = None,
    license_provenance_path: Path | None = None,
    device: str = "auto",
    dtype: str = "auto",
) -> IdmModelLock:
    """Verify exact pinned inputs and publish one immutable IDM model lock."""
    _require_path(source_root, "source_root")
    _require_path(runtime_lock_path, "runtime_lock_path")
    _require_path(output, "output")
    if source_root.is_symlink():
        raise FreezeError("pinned source root is unavailable")
    if runtime_lock_path.is_symlink():
        raise FreezeError("runtime lock is unavailable")
    source_root = source_root.resolve()
    model_root = source_root if model_root is None else _require_path(model_root, "model_root")
    runtime_python = (
        runtime_lock_path.parent / ".venv" / "bin" / "python"
        if runtime_python is None
        else _require_path(runtime_python, "runtime_python")
    )
    license_provenance_path = (
        runtime_lock_path.parent / "idm-wheel-provenance.json"
        if license_provenance_path is None
        else _require_path(license_provenance_path, "license_provenance_path")
    )

    source_revision = _verify_source_revision(source_root)
    runtime_lock_bytes = _read_regular(runtime_lock_path, "runtime lock")
    config_bytes = _read_git_blob(source_root, source_revision, IDM_MODEL_CONFIG_RELATIVE_PATH)
    checkpoint_bytes = _read_git_blob(source_root, source_revision, IDM_CHECKPOINT_RELATIVE_PATH)
    license_bytes = _read_git_blob(source_root, source_revision, "LICENSE")
    _verify_model_config(config_bytes)
    _verify_license_evidence(license_bytes, license_provenance_path)

    resolved_device = _resolve_device(device, runtime_python)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    python_version = _runtime_python_version(runtime_python)

    if model_root != source_root:
        _publish_model_inputs(model_root, config_bytes, checkpoint_bytes)
    model_root = model_root.resolve()
    runtime_lock_sha256 = hashlib.sha256(runtime_lock_bytes).hexdigest()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    model_id = f"idm-44-train-kits-{IDM_RELEASE_COMMIT[:12]}-{checkpoint_sha256[:12]}"
    lock = IdmModelLock(
        repository_url=IDM_REPOSITORY_URL,
        repository_revision=IDM_RELEASE_COMMIT,
        package_name=IDM_PACKAGE_NAME,
        package_version=IDM_PACKAGE_VERSION,
        code_license=IDM_CODE_LICENSE,
        weight_license=IDM_WEIGHT_LICENSE,
        runtime_lock_sha256=runtime_lock_sha256,
        python_version=python_version,
        model_name=IDM_MODEL_NAME,
        model_config_relative_path=IDM_MODEL_CONFIG_RELATIVE_PATH,
        model_config_sha256=config_sha256,
        model_config_byte_length=len(config_bytes),
        checkpoint_relative_path=IDM_CHECKPOINT_RELATIVE_PATH,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_length=len(checkpoint_bytes),
        model_id=model_id,
        device=resolved_device,
        dtype=resolved_dtype,
        sample_rate_hz=44100,
        input_channel_count=1,
        input_container="WAV",
        input_subtype="PCM_16",
        audio_loader_revision="soundfile-preserve-wav/v1",
        resampling="forbidden",
        mixdown="forbidden",
        mel_n_fft=1024,
        mel_hop_length=256,
        mel_n_mels=128,
        activation_rate_hz=44100 / 256,
        train_classes=IDM_TRAIN_CLASSES,
        onset_activation="sigmoid-over-native-logits-before-upstream-peak-picking",
        peak_pick_div_max=20,
        peak_pick_div_avg=10,
        peak_pick_div_wait=16,
        peak_pick_div_threshold=5,
        peak_pick_normalize=False,
        velocity_activation="exp_sigmoid(exponent=10,max_value=2,threshold=1e-7)",
        velocity_max_value=2.0,
        velocity_to_midi_revision="clamp-half-round-midi127/v1",
        native_velocity_persistence_revision="quantize-six-canonical-string/v1",
        manual_onset_override=False,
        reconstructed_stems=False,
        masking="none",
        chunking_mode="none",
        native_output_space_id=IDM_NATIVE_OUTPUT_SPACE_ID,
        native_metadata_schema_id=IDM_NATIVE_METADATA_SCHEMA_ID,
        training_data_map_id=IDM_TRAINING_DATA_MAP_ID,
    )
    if derive_idm_model_id(lock) != model_id:
        raise FreezeError("derived model ID does not match the frozen lock")

    # Validate the selected model root before publishing a lock that would bind it.
    verify_idm_model_files(lock, model_root)
    verify_idm_runtime_lock(lock, runtime_lock_path)
    content = canonical_json_bytes(model_lock_payload(lock), trailing_newline=True)
    publish_immutable_file(output, content)
    loaded = load_idm_model_lock(output)
    verify_idm_model_files(loaded, model_root)
    verify_idm_runtime_lock(loaded, runtime_lock_path)
    return loaded


def _verify_source_revision(source_root: Path) -> str:
    if source_root.is_symlink() or not source_root.is_dir():
        raise FreezeError("pinned source root is unavailable")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FreezeError("pinned source revision is unavailable") from error
    if _REVISION_RE.fullmatch(revision) is None or revision != IDM_RELEASE_COMMIT:
        raise FreezeError(
            f"source revision mismatch: expected {IDM_RELEASE_COMMIT}, got {revision}"
        )
    return revision


def _verify_model_config(content: bytes) -> None:
    try:
        import yaml

        config = yaml.safe_load(content)
    except ImportError as error:
        raise FreezeError("PyYAML is unavailable while verifying model config") from error
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise FreezeError("model config YAML is unavailable") from error
    if not isinstance(config, dict):
        raise FreezeError("model config must be a mapping")
    try:
        encoder = config["encoder"]
        transform = encoder["transform"]
        head = encoder["transcription_head"]
        decoder = config["decoder"]
        actual_facts = (
            config["sampling_rate"],
            encoder["sampling_rate"],
            decoder["sampling_rate"],
            transform["sample_rate"],
            transform["n_fft"],
            transform["hop_length"],
            transform["n_mels"],
            head["onset_activation"],
            head["velocity_activation"],
            config["train_classes"],
        )
    except (KeyError, TypeError):
        raise FreezeError("model config facts differ from the frozen IDM contract") from None
    expected_facts = (
        44100,
        44100,
        44100,
        44100,
        1024,
        256,
        128,
        "none",
        "exp_sigmoid",
        list(IDM_TRAIN_CLASSES),
    )
    if actual_facts != expected_facts:
        raise FreezeError("model config facts differ from the frozen IDM contract")


def _verify_license_evidence(license_bytes: bytes, provenance_path: Path) -> None:
    if b"Apache License" not in license_bytes or b"Version 2.0" not in license_bytes:
        raise FreezeError("source code license evidence is not Apache-2.0")
    try:
        provenance = json.loads(_read_regular(provenance_path, "license provenance"))
    except (json.JSONDecodeError, IdmModelLockError) as error:
        raise FreezeError("IDM checkpoint license provenance is unavailable") from error
    basis = provenance.get("license_basis") if isinstance(provenance, dict) else None
    if not isinstance(basis, dict):
        raise FreezeError("IDM checkpoint license provenance is unavailable")
    if (
        provenance.get("source_commit") != IDM_RELEASE_COMMIT
        or provenance.get("package_name") != IDM_PACKAGE_NAME
        or provenance.get("package_version") != IDM_PACKAGE_VERSION
        or basis.get("repository_code") != IDM_CODE_LICENSE
        or basis.get("checkpoint_provenance") != "repository-level Apache-2.0"
        or basis.get("checkpoint_notice_found") is not False
        or basis.get("note") != _LICENSE_NOTE
    ):
        raise FreezeError("IDM checkpoint license evidence is contradictory")


def _publish_model_inputs(model_root: Path, config: bytes, checkpoint: bytes) -> None:
    if model_root.is_symlink():
        raise FreezeError("model root is unavailable")
    config_path = model_root / IDM_MODEL_CONFIG_RELATIVE_PATH
    checkpoint_path = model_root / IDM_CHECKPOINT_RELATIVE_PATH
    publish_immutable_file(config_path, config)
    publish_immutable_file(checkpoint_path, checkpoint)


def _resolve_device(requested: str, runtime_python: Path) -> str:
    if requested != "auto":
        if not isinstance(requested, str) or _DEVICE_RE.fullmatch(requested) is None:
            raise FreezeError("device must be auto, cpu, mps, cuda, or cuda:<index>")
        return requested
    probe = (
        "import torch; "
        "print('cuda' if torch.cuda.is_available() else "
        "('mps' if getattr(getattr(torch, 'backends', None), 'mps', None) "
        ".is_available() else 'cpu'))"
    )
    try:
        selected = subprocess.check_output(
            [str(runtime_python), "-c", probe],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FreezeError("runtime device feasibility probe failed") from error
    if _DEVICE_RE.fullmatch(selected) is None:
        raise FreezeError("runtime device feasibility probe returned an invalid device")
    return selected


def _resolve_dtype(requested: str, device: str) -> str:
    if requested == "auto":
        return "float16" if device == "mps" else "float32"
    if requested not in _DTYPE_VALUES:
        raise FreezeError("dtype must be auto, float16, float32, or bfloat16")
    return requested


def _runtime_python_version(runtime_python: Path) -> str:
    try:
        output = subprocess.check_output(
            [str(runtime_python), "--version"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FreezeError("isolated runtime Python is unavailable") from error
    match = re.search(r"Python (3\.11\.[0-9]+)\Z", output)
    if match is None:
        raise FreezeError("isolated runtime Python must be 3.11.x")
    return match.group(1)


def _read_regular(path: Path, field: str) -> bytes:
    try:
        return read_regular_file_no_follow(path)
    except (OSError, TypeError):
        raise FreezeError(f"{field} is unavailable") from None


def _read_git_blob(source_root: Path, revision: str, relative_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "show", f"{revision}:{relative_path}"],
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FreezeError(f"pinned source file is unavailable: {relative_path}") from error


def _require_path(path: Path, field: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{field} must be a Path")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, default=Path("runtime/idm/uv.lock"))
    parser.add_argument("--runtime-python", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--license-provenance", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output", type=Path, default=Path("runtime/idm/model.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        lock = freeze_model(
            source_root=args.source_root,
            runtime_lock_path=args.runtime_lock,
            output=args.output,
            model_root=args.model_root,
            runtime_python=args.runtime_python,
            license_provenance_path=args.license_provenance,
            device=args.device,
            dtype=args.dtype,
        )
    except (FreezeError, OSError, ValueError) as error:
        print(f"freeze failed: {error}", file=sys.stderr)
        return 1
    print(lock.model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
