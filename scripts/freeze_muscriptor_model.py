#!/usr/bin/env python3
"""Freeze one authenticated MuScriptor checkpoint and model lock."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import re
import sys
from pathlib import Path
from typing import Any

from src.benchmark.artifact_io import publish_immutable_file, read_regular_file_no_follow
from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.muscriptor_model import (
    MUSCRIPTOR_NATIVE_METADATA_SCHEMA_ID,
    MUSCRIPTOR_NATIVE_OUTPUT_SPACE_ID,
    MUSCRIPTOR_RELEASE_COMMIT,
    MUSCRIPTOR_TRAINING_DATA_MAP_ID,
    MUSCRIPTOR_WEIGHT_LICENSE,
    MuscriptorModelLock,
    derive_muscriptor_model_id,
    load_muscriptor_model_lock,
    model_lock_payload,
    verify_muscriptor_checkpoint,
)

_VARIANT_RE = re.compile(r"(?:medium|small)\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_DEVICE_RE = re.compile(r"(?:cpu|mps|cuda(?::[0-9]+)?)\Z")
_DTYPE_VALUES = frozenset({"float16", "float32", "bfloat16"})
_REPO_PREFIX = "MuScriptor/muscriptor-"
_CHECKPOINT_FILENAME = "model.safetensors"
_CONFIG_FILENAME = "config.json"
_HF_WEIGHT_LICENSE = "cc-by-nc-4.0"


class FreezeError(RuntimeError):
    """The authenticated model freeze could not produce a valid lock."""


def freeze_model(
    *,
    variant: str,
    device: str,
    dtype: str,
    checkpoint_dir: Path,
    output: Path,
) -> MuscriptorModelLock:
    """Resolve, download, verify, and publish one MuScriptor model lock."""
    if _VARIANT_RE.fullmatch(variant) is None:
        raise FreezeError("variant must be medium or small")
    if not isinstance(checkpoint_dir, Path) or not isinstance(output, Path):
        raise TypeError("checkpoint_dir and output must be Paths")

    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    repo_id = _REPO_PREFIX + variant
    code_license = _verify_package()
    revision, weight_license = _resolve_huggingface_model(repo_id)

    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FreezeError("checkpoint directory is unavailable") from error
    checkpoint_path = _download_immutable(
        repo_id, revision, _CHECKPOINT_FILENAME, checkpoint_dir / _CHECKPOINT_FILENAME
    )
    config_path = _download_immutable(
        repo_id, revision, _CONFIG_FILENAME, checkpoint_dir / _CONFIG_FILENAME
    )
    checkpoint_bytes = read_regular_file_no_follow(checkpoint_path)
    config_bytes = read_regular_file_no_follow(config_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    lock_without_id = dict(
        package_name="muscriptor",
        package_version="0.3.0",
        upstream_source_commit=MUSCRIPTOR_RELEASE_COMMIT,
        code_license=code_license,
        weight_license=weight_license,
        checkpoint_variant=variant,
        checkpoint_repo_id=repo_id,
        checkpoint_revision=revision,
        checkpoint_filename=_CHECKPOINT_FILENAME,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_length=len(checkpoint_bytes),
        checkpoint_config_filename=_CONFIG_FILENAME,
        checkpoint_config_sha256=config_sha256,
        checkpoint_config_byte_length=len(config_bytes),
        device=resolved_device,
        dtype=resolved_dtype,
        input_sample_rate_hz=16000,
        chunk_duration_sec=5.0,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=1.0,
        instruments=("drums",),
        batch_size=1,
        no_eos_is_ok=True,
        beam_size=1,
        prelude_forcing=True,
        native_output_space_id=MUSCRIPTOR_NATIVE_OUTPUT_SPACE_ID,
        native_metadata_schema_id=MUSCRIPTOR_NATIVE_METADATA_SCHEMA_ID,
        training_data_map_id=MUSCRIPTOR_TRAINING_DATA_MAP_ID,
    )
    model_id = f"muscriptor-{variant}-{revision[:12]}-{checkpoint_sha256[:12]}"
    lock = MuscriptorModelLock(model_id=model_id, **lock_without_id)
    if derive_muscriptor_model_id(lock) != model_id:
        raise FreezeError("derived model ID does not match the frozen lock")

    content = canonical_json_bytes(model_lock_payload(lock), trailing_newline=True)
    publish_immutable_file(output, content)
    loaded = load_muscriptor_model_lock(output)
    verify_muscriptor_checkpoint(loaded, checkpoint_dir)
    return loaded


def _verify_package() -> str:
    try:
        version = importlib.metadata.version("muscriptor")
        metadata = importlib.metadata.metadata("muscriptor")
    except importlib.metadata.PackageNotFoundError as error:
        raise FreezeError("muscriptor==0.3.0 is not installed; use --extra muscriptor") from error
    if version != "0.3.0":
        raise FreezeError(f"installed muscriptor version is {version}, expected 0.3.0")
    license_expression = metadata.get("License-Expression")
    if license_expression != "MIT":
        raise FreezeError("MuScriptor code license metadata is not MIT")
    return license_expression


def _resolve_huggingface_model(repo_id: str) -> tuple[str, str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise FreezeError("huggingface_hub is unavailable; use --extra muscriptor") from error
    try:
        info = HfApi().model_info(repo_id)
    except Exception as error:  # noqa: BLE001 - preserve the operational HF failure
        raise FreezeError(
            f"Hugging Face model metadata unavailable for {repo_id}: {error}"
        ) from error
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise FreezeError("Hugging Face model revision is not a 40-character commit SHA")
    if getattr(info, "id", repo_id) != repo_id:
        raise FreezeError("Hugging Face model repository identity differs")
    siblings = {
        getattr(sibling, "rfilename", None) for sibling in (getattr(info, "siblings", None) or ())
    }
    if {_CHECKPOINT_FILENAME, _CONFIG_FILENAME} - siblings:
        raise FreezeError("Hugging Face model is missing model.safetensors or config.json")

    card_data = getattr(info, "cardData", None)
    if card_data is None:
        card_data = getattr(info, "card_data", None)
    license_value = _mapping_value(card_data, "license")
    if license_value != _HF_WEIGHT_LICENSE:
        raise FreezeError("Hugging Face weight license metadata is missing or contradictory")
    return revision, MUSCRIPTOR_WEIGHT_LICENSE


def _download_immutable(repo_id: str, revision: str, filename: str, destination: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise FreezeError("huggingface_hub is unavailable; use --extra muscriptor") from error
    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        content = Path(downloaded).read_bytes()
    except Exception as error:  # noqa: BLE001 - preserve auth/license/network details
        raise FreezeError(
            f"Hugging Face download failed for {repo_id}/{filename} at {revision}: {error}"
        ) from error
    publish_immutable_file(destination, content)
    return destination


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        if not isinstance(requested, str) or _DEVICE_RE.fullmatch(requested) is None:
            raise FreezeError("device must be auto, cpu, mps, cuda, or cuda:<index>")
        return requested
    try:
        import torch
    except ImportError as error:
        raise FreezeError("torch is unavailable while resolving device auto") from error
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(requested: str, device: str) -> str:
    if requested == "auto":
        return "float16" if device == "mps" else "float32"
    if requested not in _DTYPE_VALUES:
        raise FreezeError("dtype must be auto, float16, float32, or bfloat16")
    return requested


def _mapping_value(value: Any, key: str) -> object:
    if isinstance(value, dict):
        return value.get(key)
    getter = getattr(value, "get", None)
    return getter(key) if callable(getter) else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("medium", "small"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        lock = freeze_model(
            variant=args.variant,
            device=args.device,
            dtype=args.dtype,
            checkpoint_dir=args.checkpoint_dir,
            output=args.output,
        )
    except (FreezeError, OSError, ValueError) as error:
        print(f"freeze failed: {error}", file=sys.stderr)
        return 1
    print(lock.model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
