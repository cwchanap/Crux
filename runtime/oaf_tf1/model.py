"""Dependency-light OaF model and checkpoint configuration."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

_SCHEMA = "crux.oaf-model/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_EXPECTED_COMPONENT_NAMES = (
    "model.ckpt-569400.data-00000-of-00001",
    "model.ckpt-569400.index",
    "model.ckpt-569400.meta",
)
_MODEL_KEYS = frozenset(
    {
        "schema",
        "backend_id",
        "model_id",
        "architecture_id",
        "upstream_source_commit",
        "training_data_map_id",
        "native_output_space_id",
        "native_metadata_schema_id",
        "max_input_audio_frames",
        "checkpoint",
    }
)
_CHECKPOINT_KEYS = frozenset({"url", "archive_name", "archive_sha256", "components"})


class OafModelConfigError(ValueError):
    """The OaF model configuration is malformed or unsafe."""


@dataclass(frozen=True)
class OafCheckpointConfig:
    url: str
    archive_name: str
    archive_sha256: str
    components: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url:
            raise OafModelConfigError("checkpoint.url is invalid")
        if (
            not isinstance(self.archive_name, str)
            or not self.archive_name
            or self.archive_name in {".", ".."}
            or any(separator in self.archive_name for separator in ("/", "\\", ":", "\x00"))
        ):
            raise OafModelConfigError("checkpoint.archive_name is invalid")
        _require_sha256(self.archive_sha256, "checkpoint.archive_sha256")
        if not isinstance(self.components, Mapping):
            raise OafModelConfigError("checkpoint.components is invalid")
        components = dict(self.components)
        if set(components) != set(_EXPECTED_COMPONENT_NAMES):
            raise OafModelConfigError("checkpoint.components names are invalid")
        for name in _EXPECTED_COMPONENT_NAMES:
            _require_sha256(components[name], f"checkpoint.components.{name}")
        object.__setattr__(self, "components", MappingProxyType(components))


@dataclass(frozen=True)
class OafModelConfig:
    backend_id: str
    model_id: str
    architecture_id: str
    upstream_source_commit: str
    training_data_map_id: str
    native_output_space_id: str
    native_metadata_schema_id: str
    max_input_audio_frames: int | None
    checkpoint: OafCheckpointConfig

    def __post_init__(self) -> None:
        for field in (
            "backend_id",
            "model_id",
            "architecture_id",
            "training_data_map_id",
            "native_output_space_id",
            "native_metadata_schema_id",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise OafModelConfigError(f"{field} is invalid")
        if not isinstance(self.upstream_source_commit, str) or not _COMMIT.fullmatch(
            self.upstream_source_commit
        ):
            raise OafModelConfigError("upstream_source_commit is invalid")
        if self.max_input_audio_frames is not None and (
            not isinstance(self.max_input_audio_frames, int)
            or isinstance(self.max_input_audio_frames, bool)
            or self.max_input_audio_frames < 0
        ):
            raise OafModelConfigError("max_input_audio_frames is invalid")
        if not isinstance(self.checkpoint, OafCheckpointConfig):
            raise OafModelConfigError("checkpoint is invalid")


def load_model_config(path: Path = Path("runtime/oaf_tf1/model.json")) -> OafModelConfig:
    """Load and validate one repository-authored OaF model configuration."""
    try:
        content = _read_regular_file_no_follow(path)
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        raise OafModelConfigError("model configuration is invalid") from None
    if not isinstance(payload, dict) or set(payload) != _MODEL_KEYS:
        raise OafModelConfigError("model configuration keys are invalid")
    if payload.get("schema") != _SCHEMA:
        raise OafModelConfigError("model configuration schema is invalid")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_KEYS:
        raise OafModelConfigError("checkpoint keys are invalid")
    try:
        return OafModelConfig(
            backend_id=payload["backend_id"],
            model_id=payload["model_id"],
            architecture_id=payload["architecture_id"],
            upstream_source_commit=payload["upstream_source_commit"],
            training_data_map_id=payload["training_data_map_id"],
            native_output_space_id=payload["native_output_space_id"],
            native_metadata_schema_id=payload["native_metadata_schema_id"],
            max_input_audio_frames=payload["max_input_audio_frames"],
            checkpoint=OafCheckpointConfig(
                url=checkpoint["url"],
                archive_name=checkpoint["archive_name"],
                archive_sha256=checkpoint["archive_sha256"],
                components=checkpoint["components"],
            ),
        )
    except (KeyError, TypeError, OafModelConfigError) as error:
        if isinstance(error, OafModelConfigError):
            raise
        raise OafModelConfigError("model configuration fields are invalid") from None


def _require_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise OafModelConfigError(f"{field} is invalid")


def _read_regular_file_no_follow(path: Path) -> bytes:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("model configuration is not a regular file")
        return os.read(descriptor, metadata.st_size)
    finally:
        os.close(descriptor)
