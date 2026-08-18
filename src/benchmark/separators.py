"""Strict identity and canonical lock loading for the HPA-328 separators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    StrictJsonError,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)

SEPARATOR_LOCK_SCHEMA = "crux.separator-lock/v1"
SPLEETER_SEPARATOR_ID = "spleeter4-drums-v1"
HTDEMUCS_SEPARATOR_ID = "htdemucs-drums-v1"

_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_LOCK_KEYS = frozenset(
    {
        "schema",
        "separator_id",
        "repository_url",
        "repository_revision",
        "package_name",
        "package_version",
        "model_id",
        "model_files",
        "code_license",
        "model_license",
        "argv",
        "expected_drum_stem_relative_path",
        "output_container",
    }
)
_MODEL_FILE_KEYS = frozenset({"name", "sha256"})

# Commands intentionally contain only interpreter-independent arguments.  The
# freeze script records the interpreter separately, while the runner supplies
# the two path placeholders in a private working directory.
_SEPARATOR_POLICIES = {
    SPLEETER_SEPARATOR_ID: {
        "repository_url": "https://github.com/deezer/spleeter",
        "package_name": "spleeter",
        "model_id": "spleeter:4stems",
        "argv": (
            "-m",
            "spleeter",
            "separate",
            "-p",
            "spleeter:4stems",
            "-o",
            "{output_dir}",
            "{input_wav}",
        ),
        "expected_drum_stem_relative_path": "input/drums.wav",
        "output_container": "wav",
    },
    HTDEMUCS_SEPARATOR_ID: {
        "repository_url": "https://github.com/facebookresearch/demucs",
        "package_name": "demucs",
        "model_id": "htdemucs",
        "argv": (
            "-m",
            "demucs",
            "-n",
            "htdemucs",
            "-o",
            "{output_dir}",
            "{input_wav}",
        ),
        "expected_drum_stem_relative_path": "htdemucs/input/drums.wav",
        "output_container": "wav",
    },
}


class SeparatorLockError(ValueError):
    """A separator lock is malformed or violates the fixed HPA-328 contract."""


@dataclass(frozen=True)
class SeparatorModelFile:
    name: str
    sha256: str

    def __post_init__(self) -> None:
        _validate_model_file_name(self.name)
        _require_hash(self.sha256, "model file sha256")


@dataclass(frozen=True)
class SeparatorLock:
    separator_id: str
    repository_url: str
    repository_revision: str
    package_name: str
    package_version: str
    model_id: str
    model_files: tuple[SeparatorModelFile, ...]
    code_license: str
    model_license: str
    argv: tuple[str, ...]
    expected_drum_stem_relative_path: str
    output_container: str
    sha256: str

    def __post_init__(self) -> None:
        _validate_lock(self)


def load_separator_lock(path: Path) -> SeparatorLock:
    """Load one canonical separator lock and attach its exact file SHA-256."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    try:
        content = read_regular_file_no_follow(path)
    except (OSError, TypeError):
        raise SeparatorLockError("separator lock is unavailable") from None
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise SeparatorLockError("separator lock must have one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise SeparatorLockError(str(error)) from None
    if not isinstance(value, dict) or set(value) != _LOCK_KEYS:
        raise SeparatorLockError("separator lock must contain the exact key set")
    if value.get("schema") != SEPARATOR_LOCK_SCHEMA:
        raise SeparatorLockError("separator lock schema is invalid")

    try:
        raw_model_files = value["model_files"]
        if not isinstance(raw_model_files, list):
            raise SeparatorLockError("model_files must be a list")
        model_files = tuple(_model_file_from_value(item) for item in raw_model_files)
        raw_argv = value["argv"]
        if not isinstance(raw_argv, list) or any(not isinstance(item, str) for item in raw_argv):
            raise SeparatorLockError("argv must be a list of strings")
        lock = SeparatorLock(
            separator_id=value["separator_id"],
            repository_url=value["repository_url"],
            repository_revision=value["repository_revision"],
            package_name=value["package_name"],
            package_version=value["package_version"],
            model_id=value["model_id"],
            model_files=model_files,
            code_license=value["code_license"],
            model_license=value["model_license"],
            argv=tuple(raw_argv),
            expected_drum_stem_relative_path=value["expected_drum_stem_relative_path"],
            output_container=value["output_container"],
            sha256=sha256_hex(content),
        )
    except SeparatorLockError:
        raise
    except (KeyError, TypeError, ValueError):
        raise SeparatorLockError("separator lock fields are invalid") from None
    return lock


def separator_lock_payload(lock: SeparatorLock) -> dict[str, object]:
    """Return the canonical JSON payload, excluding the derived file SHA."""
    if not isinstance(lock, SeparatorLock):
        raise TypeError("lock must be a SeparatorLock")
    return {
        "schema": SEPARATOR_LOCK_SCHEMA,
        "separator_id": lock.separator_id,
        "repository_url": lock.repository_url,
        "repository_revision": lock.repository_revision,
        "package_name": lock.package_name,
        "package_version": lock.package_version,
        "model_id": lock.model_id,
        "model_files": [
            {"name": model_file.name, "sha256": model_file.sha256}
            for model_file in lock.model_files
        ],
        "code_license": lock.code_license,
        "model_license": lock.model_license,
        "argv": list(lock.argv),
        "expected_drum_stem_relative_path": lock.expected_drum_stem_relative_path,
        "output_container": lock.output_container,
    }


def _model_file_from_value(value: object) -> SeparatorModelFile:
    if not isinstance(value, dict) or set(value) != _MODEL_FILE_KEYS:
        raise SeparatorLockError("model file must contain the exact key set")
    return SeparatorModelFile(name=value["name"], sha256=value["sha256"])


def _validate_lock(lock: SeparatorLock) -> None:
    policy = _SEPARATOR_POLICIES.get(lock.separator_id)
    if policy is None:
        raise SeparatorLockError("separator_id is unsupported")
    for field in (
        "repository_url",
        "repository_revision",
        "package_name",
        "package_version",
        "model_id",
        "code_license",
        "model_license",
        "expected_drum_stem_relative_path",
        "output_container",
        "sha256",
    ):
        if not isinstance(getattr(lock, field), str) or not getattr(lock, field):
            raise SeparatorLockError(f"{field} must be a nonempty string")
    if lock.repository_url != policy["repository_url"]:
        raise SeparatorLockError("repository_url does not match separator_id")
    _require_revision(lock.repository_revision)
    if lock.package_name != policy["package_name"]:
        raise SeparatorLockError("package_name does not match separator_id")
    if lock.model_id != policy["model_id"]:
        raise SeparatorLockError("model_id does not match separator_id")
    if lock.argv != policy["argv"]:
        raise SeparatorLockError("argv does not match separator model")
    if lock.expected_drum_stem_relative_path != policy["expected_drum_stem_relative_path"]:
        raise SeparatorLockError("expected drum stem path does not match separator model")
    if lock.output_container != policy["output_container"]:
        raise SeparatorLockError("output_container does not match separator model")
    _require_hash(lock.sha256, "separator lock sha256")
    if not isinstance(lock.model_files, tuple) or not lock.model_files:
        raise SeparatorLockError("model_files must be a nonempty tuple")
    if any(not isinstance(model_file, SeparatorModelFile) for model_file in lock.model_files):
        raise SeparatorLockError("model_files contains an invalid model file")
    names = [model_file.name for model_file in lock.model_files]
    if len(names) != len(set(names)):
        raise SeparatorLockError("model file names must be unique")
    if not isinstance(lock.argv, tuple) or any(not isinstance(item, str) for item in lock.argv):
        raise SeparatorLockError("argv must be a tuple of strings")


def _validate_model_file_name(value: object) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SeparatorLockError("model file name is invalid")
    if (
        value.startswith(("/", "\\"))
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or "\\" in value
    ):
        raise SeparatorLockError("model file name must be relative")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise SeparatorLockError("model file name must be normalized")


def _require_revision(value: object) -> None:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        raise SeparatorLockError("repository_revision must be a lowercase Git revision")


def _require_hash(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise SeparatorLockError(f"{field} must be a lowercase SHA-256 hash")
    try:
        require_sha256(value, field)
    except StrictJsonError:
        raise SeparatorLockError(f"{field} must be a lowercase SHA-256 hash") from None


__all__ = [
    "HTDEMUCS_SEPARATOR_ID",
    "SEPARATOR_LOCK_SCHEMA",
    "SPLEETER_SEPARATOR_ID",
    "SeparatorLock",
    "SeparatorLockError",
    "SeparatorModelFile",
    "load_separator_lock",
    "separator_lock_payload",
]
