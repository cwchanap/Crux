"""Strict identity and canonical lock loading for the HPA-328 separators."""

from __future__ import annotations

import math
import os
import re
import signal
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import soundfile

from src.benchmark.artifact_io import (
    ArtifactPublicationError,
    publish_immutable_file,
    read_regular_file_no_follow,
)
from src.benchmark.backend_identity import (
    StrictJsonError,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)

SEPARATOR_LOCK_SCHEMA = "crux.separator-lock/v2"
SEPARATOR_ENVIRONMENT_SCHEMA = "crux.separator-environment/v1"
SPLEETER_SEPARATOR_ID = "spleeter4-drums-v1"
HTDEMUCS_SEPARATOR_ID = "htdemucs-drums-v1"
SEPARATOR_TIMEOUT_SECONDS = 1800.0
SEPARATOR_TERMINATE_GRACE_SECONDS = 5.0
STEM_NEAR_SILENT_DBFS = -80.0
STEM_CLIP_ABS = 0.9999
STEM_MAX_DURATION_DELTA_SECONDS = 0.5
STEM_MAX_DURATION_DELTA_RATIO = 0.005

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
        "interpreter_sha256",
        "environment_manifest_sha256",
        "model_root_kind",
    }
)
_MODEL_FILE_KEYS = frozenset({"name", "sha256"})
_ENVIRONMENT_KEYS = frozenset(
    {
        "schema",
        "separator_id",
        "package_name",
        "package_version",
        "python_implementation",
        "python_version",
        "python_abi",
        "platform",
        "interpreter_sha256",
        "distributions",
    }
)
_ENVIRONMENT_DISTRIBUTION_KEYS = frozenset({"name", "version", "files"})
_ENVIRONMENT_FILE_KEYS = frozenset({"root", "path", "byte_length", "sha256"})
_ENVIRONMENT_ROOT_TAGS = frozenset(
    {
        "stdlib",
        "platstdlib",
        "purelib",
        "platlib",
        "include",
        "platinclude",
        "scripts",
        "data",
    }
)
_DISTRIBUTION_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")

# Commands intentionally contain only interpreter-independent arguments.  The
# freeze script records the interpreter separately, while the runner supplies
# the private path placeholders in a temporary working directory.
_SEPARATOR_POLICIES = {
    SPLEETER_SEPARATOR_ID: {
        "repository_url": "https://github.com/deezer/spleeter",
        "package_name": "spleeter",
        "model_id": "spleeter:4stems",
        "model_root_kind": "spleeter-model-path-v1",
        "environment_discovery_keys": ("MODEL_PATH",),
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
        "model_root_kind": "demucs-local-repo-v1",
        "environment_discovery_keys": (
            "DEMUCS_MODEL_PATH",
            "DEMUCS_REPO",
            "HF_DATASETS_OFFLINE",
            "HF_ENDPOINT",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_HUB_DISABLE_TELEMETRY",
            "HF_HUB_ENDPOINT",
            "HF_HUB_OFFLINE",
            "HUGGINGFACE_HUB_CACHE",
            "TORCH_HOME",
            "TRANSFORMERS_CACHE",
            "XDG_CACHE_HOME",
        ),
        "argv": (
            "-m",
            "demucs",
            "-n",
            "htdemucs",
            "--repo",
            "{model_root}",
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


class SeparatorExecutionError(ValueError):
    """A separator process, stem artifact, or fixed QC check failed."""

    def __init__(self, detail_code: str, message: str | None = None) -> None:
        if not isinstance(detail_code, str) or not detail_code:
            raise ValueError("detail_code must be a nonempty string")
        self.detail_code = detail_code
        super().__init__(f"{detail_code}: {message or detail_code}")

    @property
    def code(self) -> str:
        """Return the stable native detail code."""
        return self.detail_code


@dataclass(frozen=True)
class StemQc:
    """Fixed quality measurements for one native separator WAV."""

    sample_rate: int
    frame_count: int
    channel_count: int
    duration_sec: float
    rms_dbfs: float
    peak_abs: float
    clipping_detected: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeparatedStem:
    """One immutable native drum stem and its fixed QC evidence."""

    separator_id: str
    source_audio_sha256: str
    separator_lock_sha256: str
    path: Path
    sha256: str
    qc: StemQc
    cache_hit: bool
    warnings: tuple[str, ...] = ()


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
    interpreter_sha256: str
    environment_manifest_sha256: str
    model_root_kind: str
    sha256: str

    def __post_init__(self) -> None:
        _validate_lock(self)


@dataclass(frozen=True)
class SeparatorEnvironmentFile:
    root: str
    path: str
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_environment_file(self)


@dataclass(frozen=True)
class SeparatorEnvironmentDistribution:
    name: str
    version: str
    files: tuple[SeparatorEnvironmentFile, ...]

    def __post_init__(self) -> None:
        _validate_environment_distribution(self)


@dataclass(frozen=True)
class SeparatorEnvironmentManifest:
    separator_id: str
    package_name: str
    package_version: str
    python_implementation: str
    python_version: str
    python_abi: str
    platform: str
    interpreter_sha256: str
    distributions: tuple[SeparatorEnvironmentDistribution, ...]
    sha256: str

    def __post_init__(self) -> None:
        _validate_environment_manifest(self)


@dataclass(frozen=True)
class AttestedSeparatorRuntime:
    """Resolved separator inputs that have passed the fixed runtime attestation."""

    interpreter: Path
    lock: SeparatorLock
    model_root: Path
    model_files: tuple[SeparatorModelFile, ...]
    environment: SeparatorEnvironmentManifest
    launch_environment: Mapping[str, str]


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
    if not isinstance(value, dict):
        raise SeparatorLockError("separator lock must contain the exact key set")
    if value.get("schema") != SEPARATOR_LOCK_SCHEMA:
        raise SeparatorLockError("separator lock schema is invalid")
    if set(value) != _LOCK_KEYS:
        raise SeparatorLockError("separator lock must contain the exact key set")

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
            interpreter_sha256=value["interpreter_sha256"],
            environment_manifest_sha256=value["environment_manifest_sha256"],
            model_root_kind=value["model_root_kind"],
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
        "interpreter_sha256": lock.interpreter_sha256,
        "environment_manifest_sha256": lock.environment_manifest_sha256,
        "model_root_kind": lock.model_root_kind,
    }


def separator_environment_manifest_payload(
    manifest: SeparatorEnvironmentManifest,
) -> dict[str, object]:
    """Return the canonical JSON payload, excluding the derived file SHA."""
    if not isinstance(manifest, SeparatorEnvironmentManifest):
        raise TypeError("manifest must be a SeparatorEnvironmentManifest")
    return {
        "schema": SEPARATOR_ENVIRONMENT_SCHEMA,
        "separator_id": manifest.separator_id,
        "package_name": manifest.package_name,
        "package_version": manifest.package_version,
        "python_implementation": manifest.python_implementation,
        "python_version": manifest.python_version,
        "python_abi": manifest.python_abi,
        "platform": manifest.platform,
        "interpreter_sha256": manifest.interpreter_sha256,
        "distributions": [
            {
                "name": distribution.name,
                "version": distribution.version,
                "files": [
                    {
                        "root": environment_file.root,
                        "path": environment_file.path,
                        "byte_length": environment_file.byte_length,
                        "sha256": environment_file.sha256,
                    }
                    for environment_file in distribution.files
                ],
            }
            for distribution in manifest.distributions
        ],
    }


def load_separator_environment_manifest(
    lock_path: Path,
    lock: SeparatorLock,
) -> SeparatorEnvironmentManifest:
    """Load the exact canonical environment sibling bound by one v2 lock."""
    if not isinstance(lock_path, Path):
        raise TypeError("lock_path must be a Path")
    if not isinstance(lock, SeparatorLock):
        raise TypeError("lock must be a SeparatorLock")

    sibling_path = lock_path.parent / "environment.json"
    try:
        content = read_regular_file_no_follow(sibling_path)
    except (OSError, TypeError):
        raise SeparatorLockError("separator lock companion environment is unavailable") from None
    if sha256_hex(content) != lock.environment_manifest_sha256:
        raise SeparatorLockError("separator lock companion environment hash does not match")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise SeparatorLockError("separator lock companion environment must have one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise SeparatorLockError(
            f"separator lock companion environment is invalid: {error}"
        ) from None
    if not isinstance(value, dict) or set(value) != _ENVIRONMENT_KEYS:
        raise SeparatorLockError(
            "separator lock companion environment must contain the exact key set"
        )
    if value.get("schema") != SEPARATOR_ENVIRONMENT_SCHEMA:
        raise SeparatorLockError("separator lock companion environment schema is invalid")

    try:
        raw_distributions = value["distributions"]
        if not isinstance(raw_distributions, list):
            raise SeparatorLockError("environment distributions must be a list")
        distributions = tuple(
            _environment_distribution_from_value(item) for item in raw_distributions
        )
        manifest = SeparatorEnvironmentManifest(
            separator_id=value["separator_id"],
            package_name=value["package_name"],
            package_version=value["package_version"],
            python_implementation=value["python_implementation"],
            python_version=value["python_version"],
            python_abi=value["python_abi"],
            platform=value["platform"],
            interpreter_sha256=value["interpreter_sha256"],
            distributions=distributions,
            sha256=sha256_hex(content),
        )
    except SeparatorLockError:
        raise
    except (KeyError, TypeError, ValueError):
        raise SeparatorLockError(
            "separator lock companion environment fields are invalid"
        ) from None

    if manifest.separator_id != lock.separator_id:
        raise SeparatorLockError("separator lock companion environment separator_id does not match")
    if manifest.package_name != lock.package_name:
        raise SeparatorLockError("separator lock companion environment package_name does not match")
    if manifest.package_version != lock.package_version:
        raise SeparatorLockError(
            "separator lock companion environment package_version does not match"
        )
    if manifest.interpreter_sha256 != lock.interpreter_sha256:
        raise SeparatorLockError(
            "separator lock companion environment interpreter_sha256 does not match"
        )
    return manifest


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
        "interpreter_sha256",
        "environment_manifest_sha256",
        "model_root_kind",
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
    if lock.model_root_kind != policy["model_root_kind"]:
        raise SeparatorLockError("model_root_kind does not match separator model")
    if lock.argv != policy["argv"]:
        raise SeparatorLockError("argv does not match separator model")
    if lock.expected_drum_stem_relative_path != policy["expected_drum_stem_relative_path"]:
        raise SeparatorLockError("expected drum stem path does not match separator model")
    if lock.output_container != policy["output_container"]:
        raise SeparatorLockError("output_container does not match separator model")
    _require_hash(lock.interpreter_sha256, "interpreter_sha256")
    _require_hash(lock.environment_manifest_sha256, "environment_manifest_sha256")
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


def _environment_file_from_value(value: object) -> SeparatorEnvironmentFile:
    if not isinstance(value, dict) or set(value) != _ENVIRONMENT_FILE_KEYS:
        raise SeparatorLockError("environment file must contain the exact key set")
    return SeparatorEnvironmentFile(
        root=value["root"],
        path=value["path"],
        byte_length=value["byte_length"],
        sha256=value["sha256"],
    )


def _environment_distribution_from_value(
    value: object,
) -> SeparatorEnvironmentDistribution:
    if not isinstance(value, dict) or set(value) != _ENVIRONMENT_DISTRIBUTION_KEYS:
        raise SeparatorLockError("environment distribution must contain the exact key set")
    raw_files = value["files"]
    if not isinstance(raw_files, list):
        raise SeparatorLockError("environment distribution files must be a list")
    return SeparatorEnvironmentDistribution(
        name=value["name"],
        version=value["version"],
        files=tuple(_environment_file_from_value(item) for item in raw_files),
    )


def _validate_environment_file(environment_file: SeparatorEnvironmentFile) -> None:
    for field in ("root", "path", "sha256"):
        value = getattr(environment_file, field)
        if not isinstance(value, str) or not value:
            raise SeparatorLockError(f"environment file {field} must be a nonempty string")
    if environment_file.root not in _ENVIRONMENT_ROOT_TAGS:
        raise SeparatorLockError("environment file root is unsupported")
    _validate_environment_relative_path(environment_file.path)
    if (
        isinstance(environment_file.byte_length, bool)
        or not isinstance(environment_file.byte_length, int)
        or environment_file.byte_length < 0
    ):
        raise SeparatorLockError("environment file byte_length must be a nonnegative integer")
    _require_hash(environment_file.sha256, "environment file sha256")


def _validate_environment_distribution(
    distribution: SeparatorEnvironmentDistribution,
) -> None:
    for field in ("name", "version"):
        value = getattr(distribution, field)
        if not isinstance(value, str) or not value:
            raise SeparatorLockError(f"environment distribution {field} must be a nonempty string")
    if _normalize_distribution_name(distribution.name) != distribution.name:
        raise SeparatorLockError("environment distribution name must be normalized")
    if not isinstance(distribution.files, tuple) or not distribution.files:
        raise SeparatorLockError("environment distribution files must be a nonempty tuple")
    if any(not isinstance(item, SeparatorEnvironmentFile) for item in distribution.files):
        raise SeparatorLockError("environment distribution contains an invalid file")
    file_keys = [(item.root, item.path) for item in distribution.files]
    if file_keys != sorted(file_keys) or len(file_keys) != len(set(file_keys)):
        raise SeparatorLockError("environment file tuples must be sorted and unique")


def _validate_environment_manifest(manifest: SeparatorEnvironmentManifest) -> None:
    for field in (
        "separator_id",
        "package_name",
        "package_version",
        "python_implementation",
        "python_version",
        "python_abi",
        "platform",
        "interpreter_sha256",
        "sha256",
    ):
        value = getattr(manifest, field)
        if not isinstance(value, str) or not value:
            raise SeparatorLockError(f"environment manifest {field} must be a nonempty string")
    _require_hash(manifest.interpreter_sha256, "interpreter_sha256")
    _require_hash(manifest.sha256, "environment manifest sha256")
    if not isinstance(manifest.distributions, tuple) or not manifest.distributions:
        raise SeparatorLockError("environment distributions must be a nonempty tuple")
    if any(
        not isinstance(item, SeparatorEnvironmentDistribution) for item in manifest.distributions
    ):
        raise SeparatorLockError("environment manifest contains an invalid distribution")
    names = [distribution.name for distribution in manifest.distributions]
    if names != sorted(names) or len(names) != len(set(names)):
        raise SeparatorLockError("environment distributions must be sorted and unique")


def _normalize_distribution_name(value: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", value).lower()
    if _DISTRIBUTION_NAME_RE.fullmatch(normalized) is None:
        raise SeparatorLockError("environment distribution name is invalid")
    return normalized


def _validate_environment_relative_path(value: object) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SeparatorLockError("environment file path is invalid")
    if (
        value.startswith(("/", "\\"))
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or "\\" in value
    ):
        raise SeparatorLockError("environment file path must be relative")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise SeparatorLockError("environment file path must be normalized")


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


_SPLEETER_MODEL_ROOT_FILES = (
    "4stems/.probe",
    "4stems/checkpoint",
    "4stems/model.data-00000-of-00001",
    "4stems/model.index",
    "4stems/model.meta",
)
_HTDEMUCS_MODEL_ROOT_FILES = (
    "955717e8-8726e21a.th",
    "htdemucs.yaml",
)


def inventory_separator_model_root(
    separator_id: str,
    model_root: Path,
) -> tuple[SeparatorModelFile, ...]:
    """Hash the exact policy-owned model-root layout for one separator."""
    if separator_id not in _SEPARATOR_POLICIES:
        raise SeparatorExecutionError("separator_id_unsupported", "separator is not supported")
    if not isinstance(model_root, Path):
        raise TypeError("model_root must be a Path")
    expected_files = (
        _SPLEETER_MODEL_ROOT_FILES
        if separator_id == SPLEETER_SEPARATOR_ID
        else _HTDEMUCS_MODEL_ROOT_FILES
    )
    try:
        return _inventory_model_root(model_root, expected_files)
    except SeparatorExecutionError:
        raise
    except (OSError, TypeError, ValueError):
        raise SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator model root does not match the fixed policy",
        ) from None


def revalidate_separator_model_root(runtime: AttestedSeparatorRuntime) -> None:
    """Re-inventory an attested model root without reading or writing evidence."""
    if not isinstance(runtime, AttestedSeparatorRuntime):
        raise TypeError("runtime must be an AttestedSeparatorRuntime")
    try:
        inventory = inventory_separator_model_root(runtime.lock.separator_id, runtime.model_root)
    except SeparatorExecutionError:
        raise
    if inventory != runtime.lock.model_files or runtime.model_files != runtime.lock.model_files:
        raise SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator model root does not match the fixed policy",
        )


def _inventory_model_root(
    model_root: Path,
    expected_files: tuple[str, ...],
) -> tuple[SeparatorModelFile, ...]:
    root_metadata = model_root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise OSError("model root is not an ordinary directory")

    expected_names = set(expected_files)
    expected_directories = {""}
    for name in expected_files:
        expected_directories.update(name.rsplit("/", 1)[:1] if "/" in name else ())

    files: dict[str, bytes] = {}
    directories: set[str] = {""}

    def walk(directory: Path, relative_directory: str) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                relative_name = (
                    f"{relative_directory}/{entry.name}" if relative_directory else entry.name
                )
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise OSError("model root contains a symlink")
                if stat.S_ISDIR(metadata.st_mode):
                    directories.add(relative_name)
                    walk(Path(entry.path), relative_name)
                elif stat.S_ISREG(metadata.st_mode):
                    files[relative_name] = _read_model_file_no_follow(Path(entry.path))
                else:
                    raise OSError("model root contains a special file")

    walk(model_root, "")
    if set(files) != expected_names or directories != expected_directories:
        raise OSError("model root file layout does not match policy")
    return tuple(
        SeparatorModelFile(name=name, sha256=sha256_hex(files[name])) for name in sorted(files)
    )


def _read_model_file_no_follow(path: Path) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow | close_on_exec)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("model file is not an ordinary file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise OSError("model file changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _build_separator_launch_environment(
    separator_id: str,
    model_root: Path,
) -> dict[str, str]:
    """Build a policy overlay that removes ambient model-discovery inputs."""
    if separator_id not in _SEPARATOR_POLICIES:
        raise SeparatorExecutionError("separator_id_unsupported", "separator is not supported")
    if not isinstance(model_root, Path):
        raise TypeError("model_root must be a Path")
    policy = _SEPARATOR_POLICIES[separator_id]
    environment = dict(os.environ)
    for key in policy["environment_discovery_keys"]:
        environment.pop(key, None)
    if separator_id == SPLEETER_SEPARATOR_ID:
        environment["MODEL_PATH"] = os.fspath(model_root)
    return environment


def run_spleeter_drums(
    source_audio_path: Path,
    *,
    source_audio_sha256: str,
    source_duration_sec: float,
    runtime: AttestedSeparatorRuntime,
    cache_root: Path,
) -> SeparatedStem:
    """Run the locked Spleeter four-stem runtime and retain its drum WAV."""
    return _run_separator_drums(
        SPLEETER_SEPARATOR_ID,
        source_audio_path,
        source_audio_sha256=source_audio_sha256,
        source_duration_sec=source_duration_sec,
        runtime=runtime,
        cache_root=cache_root,
    )


def run_htdemucs_drums(
    source_audio_path: Path,
    *,
    source_audio_sha256: str,
    source_duration_sec: float,
    runtime: AttestedSeparatorRuntime,
    cache_root: Path,
) -> SeparatedStem:
    """Run the locked standard HTDemucs runtime and retain its drum WAV."""
    return _run_separator_drums(
        HTDEMUCS_SEPARATOR_ID,
        source_audio_path,
        source_audio_sha256=source_audio_sha256,
        source_duration_sec=source_duration_sec,
        runtime=runtime,
        cache_root=cache_root,
    )


def _run_separator_drums(
    separator_id: str,
    source_audio_path: Path,
    *,
    source_audio_sha256: str,
    source_duration_sec: float,
    runtime: AttestedSeparatorRuntime,
    cache_root: Path,
) -> SeparatedStem:
    _validate_execution_inputs(
        separator_id,
        source_audio_path,
        source_audio_sha256,
        source_duration_sec,
        runtime,
        cache_root,
    )
    lock = runtime.lock
    if lock.separator_id != separator_id:
        raise SeparatorExecutionError(
            "separator_lock_mismatch",
            f"lock identifies {lock.separator_id}",
        )
    cache_path = _stem_cache_path(
        cache_root,
        separator_id=separator_id,
        source_audio_sha256=source_audio_sha256,
        separator_lock_sha256=lock.sha256,
    )

    cached_bytes = _read_cached_stem(cache_path)
    if cached_bytes is not None:
        qc = _qc_stem_bytes(cached_bytes, source_duration_sec)
        digest = sha256_hex(cached_bytes)
        return SeparatedStem(
            separator_id=separator_id,
            source_audio_sha256=source_audio_sha256,
            separator_lock_sha256=lock.sha256,
            path=cache_path,
            sha256=digest,
            qc=qc,
            cache_hit=True,
            warnings=qc.warnings,
        )

    try:
        source_bytes = read_regular_file_no_follow(source_audio_path)
    except (OSError, TypeError) as error:
        raise SeparatorExecutionError(
            "source_audio_unavailable",
            "source WAV could not be read",
        ) from error

    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=f"{separator_id}-") as temporary:
            workdir = Path(temporary)
            input_path = workdir / "input.wav"
            output_dir = workdir / "output"
            input_path.write_bytes(source_bytes)
            output_dir.mkdir()
            argv = _render_separator_argv(
                runtime,
                input_path=input_path,
                output_dir=output_dir,
            )
            _run_separator_process(argv, cwd=workdir, env=runtime.launch_environment)
            output_path = output_dir / lock.expected_drum_stem_relative_path
            try:
                stem_bytes = read_regular_file_no_follow(output_path)
            except (OSError, TypeError) as error:
                raise SeparatorExecutionError(
                    "separator_output_missing",
                    "expected drum stem WAV is unavailable",
                ) from error
            qc = _qc_stem_bytes(stem_bytes, source_duration_sec)
    except SeparatorExecutionError:
        raise
    except OSError as error:
        raise SeparatorExecutionError(
            "separator_staging_failed",
            "private separator staging failed",
        ) from error

    try:
        publish_immutable_file(cache_path, stem_bytes)
    except ArtifactPublicationError as error:
        raise SeparatorExecutionError(
            "stem_publish_conflict",
            "immutable native stem publication conflicted",
        ) from error
    except (OSError, TypeError) as error:
        raise SeparatorExecutionError(
            "stem_publish_failed",
            "immutable native stem publication failed",
        ) from error

    digest = sha256_hex(stem_bytes)
    return SeparatedStem(
        separator_id=separator_id,
        source_audio_sha256=source_audio_sha256,
        separator_lock_sha256=lock.sha256,
        path=cache_path,
        sha256=digest,
        qc=qc,
        cache_hit=False,
        warnings=qc.warnings,
    )


def _validate_execution_inputs(
    separator_id: str,
    source_audio_path: Path,
    source_audio_sha256: str,
    source_duration_sec: float,
    runtime: AttestedSeparatorRuntime,
    cache_root: Path,
) -> None:
    if separator_id not in _SEPARATOR_POLICIES:
        raise SeparatorExecutionError("separator_id_unsupported", "separator is not supported")
    if not isinstance(source_audio_path, Path):
        raise TypeError("source_audio_path must be a Path")
    if not isinstance(runtime, AttestedSeparatorRuntime):
        raise TypeError("runtime must be an AttestedSeparatorRuntime")
    if not isinstance(cache_root, Path):
        raise TypeError("cache_root must be a Path")
    try:
        require_sha256(source_audio_sha256, "source_audio_sha256")
    except (StrictJsonError, TypeError) as error:
        raise SeparatorExecutionError(
            "source_audio_identity_invalid",
            "source_audio_sha256 must be lowercase SHA-256",
        ) from error
    if (
        isinstance(source_duration_sec, bool)
        or not isinstance(source_duration_sec, (int, float))
        or not math.isfinite(float(source_duration_sec))
        or float(source_duration_sec) <= 0
    ):
        raise SeparatorExecutionError(
            "source_duration_invalid",
            "source_duration_sec must be finite and positive",
        )


def _stem_cache_path(
    cache_root: Path,
    *,
    separator_id: str,
    source_audio_sha256: str,
    separator_lock_sha256: str,
) -> Path:
    return (
        cache_root
        / "derived"
        / "stems"
        / separator_id
        / source_audio_sha256
        / separator_lock_sha256
        / "drums.wav"
    )


def _read_cached_stem(path: Path) -> bytes | None:
    try:
        return read_regular_file_no_follow(path)
    except FileNotFoundError:
        return None
    except (OSError, TypeError) as error:
        raise SeparatorExecutionError(
            "stem_cache_unavailable",
            "cached native stem is not a readable regular file",
        ) from error


def _render_separator_argv(
    runtime: AttestedSeparatorRuntime,
    *,
    input_path: Path,
    output_dir: Path,
) -> list[str]:
    if not isinstance(runtime, AttestedSeparatorRuntime):
        raise TypeError("runtime must be an AttestedSeparatorRuntime")
    replacements = {
        "{input_wav}": os.fspath(input_path),
        "{output_dir}": os.fspath(output_dir),
        "{model_root}": os.fspath(runtime.model_root),
    }
    return [
        os.fspath(runtime.interpreter),
        *(replacements.get(argument, argument) for argument in runtime.lock.argv),
    ]


def _run_separator_process(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> None:
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
    except OSError as error:
        raise SeparatorExecutionError(
            "separator_start_failed",
            "separator interpreter could not be started",
        ) from error

    try:
        process.communicate(timeout=SEPARATOR_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        _stop_separator_process_group(process)
        raise SeparatorExecutionError(
            "separator_timeout",
            "separator exceeded the fixed execution timeout",
        ) from error
    if process.returncode != 0:
        raise SeparatorExecutionError(
            "separator_nonzero_exit",
            f"separator exited with status {process.returncode}",
        )


def _stop_separator_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    leader_reaped = True
    try:
        process.wait(timeout=SEPARATOR_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        leader_reaped = False
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    if not leader_reaped:
        try:
            process.wait()
        except (OSError, subprocess.TimeoutExpired):
            pass


def _qc_stem_bytes(content: bytes, source_duration_sec: float) -> StemQc:
    try:
        samples, sample_rate = soundfile.read(BytesIO(content), always_2d=True, dtype="float32")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SeparatorExecutionError(
            "stem_decode_failed",
            "native drum stem WAV could not be decoded",
        ) from error
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise SeparatorExecutionError(
            "stem_channel_count",
            "decoded drum stem must be a two-dimensional waveform",
        )
    channel_count = int(samples.shape[1])
    if channel_count <= 0 or channel_count > 2:
        raise SeparatorExecutionError(
            "stem_channel_count",
            "decoded drum stem must have one or two channels",
        )
    if not np.isfinite(samples).all():
        raise SeparatorExecutionError(
            "stem_nonfinite",
            "decoded drum stem contains nonfinite samples",
        )
    frame_count = int(samples.shape[0])
    if frame_count <= 0 or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise SeparatorExecutionError(
            "stem_duration_invalid",
            "decoded drum stem has no positive duration",
        )
    duration_sec = frame_count / int(sample_rate)
    duration_delta = abs(duration_sec - source_duration_sec)
    duration_limit = max(
        STEM_MAX_DURATION_DELTA_SECONDS,
        source_duration_sec * STEM_MAX_DURATION_DELTA_RATIO,
    )
    if duration_delta > duration_limit:
        raise SeparatorExecutionError(
            "stem_duration_mismatch",
            "decoded drum stem duration differs from source duration",
        )

    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    rms_dbfs = -math.inf if rms <= 0 else 20.0 * math.log10(rms)
    if not rms_dbfs > STEM_NEAR_SILENT_DBFS:
        raise SeparatorExecutionError(
            "stem_near_silent",
            "decoded drum stem RMS is at or below the near-silent threshold",
        )
    peak_abs = float(np.max(np.abs(samples)))
    clipping_detected = peak_abs >= STEM_CLIP_ABS
    warnings = ("stem_clipping",) if clipping_detected else ()
    return StemQc(
        sample_rate=int(sample_rate),
        frame_count=frame_count,
        channel_count=channel_count,
        duration_sec=duration_sec,
        rms_dbfs=rms_dbfs,
        peak_abs=peak_abs,
        clipping_detected=clipping_detected,
        warnings=warnings,
    )


__all__ = [
    "AttestedSeparatorRuntime",
    "HTDEMUCS_SEPARATOR_ID",
    "SEPARATOR_ENVIRONMENT_SCHEMA",
    "SEPARATOR_LOCK_SCHEMA",
    "SEPARATOR_TERMINATE_GRACE_SECONDS",
    "SEPARATOR_TIMEOUT_SECONDS",
    "SPLEETER_SEPARATOR_ID",
    "STEM_CLIP_ABS",
    "STEM_MAX_DURATION_DELTA_RATIO",
    "STEM_MAX_DURATION_DELTA_SECONDS",
    "STEM_NEAR_SILENT_DBFS",
    "SeparatedStem",
    "SeparatorEnvironmentDistribution",
    "SeparatorEnvironmentFile",
    "SeparatorEnvironmentManifest",
    "SeparatorLock",
    "SeparatorLockError",
    "SeparatorModelFile",
    "SeparatorExecutionError",
    "StemQc",
    "inventory_separator_model_root",
    "load_separator_environment_manifest",
    "load_separator_lock",
    "revalidate_separator_model_root",
    "separator_environment_manifest_payload",
    "run_htdemucs_drums",
    "run_spleeter_drums",
    "separator_lock_payload",
]
