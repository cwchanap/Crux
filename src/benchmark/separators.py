"""Strict identity and canonical lock loading for the HPA-328 separators."""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import weakref
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType

import numpy as np
import soundfile

from src.benchmark.artifact_io import (
    ArtifactPublicationError,
    publish_immutable_file,
    read_regular_file_no_follow,
)
from src.benchmark.backend_identity import (
    StrictJsonError,
    canonical_json_bytes,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)

SEPARATOR_LOCK_SCHEMA = "crux.separator-lock/v2"
SEPARATOR_ENVIRONMENT_SCHEMA = "crux.separator-environment/v1"
ATTESTATION_FAILURE_CODES = frozenset(
    {
        "separator_lock_companion_mismatch",
        "separator_interpreter_mismatch",
        "separator_environment_mismatch",
        "separator_model_root_invalid",
        "separator_environment_probe_failed",
    }
)
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
_PYTHON_IMPORT_DISCOVERY_ENVIRONMENT_KEYS = frozenset(
    {
        "PYTHONCASEOK",
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTHONPLATLIBDIR",
        "PYTHONSAFEPATH",
        "PYTHONUSERBASE",
    }
)

# Commands intentionally contain only interpreter-independent arguments.  The
# freeze script records the interpreter separately, while the runner supplies
# the private path placeholders in a temporary working directory.
_SEPARATOR_POLICIES = {
    SPLEETER_SEPARATOR_ID: {
        "repository_url": "https://github.com/deezer/spleeter",
        "package_name": "spleeter",
        "package_version": "2.4.2",
        "model_id": "spleeter:4stems",
        "model_root_kind": "spleeter-model-path-v1",
        "code_license": "MIT",
        "model_license": "MIT",
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
        "package_version": "4.1.0",
        "model_id": "htdemucs",
        "model_root_kind": "demucs-local-repo-v1",
        "code_license": "MIT",
        "model_license": "MIT",
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
    model_root_fd: int | None = dataclass_field(default=None, init=False, repr=False, compare=False)
    model_root_launch_path: Path | None = dataclass_field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _model_root_finalizer: weakref.finalize | None = dataclass_field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def close(self) -> None:
        """Release the descriptor held to bind this runtime's model root."""
        finalizer = self._model_root_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer()
        object.__setattr__(self, "model_root_fd", None)
        object.__setattr__(self, "model_root_launch_path", None)
        object.__setattr__(self, "_model_root_finalizer", None)


_ATTESTED_RUNTIME_REGISTRY: weakref.WeakValueDictionary[int, AttestedSeparatorRuntime] = (
    weakref.WeakValueDictionary()
)


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


def _parse_separator_environment_manifest(content: bytes) -> SeparatorEnvironmentManifest:
    """Parse one canonical environment manifest from bytes without filesystem access."""
    if not isinstance(content, bytes):
        raise TypeError("environment manifest content must be bytes")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise SeparatorLockError("separator environment must have one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise SeparatorLockError(f"separator environment is invalid: {error}") from None
    if not isinstance(value, dict) or set(value) != _ENVIRONMENT_KEYS:
        raise SeparatorLockError("separator environment must contain the exact key set")
    if value.get("schema") != SEPARATOR_ENVIRONMENT_SCHEMA:
        raise SeparatorLockError("separator environment schema is invalid")

    try:
        raw_distributions = value["distributions"]
        if not isinstance(raw_distributions, list):
            raise SeparatorLockError("environment distributions must be a list")
        distributions = tuple(
            _environment_distribution_from_value(item) for item in raw_distributions
        )
        return SeparatorEnvironmentManifest(
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
        raise SeparatorLockError("separator environment fields are invalid") from None


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
    try:
        manifest = _parse_separator_environment_manifest(content)
    except SeparatorLockError as error:
        raise SeparatorLockError(
            f"separator lock companion environment is invalid: {error}"
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


def _resolve_separator_interpreter(interpreter: Path) -> tuple[Path, str]:
    """Resolve one regular interpreter and hash the resolved executable bytes."""
    if not isinstance(interpreter, Path):
        raise TypeError("interpreter must be a Path")
    try:
        resolved = interpreter.resolve(strict=True)
        digest = sha256_hex(read_regular_file_no_follow(resolved))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SeparatorExecutionError(
            "separator_interpreter_mismatch",
            "separator interpreter does not match the frozen executable",
        ) from error
    return resolved, digest


def _run_separator_environment_probe(
    interpreter: Path,
) -> SeparatorEnvironmentManifest:
    """Run and parse the standalone stdlib environment probe."""
    if not isinstance(interpreter, Path):
        raise TypeError("interpreter must be a Path")
    probe_path = Path(__file__).with_name("separator_environment_probe.py")
    try:
        result = subprocess.run(
            [os.fspath(interpreter), "-I", os.fspath(probe_path)],
            check=False,
            capture_output=True,
            env=_isolated_python_environment(),
            timeout=SEPARATOR_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SeparatorExecutionError(
            "separator_environment_probe_failed",
            "separator environment probe could not be started",
        ) from error
    if result.returncode != 0:
        raise SeparatorExecutionError(
            "separator_environment_probe_failed",
            "separator environment probe did not complete cleanly",
        )
    try:
        return _parse_separator_environment_manifest(result.stdout)
    except (SeparatorLockError, TypeError, ValueError) as error:
        raise SeparatorExecutionError(
            "separator_environment_probe_failed",
            "separator environment probe output is not canonical",
        ) from error


def _isolated_python_environment() -> dict[str, str]:
    """Remove parent variables that can alter Python import discovery."""
    environment = dict(os.environ)
    for key in _PYTHON_IMPORT_DISCOVERY_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    return environment


def _require_absolute_model_root(model_root: Path) -> Path:
    """Require and return the caller's lexical absolute model-root path."""
    if not isinstance(model_root, Path):
        raise TypeError("model_root must be a Path")
    if not model_root.is_absolute():
        raise SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator model root must be an absolute path",
        )
    return model_root


def attest_separator_runtime(
    lock_path: Path,
    interpreter: Path,
    model_root: Path,
) -> AttestedSeparatorRuntime:
    """Verify all lock-bound separator inputs before constructing a runtime."""
    if not isinstance(lock_path, Path):
        raise TypeError("lock_path must be a Path")
    if not isinstance(interpreter, Path):
        raise TypeError("interpreter must be a Path")
    _require_absolute_model_root(model_root)

    try:
        lock = load_separator_lock(lock_path)
        environment = load_separator_environment_manifest(lock_path, lock)
    except (OSError, TypeError, SeparatorLockError) as error:
        raise SeparatorExecutionError(
            "separator_lock_companion_mismatch",
            "separator lock or companion environment is not valid",
        ) from error

    resolved_interpreter, interpreter_sha256 = _resolve_separator_interpreter(interpreter)
    if interpreter_sha256 != lock.interpreter_sha256:
        raise SeparatorExecutionError(
            "separator_interpreter_mismatch",
            "separator interpreter does not match the frozen executable",
        )

    try:
        probed_environment = _run_separator_environment_probe(resolved_interpreter)
    except SeparatorExecutionError as error:
        if error.code == "separator_environment_probe_failed":
            raise
        raise SeparatorExecutionError(
            "separator_environment_probe_failed",
            "separator environment probe failed",
        ) from error
    except (OSError, TypeError, ValueError) as error:
        raise SeparatorExecutionError(
            "separator_environment_probe_failed",
            "separator environment probe failed",
        ) from error
    if not isinstance(probed_environment, SeparatorEnvironmentManifest):
        raise SeparatorExecutionError(
            "separator_environment_probe_failed",
            "separator environment probe output is invalid",
        )
    if probed_environment != environment:
        raise SeparatorExecutionError(
            "separator_environment_mismatch",
            "separator environment differs from the frozen companion",
        )

    try:
        expected_files = (
            _SPLEETER_MODEL_ROOT_FILES
            if lock.separator_id == SPLEETER_SEPARATOR_ID
            else _HTDEMUCS_MODEL_ROOT_FILES
        )
        model_files, model_root_fd = _inventory_model_root_bound(model_root, expected_files)
    except SeparatorExecutionError as error:
        if error.code == "separator_model_root_invalid":
            raise
        raise SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator model root does not match the fixed policy",
        ) from error
    except (OSError, TypeError, ValueError) as error:
        raise SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator model root does not match the fixed policy",
        ) from error
    if model_files != lock.model_files:
        _close_model_root_fd(model_root_fd)
        raise SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator model root does not match the frozen lock",
        )

    try:
        return _build_attested_separator_runtime(
            interpreter=resolved_interpreter,
            lock=lock,
            model_root=model_root,
            model_files=model_files,
            environment=probed_environment,
            model_root_fd=model_root_fd,
        )
    except SeparatorExecutionError as error:
        if error.code in ATTESTATION_FAILURE_CODES:
            raise
        raise SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator runtime inputs could not be constructed",
        ) from error
    except (OSError, TypeError, ValueError) as error:
        raise SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator runtime inputs could not be constructed",
        ) from error


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
    if lock.package_version != policy["package_version"]:
        raise SeparatorLockError("package_version does not match separator_id")
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
    _require_absolute_model_root(model_root)
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


_RUNTIME_PUBLICATION_LOCK_NAME = ".separator-publish.lock"


@contextlib.contextmanager
def _runtime_publication_lock(output: Path) -> Iterator[None]:
    """Serialize multi-file runtime publication under one exclusive lock.

    The lock+manifest pair must be published atomically so that a concurrent
    freeze with the same environment but a different revision cannot observe a
    stale ``environment_preexisting`` value and delete a manifest that belongs
    to the winning freeze.
    """
    try:
        import fcntl  # noqa: PLC0415 — lazy stdlib import; Unix-only
    except ImportError as error:  # pragma: no cover — non-Unix platform
        raise SeparatorExecutionError(
            "separator_lock_publication_failed",
            "separator publication locks are unsupported on this platform",
        ) from error
    lock_path = output.parent / _RUNTIME_PUBLICATION_LOCK_NAME
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | no_follow | close_on_exec,
            0o600,
        )
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = lock_path.lstat()
        if (
            not stat.S_ISREG(descriptor_metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or descriptor_metadata.st_dev != path_metadata.st_dev
            or descriptor_metadata.st_ino != path_metadata.st_ino
        ):
            raise OSError("publication lock is unavailable")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as error:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise SeparatorExecutionError(
            "separator_lock_publication_failed",
            "separator publication lock could not be acquired",
        ) from error
    try:
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def freeze_separator_runtime(
    *,
    separator_id: str,
    interpreter: Path,
    model_root: Path,
    repository_revision: str,
    output: Path,
) -> SeparatorLock:
    """Attest and publish one policy-owned separator runtime as a fixed pair."""
    if not isinstance(interpreter, Path):
        raise TypeError("interpreter must be a Path")
    if not isinstance(model_root, Path):
        raise TypeError("model_root must be a Path")
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")
    _require_absolute_model_root(model_root)
    policy = _SEPARATOR_POLICIES.get(separator_id)
    if policy is None:
        raise ValueError("separator_id is unsupported")
    _require_revision(repository_revision)

    resolved_interpreter, interpreter_sha256 = _resolve_separator_interpreter(interpreter)
    environment = _run_separator_environment_probe(resolved_interpreter)
    if (
        environment.separator_id != separator_id
        or environment.package_name != policy["package_name"]
        or environment.package_version != policy["package_version"]
        or environment.interpreter_sha256 != interpreter_sha256
    ):
        raise SeparatorExecutionError(
            "separator_environment_mismatch",
            "separator environment does not match the selected policy",
        )
    model_files = inventory_separator_model_root(separator_id, model_root)

    environment_bytes = canonical_json_bytes(
        separator_environment_manifest_payload(environment),
        trailing_newline=True,
    )
    environment_manifest_sha256 = sha256_hex(environment_bytes)
    lock_payload = {
        "schema": SEPARATOR_LOCK_SCHEMA,
        "separator_id": separator_id,
        "repository_url": policy["repository_url"],
        "repository_revision": repository_revision,
        "package_name": environment.package_name,
        "package_version": environment.package_version,
        "model_id": policy["model_id"],
        "model_files": [
            {"name": model_file.name, "sha256": model_file.sha256} for model_file in model_files
        ],
        "code_license": policy["code_license"],
        "model_license": policy["model_license"],
        "argv": list(policy["argv"]),
        "expected_drum_stem_relative_path": policy["expected_drum_stem_relative_path"],
        "output_container": policy["output_container"],
        "interpreter_sha256": interpreter_sha256,
        "environment_manifest_sha256": environment_manifest_sha256,
        "model_root_kind": policy["model_root_kind"],
    }
    lock_bytes = canonical_json_bytes(lock_payload, trailing_newline=True)
    environment_path = output.parent / "environment.json"
    # The exists-check and both publishes are serialized under one per-runtime
    # lock so that the lock+manifest pair is the atomic publication unit.  A
    # concurrent freeze with the same environment but a different revision
    # cannot observe a stale environment_preexisting value and delete a manifest
    # that belongs to the winning freeze.
    with _runtime_publication_lock(output):
        environment_preexisting = environment_path.exists()
        try:
            publish_immutable_file(environment_path, environment_bytes)
            publish_immutable_file(output, lock_bytes)
        except (ArtifactPublicationError, OSError, TypeError) as error:
            # Only clean up environment.json when this invocation created it.
            # publish_immutable_file treats identical existing bytes as reuse, so
            # a pre-existing manifest must never be deleted when the lock publish
            # conflicts — that would corrupt a previously valid runtime directory.
            if not environment_preexisting:
                try:
                    environment_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise SeparatorExecutionError(
                "separator_lock_publication_failed",
                "separator lock publication failed",
            ) from error

    attested = attest_separator_runtime(output, interpreter, model_root)
    try:
        if attested.lock.separator_id != separator_id:
            raise SeparatorExecutionError(
                "separator_lock_companion_mismatch",
                "published separator lock did not round-trip",
            )
        return attested.lock
    finally:
        attested.close()


def revalidate_separator_model_root(runtime: AttestedSeparatorRuntime) -> None:
    """Re-inventory an attested model root without reading or writing evidence."""
    _require_attested_runtime(runtime)
    if not isinstance(runtime.model_root_fd, int):
        raise SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator model root descriptor is unavailable",
        )
    try:
        expected_files = (
            _SPLEETER_MODEL_ROOT_FILES
            if runtime.lock.separator_id == SPLEETER_SEPARATOR_ID
            else _HTDEMUCS_MODEL_ROOT_FILES
        )
        inventory, _ = _inventory_model_root_bound(
            runtime.model_root,
            expected_files,
            root_fd=runtime.model_root_fd,
        )
    except SeparatorExecutionError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator model root does not match the fixed policy",
        ) from error
    if inventory != runtime.lock.model_files or runtime.model_files != runtime.lock.model_files:
        raise SeparatorExecutionError(
            "separator_model_root_invalid",
            "separator model root does not match the fixed policy",
        )


def _inventory_model_root(
    model_root: Path,
    expected_files: tuple[str, ...],
) -> tuple[SeparatorModelFile, ...]:
    inventory, descriptor = _inventory_model_root_bound(model_root, expected_files)
    _close_model_root_fd(descriptor)
    return inventory


def _inventory_model_root_bound(
    model_root: Path,
    expected_files: tuple[str, ...],
    *,
    root_fd: int | None = None,
) -> tuple[tuple[SeparatorModelFile, ...], int]:
    """Inventory a root while optionally retaining its bound directory fd."""
    no_follow, directory_flag, close_on_exec = _model_root_descriptor_flags()
    expected_names = set(expected_files)
    expected_directories = {""}
    for name in expected_files:
        if "/" in name:
            parts = name.split("/")
            for i in range(len(parts) - 1):
                expected_directories.add("/".join(parts[: i + 1]))
    owns_descriptor = root_fd is None
    try:
        if root_fd is None:
            root_fd = _open_model_root_path(
                model_root,
                directory_flag=directory_flag,
                no_follow=no_follow,
                close_on_exec=close_on_exec,
            )
        if not isinstance(root_fd, int):
            raise OSError("model root descriptor is unavailable")
        opened_root_metadata = os.fstat(root_fd)
        if not stat.S_ISDIR(opened_root_metadata.st_mode):
            raise OSError("model root is not an ordinary directory")
        files: dict[str, str] = {}
        directories: set[str] = {""}

        def walk(directory_fd: int, relative_directory: str) -> None:
            for entry_name in sorted(os.listdir(directory_fd)):
                if not isinstance(entry_name, str) or not entry_name:
                    raise OSError("model root entry name is invalid")
                relative_name = (
                    f"{relative_directory}/{entry_name}" if relative_directory else entry_name
                )
                entry_metadata = os.stat(
                    entry_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(entry_metadata.st_mode):
                    raise OSError("model root contains a symlink")
                if stat.S_ISDIR(entry_metadata.st_mode):
                    child_fd = os.open(
                        entry_name,
                        os.O_RDONLY | directory_flag | no_follow | close_on_exec,
                        dir_fd=directory_fd,
                    )
                    try:
                        opened_child_metadata = os.fstat(child_fd)
                        _require_same_model_identity(entry_metadata, opened_child_metadata)
                        directories.add(relative_name)
                        walk(child_fd, relative_name)
                        _require_same_model_state(
                            opened_child_metadata,
                            os.fstat(child_fd),
                        )
                        _require_same_model_identity(
                            opened_child_metadata,
                            os.stat(
                                entry_name,
                                dir_fd=directory_fd,
                                follow_symlinks=False,
                            ),
                        )
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(entry_metadata.st_mode):
                    file_fd = os.open(
                        entry_name,
                        os.O_RDONLY | no_follow | close_on_exec,
                        dir_fd=directory_fd,
                    )
                    try:
                        opened_file_metadata = os.fstat(file_fd)
                        _require_same_model_identity(entry_metadata, opened_file_metadata)
                        _, digest = _hash_model_file_descriptor(file_fd, opened_file_metadata)
                        _require_same_model_identity(
                            opened_file_metadata,
                            os.stat(
                                entry_name,
                                dir_fd=directory_fd,
                                follow_symlinks=False,
                            ),
                        )
                        files[relative_name] = digest
                    finally:
                        os.close(file_fd)
                else:
                    raise OSError("model root contains a special file")

        walk(root_fd, "")
        _require_same_model_state(opened_root_metadata, os.fstat(root_fd))
        if set(files) != expected_names or directories != expected_directories:
            raise OSError("model root file layout does not match policy")
        return (
            tuple(SeparatorModelFile(name=name, sha256=files[name]) for name in sorted(files)),
            root_fd,
        )
    except BaseException:
        if owns_descriptor and isinstance(root_fd, int):
            _close_model_root_fd(root_fd)
        raise


def _open_model_root_path(
    model_root: Path,
    *,
    directory_flag: int,
    no_follow: int,
    close_on_exec: int,
) -> int:
    """Open every absolute model-root path component without following aliases."""
    if not isinstance(model_root, Path) or not model_root.is_absolute():
        raise OSError("model root must be absolute")
    parts = model_root.parts
    if os.name == "nt" or not parts or parts[0] != os.path.sep:
        raise OSError("descriptor-relative model paths are unavailable")
    flags = os.O_RDONLY | directory_flag | no_follow | close_on_exec
    descriptor = os.open(os.path.sep, flags)
    try:
        for part in parts[1:]:
            if not part or part in {".", "..", os.path.sep}:
                raise OSError("model root path is not normalized")
            child_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child_descriptor
        return descriptor
    except BaseException:
        _close_model_root_fd(descriptor)
        raise


def _model_root_descriptor_flags() -> tuple[int, int, int]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if (
        not isinstance(no_follow, int)
        or no_follow == 0
        or not isinstance(directory_flag, int)
        or directory_flag == 0
    ):
        raise OSError("descriptor no-follow model inventory is unavailable")
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", ())
    supports_fd = getattr(os, "supports_fd", ())
    if os.open not in supports_dir_fd or os.stat not in supports_dir_fd:
        raise OSError("descriptor-relative model inventory is unavailable")
    if os.stat not in supports_follow_symlinks or os.listdir not in supports_fd:
        raise OSError("descriptor no-follow model inventory is unavailable")
    return no_follow, directory_flag, getattr(os, "O_CLOEXEC", 0)


def _close_model_root_fd(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _model_root_launch_path(descriptor: int) -> Path:
    """Return a child-visible path that resolves through a held directory fd."""
    if os.name != "posix":
        raise OSError("descriptor-backed model-root paths are unavailable")
    if sys.platform == "darwin":
        # macOS exposes directory descriptors through /dev/fd, but does not
        # permit path traversal below that symlink.  The launcher therefore
        # fchdirs to the held descriptor in the child and uses '.'.
        return Path(".")
    for prefix in (Path("/proc/self/fd"), Path("/dev/fd")):
        if prefix.is_dir():
            return prefix / str(descriptor)
    raise OSError("descriptor-backed model-root paths are unavailable")


def _model_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


def _model_state(metadata: os.stat_result) -> tuple[tuple[int, int, int], int, int, int]:
    return (
        _model_identity(metadata),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_same_model_identity(
    expected: os.stat_result,
    observed: os.stat_result,
) -> None:
    if _model_identity(expected) != _model_identity(observed):
        raise OSError("model root entry changed during inventory")


def _require_same_model_state(
    expected: os.stat_result,
    observed: os.stat_result,
) -> None:
    if _model_state(expected) != _model_state(observed):
        raise OSError("model root entry changed during inventory")


def _hash_model_file_descriptor(
    descriptor: int,
    opened_metadata: os.stat_result,
) -> tuple[int, str]:
    if not stat.S_ISREG(opened_metadata.st_mode):
        raise OSError("model file is not an ordinary file")
    digest = hashlib.sha256()
    byte_length = 0
    while True:
        content = os.read(descriptor, 1024 * 1024)
        if not content:
            break
        digest.update(content)
        byte_length += len(content)
    after_metadata = os.fstat(descriptor)
    _require_same_model_state(opened_metadata, after_metadata)
    if byte_length != after_metadata.st_size:
        raise OSError("model file changed while it was read")
    return byte_length, digest.hexdigest()


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
    for key in _PYTHON_IMPORT_DISCOVERY_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    for key in policy["environment_discovery_keys"]:
        environment.pop(key, None)
    if separator_id == SPLEETER_SEPARATOR_ID:
        environment["MODEL_PATH"] = os.fspath(model_root)
    return environment


def _build_attested_separator_runtime(
    *,
    interpreter: Path,
    lock: SeparatorLock,
    model_root: Path,
    model_files: tuple[SeparatorModelFile, ...],
    environment: SeparatorEnvironmentManifest,
    model_root_fd: int | None = None,
) -> AttestedSeparatorRuntime:
    """Construct a runtime only after its lock-bound inputs pass local checks."""
    runtime_descriptor = model_root_fd
    try:
        if not isinstance(interpreter, Path):
            raise TypeError("interpreter must be a Path")
        if not isinstance(lock, SeparatorLock):
            raise TypeError("lock must be a SeparatorLock")
        if not isinstance(model_root, Path):
            raise TypeError("model_root must be a Path")
        if not isinstance(model_files, tuple) or any(
            not isinstance(item, SeparatorModelFile) for item in model_files
        ):
            raise TypeError("model_files must be a tuple of SeparatorModelFile")
        if not isinstance(environment, SeparatorEnvironmentManifest):
            raise TypeError("environment must be a SeparatorEnvironmentManifest")
        if (
            model_files != lock.model_files
            or environment.sha256 != lock.environment_manifest_sha256
            or environment.separator_id != lock.separator_id
            or environment.package_name != lock.package_name
            or environment.package_version != lock.package_version
            or environment.interpreter_sha256 != lock.interpreter_sha256
        ):
            raise SeparatorExecutionError(
                "separator_runtime_unattested",
                "separator runtime inputs do not match the v2 lock",
            )
        if runtime_descriptor is None:
            inventory, runtime_descriptor = _inventory_model_root_bound(
                model_root,
                (
                    _SPLEETER_MODEL_ROOT_FILES
                    if lock.separator_id == SPLEETER_SEPARATOR_ID
                    else _HTDEMUCS_MODEL_ROOT_FILES
                ),
            )
        else:
            inventory = model_files
        if not isinstance(runtime_descriptor, int):
            raise OSError("separator model root descriptor is unavailable")
        if inventory != lock.model_files:
            raise SeparatorExecutionError(
                "separator_runtime_unattested",
                "separator model root does not match the v2 lock",
            )
        launch_path = _model_root_launch_path(runtime_descriptor)
        runtime = AttestedSeparatorRuntime(
            interpreter=interpreter,
            lock=lock,
            model_root=model_root,
            model_files=inventory,
            environment=environment,
            launch_environment=MappingProxyType(
                _build_separator_launch_environment(lock.separator_id, launch_path)
            ),
        )
        object.__setattr__(runtime, "model_root_fd", runtime_descriptor)
        object.__setattr__(runtime, "model_root_launch_path", launch_path)
        object.__setattr__(
            runtime,
            "_model_root_finalizer",
            weakref.finalize(runtime, _close_model_root_fd, runtime_descriptor),
        )
        _ATTESTED_RUNTIME_REGISTRY[id(runtime)] = runtime
        return runtime
    except BaseException:
        if isinstance(runtime_descriptor, int):
            _close_model_root_fd(runtime_descriptor)
        raise


def _require_attested_runtime(runtime: AttestedSeparatorRuntime) -> None:
    if not isinstance(runtime, AttestedSeparatorRuntime):
        raise TypeError("runtime must be an AttestedSeparatorRuntime")
    if _ATTESTED_RUNTIME_REGISTRY.get(id(runtime)) is not runtime:
        raise SeparatorExecutionError(
            "separator_runtime_unattested",
            "separator runtime was not constructed by the attestation gate",
        )
    if (
        not isinstance(runtime.model_root_fd, int)
        or runtime._model_root_finalizer is None
        or not runtime._model_root_finalizer.alive
    ):
        raise SeparatorExecutionError(
            "separator_runtime_unattested",
            "separator runtime model root descriptor is closed",
        )


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
            pass_fds = (runtime.model_root_fd,) if isinstance(runtime.model_root_fd, int) else ()
            _run_separator_process(
                argv,
                cwd=workdir,
                env=runtime.launch_environment,
                pass_fds=pass_fds,
                model_root_cwd_fd=(
                    runtime.model_root_fd if runtime.model_root_launch_path == Path(".") else None
                ),
            )
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
    _require_attested_runtime(runtime)
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
    _require_attested_runtime(runtime)
    model_root = runtime.model_root_launch_path or runtime.model_root
    replacements = {
        "{input_wav}": os.fspath(input_path),
        "{output_dir}": os.fspath(output_dir),
        "{model_root}": os.fspath(model_root),
    }
    return [
        os.fspath(runtime.interpreter),
        "-I",
        *(replacements.get(argument, argument) for argument in runtime.lock.argv),
    ]


def _run_separator_process(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    pass_fds: tuple[int, ...] = (),
    model_root_cwd_fd: int | None = None,
) -> None:
    popen_options: dict[str, object] = {
        "cwd": cwd,
        "env": dict(env),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": False,
        "start_new_session": True,
        "pass_fds": pass_fds,
    }
    if model_root_cwd_fd is not None:
        popen_options["preexec_fn"] = lambda: os.fchdir(model_root_cwd_fd)
    try:
        process = subprocess.Popen(argv, **popen_options)
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
    "ATTESTATION_FAILURE_CODES",
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
    "attest_separator_runtime",
    "freeze_separator_runtime",
    "inventory_separator_model_root",
    "load_separator_environment_manifest",
    "load_separator_lock",
    "revalidate_separator_model_root",
    "separator_environment_manifest_payload",
    "run_htdemucs_drums",
    "run_spleeter_drums",
    "separator_lock_payload",
]
