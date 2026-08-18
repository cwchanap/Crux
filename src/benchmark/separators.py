"""Strict identity and canonical lock loading for the HPA-328 separators."""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import tempfile
from dataclasses import dataclass
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

SEPARATOR_LOCK_SCHEMA = "crux.separator-lock/v1"
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

    @property
    def duration_seconds(self) -> float:
        """Compatibility spelling for callers that use seconds in field names."""
        return self.duration_sec

    @property
    def clipped(self) -> bool:
        return self.clipping_detected


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

    @property
    def stem_path(self) -> Path:
        return self.path

    @property
    def native_wav_path(self) -> Path:
        return self.path

    @property
    def source_sha256(self) -> str:
        return self.source_audio_sha256

    @property
    def lock_sha256(self) -> str:
        return self.separator_lock_sha256


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


def run_spleeter_drums(
    source_audio_path: Path,
    *,
    source_audio_sha256: str,
    source_duration_sec: float,
    interpreter: Path,
    lock_path: Path,
    cache_root: Path,
) -> SeparatedStem:
    """Run the locked Spleeter four-stem runtime and retain its drum WAV."""
    return _run_separator_drums(
        SPLEETER_SEPARATOR_ID,
        source_audio_path,
        source_audio_sha256=source_audio_sha256,
        source_duration_sec=source_duration_sec,
        interpreter=interpreter,
        lock_path=lock_path,
        cache_root=cache_root,
    )


def run_htdemucs_drums(
    source_audio_path: Path,
    *,
    source_audio_sha256: str,
    source_duration_sec: float,
    interpreter: Path,
    lock_path: Path,
    cache_root: Path,
) -> SeparatedStem:
    """Run the locked standard HTDemucs runtime and retain its drum WAV."""
    return _run_separator_drums(
        HTDEMUCS_SEPARATOR_ID,
        source_audio_path,
        source_audio_sha256=source_audio_sha256,
        source_duration_sec=source_duration_sec,
        interpreter=interpreter,
        lock_path=lock_path,
        cache_root=cache_root,
    )


def _run_separator_drums(
    separator_id: str,
    source_audio_path: Path,
    *,
    source_audio_sha256: str,
    source_duration_sec: float,
    interpreter: Path,
    lock_path: Path,
    cache_root: Path,
) -> SeparatedStem:
    _validate_execution_inputs(
        separator_id,
        source_audio_path,
        source_audio_sha256,
        source_duration_sec,
        interpreter,
        lock_path,
        cache_root,
    )
    lock = load_separator_lock(lock_path)
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
        qc = _read_and_qc_stem(cache_path, source_duration_sec)
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
                lock,
                interpreter=interpreter,
                input_path=input_path,
                output_dir=output_dir,
            )
            _run_separator_process(argv, cwd=workdir)
            output_path = output_dir / lock.expected_drum_stem_relative_path
            try:
                stem_bytes = read_regular_file_no_follow(output_path)
            except (OSError, TypeError) as error:
                raise SeparatorExecutionError(
                    "separator_output_missing",
                    "expected drum stem WAV is unavailable",
                ) from error
            qc = _read_and_qc_stem(output_path, source_duration_sec)
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
    interpreter: Path,
    lock_path: Path,
    cache_root: Path,
) -> None:
    if separator_id not in _SEPARATOR_POLICIES:
        raise SeparatorExecutionError("separator_id_unsupported", "separator is not supported")
    if not isinstance(source_audio_path, Path):
        raise TypeError("source_audio_path must be a Path")
    if not isinstance(interpreter, Path):
        raise TypeError("interpreter must be a Path")
    if not isinstance(lock_path, Path):
        raise TypeError("lock_path must be a Path")
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
    lock: SeparatorLock,
    *,
    interpreter: Path,
    input_path: Path,
    output_dir: Path,
) -> list[str]:
    replacements = {
        "{input_wav}": os.fspath(input_path),
        "{output_dir}": os.fspath(output_dir),
    }
    return [
        os.fspath(interpreter),
        *(replacements.get(argument, argument) for argument in lock.argv),
    ]


def _run_separator_process(argv: list[str], *, cwd: Path) -> None:
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
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
    try:
        process.wait(timeout=SEPARATOR_TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait()
    except (OSError, subprocess.TimeoutExpired):
        pass


def _read_and_qc_stem(path: Path, source_duration_sec: float) -> StemQc:
    try:
        samples, sample_rate = soundfile.read(path, always_2d=True, dtype="float32")
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
    "HTDEMUCS_SEPARATOR_ID",
    "SEPARATOR_LOCK_SCHEMA",
    "SEPARATOR_TERMINATE_GRACE_SECONDS",
    "SEPARATOR_TIMEOUT_SECONDS",
    "SPLEETER_SEPARATOR_ID",
    "STEM_CLIP_ABS",
    "STEM_MAX_DURATION_DELTA_RATIO",
    "STEM_MAX_DURATION_DELTA_SECONDS",
    "STEM_NEAR_SILENT_DBFS",
    "SeparatedStem",
    "SeparatorLock",
    "SeparatorLockError",
    "SeparatorModelFile",
    "SeparatorExecutionError",
    "StemQc",
    "load_separator_lock",
    "run_htdemucs_drums",
    "run_spleeter_drums",
    "separator_lock_payload",
]
