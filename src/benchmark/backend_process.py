# The process controller intentionally keeps its validation, state machine,
# bounded I/O, and teardown logic together for lifecycle review.
# pylint: disable=too-many-lines

from __future__ import annotations

import os
import queue
import re
import select
import signal
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backend_lock import REQUIRED_ENVIRONMENT
from src.benchmark.backends.base import BackendError, BackendFatalFailure

_PROTOCOL_SCHEMA = "crux.transcription-runner/v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CPU = re.compile(r"(?:[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_OPAQUE_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_GITHUB_RUN = re.compile(
    r"https://github\.com/[^/?#]+/[^/?#]+/actions/runs/[1-9][0-9]*/job/[1-9][0-9]*\Z"
)
_SAFE_DIAGNOSTIC = re.compile(
    rb"code=[a-z][a-z0-9_]*"
    rb"(?: tensor=[A-Za-z0-9_.:/-]+)?"
    rb"(?: count=[0-9]+)?"
    rb"(?: duration_ms=[0-9]+)?\Z"
)
_ABSOLUTE_UNIX_PATH = re.compile(rb"(?:^|[\s=])/[^\s]+")
_WINDOWS_DRIVE_PATH = re.compile(rb"(?:^|[\s=])[A-Za-z]:[\\/][^\s]+")
_WINDOWS_UNC_PATH = re.compile(rb"(?:^|[\s=])(?:\\\\|//)[^\s]+")
_FILE_URI = re.compile(rb"(?:^|[\s=])file:(?://)?/[^\s]+", re.IGNORECASE)
_PATH_BEARING_URL = re.compile(rb"(?:^|[\s=])[A-Za-z][A-Za-z0-9+.-]*://[^\s]+")
_STRUCTURAL_PATH_PATTERNS = (
    _ABSOLUTE_UNIX_PATH,
    _WINDOWS_DRIVE_PATH,
    _WINDOWS_UNC_PATH,
    _FILE_URI,
    _PATH_BEARING_URL,
)
_FORBIDDEN_DIAGNOSTIC_MARKERS = (
    b"secret",
    b"token",
    b"credential",
    b"password",
    b"traceback",
    b'file "',
    b"http://",
    b"https://",
    b"audio=",
    b"samples=",
    b"path=",
)
_SENSITIVE_TOKEN_CHARACTER = rb"A-Za-z0-9_.:/-"
_CONTAINER_PATHS = (
    "/run/crux/backend-lock.json",
    "/run/crux/runtime-lock.json",
    "/model",
    "/input",
)


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _safe_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{label} contains control characters")
    return value


def _deep_freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _plain_json(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if isinstance(value, (type(None), bool, int, str)):
        return value
    raise ValueError("evidence payload must contain only JSON values")


def _require_exact_keys(payload: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(payload) != keys:
        raise ValueError(f"{label} fields must match the exact schema")


@dataclass(frozen=True)
class NativeHostEvidence:
    kind: Literal["github_hosted", "orchestrator_signed", "approved_local"]
    payload: Mapping[str, JsonValue]
    sha256: str
    official_execution_allowed: bool

    def __post_init__(self) -> None:
        plain = _plain_json(self.payload)
        if not isinstance(plain, dict):
            raise ValueError("native host evidence payload must be an object")
        if self.kind == "github_hosted":
            _validate_github_hosted(plain)
        elif self.kind == "orchestrator_signed":
            _validate_orchestrator(plain)
        elif self.kind == "approved_local":
            _validate_approved_local(plain)
        else:
            raise ValueError("native host evidence kind is unsupported")
        try:
            require_sha256(self.sha256, "native host evidence SHA-256")
        except StrictJsonError as error:
            raise ValueError(str(error)) from None
        if sha256_hex(canonical_json_bytes(plain)) != self.sha256:
            raise ValueError("native host evidence SHA-256 does not reproduce its payload")
        if self.official_execution_allowed is not True:
            raise ValueError("accepted native evidence must allow official execution")
        object.__setattr__(self, "payload", cast(Mapping[str, JsonValue], _deep_freeze(plain)))


def _validate_github_hosted(payload: dict[str, JsonValue]) -> None:
    _require_exact_keys(
        payload,
        {
            "api_record_sha256",
            "approved_labels",
            "job_id",
            "run_url",
            "runner_arch",
            "runner_os",
            "workflow_commit",
        },
        "GitHub-hosted evidence",
    )
    if payload["runner_os"] != "Linux" or payload["runner_arch"] != "X64":
        raise ValueError("GitHub-hosted evidence must be Linux X64")
    if payload["approved_labels"] != ["Linux", "X64"]:
        raise ValueError("GitHub-hosted evidence labels are not approved")
    _positive_integer(payload["job_id"], "GitHub-hosted job ID")
    commit = _safe_text(payload["workflow_commit"], "workflow commit")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("workflow commit must be a lowercase Git identity")
    run_url = _safe_text(payload["run_url"], "immutable run URL")
    if not _GITHUB_RUN.fullmatch(run_url):
        raise ValueError("GitHub run URL must be immutable")
    _require_hash_value(payload["api_record_sha256"], "API record SHA-256")


def _validate_orchestrator(payload: dict[str, JsonValue]) -> None:
    _require_exact_keys(
        payload,
        {"attestation_sha256", "physical_architecture", "signature", "worker_id"},
        "orchestrator evidence",
    )
    if payload["physical_architecture"] != "linux/amd64":
        raise ValueError("orchestrator physical architecture must be linux/amd64")
    _safe_text(payload["signature"], "orchestrator signature")
    _safe_text(payload["worker_id"], "orchestrator worker ID")
    _require_hash_value(payload["attestation_sha256"], "orchestrator attestation SHA-256")


def _validate_approved_local(payload: dict[str, JsonValue]) -> None:
    _require_exact_keys(
        payload,
        {"approval_sha256", "daemon_id", "host_architecture", "host_os", "worker_id"},
        "approved local evidence",
    )
    if payload["host_os"] != "Linux" or payload["host_architecture"] != "x86_64":
        raise ValueError("approved local evidence must be Linux x86_64")
    _safe_text(payload["daemon_id"], "daemon ID")
    _safe_text(payload["worker_id"], "worker ID")
    _require_hash_value(payload["approval_sha256"], "local approval SHA-256")


def _require_hash_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be lowercase SHA-256")
    try:
        return require_sha256(value, label)
    except StrictJsonError as error:
        raise ValueError(str(error)) from None


@dataclass(frozen=True)
class DiagnosticHostEvidence:
    kind: Literal["bare_local", "emulated"]
    payload: Mapping[str, JsonValue]
    emulation_allowed: bool
    official_execution_allowed: bool

    def __post_init__(self) -> None:
        plain = _plain_json(self.payload)
        if not isinstance(plain, dict):
            raise ValueError("diagnostic evidence payload must be an object")
        _require_exact_keys(plain, {"host_architecture", "host_os"}, "diagnostic evidence")
        _safe_text(plain["host_architecture"], "diagnostic host architecture")
        _safe_text(plain["host_os"], "diagnostic host OS")
        if self.kind not in {"bare_local", "emulated"}:
            raise ValueError("diagnostic host evidence kind is unsupported")
        if self.kind == "bare_local" and self.emulation_allowed:
            raise ValueError("bare local evidence cannot opt into emulation")
        if self.kind == "emulated" and not self.emulation_allowed:
            raise ValueError("emulated diagnostics require explicit opt-in")
        if self.official_execution_allowed:
            raise ValueError("diagnostic evidence cannot allow official execution")
        object.__setattr__(self, "payload", cast(Mapping[str, JsonValue], _deep_freeze(plain)))


_MountIdentity = tuple[int, int, int]


# The launch contract deliberately spells out every seal-required field.
# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class RunnerLaunchProfile:
    image_manifest_digest: str
    backend_lock_path: Path
    runtime_lock_path: Path
    model_cache_path: Path
    input_root: Path
    environment: Mapping[str, str]
    uid: int
    gid: int
    cpu_limit: str
    memory_bytes: int
    pid_limit: int
    tmp_bytes: int
    shm_bytes: int
    startup_deadline_seconds: int
    request_deadline_seconds: int
    stdout_max_line_bytes: int
    stderr_read_chunk_bytes: int
    stderr_max_line_bytes: int
    stderr_ring_buffer_bytes: int
    _mount_identities: Mapping[str, _MountIdentity] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.image_manifest_digest, str) or not _DIGEST.fullmatch(
            self.image_manifest_digest
        ):
            raise ValueError("image manifest digest must be immutable sha256")
        uid = _positive_integer(self.uid, "runtime UID")
        gid = _positive_integer(self.gid, "runtime GID")
        if uid == 0 or gid == 0:
            raise ValueError("runtime UID/GID must be non-root")
        cpu = _safe_text(self.cpu_limit, "CPU limit")
        if not _CPU.fullmatch(cpu) or float(cpu) <= 0:
            raise ValueError("CPU limit must be a positive decimal")
        for name in (
            "memory_bytes",
            "pid_limit",
            "tmp_bytes",
            "shm_bytes",
            "startup_deadline_seconds",
            "request_deadline_seconds",
            "stdout_max_line_bytes",
            "stderr_read_chunk_bytes",
            "stderr_max_line_bytes",
            "stderr_ring_buffer_bytes",
        ):
            _positive_integer(getattr(self, name), name)
        if not isinstance(self.environment, Mapping) or dict(self.environment) != dict(
            REQUIRED_ENVIRONMENT
        ):
            raise ValueError("runner environment must match the exact frozen allowlist")
        environment = dict(self.environment)
        for key, value in environment.items():
            _docker_atom(key, f"environment key {key}")
            _docker_atom(value, f"environment value {key}")
        object.__setattr__(self, "environment", MappingProxyType(environment))
        identities: dict[str, _MountIdentity] = {}
        for name, expected_kind in (
            ("backend_lock_path", "file"),
            ("runtime_lock_path", "file"),
            ("model_cache_path", "directory"),
            ("input_root", "directory"),
        ):
            path = Path(getattr(self, name))
            identity = _mount_identity(path, expected_kind)
            object.__setattr__(self, name, path)
            identities[name] = identity
        object.__setattr__(self, "_mount_identities", MappingProxyType(identities))


def _docker_atom(value: str, label: str) -> str:
    _safe_text(value, label)
    if any(character in value for character in ",:"):
        raise ValueError(f"{label} alters Docker option grammar")
    return value


def _mount_identity(path: Path, expected_kind: str) -> _MountIdentity:
    if not path.is_absolute():
        raise ValueError("mount paths must be absolute")
    _docker_atom(os.fspath(path), "mount path")
    current = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            current /= component
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("mount paths must not contain symlinks")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        if expected_kind == "directory":
            flags |= getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            status = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except (OSError, ValueError) as error:
        raise ValueError("mount path must be a stable no-follow regular input") from error
    if expected_kind == "file" and not stat.S_ISREG(status.st_mode):
        raise ValueError("mount file must be regular")
    if expected_kind == "directory" and not stat.S_ISDIR(status.st_mode):
        raise ValueError("mount directory must be a directory")
    return status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode)


def _verify_mounts(profile: RunnerLaunchProfile) -> None:
    for name, expected_kind in (
        ("backend_lock_path", "file"),
        ("runtime_lock_path", "file"),
        ("model_cache_path", "directory"),
        ("input_root", "directory"),
    ):
        if (
            _mount_identity(Path(getattr(profile, name)), expected_kind)
            # pylint: disable-next=protected-access
            != profile._mount_identities[name]
        ):
            raise ValueError(f"mount identity changed: {name}")


def build_docker_command(profile: RunnerLaunchProfile) -> list[str]:
    if not isinstance(profile, RunnerLaunchProfile):
        raise TypeError("runner launch profile is required")
    _verify_mounts(profile)
    command = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--platform=linux/amd64",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--user={profile.uid}:{profile.gid}",
        f"--cpus={profile.cpu_limit}",
        f"--memory={profile.memory_bytes}",
        f"--pids-limit={profile.pid_limit}",
        f"--tmpfs=/tmp:rw,noexec,nosuid,nodev,size={profile.tmp_bytes}",
        f"--tmpfs=/dev/shm:rw,noexec,nosuid,nodev,size={profile.shm_bytes}",
    ]
    mounts = (
        (profile.backend_lock_path, _CONTAINER_PATHS[0]),
        (profile.runtime_lock_path, _CONTAINER_PATHS[1]),
        (profile.model_cache_path, _CONTAINER_PATHS[2]),
        (profile.input_root, _CONTAINER_PATHS[3]),
    )
    command.extend(
        f"--mount=type=bind,src={source},dst={destination},readonly"
        for source, destination in mounts
    )
    command.extend(f"--env={key}={profile.environment[key]}" for key in sorted(profile.environment))
    command.append(profile.image_manifest_digest)
    return command


@dataclass(frozen=True)
class RunnerResponse:
    request_id: str
    payload: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "payload", cast(Mapping[str, JsonValue], _deep_freeze(dict(self.payload)))
        )


@dataclass(frozen=True)
class _StdoutEvent:
    kind: Literal["line", "oversized", "eof", "io_error"]
    value: bytes | bool | None = None


PopenFactory = Callable[..., subprocess.Popen[bytes]]
ThreadFactory = Callable[..., Any]
ReadFunction = Callable[[int, int], bytes]


# Persistent process ownership necessarily tracks its pipes, threads, and bounded buffers.
# pylint: disable=too-many-instance-attributes
class RunnerProcess:
    def __init__(
        self,
        profile: RunnerLaunchProfile,
        process: subprocess.Popen[bytes],
        thread_factory: ThreadFactory,
        read_function: ReadFunction,
    ) -> None:
        self._profile = profile
        self._process = process
        self._read = read_function
        self._stdout_events: queue.Queue[_StdoutEvent] = queue.Queue(maxsize=2)
        self._protocol_lock = threading.Lock()
        self._protocol_condition = threading.Condition(self._protocol_lock)
        self._protocol_state = "startup_wait"
        self._protocol_failure: BackendFatalFailure | None = None
        self._stdout_batch_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._stderr_lock = threading.Lock()
        self._stderr_ring = bytearray()
        self._stderr_total_raw_bytes = 0
        self._stderr_truncated = False
        self._stderr_last_raw: bytes | None = None
        self._stderr_last_sanitized = b""
        self._sensitive_value_pattern = _compile_sensitive_value_pattern(
            profile.environment.values()
        )
        self._configured_path_bytes = tuple(
            os.fspath(path).encode("utf-8")
            for path in (
                profile.backend_lock_path,
                profile.runtime_lock_path,
                profile.model_cache_path,
                profile.input_root,
            )
        ) + tuple(path.encode("utf-8") for path in _CONTAINER_PATHS)
        self._closing = False
        self._closed = False
        self._cleanup_complete = threading.Event()
        self._stdout_thread = thread_factory(
            target=self._stdout_reader_entrypoint, name="oaf-stdout", daemon=True
        )
        self._stderr_thread = thread_factory(
            target=self._stderr_reader_entrypoint, name="oaf-stderr", daemon=True
        )
        self._handshake: Mapping[str, JsonValue] = MappingProxyType({})

    @classmethod
    def start(
        cls,
        profile: RunnerLaunchProfile,
        *,
        popen_factory: PopenFactory = subprocess.Popen,
        thread_factory: ThreadFactory = threading.Thread,
        read_function: ReadFunction = os.read,
    ) -> RunnerProcess:
        command = build_docker_command(profile)
        try:
            # The process intentionally outlives this method and is owned by RunnerProcess.
            # pylint: disable-next=consider-using-with
            process = popen_factory(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={},
                shell=False,
                start_new_session=True,
                bufsize=0,
            )
        except OSError:
            raise _failure(
                "backend_launch_failed", "The backend runner could not be launched."
            ) from None
        runner = cls(profile, process, thread_factory, read_function)
        try:
            try:
                runner._stderr_thread.start()
                runner._stdout_thread.start()
            except Exception:
                runner._abort()
                raise _failure(
                    "backend_reader_start_failed",
                    "The backend runner reader threads could not be started.",
                ) from None
            startup_expires = time.monotonic() + profile.startup_deadline_seconds
            handshake = runner._read_protocol_until(startup_expires, "backend_startup_timeout")
            if (
                handshake.get("type") != "ready"
                or handshake.get("protocol_schema") != _PROTOCOL_SCHEMA
            ):
                runner._raise_fatal(
                    "backend_protocol_mismatch",
                    "The backend runner handshake did not match the frozen protocol.",
                )
            runner._complete_protocol_exchange("startup_delivered")
            runner._handshake = cast(Mapping[str, JsonValue], _deep_freeze(dict(handshake)))
            return runner
        except BaseException:
            runner._abort()
            raise

    @property
    def handshake(self) -> Mapping[str, JsonValue]:
        return self._handshake

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def stderr_total_raw_bytes(self) -> int:
        with self._stderr_lock:
            return self._stderr_total_raw_bytes

    @property
    def stderr_retained_bytes(self) -> int:
        with self._stderr_lock:
            return len(self._stderr_ring)

    @property
    def stderr_truncated(self) -> bool:
        with self._stderr_lock:
            return self._stderr_truncated

    @property
    def stderr_text(self) -> str:
        with self._stderr_lock:
            return bytes(self._stderr_ring).decode("utf-8", errors="strict")

    @property
    def stderr_thread_alive(self) -> bool:
        return self._stderr_thread.is_alive()

    @property
    def process_returncode(self) -> int | None:
        return self._process.poll()

    def request(
        self, payload: Mapping[str, JsonValue], *, deadline_seconds: int | None = None
    ) -> RunnerResponse:
        # A context manager cannot express immediate, nonblocking acquisition.
        # pylint: disable-next=consider-using-with
        if not self._request_lock.acquire(blocking=False):
            raise _failure(
                "backend_request_in_flight",
                "The backend runner already has one request in flight.",
            )
        try:
            self._raise_protocol_failure_if_any()
            if self._closed:
                raise _failure("backend_process_closed", "The backend runner is closed.")
            deadline = (
                self._profile.request_deadline_seconds
                if deadline_seconds is None
                else _positive_integer(deadline_seconds, "request deadline")
            )
            request = dict(payload)
            request_id = request.get("request_id") or uuid.uuid4().hex
            if not isinstance(request_id, str) or not _OPAQUE_ID.fullmatch(request_id):
                raise ValueError("request ID must be an opaque ASCII identifier")
            request["request_id"] = request_id
            try:
                content = canonical_json_bytes(cast(JsonValue, request), trailing_newline=True)
            except StrictJsonError:
                raise ValueError("request must be a canonical JSON object") from None
            expires = time.monotonic() + deadline
            self._begin_request_write()
            self._write_all(content, expires)
            self._allow_request_response()
            response = self._read_protocol_until(expires, "backend_request_timeout")
            if response.get("request_id") != request_id:
                self._raise_fatal(
                    "backend_request_id_mismatch",
                    "The backend runner response used the wrong request identifier.",
                )
            if response.get("type") != "response" or not isinstance(response.get("payload"), dict):
                self._raise_fatal(
                    "backend_protocol_invalid",
                    "The backend runner emitted an invalid protocol response.",
                )
            self._complete_protocol_exchange("response_delivered")
            return RunnerResponse(
                request_id=request_id,
                payload=cast(dict[str, JsonValue], response["payload"]),
            )
        finally:
            self._request_lock.release()

    def _write_all(self, content: bytes, expires: float) -> None:
        stream = self._process.stdin
        if stream is None:
            self._raise_fatal("backend_broken_pipe", "The backend request pipe is unavailable.")
        try:
            descriptor = stream.fileno()
            os.set_blocking(descriptor, False)
        except (OSError, ValueError):
            self._raise_fatal("backend_broken_pipe", "The backend runner request pipe broke.")
        offset = 0
        while offset < len(content):
            remaining = expires - time.monotonic()
            if remaining <= 0:
                self._raise_fatal(
                    "backend_request_timeout", "The backend runner request timed out."
                )
            try:
                _, writable, _ = select.select([], [descriptor], [], remaining)
            except (OSError, ValueError):
                self._raise_fatal("backend_broken_pipe", "The backend runner request pipe broke.")
            if not writable:
                self._raise_fatal(
                    "backend_request_timeout", "The backend runner request timed out."
                )
            try:
                offset += os.write(descriptor, content[offset:])
            except (BrokenPipeError, OSError, ValueError):
                self._raise_fatal("backend_broken_pipe", "The backend runner request pipe broke.")

    def _raise_protocol_failure_if_any(self) -> None:
        with self._protocol_lock:
            failure = self._protocol_failure
        if failure is not None:
            self._cleanup_complete.wait(self._profile.request_deadline_seconds)
            raise failure

    def _begin_request_write(self) -> None:
        with self._protocol_condition:
            failure = self._protocol_failure
            valid = self._protocol_state == "idle"
            if failure is None and valid:
                self._protocol_state = "request_writing"
        if failure is not None:
            self._cleanup_complete.wait(self._profile.request_deadline_seconds)
            raise failure
        if not valid:
            self._raise_fatal(
                "backend_unexpected_response",
                "The backend runner protocol state is invalid.",
            )

    def _allow_request_response(self) -> None:
        with self._protocol_condition:
            failure = self._protocol_failure
            valid = self._protocol_state == "request_writing"
            if failure is None and valid:
                self._protocol_state = "request_wait"
            self._protocol_condition.notify_all()
        if failure is not None:
            self._cleanup_complete.wait(self._profile.request_deadline_seconds)
            raise failure
        if not valid:
            self._raise_fatal(
                "backend_unexpected_response",
                "The backend runner protocol state is invalid.",
            )

    def _complete_protocol_exchange(self, expected_state: str) -> None:
        # Synchronize with every complete line already present in the reader's
        # current bounded OS-read batch; this is state synchronization, not a
        # timing-based quiet period.
        with self._stdout_batch_lock:
            with self._protocol_condition:
                failure = self._protocol_failure
                valid = self._protocol_state == expected_state
                if failure is None and valid:
                    self._protocol_state = "idle"
                self._protocol_condition.notify_all()
        if failure is not None:
            self._cleanup_complete.wait(self._profile.request_deadline_seconds)
            raise failure
        if not valid:
            self._raise_fatal(
                "backend_unexpected_response",
                "The backend runner protocol state is invalid.",
            )

    def _read_protocol_until(self, expires: float, timeout_code: str) -> dict[str, JsonValue]:
        self._raise_protocol_failure_if_any()
        remaining = expires - time.monotonic()
        if remaining <= 0:
            self._raise_fatal(timeout_code, "The backend runner deadline expired.")
        try:
            event = self._stdout_events.get(timeout=remaining)
        except queue.Empty:
            self._raise_protocol_failure_if_any()
            self._raise_fatal(timeout_code, "The backend runner deadline expired.")
        self._raise_protocol_failure_if_any()
        if event.kind == "oversized":
            self._raise_fatal(
                "backend_protocol_oversized",
                "The backend runner exceeded the protocol line byte limit.",
            )
        if event.kind == "eof":
            if event.value:
                self._raise_fatal(
                    "backend_protocol_invalid",
                    "The backend runner ended a protocol line before its newline.",
                )
            if self._process.poll() is not None:
                self._raise_fatal("backend_process_died", "The backend runner exited unexpectedly.")
            self._raise_fatal(
                "backend_protocol_invalid", "The backend runner protocol stream ended."
            )
        if event.kind == "io_error":
            self._raise_fatal(
                "backend_protocol_io_error", "The backend runner protocol stream failed."
            )
        line = cast(bytes, event.value)
        if not line:
            self._raise_fatal(
                "backend_protocol_invalid", "The backend runner emitted a blank protocol line."
            )
        try:
            value = strict_json_loads(line)
        except StrictJsonError:
            self._raise_fatal(
                "backend_protocol_invalid", "The backend runner emitted invalid JSON."
            )
        if not isinstance(value, dict):
            self._raise_fatal(
                "backend_protocol_invalid", "The backend runner response must be a JSON object."
            )
        return value

    def _stdout_reader_entrypoint(self) -> None:
        try:
            self._drain_stdout()
        # Reader entrypoints are the final containment boundary: even injected
        # BaseException subclasses must not reach threading.excepthook.
        # pylint: disable-next=broad-exception-caught
        except BaseException:
            # Pipe teardown can race fileno/read after an intentional close.
            # No exception may escape a daemon reader and invoke Python's
            # traceback hook with host paths.
            try:
                self._emit_stdout(_StdoutEvent("io_error"))
            # pylint: disable-next=broad-exception-caught
            except BaseException:
                pass

    def _drain_stdout(self) -> None:
        stream = self._process.stdout
        if stream is None:
            raise OSError("stdout pipe is unavailable")
        descriptor = stream.fileno()
        buffered = bytearray()
        maximum = self._profile.stdout_max_line_bytes
        while True:
            chunk = self._read(descriptor, maximum + 1)
            if not chunk:
                self._emit_stdout(_StdoutEvent("eof", bool(buffered)))
                return
            with self._stdout_batch_lock:
                buffered.extend(chunk)
                while True:
                    newline = buffered.find(b"\n")
                    if newline < 0:
                        if len(buffered) > maximum:
                            self._emit_stdout(_StdoutEvent("oversized"))
                            return
                        break
                    physical_length = newline + 1
                    if physical_length > maximum:
                        self._emit_stdout(_StdoutEvent("oversized"))
                        return
                    line = bytes(buffered[:newline])
                    del buffered[:physical_length]
                    if not self._emit_stdout(_StdoutEvent("line", line)):
                        return

    # Protocol safety keeps every event/state/teardown combination explicit.
    # pylint: disable=too-many-branches,too-many-return-statements
    def _emit_stdout(self, event: _StdoutEvent) -> bool:
        abort = False
        enqueue = False
        with self._protocol_condition:
            while (
                self._protocol_state == "request_writing"
                and self._protocol_failure is None
                and not self._closing
                and not self._closed
            ):
                self._protocol_condition.wait()
            if self._protocol_failure is not None:
                return False
            teardown_in_progress = self._closing or self._closed
            if teardown_in_progress:
                if _is_intentional_teardown_event(event):
                    return False
                if event.kind == "line":
                    self._protocol_failure = _failure(
                        "backend_unexpected_response",
                        "The backend runner emitted stdout during process teardown.",
                    )
                else:
                    self._protocol_failure = _stdout_event_failure(event)
                self._protocol_state = "fatal"
                self._protocol_condition.notify_all()
                return False

            expected = self._protocol_state in {"startup_wait", "request_wait"}
            if event.kind == "line" and expected:
                if self._protocol_state == "startup_wait":
                    self._protocol_state = "startup_delivered"
                else:
                    self._protocol_state = "response_delivered"
                enqueue = True
            elif event.kind == "line":
                self._protocol_failure = _failure(
                    "backend_unexpected_response",
                    "The backend runner emitted stdout while no response was permitted.",
                )
                self._protocol_state = "fatal"
                abort = True
            else:
                self._protocol_failure = _stdout_event_failure(event)
                self._protocol_state = "fatal"
                enqueue = expected
                abort = True
            self._protocol_condition.notify_all()

        if not enqueue:
            if abort:
                self._abort()
            return False
        try:
            self._stdout_events.put_nowait(event)
        except queue.Full:
            with self._protocol_condition:
                self._protocol_failure = _failure(
                    "backend_unexpected_response",
                    "The backend runner flooded the protocol stream.",
                )
                self._protocol_state = "fatal"
                abort = not self._closing and not self._closed
                self._protocol_condition.notify_all()
            if abort:
                self._abort()
            return False
        if abort:
            self._abort()
            return False
        return True

    def _stderr_reader_entrypoint(self) -> None:
        try:
            self._drain_stderr()
        # pylint: disable-next=broad-exception-caught
        except BaseException:
            # ValueError is raised by fileno/read after another thread closes
            # the pipe. Intentional teardown is clean; an operational reader
            # failure is fatal because an undrained stderr pipe can deadlock
            # inference.
            try:
                self._record_background_failure(
                    "backend_stderr_io_error",
                    "The backend runner diagnostic stream failed.",
                )
            # pylint: disable-next=broad-exception-caught
            except BaseException:
                pass

    def _drain_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:
            raise OSError("stderr pipe is unavailable")
        descriptor = stream.fileno()
        buffered = bytearray()
        maximum = self._profile.stderr_max_line_bytes
        while True:
            chunk = self._read(descriptor, self._profile.stderr_read_chunk_bytes)
            if not chunk:
                if buffered:
                    self._retain_diagnostic(bytes(buffered), oversized=False)
                return
            with self._stderr_lock:
                self._stderr_total_raw_bytes += len(chunk)
            buffered.extend(chunk)
            while True:
                newline = buffered.find(b"\n")
                if 0 <= newline < maximum:
                    logical = bytes(buffered[:newline])
                    del buffered[: newline + 1]
                    self._retain_diagnostic(logical, oversized=False)
                    continue
                if len(buffered) >= maximum:
                    logical = bytes(buffered[:maximum])
                    del buffered[:maximum]
                    self._retain_diagnostic(logical, oversized=True)
                    continue
                break

    def _record_background_failure(self, code: str, message: str) -> None:
        with self._protocol_condition:
            if self._closing or self._closed:
                return
            if self._protocol_failure is None:
                self._protocol_failure = _failure(code, message)
            self._protocol_state = "fatal"
            self._protocol_condition.notify_all()
        try:
            self._stdout_events.put_nowait(_StdoutEvent("io_error"))
        except queue.Full:
            pass
        self._abort()

    def _retain_diagnostic(self, raw: bytes, *, oversized: bool) -> None:
        if raw == self._stderr_last_raw:
            sanitized = self._stderr_last_sanitized
        else:
            sanitized = self._sanitize_diagnostic(raw) + b"\n"
            self._stderr_last_raw = raw
            self._stderr_last_sanitized = sanitized
        capacity = self._profile.stderr_ring_buffer_bytes
        with self._stderr_lock:
            if oversized:
                self._stderr_truncated = True
            self._stderr_ring.extend(sanitized)
            if len(self._stderr_ring) > capacity:
                del self._stderr_ring[: len(self._stderr_ring) - capacity]
                self._stderr_truncated = True

    def _sanitize_diagnostic(self, raw: bytes) -> bytes:
        lowered = raw.lower()
        contains_sensitive_value = bool(
            self._sensitive_value_pattern is not None and self._sensitive_value_pattern.search(raw)
        )
        if (
            contains_sensitive_value
            or any(path in raw for path in self._configured_path_bytes)
            or any(pattern.search(raw) for pattern in _STRUCTURAL_PATH_PATTERNS)
            or any(marker in lowered for marker in _FORBIDDEN_DIAGNOSTIC_MARKERS)
            or not _SAFE_DIAGNOSTIC.fullmatch(raw)
        ):
            return b"[REDACTED]"
        return raw

    def _raise_fatal(self, code: str, message: str) -> Any:
        self._abort()
        raise _failure(code, message)

    def _abort(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closing = True
            with self._protocol_condition:
                self._protocol_condition.notify_all()
            self._signal_tree(signal.SIGKILL)
            self._reap(self._profile.request_deadline_seconds)
            self._close_pipes()
            self._join_readers(self._profile.request_deadline_seconds)
            self._closing = False
            self._closed = True
            self._cleanup_complete.set()
            with self._protocol_condition:
                self._protocol_condition.notify_all()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closing = True
            with self._protocol_condition:
                self._protocol_condition.notify_all()
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except (OSError, ValueError):
                    pass
            grace = min(1.0, float(self._profile.request_deadline_seconds))
            try:
                self._process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self._signal_tree(signal.SIGTERM)
                try:
                    self._process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    self._signal_tree(signal.SIGKILL)
                    self._reap(self._profile.request_deadline_seconds)
            self._close_pipes()
            self._join_readers(self._profile.request_deadline_seconds)
            self._closing = False
            self._closed = True
            self._cleanup_complete.set()
            with self._protocol_condition:
                self._protocol_condition.notify_all()

    def _signal_tree(self, requested_signal: signal.Signals) -> None:
        if self._process.poll() is not None:
            return
        try:
            os.killpg(self._process.pid, requested_signal)
        except (ProcessLookupError, PermissionError):
            try:
                self._process.send_signal(requested_signal)
            except ProcessLookupError:
                return

    def _reap(self, seconds: int) -> None:
        try:
            self._process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            self._signal_tree(signal.SIGKILL)
            try:
                self._process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                pass

    def _close_pipes(self) -> None:
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

    def _join_readers(self, seconds: int) -> None:
        deadline = time.monotonic() + seconds
        current = threading.current_thread()
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is current:
                continue
            try:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            except RuntimeError:
                # Thread.start may fail before a thread acquires lifecycle ownership.
                continue


def _failure(code: str, message: str) -> BackendFatalFailure:
    return BackendFatalFailure(BackendError(code=code, message=message))


def _stdout_event_failure(event: _StdoutEvent) -> BackendFatalFailure:
    if event.kind == "oversized":
        return _failure(
            "backend_protocol_oversized",
            "The backend runner exceeded the protocol line byte limit.",
        )
    if event.kind == "eof" and event.value:
        return _failure(
            "backend_protocol_invalid",
            "The backend runner ended a protocol line before its newline.",
        )
    if event.kind == "eof":
        return _failure("backend_process_died", "The backend runner exited unexpectedly.")
    return _failure(
        "backend_protocol_io_error",
        "The backend runner protocol stream failed.",
    )


def _is_intentional_teardown_event(event: _StdoutEvent) -> bool:
    return event.kind == "io_error" or (event.kind == "eof" and not event.value)


def _compile_sensitive_value_pattern(values: Iterable[str]) -> re.Pattern[bytes] | None:
    encoded = sorted(
        {value.encode("utf-8") for value in values if value},
        key=lambda value: (-len(value), value),
    )
    if not encoded:
        return None
    alternatives = b"|".join(re.escape(value) for value in encoded)
    return re.compile(
        rb"(?<!["
        + _SENSITIVE_TOKEN_CHARACTER
        + rb"])(?:"
        + alternatives
        + rb")(?!["
        + _SENSITIVE_TOKEN_CHARACTER
        + rb"])"
    )
