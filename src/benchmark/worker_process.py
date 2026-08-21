"""Minimal host-side controller for the extracted persistent OaF worker."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

_MAX_STDERR_DIAGNOSTIC_BYTES = 4096
_STDERR_READ_CHUNK_BYTES = 4096


class WorkerProcessError(RuntimeError):
    """The worker process or its line protocol failed."""


class WorkerProcess:
    """Own one subprocess and serialize its request/response exchanges."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        timeout_seconds: float,
        close_timeout_seconds: float = 30.0,
        ready: Mapping[str, Any],
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if close_timeout_seconds <= 0:
            raise ValueError("close_timeout_seconds must be positive")
        self._process = process
        self._timeout_seconds = timeout_seconds
        self._close_timeout_seconds = close_timeout_seconds
        self._request_lock = threading.Lock()
        self._closed = False
        self._ready = dict(ready)
        self._stdout_buffer = bytearray()
        self._stderr_buffer = bytearray()
        self._stderr_truncated = False
        self._stderr_lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None
        if process.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._capture_stderr,
                args=(process.stderr,),
                daemon=True,
            )
            self._stderr_thread.start()

    @classmethod
    def start(
        cls,
        command: Sequence[str] | str | Path,
        *,
        timeout_seconds: float = 30.0,
        close_timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "WorkerProcess":
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if close_timeout_seconds <= 0:
            raise ValueError("close_timeout_seconds must be positive")
        if isinstance(command, (str, Path)):
            command = [sys.executable, os.fspath(command)]
        if not command:
            raise WorkerProcessError("worker command is empty")
        try:
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.fspath(cwd) if cwd is not None else None,
                env=dict(env) if env is not None else None,
                shell=False,
                bufsize=0,
            )
        except OSError as error:
            raise WorkerProcessError("worker could not be started") from error
        worker = cls(
            process,
            timeout_seconds=timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            ready={},
        )
        try:
            ready = worker._read_record(timeout_seconds)
            if ready.get("type") != "ready":
                raise WorkerProcessError("worker ready response is invalid")
            worker._ready = ready
            return worker
        except WorkerProcessError as error:
            worker.close()
            if str(error) in {"worker exited before ready", "worker response timed out"}:
                diagnostic_error = worker._with_stderr_diagnostic(error)
                if diagnostic_error is error:
                    raise
                raise diagnostic_error from error
            raise WorkerProcessError("worker ready response is invalid") from error
        except Exception as error:
            worker.close()
            raise WorkerProcessError("worker ready response is invalid") from error

    @property
    def ready(self) -> Mapping[str, Any]:
        return dict(self._ready)

    @property
    def returncode(self) -> int | None:
        return self._process.poll()

    @property
    def closed(self) -> bool:
        return self._closed

    def request(
        self,
        audio_path: Path | str,
        *,
        audio_byte_length: int | None = None,
        audio_sha256: str | None = None,
        request_id: str | None = None,
    ) -> Mapping[str, Any]:
        if self._closed:
            raise WorkerProcessError("worker process is closed")
        with self._request_lock:
            if self._process.poll() is not None:
                self._poison()
                error = WorkerProcessError("worker exited before ready")
                diagnostic_error = self._with_stderr_diagnostic(error)
                if diagnostic_error is error:
                    raise error
                raise diagnostic_error from error
            identifier = request_id or uuid.uuid4().hex
            if not isinstance(identifier, str) or not identifier:
                raise WorkerProcessError("worker request id is invalid")
            payload = {"id": identifier, "audio_path": os.fspath(audio_path)}
            if audio_byte_length is not None or audio_sha256 is not None:
                if (
                    type(audio_byte_length) is not int
                    or audio_byte_length < 0
                    or not isinstance(audio_sha256, str)
                    or len(audio_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in audio_sha256)
                ):
                    raise WorkerProcessError("worker audio identity is invalid")
                payload.update(
                    {
                        "audio_byte_length": audio_byte_length,
                        "audio_sha256": audio_sha256,
                    }
                )
            stream = self._process.stdin
            if stream is None:
                raise WorkerProcessError("worker stdin is unavailable")
            try:
                stream.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
                stream.flush()
            except (BrokenPipeError, OSError) as error:
                self._poison()
                raise WorkerProcessError("worker request failed") from error
            try:
                response = self._read_record(self._timeout_seconds)
            except WorkerProcessError as error:
                self._poison()
                diagnostic_error = self._with_stderr_diagnostic(error)
                if diagnostic_error is error:
                    raise
                raise diagnostic_error from error
            if "error" in response:
                if response.get("id") != identifier:
                    self._poison()
                    raise WorkerProcessError("worker response id mismatch")
                error_payload = response["error"]
                if (
                    isinstance(error_payload, Mapping)
                    and isinstance(error_payload.get("code"), str)
                    and isinstance(error_payload.get("message"), str)
                ):
                    return response
                self._poison()
                raise WorkerProcessError("worker response is invalid")
            if response.get("id") != identifier:
                self._poison()
                raise WorkerProcessError("worker response id mismatch")
            events = response.get("events")
            if not isinstance(events, list):
                self._poison()
                raise WorkerProcessError("worker response is invalid")
            return response

    def _poison(self) -> None:
        """Close a process after a protocol failure so it cannot be reused."""
        self.close()

    def _capture_stderr(self, stream: Any) -> None:
        while True:
            try:
                chunk = stream.read(_STDERR_READ_CHUNK_BYTES)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            with self._stderr_lock:
                remaining = _MAX_STDERR_DIAGNOSTIC_BYTES - len(self._stderr_buffer)
                if remaining > 0:
                    self._stderr_buffer.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self._stderr_truncated = True

    def _with_stderr_diagnostic(self, error: WorkerProcessError) -> WorkerProcessError:
        if str(error) != "worker exited before ready":
            return error
        with self._stderr_lock:
            content = bytes(self._stderr_buffer)
            truncated = self._stderr_truncated
        diagnostic = content.decode("utf-8", errors="replace").strip()
        if not diagnostic:
            return error
        if truncated:
            diagnostic += " ... [stderr truncated]"
        return WorkerProcessError(f"{error}: {diagnostic}")

    def _read_record(self, timeout_seconds: float) -> dict[str, Any]:
        stream = self._process.stdout
        if stream is None:
            raise WorkerProcessError("worker stdout is unavailable")
        descriptor = stream.fileno()
        deadline = time.monotonic() + timeout_seconds
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self._stdout_buffer[:newline])
                del self._stdout_buffer[: newline + 1]
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise WorkerProcessError("worker response is invalid") from error
                if not isinstance(value, dict):
                    raise WorkerProcessError("worker response is invalid")
                return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerProcessError("worker response timed out")
            try:
                readable, _, _ = select.select([descriptor], [], [], remaining)
            except (OSError, ValueError) as error:
                raise WorkerProcessError("worker stdout failed") from error
            if not readable:
                raise WorkerProcessError("worker response timed out")
            line = os.read(descriptor, 65536)
            if not line:
                raise WorkerProcessError("worker exited before ready")
            self._stdout_buffer.extend(line)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stream = self._process.stdin
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
        deadline = time.monotonic() + self._close_timeout_seconds

        def _remaining() -> float:
            return max(deadline - time.monotonic(), 0.1)

        try:
            self._process.wait(timeout=_remaining())
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=_remaining())
            except subprocess.TimeoutExpired:
                self._process.kill()
                try:
                    self._process.wait(timeout=_remaining())
                except subprocess.TimeoutExpired as error:
                    raise WorkerProcessError("worker close timed out") from error
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=_remaining())
        for stream in (self._process.stdout, self._process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> "WorkerProcess":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()
