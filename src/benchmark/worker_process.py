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


class WorkerProcessError(RuntimeError):
    """The worker process or its line protocol failed."""


class WorkerProcess:
    """Own one subprocess and serialize its request/response exchanges."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        timeout_seconds: float,
        ready: Mapping[str, Any],
    ) -> None:
        self._process = process
        self._timeout_seconds = timeout_seconds
        self._request_lock = threading.Lock()
        self._closed = False
        self._ready = dict(ready)
        self._stdout_buffer = bytearray()

    @classmethod
    def start(
        cls,
        command: Sequence[str] | str | Path,
        *,
        timeout_seconds: float = 30.0,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "WorkerProcess":
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
        worker = cls(process, timeout_seconds=timeout_seconds, ready={})
        try:
            ready = worker._read_record(timeout_seconds)
            if (
                ready.get("type") != "ready"
                or not isinstance(ready.get("backend_id"), str)
                or ready.get("restored_tensor_count") != 78
            ):
                raise WorkerProcessError("worker ready response is invalid")
            worker._ready = ready
            return worker
        except WorkerProcessError as error:
            worker.close()
            if str(error) in {"worker exited before ready", "worker response timed out"}:
                raise
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
        self, audio_path: Path | str, *, request_id: str | None = None
    ) -> Mapping[str, Any]:
        if self._closed:
            raise WorkerProcessError("worker process is closed")
        with self._request_lock:
            if self._process.poll() is not None:
                raise WorkerProcessError("worker exited before ready")
            identifier = request_id or uuid.uuid4().hex
            if not isinstance(identifier, str) or not identifier:
                raise WorkerProcessError("worker request id is invalid")
            payload = {"id": identifier, "audio_path": os.fspath(audio_path)}
            stream = self._process.stdin
            if stream is None:
                raise WorkerProcessError("worker stdin is unavailable")
            try:
                stream.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
                stream.flush()
            except (BrokenPipeError, OSError) as error:
                raise WorkerProcessError("worker request failed") from error
            response = self._read_record(self._timeout_seconds)
            if response.get("id") != identifier:
                raise WorkerProcessError("worker response id mismatch")
            if "error" in response:
                error_payload = response["error"]
                if isinstance(error_payload, Mapping) and isinstance(
                    error_payload.get("message"), str
                ):
                    raise WorkerProcessError(error_payload["message"])
                raise WorkerProcessError("worker inference failed")
            events = response.get("events")
            if not isinstance(events, list):
                raise WorkerProcessError("worker response is invalid")
            return response

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
        try:
            self._process.wait(timeout=max(self._timeout_seconds, 0.1))
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=max(self._timeout_seconds, 0.1))
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
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
