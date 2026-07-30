#!/usr/bin/env python3
"""Native Linux Docker orchestration for the OaF calibration-only protocol."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import select
import shutil
import stat
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast
from uuid import uuid4

from runtime.oaf_tf1.protocol import ProtocolFailure, decode_native_event
from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.backend_publication import read_regular_file_no_follow
from tools.hpa320.seal_oaf_backend import (
    CalibrationBootstrapEvidence,
    CalibrationBootstrapRequest,
    CalibrationMeasurementRequest,
    CalibrationProbeResult,
    MeasurementRow,
    SealError,
)

_READY_KEYS = frozenset(
    {
        "base_system_package_evidence_sha256",
        "calibration_bootstrap_request_sha256",
        "checkpoint_acquisition_evidence_sha256",
        "checkpoint_inventory_sha256",
        "non_inference_count",
        "non_inference_inventory_sha256",
        "process_instance_id",
        "protocol_schema",
        "required_inference_count",
        "required_inference_inventory_sha256",
        "restored_inference_count",
        "runner_source_manifest_sha256",
        "runtime_image_config_digest",
        "tensorflow_abi",
        "tensorflow_build",
        "type",
        "upstream_source_manifest_sha256",
    }
)
_RESPONSE_KEYS = frozenset(
    {
        "audio_sha256",
        "inference_call_count_after",
        "inference_call_count_before",
        "native_events",
        "prediction_sha256",
        "rejected_before_inference",
        "request_id",
        "type",
    }
)
_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES=-1",
    "MKL_NUM_THREADS=1",
    "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "PYTHONHASHSEED=0",
    "PYTHONCOERCECLOCALE=0",
    "TF_NUM_INTEROP_THREADS=1",
    "TF_NUM_INTRAOP_THREADS=1",
)


@dataclass(frozen=True)
class NativeCalibrationResponse:
    """A typed protocol result and its host-observed resource metrics."""

    row: MeasurementRow
    native_events: tuple[Mapping[str, Any], ...]
    rejected_before_inference: bool


class _Diagnostics:
    def __init__(self, ring_limit: int) -> None:
        self.maximum_line_bytes = 0
        self._current_line_bytes = 0
        self._ring_limit = ring_limit
        self._ring = bytearray()
        self._lock = threading.Lock()

    def consume(self, content: bytes) -> None:
        with self._lock:
            for byte in content:
                if byte == 10:
                    self.maximum_line_bytes = max(self.maximum_line_bytes, self._current_line_bytes)
                    self._current_line_bytes = 0
                else:
                    self._current_line_bytes += 1
            self._ring.extend(content)
            if len(self._ring) > self._ring_limit:
                del self._ring[: len(self._ring) - self._ring_limit]

    def finish(self) -> None:
        with self._lock:
            self.maximum_line_bytes = max(self.maximum_line_bytes, self._current_line_bytes)

    def observed_maximum_line_bytes(self) -> int:
        with self._lock:
            return max(self.maximum_line_bytes, self._current_line_bytes)


class _CgroupMonitor:
    def __init__(self, process_id: int, interval_millis: int) -> None:
        self._process_id = process_id
        self._interval_seconds = interval_millis / 1000
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.failure: BaseException | None = None
        self.peak_cpu_millis = 0
        self.peak_rss_bytes = 0
        self.peak_pid_count = 0
        self.peak_tmp_bytes = 0
        self.peak_shm_bytes = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise SealError("calibration cgroup monitor did not stop")
        if self.failure is not None:
            raise SealError("calibration cgroup monitor failed")

    def _run(self) -> None:
        try:
            cgroup = _cgroup_root(self._process_id)
            previous_usage: int | None = None
            previous_time: int | None = None
            while not self._stop.is_set():
                current_time = time.monotonic_ns()
                usage = _cpu_usage_usec(cgroup / "cpu.stat")
                if previous_usage is not None and previous_time is not None:
                    elapsed_usec = max(1, (current_time - previous_time) // 1_000)
                    cpu_millis = math.ceil(1_000 * max(0, usage - previous_usage) / elapsed_usec)
                    self.peak_cpu_millis = max(self.peak_cpu_millis, cpu_millis)
                previous_usage = usage
                previous_time = current_time
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes, _read_nonnegative_int(cgroup / "memory.peak")
                )
                self.peak_pid_count = max(
                    self.peak_pid_count, _read_nonnegative_int(cgroup / "pids.peak")
                )
                process_root = Path("/proc") / str(self._process_id) / "root"
                self.peak_tmp_bytes = max(
                    self.peak_tmp_bytes, _allocated_tree_bytes(process_root / "tmp")
                )
                self.peak_shm_bytes = max(
                    self.peak_shm_bytes, _allocated_tree_bytes(process_root / "dev/shm")
                )
                self._stop.wait(self._interval_seconds)
        except BaseException as error:  # pylint: disable=broad-exception-caught
            if not self._stop.is_set():
                self.failure = error


class _CalibrationContainer:
    def __init__(
        self,
        *,
        bootstrap_request: CalibrationBootstrapRequest,
        bootstrap: CalibrationBootstrapEvidence,
        bootstrap_request_path: Path,
        bootstrap_evidence_path: Path,
        checkpoint_evidence_path: Path,
        base_system_evidence_path: Path,
        checkpoint_request_path: Path,
        base_system_request_path: Path,
        model_cache: Path,
        input_root: Path,
        candidate_evidence_root: Path | None,
    ) -> None:
        self._bootstrap_request = bootstrap_request
        self._bootstrap = bootstrap
        self._stdout_maximum = cast(
            int, bootstrap_request.payload["resource_ceiling"]["stdout_max_line_bytes"]
        )
        self._stderr = _Diagnostics(
            cast(
                int,
                bootstrap_request.payload["resource_ceiling"]["stderr_ring_buffer_bytes"],
            )
        )
        self._stdout_line_buffer = b""
        self._container_name = "hpa320-calibration-" + uuid4().hex
        self._runtime_digest_file = input_root.parent / (
            self._container_name + "-runtime-digest.txt"
        )
        self._runtime_digest_file.write_text(
            cast(str, bootstrap.payload["runtime_image_config_digest"]) + "\n",
            encoding="ascii",
        )
        command = _docker_create_command(
            name=self._container_name,
            bootstrap_request=bootstrap_request,
            bootstrap=bootstrap,
            bootstrap_request_path=bootstrap_request_path,
            bootstrap_evidence_path=bootstrap_evidence_path,
            checkpoint_evidence_path=checkpoint_evidence_path,
            base_system_evidence_path=base_system_evidence_path,
            checkpoint_request_path=checkpoint_request_path,
            base_system_request_path=base_system_request_path,
            model_cache=model_cache,
            input_root=input_root,
            runtime_digest_file=self._runtime_digest_file,
            candidate_evidence_root=candidate_evidence_root,
        )
        created = _run_checked(command, "calibration container creation failed")
        self.container_id = created.stdout.decode("ascii", errors="strict").strip()
        if len(self.container_id) != 64:
            self._cleanup()
            raise SealError("Docker did not return a full calibration container ID")
        self._closed = False
        self._monitor: _CgroupMonitor | None = None
        self._stderr_thread: threading.Thread | None = None
        self._started_ns = time.monotonic_ns()
        try:
            self._process = subprocess.Popen(
                ("docker", "start", "--attach", "--interactive", self.container_id),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            self._cleanup()
            raise SealError("calibration container start failed") from error
        if (
            self._process.stdin is None
            or self._process.stdout is None
            or self._process.stderr is None
        ):
            self._abort_startup()
            raise SealError("calibration container pipes are unavailable")
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        try:
            process_id = _await_container_pid(
                self.container_id,
                cast(
                    int,
                    bootstrap_request.payload["resource_ceiling"]["startup_deadline_seconds"],
                ),
            )
            self._monitor = _CgroupMonitor(
                process_id,
                cast(
                    int,
                    bootstrap_request.payload["resource_ceiling"]["monitor_interval_millis"],
                ),
            )
            self._monitor.start()
            self.ready = self._read_object(
                cast(
                    int,
                    bootstrap_request.payload["resource_ceiling"]["startup_deadline_seconds"],
                )
            )
            self.startup_millis = _ceil_millis(time.monotonic_ns() - self._started_ns)
            _validate_ready(
                self.ready,
                bootstrap_request=bootstrap_request,
                bootstrap=bootstrap,
                checkpoint_evidence_sha256=_sha256_regular(checkpoint_evidence_path),
                base_system_evidence_sha256=_sha256_regular(base_system_evidence_path),
                container_id=self.container_id,
            )
        except BaseException:
            self._abort_startup()
            raise

    def request(
        self,
        *,
        request_type: str,
        request_id: str,
        audio_path: str,
        audio_sha256: str,
        audio_frame_count: int,
        max_input_audio_frames: int,
        repetition: int,
    ) -> NativeCalibrationResponse:
        before_ns = time.monotonic_ns()
        content = canonical_json_bytes(
            {
                "audio_frame_count": audio_frame_count,
                "audio_path": audio_path,
                "audio_sha256": audio_sha256,
                "max_input_audio_frames": max_input_audio_frames,
                "request_id": request_id,
                "type": request_type,
            },
            trailing_newline=True,
        )
        try:
            cast(Any, self._process.stdin).write(content)
            cast(Any, self._process.stdin).flush()
        except OSError:
            raise SealError("calibration request pipe failed") from None
        response = self._read_object(
            cast(
                int,
                self._bootstrap_request.payload["resource_ceiling"]["request_deadline_seconds"],
            )
        )
        request_millis = _ceil_millis(time.monotonic_ns() - before_ns)
        _validate_response(
            response,
            request_type=request_type,
            request_id=request_id,
            audio_sha256=audio_sha256,
        )
        native_events = tuple(cast(Sequence[Mapping[str, Any]], response["native_events"]))
        monitor = self._require_monitor()
        row = MeasurementRow(
            input_audio_sha256=audio_sha256,
            input_frame_count=audio_frame_count,
            repetition=repetition,
            process_instance_id=self.container_id,
            inference_call_count_before=cast(int, response["inference_call_count_before"]),
            inference_call_count_after=cast(int, response["inference_call_count_after"]),
            peak_cpu_millis=monitor.peak_cpu_millis,
            peak_rss_bytes=monitor.peak_rss_bytes,
            peak_tmp_bytes=monitor.peak_tmp_bytes,
            peak_shm_bytes=monitor.peak_shm_bytes,
            peak_pid_count=monitor.peak_pid_count,
            startup_millis=self.startup_millis,
            request_millis=request_millis,
            stdout_max_line_bytes=max(
                len(canonical_json_bytes(self.ready, trailing_newline=True)) - 1,
                len(canonical_json_bytes(response, trailing_newline=True)) - 1,
            ),
            stderr_max_line_bytes=self._stderr.observed_maximum_line_bytes(),
            exit_code=0,
            signal=None,
            oom_killed=False,
            prediction_sha256=cast(str | None, response["prediction_sha256"]),
        )
        return NativeCalibrationResponse(
            row=row,
            native_events=native_events,
            rejected_before_inference=cast(bool, response["rejected_before_inference"]),
        )

    def observed_row(
        self,
        row: MeasurementRow,
        *,
        exit_code: int = 0,
        signal: int | None = None,
        oom_killed: bool = False,
    ) -> MeasurementRow:
        """Refresh one row from the latest host-observed process metrics."""

        monitor = self._require_monitor()
        return MeasurementRow(
            **{
                **row.__dict__,
                "exit_code": exit_code,
                "oom_killed": oom_killed,
                "peak_cpu_millis": max(row.peak_cpu_millis, monitor.peak_cpu_millis),
                "peak_pid_count": max(row.peak_pid_count, monitor.peak_pid_count),
                "peak_rss_bytes": max(row.peak_rss_bytes, monitor.peak_rss_bytes),
                "peak_shm_bytes": max(row.peak_shm_bytes, monitor.peak_shm_bytes),
                "peak_tmp_bytes": max(row.peak_tmp_bytes, monitor.peak_tmp_bytes),
                "signal": signal,
                "stderr_max_line_bytes": max(
                    row.stderr_max_line_bytes,
                    self._stderr.observed_maximum_line_bytes(),
                ),
            }
        )

    def close(self) -> tuple[int, int | None, bool]:
        if self._closed:
            return (0, None, False)
        self._closed = True
        try:
            cast(Any, self._process.stdin).close()
            return_code = self._process.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self._terminate()
            raise SealError("calibration container did not exit cleanly") from None
        monitor_failure = False
        try:
            self._require_monitor().stop()
        except SealError:
            monitor_failure = True
        stderr_thread = self._stderr_thread
        if stderr_thread is not None:
            stderr_thread.join(timeout=5)
            if stderr_thread.is_alive():
                monitor_failure = True
        self._stderr.finish()
        try:
            state = _container_state(self.container_id)
        finally:
            self._cleanup()
        if monitor_failure:
            raise SealError("calibration monitor or diagnostic drain failed")
        exit_code = cast(int, state["ExitCode"])
        oom_killed = cast(bool, state["OOMKilled"])
        signal = exit_code - 128 if exit_code >= 129 else None
        if return_code != exit_code:
            raise SealError("Docker attach and container exit codes differ")
        return exit_code, signal, oom_killed

    def _read_object(self, deadline_seconds: int) -> Mapping[str, Any]:
        deadline = time.monotonic() + deadline_seconds
        stdout_fd = cast(Any, self._process.stdout).fileno()
        while True:
            newline_index = self._stdout_line_buffer.find(b"\n")
            if newline_index >= 0:
                content = self._stdout_line_buffer[: newline_index + 1]
                self._stdout_line_buffer = self._stdout_line_buffer[newline_index + 1 :]
                if len(content) > self._stdout_maximum:
                    raise SealError("calibration protocol line is oversized")
                if not content.endswith(b"\n") or content.endswith(b"\n\n"):
                    raise SealError("calibration protocol line framing is invalid")
                try:
                    value = strict_json_loads(content[:-1], require_canonical=True)
                except ValueError:
                    raise SealError("calibration protocol response is invalid") from None
                if not isinstance(value, Mapping):
                    raise SealError("calibration protocol response is not an object")
                return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SealError("calibration protocol deadline expired")
            if len(self._stdout_line_buffer) > self._stdout_maximum:
                raise SealError("calibration protocol line is oversized")
            readable, _, _ = select.select([stdout_fd], [], [], min(remaining, 1.0))
            if not readable:
                if self._process.poll() is not None:
                    raise SealError("calibration container exited before its response")
                continue
            chunk = os.read(stdout_fd, self._stdout_maximum + 1)
            if not chunk:
                raise SealError("calibration protocol stream closed before its response")
            self._stdout_line_buffer += chunk

    def _drain_stderr(self) -> None:
        stderr = cast(Any, self._process.stderr)
        chunk_bytes = cast(
            int,
            self._bootstrap_request.payload["resource_ceiling"]["stderr_read_chunk_bytes"],
        )
        while True:
            content = stderr.read(chunk_bytes)
            if not content:
                return
            self._stderr.consume(content)

    def _terminate(self) -> None:
        self._closed = True
        self._abort_startup()

    def _abort_startup(self) -> None:
        _run_checked(
            ("docker", "kill", self.container_id),
            "calibration container kill failed",
            tolerate_failure=True,
        )
        process = getattr(self, "_process", None)
        if process is not None:
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        monitor = self._monitor
        if monitor is not None:
            try:
                monitor.stop()
            except SealError:
                pass
        stderr_thread = self._stderr_thread
        if stderr_thread is not None:
            stderr_thread.join(timeout=5)
        self._stderr.finish()
        self._cleanup()

    def _require_monitor(self) -> _CgroupMonitor:
        if self._monitor is None:
            raise SealError("calibration cgroup monitor is unavailable")
        return self._monitor

    def _cleanup(self) -> None:
        _run_checked(
            ("docker", "rm", "--force", self._container_name),
            "calibration container cleanup failed",
            tolerate_failure=True,
        )
        self._runtime_digest_file.unlink(missing_ok=True)


class NativeCalibrationRunner:
    """Materialize exact fixtures and expose callbacks used by seal orchestration."""

    def __init__(
        self,
        *,
        repository_root: Path,
        bootstrap_request: CalibrationBootstrapRequest,
        bootstrap: CalibrationBootstrapEvidence,
        bootstrap_request_path: Path,
        bootstrap_evidence_path: Path,
        checkpoint_evidence_path: Path,
        base_system_evidence_path: Path,
        model_cache: Path,
        max_input_audio_frames: int,
        candidate_evidence_root: Path | None = None,
    ) -> None:
        _require_native_linux_amd64()
        self._repository_root = Path(repository_root)
        self._bootstrap_request = bootstrap_request
        self._bootstrap = bootstrap
        self._bootstrap_request_path = Path(bootstrap_request_path).resolve()
        self._bootstrap_evidence_path = Path(bootstrap_evidence_path).resolve()
        self._checkpoint_evidence_path = Path(checkpoint_evidence_path).resolve()
        self._base_system_evidence_path = Path(base_system_evidence_path).resolve()
        self._model_cache = Path(model_cache).resolve()
        self._maximum = max_input_audio_frames
        self._candidate_evidence_root = (
            None if candidate_evidence_root is None else Path(candidate_evidence_root).resolve()
        )
        self._temporary = Path(tempfile.mkdtemp(prefix=".hpa320-native-runner-"))
        self._input_root = self._temporary / "input"
        self._input_root.mkdir()
        self._persistent: _CalibrationContainer | None = None
        self._ready_identity: Mapping[str, Any] | None = None

    def close(self) -> None:
        try:
            if self._persistent is not None:
                exit_code, signal, oom = self._persistent.close()
                if exit_code != 0 or signal is not None or oom:
                    raise SealError("persistent calibration container failed")
        finally:
            self._persistent = None
            shutil.rmtree(self._temporary, ignore_errors=True)

    def measure(
        self,
        request: CalibrationMeasurementRequest,
        frame_count: int,
        repetition: int,
    ) -> MeasurementRow:
        fixture = next(
            cast(Mapping[str, Any], row)
            for row in cast(Sequence[Mapping[str, Any]], request.payload["fixtures"])
            if row["audio_frame_count"] == frame_count
        )
        relative, digest = self._materialize_fixture(frame_count, fixture)
        container = self._container()
        try:
            response = container.request(
                request_type="measure",
                request_id=f"measure-{frame_count}-{repetition}",
                audio_path=relative,
                audio_sha256=digest,
                audio_frame_count=frame_count,
                max_input_audio_frames=self._maximum,
                repetition=repetition,
            )
            exit_code, signal, oom = container.close()
        except BaseException:
            container._terminate()  # pylint: disable=protected-access
            raise
        return container.observed_row(
            response.row,
            exit_code=exit_code,
            signal=signal,
            oom_killed=oom,
        )

    def probe(self, frame_count: int, persistent: bool, ordinal: int) -> CalibrationProbeResult:
        relative, digest = self._materialize_fixture(frame_count, None)
        container = self._persistent if persistent else None
        if container is None:
            container = self._container()
            if persistent:
                self._persistent = container
        response = container.request(
            request_type="calibration_probe",
            request_id=f"probe-{'persistent' if persistent else 'fresh'}-{ordinal}",
            audio_path=relative,
            audio_sha256=digest,
            audio_frame_count=frame_count,
            max_input_audio_frames=self._maximum,
            repetition=ordinal,
        )
        if not persistent:
            exit_code, signal, oom = container.close()
            response = NativeCalibrationResponse(
                row=container.observed_row(
                    response.row,
                    exit_code=exit_code,
                    signal=signal,
                    oom_killed=oom,
                ),
                native_events=response.native_events,
                rejected_before_inference=response.rejected_before_inference,
            )
        else:
            response = NativeCalibrationResponse(
                row=container.observed_row(response.row),
                native_events=response.native_events,
                rejected_before_inference=response.rejected_before_inference,
            )
        return CalibrationProbeResult(
            row=response.row,
            rejected_before_inference=response.rejected_before_inference,
        )

    def smoke(self) -> tuple[Mapping[str, Any], ...]:
        """Run the checked-in canonical smoke WAV in its own fresh process."""

        relative, digest, frame_count = self._materialize_canonical_smoke()
        container = self._container()
        try:
            response = container.request(
                request_type="calibration_probe",
                request_id="smoke-canonical",
                audio_path=relative,
                audio_sha256=digest,
                audio_frame_count=frame_count,
                max_input_audio_frames=self._maximum,
                repetition=0,
            )
            exit_code, signal, oom = container.close()
        except BaseException:
            container._terminate()  # pylint: disable=protected-access
            raise
        if (
            response.rejected_before_inference
            or not response.native_events
            or exit_code != 0
            or signal is not None
            or oom
        ):
            raise SealError("canonical calibration smoke inference failed")
        self._validate_tensor_evidence()
        return response.native_events

    def _container(self) -> _CalibrationContainer:
        container = _CalibrationContainer(
            bootstrap_request=self._bootstrap_request,
            bootstrap=self._bootstrap,
            bootstrap_request_path=self._bootstrap_request_path,
            bootstrap_evidence_path=self._bootstrap_evidence_path,
            checkpoint_evidence_path=self._checkpoint_evidence_path,
            base_system_evidence_path=self._base_system_evidence_path,
            checkpoint_request_path=(
                self._repository_root / "config/benchmark/backends/"
                "magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json"
            ).resolve(),
            base_system_request_path=(
                self._repository_root / "runtime/oaf_tf1/base-system-package-request.json"
            ).resolve(),
            model_cache=self._model_cache,
            input_root=self._input_root,
            candidate_evidence_root=self._candidate_evidence_root,
        )
        identity = {
            field: container.ready[field]
            for field in (
                "checkpoint_inventory_sha256",
                "non_inference_inventory_sha256",
                "required_inference_inventory_sha256",
            )
        }
        if self._ready_identity is None:
            self._ready_identity = identity
        elif identity != self._ready_identity:
            container._terminate()  # pylint: disable=protected-access
            raise SealError("calibration tensor identity changed between processes")
        return container

    def _materialize_fixture(
        self, frame_count: int, expected: Mapping[str, Any] | None
    ) -> tuple[str, str]:
        source_relative = "tests/fixtures/oaf_tf1_smoke/canonical.wav"
        source = self._repository_root / source_relative
        content = _derive_fixture(source.read_bytes(), frame_count)
        digest = hashlib.sha256(content).hexdigest()
        if expected is not None and (
            expected["input_audio_sha256"] != digest or expected["wav_byte_length"] != len(content)
        ):
            raise SealError("derived calibration fixture differs from its request")
        relative = f"fixtures/{frame_count}.wav"
        target = self._input_root / relative
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if target.read_bytes() != content:
                raise SealError("calibration fixture path already differs")
        else:
            target.write_bytes(content)
        return relative, digest

    def _materialize_canonical_smoke(self) -> tuple[str, str, int]:
        source = self._repository_root / "tests/fixtures/oaf_tf1_smoke/canonical.wav"
        content = read_regular_file_no_follow(source)
        if (
            len(content) < 46
            or content[:4] != b"RIFF"
            or content[8:16] != b"WAVEfmt "
            or content[36:40] != b"data"
            or len(content[44:]) % 2
        ):
            raise SealError("canonical calibration smoke fixture is invalid")
        frame_count = len(content[44:]) // 2
        if frame_count <= 0 or frame_count > self._maximum:
            raise SealError("canonical calibration smoke fixture exceeds its authority")
        relative = "fixtures/canonical-smoke.wav"
        target = self._input_root / relative
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if read_regular_file_no_follow(target) != content:
                raise SealError("canonical calibration smoke fixture already differs")
        else:
            target.write_bytes(content)
        return relative, hashlib.sha256(content).hexdigest(), frame_count

    def _validate_tensor_evidence(self) -> None:
        if self._candidate_evidence_root is None or self._ready_identity is None:
            raise SealError("calibration tensor evidence authority is unavailable")
        try:
            content = read_regular_file_no_follow(
                self._candidate_evidence_root / "tensor-coverage.json"
            )
        except OSError:
            raise SealError("calibration tensor evidence is missing or unsafe") from None
        if not content.endswith(b"\n") or content.endswith(b"\n\n"):
            raise SealError("calibration tensor evidence is not canonical JSON")
        try:
            payload = strict_json_loads(content[:-1], require_canonical=True)
        except ValueError:
            raise SealError("calibration tensor evidence is invalid") from None
        _validate_tensor_coverage_against_ready(payload, self._ready_identity)


def _docker_create_command(
    *,
    name: str,
    bootstrap_request: CalibrationBootstrapRequest,
    bootstrap: CalibrationBootstrapEvidence,
    bootstrap_request_path: Path,
    bootstrap_evidence_path: Path,
    checkpoint_evidence_path: Path,
    base_system_evidence_path: Path,
    checkpoint_request_path: Path,
    base_system_request_path: Path,
    model_cache: Path,
    input_root: Path,
    runtime_digest_file: Path,
    candidate_evidence_root: Path | None,
) -> tuple[str, ...]:
    ceiling = cast(Mapping[str, int], bootstrap_request.payload["resource_ceiling"])
    command = [
        "docker",
        "create",
        "--name",
        name,
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{bootstrap_request.runtime_uid}:{bootstrap_request.runtime_gid}",
        "--cpus",
        str(ceiling["cpu_limit_millis"] / 1_000),
        "--memory",
        str(ceiling["memory_limit_bytes"]),
        "--pids-limit",
        str(ceiling["pid_limit"]),
        "--tmpfs",
        (
            "/tmp:rw,nosuid,nodev,size="
            f"{ceiling['tmp_bytes']},uid={bootstrap_request.runtime_uid},"
            f"gid={bootstrap_request.runtime_gid},mode=1777"
        ),
        "--tmpfs",
        (
            "/dev/shm:rw,nosuid,nodev,size="
            f"{ceiling['shm_bytes']},uid={bootstrap_request.runtime_uid},"
            f"gid={bootstrap_request.runtime_gid},mode=1777"
        ),
    ]
    mounts = (
        (
            bootstrap_request_path,
            "/run/crux/calibration-bootstrap-request.json",
            True,
        ),
        (
            bootstrap_evidence_path,
            "/run/crux/calibration-bootstrap-evidence.json",
            True,
        ),
        (
            checkpoint_request_path,
            "/run/crux/checkpoint-acquisition-request.json",
            True,
        ),
        (
            checkpoint_evidence_path,
            "/run/crux/checkpoint-acquisition-evidence.json",
            True,
        ),
        (
            base_system_request_path,
            "/run/crux/base-system-package-request.json",
            True,
        ),
        (
            base_system_evidence_path,
            "/run/crux/base-system-package-evidence.json",
            True,
        ),
        (runtime_digest_file.resolve(), "/run/crux/runtime-image-config-digest.txt", True),
        (model_cache, "/model", True),
        (input_root.resolve(), "/input", True),
    )
    for source, destination, readonly in mounts:
        command.extend(
            (
                "--mount",
                "type=bind,src="
                + os.fspath(source)
                + ",dst="
                + destination
                + (",readonly" if readonly else ""),
            )
        )
    if candidate_evidence_root is not None:
        command.extend(
            (
                "--mount",
                "type=bind,src=" + os.fspath(candidate_evidence_root) + ",dst=/output",
            )
        )
    command.extend(
        (
            "--entrypoint",
            "/usr/bin/env",
            cast(str, bootstrap.payload["runtime_image_config_digest"]),
            "-i",
            *_ENVIRONMENT,
            "/opt/crux/venv/bin/python",
            "-s",
            "/opt/crux/runtime/calibration_entrypoint.py",
        )
    )
    return tuple(command)


def _validate_ready(
    value: Mapping[str, Any],
    *,
    bootstrap_request: CalibrationBootstrapRequest,
    bootstrap: CalibrationBootstrapEvidence,
    checkpoint_evidence_sha256: str,
    base_system_evidence_sha256: str,
    container_id: str,
) -> None:
    if (
        set(value) != _READY_KEYS
        or value["type"] != "ready"
        or value["protocol_schema"] != "crux.oaf-calibration-runner/v1"
        or value["calibration_bootstrap_request_sha256"] != bootstrap_request.sha256
        or value["checkpoint_acquisition_evidence_sha256"] != checkpoint_evidence_sha256
        or value["base_system_package_evidence_sha256"] != base_system_evidence_sha256
        or value["runner_source_manifest_sha256"]
        != bootstrap_request.payload["runner_source_manifest_sha256"]
        or value["upstream_source_manifest_sha256"]
        != bootstrap_request.payload["upstream_source_manifest_sha256"]
        or value["runtime_image_config_digest"] != bootstrap.payload["runtime_image_config_digest"]
        or value["process_instance_id"] != container_id[:12]
        or value["required_inference_count"] != 78
        or value["restored_inference_count"] != 78
        or value["non_inference_count"] != 52
        or value["tensorflow_abi"] != "cp37-cp37m-manylinux2010_x86_64"
        or value["tensorflow_build"] != "v1.15.5-0-g590d6eef7e"
        or any(
            not isinstance(value[field], str)
            or len(cast(str, value[field])) != 64
            or any(character not in "0123456789abcdef" for character in cast(str, value[field]))
            for field in (
                "checkpoint_inventory_sha256",
                "required_inference_inventory_sha256",
                "non_inference_inventory_sha256",
            )
        )
    ):
        raise SealError("calibration ready response is unrelated or invalid")


def _validate_response(
    value: Mapping[str, Any],
    *,
    request_type: str,
    request_id: str,
    audio_sha256: str,
) -> None:
    if (
        set(value) != _RESPONSE_KEYS
        or value["type"] != request_type
        or value["request_id"] != request_id
        or value["audio_sha256"] != audio_sha256
        or not isinstance(value["native_events"], list)
        or type(value["rejected_before_inference"]) is not bool
        or type(value["inference_call_count_before"]) is not int
        or type(value["inference_call_count_after"]) is not int
    ):
        raise SealError("calibration response is unrelated or invalid")
    prediction = value["prediction_sha256"]
    if value["rejected_before_inference"]:
        if (
            value["native_events"] != []
            or prediction is not None
            or value["inference_call_count_before"] != value["inference_call_count_after"]
        ):
            raise SealError("calibration rejection occurred after inference")
    else:
        try:
            for event in value["native_events"]:
                decode_native_event(event)
        except ProtocolFailure:
            raise SealError("calibration response native event is invalid") from None
        content = canonical_json_bytes(value["native_events"])
        if (
            not isinstance(prediction, str)
            or prediction != hashlib.sha256(content).hexdigest()
            or value["inference_call_count_after"] != value["inference_call_count_before"] + 1
        ):
            raise SealError("calibration inference response identity is invalid")


def _validate_tensor_coverage_against_ready(
    value: object,
    ready_identity: Mapping[str, Any],
) -> None:
    keys = frozenset(
        {
            "active_predict_dropout",
            "checkpoint_inventory",
            "non_inference_inventory",
            "note_sequence_byte_parity",
            "required_inference_inventory",
            "schema",
            "uninitialized_required",
        }
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != keys
        or value["schema"] != "crux.oaf-tensor-coverage/v1"
        or value["active_predict_dropout"] is not False
        or value["note_sequence_byte_parity"] is not True
        or value["uninitialized_required"] != []
    ):
        raise SealError("calibration tensor evidence fields are invalid")
    inventories = (
        ("checkpoint_inventory", "checkpoint_inventory_sha256", 130),
        ("required_inference_inventory", "required_inference_inventory_sha256", 78),
        ("non_inference_inventory", "non_inference_inventory_sha256", 52),
    )
    for field, identity_field, count in inventories:
        inventory = value[field]
        if (
            not isinstance(inventory, list)
            or len(inventory) != count
            or hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
            != ready_identity[identity_field]
        ):
            raise SealError("calibration tensor evidence differs from the ready handshake")


def _sha256_regular(path: Path) -> str:
    try:
        return hashlib.sha256(read_regular_file_no_follow(Path(path))).hexdigest()
    except OSError:
        raise SealError("calibration evidence path is missing or unsafe") from None


def _derive_fixture(source: bytes, frame_count: int) -> bytes:
    if (
        len(source) < 46
        or source[:4] != b"RIFF"
        or source[8:16] != b"WAVEfmt "
        or source[36:40] != b"data"
        or type(frame_count) is not int
        or frame_count <= 0
    ):
        raise SealError("canonical calibration fixture source is invalid")
    pcm = source[44:]
    if not pcm or len(pcm) % 2:
        raise SealError("canonical calibration fixture PCM is invalid")
    byte_count = frame_count * 2
    repeated = (pcm * ((byte_count + len(pcm) - 1) // len(pcm)))[:byte_count]
    return (
        source[:4]
        + struct.pack("<I", 36 + byte_count)
        + source[8:40]
        + struct.pack("<I", byte_count)
        + repeated
    )


def _await_container_pid(container_id: str, deadline_seconds: int) -> int:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        result = _run_checked(
            ("docker", "inspect", "--format", "{{.State.Pid}}", container_id),
            "calibration container PID inspection failed",
        )
        try:
            process_id = int(result.stdout.strip())
        except ValueError:
            process_id = 0
        if process_id > 0:
            return process_id
        time.sleep(0.01)
    raise SealError("calibration container PID did not become available")


def _container_state(container_id: str) -> Mapping[str, Any]:
    result = _run_checked(
        ("docker", "inspect", "--format", "{{json .State}}", container_id),
        "calibration container state inspection failed",
    )
    try:
        value = json.loads(result.stdout)
    except ValueError:
        raise SealError("calibration container state is invalid") from None
    if (
        not isinstance(value, Mapping)
        or type(value.get("ExitCode")) is not int
        or type(value.get("OOMKilled")) is not bool
    ):
        raise SealError("calibration container state is incomplete")
    return value


def _cgroup_root(process_id: int) -> Path:
    content = (Path("/proc") / str(process_id) / "cgroup").read_text(encoding="ascii")
    rows = [line for line in content.splitlines() if line.startswith("0::")]
    if len(rows) != 1:
        raise SealError("calibration requires one cgroup v2 hierarchy")
    relative = rows[0][3:]
    if not relative.startswith("/") or ".." in Path(relative).parts:
        raise SealError("calibration cgroup path is invalid")
    root = Path("/sys/fs/cgroup") / relative.removeprefix("/")
    for name in ("cpu.stat", "memory.peak", "pids.peak"):
        if not (root / name).is_file():
            raise SealError("calibration cgroup metrics are unavailable")
    return root


def _cpu_usage_usec(path: Path) -> int:
    rows = {}
    for line in path.read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition(" ")
        if separator:
            rows[key] = value
    try:
        usage = int(rows["usage_usec"])
    except (KeyError, ValueError):
        raise SealError("calibration cpu.stat is invalid") from None
    if usage < 0:
        raise SealError("calibration cpu.stat is negative")
    return usage


def _read_nonnegative_int(path: Path) -> int:
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        raise SealError("calibration cgroup counter is invalid") from None
    if value < 0:
        raise SealError("calibration cgroup counter is negative")
    return value


def _allocated_tree_bytes(root: Path) -> int:
    total = 0
    for directory, names, files in os.walk(root, followlinks=False):
        if names:
            names[:] = [name for name in names if not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            try:
                metadata = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_blocks * 512
    return total


def _ceil_millis(nanoseconds: int) -> int:
    return max(1, (nanoseconds + 999_999) // 1_000_000)


def _run_checked(
    command: tuple[str, ...],
    label: str,
    *,
    tolerate_failure: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        if tolerate_failure:
            return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"")
        raise SealError(label) from error
    if result.returncode != 0 and not tolerate_failure:
        raise SealError(label)
    return result


def _require_native_linux_amd64() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "amd64",
        "x86_64",
    }:
        raise SealError("native calibration execution requires Linux amd64")
