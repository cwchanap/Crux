from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from runtime.oaf_tf1.protocol import serve_requests
from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.backend_lock import REQUIRED_ENVIRONMENT
from src.benchmark.backend_process import (
    DiagnosticHostEvidence,
    NativeHostEvidence,
    RunnerLaunchProfile,
    RunnerProcess,
    build_docker_command,
)
from src.benchmark.backends.base import BackendFatalFailure

FAKE_RUNNER = Path(__file__).parents[1] / "fixtures" / "fake_oaf_runner.py"


def _canonical_sha256(payload: dict[str, object]) -> str:
    from src.benchmark.backend_identity import canonical_json_bytes

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _write_mounts(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    backend = tmp_path / "backend.json"
    runtime = tmp_path / "runtime.json"
    seal = tmp_path / "seal-evidence.json"
    model = tmp_path / "model"
    inputs = tmp_path / "inputs"
    backend.write_bytes(b"{}\n")
    runtime.write_bytes(b"{}\n")
    seal.write_bytes(b"{}\n")
    model.mkdir()
    inputs.mkdir()
    return backend, runtime, seal, model, inputs


def profile(tmp_path: Path, **changes: object) -> RunnerLaunchProfile:
    backend, runtime, seal, model, inputs = _write_mounts(tmp_path)
    values: dict[str, object] = {
        "image_manifest_digest": f"sha256:{'4' * 64}",
        "backend_lock_path": backend,
        "runtime_lock_path": runtime,
        "seal_evidence_path": seal,
        "model_cache_path": model,
        "input_root": inputs,
        "environment": dict(REQUIRED_ENVIRONMENT),
        "uid": 10001,
        "gid": 10002,
        "cpu_limit": "2.5",
        "memory_bytes": 1073741824,
        "pid_limit": 64,
        "tmp_bytes": 134217728,
        "shm_bytes": 67108864,
        "startup_deadline_seconds": 1,
        "request_deadline_seconds": 1,
        "stdout_max_line_bytes": 2048,
        "stderr_read_chunk_bytes": 17,
        "stderr_max_line_bytes": 256,
        "stderr_ring_buffer_bytes": 4096,
    }
    values.update(changes)
    return RunnerLaunchProfile(**values)  # type: ignore[arg-type]


def fake_popen(mode: str):
    def factory(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        assert command[:4] == ["docker", "run", "--rm", "-i"]
        assert kwargs["env"] == {}
        assert kwargs["shell"] is False
        assert kwargs["start_new_session"] is True
        return subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, str(FAKE_RUNNER), mode],
            stdin=kwargs["stdin"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
            env={},
            shell=False,
            start_new_session=True,
        )

    return factory


def capturing_fake_popen(mode: str, captured: list[subprocess.Popen[bytes]]):
    base_factory = fake_popen(mode)

    def factory(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        process = base_factory(command, **kwargs)
        captured.append(process)
        return process

    return factory


def test_docker_command_contains_exact_hardening_and_fresh_environment(tmp_path: Path) -> None:
    launch = profile(tmp_path)

    command = build_docker_command(launch)

    assert command[:4] == ["docker", "run", "--rm", "-i"]
    assert command[4:15] == [
        "--platform=linux/amd64",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=10001:10002",
        "--cpus=2.5",
        "--memory=1073741824",
        "--pids-limit=64",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=134217728",
        "--tmpfs=/dev/shm:rw,noexec,nosuid,nodev,size=67108864",
    ]
    env_args = [item for item in command if item.startswith("--env=")]
    assert env_args == [
        f"--env={key}={REQUIRED_ENVIRONMENT[key]}" for key in sorted(REQUIRED_ENVIRONMENT)
    ]
    assert command[-1] == launch.image_manifest_digest
    expected_mounts = {
        launch.backend_lock_path: "/run/crux/backend-lock.json",
        launch.runtime_lock_path: "/run/crux/runtime-lock.json",
        launch.seal_evidence_path: "/run/crux/seal-evidence.json",
        launch.model_cache_path: "/model",
        launch.input_root: "/input",
    }
    for source, destination in expected_mounts.items():
        assert f"--mount=type=bind,src={source},dst={destination},readonly" in command


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uid", 0),
        ("uid", True),
        ("gid", -1),
        ("cpu_limit", "2,5"),
        ("cpu_limit", "0"),
        ("memory_bytes", 0),
        ("pid_limit", True),
        ("stdout_max_line_bytes", 0),
        ("stderr_read_chunk_bytes", -1),
    ],
)
def test_launch_profile_rejects_invalid_limits(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        profile(tmp_path, **{field: value})


def test_launch_profile_requires_exact_environment_and_safe_mounts(tmp_path: Path) -> None:
    environment = dict(REQUIRED_ENVIRONMENT)
    environment["EXTRA"] = "unsafe"
    with pytest.raises(ValueError, match="environment"):
        profile(tmp_path / "extra", environment=environment)

    launch = profile(tmp_path / "symlink-target")
    symlink = tmp_path / "model-link"
    symlink.symlink_to(launch.model_cache_path, target_is_directory=True)
    with pytest.raises(ValueError, match="mount"):
        profile(tmp_path / "symlink-profile", model_cache_path=symlink)


def test_launch_profile_is_frozen_and_rechecks_mount_identity(tmp_path: Path) -> None:
    launch = profile(tmp_path)
    with pytest.raises(FrozenInstanceError):
        launch.uid = 42  # type: ignore[misc]

    replaced = launch.backend_lock_path.with_suffix(".new")
    replaced.write_bytes(b"changed\n")
    os.replace(replaced, launch.backend_lock_path)
    with pytest.raises(ValueError, match="changed"):
        build_docker_command(launch)


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        (
            "github_hosted",
            {
                "api_record_sha256": "a" * 64,
                "approved_labels": ["Linux", "X64"],
                "job_id": 123,
                "run_url": "https://github.com/acme/crux/actions/runs/456/job/123",
                "runner_arch": "X64",
                "runner_os": "Linux",
                "workflow_commit": "b" * 40,
            },
        ),
        (
            "orchestrator_signed",
            {
                "attestation_sha256": "c" * 64,
                "physical_architecture": "linux/amd64",
                "signature": "ed25519:abc",
                "worker_id": "worker-01",
            },
        ),
        (
            "approved_local",
            {
                "approval_sha256": "d" * 64,
                "daemon_id": "docker-01",
                "host_architecture": "x86_64",
                "host_os": "Linux",
                "worker_id": "seal-host-01",
            },
        ),
    ],
)
def test_native_host_evidence_accepts_only_exact_official_forms(
    kind: str, payload: dict[str, object]
) -> None:
    evidence = NativeHostEvidence(
        kind=kind,  # type: ignore[arg-type]
        payload=payload,
        sha256=_canonical_sha256(payload),
        official_execution_allowed=True,
    )

    assert evidence.official_execution_allowed is True
    with pytest.raises(TypeError):
        evidence.payload["extra"] = "no"  # type: ignore[index]
    if kind == "github_hosted":
        with pytest.raises((TypeError, AttributeError)):
            evidence.payload["approved_labels"].append("ARM64")  # type: ignore[union-attr]


def test_native_evidence_rejects_hash_shape_and_policy_drift() -> None:
    payload = {
        "api_record_sha256": "a" * 64,
        "approved_labels": ["Linux", "X64"],
        "job_id": 123,
        "run_url": "https://github.com/acme/crux/actions/runs/456/job/123",
        "runner_arch": "X64",
        "runner_os": "Linux",
        "workflow_commit": "b" * 40,
    }
    for changed in [
        {**payload, "runner_arch": "ARM64"},
        {**payload, "extra": "field"},
    ]:
        with pytest.raises(ValueError):
            NativeHostEvidence(
                kind="github_hosted",
                payload=changed,
                sha256=_canonical_sha256(changed),
                official_execution_allowed=True,
            )
    with pytest.raises(ValueError, match="SHA-256"):
        NativeHostEvidence(
            kind="github_hosted",
            payload=payload,
            sha256="0" * 64,
            official_execution_allowed=True,
        )
    with pytest.raises(ValueError, match="official"):
        NativeHostEvidence(
            kind="github_hosted",
            payload=payload,
            sha256=_canonical_sha256(payload),
            official_execution_allowed=False,
        )


def test_bare_local_and_emulation_are_explicit_diagnostic_only() -> None:
    local = DiagnosticHostEvidence(
        kind="bare_local",
        payload={"host_architecture": "x86_64", "host_os": "Linux"},
        emulation_allowed=False,
        official_execution_allowed=False,
    )
    emulated = DiagnosticHostEvidence(
        kind="emulated",
        payload={"host_architecture": "arm64", "host_os": "Darwin"},
        emulation_allowed=True,
        official_execution_allowed=False,
    )
    assert not local.official_execution_allowed
    assert emulated.emulation_allowed
    with pytest.raises(ValueError, match="official"):
        DiagnosticHostEvidence(
            kind="emulated",
            payload={"host_architecture": "arm64", "host_os": "Darwin"},
            emulation_allowed=True,
            official_execution_allowed=True,
        )


def test_runner_starts_and_serves_one_correlated_request(tmp_path: Path) -> None:
    runner = RunnerProcess.start(profile(tmp_path), popen_factory=fake_popen("success"))
    try:
        assert runner.handshake["type"] == "ready"
        response = runner.request({"type": "ping", "request_id": "request-1"})
        assert response.payload == {"type": "pong"}
        assert response.request_id == "request-1"
    finally:
        runner.close()
        runner.close()
    assert not runner.stderr_thread_alive
    with runner._protocol_lock:  # pylint: disable=protected-access
        assert runner._protocol_failure is None  # pylint: disable=protected-access


def test_verify_backend_runtime_protocol_envelope_matches_runner_process(
    tmp_path: Path,
) -> None:
    pcm = b"\x00\x00" * 4
    audio_content = (
        b"RIFF"
        + (36 + len(pcm)).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (44100).to_bytes(4, "little")
        + (88200).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + len(pcm).to_bytes(4, "little")
        + pcm
    )
    input_root = profile(tmp_path / "profile").input_root
    (input_root / "audio.wav").write_bytes(audio_content)
    audio_sha256 = hashlib.sha256(audio_content).hexdigest()
    request_id = "task7-integration"
    descriptor_sha256 = "b" * 64
    runtime_stdin = io.BytesIO(
        canonical_json_bytes(
            {
                "audio_path": "audio.wav",
                "audio_sha256": audio_sha256,
                "backend_descriptor_sha256": descriptor_sha256,
                "request_id": request_id,
                "type": "transcribe",
            },
            trailing_newline=True,
        )
    )
    runtime_stdout = io.BytesIO()

    class RuntimeBackend:
        @staticmethod
        def transcribe(_verified: object) -> list[dict[str, object]]:
            return [{"frame_index": 0}]

    serve_requests(
        stdin=runtime_stdin,
        stdout=runtime_stdout,
        backend=RuntimeBackend(),
        input_root=input_root,
        descriptor_sha256=descriptor_sha256,
        max_input_audio_frames=4,
        stdout_max_line_bytes=2048,
    )
    response_line = runtime_stdout.getvalue()
    ready_line = canonical_json_bytes(
        {
            "protocol_schema": "crux.transcription-runner/v1",
            "type": "ready",
        },
        trailing_newline=True,
    )

    def runtime_output_popen(_command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        script = (
            "import sys;"
            "sys.stdout.buffer.write(bytes.fromhex(sys.argv[1]));"
            "sys.stdout.buffer.flush();"
            "sys.stdin.buffer.readline();"
            "sys.stdout.buffer.write(bytes.fromhex(sys.argv[2]));"
            "sys.stdout.buffer.flush();"
            "sys.stdin.buffer.read()"
        )
        return subprocess.Popen(  # pylint: disable=consider-using-with
            [
                sys.executable,
                "-c",
                script,
                ready_line.hex(),
                response_line.hex(),
            ],
            stdin=kwargs["stdin"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
            env={},
            shell=False,
            start_new_session=True,
        )

    runner = RunnerProcess.start(
        profile(tmp_path / "runner"),
        popen_factory=runtime_output_popen,
    )
    try:
        response = runner.request(
            {
                "audio_path": "audio.wav",
                "audio_sha256": audio_sha256,
                "backend_descriptor_sha256": descriptor_sha256,
                "request_id": request_id,
                "type": "transcribe",
            }
        )
    finally:
        runner.close()

    assert response.request_id == request_id
    assert set(response.payload) == {
        "audio_sha256",
        "backend_descriptor_sha256",
        "native_events",
        "type",
    }
    assert response.payload["audio_sha256"] == audio_sha256
    assert response.payload["backend_descriptor_sha256"] == descriptor_sha256
    assert response.payload["type"] == "transcription_result"
    assert [dict(event) for event in response.payload["native_events"]] == [  # type: ignore[union-attr]
        {"frame_index": 0}
    ]


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("startup_timeout", "backend_startup_timeout"),
        ("startup_death", "backend_process_died"),
        ("startup_malformed", "backend_protocol_invalid"),
        ("startup_stray", "backend_protocol_invalid"),
        ("startup_eof", "backend_protocol_invalid"),
        ("startup_oversized", "backend_protocol_oversized"),
        ("startup_flood", "backend_unexpected_response"),
    ],
)
def test_startup_failures_are_fatal_and_reaped(tmp_path: Path, mode: str, code: str) -> None:
    launch = profile(tmp_path, stdout_max_line_bytes=128)
    with pytest.raises(BackendFatalFailure) as raised:
        RunnerProcess.start(launch, popen_factory=fake_popen(mode))
    assert raised.value.error.code == code


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("request_timeout", "backend_request_timeout"),
        ("request_death", "backend_process_died"),
        ("request_malformed", "backend_protocol_invalid"),
        ("request_wrong_id", "backend_request_id_mismatch"),
        ("request_eof", "backend_protocol_invalid"),
        ("request_oversized", "backend_protocol_oversized"),
        ("request_extra", "backend_unexpected_response"),
    ],
)
def test_request_failures_are_fatal_and_invalidate_runner(
    tmp_path: Path, mode: str, code: str
) -> None:
    runner = RunnerProcess.start(
        profile(tmp_path, stdout_max_line_bytes=128), popen_factory=fake_popen(mode)
    )
    with pytest.raises(BackendFatalFailure) as raised:
        runner.request({"type": "ping", "request_id": "request-1"})
    assert raised.value.error.code == code
    assert runner.closed


def test_same_batch_duplicate_fails_current_operation(tmp_path: Path) -> None:
    runner = RunnerProcess.start(
        profile(tmp_path, stdout_max_line_bytes=128),
        popen_factory=fake_popen("request_duplicate"),
    )

    with pytest.raises(BackendFatalFailure) as raised:
        runner.request({"type": "ping", "request_id": "request-1"})

    assert raised.value.error.code == "backend_unexpected_response"
    assert runner.closed


def test_concurrent_request_is_rejected_without_killing_runner(tmp_path: Path) -> None:
    runner = RunnerProcess.start(profile(tmp_path), popen_factory=fake_popen("hold"))
    result: list[object] = []
    first = threading.Thread(
        target=lambda: result.append(runner.request({"type": "ping", "request_id": "request-1"}))
    )
    first.start()
    time.sleep(0.05)
    with pytest.raises(BackendFatalFailure) as raised:
        runner.request({"type": "ping", "request_id": "request-2"})
    assert raised.value.error.code == "backend_request_in_flight"
    assert not runner.closed
    first.join(timeout=2)
    runner.close()
    assert len(result) == 1


def test_stderr_over_pipe_and_ring_capacity_does_not_deadlock(tmp_path: Path) -> None:
    runner = RunnerProcess.start(
        profile(tmp_path, stderr_ring_buffer_bytes=4096),
        popen_factory=fake_popen("stderr_stress"),
    )
    started = time.monotonic()
    response = runner.request({"type": "ping", "request_id": "request-1"}, deadline_seconds=2)
    elapsed = time.monotonic() - started
    runner.close()

    assert response.payload == {"type": "pong"}
    assert elapsed < 2
    assert runner.stderr_total_raw_bytes > 1024 * 1024
    assert runner.stderr_truncated is True
    assert runner.stderr_retained_bytes <= 4096


def test_stderr_sanitizes_sensitive_content_across_chunks(tmp_path: Path) -> None:
    runner = RunnerProcess.start(
        profile(tmp_path, environment={**REQUIRED_ENVIRONMENT, "PYTHONHASHSEED": "0"}),
        popen_factory=fake_popen("stderr_sensitive"),
    )
    runner.request({"type": "ping", "request_id": "request-1"})
    runner.close()
    retained = runner.stderr_text

    for secret in [
        "super-secret",
        "/private/secret",
        "user:pass",
        "?token=x",
        "0.123",
        "Traceback",
    ]:
        assert secret not in retained
    assert "code=restore_ok tensor=onsets/count count=78 duration_ms=42" in retained
    assert "[REDACTED]" in retained


def test_stderr_structurally_redacts_unknown_absolute_paths_across_chunks(
    tmp_path: Path,
) -> None:
    runner = RunnerProcess.start(profile(tmp_path), popen_factory=fake_popen("stderr_paths"))
    runner.request({"type": "ping", "request_id": "request-1"})
    runner.close()
    retained = runner.stderr_text

    for leaked in [
        "/unknown/private/model.ckpt",
        r"C:\Users\alice\model.ckpt",
        r"\\server\share\model.ckpt",
        "file:///unknown/private/model.ckpt",
        "https://example.test/private/model.ckpt",
    ]:
        assert leaked not in retained
    assert "tensor=onsets/conv/kernel:0" in retained


def test_oversized_unterminated_stderr_is_bounded_and_sanitized(tmp_path: Path) -> None:
    runner = RunnerProcess.start(
        profile(tmp_path, stderr_max_line_bytes=64, stderr_ring_buffer_bytes=512),
        popen_factory=fake_popen("stderr_unterminated"),
    )
    runner.request({"type": "ping", "request_id": "request-1"})
    runner.close()
    assert runner.stderr_retained_bytes <= 512
    assert runner.stderr_truncated
    assert "secret=" not in runner.stderr_text


def test_close_forces_kill_and_is_idempotent(tmp_path: Path) -> None:
    runner = RunnerProcess.start(profile(tmp_path), popen_factory=fake_popen("ignore_term"))
    runner.request({"type": "ping", "request_id": "request-1"})
    runner.close()
    runner.close()
    assert runner.closed
    assert runner.process_returncode is not None
    assert not runner.stderr_thread_alive


@pytest.mark.parametrize("mode", ["delayed_startup_stray", "delayed_request_duplicate"])
def test_delayed_stdout_while_idle_atomically_invalidates_runner(tmp_path: Path, mode: str) -> None:
    runner = RunnerProcess.start(profile(tmp_path), popen_factory=fake_popen(mode))
    if mode == "delayed_request_duplicate":
        response = runner.request({"type": "ping", "request_id": "request-1"})
        assert response.payload == {"type": "pong"}

    deadline = time.monotonic() + 2
    while not runner.closed and time.monotonic() < deadline:
        time.sleep(0.01)

    assert runner.closed
    assert runner.process_returncode is not None
    with pytest.raises(BackendFatalFailure) as raised:
        runner.request({"type": "ping", "request_id": "request-2"})
    assert raised.value.error.code == "backend_unexpected_response"


def test_delayed_duplicate_racing_close_retains_protocol_failure(tmp_path: Path) -> None:
    runner = RunnerProcess.start(
        profile(tmp_path),
        popen_factory=fake_popen("delayed_request_duplicate"),
    )
    response = runner.request({"type": "ping", "request_id": "request-1"})
    assert response.payload == {"type": "pong"}

    runner.close()

    assert runner.closed
    assert runner.process_returncode is not None
    with runner._protocol_lock:  # pylint: disable=protected-access
        failure = runner._protocol_failure  # pylint: disable=protected-access
    assert failure is not None
    assert failure.error.code == "backend_unexpected_response"
    with pytest.raises(BackendFatalFailure) as raised:
        runner.request({"type": "ping", "request_id": "request-2"})
    assert raised.value.error.code == "backend_unexpected_response"


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("delayed_oversized", "backend_protocol_oversized"),
        ("delayed_partial_eof", "backend_protocol_invalid"),
        ("delayed_clean_eof", "backend_process_died"),
    ],
)
def test_delayed_terminal_stdout_event_atomically_invalidates_runner(
    tmp_path: Path, mode: str, code: str
) -> None:
    runner = RunnerProcess.start(
        profile(tmp_path, stdout_max_line_bytes=128),
        popen_factory=fake_popen(mode),
    )

    deadline = time.monotonic() + 2
    while not runner.closed and time.monotonic() < deadline:
        time.sleep(0.01)

    assert runner.closed
    assert runner.process_returncode is not None
    with pytest.raises(BackendFatalFailure) as raised:
        runner.request({"type": "ping", "request_id": "request-2"})
    assert raised.value.error.code == code


def test_injected_stdout_io_error_is_stable_and_reaps_child(tmp_path: Path) -> None:
    descriptors: dict[str, int] = {}
    base_factory = fake_popen("startup_timeout")

    def factory(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        process = base_factory(command, **kwargs)
        assert process.stdout is not None
        descriptors["stdout"] = process.stdout.fileno()
        return process

    def injected_read(descriptor: int, size: int) -> bytes:
        if descriptor == descriptors["stdout"]:
            raise OSError("injected stdout read failure")
        return os.read(descriptor, size)

    with pytest.raises(BackendFatalFailure) as raised:
        RunnerProcess.start(
            profile(tmp_path),
            popen_factory=factory,
            read_function=injected_read,
        )
    assert raised.value.error.code == "backend_protocol_io_error"


def test_injected_stdout_io_error_wakes_expected_response_waiter(tmp_path: Path) -> None:
    descriptors: dict[str, int] = {}
    stdout_reads = [0]
    second_read_entered = threading.Event()
    release = threading.Event()
    base_factory = fake_popen("request_timeout")

    def factory(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        process = base_factory(command, **kwargs)
        assert process.stdout is not None
        descriptors["stdout"] = process.stdout.fileno()
        return process

    def injected_read(descriptor: int, size: int) -> bytes:
        if descriptor == descriptors["stdout"]:
            stdout_reads[0] += 1
            if stdout_reads[0] > 1:
                second_read_entered.set()
                release.wait(timeout=3)
                raise OSError("injected stdout read failure")
        return os.read(descriptor, size)

    runner = RunnerProcess.start(
        profile(tmp_path, request_deadline_seconds=2),
        popen_factory=factory,
        read_function=injected_read,
    )
    assert second_read_entered.wait(timeout=1)
    failures: list[BackendFatalFailure] = []

    def request() -> None:
        try:
            runner.request({"type": "ping", "request_id": "request-1"})
        except BackendFatalFailure as failure:
            failures.append(failure)

    requester = threading.Thread(target=request)
    requester.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with runner._protocol_lock:  # pylint: disable=protected-access
            if runner._protocol_state == "request_wait":  # pylint: disable=protected-access
                break
        time.sleep(0.01)
    else:
        pytest.fail("request did not enter response-wait state")
    release.set()
    requester.join(timeout=2)

    assert not requester.is_alive()
    assert len(failures) == 1
    assert failures[0].error.code == "backend_protocol_io_error"
    assert runner.closed


def test_operational_stderr_reader_error_is_stable_and_reaps_child(tmp_path: Path) -> None:
    descriptors: dict[str, int] = {}
    base_factory = fake_popen("startup_timeout")

    def factory(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        process = base_factory(command, **kwargs)
        assert process.stderr is not None
        descriptors["stderr"] = process.stderr.fileno()
        return process

    def injected_read(descriptor: int, size: int) -> bytes:
        if descriptor == descriptors["stderr"]:
            raise RuntimeError("injected stderr read failure")
        return os.read(descriptor, size)

    with pytest.raises(BackendFatalFailure) as raised:
        RunnerProcess.start(
            profile(tmp_path),
            popen_factory=factory,
            read_function=injected_read,
        )
    assert raised.value.error.code == "backend_stderr_io_error"


def test_busy_close_suppresses_reader_traceback_and_path_leakage(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    descriptors: dict[str, int] = {}
    entered = threading.Event()
    release = threading.Event()
    captured: list[subprocess.Popen[bytes]] = []
    base_factory = capturing_fake_popen("busy_stderr", captured)

    def factory(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        process = base_factory(command, **kwargs)
        assert process.stderr is not None
        descriptors["stderr"] = process.stderr.fileno()
        return process

    def injected_read(descriptor: int, size: int) -> bytes:
        if descriptor == descriptors["stderr"]:
            entered.set()
            release.wait(timeout=3)
            raise ValueError("I/O operation on closed file")
        return os.read(descriptor, size)

    runner = RunnerProcess.start(
        profile(tmp_path),
        popen_factory=factory,
        read_function=injected_read,
    )
    assert entered.wait(timeout=1)
    closer = threading.Thread(target=runner.close)
    closer.start()
    assert captured[0].stderr is not None
    deadline = time.monotonic() + 2
    while not captured[0].stderr.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert captured[0].stderr.closed
    release.set()
    closer.join(timeout=3)

    assert not closer.is_alive()
    assert runner.closed
    assert not runner.stderr_thread_alive
    captured_stderr = capfd.readouterr().err
    assert "Exception in thread" not in captured_stderr
    assert "Traceback" not in captured_stderr
    assert str(Path.cwd()) not in captured_stderr


class _StartFailingThread:
    def __init__(
        self,
        inner: threading.Thread,
        *,
        starts: list[int],
        fail_at: int,
    ) -> None:
        self._inner = inner
        self._starts = starts
        self._fail_at = fail_at

    def start(self) -> None:
        self._starts[0] += 1
        if self._starts[0] == self._fail_at:
            raise RuntimeError("injected thread start failure")
        self._inner.start()

    def join(self, timeout: float | None = None) -> None:
        self._inner.join(timeout)

    def is_alive(self) -> bool:
        return self._inner.is_alive()


@pytest.mark.parametrize("fail_at", [1, 2])
def test_reader_thread_start_failure_is_typed_and_reaps_child(tmp_path: Path, fail_at: int) -> None:
    captured: list[subprocess.Popen[bytes]] = []
    starts = [0]

    def thread_factory(**kwargs: Any) -> _StartFailingThread:
        return _StartFailingThread(
            threading.Thread(**kwargs),
            starts=starts,
            fail_at=fail_at,
        )

    with pytest.raises(BackendFatalFailure) as raised:
        RunnerProcess.start(
            profile(tmp_path),
            popen_factory=capturing_fake_popen("startup_timeout", captured),
            thread_factory=thread_factory,
        )

    assert raised.value.error.code == "backend_reader_start_failed"
    assert len(captured) == 1
    assert captured[0].poll() is not None
    for stream in (captured[0].stdin, captured[0].stdout, captured[0].stderr):
        assert stream is not None and stream.closed
