from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import src.benchmark.worker_process as worker_process
from src.benchmark.worker_process import WorkerProcess, WorkerProcessError


class _FakeStream:
    def close(self) -> None:
        pass


class _FakePopen:
    def __init__(self) -> None:
        self.stdin = _FakeStream()
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()
        self.wait_calls: list[float | None] = []
        self.killed = False
        self.reap_after_kill = True

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)
        if timeout is not None and not (self.killed and self.reap_after_kill):
            raise subprocess.TimeoutExpired("fake-worker", timeout)

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        self.killed = True


class _RecordingThread:
    instances: list[_RecordingThread] = []

    def __init__(self, **_kwargs: object) -> None:
        self.join_calls: list[float | None] = []
        self.instances.append(self)

    def start(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


def test_worker_process_close_uses_independent_close_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingThread.instances.clear()
    monkeypatch.setattr(worker_process.threading, "Thread", _RecordingThread)
    monkeypatch.setattr(worker_process.time, "monotonic", lambda: 0.0)
    process = _FakePopen()
    worker = WorkerProcess(
        process,
        timeout_seconds=3600.0,
        close_timeout_seconds=30.0,
        ready={"type": "ready"},
    )

    worker.close()

    assert process.wait_calls == [30.0, 30.0, 30.0]
    assert _RecordingThread.instances[0].join_calls == [30.0]
    assert 3600.0 not in process.wait_calls
    assert 3600.0 not in _RecordingThread.instances[0].join_calls


def test_worker_process_close_deadline_shrinks_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The close timeout is one monotonic deadline, not a per-phase timeout."""
    _RecordingThread.instances.clear()
    monkeypatch.setattr(worker_process.threading, "Thread", _RecordingThread)
    clock_values = iter([0.0, 10.0, 20.0, 30.0, 40.0])
    monkeypatch.setattr(worker_process.time, "monotonic", lambda: next(clock_values))
    process = _FakePopen()
    worker = WorkerProcess(
        process,
        timeout_seconds=3600.0,
        close_timeout_seconds=30.0,
        ready={"type": "ready"},
    )

    worker.close()

    # deadline = 0.0 + 30.0 = 30.0; each phase gets only the remaining budget.
    assert process.wait_calls == [20.0, 10.0, 0.1]
    assert _RecordingThread.instances[0].join_calls == [0.1]


def test_worker_process_close_raises_when_killed_process_never_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A killed process that cannot be reaped raises instead of hanging close()."""
    _RecordingThread.instances.clear()
    monkeypatch.setattr(worker_process.threading, "Thread", _RecordingThread)
    monkeypatch.setattr(worker_process.time, "monotonic", lambda: 0.0)
    process = _FakePopen()
    process.reap_after_kill = False
    worker = WorkerProcess(
        process,
        timeout_seconds=3600.0,
        close_timeout_seconds=30.0,
        ready={"type": "ready"},
    )

    with pytest.raises(WorkerProcessError, match="worker close timed out"):
        worker.close()

    assert process.wait_calls == [30.0, 30.0, 30.0]


@pytest.mark.parametrize(
    ("timeout_seconds", "close_timeout_seconds", "message"),
    [
        (0.0, 30.0, "timeout_seconds must be positive"),
        (30.0, 0.0, "close_timeout_seconds must be positive"),
    ],
)
def test_worker_process_rejects_non_positive_timeouts(
    timeout_seconds: float,
    close_timeout_seconds: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        WorkerProcess(
            _FakePopen(),
            timeout_seconds=timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            ready={"type": "ready"},
        )


@pytest.mark.parametrize(
    ("timeout_seconds", "close_timeout_seconds", "message"),
    [
        (0.0, 30.0, "timeout_seconds must be positive"),
        (30.0, 0.0, "close_timeout_seconds must be positive"),
    ],
)
def test_worker_process_start_rejects_non_positive_timeouts(
    timeout_seconds: float,
    close_timeout_seconds: float,
    message: str,
) -> None:
    """WorkerProcess.start validates both timeouts before spawning a subprocess."""
    with pytest.raises(ValueError, match=message):
        WorkerProcess.start(
            [sys.executable, "-c", "pass"],
            timeout_seconds=timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
        )


def _script(
    tmp_path: Path, output: list[dict[str, object]], *, exit_after_ready: bool = False
) -> Path:
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import json, sys\n"
        f"ready = {output[0]!r}\n"
        "print(json.dumps(ready), flush=True)\n"
        + ("raise SystemExit(0)\n" if exit_after_ready else "")
        + "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        + "    response = dict(request.get('response', {'events': []}))\n"
        + "    response.setdefault('id', request['id'])\n"
        + "    print(json.dumps(response), flush=True)\n"
    )
    return script


def test_worker_process_reuses_child_for_two_requests_and_close(tmp_path: Path) -> None:
    script = _script(
        tmp_path,
        [{"type": "ready", "backend_id": "test", "restored_tensor_count": 78}],
    )
    process = WorkerProcess.start([sys.executable, str(script)], timeout_seconds=1)
    try:
        assert process.ready["type"] == "ready"
        assert process.request("one.wav")["events"] == []
        assert process.request("two.wav")["events"] == []
    finally:
        process.close()
    assert process.returncode is not None


@pytest.mark.parametrize(
    ("script_output", "message"),
    [
        ({}, "worker ready response is invalid"),
    ],
)
def test_worker_process_rejects_malformed_ready(
    tmp_path: Path, script_output: dict[str, object], message: str
) -> None:
    script = _script(tmp_path, [script_output])
    with pytest.raises(WorkerProcessError, match=message):
        WorkerProcess.start([sys.executable, str(script)], timeout_seconds=1)


def test_worker_process_surfaces_bounded_stderr_when_exiting_before_ready(
    tmp_path: Path,
) -> None:
    script = tmp_path / "failing_worker.py"
    script.write_text(
        "import sys\n"
        "sys.stderr.write('checkpoint load failed\\n' + ('x' * 10000))\n"
        "sys.stderr.flush()\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkerProcessError) as caught:
        WorkerProcess.start([sys.executable, str(script)], timeout_seconds=1)

    message = str(caught.value)
    assert message.startswith("worker exited before ready: checkpoint load failed")
    assert len(message) < 5_000


def test_worker_process_accepts_generic_ready_record(tmp_path: Path) -> None:
    script = _script(tmp_path, [{"type": "ready"}])
    process = WorkerProcess.start([sys.executable, str(script)], timeout_seconds=1)
    try:
        assert process.ready == {"type": "ready"}
        assert process.request("one.wav")["events"] == []
    finally:
        process.close()


def test_worker_process_reports_early_exit(tmp_path: Path) -> None:
    script = _script(
        tmp_path,
        [{"type": "ready", "backend_id": "test", "restored_tensor_count": 78}],
        exit_after_ready=True,
    )
    process = WorkerProcess.start([sys.executable, str(script)], timeout_seconds=1)
    with pytest.raises(WorkerProcessError, match="worker exited before ready"):
        process.request("one.wav")
    process.close()


def test_worker_process_reports_wrong_response_id(tmp_path: Path) -> None:
    script = tmp_path / "wrong_worker.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'type':'ready','backend_id':'test','restored_tensor_count':78}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    print(json.dumps({'id':'wrong','events':[]}), flush=True)\n"
    )
    process = WorkerProcess.start([sys.executable, str(script)], timeout_seconds=1)
    try:
        with pytest.raises(WorkerProcessError, match="worker response id mismatch"):
            process.request("one.wav")
        assert process.closed
        with pytest.raises(WorkerProcessError, match="worker process is closed"):
            process.request("two.wav")
    finally:
        process.close()


def test_worker_process_poisoned_after_timeout(tmp_path: Path) -> None:
    script = tmp_path / "slow_worker.py"
    script.write_text(
        "import json, sys, time\n"
        "print(json.dumps({'type':'ready'}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    time.sleep(0.2)\n"
        "    print(json.dumps({'id':request['id'],'events':[]}), flush=True)\n"
    )
    process = WorkerProcess.start([sys.executable, str(script)], timeout_seconds=0.05)
    try:
        with pytest.raises(WorkerProcessError, match="worker response timed out"):
            process.request("one.wav")
        assert process.closed
        with pytest.raises(WorkerProcessError, match="worker process is closed"):
            process.request("two.wav")
    finally:
        process.close()


def test_worker_process_poisoned_after_malformed_output(tmp_path: Path) -> None:
    script = tmp_path / "malformed_worker.py"
    script.write_text(
        "import sys\n"
        'print(\'{"type":"ready"}\', flush=True)\n'
        "for _line in sys.stdin:\n"
        "    print('not-json', flush=True)\n"
    )
    process = WorkerProcess.start([sys.executable, str(script)], timeout_seconds=1)
    try:
        with pytest.raises(WorkerProcessError, match="worker response is invalid"):
            process.request("one.wav")
        assert process.closed
        with pytest.raises(WorkerProcessError, match="worker process is closed"):
            process.request("two.wav")
    finally:
        process.close()


def test_worker_process_request_attaches_valid_audio_identity(tmp_path: Path) -> None:
    script = tmp_path / "echo_identity_worker.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'type':'ready'}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    response = {'id': request['id'], 'events': [], "
        "'audio_byte_length': request.get('audio_byte_length'), "
        "'audio_sha256': request.get('audio_sha256')}\n"
        "    print(json.dumps(response), flush=True)\n",
        encoding="utf-8",
    )
    process = WorkerProcess.start([sys.executable, str(script)], timeout_seconds=1)
    try:
        response = process.request(
            "one.wav",
            audio_byte_length=4096,
            audio_sha256="a" * 64,
        )
        assert response["audio_byte_length"] == 4096
        assert response["audio_sha256"] == "a" * 64
    finally:
        process.close()


def test_worker_process_request_rejects_invalid_audio_identity(tmp_path: Path) -> None:
    """Invalid audio identity is rejected host-side without poisoning the child."""
    script = _script(tmp_path, [{"type": "ready"}])
    process = WorkerProcess.start([sys.executable, str(script)], timeout_seconds=1)
    try:
        for audio_byte_length, audio_sha256 in [
            (-1, "a" * 64),
            (4096, "short"),
            (4096, "g" * 64),
            ("4096", "a" * 64),
        ]:
            with pytest.raises(WorkerProcessError, match="worker audio identity is invalid"):
                process.request(
                    "one.wav",
                    audio_byte_length=audio_byte_length,  # type: ignore[arg-type]
                    audio_sha256=audio_sha256,
                )
            assert not process.closed
    finally:
        process.close()


def test_worker_process_preserves_correlated_error_without_poisoning_child(
    tmp_path: Path,
) -> None:
    script = tmp_path / "error_then_success_worker.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'type':'ready'}), flush=True)\n"
        "first = True\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if first:\n"
        "        first = False\n"
        "        print(json.dumps({'id':request['id'],'error':{'code':'audio_unavailable','message':'audio unavailable'}}), flush=True)\n"
        "    else:\n"
        "        print(json.dumps({'id':request['id'],'events':[]}), flush=True)\n",
        encoding="utf-8",
    )
    process = WorkerProcess.start([sys.executable, str(script)], timeout_seconds=1)
    try:
        response = process.request("one.wav")
        assert response["error"] == {
            "code": "audio_unavailable",
            "message": "audio unavailable",
        }
        assert not process.closed
        assert process.request("two.wav")["events"] == []
    finally:
        process.close()
