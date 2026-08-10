from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.benchmark.worker_process import WorkerProcess, WorkerProcessError


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
        ({"type": "ready"}, "worker ready response is invalid"),
    ],
)
def test_worker_process_rejects_malformed_ready(
    tmp_path: Path, script_output: dict[str, object], message: str
) -> None:
    script = _script(tmp_path, [script_output])
    with pytest.raises(WorkerProcessError, match=message):
        WorkerProcess.start([sys.executable, str(script)], timeout_seconds=1)


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
    finally:
        process.close()
