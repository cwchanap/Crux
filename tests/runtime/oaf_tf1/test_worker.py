from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from runtime.oaf_tf1.model import OafModelError, OafNativeEvent
from runtime.oaf_tf1.worker import serve_requests


def _requests() -> str:
    return "".join(
        json.dumps({"id": request_id, "audio_path": audio_path}) + "\n"
        for request_id, audio_path in (("request-1", "one.wav"), ("request-2", "two.wav"))
    )


def test_worker_loads_once_and_serves_sequential_requests(tmp_path: Path) -> None:
    seen: list[str] = []
    factories: list[tuple[Path, object]] = []

    class FakeModel:
        restored_tensor_count = 78

        def transcribe(self, audio_path: Path) -> tuple[OafNativeEvent, ...]:
            seen.append(audio_path.name)
            return ()

    def factory(checkpoint_dir: Path, config=None) -> FakeModel:
        factories.append((checkpoint_dir, config))
        return FakeModel()

    output = io.StringIO()
    assert (
        serve_requests(
            io.StringIO(_requests()),
            output,
            checkpoint_dir=tmp_path,
            model_factory=factory,
        )
        == 0
    )

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[0] == {
        "type": "ready",
        "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
        "restored_tensor_count": 78,
    }
    assert [record["id"] for record in records[1:]] == ["request-1", "request-2"]
    assert len(factories) == 1
    assert seen == ["one.wav", "two.wav"]


def test_worker_reports_invalid_request_and_inference_failure(tmp_path: Path) -> None:
    class FakeModel:
        restored_tensor_count = 78

        def transcribe(self, _audio_path: Path) -> tuple[OafNativeEvent, ...]:
            raise OafModelError("inference failed")

    output = io.StringIO()
    serve_requests(
        io.StringIO("not-json\n{" + '"id":"request-1","audio_path":"song.wav"}' + "\n"),
        output,
        checkpoint_dir=tmp_path,
        model_factory=lambda *_args, **_kwargs: FakeModel(),
    )

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[1]["error"]["code"] == "invalid_request"
    assert records[2]["error"]["code"] == "inference_failed"


@pytest.mark.parametrize("audio_path", ["/tmp/song.wav", "songs/../song.wav"])
def test_worker_rejects_unsafe_audio_paths(tmp_path: Path, audio_path: str) -> None:
    class FakeModel:
        restored_tensor_count = 78

    output = io.StringIO()
    serve_requests(
        io.StringIO(json.dumps({"id": "request-1", "audio_path": audio_path}) + "\n"),
        output,
        checkpoint_dir=tmp_path,
        model_factory=lambda *_args, **_kwargs: FakeModel(),
    )
    record = json.loads(output.getvalue().splitlines()[1])
    assert record["error"]["code"] == "invalid_request"
