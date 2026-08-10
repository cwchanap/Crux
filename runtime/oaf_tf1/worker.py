"""Small synchronous line-JSON worker for the extracted OaF model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, TextIO

try:
    from .model import OafModel, OafModelConfig, OafModelError, OafNativeEvent, load_model_config
except ImportError:  # pragma: no cover - direct execution in the runtime image
    from model import OafModel, OafModelConfig, OafModelError, OafNativeEvent, load_model_config

DEFAULT_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"


def _event_payload(event: Any) -> dict[str, Any]:
    if isinstance(event, OafNativeEvent):
        return {
            "time_sec": event.time_sec,
            "native_class_id": event.native_class_id,
            "model_output_bin": event.model_output_bin,
            "native_midi_note": event.native_midi_note,
            "upstream_8hit_group_id": event.upstream_8hit_group_id,
            "confidence": event.confidence,
            "velocity_midi": event.velocity_midi,
        }
    if isinstance(event, dict):
        return dict(event)
    raise OafModelError("model returned an invalid native event")


def _write(output: TextIO, payload: dict[str, Any]) -> None:
    output.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    output.flush()


def _valid_request(payload: object) -> tuple[str, str]:
    if not isinstance(payload, dict) or set(payload) != {"id", "audio_path"}:
        raise ValueError("request must contain id and audio_path")
    request_id = payload["id"]
    audio_path = payload["audio_path"]
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request id is invalid")
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError("audio path is invalid")
    path = Path(audio_path)
    if path.is_absolute() or "\\" in audio_path:
        raise ValueError("audio path is invalid")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("audio path is invalid")
    return request_id, audio_path


def _request_id(payload: object) -> str | None:
    """Return a usable request ID before validating the rest of a request."""
    if not isinstance(payload, dict):
        return None
    request_id = payload.get("id")
    if isinstance(request_id, str) and request_id:
        return request_id
    return None


def serve_requests(
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    *,
    checkpoint_dir: Path | None = None,
    config: OafModelConfig | None = None,
    model_factory: Callable[..., OafModel] = OafModel.load,
) -> int:
    """Load one model and process newline-delimited requests until EOF."""
    checkpoint_dir = Path(checkpoint_dir or os.environ.get("OAF_CHECKPOINT_DIR", "/model"))
    loaded_config = config or load_model_config()
    model = model_factory(checkpoint_dir, loaded_config)
    _write(
        stdout,
        {
            "type": "ready",
            "backend_id": loaded_config.backend_id or DEFAULT_BACKEND_ID,
            "restored_tensor_count": model.restored_tensor_count,
        },
    )
    for raw_line in stdin:
        request_id = None
        try:
            payload = json.loads(raw_line)
            request_id = _request_id(payload)
            request_id, audio_path = _valid_request(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            error = {"error": {"code": "invalid_request", "message": "invalid request"}}
            if request_id is not None:
                error["id"] = request_id
            _write(
                stdout,
                error,
            )
            continue
        try:
            events = model.transcribe(Path(audio_path))
            _write(
                stdout,
                {"id": request_id, "events": [_event_payload(event) for event in events]},
            )
        except OafModelError:
            _write(
                stdout,
                {
                    "id": request_id,
                    "error": {"code": "inference_failed", "message": "inference failed"},
                },
            )
        except Exception:
            _write(
                stdout,
                {
                    "id": request_id,
                    "error": {"code": "inference_failed", "message": "inference failed"},
                },
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the extracted OaF model worker")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/model"))
    args = parser.parse_args(argv)
    return serve_requests(checkpoint_dir=args.checkpoint_dir)


run_worker = serve_requests


if __name__ == "__main__":
    raise SystemExit(main())
