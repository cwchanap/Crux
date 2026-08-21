"""Standalone persistent worker for the frozen Inverse Drum Machine model."""

# The isolated runtime owns these dependencies; keep them lazy for host-side imports.
# pylint: disable=import-error,import-outside-toplevel

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, TextIO

BACKEND_ID = "idm-44-train-kits-v1"
MODEL_ID = "idm-44-train-kits-456656868538-5856a9bee7c6"
TRAIN_CLASSES = (
    "CY_CR",
    "CY_RD",
    "HH_CHH",
    "HH_OHH",
    "KD",
    "SD",
    "TT_HFT",
    "TT_HMT",
    "TT_LMT",
)
SAMPLE_RATE_HZ = 44100
ACTIVATION_RATE_HZ = 44100 / 256
MODEL_CONFIG_RELATIVE_PATH = Path("pretrained/idm-44-train-kits/checkpoints/model.yaml")
CHECKPOINT_RELATIVE_PATH = Path(
    "pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt"
)
_WHEEL_PROJECT: Any | None = None


def _write(output: TextIO, payload: dict[str, Any]) -> None:
    output.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    output.flush()


def _request_id(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    request_id = payload.get("id")
    return request_id if isinstance(request_id, str) and request_id else None


def _valid_request(payload: object) -> tuple[str, str]:
    if not isinstance(payload, dict) or set(payload) != {"id", "audio_path"}:
        raise ValueError("request is invalid")
    request_id = payload["id"]
    audio_path = payload["audio_path"]
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request id is invalid")
    if not isinstance(audio_path, str) or not audio_path or "\x00" in audio_path:
        raise ValueError("audio path is invalid")
    path = Path(audio_path)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("audio path is invalid")
    return request_id, audio_path


def _load_model(
    model_root: Path,
    *,
    wheel_path: Path | None = None,
    wheel_sha256: str | None = None,
) -> tuple[Any, Any, Any]:
    import torch

    global _WHEEL_PROJECT
    if wheel_path is None or wheel_sha256 is None or wheel_path.is_symlink():
        raise ValueError("attested IDM wheel is required")
    wheel_bytes = wheel_path.read_bytes()
    if hashlib.sha256(wheel_bytes).hexdigest() != wheel_sha256:
        raise ValueError("attested IDM wheel digest does not match")

    if _WHEEL_PROJECT is not None:
        _WHEEL_PROJECT.cleanup()
    _WHEEL_PROJECT = tempfile.TemporaryDirectory(prefix="crux-idm-wheel-")
    project_root = Path(_WHEEL_PROJECT.name)
    with zipfile.ZipFile(_bytes_as_file(wheel_bytes)) as wheel:
        wheel.extractall(project_root)
    (project_root / ".project-root").touch()
    model_data_root = project_root / MODEL_CONFIG_RELATIVE_PATH.parent
    model_data_root.mkdir(parents=True)
    for relative_path in (MODEL_CONFIG_RELATIVE_PATH, CHECKPOINT_RELATIVE_PATH):
        (model_data_root / relative_path.name).symlink_to(
            (model_root / relative_path).resolve(strict=True)
        )
    os.chdir(project_root)
    sys.path.insert(0, os.fspath(project_root))
    from idm.inference import load_model

    with contextlib.redirect_stdout(sys.stderr):
        model, _ = load_model("idm-44-train-kits", torch.device("cpu"), log_dir=Path("pretrained"))
    model.eval()
    return model, torch, torch.device("cpu")


def _load_audio(path: str, torch: Any, device: Any) -> Any:
    import soundfile

    try:
        info = soundfile.info(path)
        if (
            info.format != "WAV"
            or info.subtype != "PCM_16"
            or info.samplerate != SAMPLE_RATE_HZ
            or info.channels != 1
        ):
            raise ValueError("audio format is invalid")
        samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    except Exception as error:
        raise ValueError("audio format is invalid") from error
    if sample_rate != SAMPLE_RATE_HZ or samples.shape[1] != 1:
        raise ValueError("audio format is invalid")
    return torch.from_numpy(samples[:, 0]).to(device).unsqueeze(0)


def _events(model: Any, audio: Any, torch: Any) -> list[dict[str, Any]]:
    with torch.inference_mode():
        encoder_outputs = model.encoder(audio)
        activations = encoder_outputs["activations"]
        onset_logits = activations["onset"]
        onset_scores = model.decoder.activation(onset_logits)
        picked = model.decoder.peak_picking_val(
            onset_scores,
            activation_rate=encoder_outputs["activation_rate"],
        )
        velocity_values = activations["velocity"]

    activation_rate = float(encoder_outputs["activation_rate"])
    if not math.isfinite(activation_rate) or activation_rate <= 0:
        raise RuntimeError("activation rate is invalid")
    selected = torch.nonzero(picked[0] > 0, as_tuple=False)
    events: list[dict[str, Any]] = []
    for class_index_tensor, frame_index_tensor in selected.tolist():
        class_index = int(class_index_tensor)
        frame_index = int(frame_index_tensor)
        onset_score = float(onset_scores[0, class_index, frame_index].item())
        native_velocity = float(velocity_values[0, class_index, frame_index].item())
        if not math.isfinite(onset_score) or not math.isfinite(native_velocity):
            raise RuntimeError("model emitted a nonfinite event")
        events.append(
            {
                "class_index": class_index,
                "native_class_id": TRAIN_CLASSES[class_index],
                "frame_index": frame_index,
                "time_sec": frame_index / activation_rate,
                "onset_score": onset_score,
                "native_velocity": native_velocity,
            }
        )
    return events


def serve_requests(
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    *,
    model_root: Path,
    wheel_path: Path | None = None,
    wheel_sha256: str | None = None,
    model_loader: Callable[..., tuple[Any, Any, Any]] = _load_model,
) -> int:
    if model_loader is _load_model:
        model, torch, device = model_loader(
            model_root,
            wheel_path=wheel_path,
            wheel_sha256=wheel_sha256,
        )
    else:
        model, torch, device = model_loader(model_root)
    _write(
        stdout,
        {
            "type": "ready",
            "backend_id": BACKEND_ID,
            "model_id": MODEL_ID,
            "train_classes": list(TRAIN_CLASSES),
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "activation_rate_hz": ACTIVATION_RATE_HZ,
        },
    )
    for raw_line in stdin:
        request_id = None
        try:
            payload = json.loads(raw_line)
            request_id = _request_id(payload)
            request_id, audio_path = _valid_request(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            error: dict[str, Any] = {
                "error": {"code": "invalid_request", "message": "invalid request"}
            }
            if request_id is not None:
                error["id"] = request_id
            _write(stdout, error)
            continue

        try:
            audio = _load_audio(audio_path, torch, device)
            events = _events(model, audio, torch)
        except ValueError:
            _write(
                stdout,
                {
                    "id": request_id,
                    "error": {"code": "invalid_request", "message": "invalid request"},
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
        else:
            _write(stdout, {"id": request_id, "events": events})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the isolated IDM model worker")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--wheel-path", type=Path, required=True)
    parser.add_argument("--wheel-sha256", required=True)
    args = parser.parse_args(argv)
    return serve_requests(
        model_root=args.model_root,
        wheel_path=args.wheel_path,
        wheel_sha256=args.wheel_sha256,
    )


def _bytes_as_file(content: bytes) -> Any:
    from io import BytesIO

    return BytesIO(content)


if __name__ == "__main__":
    raise SystemExit(main())
