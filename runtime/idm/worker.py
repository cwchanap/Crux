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
import stat
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
    site_packages: Path | None = None,
    model_config_path: Path | None = None,
    model_config_sha256: str | None = None,
    model_config_byte_length: int | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str | None = None,
    checkpoint_byte_length: int | None = None,
) -> tuple[Any, Any, Any]:
    global _WHEEL_PROJECT
    if sys.flags.isolated != 1 or sys.flags.no_site != 1:
        raise ValueError("IDM worker requires isolated no-site interpreter startup")
    if any(name == "idm" or name.startswith("idm.") for name in sys.modules):
        raise ValueError("IDM package was imported before attestation")
    if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise ValueError("Python customization was imported before attestation")
    if (
        wheel_path is None
        or wheel_sha256 is None
        or site_packages is None
        or model_config_path is None
        or model_config_sha256 is None
        or model_config_byte_length is None
        or checkpoint_path is None
        or checkpoint_sha256 is None
        or checkpoint_byte_length is None
    ):
        raise ValueError("attested IDM runtime and model files are required")
    wheel_bytes = _read_regular_file_no_follow(wheel_path)
    if hashlib.sha256(wheel_bytes).hexdigest() != wheel_sha256:
        raise ValueError("attested IDM wheel digest does not match")

    if _WHEEL_PROJECT is not None:
        _WHEEL_PROJECT.cleanup()
    _WHEEL_PROJECT = tempfile.TemporaryDirectory(prefix="crux-idm-wheel-")
    project_root = Path(_WHEEL_PROJECT.name)
    with zipfile.ZipFile(_bytes_as_file(wheel_bytes)) as wheel:
        wheel.extractall(project_root)
    (project_root / ".project-root").touch()
    _stage_model_files(
        project_root,
        model_config_path,
        model_config_sha256,
        model_config_byte_length,
        checkpoint_path,
        checkpoint_sha256,
        checkpoint_byte_length,
    )
    _configure_isolated_imports(site_packages, project_root)
    import torch

    os.chdir(project_root)
    from idm.inference import load_model

    with contextlib.redirect_stdout(sys.stderr):
        model, _ = load_model("idm-44-train-kits", torch.device("cpu"), log_dir=Path("pretrained"))
    model.eval()
    return model, torch, torch.device("cpu")


def _read_regular_file_no_follow(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_length: int | None = None,
) -> bytes:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("no-follow file reads are unavailable")
    descriptor = os.open(
        path,
        os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("attested file is not regular")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        length = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                content = b"".join(chunks)
                if expected_length is not None and length != expected_length:
                    raise ValueError("attested file length does not match")
                if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                    raise ValueError("attested file digest does not match")
                return content
            chunks.append(chunk)
            digest.update(chunk)
            length += len(chunk)
    finally:
        os.close(descriptor)


def _read_attested_file(path: Path, digest: str, length: int) -> bytes:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(length) is not int
        or length < 0
    ):
        raise ValueError("attested file identity is invalid")
    return _read_regular_file_no_follow(
        path,
        expected_sha256=digest,
        expected_length=length,
    )


def _write_private_file(path: Path, content: bytes) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("no-follow file writes are unavailable")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short private file write")
            view = view[written:]
    finally:
        os.close(descriptor)
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("staged model file is not private")


def _stage_model_files(
    project_root: Path,
    model_config_path: Path,
    model_config_sha256: str,
    model_config_byte_length: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_byte_length: int,
) -> None:
    model_config = _read_attested_file(
        model_config_path,
        model_config_sha256,
        model_config_byte_length,
    )
    checkpoint = _read_attested_file(
        checkpoint_path,
        checkpoint_sha256,
        checkpoint_byte_length,
    )
    model_data_root = project_root / MODEL_CONFIG_RELATIVE_PATH.parent
    model_data_root.mkdir(parents=True)
    _write_private_file(model_data_root / MODEL_CONFIG_RELATIVE_PATH.name, model_config)
    _write_private_file(model_data_root / CHECKPOINT_RELATIVE_PATH.name, checkpoint)


def _configure_isolated_imports(site_packages: Path, wheel_project: Path) -> None:
    if (
        not isinstance(site_packages, Path)
        or site_packages.is_symlink()
        or not site_packages.is_dir()
    ):
        raise ValueError("isolated runtime site-packages are unavailable")
    sys.path.insert(0, os.fspath(wheel_project))
    sys.path.insert(1, os.fspath(site_packages))


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
    site_packages: Path | None = None,
    model_config_path: Path | None = None,
    model_config_sha256: str | None = None,
    model_config_byte_length: int | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str | None = None,
    checkpoint_byte_length: int | None = None,
    model_loader: Callable[..., tuple[Any, Any, Any]] = _load_model,
) -> int:
    if model_loader is _load_model:
        model, torch, device = model_loader(
            model_root,
            wheel_path=wheel_path,
            wheel_sha256=wheel_sha256,
            site_packages=site_packages,
            model_config_path=model_config_path,
            model_config_sha256=model_config_sha256,
            model_config_byte_length=model_config_byte_length,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_byte_length=checkpoint_byte_length,
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
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--model-config-path", type=Path, required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--model-config-byte-length", type=int, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-byte-length", type=int, required=True)
    args = parser.parse_args(argv)
    return serve_requests(
        model_root=args.model_root,
        wheel_path=args.wheel_path,
        wheel_sha256=args.wheel_sha256,
        site_packages=args.site_packages,
        model_config_path=args.model_config_path,
        model_config_sha256=args.model_config_sha256,
        model_config_byte_length=args.model_config_byte_length,
        checkpoint_path=args.checkpoint_path,
        checkpoint_sha256=args.checkpoint_sha256,
        checkpoint_byte_length=args.checkpoint_byte_length,
    )


def _bytes_as_file(content: bytes) -> Any:
    from io import BytesIO

    return BytesIO(content)


if __name__ == "__main__":
    raise SystemExit(main())
