"""Strict stdlib-only request and WAV handling for calibration execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROTOCOL_SCHEMA = "crux.oaf-calibration-runner/v1"
_REQUEST_KEYS = frozenset(
    {
        "audio_frame_count",
        "audio_path",
        "audio_sha256",
        "max_input_audio_frames",
        "request_id",
        "type",
    }
)
_REQUEST_TYPES = frozenset({"measure", "calibration_probe"})
_OPAQUE_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CalibrationProtocolFailure(ValueError):
    """A calibration request failed before model inference."""


@dataclass(frozen=True)
class CalibrationRequest:
    request_type: str
    request_id: str
    audio_path: str
    audio_sha256: str
    audio_frame_count: int
    max_input_audio_frames: int


@dataclass(frozen=True)
class VerifiedCalibrationWav:
    relative_path: str
    content: bytes
    sha256: str
    audio_frame_count: int


def validate_calibration_request(
    content: bytes,
    *,
    authorized_max_input_audio_frames: int,
) -> CalibrationRequest:
    """Strict-parse one canonical request and enforce the reviewed frame ceiling."""

    if type(authorized_max_input_audio_frames) is not int or authorized_max_input_audio_frames <= 0:
        raise CalibrationProtocolFailure("authorized frame bound is invalid")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise CalibrationProtocolFailure("calibration request framing is invalid")
    try:
        text = content[:-1].decode("utf-8", errors="strict")
        payload = json.loads(text, object_pairs_hook=_object_from_pairs)
    except (UnicodeError, ValueError):
        raise CalibrationProtocolFailure("calibration request JSON is invalid") from None
    if not isinstance(payload, Mapping) or set(payload) != _REQUEST_KEYS:
        raise CalibrationProtocolFailure("calibration request fields are invalid")
    canonical = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if canonical != content:
        raise CalibrationProtocolFailure("calibration request bytes are not canonical")
    request_type = payload["type"]
    request_id = payload["request_id"]
    audio_sha256 = payload["audio_sha256"]
    audio_frames = payload["audio_frame_count"]
    request_maximum = payload["max_input_audio_frames"]
    if request_type not in _REQUEST_TYPES:
        raise CalibrationProtocolFailure("calibration request type is invalid")
    if not isinstance(request_id, str) or not _OPAQUE_ID.fullmatch(request_id):
        raise CalibrationProtocolFailure("calibration request ID is invalid")
    if not isinstance(audio_sha256, str) or not _SHA256.fullmatch(audio_sha256):
        raise CalibrationProtocolFailure("calibration request audio hash is invalid")
    if (
        type(audio_frames) is not int
        or audio_frames <= 0
        or type(request_maximum) is not int
        or request_maximum <= 0
        or request_maximum != authorized_max_input_audio_frames
        or (audio_frames > request_maximum and request_type != "calibration_probe")
    ):
        raise CalibrationProtocolFailure("calibration request exceeds the authorized frame bound")
    return CalibrationRequest(
        request_type=request_type,
        request_id=request_id,
        audio_path=_validate_relative_wav_path(payload["audio_path"]),
        audio_sha256=audio_sha256,
        audio_frame_count=audio_frames,
        max_input_audio_frames=request_maximum,
    )


def read_verified_calibration_wav(
    request: CalibrationRequest,
    input_root: Path,
) -> VerifiedCalibrationWav:
    """Read one no-follow canonical WAV and reproduce its declared hash and frames."""

    descriptor: int | None = None
    try:
        descriptor = _open_beneath(Path(input_root), request.audio_path)
        metadata = os.fstat(descriptor)
        if metadata.st_nlink != 1:
            raise OSError("calibration audio is multiply linked")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
    except OSError:
        raise CalibrationProtocolFailure(
            "calibration audio path is unavailable or unsafe"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    digest = hashlib.sha256(content).hexdigest()
    if digest != request.audio_sha256:
        raise CalibrationProtocolFailure("calibration audio hash does not match the request")
    frames = _parse_canonical_wav(content, request.max_input_audio_frames)
    if frames != request.audio_frame_count:
        raise CalibrationProtocolFailure("calibration audio frame count does not match the request")
    return VerifiedCalibrationWav(
        relative_path=request.audio_path,
        content=content,
        sha256=digest,
        audio_frame_count=frames,
    )


def _validate_relative_wav_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or value.startswith("/")
        or "\\" in value
        or unicodedata.normalize("NFC", value) != value
        or not value.endswith(".wav")
        or any(component in {"", ".", ".."} for component in value.split("/"))
    ):
        raise CalibrationProtocolFailure("calibration audio path is invalid")
    return value


def _open_beneath(root: Path, relative_path: str) -> int:
    root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.open(root, root_flags)
    try:
        for index, component in enumerate(relative_path.split("/")):
            final = index == len(relative_path.split("/")) - 1
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if not final:
                flags |= getattr(os, "O_DIRECTORY", 0)
            following = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = following
        if not stat.S_ISREG(os.fstat(current).st_mode):
            raise OSError("calibration audio is not regular")
        return current
    except BaseException:
        os.close(current)
        raise


def _parse_canonical_wav(content: bytes, maximum_frames: int) -> int:
    if len(content) < 44:
        raise CalibrationProtocolFailure("calibration WAV is truncated")
    try:
        riff_size = struct.unpack("<I", content[4:8])[0]
        fmt = struct.unpack("<IHHIIHH", content[16:36])
        data_size = struct.unpack("<I", content[40:44])[0]
    except struct.error:
        raise CalibrationProtocolFailure("calibration WAV header is invalid") from None
    if (
        content[:4] != b"RIFF"
        or content[8:16] != b"WAVEfmt "
        or riff_size != len(content) - 8
        or fmt != (16, 1, 1, 44100, 88200, 2, 16)
        or content[36:40] != b"data"
        or data_size <= 0
        or data_size % 2
        or data_size != len(content) - 44
    ):
        raise CalibrationProtocolFailure("calibration WAV contract is invalid")
    frames = data_size // 2
    if frames > maximum_frames:
        raise CalibrationProtocolFailure("calibration WAV exceeds the authorized frame bound")
    return frames


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value
