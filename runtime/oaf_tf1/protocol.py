"""Strict stdlib-only wire and mounted-input handling for the frozen runner."""

# Canonical JSON and WAV parsing are deliberately explicit fail-closed state
# machines; keeping the branches local makes each accepted shape reviewable.
# pylint: disable=too-many-arguments,too-many-boolean-expressions,too-many-branches
# pylint: disable=too-many-locals,too-many-return-statements,too-many-statements
# pylint: disable=try-except-raise

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO, FrozenSet, Mapping, Optional, Sequence, Tuple

PROTOCOL_SCHEMA = "crux.transcription-runner/v1"
_OPAQUE_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_DIAGNOSTIC = re.compile(
    rb"code=[a-z][a-z0-9_]*"
    rb"(?: tensor=[A-Za-z0-9_.:/-]+)?"
    rb"(?: count=[0-9]+)?"
    rb"(?: duration_ms=[0-9]+)?\Z"
)
_FORBIDDEN_DIAGNOSTIC = (
    b"traceback",
    b'file "',
    b"secret",
    b"token",
    b"credential",
    b"password",
    b"http://",
    b"https://",
    b"audio=",
    b"samples=",
    b"path=",
)
_ABSOLUTE_PATH = re.compile(rb"(?:^|[\s=])(?:/|[A-Za-z]:[\\/]|\\\\)[^\s]*")
_REQUEST_KEYS = frozenset(
    {
        "audio_path",
        "audio_sha256",
        "backend_descriptor_sha256",
        "request_id",
        "type",
    }
)


class ProtocolFailure(Exception):
    """A stable runner failure safe to expose over the protocol."""

    def __init__(self, code: str, message: str, *, fatal: bool) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fatal = fatal


@dataclass(frozen=True)
class AuthenticatedObject:
    payload: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True)
class TranscribeRequest:
    request_id: str
    audio_path: str
    audio_sha256: str
    backend_descriptor_sha256: str


@dataclass(frozen=True)
class VerifiedWav:
    relative_path: str
    content: bytes
    sha256: str
    audio_frame_count: int


def _pairs_object(pairs: Sequence[Tuple[str, Any]]) -> Mapping[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("nonfinite JSON value")


def _strict_json_loads(content: bytes) -> Any:
    try:
        text = content.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_pairs_object,
            parse_float=Decimal,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError):
        raise ProtocolFailure(
            "protocol_input_invalid",
            "The runner received invalid canonical JSON.",
            fatal=True,
        ) from None


def _render_json(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("nonfinite Decimal")
        rendered = format(value, "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return rendered or "0"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite float")
        return json.dumps(value, allow_nan=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_render_json(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object key is not a string")
        return (
            "{"
            + ",".join(_render_json(key) + ":" + _render_json(value[key]) for key in sorted(value))
            + "}"
        )
    raise ValueError("unsupported JSON value")


def canonical_json_bytes(value: Any, *, trailing_newline: bool) -> bytes:
    try:
        content = _render_json(value).encode("utf-8")
    except (TypeError, ValueError):
        raise ProtocolFailure(
            "protocol_output_invalid",
            "The runner could not encode a protocol object.",
            fatal=True,
        ) from None
    if trailing_newline:
        content += b"\n"
    return content


def canonical_json_line(value: Any, maximum_bytes: int) -> bytes:
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int):
        raise ProtocolFailure(
            "protocol_output_invalid",
            "The protocol output bound is invalid.",
            fatal=True,
        )
    content = canonical_json_bytes(value, trailing_newline=True)
    if maximum_bytes <= 0 or len(content) > maximum_bytes:
        raise ProtocolFailure(
            "protocol_output_oversized",
            "The runner protocol object exceeds the locked byte bound.",
            fatal=True,
        )
    return content


def _read_regular_no_follow(path: Path, maximum_bytes: int) -> bytes:
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    root_fd = os.open(path.anchor, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    current_fd = root_fd
    try:
        for index, component in enumerate(path.parts[1:]):
            final = index == len(path.parts[1:]) - 1
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if not final:
                flags |= getattr(os, "O_DIRECTORY", 0)
            next_fd = os.open(component, flags, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        status = os.fstat(current_fd)
        if not stat.S_ISREG(status.st_mode):
            raise OSError("not a regular file")
        chunks = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(current_fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > maximum_bytes:
            raise OSError("file exceeds bound")
        final_status = os.fstat(current_fd)
        if (
            final_status.st_dev,
            final_status.st_ino,
            final_status.st_size,
        ) != (status.st_dev, status.st_ino, status.st_size):
            raise OSError("file identity changed")
        return content
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def load_authenticated_object(
    path: Path,
    *,
    label: str,
    exact_keys: FrozenSet[str],
    expected_schema: str,
    expected_sha256: Optional[str] = None,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> AuthenticatedObject:
    try:
        content = _read_regular_no_follow(Path(path), maximum_bytes)
        payload = _strict_json_loads(content)
        if not isinstance(payload, Mapping):
            raise ValueError("mounted JSON is not an object")
        if set(payload) != set(exact_keys):
            raise ValueError("mounted JSON keys differ")
        if payload.get("schema") != expected_schema:
            raise ValueError("mounted JSON schema differs")
        if canonical_json_bytes(payload, trailing_newline=True) != content:
            raise ValueError("mounted JSON is not canonical")
        digest = hashlib.sha256(content).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("mounted JSON hash differs")
    except (OSError, ProtocolFailure, TypeError, ValueError):
        raise ProtocolFailure(
            "mounted_identity_invalid",
            "A mounted runner identity is invalid: " + label + ".",
            fatal=True,
        ) from None
    return AuthenticatedObject(payload=payload, sha256=digest)


def _require_sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProtocolFailure(code, "A request SHA-256 value is invalid.", fatal=False)
    return value


def _validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProtocolFailure("input_path_invalid", "The input path is invalid.", fatal=False)
    if value.startswith("/") or "\\" in value or unicodedata.normalize("NFC", value) != value:
        raise ProtocolFailure("input_path_invalid", "The input path is invalid.", fatal=False)
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ProtocolFailure("input_path_invalid", "The input path is invalid.", fatal=False)
    if not value.endswith(".wav"):
        raise ProtocolFailure("input_path_invalid", "The input path is invalid.", fatal=False)
    return value


def validate_transcribe_request(
    payload: Mapping[str, Any], *, expected_descriptor_sha256: str
) -> TranscribeRequest:
    if not isinstance(payload, Mapping) or set(payload) != set(_REQUEST_KEYS):
        raise ProtocolFailure(
            "request_invalid",
            "The transcription request does not match the protocol schema.",
            fatal=False,
        )
    if payload.get("type") != "transcribe":
        raise ProtocolFailure(
            "request_invalid",
            "The transcription request type is invalid.",
            fatal=False,
        )
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not _OPAQUE_ID.fullmatch(request_id):
        raise ProtocolFailure(
            "request_invalid",
            "The transcription request ID is invalid.",
            fatal=False,
        )
    descriptor_sha256 = _require_sha256(payload.get("backend_descriptor_sha256"), "request_invalid")
    if descriptor_sha256 != expected_descriptor_sha256:
        raise ProtocolFailure(
            "request_descriptor_mismatch",
            "The request backend descriptor does not match the ready runner.",
            fatal=False,
        )
    return TranscribeRequest(
        request_id=request_id,
        audio_path=_validate_relative_path(payload.get("audio_path")),
        audio_sha256=_require_sha256(payload.get("audio_sha256"), "request_invalid"),
        backend_descriptor_sha256=descriptor_sha256,
    )


def _open_beneath(root: Path, relative_path: str) -> int:
    root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(root, root_flags)
    try:
        components = relative_path.split("/")
        for index, component in enumerate(components):
            final = index == len(components) - 1
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if not final:
                flags |= getattr(os, "O_DIRECTORY", 0)
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        status = os.fstat(current_fd)
        if not stat.S_ISREG(status.st_mode):
            raise OSError("input is not regular")
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _parse_canonical_wav(content: bytes, maximum_frames: int) -> int:
    if len(content) < 44:
        raise ValueError("WAV is truncated")
    if content[:4] != b"RIFF" or content[8:16] != b"WAVEfmt ":
        raise ValueError("WAV identity differs")
    riff_size = struct.unpack("<I", content[4:8])[0]
    fmt_size, audio_format, channels, rate, byte_rate, block_align, bits = struct.unpack(
        "<IHHIIHH", content[16:36]
    )
    if (
        riff_size != len(content) - 8
        or fmt_size != 16
        or audio_format != 1
        or channels != 1
        or rate != 44100
        or byte_rate != 88200
        or block_align != 2
        or bits != 16
        or content[36:40] != b"data"
    ):
        raise ValueError("WAV contract differs")
    data_size = struct.unpack("<I", content[40:44])[0]
    if data_size <= 0 or data_size % 2 or data_size != len(content) - 44:
        raise ValueError("WAV data differs")
    frames = data_size // 2
    if frames > maximum_frames:
        raise ProtocolFailure(
            "input_too_long",
            "The canonical input exceeds the locked frame bound.",
            fatal=False,
        )
    return frames


def read_verified_canonical_wav(
    input_root: Path,
    relative_path: str,
    expected_sha256: str,
    maximum_frames: int,
) -> VerifiedWav:
    _validate_relative_path(relative_path)
    _require_sha256(expected_sha256, "request_invalid")
    if (
        isinstance(maximum_frames, bool)
        or not isinstance(maximum_frames, int)
        or maximum_frames <= 0
    ):
        raise ProtocolFailure(
            "mounted_identity_invalid",
            "The locked input frame bound is invalid.",
            fatal=True,
        )
    descriptor = -1
    try:
        descriptor = _open_beneath(Path(input_root), relative_path)
        status = os.fstat(descriptor)
        maximum_bytes = 44 + maximum_frames * 2
        if status.st_size > maximum_bytes:
            header = os.read(descriptor, 44)
            try:
                if len(header) != 44:
                    raise ValueError("WAV header is truncated")
                data_size = struct.unpack("<I", header[40:44])[0]
                declared_size = struct.unpack("<I", header[4:8])[0] + 8
                canonical_size = 44 + data_size
                if (
                    header[:4] != b"RIFF"
                    or header[8:16] != b"WAVEfmt "
                    or header[36:40] != b"data"
                    or declared_size != status.st_size
                    or canonical_size != status.st_size
                ):
                    raise ValueError("WAV size is noncanonical")
                if data_size // 2 > maximum_frames:
                    raise ProtocolFailure(
                        "input_too_long",
                        "The canonical input exceeds the locked frame bound.",
                        fatal=False,
                    )
                raise ValueError("WAV exceeds its canonical size")
            except ProtocolFailure:
                raise
            except (ValueError, struct.error):
                raise ProtocolFailure(
                    "input_wav_invalid",
                    "The canonical input does not match the locked WAV contract.",
                    fatal=False,
                ) from None
        os.lseek(descriptor, 0, os.SEEK_SET)
        content_parts = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            content_parts.append(chunk)
            remaining -= len(chunk)
        content = b"".join(content_parts)
        final_status = os.fstat(descriptor)
        if (status.st_dev, status.st_ino, status.st_size) != (
            final_status.st_dev,
            final_status.st_ino,
            final_status.st_size,
        ):
            raise OSError("input identity changed")
    except OSError:
        raise ProtocolFailure(
            "input_path_invalid",
            "The canonical input path is unavailable or unsafe.",
            fatal=False,
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_sha256:
        raise ProtocolFailure(
            "input_hash_mismatch",
            "The canonical input hash does not match the request.",
            fatal=False,
        )
    try:
        frames = _parse_canonical_wav(content, maximum_frames)
    except ProtocolFailure:
        raise
    except (ValueError, struct.error):
        raise ProtocolFailure(
            "input_wav_invalid",
            "The canonical input does not match the locked WAV contract.",
            fatal=False,
        ) from None
    return VerifiedWav(
        relative_path=relative_path,
        content=content,
        sha256=digest,
        audio_frame_count=frames,
    )


def sanitize_diagnostic(raw: bytes) -> bytes:
    lowered = raw.lower()
    if (
        _ABSOLUTE_PATH.search(raw)
        or any(marker in lowered for marker in _FORBIDDEN_DIAGNOSTIC)
        or not _SAFE_DIAGNOSTIC.fullmatch(raw)
    ):
        return b"[REDACTED]"
    return raw


def _write_line(stdout: BinaryIO, payload: Mapping[str, Any], maximum_bytes: int) -> None:
    stdout.write(canonical_json_line(payload, maximum_bytes))
    stdout.flush()


def serve_requests(
    *,
    stdin: BinaryIO,
    stdout: BinaryIO,
    backend: Any,
    input_root: Path,
    descriptor_sha256: str,
    max_input_audio_frames: int,
    stdout_max_line_bytes: int,
) -> None:
    while True:
        line = stdin.readline(stdout_max_line_bytes + 1)
        if not line:
            return
        if len(line) > stdout_max_line_bytes or not line.endswith(b"\n"):
            raise ProtocolFailure(
                "protocol_input_invalid",
                "The request protocol line is invalid.",
                fatal=True,
            )
        value = _strict_json_loads(line[:-1])
        if canonical_json_bytes(value, trailing_newline=False) != line[:-1]:
            raise ProtocolFailure(
                "protocol_input_invalid",
                "The request protocol object is not canonical.",
                fatal=True,
            )
        if not isinstance(value, Mapping):
            raise ProtocolFailure(
                "protocol_input_invalid",
                "The request protocol value is not an object.",
                fatal=True,
            )
        request_id = value.get("request_id")
        try:
            request = validate_transcribe_request(
                value, expected_descriptor_sha256=descriptor_sha256
            )
            verified = read_verified_canonical_wav(
                input_root,
                request.audio_path,
                request.audio_sha256,
                max_input_audio_frames,
            )
            native_events = backend.transcribe(verified)
            payload = {
                "audio_sha256": verified.sha256,
                "backend_descriptor_sha256": descriptor_sha256,
                "native_events": native_events,
                "type": "transcription_result",
            }
            response_request_id = request.request_id
        except ProtocolFailure as error:
            if error.fatal:
                raise
            payload = {
                "code": error.code,
                "message": error.message,
                "type": "transcription_error",
            }
            response_request_id = request_id if isinstance(request_id, str) else ""
        response = {
            "payload": payload,
            "request_id": response_request_id,
            "type": "response",
        }
        _write_line(stdout, response, stdout_max_line_bytes)
