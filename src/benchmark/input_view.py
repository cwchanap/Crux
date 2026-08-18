from __future__ import annotations

import os
import stat
import struct
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Literal

import librosa
import soundfile

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import require_sha256, sha256_hex, strict_json_loads
from src.benchmark.backends import CanonicalAudio
from src.benchmark.corpus_cache import ResolvedSourceAudio

_MANIFEST_SCHEMA = "crux.input-view-manifest/v1"
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "source_audio_id",
        "source_audio_sha256",
        "source_path",
        "input_view_id",
        "input_audio_sha256",
        "input_audio_path",
    }
)
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


@dataclass(frozen=True)
class InputViewManifest:
    schema: Literal["crux.input-view-manifest/v1"]
    source_audio_id: str
    source_audio_sha256: str
    source_path: str
    input_view_id: str
    input_audio_sha256: str
    input_audio_path: str


@dataclass(frozen=True)
class _CanonicalWavInfo:
    byte_length: int
    sample_rate: int
    channel_count: int
    sample_width_bytes: int
    audio_frame_count: int


# Keep every physical WAV invariant adjacent in this audited linear parser.
# pylint: disable-next=too-many-branches,too-many-locals,too-many-statements
def parse_canonical_wav(
    content: bytes,
    max_input_audio_frames: int | None,
) -> _CanonicalWavInfo:
    # Exact type comparison rejects bool, which isinstance(..., int) would accept.
    # pylint: disable-next=unidiomatic-typecheck
    if max_input_audio_frames is not None and type(max_input_audio_frames) is not int:
        raise ValueError("max_input_audio_frames must be an integer or null")
    if len(content) < 12:
        raise ValueError("invalid RIFF header")

    riff_id, riff_size, wave_id = struct.unpack_from("<4sI4s", content)
    if riff_id != b"RIFF":
        raise ValueError("canonical WAV must start with RIFF")
    if wave_id != b"WAVE":
        raise ValueError("canonical WAV must use WAVE form")
    if riff_size != len(content) - 8:
        raise ValueError("canonical WAV RIFF size does not match exact content length")

    if len(content) < 20:
        raise ValueError("canonical WAV is missing the fmt chunk")
    fmt_id, fmt_size = struct.unpack_from("<4sI", content, 12)
    if fmt_id != b"fmt ":
        raise ValueError("canonical WAV must begin with a fmt chunk")
    if fmt_size != 16:
        raise ValueError("canonical WAV fmt chunk must contain exactly 16 bytes")
    if len(content) < 36:
        raise ValueError("canonical WAV has a truncated fmt chunk")

    (
        audio_format,
        channel_count,
        sample_rate,
        byte_rate,
        block_alignment,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", content, 20)
    if audio_format != 1:
        raise ValueError("canonical WAV must use integer PCM format 1")
    if channel_count != 1:
        raise ValueError("canonical WAV must be mono")
    if sample_rate != 44100:
        raise ValueError("canonical WAV sample rate must be 44100 Hz")
    if byte_rate != 88200:
        raise ValueError("canonical WAV byte rate must be 88200")
    if block_alignment != 2:
        raise ValueError("canonical WAV block alignment must be 2")
    if bits_per_sample != 16:
        raise ValueError("canonical WAV must be 16-bit PCM")

    if len(content) < 44:
        raise ValueError("canonical WAV is missing the data chunk")
    data_id, data_length = struct.unpack_from("<4sI", content, 36)
    if data_id != b"data":
        raise ValueError("canonical WAV fmt chunk must be followed by the data chunk")
    expected_length = 44 + data_length
    if len(content) < expected_length:
        raise ValueError("canonical WAV data length exceeds available bytes")
    if len(content) > expected_length:
        raise ValueError("canonical WAV contains trailing bytes or extra chunks")
    if data_length == 0:
        raise ValueError("canonical WAV data length must be positive")
    if data_length % 2 != 0:
        raise ValueError("canonical WAV data length must be even")

    audio_frame_count = data_length // 2
    if max_input_audio_frames is not None and audio_frame_count > max_input_audio_frames:
        raise ValueError("canonical WAV frame count exceeds max_input_audio_frames")

    return _CanonicalWavInfo(
        byte_length=len(content),
        sample_rate=sample_rate,
        channel_count=channel_count,
        sample_width_bytes=bits_per_sample // 8,
        audio_frame_count=audio_frame_count,
    )


def load_direct_audio(
    audio_path: Path,
    *,
    source_audio_id: str,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    content = read_regular_file_no_follow(audio_path)
    return load_direct_audio_bytes(
        audio_path,
        content,
        source_audio_id=source_audio_id,
        input_view_id=input_view_id,
        max_input_audio_frames=max_input_audio_frames,
    )


def load_direct_audio_bytes(
    audio_path: Path,
    content: bytes,
    *,
    source_audio_id: str,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    _require_nonempty_id(source_audio_id, "source_audio_id")
    _require_nonempty_id(input_view_id, "input_view_id")
    wav = parse_canonical_wav(content, max_input_audio_frames)
    audio_sha256 = sha256_hex(content)
    return CanonicalAudio(
        path=audio_path,
        source_audio_id=source_audio_id,
        source_audio_sha256=audio_sha256,
        input_view_id=input_view_id,
        input_audio_sha256=audio_sha256,
        byte_length=wav.byte_length,
        sample_rate=wav.sample_rate,
        channel_count=wav.channel_count,
        sample_width_bytes=wav.sample_width_bytes,
        audio_frame_count=wav.audio_frame_count,
    )


def load_materialized_audio(
    path: Path,
    *,
    source_audio_id: str,
    source_audio_sha256: str,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    _require_nonempty_id(source_audio_id, "source_audio_id")
    require_sha256(source_audio_sha256, "source_audio_sha256")
    _require_nonempty_id(input_view_id, "input_view_id")
    content = read_regular_file_no_follow(path)
    wav = parse_canonical_wav(content, max_input_audio_frames)
    return CanonicalAudio(
        path=path,
        source_audio_id=source_audio_id,
        source_audio_sha256=source_audio_sha256,
        input_view_id=input_view_id,
        input_audio_sha256=sha256_hex(content),
        byte_length=wav.byte_length,
        sample_rate=wav.sample_rate,
        channel_count=wav.channel_count,
        sample_width_bytes=wav.sample_width_bytes,
        audio_frame_count=wav.audio_frame_count,
    )


def materialize_full_mix_audio(
    source_audio: ResolvedSourceAudio,
    output_path: Path,
    *,
    input_root: Path,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    """Materialize one temporary canonical audio input beneath ``input_root``."""
    if not isinstance(source_audio, ResolvedSourceAudio):
        raise TypeError("source_audio must be ResolvedSourceAudio")
    source_input = (
        BytesIO(source_audio.content) if source_audio.content is not None else source_audio.path
    )
    return _materialize_canonical_audio(
        source_audio,
        source_input,
        output_path,
        input_root=input_root,
        input_view_id=input_view_id,
        max_input_audio_frames=max_input_audio_frames,
    )


def materialize_derived_audio(
    source_audio: ResolvedSourceAudio,
    derived_audio_path: Path,
    output_path: Path,
    *,
    input_root: Path,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    """Canonicalize one retained derived stem with authoritative source identity."""
    if not isinstance(derived_audio_path, Path):
        raise TypeError("derived_audio_path must be a Path")
    return _materialize_canonical_audio(
        source_audio,
        derived_audio_path,
        output_path,
        input_root=input_root,
        input_view_id=input_view_id,
        max_input_audio_frames=max_input_audio_frames,
    )


def _materialize_canonical_audio(
    source_audio: ResolvedSourceAudio,
    source_input: BytesIO | Path,
    output_path: Path,
    *,
    input_root: Path,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    """Canonicalize one source input while retaining the verified source identity."""
    if not isinstance(source_audio, ResolvedSourceAudio):
        raise TypeError("source_audio must be ResolvedSourceAudio")
    if not isinstance(output_path, Path) or not isinstance(input_root, Path):
        raise TypeError("output_path and input_root must be Paths")
    root = input_root.resolve()
    destination = output_path.resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        raise ValueError("canonical input must be beneath input_root") from None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples, _ = librosa.load(source_input, sr=44100, mono=True, res_type="soxr_hq")
    soundfile.write(output_path, samples, 44100, format="WAV", subtype="PCM_16")
    return load_materialized_audio(
        path=output_path,
        source_audio_id=source_audio.source_audio_id,
        source_audio_sha256=source_audio.source_audio_sha256,
        input_view_id=input_view_id,
        max_input_audio_frames=max_input_audio_frames,
    )


def load_derived_audio(
    audio_path: Path,
    manifest_path: Path,
    *,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    manifest = _load_manifest(manifest_path)

    manifest_root = manifest_path.parent.resolve(strict=True)
    try:
        manifest_root_fd = os.open(manifest_root, _DIRECTORY_OPEN_FLAGS)
    except OSError:
        raise ValueError("manifest directory must remain a readable directory") from None
    try:
        source_path = _resolve_manifest_path(
            manifest_root,
            manifest.source_path,
            "source_path",
        )
        input_path = _resolve_manifest_path(
            manifest_root,
            manifest.input_audio_path,
            "input_audio_path",
        )

        try:
            resolved_audio_path = audio_path.resolve(strict=True)
        except OSError:
            raise ValueError("audio_path must name a readable canonical artifact") from None
        if resolved_audio_path != input_path:
            raise ValueError("audio_path does not match manifest input_audio_path")

        source_content = _read_manifest_artifact(
            manifest_root_fd,
            source_path.relative_to(manifest_root),
            "source_path",
        )
        input_content = _read_manifest_artifact(
            manifest_root_fd,
            input_path.relative_to(manifest_root),
            "input_audio_path",
        )
    finally:
        _close_fd(manifest_root_fd)

    return load_derived_audio_bytes(
        input_path,
        manifest,
        source_content=source_content,
        input_content=input_content,
        max_input_audio_frames=max_input_audio_frames,
    )


def load_derived_audio_bytes(
    audio_path: Path,
    manifest: InputViewManifest,
    *,
    source_content: bytes,
    input_content: bytes,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    if sha256_hex(source_content) != manifest.source_audio_sha256:
        raise ValueError("source_audio_sha256 does not match source_path bytes")
    if sha256_hex(input_content) != manifest.input_audio_sha256:
        raise ValueError("input_audio_sha256 does not match input_audio_path bytes")
    wav = parse_canonical_wav(input_content, max_input_audio_frames)
    return _canonical_audio(manifest, audio_path, wav)


def _canonical_audio(
    manifest: InputViewManifest,
    input_path: Path,
    wav: _CanonicalWavInfo,
) -> CanonicalAudio:
    return CanonicalAudio(
        path=input_path,
        source_audio_id=manifest.source_audio_id,
        source_audio_sha256=manifest.source_audio_sha256,
        input_view_id=manifest.input_view_id,
        input_audio_sha256=manifest.input_audio_sha256,
        byte_length=wav.byte_length,
        sample_rate=wav.sample_rate,
        channel_count=wav.channel_count,
        sample_width_bytes=wav.sample_width_bytes,
        audio_frame_count=wav.audio_frame_count,
    )


def _load_manifest(
    manifest_path: Path,
) -> InputViewManifest:
    return parse_input_view_manifest(read_regular_file_no_follow(manifest_path))


def parse_input_view_manifest(content: bytes) -> InputViewManifest:
    value = strict_json_loads(content)
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise ValueError("input-view manifest must contain the exact seven-key set")
    if any(not isinstance(field_value, str) for field_value in value.values()):
        raise ValueError("input-view manifest fields must be strings")

    fields = {
        key: field_value for key, field_value in value.items() if isinstance(field_value, str)
    }
    if fields["schema"] != _MANIFEST_SCHEMA:
        raise ValueError(f"schema must be {_MANIFEST_SCHEMA}")
    _require_nonempty_id(fields["source_audio_id"], "source_audio_id")
    _require_nonempty_id(fields["input_view_id"], "input_view_id")
    require_sha256(fields["source_audio_sha256"], "source_audio_sha256")
    require_sha256(fields["input_audio_sha256"], "input_audio_sha256")

    return InputViewManifest(
        schema=_MANIFEST_SCHEMA,
        source_audio_id=fields["source_audio_id"],
        source_audio_sha256=fields["source_audio_sha256"],
        source_path=fields["source_path"],
        input_view_id=fields["input_view_id"],
        input_audio_sha256=fields["input_audio_sha256"],
        input_audio_path=fields["input_audio_path"],
    )


def validate_schema_golden(schema: str, content: bytes) -> None:
    if schema != _MANIFEST_SCHEMA:
        raise ValueError("unsupported schema golden")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("schema golden must have one final newline")
    strict_json_loads(content[:-1], require_canonical=True)
    parse_input_view_manifest(content)


def input_view_artifact_paths(
    manifest_path: Path,
    manifest: InputViewManifest,
) -> tuple[Path, Path]:
    manifest_root = manifest_path.parent
    return (
        _anchored_manifest_path(
            manifest_root,
            manifest.source_path,
            "source_path",
        ),
        _anchored_manifest_path(
            manifest_root,
            manifest.input_audio_path,
            "input_audio_path",
        ),
    )


def _resolve_manifest_path(root: Path, raw_path: str, field: str) -> Path:
    path = PurePosixPath(raw_path)
    if not raw_path or path.is_absolute():
        raise ValueError(f"{field} must be a nonempty POSIX relative path")
    if ".." in path.parts:
        raise ValueError(f"{field} must not contain parent traversal")

    try:
        resolved = root.joinpath(*path.parts).resolve(strict=True)
    except OSError:
        raise ValueError(f"{field} must name a readable artifact") from None
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field} symlink escapes the manifest directory")
    return resolved


def _anchored_manifest_path(root: Path, raw_path: str, field: str) -> Path:
    path = PurePosixPath(raw_path)
    if not raw_path or path.is_absolute():
        raise ValueError(f"{field} must be a nonempty POSIX relative path")
    if any(component in {"", ".", ".."} for component in path.parts):
        raise ValueError(f"{field} contains an invalid path component")
    return root.joinpath(*path.parts)


def _read_manifest_artifact(root_fd: int, relative_path: Path, field: str) -> bytes:
    if not relative_path.parts:
        raise ValueError(f"{field} must name an in-root regular artifact")

    opened_fds: list[int] = []
    try:
        current_fd = os.dup(root_fd)
        opened_fds.append(current_fd)
        for component in relative_path.parts[:-1]:
            current_fd = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=current_fd,
            )
            opened_fds.append(current_fd)

        file_fd = os.open(
            relative_path.parts[-1],
            _FILE_OPEN_FLAGS,
            dir_fd=current_fd,
        )
        opened_fds.append(file_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise OSError

        chunks: list[bytes] = []
        while chunk := os.read(file_fd, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError:
        raise ValueError(f"{field} must remain a readable in-root regular artifact") from None
    finally:
        for descriptor in reversed(opened_fds):
            _close_fd(descriptor)


def _close_fd(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _require_nonempty_id(value: str, field: str) -> None:
    if not value:
        raise ValueError(f"{field} must not be empty")
