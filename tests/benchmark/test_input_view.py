from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields
from hashlib import sha256
from pathlib import Path

import pytest

from src.benchmark.corpus_cache import ResolvedSourceAudio
from src.benchmark.input_view import (
    InputViewManifest,
    load_derived_audio,
    load_direct_audio,
    load_materialized_audio,
    materialize_derived_audio,
    materialize_full_mix_audio,
    parse_canonical_wav,
)

MANIFEST_SCHEMA = "crux.input-view-manifest/v1"


def canonical_wav_bytes(
    *,
    sample_frames: int = 1,
    audio_format: int = 1,
    channel_count: int = 1,
    sample_rate: int = 44100,
    byte_rate: int = 88200,
    block_alignment: int = 2,
    bits_per_sample: int = 16,
) -> bytes:
    fmt_payload = struct.pack(
        "<HHIIHH",
        audio_format,
        channel_count,
        sample_rate,
        byte_rate,
        block_alignment,
        bits_per_sample,
    )
    data = b"\0\0" * sample_frames
    chunks = b"fmt " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    chunks += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def manifest_payload(root: Path, source: Path, canonical: Path) -> dict[str, str]:
    return {
        "schema": MANIFEST_SCHEMA,
        "source_audio_id": "song-42-source-v1",
        "source_audio_sha256": sha256(source.read_bytes()).hexdigest(),
        "source_path": source.relative_to(root).as_posix(),
        "input_view_id": "full-mix-canonical-wav-v1",
        "input_audio_sha256": sha256(canonical.read_bytes()).hexdigest(),
        "input_audio_path": canonical.relative_to(root).as_posix(),
    }


def write_manifest(root: Path, payload: dict[str, object]) -> Path:
    manifest = root / "input-view.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def write_input_view_manifest(root: Path, source: Path, canonical: Path) -> Path:
    return write_manifest(root, manifest_payload(root, source, canonical))


def rewrite_u32(content: bytes, offset: int, value: int) -> bytes:
    changed = bytearray(content)
    struct.pack_into("<I", changed, offset, value)
    return bytes(changed)


def install_swap_before_artifact_open(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_path: Path,
    relative_open_name: str,
    swap: Callable[[], None],
    relative_open_occurrence: int = 1,
) -> None:
    original_read_bytes = Path.read_bytes
    original_os_open = os.open
    swapped = False
    relative_open_count = 0

    def swap_once() -> None:
        nonlocal swapped
        if not swapped:
            swap()
            swapped = True

    def read_bytes(path: Path) -> bytes:
        if path == target_path:
            swap_once()
        return original_read_bytes(path)

    def open_file(
        path: str | bytes,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal relative_open_count
        if dir_fd is not None and path == relative_open_name:
            relative_open_count += 1
            if relative_open_count == relative_open_occurrence:
                swap_once()
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(os, "open", open_file)


def test_direct_audio_hashes_one_file_as_source_and_input(tmp_path: Path) -> None:
    audio_path = tmp_path / "input.wav"
    content = canonical_wav_bytes(sample_frames=441)
    audio_path.write_bytes(content)

    audio = load_direct_audio(
        audio_path,
        source_audio_id="song-42-source-v1",
        input_view_id="full-mix-canonical-wav-v1",
        max_input_audio_frames=441,
    )

    assert audio.path == audio_path
    assert audio.source_audio_id == "song-42-source-v1"
    assert audio.input_view_id == "full-mix-canonical-wav-v1"
    assert audio.source_audio_sha256 == sha256(content).hexdigest()
    assert audio.source_audio_sha256 == audio.input_audio_sha256
    assert audio.byte_length == len(content)
    assert audio.audio_frame_count == 441
    assert audio.sample_rate == 44100
    assert audio.channel_count == 1
    assert audio.sample_width_bytes == 2


@pytest.mark.parametrize("field", ["source_audio_id", "input_view_id"])
def test_direct_audio_rejects_empty_stable_ids(tmp_path: Path, field: str) -> None:
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(canonical_wav_bytes())
    arguments = {
        "source_audio_id": "source-v1",
        "input_view_id": "canonical-v1",
    }
    arguments[field] = ""

    with pytest.raises(ValueError, match=field):
        load_direct_audio(audio_path, max_input_audio_frames=1, **arguments)


def test_materialize_full_mix_audio_preserves_identity_and_canonical_metadata(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.wav"
    source_content = canonical_wav_bytes(sample_frames=441)
    source_path.write_bytes(source_content)
    source = ResolvedSourceAudio(
        path=source_path,
        source_audio_id="song-42-source-v1",
        source_audio_sha256=sha256(source_content).hexdigest(),
        duration_sec=0.01,
    )
    input_root = tmp_path / "inputs"
    output_path = input_root / "42" / "full-mix.wav"

    audio = materialize_full_mix_audio(
        source,
        output_path,
        input_root=input_root,
        input_view_id="full-mix-canonical-wav-v1",
        max_input_audio_frames=441,
    )

    assert audio.path == output_path
    assert audio.source_audio_id == source.source_audio_id
    assert audio.source_audio_sha256 == source.source_audio_sha256
    assert audio.input_view_id == "full-mix-canonical-wav-v1"
    assert audio.input_audio_sha256 == sha256(output_path.read_bytes()).hexdigest()
    assert audio.sample_rate == 44100
    assert audio.channel_count == 1
    assert audio.sample_width_bytes == 2
    assert audio.audio_frame_count == 441


def test_materialize_derived_audio_uses_retained_stem_and_authoritative_source_identity(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.flac"
    source_content = b"authoritative source bytes"
    source_path.write_bytes(source_content)
    source = ResolvedSourceAudio(
        path=source_path,
        source_audio_id="song-42-source-v1",
        source_audio_sha256=sha256(source_content).hexdigest(),
        duration_sec=0.01,
    )
    derived_audio_path = tmp_path / "retained" / "drums.wav"
    derived_content = canonical_wav_bytes(sample_frames=441)
    derived_audio_path.parent.mkdir()
    derived_audio_path.write_bytes(derived_content)
    input_root = tmp_path / "inputs"
    output_path = input_root / "42" / "spleeter.wav"

    audio = materialize_derived_audio(
        source,
        derived_audio_path,
        output_path,
        input_root=input_root,
        input_view_id="crux.oaf-spleeter4-drums-mono44k1-pcm16/v1",
        max_input_audio_frames=441,
    )

    assert audio.path == output_path
    assert audio.source_audio_id == source.source_audio_id
    assert audio.source_audio_sha256 == source.source_audio_sha256
    assert audio.input_view_id == "crux.oaf-spleeter4-drums-mono44k1-pcm16/v1"
    assert audio.input_audio_sha256 == sha256(output_path.read_bytes()).hexdigest()
    assert audio.input_audio_sha256 != source.source_audio_sha256
    assert audio.audio_frame_count == 441


def test_materialized_audio_preserves_source_identity_and_hashes_input(tmp_path: Path) -> None:
    audio_path = tmp_path / "materialized.wav"
    content = canonical_wav_bytes(sample_frames=441)
    audio_path.write_bytes(content)
    source_digest = sha256(b"authoritative source bytes").hexdigest()

    audio = load_materialized_audio(
        path=audio_path,
        source_audio_id="song-42-source-v1",
        source_audio_sha256=source_digest,
        input_view_id="crux-oaf-full-mix-mono44k1-pcm16/v1",
        max_input_audio_frames=441,
    )

    assert audio.path == audio_path
    assert audio.source_audio_id == "song-42-source-v1"
    assert audio.source_audio_sha256 == source_digest
    assert audio.input_audio_sha256 == sha256(content).hexdigest()
    assert audio.input_view_id == "crux-oaf-full-mix-mono44k1-pcm16/v1"
    assert audio.byte_length == len(content)
    assert audio.sample_rate == 44100
    assert audio.channel_count == 1
    assert audio.sample_width_bytes == 2
    assert audio.audio_frame_count == 441


def test_materialized_audio_rejects_malformed_source_sha256(tmp_path: Path) -> None:
    audio_path = tmp_path / "materialized.wav"
    audio_path.write_bytes(canonical_wav_bytes())

    with pytest.raises(ValueError, match="source_audio_sha256"):
        load_materialized_audio(
            audio_path,
            source_audio_id="song-42-source-v1",
            source_audio_sha256="not-a-sha256",
            input_view_id="crux-oaf-full-mix-mono44k1-pcm16/v1",
            max_input_audio_frames=1,
        )


def test_materialized_audio_enforces_frame_limit(tmp_path: Path) -> None:
    audio_path = tmp_path / "materialized.wav"
    audio_path.write_bytes(canonical_wav_bytes(sample_frames=2))

    with pytest.raises(ValueError, match="frame count exceeds max_input_audio_frames"):
        load_materialized_audio(
            audio_path,
            source_audio_id="song-42-source-v1",
            source_audio_sha256="a" * 64,
            input_view_id="crux-oaf-full-mix-mono44k1-pcm16/v1",
            max_input_audio_frames=1,
        )


def test_derived_audio_rehashes_source_and_input(tmp_path: Path) -> None:
    source = tmp_path / "source.flac"
    canonical = tmp_path / "canonical.wav"
    source.write_bytes(b"source bytes")
    canonical.write_bytes(canonical_wav_bytes(sample_frames=882))
    manifest = write_input_view_manifest(tmp_path, source, canonical)

    audio = load_derived_audio(canonical, manifest, max_input_audio_frames=882)

    assert audio.path == canonical.resolve()
    assert audio.source_audio_id == "song-42-source-v1"
    assert audio.input_view_id == "full-mix-canonical-wav-v1"
    assert audio.source_audio_sha256 == sha256(source.read_bytes()).hexdigest()
    assert audio.input_audio_sha256 == sha256(canonical.read_bytes()).hexdigest()
    assert audio.audio_frame_count == 882


def test_input_view_manifest_is_an_immutable_exact_record() -> None:
    manifest = InputViewManifest(
        schema=MANIFEST_SCHEMA,
        source_audio_id="source-v1",
        source_audio_sha256="a" * 64,
        source_path="source.flac",
        input_view_id="view-v1",
        input_audio_sha256="b" * 64,
        input_audio_path="canonical.wav",
    )

    assert tuple(field.name for field in fields(manifest)) == (
        "schema",
        "source_audio_id",
        "source_audio_sha256",
        "source_path",
        "input_view_id",
        "input_audio_sha256",
        "input_audio_path",
    )
    with pytest.raises(FrozenInstanceError):
        manifest.source_audio_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "RIFF"),
        (b"RIF", "RIFF"),
        (b"NOPE" + canonical_wav_bytes()[4:], "RIFF"),
        (canonical_wav_bytes()[:8] + b"NOPE" + canonical_wav_bytes()[12:], "WAVE"),
    ],
)
def test_canonical_wav_rejects_malformed_riff_headers(content: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_canonical_wav(content, max_input_audio_frames=None)


def test_canonical_wav_rejects_riff_size_mismatch() -> None:
    content = canonical_wav_bytes()

    for declared_size in (len(content) - 9, len(content) - 7):
        with pytest.raises(ValueError, match="RIFF size"):
            parse_canonical_wav(rewrite_u32(content, 4, declared_size), None)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            rewrite_u32(canonical_wav_bytes()[:12], 4, 4),
            "fmt",
        ),
        (
            rewrite_u32(canonical_wav_bytes()[:20], 4, 12),
            "fmt",
        ),
        (
            canonical_wav_bytes()[:12] + b"JUNK" + canonical_wav_bytes()[16:],
            "fmt",
        ),
        (
            rewrite_u32(canonical_wav_bytes(), 16, 15),
            "fmt",
        ),
        (
            rewrite_u32(canonical_wav_bytes(), 16, 17),
            "fmt",
        ),
        (
            canonical_wav_bytes()[:36] + b"JUNK" + canonical_wav_bytes()[40:],
            "data",
        ),
        (
            rewrite_u32(canonical_wav_bytes()[:36], 4, 28),
            "data",
        ),
    ],
)
def test_canonical_wav_requires_exact_fmt_then_data_chunks(
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_canonical_wav(content, max_input_audio_frames=None)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (canonical_wav_bytes(audio_format=3), "PCM"),
        (canonical_wav_bytes(channel_count=2), "mono"),
        (canonical_wav_bytes(sample_rate=48000), "44100"),
        (canonical_wav_bytes(byte_rate=1), "byte rate"),
        (canonical_wav_bytes(block_alignment=1), "block alignment"),
        (canonical_wav_bytes(bits_per_sample=24), "16-bit"),
    ],
)
def test_canonical_wav_rejects_noncanonical_pcm_fields(content: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_canonical_wav(content, max_input_audio_frames=None)


def test_canonical_wav_rejects_zero_frames() -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_canonical_wav(canonical_wav_bytes(sample_frames=0), None)


def test_canonical_wav_rejects_odd_data_length() -> None:
    content = canonical_wav_bytes()
    odd_content = content[:-1]
    odd_content = rewrite_u32(odd_content, 4, len(odd_content) - 8)
    odd_content = rewrite_u32(odd_content, 40, 1)

    with pytest.raises(ValueError, match="even"):
        parse_canonical_wav(odd_content, None)


def test_canonical_wav_rejects_truncated_data() -> None:
    content = canonical_wav_bytes(sample_frames=2)
    content = rewrite_u32(content[:-2], 4, len(content) - 10)

    with pytest.raises(ValueError, match="data length"):
        parse_canonical_wav(content, None)


def test_canonical_wav_rejects_trailing_bytes_even_if_riff_size_includes_them() -> None:
    content = canonical_wav_bytes() + b"JUNK"
    content = rewrite_u32(content, 4, len(content) - 8)

    with pytest.raises(ValueError, match="trailing"):
        parse_canonical_wav(content, None)


def test_canonical_wav_rejects_extra_chunk_between_fmt_and_data() -> None:
    content = canonical_wav_bytes()
    extra = b"JUNK\0\0\0\0"
    content = content[:36] + extra + content[36:]
    content = rewrite_u32(content, 4, len(content) - 8)

    with pytest.raises(ValueError, match="data"):
        parse_canonical_wav(content, None)


@pytest.mark.parametrize(
    ("sample_frames", "max_frames", "accepted"),
    [
        (2, 3, True),
        (3, 3, True),
        (4, 3, False),
    ],
)
def test_canonical_wav_enforces_frame_bound_edges(
    sample_frames: int,
    max_frames: int,
    accepted: bool,
) -> None:
    content = canonical_wav_bytes(sample_frames=sample_frames)

    if accepted:
        assert parse_canonical_wav(content, max_frames).audio_frame_count == sample_frames
    else:
        with pytest.raises(ValueError, match="frame"):
            parse_canonical_wav(content, max_frames)


def test_canonical_wav_allows_null_adapter_bound() -> None:
    metadata = parse_canonical_wav(canonical_wav_bytes(sample_frames=4), None)

    assert metadata.audio_frame_count == 4


@pytest.mark.parametrize("bound", [True, False, 1.0, "1"])
def test_canonical_wav_rejects_bool_and_non_integer_frame_bounds(bound: object) -> None:
    with pytest.raises(ValueError, match="integer or null"):
        parse_canonical_wav(
            canonical_wav_bytes(),
            bound,  # type: ignore[arg-type]
        )


def test_null_adapter_bound_keeps_all_other_wav_checks() -> None:
    with pytest.raises(ValueError, match="mono"):
        parse_canonical_wav(canonical_wav_bytes(channel_count=2), None)


def test_direct_audio_rejects_audio_over_backend_bound(tmp_path: Path) -> None:
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(canonical_wav_bytes(sample_frames=4))

    with pytest.raises(ValueError, match="frame"):
        load_direct_audio(
            audio_path,
            source_audio_id="source-v1",
            input_view_id="view-v1",
            max_input_audio_frames=3,
        )


@pytest.mark.parametrize("field", ["source_path", "input_audio_path"])
def test_derived_audio_rejects_absolute_manifest_paths(tmp_path: Path, field: str) -> None:
    source = tmp_path / "source.flac"
    canonical = tmp_path / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    payload = manifest_payload(tmp_path, source, canonical)
    payload[field] = str((tmp_path / payload[field]).resolve())
    manifest = write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match="relative"):
        load_derived_audio(canonical, manifest, max_input_audio_frames=1)


@pytest.mark.parametrize("field", ["source_path", "input_audio_path"])
def test_derived_audio_rejects_parent_manifest_paths(tmp_path: Path, field: str) -> None:
    root = tmp_path / "view"
    root.mkdir()
    source = root / "source.flac"
    canonical = root / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    payload = manifest_payload(root, source, canonical)
    payload[field] = "../outside.wav"
    manifest = write_manifest(root, payload)

    with pytest.raises(ValueError, match="parent"):
        load_derived_audio(canonical, manifest, max_input_audio_frames=1)


@pytest.mark.parametrize("field", ["source_path", "input_audio_path"])
def test_derived_audio_rejects_symlink_escape(tmp_path: Path, field: str) -> None:
    root = tmp_path / "view"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    source = root / "source.flac"
    canonical = root / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    escaped_source = outside / "source.flac"
    escaped_input = outside / "canonical.wav"
    escaped_source.write_bytes(b"outside source")
    escaped_input.write_bytes(canonical_wav_bytes())
    (root / "escape").symlink_to(outside, target_is_directory=True)
    payload = manifest_payload(root, source, canonical)
    payload[field] = "escape/" + ("source.flac" if field == "source_path" else "canonical.wav")
    manifest = write_manifest(root, payload)
    audio_path = escaped_input if field == "input_audio_path" else canonical

    with pytest.raises(ValueError, match="escape"):
        load_derived_audio(audio_path, manifest, max_input_audio_frames=1)


@pytest.mark.parametrize("field", ["source_path", "input_audio_path"])
def test_derived_audio_rejects_leaf_symlink_escape(tmp_path: Path, field: str) -> None:
    root = tmp_path / "view"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    source = root / "source.flac"
    canonical = root / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    escaped_source = outside / "source.flac"
    escaped_input = outside / "canonical.wav"
    escaped_source.write_bytes(b"outside source")
    escaped_input.write_bytes(canonical_wav_bytes())
    link = root / ("source-link.flac" if field == "source_path" else "input-link.wav")
    link.symlink_to(escaped_source if field == "source_path" else escaped_input)
    payload = manifest_payload(root, source, canonical)
    payload[field] = link.name
    manifest = write_manifest(root, payload)
    audio_path = escaped_input if field == "input_audio_path" else canonical

    with pytest.raises(ValueError, match="escape"):
        load_derived_audio(audio_path, manifest, max_input_audio_frames=1)


def test_derived_audio_accepts_symlink_that_remains_below_manifest_root(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    source = artifacts / "source.flac"
    canonical = artifacts / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    (tmp_path / "safe").symlink_to(artifacts, target_is_directory=True)
    payload = manifest_payload(tmp_path, source, canonical)
    payload["source_path"] = "safe/source.flac"
    payload["input_audio_path"] = "safe/canonical.wav"
    manifest = write_manifest(tmp_path, payload)

    audio = load_derived_audio(canonical, manifest, max_input_audio_frames=1)

    assert audio.source_audio_sha256 == sha256(b"source").hexdigest()


def test_derived_audio_accepts_leaf_symlinks_that_remain_below_manifest_root(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    source = artifacts / "source.flac"
    canonical = artifacts / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    source_link = tmp_path / "source-link.flac"
    input_link = tmp_path / "input-link.wav"
    source_link.symlink_to(source)
    input_link.symlink_to(canonical)
    payload = manifest_payload(tmp_path, source, canonical)
    payload["source_path"] = source_link.name
    payload["input_audio_path"] = input_link.name
    manifest = write_manifest(tmp_path, payload)

    audio = load_derived_audio(canonical, manifest, max_input_audio_frames=1)

    assert audio.source_audio_sha256 == sha256(b"source").hexdigest()
    assert audio.input_audio_sha256 == sha256(canonical.read_bytes()).hexdigest()


def test_derived_audio_rejects_leaf_swap_before_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "view"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    source = root / "source.flac"
    canonical = root / "canonical.wav"
    escaped_source = outside / "source.flac"
    source.write_bytes(b"inside source")
    canonical.write_bytes(canonical_wav_bytes())
    escaped_source.write_bytes(b"outside source")
    payload = manifest_payload(root, source, canonical)
    payload["source_audio_sha256"] = sha256(escaped_source.read_bytes()).hexdigest()
    manifest = write_manifest(root, payload)

    def swap_source() -> None:
        source.unlink()
        source.symlink_to(escaped_source)

    install_swap_before_artifact_open(
        monkeypatch,
        target_path=source,
        relative_open_name=source.name,
        swap=swap_source,
    )

    with pytest.raises(ValueError, match="source_path.*in-root regular artifact"):
        load_derived_audio(canonical, manifest, max_input_audio_frames=1)


def test_derived_audio_rejects_directory_swap_before_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "view"
    artifacts = root / "artifacts"
    outside = tmp_path / "outside"
    root.mkdir()
    artifacts.mkdir()
    outside.mkdir()
    source = artifacts / "source.flac"
    canonical = artifacts / "canonical.wav"
    escaped_input = outside / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    escaped_input.write_bytes(canonical_wav_bytes(sample_frames=2))
    payload = manifest_payload(root, source, canonical)
    payload["input_audio_sha256"] = sha256(escaped_input.read_bytes()).hexdigest()
    manifest = write_manifest(root, payload)

    def swap_artifacts() -> None:
        artifacts.rename(root / "original-artifacts")
        artifacts.symlink_to(outside, target_is_directory=True)

    install_swap_before_artifact_open(
        monkeypatch,
        target_path=canonical,
        relative_open_name=artifacts.name,
        swap=swap_artifacts,
        relative_open_occurrence=2,
    )

    with pytest.raises(ValueError, match="input_audio_path.*in-root regular artifact"):
        load_derived_audio(canonical, manifest, max_input_audio_frames=2)


def test_derived_audio_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    source = tmp_path / "source.flac"
    canonical = tmp_path / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    payload = manifest_payload(tmp_path, source, canonical)
    manifest = tmp_path / "input-view.json"
    manifest.write_text(
        json.dumps(payload)[:-1] + ',"schema":"crux.input-view-manifest/v1"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_derived_audio(canonical, manifest, max_input_audio_frames=1)


@pytest.mark.parametrize(
    "payload_change",
    [
        {"unknown": "field"},
        {"source_audio_id": None},
    ],
)
def test_derived_audio_rejects_unknown_or_non_string_manifest_fields(
    tmp_path: Path,
    payload_change: dict[str, object],
) -> None:
    source = tmp_path / "source.flac"
    canonical = tmp_path / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    payload: dict[str, object] = manifest_payload(tmp_path, source, canonical)
    payload.update(payload_change)
    manifest = write_manifest(tmp_path, payload)

    with pytest.raises(ValueError):
        load_derived_audio(canonical, manifest, max_input_audio_frames=1)


def test_derived_audio_rejects_missing_manifest_key(tmp_path: Path) -> None:
    source = tmp_path / "source.flac"
    canonical = tmp_path / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    payload = manifest_payload(tmp_path, source, canonical)
    del payload["input_view_id"]
    manifest = write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match="exact"):
        load_derived_audio(canonical, manifest, max_input_audio_frames=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "crux.input-view-manifest/v2"),
        ("source_audio_id", ""),
        ("input_view_id", ""),
        ("source_audio_sha256", "A" * 64),
        ("input_audio_sha256", "not-a-hash"),
    ],
)
def test_derived_audio_rejects_invalid_manifest_identities(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    source = tmp_path / "source.flac"
    canonical = tmp_path / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    payload = manifest_payload(tmp_path, source, canonical)
    payload[field] = value
    manifest = write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match=field):
        load_derived_audio(canonical, manifest, max_input_audio_frames=1)


@pytest.mark.parametrize("field", ["source_audio_sha256", "input_audio_sha256"])
def test_derived_audio_independently_rejects_each_hash_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    source = tmp_path / "source.flac"
    canonical = tmp_path / "canonical.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    payload = manifest_payload(tmp_path, source, canonical)
    payload[field] = "0" * 64
    manifest = write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match=field):
        load_derived_audio(canonical, manifest, max_input_audio_frames=1)


def test_derived_audio_requires_audio_path_to_match_manifest_canonical_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.flac"
    canonical = tmp_path / "canonical.wav"
    other = tmp_path / "other.wav"
    source.write_bytes(b"source")
    canonical.write_bytes(canonical_wav_bytes())
    other.write_bytes(canonical_wav_bytes())
    manifest = write_input_view_manifest(tmp_path, source, canonical)

    with pytest.raises(ValueError, match="audio_path"):
        load_derived_audio(other, manifest, max_input_audio_frames=1)
