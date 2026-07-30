from __future__ import annotations

import hashlib
import importlib.util
import math
import shutil
import stat
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from runtime.oaf_tf1 import oaf_backend
from runtime.oaf_tf1.apply_instrumentation_patch import (
    InstrumentationPatchError,
    apply_reviewed_patch,
)
from runtime.oaf_tf1.oaf_backend import (
    AdapterItemFailure,
    frame_time_seconds,
    native_event_from_capture,
    velocity_to_midi,
)
from tools.hpa320 import generate_runner_source_manifest as runner_manifest
from tools.hpa320.generate_runner_source_manifest import (
    build_runner_source_manifest,
    write_runner_source_manifest,
)

_IMAGE_RUNTIME_ROOT = Path("/opt/crux/runtime/oaf_tf1")
RUNTIME_ROOT = (
    _IMAGE_RUNTIME_ROOT if _IMAGE_RUNTIME_ROOT.is_dir() else Path(__file__).resolve().parents[1]
)
REPOSITORY_ROOT = RUNTIME_ROOT.parents[1]
UPSTREAM_ROOT = (
    Path("/opt/crux/upstream") if Path("/opt/crux/upstream").is_dir() else RUNTIME_ROOT / "vendor"
)
PATCH_PATH = RUNTIME_ROOT / "patches" / "capture-emitted-frame.patch"


def _load_source_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create a source loader for {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _isolated_magenta_imports(patched_vendor: Path) -> Iterator[None]:
    module_names = (
        "magenta",
        "magenta.models",
        "magenta.models.onsets_frames_transcription",
        "magenta.models.onsets_frames_transcription.data",
        "magenta.music",
        "magenta.music.chord_symbols_lib",
        "magenta.music.constants",
        "magenta.music.protobuf",
        "magenta.music.protobuf.music_pb2",
        "magenta.music.sequences_lib",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    try:
        magenta = types.ModuleType("magenta")
        magenta.__path__ = [str(patched_vendor / "magenta")]
        music = types.ModuleType("magenta.music")
        music.__path__ = [str(patched_vendor / "magenta" / "music")]
        protobuf = types.ModuleType("magenta.music.protobuf")
        protobuf.__path__ = [str(patched_vendor / "magenta" / "music" / "protobuf")]
        models = types.ModuleType("magenta.models")
        models.__path__ = [str(patched_vendor / "magenta" / "models")]
        transcription = types.ModuleType("magenta.models.onsets_frames_transcription")
        transcription.__path__ = [
            str(patched_vendor / "magenta" / "models" / "onsets_frames_transcription")
        ]
        chord_symbols = types.ModuleType("magenta.music.chord_symbols_lib")
        data = types.ModuleType("magenta.models.onsets_frames_transcription.data")

        sys.modules.update(
            {
                "magenta": magenta,
                "magenta.models": models,
                "magenta.models.onsets_frames_transcription": transcription,
                "magenta.models.onsets_frames_transcription.data": data,
                "magenta.music": music,
                "magenta.music.chord_symbols_lib": chord_symbols,
                "magenta.music.protobuf": protobuf,
            }
        )
        magenta.models = models
        magenta.music = music
        models.onsets_frames_transcription = transcription
        transcription.data = data
        music.chord_symbols_lib = chord_symbols
        music.protobuf = protobuf

        constants = _load_source_module(
            "magenta.music.constants",
            patched_vendor / "magenta" / "music" / "constants.py",
        )
        music.constants = constants
        music_pb2 = _load_source_module(
            "magenta.music.protobuf.music_pb2",
            patched_vendor / "magenta" / "music" / "protobuf" / "music_pb2.py",
        )
        protobuf.music_pb2 = music_pb2
        yield
    finally:
        for name in reversed(module_names):
            prior = previous[name]
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_patch_applies_only_to_private_copy_and_emits_deterministic_manifest(
    tmp_path: Path,
) -> None:
    before = _file_hashes(UPSTREAM_ROOT)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = apply_reviewed_patch(UPSTREAM_ROOT, first, PATCH_PATH)
    second_manifest = apply_reviewed_patch(UPSTREAM_ROOT, second, PATCH_PATH)

    assert _file_hashes(UPSTREAM_ROOT) == before
    assert first_manifest == second_manifest
    assert first_manifest["schema"] == "crux.oaf-instrumented-source-manifest/v1"
    assert first_manifest["patch_sha256"] == hashlib.sha256(PATCH_PATH.read_bytes()).hexdigest()
    changed = [
        entry["path"]
        for entry in first_manifest["files"]
        if before[entry["path"]] != entry["sha256"]
    ]
    assert changed == [
        "magenta/models/onsets_frames_transcription/infer_util.py",
        "magenta/music/sequences_lib.py",
    ]
    for relative_path in changed:
        assert stat.S_IMODE((first / relative_path).stat().st_mode) == stat.S_IMODE(
            (UPSTREAM_ROOT / relative_path).stat().st_mode
        )


def test_patch_applier_rejects_unreviewed_patch_before_creating_destination(
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "tampered.patch"
    tampered.write_bytes(PATCH_PATH.read_bytes() + b"\n")
    destination = tmp_path / "output"

    with pytest.raises(InstrumentationPatchError, match="patch identity"):
        apply_reviewed_patch(UPSTREAM_ROOT, destination, tampered)

    assert not destination.exists()


def test_patch_applier_rejects_symlinked_source_and_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(UPSTREAM_ROOT, source)
    (source / "magenta" / "music" / "sequences_lib.py").unlink()
    (source / "magenta" / "music" / "sequences_lib.py").symlink_to(
        UPSTREAM_ROOT / "magenta" / "music" / "sequences_lib.py"
    )

    with pytest.raises(InstrumentationPatchError, match="regular no-follow"):
        apply_reviewed_patch(source, tmp_path / "output", PATCH_PATH)

    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(InstrumentationPatchError, match="destination"):
        apply_reviewed_patch(UPSTREAM_ROOT, destination, PATCH_PATH)


def test_patch_applier_rejects_preimage_drift_and_crlf_changes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(UPSTREAM_ROOT, source)
    target = source / "magenta" / "music" / "sequences_lib.py"
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(InstrumentationPatchError, match="preimage"):
        apply_reviewed_patch(source, tmp_path / "output", PATCH_PATH)


def test_runner_source_manifest_is_deterministic_and_excludes_itself() -> None:
    if not (REPOSITORY_ROOT / "runtime/oaf_tf1/Dockerfile").is_file():
        pytest.skip("repository-layout manifest generation is a host-side check")
    first = build_runner_source_manifest(REPOSITORY_ROOT)
    second = build_runner_source_manifest(REPOSITORY_ROOT)

    assert first == second
    assert first["schema"] == "crux.oaf-runner-source-manifest/v1"
    paths = [entry["path"] for entry in first["files"]]
    assert paths == sorted(paths, key=lambda path: path.encode("utf-8"))
    assert "runtime/oaf_tf1/runner-source-manifest.json" not in paths
    assert "runtime/oaf_tf1/Dockerfile" in paths
    assert "runtime/oaf_tf1/apply_instrumentation_patch.py" in paths
    assert "runtime/oaf_tf1/patches/capture-emitted-frame.patch" in paths
    assert "tools/hpa320/generate_runner_source_manifest.py" in paths


def test_written_runner_source_manifest_is_world_readable(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    source = repository / "runtime" / "oaf_tf1" / "Dockerfile"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"FROM scratch\n")
    monkeypatch.setattr(
        runner_manifest,
        "SOURCE_PATHS",
        ("runtime/oaf_tf1/Dockerfile",),
    )
    output = tmp_path / "runner-source-manifest.json"

    write_runner_source_manifest(repository, output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o644


def test_frame_time_uses_locked_binary64_evaluation_order() -> None:
    assert frame_time_seconds(15).hex() == "0x1.64a893adcd260p-3"
    assert frame_time_seconds(15) != 15 * 512 / 44100


@pytest.mark.parametrize(
    ("raw_velocity", "expected"),
    [
        (-0.25, 0),
        (0.0, 0),
        (0.499, 63),
        (1.0, 127),
        (1.25, 127),
    ],
)
def test_velocity_clamps_multiplies_and_truncates(raw_velocity: float, expected: int) -> None:
    assert velocity_to_midi(raw_velocity) == expected


@pytest.mark.parametrize("raw_velocity", [math.nan, math.inf, -math.inf])
def test_velocity_rejects_nonfinite_values_before_clamping(raw_velocity: float) -> None:
    with pytest.raises(AdapterItemFailure) as error:
        velocity_to_midi(raw_velocity)

    assert error.value.code == "nonfinite_velocity"


def test_native_event_preserves_frame_bin_pitch_group_confidence_and_velocity() -> None:
    event = native_event_from_capture(
        start_frame=15,
        native_midi_note=38,
        raw_velocity=0.499,
        raw_confidence=0.625,
        training_groups=(
            {
                "base_midi": 38,
                "group_id": "snare",
                "member_pitches": [38, 40, 37, 39],
                "output_bin": 17,
            },
        ),
    )

    assert event == {
        "confidence_binary64": "3fe4000000000000",
        "frame_index": 15,
        "model_output_bin": 17,
        "native_class_id": "midi_38",
        "native_midi_note": 38,
        "time_sec_binary64": "3fc64a893adcd260",
        "upstream_8hit_group_id": "snare",
        "velocity_midi": 63,
    }


def test_instrumented_conversion_is_byte_identical_and_captures_emission_metadata() -> None:
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("tensorflow")
    patched_vendor = Path("/opt/crux/vendor")
    if not patched_vendor.is_dir():
        pytest.skip("instrumented image-build copy is available only in the runtime image")
    with _isolated_magenta_imports(patched_vendor):
        patched_sequences_lib = _load_source_module(
            "magenta.music.sequences_lib",
            patched_vendor / "magenta" / "music" / "sequences_lib.py",
        )
        upstream_sequences_lib = oaf_backend.load_uninstrumented_sequences_module()

        onsets = numpy.zeros((4, 88), dtype=numpy.bool_)
        velocities = numpy.zeros((4, 88), dtype=numpy.float32)
        onsets[1, 15] = True
        onsets[3, 17] = True
        velocities[1, 15] = numpy.float32(0.25)
        velocities[3, 17] = numpy.float32(0.75)

        original = upstream_sequences_lib.pianoroll_onsets_to_note_sequence(
            onsets=onsets,
            frames_per_second=44100 / 512,
            min_midi_pitch=21,
            velocity_values=velocities,
            velocity_scale=127,
            velocity_bias=0,
        )
        instrumented, captured = patched_sequences_lib.pianoroll_onsets_to_note_sequence(
            onsets=onsets,
            frames_per_second=44100 / 512,
            min_midi_pitch=21,
            velocity_values=velocities,
            velocity_scale=127,
            velocity_bias=0,
            capture_emitted_frames=True,
        )

        assert Path("/opt/crux/upstream") in Path(upstream_sequences_lib.__file__).resolve().parents
        assert patched_vendor in Path(patched_sequences_lib.__file__).resolve().parents
        assert instrumented.SerializeToString() == original.SerializeToString()
        assert captured == ((1, 36, numpy.float32(0.25)), (3, 38, numpy.float32(0.75)))


def test_infer_util_pairs_confidence_from_selected_onset_cell() -> None:
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("tensorflow")
    patched_vendor = Path("/opt/crux/vendor")
    if not patched_vendor.is_dir():
        pytest.skip("instrumented image-build copy is available only in the runtime image")
    with _isolated_magenta_imports(patched_vendor):
        infer_util = _load_source_module(
            "magenta.models.onsets_frames_transcription.infer_util",
            patched_vendor / "magenta" / "models" / "onsets_frames_transcription" / "infer_util.py",
        )

        captures = ((1, 36, numpy.float32(0.25)), (3, 38, numpy.float32(0.75)))
        onset_probs = numpy.zeros((4, 88), dtype=numpy.float32)
        onset_probs[1, 15] = numpy.float32(0.625)
        onset_probs[3, 17] = numpy.float32(0.875)

        assert infer_util.pair_emitted_frame_confidence(captures, onset_probs, min_pitch=21) == (
            (1, 36, numpy.float32(0.25), numpy.float32(0.625)),
            (3, 38, numpy.float32(0.75), numpy.float32(0.875)),
        )
