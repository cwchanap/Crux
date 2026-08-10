from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from click.testing import CliRunner

from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    OAF_DESCRIPTOR_SCHEMA,
    build_descriptor,
    sha256_hex,
)
from src.benchmark.backends import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.oaf_smoke_oracle import render_smoke_oracle
from src.cli.main import main


def _descriptor() -> object:
    payload = {
        "architecture_id": "magenta-oaf-model-tpu-drums-v1",
        "backend_id": OAF_BACKEND_ID,
        "descriptor_schema": OAF_DESCRIPTOR_SCHEMA,
        "model_id": "magenta-egmd-ckpt-569400-v1",
        "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
        "native_output_space_id": "magenta-oaf-midi88-a0-v1",
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
        "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
    }
    return build_descriptor(payload, frozenset(payload), OAF_DESCRIPTOR_SCHEMA)


def _audio(tmp_path: Path) -> CanonicalAudio:
    content = (
        struct.pack("<4sI4s", b"RIFF", 40, b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16)
        + struct.pack("<4sI", b"data", 4)
        + b"\x00\x00\x00\x00"
    )
    path = tmp_path / "canonical.wav"
    path.write_bytes(content)
    digest = sha256_hex(content)
    return CanonicalAudio(
        path, "oaf-smoke", digest, "oaf-smoke-v1", digest, len(content), 44100, 1, 2, 2
    )


def _prediction(tmp_path: Path) -> NativePrediction:
    audio = _audio(tmp_path)
    events = (
        NativeEvent(
            time_sec=0.5,
            native_class_id="midi_46",
            model_output_bin=25,
            native_midi_note=46,
            native_metadata={"upstream_8hit_group_id": "hihat"},
            confidence=0.75,
            velocity_midi=96,
        ),
        NativeEvent(
            time_sec=1.25,
            native_class_id="midi_75",
            model_output_bin=54,
            native_midi_note=75,
            native_metadata={"upstream_8hit_group_id": "sticks"},
            confidence=0.5,
            velocity_midi=80,
        ),
    )
    return NativePrediction(audio, _descriptor(), events)


@dataclass
class _FakeBackend:
    prediction: NativePrediction

    def descriptor(self):
        return self.prediction.descriptor

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        return NativePrediction(audio, self.prediction.descriptor, self.prediction.events)

    def close(self) -> None:
        return None


class _FakeRegistry:
    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend

    def create(self, backend_id: str | None = None, **kwargs: object) -> _FakeBackend:
        assert backend_id == "oaf"
        del kwargs
        return self.backend


def test_smoke_backend_times_only_backend_inference_and_publishes_v2(
    tmp_path: Path, monkeypatch
) -> None:
    output_root = tmp_path / "input"
    output_root.mkdir()
    prediction = _prediction(output_root)
    import src.cli.benchmark as benchmark_module

    fake_registry = _FakeRegistry(_FakeBackend(prediction))
    monkeypatch.setattr(benchmark_module, "default_backend_registry", lambda: fake_registry)
    monkeypatch.setattr(
        benchmark_module,
        "load_model_config",
        lambda: type(
            "Config",
            (),
            {
                "max_input_audio_frames": None,
                "checkpoint": type("Checkpoint", (), {"archive_sha256": "a" * 64})(),
            },
        )(),
    )
    monkeypatch.setattr(
        benchmark_module, "load_direct_audio", lambda *args, **kwargs: prediction.audio
    )
    monkeypatch.setattr(benchmark_module.time, "perf_counter", iter((10.0, 12.0)).__next__)

    result = CliRunner().invoke(
        main,
        ["benchmark", "smoke-backend", "--backend", "oaf"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    summary = json.loads(result.output)
    assert summary["status"] == "ok"
    assert summary["mapped_event_count"] == 1
    assert summary["unmapped_event_count"] == 1
    assert summary["inference_elapsed_seconds"] == 2.0
    assert summary["real_time_factor"] == 2.0 / (
        prediction.audio.audio_frame_count / prediction.audio.sample_rate
    )
    assert summary["oracle_status"] == "not_checked"
    prediction_path = Path(summary["prediction_path"])
    assert prediction_path.exists()
    assert b'"schema":"crux.drum-prediction-events/v2"' in prediction_path.read_bytes()


def test_smoke_backend_matches_optional_oracle_and_rejects_missing_or_mismatching(
    tmp_path: Path, monkeypatch
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    prediction = _prediction(input_root)
    oracle_path = tmp_path / "oracle.json"
    oracle_path.write_bytes(render_smoke_oracle(map_oaf_prediction(prediction)[0]))

    import src.cli.benchmark as benchmark_module

    fake_registry = _FakeRegistry(_FakeBackend(prediction))
    monkeypatch.setattr(benchmark_module, "default_backend_registry", lambda: fake_registry)
    monkeypatch.setattr(
        benchmark_module,
        "load_model_config",
        lambda: type(
            "Config",
            (),
            {
                "max_input_audio_frames": None,
                "checkpoint": type("Checkpoint", (), {"archive_sha256": "a" * 64})(),
            },
        )(),
    )
    monkeypatch.setattr(
        benchmark_module, "load_direct_audio", lambda *args, **kwargs: prediction.audio
    )
    monkeypatch.setattr(benchmark_module.time, "perf_counter", iter((10.0, 12.0)).__next__)

    matching = CliRunner().invoke(
        main,
        ["benchmark", "smoke-backend", "--backend", "oaf", "--oracle", str(oracle_path)],
        catch_exceptions=False,
    )
    assert matching.exit_code == 0
    assert json.loads(matching.output)["oracle_status"] == "matched"

    missing = CliRunner().invoke(
        main,
        ["benchmark", "smoke-backend", "--backend", "oaf", "--oracle", str(tmp_path / "missing")],
    )
    assert missing.exit_code != 0

    oracle_path.write_bytes(oracle_path.read_bytes().replace(b"midi_46", b"midi_47"))
    monkeypatch.setattr(benchmark_module.time, "perf_counter", iter((10.0, 12.0)).__next__)
    mismatching = CliRunner().invoke(
        main,
        ["benchmark", "smoke-backend", "--backend", "oaf", "--oracle", str(oracle_path)],
    )
    assert mismatching.exit_code != 0
