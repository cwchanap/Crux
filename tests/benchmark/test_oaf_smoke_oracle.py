from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import pytest

from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    OAF_DESCRIPTOR_SCHEMA,
    build_descriptor,
    sha256_hex,
)
from src.benchmark.backends import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.oaf_smoke_oracle import (
    SmokeOracleEvent,
    assert_smoke_oracle_matches,
    read_smoke_oracle,
    render_smoke_oracle,
)


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


def _prediction(tmp_path: Path) -> object:
    audio = _audio(tmp_path)
    native = (
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
            confidence=None,
            velocity_midi=None,
        ),
    )
    prediction = NativePrediction(audio, _descriptor(), native)
    return map_oaf_prediction(prediction)[0]


def test_render_read_oracle_round_trips_retained_native_events(tmp_path: Path) -> None:
    prediction = _prediction(tmp_path)

    content = render_smoke_oracle(prediction)
    oracle = read_smoke_oracle(content)

    assert oracle.schema == "crux.oaf-smoke-oracle/v2"
    assert oracle.backend_id == OAF_BACKEND_ID
    assert oracle.input_audio_sha256 == prediction.audio.input_audio_sha256
    assert oracle.native_events[0].time_sec_binary64 == struct.pack(">d", 0.5).hex()
    assert oracle.native_events[0].confidence_binary64 == struct.pack(">d", 0.75).hex()
    assert oracle.native_events[1].confidence_binary64 is None
    assert_smoke_oracle_matches(prediction, oracle)


def test_oracle_compare_rejects_every_changed_event_field(tmp_path: Path) -> None:
    prediction = _prediction(tmp_path)
    oracle = read_smoke_oracle(render_smoke_oracle(prediction))
    event = oracle.native_events[0]
    fields = (
        "time_sec_binary64",
        "native_class_id",
        "model_output_bin",
        "native_midi_note",
        "upstream_8hit_group_id",
        "confidence_binary64",
        "velocity_midi",
    )

    for field in fields:
        value = getattr(event, field)
        if isinstance(value, str):
            replacement = ("0" if value[0] != "0" else "1") + value[1:]
        elif value is None:
            replacement = 0
        else:
            replacement = value + 1
        changed = replace(event, **{field: replacement})
        mismatching = replace(oracle, native_events=(changed, *oracle.native_events[1:]))
        with pytest.raises(AssertionError):
            assert_smoke_oracle_matches(prediction, mismatching)


def test_oracle_golden_has_exact_schema_and_event_keys(tmp_path: Path) -> None:
    prediction = _prediction(tmp_path)
    content = render_smoke_oracle(prediction)

    assert content.endswith(b"\n")
    document = read_smoke_oracle(content)
    assert document.native_events[0].native_class_id == "midi_46"


def test_oracle_compare_rejects_backend_or_input_identity_mismatch(tmp_path: Path) -> None:
    prediction = _prediction(tmp_path)
    oracle = read_smoke_oracle(render_smoke_oracle(prediction))

    with pytest.raises(AssertionError):
        assert_smoke_oracle_matches(prediction, replace(oracle, backend_id="other"))
    with pytest.raises(AssertionError):
        assert_smoke_oracle_matches(prediction, replace(oracle, input_audio_sha256="0" * 64))


def test_checked_in_smoke_oracle_matches_run5_native_bit_patterns() -> None:
    fixture_path = Path(__file__).parents[2] / "tests/fixtures/oaf_tf1_smoke/smoke-oracle.json"

    oracle = read_smoke_oracle(fixture_path.read_bytes())

    assert oracle.schema == "crux.oaf-smoke-oracle/v2"
    assert oracle.backend_id == OAF_BACKEND_ID
    assert oracle.input_audio_sha256 == (
        "8ec2aed65945b7002e17b51818495ca754a519c39940f739cdcd1403eb661673"
    )
    assert oracle.native_events == (
        SmokeOracleEvent(
            time_sec_binary64="3fb7c6fbd273d5bb",
            native_class_id="midi_38",
            model_output_bin=17,
            native_midi_note=38,
            upstream_8hit_group_id="snare",
            confidence_binary64="3fe3a19c9d5a187a",
            velocity_midi=116,
        ),
        SmokeOracleEvent(
            time_sec_binary64="3fd943675ddd2ae9",
            native_class_id="midi_38",
            model_output_bin=17,
            native_midi_note=38,
            upstream_8hit_group_id="snare",
            confidence_binary64="3feb05caafbc1ace",
            velocity_midi=118,
        ),
        SmokeOracleEvent(
            time_sec_binary64="3fe64a89fc6da448",
            native_class_id="midi_38",
            model_output_bin=17,
            native_midi_note=38,
            upstream_8hit_group_id="snare",
            confidence_binary64="3feb67ddca4b124d",
            velocity_midi=107,
        ),
    )
