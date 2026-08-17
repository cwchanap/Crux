from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    OAF_DESCRIPTOR_SCHEMA,
    StrictJsonError,
    build_descriptor,
)
from src.benchmark.backends import CanonicalAudio, NativeEvent
from src.benchmark.prediction_artifact import (
    MappedPrediction,
    MappedPredictionEvent,
    PredictionArtifactError,
    prediction_path,
    read_prediction_artifact,
    render_prediction_artifact,
)

OAF_PREDICTION_MAP_ID = "crux.prediction-map/oaf-drums-v1"

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _oaf_descriptor():
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


def _oaf_native_event() -> NativeEvent:
    return NativeEvent(
        time_sec=0.5,
        native_class_id="midi_46",
        model_output_bin=25,
        native_midi_note=46,
        native_metadata={"upstream_8hit_group_id": "hihat"},
        confidence=0.75,
        velocity_midi=96,
    )


def _oaf_prediction(native: NativeEvent | None = None) -> MappedPrediction:
    return MappedPrediction(
        audio=CanonicalAudio(
            path=Path(),
            source_audio_id="song",
            source_audio_sha256="f" * 64,
            input_view_id="full-mix-v1",
            input_audio_sha256="b" * 64,
            byte_length=46,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=1,
        ),
        descriptor=_oaf_descriptor(),
        events=(
            MappedPredictionEvent(
                native=_oaf_native_event() if native is None else native,
                canonical_class="hihat",
                common_class="hihat",
                mapping_status="mapped",
                prediction_map_version=OAF_PREDICTION_MAP_ID,
            ),
        ),
    )


def test_oaf_prediction_round_trip_is_byte_identical() -> None:
    content = render_prediction_artifact(_oaf_prediction())

    round_tripped = read_prediction_artifact(content)

    assert render_prediction_artifact(round_tripped.prediction) == content


@pytest.mark.parametrize(
    "change",
    [
        {"model_output_bin": None},
        {"native_midi_note": None},
        {"confidence": None},
        {"velocity_midi": None},
    ],
)
def test_oaf_prediction_rejects_nullability_violation(change: dict[str, object]) -> None:
    native = replace(_oaf_native_event(), **change)

    with pytest.raises(PredictionArtifactError, match="oaf_event_nullability"):
        render_prediction_artifact(_oaf_prediction(native))


def test_oaf_prediction_rejects_native_midi_note_identity_mismatch() -> None:
    native = replace(_oaf_native_event(), native_midi_note=47)

    with pytest.raises(PredictionArtifactError, match="oaf_native_identity"):
        render_prediction_artifact(_oaf_prediction(native))


def test_oaf_prediction_rejects_native_class_id_identity_mismatch() -> None:
    native = replace(_oaf_native_event(), native_class_id="midi_47")

    with pytest.raises(PredictionArtifactError, match="oaf_native_identity"):
        render_prediction_artifact(_oaf_prediction(native))


def test_prediction_path_rejects_non_str_source_audio_sha256() -> None:
    with pytest.raises(StrictJsonError, match="source_audio_sha256"):
        prediction_path(
            Path("/tmp"),
            simfile_id=1,
            source_audio_sha256=None,
            backend_descriptor_sha256=SHA_B,
            inference_config_sha256=SHA_C,
        )


def test_prediction_path_rejects_non_path_output_dir() -> None:
    with pytest.raises(TypeError, match="output_dir must be a Path"):
        prediction_path(
            "/tmp",
            simfile_id=1,
            source_audio_sha256=SHA_A,
            backend_descriptor_sha256=SHA_B,
            inference_config_sha256=SHA_C,
        )


def test_prediction_path_rejects_non_positive_simfile_id() -> None:
    with pytest.raises(ValueError, match="simfile_id must be a positive integer"):
        prediction_path(
            Path("/tmp"),
            simfile_id=0,
            source_audio_sha256=SHA_A,
            backend_descriptor_sha256=SHA_B,
            inference_config_sha256=SHA_C,
        )
