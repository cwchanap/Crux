from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.benchmark import backend_identity
from src.benchmark.backend_identity import build_descriptor
from src.benchmark.backends import CanonicalAudio, NativeEvent
from src.benchmark.prediction_artifact import (
    MappedPrediction,
    MappedPredictionEvent,
    PredictionArtifactError,
    prediction_artifact_matches_audio,
    prediction_artifact_matches_run_row,
    prediction_path,
    read_prediction_artifact,
    render_prediction_artifact,
)

MUSCRIPTOR_BACKEND_ID = "muscriptor-v0.3.0-drums-v1"
MUSCRIPTOR_DESCRIPTOR_SCHEMA = "crux.transcription-backend-descriptor/v2"
MUSCRIPTOR_PREDICTION_MAP_ID = "crux.prediction-map/muscriptor-drums-v1"


def _muscriptor_descriptor():
    assert backend_identity.MUSCRIPTOR_BACKEND_ID == MUSCRIPTOR_BACKEND_ID
    assert backend_identity.MUSCRIPTOR_DESCRIPTOR_SCHEMA == MUSCRIPTOR_DESCRIPTOR_SCHEMA
    payload = {
        "architecture_id": "muscriptor-transformer-v0.3.0",
        "backend_id": MUSCRIPTOR_BACKEND_ID,
        "descriptor_schema": MUSCRIPTOR_DESCRIPTOR_SCHEMA,
        "model_id": "muscriptor-medium-0123456789ab-fedcba987654",
        "native_metadata_schema_id": "muscriptor-note-start-metadata-v1",
        "native_output_space_id": "muscriptor-drums-midi128-v1",
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": "muscriptor-training-data-v0.3.0",
        "upstream_source_commit": "d73147e75e5b9b0c0a79ebe154587db4fd603e0c",
    }
    return build_descriptor(payload, frozenset(payload), MUSCRIPTOR_DESCRIPTOR_SCHEMA)


def _muscriptor_native_event() -> NativeEvent:
    return NativeEvent(
        time_sec=1.25,
        native_class_id="drums:midi_38",
        model_output_bin=None,
        native_midi_note=38,
        native_metadata={"instrument_group": "drums"},
        confidence=None,
        velocity_midi=None,
    )


def _muscriptor_prediction(native: NativeEvent | None = None) -> MappedPrediction:
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
        descriptor=_muscriptor_descriptor(),
        events=(
            MappedPredictionEvent(
                native=_muscriptor_native_event() if native is None else native,
                canonical_class="snare",
                common_class="snare",
                mapping_status="mapped",
                prediction_map_version=MUSCRIPTOR_PREDICTION_MAP_ID,
            ),
        ),
    )


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def test_prediction_path_is_source_keyed_and_reference_independent(tmp_path: Path) -> None:
    path = prediction_path(
        tmp_path,
        simfile_id=10,
        source_audio_sha256=SHA_A,
        backend_descriptor_sha256=SHA_B,
        inference_config_sha256=SHA_C,
    )

    assert path == (tmp_path / "predictions" / "10" / SHA_A / SHA_B / f"{SHA_C}.jsonl")
    assert "input_audio_sha256" not in str(path)
    assert "reference_manifest" not in str(path)


def test_muscriptor_prediction_round_trip_is_byte_identical() -> None:
    content = render_prediction_artifact(_muscriptor_prediction())

    round_tripped = read_prediction_artifact(content)

    assert render_prediction_artifact(round_tripped.prediction) == content


def test_prediction_matchers_accept_a_non_full_mix_view_and_explicit_row_policy() -> None:
    prediction = _muscriptor_prediction()
    derived_audio = replace(
        prediction.audio,
        input_view_id="crux.oaf-spleeter4-drums-mono44k1-pcm16/v1",
        input_audio_sha256="c" * 64,
    )
    artifact = read_prediction_artifact(
        render_prediction_artifact(replace(prediction, audio=derived_audio))
    )

    assert prediction_artifact_matches_audio(
        artifact,
        source_audio_id="song",
        source_audio_sha256="f" * 64,
        audio=derived_audio,
        descriptor=prediction.descriptor,
        prediction_map_version=MUSCRIPTOR_PREDICTION_MAP_ID,
    )
    assert not prediction_artifact_matches_audio(
        artifact,
        source_audio_id="song",
        source_audio_sha256="f" * 64,
        audio=replace(derived_audio, input_audio_sha256="d" * 64),
        descriptor=prediction.descriptor,
        prediction_map_version=MUSCRIPTOR_PREDICTION_MAP_ID,
    )
    row = {
        "prediction_artifact_sha256": artifact.artifact_sha256,
        "source_audio_id": "song",
        "source_audio_sha256": "f" * 64,
        "input_view_id": "crux.oaf-spleeter4-drums-mono44k1-pcm16/v1",
        "input_audio_sha256": "c" * 64,
    }
    assert prediction_artifact_matches_run_row(
        artifact,
        row,
        expected_input_view_id="crux.oaf-spleeter4-drums-mono44k1-pcm16/v1",
    )
    assert not prediction_artifact_matches_run_row(
        artifact,
        row,
        expected_input_view_id="full-mix-v1",
    )
    assert not prediction_artifact_matches_run_row(
        artifact,
        {**row, "input_view_id": "full-mix-v1"},
        expected_input_view_id="crux.oaf-spleeter4-drums-mono44k1-pcm16/v1",
    )


@pytest.mark.parametrize(
    "change",
    [
        {"model_output_bin": 17},
        {"native_midi_note": None},
        {"native_midi_note": -1},
        {"native_midi_note": 128},
        {"confidence": 0.5},
        {"velocity_midi": 100},
        {"native_metadata": {"instrument_group": "piano"}},
        {"native_class_id": "drums:midi_36"},
    ],
)
def test_muscriptor_prediction_rejects_native_event_invariant_violations(
    change: dict[str, object],
) -> None:
    native = replace(_muscriptor_native_event(), **change)

    with pytest.raises(PredictionArtifactError):
        render_prediction_artifact(_muscriptor_prediction(native))
