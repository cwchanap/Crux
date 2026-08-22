from dataclasses import replace
from pathlib import Path

import pytest

from src.benchmark.backend_identity import build_descriptor
from src.benchmark.backends import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.mapping import (
    DEFAULT_MIDI_NOTE_MAP,
    DTX_LANE_MAP,
    MUSCRIPTOR_PREDICTION_MAP,
    map_dtx_events,
    map_idm_prediction,
    map_midi_events,
    map_muscriptor_prediction,
)
from src.benchmark.models import BenchmarkEvent
from src.benchmark.prediction_artifact import read_prediction_artifact, render_prediction_artifact
from src.benchmark.taxonomy import (
    IDM_PREDICTION_MAP,
    MUSCRIPTOR_PREDICTION_MAP_ID,
    ClassMapping,
)

IDM_BACKEND_ID = "idm-44-train-kits-v1"
IDM_DESCRIPTOR_SCHEMA = "crux.transcription-backend-descriptor/v2"
IDM_PREDICTION_MAP_ID = "crux.prediction-map/idm-44-train-kits-v1"


def test_default_dtx_mapping_supports_drumery_editor_lanes():
    assert DTX_LANE_MAP["13"].canonical_class == "kick"
    assert DTX_LANE_MAP["12"].canonical_class == "snare"
    assert DTX_LANE_MAP["11"].canonical_class == "closed_hihat"


def test_default_midi_mapping_supports_current_transcriber_notes():
    assert DEFAULT_MIDI_NOTE_MAP[36] == "kick"
    assert DEFAULT_MIDI_NOTE_MAP[38] == "snare"


def test_default_midi_note_47_maps_to_existing_dtx_tom_class():
    """MIDI note 47 must map to a detailed class that DTX_LANE_MAP
    can produce, otherwise mid-tom predictions are guaranteed false-positives."""
    detailed = DEFAULT_MIDI_NOTE_MAP[47]
    dtx_classes = {m.canonical_class for m in DTX_LANE_MAP.values()}
    assert detailed in dtx_classes, f"{detailed!r} not in DTX detailed classes {dtx_classes}"


def test_map_dtx_event_replaces_lane_class_and_stores_common_metadata():
    event = BenchmarkEvent("song", 0.0, "13", "ground_truth", {"lane_id": "13"})

    mapped, diagnostics = map_dtx_events([event])

    assert mapped[0].canonical_class == "kick"
    assert mapped[0].metadata["common_class"] == "kick"
    assert diagnostics.unmapped == {}


def test_map_dtx_events_respects_explicit_empty_lane_map():
    event = BenchmarkEvent("song", 0.0, "13", "ground_truth", {"lane_id": "13"})

    mapped, diagnostics = map_dtx_events([event], lane_map={})

    assert mapped == []
    assert diagnostics.unmapped == {"13": 1}


def test_default_dtx_mapping_ignores_converter_style_non_drum_lanes():
    event = BenchmarkEvent("song", 0.0, "01", "ground_truth", {"lane_id": "01"})

    mapped, diagnostics = map_dtx_events([event])

    assert mapped == []
    assert diagnostics.unmapped == {"01": 1}


def test_map_midi_events_maps_known_note_and_preserves_event_structure():
    event = BenchmarkEvent("song", 1.25, "unknown", "prediction", {"midi_note": 36})

    mapped, diagnostics = map_midi_events([event])

    assert mapped == [BenchmarkEvent("song", 1.25, "kick", "prediction", {"midi_note": 36})]
    assert diagnostics.unmapped == {}


def test_map_midi_events_respects_explicit_empty_note_map():
    event = BenchmarkEvent("song", 0.0, "unknown", "prediction", {"midi_note": 36})

    mapped, diagnostics = map_midi_events([event], note_map={})

    assert mapped == []
    assert diagnostics.unmapped == {"36": 1}


def build_muscriptor_descriptor():
    payload = {
        "architecture_id": "muscriptor-transformer-v0.3.0",
        "backend_id": "muscriptor-v0.3.0-drums-v1",
        "descriptor_schema": "crux.transcription-backend-descriptor/v2",
        "model_id": "muscriptor-medium-0123456789ab-fedcba987654",
        "native_metadata_schema_id": "muscriptor-note-start-metadata-v1",
        "native_output_space_id": "muscriptor-drums-midi128-v1",
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": "muscriptor-training-data-v0.3.0",
        "upstream_source_commit": "d73147e75e5b9b0c0a79ebe154587db4fd603e0c",
    }
    return build_descriptor(payload, frozenset(payload), payload["descriptor_schema"])


def build_idm_descriptor():
    payload = {
        "architecture_id": "inverse-drum-machine-v0.1.0",
        "backend_id": IDM_BACKEND_ID,
        "descriptor_schema": IDM_DESCRIPTOR_SCHEMA,
        "model_id": "idm-44-train-kits-0123456789ab-fedcba987654",
        "native_metadata_schema_id": "idm-peak-event-metadata-v1",
        "native_output_space_id": "idm-44-train-kits-9class-v1",
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": "idm-training-contract-44-train-kits-v1",
        "upstream_source_commit": "456656868538205ef756912c7cf5b0fd936de8af",
    }
    return build_descriptor(payload, frozenset(payload), IDM_DESCRIPTOR_SCHEMA)


def build_muscriptor_prediction(notes: tuple[int, ...]) -> NativePrediction:
    return NativePrediction(
        audio=CanonicalAudio(
            path=Path(),
            source_audio_id="song",
            source_audio_sha256="a" * 64,
            input_view_id="full-mix-v1",
            input_audio_sha256="b" * 64,
            byte_length=46,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=1,
        ),
        descriptor=build_muscriptor_descriptor(),
        events=tuple(
            NativeEvent(
                time_sec=float(index),
                native_class_id=f"drums:midi_{note}",
                model_output_bin=None,
                native_midi_note=note,
                native_metadata={"instrument_group": "drums"},
                confidence=None,
                velocity_midi=None,
            )
            for index, note in enumerate(notes)
        ),
    )


def build_idm_prediction(native_classes: tuple[str, ...]) -> NativePrediction:
    return NativePrediction(
        audio=CanonicalAudio(
            path=Path(),
            source_audio_id="song",
            source_audio_sha256="a" * 64,
            input_view_id="full-mix-v1",
            input_audio_sha256="b" * 64,
            byte_length=46,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=1,
        ),
        descriptor=build_idm_descriptor(),
        events=tuple(
            NativeEvent(
                time_sec=float(index),
                native_class_id=native_class,
                model_output_bin=index,
                native_midi_note=None,
                native_metadata={"frame_index": str(index), "native_velocity": "1"},
                confidence=0.8,
                velocity_midi=90,
            )
            for index, native_class in enumerate(native_classes)
        ),
    )


def test_muscriptor_map_is_one_string_keyed_frozen_pitch_map():
    assert MUSCRIPTOR_PREDICTION_MAP.map_id == MUSCRIPTOR_PREDICTION_MAP_ID
    assert all(isinstance(key, str) for key in MUSCRIPTOR_PREDICTION_MAP.classes)
    assert len(MUSCRIPTOR_PREDICTION_MAP.classes) == 18
    assert MUSCRIPTOR_PREDICTION_MAP.classes["35"].canonical_class == "kick"
    assert MUSCRIPTOR_PREDICTION_MAP.classes["40"].canonical_class == "snare"
    assert MUSCRIPTOR_PREDICTION_MAP.classes["44"].common_class == "hihat"
    assert MUSCRIPTOR_PREDICTION_MAP.classes["53"].canonical_class == "ride"
    assert MUSCRIPTOR_PREDICTION_MAP.classes["57"].canonical_class == "crash"


def test_map_muscriptor_prediction_preserves_mapped_aliases_and_unmapped_hits():
    prediction = build_muscriptor_prediction((35, 40, 44, 53, 57, 37, 55))

    mapped, diagnostics = map_muscriptor_prediction(prediction)

    assert [event.native.native_midi_note for event in mapped.events] == [
        35,
        40,
        44,
        53,
        57,
        37,
        55,
    ]
    assert [(event.canonical_class, event.common_class) for event in mapped.events] == [
        ("kick", "kick"),
        ("snare", "snare"),
        ("closed_hihat", "hihat"),
        ("ride", "ride"),
        ("crash", "crash"),
        (None, None),
        (None, None),
    ]
    assert [event.mapping_status for event in mapped.events] == [
        "mapped",
        "mapped",
        "mapped",
        "mapped",
        "mapped",
        "unmapped",
        "unmapped",
    ]
    assert diagnostics.unmapped == {"37": 1, "55": 1}
    assert all(
        event.prediction_map_version == MUSCRIPTOR_PREDICTION_MAP_ID for event in mapped.events
    )
    persisted = read_prediction_artifact(render_prediction_artifact(mapped)).prediction.events
    assert len(persisted) == 7
    assert [event.mapping_status for event in persisted[-2:]] == ["unmapped", "unmapped"]


@pytest.mark.parametrize(
    "change",
    [
        {"backend_id": "other-backend"},
        {"native_output_space_id": "other-output-space"},
    ],
)
def test_map_muscriptor_prediction_rejects_identity_mismatch(change: dict[str, str]):
    prediction = build_muscriptor_prediction((38,))
    descriptor = replace(
        prediction.descriptor,
        payload={**prediction.descriptor.payload, **change},
    )

    with pytest.raises(ValueError):
        map_muscriptor_prediction(replace(prediction, descriptor=descriptor))


def test_idm_prediction_map_uses_native_class_ids_for_all_fixed_classes_and_unmapped_evidence():
    prediction_map = IDM_PREDICTION_MAP
    assert prediction_map.map_id == IDM_PREDICTION_MAP_ID
    assert prediction_map.classes["TT_LMT"] == ClassMapping(None, "tom")

    native_classes = (
        "KD",
        "SD",
        "HH_CHH",
        "HH_OHH",
        "CY_CR",
        "CY_RD",
        "TT_HMT",
        "TT_LMT",
        "TT_HFT",
        "UNEXPECTED",
    )
    mapped, diagnostics = map_idm_prediction(build_idm_prediction(native_classes), prediction_map)

    assert [(event.canonical_class, event.common_class) for event in mapped.events] == [
        ("kick", "kick"),
        ("snare", "snare"),
        ("closed_hihat", "hihat"),
        ("open_hihat", "hihat"),
        ("crash", "crash"),
        ("ride", "ride"),
        ("high_tom", "tom"),
        (None, "tom"),
        ("low_or_floor_tom", "tom"),
        (None, None),
    ]
    assert [event.mapping_status for event in mapped.events] == [
        "mapped",
        "mapped",
        "mapped",
        "mapped",
        "mapped",
        "mapped",
        "mapped",
        "mapped",
        "mapped",
        "unmapped",
    ]
    assert diagnostics.unmapped == {"UNEXPECTED": 1}
    assert all(event.prediction_map_version == IDM_PREDICTION_MAP_ID for event in mapped.events)


@pytest.mark.parametrize(
    "change",
    [
        {"backend_id": "other-backend"},
        {"native_output_space_id": "other-output-space"},
    ],
)
def test_map_idm_prediction_rejects_identity_mismatch(change: dict[str, str]):
    prediction = build_idm_prediction(("KD",))
    descriptor = replace(
        prediction.descriptor,
        payload={**prediction.descriptor.payload, **change},
    )

    with pytest.raises(ValueError):
        map_idm_prediction(replace(prediction, descriptor=descriptor), IDM_PREDICTION_MAP)
