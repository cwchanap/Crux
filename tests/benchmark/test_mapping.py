from src.benchmark.mapping import (
    DEFAULT_DTX_LANE_MAP,
    DEFAULT_MIDI_NOTE_MAP,
    map_dtx_events,
    map_midi_events,
)
from src.benchmark.models import BenchmarkEvent


def test_default_dtx_mapping_supports_drumery_editor_lanes():
    assert DEFAULT_DTX_LANE_MAP["13"].collapsed_class == "kick"
    assert DEFAULT_DTX_LANE_MAP["12"].collapsed_class == "snare"
    assert DEFAULT_DTX_LANE_MAP["11"].collapsed_class == "closed_hihat"


def test_default_midi_mapping_supports_current_transcriber_notes():
    assert DEFAULT_MIDI_NOTE_MAP[36] == "kick"
    assert DEFAULT_MIDI_NOTE_MAP[38] == "snare"


def test_map_dtx_event_replaces_lane_class_and_preserves_native_metadata():
    event = BenchmarkEvent("song", 0.0, "13", "ground_truth", {"lane_id": "13"})

    mapped, diagnostics = map_dtx_events([event])

    assert mapped[0].canonical_class == "kick"
    assert mapped[0].metadata["native_class"] == "kick"
    assert diagnostics.unmapped == {}


def test_map_dtx_events_respects_explicit_empty_lane_map():
    event = BenchmarkEvent("song", 0.0, "13", "ground_truth", {"lane_id": "13"})

    mapped, diagnostics = map_dtx_events([event], lane_map={})

    assert mapped == []
    assert diagnostics.unmapped == {"13": 1}


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
