from pathlib import Path

import pretty_midi

from src.benchmark.midi_io import parse_prediction_midi, write_reference_midi
from src.benchmark.models import BenchmarkEvent


def test_parse_prediction_midi_extracts_drum_note_onsets(tmp_path: Path):
    path = tmp_path / "prediction.mid"
    midi = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=1.25, end=1.35))
    midi.instruments.append(drums)
    midi.write(str(path))

    events = parse_prediction_midi(path, chart_id="song")

    assert events == [
        BenchmarkEvent("song", 1.25, "36", "prediction", {"midi_note": 36, "velocity": 100})
    ]


def test_write_reference_midi_handles_simultaneous_hits(tmp_path: Path):
    path = tmp_path / "reference.mid"
    events = [
        BenchmarkEvent("song", 0.0, "kick", "ground_truth"),
        BenchmarkEvent("song", 0.0, "snare", "ground_truth"),
    ]

    write_reference_midi(events, path)

    parsed = pretty_midi.PrettyMIDI(str(path))
    notes = sorted(note.pitch for note in parsed.instruments[0].notes)
    assert notes == [36, 38]


def test_write_reference_midi_skips_unmapped_classes_with_warning(tmp_path: Path, caplog):
    import logging

    path = tmp_path / "reference.mid"
    events = [
        BenchmarkEvent("song", 0.0, "kick", "ground_truth"),
        BenchmarkEvent("song", 1.0, "unknown_class", "ground_truth"),
    ]

    with caplog.at_level(logging.WARNING, logger="src.benchmark.midi_io"):
        write_reference_midi(events, path)

    assert "unmapped canonical class 'unknown_class'" in caplog.text

    parsed = pretty_midi.PrettyMIDI(str(path))
    notes = [note.pitch for note in parsed.instruments[0].notes]
    assert notes == [36]
