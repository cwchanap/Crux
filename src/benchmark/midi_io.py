from __future__ import annotations

import logging
from pathlib import Path

import pretty_midi

from src.benchmark.models import BenchmarkEvent

logger = logging.getLogger(__name__)

REFERENCE_CLASS_TO_MIDI = {
    "kick": 36,
    "snare": 38,
    "closed_hihat": 42,
    "open_hihat": 46,
    "crash": 49,
    "ride": 51,
    "low_tom": 45,
    "mid_tom": 47,
    "high_tom": 50,
}


def parse_prediction_midi(path: Path, chart_id: str) -> list[BenchmarkEvent]:
    midi = pretty_midi.PrettyMIDI(str(path))
    events: list[BenchmarkEvent] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            continue
        for note in instrument.notes:
            events.append(
                BenchmarkEvent(
                    chart_id=chart_id,
                    time_sec=float(note.start),
                    canonical_class=str(note.pitch),
                    source="prediction",
                    metadata={"midi_note": int(note.pitch), "velocity": int(note.velocity)},
                )
            )
    if not events and midi.instruments:
        non_drum = [i.name for i in midi.instruments if not i.is_drum]
        logger.warning(
            "parse_prediction_midi: no drum track found in %s (chart_id=%r); "
            "non-drum instruments: %s",
            path,
            chart_id,
            non_drum or "<none>",
        )
    return sorted(events)


def write_reference_midi(events: list[BenchmarkEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    for event in sorted(events):
        midi_note = REFERENCE_CLASS_TO_MIDI.get(event.canonical_class)
        if midi_note is None:
            logger.warning(
                "Skipping unmapped canonical class %r in write_reference_midi",
                event.canonical_class,
            )
            continue
        start = max(0.0, event.time_sec)
        drums.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=midi_note,
                start=start,
                end=start + 0.1,
            )
        )
    midi.instruments.append(drums)
    midi.write(str(path))
