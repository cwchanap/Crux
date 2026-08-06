from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.benchmark.dtx_text import decode_dtxmania_text
from src.benchmark.models import DtxEvent

LINE_RE = re.compile(r"^[#*](?P<measure>\d{3})(?P<channel>[0-9A-Za-z]{2})\s*:\s*(?P<value>.*)$")
HEADER_RE = re.compile(r"^[#*](?P<key>[A-Za-z0-9_]+)\s*:?\s*(?P<value>.*)$")
_ASCII_DECIMAL_RE = re.compile(r"^[0-9]+$")
_STRING_VALUE_HEADER_KEYS = frozenset({"TITLE", "ARTIST"})


@dataclass(frozen=True)
class DtxBpmEvent:
    measure: int
    position: float
    bpm: float
    source_channel: str
    source_order: int = 0


@dataclass(frozen=True)
class ParsedDtxChart:
    chart_id: str
    title: str = ""
    artist: str = ""
    dlevel_raw: str | None = None
    dlevel_normalized: int | None = None
    base_bpm: float = 120.0
    bpm_table: dict[str, float] = field(default_factory=dict)
    wav_table: dict[str, str] = field(default_factory=dict)
    volume_table: dict[str, float] = field(default_factory=dict)
    position_table: dict[str, float] = field(default_factory=dict)
    measure_lengths: dict[int, float] = field(default_factory=dict)
    events: list[DtxEvent] = field(default_factory=list)
    bpm_events: list[DtxBpmEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_dtx_file(path: Path, chart_id: str | None = None) -> ParsedDtxChart:
    text = decode_dtxmania_text(path.read_bytes(), source_name=str(path), kind="dtx")
    return parse_dtx_text(text, chart_id=chart_id or path.stem)


def _is_string_value_header_key(key: str) -> bool:
    """Return True if the header key stores a free-form string that may contain semicolons."""
    return key in _STRING_VALUE_HEADER_KEYS or (key.startswith("WAV") and len(key) == 5)


def parse_dtx_text(text: str, chart_id: str) -> ParsedDtxChart:
    text = text.removeprefix("\ufeff")
    title = ""
    artist = ""
    dlevel_raw: str | None = None
    dlevel_normalized: int | None = None
    base_bpm = 120.0
    bpm_table: dict[str, float] = {}
    wav_table: dict[str, str] = {}
    volume_table: dict[str, float] = {}
    position_table: dict[str, float] = {}
    measure_lengths: dict[int, float] = {}
    events: list[DtxEvent] = []
    bpm_events: list[DtxBpmEvent] = []
    pending_table_bpm_events: list[tuple[int, str, int]] = []
    warnings: list[str] = []
    source_counter = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        # Preserve semicolons in header lines with free-form string values
        # (TITLE, ARTIST, WAVxx) since they can legitimately appear in
        # titles, artist names and filenames.  For everything else (data
        # lines, numeric headers) semicolons are comment delimiters.
        header_peek = HEADER_RE.match(line)
        if header_peek and _is_string_value_header_key(header_peek.group("key").upper()):
            value_without_comment = line
        else:
            value_without_comment = line.split(";", 1)[0].strip()

        match = LINE_RE.match(value_without_comment)
        if match:
            measure = int(match.group("measure"))
            channel = match.group("channel").upper()
            value = match.group("value").strip().replace("_", "")
            if channel == "02":
                measure_lengths[measure] = _parse_positive_float(value, f"measure length {measure}")
            elif channel == "03":
                bpm_events.extend(_parse_direct_bpm_events(measure, value, source_counter))
                source_counter += 1
            elif channel == "08":
                pending_table_bpm_events.append((measure, value, source_counter))
                source_counter += 1
            else:
                events.extend(_parse_note_events(chart_id, measure, channel, value))
            continue

        header = HEADER_RE.match(value_without_comment)
        if not header:
            continue
        key = header.group("key").upper()
        value = header.group("value").strip().strip('"')
        if key == "TITLE":
            title = value
        elif key == "ARTIST":
            artist = value
        elif key == "DLEVEL":
            dlevel_raw = value
            dlevel_normalized = _normalize_dlevel(value)
            if dlevel_normalized is None:
                warnings.append(f"ignoring invalid DLEVEL value: {value!r}")
        elif key == "BPM":
            base_bpm = _parse_positive_float(value, "base BPM")
        elif key.startswith("BPM") and len(key) == 5:
            bpm_table[key[3:].upper()] = _parse_positive_float(value, key)
        elif key.startswith("WAV") and len(key) == 5:
            wav_table[key[3:].upper()] = value
        elif key.startswith("VOLUME") and len(key) == 8:
            try:
                volume_table[key[6:].upper()] = float(value)
            except ValueError:
                warnings.append(f"ignoring non-numeric VOLUME value for {key}: {value!r}")
        elif key.startswith("POSITION") and len(key) == 10:
            try:
                position_table[key[8:].upper()] = float(value)
            except ValueError:
                warnings.append(f"ignoring non-numeric POSITION value for {key}: {value!r}")

    for measure, value, order in pending_table_bpm_events:
        bpm_events.extend(_parse_table_bpm_events(measure, value, bpm_table, warnings, order))

    return ParsedDtxChart(
        chart_id=chart_id,
        title=title,
        artist=artist,
        dlevel_raw=dlevel_raw,
        dlevel_normalized=dlevel_normalized,
        base_bpm=base_bpm,
        bpm_table=bpm_table,
        wav_table=wav_table,
        volume_table=volume_table,
        position_table=position_table,
        measure_lengths=measure_lengths,
        events=sorted(events, key=lambda event: (event.measure, event.position, event.lane_id)),
        bpm_events=sorted(
            bpm_events, key=lambda event: (event.measure, event.position, event.source_order)
        ),
        warnings=warnings,
    )


def _parse_positive_float(value: str, label: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _normalize_dlevel(value: str) -> int | None:
    if _ASCII_DECIMAL_RE.fullmatch(value) is None:
        return None
    try:
        normalized = int(value)
    except ValueError:
        return None
    return normalized if normalized <= 100 else None


def _chunks(value: str) -> list[str]:
    if len(value) % 2:
        raise ValueError(f"DTX pattern has odd length: {value}")
    return [value[index : index + 2].upper() for index in range(0, len(value), 2)]


def _parse_note_events(chart_id: str, measure: int, channel: str, value: str) -> list[DtxEvent]:
    chunks = _chunks(value)
    if not chunks:
        return []
    return [
        DtxEvent(chart_id, measure, index / len(chunks), channel, note_id)
        for index, note_id in enumerate(chunks)
        if note_id != "00"
    ]


def _parse_direct_bpm_events(measure: int, value: str, source_order: int = 0) -> list[DtxBpmEvent]:
    """Parse DTX channel 03 direct BPM events (2-digit hexadecimal values)."""
    chunks = _chunks(value)
    return [
        DtxBpmEvent(measure, index / len(chunks), float(int(note_id, 16)), "03", source_order)
        for index, note_id in enumerate(chunks)
        if note_id != "00"
    ]


def _parse_table_bpm_events(
    measure: int,
    value: str,
    bpm_table: dict[str, float],
    warnings: list[str],
    source_order: int = 0,
) -> list[DtxBpmEvent]:
    chunks = _chunks(value)
    events: list[DtxBpmEvent] = []
    for index, note_id in enumerate(chunks):
        if note_id == "00":
            continue
        bpm = bpm_table.get(note_id)
        if bpm is None:
            warnings.append(f"BPM event references unknown #BPM{note_id}")
            continue
        events.append(DtxBpmEvent(measure, index / len(chunks), bpm, "08", source_order))
    return events
