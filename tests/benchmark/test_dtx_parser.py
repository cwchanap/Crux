from pathlib import Path

import pytest

from src.benchmark.dtx_parser import parse_dtx_file, parse_dtx_text


def test_parse_notes_with_lf_or_crlf():
    for newline in ("\n", "\r\n"):
        text = (
            f"#TITLE: Song{newline}"
            f"#BPM: 120{newline}"
            f"#00011: 01000002{newline}"
            f"#00012: 00010000{newline}"
        )

        chart = parse_dtx_text(text, chart_id="song")

        assert chart.title == "Song"
        assert chart.base_bpm == 120.0
        assert [
            (event.measure, event.position, event.lane_id, event.note_id) for event in chart.events
        ] == [
            (0, 0.0, "11", "01"),
            (0, 0.25, "12", "01"),
            (0, 0.75, "11", "02"),
        ]


def test_parse_render_chip_metadata_headers():
    text = "\n".join(
        [
            "#WAV01: kick.wav",
            "#VOLUME01: 90",
            "#POSITION01: 0.25",
            "#BPM: 120",
        ]
    )

    chart = parse_dtx_text(text, chart_id="render-metadata")

    assert chart.wav_table == {"01": "kick.wav"}
    assert chart.volume_table == {"01": 90.0}
    assert chart.position_table == {"01": 0.25}


def test_parse_bpm_table_and_bpm_channels():
    text = "\n".join(
        [
            "#BPM: 120",
            "#BPM01: 180",
            "#00008: 0100",
            "#00103: 3C00",
        ]
    )

    chart = parse_dtx_text(text, chart_id="tempo")

    assert chart.bpm_table == {"01": 180.0}
    # Channel 03 "3C" is hexadecimal = 60 BPM (not base-36 = 120)
    assert [(event.measure, event.position, event.bpm) for event in chart.bpm_events] == [
        (0, 0.0, 180.0),
        (1, 0.0, 60.0),
    ]


def test_parse_bpm_source_order_channel03_after_channel08():
    """When channel 08 appears before channel 03 at the same beat, the later
    channel 03 line (60 BPM) must sort after the table event (180 BPM) so
    that downstream consumers use the correct last-written tempo."""
    text = "\n".join(
        [
            "#BPM: 120",
            "#BPM01: 180",
            "#00008: 0100",  # table BPM → 180, source_order=0
            "#00003: 3C00",  # direct BPM → 60,  source_order=1 (later in file)
        ]
    )

    chart = parse_dtx_text(text, chart_id="source-order")

    assert len(chart.bpm_events) == 2
    # Both at measure 0, position 0.0.  Source order preserves file order,
    # so the direct BPM (60) must come last and win when downstream code
    # picks the last event at a given position.
    assert [(ev.bpm, ev.source_channel) for ev in chart.bpm_events] == [
        (180.0, "08"),
        (60.0, "03"),
    ]


def test_parse_bpm_source_order_channel08_after_channel03():
    """When channel 03 appears before channel 08 at the same beat, the later
    channel 08 line must sort after the direct BPM event."""
    text = "\n".join(
        [
            "#BPM: 120",
            "#BPM01: 180",
            "#00003: 3C00",  # direct BPM → 60,  source_order=0
            "#00008: 0100",  # table BPM → 180, source_order=1 (later in file)
        ]
    )

    chart = parse_dtx_text(text, chart_id="source-order-reverse")

    assert len(chart.bpm_events) == 2
    assert [(ev.bpm, ev.source_channel) for ev in chart.bpm_events] == [
        (60.0, "03"),
        (180.0, "08"),
    ]


def test_parse_direct_bpm_channel_03_hex_values():
    """Channel 03 BPM values are 2-digit hexadecimal (base 16)."""
    text = "\n".join(
        [
            "#BPM: 100",
            "#00003: 7F00",  # 0x7F = 127
            "#00103: FF00",  # 0xFF = 255
            "#00203: 5000",  # 0x50 = 80
        ]
    )

    chart = parse_dtx_text(text, chart_id="hex-bpm")

    assert [(ev.measure, ev.bpm) for ev in chart.bpm_events] == [
        (0, 127.0),
        (1, 255.0),
        (2, 80.0),
    ]


def test_parse_bpm_table_reference_defined_later():
    text = "\n".join(
        [
            "#BPM: 120",
            "#00008: 0100",
            "#BPM01: 180",
        ]
    )

    chart = parse_dtx_text(text, chart_id="tempo-late")

    assert chart.bpm_table == {"01": 180.0}
    assert [(event.measure, event.position, event.bpm) for event in chart.bpm_events] == [
        (0, 0.0, 180.0),
    ]
    assert chart.warnings == []


def test_parse_measure_length_carries_forward_metadata():
    text = "#BPM: 120\n#00102: 0.5\n#00302: 1.5\n#00111: 0100\n"

    chart = parse_dtx_text(text, chart_id="length")

    assert chart.measure_lengths == {1: 0.5, 3: 1.5}


def test_parse_text_strips_utf8_bom_from_first_header():
    text = "\ufeff#TITLE: Song\n#BPM: 120\n"

    chart = parse_dtx_text(text, chart_id="bom")

    assert chart.title == "Song"
    assert chart.base_bpm == 120.0


def test_parse_file_decodes_shift_jis(tmp_path: Path):
    path = tmp_path / "sjis.dtx"
    path.write_bytes("#TITLE: テスト\r\n#BPM: 120\r\n".encode("shift-jis"))

    chart = parse_dtx_file(path, chart_id="sjis")

    assert chart.title == "テスト"


def test_parse_preserves_semicolons_in_string_value_headers():
    text = "\n".join(
        [
            "#TITLE: Song; Part II",
            "#ARTIST: Band; feat. Singer",
            "#WAV01: kick;snare.wav",
            "#BPM: 120",
        ]
    )

    chart = parse_dtx_text(text, chart_id="semicolons")

    assert chart.title == "Song; Part II"
    assert chart.artist == "Band; feat. Singer"
    assert chart.wav_table == {"01": "kick;snare.wav"}


def test_parse_strips_semicolon_comments_from_data_lines():
    text = "\n".join(
        [
            "#BPM: 120",
            "#00011: 01000002; note data comment",
            "#00103: 3C00; bpm comment",
        ]
    )

    chart = parse_dtx_text(text, chart_id="data-comments")

    assert len(chart.events) == 2
    assert [(ev.measure, ev.position, ev.lane_id, ev.note_id) for ev in chart.events] == [
        (0, 0.0, "11", "01"),
        (0, 0.75, "11", "02"),
    ]
    assert len(chart.bpm_events) == 1
    # Channel 03 "3C" is hexadecimal = 60 BPM
    assert chart.bpm_events[0].bpm == 60.0


def test_parse_dtx_raises_on_odd_length_pattern():
    with pytest.raises(ValueError, match="odd length"):
        parse_dtx_text("#BPM: 120\n#00011: 010\n", chart_id="test")


def test_parse_file_rejects_gibberish_shift_jis_decode(tmp_path: Path):
    """A UTF-16LE file should not be silently decoded as shift-JIS gibberish.

    Without post-decode validation, shift-JIS would accept the raw bytes and
    return an empty chart (default BPM, zero events) instead of trying the
    later UTF-16 encodings in the list.
    """
    content = "#TITLE: UTF16 Song\r\n#BPM: 140\r\n#00011: 0100\r\n"
    path = tmp_path / "utf16.dtx"
    path.write_bytes(content.encode("utf-16le"))

    chart = parse_dtx_file(path, chart_id="utf16")

    assert chart.title == "UTF16 Song"
    assert chart.base_bpm == 140.0
    assert len(chart.events) == 1


def test_parse_file_raises_when_no_encoding_produces_valid_dtx(tmp_path: Path):
    """Random binary content that decodes under multiple encodings but contains
    no DTX lines should still raise."""
    path = tmp_path / "garbage.dtx"
    path.write_bytes(bytes(range(256)))

    with pytest.raises((UnicodeDecodeError, ValueError)):
        parse_dtx_file(path, chart_id="garbage")


def test_parse_file_prefers_utf8_encoding_over_shift_jis(tmp_path: Path):
    """A UTF-8 encoded chart with non-ASCII text must not be garbled by a
    Shift-JIS decode.  UTF-8 is tried first, so the correct title survives."""
    # " café " in UTF-8 is valid but decodes to garbled text under Shift-JIS.
    content = "#TITLE: café\n#BPM: 120\n#00011: 0100\n"
    path = tmp_path / "utf8.dtx"
    path.write_bytes(content.encode("utf-8"))

    chart = parse_dtx_file(path, chart_id="utf8")

    assert chart.title == "café"
    assert chart.base_bpm == 120.0
    assert len(chart.events) == 1


def test_parse_file_decodes_bomless_utf16be(tmp_path: Path) -> None:
    content = "#TITLE: UTF16BE Song\r\n#BPM: 140\r\n#00011: 0100\r\n"
    path = tmp_path / "utf16be.dtx"
    path.write_bytes(content.encode("utf-16be"))

    chart = parse_dtx_file(path, chart_id="utf16be")

    assert chart.title == "UTF16BE Song"
    assert chart.base_bpm == 140.0
    assert len(chart.events) == 1


def test_parse_dtx_retains_numeric_dlevel() -> None:
    chart = parse_dtx_text("#DLEVEL: 87\n#BPM: 120\n", "song")

    assert chart.dlevel_raw == "87"
    assert chart.dlevel_normalized == 87


def test_parse_dtx_dlevel_last_directive_wins() -> None:
    chart = parse_dtx_text("#DLEVEL: 24\n#DLEVEL: 87\n#BPM: 120\n", "song")

    assert chart.dlevel_raw == "87"
    assert chart.dlevel_normalized == 87


@pytest.mark.parametrize(("raw_value", "normalized"), [("0", 0), ("100", 100)])
def test_parse_dtx_normalizes_ascii_dlevel_boundaries(raw_value: str, normalized: int) -> None:
    chart = parse_dtx_text(f"#DLEVEL: {raw_value}\n#BPM: 120\n", "song")

    assert chart.dlevel_raw == raw_value
    assert chart.dlevel_normalized == normalized


@pytest.mark.parametrize("raw_value", ["87.5", "１２", "-1", "101"])
def test_parse_dtx_warns_for_invalid_dlevel(raw_value: str) -> None:
    chart = parse_dtx_text(f"#DLEVEL: {raw_value}\n#BPM: 120\n", "song")

    assert chart.dlevel_raw == raw_value
    assert chart.dlevel_normalized is None
    assert any("DLEVEL" in warning and raw_value in warning for warning in chart.warnings)


def test_parse_star_prefixed_headers():
    """DTX charts may use * instead of # as the directive prefix."""
    text = "\n".join(
        [
            "*TITLE: Star Song",
            "*ARTIST: Star Artist",
            "*BPM: 180",
            "*BPM01: 200",
            "*WAV01: kick.wav",
            "*VOLUME01: 80",
            "*POSITION01: 0.5",
            "*00011: 0100",
        ]
    )

    chart = parse_dtx_text(text, chart_id="star-headers")

    assert chart.title == "Star Song"
    assert chart.artist == "Star Artist"
    assert chart.base_bpm == 180.0
    assert chart.bpm_table == {"01": 200.0}
    assert chart.wav_table == {"01": "kick.wav"}
    assert chart.volume_table == {"01": 80.0}
    assert chart.position_table == {"01": 0.5}
    assert len(chart.events) == 1
    assert chart.events[0].lane_id == "11"
    assert chart.events[0].note_id == "01"
