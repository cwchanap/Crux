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


def test_parse_file_tries_shift_jis(tmp_path: Path):
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
