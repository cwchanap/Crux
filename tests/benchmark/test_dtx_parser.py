from pathlib import Path

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
    assert [(event.measure, event.position, event.bpm) for event in chart.bpm_events] == [
        (0, 0.0, 180.0),
        (1, 0.0, 120.0),
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
