import pytest

from src.benchmark.dtx_parser import parse_dtx_text
from src.benchmark.timing import dtx_events_to_timed_events


def test_constant_bpm_timing():
    chart = parse_dtx_text("#BPM: 120\n#00111: 0100\n", chart_id="song")

    timed = dtx_events_to_timed_events(chart)

    assert timed[0].time_sec == 2.0


def test_measure_length_changes_timing():
    chart = parse_dtx_text("#BPM: 120\n#00102: 0.5\n#00111: 0100\n#00211: 0100\n", "song")

    timed = dtx_events_to_timed_events(chart)

    assert [event.time_sec for event in timed] == [2.0, 3.0]


def test_bpm_change_affects_following_segments():
    chart = parse_dtx_text("#BPM: 120\n#BPM01: 60\n#00108: 0100\n#00211: 0100\n", "song")

    timed = dtx_events_to_timed_events(chart)

    assert timed[0].time_sec == 6.0


def test_same_beat_bpm_events_raise_error():
    chart = parse_dtx_text(
        "#BPM: 120\n#BPM01: 150\n#00103: 0100\n#00108: 0100\n#00211: 0100\n",
        "song",
    )

    with pytest.raises(ValueError, match="duplicate tempo events at beat 4.0"):
        dtx_events_to_timed_events(chart)


def test_beat_zero_bpm_event_overrides_seeded_base_tempo():
    chart = parse_dtx_text("#BPM: 120\n#BPM01: 60\n#00008: 01\n#00111: 01\n", "song")

    timed = dtx_events_to_timed_events(chart)

    assert timed[0].time_sec == 4.0
