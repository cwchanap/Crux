from src.benchmark.dtx_parser import parse_dtx_text
from src.benchmark.timing import (
    build_dtx_timing_map,
    dtx_events_to_chart_time_events,
)


def test_constant_bpm_timing():
    chart = parse_dtx_text("#BPM: 120\n#00111: 0100\n", chart_id="song")

    timed = dtx_events_to_chart_time_events(chart)

    assert timed[0].time_sec == 2.0


def test_dtx_measure_length_persists_until_superseded():
    """DTX channel 02 is sticky: once a measure length is set it persists for
    all later measures until superseded by another explicit length (backed by
    DTXManiaXG source).  Measure 0 with no explicit length stays 1.0."""
    chart = parse_dtx_text(
        "#BPM: 120\n"
        "#00102: 0.5\n"
        "#00111: 01\n"
        "#00211: 01\n"
        "#00311: 01\n"
        "#00402: 1.0\n"
        "#00411: 01\n"
        "#00511: 01\n",
        "song",
    )

    assert [event.time_sec for event in dtx_events_to_chart_time_events(chart)] == [
        2.0,
        3.0,
        4.0,
        5.0,
        7.0,
    ]


def test_dtx_measure_length_multiple_sticky_changes():
    """Multiple sticky changes 0.5 -> 1.5 -> 1.0 carry forward until the next
    explicit length."""
    chart = parse_dtx_text(
        "#BPM: 120\n"
        "#00102: 0.5\n"
        "#00302: 1.5\n"
        "#00502: 1.0\n"
        "#00111: 01\n"
        "#00211: 01\n"
        "#00311: 01\n"
        "#00411: 01\n"
        "#00511: 01\n"
        "#00611: 01\n",
        "song",
    )

    assert [event.time_sec for event in dtx_events_to_chart_time_events(chart)] == [
        2.0,
        3.0,
        4.0,
        7.0,
        10.0,
        12.0,
    ]


def test_bpm_change_inside_sticky_shortened_measure():
    """A BPM change at a fractional position inside a sticky shortened measure
    must use the resolved (sticky) measure length for its in-measure beat, and
    the following note lands at 4.5 seconds."""
    chart = parse_dtx_text(
        "#BPM: 120\n"
        "#BPM01: 30\n"
        "#00102: 0.5\n"
        "#00108: 0001\n"  # BPM 30 at measure 1, position 0.5 (inside the shortened measure)
        "#00211: 01\n",
        "song",
    )

    assert dtx_events_to_chart_time_events(chart)[0].time_sec == 4.5


def test_bgm_and_playable_events_share_one_timing_map():
    """BGM (channel 01) and playable events resolve through a single
    :class:`DtxTimingMap`, so they produce identical times at the same measure
    and position even under a sticky shortened measure."""
    chart = parse_dtx_text(
        "#BPM: 120\n#00102: 0.5\n#00201: 01\n#00211: 01\n",
        "song",
    )

    timing_map = build_dtx_timing_map(chart)
    bgm_time = timing_map.time_sec(chart.bgm_events[0])
    playable_time = dtx_events_to_chart_time_events(chart)[0].time_sec

    assert bgm_time == playable_time == 3.0


def test_bpm_change_affects_following_segments():
    chart = parse_dtx_text("#BPM: 120\n#BPM01: 60\n#00108: 0100\n#00211: 0100\n", "song")

    timed = dtx_events_to_chart_time_events(chart)

    assert timed[0].time_sec == 6.0


def test_same_beat_bpm_events_raise_error():
    chart = parse_dtx_text(
        "#BPM: 120\n#BPM01: 150\n#00103: 0100\n#00108: 0100\n#00211: 0100\n",
        "song",
    )

    dtx_events_to_chart_time_events(chart)

    assert any("duplicate tempo at beat 4.0" in w for w in chart.warnings)


def test_beat_zero_bpm_event_overrides_seeded_base_tempo():
    chart = parse_dtx_text("#BPM: 120\n#BPM01: 60\n#00008: 01\n#00111: 01\n", "song")

    timed = dtx_events_to_chart_time_events(chart)

    assert timed[0].time_sec == 4.0


def test_multiple_beat_zero_bpm_events_raise_error():
    chart = parse_dtx_text(
        "#BPM: 120\n#BPM01: 60\n#00003: 01\n#00008: 01\n#00111: 01\n",
        "song",
    )

    dtx_events_to_chart_time_events(chart)

    assert any("duplicate tempo at beat 0.0" in w for w in chart.warnings)
