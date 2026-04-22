from pathlib import Path

import pytest

import src.benchmark.render_audio as render_audio
from src.benchmark.models import BenchmarkEvent
from src.benchmark.render_audio import plan_render_corpus, plan_render_song


def test_plan_render_corpus_uses_highest_level_chart_rule_without_drum_stem(
    tmp_path: Path,
):
    raw = tmp_path / "raw"
    song = raw / "Song One"
    song.mkdir(parents=True)
    (song / "bas.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "ext.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "mas.dtx").write_text("#BPM: 120\n", encoding="utf-8")

    result = plan_render_corpus(raw)

    assert result.invalid_items == []
    assert len(result.valid_items) == 1
    assert result.valid_items[0].selected_chart.name == "mas.dtx"
    assert result.valid_items[0].selected_chart_level == "mas"


def test_plan_render_song_schedules_sample_placements_with_absolute_times(tmp_path: Path):
    song = tmp_path / "Song Two"
    song.mkdir()
    (song / "mas.dtx").write_text(
        "\n".join(
            [
                "#BPM: 120",
                "#WAV01: kick.wav",
                "#WAV02: snare.wav",
                "#00111: 01",
                "#00211: 02",
            ]
        ),
        encoding="utf-8",
    )
    (song / "kick.wav").write_bytes(b"kick")
    (song / "snare.wav").write_bytes(b"snare")

    plan, invalid = plan_render_song(song)

    assert invalid is None
    assert plan is not None
    assert [
        (placement.time_sec, placement.sample_name, placement.sample_path.name, placement.note_id)
        for placement in plan.placements
    ] == [
        (2.0, "kick.wav", "kick.wav", "01"),
        (4.0, "snare.wav", "snare.wav", "02"),
    ]


def test_plan_render_song_marks_missing_chip_or_sample_references_invalid(tmp_path: Path):
    song = tmp_path / "Song Three"
    song.mkdir()
    (song / "mas.dtx").write_text(
        "\n".join(
            [
                "#BPM: 120",
                "#WAV02: missing_sample.wav",
                "#00111: 01",
                "#00211: 02",
            ]
        ),
        encoding="utf-8",
    )

    plan, invalid = plan_render_song(song)

    assert plan is None
    assert invalid is not None
    assert invalid.reason == "missing chip/sample references"
    assert invalid.details["missing_note_ids"] == ["01"]
    assert invalid.details["missing_sample_names"] == ["missing_sample.wav"]


def test_plan_render_corpus_collects_timing_failures_as_invalid_items(tmp_path: Path):
    raw = tmp_path / "raw"
    good_song = raw / "Good Song"
    bad_song = raw / "Bad Song"
    good_song.mkdir(parents=True)
    bad_song.mkdir(parents=True)

    (good_song / "mas.dtx").write_text(
        "\n".join(
            [
                "#BPM: 120",
                "#WAV01: kick.wav",
                "#00111: 01",
            ]
        ),
        encoding="utf-8",
    )
    (good_song / "kick.wav").write_bytes(b"kick")
    (bad_song / "mas.dtx").write_text(
        "\n".join(
            [
                "#BPM: 120",
                "#BPM01: 150",
                "#00103: 01",
                "#00108: 01",
                "#00211: 01",
            ]
        ),
        encoding="utf-8",
    )

    result = plan_render_corpus(raw)

    assert len(result.valid_items) == 1
    assert result.valid_items[0].song_id == "Good Song"
    assert len(result.invalid_items) == 1
    assert result.invalid_items[0].raw_folder == bad_song
    assert result.invalid_items[0].reason == "failed to parse or time chart"
    assert result.invalid_items[0].details["exception_class"] == "ValueError"


def test_plan_render_song_rejects_directory_sample_paths(tmp_path: Path):
    song = tmp_path / "Song Four"
    song.mkdir()
    (song / "mas.dtx").write_text(
        "\n".join(
            [
                "#BPM: 120",
                "#WAV01: kick.wav",
                "#00111: 01",
            ]
        ),
        encoding="utf-8",
    )
    (song / "kick.wav").mkdir()

    plan, invalid = plan_render_song(song)

    assert plan is None
    assert invalid is not None
    assert invalid.reason == "missing chip/sample references"
    assert invalid.details["missing_note_ids"] == []
    assert invalid.details["missing_sample_names"] == ["kick.wav"]


@pytest.mark.parametrize("sample_ref", ["../escape.wav", "ABSOLUTE"])
def test_plan_render_song_rejects_sample_paths_outside_song_dir(tmp_path: Path, sample_ref: str):
    root = tmp_path / "raw"
    song = root / "Song Five"
    song.mkdir(parents=True)
    outside_sample = root / "escape.wav"
    outside_sample.write_bytes(b"escape")

    sample_name = str(outside_sample.resolve()) if sample_ref == "ABSOLUTE" else sample_ref
    (song / "mas.dtx").write_text(
        "\n".join(
            [
                "#BPM: 120",
                f"#WAV01: {sample_name}",
                "#00111: 01",
            ]
        ),
        encoding="utf-8",
    )

    plan, invalid = plan_render_song(song)

    assert plan is None
    assert invalid is not None
    assert invalid.reason == "missing chip/sample references"
    assert invalid.details["missing_note_ids"] == []
    assert invalid.details["missing_sample_names"] == [sample_name]


def test_plan_render_song_marks_missing_note_id_metadata_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    song = tmp_path / "Song Six"
    song.mkdir()
    (song / "mas.dtx").write_text(
        "\n".join(
            [
                "#BPM: 120",
                "#WAV01: kick.wav",
                "#00111: 01",
            ]
        ),
        encoding="utf-8",
    )
    (song / "kick.wav").write_bytes(b"kick")

    monkeypatch.setattr(
        render_audio,
        "dtx_events_to_timed_events",
        lambda _chart: [
            BenchmarkEvent(
                chart_id="Song Six",
                time_sec=2.0,
                canonical_class="11",
                source="ground_truth",
                metadata={"lane_id": "11"},
            )
        ],
    )

    plan, invalid = plan_render_song(song)

    assert plan is None
    assert invalid is not None
    assert invalid.reason == "missing chip/sample references"
    assert invalid.details["missing_note_ids"] == []
    assert invalid.details["missing_event_metadata"] == ["note_id"]


def test_plan_render_song_preserves_empty_sample_name_in_diagnostics(tmp_path: Path):
    song = tmp_path / "Song Seven"
    song.mkdir()
    (song / "mas.dtx").write_text(
        "\n".join(
            [
                "#BPM: 120",
                "#WAV01:",
                "#00111: 01",
            ]
        ),
        encoding="utf-8",
    )

    plan, invalid = plan_render_song(song)

    assert plan is None
    assert invalid is not None
    assert invalid.reason == "missing chip/sample references"
    assert invalid.details["missing_note_ids"] == []
    assert invalid.details["missing_sample_names"] == [""]


def test_plan_render_song_propagates_unexpected_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    song = tmp_path / "Song Eight"
    song.mkdir()
    (song / "mas.dtx").write_text("#BPM: 120\n", encoding="utf-8")

    def raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(render_audio, "parse_dtx_file", raise_runtime_error)

    with pytest.raises(RuntimeError, match="boom"):
        plan_render_song(song)


def test_plan_render_song_converts_expected_io_chart_failures_to_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    song = tmp_path / "Song Nine"
    song.mkdir()
    (song / "mas.dtx").write_text("#BPM: 120\n", encoding="utf-8")

    def raise_file_not_found_error(*_args, **_kwargs):
        raise FileNotFoundError("missing chart")

    monkeypatch.setattr(render_audio, "parse_dtx_file", raise_file_not_found_error)

    plan, invalid = plan_render_song(song)

    assert plan is None
    assert invalid is not None
    assert invalid.reason == "failed to parse or time chart"
    assert invalid.details["exception_class"] == "FileNotFoundError"
