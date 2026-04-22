from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import src.benchmark.render_audio as render_audio
from src.benchmark.models import BenchmarkEvent
from src.benchmark.render_audio import (
    plan_render_corpus,
    plan_render_song,
    render_plan_item,
)


def _write_sample(path: Path, samples: list[float], sample_rate: int = 8000) -> None:
    sf.write(
        path,
        np.asarray(samples, dtype=np.float32),
        sample_rate,
        format="WAV",
        subtype="FLOAT",
    )


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    return np.asarray(audio, dtype=np.float32), sample_rate


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
    _write_sample(song / "kick.wav", [1.0, 0.0, 0.0, 0.0])
    _write_sample(song / "snare.wav", [0.0, 1.0, 0.0, 0.0])

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
    _write_sample(good_song / "kick.wav", [1.0, 0.0, 0.0, 0.0])
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


def test_plan_render_song_rejects_unreadable_audio_sample_files(tmp_path: Path):
    song = tmp_path / "Song Four B"
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
    (song / "kick.wav").write_text("not audio", encoding="utf-8")

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


def test_render_plan_item_writes_sample_at_expected_location(tmp_path: Path):
    song = tmp_path / "Song Ten"
    song.mkdir()
    sample = song / "kick.wav"
    _write_sample(sample, [1.0, 0.0, 0.0, 0.0])
    plan = render_audio.RenderPlanItem(
        song_id="Song Ten",
        raw_folder=song,
        selected_chart=song / "mas.dtx",
        selected_chart_level="mas",
        placements=[
            render_audio.ScheduledSamplePlacement(
                time_sec=1.0,
                sample_name="kick.wav",
                sample_path=sample,
                lane_id="11",
                note_id="01",
                volume_scale=1.0,
            )
        ],
    )
    output = song / "out.wav"

    render_plan_item(plan, output)

    audio, sample_rate = _read_audio(output)
    assert sample_rate == 8000
    assert len(audio) == 8004
    assert np.isclose(audio[8000], 1.0)
    assert np.allclose(audio[:8000], 0.0)
    assert np.allclose(audio[8001:], 0.0)


def test_render_plan_item_mix_overlapping_hits_additively(tmp_path: Path):
    song = tmp_path / "Song Eleven"
    song.mkdir()
    sample = song / "kick.wav"
    _write_sample(sample, [1.0, 0.0, 0.0, 0.0])
    plan = render_audio.RenderPlanItem(
        song_id="Song Eleven",
        raw_folder=song,
        selected_chart=song / "mas.dtx",
        selected_chart_level="mas",
        placements=[
            render_audio.ScheduledSamplePlacement(
                time_sec=1.0,
                sample_name="kick.wav",
                sample_path=sample,
                lane_id="11",
                note_id="01",
                volume_scale=1.0,
            ),
            render_audio.ScheduledSamplePlacement(
                time_sec=1.0,
                sample_name="kick.wav",
                sample_path=sample,
                lane_id="11",
                note_id="01",
                volume_scale=1.0,
            ),
        ],
    )
    output = song / "out.wav"

    render_plan_item(plan, output)

    audio, _ = _read_audio(output)
    assert np.isclose(audio[8000], 2.0)


def test_render_plan_item_applies_volume_scaling(tmp_path: Path):
    song = tmp_path / "Song Twelve"
    song.mkdir()
    sample = song / "kick.wav"
    _write_sample(sample, [1.0, 0.0, 0.0, 0.0])
    (song / "mas.dtx").write_text(
        "\n".join(
            [
                "#BPM: 240",
                "#VOLUME01: 50",
                "#WAV01: kick.wav",
                "#00111: 01",
            ]
        ),
        encoding="utf-8",
    )
    plan, invalid = plan_render_song(song)
    assert invalid is None
    assert plan is not None
    output = song / "out.wav"

    render_plan_item(plan, output)

    audio, _ = _read_audio(output)
    assert np.isclose(audio[8000], 0.5)


def test_render_plan_item_preserves_stereo_samples(tmp_path: Path):
    song = tmp_path / "Song Thirteen"
    song.mkdir()
    sample = song / "stereo.wav"
    _write_sample(sample, [[1.0, 0.0], [0.0, 1.0]])
    plan = render_audio.RenderPlanItem(
        song_id="Song Thirteen",
        raw_folder=song,
        selected_chart=song / "mas.dtx",
        selected_chart_level="mas",
        placements=[
            render_audio.ScheduledSamplePlacement(
                time_sec=1.0,
                sample_name="stereo.wav",
                sample_path=sample,
                lane_id="11",
                note_id="01",
                volume_scale=1.0,
            )
        ],
    )
    output = song / "out.wav"

    render_plan_item(plan, output)

    audio, sample_rate = _read_audio(output)
    assert sample_rate == 8000
    assert audio.shape == (8002, 2)
    assert np.allclose(audio[:8000], 0.0)
    assert np.isclose(audio[8000, 0], 1.0)
    assert np.isclose(audio[8000, 1], 0.0)
    assert np.isclose(audio[8001, 0], 0.0)
    assert np.isclose(audio[8001, 1], 1.0)
