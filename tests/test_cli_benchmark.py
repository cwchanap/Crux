from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from click.testing import CliRunner

from src.cli.main import main


def write_prediction(path: Path):
    midi = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    midi.instruments.append(drums)
    midi.write(str(path))


def _write_render_song(song: Path, chart_name: str = "mas.dtx") -> None:
    song.mkdir(parents=True)
    (song / chart_name).write_text(
        "\n".join(["#BPM: 120", "#WAV01: drum.mp3", "#00111: 01"]),
        encoding="utf-8",
    )
    # Write a valid WAV container to the configured sample filename.
    sf.write(
        song / "drum.mp3",
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        8000,
        format="WAV",
        subtype="FLOAT",
    )


def test_prepare_corpus_command_writes_parsed_files(tmp_path: Path):
    raw = tmp_path / "raw"
    song = raw / "Soukyuu e no shouka"
    output = tmp_path / "parsed"
    song.mkdir(parents=True)
    (song / "adv.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "mas.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "2 Drums.mp3").write_bytes(b"drums")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "prepare-corpus",
            "--raw-dir",
            str(raw),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert (output / "charts" / "Soukyuu e no shouka.dtx").exists()
    assert (output / "audio" / "Soukyuu e no shouka.mp3").exists()
    assert (output / "manifest.json").exists()


def test_prepare_corpus_defaults_output_dir_from_raw_dir_name(tmp_path: Path, monkeypatch):
    raw = tmp_path / "Test DTX"
    song = raw / "Soukyuu e no shouka"
    expected_output = tmp_path / "artifacts" / "benchmark" / "Test DTX"
    monkeypatch.chdir(tmp_path)
    song.mkdir(parents=True)
    (song / "mas.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (song / "drum.mp3").write_bytes(b"drums")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "prepare-corpus",
            "--raw-dir",
            str(raw),
        ],
    )

    assert result.exit_code == 0
    assert (expected_output / "charts" / "Soukyuu e no shouka.dtx").exists()
    assert (expected_output / "audio" / "Soukyuu e no shouka.mp3").exists()
    assert (expected_output / "manifest.json").exists()


def test_render_audio_command_writes_one_song_from_song_dir(tmp_path: Path):
    song = tmp_path / "Song Render"
    output = tmp_path / "rendered"
    _write_render_song(song)

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "render-audio",
            "--song-dir",
            str(song),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert (output / "audio" / "Song Render.wav").exists()


def test_render_audio_command_writes_batch_from_raw_dir(tmp_path: Path):
    raw = tmp_path / "raw"
    output = tmp_path / "rendered"
    _write_render_song(raw / "Song A")
    _write_render_song(raw / "Song B")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "render-audio",
            "--raw-dir",
            str(raw),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert (output / "audio" / "Song A.wav").exists()
    assert (output / "audio" / "Song B.wav").exists()


def test_render_audio_defaults_output_dir_from_input_dir_name(tmp_path: Path, monkeypatch):
    raw = tmp_path / "Test DTX"
    monkeypatch.chdir(tmp_path)
    _write_render_song(raw / "Song Default")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "render-audio",
            "--raw-dir",
            str(raw),
        ],
    )

    assert result.exit_code == 0
    assert (
        tmp_path / "artifacts" / "benchmark" / "Test DTX" / "audio" / "Song Default.wav"
    ).exists()


def test_score_midi_command_runs(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    output = tmp_path / "out"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    write_prediction(predictions / "foo.mid")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "score-midi",
            "--charts-dir",
            str(charts),
            "--predictions-dir",
            str(predictions),
            "--output-dir",
            str(output),
            "--tolerance-ms",
            "50",
        ],
    )

    assert result.exit_code == 0
    assert (output / "summary.json").exists()


def test_score_midi_defaults_output_dir_from_run_name(tmp_path: Path, monkeypatch):
    charts = tmp_path / "parsed" / "charts"
    predictions = tmp_path / "parsed" / "predictions"
    expected_output = tmp_path / "artifacts" / "benchmark" / "soukyuu-stem-test"
    monkeypatch.chdir(tmp_path)
    charts.mkdir(parents=True)
    predictions.mkdir(parents=True)
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    write_prediction(predictions / "foo.mid")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "score-midi",
            "--charts-dir",
            str(charts),
            "--predictions-dir",
            str(predictions),
            "--run-name",
            "soukyuu-stem-test",
            "--tolerance-ms",
            "50",
        ],
    )

    assert result.exit_code == 0
    assert (expected_output / "summary.json").exists()


def test_validate_corpus_reports_missing_prediction(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "validate-corpus",
            "--charts-dir",
            str(charts),
            "--predictions-dir",
            str(predictions),
        ],
    )

    assert result.exit_code == 1
    assert "missing prediction MIDI" in result.output


def test_export_reference_midi_command_writes_files(tmp_path: Path):
    charts = tmp_path / "charts"
    output = tmp_path / "out"
    charts.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "export-reference-midi",
            "--charts-dir",
            str(charts),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert (output / "foo.mid").exists()


def test_inspect_dtx_outputs_counts(tmp_path: Path):
    chart = tmp_path / "foo.dtx"
    chart.write_text("#BPM: 120\n#BPM01: 180\n#00008: 0100\n#00013: 0100\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["benchmark", "inspect-dtx", str(chart)])

    assert result.exit_code == 0
    assert "chart_id: foo" in result.output
    assert "events: 1" in result.output
    assert "bpm_events: 1" in result.output
