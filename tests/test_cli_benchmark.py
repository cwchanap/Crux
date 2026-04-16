from pathlib import Path

import pretty_midi
from click.testing import CliRunner

from src.cli.main import main


def write_prediction(path: Path):
    midi = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    midi.instruments.append(drums)
    midi.write(str(path))


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
