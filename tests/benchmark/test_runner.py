from pathlib import Path
from unittest.mock import patch

import pretty_midi
import pytest

from src.benchmark.runner import run_score_midi, run_transcribe_and_score


def write_prediction(path: Path):
    midi = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    midi.instruments.append(drums)
    midi.write(str(path))


def test_run_score_midi_writes_summary(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    output = tmp_path / "out"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    write_prediction(predictions / "foo.mid")

    run_score_midi(charts, predictions, output, tolerance_ms=[50], align=True)

    assert (output / "summary.json").exists()


def test_run_score_midi_raises_on_missing_prediction(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    output = tmp_path / "out"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (charts / "bar.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    write_prediction(predictions / "foo.mid")
    # bar.mid is missing

    with pytest.raises(ValueError, match="corpus validation failed"):
        run_score_midi(charts, predictions, output, tolerance_ms=[50], align=True)


def test_transcribe_and_score_uses_injected_transcriber(tmp_path: Path):
    charts = tmp_path / "charts"
    audio = tmp_path / "audio"
    output = tmp_path / "out"
    charts.mkdir()
    audio.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (audio / "foo.wav").write_bytes(b"fake wav")

    def fake_transcribe(audio_path: Path) -> bytes:
        midi_path = tmp_path / "prediction.mid"
        write_prediction(midi_path)
        return midi_path.read_bytes()

    reports = run_transcribe_and_score(charts, audio, output, [50], transcribe=fake_transcribe)

    assert reports[0].chart_id == "foo"
    assert (output / "predictions" / "foo.mid").exists()


def test_transcribe_and_score_skips_missing_audio(tmp_path: Path):
    """Charts with missing audio are skipped; remaining charts are still scored."""
    charts = tmp_path / "charts"
    audio = tmp_path / "audio"
    output = tmp_path / "out"
    charts.mkdir()
    audio.mkdir()

    # Two charts, but only "found" has audio
    (charts / "found.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (charts / "missing.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (audio / "found.wav").write_bytes(b"fake wav")

    def fake_transcribe(audio_path: Path) -> bytes:
        return write_prediction_bytes()

    reports = run_transcribe_and_score(charts, audio, output, [50], transcribe=fake_transcribe)

    # "found" was transcribed and scored; "missing" was skipped
    assert (output / "predictions" / "found.mid").exists()
    assert not (output / "predictions" / "missing.mid").exists()
    # run_score_midi still ran for the available prediction
    assert any(r.chart_id == "found" for r in reports)


def test_transcribe_and_score_clears_stale_predictions_on_rerun(tmp_path: Path):
    """Rerunning transcribe-and-score into the same output_dir clears stale artefacts."""
    charts = tmp_path / "charts"
    audio = tmp_path / "audio"
    output = tmp_path / "out"
    charts.mkdir()
    audio.mkdir()

    # First run: two charts with audio
    (charts / "alpha.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (charts / "bravo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (audio / "alpha.wav").write_bytes(b"fake wav")
    (audio / "bravo.wav").write_bytes(b"fake wav")

    def fake_transcribe(audio_path: Path) -> bytes:
        return write_prediction_bytes()

    run_transcribe_and_score(charts, audio, output, [50], transcribe=fake_transcribe)
    assert (output / "predictions" / "alpha.mid").exists()
    assert (output / "predictions" / "bravo.mid").exists()

    # Remove bravo's audio to simulate a chart being skipped on rerun
    (audio / "bravo.wav").unlink()

    run_transcribe_and_score(charts, audio, output, [50], transcribe=fake_transcribe)

    # Stale bravo prediction should be gone; only alpha remains
    assert (output / "predictions" / "alpha.mid").exists()
    assert not (output / "predictions" / "bravo.mid").exists()


def test_default_transcribe_creates_single_transcriber(tmp_path: Path):
    """When no transcribe callback is provided, DrumTranscriber is constructed once."""
    charts = tmp_path / "charts"
    audio = tmp_path / "audio"
    output = tmp_path / "out"
    charts.mkdir()
    audio.mkdir()
    (charts / "a.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (charts / "b.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (audio / "a.wav").write_bytes(b"fake")
    (audio / "b.wav").write_bytes(b"fake")

    with patch("src.benchmark.runner._create_shared_transcriber") as mock_create:
        mock_transcriber = mock_create.return_value
        midi_bytes = write_prediction_bytes()
        mock_transcriber.transcribe.side_effect = lambda *a, **kw: _async_return(midi_bytes)
        reports = run_transcribe_and_score(charts, audio, output, [50])

    mock_create.assert_called_once()
    assert mock_transcriber.transcribe.call_count == 2
    assert len(reports) == 4  # 2 charts × 2 report types (raw + aligned)


def write_prediction_bytes() -> bytes:
    """Return a minimal prediction MIDI as raw bytes."""
    midi = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    midi.instruments.append(drums)

    import io

    buf = io.BytesIO()
    midi.write(buf)
    buf.seek(0)
    return buf.read()


async def _async_return(value):
    """Helper to make a coroutine that returns *value*."""
    return value


def test_run_score_midi_no_align_emits_only_raw_reports(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    output = tmp_path / "out"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    write_prediction(predictions / "foo.mid")

    reports = run_score_midi(charts, predictions, output, tolerance_ms=[50], align=False)

    modes = {report.mode for report in reports}
    assert modes == {"raw"}, f"expected only 'raw' mode, got {modes}"
    assert len(reports) == 1
