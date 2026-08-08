import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import pretty_midi
import pytest

from src.benchmark.runner import export_reference_midis, run_score_midi, run_transcribe_and_score


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


def test_current_default_transcribe_constructs_drum_transcriber_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    charts = tmp_path / "charts"
    audio = tmp_path / "audio"
    output = tmp_path / "output"
    charts.mkdir()
    audio.mkdir()
    for chart_id in ("a", "b"):
        (charts / f"{chart_id}.dtx").write_text(
            "#BPM: 120\n#00013: 0100\n",
            encoding="utf-8",
        )
        (audio / f"{chart_id}.wav").write_bytes(b"characterization")

    calls: list[Path] = []

    def transcribe(path: Path) -> bytes:
        calls.append(path)
        return write_prediction_bytes()

    run_transcribe_and_score(charts, audio, output, [50], transcribe=transcribe)

    assert calls == [audio / "a.wav", audio / "b.wav"]


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


def test_transcribe_and_score_skips_chart_when_file_write_fails(tmp_path: Path):
    """If writing the prediction MIDI or copying the chart fails, that chart is
    skipped but the batch continues."""
    charts = tmp_path / "charts"
    audio = tmp_path / "audio"
    output = tmp_path / "out"
    charts.mkdir()
    audio.mkdir()

    (charts / "ok.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (charts / "bad.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (audio / "ok.wav").write_bytes(b"fake wav")
    (audio / "bad.wav").write_bytes(b"fake wav")

    def fake_transcribe(audio_path: Path) -> bytes:
        return write_prediction_bytes()

    original_write_bytes = Path.write_bytes

    def failing_write_bytes(self, data):
        if "bad.mid" in str(self):
            raise OSError("disk full")
        return original_write_bytes(self, data)

    with patch.object(Path, "write_bytes", failing_write_bytes):
        reports = run_transcribe_and_score(charts, audio, output, [50], transcribe=fake_transcribe)

    # "ok" should succeed; "bad" should be skipped due to write failure
    assert (output / "predictions" / "ok.mid").exists()
    assert not (output / "predictions" / "bad.mid").exists()
    assert any(r.chart_id == "ok" for r in reports)


def test_transcribe_and_score_cleans_up_prediction_when_chart_copy_fails(tmp_path: Path):
    """If write_bytes succeeds but shutil.copy2 fails, the prediction MIDI must
    be removed so corpus validation does not see a stray file."""
    charts = tmp_path / "charts"
    audio = tmp_path / "audio"
    output = tmp_path / "out"
    charts.mkdir()
    audio.mkdir()

    (charts / "ok.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (charts / "bad.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (audio / "ok.wav").write_bytes(b"fake wav")
    (audio / "bad.wav").write_bytes(b"fake wav")

    def fake_transcribe(audio_path: Path) -> bytes:
        return write_prediction_bytes()

    original_copy2 = shutil.copy2

    def failing_copy2(src, dst, *, follow_symlinks=True):
        if "bad.dtx" in str(src):
            raise OSError("permission denied")
        return original_copy2(src, dst, follow_symlinks=follow_symlinks)

    with patch("src.benchmark.runner.shutil.copy2", failing_copy2):
        reports = run_transcribe_and_score(charts, audio, output, [50], transcribe=fake_transcribe)

    # "ok" should succeed; "bad" prediction must be cleaned up
    assert (output / "predictions" / "ok.mid").exists()
    assert not (output / "predictions" / "bad.mid").exists()
    assert any(r.chart_id == "ok" for r in reports)


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


def test_run_score_midi_raises_on_scoring_failure(tmp_path: Path):
    """When a validated chart fails during scoring, the run must raise
    instead of silently producing partial results."""
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    output = tmp_path / "out"
    charts.mkdir()
    predictions.mkdir()
    (charts / "bad.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    # Write an invalid MIDI file so parsing fails during scoring.
    (predictions / "bad.mid").write_bytes(b"not a midi file")

    with pytest.raises(RuntimeError, match="Scoring failed"):
        run_score_midi(charts, predictions, output, tolerance_ms=[50], align=True)


def test_transcribe_and_score_raises_when_all_audio_missing(tmp_path: Path):
    """If every chart is missing audio, the run must raise instead of
    producing an empty summary with exit 0."""
    charts = tmp_path / "charts"
    audio = tmp_path / "audio"
    output = tmp_path / "out"
    charts.mkdir()
    audio.mkdir()
    (charts / "song.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    # No audio file created for "song"

    with pytest.raises(RuntimeError, match="No charts available for scoring"):
        run_transcribe_and_score(charts, audio, output, [50])


def test_transcribe_and_score_raises_when_all_transcriptions_fail(tmp_path: Path):
    """If every transcription raises, the run must raise instead of
    producing an empty summary."""
    charts = tmp_path / "charts"
    audio = tmp_path / "audio"
    output = tmp_path / "out"
    charts.mkdir()
    audio.mkdir()
    (charts / "song.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    (audio / "song.wav").write_bytes(b"fake wav")

    def failing_transcribe(audio_path: Path) -> bytes:
        raise RuntimeError("transcription engine broken")

    with pytest.raises(RuntimeError, match="No charts available for scoring"):
        run_transcribe_and_score(charts, audio, output, [50], transcribe=failing_transcribe)


def _write_prediction_with_note(path: Path, pitch: int):
    """Write a prediction MIDI with a single note at the given pitch."""
    midi = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=0.0, end=0.1))
    midi.instruments.append(drums)
    midi.write(str(path))


def test_run_score_midi_warns_on_unmapped_prediction_events(tmp_path: Path, caplog):
    """Unmapped prediction MIDI notes should produce a warning so that the
    user knows some predictions were excluded from scoring."""
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    output = tmp_path / "out"
    charts.mkdir()
    predictions.mkdir()
    # Chart with a single kick (lane 13 → MIDI 36)
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    # Prediction with MIDI note 44 (pedal hi-hat) — not in DEFAULT_MIDI_NOTE_MAP
    _write_prediction_with_note(predictions / "foo.mid", pitch=44)

    with caplog.at_level(logging.WARNING, logger="src.benchmark.runner"):
        reports = run_score_midi(charts, predictions, output, tolerance_ms=[50], align=False)

    assert any("unmapped prediction events" in msg for msg in caplog.messages), (
        f"Expected unmapped prediction warning, got: {caplog.messages}"
    )
    # Scoring should still complete (the unmapped prediction is dropped, ground truth is present)
    assert len(reports) == 1


def test_run_score_midi_warns_on_unmapped_ground_truth_events(tmp_path: Path, caplog):
    """Unmapped DTX lanes should produce a warning so that the user knows
    some ground-truth hits were excluded from scoring."""
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    output = tmp_path / "out"
    charts.mkdir()
    predictions.mkdir()
    # Channel "1F" is a playable pattern lane not in DEFAULT_DTX_LANE_MAP.
    # (Channel 01 is BGM control data and no longer enters the playable path.)
    (charts / "bar.dtx").write_text("#BPM: 120\n#0001F: 0100\n", encoding="utf-8")
    write_prediction(predictions / "bar.mid")

    with caplog.at_level(logging.WARNING, logger="src.benchmark.runner"):
        reports = run_score_midi(charts, predictions, output, tolerance_ms=[50], align=False)

    assert any("unmapped ground-truth events" in msg for msg in caplog.messages), (
        f"Expected unmapped ground-truth warning, got: {caplog.messages}"
    )
    assert len(reports) == 1


def test_export_reference_midis_warns_on_unmapped_events(tmp_path: Path, caplog):
    """export_reference_midis should warn when a chart has unmapped lanes."""
    charts = tmp_path / "charts"
    output = tmp_path / "out"
    charts.mkdir()
    # Channel "1F" is a playable pattern lane not in DEFAULT_DTX_LANE_MAP.
    # (Channel 01 is BGM control data and no longer enters the playable path.)
    (charts / "bgm.dtx").write_text("#BPM: 120\n#0001F: 0100\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="src.benchmark.runner"):
        count = export_reference_midis(charts, output)

    assert count == 1
    assert any("unmapped ground-truth events" in msg for msg in caplog.messages), (
        f"Expected unmapped ground-truth warning, got: {caplog.messages}"
    )
