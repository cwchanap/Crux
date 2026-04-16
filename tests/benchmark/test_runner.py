from pathlib import Path

import pretty_midi

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
