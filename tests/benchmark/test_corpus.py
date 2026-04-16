from pathlib import Path

from src.benchmark.corpus import discover_score_midi_items, validate_score_midi_corpus


def test_discovers_items_by_matching_basenames(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.txt").write_text("#BPM: 120\n", encoding="utf-8")
    (predictions / "foo.midi").write_bytes(b"MThd")

    items = discover_score_midi_items(charts, predictions)

    assert len(items) == 1
    assert items[0].chart_id == "foo"
    assert items[0].dtx_path == charts / "foo.txt"
    assert items[0].prediction_midi_path == predictions / "foo.midi"


def test_validation_reports_missing_prediction(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    charts.mkdir()
    predictions.mkdir()
    (charts / "missing.dtx").write_text("#BPM: 120\n", encoding="utf-8")

    result = validate_score_midi_corpus(charts, predictions)

    assert result.valid_items == []
    assert result.errors == ["missing prediction MIDI for chart_id missing"]
    assert result.is_valid is False


def test_validation_reports_duplicate_stems(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (charts / "foo.txt").write_text("#BPM: 120\n", encoding="utf-8")
    (predictions / "foo.mid").write_bytes(b"MThd")

    result = validate_score_midi_corpus(charts, predictions)

    assert result.valid_items == []
    assert result.errors == ["duplicate files for chart_id foo"]
    assert result.is_valid is False
