from pathlib import Path

from src.benchmark.corpus import discover_score_midi_items, validate_score_midi_corpus


def test_discovers_items_by_matching_basenames(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n", encoding="utf-8")
    (predictions / "foo.mid").write_bytes(b"MThd")

    items = discover_score_midi_items(charts, predictions)

    assert len(items) == 1
    assert items[0].chart_id == "foo"
    assert items[0].dtx_path == charts / "foo.dtx"
    assert items[0].prediction_midi_path == predictions / "foo.mid"


def test_validation_reports_missing_prediction(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    charts.mkdir()
    predictions.mkdir()
    (charts / "missing.dtx").write_text("#BPM: 120\n", encoding="utf-8")

    result = validate_score_midi_corpus(charts, predictions)

    assert result.valid_items == []
    assert result.errors == ["missing prediction MIDI for chart_id missing"]
