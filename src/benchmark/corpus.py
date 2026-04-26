from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScoreMidiItem:
    chart_id: str
    dtx_path: Path
    prediction_midi_path: Path


@dataclass(frozen=True)
class CorpusValidationResult:
    valid_items: list[ScoreMidiItem]
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _by_stem(directory: Path, suffixes: set[str]) -> tuple[dict[str, Path], list[str]]:
    if not directory.exists() or not directory.is_dir():
        return {}, [f"directory not found: {directory}"]

    paths: dict[str, Path] = {}
    errors: list[str] = []
    duplicate_stems: set[str] = set()
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            stem_lower = path.stem.lower()
            if stem_lower in paths:
                duplicate_stems.add(stem_lower)
                errors.append(f"duplicate files for chart_id {stem_lower}")
                continue
            paths[stem_lower] = path

    for stem in duplicate_stems:
        paths.pop(stem, None)

    return paths, errors


def discover_score_midi_items(charts_dir: Path, predictions_dir: Path) -> list[ScoreMidiItem]:
    return validate_score_midi_corpus(charts_dir, predictions_dir).valid_items


def validate_score_midi_corpus(charts_dir: Path, predictions_dir: Path) -> CorpusValidationResult:
    errors: list[str] = []
    valid_items: list[ScoreMidiItem] = []

    chart_paths, chart_errors = _by_stem(charts_dir, {".dtx", ".txt"})
    prediction_paths, prediction_errors = _by_stem(predictions_dir, {".mid", ".midi"})
    errors.extend(chart_errors)
    errors.extend(prediction_errors)

    for chart_id, dtx_path in chart_paths.items():
        prediction_path = prediction_paths.get(chart_id)
        if prediction_path is None:
            errors.append(f"missing prediction MIDI for chart_id {chart_id}")
            continue
        valid_items.append(ScoreMidiItem(chart_id, dtx_path, prediction_path))

    chart_stems = set(chart_paths.keys())
    for prediction_id in prediction_paths:
        if prediction_id not in chart_stems:
            errors.append(f"stray prediction MIDI with no matching chart: {prediction_id}")

    return CorpusValidationResult(valid_items=valid_items, errors=errors)
