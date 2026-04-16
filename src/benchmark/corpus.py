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


def _by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            paths[path.stem] = path
    return paths


def discover_score_midi_items(charts_dir: Path, predictions_dir: Path) -> list[ScoreMidiItem]:
    return validate_score_midi_corpus(charts_dir, predictions_dir).valid_items


def validate_score_midi_corpus(charts_dir: Path, predictions_dir: Path) -> CorpusValidationResult:
    errors: list[str] = []
    valid_items: list[ScoreMidiItem] = []

    chart_paths = _by_stem(charts_dir, {".dtx", ".txt"})
    prediction_paths = _by_stem(predictions_dir, {".mid", ".midi"})

    for chart_id, dtx_path in chart_paths.items():
        prediction_path = prediction_paths.get(chart_id)
        if prediction_path is None:
            errors.append(f"missing prediction MIDI for chart_id {chart_id}")
            continue
        valid_items.append(ScoreMidiItem(chart_id, dtx_path, prediction_path))

    return CorpusValidationResult(valid_items=valid_items, errors=errors)
