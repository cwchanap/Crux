from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

CHART_LEVEL_PRIORITY = ("mas", "ext", "adv", "bas")
DRUM_AUDIO_FILENAMES = ("2 Drums.mp3", "drum.mp3")
CHART_SUFFIXES = {".dtx", ".txt"}


@dataclass(frozen=True)
class PreparedCorpusSelection:
    song_id: str
    raw_folder: Path
    selected_chart: Path
    selected_chart_level: str
    selected_audio: Path


@dataclass(frozen=True)
class InvalidPreparedCorpusItem:
    raw_folder: Path
    reason: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PrepareCorpusScanResult:
    valid_items: list[PreparedCorpusSelection]
    invalid_items: list[InvalidPreparedCorpusItem]


@dataclass(frozen=True)
class PrepareCorpusResult:
    valid_items: list[PreparedCorpusSelection]
    invalid_items: list[InvalidPreparedCorpusItem]
    manifest_path: Path
    invalid_report_path: Path


def scan_raw_corpus(raw_dir: Path) -> PrepareCorpusScanResult:
    valid_items: list[PreparedCorpusSelection] = []
    invalid_items: list[InvalidPreparedCorpusItem] = []

    for song_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        selection, invalid = _select_song_folder(song_dir)
        if invalid is not None:
            invalid_items.append(invalid)
            continue
        if selection is not None:
            valid_items.append(selection)

    return PrepareCorpusScanResult(valid_items=valid_items, invalid_items=invalid_items)


def prepare_corpus(raw_dir: Path, output_dir: Path) -> PrepareCorpusResult:
    scan_result = scan_raw_corpus(raw_dir)
    charts_dir = output_dir / "charts"
    audio_dir = output_dir / "audio"
    # Clear stale files from previous runs so removed/invalid songs don't
    # leak into downstream commands that discover inputs by scanning these
    # directories rather than reading manifest.json.
    if charts_dir.exists():
        shutil.rmtree(charts_dir)
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []
    prepared_items: list[PreparedCorpusSelection] = []
    copy_failures: list[InvalidPreparedCorpusItem] = []
    for item in scan_result.valid_items:
        parsed_chart_path = charts_dir / f"{item.song_id}{item.selected_chart.suffix.lower()}"
        parsed_audio_path = audio_dir / f"{item.song_id}{item.selected_audio.suffix.lower()}"
        try:
            shutil.copy2(item.selected_chart, parsed_chart_path)
            shutil.copy2(item.selected_audio, parsed_audio_path)
        except OSError as exc:
            # Remove partially copied chart so it doesn't surface as an
            # orphan with missing audio in downstream directory-scanning.
            if parsed_chart_path.exists():
                parsed_chart_path.unlink(missing_ok=True)
            copy_failures.append(
                InvalidPreparedCorpusItem(
                    raw_folder=item.raw_folder,
                    reason="failed to copy corpus files",
                    details={"exception": str(exc)},
                )
            )
            continue
        prepared_items.append(item)
        manifest_entries.append(
            {
                "song_id": item.song_id,
                "raw_folder": str(item.raw_folder),
                "selected_chart": item.selected_chart.name,
                "selected_chart_level": item.selected_chart_level,
                "selected_audio": item.selected_audio.name,
                "parsed_chart_path": str(parsed_chart_path),
                "parsed_audio_path": str(parsed_audio_path),
            }
        )

    all_invalid_items = list(scan_result.invalid_items) + copy_failures
    invalid_entries = [
        {
            "raw_folder": str(item.raw_folder),
            "reason": item.reason,
            "details": item.details,
        }
        for item in all_invalid_items
    ]

    manifest_path = output_dir / "manifest.json"
    invalid_report_path = output_dir / "invalid.json"
    manifest_path.write_text(
        json.dumps({"items": manifest_entries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    invalid_report_path.write_text(
        json.dumps({"items": invalid_entries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return PrepareCorpusResult(
        valid_items=prepared_items,
        invalid_items=all_invalid_items,
        manifest_path=manifest_path,
        invalid_report_path=invalid_report_path,
    )


def _select_song_folder(
    song_dir: Path,
) -> tuple[PreparedCorpusSelection | None, InvalidPreparedCorpusItem | None]:
    selected_chart, invalid = _select_chart(song_dir)
    if invalid is not None:
        return None, invalid

    selected_audio, invalid = _select_audio(song_dir)
    if invalid is not None:
        return None, invalid

    assert selected_chart is not None
    assert selected_audio is not None
    return (
        PreparedCorpusSelection(
            song_id=song_dir.name,
            raw_folder=song_dir,
            selected_chart=selected_chart[0],
            selected_chart_level=selected_chart[1],
            selected_audio=selected_audio,
        ),
        None,
    )


def _select_chart(
    song_dir: Path,
) -> tuple[tuple[Path, str] | None, InvalidPreparedCorpusItem | None]:
    chart_paths_by_level: dict[str, list[Path]] = {level: [] for level in CHART_LEVEL_PRIORITY}
    for path in sorted(song_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in CHART_SUFFIXES:
            stem = path.stem.lower()
            if stem in chart_paths_by_level:
                chart_paths_by_level[stem].append(path)

    for level in CHART_LEVEL_PRIORITY:
        candidates = chart_paths_by_level[level]
        if len(candidates) > 1:
            return None, InvalidPreparedCorpusItem(
                raw_folder=song_dir,
                reason="multiple chart files for selected level",
                details={"level": level, "candidates": [path.name for path in candidates]},
            )
        if len(candidates) == 1:
            return (candidates[0], level), None

    return None, InvalidPreparedCorpusItem(
        raw_folder=song_dir,
        reason="no recognized chart file",
        details={"expected_levels": list(CHART_LEVEL_PRIORITY)},
    )


def _select_audio(song_dir: Path) -> tuple[Path | None, InvalidPreparedCorpusItem | None]:
    allowed_names = {name.lower() for name in DRUM_AUDIO_FILENAMES}
    candidates = [
        path
        for path in sorted(song_dir.iterdir())
        if path.is_file() and path.name.lower() in allowed_names
    ]
    if len(candidates) > 1:
        return None, InvalidPreparedCorpusItem(
            raw_folder=song_dir,
            reason="multiple allowed drum audio files",
            details={"candidates": [path.name for path in candidates]},
        )
    if not candidates:
        return None, InvalidPreparedCorpusItem(
            raw_folder=song_dir,
            reason="missing allowed drum audio file",
            details={"allowed_filenames": list(DRUM_AUDIO_FILENAMES)},
        )
    return candidates[0], None
