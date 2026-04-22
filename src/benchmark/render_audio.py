from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.benchmark.dtx_parser import parse_dtx_file
from src.benchmark.models import BenchmarkEvent
from src.benchmark.prepare import _select_chart
from src.benchmark.timing import dtx_events_to_timed_events


@dataclass(frozen=True)
class ScheduledSamplePlacement:
    time_sec: float
    sample_name: str
    sample_path: Path
    lane_id: str
    note_id: str


@dataclass(frozen=True)
class RenderPlanItem:
    song_id: str
    raw_folder: Path
    selected_chart: Path
    selected_chart_level: str
    placements: list[ScheduledSamplePlacement]


@dataclass(frozen=True)
class InvalidRenderPlanItem:
    raw_folder: Path
    reason: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderPlanResult:
    valid_items: list[RenderPlanItem]
    invalid_items: list[InvalidRenderPlanItem]


def plan_render_corpus(raw_dir: Path) -> RenderPlanResult:
    valid_items: list[RenderPlanItem] = []
    invalid_items: list[InvalidRenderPlanItem] = []

    for song_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        plan, invalid = plan_render_song(song_dir)
        if invalid is not None:
            invalid_items.append(invalid)
            continue
        if plan is not None:
            valid_items.append(plan)

    return RenderPlanResult(valid_items=valid_items, invalid_items=invalid_items)


def plan_render_song(song_dir: Path) -> tuple[RenderPlanItem | None, InvalidRenderPlanItem | None]:
    selected_chart, invalid = _select_chart(song_dir)
    if invalid is not None:
        return None, InvalidRenderPlanItem(
            raw_folder=invalid.raw_folder,
            reason=invalid.reason,
            details=invalid.details,
        )
    assert selected_chart is not None
    return _plan_render_selection(
        song_dir=song_dir,
        selected_chart=selected_chart[0],
        selected_chart_level=selected_chart[1],
    )


def _plan_render_selection(
    song_dir: Path,
    selected_chart: Path,
    selected_chart_level: str,
) -> tuple[RenderPlanItem | None, InvalidRenderPlanItem | None]:
    try:
        chart = parse_dtx_file(selected_chart, chart_id=song_dir.name)
        timed_events = dtx_events_to_timed_events(chart)
    except (OSError, ValueError) as exc:
        return None, InvalidRenderPlanItem(
            raw_folder=song_dir,
            reason="failed to parse or time chart",
            details={
                "exception_class": exc.__class__.__name__,
                "message": str(exc),
            },
        )
    placements: list[ScheduledSamplePlacement] = []
    missing_note_ids: list[str] = []
    missing_sample_names: list[str] = []
    missing_event_metadata: list[str] = []

    for event in timed_events:
        placement, missing_note_id, missing_sample_name, missing_metadata = _plan_sample_placement(
            song_dir, chart.wav_table, event
        )
        if missing_metadata is not None:
            _append_unique(missing_event_metadata, missing_metadata)
            continue
        if missing_note_id is not None:
            _append_unique(missing_note_ids, missing_note_id)
            continue
        if missing_sample_name is not None:
            _append_unique(missing_sample_names, missing_sample_name)
            continue
        assert placement is not None
        placements.append(placement)

    if missing_note_ids or missing_sample_names or missing_event_metadata:
        return None, InvalidRenderPlanItem(
            raw_folder=song_dir,
            reason="missing chip/sample references",
            details={
                "missing_note_ids": missing_note_ids,
                "missing_sample_names": missing_sample_names,
                "missing_event_metadata": missing_event_metadata,
            },
        )

    return (
        RenderPlanItem(
            song_id=song_dir.name,
            raw_folder=song_dir,
            selected_chart=selected_chart,
            selected_chart_level=selected_chart_level,
            placements=placements,
        ),
        None,
    )


def _plan_sample_placement(
    song_dir: Path, wav_table: dict[str, str], event: BenchmarkEvent
) -> tuple[ScheduledSamplePlacement | None, str | None, str | None, str | None]:
    note_id_value = event.metadata.get("note_id")
    if note_id_value is None:
        return None, None, None, "note_id"

    note_id = str(note_id_value).upper()
    sample_name = wav_table.get(note_id)
    if sample_name is None:
        return None, note_id, None, None

    sample_path = _resolve_sample_path(song_dir, sample_name)
    if sample_path is None:
        return None, None, sample_name, None

    return (
        ScheduledSamplePlacement(
            time_sec=event.time_sec,
            sample_name=sample_name,
            sample_path=sample_path,
            lane_id=str(event.metadata.get("lane_id", "")).upper(),
            note_id=note_id,
        ),
        None,
        None,
        None,
    )


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _resolve_sample_path(song_dir: Path, sample_name: str) -> Path | None:
    song_dir_resolved = song_dir.resolve()
    sample_path = (song_dir / sample_name).resolve(strict=False)
    try:
        sample_path.relative_to(song_dir_resolved)
    except ValueError:
        return None
    if not sample_path.is_file():
        return None
    return sample_path
