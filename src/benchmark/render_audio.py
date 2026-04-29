from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from src.benchmark.dtx_parser import parse_dtx_file
from src.benchmark.mapping import DEFAULT_DTX_LANE_MAP
from src.benchmark.models import BenchmarkEvent
from src.benchmark.prepare import _select_chart
from src.benchmark.timing import dtx_events_to_timed_events

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledSamplePlacement:
    time_sec: float
    sample_name: str
    sample_path: Path
    lane_id: str
    note_id: str
    volume_scale: float = 1.0


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


@dataclass(frozen=True)
class RenderedAudioItem:
    song_id: str
    raw_folder: Path
    selected_chart: Path
    selected_chart_level: str
    audio_path: Path
    render_path: Path


@dataclass(frozen=True)
class RenderAudioResult:
    valid_items: list[RenderedAudioItem]
    invalid_items: list[InvalidRenderPlanItem]
    manifest_path: Path
    invalid_report_path: Path


def render_audio_corpus(raw_dir: Path, output_dir: Path) -> RenderAudioResult:
    plan_result = plan_render_corpus(raw_dir)
    return _render_plans(plan_result.valid_items, plan_result.invalid_items, output_dir)


def render_audio_song(song_dir: Path, output_dir: Path) -> RenderAudioResult:
    plan, invalid = plan_render_song(song_dir)
    valid_plans = [plan] if plan is not None else []
    invalid_items = [invalid] if invalid is not None else []
    return _render_plans(valid_plans, invalid_items, output_dir)


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


def render_plan_item(plan_item: RenderPlanItem, output_path: Path) -> Path:
    if not plan_item.placements:
        sample_rate = 44100
        rendered_audio = np.zeros(1, dtype=np.float32)
    else:
        rendered_audio, sample_rate = _render_placements(plan_item.placements)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, rendered_audio, sample_rate, format="WAV", subtype="FLOAT")
    return output_path


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
        lane_id = str(event.metadata.get("lane_id", "")).upper()
        if lane_id not in DEFAULT_DTX_LANE_MAP:
            continue
        placement, missing_note_id, missing_sample_name, missing_metadata = _plan_sample_placement(
            song_dir, chart.wav_table, chart.volume_table, event
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
    if not placements:
        return None, InvalidRenderPlanItem(
            raw_folder=song_dir,
            reason="no renderable note events",
            details={},
        )

    # Validate sample rate consistency
    sample_rate_check = _validate_placement_sample_rates(placements)
    if sample_rate_check is not None:
        return None, InvalidRenderPlanItem(
            raw_folder=song_dir,
            reason=sample_rate_check["reason"],
            details=sample_rate_check["details"],
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
    song_dir: Path, wav_table: dict[str, str], volume_table: dict[str, float], event: BenchmarkEvent
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

    volume_scale = volume_table.get(note_id, 100.0) / 100.0

    return (
        ScheduledSamplePlacement(
            time_sec=event.time_sec,
            sample_name=sample_name,
            sample_path=sample_path,
            lane_id=str(event.metadata.get("lane_id", "")).upper(),
            note_id=note_id,
            volume_scale=volume_scale,
        ),
        None,
        None,
        None,
    )


def _validate_placement_sample_rates(
    placements: list[ScheduledSamplePlacement],
) -> dict[str, object] | None:
    if not placements:
        return None

    sample_rate: int | None = None
    for placement in placements:
        try:
            info = sf.info(placement.sample_path)
            placement_sample_rate = info.samplerate
        except Exception as exc:
            return {
                "reason": "sample rate validation failed",
                "details": {
                    "message": f"could not read sample rate from {placement.sample_path.name}: {exc}",
                },
            }

        if sample_rate is None:
            sample_rate = placement_sample_rate
        elif sample_rate != placement_sample_rate:
            return {
                "reason": "sample rate mismatch",
                "details": {
                    "message": f"sample rate mismatch: expected {sample_rate} Hz but "
                    f"{placement.sample_path.name} has {placement_sample_rate} Hz",
                },
            }
    return None


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _resolve_sample_path(song_dir: Path, sample_name: str) -> Path | None:
    song_dir_resolved = song_dir.resolve()
    sample_path = (song_dir / sample_name).resolve(strict=False)
    try:
        sample_path.relative_to(song_dir_resolved)
    except ValueError:
        logger.warning(
            "Sample path escapes song directory (song_dir=%s, sample=%s); skipping",
            song_dir_resolved,
            sample_path,
        )
        return None
    if not sample_path.is_file():
        return None
    try:
        sf.info(sample_path)
    except (OSError, RuntimeError, ValueError, sf.LibsndfileError) as exc:
        logger.debug(
            "Sample file exists but is unreadable (path=%s): %s",
            sample_path,
            exc,
        )
        return None
    return sample_path


def _render_plans(
    valid_plans: list[RenderPlanItem],
    invalid_items: list[InvalidRenderPlanItem],
    output_dir: Path,
) -> RenderAudioResult:
    rendered_items: list[RenderedAudioItem] = []
    final_invalid_items = list(invalid_items)

    for plan in valid_plans:
        try:
            rendered_items.append(_render_outputs_for_plan(plan, output_dir))
        except (OSError, RuntimeError, ValueError) as exc:
            final_invalid_items.append(
                InvalidRenderPlanItem(
                    raw_folder=plan.raw_folder,
                    reason="failed to render audio",
                    details={
                        "exception_class": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
            )

    manifest_path, invalid_report_path = _write_render_reports(
        rendered_items, final_invalid_items, output_dir
    )
    return RenderAudioResult(
        valid_items=rendered_items,
        invalid_items=final_invalid_items,
        manifest_path=manifest_path,
        invalid_report_path=invalid_report_path,
    )


def _render_outputs_for_plan(plan_item: RenderPlanItem, output_dir: Path) -> RenderedAudioItem:
    audio_path = output_dir / "audio" / f"{plan_item.song_id}.wav"
    render_path = output_dir / "renders" / f"{plan_item.song_id}.wav"
    render_plan_item(plan_item, audio_path)
    render_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audio_path, render_path)
    return RenderedAudioItem(
        song_id=plan_item.song_id,
        raw_folder=plan_item.raw_folder,
        selected_chart=plan_item.selected_chart,
        selected_chart_level=plan_item.selected_chart_level,
        audio_path=audio_path,
        render_path=render_path,
    )


def _write_render_reports(
    valid_items: list[RenderedAudioItem],
    invalid_items: list[InvalidRenderPlanItem],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    invalid_report_path = output_dir / "invalid.json"
    manifest_entries = [
        {
            "audio_path": str(item.audio_path),
            "raw_folder": str(item.raw_folder),
            "render_path": str(item.render_path),
            "selected_chart": item.selected_chart.name,
            "selected_chart_level": item.selected_chart_level,
            "song_id": item.song_id,
        }
        for item in valid_items
    ]
    invalid_entries = [
        {
            "details": item.details,
            "raw_folder": str(item.raw_folder),
            "reason": item.reason,
        }
        for item in invalid_items
    ]
    manifest_path.write_text(
        json.dumps({"items": manifest_entries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    invalid_report_path.write_text(
        json.dumps({"items": invalid_entries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path, invalid_report_path


def _render_placements(placements: list[ScheduledSamplePlacement]) -> tuple[np.ndarray, int]:
    sample_rate: int | None = None
    cached_samples: dict[Path, np.ndarray] = {}
    max_frame = 0
    output_channels = 1

    for placement in placements:
        if placement.sample_path not in cached_samples:
            sample, placement_sample_rate = _load_sample(placement.sample_path)
            cached_samples[placement.sample_path] = sample
            if sample_rate is None:
                sample_rate = placement_sample_rate
            elif sample_rate != placement_sample_rate:
                raise ValueError(
                    f"sample rate mismatch: expected {sample_rate} Hz but "
                    f"{placement.sample_path.name} has {placement_sample_rate} Hz"
                )
            output_channels = max(output_channels, sample.shape[1])
        else:
            sample = cached_samples[placement.sample_path]

        start_frame = max(0, int(round(placement.time_sec * sample_rate)))
        max_frame = max(max_frame, start_frame + sample.shape[0])

    assert sample_rate is not None
    if output_channels == 1:
        rendered_audio = np.zeros(max(1, max_frame), dtype=np.float32)
    else:
        rendered_audio = np.zeros((max(1, max_frame), output_channels), dtype=np.float32)

    for placement in placements:
        sample = cached_samples[placement.sample_path]
        start_frame = max(0, int(round(placement.time_sec * sample_rate)))
        end_frame = start_frame + sample.shape[0]
        if output_channels == 1:
            rendered_audio[start_frame:end_frame] += sample[:, 0] * placement.volume_scale
            continue
        if sample.shape[1] == 1:
            sample_to_mix = np.repeat(sample, output_channels, axis=1)
        elif sample.shape[1] == output_channels:
            sample_to_mix = sample
        else:
            raise ValueError("sample channel count mismatch across rendered placements")
        rendered_audio[start_frame:end_frame, :] += sample_to_mix * placement.volume_scale

    return rendered_audio, sample_rate


def _load_sample(sample_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(sample_path, always_2d=True, dtype="float32")
    return np.asarray(audio, dtype=np.float32), sample_rate
