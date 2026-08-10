from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import soundfile as sf

from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.corpus_manifest import ManifestRowView
from src.benchmark.dtx_parser import DtxBgmEvent, ParsedDtxChart
from src.benchmark.models import DtxEvent
from src.benchmark.r2_corpus_models import (
    CACHE_PROFILE,
    MAX_SIMFILE_ID,
    ProvenanceRecord,
    RemoteObject,
    SimfileInventory,
)
from src.benchmark.reference_chart_manifest import ReferenceChartRowView
from src.benchmark.reference_timing import (
    REFERENCE_EVENT_SCHEMA,
    TIMING_REASON_CODES,
    AudioRelativeReference,
    BgmReferenceGroup,
    BgmReferenceSet,
    BgmResolution,
    NativeReferenceEvent,
    SourceAudioInfo,
    TimingReasonCode,
    build_audio_relative_events,
    inspect_source_audio,
    publish_immutable_content,
    read_reference_events,
    render_reference_events,
    resolve_bgm_reference_groups,
    select_bgm_reference,
    validate_schema_golden,
)
from src.benchmark.timing import DtxTimingMap, build_dtx_timing_map

_FIXED_TIME = datetime(2026, 8, 5, tzinfo=timezone.utc)
_CHART_ID = "42/real.dtx"


def _remote(key: str, *, simfile_id: int = 42) -> RemoteObject:
    body = f"body-{key}".encode()
    digest = sha256(body).hexdigest()
    return RemoteObject(
        key=f"{simfile_id}/{key}",
        size=len(body),
        etag=f"etag-{key}",
        etag_is_weak=False,
        last_modified=_FIXED_TIME,
        content_type="application/octet-stream",
        cache_status="verified",
        sha256=digest,
        cache_path=f"sha256/{digest[:2]}/{digest}",
    )


def _row_view(
    objects: tuple[RemoteObject, ...],
    *,
    simfile_id: int = 42,
    object_prefix: str | None = None,
) -> ReferenceChartRowView:
    prefix = object_prefix if object_prefix is not None else f"{simfile_id}/"
    inventory = SimfileInventory(simfile_id, prefix, objects, "complete")
    source = ManifestRowView(
        inventory=inventory,
        provenance=ProvenanceRecord(),
        corpus_version="sha256:" + "a" * 64,
        cache_profile=CACHE_PROFILE,
        source_endpoint_sha256="f" * 64,
        source_bucket="simfile-dtx",
        source_discovery_method="r2_list_objects_v2",
    )
    selected = objects[0] if objects else None
    return ReferenceChartRowView(
        source=source,
        corpus_version="sha256:" + "b" * 64,
        selection_status="selected",
        selection_reason_codes=(),
        selection_warnings=(),
        selected_chart=selected,
        selected_chart_content_hash=getattr(selected, "sha256", None),
    )


def _chart(
    bgm_events: list[DtxBgmEvent],
    wav_table: dict[str, str],
    *,
    chart_id: str = _CHART_ID,
) -> ParsedDtxChart:
    return ParsedDtxChart(chart_id=chart_id, bgm_events=bgm_events, wav_table=wav_table)


def _bgm(measure: int, position: float, note_id: str, *, order: int = 0) -> DtxBgmEvent:
    return DtxBgmEvent(_CHART_ID, measure, position, note_id, order)


# ---------------------------------------------------------------------------
# Closed reason-code contract
# ---------------------------------------------------------------------------


def test_timing_reason_codes_matches_the_literal_arguments() -> None:
    assert TIMING_REASON_CODES == frozenset(get_args(TimingReasonCode))


def test_timing_reason_codes_are_unique() -> None:
    args = list(get_args(TimingReasonCode))
    assert len(args) == len(set(args))


# ---------------------------------------------------------------------------
# Shared-resolver outcome mapping
# ---------------------------------------------------------------------------


def test_exact_resolution_groups_one_event() -> None:
    chart = _chart([_bgm(0, 0.0, "01")], {"01": "bgm.ogg"})
    audio = _remote("bgm.ogg")
    row = _row_view((_remote("real.dtx"), audio))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=False
    )

    assert result.reason_codes == ()
    assert result.bgm_event_count == 1
    (group,) = result.groups
    assert isinstance(group, BgmReferenceGroup)
    assert group.remote is audio
    assert group.measure == 0
    assert group.position == 0.0
    assert group.events == (_bgm(0, 0.0, "01"),)


def test_casefold_resolution_uses_the_folded_remote() -> None:
    chart = _chart([_bgm(0, 0.0, "01")], {"01": "bgm.ogg"})
    audio = _remote("Bgm.Ogg")
    row = _row_view((_remote("real.dtx"), audio))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=False
    )

    assert result.reason_codes == ()
    (group,) = result.groups
    assert group.remote is audio


def test_invalid_path_records_unsafe_bgm_audio_path() -> None:
    chart = _chart([_bgm(0, 0.0, "01")], {"01": "../escape.ogg"})
    row = _row_view((_remote("real.dtx"),))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=False
    )

    assert result.groups == ()
    assert result.reason_codes == ("unsafe_bgm_audio_path",)
    assert set(result.reason_codes) <= TIMING_REASON_CODES


def test_ambiguous_key_records_source_audio_key_ambiguous() -> None:
    chart = _chart([_bgm(0, 0.0, "01")], {"01": "bgm.ogg"})
    # Two inventory objects share the same key -> the resolver reports ambiguous.
    row = _row_view((_remote("real.dtx"), _remote("bgm.ogg"), _remote("bgm.ogg")))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=False
    )

    assert result.groups == ()
    assert result.reason_codes == ("source_audio_key_ambiguous",)
    assert set(result.reason_codes) <= TIMING_REASON_CODES


def test_missing_without_fallback_records_source_audio_missing() -> None:
    chart = _chart([_bgm(0, 0.0, "01")], {"01": "absent.ogg"})
    row = _row_view((_remote("real.dtx"),))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=False
    )

    assert result.groups == ()
    assert result.reason_codes == ("source_audio_missing",)
    assert set(result.reason_codes) <= TIMING_REASON_CODES


def test_missing_with_root_fallback_recovers_audio_at_the_simfile_root() -> None:
    chart = _chart([_bgm(0, 0.0, "01")], {"01": "bgm.ogg"})
    # The chart lives in a sub-directory but the audio sits at the simfile root.
    chart_remote = _remote("sub/real.dtx")
    audio = _remote("bgm.ogg")
    row = _row_view((chart_remote, audio))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/sub/real.dtx", row=row, allow_root_fallback=True
    )

    assert result.reason_codes == ()
    (group,) = result.groups
    assert group.remote is audio
    assert group.measure == 0


def test_missing_with_root_fallback_still_missing_records_source_audio_missing() -> None:
    chart = _chart([_bgm(0, 0.0, "01")], {"01": "absent.ogg"})
    row = _row_view((_remote("real.dtx"),))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=True
    )

    assert result.groups == ()
    assert result.reason_codes == ("source_audio_missing",)


def test_missing_with_root_fallback_ambiguous_records_source_audio_key_ambiguous() -> None:
    # The chart is in a sub-directory (first attempt misses), but the simfile
    # root holds two objects with the same key, so the root retry is ambiguous.
    chart = _chart([_bgm(0, 0.0, "01")], {"01": "bgm.ogg"})
    row = _row_view((_remote("sub/real.dtx"), _remote("bgm.ogg"), _remote("bgm.ogg")))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/sub/real.dtx", row=row, allow_root_fallback=True
    )

    assert result.groups == ()
    assert result.reason_codes == ("source_audio_key_ambiguous",)
    assert set(result.reason_codes) <= TIMING_REASON_CODES


def test_missing_with_root_fallback_invalid_path_records_unsafe_bgm_audio_path() -> None:
    # The relative path is safe at the chart's depth but escapes the prefix when
    # retried against the simfile root.
    chart = _chart([_bgm(0, 0.0, "01")], {"01": "../x.ogg"})
    row = _row_view((_remote("a/b/real.dtx"),))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/a/b/real.dtx", row=row, allow_root_fallback=True
    )

    assert result.groups == ()
    assert result.reason_codes == ("unsafe_bgm_audio_path",)
    assert set(result.reason_codes) <= TIMING_REASON_CODES


# ---------------------------------------------------------------------------
# Grouping by (remote.key, measure, position) — never floating-point time
# ---------------------------------------------------------------------------


def test_repeated_tokens_at_one_identity_collapse_into_one_group() -> None:
    # Two distinct note IDs that both point at the same file, landing on the
    # same (measure, position), collapse to one group retaining both events.
    events = [_bgm(0, 0.0, "01", order=0), _bgm(0, 0.0, "02", order=1)]
    chart = _chart(events, {"01": "bgm.ogg", "02": "bgm.ogg"})
    audio = _remote("bgm.ogg")
    row = _row_view((_remote("real.dtx"), audio))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=False
    )

    assert result.bgm_event_count == 2
    assert result.reason_codes == ()
    (group,) = result.groups
    assert group.remote is audio
    assert group.measure == 0
    assert group.position == 0.0
    assert len(group.events) == 2
    assert group.events == tuple(events)


def test_same_file_at_two_positions_yields_two_groups() -> None:
    events = [_bgm(0, 0.0, "01", order=0), _bgm(1, 0.5, "01", order=1)]
    chart = _chart(events, {"01": "bgm.ogg"})
    audio = _remote("bgm.ogg")
    row = _row_view((_remote("real.dtx"), audio))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=False
    )

    assert result.bgm_event_count == 2
    assert result.reason_codes == ()
    assert len(result.groups) == 2
    assert {(group.measure, group.position) for group in result.groups} == {(0, 0.0), (1, 0.5)}
    assert {group.remote for group in result.groups} == {audio}
    # No winner is chosen: both groups survive.
    assert all(group.events == (event,) for group, event in zip(result.groups, events, strict=True))


def test_different_files_at_one_position_yield_two_groups() -> None:
    events = [_bgm(0, 0.0, "01", order=0), _bgm(0, 0.0, "02", order=1)]
    chart = _chart(events, {"01": "a.ogg", "02": "b.ogg"})
    audio_a = _remote("a.ogg")
    audio_b = _remote("b.ogg")
    row = _row_view((_remote("real.dtx"), audio_a, audio_b))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=False
    )

    assert result.bgm_event_count == 2
    assert result.reason_codes == ()
    assert len(result.groups) == 2
    assert {group.remote for group in result.groups} == {audio_a, audio_b}
    assert {group.measure for group in result.groups} == {0}


def test_unresolved_wav_note_id_records_unresolved_bgm_wav() -> None:
    chart = _chart([_bgm(0, 0.0, "99")], {"01": "bgm.ogg"})
    row = _row_view((_remote("real.dtx"), _remote("bgm.ogg")))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=False
    )

    assert result.groups == ()
    assert result.reason_codes == ("unresolved_bgm_wav",)
    assert result.bgm_event_count == 1
    assert set(result.reason_codes) <= TIMING_REASON_CODES


def test_zero_bgm_events_records_bgm_event_missing() -> None:
    chart = _chart([], {"01": "bgm.ogg"})
    row = _row_view((_remote("real.dtx"),))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=False
    )

    assert isinstance(result, BgmReferenceSet)
    assert result.groups == ()
    assert result.reason_codes == ("bgm_event_missing",)
    assert result.bgm_event_count == 0
    assert set(result.reason_codes) <= TIMING_REASON_CODES


def test_partial_failure_keeps_resolved_groups_and_records_reasons() -> None:
    # One event resolves, one is unresolved, and one points at a missing file.
    events = [_bgm(0, 0.0, "01", order=0), _bgm(1, 0.0, "99", order=1), _bgm(2, 0.0, "02", order=2)]
    chart = _chart(events, {"01": "bgm.ogg", "02": "absent.ogg"})
    audio = _remote("bgm.ogg")
    row = _row_view((_remote("real.dtx"), audio))

    result = resolve_bgm_reference_groups(
        chart, selected_chart_key="42/real.dtx", row=row, allow_root_fallback=False
    )

    assert result.bgm_event_count == 3
    assert set(result.reason_codes) == {"unresolved_bgm_wav", "source_audio_missing"}
    assert set(result.reason_codes) <= TIMING_REASON_CODES
    (group,) = result.groups
    assert group.remote is audio
    assert group.events == (events[0],)


# ---------------------------------------------------------------------------
# Frozen conservative selector: select_bgm_reference / BgmResolution
# ---------------------------------------------------------------------------


def _resolve_for_select(
    chart: ParsedDtxChart, row: ReferenceChartRowView, *, allow_root_fallback: bool = True
) -> tuple[BgmReferenceSet, DtxTimingMap]:
    references = resolve_bgm_reference_groups(
        chart,
        selected_chart_key="42/real.dtx",
        row=row,
        allow_root_fallback=allow_root_fallback,
    )
    return references, build_dtx_timing_map(chart)


def test_select_bgm_reference_zero_groups_returns_bgm_event_missing() -> None:
    chart = _chart([], {"01": "bgm.ogg"})
    row = _row_view((_remote("real.dtx"),))
    references, timing_map = _resolve_for_select(chart, row)

    resolution = select_bgm_reference(references, timing_map)

    assert isinstance(resolution, BgmResolution)
    assert resolution.selected_event is None
    assert resolution.chart_time_sec is None
    assert resolution.reason_codes == ("bgm_event_missing",)
    assert resolution.warnings == ()
    assert set(resolution.reason_codes) <= TIMING_REASON_CODES


def test_select_bgm_reference_zero_groups_with_events_does_not_add_bgm_event_missing() -> None:
    # A BGM event exists but its WAV note_id is absent from the wav_table, so
    # zero groups survive and the resolver emits unresolved_bgm_wav.  The
    # selector must NOT additionally emit bgm_event_missing — the event exists,
    # it just failed to resolve.
    chart = _chart([_bgm(0, 0.0, "99")], {"01": "bgm.ogg"})
    row = _row_view((_remote("real.dtx"), _remote("bgm.ogg")))
    references, timing_map = _resolve_for_select(chart, row)

    assert references.groups == ()
    assert references.bgm_event_count == 1

    resolution = select_bgm_reference(references, timing_map)

    assert isinstance(resolution, BgmResolution)
    assert resolution.selected_event is None
    assert resolution.chart_time_sec is None
    assert resolution.reason_codes == ()
    assert resolution.warnings == ()


def test_select_bgm_reference_one_group_picks_lowest_source_order_event() -> None:
    event = _bgm(1, 0.5, "01", order=3)
    chart = _chart([event], {"01": "bgm.ogg"})
    audio = _remote("bgm.ogg")
    row = _row_view((_remote("real.dtx"), audio))
    references, timing_map = _resolve_for_select(chart, row)

    resolution = select_bgm_reference(references, timing_map)

    assert resolution.reason_codes == ()
    assert resolution.warnings == ()
    assert resolution.selected_event is event
    # Measure 1, position 0.5, default 1.0 measure length, 120 BPM -> 3.0s.
    assert resolution.chart_time_sec == timing_map.time_sec(event)
    assert resolution.chart_time_sec == 3.0


def test_select_bgm_reference_repeated_tokens_emit_one_deterministic_warning() -> None:
    # Two tokens at the same identity collapse to one group retaining both.
    low = _bgm(0, 0.0, "01", order=1)
    high = _bgm(0, 0.0, "02", order=2)
    chart = _chart([high, low], {"01": "bgm.ogg", "02": "bgm.ogg"})
    audio = _remote("bgm.ogg")
    row = _row_view((_remote("real.dtx"), audio))
    references, timing_map = _resolve_for_select(chart, row)
    (group,) = references.groups
    assert len(group.events) == 2

    resolution = select_bgm_reference(references, timing_map)

    assert resolution.reason_codes == ()
    # Lowest source_order wins.
    assert resolution.selected_event is low
    assert resolution.chart_time_sec == timing_map.time_sec(low)
    # Exactly one deterministic warning summarizing the repeated tokens.
    assert len(resolution.warnings) == 1
    assert resolution.warnings == ("repeated_bgm_tokens:01,02:count=2",)


def test_select_bgm_reference_multiple_groups_returns_ambiguous_bgm_start() -> None:
    # Same file at two positions -> two groups (no winner chosen).
    events = [_bgm(0, 0.0, "01", order=0), _bgm(1, 0.5, "01", order=1)]
    chart = _chart(events, {"01": "bgm.ogg"})
    audio = _remote("bgm.ogg")
    row = _row_view((_remote("real.dtx"), audio))
    references, timing_map = _resolve_for_select(chart, row)
    assert len(references.groups) > 1

    resolution = select_bgm_reference(references, timing_map)

    assert resolution.selected_event is None
    assert resolution.chart_time_sec is None
    assert resolution.reason_codes == ("ambiguous_bgm_start",)
    assert resolution.warnings == ()
    assert set(resolution.reason_codes) <= TIMING_REASON_CODES


def test_select_bgm_reference_is_deterministic_across_calls() -> None:
    low = _bgm(0, 0.0, "01", order=1)
    high = _bgm(0, 0.0, "02", order=2)
    chart = _chart([high, low], {"01": "bgm.ogg", "02": "bgm.ogg"})
    audio = _remote("bgm.ogg")
    row = _row_view((_remote("real.dtx"), audio))
    references, timing_map = _resolve_for_select(chart, row)

    first = select_bgm_reference(references, timing_map)
    second = select_bgm_reference(references, timing_map)

    assert first == second
    assert first.selected_event is low


# ===========================================================================
# HPA-323 Task 5: source-audio metadata, bounded events, schema golden
# ===========================================================================


_AUDIO_PROBE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    ValueError,
    sf.LibsndfileError,
)

_IDENTITY_KWARGS = {
    "simfile_id": 42,
    "selected_chart_key": "42/real.dtx",
    "selected_chart_content_hash": "a" * 64,
    "source_audio_key": "42/bgm.ogg",
    "source_audio_content_hash": "b" * 64,
}


class _FakeTimingMap:
    """Minimal stand-in for :class:`DtxTimingMap` with per-event chart times.

    lets the bounds tests control ``chart_time_sec`` exactly (keyed by
    ``source_order``) instead of depending on the BPM/measure math.
    """

    def __init__(self, times: dict[int, float]) -> None:
        self._times = times

    def time_sec(self, event: DtxEvent) -> float:
        return self._times[event.source_order]


def _note(source_order: int, *, measure: int = 0, position: float = 0.0) -> DtxEvent:
    return DtxEvent(_CHART_ID, measure, position, "11", "01", source_order)


def _chart_with_events(events: list[DtxEvent]) -> ParsedDtxChart:
    return ParsedDtxChart(chart_id=_CHART_ID, events=events)


def _audio(
    *, duration_sec: float = 10.0, sample_rate: int = 100, frames: int | None = None
) -> SourceAudioInfo:
    resolved_frames = int(duration_sec * sample_rate) if frames is None else frames
    return SourceAudioInfo(
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        channels=2,
        frames=resolved_frames,
    )


# ---------------------------------------------------------------------------
# Step 1: inspect_source_audio metadata
# ---------------------------------------------------------------------------


def test_inspect_source_audio_returns_header_metadata(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(22050, dtype=np.float32), 22050, format="WAV", subtype="FLOAT")

    info = inspect_source_audio(path)

    assert isinstance(info, SourceAudioInfo)
    assert info.frames == 22050
    assert info.sample_rate == 22050
    assert info.channels == 1
    assert info.duration_sec == pytest.approx(1.0)


def test_inspect_source_audio_reports_zero_frame_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, dtype=np.float32), 8000, format="WAV", subtype="FLOAT")

    info = inspect_source_audio(path)

    assert info.frames == 0
    assert info.sample_rate == 8000
    assert info.duration_sec == 0.0


def test_inspect_source_audio_raises_for_unreadable_body(tmp_path: Path) -> None:
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"not a wav file at all")

    with pytest.raises(_AUDIO_PROBE_ERRORS):
        inspect_source_audio(path)


# ---------------------------------------------------------------------------
# Step 2: build_audio_relative_events bounds logic
# ---------------------------------------------------------------------------


def test_build_audio_relative_events_shifts_by_bgm_anchor_with_real_timing() -> None:
    # base_bpm=120, default measure length -> time_sec = 2 * (measure + position).
    event = DtxEvent(_CHART_ID, 2, 0.5, "11", "01", 0)
    chart = ParsedDtxChart(chart_id=_CHART_ID, events=[event])
    timing_map = build_dtx_timing_map(chart)

    result = build_audio_relative_events(
        chart,
        timing_map,
        bgm_chart_time_sec=2.0,
        audio=_audio(duration_sec=10.0, sample_rate=100),
        **_IDENTITY_KWARGS,
    )

    assert result.pre_audio_event_count == 0
    assert result.post_audio_event_count == 0
    assert result.reason_codes == ()
    (kept,) = result.events
    assert kept.chart_time_sec == 5.0  # 2 * (2 + 0.5)
    assert kept.audio_time_sec == 3.0  # 5.0 - 2.0 bgm shift


def test_build_audio_relative_events_exact_zero_and_exact_duration_are_in_bounds() -> None:
    chart = _chart_with_events([_note(0), _note(1)])
    timing_map = _FakeTimingMap({0: 5.0, 1: 15.0})

    result = build_audio_relative_events(
        chart,
        timing_map,
        bgm_chart_time_sec=5.0,
        audio=_audio(duration_sec=10.0, sample_rate=100),
        **_IDENTITY_KWARGS,
    )

    times = sorted(event.audio_time_sec for event in result.events)
    assert times == [0.0, 10.0]
    assert result.pre_audio_event_count == 0
    assert result.post_audio_event_count == 0
    assert result.reason_codes == ()


def test_build_audio_relative_events_clamps_one_frame_below_zero() -> None:
    # one_frame = 1/100 = 0.01; -0.005 is within the band -> clamp to 0.0.
    chart = _chart_with_events([_note(0)])
    timing_map = _FakeTimingMap({0: 4.995})

    result = build_audio_relative_events(
        chart,
        timing_map,
        bgm_chart_time_sec=5.0,
        audio=_audio(duration_sec=10.0, sample_rate=100),
        **_IDENTITY_KWARGS,
    )

    (kept,) = result.events
    assert kept.audio_time_sec == 0.0
    assert result.pre_audio_event_count == 0


def test_build_audio_relative_events_clamps_one_frame_above_duration() -> None:
    chart = _chart_with_events([_note(0)])
    timing_map = _FakeTimingMap({0: 15.005})

    result = build_audio_relative_events(
        chart,
        timing_map,
        bgm_chart_time_sec=5.0,
        audio=_audio(duration_sec=10.0, sample_rate=100),
        **_IDENTITY_KWARGS,
    )

    (kept,) = result.events
    assert kept.audio_time_sec == 10.0
    assert result.post_audio_event_count == 0


def test_build_audio_relative_events_excludes_materially_negative_with_pre_counter() -> None:
    # -0.02 < -one_frame (0.01) -> excluded, pre counter incremented.  A second
    # in-bounds event keeps the retained set non-empty so this isolates the
    # pre-counter from the zero-retained (no_in_bounds) rule.
    chart = _chart_with_events([_note(0), _note(1)])
    timing_map = _FakeTimingMap({0: 4.98, 1: 5.0})

    result = build_audio_relative_events(
        chart,
        timing_map,
        bgm_chart_time_sec=5.0,
        audio=_audio(duration_sec=10.0, sample_rate=100),
        **_IDENTITY_KWARGS,
    )

    assert len(result.events) == 1
    assert result.pre_audio_event_count == 1
    assert result.post_audio_event_count == 0
    assert result.reason_codes == ()


def test_build_audio_relative_events_excludes_materially_late_with_post_counter() -> None:
    # duration + 0.02 > duration + one_frame -> excluded, post counter incremented.
    chart = _chart_with_events([_note(0), _note(1)])
    timing_map = _FakeTimingMap({0: 15.02, 1: 5.0})

    result = build_audio_relative_events(
        chart,
        timing_map,
        bgm_chart_time_sec=5.0,
        audio=_audio(duration_sec=10.0, sample_rate=100),
        **_IDENTITY_KWARGS,
    )

    assert len(result.events) == 1
    assert result.post_audio_event_count == 1
    assert result.pre_audio_event_count == 0
    assert result.reason_codes == ()


def test_build_audio_relative_events_non_finite_time_records_reason_without_counters() -> None:
    chart = _chart_with_events([_note(0)])
    timing_map = _FakeTimingMap({0: float("nan")})

    result = build_audio_relative_events(
        chart,
        timing_map,
        bgm_chart_time_sec=5.0,
        audio=_audio(duration_sec=10.0, sample_rate=100),
        **_IDENTITY_KWARGS,
    )

    assert isinstance(result, AudioRelativeReference)
    assert result.events == ()
    assert result.pre_audio_event_count == 0
    assert result.post_audio_event_count == 0
    assert result.reason_codes == ("non_finite_reference_time",)
    assert set(result.reason_codes) <= TIMING_REASON_CODES


def test_build_audio_relative_events_zero_retained_records_no_in_bounds() -> None:
    chart = _chart_with_events([_note(0), _note(1)])
    timing_map = _FakeTimingMap({0: 0.0, 1: 100.0})  # one far pre, one far post.

    result = build_audio_relative_events(
        chart,
        timing_map,
        bgm_chart_time_sec=50.0,
        audio=_audio(duration_sec=10.0, sample_rate=100),
        **_IDENTITY_KWARGS,
    )

    assert result.events == ()
    assert result.pre_audio_event_count == 1
    assert result.post_audio_event_count == 1
    assert result.reason_codes == ("no_in_bounds_reference_events",)
    assert set(result.reason_codes) <= TIMING_REASON_CODES


def test_build_audio_relative_events_non_finite_suppresses_no_in_bounds_reason() -> None:
    # A non-finite failure with zero retained must not also raise no_in_bounds.
    chart = _chart_with_events([_note(0)])
    timing_map = _FakeTimingMap({0: float("inf")})

    result = build_audio_relative_events(
        chart,
        timing_map,
        bgm_chart_time_sec=float("inf"),
        audio=_audio(duration_sec=10.0, sample_rate=100),
        **_IDENTITY_KWARGS,
    )

    assert result.events == ()
    assert result.reason_codes == ("non_finite_reference_time",)


def test_build_audio_relative_events_preserves_identity_and_sorts_native_order() -> None:
    # Out-of-(measure, position, source_order) entry order; retained set must be
    # sorted and preserve every native identity field verbatim.
    events = [
        DtxEvent(_CHART_ID, 5, 0.5, "18", "AA", 2),
        DtxEvent(_CHART_ID, 1, 0.25, "11", "01", 0),
        DtxEvent(_CHART_ID, 1, 0.5, "12", "02", 1),
    ]
    chart = _chart_with_events(events)
    timing_map = _FakeTimingMap({0: 6.0, 1: 6.5, 2: 7.0})

    result = build_audio_relative_events(
        chart,
        timing_map,
        bgm_chart_time_sec=5.0,
        audio=_audio(duration_sec=10.0, sample_rate=100),
        **_IDENTITY_KWARGS,
    )

    assert [kept.source_order for kept in result.events] == [0, 1, 2]
    assert [(kept.measure, kept.position) for kept in result.events] == [
        (1, 0.25),
        (1, 0.5),
        (5, 0.5),
    ]
    source = {e.source_order: e for e in events}
    for kept in result.events:
        original = source[kept.source_order]
        assert kept.measure == original.measure
        assert kept.position == original.position
        assert kept.lane_id == original.lane_id
        assert kept.note_id == original.note_id
        assert kept.simfile_id == _IDENTITY_KWARGS["simfile_id"]
        assert kept.selected_chart_key == _IDENTITY_KWARGS["selected_chart_key"]
        assert kept.selected_chart_content_hash == _IDENTITY_KWARGS["selected_chart_content_hash"]
        assert kept.source_audio_key == _IDENTITY_KWARGS["source_audio_key"]
        assert kept.source_audio_content_hash == _IDENTITY_KWARGS["source_audio_content_hash"]
        assert kept.chart_time_sec == timing_map.time_sec(original)


# ---------------------------------------------------------------------------
# Step 4: deterministic canonical JSONL rendering
# ---------------------------------------------------------------------------


def _sample_events() -> tuple[NativeReferenceEvent, ...]:
    return (
        NativeReferenceEvent(
            source_order=1,
            measure=2,
            position=0.5,
            lane_id="12",
            note_id="02",
            chart_time_sec=4.0,
            audio_time_sec=3.0,
            **_IDENTITY_KWARGS,  # type: ignore[arg-type]
        ),
        NativeReferenceEvent(
            source_order=0,
            measure=1,
            position=0.0,
            lane_id="11",
            note_id="01",
            chart_time_sec=2.0,
            audio_time_sec=1.0,
            **_IDENTITY_KWARGS,  # type: ignore[arg-type]
        ),
    )


def test_render_reference_events_is_byte_identical_across_renders() -> None:
    events = _sample_events()

    first = render_reference_events(events)
    second = render_reference_events(events)

    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    assert first.endswith(b"\n")
    assert not first.endswith(b"\n\n")


def test_render_reference_events_sorts_by_deterministic_native_identity() -> None:
    events = _sample_events()

    content = render_reference_events(events)

    lines = content.splitlines(keepends=True)
    assert len(lines) == 2
    rows = [strict_json_loads(line[:-1], require_canonical=True) for line in lines]
    # Sorted by (measure, position, source_order) -> source_order 0 first.
    assert rows[0]["source_order"] == 0
    assert rows[1]["source_order"] == 1
    assert rows[0]["measure"] == 1
    assert rows[1]["measure"] == 2


def test_render_reference_events_emits_canonical_jsonl_each_line() -> None:
    content = render_reference_events(_sample_events())

    for line in content.splitlines(keepends=True):
        assert line.endswith(b"\n")
        strict_json_loads(line[:-1], require_canonical=True)


def test_read_reference_events_round_trips_rendered_content() -> None:
    content = render_reference_events(_sample_events())

    events = read_reference_events(content)

    assert events == tuple(
        sorted(
            _sample_events(),
            key=lambda event: (
                event.measure,
                event.position,
                event.source_order,
                event.lane_id,
                event.note_id,
            ),
        )
    )
    assert render_reference_events(events) == content


def test_read_reference_events_round_trips_base36_note_id() -> None:
    event = NativeReferenceEvent(
        source_order=0,
        measure=1,
        position=0.5,
        lane_id="14",
        note_id="0L",
        chart_time_sec=2.5,
        audio_time_sec=1.5,
        **_IDENTITY_KWARGS,  # type: ignore[arg-type]
    )
    content = render_reference_events((event,))

    assert read_reference_events(content) == (event,)
    assert b'"note_id":"0L"' in content


@pytest.mark.parametrize("note_id", ["0l", "0/", "000"])
def test_read_reference_events_rejects_noncanonical_dtx_note_id(note_id: str) -> None:
    event = NativeReferenceEvent(
        source_order=0,
        measure=1,
        position=0.5,
        lane_id="14",
        note_id=note_id,
        chart_time_sec=2.5,
        audio_time_sec=1.5,
        **_IDENTITY_KWARGS,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError):
        read_reference_events(render_reference_events((event,)))


@pytest.mark.parametrize(
    "content_builder",
    [
        pytest.param(
            lambda content: content.replace(b'":', b'" :', 1),
            id="non-canonical-json",
        ),
        pytest.param(
            lambda content: canonical_json_bytes(
                {
                    **strict_json_loads(content.splitlines()[0], require_canonical=True),
                    "unexpected": "value",
                },
                trailing_newline=True,
            ),
            id="wrong-keys",
        ),
        pytest.param(
            lambda content: canonical_json_bytes(
                {
                    **strict_json_loads(content.splitlines()[0], require_canonical=True),
                    "audio_time_sec": "not-a-time",
                },
                trailing_newline=True,
            ),
            id="invalid-time",
        ),
        pytest.param(
            lambda content: content.splitlines(keepends=True)[1]
            + content.splitlines(keepends=True)[0],
            id="out-of-order",
        ),
        pytest.param(
            lambda content: content.splitlines(keepends=True)[0]
            + content.splitlines(keepends=True)[0],
            id="duplicate-native-identity",
        ),
    ],
)
def test_read_reference_events_rejects_invalid_content(content_builder) -> None:
    content = content_builder(render_reference_events(_sample_events()))

    with pytest.raises(ValueError):
        read_reference_events(content)


# ---------------------------------------------------------------------------
# Step 5: publish_immutable_content delegates to the existing publisher
# ---------------------------------------------------------------------------


def test_publish_immutable_content_writes_immutable_bytes(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    content = render_reference_events(_sample_events())
    expected_sha256 = hashlib.sha256(content).hexdigest()

    publish_immutable_content(path, content, expected_sha256)

    assert path.read_bytes() == content


def test_publish_immutable_content_rejects_hash_mismatch(tmp_path: Path) -> None:
    from src.benchmark.corpus_manifest import ManifestPublicationError

    path = tmp_path / "events.jsonl"
    content = render_reference_events(_sample_events())

    with pytest.raises(ManifestPublicationError):
        publish_immutable_content(path, content, "0" * 64)


# ---------------------------------------------------------------------------
# Step 6: schema-golden registration + validator behavior
# ---------------------------------------------------------------------------

_GOLDEN_PATH = Path(__file__).parent / "schema_goldens" / "crux.dtx-reference-event-v1.jsonl"


def _golden_event_row() -> dict[str, object]:
    return {
        "simfile_id": 42,
        "selected_chart_key": "42/real.dtx",
        "selected_chart_content_hash": "a" * 64,
        "source_audio_key": "42/bgm.ogg",
        "source_audio_content_hash": "b" * 64,
        "source_order": 0,
        "measure": 1,
        "position": Decimal("0.25"),
        "lane_id": "11",
        "note_id": "01",
        "chart_time_sec": Decimal("2.5"),
        "audio_time_sec": Decimal("1.5"),
    }


def test_reference_event_schema_golden_is_registered_and_valid() -> None:
    from tests.benchmark.test_schema_goldens import (
        load_schema_golden_manifest,
        validate_schema_golden_entry,
    )

    repository_root = Path(__file__).parents[2]
    entries = load_schema_golden_manifest(repository_root)
    entry = next(item for item in entries if item.schema == "crux.dtx-reference-event/v1")
    assert entry.golden_path.as_posix() == (
        "tests/benchmark/schema_goldens/crux.dtx-reference-event-v1.jsonl"
    )
    assert entry.validator_modules == ("src.benchmark.reference_timing",)

    validate_schema_golden_entry(entry, repository_root)


def test_reference_event_golden_was_generated_by_render_reference_events() -> None:
    # Proves the on-disk golden is byte-identical to the canonical renderer.
    events = (
        NativeReferenceEvent(
            source_order=0,
            measure=1,
            position=0.25,
            lane_id="11",
            note_id="01",
            chart_time_sec=2.5,
            audio_time_sec=1.5,
            **_IDENTITY_KWARGS,  # type: ignore[arg-type]
        ),
        NativeReferenceEvent(
            source_order=1,
            measure=2,
            position=0.5,
            lane_id="12",
            note_id="02",
            chart_time_sec=4.5,
            audio_time_sec=3.5,
            **_IDENTITY_KWARGS,  # type: ignore[arg-type]
        ),
    )

    assert _GOLDEN_PATH.read_bytes() == render_reference_events(events)


def test_validate_reference_event_golden_rejects_unknown_key() -> None:
    row = _golden_event_row()
    row["unexpected"] = "unexpected"
    content = canonical_json_bytes(row, trailing_newline=True)

    with pytest.raises(ValueError):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


def test_validate_reference_event_golden_rejects_missing_key() -> None:
    row = _golden_event_row()
    del row["audio_time_sec"]
    content = canonical_json_bytes(row, trailing_newline=True)

    with pytest.raises(ValueError):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


def test_validate_reference_event_golden_rejects_bad_sha256() -> None:
    row = _golden_event_row()
    row["source_audio_content_hash"] = "not-a-sha256"
    content = canonical_json_bytes(row, trailing_newline=True)

    with pytest.raises(ValueError):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


def test_validate_reference_event_golden_rejects_non_finite_time() -> None:
    valid = canonical_json_bytes(_golden_event_row(), trailing_newline=True)
    nonfinite = valid.replace(b'"audio_time_sec":1.5', b'"audio_time_sec":Infinity')

    with pytest.raises(ValueError):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, nonfinite)


@pytest.mark.parametrize("invalid", ["soon", True])
def test_validate_reference_event_golden_rejects_wrong_type_time(invalid: object) -> None:
    # Whole-number times legitimately render to integer tokens that parse back
    # as ``int`` (see the acceptance test below), so a wrong-type time is probed
    # with a string and a bool rather than an integer.  Both must be rejected.
    row = _golden_event_row()
    row["audio_time_sec"] = invalid
    content = canonical_json_bytes(row, trailing_newline=True)

    with pytest.raises(ValueError):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


def test_validate_reference_event_golden_accepts_whole_number_times() -> None:
    # Regression: the canonical renderer collapses whole-number Decimals to
    # bare-integer tokens (Decimal("0.0") -> "0"), which strict_json_loads parses
    # back as ``int``.  build_audio_relative_events realistically emits such
    # whole-number times — the COMMON clamp-to-zero outcome (audio_time_sec=0.0),
    # the exact-duration outcome (audio_time_sec=duration_sec), position=0.0, and
    # whole-number chart_time_sec.  A byte stream produced by
    # render_reference_events must therefore PASS validate_schema_golden.
    chart = _chart_with_events(
        [_note(0, measure=0, position=0.0), _note(1, measure=0, position=0.0)]
    )
    timing_map = _FakeTimingMap({0: 5.0, 1: 15.0})
    result = build_audio_relative_events(
        chart,
        timing_map,
        bgm_chart_time_sec=5.0,
        audio=_audio(duration_sec=10.0, sample_rate=100),
        **_IDENTITY_KWARGS,
    )

    # Sanity: the realistic whole-number outcomes the renderer collapses.
    audio_times = sorted(event.audio_time_sec for event in result.events)
    assert audio_times == [0.0, 10.0]
    assert any(event.position == 0.0 for event in result.events)
    assert any(float(event.chart_time_sec).is_integer() for event in result.events)

    rendered = render_reference_events(result.events)

    # Must not raise: the rendered byte stream validates end-to-end.
    validate_schema_golden(REFERENCE_EVENT_SCHEMA, rendered)


def test_validate_reference_event_golden_rejects_non_canonical_bytes() -> None:
    valid = canonical_json_bytes(_golden_event_row(), trailing_newline=True)
    # Swap two sorted-adjacent keys -> valid JSON, non-canonical ordering.
    noncanonical = valid.replace(
        b'"audio_time_sec":1.5,"chart_time_sec":2.5',
        b'"chart_time_sec":2.5,"audio_time_sec":1.5',
    )

    with pytest.raises(ValueError):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, noncanonical)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("simfile_id", -1),
        ("simfile_id", MAX_SIMFILE_ID + 1),
        ("source_order", -1),
        ("measure", -1),
        ("position", Decimal("-0.1")),
        ("position", Decimal("1.0")),
        ("lane_id", "1"),
        ("lane_id", "GG"),
        ("note_id", "000"),
        ("note_id", "0/"),
        ("selected_chart_key", "../escape.dtx"),
        ("selected_chart_key", "/42/real.dtx"),
        ("selected_chart_key", "42\\real.dtx"),
        ("source_audio_key", "42/../audio.ogg"),
        ("source_audio_key", "42/audio.ogg\x00"),
    ],
)
def test_validate_reference_event_golden_rejects_invalid_identity_fields(
    field: str,
    value: object,
) -> None:
    row = _golden_event_row()
    row[field] = value
    content = canonical_json_bytes(row, trailing_newline=True)

    with pytest.raises(ValueError):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


def test_validate_reference_event_golden_rejects_non_hex_lane_with_base36_note() -> None:
    row = _golden_event_row()
    row["lane_id"] = "GG"
    row["note_id"] = "0L"
    content = canonical_json_bytes(row, trailing_newline=True)

    with pytest.raises(ValueError, match="lane_id must be a two-digit uppercase hexadecimal ID"):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lane_id", "1a", "uppercase hexadecimal"),
        ("note_id", "0l", "uppercase base-36"),
    ],
)
def test_validate_reference_event_golden_rejects_lowercase_dtx_ids(
    field: str, value: str, message: str
) -> None:
    row = _golden_event_row()
    row[field] = value
    content = canonical_json_bytes(row, trailing_newline=True)

    with pytest.raises(ValueError, match=message):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


def test_validate_reference_event_golden_rejects_duplicate_native_identity() -> None:
    row = _golden_event_row()
    content = b"".join(canonical_json_bytes(row, trailing_newline=True) for _ in range(2))

    with pytest.raises(ValueError, match="duplicate"):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


def test_validate_reference_event_golden_rejects_out_of_order_rows() -> None:
    content = render_reference_events(_sample_events())
    lines = content.splitlines(keepends=True)
    out_of_order = lines[1] + lines[0]

    with pytest.raises(ValueError, match="order"):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, out_of_order)


def test_validate_reference_event_golden_rejects_unsupported_schema() -> None:
    content = canonical_json_bytes(_golden_event_row(), trailing_newline=True)

    with pytest.raises(ValueError):
        validate_schema_golden("crux.other/v1", content)


# ---------------------------------------------------------------------------
# Coverage: remaining validate_schema_golden rejection paths
# ---------------------------------------------------------------------------


def test_validate_reference_event_golden_rejects_content_without_trailing_newline() -> None:
    # Line 614: content that does not end with a newline.
    content = canonical_json_bytes(_golden_event_row())  # no trailing newline

    with pytest.raises(ValueError, match="one final newline"):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


def test_validate_reference_event_golden_rejects_double_trailing_newline() -> None:
    # Line 614: content that ends with a double newline.
    content = canonical_json_bytes(_golden_event_row(), trailing_newline=True) + b"\n"

    with pytest.raises(ValueError, match="one final newline"):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


def test_validate_reference_event_golden_rejects_blank_line() -> None:
    # Line 618: a blank line in the content.
    valid = canonical_json_bytes(_golden_event_row(), trailing_newline=True)
    content = valid + b"\n" + valid

    with pytest.raises(ValueError, match="one record per line"):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


def test_validate_reference_event_golden_rejects_non_byte_identical_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Line 632: content that passes strict_json_loads but is not byte-identical
    # to the canonical re-render.  We bypass the canonical check in
    # strict_json_loads so a non-canonical-but-valid-JSON line reaches the
    # byte-identity guard.
    import src.benchmark.reference_timing as rt

    real_loads = rt.strict_json_loads

    def lenient_loads(content: bytes, *, require_canonical: bool = False) -> object:
        return real_loads(content, require_canonical=False)

    monkeypatch.setattr(rt, "strict_json_loads", lenient_loads)

    valid = canonical_json_bytes(_golden_event_row())
    non_canonical = valid.replace(b'":', b'" :', 1) + b"\n"

    with pytest.raises(ValueError, match="byte-identical"):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, non_canonical)


@pytest.mark.parametrize("field", ["simfile_id", "source_order", "measure"])
def test_validate_reference_event_golden_rejects_non_integer_identity_field(
    field: str,
) -> None:
    # Line 641: an int-key field that parses as Decimal (float in JSON) is not
    # an int and must be rejected.
    row = _golden_event_row()
    row[field] = Decimal("1.5")
    content = canonical_json_bytes(row, trailing_newline=True)

    with pytest.raises(ValueError, match="must be an integer"):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


@pytest.mark.parametrize("field", ["selected_chart_key", "source_audio_key", "lane_id", "note_id"])
def test_validate_reference_event_golden_rejects_empty_string_field(field: str) -> None:
    # Line 650: a string-key field that is empty must be rejected.
    row = _golden_event_row()
    row[field] = ""
    content = canonical_json_bytes(row, trailing_newline=True)

    with pytest.raises(ValueError, match="must be a non-empty string"):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)


def test_validate_reference_event_row_rejects_non_finite_decimal_time() -> None:
    # Line 673: a time field that is a non-finite Decimal passes the type check
    # (int/Decimal) but fails the isfinite guard.  This cannot be reached
    # through validate_schema_golden because strict_json_loads rejects
    # non-finite JSON constants, so the internal validator is exercised directly.
    from src.benchmark.reference_timing import _validate_reference_event_row

    row = _golden_event_row()
    row["audio_time_sec"] = Decimal("Infinity")

    with pytest.raises(ValueError, match="must be a finite number"):
        _validate_reference_event_row(row)


def test_validate_reference_event_golden_rejects_mixed_source_identity() -> None:
    # Line 706: two valid rows with different simfile_id (and therefore
    # different identity prefixes) must be rejected.
    row_a = _golden_event_row()
    row_b = _golden_event_row()
    row_b["simfile_id"] = 43
    row_b["selected_chart_key"] = "43/real.dtx"
    row_b["source_audio_key"] = "43/bgm.ogg"
    # Use a different measure so the native-identity ordering check passes.
    row_b["measure"] = 2
    content = b"".join(canonical_json_bytes(row, trailing_newline=True) for row in (row_a, row_b))

    with pytest.raises(ValueError, match="mixed source identity"):
        validate_schema_golden(REFERENCE_EVENT_SCHEMA, content)
