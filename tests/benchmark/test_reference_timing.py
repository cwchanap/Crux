from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import get_args

from src.benchmark.corpus_manifest import ManifestRowView
from src.benchmark.dtx_parser import DtxBgmEvent, ParsedDtxChart
from src.benchmark.r2_corpus_models import (
    CACHE_PROFILE,
    ProvenanceRecord,
    RemoteObject,
    SimfileInventory,
)
from src.benchmark.reference_chart_manifest import ReferenceChartRowView
from src.benchmark.reference_timing import (
    TIMING_REASON_CODES,
    BgmReferenceGroup,
    BgmReferenceSet,
    BgmResolution,
    TimingReasonCode,
    resolve_bgm_reference_groups,
    select_bgm_reference,
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
