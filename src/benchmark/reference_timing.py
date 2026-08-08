"""Reference-timing inputs derived from a validated reference-chart row.

This module is policy-neutral on purpose.  It exposes:

* the closed :data:`TimingReasonCode` contract shared by every timing consumer;
* a typed :class:`BgmReferenceSet` that groups channel-01 BGM control events by
  their stable ``(remote.key, measure, position)`` identity; and
* :func:`resolve_bgm_reference_groups`, which resolves each BGM event's source
  audio through the shared HPA-322 object-key resolver.

Group selection (choosing a single BGM group when several qualify) is deferred
to the corpus diagnostic (HPA-323 Task 4).  No winner is picked here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from src.benchmark.dtx_parser import DtxBgmEvent, ParsedDtxChart
from src.benchmark.inventory_object_keys import resolve_inventory_object_key
from src.benchmark.r2_corpus_models import RemoteObject
from src.benchmark.reference_chart_manifest import ReferenceChartRowView
from src.benchmark.timing import DtxTimingMap

TimingReasonCode = Literal[
    "upstream_chart_selection_unavailable",
    "selected_chart_cache_invalid",
    "selected_chart_parse_failed",
    "timing_map_invalid",
    "bgm_event_missing",
    "unresolved_bgm_wav",
    "unsafe_bgm_audio_path",
    "source_audio_missing",
    "source_audio_key_ambiguous",
    "ambiguous_bgm_start",
    "source_audio_download_failed",
    "source_audio_cache_invalid",
    "source_audio_decode_failed",
    "non_finite_reference_time",
    "no_in_bounds_reference_events",
]

#: Every stable reason code timing consumers may emit.  Kept in sync with
#: :data:`TimingReasonCode` via the ``get_args`` invariant test.
TIMING_REASON_CODES: frozenset[str] = frozenset(get_args(TimingReasonCode))


@dataclass(frozen=True)
class BgmReferenceGroup:
    """One BGM identity: a resolved source audio object at one chart position.

    ``events`` retains every channel-01 token that collapsed onto this
    ``(remote.key, measure, position)`` identity so downstream diagnostics can
    report the exact source tokens without re-resolving them.
    """

    remote: RemoteObject
    measure: int
    position: float
    events: tuple[DtxBgmEvent, ...]


@dataclass(frozen=True)
class BgmReferenceSet:
    """Policy-neutral collection of resolved BGM reference groups.

    ``groups`` is unordered with respect to winner selection: when more than one
    group qualifies, every group is returned and the caller (post Task 4)
    decides how to reduce them.  ``reason_codes`` only ever holds values from
    :data:`TimingReasonCode`; no free-form strings.
    """

    groups: tuple[BgmReferenceGroup, ...]
    reason_codes: tuple[TimingReasonCode, ...]
    warnings: tuple[str, ...]
    bgm_event_count: int


def resolve_bgm_reference_groups(
    chart: ParsedDtxChart,
    *,
    selected_chart_key: str,
    row: ReferenceChartRowView,
    allow_root_fallback: bool,
) -> BgmReferenceSet:
    """Resolve channel-01 BGM events into policy-neutral reference groups.

    Each BGM event's ``note_id`` is mapped through ``chart.wav_table`` to a
    relative audio path, then resolved against the already-validated inventory
    (:meth:`row.source.inventory`) using the shared HPA-322 object-key resolver.
    Events that resolve to the same :class:`RemoteObject` at the same
    ``(measure, position)`` collapse into a single :class:`BgmReferenceGroup`
    and retain their source events.

    Grouping keys on ``(remote.key, measure, position)`` — never floating-point
    seconds.  No winner is chosen when several groups qualify.
    """
    inventory = row.source.inventory
    objects = inventory.objects
    object_prefix = inventory.object_prefix
    chart_dir = _chart_object_dir(selected_chart_key)
    bgm_events = chart.bgm_events

    if not bgm_events:
        return BgmReferenceSet(
            groups=(),
            reason_codes=("bgm_event_missing",),
            warnings=(),
            bgm_event_count=0,
        )

    reason_codes: set[TimingReasonCode] = set()
    grouped: dict[tuple[str, int, float], tuple[RemoteObject, list[DtxBgmEvent]]] = {}

    for event in bgm_events:
        wav_path = chart.wav_table.get(event.note_id)
        if wav_path is None:
            reason_codes.add("unresolved_bgm_wav")
            continue
        remote, reason = _resolve_bgm_remote(
            wav_path,
            chart_dir=chart_dir,
            object_prefix=object_prefix,
            objects=objects,
            allow_root_fallback=allow_root_fallback,
        )
        if remote is None:
            assert reason is not None
            reason_codes.add(reason)
            continue
        group_key = (remote.key, event.measure, event.position)
        existing = grouped.get(group_key)
        if existing is None:
            grouped[group_key] = (remote, [event])
        else:
            existing[1].append(event)

    groups = tuple(
        BgmReferenceGroup(
            remote=data[0],
            measure=group_key[1],
            position=group_key[2],
            events=tuple(data[1]),
        )
        for group_key, data in grouped.items()
    )
    groups = tuple(
        sorted(groups, key=lambda group: (group.measure, group.position, group.remote.key))
    )
    return BgmReferenceSet(
        groups=groups,
        reason_codes=tuple(sorted(reason_codes)),
        warnings=(),
        bgm_event_count=len(bgm_events),
    )


def _resolve_bgm_remote(
    wav_path: str,
    *,
    chart_dir: str,
    object_prefix: str,
    objects: tuple[RemoteObject, ...],
    allow_root_fallback: bool,
) -> tuple[RemoteObject | None, TimingReasonCode | None]:
    """Resolve one BGM audio path to a remote, mapping every resolver outcome.

    Returns ``(remote, None)`` on success and ``(None, reason)`` on failure.
    Outcome mapping for :func:`resolve_inventory_object_key`:

    * ``exact`` / ``casefold`` -> the matched :class:`RemoteObject`;
    * ``invalid_path`` -> ``unsafe_bgm_audio_path``;
    * ``ambiguous`` -> ``source_audio_key_ambiguous``;
    * ``missing`` -> one root-relative retry (when ``allow_root_fallback``),
      otherwise ``source_audio_missing``.
    """
    result = resolve_inventory_object_key(
        wav_path,
        base_object_key_dir=chart_dir,
        object_prefix=object_prefix,
        objects=objects,
    )
    if result.status in ("exact", "casefold"):
        return result.remote, None
    if result.status == "invalid_path":
        return None, "unsafe_bgm_audio_path"
    if result.status == "ambiguous":
        return None, "source_audio_key_ambiguous"

    # ``missing``: optionally retry once against the simfile root, then surface
    # ``source_audio_missing`` if still unresolved.
    if allow_root_fallback:
        root_result = resolve_inventory_object_key(
            wav_path,
            base_object_key_dir=object_prefix,
            object_prefix=object_prefix,
            objects=objects,
        )
        if root_result.status in ("exact", "casefold"):
            return root_result.remote, None
        if root_result.status == "invalid_path":
            return None, "unsafe_bgm_audio_path"
        if root_result.status == "ambiguous":
            return None, "source_audio_key_ambiguous"
    return None, "source_audio_missing"


def _chart_object_dir(selected_chart_key: str) -> str:
    """Return the object-key directory of the selected chart.

    For ``"42/real.dtx"`` this is ``"42"``; for ``"42/sub/real.dtx"`` it is
    ``"42/sub"``.  Chart keys always carry the simfile object prefix, so a
    separating ``/`` is always present.
    """
    return selected_chart_key.rsplit("/", 1)[0]


@dataclass(frozen=True)
class BgmResolution:
    """Frozen result of reducing a :class:`BgmReferenceSet` to one reference.

    Carries the selected BGM event (``None`` when no winner could be chosen),
    its chart time in seconds (``None`` when no event is selected), and a
    closed-set ``reason_codes`` / ``warnings`` payload.  ``reason_codes`` only
    ever holds values from :data:`TimingReasonCode`; ``warnings`` are stable,
    deterministic strings.
    """

    selected_event: DtxBgmEvent | None
    chart_time_sec: float | None
    reason_codes: tuple[TimingReasonCode, ...]
    warnings: tuple[str, ...]


def select_bgm_reference(
    references: BgmReferenceSet,
    timing_map: DtxTimingMap,
) -> BgmResolution:
    """Conservatively reduce resolved BGM groups to a single reference point.

    Frozen conservative policy (HPA-323 Task 4 default):

    * **zero groups**  -> ``bgm_event_missing`` (no event selected);
    * **one group**    -> the lowest-``source_order`` event in that group, with
      ``chart_time_sec`` resolved through ``timing_map``; repeated tokens at
      the same identity emit exactly one deterministic warning;
    * **many groups**  -> ``ambiguous_bgm_start`` (no event selected; the row
      is quarantined for inspection).

    The earliest group is never chosen merely to raise yield.  Root-fallback
    handling stays gated on :func:`resolve_bgm_reference_groups`'s
    ``allow_root_fallback`` flag; no unconditional ``used_root_fallback`` field
    is introduced here.  Evidence-based finalization (a real corpus run) may
    revisit this default; until then the conservative quarantine is retained.
    """
    group_count = len(references.groups)
    if group_count == 0:
        return BgmResolution(
            selected_event=None,
            chart_time_sec=None,
            reason_codes=("bgm_event_missing",),
            warnings=(),
        )
    if group_count > 1:
        return BgmResolution(
            selected_event=None,
            chart_time_sec=None,
            reason_codes=("ambiguous_bgm_start",),
            warnings=(),
        )
    group = references.groups[0]
    selected = min(group.events, key=lambda event: event.source_order)
    warnings: tuple[str, ...] = ()
    if len(group.events) > 1:
        token_summary = ",".join(sorted({event.note_id for event in group.events}))
        warnings = (f"repeated_bgm_tokens:{token_summary}:count={len(group.events)}",)
    return BgmResolution(
        selected_event=selected,
        chart_time_sec=timing_map.time_sec(selected),
        reason_codes=(),
        warnings=warnings,
    )
