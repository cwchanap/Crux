"""Reference-timing inputs derived from a validated reference-chart row.

This module is policy-neutral on purpose.  It exposes:

* the closed :data:`TimingReasonCode` contract shared by every timing consumer;
* a typed :class:`BgmReferenceSet` that groups channel-01 BGM control events by
  their stable ``(remote.key, measure, position)`` identity; and
* :func:`resolve_bgm_reference_groups`, which resolves each BGM event's source
  audio through the shared HPA-322 object-key resolver.

Group selection (choosing a single BGM group when several qualify) is deferred
to the corpus diagnostic (HPA-323 Task 4).  No winner is picked here.

HPA-323 Task 5 adds:

* :func:`inspect_source_audio` — header-only source-audio metadata via
  ``soundfile.info`` (no waveform decode);
* :func:`build_audio_relative_events` — bounded audio-relative native reference
  events with a one-frame tolerance band; and
* :func:`render_reference_events` / :data:`REFERENCE_EVENT_SCHEMA` plus
  :func:`validate_schema_golden` for the ``crux.dtx-reference-event/v1``
  canonical JSONL schema golden.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Literal, get_args

import soundfile as sf

from src.benchmark.backend_identity import (
    StrictJsonError,
    canonical_json_bytes,
    strict_json_loads,
)
from src.benchmark.corpus_manifest import publish_immutable_bytes
from src.benchmark.dtx_parser import DtxBgmEvent, ParsedDtxChart
from src.benchmark.inventory_object_keys import resolve_inventory_object_key
from src.benchmark.r2_corpus_models import MAX_SIMFILE_ID, RemoteObject
from src.benchmark.reference_chart_manifest import ReferenceChartRowView
from src.benchmark.timing import DtxTimingMap

#: Canonical schema id for one bounded native reference event per JSONL row.
REFERENCE_EVENT_SCHEMA = "crux.dtx-reference-event/v1"

#: The exception family ``soundfile.info`` raises for an unreadable body.
#: Mirrors :mod:`src.benchmark.render_audio` so audio probing is uniform.
_AUDIO_PROBE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    ValueError,
    sf.LibsndfileError,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DTX_ID_RE = re.compile(r"[0-9A-F]{2}")

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


# ---------------------------------------------------------------------------
# HPA-323 Task 5: source-audio metadata + bounded audio-relative events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceAudioInfo:
    """Header-only metadata for one source-audio file (no waveform decode).

    ``duration_sec`` is derived as ``frames / sample_rate`` so one audio frame
    is exactly ``1.0 / sample_rate`` seconds — that frame is the tolerance band
    used by :func:`build_audio_relative_events`.
    """

    duration_sec: float
    sample_rate: int
    channels: int
    frames: int


@dataclass(frozen=True)
class NativeReferenceEvent:
    """One bounded native reference event expressed in the source-audio clock.

    Identity fields (``simfile_id`` ... ``note_id``) are preserved verbatim from
    the source chart.  ``chart_time_sec`` is the event's resolved chart time;
    ``audio_time_sec`` is the BGM-anchored, clamped time in the source-audio
    frame (``chart_time_sec - bgm_chart_time_sec``, clamped to one-frame of
    ``[0, duration_sec]``).
    """

    simfile_id: int
    selected_chart_key: str
    selected_chart_content_hash: str
    source_audio_key: str
    source_audio_content_hash: str
    source_order: int
    measure: int
    position: float
    lane_id: str
    note_id: str
    chart_time_sec: float
    audio_time_sec: float


@dataclass(frozen=True)
class AudioRelativeReference:
    """Bounded audio-relative reference events plus exclusion bookkeeping.

    ``events`` holds only in-bounds (clamped) events in deterministic native
    identity order.  ``pre_audio_event_count`` / ``post_audio_event_count``
    count the events that fell materially outside ``[0, duration_sec]``.

    ``reason_codes`` is the minimal carrier for the two conditions a caller must
    be able to detect without re-scanning ``events``:

    * ``non_finite_reference_time`` — at least one event had a non-finite audio
      time (it is dropped, not counted in the pre/post counters);
    * ``no_in_bounds_reference_events`` — zero events were retained and no
      non-finite failure was recorded.
    """

    events: tuple[NativeReferenceEvent, ...]
    pre_audio_event_count: int
    post_audio_event_count: int
    reason_codes: tuple[TimingReasonCode, ...]


def inspect_source_audio(path: Path) -> SourceAudioInfo:
    """Return header-only metadata for ``path`` via :func:`soundfile.info`.

    No waveform is decoded.  The unreadable-body failure family is the same as
    :mod:`src.benchmark.render_audio`:
    ``(OSError, RuntimeError, ValueError, soundfile.LibsndfileError)``.
    ``duration_sec`` is ``frames / sample_rate``.
    """
    info = sf.info(str(path))
    frames = int(info.frames)
    sample_rate = int(info.samplerate)
    channels = int(info.channels)
    duration_sec = frames / sample_rate if sample_rate else 0.0
    return SourceAudioInfo(
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        channels=channels,
        frames=frames,
    )


def build_audio_relative_events(
    chart: ParsedDtxChart,
    timing_map: DtxTimingMap,
    *,
    simfile_id: int,
    selected_chart_key: str,
    selected_chart_content_hash: str,
    source_audio_key: str,
    source_audio_content_hash: str,
    bgm_chart_time_sec: float,
    audio: SourceAudioInfo,
) -> AudioRelativeReference:
    """Build bounded, audio-relative native reference events for one chart.

    For each playable event in ``chart.events`` the chart time is resolved
    through ``timing_map`` and shifted by the BGM anchor into the source-audio
    frame: ``audio_time_sec = chart_time_sec - bgm_chart_time_sec``.  One audio
    frame (``1.0 / audio.sample_rate``) is the tolerance band:

    * non-finite ``audio_time_sec``  -> ``non_finite_reference_time``, dropped
      (not counted in the pre/post counters);
    * ``audio_time_sec`` within one frame below ``0``  -> clamped to ``0.0``;
    * ``audio_time_sec`` within one frame above ``duration_sec`` -> clamped to
      ``duration_sec``;
    * materially below ``0`` (``< -one_frame``) -> dropped, ``pre_audio``++;
    * materially above ``duration_sec`` (``> duration_sec + one_frame``) ->
      dropped, ``post_audio``++;
    * zero events retained and no non-finite failure -> ``no_in_bounds``.

    Retained events preserve ``source_order`` and every native identity field;
    ``chart_time_sec`` is untouched and ``audio_time_sec`` is the clamped value.
    Retained events are sorted by ``(measure, position, source_order)`` — a
    deterministic native-identity key (ties are stable against ``chart.events``
    order).
    """
    one_frame = 1.0 / audio.sample_rate if audio.sample_rate else 0.0
    duration_sec = audio.duration_sec

    reason_codes: set[TimingReasonCode] = set()
    retained: list[NativeReferenceEvent] = []
    pre_count = 0
    post_count = 0

    for event in chart.events:
        chart_time_sec = timing_map.time_sec(event)
        audio_time_sec = chart_time_sec - bgm_chart_time_sec

        if not math.isfinite(audio_time_sec):
            reason_codes.add("non_finite_reference_time")
            continue
        if audio_time_sec < -one_frame:
            pre_count += 1
            continue
        if audio_time_sec > duration_sec + one_frame:
            post_count += 1
            continue
        clamped = max(0.0, min(duration_sec, audio_time_sec))
        retained.append(
            NativeReferenceEvent(
                simfile_id=simfile_id,
                selected_chart_key=selected_chart_key,
                selected_chart_content_hash=selected_chart_content_hash,
                source_audio_key=source_audio_key,
                source_audio_content_hash=source_audio_content_hash,
                source_order=event.source_order,
                measure=event.measure,
                position=event.position,
                lane_id=event.lane_id,
                note_id=event.note_id,
                chart_time_sec=chart_time_sec,
                audio_time_sec=clamped,
            )
        )

    if not retained and "non_finite_reference_time" not in reason_codes:
        reason_codes.add("no_in_bounds_reference_events")

    events = tuple(
        sorted(retained, key=lambda item: (item.measure, item.position, item.source_order))
    )
    return AudioRelativeReference(
        events=events,
        pre_audio_event_count=pre_count,
        post_audio_event_count=post_count,
        reason_codes=tuple(sorted(reason_codes)),
    )


# ---------------------------------------------------------------------------
# Canonical JSONL rendering + immutable publication
# ---------------------------------------------------------------------------


def _event_native_identity(event: NativeReferenceEvent) -> tuple[int, float, int, str, str]:
    return (
        event.measure,
        event.position,
        event.source_order,
        event.lane_id,
        event.note_id,
    )


def _event_to_canonical_row(event: NativeReferenceEvent) -> dict[str, object]:
    """Render one event as a canonical-JSON-ready row.

    Float fields are converted to :class:`~decimal.Decimal` so the row renders
    through :func:`canonical_json_bytes` in the exact canonical form that
    :func:`strict_json_loads` (``require_canonical=True``) expects — whole-number
    floats collapse to their bare-integer canonical text (``3.0`` -> ``3``).
    """
    return {
        "simfile_id": event.simfile_id,
        "selected_chart_key": event.selected_chart_key,
        "selected_chart_content_hash": event.selected_chart_content_hash,
        "source_audio_key": event.source_audio_key,
        "source_audio_content_hash": event.source_audio_content_hash,
        "source_order": event.source_order,
        "measure": event.measure,
        "position": Decimal(str(event.position)),
        "lane_id": event.lane_id,
        "note_id": event.note_id,
        "chart_time_sec": Decimal(str(event.chart_time_sec)),
        "audio_time_sec": Decimal(str(event.audio_time_sec)),
    }


def render_reference_events(
    events: tuple[NativeReferenceEvent, ...] | list[NativeReferenceEvent],
) -> bytes:
    """Render events as canonical JSONL (one event per line, one final newline).

    Events are sorted by deterministic native identity
    (``measure, position, source_order``).  Each row is encoded through the
    shared canonical JSON helper (:func:`canonical_json_bytes`), so repeated
    rendering of the same events is byte-identical and SHA-256-stable.
    """
    ordered = sorted(events, key=_event_native_identity)
    return b"".join(
        canonical_json_bytes(_event_to_canonical_row(event), trailing_newline=True)
        for event in ordered
    )


def publish_immutable_content(path: Path, content: bytes, expected_sha256: str) -> None:
    """Publish ``content`` at ``path`` immutably, delegating all durability.

    Thin delegation to :func:`corpus_manifest.publish_immutable_bytes` — the
    public surface of the existing hash-checked, fsync + hardlink + verify
    publisher.  No durability or conflict-handling logic is duplicated here, and
    no leading-underscore symbol is imported across the module boundary.  Raises
    :class:`~src.benchmark.corpus_manifest.ManifestPublicationError` on failure.
    """
    publish_immutable_bytes(path, content, expected_sha256)


# ---------------------------------------------------------------------------
# Schema-golden support for crux.dtx-reference-event/v1
# ---------------------------------------------------------------------------

_REFERENCE_EVENT_KEYS: frozenset[str] = frozenset(
    {
        "simfile_id",
        "selected_chart_key",
        "selected_chart_content_hash",
        "source_audio_key",
        "source_audio_content_hash",
        "source_order",
        "measure",
        "position",
        "lane_id",
        "note_id",
        "chart_time_sec",
        "audio_time_sec",
    }
)
_REFERENCE_EVENT_INT_KEYS: frozenset[str] = frozenset({"simfile_id", "source_order", "measure"})
_REFERENCE_EVENT_STRING_KEYS: frozenset[str] = frozenset(
    {"selected_chart_key", "source_audio_key", "lane_id", "note_id"}
)
_REFERENCE_EVENT_HASH_KEYS: frozenset[str] = frozenset(
    {"selected_chart_content_hash", "source_audio_content_hash"}
)
_REFERENCE_EVENT_TIME_KEYS: frozenset[str] = frozenset(
    {"position", "chart_time_sec", "audio_time_sec"}
)


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Validate a ``crux.dtx-reference-event/v1`` canonical JSONL golden.

    Asserts canonical JSONL (one final newline, no blank lines, each line
    canonical via :func:`strict_json_loads` with ``require_canonical=True``),
    the exact 12-key row set, finite ``int``/``Decimal`` time fields (both
    canonical tokens the renderer emits for whole- and fractional-number times),
    integer identity counters, lowercase-SHA-256 hash fields, and byte-identity
    against a re-render through the canonical JSON helper.  Raises
    :class:`ValueError` (or its :class:`StrictJsonError` subclass) on any drift.
    """
    if schema != REFERENCE_EVENT_SCHEMA:
        raise ValueError("unsupported schema golden")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("reference event golden must be canonical JSONL with one final newline")

    lines = content.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") or line == b"\n" for line in lines):
        raise ValueError("reference event golden must contain one record per line")

    try:
        rows = tuple(strict_json_loads(line[:-1], require_canonical=True) for line in lines)
    except StrictJsonError:
        raise ValueError("reference event golden must be canonical JSONL") from None

    for row in rows:
        _validate_reference_event_row(row)

    _validate_reference_event_sequence(rows)

    regenerated = b"".join(canonical_json_bytes(row, trailing_newline=True) for row in rows)
    if regenerated != content:
        raise ValueError("reference event golden is not byte-identical to its canonical render")


def _validate_reference_event_row(row: object) -> None:
    if not isinstance(row, dict) or set(row) != _REFERENCE_EVENT_KEYS:
        raise ValueError("reference event row has an invalid key set")
    for key in _REFERENCE_EVENT_INT_KEYS:
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    simfile_id = row["simfile_id"]
    if not 0 <= simfile_id <= MAX_SIMFILE_ID:
        raise ValueError("simfile_id is outside the repository bounds")
    for key in ("source_order", "measure"):
        if row[key] < 0:
            raise ValueError(f"{key} must be non-negative")
    for key in _REFERENCE_EVENT_STRING_KEYS:
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f"{key} must be a non-empty string")
    for key in ("lane_id", "note_id"):
        value = row[key]
        if _DTX_ID_RE.fullmatch(value) is None:
            raise ValueError(f"{key} must be a two-digit uppercase hexadecimal ID")
    for key in ("selected_chart_key", "source_audio_key"):
        if not _is_safe_event_object_key(row[key], simfile_id):
            raise ValueError(f"{key} must be a safe simfile object key")
    for key in _REFERENCE_EVENT_HASH_KEYS:
        value = row[key]
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{key} must be a lowercase SHA-256")
    for key in _REFERENCE_EVENT_TIME_KEYS:
        value = row[key]
        # The canonical renderer (backend_identity._render_decimal) collapses
        # whole-number Decimals to bare-integer tokens (Decimal("0.0") -> "0"),
        # and strict_json_loads parses integer tokens back as ``int`` (float
        # tokens come back as ``Decimal`` via parse_float).  Both representations
        # are legitimate for a time field, so accept ``int`` and ``Decimal``
        # (rejecting ``bool``, which is an ``int`` subclass) and require finite.
        if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
            raise ValueError(f"{key} must be a finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{key} must be a finite number")
    position = row["position"]
    if not 0 <= position < 1:
        raise ValueError("position must be in the half-open range [0, 1)")


def _is_safe_event_object_key(value: object, simfile_id: int) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        return False
    if any(part in {"", ".", ".."} for part in path.parts) or len(path.parts) < 2:
        return False
    first_segment = path.parts[0]
    return (
        first_segment.isascii() and first_segment.isdecimal() and int(first_segment) == simfile_id
    )


def _validate_reference_event_sequence(rows: tuple[object, ...]) -> None:
    typed_rows = tuple(row for row in rows if isinstance(row, dict))
    identity_prefixes = {
        (
            row["simfile_id"],
            row["selected_chart_key"],
            row["selected_chart_content_hash"],
            row["source_audio_key"],
            row["source_audio_content_hash"],
        )
        for row in typed_rows
    }
    if len(identity_prefixes) != 1:
        raise ValueError("reference event rows have mixed source identity")

    ordered_keys = tuple(
        (
            row["measure"],
            row["position"],
            row["source_order"],
            row["lane_id"],
            row["note_id"],
        )
        for row in typed_rows
    )
    if ordered_keys != tuple(sorted(ordered_keys)):
        raise ValueError("reference event rows are out of order")
    if len(ordered_keys) != len(set(ordered_keys)):
        raise ValueError("reference event rows contain a duplicate native identity")
