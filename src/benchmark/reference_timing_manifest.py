"""Reference-timing manifest contract and orchestration (HPA-323 Tasks 6a/6b).

Task 6a defined the *pure contract* for the reference-timing manifest — loading
a validated HPA-322 reference-chart manifest, remapping each row into a
timing-lineage row, rendering that row through the shared canonical-JSONL
publisher, accounting for the run outcome, and validating the
``crux.reference-timing-manifest/v1`` schema golden.

Task 6b adds the orchestration (:func:`run_reference_timing`) that populates
those rows with real chart / audio / event data.  The flow is:

1. load the HPA-322 manifest once through :func:`load_reference_chart_manifest`;
2. first pass per row — verify the selected DTX body once, parse it, build the
   timing map, resolve BGM groups and apply the Task 4 frozen selection policy,
   then resolve the selected source-audio cache body when already verified or
   queue the exact audio key for a targeted R2 fill;
3. targeted fill — only when audio misses exist, lazily touch the optional R2
   dependency, validate the resolved config identity against the embedded
   source endpoint/bucket, fill the exact selected audio keys, merge the
   returned inventories by simfile ID, and reverify only rows whose inventory
   changed;
4. metadata + event publication — inspect the resolved audio, build the bounded
   audio-relative native events, and publish ``events/<sha256>.jsonl``
   immutably (event publication failure is fatal);
5. render + publish the timing manifest and latest pointer, then build the
   :class:`ReferenceTimingOutcome`.

A complete-cache run never touches the optional R2 store: the dependency check,
config resolution, store factory, cache-index load, and exact-key sync are all
gated behind the audio-miss branch, so the offline acceptance path stays free
of the ``boto3`` extra.

Lineage (Brief Step 2):  every HPA-322 field is carried through verbatim
except the top-level ``corpus_version`` (re-derived by ``render_manifest``).
``source_manifest_sha256`` / ``source_corpus_version`` — which already point at
the upstream HPA-321 corpus manifest inside the HPA-322 row — are preserved
unchanged.  Two new fields record the HPA-322 manifest identity itself:

* ``source_reference_chart_manifest_sha256`` — exact input-byte SHA-256;
* ``source_reference_chart_version``         — the HPA-322 input ``corpus_version``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Literal, get_args

import soundfile as sf

from src.benchmark.backend_identity import StrictJsonError, require_sha256, strict_json_loads
from src.benchmark.corpus_cache import (
    CacheIndexStore,
    cache_writer_lock,
    read_verified_cache_body,
    resolve_verified_cache_body,
    sync_explicit_cache_keys,
)
from src.benchmark.corpus_manifest import (
    ManifestPublicationError,
    ManifestRowView,
    publish_latest_manifest,
    publish_manifest,
    render_manifest,
)
from src.benchmark.dtx_parser import ParsedDtxChart, parse_dtx_bytes
from src.benchmark.durability import ensure_durable_directory
from src.benchmark.r2_corpus_models import (
    CacheSyncResult,
    PublishedManifest,
    R2Config,
    RemoteObject,
    SimfileInventory,
)
from src.benchmark.r2_inventory import (
    R2ObjectStore,
    R2StoreError,
    create_boto3_store,
    ensure_r2_dependency,
)
from src.benchmark.reference_chart_manifest import (
    REFERENCE_CHART_MANIFEST_SCHEMA,
    ReferenceChartRowView,
    reference_chart_row_view_from_row,
)
from src.benchmark.reference_timing import (
    BgmResolution,
    TimingReasonCode,
    build_audio_relative_events,
    inspect_source_audio,
    publish_immutable_content,
    render_reference_events,
    resolve_bgm_reference_groups,
    select_bgm_reference,
)
from src.benchmark.timing import DtxTimingMap, build_dtx_timing_map

#: Canonical schema id for the derived reference-timing manifest rows.
REFERENCE_TIMING_MANIFEST_SCHEMA = "crux.reference-timing-manifest/v1"

#: The timing-semantics family the rows were produced under.
TIMING_SEMANTICS_VERSION = "crux.dtx-audio-timing/v1"

#: Every stable timing reason code (mirrors :data:`TimingReasonCode` via ``get_args``).
_TIMING_REASON_CODES: frozenset[str] = frozenset(get_args(TimingReasonCode))

#: Timing-specific keys added to every derived row.
_TIMING_SPECIFIC_KEYS: frozenset[str] = frozenset(
    {
        "timing_semantics_version",
        "timing_status",
        "timing_reason_codes",
        "timing_warnings",
        "source_audio_key",
        "source_audio_content_hash",
        "reference_events_cache_path",
    }
)

#: Lineage keys added to every derived row (the HPA-322 manifest identity).
_TIMING_LINEAGE_KEYS: frozenset[str] = frozenset(
    {
        "source_reference_chart_manifest_sha256",
        "source_reference_chart_version",
    }
)

_TIMING_STATUSES: frozenset[str] = frozenset({"ready", "quarantined"})

#: The exception family :func:`inspect_source_audio` raises for an unreadable
#: body.  Mirrors :mod:`src.benchmark.reference_timing` so audio probing is
#: uniform across the timing pipeline.
_AUDIO_PROBE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    ValueError,
    sf.LibsndfileError,
)

#: The directory (under ``output_dir``) that holds the immutable events
#: artifacts, and the manifest row prefix that points at them.
_EVENTS_DIR_NAME = "events"

StoreFactory = Callable[[R2Config], R2ObjectStore]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ReferenceTimingRequest:
    manifest_path: Path
    cache_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class ReferenceTimingOutcome:
    status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    manifest: PublishedManifest | None
    ready_count: int
    quarantined_count: int
    upstream_quarantined_count: int
    events_published: int


@dataclass(frozen=True)
class TimingRowResolution:
    """The per-row timing decision consumed by :func:`build_timing_row`.

    ``status`` is the timing-layer verdict (independent of the upstream HPA-322
    ``selection_status``): an HPA-322 *selected* row may still be timing
    *quarantined* when timing analysis fails.  ``reason_codes`` only ever holds
    values from :data:`TimingReasonCode`.  The source-audio / events identity is
    populated for ``ready`` rows and ``None`` for ``quarantined`` rows.
    """

    status: Literal["ready", "quarantined"]
    reason_codes: tuple[TimingReasonCode, ...]
    warnings: tuple[str, ...]
    source_audio_key: str | None
    source_audio_content_hash: str | None
    reference_events_cache_path: str | None


def upstream_chart_unavailable_resolution() -> TimingRowResolution:
    """The canonical timing resolution for an upstream-quarantined HPA-322 row.

    When the HPA-322 row was already quarantined at chart selection, timing
    analysis cannot run: there is no selected chart to time.  The row carries
    the single reason code ``upstream_chart_selection_unavailable`` and null
    source-audio / events identity.
    """
    return TimingRowResolution(
        status="quarantined",
        reason_codes=("upstream_chart_selection_unavailable",),
        warnings=(),
        source_audio_key=None,
        source_audio_content_hash=None,
        reference_events_cache_path=None,
    )


@dataclass(frozen=True)
class _ValidatedReferenceChartRow:
    source_row: Mapping[str, object]
    view: ReferenceChartRowView


@dataclass(frozen=True)
class _LoadedReferenceChartManifest:
    source_reference_chart_manifest_sha256: str
    source_reference_chart_version: str
    rows: tuple[_ValidatedReferenceChartRow, ...]


def load_reference_chart_manifest(path: Path) -> _LoadedReferenceChartManifest:
    """Load and validate a canonical HPA-322 reference-chart manifest.

    Mirrors the canonical-JSONL loader used for HPA-321 source manifests but
    validates each row through :func:`reference_chart_row_view_from_row` (the
    merged HPA-322 reference-chart validator) — never the HPA-321-only
    :func:`manifest_row_view_from_row`.  Records the exact input-byte SHA-256
    and the shared HPA-322 ``corpus_version`` for downstream lineage.
    """
    try:
        content = path.read_bytes()
    except OSError:
        raise ValueError("reference chart manifest is unavailable") from None

    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("reference chart manifest must contain canonical JSONL records")

    rows: list[_ValidatedReferenceChartRow] = []
    simfile_ids: set[int] = set()
    source_identity: tuple[str, str, str, str, str] | None = None
    for line in content.splitlines(keepends=True):
        if not line.endswith(b"\n") or line == b"\n":
            raise ValueError("reference chart manifest must contain canonical JSONL records")
        try:
            source_row = strict_json_loads(line[:-1], require_canonical=True)
        except StrictJsonError:
            raise ValueError(
                "reference chart manifest must contain canonical JSONL records"
            ) from None
        if (
            not isinstance(source_row, dict)
            or source_row.get("schema_version") != REFERENCE_CHART_MANIFEST_SCHEMA
        ):
            raise ValueError("reference chart manifest contains an unsupported row")
        try:
            view = reference_chart_row_view_from_row(source_row)
        except ValueError:
            raise ValueError(
                "reference chart manifest contains an invalid reference chart row"
            ) from None

        identity = (
            view.corpus_version,
            view.source.source_endpoint_sha256,
            view.source.source_bucket,
            view.source.cache_profile,
            view.source.source_discovery_method,
        )
        if source_identity is None:
            source_identity = identity
        elif identity != source_identity:
            raise ValueError("reference chart manifest contains mixed source identity")
        if view.simfile_id in simfile_ids:
            raise ValueError("reference chart manifest contains duplicate simfile IDs")
        simfile_ids.add(view.simfile_id)
        rows.append(_ValidatedReferenceChartRow(MappingProxyType(source_row), view))

    if not rows:
        raise ValueError("reference chart manifest contains no records")
    normalized_rows = tuple(
        {key: value for key, value in validated.source_row.items() if key != "corpus_version"}
        for validated in rows
    )
    rendered = render_manifest(normalized_rows)
    assert source_identity is not None
    if rendered.content != content or rendered.corpus_version != source_identity[0]:
        raise ValueError("reference chart manifest has an invalid derived corpus version")
    return _LoadedReferenceChartManifest(
        source_reference_chart_manifest_sha256=sha256(content).hexdigest(),
        source_reference_chart_version=source_identity[0],
        rows=tuple(rows),
    )


def build_timing_row(
    validated: _ValidatedReferenceChartRow,
    *,
    source_reference_chart_manifest_sha256: str,
    source_reference_chart_version: str,
    timing: TimingRowResolution,
) -> dict[str, object]:
    """Remap one validated HPA-322 row into a derived timing-manifest row.

    Lineage remap (Brief Step 2):

    * ``schema_version`` is set to :data:`REFERENCE_TIMING_MANIFEST_SCHEMA`;
    * the HPA-322 top-level ``corpus_version`` is removed (``render_manifest``
      re-derives it);
    * ``source_manifest_sha256`` / ``source_corpus_version`` are carried through
      unchanged from the HPA-322 row;
    * ``source_reference_chart_manifest_sha256`` /
      ``source_reference_chart_version`` record the HPA-322 manifest identity.

    Every other HPA-322 field is passed through verbatim.  Reason codes and
    warnings are sorted so canonical rendering is byte-stable.
    """
    row = dict(validated.source_row)
    row.pop("corpus_version")
    row["schema_version"] = REFERENCE_TIMING_MANIFEST_SCHEMA
    row["source_reference_chart_manifest_sha256"] = source_reference_chart_manifest_sha256
    row["source_reference_chart_version"] = source_reference_chart_version
    row["timing_semantics_version"] = TIMING_SEMANTICS_VERSION
    row["timing_status"] = timing.status
    row["timing_reason_codes"] = sorted(timing.reason_codes)
    row["timing_warnings"] = sorted(timing.warnings)
    row["source_audio_key"] = timing.source_audio_key
    row["source_audio_content_hash"] = timing.source_audio_content_hash
    row["reference_events_cache_path"] = timing.reference_events_cache_path
    _validate_timing_status_shape(row)
    return row


def build_reference_timing_outcome(
    *,
    manifest: PublishedManifest,
    total_input_rows: int,
    ready_count: int,
    quarantined_count: int,
    upstream_quarantined_count: int,
    events_published: int,
) -> ReferenceTimingOutcome:
    """Build a successful :class:`ReferenceTimingOutcome`, enforcing invariants.

    Raises ``ValueError`` if the pure accounting invariants do not hold:

    * ``ready_count + quarantined_count == total_input_rows``;
    * ``upstream_quarantined_count <= quarantined_count``;
    * ``events_published == ready_count``.

    The exit convention is preserved: no quarantines -> ``0``; any quarantine
    with a published manifest -> ``1``.  Fatal loading / publication failures
    use :func:`failed_reference_timing_outcome` (exit ``2``).
    """
    if ready_count + quarantined_count != total_input_rows:
        raise ValueError("reference timing outcome must balance input rows")
    if upstream_quarantined_count > quarantined_count:
        raise ValueError("reference timing upstream quarantine exceeds total quarantine")
    if events_published != ready_count:
        raise ValueError("reference timing events published must equal ready rows")
    if quarantined_count == 0:
        status: Literal["complete", "partial"] = "complete"
        exit_code: Literal[0, 1, 2] = 0
    else:
        status = "partial"
        exit_code = 1
    return ReferenceTimingOutcome(
        status=status,
        exit_code=exit_code,
        manifest=manifest,
        ready_count=ready_count,
        quarantined_count=quarantined_count,
        upstream_quarantined_count=upstream_quarantined_count,
        events_published=events_published,
    )


def failed_reference_timing_outcome() -> ReferenceTimingOutcome:
    """The exit-2 outcome for a fatal loading / publication-prep failure."""
    return ReferenceTimingOutcome(
        status="failed",
        exit_code=2,
        manifest=None,
        ready_count=0,
        quarantined_count=0,
        upstream_quarantined_count=0,
        events_published=0,
    )


# ---------------------------------------------------------------------------
# HPA-323 Task 6b: cache/R2 orchestration, event publication, manifest run
# ---------------------------------------------------------------------------


@dataclass
class _RowTimingState:
    """Mutable per-row working state carried across the orchestration phases.

    ``resolution`` is finalised by the end of :func:`_finalise_row_state`; until
    then the row is either upstream-quarantined (``is_upstream``), pending audio
    fill (``pending_audio_key`` set), or awaiting event publication
    (``audio_path`` set with no resolution yet).
    """

    validated: _ValidatedReferenceChartRow
    is_upstream: bool
    chart: ParsedDtxChart | None = None
    timing_map: DtxTimingMap | None = None
    bgm: BgmResolution | None = None
    audio_remote: RemoteObject | None = None
    audio_path: Path | None = None
    pending_audio_key: str | None = None
    resolution: TimingRowResolution | None = None


def _timing_quarantine(
    reason_codes: tuple[TimingReasonCode, ...] | tuple[str, ...],
    *,
    warnings: tuple[str, ...] = (),
) -> TimingRowResolution:
    return TimingRowResolution(
        status="quarantined",
        reason_codes=tuple(reason_codes),  # type: ignore[arg-type]
        warnings=warnings,
        source_audio_key=None,
        source_audio_content_hash=None,
        reference_events_cache_path=None,
    )


def run_reference_timing(
    request: ReferenceTimingRequest,
    *,
    environ: Mapping[str, str] | None = None,
    dependency_check: Callable[[], None] = ensure_r2_dependency,
    store_factory: StoreFactory = create_boto3_store,
    clock: Clock = _utc_now,
) -> ReferenceTimingOutcome:
    """Build and publish the reference-timing manifest + events for one run.

    Dependency injection mirrors :func:`sync_r2_corpus`: ``environ``,
    ``dependency_check``, ``store_factory``, and ``clock`` default to the
    production R2 machinery but accept fakes so the no-R2 and targeted-fill
    paths are testable offline.  The optional R2 store is only ever touched
    when at least one selected audio body is missing from the cache.

    Returns :func:`failed_reference_timing_outcome` (exit ``2``) when manifest
    loading, R2 config/index resolution, or event/manifest publication fails
    fatally; otherwise a balanced :class:`ReferenceTimingOutcome` whose exit
    code is ``0`` (all rows ready) or ``1`` (any row quarantined).
    """
    try:
        loaded = load_reference_chart_manifest(request.manifest_path)
    except ValueError:
        return failed_reference_timing_outcome()
    try:
        return _run_reference_timing(
            loaded=loaded,
            request=request,
            environ=os.environ if environ is None else environ,
            dependency_check=dependency_check,
            store_factory=store_factory,
            clock=clock,
        )
    except (ManifestPublicationError, R2StoreError, ValueError, RuntimeError, OSError):
        return failed_reference_timing_outcome()


def _run_reference_timing(
    *,
    loaded: _LoadedReferenceChartManifest,
    request: ReferenceTimingRequest,
    environ: Mapping[str, str],
    dependency_check: Callable[[], None],
    store_factory: StoreFactory,
    clock: Clock,
) -> ReferenceTimingOutcome:
    cache_dir = request.cache_dir
    output_dir = request.output_dir

    # Phase 1: first pass per row — verify chart bytes once, parse, build the
    # timing map, resolve BGM groups, and resolve already-verified audio or
    # queue the exact audio key for a targeted fill.
    states: list[_RowTimingState] = [
        _first_pass_row(validated, cache_dir=cache_dir) for validated in loaded.rows
    ]

    # Phase 2/3: targeted R2 fill — only when audio misses exist.  A complete
    # cache never reaches this block, so the optional R2 dependency, config,
    # store factory, cache-index load, and exact-key sync all stay untouched.
    pending = [state for state in states if state.pending_audio_key is not None]
    if pending:
        _fill_pending_audio(
            pending,
            loaded=loaded,
            cache_dir=cache_dir,
            environ=environ,
            dependency_check=dependency_check,
            store_factory=store_factory,
        )

    # Phase 4: metadata + event publication for resolved rows, then finalise
    # every row's timing resolution.  Event publication failure is fatal and
    # propagates to :func:`run_reference_timing` as an exit-2 outcome.
    for state in states:
        _finalise_row_state(state, output_dir=output_dir)

    ready_count = 0
    quarantined_count = 0
    upstream_quarantined_count = 0
    timing_rows: list[dict[str, object]] = []
    for state in states:
        assert state.resolution is not None  # every row is finalised above
        if state.is_upstream:
            upstream_quarantined_count += 1
        if state.resolution.status == "ready":
            ready_count += 1
        else:
            quarantined_count += 1
        timing_rows.append(
            build_timing_row(
                state.validated,
                source_reference_chart_manifest_sha256=loaded.source_reference_chart_manifest_sha256,
                source_reference_chart_version=loaded.source_reference_chart_version,
                timing=state.resolution,
            )
        )

    # Phase 5/6: render + publish the timing manifest and the latest pointer.
    rendered = render_manifest(tuple(timing_rows))
    published = publish_manifest(output_dir, rendered)
    overall_status: Literal["complete", "partial"] = (
        "complete" if quarantined_count == 0 else "partial"
    )
    publish_latest_manifest(output_dir, published, overall_status, clock())
    return build_reference_timing_outcome(
        manifest=published,
        total_input_rows=len(states),
        ready_count=ready_count,
        quarantined_count=quarantined_count,
        upstream_quarantined_count=upstream_quarantined_count,
        events_published=ready_count,
    )


def _first_pass_row(
    validated: _ValidatedReferenceChartRow,
    *,
    cache_dir: Path,
) -> _RowTimingState:
    """Run the first-pass timing analysis for one row.

    Upstream HPA-322 quarantines carry straight through as
    ``upstream_chart_selection_unavailable``.  Selected rows verify the chart
    bytes once through :func:`read_verified_cache_body`, parse them, build the
    timing map, resolve BGM groups, and apply the frozen selection policy.  The
    selected source-audio body is resolved once when already verified, else
    queued for fill.  Row-local failures map to :data:`TimingReasonCode`
    without aborting sibling rows.
    """
    view = validated.view
    state = _RowTimingState(validated=validated, is_upstream=view.selection_status != "selected")
    if state.is_upstream:
        state.resolution = upstream_chart_unavailable_resolution()
        return state

    assert view.selected_chart is not None  # the merged HPA-322 validator guarantees this
    endpoint = view.source.source_endpoint_sha256
    bucket = view.source.source_bucket

    try:
        chart_bytes = read_verified_cache_body(
            cache_dir,
            view.selected_chart,
            source_endpoint_sha256=endpoint,
            bucket=bucket,
            expected_sha256=view.selected_chart_content_hash,
        )
    except ValueError:
        state.resolution = _timing_quarantine(("selected_chart_cache_invalid",))
        return state

    try:
        chart = parse_dtx_bytes(
            chart_bytes,
            chart_id=view.selected_chart.key,
            source_name=view.selected_chart.key,
        )
    except Exception:  # any decode/parse failure quarantines this row only
        state.resolution = _timing_quarantine(("selected_chart_parse_failed",))
        return state

    try:
        timing_map = build_dtx_timing_map(chart)
    except Exception:
        state.resolution = _timing_quarantine(("timing_map_invalid",))
        return state

    references = resolve_bgm_reference_groups(
        chart,
        selected_chart_key=view.selected_chart.key,
        row=view,
        allow_root_fallback=True,
    )
    bgm = select_bgm_reference(references, timing_map)
    combined_reasons = tuple(sorted(set(references.reason_codes) | set(bgm.reason_codes)))
    if combined_reasons:
        state.resolution = _timing_quarantine(combined_reasons, warnings=bgm.warnings)
        return state

    # Exactly one BGM group survived (the frozen policy guarantees this when no
    # reason codes are present); its remote is the selected source audio.
    (group,) = references.groups
    audio_remote = group.remote
    state.chart = chart
    state.timing_map = timing_map
    state.bgm = bgm
    state.audio_remote = audio_remote

    if audio_remote.cache_status == "verified":
        try:
            state.audio_path = resolve_verified_cache_body(
                cache_dir,
                audio_remote,
                source_endpoint_sha256=endpoint,
                bucket=bucket,
            )
        except ValueError:
            state.resolution = _timing_quarantine(("source_audio_cache_invalid",))
    else:
        state.pending_audio_key = audio_remote.key
    return state


def _fill_pending_audio(
    pending: list[_RowTimingState],
    *,
    loaded: _LoadedReferenceChartManifest,
    cache_dir: Path,
    environ: Mapping[str, str],
    dependency_check: Callable[[], None],
    store_factory: StoreFactory,
) -> None:
    """Fill the exact selected audio keys through the optional R2 store.

    Validates the resolved R2 config identity against the embedded source
    endpoint/bucket, loads the existing cache index under the writer lock,
    builds and validates the store, fills only the exact selected audio keys,
    merges the returned inventories by simfile ID, and reverifies only the
    rows whose inventory changed.  Already-verified rows (resolved in the first
    pass) bypass this entirely.
    """
    dependency_check()
    config = R2Config.from_environ(environ)

    source = loaded.rows[0].view.source
    if (
        config.source_endpoint_sha256 != source.source_endpoint_sha256
        or config.bucket != source.source_bucket
    ):
        raise ValueError("R2 config identity does not match the reference chart source")

    selected_keys = frozenset(
        state.pending_audio_key for state in pending if state.pending_audio_key
    )
    inventories_by_id: dict[int, SimfileInventory] = {}
    for state in pending:
        inventory = state.validated.view.source.inventory
        inventories_by_id.setdefault(inventory.simfile_id, inventory)
    simfiles = tuple(inventories_by_id.values())

    with cache_writer_lock(cache_dir):
        index = CacheIndexStore.load(cache_dir)
        store = store_factory(config)
        store.validate_bucket()
        result = sync_explicit_cache_keys(
            simfiles,
            store,
            index,
            config,
            selected_keys=selected_keys,
        )

    _merge_filled_audio(pending, result, cache_dir=cache_dir, source=source)


def _merge_filled_audio(
    pending: list[_RowTimingState],
    result: CacheSyncResult,
    *,
    cache_dir: Path,
    source: ManifestRowView,
) -> None:
    """Merge rebuilt inventories and reverify only rows whose audio changed.

    Maps download failures to ``source_audio_download_failed`` and a verified
    body that cannot be re-resolved to ``source_audio_cache_invalid``.
    Unrelated object records are preserved exactly because the rebuilt
    inventory replaces the row's audio remote in place.
    """
    rebuilt_by_id = {inventory.simfile_id: inventory for inventory in result.simfiles}
    for state in pending:
        inventory = state.validated.view.source.inventory
        rebuilt = rebuilt_by_id.get(inventory.simfile_id, inventory)
        assert state.pending_audio_key is not None
        audio = next(
            (remote for remote in rebuilt.objects if remote.key == state.pending_audio_key),
            None,
        )
        if audio is None or audio.cache_status != "verified":
            state.resolution = _timing_quarantine(("source_audio_download_failed",))
            continue
        try:
            state.audio_path = resolve_verified_cache_body(
                cache_dir,
                audio,
                source_endpoint_sha256=source.source_endpoint_sha256,
                bucket=source.source_bucket,
            )
            state.audio_remote = audio
        except ValueError:
            state.resolution = _timing_quarantine(("source_audio_cache_invalid",))


def _finalise_row_state(state: _RowTimingState, *, output_dir: Path) -> None:
    """Inspect resolved audio, build + publish events, and finalise the row.

    Rows already carrying a resolution (upstream quarantine or a first-pass /
    fill failure) pass through unchanged.  Resolved rows probe the source
    audio, build the bounded audio-relative events, publish the immutable
    events artifact, and become ``ready``; any reason code from the bounded
    builder quarantines the row instead.
    """
    if state.resolution is not None:
        return
    if state.audio_path is None or state.audio_remote is None:
        # A pending fill that did not resolve should already carry a resolution;
        # stay defensive and quarantine rather than crash the whole run.
        state.resolution = _timing_quarantine(("source_audio_download_failed",))
        return

    view = state.validated.view
    assert state.chart is not None
    assert state.timing_map is not None
    assert state.bgm is not None
    assert state.bgm.chart_time_sec is not None
    assert view.selected_chart is not None
    audio_remote = state.audio_remote
    audio_key = audio_remote.key
    audio_content_hash = audio_remote.sha256
    assert isinstance(audio_content_hash, str)

    try:
        audio_info = inspect_source_audio(state.audio_path)
    except _AUDIO_PROBE_ERRORS:
        state.resolution = _timing_quarantine(("source_audio_decode_failed",))
        return

    events = build_audio_relative_events(
        state.chart,
        state.timing_map,
        simfile_id=view.simfile_id,
        selected_chart_key=view.selected_chart.key,
        selected_chart_content_hash=view.selected_chart_content_hash,
        source_audio_key=audio_key,
        source_audio_content_hash=audio_content_hash,
        bgm_chart_time_sec=state.bgm.chart_time_sec,
        audio=audio_info,
    )
    if events.reason_codes:
        state.resolution = _timing_quarantine(events.reason_codes)
        return

    content = render_reference_events(events.events)
    events_sha256 = sha256(content).hexdigest()
    events_relative_path = f"{_EVENTS_DIR_NAME}/{events_sha256}.jsonl"
    events_dir = output_dir / _EVENTS_DIR_NAME
    ensure_durable_directory(events_dir)
    publish_immutable_content(
        events_dir / f"{events_sha256}.jsonl",
        content,
        events_sha256,
    )
    state.resolution = TimingRowResolution(
        status="ready",
        reason_codes=(),
        warnings=state.bgm.warnings,
        source_audio_key=audio_key,
        source_audio_content_hash=audio_content_hash,
        reference_events_cache_path=events_relative_path,
    )


# ---------------------------------------------------------------------------
# Schema-golden validation for crux.reference-timing-manifest/v1
# ---------------------------------------------------------------------------


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Validate a ``crux.reference-timing-manifest/v1`` canonical JSONL golden.

    Asserts canonical JSONL (one final newline, no blank lines, each line
    canonical), exactly two records (one ready and one quarantined), a single
    shared HPA-322 source identity, a single derived ``corpus_version`` that
    round-trips through :func:`render_manifest`, and — per row — a valid
    timing payload whose underlying HPA-322 reference-chart fields re-validate
    through :func:`reference_chart_row_view_from_row`.  Reason codes are
    validated against ``get_args(TimingReasonCode)``.
    """
    if schema != REFERENCE_TIMING_MANIFEST_SCHEMA:
        raise ValueError("unsupported schema golden")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("reference timing manifest golden must be canonical JSONL")

    lines = content.splitlines(keepends=True)
    if len(lines) != 2 or any(not line.endswith(b"\n") or line == b"\n" for line in lines):
        raise ValueError("reference timing manifest golden must contain exactly two records")
    try:
        rows = tuple(strict_json_loads(line[:-1], require_canonical=True) for line in lines)
    except StrictJsonError:
        raise ValueError("reference timing manifest golden must be canonical JSONL") from None
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("reference timing manifest golden rows must be objects")

    for row in rows:
        _validate_timing_manifest_row(row)

    statuses = [row["timing_status"] for row in rows]
    if sorted(statuses) != ["quarantined", "ready"]:
        raise ValueError(
            "reference timing manifest golden requires one ready and one quarantined row"
        )

    source_identities = {
        (
            row["source_reference_chart_manifest_sha256"],
            row["source_reference_chart_version"],
            row["source_endpoint_sha256"],
            row["source_bucket"],
            row["cache_profile"],
            row["source_discovery_method"],
        )
        for row in rows
    }
    if len(source_identities) != 1:
        raise ValueError("reference timing manifest golden contains mixed source identity")

    derived_versions = {row["corpus_version"] for row in rows}
    if len(derived_versions) != 1:
        raise ValueError("reference timing manifest golden contains mixed corpus version")
    (derived_version,) = derived_versions
    if not _is_corpus_version(derived_version):
        raise ValueError("reference timing manifest golden has an invalid corpus version")
    normalized_rows = tuple(
        {key: value for key, value in row.items() if key != "corpus_version"} for row in rows
    )
    rendered = render_manifest(normalized_rows)
    if rendered.corpus_version != derived_version or rendered.content != content:
        raise ValueError("reference timing manifest golden has an invalid derived corpus version")


def _validate_timing_manifest_row(row: Mapping[str, object]) -> None:
    # Timing-specific + lineage keys must be present before the HPA-322
    # reconstruction below reads ``source_reference_chart_version``.
    if not (_TIMING_SPECIFIC_KEYS | _TIMING_LINEAGE_KEYS) <= set(row):
        raise ValueError("reference timing manifest row has an invalid key set")
    if "schema_version" not in row or "corpus_version" not in row:
        raise ValueError("reference timing manifest row has an invalid key set")

    # Reconstruct the HPA-322 row and delegate the complete key-set, schema,
    # digest, cache-path, DLEVEL, and selected/nullability contract to the
    # merged reference-chart validator.  This both validates the pass-through
    # payload and rejects any unknown / missing key.
    hpa322_row = {
        key: value
        for key, value in row.items()
        if key not in _TIMING_SPECIFIC_KEYS
        and key not in _TIMING_LINEAGE_KEYS
        and key != "corpus_version"
    }
    hpa322_row["schema_version"] = REFERENCE_CHART_MANIFEST_SCHEMA
    hpa322_row["corpus_version"] = row["source_reference_chart_version"]
    try:
        reference_chart_row_view_from_row(hpa322_row)
    except ValueError:
        raise ValueError(
            "reference timing manifest row has an invalid reference chart payload"
        ) from None

    if row["schema_version"] != REFERENCE_TIMING_MANIFEST_SCHEMA:
        raise ValueError("reference timing manifest row has an unsupported schema")
    if row["timing_semantics_version"] != TIMING_SEMANTICS_VERSION:
        raise ValueError(
            "reference timing manifest row has an unsupported timing semantics version"
        )
    _require_sha256_value(
        row["source_reference_chart_manifest_sha256"],
        "source_reference_chart_manifest_sha256",
    )
    if not _is_corpus_version(row["source_reference_chart_version"]):
        raise ValueError(
            "reference timing manifest row has an invalid source reference chart version"
        )
    if not _is_corpus_version(row["corpus_version"]):
        raise ValueError("reference timing manifest row has an invalid corpus version")
    warnings = row["timing_warnings"]
    if not isinstance(warnings, list) or any(not isinstance(warning, str) for warning in warnings):
        raise ValueError("reference timing manifest row has invalid timing warnings")
    _validate_timing_status_shape(row)


def _validate_timing_status_shape(row: Mapping[str, object]) -> None:
    """Validate the ready/quarantined shape shared by builder and validator."""
    status = row["timing_status"]
    if not isinstance(status, str) or status not in _TIMING_STATUSES:
        raise ValueError("reference timing row has an invalid timing status")
    _validate_timing_reason_codes(row["timing_reason_codes"])
    source_audio_fields = (
        "source_audio_key",
        "source_audio_content_hash",
        "reference_events_cache_path",
    )
    if status == "ready":
        if row["timing_reason_codes"]:
            raise ValueError("ready reference timing row must not carry reason codes")
        for field in source_audio_fields:
            value = row[field]
            if not isinstance(value, str) or not value:
                raise ValueError("ready reference timing row is missing source audio identity")
        _require_sha256_value(row["source_audio_content_hash"], "source_audio_content_hash")
        return
    if not row["timing_reason_codes"]:
        raise ValueError("quarantined reference timing row must carry reason codes")
    for field in source_audio_fields:
        if row[field] is not None:
            raise ValueError("quarantined reference timing row must null source audio identity")


def _validate_timing_reason_codes(value: object) -> None:
    if not isinstance(value, list) or any(
        not isinstance(reason, str) or reason not in _TIMING_REASON_CODES for reason in value
    ):
        raise ValueError("reference timing row has invalid reason codes")


def _require_sha256_value(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be lowercase SHA-256")
    try:
        return require_sha256(value, field)
    except StrictJsonError:
        raise ValueError(f"{field} must be lowercase SHA-256") from None


def _is_corpus_version(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    try:
        _require_sha256_value(value.removeprefix("sha256:"), "corpus_version")
    except ValueError:
        return False
    return True
