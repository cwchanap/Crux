"""Committed, reproducible corpus diagnostic for HPA-323 reference timing.

This tool measures the BGM-layout, audio-format, and channel-``02`` timing
behavior of a validated HPA-322 reference-chart manifest so the timing policy
can be frozen before native reference events are published (HPA-323 Task 4).

It is intentionally **committed and re-runnable** rather than a throwaway
script: every input it needs (manifest, local cache, R2 credentials) is
supplied by the operator, and it only ever calls **production seams** -- the
typed reference-row view, the verified-cache readers, the typed DTX parser,
the policy-neutral BGM group resolver, the exact-key cache fill, the shared
object-key resolver, and the single sticky timing engine.  No private HPA-322
validator or private path/casefold helper is reachable from here.

.. note:: DEFERRED.  Brief Step 7 (run the real corpus diagnostic) could not
    execute in the worktree where this code was authored: there is no
    ``artifacts/`` corpus, no HPA-322 output manifest, and no R2 credentials.
    The committed tool is reproducible, so the operator runs it later against
    real data.  Evidence-based policy finalization (Step 8 review) is pending
    that run; the conservative multi-group quarantine (``select_bgm_reference``
    in :mod:`src.benchmark.reference_timing`) is the frozen default until then.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import click

from src.benchmark.backend_identity import canonical_json_bytes, quantize_six
from src.benchmark.corpus_cache import (
    CacheIndexStore,
    read_verified_cache_body,
    resolve_verified_cache_body,
    sync_explicit_cache_keys,
)
from src.benchmark.dtx_parser import ParsedDtxChart, parse_dtx_bytes
from src.benchmark.inventory_object_keys import resolve_inventory_object_key
from src.benchmark.r2_corpus_models import (
    CacheSyncResult,
    R2Config,
    RemoteObject,
    SimfileInventory,
)
from src.benchmark.r2_inventory import R2ObjectStore, create_boto3_store
from src.benchmark.reference_chart_manifest import (
    ReferenceChartRowView,
    reference_chart_row_view_from_row,
)
from src.benchmark.reference_timing import BgmReferenceSet, resolve_bgm_reference_groups
from src.benchmark.timing import DtxTimingMap, build_dtx_timing_map

#: Channel-``02`` measures below this absolute delta (seconds) are treated as
#: unchanged between the corrected and diagnostic-legacy timing maps.
_CHANNEL_02_DELTA_EPSILON = 1e-9
#: Maximum number of multi-group and channel-``02`` examples retained.
_MAX_EXAMPLES = 25
#: BGM resolution reason codes that indicate at least one BGM event's source
#: audio could not be resolved to a remote object.
_BGM_RESOLUTION_FAILURE_CODES = frozenset(
    {
        "unresolved_bgm_wav",
        "unsafe_bgm_audio_path",
        "source_audio_missing",
        "source_audio_key_ambiguous",
    }
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Inputs supplied by the operator."""

    manifest_path: Path
    cache_dir: Path
    output_path: Path
    audio_sample_limit: int = 50
    #: Forwarded to :func:`resolve_bgm_reference_groups`; stays policy-neutral.
    allow_root_fallback: bool = True


@dataclass(frozen=True)
class AudioProbeOutcome:
    """Result of probing one resolved audio body with ``soundfile.info``."""

    decodable: bool
    error: str | None


def _default_probe_audio(path: Path) -> AudioProbeOutcome:
    """Probe a content-addressed, extensionless body via ``soundfile.info``.

    Matches :mod:`src.benchmark.render_audio`'s exception envelope so decoder
    failures are reported without aborting the diagnostic.
    """
    import soundfile as sf  # heavy optional dependency: imported lazily

    try:
        sf.info(path)
    except (OSError, RuntimeError, ValueError, sf.LibsndfileError) as error:
        return AudioProbeOutcome(decodable=False, error=type(error).__name__)
    return AudioProbeOutcome(decodable=True, error=None)


@dataclass(frozen=True)
class AnalysisDeps:
    """Injectable production seams plus the R2 connection used for cache fills.

    Every callable field defaults to the real production function, so the
    committed tool only ever calls production seams.  Tests override the I/O
    fields with offline fakes and assert each seam is invoked.
    """

    r2_config: R2Config
    store: R2ObjectStore
    index: CacheIndexStore
    build_row_view: Callable[..., ReferenceChartRowView] = reference_chart_row_view_from_row
    read_chart_body: Callable[..., bytes] = read_verified_cache_body
    parse_chart: Callable[..., ParsedDtxChart] = parse_dtx_bytes
    resolve_bgm_groups: Callable[..., BgmReferenceSet] = resolve_bgm_reference_groups
    build_timing_map: Callable[..., DtxTimingMap] = build_dtx_timing_map
    resolve_audio_body: Callable[..., Path] = resolve_verified_cache_body
    probe_audio: Callable[[Path], AudioProbeOutcome] = _default_probe_audio
    sync_explicit: Callable[..., CacheSyncResult] = sync_explicit_cache_keys


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def _load_row_views(
    manifest_path: Path,
    build_row_view: Callable[..., ReferenceChartRowView],
) -> tuple[ReferenceChartRowView, ...]:
    """Read canonical JSONL and build a typed view per row.

    Rows are returned in manifest order (already canonical / deterministic).
    Malformed records fail loudly through the row validator.
    """
    content = manifest_path.read_bytes()
    if not content.endswith(b"\n"):
        raise ValueError("reference-chart manifest must be canonical JSONL")
    views: list[ReferenceChartRowView] = []
    for line in content.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError("reference-chart manifest must be canonical JSONL")
        row = _json_object(line[:-1])
        views.append(build_row_view(row))
    if not views:
        raise ValueError("reference-chart manifest contains no records")
    return tuple(views)


def _json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("reference-chart manifest must be canonical JSONL") from None
    if not isinstance(value, dict):
        raise ValueError("reference-chart manifest must be canonical JSONL")
    return value


# ---------------------------------------------------------------------------
# Path-resolution classification (shared HPA-322 object-key resolver only)
# ---------------------------------------------------------------------------


def _classify_bgm_resolution(
    chart: ParsedDtxChart,
    row: ReferenceChartRowView,
    selected_chart_key: str,
    allow_root_fallback: bool,
) -> tuple[bool, bool]:
    """Classify whether any BGM event needed a case-insensitive or root retry.

    Mirrors :func:`resolve_bgm_reference_groups`'s outcome mapping using the
    same shared :func:`resolve_inventory_object_key`.  Read-only: never affects
    grouping.  Returns ``(casefold_needed, root_fallback_needed)``.
    """
    inventory = row.source.inventory
    objects = inventory.objects
    object_prefix = inventory.object_prefix
    chart_dir = selected_chart_key.rsplit("/", 1)[0]
    casefold_needed = False
    root_fallback_needed = False
    for event in chart.bgm_events:
        wav_path = chart.wav_table.get(event.note_id)
        if wav_path is None:
            continue
        result = resolve_inventory_object_key(
            wav_path,
            base_object_key_dir=chart_dir,
            object_prefix=object_prefix,
            objects=objects,
        )
        if result.status == "casefold":
            casefold_needed = True
        if result.status in ("exact", "casefold"):
            continue
        if result.status == "missing" and allow_root_fallback:
            root_result = resolve_inventory_object_key(
                wav_path,
                base_object_key_dir=object_prefix,
                object_prefix=object_prefix,
                objects=objects,
            )
            if root_result.status in ("exact", "casefold"):
                root_fallback_needed = True
                if root_result.status == "casefold":
                    casefold_needed = True
    return casefold_needed, root_fallback_needed


def _authored_extension(wav_path: str) -> str:
    """Return the lowercased suffix of an authored BGM ``#WAVxx`` string.

    Extensions are used for *counting only*; they never decide decodability.
    """
    return Path(wav_path).suffix.lower()


def _referenced_bgm_extensions(chart: ParsedDtxChart) -> tuple[str, ...]:
    """Lowercased extensions of the distinct BGM-referenced ``#WAVxx`` values."""
    referenced: set[str] = set()
    for event in chart.bgm_events:
        wav_path = chart.wav_table.get(event.note_id)
        if wav_path:
            referenced.add(wav_path)
    return tuple(sorted(_authored_extension(path) for path in referenced))


def _candidate_extension(chart: ParsedDtxChart, remote_key: str) -> str:
    """Best-effort authored extension for a resolved candidate audio object.

    Resolved remotes are content-addressed and extensionless, so the authored
    extension is recovered from the BGM-referenced ``#WAVxx`` string(s) whose
    basename matches the object key tail.  The lexicographically smallest
    suffix wins so the value is deterministic.  Empty when unrecoverable.
    """
    tail = remote_key.rsplit("/", 1)[-1].lower()
    suffixes: list[str] = []
    for event in chart.bgm_events:
        wav_path = chart.wav_table.get(event.note_id)
        if wav_path and Path(wav_path).name.lower() == tail:
            suffixes.append(_authored_extension(wav_path))
    suffixes.sort()
    return suffixes[0] if suffixes else ""


# ---------------------------------------------------------------------------
# Channel-02 blast-radius diagnostic (single sticky timing engine)
# ---------------------------------------------------------------------------


def _chart_max_measure(chart: ParsedDtxChart) -> int:
    measures = [event.measure for event in chart.events]
    measures.extend(event.measure for event in chart.bgm_events)
    measures.extend(event.measure for event in chart.bpm_events)
    measures.extend(chart.measure_lengths)
    return max(measures, default=0)


def _diagnostic_legacy_chart(chart: ParsedDtxChart) -> ParsedDtxChart:
    """Copy where every measure up to the chart max has an explicit length.

    Implicit measures get ``1.0``; running the *same* corrected
    :func:`build_dtx_timing_map` on this copy reproduces the legacy
    per-measure-reset behavior through the single sticky engine, with no
    second timing implementation.
    """
    max_measure = _chart_max_measure(chart)
    diagnostic_lengths = {
        measure: chart.measure_lengths.get(measure, 1.0) for measure in range(max_measure + 1)
    }
    return replace(chart, measure_lengths=diagnostic_lengths)


# ---------------------------------------------------------------------------
# Report rendering helpers
# ---------------------------------------------------------------------------


def _q6(value: float) -> Any:
    """Quantize a float to the repo canonical 6-place Decimal for output."""
    return quantize_six(value)


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts)}


def _multi_group_example(
    row: ReferenceChartRowView,
    chart_key: str,
    references: BgmReferenceSet,
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for group in references.groups:
        groups.append(
            {
                "measure": group.measure,
                "note_ids": sorted(event.note_id for event in group.events),
                "object_key": group.remote.key,
                "position": _q6(group.position),
            }
        )
    groups.sort(key=lambda item: (item["measure"], item["object_key"], item["position"]))
    return {
        "chart_key": chart_key,
        "groups": groups,
        "simfile_id": row.simfile_id,
    }


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def run_reference_timing_analysis(
    config: AnalysisConfig,
    deps: AnalysisDeps,
) -> dict[str, Any]:
    """Run the full corpus diagnostic and return the canonical report dict.

    The returned dict is JSON-serializable via :func:`canonical_json_bytes` and
    byte-stable across re-runs against the same fixture (all collections are
    deterministically ordered; floats are quantized to six places).
    """
    row_views = _load_row_views(config.manifest_path, deps.build_row_view)
    selected_rows = tuple(view for view in row_views if view.selection_status == "selected")
    quarantined_rows = tuple(view for view in row_views if view.selection_status == "quarantined")
    # Process selected rows in simfile-id order for deterministic output.
    selected_rows = tuple(sorted(selected_rows, key=lambda view: view.simfile_id))

    rows_with_0_bgm_groups = 0
    rows_with_1_bgm_group = 0
    rows_with_multiple_bgm_groups = 0
    rows_with_unresolved_wav = 0
    rows_needing_case_insensitive_match = 0
    rows_needing_simfile_root_fallback = 0
    multi_group_examples: list[dict[str, Any]] = []
    bgm_extension_counts: dict[str, int] = {}
    charts_with_channel_02 = 0
    charts_with_multiple_channel_02_changes = 0
    max_channel_02_time_delta_sec = 0.0
    channel_02_delta_examples: list[dict[str, Any]] = []

    # Resolved candidate audio objects keyed by remote.key for stable sampling.
    candidate_audio: dict[str, tuple[RemoteObject, str, int, str]] = {}
    # Simfile inventories captured per simfile-id for exact-key cache fills.
    inventories_by_simfile: dict[int, SimfileInventory] = {}

    endpoint_sha = deps.r2_config.source_endpoint_sha256
    bucket = deps.r2_config.bucket

    for row in selected_rows:
        assert row.selected_chart is not None
        chart_key = row.selected_chart.key
        chart_raw = deps.read_chart_body(
            config.cache_dir,
            row.selected_chart,
            source_endpoint_sha256=endpoint_sha,
            bucket=bucket,
            expected_sha256=row.selected_chart_content_hash,
        )
        chart = deps.parse_chart(chart_raw, chart_id=chart_key, source_name=chart_key)
        references = deps.resolve_bgm_groups(
            chart,
            selected_chart_key=chart_key,
            row=row,
            allow_root_fallback=config.allow_root_fallback,
        )

        inventories_by_simfile.setdefault(row.simfile_id, row.source.inventory)

        group_count = len(references.groups)
        if group_count == 0:
            rows_with_0_bgm_groups += 1
        elif group_count == 1:
            rows_with_1_bgm_group += 1
        else:
            rows_with_multiple_bgm_groups += 1

        if set(references.reason_codes) & _BGM_RESOLUTION_FAILURE_CODES:
            rows_with_unresolved_wav += 1

        casefold_needed, root_fallback_needed = _classify_bgm_resolution(
            chart, row, chart_key, config.allow_root_fallback
        )
        if casefold_needed:
            rows_needing_case_insensitive_match += 1
        if root_fallback_needed:
            rows_needing_simfile_root_fallback += 1

        if group_count > 1 and len(multi_group_examples) < _MAX_EXAMPLES:
            multi_group_examples.append(_multi_group_example(row, chart_key, references))

        for extension in _referenced_bgm_extensions(chart):
            bgm_extension_counts[extension] = bgm_extension_counts.get(extension, 0) + 1

        for group in references.groups:
            candidate_audio.setdefault(
                group.remote.key,
                (
                    group.remote,
                    _candidate_extension(chart, group.remote.key),
                    row.simfile_id,
                    chart_key,
                ),
            )

        # Channel-02 blast radius: corrected vs diagnostic-legacy timing.
        if chart.measure_lengths:
            charts_with_channel_02 += 1
            if len(chart.measure_lengths) > 1:
                charts_with_multiple_channel_02_changes += 1
        corrected_map = deps.build_timing_map(chart)
        legacy_map = deps.build_timing_map(_diagnostic_legacy_chart(chart))
        for event in chart.events:
            corrected_sec = corrected_map.time_sec(event)
            legacy_sec = legacy_map.time_sec(event)
            delta_sec = abs(corrected_sec - legacy_sec)
            if delta_sec > max_channel_02_time_delta_sec:
                max_channel_02_time_delta_sec = delta_sec
            if delta_sec > _CHANNEL_02_DELTA_EPSILON:
                channel_02_delta_examples.append(
                    {
                        "chart_key": chart_key,
                        "corrected_sec": _q6(corrected_sec),
                        "delta_sec": _q6(delta_sec),
                        "legacy_sec": _q6(legacy_sec),
                        "simfile_id": row.simfile_id,
                        "source_order": event.source_order,
                    }
                )

    channel_02_delta_examples.sort(
        key=lambda item: (
            -float(item["delta_sec"]),
            item["simfile_id"],
            item["chart_key"],
            item["source_order"],
        )
    )
    channel_02_delta_examples = channel_02_delta_examples[:_MAX_EXAMPLES]

    audio_sample = _sample_audio(candidate_audio, config.audio_sample_limit)
    sampled = _probe_sampled_audio(
        config, deps, audio_sample, endpoint_sha, bucket, inventories_by_simfile
    )
    (
        sampled_audio_count,
        sampled_audio_decodable_count,
        sampled_audio_undecodable_count,
        sampled_audio_undecodable_by_extension,
    ) = sampled

    return {
        "bgm_extension_counts": _sorted_counts(bgm_extension_counts),
        "channel_02_delta_examples": channel_02_delta_examples,
        "charts_with_channel_02": charts_with_channel_02,
        "charts_with_multiple_channel_02_changes": charts_with_multiple_channel_02_changes,
        "max_channel_02_time_delta_sec": _q6(max_channel_02_time_delta_sec),
        "multi_group_examples": multi_group_examples,
        "rows_needing_case_insensitive_match": rows_needing_case_insensitive_match,
        "rows_needing_simfile_root_fallback": rows_needing_simfile_root_fallback,
        "rows_with_0_bgm_groups": rows_with_0_bgm_groups,
        "rows_with_1_bgm_group": rows_with_1_bgm_group,
        "rows_with_multiple_bgm_groups": rows_with_multiple_bgm_groups,
        "rows_with_unresolved_wav": rows_with_unresolved_wav,
        "sampled_audio_count": sampled_audio_count,
        "sampled_audio_decodable_count": sampled_audio_decodable_count,
        "sampled_audio_undecodable_by_extension": _sorted_counts(
            sampled_audio_undecodable_by_extension
        ),
        "sampled_audio_undecodable_count": sampled_audio_undecodable_count,
        "selected_rows": len(selected_rows),
        "upstream_quarantined_rows": len(quarantined_rows),
    }


def _sample_audio(
    candidate_audio: dict[str, tuple[RemoteObject, str, int, str]],
    audio_sample_limit: int,
) -> list[tuple[str, tuple[RemoteObject, str, int, str]]]:
    """Deterministically cap the candidate set to ``audio_sample_limit`` keys."""
    if audio_sample_limit <= 0:
        return []
    return sorted(candidate_audio.items())[:audio_sample_limit]


def _probe_sampled_audio(
    config: AnalysisConfig,
    deps: AnalysisDeps,
    audio_sample: list[tuple[str, tuple[RemoteObject, str, int, str]]],
    endpoint_sha: str,
    bucket: str,
    inventories_by_simfile: dict[int, SimfileInventory],
) -> tuple[int, int, int, dict[str, int]]:
    """Fill missing verified bodies, then probe decodability.

    Ordering is strict: every sampled body is resolved (filling missing exact
    keys through :func:`sync_explicit_cache_keys` first) *before*
    :func:`probe_audio` runs, so ``resolve_verified_cache_body`` always
    precedes ``soundfile.info``.
    """
    if not audio_sample:
        return 0, 0, 0, {}

    resolved_paths: dict[str, Path | None] = {}
    for key, (remote, _ext, _simfile_id, _chart_key) in audio_sample:
        try:
            resolved_paths[key] = deps.resolve_audio_body(
                config.cache_dir,
                remote,
                source_endpoint_sha256=endpoint_sha,
                bucket=bucket,
            )
        except ValueError:
            resolved_paths[key] = None

    missing_keys = frozenset(key for key, path in resolved_paths.items() if path is None)
    if missing_keys:
        simfile_ids = {simfile_id for _key, (_remote, _ext, simfile_id, _chart_key) in audio_sample}
        inventories = tuple(
            inventories_by_simfile[simfile_id] for simfile_id in sorted(simfile_ids)
        )
        deps.sync_explicit(
            inventories,
            deps.store,
            deps.index,
            deps.r2_config,
            missing_keys,
        )
        for key, (remote, _ext, _simfile_id, _chart_key) in audio_sample:
            if key not in missing_keys:
                continue
            try:
                resolved_paths[key] = deps.resolve_audio_body(
                    config.cache_dir,
                    remote,
                    source_endpoint_sha256=endpoint_sha,
                    bucket=bucket,
                )
            except ValueError:
                resolved_paths[key] = None

    sampled_audio_count = 0
    sampled_audio_decodable_count = 0
    sampled_audio_undecodable_count = 0
    sampled_audio_undecodable_by_extension: dict[str, int] = {}
    for key, (_remote, extension, _simfile_id, _chart_key) in audio_sample:
        sampled_audio_count += 1
        path = resolved_paths[key]
        if path is None:
            outcome = AudioProbeOutcome(decodable=False, error="verified body unavailable")
        else:
            outcome = deps.probe_audio(path)
        if outcome.decodable:
            sampled_audio_decodable_count += 1
        else:
            sampled_audio_undecodable_count += 1
            label = extension or "<none>"
            sampled_audio_undecodable_by_extension[label] = (
                sampled_audio_undecodable_by_extension.get(label, 0) + 1
            )
    return (
        sampled_audio_count,
        sampled_audio_decodable_count,
        sampled_audio_undecodable_count,
        sampled_audio_undecodable_by_extension,
    )


def render_report(report: dict[str, Any]) -> bytes:
    """Canonical sorted JSON for the report, with one final newline."""
    return canonical_json_bytes(report, trailing_newline=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Validated HPA-322 reference-chart manifest (canonical JSONL).",
)
@click.option(
    "--cache-dir",
    "cache_dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Local r2-corpus cache root used for verified body reads/fills.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Destination path for the canonical JSON diagnostic report.",
)
@click.option(
    "--audio-sample-limit",
    type=click.IntRange(min=0),
    default=50,
    show_default=True,
    help="Maximum unique resolved audio objects to probe for decodability.",
)
@click.option(
    "--disallow-root-fallback",
    is_flag=True,
    default=False,
    help="Disable the simfile-root retry when resolving BGM audio paths.",
)
def main(
    manifest_path: Path,
    cache_dir: Path,
    output_path: Path,
    audio_sample_limit: int,
    disallow_root_fallback: bool,
) -> None:
    """Measure reference-timing corpus behavior and freeze the diagnostic."""
    r2_config = R2Config.from_environ(os.environ)
    store = create_boto3_store(r2_config)
    index = CacheIndexStore.load(Path(cache_dir))
    deps = AnalysisDeps(r2_config=r2_config, store=store, index=index)
    analysis_config = AnalysisConfig(
        manifest_path=Path(manifest_path),
        cache_dir=Path(cache_dir),
        output_path=Path(output_path),
        audio_sample_limit=audio_sample_limit,
        allow_root_fallback=not disallow_root_fallback,
    )
    report = run_reference_timing_analysis(analysis_config, deps)
    output_bytes = render_report(report)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(output_bytes)
    click.echo(f"reference_timing_report_sha256={sha256(output_bytes).hexdigest()}")


if __name__ == "__main__":
    main()
