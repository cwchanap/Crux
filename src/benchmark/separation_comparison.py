"""Published HPA-328 comparisons for the fixed OaF separation views."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    StrictJsonError,
    canonical_json_bytes,
    strict_json_loads,
)
from src.benchmark.cohort_scoring import SCORING_VERSION, CohortIdentity
from src.benchmark.oaf_corpus_run import OAF_FULL_MIX_INPUT_VIEW_ID
from src.benchmark.published_comparison import (
    ComparisonIntegrityError,
    PublishedRunEvidence,
    PublishedRunItem,
    comparison_summary,
    csv_decimal,
    metric_delta,
    pairable_success_ids,
    paired_class_rows,
    paired_song_rows,
    write_comparison_artifacts,
    write_markdown,
)
from src.benchmark.reference_set_manifest import load_reference_set_manifest
from src.benchmark.reference_timing_manifest import load_reference_timing_manifest
from src.benchmark.reports import PublishedCohortReports, read_cohort_reports
from src.benchmark.reviewed_subset import load_reviewed_subset_manifest

SEPARATION_COMPARISON_SCHEMA = "crux.oaf-separation-comparison/v1"
SEPARATION_COMPARISON_TITLE = "HPA-328 OaF Separation Published Comparison"
SEPARATION_RUN_SCHEMA = "crux.oaf-separation-run/v1"
SPLEETER_INPUT_VIEW_ID = "crux.oaf-spleeter4-drums-mono44k1-pcm16/v1"
HTDEMUCS_INPUT_VIEW_ID = "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"
_SIX_PLACES = Decimal("0.000001")
_MODES = {"raw": 0, "aligned": 1}
_VIEW_IDS = {
    "full_mix": OAF_FULL_MIX_INPUT_VIEW_ID,
    "spleeter": SPLEETER_INPUT_VIEW_ID,
    "htdemucs": HTDEMUCS_INPUT_VIEW_ID,
}
_REPORT_NAMES = (
    "summary.json",
    "items.csv",
    "per_song.csv",
    "per_class.csv",
)


@dataclass(frozen=True)
class SeparationComparisonRequest:
    """Inputs for one persisted HPA-328 comparison."""

    run_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    subset_manifest_path: Path
    output_dir: Path
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        for field in (
            "run_path",
            "reference_manifest_path",
            "timing_manifest_path",
            "subset_manifest_path",
            "output_dir",
        ):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a Path")
        if self.cache_dir is not None and not isinstance(self.cache_dir, Path):
            raise TypeError("cache_dir must be a Path or None")


@dataclass(frozen=True)
class SeparationComparisonOutcome:
    """Published comparison paths and pair counts."""

    output_dir: Path
    pairable_success_counts: dict[str, int]
    paired_song_counts: dict[str, int]
    paired_class_counts: dict[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a Path")
        for field in (
            "pairable_success_counts",
            "paired_song_counts",
            "paired_class_counts",
        ):
            value = getattr(self, field)
            if not isinstance(value, dict):
                raise TypeError(f"{field} must be a dict")
            if any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in value.values()
            ):
                raise ValueError(f"{field} contains an invalid count")


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(_SIX_PLACES, rounding=ROUND_HALF_EVEN)


def _metric(
    tp: int,
    fp: int,
    fn: int,
    duration_seconds: Decimal | None,
) -> dict[str, object]:
    precision = _quantize(Decimal(tp) / Decimal(tp + fp)) if tp + fp else None
    recall = _quantize(Decimal(tp) / Decimal(tp + fn)) if tp + fn else None
    f1 = _quantize(Decimal(2 * tp) / Decimal(2 * tp + fp + fn)) if 2 * tp + fp + fn else None
    minutes = duration_seconds / Decimal("60") if duration_seconds is not None else None
    fp_per_minute = (
        _quantize(Decimal(fp) / minutes) if minutes is not None and minutes > 0 else None
    )
    fn_per_minute = (
        _quantize(Decimal(fn) / minutes) if minutes is not None and minutes > 0 else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_per_minute": fp_per_minute,
        "fn_per_minute": fn_per_minute,
    }


def _positive_finite_duration(value: object) -> Decimal | None:
    try:
        duration = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not duration.is_finite() or duration <= 0:
        return None
    return duration


def _song_rows(reports: PublishedCohortReports) -> dict[tuple[str, int, str], object]:
    return {(row.simfile_id, row.tolerance_ms, row.mode): row for row in reports.songs}


def _class_rows(reports: PublishedCohortReports) -> dict[tuple[str, int, str, str], object]:
    return {
        (row.simfile_id, row.tolerance_ms, row.mode, row.common_class): row
        for row in reports.classes
    }


def aggregate_paired_event_micro(
    left_reports: object,
    right_reports: object,
    pairable_ids: set[str],
    source_durations: Mapping[str, object],
    *,
    left_label: str = "full_mix",
    right_label: str = "view",
) -> list[dict[str, object]]:
    """Sum published per-song counts before deriving paired event metrics."""
    if not isinstance(left_label, str) or not left_label:
        raise ValueError("left_label must be a nonempty string")
    if not isinstance(right_label, str) or not right_label:
        raise ValueError("right_label must be a nonempty string")
    left = _song_rows(left_reports)  # type: ignore[arg-type]
    right = _song_rows(right_reports)  # type: ignore[arg-type]
    left_keys = {key for key in left if key[0] in pairable_ids}
    right_keys = {key for key in right if key[0] in pairable_ids}
    if left_keys != right_keys:
        raise ComparisonIntegrityError("paired event-micro score key grid mismatch")
    grouped: dict[tuple[int, str], list[tuple[object, object]]] = {}
    for key in sorted(left_keys, key=lambda value: (int(value[0]), value[1], _MODES[value[2]])):
        grouped.setdefault((key[1], key[2]), []).append((left[key], right[key]))
    duration_seconds = sum(
        (
            duration
            for key, value in source_durations.items()
            if key in pairable_ids
            for duration in (_positive_finite_duration(value),)
            if duration is not None
        ),
        Decimal(0),
    )
    duration = duration_seconds if duration_seconds > 0 else None
    rows: list[dict[str, object]] = []
    for tolerance_mode in sorted(grouped, key=lambda value: (value[0], _MODES[value[1]])):
        grouped_rows = grouped[tolerance_mode]
        metrics: dict[str, dict[str, object]] = {}
        for label, index in ((left_label, 0), (right_label, 1)):
            tp = sum(getattr(pair[index], "true_positives") for pair in grouped_rows)
            fp = sum(getattr(pair[index], "false_positives") for pair in grouped_rows)
            fn = sum(getattr(pair[index], "false_negatives") for pair in grouped_rows)
            metrics[label] = _metric(tp, fp, fn, duration)
        left_metric = metrics[left_label]
        right_metric = metrics[right_label]
        rows.append(
            {
                "tolerance_ms": tolerance_mode[0],
                "mode": tolerance_mode[1],
                "pairable_song_count": len(grouped_rows),
                "duration_seconds": duration,
                left_label: left_metric,
                right_label: right_metric,
                "delta": {
                    metric: metric_delta(
                        left_metric[metric],
                        right_metric[metric],  # type: ignore[arg-type]
                    )
                    for metric in ("precision", "recall", "f1")
                },
            }
        )
    return rows


def _strict_report_identity(report_dir: Path) -> CohortIdentity:
    try:
        summary = strict_json_loads(
            read_regular_file_no_follow(report_dir / "summary.json"),
            require_canonical=True,
        )
    except (OSError, StrictJsonError) as error:
        raise ComparisonIntegrityError(f"cannot read HPA-325 identity: {error}") from error
    if not isinstance(summary, Mapping) or not isinstance(summary.get("identity"), Mapping):
        raise ComparisonIntegrityError("HPA-325 summary identity is malformed")
    try:
        return CohortIdentity(**summary["identity"])  # type: ignore[arg-type]
    except (TypeError, ValueError, StrictJsonError) as error:
        raise ComparisonIntegrityError(f"HPA-325 summary identity is malformed: {error}") from error


def _expected_cohort_id(
    snapshot: Mapping[str, object],
    subset_manifest: object,
    view_name: str,
) -> str:
    if view_name == "full_mix":
        parent_run_id = snapshot.get("parent_oaf_run_id")
        if not isinstance(parent_run_id, str) or not parent_run_id:
            raise ComparisonIntegrityError("parent OaF run identity is unavailable")
        payload = {
            "parent_run_id": parent_run_id,
            "reviewed_subset_manifest_sha256": getattr(subset_manifest, "manifest_sha256"),
        }
    else:
        run_id = snapshot.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ComparisonIntegrityError("separation run identity is unavailable")
        payload = {"parent_oaf_run_id": run_id, "input_view_id": _VIEW_IDS[view_name]}
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _validated_identity(
    report_dir: Path,
    snapshot: Mapping[str, object],
    reference_manifest: object,
    timing_manifest: object,
    subset_manifest: object,
    view_name: str,
) -> CohortIdentity:
    identity = _strict_report_identity(report_dir)
    expected_values = {
        "cohort_id": _expected_cohort_id(snapshot, subset_manifest, view_name),
        "reference_manifest_sha256": getattr(reference_manifest, "manifest_sha256"),
        "reference_timing_version": getattr(timing_manifest, "corpus_version"),
        "taxonomy_version": identity.taxonomy_version,
        "lane_map_version": identity.lane_map_version,
        "backend_id": OAF_BACKEND_ID,
        "model_id": identity.model_id,
        "model_lock_sha256": snapshot.get("oaf_model_lock_sha256"),
        "backend_descriptor_sha256": snapshot.get("oaf_backend_descriptor_sha256"),
        "prediction_map_version": identity.prediction_map_version,
        "input_view_id": _VIEW_IDS[view_name],
        "scoring_version": SCORING_VERSION,
    }
    if identity.backend_id != OAF_BACKEND_ID:
        raise ComparisonIntegrityError("HPA-328 comparison requires OaF reports")
    for field, expected in expected_values.items():
        if getattr(identity, field) != expected:
            raise ComparisonIntegrityError(f"HPA-328 report identity mismatch for {field}")
    return identity


def _status(value: object) -> str:
    if value in {"inferred", "resumed", "success"}:
        return "success"
    if value in {"failed", "skipped", "quarantined"}:
        return str(value)
    return "failed"


def _view_evidence(
    snapshot: Mapping[str, object],
    reports: PublishedCohortReports,
    view_name: str,
) -> PublishedRunEvidence:
    raw_items = snapshot.get("items")
    if not isinstance(raw_items, list):
        raise ComparisonIntegrityError("separation run items are unavailable")
    items: dict[str, PublishedRunItem] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            raise ComparisonIntegrityError("separation run item is malformed")
        raw_id = raw_item.get("simfile_id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id <= 0:
            raise ComparisonIntegrityError("separation run item simfile_id is malformed")
        simfile_id = str(raw_id)
        raw_view = raw_item.get(view_name)
        if not isinstance(raw_view, Mapping):
            raise ComparisonIntegrityError(f"{view_name} run evidence is malformed")
        input_evidence = raw_view.get("input")
        input_hash = (
            input_evidence.get("input_audio_sha256")
            if isinstance(input_evidence, Mapping)
            else None
        )
        source_hash = raw_item.get("source_audio_sha256")
        if source_hash is not None and not isinstance(source_hash, str):
            raise ComparisonIntegrityError("source_audio_sha256 is malformed")
        if input_hash is not None and not isinstance(input_hash, str):
            raise ComparisonIntegrityError("input_audio_sha256 is malformed")
        items[simfile_id] = PublishedRunItem(
            simfile_id,
            _status(raw_view.get("status")),
            source_hash,
            input_hash,
        )
    report_statuses = {row.simfile_id: row.status for row in reports.items}
    if set(report_statuses) != set(items):
        raise ComparisonIntegrityError(f"{view_name} report population does not match run")
    if any(report_statuses[item_id] != item.status for item_id, item in items.items()):
        raise ComparisonIntegrityError(f"{view_name} report status does not match run")
    return PublishedRunEvidence(
        identity=reports.identity,
        items=items,
        reports=reports,
        snapshot=snapshot,
        label=view_name,
    )


def _durations(snapshot: Mapping[str, object]) -> dict[str, Decimal]:
    raw_items = snapshot.get("items")
    if not isinstance(raw_items, list):
        raise ComparisonIntegrityError("separation run items are unavailable")
    durations: dict[str, Decimal] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            continue
        raw_id = raw_item.get("simfile_id")
        value = raw_item.get("source_duration_sec")
        duration = _positive_finite_duration(value)
        if isinstance(raw_id, int) and raw_id > 0 and duration is not None:
            durations[str(raw_id)] = duration
    return durations


def _finite_seconds(value: object) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(0)
    if not parsed.is_finite() or parsed < 0:
        return Decimal(0)
    return parsed


def _resolve_artifact(value: object, *, roots: tuple[Path, ...]) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    if raw.is_absolute():
        return raw
    for root in roots:
        candidate = root / raw
        if candidate.exists():
            return candidate
    return None


def _artifact_bytes(path: Path | None) -> int:
    if path is None:
        return 0
    try:
        return len(read_regular_file_no_follow(path))
    except (OSError, TypeError) as error:
        raise ComparisonIntegrityError(f"retained artifact is unreadable: {path}") from error


def _report_bytes(report_dir: Path) -> int:
    total = 0
    for name in _REPORT_NAMES + ("event_diagnostics.jsonl", "summary.md"):
        total += _artifact_bytes(report_dir / name)
    return total


def _resources(
    snapshot: Mapping[str, object],
    run_dir: Path,
    cache_dir: Path | None,
    view_name: str,
    report_dir: Path,
) -> dict[str, object]:
    raw_items = snapshot.get("items")
    if not isinstance(raw_items, list):
        raise ComparisonIntegrityError("separation run items are unavailable")
    separator_seconds = Decimal(0)
    oaf_seconds = Decimal(0)
    stem_bytes = 0
    prediction_bytes = 0
    roots = tuple(root for root in (cache_dir, run_dir, run_dir.parent.parent) if root is not None)
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            continue
        raw_view = raw_item.get(view_name)
        if not isinstance(raw_view, Mapping):
            continue
        runtime = raw_view.get("runtime")
        if isinstance(runtime, Mapping):
            separator_seconds += _finite_seconds(runtime.get("separator_wall_time_sec"))
            oaf_value = runtime.get("wall_time_sec")
            if oaf_value is None:
                oaf_value = runtime.get("inference_elapsed_seconds")
            oaf_seconds += _finite_seconds(oaf_value)
        stem = raw_view.get("stem")
        if isinstance(stem, Mapping):
            stem_bytes += _artifact_bytes(_resolve_artifact(stem.get("path"), roots=roots))
        prediction = raw_view.get("prediction")
        if isinstance(prediction, Mapping):
            prediction_bytes += _artifact_bytes(
                _resolve_artifact(prediction.get("path"), roots=roots)
            )
    return {
        "separator_wall_time_sec": _quantize(separator_seconds),
        "oaf_wall_time_sec": _quantize(oaf_seconds),
        "retained_stem_bytes": stem_bytes,
        "retained_prediction_bytes": prediction_bytes,
        "retained_report_bytes": _report_bytes(report_dir),
    }


def _failure_histogram(snapshot: Mapping[str, object], view_name: str) -> dict[str, int]:
    raw_items = snapshot.get("items")
    if not isinstance(raw_items, list):
        raise ComparisonIntegrityError("separation run items are unavailable")
    counts: Counter[str] = Counter()
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            continue
        view = raw_item.get(view_name)
        if not isinstance(view, Mapping):
            continue
        code = view.get("failure_code")
        if isinstance(code, str) and code:
            counts[code] += 1
    return dict(sorted(counts.items()))


def _pair_summary(
    full_mix: PublishedRunEvidence,
    view: PublishedRunEvidence,
    *,
    pairable_ids: set[str],
    exclusions: Mapping[str, int],
    source_durations: Mapping[str, Decimal],
    reference_manifest: object,
    timing_manifest: object,
    subset_path: Path,
    subset_manifest: object,
    identity: Mapping[str, object],
    view_name: str,
) -> dict[str, object]:
    full_songs = _song_rows(full_mix.reports)
    view_songs = _song_rows(view.reports)
    full_classes = _class_rows(full_mix.reports)
    view_classes = _class_rows(view.reports)
    song_rows = paired_song_rows(
        full_songs,
        view_songs,
        pairable_ids,
        left_label="full_mix",
        right_label=view_name,
    )
    class_rows = paired_class_rows(
        full_classes,
        view_classes,
        pairable_ids,
        left_label="full_mix",
        right_label=view_name,
    )
    base = comparison_summary(
        full_mix,
        view,
        pairable_ids,
        exclusions,
        song_rows,
        class_rows,
        reference_manifest,
        timing_manifest,
        subset_path,
        subset_manifest,
        schema=SEPARATION_COMPARISON_SCHEMA,
        identity=identity,
        left_label="full_mix",
        right_label=view_name,
    )
    event_micro = aggregate_paired_event_micro(
        full_mix.reports,
        view.reports,
        pairable_ids,
        source_durations,
        left_label="full_mix",
        right_label=view_name,
    )
    return {
        "base": base,
        "song_rows": song_rows,
        "class_rows": class_rows,
        "event_micro": event_micro,
        "pairable_ids": pairable_ids,
    }


def _comparison_identity(
    snapshot: Mapping[str, object],
    reference_manifest: object,
    timing_manifest: object,
    subset_manifest: object,
) -> dict[str, object]:
    return {
        "run_id": snapshot["run_id"],
        "parent_oaf_run_id": snapshot["parent_oaf_run_id"],
        "reference_manifest_sha256": getattr(reference_manifest, "manifest_sha256"),
        "reference_manifest_version": getattr(reference_manifest, "corpus_version"),
        "reference_timing_manifest_sha256": getattr(timing_manifest, "manifest_sha256"),
        "reference_timing_version": getattr(timing_manifest, "corpus_version"),
        "reviewed_subset_manifest_sha256": getattr(subset_manifest, "manifest_sha256"),
        "input_views": dict(_VIEW_IDS),
        "scoring_version": SCORING_VERSION,
    }


def _markdown_evidence(
    lines: list[str],
    view_name: str,
    pair: Mapping[str, object],
    resources: Mapping[str, object],
    failure_histogram: Mapping[str, int],
) -> None:
    lines.extend(
        [
            "",
            "### HPA-328 Evidence",
            "",
            f"- pairable_success_count: {len(pair['pairable_ids'])}",
            f"- failure_code_histogram: `{dict(failure_histogram)}`",
            f"- separator_wall_time_sec: {resources['separator_wall_time_sec']}",
            f"- oaf_wall_time_sec: {resources['oaf_wall_time_sec']}",
            f"- retained_stem_bytes: {resources['retained_stem_bytes']}",
            f"- retained_prediction_bytes: {resources['retained_prediction_bytes']}",
            f"- retained_report_bytes: {resources['retained_report_bytes']}",
            "",
            "#### Paired Event Micro",
            "",
            "| tolerance_ms | mode | songs | full_mix F1 | view F1 | full_mix FP/min | view FP/min | full_mix FN/min | view FN/min |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pair["event_micro"]:
        full_metric = row["full_mix"]
        view_metric = row[view_name]
        lines.append(
            f"| {row['tolerance_ms']} | {row['mode']} | {row['pairable_song_count']} | "
            f"{csv_decimal(full_metric['f1'])} | {csv_decimal(view_metric['f1'])} | "
            f"{csv_decimal(full_metric['fp_per_minute'])} | "
            f"{csv_decimal(view_metric['fp_per_minute'])} | "
            f"{csv_decimal(full_metric['fn_per_minute'])} | "
            f"{csv_decimal(view_metric['fn_per_minute'])} |"
        )


def _write_outputs(
    output_dir: Path,
    pair_results: Mapping[str, Mapping[str, object]],
    summary: Mapping[str, object],
    resources: Mapping[str, Mapping[str, object]],
    failure_histograms: Mapping[str, Mapping[str, int]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".separation-comparison-", dir=output_dir.parent) as stage_name:
        stage = Path(stage_name)
        for view_name in ("spleeter", "htdemucs"):
            pair = pair_results[view_name]
            pair_stage = stage / view_name
            write_comparison_artifacts(
                pair_stage,
                pair["song_rows"],  # type: ignore[arg-type]
                pair["class_rows"],  # type: ignore[arg-type]
                pair["base"],  # type: ignore[arg-type]
                title=f"Full Mix vs {view_name.title()} Published Comparison",
                left_label="full_mix",
                right_label=view_name,
            )
            destination = output_dir / view_name
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("paired_per_song.csv", "paired_per_class.csv"):
                os.replace(pair_stage / name, destination / name)

        markdown_path = stage / "summary.md"
        lines = [f"# {SEPARATION_COMPARISON_TITLE}", ""]
        identity = summary["identity"]
        lines.extend(["## Identity", ""])
        lines.extend(f"- {key}: `{value}`" for key, value in identity.items())
        for view_name in ("spleeter", "htdemucs"):
            pair = pair_results[view_name]
            pair_markdown = stage / f"{view_name}-pair.md"
            write_markdown(
                pair_markdown,
                pair["base"],  # type: ignore[arg-type]
                title=f"Full Mix vs {view_name.title()} Published Comparison",
                left_label="full_mix",
                right_label=view_name,
            )
            generated = pair_markdown.read_text(encoding="utf-8").splitlines()
            lines.extend(["", f"## Full Mix vs {view_name.title()}", "", *generated[2:]])
            _markdown_evidence(
                lines,
                view_name,
                pair,
                resources[view_name],
                failure_histograms[view_name],
            )
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (stage / "summary.json").write_bytes(canonical_json_bytes(summary))
        os.replace(stage / "summary.json", output_dir / "summary.json")
        os.replace(markdown_path, output_dir / "summary.md")


def compare_oaf_separation(request: SeparationComparisonRequest) -> SeparationComparisonOutcome:
    """Join full-mix and both derived HPA-325 report populations."""
    if not isinstance(request, SeparationComparisonRequest):
        raise TypeError("request must be SeparationComparisonRequest")
    try:
        from src.benchmark.separation_pilot import parse_oaf_separation_run

        reference_manifest = load_reference_set_manifest(request.reference_manifest_path)
        timing_manifest = load_reference_timing_manifest(request.timing_manifest_path)
        subset_manifest = load_reviewed_subset_manifest(request.subset_manifest_path)
        snapshot = parse_oaf_separation_run(read_regular_file_no_follow(request.run_path))
        if snapshot.get("schema") != SEPARATION_RUN_SCHEMA:
            raise ComparisonIntegrityError("run snapshot schema is invalid")
        if snapshot.get("reference_manifest_sha256") != reference_manifest.manifest_sha256:
            raise ComparisonIntegrityError("run/reference manifest lineage does not match")
        if snapshot.get("reference_timing_manifest_sha256") != timing_manifest.manifest_sha256:
            raise ComparisonIntegrityError("run/timing manifest lineage does not match")
        if snapshot.get("reviewed_subset_manifest_sha256") != subset_manifest.manifest_sha256:
            raise ComparisonIntegrityError("run/subset manifest lineage does not match")

        run_dir = request.run_path.parent
        reports_by_view: dict[str, PublishedCohortReports] = {}
        identities: dict[str, CohortIdentity] = {}
        for view_name in ("full_mix", "spleeter", "htdemucs"):
            report_dir = run_dir / "views" / view_name / "reports"
            identity = _validated_identity(
                report_dir,
                snapshot,
                reference_manifest,
                timing_manifest,
                subset_manifest,
                view_name,
            )
            reports_by_view[view_name] = read_cohort_reports(
                report_dir,
                expected_identity=identity,
            )
            identities[view_name] = identity
        if len({identity.model_id for identity in identities.values()}) != 1:
            raise ComparisonIntegrityError("OaF model identities do not match")
        full_mix = _view_evidence(snapshot, reports_by_view["full_mix"], "full_mix")
        durations = _durations(snapshot)
        comparison_identity = _comparison_identity(
            snapshot,
            reference_manifest,
            timing_manifest,
            subset_manifest,
        )
        pair_results: dict[str, Mapping[str, object]] = {}
        resources: dict[str, Mapping[str, object]] = {}
        failure_histograms: dict[str, Mapping[str, int]] = {}
        for view_name in ("spleeter", "htdemucs"):
            view = _view_evidence(snapshot, reports_by_view[view_name], view_name)
            pairable_ids, exclusions = pairable_success_ids(
                full_mix,
                view,
                None,
                require_identical_input_hash=False,
                left_label="full_mix",
                right_label=view_name,
            )
            pair = _pair_summary(
                full_mix,
                view,
                pairable_ids=pairable_ids,
                exclusions=exclusions,
                source_durations=durations,
                reference_manifest=reference_manifest,
                timing_manifest=timing_manifest,
                subset_path=request.subset_manifest_path,
                subset_manifest=subset_manifest,
                identity=comparison_identity,
                view_name=view_name,
            )
            pair_results[view_name] = pair
            report_dir = run_dir / "views" / view_name / "reports"
            resources[view_name] = _resources(
                snapshot,
                run_dir,
                request.cache_dir,
                view_name,
                report_dir,
            )
            failure_histograms[view_name] = _failure_histogram(snapshot, view_name)

        full_population = pair_results["spleeter"]["base"]["models"]["full_mix"]
        summary_models: dict[str, object] = {"full_mix": full_population}
        for view_name in ("spleeter", "htdemucs"):
            model = dict(pair_results[view_name]["base"]["models"][view_name])
            model["resources"] = resources[view_name]
            model["failure_code_histogram"] = failure_histograms[view_name]
            summary_models[view_name] = model
        summary = {
            "schema": SEPARATION_COMPARISON_SCHEMA,
            "identity": comparison_identity,
            "models": summary_models,
            "pairing": {
                view_name: pair_results[view_name]["base"]["pairing"]
                for view_name in ("spleeter", "htdemucs")
            },
            "comparisons": {
                view_name: {
                    "event_micro": pair_results[view_name]["event_micro"],
                    "aggregates": pair_results[view_name]["base"]["aggregates"],
                }
                for view_name in ("spleeter", "htdemucs")
            },
        }
        _write_outputs(
            request.output_dir,
            pair_results,
            summary,
            resources,
            failure_histograms,
        )
        return SeparationComparisonOutcome(
            output_dir=request.output_dir,
            pairable_success_counts={
                view_name: len(pair_results[view_name]["pairable_ids"])
                for view_name in ("spleeter", "htdemucs")
            },
            paired_song_counts={
                view_name: len(pair_results[view_name]["song_rows"])
                for view_name in ("spleeter", "htdemucs")
            },
            paired_class_counts={
                view_name: len(pair_results[view_name]["class_rows"])
                for view_name in ("spleeter", "htdemucs")
            },
        )
    except ComparisonIntegrityError:
        raise
    except (OSError, StrictJsonError, TypeError, ValueError) as error:
        raise ComparisonIntegrityError(str(error)) from error


__all__ = [
    "SEPARATION_COMPARISON_SCHEMA",
    "SEPARATION_COMPARISON_TITLE",
    "SeparationComparisonOutcome",
    "SeparationComparisonRequest",
    "aggregate_paired_event_micro",
    "compare_oaf_separation",
]
