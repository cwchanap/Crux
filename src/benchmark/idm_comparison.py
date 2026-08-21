"""Publish the strict HPA-396 OaF-versus-IDM stem comparison."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from statistics import median

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    IDM_BACKEND_ID,
    OAF_BACKEND_ID,
    StrictJsonError,
    require_sha256,
)
from src.benchmark.cohort_scoring import SCORING_VERSION, CohortIdentity
from src.benchmark.idm_pilot_run import (
    IDM_PILOT_RUN_SCHEMA,
    IDM_STEM_INPUT_VIEW_ID,
    parse_idm_pilot_run,
)
from src.benchmark.prediction_artifact import PredictionArtifactError, read_prediction_artifact
from src.benchmark.published_comparison import (
    ComparisonIntegrityError,
    PublishedRunEvidence,
    PublishedRunItem,
    comparison_summary,
    pairable_success_ids,
    paired_class_rows,
    paired_song_rows,
    runtime,
    selected_rows,
    write_comparison_artifacts,
)
from src.benchmark.reports import PublishedCohortReports, ReportIntegrityError, read_cohort_reports
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, IDM_PREDICTION_MAP_ID, TAXONOMY_VERSION

IDM_COMPARISON_SCHEMA = "crux.oaf-idm-comparison/v1"
IDM_COMPARISON_TITLE = "HPA-396 OaF/IDM Stem Published Comparison"
_SIX_PLACES = Decimal("0.000001")


@dataclass(frozen=True)
class IdmComparisonRequest:
    """Inputs for one persisted HPA-396 stem comparison."""

    run_path: Path
    output_dir: Path

    def __post_init__(self) -> None:
        for field in ("run_path", "output_dir"):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a Path")


def _fail(message: str) -> None:
    raise ComparisonIntegrityError(message)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a nonempty string")
    return value


def _hash(value: object, field: str) -> str:
    try:
        return require_sha256(_text(value, field), field)
    except (StrictJsonError, ValueError) as error:
        _fail(str(error))
    raise AssertionError("unreachable")


def _load_snapshot(path: Path) -> Mapping[str, object]:
    try:
        snapshot = parse_idm_pilot_run(read_regular_file_no_follow(path))
    except (OSError, StrictJsonError, TypeError, ValueError) as error:
        _fail(f"invalid IDM run snapshot: {error}")
    if snapshot.get("schema") != IDM_PILOT_RUN_SCHEMA:
        _fail("run snapshot schema is invalid")
    return snapshot


def _validate_snapshot_identity(snapshot: Mapping[str, object]) -> None:
    expected_input_view = IDM_STEM_INPUT_VIEW_ID
    if snapshot.get("input_view_id") != expected_input_view:
        _fail("IDM stem comparison requires the frozen stem input view")
    _hash(snapshot.get("reference_manifest_sha256"), "reference_manifest_sha256")
    _hash(
        snapshot.get("reference_timing_manifest_sha256"),
        "reference_timing_manifest_sha256",
    )
    _text(snapshot.get("reference_manifest_version"), "reference_manifest_version")
    _text(snapshot.get("reference_timing_version"), "reference_timing_version")

    descriptor = snapshot.get("backend_descriptor")
    if not isinstance(descriptor, Mapping) or descriptor.get("backend_id") != IDM_BACKEND_ID:
        _fail("run snapshot backend descriptor identity mismatch")

    for field, expected in (
        ("taxonomy_version", TAXONOMY_VERSION),
        ("lane_map_version", DTX_LANE_MAP_VERSION),
        ("scoring_version", SCORING_VERSION),
        ("prediction_map_version", IDM_PREDICTION_MAP_ID),
    ):
        if field in snapshot and snapshot[field] != expected:
            _fail(f"run snapshot {field} identity mismatch")

    inference_config = snapshot.get("inference_config")
    if inference_config is not None:
        if not isinstance(inference_config, Mapping):
            _fail("run snapshot inference_config is malformed")
        for field, expected in (
            ("input_view_id", expected_input_view),
            ("prediction_map_version", IDM_PREDICTION_MAP_ID),
        ):
            if field in inference_config and inference_config[field] != expected:
                _fail(f"run snapshot inference_config {field} identity mismatch")


def _identity(snapshot: Mapping[str, object], *, label: str) -> CohortIdentity:
    run_id = _text(snapshot.get("run_id"), "run_id")
    if label == "oaf":
        backend_id = OAF_BACKEND_ID
        model_id = _text(snapshot.get("oaf_model_id"), "oaf_model_id")
        model_lock = _hash(snapshot.get("oaf_model_lock_sha256"), "oaf_model_lock_sha256")
        descriptor = _hash(
            snapshot.get("oaf_backend_descriptor_sha256"),
            "oaf_backend_descriptor_sha256",
        )
        prediction_map = _text(
            snapshot.get("oaf_prediction_map_version"), "oaf_prediction_map_version"
        )
    elif label == "idm":
        backend_id = IDM_BACKEND_ID
        model_id = _text(snapshot.get("model_id"), "model_id")
        model_lock = _hash(snapshot.get("model_lock_sha256"), "model_lock_sha256")
        descriptor = _hash(snapshot.get("backend_descriptor_sha256"), "backend_descriptor_sha256")
        prediction_map = IDM_PREDICTION_MAP_ID
    else:  # pragma: no cover - labels are private call-site constants.
        raise ValueError("unsupported comparison label")

    try:
        return CohortIdentity(
            cohort_id=f"{run_id}:{label}",
            reference_manifest_sha256=_hash(
                snapshot.get("reference_manifest_sha256"), "reference_manifest_sha256"
            ),
            reference_timing_version=_text(
                snapshot.get("reference_timing_version"), "reference_timing_version"
            ),
            taxonomy_version=TAXONOMY_VERSION,
            lane_map_version=DTX_LANE_MAP_VERSION,
            backend_id=backend_id,
            model_id=model_id,
            model_lock_sha256=model_lock,
            backend_descriptor_sha256=descriptor,
            prediction_map_version=prediction_map,
            input_view_id=IDM_STEM_INPUT_VIEW_ID,
            scoring_version=SCORING_VERSION,
        )
    except (StrictJsonError, TypeError, ValueError) as error:
        _fail(f"{label} report identity is invalid: {error}")
    raise AssertionError("unreachable")


def _validate_pair_identity(left: CohortIdentity, right: CohortIdentity) -> None:
    for field in (
        "reference_manifest_sha256",
        "reference_timing_version",
        "taxonomy_version",
        "lane_map_version",
        "input_view_id",
        "scoring_version",
    ):
        if getattr(left, field) != getattr(right, field):
            _fail(f"paired report identity mismatch for {field}")


def _load_reports(
    report_dir: Path,
    *,
    expected_identity: CohortIdentity,
) -> PublishedCohortReports:
    try:
        return read_cohort_reports(report_dir, expected_identity=expected_identity)
    except (OSError, ReportIntegrityError, TypeError, ValueError) as error:
        _fail(str(error))
    raise AssertionError("unreachable")


def _run_items(
    reports: PublishedCohortReports,
    snapshot_items: Mapping[str, Mapping[str, object]],
    *,
    label: str,
    validate_snapshot_status: bool = False,
) -> dict[str, PublishedRunItem]:
    report_items = {row.simfile_id: row for row in reports.items}
    if set(report_items) != set(snapshot_items):
        _fail("published report population does not match IDM run snapshot")
    result: dict[str, PublishedRunItem] = {}
    for simfile_id in sorted(report_items, key=int):
        row = report_items[simfile_id]
        raw = snapshot_items[simfile_id]
        source_hash = raw.get(f"{label}_source_audio_sha256", raw.get("source_audio_sha256"))
        input_hash = raw.get(f"{label}_input_audio_sha256", raw.get("input_audio_sha256"))
        if source_hash is not None and not isinstance(source_hash, str):
            _fail("source_audio_sha256 is malformed")
        if input_hash is not None and not isinstance(input_hash, str):
            _fail("input_audio_sha256 is malformed")
        if validate_snapshot_status:
            disposition = raw.get("execution_disposition")
            expected_status = {
                "inferred": "success",
                "resumed": "success",
                "failed": "failed",
                "quarantined": "quarantined",
            }.get(disposition)
            if expected_status != row.status:
                _fail("IDM items report status does not match run snapshot")
        result[simfile_id] = PublishedRunItem(
            simfile_id=simfile_id,
            status=row.status,
            source_audio_sha256=source_hash,
            input_audio_sha256=input_hash,
        )
    return result


def _evidence(
    identity: CohortIdentity,
    reports: PublishedCohortReports,
    items: dict[str, PublishedRunItem],
    *,
    label: str,
    snapshot: Mapping[str, object],
) -> PublishedRunEvidence:
    return PublishedRunEvidence(
        identity=identity,
        items=items,
        reports=reports,
        snapshot=snapshot,
        label=label,
    )


def _coverage(reports: PublishedCohortReports) -> dict[str, object]:
    successful = [row for row in reports.items if row.status == "success"]
    native_count = sum(row.prediction_native_event_count or 0 for row in successful)
    mapped_count = sum(row.prediction_mapped_event_count or 0 for row in successful)
    unmapped_count = sum(row.prediction_unmapped_event_count or 0 for row in successful)
    native_classes: Counter[str] = Counter()
    for row in successful:
        native_classes.update(dict(row.prediction_native_class_counts))
    mapping_coverage = (
        (Decimal(mapped_count) / Decimal(native_count)).quantize(
            _SIX_PLACES, rounding=ROUND_HALF_EVEN
        )
        if native_count
        else None
    )
    return {
        "successful_song_count": len(successful),
        "prediction_native_event_count": native_count,
        "prediction_mapped_event_count": mapped_count,
        "prediction_unmapped_event_count": unmapped_count,
        "prediction_mapping_coverage": mapping_coverage,
        "prediction_native_class_counts": dict(sorted(native_classes.items())),
    }


def _native_failure_histogram(snapshot: Mapping[str, object]) -> dict[str, int]:
    raw = snapshot.get("native_failure_counts", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        _fail("native_failure_counts must be an object")
    result: dict[str, int] = {}
    for code, count in raw.items():
        if not isinstance(code, str) or not code:
            _fail("native_failure_counts contains an invalid code")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _fail("native_failure_counts contains an invalid count")
        result[code] = count
    return dict(sorted(result.items()))


def _runtime_diagnostics(snapshot: Mapping[str, object]) -> dict[str, object]:
    result = dict(runtime(snapshot))
    values: list[Decimal] = []
    raw_items = snapshot.get("items", [])
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, Mapping) or raw.get("execution_disposition") not in {
                "inferred",
                "resumed",
            }:
                continue
            value = raw.get("rtf")
            if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            decimal = Decimal(str(value))
            if decimal.is_finite() and decimal >= 0:
                values.append(decimal.quantize(_SIX_PLACES, rounding=ROUND_HALF_EVEN))
    if values:
        ordered = sorted(values)
        result["rtf_distribution"] = {
            "available_count": len(ordered),
            "minimum": ordered[0],
            "p10": ordered[min(len(ordered) - 1, math.ceil((len(ordered) - 1) * 0.10))],
            "median": median(ordered),
            "p90": ordered[min(len(ordered) - 1, math.ceil((len(ordered) - 1) * 0.90))],
            "maximum": ordered[-1],
        }
    return result


def _owned_prediction_path(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        return None
    candidate = root.joinpath(*parsed.parts)
    try:
        root_resolved = root.resolve()
        candidate.resolve().relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _velocity_diagnostics(
    snapshot: Mapping[str, object],
    *,
    run_roots: tuple[Path, ...],
    total_event_count: int,
) -> dict[str, object]:
    values: list[Decimal] = []
    raw_items = snapshot.get("items", [])
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, Mapping) or raw.get("execution_disposition") not in {
                "inferred",
                "resumed",
            }:
                continue
            artifact = None
            for run_root in run_roots:
                path = _owned_prediction_path(run_root, raw.get("prediction_path"))
                if path is None:
                    continue
                try:
                    artifact = read_prediction_artifact(read_regular_file_no_follow(path))
                except (OSError, PredictionArtifactError, StrictJsonError, TypeError, ValueError):
                    continue
                break
            if artifact is None:
                continue
            for event in artifact.prediction.events:
                velocity = event.native.native_metadata.get("native_velocity")
                if not isinstance(velocity, str):
                    continue
                try:
                    value = Decimal(velocity)
                except (InvalidOperation, ValueError):
                    continue
                if value.is_finite():
                    values.append(value)

    ordered = sorted(values)
    distribution: dict[str, object] | None = None
    if ordered:
        distribution = {
            "minimum": ordered[0],
            "p10": ordered[min(len(ordered) - 1, math.ceil((len(ordered) - 1) * 0.10))],
            "median": median(ordered),
            "p90": ordered[min(len(ordered) - 1, math.ceil((len(ordered) - 1) * 0.90))],
            "maximum": ordered[-1],
        }
    return {
        "event_count": total_event_count,
        "available_event_count": len(ordered),
        "availability": (
            (Decimal(len(ordered)) / Decimal(total_event_count)).quantize(
                _SIX_PLACES, rounding=ROUND_HALF_EVEN
            )
            if total_event_count
            else None
        ),
        "native_velocity_distribution": distribution,
    }


def _append_diagnostics_markdown(path: Path, summary: Mapping[str, object]) -> None:
    models = summary.get("models")
    if not isinstance(models, Mapping) or not isinstance(models.get("idm"), Mapping):
        return
    idm = models["idm"]
    coverage = idm.get("coverage", {})
    runtime_info = idm.get("runtime", {})
    velocity = idm.get("velocity", {})
    lines = [
        "",
        "## IDM Diagnostics",
        "",
        f"- native_failure_histogram: `{idm.get('native_failure_histogram', {})}`",
        f"- coverage: `{coverage}`",
        f"- runtime: `{runtime_info}`",
        f"- velocity: `{velocity}`",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def compare_oaf_idm(request: IdmComparisonRequest) -> Path:
    """Join the persisted OaF and IDM HPA-396 stem reports without rescoring."""
    if not isinstance(request, IdmComparisonRequest):
        raise TypeError("request must be IdmComparisonRequest")
    try:
        snapshot = _load_snapshot(request.run_path)
        _validate_snapshot_identity(snapshot)
        raw_items = snapshot.get("items")
        if not isinstance(raw_items, list):
            _fail("run snapshot items must be an array")
        snapshot_items: dict[str, Mapping[str, object]] = {}
        for raw in raw_items:
            if not isinstance(raw, Mapping) or type(raw.get("simfile_id")) is not int:
                _fail("run snapshot item is malformed")
            simfile_id = str(raw["simfile_id"])
            if simfile_id in snapshot_items:
                _fail("run snapshot contains duplicate simfile_id")
            snapshot_items[simfile_id] = raw

        oaf_identity = _identity(snapshot, label="oaf")
        idm_identity = _identity(snapshot, label="idm")
        _validate_pair_identity(oaf_identity, idm_identity)
        report_root = request.run_path.parent / "reports"
        oaf_reports = _load_reports(report_root / "oaf", expected_identity=oaf_identity)
        idm_reports = _load_reports(report_root / "idm", expected_identity=idm_identity)
        oaf_items = _run_items(oaf_reports, snapshot_items, label="oaf")
        idm_items = _run_items(
            idm_reports,
            snapshot_items,
            label="idm",
            validate_snapshot_status=True,
        )
        # The OaF report receives the same source/input ledger as IDM.  Its
        # publication status is taken from the report itself because retained
        # OaF evidence and IDM inference can fail independently.
        oaf = _evidence(
            oaf_identity,
            oaf_reports,
            oaf_items,
            label="oaf",
            snapshot={},
        )
        idm = _evidence(
            idm_identity,
            idm_reports,
            idm_items,
            label="idm",
            snapshot=snapshot,
        )
        pairable_ids, exclusions = pairable_success_ids(
            oaf,
            idm,
            None,
            require_identical_input_hash=True,
            left_label="oaf",
            right_label="idm",
        )
        oaf_songs = selected_rows(
            {(row.simfile_id, row.tolerance_ms, row.mode): row for row in oaf_reports.songs},
            None,
        )
        idm_songs = selected_rows(
            {(row.simfile_id, row.tolerance_ms, row.mode): row for row in idm_reports.songs},
            None,
        )
        oaf_classes = selected_rows(
            {
                (row.simfile_id, row.tolerance_ms, row.mode, row.common_class): row
                for row in oaf_reports.classes
            },
            None,
        )
        idm_classes = selected_rows(
            {
                (row.simfile_id, row.tolerance_ms, row.mode, row.common_class): row
                for row in idm_reports.classes
            },
            None,
        )
        song_rows = paired_song_rows(
            oaf_songs,
            idm_songs,
            pairable_ids,
            left_label="oaf",
            right_label="idm",
        )
        class_rows = paired_class_rows(
            oaf_classes,
            idm_classes,
            pairable_ids,
            left_label="oaf",
            right_label="idm",
        )
        summary = comparison_summary(
            oaf,
            idm,
            pairable_ids,
            exclusions,
            song_rows,
            class_rows,
            snapshot,
            snapshot,
            None,
            None,
            schema=IDM_COMPARISON_SCHEMA,
            identity={
                "run_id": snapshot["run_id"],
                "reference_manifest_sha256": snapshot["reference_manifest_sha256"],
                "reference_manifest_version": snapshot["reference_manifest_version"],
                "reference_timing_manifest_sha256": snapshot["reference_timing_manifest_sha256"],
                "reference_timing_version": snapshot["reference_timing_version"],
                "taxonomy_version": TAXONOMY_VERSION,
                "lane_map_version": DTX_LANE_MAP_VERSION,
                "input_view_id": IDM_STEM_INPUT_VIEW_ID,
                "scoring_version": SCORING_VERSION,
            },
            left_label="oaf",
            right_label="idm",
        )
        models = summary["models"]
        if not isinstance(models, dict):
            _fail("comparison summary models must be a mapping")
        idm_model = models.get("idm")
        if not isinstance(idm_model, dict):
            _fail("comparison summary IDM model is malformed")
        for label, reports in (("oaf", oaf_reports), ("idm", idm_reports)):
            model = models.get(label)
            if isinstance(model, dict):
                model["reason_counts"] = dict(reports.population.reason_counts)
        coverage = _coverage(idm_reports)
        idm_model["coverage"] = coverage
        idm_model["native_failure_histogram"] = _native_failure_histogram(snapshot)
        idm_model["runtime"] = _runtime_diagnostics(snapshot)
        idm_model["velocity"] = _velocity_diagnostics(
            snapshot,
            run_roots=(
                request.run_path.parent,
                request.run_path.parent.parent,
                request.run_path.parent.parent.parent,
            ),
            total_event_count=int(coverage["prediction_native_event_count"]),
        )
        write_comparison_artifacts(
            request.output_dir,
            song_rows,
            class_rows,
            summary,
            title=IDM_COMPARISON_TITLE,
            left_label="oaf",
            right_label="idm",
        )
        _append_diagnostics_markdown(request.output_dir / "summary.md", summary)
        return request.output_dir
    except ComparisonIntegrityError:
        raise
    except (OSError, StrictJsonError, ReportIntegrityError, TypeError, ValueError) as error:
        raise ComparisonIntegrityError(str(error)) from error


__all__ = [
    "IDM_COMPARISON_SCHEMA",
    "IDM_COMPARISON_TITLE",
    "ComparisonIntegrityError",
    "IdmComparisonRequest",
    "compare_oaf_idm",
]
