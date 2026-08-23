"""Compose the persisted HPA-562 pairwise comparison publications."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import StrictJsonError, require_sha256, strict_json_loads
from src.benchmark.idm_comparison import (
    IDM_COMPARISON_SCHEMA,
    IdmComparisonRequest,
    compare_oaf_idm,
)
from src.benchmark.idm_pilot_run import IDM_STEM_INPUT_VIEW_ID
from src.benchmark.muscriptor_comparison import (
    COMPARISON_SCHEMA,
    ComparisonRequest,
    compare_oaf_muscriptor,
)
from src.benchmark.published_comparison import ComparisonIntegrityError
from src.benchmark.separation_comparison import (
    HTDEMUCS_INPUT_VIEW_ID,
    SEPARATION_COMPARISON_SCHEMA,
    SeparationComparisonRequest,
    compare_oaf_separation,
)

PAIRED_BENCHMARK_PUBLICATION_SCHEMA = "crux.paired-benchmark-publication/v1"

_COMPARISON_DIRS = {
    "oaf_muscriptor_full_mix": Path("comparisons/oaf-muscriptor"),
    "oaf_separation_pilot": Path("comparisons/oaf-separation"),
    "oaf_idm_htdemucs": Path("comparisons/oaf-idm"),
}
_SHARED_FIELDS = (
    "reference_manifest_sha256",
    "reference_manifest_version",
    "reference_timing_manifest_sha256",
    "reference_timing_version",
    "taxonomy_version",
    "lane_map_version",
    "scoring_version",
)


@dataclass(frozen=True)
class CrossComparisonRequest:
    """Inputs required to regenerate and compose all HPA-562 comparisons."""

    oaf_run_path: Path
    muscriptor_run_path: Path
    separation_run_path: Path
    idm_run_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    subset_manifest_path: Path
    output_dir: Path
    separation_cache_dir: Path | None = None

    def __post_init__(self) -> None:
        for field in (
            "oaf_run_path",
            "muscriptor_run_path",
            "separation_run_path",
            "idm_run_path",
            "reference_manifest_path",
            "timing_manifest_path",
            "subset_manifest_path",
            "output_dir",
        ):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a Path")
        if self.separation_cache_dir is not None and not isinstance(
            self.separation_cache_dir, Path
        ):
            raise TypeError("separation_cache_dir must be a Path or None")


@dataclass(frozen=True)
class CrossComparisonOutcome:
    """Published paths and pairable-success counts for HPA-562."""

    output_dir: Path
    headline_matrix_path: Path
    comparison_paths: dict[str, Path]
    pairable_success_counts: dict[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a Path")
        if not isinstance(self.headline_matrix_path, Path) or not self.headline_matrix_path.parts:
            raise TypeError("headline_matrix_path must be a nonempty Path")
        if not isinstance(self.comparison_paths, dict) or not self.comparison_paths:
            raise ValueError("comparison_paths must be nonempty")
        for comparison_id, path in self.comparison_paths.items():
            if not isinstance(comparison_id, str) or not comparison_id:
                raise TypeError("comparison_paths keys must be nonempty strings")
            if not isinstance(path, Path) or not path.parts:
                raise TypeError("comparison_paths values must be nonempty Paths")
        if not isinstance(self.pairable_success_counts, dict):
            raise TypeError("pairable_success_counts must be a dict")
        for comparison_id, count in self.pairable_success_counts.items():
            if not isinstance(comparison_id, str) or not comparison_id:
                raise TypeError("pairable_success_counts keys must be nonempty strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("pairable_success_counts contains an invalid count")


def _read_summary(path: Path, *, expected_schema: str) -> Mapping[str, object]:
    try:
        content = read_regular_file_no_follow(path)
        value = strict_json_loads(content, require_canonical=True)
    except (OSError, StrictJsonError) as error:
        raise ComparisonIntegrityError(f"invalid comparison summary: {error}") from error
    if not isinstance(value, Mapping) or value.get("schema") != expected_schema:
        raise ComparisonIntegrityError("comparison summary schema mismatch")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ComparisonIntegrityError(f"{label} must be a mapping")
    return value


def _model(summary: Mapping[str, object], model_key: str) -> Mapping[str, object]:
    models = _mapping(summary.get("models"), "comparison summary models")
    value = models.get(model_key)
    if not isinstance(value, Mapping):
        raise ComparisonIntegrityError(f"comparison summary models[{model_key!r}] is malformed")
    return value


def _validate_shared_identity(summaries: Mapping[str, Mapping[str, object]]) -> None:
    identities: dict[str, Mapping[str, object]] = {}
    for comparison_id, summary in summaries.items():
        identity = summary.get("identity")
        identities[comparison_id] = _mapping(identity, f"{comparison_id} comparison identity")

    for field in _SHARED_FIELDS:
        values: list[str] = []
        for identity in identities.values():
            value = identity.get(field)
            if not isinstance(value, str) or not value:
                raise ComparisonIntegrityError(f"{field} must be a nonempty value")
            values.append(value)
        if len(set(values)) != 1:
            raise ComparisonIntegrityError(f"{field} mismatch")


def _validate_oaf_identity(summaries: Mapping[str, Mapping[str, object]]) -> None:
    muscriptor_oaf = _model(summaries["oaf_muscriptor_full_mix"], "oaf")
    separation_full_mix = _model(summaries["oaf_separation_pilot"], "full_mix")
    separation_htdemucs = _model(summaries["oaf_separation_pilot"], "htdemucs")
    idm_oaf = _model(summaries["oaf_idm_htdemucs"], "oaf")

    locks: list[str] = []
    for label, model in (
        ("MuScriptor OaF", muscriptor_oaf),
        ("separation full_mix", separation_full_mix),
        ("IDM OaF", idm_oaf),
    ):
        value = model.get("model_lock_sha256")
        if not isinstance(value, str):
            raise ComparisonIntegrityError(f"{label} model_lock_sha256 is malformed")
        try:
            locks.append(require_sha256(value, "model_lock_sha256"))
        except (StrictJsonError, TypeError) as error:
            raise ComparisonIntegrityError(
                f"{label} model_lock_sha256 is malformed: {error}"
            ) from error
    if len(set(locks)) != 1:
        raise ComparisonIntegrityError("model_lock_sha256 mismatch")

    separation_view = separation_htdemucs.get("input_view_id")
    idm_view = idm_oaf.get("input_view_id")
    if (
        separation_view != HTDEMUCS_INPUT_VIEW_ID
        or idm_view != IDM_STEM_INPUT_VIEW_ID
        or separation_view != idm_view
    ):
        raise ComparisonIntegrityError("HTDemucs input_view_id mismatch")


def publish_cross_comparisons(request: CrossComparisonRequest) -> CrossComparisonOutcome:
    """Run the three authoritative comparisons and publish one staged bundle."""
    if not isinstance(request, CrossComparisonRequest):
        raise TypeError("request must be CrossComparisonRequest")
    if request.output_dir.exists():
        raise ComparisonIntegrityError("output_dir already exists")

    comparison_paths = {
        comparison_id: request.output_dir / relative_path
        for comparison_id, relative_path in _COMPARISON_DIRS.items()
    }
    request.output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(
            prefix=".paired-comparison-", dir=request.output_dir.parent
        ) as stage_name:
            stage = Path(stage_name)
            compare_oaf_muscriptor(
                ComparisonRequest(
                    oaf_run_path=request.oaf_run_path,
                    muscriptor_run_path=request.muscriptor_run_path,
                    reference_manifest_path=request.reference_manifest_path,
                    timing_manifest_path=request.timing_manifest_path,
                    output_dir=stage / _COMPARISON_DIRS["oaf_muscriptor_full_mix"],
                    subset_manifest_path=None,
                )
            )
            compare_oaf_separation(
                SeparationComparisonRequest(
                    run_path=request.separation_run_path,
                    reference_manifest_path=request.reference_manifest_path,
                    timing_manifest_path=request.timing_manifest_path,
                    subset_manifest_path=request.subset_manifest_path,
                    output_dir=stage / _COMPARISON_DIRS["oaf_separation_pilot"],
                    cache_dir=request.separation_cache_dir,
                )
            )
            compare_oaf_idm(
                IdmComparisonRequest(
                    run_path=request.idm_run_path,
                    output_dir=stage / _COMPARISON_DIRS["oaf_idm_htdemucs"],
                )
            )

            summaries = {
                "oaf_muscriptor_full_mix": _read_summary(
                    stage / _COMPARISON_DIRS["oaf_muscriptor_full_mix"] / "summary.json",
                    expected_schema=COMPARISON_SCHEMA,
                ),
                "oaf_separation_pilot": _read_summary(
                    stage / _COMPARISON_DIRS["oaf_separation_pilot"] / "summary.json",
                    expected_schema=SEPARATION_COMPARISON_SCHEMA,
                ),
                "oaf_idm_htdemucs": _read_summary(
                    stage / _COMPARISON_DIRS["oaf_idm_htdemucs"] / "summary.json",
                    expected_schema=IDM_COMPARISON_SCHEMA,
                ),
            }
            _validate_shared_identity(summaries)
            _validate_oaf_identity(summaries)
            os.rename(stage, request.output_dir)
    except ComparisonIntegrityError:
        raise
    except (OSError, StrictJsonError, TypeError, ValueError) as error:
        raise ComparisonIntegrityError(str(error)) from error

    return CrossComparisonOutcome(
        output_dir=request.output_dir,
        headline_matrix_path=request.output_dir / "headline_matrix.csv",
        comparison_paths=comparison_paths,
        pairable_success_counts={},
    )


__all__ = [
    "COMPARISON_SCHEMA",
    "CrossComparisonOutcome",
    "CrossComparisonRequest",
    "IDM_COMPARISON_SCHEMA",
    "PAIRED_BENCHMARK_PUBLICATION_SCHEMA",
    "SEPARATION_COMPARISON_SCHEMA",
    "publish_cross_comparisons",
]
