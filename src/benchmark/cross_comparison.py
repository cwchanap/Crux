"""Compose the persisted HPA-562 pairwise comparison publications."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    StrictJsonError,
    canonical_json_bytes,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
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
from src.benchmark.muscriptor_corpus_run import MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID
from src.benchmark.oaf_corpus_run import OAF_FULL_MIX_INPUT_VIEW_ID
from src.benchmark.published_comparison import ComparisonIntegrityError, write_csv
from src.benchmark.separation_comparison import (
    HTDEMUCS_INPUT_VIEW_ID,
    SEPARATION_COMPARISON_SCHEMA,
    SPLEETER_INPUT_VIEW_ID,
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
_POPULATION_FIELDS = (
    "total_count",
    "eligible_count",
    "success_count",
    "failed_count",
    "skipped_count",
    "quarantined_count",
)
_HEADLINE_FIELDS = (
    "scope",
    "model",
    "input_view_id",
    "total_count",
    "eligible_count",
    "success_count",
    "failed_count",
    "skipped_count",
    "quarantined_count",
    "comparison_ids",
)
_HEADLINE_SOURCES = (
    (
        "broad_full_mix",
        "oaf",
        OAF_FULL_MIX_INPUT_VIEW_ID,
        "oaf_muscriptor_full_mix",
        "oaf",
    ),
    (
        "broad_full_mix",
        "muscriptor",
        MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
        "oaf_muscriptor_full_mix",
        "muscriptor",
    ),
    (
        "reviewed_pilot",
        "oaf",
        OAF_FULL_MIX_INPUT_VIEW_ID,
        "oaf_separation_pilot",
        "full_mix",
    ),
    (
        "reviewed_pilot",
        "oaf",
        SPLEETER_INPUT_VIEW_ID,
        "oaf_separation_pilot",
        "spleeter",
    ),
    (
        "reviewed_pilot",
        "oaf",
        HTDEMUCS_INPUT_VIEW_ID,
        "oaf_separation_pilot",
        "htdemucs",
    ),
    (
        "reviewed_pilot",
        "idm",
        IDM_STEM_INPUT_VIEW_ID,
        "oaf_idm_htdemucs",
        "idm",
    ),
)
_EXPECTED_ARTIFACTS = {
    "oaf_muscriptor_full_mix": (
        "summary.json",
        "summary.md",
        "paired_per_song.csv",
        "paired_per_class.csv",
    ),
    "oaf_separation_pilot": (
        "summary.json",
        "summary.md",
        "spleeter/paired_per_song.csv",
        "spleeter/paired_per_class.csv",
        "htdemucs/paired_per_song.csv",
        "htdemucs/paired_per_class.csv",
    ),
    "oaf_idm_htdemucs": (
        "summary.json",
        "summary.md",
        "paired_per_song.csv",
        "paired_per_class.csv",
    ),
}
_PAIR_COUNT_SOURCES = {
    "oaf_muscriptor_full_mix": (
        "oaf_muscriptor_full_mix",
        ("pairing", "pairable_success_intersection"),
    ),
    "oaf_separation_pilot.spleeter": (
        "oaf_separation_pilot",
        ("pairing", "spleeter", "pairable_success_intersection"),
    ),
    "oaf_separation_pilot.htdemucs": (
        "oaf_separation_pilot",
        ("pairing", "htdemucs", "pairable_success_intersection"),
    ),
    "oaf_idm_htdemucs": (
        "oaf_idm_htdemucs",
        ("pairing", "pairable_success_intersection"),
    ),
}


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


def _validate_oaf_identity(summaries: Mapping[str, Mapping[str, object]]) -> str:
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
    return locks[0]


def _validate_population(
    model: Mapping[str, object], *, comparison_id: str, model_key: str
) -> dict[str, int]:
    value = model.get("population")
    if not isinstance(value, Mapping):
        raise ComparisonIntegrityError(
            f"{comparison_id} models[{model_key!r}] population is malformed"
        )
    if set(value) != set(_POPULATION_FIELDS):
        raise ComparisonIntegrityError(
            f"{comparison_id} models[{model_key!r}] population fields are invalid"
        )
    result: dict[str, int] = {}
    for field in _POPULATION_FIELDS:
        count = value[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ComparisonIntegrityError(
                f"{comparison_id} models[{model_key!r}] population.{field} is invalid"
            )
        result[field] = count
    return result


def _headline_rows(summaries: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope, model_name, expected_view, comparison_id, model_key in _HEADLINE_SOURCES:
        model = _model(summaries[comparison_id], model_key)
        input_view_id = model.get("input_view_id")
        if input_view_id != expected_view:
            raise ComparisonIntegrityError(
                f"{comparison_id} models[{model_key!r}] input_view_id mismatch"
            )
        population = _validate_population(
            model,
            comparison_id=comparison_id,
            model_key=model_key,
        )
        rows.append(
            {
                "scope": scope,
                "model": model_name,
                "input_view_id": expected_view,
                **population,
                "comparison_ids": comparison_id,
            }
        )
    return rows


def _pairable_success_counts(
    summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for count_key, (comparison_id, path) in _PAIR_COUNT_SOURCES.items():
        value: object = summaries[comparison_id]
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                raise ComparisonIntegrityError(f"{count_key} missing {'.'.join(path)}")
            value = value[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ComparisonIntegrityError(f"{count_key} {path[-1]} must be a non-negative integer")
        counts[count_key] = value
    return counts


def _artifact_index(stage_root: Path, comparison_id: str) -> list[dict[str, str]]:
    expected = _EXPECTED_ARTIFACTS[comparison_id]
    expected_files = set(expected)
    expected_dirs = {
        parent.as_posix()
        for parent in (Path(relative).parent for relative in expected)
        if parent != Path(".")
    }
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()

    def _raise_walk_error(error: OSError) -> None:
        raise error

    try:
        for current, dirnames, filenames in os.walk(stage_root, onerror=_raise_walk_error):
            current_dir = Path(current)
            for name in dirnames:
                path = current_dir / name
                relative = path.relative_to(stage_root).as_posix()
                if path.is_symlink():
                    actual_files.add(relative)
                else:
                    actual_dirs.add(relative)
            for name in filenames:
                actual_files.add((current_dir / name).relative_to(stage_root).as_posix())
    except OSError as error:
        raise ComparisonIntegrityError(
            f"cannot inspect comparison artifacts for {comparison_id}: {error}"
        ) from error
    missing = [relative for relative in expected if relative not in actual_files]
    if missing:
        raise ComparisonIntegrityError(
            f"missing expected comparison artifact: {comparison_id}/{missing[0]}"
        )
    unexpected = sorted((actual_files - expected_files) | (actual_dirs - expected_dirs))
    if unexpected:
        raise ComparisonIntegrityError(
            f"unexpected comparison artifact: {comparison_id}/{unexpected[0]}"
        )

    artifacts: list[dict[str, str]] = []
    for relative in expected:
        path = stage_root / relative
        try:
            digest = sha256_hex(read_regular_file_no_follow(path))
        except (OSError, TypeError) as error:
            raise ComparisonIntegrityError(
                f"invalid comparison artifact: {comparison_id}/{relative}: {error}"
            ) from error
        artifacts.append(
            {
                "path": (_COMPARISON_DIRS[comparison_id] / relative).as_posix(),
                "sha256": digest,
            }
        )
    return artifacts


def _scope_identities(
    summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    separation_identity = _mapping(
        summaries["oaf_separation_pilot"].get("identity"),
        "oaf_separation_pilot comparison identity",
    )
    reviewed_subset_hash = separation_identity.get("reviewed_subset_manifest_sha256")
    if not isinstance(reviewed_subset_hash, str):
        raise ComparisonIntegrityError("reviewed_subset_manifest_sha256 is malformed")
    try:
        reviewed_subset_hash = require_sha256(
            reviewed_subset_hash, "reviewed_subset_manifest_sha256"
        )
    except (StrictJsonError, TypeError) as error:
        raise ComparisonIntegrityError(
            f"reviewed_subset_manifest_sha256 is malformed: {error}"
        ) from error
    return {
        "oaf_muscriptor_full_mix": {},
        "oaf_separation_pilot": {
            "reviewed_subset_manifest_sha256": reviewed_subset_hash,
            "reviewed_subset_cross_verified": True,
        },
        "oaf_idm_htdemucs": {
            "pilot_lineage": "validated_hpa396_run",
            "reviewed_subset_cross_verified": False,
        },
    }


def _write_top_level_markdown(
    path: Path,
    *,
    identity: Mapping[str, object],
    rows: list[Mapping[str, object]],
    counts: Mapping[str, int],
    comparisons: Mapping[str, Mapping[str, object]],
) -> None:
    lines = [
        "# HPA-562 Paired Benchmark Publication",
        "",
        "## Identity",
        "",
        *[f"- {key}: `{value}`" for key, value in identity.items()],
        "",
        "## Headline Matrix",
        "",
        "| scope | model | input_view_id | total | eligible | success | failed | skipped | quarantined | comparison_ids |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scope']} | {row['model']} | {row['input_view_id']} | "
            f"{row['total_count']} | {row['eligible_count']} | {row['success_count']} | "
            f"{row['failed_count']} | {row['skipped_count']} | {row['quarantined_count']} | "
            f"{row['comparison_ids']} |"
        )
    lines.extend(["", "## Pairable Success Counts", ""])
    lines.extend(f"- {key}: {value}" for key, value in counts.items())
    lines.extend(["", "## Comparisons", ""])
    lines.extend(
        f"- {comparison_id}: `{entry['path']}`" for comparison_id, entry in comparisons.items()
    )
    lines.extend(
        [
            "",
            "## Scope Cautions",
            "",
            "Broad full-mix and reviewed-pilot rows have different populations and must not be ranked as one leaderboard.",
            "The IDM pilot lineage is validated inside its HPA-396 run; HPA-562 does not cross-verify its reviewed-subset identity against the HPA-328 separation publication.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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
            oaf_model_lock_sha256 = _validate_oaf_identity(summaries)
            headline_rows = _headline_rows(summaries)
            pairable_success_counts = _pairable_success_counts(summaries)
            artifacts = {
                comparison_id: _artifact_index(
                    stage / _COMPARISON_DIRS[comparison_id], comparison_id
                )
                for comparison_id in _COMPARISON_DIRS
            }
            scope_identities = _scope_identities(summaries)
            shared_identity = dict(
                _mapping(
                    summaries["oaf_muscriptor_full_mix"].get("identity"),
                    "oaf_muscriptor_full_mix comparison identity",
                )
            )
            top_identity = {field: shared_identity[field] for field in _SHARED_FIELDS}
            top_identity["oaf_model_lock_sha256"] = oaf_model_lock_sha256

            headline_path = stage / "headline_matrix.csv"
            write_csv(headline_path, _HEADLINE_FIELDS, headline_rows)
            headline_hash = sha256_hex(read_regular_file_no_follow(headline_path))
            comparisons = {
                comparison_id: {
                    "path": _COMPARISON_DIRS[comparison_id].as_posix(),
                    "artifacts": artifacts[comparison_id],
                    "scope_identity": scope_identities[comparison_id],
                }
                for comparison_id in _COMPARISON_DIRS
            }
            top_summary = {
                "schema": PAIRED_BENCHMARK_PUBLICATION_SCHEMA,
                "identity": top_identity,
                "pairable_success_counts": pairable_success_counts,
                "comparisons": comparisons,
                "headline_matrix": {
                    "path": "headline_matrix.csv",
                    "sha256": headline_hash,
                },
            }
            (stage / "summary.json").write_bytes(canonical_json_bytes(top_summary))
            _write_top_level_markdown(
                stage / "summary.md",
                identity=top_identity,
                rows=headline_rows,
                counts=pairable_success_counts,
                comparisons=comparisons,
            )
            os.rename(stage, request.output_dir)
    except ComparisonIntegrityError:
        raise
    except (OSError, StrictJsonError, TypeError, ValueError) as error:
        raise ComparisonIntegrityError(str(error)) from error

    return CrossComparisonOutcome(
        output_dir=request.output_dir,
        headline_matrix_path=request.output_dir / "headline_matrix.csv",
        comparison_paths=comparison_paths,
        pairable_success_counts=pairable_success_counts,
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
