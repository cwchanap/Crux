from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.benchmark.backend_identity import (
    MUSCRIPTOR_BACKEND_ID,
    OAF_BACKEND_ID,
    canonical_json_bytes,
)
from src.benchmark.muscriptor_comparison import (
    ComparisonIntegrityError,
    ComparisonRequest,
    compare_oaf_muscriptor,
)
from src.benchmark.muscriptor_corpus_run import (
    MUSCRIPTOR_CORPUS_RUN_SCHEMA,
    render_muscriptor_corpus_run,
)
from src.benchmark.oaf_corpus_run import OAF_CORPUS_RUN_SCHEMA, render_oaf_corpus_run
from src.benchmark.reports import REPORT_SCHEMA
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION

_REPORT_IDENTITY = (
    "cohort_id",
    "model_id",
    "model_lock_sha256",
    "prediction_map_version",
    "input_view_id",
    "scoring_version",
)
_SONG_FIELDS = (
    *_REPORT_IDENTITY,
    "simfile_id",
    "tolerance_ms",
    "mode",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "prediction_to_reference_ratio",
    "median_abs_error_ms",
    "p95_abs_error_ms",
    "offset_ms",
    "warnings",
)
_CLASS_FIELDS = (
    *_REPORT_IDENTITY,
    "simfile_id",
    "tolerance_ms",
    "mode",
    "common_class",
    "tp",
    "fp",
    "fn",
    "reference_support",
    "prediction_support",
    "precision",
    "recall",
    "f1",
)
_ITEM_FIELDS = (
    "cohort_id",
    "simfile_id",
    "status",
    "failure_reason",
    "warnings",
    "reference_native_event_count",
    "reference_common_event_count",
    "reference_ignored_event_count",
    "reference_unmapped_event_count",
    "reference_duplicate_collapsed_count",
    "prediction_native_event_count",
    "prediction_mapped_event_count",
    "prediction_unmapped_event_count",
    "prediction_mapping_coverage",
    "prediction_native_class_counts",
)


@pytest.fixture
def manifest_loaders(monkeypatch: pytest.MonkeyPatch):
    import src.benchmark.muscriptor_comparison as comparison

    monkeypatch.setattr(
        comparison,
        "load_reference_set_manifest",
        lambda _path: SimpleNamespace(manifest_sha256="a" * 64, corpus_version="hpa324-v1"),
    )
    monkeypatch.setattr(
        comparison,
        "load_reference_timing_manifest",
        lambda _path: SimpleNamespace(manifest_sha256="b" * 64, corpus_version="hpa323-v1"),
    )


def _snapshot(
    schema: str,
    run_id: str,
    model_id: str,
    model_lock: str,
    prediction_map: str,
    *,
    input_view: str = "crux.full-mix/v1",
    input_audio: str = "c" * 64,
    source_audio: str = "d" * 64,
    disposition: str = "inferred",
    item_ids: tuple[int, ...] = (1,),
    item_dispositions: dict[int, str] | None = None,
) -> bytes:
    dispositions = item_dispositions or {simfile_id: disposition for simfile_id in item_ids}
    items = [
        {
            "simfile_id": simfile_id,
            "execution_disposition": dispositions[simfile_id],
            "source_audio_sha256": source_audio,
            "input_audio_sha256": input_audio,
        }
        for simfile_id in item_ids
    ]
    snapshot = {
        "schema": schema,
        "run_id": run_id,
        "reference_manifest_sha256": "a" * 64,
        "reference_manifest_version": "hpa324-v1",
        "reference_timing_manifest_sha256": "b" * 64,
        "reference_timing_version": "hpa323-v1",
        "backend_descriptor": {
            "backend_id": (
                OAF_BACKEND_ID if schema == OAF_CORPUS_RUN_SCHEMA else MUSCRIPTOR_BACKEND_ID
            )
        },
        "backend_descriptor_sha256": "a" * 64,
        "model_id": model_id,
        "model_lock_sha256": model_lock,
        "prediction_map_version": prediction_map,
        "input_view_id": input_view,
        "items": items,
        "overall_status": "complete",
        "success_count": sum(dispositions[simfile_id] == "inferred" for simfile_id in item_ids),
        "failed_count": sum(dispositions[simfile_id] == "failed" for simfile_id in item_ids),
        "skipped_count": sum(dispositions[simfile_id] == "skipped" for simfile_id in item_ids),
        "quarantined_count": sum(
            dispositions[simfile_id] == "quarantined" for simfile_id in item_ids
        ),
    }
    if schema == OAF_CORPUS_RUN_SCHEMA:
        snapshot["inference_config"] = {"prediction_map_version": prediction_map}
        return render_oaf_corpus_run(snapshot)
    return render_muscriptor_corpus_run(snapshot)


def _write_run(
    root: Path,
    *,
    model: str,
    run_id: str,
    schema: str,
    lock: str,
    prediction_map: str,
    **kwargs: object,
) -> Path:
    run_path = root / "run.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_bytes(
        _snapshot(
            schema,
            run_id,
            model,
            lock,
            prediction_map,
            **kwargs,
        )
    )
    return run_path


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _reports(
    root: Path,
    cohort: str,
    model: str,
    lock: str,
    prediction_map: str,
    *,
    precision: str,
    item_ids: tuple[int, ...] = (1,),
    failed_item_ids: frozenset[int] = frozenset(),
):
    reports = root / "reports"
    identity = {
        "cohort_id": cohort,
        "model_id": model,
        "model_lock_sha256": lock,
        "prediction_map_version": prediction_map,
        "input_view_id": "crux.full-mix/v1",
        "scoring_version": "crux.single-cohort-scoring/v1",
    }
    _write_csv(
        reports / "items.csv",
        _ITEM_FIELDS,
        [
            {
                **{field: "" for field in _ITEM_FIELDS},
                "cohort_id": cohort,
                "simfile_id": str(simfile_id),
                "status": "failed" if simfile_id in failed_item_ids else "success",
                "failure_reason": ("prediction_missing" if simfile_id in failed_item_ids else ""),
                "reference_native_event_count": "1",
                "reference_common_event_count": "1",
                "reference_ignored_event_count": "0",
                "reference_unmapped_event_count": "0",
                "reference_duplicate_collapsed_count": "0",
                "prediction_native_event_count": ("" if simfile_id in failed_item_ids else "1"),
                "prediction_mapped_event_count": "" if simfile_id in failed_item_ids else "1",
                "prediction_unmapped_event_count": "" if simfile_id in failed_item_ids else "0",
                "prediction_mapping_coverage": "" if simfile_id in failed_item_ids else "1",
                "prediction_native_class_counts": "" if simfile_id in failed_item_ids else "kick=1",
            }
            for simfile_id in item_ids
        ],
    )
    _write_csv(
        reports / "per_song.csv",
        _SONG_FIELDS,
        (
            [
                {
                    **identity,
                    "simfile_id": str(simfile_id),
                    "tolerance_ms": "50",
                    "mode": mode,
                    "tp": "1",
                    "fp": "1",
                    "fn": "1",
                    "precision": precision,
                    "recall": "0.4",
                    "f1": "0.4",
                    "prediction_to_reference_ratio": "1",
                    "median_abs_error_ms": "",
                    "p95_abs_error_ms": "",
                    "offset_ms": "0",
                    "warnings": "",
                }
                for simfile_id in item_ids
                if simfile_id not in failed_item_ids
                for mode in ("raw", "aligned")
            ]
        ),
    )
    _write_csv(
        reports / "per_class.csv",
        _CLASS_FIELDS,
        (
            [
                {
                    **identity,
                    "simfile_id": str(simfile_id),
                    "tolerance_ms": "50",
                    "mode": mode,
                    "common_class": "kick",
                    "tp": "1",
                    "fp": "0",
                    "fn": "1",
                    "reference_support": "2",
                    "prediction_support": "1",
                    "precision": precision,
                    "recall": "0.5",
                    "f1": "0.5",
                }
                for simfile_id in item_ids
                if simfile_id not in failed_item_ids
                for mode in ("raw", "aligned")
            ]
        ),
    )
    success_count = len(set(item_ids) - set(failed_item_ids))
    backend_id = MUSCRIPTOR_BACKEND_ID if "muscriptor" in model else OAF_BACKEND_ID
    summary_identity = {
        "cohort_id": cohort,
        "reference_manifest_sha256": "a" * 64,
        "reference_timing_version": "hpa323-v1",
        "taxonomy_version": TAXONOMY_VERSION,
        "lane_map_version": DTX_LANE_MAP_VERSION,
        "backend_id": backend_id,
        "model_id": model,
        "model_lock_sha256": lock,
        "backend_descriptor_sha256": "a" * 64,
        "prediction_map_version": prediction_map,
        "input_view_id": "crux.full-mix/v1",
        "scoring_version": "crux.single-cohort-scoring/v1",
    }
    metric = {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": Decimal("0.5"),
        "recall": Decimal("0.5"),
        "f1": Decimal("0.5"),
    }
    aggregate = {
        "tolerance_ms": 50,
        "mode": "raw",
        "event_micro": metric,
        "song_macro_f1": Decimal("0.5") if success_count else None,
        "class_macro_f1": Decimal("0.5") if success_count else None,
        "song_f1_distribution": {
            field: Decimal("0.5") if success_count else None
            for field in ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")
        },
        "per_class": [
            {
                "common_class": "kick",
                **metric,
                "reference_support": 2,
                "prediction_support": 1,
            }
        ]
        if success_count
        else [],
        "successful_song_count": success_count,
    }
    summary = {
        "schema": REPORT_SCHEMA,
        "identity": summary_identity,
        "tolerances_ms": [50],
        "population": {
            "total_count": len(item_ids),
            "success_count": success_count,
            "failed_count": len(failed_item_ids),
            "skipped_count": 0,
            "quarantined_count": 0,
            "reason_counts": {"prediction_missing": len(failed_item_ids)}
            if failed_item_ids
            else {},
        },
        "aggregates": [
            aggregate,
            {**aggregate, "mode": "aligned"},
        ],
    }
    (reports / "summary.json").write_bytes(canonical_json_bytes(summary))


def _request(tmp_path: Path, oaf: Path, muscriptor: Path, subset: Path | None = None):
    return ComparisonRequest(
        oaf_run_path=oaf,
        muscriptor_run_path=muscriptor,
        reference_manifest_path=tmp_path / "hpa324.jsonl",
        timing_manifest_path=tmp_path / "hpa323.jsonl",
        output_dir=tmp_path / "comparison",
        subset_manifest_path=subset,
    )


def test_compare_joins_published_song_and_class_rows_without_rescoring(
    tmp_path: Path, manifest_loaders
) -> None:
    oaf_root = tmp_path / "oaf"
    muscriptor_root = tmp_path / "muscriptor"
    oaf = _write_run(
        oaf_root,
        model="oaf-model",
        run_id="oaf-run",
        schema=OAF_CORPUS_RUN_SCHEMA,
        lock="e" * 64,
        prediction_map="oaf-map",
    )
    muscriptor = _write_run(
        muscriptor_root,
        model="muscriptor-model",
        run_id="muscriptor-run",
        schema=MUSCRIPTOR_CORPUS_RUN_SCHEMA,
        lock="f" * 64,
        prediction_map="muscriptor-map",
    )
    _reports(oaf_root, "oaf-run", "oaf-model", "e" * 64, "oaf-map", precision="0.5")
    _reports(
        muscriptor_root,
        "muscriptor-run",
        "muscriptor-model",
        "f" * 64,
        "muscriptor-map",
        precision="0.8",
    )

    result = compare_oaf_muscriptor(_request(tmp_path, oaf, muscriptor))

    assert result.pairable_success_count == 1
    with (result.output_dir / "paired_per_song.csv").open(newline="", encoding="utf-8") as handle:
        song_rows = list(csv.DictReader(handle))
    assert song_rows[0] == {
        "simfile_id": "1",
        "tolerance_ms": "50",
        "mode": "raw",
        "oaf_precision": "0.5",
        "muscriptor_precision": "0.8",
        "delta_precision": "0.3",
        "oaf_recall": "0.4",
        "muscriptor_recall": "0.4",
        "delta_recall": "0",
        "oaf_f1": "0.4",
        "muscriptor_f1": "0.4",
        "delta_f1": "0",
    }
    with (result.output_dir / "paired_per_class.csv").open(newline="", encoding="utf-8") as handle:
        class_rows = list(csv.DictReader(handle))
    assert class_rows[0]["oaf_reference_support"] == "2"
    assert class_rows[0]["muscriptor_prediction_support"] == "1"
    assert class_rows[0]["delta_precision"] == "0.3"

    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "crux.oaf-muscriptor-comparison/v1"
    assert summary["pairing"]["pairable_success_intersection"] == 1
    assert summary["aggregates"]["song"][0]["mean_delta_precision"] == 0.3
    assert (
        (result.output_dir / "summary.md")
        .read_text(encoding="utf-8")
        .startswith("# OaF/MuScriptor Published Report Comparison\n")
    )
    assert "summary.md" in {path.name for path in result.output_dir.iterdir()}


def test_subset_filters_pairing_but_keeps_full_model_populations(
    tmp_path: Path,
    manifest_loaders,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oaf_root = tmp_path / "oaf"
    muscriptor_root = tmp_path / "muscriptor"
    common_kwargs = {
        "item_ids": (1, 2),
        "item_dispositions": {1: "inferred", 2: "failed"},
    }
    oaf = _write_run(
        oaf_root,
        model="oaf-model",
        run_id="oaf-run",
        schema=OAF_CORPUS_RUN_SCHEMA,
        lock="e" * 64,
        prediction_map="oaf-map",
        **common_kwargs,
    )
    muscriptor = _write_run(
        muscriptor_root,
        model="muscriptor-model",
        run_id="muscriptor-run",
        schema=MUSCRIPTOR_CORPUS_RUN_SCHEMA,
        lock="f" * 64,
        prediction_map="muscriptor-map",
        **common_kwargs,
    )
    _reports(
        oaf_root,
        "oaf-run",
        "oaf-model",
        "e" * 64,
        "oaf-map",
        precision="0.5",
        item_ids=(1, 2),
        failed_item_ids=frozenset({2}),
    )
    _reports(
        muscriptor_root,
        "muscriptor-run",
        "muscriptor-model",
        "f" * 64,
        "muscriptor-map",
        precision="0.8",
        item_ids=(1, 2),
        failed_item_ids=frozenset({2}),
    )

    import src.benchmark.muscriptor_comparison as comparison

    monkeypatch.setattr(
        comparison,
        "load_reviewed_subset_manifest",
        lambda _path: SimpleNamespace(
            manifest_sha256="s" * 64,
            corpus_version="hpa327-v1",
            review_policy_version="crux.review-policy/v1",
            review_ledger_sha256="t" * 64,
            source_reference_manifest_sha256="a" * 64,
            source_reference_manifest_version="hpa324-v1",
            source_timing_manifest_sha256="b" * 64,
            source_timing_manifest_version="hpa323-v1",
            rows=(SimpleNamespace(view=SimpleNamespace(simfile_id=1)),),
        ),
    )
    result = compare_oaf_muscriptor(
        _request(tmp_path, oaf, muscriptor, subset=tmp_path / "subset.json")
    )

    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["models"]["oaf"]["population"] == {
        "total_count": 2,
        "eligible_count": 2,
        "success_count": 1,
        "failed_count": 1,
        "skipped_count": 0,
        "quarantined_count": 0,
    }
    assert summary["models"]["muscriptor"]["population"] == summary["models"]["oaf"]["population"]
    assert summary["subset_manifest"]["path"] == str(tmp_path / "subset.json")
    assert summary["subset_manifest"]["manifest_sha256"] == "s" * 64
    assert result.pairable_success_count == 1


def test_compare_rejects_duplicate_score_keys_and_bad_numbers(
    tmp_path: Path, manifest_loaders
) -> None:
    oaf_root = tmp_path / "oaf"
    muscriptor_root = tmp_path / "muscriptor"
    oaf = _write_run(
        oaf_root,
        model="oaf-model",
        run_id="oaf-run",
        schema=OAF_CORPUS_RUN_SCHEMA,
        lock="e" * 64,
        prediction_map="oaf-map",
    )
    muscriptor = _write_run(
        muscriptor_root,
        model="muscriptor-model",
        run_id="muscriptor-run",
        schema=MUSCRIPTOR_CORPUS_RUN_SCHEMA,
        lock="f" * 64,
        prediction_map="muscriptor-map",
    )
    _reports(oaf_root, "oaf-run", "oaf-model", "e" * 64, "oaf-map", precision="0.5")
    _reports(
        muscriptor_root,
        "muscriptor-run",
        "muscriptor-model",
        "f" * 64,
        "muscriptor-map",
        precision="0.8",
    )
    with (oaf_root / "reports" / "per_song.csv").open("a", encoding="utf-8") as handle:
        handle.write(
            (oaf_root / "reports" / "per_song.csv").read_text(encoding="utf-8").splitlines()[1]
            + "\n"
        )

    with pytest.raises(ComparisonIntegrityError, match="duplicate"):
        compare_oaf_muscriptor(_request(tmp_path, oaf, muscriptor))

    lines = (oaf_root / "reports" / "per_song.csv").read_text(encoding="utf-8").splitlines()
    (oaf_root / "reports" / "per_song.csv").write_text(
        "\n".join([lines[0], lines[1].replace(",0.5,", ",not-a-number,")]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ComparisonIntegrityError, match="numeric"):
        compare_oaf_muscriptor(_request(tmp_path, oaf, muscriptor))


def test_compare_rejects_report_identity_mismatch_in_published_rows(
    tmp_path: Path, manifest_loaders
) -> None:
    oaf_root = tmp_path / "oaf"
    muscriptor_root = tmp_path / "muscriptor"
    oaf = _write_run(
        oaf_root,
        model="oaf-model",
        run_id="oaf-run",
        schema=OAF_CORPUS_RUN_SCHEMA,
        lock="e" * 64,
        prediction_map="oaf-map",
    )
    muscriptor = _write_run(
        muscriptor_root,
        model="muscriptor-model",
        run_id="muscriptor-run",
        schema=MUSCRIPTOR_CORPUS_RUN_SCHEMA,
        lock="f" * 64,
        prediction_map="muscriptor-map",
    )
    _reports(oaf_root, "oaf-run", "oaf-model", "e" * 64, "oaf-map", precision="0.5")
    _reports(
        muscriptor_root,
        "muscriptor-run",
        "muscriptor-model",
        "f" * 64,
        "muscriptor-map",
        precision="0.8",
    )
    # The altered cohort identity reaches per_song.csv, so the joined report
    # identity no longer matches the run snapshot.
    song_path = oaf_root / "reports" / "per_song.csv"
    song_path.write_text(
        song_path.read_text(encoding="utf-8").replace("oaf-run,", "other-run,"),
        encoding="utf-8",
    )

    with pytest.raises(ComparisonIntegrityError, match="identity mismatch"):
        compare_oaf_muscriptor(_request(tmp_path, oaf, muscriptor))


@pytest.mark.parametrize(
    ("oaf_schema", "muscriptor_schema", "message"),
    [
        (MUSCRIPTOR_CORPUS_RUN_SCHEMA, OAF_CORPUS_RUN_SCHEMA, "--oaf-run.*OaF"),
        (OAF_CORPUS_RUN_SCHEMA, OAF_CORPUS_RUN_SCHEMA, "--muscriptor-run.*MuScriptor"),
        (MUSCRIPTOR_CORPUS_RUN_SCHEMA, MUSCRIPTOR_CORPUS_RUN_SCHEMA, "--oaf-run.*OaF"),
    ],
)
def test_compare_rejects_swapped_or_same_family_runs(
    tmp_path: Path,
    manifest_loaders,
    oaf_schema: str,
    muscriptor_schema: str,
    message: str,
) -> None:
    oaf_root = tmp_path / "oaf"
    muscriptor_root = tmp_path / "muscriptor"
    oaf = _write_run(
        oaf_root,
        model="oaf-model",
        run_id="oaf-run",
        schema=oaf_schema,
        lock="e" * 64,
        prediction_map="oaf-map",
    )
    muscriptor = _write_run(
        muscriptor_root,
        model="muscriptor-model",
        run_id="muscriptor-run",
        schema=muscriptor_schema,
        lock="f" * 64,
        prediction_map="muscriptor-map",
    )
    _reports(oaf_root, "oaf-run", "oaf-model", "e" * 64, "oaf-map", precision="0.5")
    _reports(
        muscriptor_root,
        "muscriptor-run",
        "muscriptor-model",
        "f" * 64,
        "muscriptor-map",
        precision="0.8",
    )

    with pytest.raises(ComparisonIntegrityError, match=message):
        compare_oaf_muscriptor(_request(tmp_path, oaf, muscriptor))


def test_compare_rejects_canonical_input_mismatch_before_joining(
    tmp_path: Path, manifest_loaders
) -> None:
    oaf_root = tmp_path / "oaf"
    muscriptor_root = tmp_path / "muscriptor"
    oaf = _write_run(
        oaf_root,
        model="oaf-model",
        run_id="oaf-run",
        schema=OAF_CORPUS_RUN_SCHEMA,
        lock="e" * 64,
        prediction_map="oaf-map",
        input_audio="1" * 64,
    )
    muscriptor = _write_run(
        muscriptor_root,
        model="muscriptor-model",
        run_id="muscriptor-run",
        schema=MUSCRIPTOR_CORPUS_RUN_SCHEMA,
        lock="f" * 64,
        prediction_map="muscriptor-map",
        input_audio="2" * 64,
    )
    _reports(oaf_root, "oaf-run", "oaf-model", "e" * 64, "oaf-map", precision="0.5")
    _reports(
        muscriptor_root,
        "muscriptor-run",
        "muscriptor-model",
        "f" * 64,
        "muscriptor-map",
        precision="0.8",
    )

    with pytest.raises(ComparisonIntegrityError, match="canonical-input"):
        compare_oaf_muscriptor(_request(tmp_path, oaf, muscriptor))


def test_compare_rejects_one_sided_missing_per_song_row(tmp_path: Path, manifest_loaders) -> None:
    """A missing per_song row on one side is corrupted evidence, not an exclusion."""
    oaf_root = tmp_path / "oaf"
    muscriptor_root = tmp_path / "muscriptor"
    oaf = _write_run(
        oaf_root,
        model="oaf-model",
        run_id="oaf-run",
        schema=OAF_CORPUS_RUN_SCHEMA,
        lock="e" * 64,
        prediction_map="oaf-map",
    )
    muscriptor = _write_run(
        muscriptor_root,
        model="muscriptor-model",
        run_id="muscriptor-run",
        schema=MUSCRIPTOR_CORPUS_RUN_SCHEMA,
        lock="f" * 64,
        prediction_map="muscriptor-map",
    )
    _reports(oaf_root, "oaf-run", "oaf-model", "e" * 64, "oaf-map", precision="0.5")
    _reports(
        muscriptor_root,
        "muscriptor-run",
        "muscriptor-model",
        "f" * 64,
        "muscriptor-map",
        precision="0.8",
    )
    # Remove the only per_song row from MuScriptor, simulating a one-sided
    # missing score row for an otherwise pairable successful song.
    song_path = muscriptor_root / "reports" / "per_song.csv"
    lines = song_path.read_text(encoding="utf-8").splitlines()
    song_path.write_text(lines[0] + "\n", encoding="utf-8")

    with pytest.raises(ComparisonIntegrityError, match="per_song.*score grid"):
        compare_oaf_muscriptor(_request(tmp_path, oaf, muscriptor))


def test_compare_persists_subset_identity_in_summary(
    tmp_path: Path,
    manifest_loaders,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """summary.json records the immutable subset identity, not only the path."""
    oaf_root = tmp_path / "oaf"
    muscriptor_root = tmp_path / "muscriptor"
    common_kwargs = {
        "item_ids": (1, 2),
        "item_dispositions": {1: "inferred", 2: "failed"},
    }
    oaf = _write_run(
        oaf_root,
        model="oaf-model",
        run_id="oaf-run",
        schema=OAF_CORPUS_RUN_SCHEMA,
        lock="e" * 64,
        prediction_map="oaf-map",
        **common_kwargs,
    )
    muscriptor = _write_run(
        muscriptor_root,
        model="muscriptor-model",
        run_id="muscriptor-run",
        schema=MUSCRIPTOR_CORPUS_RUN_SCHEMA,
        lock="f" * 64,
        prediction_map="muscriptor-map",
        **common_kwargs,
    )
    _reports(
        oaf_root,
        "oaf-run",
        "oaf-model",
        "e" * 64,
        "oaf-map",
        precision="0.5",
        item_ids=(1, 2),
        failed_item_ids=frozenset({2}),
    )
    _reports(
        muscriptor_root,
        "muscriptor-run",
        "muscriptor-model",
        "f" * 64,
        "muscriptor-map",
        precision="0.8",
        item_ids=(1, 2),
        failed_item_ids=frozenset({2}),
    )

    import src.benchmark.muscriptor_comparison as comparison

    subset_sha = "a" * 64
    subset_corpus_version = "hpa327-v1"
    subset_policy_version = "crux.review-policy/v1"
    subset_ledger_sha = "b" * 64
    monkeypatch.setattr(
        comparison,
        "load_reviewed_subset_manifest",
        lambda _path: SimpleNamespace(
            manifest_sha256=subset_sha,
            corpus_version=subset_corpus_version,
            review_policy_version=subset_policy_version,
            review_ledger_sha256=subset_ledger_sha,
            source_reference_manifest_sha256="a" * 64,
            source_reference_manifest_version="hpa324-v1",
            source_timing_manifest_sha256="b" * 64,
            source_timing_manifest_version="hpa323-v1",
            rows=(SimpleNamespace(view=SimpleNamespace(simfile_id=1)),),
        ),
    )
    result = compare_oaf_muscriptor(
        _request(tmp_path, oaf, muscriptor, subset=tmp_path / "subset.json")
    )

    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    subset_manifest = summary["subset_manifest"]
    assert subset_manifest["path"] == str(tmp_path / "subset.json")
    assert subset_manifest["manifest_sha256"] == subset_sha
    assert subset_manifest["corpus_version"] == subset_corpus_version
    assert subset_manifest["review_policy_version"] == subset_policy_version
    assert subset_manifest["review_ledger_sha256"] == subset_ledger_sha
