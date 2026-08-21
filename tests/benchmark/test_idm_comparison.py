from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.cohort_scoring import SCORING_VERSION
from src.benchmark.idm_comparison import (
    ComparisonIntegrityError,
    IdmComparisonRequest,
    compare_oaf_idm,
)
from src.benchmark.idm_pilot_run import render_idm_pilot_run
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION

_REPORT_IDENTITY = (
    "cohort_id",
    "model_id",
    "model_lock_sha256",
    "prediction_map_version",
    "input_view_id",
    "scoring_version",
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


def _snapshot(
    *,
    input_audio_sha256: str = "c" * 64,
    input_view_id: str = "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
    oaf_input_audio_sha256: str | None = None,
) -> bytes:
    items = [
        {
            "simfile_id": 1,
            "source_audio_id": "song-1",
            "source_audio_sha256": "d" * 64,
            "input_audio_sha256": input_audio_sha256,
            "input_view_id": input_view_id,
            "execution_disposition": "inferred",
            "prediction_path": "predictions/1.jsonl",
            "prediction_artifact_sha256": "e" * 64,
            "wall_time_sec": 2.5,
            "rtf": 0.25,
        },
        {
            "simfile_id": 2,
            "source_audio_id": "song-2",
            "source_audio_sha256": "f" * 64,
            "input_audio_sha256": input_audio_sha256,
            "input_view_id": input_view_id,
            "execution_disposition": "failed",
            "native_failure_code": "worker_protocol_failed",
            "cohort_failure_reason": "backend_unavailable",
        },
    ]
    if oaf_input_audio_sha256 is not None:
        for item in items:
            item["oaf_input_audio_sha256"] = oaf_input_audio_sha256
    return render_idm_pilot_run(
        {
            "schema": "crux.idm-stem-pilot-run/v1",
            "run_id": "idm-test-run",
            "reference_manifest_sha256": "a" * 64,
            "reference_manifest_version": "hpa324-v1",
            "reference_timing_manifest_sha256": "b" * 64,
            "reference_timing_version": "hpa323-v1",
            "backend_descriptor_sha256": "1" * 64,
            "backend_descriptor": {"backend_id": "idm-44-train-kits-v1"},
            "model_id": "idm-44-train-kits-model",
            "model_lock_sha256": "2" * 64,
            "prediction_map_version": "crux.prediction-map/idm-44-train-kits-v1",
            "input_view_id": input_view_id,
            "oaf_model_id": "oaf-model",
            "oaf_model_lock_sha256": "3" * 64,
            "oaf_backend_descriptor_sha256": "4" * 64,
            "oaf_prediction_map_version": "crux.prediction-map/oaf-v1",
            "native_failure_counts": {"worker_protocol_failed": 1},
            "aggregate_rtf": 0.125,
            "peak_process_rss_bytes": 1234,
            "items": items,
            "success_count": 1,
            "failed_count": 1,
            "skipped_count": 0,
            "quarantined_count": 0,
            "overall_status": "partial",
        }
    )


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(root: Path, *, label: str, failed: bool = True) -> None:
    cohort_id = f"idm-test-run:{label}"
    model_id = "oaf-model" if label == "oaf" else "idm-44-train-kits-model"
    model_lock = "3" * 64 if label == "oaf" else "2" * 64
    prediction_map = (
        "crux.prediction-map/oaf-v1"
        if label == "oaf"
        else "crux.prediction-map/idm-44-train-kits-v1"
    )
    identity = {
        "cohort_id": cohort_id,
        "model_id": model_id,
        "model_lock_sha256": model_lock,
        "prediction_map_version": prediction_map,
        "input_view_id": "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
        "scoring_version": SCORING_VERSION,
    }
    statuses = {"1": "success", "2": "failed" if failed else "success"}
    _write_csv(
        root / "items.csv",
        _ITEM_FIELDS,
        [
            {
                **{field: "" for field in _ITEM_FIELDS},
                "cohort_id": cohort_id,
                "simfile_id": simfile_id,
                "status": status,
                "failure_reason": "backend_unavailable" if status == "failed" else "",
                "reference_native_event_count": "1",
                "reference_common_event_count": "1",
                "reference_ignored_event_count": "0",
                "reference_unmapped_event_count": "0",
                "reference_duplicate_collapsed_count": "0",
                "prediction_native_event_count": "2" if status == "success" else "",
                "prediction_mapped_event_count": "1" if status == "success" else "",
                "prediction_unmapped_event_count": "1" if status == "success" else "",
                "prediction_mapping_coverage": "0.5" if status == "success" else "",
                "prediction_native_class_counts": "KD=1|TT_LMT=1" if status == "success" else "",
            }
            for simfile_id, status in statuses.items()
        ],
    )
    score_rows = []
    class_rows = []
    for simfile_id, status in statuses.items():
        if status != "success":
            continue
        for mode in ("raw", "aligned"):
            score_rows.append(
                {
                    **identity,
                    "simfile_id": simfile_id,
                    "tolerance_ms": "50",
                    "mode": mode,
                    "tp": "1",
                    "fp": "0",
                    "fn": "0",
                    "precision": "1",
                    "recall": "1",
                    "f1": "1",
                    "prediction_to_reference_ratio": "2",
                    "median_abs_error_ms": "0",
                    "p95_abs_error_ms": "0",
                    "offset_ms": "0",
                    "warnings": "",
                }
            )
            class_rows.append(
                {
                    **identity,
                    "simfile_id": simfile_id,
                    "tolerance_ms": "50",
                    "mode": mode,
                    "common_class": "kick",
                    "tp": "1",
                    "fp": "0",
                    "fn": "0",
                    "reference_support": "1",
                    "prediction_support": "1",
                    "precision": "1",
                    "recall": "1",
                    "f1": "1",
                }
            )
    _write_csv(root / "per_song.csv", _SONG_FIELDS, score_rows)
    _write_csv(root / "per_class.csv", _CLASS_FIELDS, class_rows)
    metric = {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "precision": 1,
        "recall": 1,
        "f1": 1,
    }
    aggregate = {
        "tolerance_ms": 50,
        "mode": "raw",
        "event_micro": metric,
        "song_macro_f1": 1,
        "class_macro_f1": 1,
        "song_f1_distribution": {
            field: 1 for field in ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")
        },
        "per_class": [
            {
                "common_class": "kick",
                **metric,
                "reference_support": 1,
                "prediction_support": 1,
            }
        ],
        "successful_song_count": 1,
    }
    summary = {
        "schema": "crux.single-cohort-report/v1",
        "identity": {
            "reference_manifest_sha256": "a" * 64,
            "reference_timing_version": "hpa323-v1",
            "taxonomy_version": TAXONOMY_VERSION,
            "lane_map_version": DTX_LANE_MAP_VERSION,
            "backend_id": (
                "magenta-egmd-tf1-94529798-8hit-v1" if label == "oaf" else "idm-44-train-kits-v1"
            ),
            "backend_descriptor_sha256": "4" * 64 if label == "oaf" else "1" * 64,
            **identity,
        },
        "tolerances_ms": [50],
        "population": {
            "total_count": 2,
            "success_count": 1,
            "failed_count": 1,
            "skipped_count": 0,
            "quarantined_count": 0,
            "reason_counts": {"backend_unavailable": 1},
        },
        "aggregates": [aggregate, {**aggregate, "mode": "aligned"}],
    }
    (root / "summary.json").write_bytes(canonical_json_bytes(summary))


@pytest.fixture
def idm_run(tmp_path: Path) -> Path:
    run_path = tmp_path / "run" / "run.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_bytes(_snapshot())
    _write_reports(run_path.parent / "reports" / "oaf", label="oaf")
    _write_reports(run_path.parent / "reports" / "idm", label="idm")
    return run_path


def test_compare_oaf_idm_publishes_deterministic_pair_and_diagnostics(
    idm_run: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "comparison"
    request = IdmComparisonRequest(run_path=idm_run, output_dir=output_dir)

    result = compare_oaf_idm(request)

    assert result == output_dir
    assert {path.name for path in output_dir.iterdir()} == {
        "summary.json",
        "summary.md",
        "paired_per_song.csv",
        "paired_per_class.csv",
    }
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["pairing"]["pairable_success_intersection"] == 1
    assert summary["models"]["oaf"]["population"]["failed_count"] == 1
    assert summary["models"]["idm"]["population"]["success_count"] == 1
    assert summary["models"]["idm"]["native_failure_histogram"] == {"worker_protocol_failed": 1}
    assert summary["models"]["idm"]["coverage"]["prediction_mapped_event_count"] == 1
    assert summary["models"]["idm"]["coverage"]["prediction_unmapped_event_count"] == 1
    assert summary["models"]["idm"]["runtime"]["aggregate_rtf"] == 0.125
    first_outputs = {
        name: (output_dir / name).read_bytes()
        for name in ("summary.json", "summary.md", "paired_per_song.csv", "paired_per_class.csv")
    }

    compare_oaf_idm(request)

    assert first_outputs == {
        name: (output_dir / name).read_bytes()
        for name in ("summary.json", "summary.md", "paired_per_song.csv", "paired_per_class.csv")
    }
    with (output_dir / "paired_per_song.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["idm_f1"] == "1"


def test_compare_oaf_idm_rejects_stem_input_view_mismatch(idm_run: Path, tmp_path: Path) -> None:
    idm_run.write_bytes(_snapshot(input_view_id="crux.full-mix/v1"))

    with pytest.raises(ComparisonIntegrityError, match="frozen stem input view"):
        compare_oaf_idm(IdmComparisonRequest(run_path=idm_run, output_dir=tmp_path / "comparison"))


def test_compare_oaf_idm_rejects_identical_input_hash_mismatch(
    idm_run: Path, tmp_path: Path
) -> None:
    idm_run.write_bytes(_snapshot(oaf_input_audio_sha256="9" * 64))

    with pytest.raises(ComparisonIntegrityError, match="canonical-input.*input_audio_sha256"):
        compare_oaf_idm(IdmComparisonRequest(run_path=idm_run, output_dir=tmp_path / "comparison"))
