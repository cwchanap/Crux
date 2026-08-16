from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
) -> bytes:
    snapshot = {
        "schema": schema,
        "run_id": run_id,
        "reference_manifest_sha256": "a" * 64,
        "reference_manifest_version": "hpa324-v1",
        "reference_timing_manifest_sha256": "b" * 64,
        "reference_timing_version": "hpa323-v1",
        "model_id": model_id,
        "model_lock_sha256": model_lock,
        "prediction_map_version": prediction_map,
        "input_view_id": input_view,
        "items": [
            {
                "simfile_id": 1,
                "execution_disposition": disposition,
                "source_audio_sha256": source_audio,
                "input_audio_sha256": input_audio,
            }
        ],
        "overall_status": "complete",
        "success_count": 1 if disposition == "inferred" else 0,
        "failed_count": 1 if disposition == "failed" else 0,
        "skipped_count": 1 if disposition == "skipped" else 0,
        "quarantined_count": 1 if disposition == "quarantined" else 0,
    }
    if schema == OAF_CORPUS_RUN_SCHEMA:
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
    **kwargs: str,
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
    root: Path, cohort: str, model: str, lock: str, prediction_map: str, *, precision: str
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
                "simfile_id": "1",
                "status": "success",
                "reference_native_event_count": "1",
                "reference_common_event_count": "1",
                "reference_ignored_event_count": "0",
                "reference_unmapped_event_count": "0",
                "reference_duplicate_collapsed_count": "0",
            }
        ],
    )
    _write_csv(
        reports / "per_song.csv",
        _SONG_FIELDS,
        [
            {
                **identity,
                "simfile_id": "1",
                "tolerance_ms": "50",
                "mode": "raw",
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
        ],
    )
    _write_csv(
        reports / "per_class.csv",
        _CLASS_FIELDS,
        [
            {
                **identity,
                "simfile_id": "1",
                "tolerance_ms": "50",
                "mode": "raw",
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
        ],
    )


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
    assert summary["pairing"]["pairable_success_intersection"] == 1
    assert summary["aggregates"]["song"][0]["mean_delta_precision"] == 0.3
    assert "summary.md" in {path.name for path in result.output_dir.iterdir()}


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
