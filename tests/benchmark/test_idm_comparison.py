from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.benchmark.backend_identity import build_descriptor, canonical_json_bytes, sha256_hex
from src.benchmark.backends import CanonicalAudio, NativeEvent
from src.benchmark.cohort_scoring import SCORING_VERSION
from src.benchmark.idm_comparison import (
    ComparisonIntegrityError,
    IdmComparisonRequest,
    compare_oaf_idm,
)
from src.benchmark.idm_pilot_run import build_run_id, render_idm_pilot_run
from src.benchmark.prediction_artifact import (
    MappedPrediction,
    MappedPredictionEvent,
    render_prediction_artifact,
)
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

_HANDOFF_SHA = "5" * 64
_HANDOFF_VERSION = "sha256:" + "6" * 64
_REFERENCE_SHA = "a" * 64
_REFERENCE_VERSION = "sha256:" + "7" * 64
_TIMING_SHA = "b" * 64
_TIMING_VERSION = "sha256:" + "8" * 64
_DESCRIPTOR_SHA = "1" * 64
_MODEL_LOCK_SHA = "2" * 64
_INFERENCE_CONFIG_SHA = "9" * 64
_BASE_RUN_ID = build_run_id(
    _HANDOFF_SHA,
    _HANDOFF_VERSION,
    _REFERENCE_SHA,
    _REFERENCE_VERSION,
    _TIMING_SHA,
    _TIMING_VERSION,
    _DESCRIPTOR_SHA,
    _MODEL_LOCK_SHA,
    _INFERENCE_CONFIG_SHA,
    "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
)


def _snapshot(
    *,
    source_audio_sha256: str | None = "d" * 64,
    input_audio_sha256: str = "c" * 64,
    input_view_id: str = "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
    oaf_input_audio_sha256: str | None = None,
    reference_manifest_sha256: str = _REFERENCE_SHA,
    reference_timing_manifest_sha256: str = _TIMING_SHA,
    run_id: str = _BASE_RUN_ID,
    native_failure_counts: dict[str, int] | None = None,
) -> bytes:
    items = [
        {
            "simfile_id": 1,
            "source_audio_id": "song-1",
            "source_audio_sha256": source_audio_sha256,
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
            "run_id": run_id,
            "handoff_manifest_sha256": _HANDOFF_SHA,
            "handoff_manifest_version": _HANDOFF_VERSION,
            "reference_manifest_sha256": reference_manifest_sha256,
            "reference_manifest_version": _REFERENCE_VERSION,
            "reference_timing_manifest_sha256": reference_timing_manifest_sha256,
            "reference_timing_version": _TIMING_VERSION,
            "backend_descriptor_sha256": _DESCRIPTOR_SHA,
            "backend_descriptor": {"backend_id": "idm-44-train-kits-v1"},
            "model_id": "idm-44-train-kits-model",
            "model_lock_sha256": _MODEL_LOCK_SHA,
            "inference_config_sha256": _INFERENCE_CONFIG_SHA,
            "prediction_map_version": "crux.prediction-map/idm-44-train-kits-v1",
            "input_view_id": input_view_id,
            "oaf_model_id": "oaf-model",
            "oaf_model_lock_sha256": "3" * 64,
            "oaf_backend_descriptor_sha256": "4" * 64,
            "oaf_prediction_map_version": "crux.prediction-map/oaf-v1",
            "native_failure_counts": (
                {"worker_protocol_failed": 1}
                if native_failure_counts is None
                else native_failure_counts
            ),
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


def _write_reports(
    root: Path, *, label: str, failed: bool = True, run_id: str = _BASE_RUN_ID
) -> None:
    cohort_id = f"{run_id}:{label}"
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
            "reference_manifest_sha256": _REFERENCE_SHA,
            "reference_timing_version": _TIMING_VERSION,
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


def _idm_prediction_bytes(
    *,
    source_audio_id: str = "song-1",
    source_audio_sha256: str = "d" * 64,
    input_audio_sha256: str = "c" * 64,
    native_velocity: str = "1.337421",
) -> bytes:
    descriptor_payload = {
        "architecture_id": "inverse-drum-machine-v0.1.0",
        "backend_id": "idm-44-train-kits-v1",
        "descriptor_schema": "crux.transcription-backend-descriptor/v2",
        "model_id": "idm-44-train-kits-0123456789ab-fedcba987654",
        "native_metadata_schema_id": "idm-peak-event-metadata-v1",
        "native_output_space_id": "idm-44-train-kits-9class-v1",
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": "idm-training-contract-44-train-kits-v1",
        "upstream_source_commit": "456656868538205ef756912c7cf5b0fd936de8af",
    }
    prediction = MappedPrediction(
        audio=CanonicalAudio(
            path=Path(),
            source_audio_id=source_audio_id,
            source_audio_sha256=source_audio_sha256,
            input_view_id="crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
            input_audio_sha256=input_audio_sha256,
            byte_length=46,
            sample_rate=44100,
            channel_count=1,
            sample_width_bytes=2,
            audio_frame_count=1,
        ),
        descriptor=build_descriptor(
            descriptor_payload,
            frozenset(descriptor_payload),
            "crux.transcription-backend-descriptor/v2",
        ),
        events=(
            MappedPredictionEvent(
                native=NativeEvent(
                    time_sec=1.25,
                    native_class_id="KD",
                    model_output_bin=4,
                    native_midi_note=None,
                    native_metadata={
                        "frame_index": "215",
                        "native_velocity": native_velocity,
                    },
                    confidence=0.83,
                    velocity_midi=85,
                ),
                canonical_class="kick",
                common_class="kick",
                mapping_status="mapped",
                prediction_map_version="crux.prediction-map/idm-44-train-kits-v1",
            ),
        ),
    )
    return render_prediction_artifact(prediction)


@pytest.fixture
def idm_run(tmp_path: Path) -> Path:
    run_path = tmp_path / "output" / "runs" / _BASE_RUN_ID / "run.json"
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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_audio_sha256", None),
        ("source_audio_sha256", "not-a-sha"),
        ("input_audio_sha256", None),
        ("input_audio_sha256", "not-a-sha"),
    ),
)
def test_compare_oaf_idm_rejects_missing_or_malformed_success_hash(
    idm_run: Path, tmp_path: Path, field: str, value: object
) -> None:
    idm_run.write_bytes(_snapshot(**{field: value}))

    with pytest.raises(
        ComparisonIntegrityError,
        match=f"successful .*{field}",
    ):
        compare_oaf_idm(IdmComparisonRequest(run_path=idm_run, output_dir=tmp_path / "comparison"))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reference_manifest_sha256", "0" * 64),
        ("reference_timing_manifest_sha256", "0" * 64),
    ),
)
def test_compare_oaf_idm_rejects_mutated_lineage_with_stale_run_id(
    idm_run: Path, tmp_path: Path, field: str, value: str
) -> None:
    idm_run.write_bytes(_snapshot(run_id=_BASE_RUN_ID, **{field: value}))

    with pytest.raises(ComparisonIntegrityError, match="run_id identity"):
        compare_oaf_idm(IdmComparisonRequest(run_path=idm_run, output_dir=tmp_path / "comparison"))


def test_compare_oaf_idm_rejects_inconsistent_native_failure_histogram(
    idm_run: Path, tmp_path: Path
) -> None:
    idm_run.write_bytes(_snapshot(native_failure_counts={"worker_protocol_failed": 2}))

    with pytest.raises(ComparisonIntegrityError, match="native_failure_counts.*failed"):
        compare_oaf_idm(IdmComparisonRequest(run_path=idm_run, output_dir=tmp_path / "comparison"))


def test_compare_oaf_idm_derives_native_failure_histogram_when_unstored(
    idm_run: Path, tmp_path: Path
) -> None:
    snapshot = json.loads(_snapshot().decode("utf-8"))
    del snapshot["native_failure_counts"]
    idm_run.write_bytes(render_idm_pilot_run(snapshot))

    compare_oaf_idm(IdmComparisonRequest(run_path=idm_run, output_dir=tmp_path / "comparison"))

    summary = json.loads((tmp_path / "comparison" / "summary.json").read_text(encoding="utf-8"))
    assert summary["models"]["idm"]["native_failure_histogram"] == {"worker_protocol_failed": 1}


def _set_prediction_artifact_sha(run_path: Path, artifact: bytes, *, source_audio_id: str) -> None:
    snapshot = json.loads(run_path.read_text(encoding="utf-8"))
    row = snapshot["items"][0]
    row["source_audio_id"] = source_audio_id
    row["prediction_artifact_sha256"] = sha256_hex(artifact)
    run_path.write_bytes(render_idm_pilot_run(snapshot))


def test_compare_oaf_idm_reports_velocity_from_task4_output_root(
    idm_run: Path, tmp_path: Path
) -> None:
    artifact = _idm_prediction_bytes()
    _set_prediction_artifact_sha(idm_run, artifact, source_audio_id="song-1")
    target = idm_run.parents[2] / "predictions" / "1.jsonl"
    target.parent.mkdir(parents=True)
    target.write_bytes(artifact)

    compare_oaf_idm(IdmComparisonRequest(run_path=idm_run, output_dir=tmp_path / "comparison"))

    summary = json.loads((tmp_path / "comparison" / "summary.json").read_text(encoding="utf-8"))
    velocity = summary["models"]["idm"]["velocity"]
    assert velocity["available_event_count"] == 1
    assert velocity["native_velocity_distribution"]["median"] == 1.337421


def test_compare_oaf_idm_ignores_prediction_shadow_under_run_directory(
    idm_run: Path, tmp_path: Path
) -> None:
    artifact = _idm_prediction_bytes()
    _set_prediction_artifact_sha(idm_run, artifact, source_audio_id="song-1")
    target = idm_run.parent / "predictions" / "1.jsonl"
    target.parent.mkdir(parents=True)
    target.write_bytes(artifact)

    compare_oaf_idm(IdmComparisonRequest(run_path=idm_run, output_dir=tmp_path / "comparison"))

    summary = json.loads((tmp_path / "comparison" / "summary.json").read_text(encoding="utf-8"))
    assert summary["models"]["idm"]["velocity"]["available_event_count"] == 0


def test_compare_oaf_idm_rejects_prediction_artifact_wrong_for_run_row(
    idm_run: Path, tmp_path: Path
) -> None:
    artifact = _idm_prediction_bytes(source_audio_id="different-song")
    target = idm_run.parents[2] / "predictions" / "1.jsonl"
    target.parent.mkdir(parents=True)
    target.write_bytes(artifact)

    compare_oaf_idm(IdmComparisonRequest(run_path=idm_run, output_dir=tmp_path / "comparison"))

    summary = json.loads((tmp_path / "comparison" / "summary.json").read_text(encoding="utf-8"))
    assert summary["models"]["idm"]["velocity"]["available_event_count"] == 0
