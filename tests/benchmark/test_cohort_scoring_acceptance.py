from __future__ import annotations

import builtins
import csv
import dataclasses
import struct
import sys
from pathlib import Path

import pytest

from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    OAF_DESCRIPTOR_SCHEMA,
    build_descriptor,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backends import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.cohort_scoring import (
    CohortIdentity,
    CohortItem,
    CohortScoreResult,
    coverage_from_artifacts,
    score_cohort,
)
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.prediction_artifact import (
    PredictionArtifact,
    read_prediction_artifact,
    render_prediction_artifact,
)
from src.benchmark.reference_set import ReferenceMappingResult, map_reference_events
from src.benchmark.reference_timing import NativeReferenceEvent
from src.benchmark.reports import write_cohort_reports
from src.benchmark.scorer_input import (
    prediction_to_benchmark_events,
    reference_to_benchmark_events,
)
from src.benchmark.taxonomy import (
    DTX_LANE_MAP_VERSION,
    TAXONOMY_VERSION,
)

_FORBIDDEN_RUNTIME_MODULES = (
    "src.benchmark.backends.oaf",
    "src.benchmark.worker_process",
    "src.worker",
)


def _descriptor():
    payload = {
        "architecture_id": "magenta-oaf-model-tpu-drums-v1",
        "backend_id": OAF_BACKEND_ID,
        "descriptor_schema": OAF_DESCRIPTOR_SCHEMA,
        "model_id": "magenta-egmd-ckpt-569400-v1",
        "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
        "native_output_space_id": "magenta-oaf-midi88-a0-v1",
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
        "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
    }
    return build_descriptor(payload, frozenset(payload), OAF_DESCRIPTOR_SCHEMA)


def _native_reference(lane_id: str, time_sec: float, source_order: int) -> NativeReferenceEvent:
    return NativeReferenceEvent(
        simfile_id=7,
        selected_chart_key="7/chart.dtx",
        selected_chart_content_hash="a" * 64,
        source_audio_key="7/audio.wav",
        source_audio_content_hash="b" * 64,
        source_order=source_order,
        measure=1,
        position=float(source_order),
        lane_id=lane_id,
        note_id=f"{lane_id}-{source_order}",
        chart_time_sec=time_sec,
        audio_time_sec=time_sec,
    )


def _reference_mapping() -> ReferenceMappingResult:
    return map_reference_events(
        (
            _native_reference("14", 1.0, 0),
            _native_reference("15", 1.0, 1),
            _native_reference("13", 2.0, 2),
        )
    )


def _audio(tmp_path: Path) -> CanonicalAudio:
    content = (
        struct.pack("<4sI4s", b"RIFF", 40, b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16)
        + struct.pack("<4sI", b"data", 4)
        + b"\x00\x00\x00\x00"
    )
    digest = sha256_hex(content)
    path = tmp_path / "song.wav"
    path.write_bytes(content)
    return CanonicalAudio(
        path=path,
        source_audio_id="7",
        source_audio_sha256=digest,
        input_view_id="full-mix-v1",
        input_audio_sha256=digest,
        byte_length=len(content),
        sample_rate=44100,
        channel_count=1,
        sample_width_bytes=2,
        audio_frame_count=2,
    )


def _prediction_artifact(tmp_path: Path) -> PredictionArtifact:
    native_events = (
        NativeEvent(
            time_sec=1.0,
            native_class_id="midi_48",
            model_output_bin=27,
            native_midi_note=48,
            native_metadata={"upstream_8hit_group_id": "toms"},
            confidence=0.9,
            velocity_midi=100,
        ),
        NativeEvent(
            time_sec=2.0,
            native_class_id="midi_36",
            model_output_bin=15,
            native_midi_note=36,
            native_metadata={"upstream_8hit_group_id": "kick"},
            confidence=0.8,
            velocity_midi=101,
        ),
        NativeEvent(
            time_sec=3.0,
            native_class_id="midi_75",
            model_output_bin=54,
            native_midi_note=75,
            native_metadata={"upstream_8hit_group_id": "sticks"},
            confidence=0.7,
            velocity_midi=102,
        ),
    )
    mapped, diagnostics = map_oaf_prediction(
        NativePrediction(_audio(tmp_path), _descriptor(), native_events)
    )
    assert diagnostics.unmapped == {"sticks": 1}
    path = tmp_path / "prediction.jsonl"
    path.write_bytes(render_prediction_artifact(mapped))
    return read_prediction_artifact(path.read_bytes())


def _identity(artifact: PredictionArtifact) -> CohortIdentity:
    descriptor = artifact.prediction.descriptor
    prediction_map_version = artifact.prediction.events[0].prediction_map_version
    return CohortIdentity(
        cohort_id="oaf-hpa-325-acceptance-v1",
        reference_manifest_sha256="c" * 64,
        reference_timing_version="crux.reference-timing/v1",
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=descriptor.payload["backend_id"],
        model_id=descriptor.payload["model_id"],
        model_lock_sha256="d" * 64,
        backend_descriptor_sha256=descriptor.sha256,
        prediction_map_version=prediction_map_version,
        input_view_id=artifact.prediction.audio.input_view_id,
    )


def _item(
    reference: ReferenceMappingResult,
    artifact: PredictionArtifact,
    identity: CohortIdentity,
) -> CohortItem:
    reference_events = reference_to_benchmark_events("7", reference.common_events)
    prediction_events = prediction_to_benchmark_events(artifact)
    assert all(
        event.metadata["input_view_id"] == identity.input_view_id
        and event.metadata["prediction_map_version"] == identity.prediction_map_version
        for event in prediction_events
    )
    return CohortItem(
        simfile_id="7",
        status="success",
        reference_events=reference_events,
        prediction_events=prediction_events,
        coverage=coverage_from_artifacts(reference, artifact),
    )


def _forbidden_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if any(
            name == module or name.startswith(module + ".") for module in _FORBIDDEN_RUNTIME_MODULES
        )
    }


def _install_no_inference_import_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        del globals, locals
        if any(
            name == module or name.startswith(module + ".") for module in _FORBIDDEN_RUNTIME_MODULES
        ):
            raise AssertionError(f"inference runtime import attempted: {name}")
        if name == "src.benchmark.backends" and "oaf" in fromlist:
            raise AssertionError("inference runtime import attempted: src.benchmark.backends.oaf")
        if name == "src.benchmark" and "worker_process" in fromlist:
            raise AssertionError("inference runtime import attempted: src.benchmark.worker_process")
        return original_import(name, fromlist=fromlist, level=level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _assert_reports(
    result: CohortScoreResult,
    report_dir: Path,
    identity: CohortIdentity,
) -> None:
    from src.benchmark.reports import ReportArtifacts

    artifacts = write_cohort_reports(result, report_dir)
    assert isinstance(artifacts, ReportArtifacts)

    summary = strict_json_loads(artifacts.summary_json.read_bytes(), require_canonical=True)
    assert set(summary) == {"schema", "identity", "tolerances_ms", "population", "aggregates"}
    assert "items" not in summary
    assert summary["identity"]["cohort_id"] == identity.cohort_id
    assert summary["population"] == {
        "failed_count": 0,
        "quarantined_count": 0,
        "reason_counts": {},
        "skipped_count": 0,
        "success_count": 1,
        "total_count": 1,
    }

    with artifacts.items_csv.open(newline="", encoding="utf-8") as handle:
        item_rows = list(csv.DictReader(handle))
    assert len(item_rows) == 1
    item_row = item_rows[0]
    assert item_row["reference_duplicate_collapsed_count"] == "1"
    assert item_row["reference_native_event_count"] == "3"
    assert item_row["reference_common_event_count"] == "2"
    assert item_row["prediction_native_event_count"] == "3"
    assert item_row["prediction_mapped_event_count"] == "2"
    assert item_row["prediction_unmapped_event_count"] == "1"
    assert item_row["prediction_mapping_coverage"] == "0.666667"

    with artifacts.per_song_csv.open(newline="", encoding="utf-8") as handle:
        song_rows = list(csv.DictReader(handle))
    assert len(song_rows) == len(result.tolerances_ms) * 2
    assert {row["simfile_id"] for row in song_rows} == {"7"}
    assert {
        "cohort_id",
        "model_id",
        "model_lock_sha256",
        "prediction_map_version",
        "input_view_id",
        "scoring_version",
    } <= song_rows[0].keys()
    assert all("backend_descriptor_sha256" not in row for row in song_rows)

    with artifacts.per_class_csv.open(newline="", encoding="utf-8") as handle:
        class_rows = list(csv.DictReader(handle))
    assert len(class_rows) == len(result.tolerances_ms) * 2 * 2
    assert {row["simfile_id"] for row in class_rows} == {"7"}
    assert all("scope" not in row for row in class_rows)
    assert all(row["tolerance_ms"] in {"30", "50", "100"} for row in class_rows)

    diagnostic_lines = artifacts.event_diagnostics_jsonl.read_bytes().splitlines()
    assert len(diagnostic_lines) == len(result.event_diagnostics)
    assert diagnostic_lines
    expected_diagnostic_keys = {
        "cohort_id",
        "simfile_id",
        "tolerance_ms",
        "mode",
        "outcome",
        "common_class",
        "reference_time_sec",
        "prediction_time_sec",
        "scored_prediction_time_sec",
        "timing_error_sec",
    }
    for line in diagnostic_lines:
        diagnostic = strict_json_loads(line, require_canonical=True)
        assert set(diagnostic) == expected_diagnostic_keys
        assert diagnostic["cohort_id"] == identity.cohort_id
        assert diagnostic["simfile_id"] == "7"


def test_persisted_artifacts_rescore_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_runtime_modules = _forbidden_modules()
    _install_no_inference_import_guard(monkeypatch)

    reference = _reference_mapping()
    assert len(reference.mapped_events) == 3
    assert len(reference.common_events) == 2
    assert reference.diagnostics.duplicate_common_event_count == 1

    reference_events = reference_to_benchmark_events("7", reference.common_events)
    assert [(event.canonical_class, event.time_sec) for event in reference_events] == [
        ("tom", 1.0),
        ("kick", 2.0),
    ]

    artifact = _prediction_artifact(tmp_path)
    prediction_events = prediction_to_benchmark_events(artifact)
    assert [(event.canonical_class, event.time_sec) for event in prediction_events] == [
        ("tom", 1.0),
        ("kick", 2.0),
    ]

    coverage = coverage_from_artifacts(reference, artifact)
    assert coverage.reference_duplicate_collapsed_count == 1
    assert coverage.reference_native_event_count == (
        coverage.reference_common_event_count
        + coverage.reference_duplicate_collapsed_count
        + coverage.reference_ignored_event_count
        + coverage.reference_unmapped_event_count
    )
    assert coverage.prediction_native_event_count == (
        coverage.prediction_mapped_event_count + coverage.prediction_unmapped_event_count
    )
    assert coverage.prediction_native_event_count == 3
    assert coverage.prediction_mapped_event_count == 2
    assert coverage.prediction_unmapped_event_count == 1

    identity = _identity(artifact)
    item = _item(reference, artifact, identity)
    selected = score_cohort(identity, (item,), diagnostics_for=("7",))
    assert selected.event_diagnostics
    assert {row.simfile_id for row in selected.event_diagnostics} == {"7"}

    default = score_cohort(identity, (item,))
    assert default.event_diagnostics == ()
    assert selected.aggregates == default.aggregates
    assert selected.song_scores == default.song_scores
    _assert_reports(selected, tmp_path / "reports", identity)

    changed_map = identity.prediction_map_version + "/rescored"
    changed_identity = dataclasses.replace(identity, prediction_map_version=changed_map)
    changed_events = tuple(
        dataclasses.replace(
            event,
            metadata={**event.metadata, "prediction_map_version": changed_map},
        )
        for event in item.prediction_events or ()
    )
    rescored = score_cohort(
        changed_identity,
        (dataclasses.replace(item, prediction_events=changed_events),),
    )
    assert rescored.event_diagnostics == ()
    assert rescored.aggregates == default.aggregates
    assert rescored.song_scores == default.song_scores

    assert _forbidden_modules() == before_runtime_modules
