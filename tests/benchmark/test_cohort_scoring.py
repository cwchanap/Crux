from __future__ import annotations

import dataclasses
import struct
from pathlib import Path
from typing import get_args

import pytest

from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    OAF_DESCRIPTOR_SCHEMA,
    build_descriptor,
    sha256_hex,
)
from src.benchmark.backends import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.cohort_scoring import (
    COHORT_FAILURE_REASONS,
    DEFAULT_TOLERANCES_MS,
    SCORING_VERSION,
    CohortCoverage,
    CohortIdentity,
    CohortItem,
    _score_success_items,
    coverage_from_artifacts,
    validate_cohort_items,
)
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.models import BenchmarkEvent
from src.benchmark.prediction_artifact import (
    PredictionArtifact,
    read_prediction_artifact,
    render_prediction_artifact,
)
from src.benchmark.reference_set import map_reference_events
from src.benchmark.reference_timing import NativeReferenceEvent
from src.benchmark.scorer_input import (
    prediction_to_benchmark_events,
    reference_to_benchmark_events,
)
from src.benchmark.taxonomy import (
    DTX_LANE_MAP_VERSION,
    OAF_PREDICTION_MAP_ID,
    TAXONOMY_VERSION,
)


def identity() -> CohortIdentity:
    return CohortIdentity(
        cohort_id="oaf-full-mix-v1",
        reference_manifest_sha256="a" * 64,
        reference_timing_version="sha256:" + "b" * 64,
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=OAF_BACKEND_ID,
        model_id="magenta-egmd-ckpt-569400-v1",
        model_lock_sha256="c" * 64,
        backend_descriptor_sha256="d" * 64,
        prediction_map_version=OAF_PREDICTION_MAP_ID,
        input_view_id="full-mix-v1",
    )


def native_reference(
    lane_id: str, audio_time_sec: float, source_order: int
) -> NativeReferenceEvent:
    return NativeReferenceEvent(
        simfile_id=42,
        selected_chart_key="42/chart.dtx",
        selected_chart_content_hash="e" * 64,
        source_audio_key="42/audio.wav",
        source_audio_content_hash="f" * 64,
        source_order=source_order,
        measure=1,
        position=float(source_order),
        lane_id=lane_id,
        note_id=f"{lane_id}-{source_order}",
        chart_time_sec=audio_time_sec,
        audio_time_sec=audio_time_sec,
    )


def reference_mapping():
    return map_reference_events(
        (
            native_reference("14", 1.0, 0),
            native_reference("15", 1.0, 1),
            native_reference("13", 2.0, 2),
            native_reference("54", 3.0, 3),
        )
    )


def descriptor():
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


def prediction_artifact(tmp_path: Path) -> PredictionArtifact:
    audio_content = (
        struct.pack("<4sI4s", b"RIFF", 40, b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16)
        + struct.pack("<4sI", b"data", 4)
        + b"\x00\x00\x00\x00"
    )
    digest = sha256_hex(audio_content)
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(audio_content)
    audio = CanonicalAudio(
        audio_path,
        "song",
        digest,
        "full-mix-v1",
        digest,
        len(audio_content),
        44100,
        1,
        2,
        2,
    )
    native_events = (
        NativeEvent(
            time_sec=1.0,
            native_class_id="midi_46",
            model_output_bin=25,
            native_midi_note=46,
            native_metadata={"upstream_8hit_group_id": "hihat"},
            confidence=0.8,
            velocity_midi=96,
        ),
        NativeEvent(
            time_sec=1.5,
            native_class_id="midi_42",
            model_output_bin=21,
            native_midi_note=42,
            native_metadata={"upstream_8hit_group_id": "sticks"},
            confidence=0.7,
            velocity_midi=95,
        ),
    )
    mapped, _ = map_oaf_prediction(NativePrediction(audio, descriptor(), native_events))
    return read_prediction_artifact(render_prediction_artifact(mapped))


def no_prediction_coverage() -> CohortCoverage:
    return CohortCoverage(2, 2, 0, 0, 0, None, None, None)


def item(
    *,
    simfile_id: str = "42",
    status: str = "success",
    reference_events: tuple[BenchmarkEvent, ...] | None = None,
    prediction_events: tuple[BenchmarkEvent, ...] | None = (),
    coverage: CohortCoverage | None = None,
    failure_reason: str | None = None,
) -> CohortItem:
    if reference_events is None:
        reference_events = (
            BenchmarkEvent(simfile_id, 1.0, "tom", "ground_truth"),
            BenchmarkEvent(simfile_id, 2.0, "kick", "ground_truth"),
        )
    return CohortItem(
        simfile_id=simfile_id,
        status=status,  # type: ignore[arg-type]
        reference_events=reference_events,
        prediction_events=prediction_events,
        coverage=coverage
        or (
            no_prediction_coverage()
            if prediction_events is None
            else CohortCoverage(2, 2, 0, 0, 0, 0, 0, 0)
        ),
        failure_reason=failure_reason,  # type: ignore[arg-type]
    )


def test_cohort_failure_reason_set_is_closed() -> None:
    assert COHORT_FAILURE_REASONS == {
        "reference_quarantined",
        "backend_unavailable",
        "inference_failed",
        "prediction_artifact_invalid",
        "prediction_missing",
        "explicitly_skipped",
    }
    from src.benchmark.cohort_scoring import CohortFailureReason

    assert COHORT_FAILURE_REASONS == frozenset(get_args(CohortFailureReason))


@pytest.mark.parametrize(
    "field",
    ["reference_manifest_sha256", "model_lock_sha256", "backend_descriptor_sha256"],
)
def test_identity_reuses_shared_sha256_validation(field: str) -> None:
    values = dataclasses.asdict(identity())
    values[field] = "ABC"
    with pytest.raises(ValueError, match=f"{field} must be lowercase SHA-256"):
        CohortIdentity(**values)


def test_identity_requires_nonempty_fields_and_exact_scoring_version() -> None:
    values = dataclasses.asdict(identity())
    values["model_id"] = ""
    with pytest.raises(ValueError, match="model_id must be a nonempty string"):
        CohortIdentity(**values)

    values = dataclasses.asdict(identity())
    values["scoring_version"] = "other/v1"
    with pytest.raises(ValueError, match=SCORING_VERSION):
        CohortIdentity(**values)


def test_coverage_from_reference_and_prediction_artifacts(tmp_path: Path) -> None:
    reference = reference_mapping()
    coverage = coverage_from_artifacts(reference, prediction_artifact(tmp_path))

    assert coverage.reference_native_event_count == 4
    assert coverage.reference_common_event_count == 2
    assert coverage.reference_duplicate_collapsed_count == 1
    assert coverage.reference_ignored_event_count == 1
    assert coverage.reference_unmapped_event_count == 0
    assert (
        coverage.reference_native_event_count
        == coverage.reference_common_event_count
        + coverage.reference_duplicate_collapsed_count
        + coverage.reference_ignored_event_count
        + coverage.reference_unmapped_event_count
    )
    assert coverage.prediction_native_event_count == 2
    assert coverage.prediction_mapped_event_count == 1
    assert coverage.prediction_unmapped_event_count == 1
    assert coverage.prediction_native_class_counts == (("midi_42", 1), ("midi_46", 1))


def test_coverage_without_prediction_has_no_prediction_counts() -> None:
    coverage = coverage_from_artifacts(reference_mapping(), None)

    assert coverage.prediction_native_event_count is None
    assert coverage.prediction_mapped_event_count is None
    assert coverage.prediction_unmapped_event_count is None
    assert coverage.prediction_native_class_counts == ()


def test_validate_cohort_items_accepts_success_and_closed_failure_shapes() -> None:
    successful = item()
    failed = item(
        simfile_id="43",
        status="failed",
        prediction_events=None,
        failure_reason="backend_unavailable",
    )
    skipped = item(
        simfile_id="44",
        status="skipped",
        prediction_events=None,
        failure_reason="explicitly_skipped",
    )
    quarantined = item(
        simfile_id="45",
        status="quarantined",
        prediction_events=None,
        failure_reason="reference_quarantined",
    )

    assert validate_cohort_items(identity(), (successful, failed, skipped, quarantined)) is None


def test_validate_cohort_items_rejects_duplicate_ids_and_invalid_status_shapes() -> None:
    with pytest.raises(ValueError, match="simfile_id values must be unique"):
        validate_cohort_items(identity(), (item(), item(simfile_id="42")))

    with pytest.raises(ValueError, match="success item requires nonempty reference_events"):
        validate_cohort_items(
            identity(),
            (
                item(
                    reference_events=(),
                    coverage=CohortCoverage(0, 0, 0, 0, 0, 0, 0, 0),
                ),
            ),
        )

    with pytest.raises(ValueError, match="success item requires failure_reason to be None"):
        validate_cohort_items(identity(), (item(failure_reason="prediction_missing"),))

    with pytest.raises(ValueError, match="failed item requires a prediction failure reason"):
        validate_cohort_items(identity(), (item(status="failed", prediction_events=None),))


def test_validate_cohort_items_rejects_unbalanced_coverage() -> None:
    invalid = item(
        coverage=CohortCoverage(
            reference_native_event_count=2,
            reference_common_event_count=1,
            reference_ignored_event_count=0,
            reference_unmapped_event_count=0,
            reference_duplicate_collapsed_count=0,
            prediction_native_event_count=0,
            prediction_mapped_event_count=0,
            prediction_unmapped_event_count=0,
        ),
    )
    with pytest.raises(ValueError, match="reference coverage counts do not balance"):
        validate_cohort_items(identity(), (invalid,))


def test_non_success_items_reject_prediction_native_class_counts() -> None:
    invalid = item(
        simfile_id="46",
        status="skipped",
        prediction_events=None,
        failure_reason="explicitly_skipped",
        coverage=CohortCoverage(
            reference_native_event_count=2,
            reference_common_event_count=2,
            reference_ignored_event_count=0,
            reference_unmapped_event_count=0,
            reference_duplicate_collapsed_count=0,
            prediction_native_event_count=None,
            prediction_mapped_event_count=None,
            prediction_unmapped_event_count=None,
            prediction_native_class_counts=(("midi_42", 1),),
        ),
    )
    with pytest.raises(ValueError, match="must not have prediction native class counts"):
        validate_cohort_items(identity(), (invalid,))


def test_success_requires_prediction_coverage_to_match_events() -> None:
    invalid = item(
        prediction_events=(BenchmarkEvent("42", 1.0, "tom", "prediction"),),
        coverage=CohortCoverage(2, 2, 0, 0, 0, 0, 0, 0),
    )
    with pytest.raises(ValueError, match="prediction mapped count must match prediction_events"):
        validate_cohort_items(identity(), (invalid,))


def test_validate_cohort_items_reconciles_success_prediction_metadata(tmp_path: Path) -> None:
    artifact = prediction_artifact(tmp_path)
    reference = reference_mapping()
    prediction_events = prediction_to_benchmark_events(artifact)
    valid = CohortItem(
        simfile_id="42",
        status="success",
        reference_events=reference_to_benchmark_events("42", reference.common_events),
        prediction_events=prediction_events,
        coverage=coverage_from_artifacts(reference, artifact),
    )
    assert validate_cohort_items(identity(), (valid,)) is None

    mixed_view_event = dataclasses.replace(
        prediction_events[0],
        metadata={**prediction_events[0].metadata, "input_view_id": "stem-v1"},
    )
    with pytest.raises(ValueError, match="input_view_id"):
        validate_cohort_items(
            identity(),
            (dataclasses.replace(valid, prediction_events=(mixed_view_event,)),),
        )

    mixed_map_event = dataclasses.replace(
        prediction_events[0],
        metadata={**prediction_events[0].metadata, "prediction_map_version": "other/v1"},
    )
    with pytest.raises(ValueError, match="prediction_map_version"):
        validate_cohort_items(
            identity(),
            (dataclasses.replace(valid, prediction_events=(mixed_map_event,)),),
        )

    missing_metadata_event = dataclasses.replace(prediction_events[0], metadata={})
    with pytest.raises(ValueError, match="input_view_id"):
        validate_cohort_items(
            identity(),
            (dataclasses.replace(valid, prediction_events=(missing_metadata_event,)),),
        )


def scoring_item(
    simfile_id: str,
    reference_events: tuple[BenchmarkEvent, ...],
    prediction_events: tuple[BenchmarkEvent, ...],
    *,
    warnings: tuple[str, ...] = (),
) -> CohortItem:
    return CohortItem(
        simfile_id=simfile_id,
        status="success",
        reference_events=reference_events,
        prediction_events=prediction_events,
        coverage=CohortCoverage(
            reference_native_event_count=len(reference_events),
            reference_common_event_count=len(reference_events),
            reference_ignored_event_count=0,
            reference_unmapped_event_count=0,
            reference_duplicate_collapsed_count=0,
            prediction_native_event_count=len(prediction_events),
            prediction_mapped_event_count=len(prediction_events),
            prediction_unmapped_event_count=0,
        ),
        warnings=warnings,
    )


def scoring_event(
    simfile_id: str,
    time_sec: float,
    canonical_class: str,
    source: str,
    metadata: dict[str, object] | None = None,
) -> BenchmarkEvent:
    return BenchmarkEvent(
        chart_id=simfile_id,
        time_sec=time_sec,
        canonical_class=canonical_class,
        source=source,
        metadata=metadata or {},
    )


def test_score_success_items_uses_fixed_tolerance_mode_matrix() -> None:
    success = scoring_item(
        "1",
        (scoring_event("1", 1.0, "kick", "ground_truth"),),
        (scoring_event("1", 1.04, "kick", "prediction"),),
    )

    scores, diagnostics = _score_success_items(
        (success,), DEFAULT_TOLERANCES_MS, diagnostics_for=frozenset()
    )

    assert diagnostics == ()
    assert [(score.tolerance_ms, score.mode) for score in scores] == [
        (30, "raw"),
        (30, "aligned"),
        (50, "raw"),
        (50, "aligned"),
        (100, "raw"),
        (100, "aligned"),
    ]
    assert scores[0].summary.f1 == 0.0
    assert scores[1].summary.f1 == 1.0
    assert scores[2].summary.f1 == 1.0
    assert scores[4].summary.f1 == 1.0


def test_score_success_items_reconciles_per_class_rows() -> None:
    success = scoring_item(
        "1",
        (
            scoring_event("1", 1.0, "kick", "ground_truth"),
            scoring_event("1", 2.0, "snare", "ground_truth"),
        ),
        (
            scoring_event("1", 1.0, "kick", "prediction"),
            scoring_event("1", 3.0, "hihat", "prediction"),
        ),
    )

    scores, _ = _score_success_items((success,), (50,), diagnostics_for=frozenset())

    for score in scores:
        assert sum(row.summary.true_positives for row in score.per_class) == (
            score.summary.true_positives
        )
        assert sum(row.summary.false_positives for row in score.per_class) == (
            score.summary.false_positives
        )
        assert sum(row.summary.false_negatives for row in score.per_class) == (
            score.summary.false_negatives
        )
        assert [row.common_class for row in score.per_class] == sorted(
            row.common_class for row in score.per_class
        )
        assert {row.common_class for row in score.per_class} == {"hihat", "kick", "snare"}
        hihat = next(row for row in score.per_class if row.common_class == "hihat")
        assert hihat.reference_support == 0
        assert hihat.prediction_support == 1


def test_score_success_items_is_independent_of_optional_prediction_metadata() -> None:
    reference = (scoring_event("1", 1.0, "kick", "ground_truth"),)
    bare = scoring_item("1", reference, (scoring_event("1", 1.0, "kick", "prediction"),))
    detailed = scoring_item(
        "1",
        reference,
        (
            scoring_event(
                "1",
                1.0,
                "kick",
                "prediction",
                {"confidence": 0.9, "velocity_midi": 100},
            ),
        ),
    )

    bare_scores, _ = _score_success_items((bare,), (50,), diagnostics_for=frozenset())
    detailed_scores, _ = _score_success_items((detailed,), (50,), diagnostics_for=frozenset())

    assert [
        (score.summary.true_positives, score.summary.false_positives, score.summary.f1)
        for score in bare_scores
    ] == [
        (score.summary.true_positives, score.summary.false_positives, score.summary.f1)
        for score in detailed_scores
    ]


def test_score_success_items_default_diagnostics_are_empty() -> None:
    success_a = scoring_item(
        "1",
        (scoring_event("1", 1.0, "kick", "ground_truth"),),
        (scoring_event("1", 1.0, "kick", "prediction"),),
    )
    success_b = scoring_item(
        "2",
        (scoring_event("2", 1.0, "snare", "ground_truth"),),
        (scoring_event("2", 1.0, "snare", "prediction"),),
    )

    song_scores, diagnostics = _score_success_items(
        (success_a, success_b), DEFAULT_TOLERANCES_MS, diagnostics_for=frozenset()
    )

    assert song_scores
    assert diagnostics == ()


def test_score_success_items_materializes_only_selected_song_diagnostics() -> None:
    success_a = scoring_item(
        "1",
        (scoring_event("1", 1.0, "kick", "ground_truth"),),
        (scoring_event("1", 1.1, "kick", "prediction"),),
    )
    success_b = scoring_item(
        "2",
        (scoring_event("2", 1.0, "kick", "ground_truth"),),
        (scoring_event("2", 1.1, "kick", "prediction"),),
    )

    _, diagnostics = _score_success_items((success_a, success_b), (30,), frozenset({"2"}))

    assert diagnostics
    assert {diagnostic.simfile_id for diagnostic in diagnostics} == {"2"}
    aligned = next(
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.mode == "aligned" and diagnostic.outcome == "matched"
    )
    assert aligned.reference_time_sec == pytest.approx(1.0)
    assert aligned.prediction_time_sec == pytest.approx(1.1)
    assert aligned.scored_prediction_time_sec == pytest.approx(1.0)
    assert aligned.timing_error_sec == pytest.approx(0.0)


def test_aligned_diagnostics_preserve_exact_binary64_prediction_time() -> None:
    success = scoring_item(
        "1",
        (scoring_event("1", 0.3, "kick", "ground_truth"),),
        (scoring_event("1", 0.1, "kick", "prediction"),),
    )

    _, diagnostics = _score_success_items((success,), (50,), frozenset({"1"}))

    aligned = next(
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.mode == "aligned" and diagnostic.outcome == "matched"
    )
    assert aligned.prediction_time_sec == 0.1
