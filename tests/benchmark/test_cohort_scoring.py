from __future__ import annotations

import dataclasses
import struct
from decimal import Decimal
from pathlib import Path
from typing import Callable, get_args

import pytest

from src.benchmark import cohort_scoring
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
    F1Distribution,
    _score_success_items,
    coverage_from_artifacts,
    score_cohort,
    validate_cohort_items,
)
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.models import BenchmarkEvent, ScoreSummary
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
    DTX_LANE_MAP,
    DTX_LANE_MAP_VERSION,
    OAF_PREDICTION_MAP_ID,
    TAXONOMY_VERSION,
    ClassMapping,
)


def build_identity() -> CohortIdentity:
    return CohortIdentity(
        cohort_id="oaf-full-mix-v1",
        reference_manifest_sha256="a" * 64,
        reference_timing_version="sha256:" + "b" * 64,
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=OAF_BACKEND_ID,
        model_id="magenta-egmd-ckpt-569400-v1",
        model_lock_sha256="c" * 64,
        backend_descriptor_sha256=build_descriptor_payload().sha256,
        prediction_map_version=OAF_PREDICTION_MAP_ID,
        input_view_id="full-mix-v1",
    )


def build_native_reference(
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


def build_reference_mapping():
    return map_reference_events(
        (
            build_native_reference("14", 1.0, 0),
            build_native_reference("15", 1.0, 1),
            build_native_reference("13", 2.0, 2),
            build_native_reference("54", 3.0, 3),
        )
    )


def build_descriptor_payload():
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


def build_prediction_artifact(tmp_path: Path) -> PredictionArtifact:
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
        "f" * 64,
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
    mapped, _ = map_oaf_prediction(
        NativePrediction(audio, build_descriptor_payload(), native_events)
    )
    return read_prediction_artifact(render_prediction_artifact(mapped))


def synthetic_reference_mapping(
    simfile_id: str,
    events: tuple[BenchmarkEvent, ...],
):
    lane_by_class = {
        "kick": "13",
        "snare": "12",
        "hihat": "11",
        "tom": "14",
        "crash": "16",
        "ride": "19",
    }
    native_events = tuple(
        NativeReferenceEvent(
            simfile_id=simfile_id,
            selected_chart_key=f"{simfile_id}/chart.dtx",
            selected_chart_content_hash="e" * 64,
            source_audio_key=f"{simfile_id}/audio.wav",
            source_audio_content_hash="f" * 64,
            source_order=index,
            measure=1,
            position=float(index),
            lane_id=lane_by_class[event.canonical_class],
            note_id=f"{event.canonical_class}-{index}",
            chart_time_sec=event.time_sec,
            audio_time_sec=event.time_sec,
        )
        for index, event in enumerate(events)
    )
    return map_reference_events(native_events)


def synthetic_prediction_artifact(
    simfile_id: str,
    events: tuple[BenchmarkEvent, ...],
) -> PredictionArtifact:
    group_by_class = {
        "kick": "kick",
        "snare": "snare",
        "hihat": "hihat",
        "tom": "toms",
        "crash": "crash",
        "ride": "ride",
    }
    audio = CanonicalAudio(
        Path(),
        simfile_id,
        "f" * 64,
        "full-mix-v1",
        "b" * 64,
        46,
        44100,
        1,
        2,
        1,
    )
    native_events = tuple(
        NativeEvent(
            time_sec=event.time_sec,
            native_class_id=f"midi_{index + 22}",
            model_output_bin=index + 1,
            native_midi_note=index + 22,
            native_metadata={"upstream_8hit_group_id": group_by_class[event.canonical_class]},
            confidence=0.9,
            velocity_midi=100,
        )
        for index, event in enumerate(events)
    )
    mapped, _ = map_oaf_prediction(
        NativePrediction(audio, build_descriptor_payload(), native_events)
    )
    return read_prediction_artifact(render_prediction_artifact(mapped))


def no_prediction_coverage() -> CohortCoverage:
    return CohortCoverage(2, 2, 0, 0, 0, None, None, None)


def build_artifact_identity(simfile_id: str) -> cohort_scoring.CohortArtifactIdentity:
    return cohort_scoring.CohortArtifactIdentity(
        simfile_id=simfile_id,
        backend_id=build_identity().backend_id,
        model_id=build_identity().model_id,
        backend_descriptor_sha256=build_identity().backend_descriptor_sha256,
        input_view_id=build_identity().input_view_id,
        prediction_map_version=build_identity().prediction_map_version,
    )


def build_item(
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
        artifact_identity=(
            build_artifact_identity(simfile_id) if prediction_events is not None else None
        ),
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
    values = dataclasses.asdict(build_identity())
    values[field] = "ABC"
    with pytest.raises(ValueError, match=f"{field} must be lowercase SHA-256"):
        CohortIdentity(**values)


def test_identity_requires_nonempty_fields_and_exact_scoring_version() -> None:
    values = dataclasses.asdict(build_identity())
    values["model_id"] = ""
    with pytest.raises(ValueError, match="model_id must be a nonempty string"):
        CohortIdentity(**values)

    values = dataclasses.asdict(build_identity())
    values["scoring_version"] = "other/v1"
    with pytest.raises(ValueError, match=SCORING_VERSION):
        CohortIdentity(**values)


def test_coverage_from_reference_and_prediction_artifacts(tmp_path: Path) -> None:
    reference = build_reference_mapping()
    coverage = coverage_from_artifacts(reference, build_prediction_artifact(tmp_path))

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
    coverage = coverage_from_artifacts(build_reference_mapping(), None)

    assert coverage.prediction_native_event_count is None
    assert coverage.prediction_mapped_event_count is None
    assert coverage.prediction_unmapped_event_count is None
    assert coverage.prediction_native_class_counts == ()


def test_validate_cohort_items_accepts_success_and_closed_failure_shapes() -> None:
    successful = cohort_scoring_item(
        "42",
        (
            build_scoring_event("42", 1.0, "tom", "ground_truth"),
            build_scoring_event("42", 2.0, "kick", "ground_truth"),
        ),
        (),
    )
    failed = build_item(
        simfile_id="43",
        status="failed",
        prediction_events=None,
        failure_reason="backend_unavailable",
    )
    skipped = build_item(
        simfile_id="44",
        status="skipped",
        prediction_events=None,
        failure_reason="explicitly_skipped",
    )
    quarantined = build_item(
        simfile_id="45",
        status="quarantined",
        prediction_events=None,
        failure_reason="reference_quarantined",
    )

    assert (
        validate_cohort_items(build_identity(), (successful, failed, skipped, quarantined)) is None
    )


def test_validate_cohort_items_rejects_duplicate_ids_and_invalid_status_shapes() -> None:
    successful = cohort_scoring_item(
        "42",
        (build_scoring_event("42", 1.0, "kick", "ground_truth"),),
        (),
    )
    with pytest.raises(ValueError, match="simfile_id values must be unique"):
        validate_cohort_items(build_identity(), (successful, successful))

    with pytest.raises(ValueError, match="success item requires nonempty reference_events"):
        validate_cohort_items(
            build_identity(),
            (
                build_item(
                    reference_events=(),
                    coverage=CohortCoverage(0, 0, 0, 0, 0, 0, 0, 0),
                ),
            ),
        )

    with pytest.raises(ValueError, match="success item requires failure_reason to be None"):
        validate_cohort_items(build_identity(), (build_item(failure_reason="prediction_missing"),))

    with pytest.raises(ValueError, match="failed item requires a prediction failure reason"):
        validate_cohort_items(
            build_identity(), (build_item(status="failed", prediction_events=None),)
        )


def test_validate_cohort_items_rejects_unbalanced_coverage() -> None:
    invalid = build_item(
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
        validate_cohort_items(build_identity(), (invalid,))


def test_non_success_items_reject_prediction_native_class_counts() -> None:
    invalid = build_item(
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
        validate_cohort_items(build_identity(), (invalid,))


def test_success_requires_prediction_coverage_to_match_events() -> None:
    valid = cohort_scoring_item(
        "42",
        (build_scoring_event("42", 1.0, "tom", "ground_truth"),),
        (build_scoring_event("42", 1.0, "tom", "prediction"),),
    )
    invalid = dataclasses.replace(
        valid,
        coverage=dataclasses.replace(valid.coverage, prediction_mapped_event_count=0),
    )
    with pytest.raises(ValueError, match="coverage"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_cohort_items_reconciles_success_prediction_metadata(tmp_path: Path) -> None:
    artifact = _artifact_for_song(tmp_path, "42")
    reference = build_reference_mapping()
    prediction_events = prediction_to_benchmark_events(artifact)
    valid = cohort_scoring.cohort_item_from_artifacts(
        build_identity(),
        "42",
        reference,
        artifact,
    )
    assert validate_cohort_items(build_identity(), (valid,)) is None

    mixed_view_event = dataclasses.replace(
        prediction_events[0],
        metadata={**prediction_events[0].metadata, "input_view_id": "stem-v1"},
    )
    with pytest.raises(ValueError, match="input_view_id"):
        validate_cohort_items(
            build_identity(),
            (dataclasses.replace(valid, prediction_events=(mixed_view_event,)),),
        )

    mixed_map_event = dataclasses.replace(
        prediction_events[0],
        metadata={**prediction_events[0].metadata, "prediction_map_version": "other/v1"},
    )
    with pytest.raises(ValueError, match="prediction_map_version"):
        validate_cohort_items(
            build_identity(),
            (dataclasses.replace(valid, prediction_events=(mixed_map_event,)),),
        )

    missing_metadata_event = dataclasses.replace(prediction_events[0], metadata={})
    with pytest.raises(ValueError, match="input_view_id"):
        validate_cohort_items(
            build_identity(),
            (dataclasses.replace(valid, prediction_events=(missing_metadata_event,)),),
        )


def _artifact_for_song(tmp_path: Path, simfile_id: str) -> PredictionArtifact:
    artifact = build_prediction_artifact(tmp_path)
    audio = dataclasses.replace(artifact.prediction.audio, source_audio_id=simfile_id)
    mapped = dataclasses.replace(artifact.prediction, audio=audio)
    return read_prediction_artifact(render_prediction_artifact(mapped))


def _empty_artifact_for_song(tmp_path: Path, simfile_id: str) -> PredictionArtifact:
    artifact = _artifact_for_song(tmp_path, simfile_id)
    mapped = dataclasses.replace(artifact.prediction, events=())
    return read_prediction_artifact(render_prediction_artifact(mapped))


def test_cohort_item_from_artifacts_binds_song_and_descriptor_provenance(
    tmp_path: Path,
) -> None:
    reference = build_reference_mapping()
    artifact = _artifact_for_song(tmp_path, "42")

    item = cohort_scoring.cohort_item_from_artifacts(
        build_identity(),
        "42",
        reference,
        artifact,
    )

    assert item.artifact_identity is not None
    assert item.artifact_identity.simfile_id == "42"
    assert item.artifact_identity.backend_id == build_identity().backend_id
    assert item.artifact_identity.model_id == build_identity().model_id
    assert (
        item.artifact_identity.backend_descriptor_sha256
        == build_identity().backend_descriptor_sha256
    )
    assert validate_cohort_items(build_identity(), (item,)) is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend_id", "other-backend", "backend_id"),
        ("model_id", "other-model", "model_id"),
        ("backend_descriptor_sha256", "e" * 64, "backend_descriptor_sha256"),
        ("input_view_id", "stem-v1", "input_view_id"),
        ("prediction_map_version", "other-map", "prediction_map_version"),
    ],
)
def test_validate_cohort_items_rejects_mixed_artifact_provenance(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    item = cohort_scoring.cohort_item_from_artifacts(
        build_identity(),
        "42",
        build_reference_mapping(),
        _artifact_for_song(tmp_path, "42"),
    )
    evidence = dataclasses.replace(item.artifact_identity, **{field: value})
    mixed = dataclasses.replace(item, artifact_identity=evidence)

    with pytest.raises(ValueError, match=message):
        validate_cohort_items(build_identity(), (mixed,))


def test_cohort_item_from_artifacts_rejects_wrong_song_reference_and_prediction(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="reference.*simfile_id"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(),
            "8",
            build_reference_mapping(),
            _artifact_for_song(tmp_path, "8"),
        )

    with pytest.raises(ValueError, match="prediction.*source_audio_id"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(),
            "42",
            build_reference_mapping(),
            _artifact_for_song(tmp_path, "8"),
        )


def test_cohort_item_rejects_mismatched_source_audio_hash(tmp_path: Path) -> None:
    """A stale prediction for an older audio revision must not score against a newer reference."""
    reference = build_reference_mapping()
    artifact = _artifact_for_song(tmp_path, "42")
    mismatched_audio = dataclasses.replace(artifact.prediction.audio, source_audio_sha256="e" * 64)
    mismatched_prediction = dataclasses.replace(artifact.prediction, audio=mismatched_audio)
    mismatched_artifact = read_prediction_artifact(
        render_prediction_artifact(mismatched_prediction)
    )
    with pytest.raises(ValueError, match="source_audio_content_hash.*source_audio_sha256"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(), "42", reference, mismatched_artifact
        )


def test_identity_rejects_wrong_taxonomy_version() -> None:
    values = dataclasses.asdict(build_identity())
    values["taxonomy_version"] = "crux.dtx-taxonomy/v1"
    with pytest.raises(ValueError, match="taxonomy_version must be"):
        CohortIdentity(**values)


def test_identity_rejects_wrong_lane_map_version() -> None:
    values = dataclasses.asdict(build_identity())
    values["lane_map_version"] = "crux.other-lane-map/v1"
    with pytest.raises(ValueError, match="lane_map_version must be"):
        CohortIdentity(**values)


def test_validate_rejects_reference_mapped_with_custom_lane_map(tmp_path: Path) -> None:
    """A reference mapped with a custom lane_map must not be reported as crux.dtx-lane-map/v1."""
    native_events = (
        build_native_reference("14", 1.0, 0),
        build_native_reference("15", 1.0, 1),
        build_native_reference("13", 2.0, 2),
    )
    custom_lane_map = {
        "14": ClassMapping("snare", "snare"),
        "15": ClassMapping("low_or_floor_tom", "tom"),
        "13": ClassMapping("kick", "kick"),
    }
    custom_reference = map_reference_events(native_events, lane_map=custom_lane_map)
    artifact = _artifact_for_song(tmp_path, "42")
    with pytest.raises(ValueError, match="frozen DTX lane map"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(), "42", custom_reference, artifact
        )


def test_validate_cohort_items_rejects_reference_and_prediction_chart_or_source_mismatch(
    tmp_path: Path,
) -> None:
    item = cohort_scoring.cohort_item_from_artifacts(
        build_identity(),
        "42",
        build_reference_mapping(),
        _artifact_for_song(tmp_path, "42"),
    )

    bad_reference_chart = dataclasses.replace(item.reference_events[0], chart_id="8")
    with pytest.raises(ValueError, match="reference event chart_id"):
        validate_cohort_items(
            build_identity(),
            (
                dataclasses.replace(
                    item, reference_events=(bad_reference_chart, *item.reference_events[1:])
                ),
            ),
        )

    bad_reference_source = dataclasses.replace(item.reference_events[0], source="prediction")
    with pytest.raises(ValueError, match="reference event source"):
        validate_cohort_items(
            build_identity(),
            (
                dataclasses.replace(
                    item, reference_events=(bad_reference_source, *item.reference_events[1:])
                ),
            ),
        )

    bad_prediction_chart = dataclasses.replace(item.prediction_events[0], chart_id="8")
    with pytest.raises(ValueError, match="prediction event chart_id"):
        validate_cohort_items(
            build_identity(),
            (
                dataclasses.replace(
                    item, prediction_events=(bad_prediction_chart, *item.prediction_events[1:])
                ),
            ),
        )

    bad_prediction_source = dataclasses.replace(item.prediction_events[0], source="ground_truth")
    with pytest.raises(ValueError, match="prediction event source"):
        validate_cohort_items(
            build_identity(),
            (
                dataclasses.replace(
                    item, prediction_events=(bad_prediction_source, *item.prediction_events[1:])
                ),
            ),
        )


def test_zero_event_artifact_is_identity_bound_without_event_metadata(tmp_path: Path) -> None:
    item = cohort_scoring.cohort_item_from_artifacts(
        build_identity(),
        "42",
        build_reference_mapping(),
        _empty_artifact_for_song(tmp_path, "42"),
    )

    assert item.prediction_events == ()
    assert item.artifact_identity is not None
    assert item.artifact_identity.prediction_map_version == build_identity().prediction_map_version
    assert validate_cohort_items(build_identity(), (item,)) is None

    mixed_identity = dataclasses.replace(build_identity(), prediction_map_version="other-map")
    with pytest.raises(ValueError, match="prediction_map_version"):
        validate_cohort_items(mixed_identity, (item,))


def test_success_item_without_artifact_provenance_is_rejected() -> None:
    with pytest.raises(ValueError, match="artifact_identity"):
        validate_cohort_items(
            build_identity(), (dataclasses.replace(build_item(), artifact_identity=None),)
        )


def test_score_cohort_rejects_forged_artifact_identity_for_empty_prediction() -> None:
    """A caller-created identity must not bless an unbacked empty artifact."""
    forged = build_item(
        prediction_events=(),
        coverage=CohortCoverage(2, 2, 0, 0, 0, 0, 0, 0),
    )

    with pytest.raises(ValueError, match="artifact"):
        score_cohort(build_identity(), (forged,), tolerances_ms=(50,))


def test_score_cohort_rejects_prediction_projection_substitution(tmp_path: Path) -> None:
    artifact = _artifact_for_song(tmp_path, "42")
    reference = build_reference_mapping()
    valid = cohort_scoring.cohort_item_from_artifacts(build_identity(), "42", reference, artifact)
    tampered_prediction = dataclasses.replace(artifact.prediction, events=())
    tampered_artifact = dataclasses.replace(artifact, prediction=tampered_prediction)
    assert tampered_artifact.prediction.events == ()
    assert tampered_artifact.content == artifact.content
    assert tampered_artifact.event_count == artifact.event_count == 2
    assert tampered_artifact.prefix_sha256 == artifact.prefix_sha256
    assert tampered_artifact.artifact_sha256 == artifact.artifact_sha256
    tampered = dataclasses.replace(
        valid,
        prediction_events=(),
        coverage=coverage_from_artifacts(reference, tampered_artifact),
        prediction_artifact=tampered_artifact,
    )

    with pytest.raises(
        ValueError, match="prediction artifact fields do not match canonical content"
    ):
        score_cohort(build_identity(), (tampered,), tolerances_ms=(50,))


def test_score_cohort_rejects_reference_common_projection_substitution(tmp_path: Path) -> None:
    artifact = _artifact_for_song(tmp_path, "42")
    reference = build_reference_mapping()
    valid = cohort_scoring.cohort_item_from_artifacts(build_identity(), "42", reference, artifact)
    forged_common = dataclasses.replace(
        reference.common_events[0],
        canonical_audio_time=Decimal("9.000000"),
    )
    tampered_reference = dataclasses.replace(
        reference,
        common_events=(forged_common, *reference.common_events[1:]),
    )
    tampered = dataclasses.replace(
        valid,
        reference_events=reference_to_benchmark_events("42", tampered_reference.common_events),
        reference_artifact=tampered_reference,
    )

    with pytest.raises(
        ValueError, match="reference common_events do not match mapped event projection"
    ):
        score_cohort(build_identity(), (tampered,), tolerances_ms=(50,))


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
        artifact_identity=build_artifact_identity(simfile_id),
    )


def build_scoring_event(
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


def cohort_scoring_item(
    simfile_id: str,
    reference_events: tuple[BenchmarkEvent, ...],
    prediction_events: tuple[BenchmarkEvent, ...],
    *,
    warnings: tuple[str, ...] = (),
) -> CohortItem:
    return cohort_scoring.cohort_item_from_artifacts(
        build_identity(),
        simfile_id,
        synthetic_reference_mapping(simfile_id, reference_events),
        synthetic_prediction_artifact(simfile_id, prediction_events),
        warnings=warnings,
    )


def test_score_success_items_uses_fixed_tolerance_mode_matrix() -> None:
    success = scoring_item(
        "1",
        (build_scoring_event("1", 1.0, "kick", "ground_truth"),),
        (build_scoring_event("1", 1.04, "kick", "prediction"),),
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
            build_scoring_event("1", 1.0, "kick", "ground_truth"),
            build_scoring_event("1", 2.0, "snare", "ground_truth"),
        ),
        (
            build_scoring_event("1", 1.0, "kick", "prediction"),
            build_scoring_event("1", 3.0, "hihat", "prediction"),
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
    reference = (build_scoring_event("1", 1.0, "kick", "ground_truth"),)
    bare = scoring_item("1", reference, (build_scoring_event("1", 1.0, "kick", "prediction"),))
    detailed = scoring_item(
        "1",
        reference,
        (
            build_scoring_event(
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
        (build_scoring_event("1", 1.0, "kick", "ground_truth"),),
        (build_scoring_event("1", 1.0, "kick", "prediction"),),
    )
    success_b = scoring_item(
        "2",
        (build_scoring_event("2", 1.0, "snare", "ground_truth"),),
        (build_scoring_event("2", 1.0, "snare", "prediction"),),
    )

    song_scores, diagnostics = _score_success_items(
        (success_a, success_b), DEFAULT_TOLERANCES_MS, diagnostics_for=frozenset()
    )

    assert song_scores
    assert diagnostics == ()


def test_score_success_items_materializes_only_selected_song_diagnostics() -> None:
    success_a = scoring_item(
        "1",
        (build_scoring_event("1", 1.0, "kick", "ground_truth"),),
        (build_scoring_event("1", 1.1, "kick", "prediction"),),
    )
    success_b = scoring_item(
        "2",
        (build_scoring_event("2", 1.0, "kick", "ground_truth"),),
        (build_scoring_event("2", 1.1, "kick", "prediction"),),
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
        (build_scoring_event("1", 0.3, "kick", "ground_truth"),),
        (build_scoring_event("1", 0.1, "kick", "prediction"),),
    )

    _, diagnostics = _score_success_items((success,), (50,), frozenset({"1"}))

    aligned = next(
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.mode == "aligned" and diagnostic.outcome == "matched"
    )
    assert aligned.prediction_time_sec == 0.1


def test_score_cohort_canonicalizes_items_and_song_rows() -> None:
    item_1 = cohort_scoring_item(
        "1",
        (build_scoring_event("1", 1.0, "kick", "ground_truth"),),
        (build_scoring_event("1", 1.0, "kick", "prediction"),),
    )
    item_2 = cohort_scoring_item(
        "2",
        (build_scoring_event("2", 1.0, "snare", "ground_truth"),),
        (build_scoring_event("2", 1.0, "snare", "prediction"),),
    )

    result = score_cohort(build_identity(), (item_2, item_1), tolerances_ms=(50, 100))

    assert [item.simfile_id for item in result.items] == ["1", "2"]
    assert [(row.simfile_id, row.tolerance_ms, row.mode) for row in result.song_scores] == [
        ("1", 50, "raw"),
        ("1", 50, "aligned"),
        ("1", 100, "raw"),
        ("1", 100, "aligned"),
        ("2", 50, "raw"),
        ("2", 50, "aligned"),
        ("2", 100, "raw"),
        ("2", 100, "aligned"),
    ]
    assert result.event_diagnostics == ()
    assert result.tolerances_ms == (50, 100)


@pytest.mark.parametrize(
    ("tolerances_ms", "message"),
    [
        ((0,), "positive"),
        ((50, 50), "unique"),
        ((100, 50), "sorted"),
    ],
)
def test_score_cohort_rejects_invalid_tolerances(
    tolerances_ms: tuple[int, ...], message: str
) -> None:
    item_1 = cohort_scoring_item(
        "1",
        (build_scoring_event("1", 1.0, "kick", "ground_truth"),),
        (build_scoring_event("1", 1.0, "kick", "prediction"),),
    )

    with pytest.raises(ValueError, match=message):
        score_cohort(build_identity(), (item_1,), tolerances_ms=tolerances_ms)


def test_score_cohort_rejects_invalid_diagnostics_requests() -> None:
    success = cohort_scoring_item(
        "1",
        (build_scoring_event("1", 1.0, "kick", "ground_truth"),),
        (build_scoring_event("1", 1.0, "kick", "prediction"),),
    )
    failed = build_item(
        simfile_id="2",
        status="failed",
        prediction_events=None,
        failure_reason="inference_failed",
    )

    with pytest.raises(ValueError, match="unique"):
        score_cohort(build_identity(), (success,), diagnostics_for=("1", "1"))
    with pytest.raises(ValueError, match="nonempty"):
        score_cohort(build_identity(), (success,), diagnostics_for=("",))
    with pytest.raises(ValueError, match="successful"):
        score_cohort(build_identity(), (success, failed), diagnostics_for=("2",))
    with pytest.raises(ValueError, match="names an input item"):
        score_cohort(build_identity(), (success,), diagnostics_for=("missing",))


def test_score_cohort_materializes_diagnostics_only_for_requested_successes() -> None:
    success_1 = cohort_scoring_item(
        "1",
        (build_scoring_event("1", 1.0, "kick", "ground_truth"),),
        (build_scoring_event("1", 1.1, "kick", "prediction"),),
    )
    success_2 = cohort_scoring_item(
        "2",
        (build_scoring_event("2", 1.0, "snare", "ground_truth"),),
        (build_scoring_event("2", 1.1, "snare", "prediction"),),
    )

    result = score_cohort(
        build_identity(), (success_2, success_1), tolerances_ms=(30,), diagnostics_for=("2",)
    )

    assert result.event_diagnostics
    assert {row.simfile_id for row in result.event_diagnostics} == {"2"}


def test_score_cohort_retains_full_population_and_closed_reason_counts() -> None:
    successful = cohort_scoring_item(
        "1",
        (build_scoring_event("1", 1.0, "kick", "ground_truth"),),
        (build_scoring_event("1", 1.0, "kick", "prediction"),),
        warnings=("warning-only",),
    )
    failed = build_item(
        simfile_id="2",
        status="failed",
        prediction_events=None,
        failure_reason="inference_failed",
        coverage=no_prediction_coverage(),
    )
    skipped = build_item(
        simfile_id="3",
        status="skipped",
        prediction_events=None,
        failure_reason="explicitly_skipped",
        coverage=no_prediction_coverage(),
    )
    quarantined = build_item(
        simfile_id="4",
        status="quarantined",
        prediction_events=None,
        failure_reason="reference_quarantined",
        coverage=no_prediction_coverage(),
    )

    result = score_cohort(build_identity(), (quarantined, successful, skipped, failed))

    assert result.population.total_count == 4
    assert result.population.success_count == 1
    assert result.population.failed_count == 1
    assert result.population.skipped_count == 1
    assert result.population.quarantined_count == 1
    assert result.population.reason_counts == (
        ("explicitly_skipped", 1),
        ("inference_failed", 1),
        ("reference_quarantined", 1),
    )


def test_score_cohort_aggregates_event_micro_song_macro_and_class_macro() -> None:
    song_a = cohort_scoring_item(
        "a",
        (
            build_scoring_event("a", 1.0, "kick", "ground_truth"),
            build_scoring_event("a", 2.0, "snare", "ground_truth"),
        ),
        (build_scoring_event("a", 1.0, "kick", "prediction"),),
    )
    song_b = cohort_scoring_item(
        "b",
        (build_scoring_event("b", 1.0, "kick", "ground_truth"),),
        (
            build_scoring_event("b", 1.0, "kick", "prediction"),
            build_scoring_event("b", 2.0, "kick", "prediction"),
        ),
    )

    result = score_cohort(build_identity(), (song_b, song_a), tolerances_ms=(50,))
    aggregate = next(row for row in result.aggregates if row.mode == "raw")

    assert (
        aggregate.event_micro.true_positives,
        aggregate.event_micro.false_positives,
        aggregate.event_micro.false_negatives,
    ) == (2, 1, 1)
    assert aggregate.event_micro.precision == pytest.approx(2 / 3)
    assert aggregate.event_micro.recall == pytest.approx(2 / 3)
    assert aggregate.event_micro.f1 == pytest.approx(2 / 3)
    assert aggregate.song_macro_f1 == pytest.approx(2 / 3)
    assert aggregate.class_macro_f1 == pytest.approx(0.4)
    assert aggregate.successful_song_count == 2
    assert [row.common_class for row in aggregate.per_class] == ["kick", "snare"]
    kick = aggregate.per_class[0]
    assert (
        kick.summary.true_positives,
        kick.summary.false_positives,
        kick.summary.false_negatives,
    ) == (
        2,
        1,
        0,
    )
    assert kick.summary.f1 == pytest.approx(0.8)


def test_score_cohort_uses_percentile_convention_for_song_f1_distribution() -> None:
    songs = (
        cohort_scoring_item(
            "0.0",
            (build_scoring_event("0.0", 1.0, "kick", "ground_truth"),),
            (),
        ),
        cohort_scoring_item(
            "0.25",
            (build_scoring_event("0.25", 1.0, "kick", "ground_truth"),),
            (
                build_scoring_event("0.25", 1.0, "kick", "prediction"),
                *tuple(
                    build_scoring_event("0.25", float(index), "kick", "prediction")
                    for index in range(2, 8)
                ),
            ),
        ),
        cohort_scoring_item(
            "0.5",
            (build_scoring_event("0.5", 1.0, "kick", "ground_truth"),),
            (
                build_scoring_event("0.5", 1.0, "kick", "prediction"),
                build_scoring_event("0.5", 2.0, "kick", "prediction"),
                build_scoring_event("0.5", 3.0, "kick", "prediction"),
            ),
        ),
        cohort_scoring_item(
            "0.75",
            tuple(
                build_scoring_event("0.75", float(index), "kick", "ground_truth")
                for index in range(1, 5)
            ),
            (
                *(
                    build_scoring_event("0.75", float(index), "kick", "prediction")
                    for index in range(1, 4)
                ),
                build_scoring_event("0.75", 5.0, "kick", "prediction"),
            ),
        ),
        cohort_scoring_item(
            "1.0",
            (build_scoring_event("1.0", 1.0, "kick", "ground_truth"),),
            (build_scoring_event("1.0", 1.0, "kick", "prediction"),),
        ),
    )

    result = score_cohort(build_identity(), songs, tolerances_ms=(50,))
    distribution = next(row for row in result.aggregates if row.mode == "raw").song_f1_distribution

    assert distribution.minimum == 0.0
    assert distribution.p10 == 0.25
    assert distribution.p25 == 0.25
    assert distribution.median == 0.5
    assert distribution.p75 == 0.75
    assert distribution.p90 == 1.0
    assert distribution.maximum == 1.0


def test_score_cohort_zero_success_keeps_population_and_undefined_aggregates() -> None:
    failed = build_item(
        simfile_id="1",
        status="failed",
        prediction_events=None,
        failure_reason="prediction_missing",
        coverage=no_prediction_coverage(),
    )
    quarantined = build_item(
        simfile_id="2",
        status="quarantined",
        prediction_events=None,
        failure_reason="reference_quarantined",
        coverage=no_prediction_coverage(),
    )

    result = score_cohort(build_identity(), (quarantined, failed))

    assert len(result.aggregates) == 6
    assert result.population.total_count == 2
    assert result.population.success_count == 0
    assert result.population.failed_count == 1
    assert result.population.quarantined_count == 1
    assert result.event_diagnostics == ()
    for aggregate in result.aggregates:
        assert aggregate.event_micro == ScoreSummary(0, 0, 0)
        assert aggregate.event_micro.precision is None
        assert aggregate.event_micro.recall is None
        assert aggregate.event_micro.f1 is None
        assert aggregate.song_macro_f1 is None
        assert aggregate.class_macro_f1 is None
        assert aggregate.song_f1_distribution == F1Distribution(
            None, None, None, None, None, None, None
        )
        assert aggregate.per_class == ()
        assert aggregate.successful_song_count == 0


# ---------------------------------------------------------------------------
# Coverage gap tests: validation error paths in cohort_scoring.py
# ---------------------------------------------------------------------------


def test_artifact_identity_rejects_empty_field() -> None:
    with pytest.raises(ValueError, match="simfile_id must be a nonempty string"):
        cohort_scoring.CohortArtifactIdentity(
            simfile_id="",
            backend_id=OAF_BACKEND_ID,
            model_id="model",
            backend_descriptor_sha256="e" * 64,
            input_view_id="full-mix-v1",
            prediction_map_version=OAF_PREDICTION_MAP_ID,
        )


@pytest.mark.parametrize(
    ("expected", "match", "build_args"),
    [
        # Validation stops at the identity check, so reference/prediction
        # fixtures are never reached and need not be built.
        pytest.param(
            TypeError,
            "identity must be CohortIdentity",
            lambda tmp_path: ("not identity", "42", None, None),
            id="bad-identity",
        ),
        pytest.param(
            ValueError,
            "simfile_id must be a nonempty string",
            lambda tmp_path: (build_identity(), "", None, None),
            id="empty-simfile-id",
        ),
        # Validation stops at the reference check, so the prediction fixture
        # is never reached and need not be built.
        pytest.param(
            TypeError,
            "reference must be ReferenceMappingResult",
            lambda tmp_path: (build_identity(), "42", "not ref", None),
            id="bad-reference",
        ),
        # The prediction check runs last, so a real reference fixture is
        # required to reach it.
        pytest.param(
            TypeError,
            "prediction must be PredictionArtifact",
            lambda tmp_path: (
                build_identity(),
                "42",
                build_reference_mapping(),
                "not pred",
            ),
            id="bad-prediction",
        ),
    ],
)
def test_cohort_item_from_artifacts_rejects_bad_argument_types(
    tmp_path: Path,
    expected: type[BaseException],
    match: str,
    build_args: Callable[[Path], tuple],
) -> None:
    identity, simfile_id, ref, pred = build_args(tmp_path)
    with pytest.raises(expected, match=match):
        cohort_scoring.cohort_item_from_artifacts(identity, simfile_id, ref, pred)


def test_cohort_item_from_artifacts_rejects_non_tuple_warnings(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="warnings must be a tuple of strings"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(),
            "42",
            build_reference_mapping(),
            _artifact_for_song(tmp_path, "42"),
            warnings=["not a tuple"],
        )


def test_coverage_from_artifacts_rejects_bad_types() -> None:
    with pytest.raises(TypeError, match="reference must be ReferenceMappingResult"):
        coverage_from_artifacts("not a reference", None)

    with pytest.raises(TypeError, match="prediction must be PredictionArtifact or None"):
        coverage_from_artifacts(build_reference_mapping(), "not a prediction")


def test_validate_cohort_items_rejects_bad_identity_type() -> None:
    with pytest.raises(TypeError, match="identity must be CohortIdentity"):
        validate_cohort_items("not identity", ())


def test_validate_cohort_items_rejects_non_cohort_item() -> None:
    with pytest.raises(TypeError, match="cohort items must be CohortItem"):
        validate_cohort_items(build_identity(), ("not item",))


def test_validate_cohort_items_rejects_empty_simfile_id() -> None:
    invalid = dataclasses.replace(build_item(), simfile_id="")
    with pytest.raises(ValueError, match="simfile_id must be a nonempty string"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_cohort_items_rejects_invalid_status() -> None:
    invalid = dataclasses.replace(build_item(), status="other")
    with pytest.raises(ValueError, match="invalid status"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_success_item_rejects_none_prediction_events() -> None:
    invalid = build_item(prediction_events=None)
    with pytest.raises(ValueError, match="success item requires prediction_events"):
        validate_cohort_items(build_identity(), (invalid,))


def test_non_success_item_rejects_artifact_evidence() -> None:
    invalid = dataclasses.replace(
        build_item(
            simfile_id="43",
            status="failed",
            prediction_events=None,
            failure_reason="backend_unavailable",
        ),
        artifact_identity=build_artifact_identity("43"),
    )
    with pytest.raises(ValueError, match="must not have artifact evidence"):
        validate_cohort_items(build_identity(), (invalid,))


def test_non_success_item_rejects_prediction_events() -> None:
    invalid = CohortItem(
        simfile_id="43",
        status="failed",
        reference_events=(BenchmarkEvent("43", 1.0, "kick", "ground_truth"),),
        prediction_events=(BenchmarkEvent("43", 1.0, "kick", "prediction"),),
        coverage=CohortCoverage(1, 1, 0, 0, 0, None, None, None),
        failure_reason="backend_unavailable",
    )
    with pytest.raises(ValueError, match="must not have prediction_events"):
        validate_cohort_items(build_identity(), (invalid,))


def test_non_success_item_rejects_prediction_coverage() -> None:
    invalid = CohortItem(
        simfile_id="43",
        status="failed",
        reference_events=(BenchmarkEvent("43", 1.0, "kick", "ground_truth"),),
        prediction_events=None,
        coverage=CohortCoverage(1, 1, 0, 0, 0, 1, 1, 0),
        failure_reason="backend_unavailable",
    )
    with pytest.raises(ValueError, match="must not have prediction coverage"):
        validate_cohort_items(build_identity(), (invalid,))


def test_skipped_item_rejects_wrong_failure_reason() -> None:
    invalid = CohortItem(
        simfile_id="44",
        status="skipped",
        reference_events=(BenchmarkEvent("44", 1.0, "kick", "ground_truth"),),
        prediction_events=None,
        coverage=CohortCoverage(1, 1, 0, 0, 0, None, None, None),
        failure_reason="inference_failed",
    )
    with pytest.raises(ValueError, match="skipped item requires explicitly_skipped"):
        validate_cohort_items(build_identity(), (invalid,))


def test_quarantined_item_rejects_wrong_failure_reason() -> None:
    invalid = CohortItem(
        simfile_id="45",
        status="quarantined",
        reference_events=(BenchmarkEvent("45", 1.0, "kick", "ground_truth"),),
        prediction_events=None,
        coverage=CohortCoverage(1, 1, 0, 0, 0, None, None, None),
        failure_reason="inference_failed",
    )
    with pytest.raises(ValueError, match="quarantined item requires reference_quarantined"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_artifact_identity_rejects_bad_type() -> None:
    with pytest.raises(TypeError, match="artifact_identity must be CohortArtifactIdentity"):
        cohort_scoring._validate_artifact_identity(build_identity(), "not identity", "42")


def test_validate_artifact_identity_rejects_simfile_id_mismatch() -> None:
    artifact_identity = build_artifact_identity("42")
    with pytest.raises(ValueError, match="artifact simfile_id does not match"):
        cohort_scoring._validate_artifact_identity(build_identity(), artifact_identity, "99")


def test_artifact_identity_rejects_mixed_source_audio_content_hash(
    tmp_path: Path,
) -> None:
    native_events = (
        build_native_reference("13", 1.0, 0),
        NativeReferenceEvent(
            simfile_id=42,
            selected_chart_key="42/chart.dtx",
            selected_chart_content_hash="e" * 64,
            source_audio_key="42/audio.wav",
            source_audio_content_hash="a" * 64,
            source_order=1,
            measure=1,
            position=1.0,
            lane_id="12",
            note_id="snare-1",
            chart_time_sec=2.0,
            audio_time_sec=2.0,
        ),
    )
    mixed_reference = map_reference_events(native_events)
    with pytest.raises(ValueError, match="mixed source_audio_content_hash"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(),
            "42",
            mixed_reference,
            _artifact_for_song(tmp_path, "42"),
        )


def test_artifact_identity_rejects_invalid_backend_id(tmp_path: Path) -> None:
    artifact = _artifact_for_song(tmp_path, "42")
    tampered = dataclasses.replace(
        artifact,
        prediction=dataclasses.replace(
            artifact.prediction,
            descriptor=dataclasses.replace(
                artifact.prediction.descriptor,
                payload={**artifact.prediction.descriptor.payload, "backend_id": ""},
            ),
        ),
    )
    with pytest.raises(ValueError, match="backend_id is invalid"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(), "42", build_reference_mapping(), tampered
        )


def test_artifact_identity_rejects_invalid_model_id(tmp_path: Path) -> None:
    artifact = _artifact_for_song(tmp_path, "42")
    tampered = dataclasses.replace(
        artifact,
        prediction=dataclasses.replace(
            artifact.prediction,
            descriptor=dataclasses.replace(
                artifact.prediction.descriptor,
                payload={**artifact.prediction.descriptor.payload, "model_id": ""},
            ),
        ),
    )
    with pytest.raises(ValueError, match="model_id is invalid"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(), "42", build_reference_mapping(), tampered
        )


def test_artifact_identity_rejects_mixed_prediction_map_versions(
    tmp_path: Path,
) -> None:
    artifact = _artifact_for_song(tmp_path, "42")
    events = artifact.prediction.events
    tampered_event = dataclasses.replace(events[0], prediction_map_version="other-map")
    tampered = dataclasses.replace(
        artifact,
        prediction=dataclasses.replace(
            artifact.prediction,
            events=(tampered_event, *events[1:]),
        ),
    )
    with pytest.raises(ValueError, match="mixed prediction_map_version"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(), "42", build_reference_mapping(), tampered
        )


def test_artifact_identity_rejects_empty_non_oaf_prediction(tmp_path: Path) -> None:
    artifact = _empty_artifact_for_song(tmp_path, "42")
    tampered = dataclasses.replace(
        artifact,
        prediction=dataclasses.replace(
            artifact.prediction,
            descriptor=dataclasses.replace(
                artifact.prediction.descriptor,
                payload={**artifact.prediction.descriptor.payload, "backend_id": "other"},
            ),
        ),
    )
    with pytest.raises(ValueError, match="no prediction_map_version"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(), "42", build_reference_mapping(), tampered
        )


def test_success_item_rejects_missing_prediction_artifact_evidence(
    tmp_path: Path,
) -> None:
    item = cohort_scoring.cohort_item_from_artifacts(
        build_identity(), "42", build_reference_mapping(), _artifact_for_song(tmp_path, "42")
    )
    tampered = dataclasses.replace(item, prediction_artifact=None)
    with pytest.raises(ValueError, match="prediction artifact evidence"):
        validate_cohort_items(build_identity(), (tampered,))


def test_success_item_rejects_reference_duplicate_diagnostics_mismatch(
    tmp_path: Path,
) -> None:
    item = cohort_scoring.cohort_item_from_artifacts(
        build_identity(), "42", build_reference_mapping(), _artifact_for_song(tmp_path, "42")
    )
    tampered_diagnostics = dataclasses.replace(
        item.reference_artifact.diagnostics,
        duplicate_common_event_count=item.reference_artifact.diagnostics.duplicate_common_event_count
        + 1,
    )
    tampered = dataclasses.replace(
        item,
        reference_artifact=dataclasses.replace(
            item.reference_artifact, diagnostics=tampered_diagnostics
        ),
    )
    with pytest.raises(ValueError, match="reference duplicate diagnostics"):
        validate_cohort_items(build_identity(), (tampered,))


def test_success_item_rejects_reference_mapped_with_unknown_lane(
    tmp_path: Path,
) -> None:
    custom_lane_map = {**dict(DTX_LANE_MAP), "99": ClassMapping("kick", "kick")}
    native_events = (build_native_reference("99", 1.0, 0),)
    custom_reference = map_reference_events(native_events, lane_map=custom_lane_map)
    with pytest.raises(ValueError, match="not mapped using the frozen DTX lane map"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(), "42", custom_reference, _artifact_for_song(tmp_path, "42")
        )


def test_success_item_rejects_reference_common_class_mismatch(
    tmp_path: Path,
) -> None:
    custom_lane_map = {**dict(DTX_LANE_MAP), "13": ClassMapping("kick", "snare")}
    native_events = (build_native_reference("13", 1.0, 0),)
    custom_reference = map_reference_events(native_events, lane_map=custom_lane_map)
    with pytest.raises(ValueError, match="common_class does not match frozen DTX lane map"):
        cohort_scoring.cohort_item_from_artifacts(
            build_identity(), "42", custom_reference, _artifact_for_song(tmp_path, "42")
        )


def test_success_item_rejects_artifact_identity_not_matching_evidence(
    tmp_path: Path,
) -> None:
    item = cohort_scoring.cohort_item_from_artifacts(
        build_identity(), "42", build_reference_mapping(), _artifact_for_song(tmp_path, "42")
    )
    modified_identity = dataclasses.replace(build_identity(), input_view_id="other-view")
    modified_artifact_identity = dataclasses.replace(
        item.artifact_identity, input_view_id="other-view"
    )
    tampered = dataclasses.replace(item, artifact_identity=modified_artifact_identity)
    with pytest.raises(ValueError, match="artifact_identity does not match persisted"):
        validate_cohort_items(modified_identity, (tampered,))


def test_success_item_rejects_reference_events_not_matching_artifact(
    tmp_path: Path,
) -> None:
    item = cohort_scoring.cohort_item_from_artifacts(
        build_identity(), "42", build_reference_mapping(), _artifact_for_song(tmp_path, "42")
    )
    tampered_ref_event = dataclasses.replace(item.reference_events[0], time_sec=99.0)
    tampered = dataclasses.replace(
        item, reference_events=(tampered_ref_event, *item.reference_events[1:])
    )
    with pytest.raises(ValueError, match="reference_events do not match"):
        validate_cohort_items(build_identity(), (tampered,))


def test_success_item_rejects_prediction_events_not_matching_artifact(
    tmp_path: Path,
) -> None:
    item = cohort_scoring.cohort_item_from_artifacts(
        build_identity(), "42", build_reference_mapping(), _artifact_for_song(tmp_path, "42")
    )
    tampered_pred_event = dataclasses.replace(item.prediction_events[0], time_sec=99.0)
    tampered = dataclasses.replace(
        item, prediction_events=(tampered_pred_event, *item.prediction_events[1:])
    )
    with pytest.raises(ValueError, match="prediction_events do not match"):
        validate_cohort_items(build_identity(), (tampered,))


def test_success_item_rejects_coverage_not_matching_artifact(
    tmp_path: Path,
) -> None:
    item = cohort_scoring.cohort_item_from_artifacts(
        build_identity(), "42", build_reference_mapping(), _artifact_for_song(tmp_path, "42")
    )
    tampered_coverage = dataclasses.replace(
        item.coverage,
        prediction_native_class_counts=(("other_class", 99),),
    )
    tampered = dataclasses.replace(item, coverage=tampered_coverage)
    with pytest.raises(ValueError, match="coverage does not match"):
        validate_cohort_items(build_identity(), (tampered,))


def test_validate_event_sources_rejects_non_benchmark_reference_event() -> None:
    invalid = dataclasses.replace(build_item(), reference_events=("not an event",))
    with pytest.raises(TypeError, match="reference_events must contain BenchmarkEvent"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_event_sources_rejects_non_benchmark_prediction_event() -> None:
    invalid = dataclasses.replace(build_item(), prediction_events=("not an event",))
    with pytest.raises(TypeError, match="prediction_events must contain BenchmarkEvent"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_coverage_rejects_non_cohort_coverage() -> None:
    invalid = dataclasses.replace(build_item(prediction_events=None), coverage="not coverage")
    with pytest.raises(TypeError, match="coverage must be CohortCoverage"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_coverage_rejects_negative_reference_count() -> None:
    bad_coverage = CohortCoverage(-1, 0, 0, 0, 0, None, None, None)
    invalid = dataclasses.replace(build_item(prediction_events=None), coverage=bad_coverage)
    with pytest.raises(ValueError, match="reference coverage counts must be nonnegative"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_coverage_rejects_reference_common_count_mismatch() -> None:
    bad_coverage = CohortCoverage(3, 3, 0, 0, 0, None, None, None)
    invalid = dataclasses.replace(build_item(prediction_events=None), coverage=bad_coverage)
    with pytest.raises(ValueError, match="reference common count must match"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_coverage_rejects_partial_prediction_counts() -> None:
    bad_coverage = CohortCoverage(2, 2, 0, 0, 0, 1, None, None)
    invalid = dataclasses.replace(build_item(prediction_events=None), coverage=bad_coverage)
    with pytest.raises(ValueError, match="prediction coverage counts must be all present"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_coverage_rejects_negative_prediction_count() -> None:
    bad_coverage = CohortCoverage(2, 2, 0, 0, 0, -1, 0, 0)
    invalid = dataclasses.replace(build_item(prediction_events=None), coverage=bad_coverage)
    with pytest.raises(ValueError, match="prediction coverage counts must be nonnegative"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_coverage_rejects_empty_native_class_id() -> None:
    bad_coverage = CohortCoverage(2, 2, 0, 0, 0, 1, 1, 0, (("", 1),))
    invalid = dataclasses.replace(build_item(prediction_events=None), coverage=bad_coverage)
    with pytest.raises(ValueError, match="prediction native class ids must be nonempty"):
        validate_cohort_items(build_identity(), (invalid,))


def test_validate_coverage_rejects_negative_native_class_count() -> None:
    bad_coverage = CohortCoverage(2, 2, 0, 0, 0, 1, 1, 0, (("midi_36", -1),))
    invalid = dataclasses.replace(build_item(prediction_events=None), coverage=bad_coverage)
    with pytest.raises(ValueError, match="prediction native class counts must be nonnegative"):
        validate_cohort_items(build_identity(), (invalid,))


def test_score_cohort_rejects_non_cohort_identity() -> None:
    with pytest.raises(TypeError, match="identity must be CohortIdentity"):
        score_cohort("not identity", ())


@pytest.mark.parametrize(
    ("tolerances_ms", "match"),
    [
        ([50], "tolerances_ms must be a tuple"),
        ((), "tolerances_ms must not be empty"),
        ((True,), "tolerances_ms must contain positive integers"),
    ],
)
def test_score_cohort_rejects_invalid_tolerance_types(tolerances_ms: object, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        score_cohort(build_identity(), (), tolerances_ms=tolerances_ms)  # type: ignore[arg-type]


def test_score_cohort_rejects_non_tuple_diagnostics_for() -> None:
    with pytest.raises(TypeError, match="diagnostics_for must be a tuple"):
        score_cohort(build_identity(), (), diagnostics_for=["1"])  # type: ignore[arg-type]


def test_score_cohort_rejects_non_string_diagnostics_for_id() -> None:
    with pytest.raises(ValueError, match="diagnostics_for IDs must be strings"):
        score_cohort(build_identity(), (), diagnostics_for=(42,))  # type: ignore[arg-type]


def test_original_prediction_time_rejects_aligned_without_provenance() -> None:
    event = BenchmarkEvent("1", 1.0, "kick", "prediction", metadata={})
    with pytest.raises(ValueError, match="original time provenance"):
        cohort_scoring._original_prediction_time(event, "aligned")
