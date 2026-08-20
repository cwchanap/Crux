from __future__ import annotations

import csv
import math
from decimal import Decimal
from pathlib import Path

import pytest

from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    OAF_DESCRIPTOR_SCHEMA,
    StrictJsonError,
    build_descriptor,
    canonical_json_bytes,
    strict_json_loads,
)
from src.benchmark.backends import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.cohort_scoring import (
    CohortCoverage,
    CohortIdentity,
    CohortItem,
    cohort_item_from_artifacts,
    score_cohort,
)
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.models import BenchmarkEvent
from src.benchmark.prediction_artifact import (
    read_prediction_artifact,
    render_prediction_artifact,
)
from src.benchmark.reference_set import map_reference_events
from src.benchmark.reference_timing import NativeReferenceEvent
from src.benchmark.reports import (
    REPORT_SCHEMA,
    ReportArtifacts,
    ReportIntegrityError,
    _csv_decimal,
    _report_decimal,
    read_cohort_reports,
    write_cohort_reports,
)
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION


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


def _identity() -> CohortIdentity:
    return CohortIdentity(
        cohort_id="oaf-full-mix-v1",
        reference_manifest_sha256="a" * 64,
        reference_timing_version="sha256:" + "b" * 64,
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=OAF_BACKEND_ID,
        model_id="magenta-egmd-ckpt-569400-v1",
        model_lock_sha256="c" * 64,
        backend_descriptor_sha256=_descriptor().sha256,
        prediction_map_version="crux.prediction-map/oaf-egmd-8hit-v1",
        input_view_id="full-mix-v1",
    )


def _item(
    simfile_id: str,
    *,
    prediction_events: tuple[BenchmarkEvent, ...] | None,
    status: str = "success",
    failure_reason: str | None = None,
    warnings: tuple[str, ...] = (),
    prediction_native_event_count: int | None = None,
    prediction_mapped_event_count: int | None = None,
    prediction_unmapped_event_count: int | None = None,
    prediction_native_class_counts: tuple[tuple[str, int], ...] = (),
) -> CohortItem:
    reference_events = (BenchmarkEvent(simfile_id, 0.5, "kick", "ground_truth"),)
    if status == "success":
        assert prediction_events is not None
        reference = map_reference_events(
            (
                NativeReferenceEvent(
                    simfile_id=int(simfile_id),
                    selected_chart_key=f"{simfile_id}/chart.dtx",
                    selected_chart_content_hash="e" * 64,
                    source_audio_key=f"{simfile_id}/audio.wav",
                    source_audio_content_hash="f" * 64,
                    source_order=0,
                    measure=1,
                    position=0.0,
                    lane_id="13",
                    note_id="kick-0",
                    chart_time_sec=0.5,
                    audio_time_sec=0.5,
                ),
            )
        )
        group_by_class = {
            "kick": "kick",
            "snare": "snare",
            "hihat": "hihat",
            "tom": "toms",
            "crash": "crash",
            "ride": "ride",
        }
        native_events = [
            NativeEvent(
                time_sec=event.time_sec,
                native_class_id="midi_36",
                model_output_bin=15,
                native_midi_note=36,
                native_metadata={"upstream_8hit_group_id": group_by_class[event.canonical_class]},
                confidence=0.9,
                velocity_midi=100,
            )
            for index, event in enumerate(prediction_events)
        ]
        extra_native_count = (prediction_native_event_count or 0) - len(native_events)
        native_events.extend(
            NativeEvent(
                time_sec=9.0 + index,
                native_class_id="midi_75",
                model_output_bin=54,
                native_midi_note=75,
                native_metadata={"upstream_8hit_group_id": "sticks"},
                confidence=0.9,
                velocity_midi=100,
            )
            for index in range(max(0, extra_native_count))
        )
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
        mapped, _ = map_oaf_prediction(NativePrediction(audio, _descriptor(), tuple(native_events)))
        artifact = read_prediction_artifact(render_prediction_artifact(mapped))
        return cohort_item_from_artifacts(
            _identity(),
            simfile_id,
            reference,
            artifact,
            warnings=warnings,
        )
    return CohortItem(
        simfile_id=simfile_id,
        status=status,  # type: ignore[arg-type]
        reference_events=reference_events,
        prediction_events=prediction_events,
        coverage=CohortCoverage(
            reference_native_event_count=1,
            reference_common_event_count=1,
            reference_ignored_event_count=0,
            reference_unmapped_event_count=0,
            reference_duplicate_collapsed_count=0,
            prediction_native_event_count=prediction_native_event_count,
            prediction_mapped_event_count=prediction_mapped_event_count,
            prediction_unmapped_event_count=prediction_unmapped_event_count,
            prediction_native_class_counts=prediction_native_class_counts,
        ),
        warnings=warnings,
        failure_reason=failure_reason,  # type: ignore[arg-type]
    )


def _prediction(simfile_id: str, time_sec: float) -> BenchmarkEvent:
    return BenchmarkEvent(
        simfile_id,
        time_sec,
        "kick",
        "prediction",
        metadata={
            "input_view_id": "full-mix-v1",
            "prediction_map_version": "map-v1",
        },
    )


def _result(*, diagnostics_for: tuple[str, ...] = (), reverse_items: bool = False):
    success = _item(
        "1",
        prediction_events=(_prediction("1", 0.5), _prediction("1", 1.0)),
        warnings=("z-warning", "a-warning"),
        prediction_native_event_count=2,
    )
    partial = _item(
        "4",
        prediction_events=(_prediction("4", 0.5),),
        prediction_native_event_count=2,
    )
    empty = _item(
        "2",
        prediction_events=(),
        prediction_native_event_count=0,
    )
    failed = _item(
        "3",
        prediction_events=None,
        status="failed",
        failure_reason="prediction_missing",
    )
    items = (failed, empty, success, partial)
    if reverse_items:
        items = tuple(reversed(items))
    return score_cohort(_identity(), items, tolerances_ms=(50,), diagnostics_for=diagnostics_for)


def test_report_decimal_boundary() -> None:
    assert _report_decimal(0.5) == Decimal("0.500000")
    assert _report_decimal(None) is None
    assert _csv_decimal(0.5) == "0.5"
    assert _csv_decimal(None) == ""
    with pytest.raises(StrictJsonError):
        _report_decimal(math.nan)


def test_write_cohort_reports_outputs_six_contract_files(tmp_path: Path) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)

    assert artifacts.summary_json == tmp_path / "summary.json"
    assert artifacts.items_csv == tmp_path / "items.csv"
    assert artifacts.per_song_csv == tmp_path / "per_song.csv"
    assert artifacts.per_class_csv == tmp_path / "per_class.csv"
    assert artifacts.event_diagnostics_jsonl == tmp_path / "event_diagnostics.jsonl"
    assert artifacts.summary_markdown == tmp_path / "summary.md"

    summary = strict_json_loads(artifacts.summary_json.read_bytes(), require_canonical=True)
    assert set(summary) == {"schema", "identity", "tolerances_ms", "population", "aggregates"}
    assert summary["schema"] == REPORT_SCHEMA
    assert "items" not in summary
    assert summary["aggregates"][0]["event_micro"]["precision"] == Decimal("0.666667")
    assert summary["aggregates"][0]["song_macro_f1"] is not None

    with artifacts.items_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["simfile_id"] == "1"
    assert rows[0]["warnings"] == "a-warning|z-warning"
    assert rows[0]["prediction_mapping_coverage"] == "1"
    partial_row = next(row for row in rows if row["simfile_id"] == "4")
    assert partial_row["prediction_mapping_coverage"] == "0.5"
    empty_row = next(row for row in rows if row["simfile_id"] == "2")
    assert empty_row["prediction_mapping_coverage"] == ""
    assert all(row["prediction_mapping_coverage"] == "" for row in rows if row["simfile_id"] == "3")


def test_per_song_and_per_class_are_song_scoped(tmp_path: Path) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)
    with artifacts.per_song_csv.open(newline="", encoding="utf-8") as handle:
        song_rows = list(csv.DictReader(handle))
    with artifacts.per_class_csv.open(newline="", encoding="utf-8") as handle:
        class_rows = list(csv.DictReader(handle))

    assert song_rows
    assert song_rows[0]["precision"] == "0.5"
    assert "backend_descriptor_sha256" not in song_rows[0]
    assert class_rows
    assert "scope" not in class_rows[0]
    assert {row["simfile_id"] for row in class_rows} == {"1", "2", "4"}


def test_event_diagnostics_are_slim_canonical_jsonl(tmp_path: Path) -> None:
    artifacts = write_cohort_reports(_result(diagnostics_for=("1",)), tmp_path)
    lines = artifacts.event_diagnostics_jsonl.read_bytes().splitlines()
    assert lines
    expected = (
        b'{"cohort_id":"oaf-full-mix-v1","common_class":"kick","mode":"raw",'
        b'"outcome":"matched","prediction_time_sec":0.5,"reference_time_sec":0.5,'
        b'"scored_prediction_time_sec":0.5,"simfile_id":"1","timing_error_sec":0,'
        b'"tolerance_ms":50}'
    )
    assert expected in lines
    assert set(strict_json_loads(lines[0], require_canonical=True)) == {
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


def test_default_diagnostics_are_empty(tmp_path: Path) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)
    assert artifacts.event_diagnostics_jsonl.read_bytes() == b""


_REPORT_ARTIFACT_FIELDS = (
    "summary_json",
    "items_csv",
    "per_song_csv",
    "per_class_csv",
    "event_diagnostics_jsonl",
    "summary_markdown",
)


def _assert_same_bytes(first: ReportArtifacts, second: ReportArtifacts) -> None:
    for first_path, second_path in zip(
        (getattr(first, field) for field in _REPORT_ARTIFACT_FIELDS),
        (getattr(second, field) for field in _REPORT_ARTIFACT_FIELDS),
        strict=True,
    ):
        assert first_path.read_bytes() == second_path.read_bytes()


def test_report_bytes_are_deterministic_for_same_result(tmp_path: Path) -> None:
    first = write_cohort_reports(_result(diagnostics_for=("1",)), tmp_path / "first")
    second = write_cohort_reports(_result(diagnostics_for=("1",)), tmp_path / "second")
    _assert_same_bytes(first, second)


def test_reversed_input_order_keeps_all_report_bytes_identical(tmp_path: Path) -> None:
    first = write_cohort_reports(_result(diagnostics_for=("1",)), tmp_path / "first")
    second = write_cohort_reports(
        _result(diagnostics_for=("1",), reverse_items=True), tmp_path / "second"
    )
    _assert_same_bytes(first, second)


def _zero_success_result():
    failed = _item(
        "1",
        prediction_events=None,
        status="failed",
        failure_reason="prediction_missing",
    )
    quarantined = _item(
        "2",
        prediction_events=None,
        status="quarantined",
        failure_reason="reference_quarantined",
    )
    return score_cohort(_identity(), (failed, quarantined), tolerances_ms=(50,))


def test_write_cohort_reports_renders_empty_markdown_sections_for_zero_success(
    tmp_path: Path,
) -> None:
    artifacts = write_cohort_reports(_zero_success_result(), tmp_path)
    markdown = artifacts.summary_markdown.read_text(encoding="utf-8")
    assert "No supported classes." in markdown
    assert "No successful songs." in markdown


def test_write_cohort_reports_rejects_non_cohort_score_result(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="result must be CohortScoreResult"):
        write_cohort_reports("not a result", tmp_path)  # type: ignore[arg-type]


def test_read_cohort_reports_parses_all_published_rows_and_summary_aggregates(
    tmp_path: Path,
) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)

    reports = read_cohort_reports(tmp_path, expected_identity=_identity())

    assert reports.identity == _identity()
    assert reports.population.total_count == 4
    assert reports.population.success_count == 3
    assert [row.simfile_id for row in reports.items] == ["1", "2", "3", "4"]
    assert reports.items[2].failure_reason == "prediction_missing"
    assert reports.songs[0].simfile_id == "1"
    assert reports.songs[0].precision == Decimal("0.5")
    assert reports.classes[0].common_class == "kick"
    assert reports.aggregates[0].event_micro.true_positives == 2
    assert reports.aggregates[0].event_micro.false_positives == 1
    assert artifacts.summary_json.exists()


def test_read_cohort_reports_uses_summary_event_micro_without_recomputing_csv(
    tmp_path: Path,
) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)
    summary = strict_json_loads(artifacts.summary_json.read_bytes(), require_canonical=True)
    summary["aggregates"][0]["event_micro"]["tp"] = 99
    artifacts.summary_json.write_bytes(canonical_json_bytes(summary))

    reports = read_cohort_reports(tmp_path, expected_identity=_identity())

    assert reports.aggregates[0].event_micro.true_positives == 99


def test_read_cohort_reports_rejects_invalid_csv_schema(tmp_path: Path) -> None:
    write_cohort_reports(_result(), tmp_path)
    path = tmp_path / "items.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("wrong\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")

    with pytest.raises(ReportIntegrityError, match="invalid column schema"):
        read_cohort_reports(tmp_path, expected_identity=_identity())


def test_read_cohort_reports_rejects_bad_number(tmp_path: Path) -> None:
    write_cohort_reports(_result(), tmp_path)
    path = tmp_path / "per_song.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace(",0.5,", ",not-a-number,")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ReportIntegrityError, match="numeric"):
        read_cohort_reports(tmp_path, expected_identity=_identity())


def test_read_cohort_reports_rejects_identity_mismatch(tmp_path: Path) -> None:
    write_cohort_reports(_result(), tmp_path)
    path = tmp_path / "per_song.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("oaf-full-mix-v1,", "other-cohort,", 1),
        encoding="utf-8",
    )

    with pytest.raises(ReportIntegrityError, match="identity mismatch"):
        read_cohort_reports(tmp_path, expected_identity=_identity())


def test_read_cohort_reports_rejects_duplicate_score_rows(tmp_path: Path) -> None:
    write_cohort_reports(_result(), tmp_path)
    path = tmp_path / "per_song.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")

    with pytest.raises(ReportIntegrityError, match="duplicate"):
        read_cohort_reports(tmp_path, expected_identity=_identity())


def test_read_cohort_reports_rejects_score_row_for_non_success_item(tmp_path: Path) -> None:
    write_cohort_reports(_result(), tmp_path)
    path = tmp_path / "per_song.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    failed_row = lines[1].replace(",1,", ",3,", 1)
    path.write_text("\n".join(lines + [failed_row]) + "\n", encoding="utf-8")

    with pytest.raises(ReportIntegrityError, match="non-success item"):
        read_cohort_reports(tmp_path, expected_identity=_identity())


def test_read_cohort_reports_rejects_incomplete_success_song_grid(tmp_path: Path) -> None:
    write_cohort_reports(_result(), tmp_path)
    path = tmp_path / "per_song.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], *lines[2:]]) + "\n", encoding="utf-8")

    with pytest.raises(ReportIntegrityError, match="score grid"):
        read_cohort_reports(tmp_path, expected_identity=_identity())


def test_read_cohort_reports_rejects_incomplete_aggregate_mode_grid(tmp_path: Path) -> None:
    artifacts = write_cohort_reports(_result(), tmp_path)
    summary = strict_json_loads(artifacts.summary_json.read_bytes(), require_canonical=True)
    summary["aggregates"] = summary["aggregates"][:1]
    artifacts.summary_json.write_bytes(canonical_json_bytes(summary))

    with pytest.raises(ReportIntegrityError, match="mode grid"):
        read_cohort_reports(tmp_path, expected_identity=_identity())


@pytest.mark.parametrize(
    ("needle", "replacement"),
    (
        (",50,raw,", ",050,raw,"),
        (",0.5,", ",0.50,"),
        (
            ",50,raw,1,0,0,1,1,1,1,0,",
            ",50,raw,1,0,0,1e-1,1,1,1,0,",
        ),
    ),
)
def test_read_cohort_reports_rejects_noncanonical_csv_numbers(
    tmp_path: Path, needle: str, replacement: str
) -> None:
    write_cohort_reports(_result(), tmp_path)
    path = tmp_path / "per_song.csv"
    content = path.read_text(encoding="utf-8")
    assert needle in content
    path.write_text(content.replace(needle, replacement, 1), encoding="utf-8")

    with pytest.raises(ReportIntegrityError, match="canonical"):
        read_cohort_reports(tmp_path, expected_identity=_identity())


def _add_snare_to_aggregate_per_class(artifacts_dir: Path) -> None:
    """Add a snare class to the aggregate per_class in summary.json."""
    summary = strict_json_loads(
        (artifacts_dir / "summary.json").read_bytes(), require_canonical=True
    )
    for aggregate in summary["aggregates"]:
        aggregate["per_class"] = sorted(
            [*aggregate["per_class"], _snare_aggregate_class()],
            key=lambda entry: entry["common_class"],
        )
    (artifacts_dir / "summary.json").write_bytes(canonical_json_bytes(summary))


def _snare_aggregate_class() -> dict[str, object]:
    return {
        "common_class": "snare",
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": None,
        "recall": None,
        "f1": None,
        "reference_support": 0,
        "prediction_support": 0,
    }


def _per_class_row(simfile_id: str, common_class: str) -> str:
    identity = _identity()
    return (
        f"{identity.cohort_id},{identity.model_id},{identity.model_lock_sha256},"
        f"{identity.prediction_map_version},{identity.input_view_id},"
        f"{identity.scoring_version},{simfile_id},50,raw,{common_class},"
        "0,0,0,0,0,,,"
    )


def test_read_cohort_reports_accepts_heterogeneous_per_class(tmp_path: Path) -> None:
    """A song may omit classes that another song in the cohort contains."""
    write_cohort_reports(_result(), tmp_path)
    _add_snare_to_aggregate_per_class(tmp_path)

    # Add snare rows for song "1" under both modes; songs "2" and "4" remain
    # kick-only.  The class set for song "1" is {kick, snare} across both
    # tolerances/modes, while songs "2" and "4" have {kick} — a valid
    # heterogeneous cohort.
    path = tmp_path / "per_class.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    snare_raw = _per_class_row("1", "snare")
    snare_aligned = _per_class_row("1", "snare").replace(",50,raw,", ",50,aligned,")
    path.write_text(
        "\n".join([lines[0], lines[1], snare_raw, snare_aligned, *lines[2:]]) + "\n",
        encoding="utf-8",
    )

    reports = read_cohort_reports(tmp_path, expected_identity=_identity())

    assert {row.common_class for row in reports.classes} == {"kick", "snare"}
    assert {row.simfile_id for row in reports.classes if row.common_class == "snare"} == {"1"}


def test_read_cohort_reports_rejects_per_class_absent_from_aggregates(
    tmp_path: Path,
) -> None:
    write_cohort_reports(_result(), tmp_path)

    path = tmp_path / "per_class.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    extra = _per_class_row("1", "snare")
    path.write_text("\n".join([lines[0], lines[1], extra, *lines[2:]]) + "\n", encoding="utf-8")

    with pytest.raises(ReportIntegrityError, match="absent from aggregates"):
        read_cohort_reports(tmp_path, expected_identity=_identity())


def test_read_cohort_reports_rejects_per_class_for_unexpected_combination(
    tmp_path: Path,
) -> None:
    write_cohort_reports(_result(), tmp_path)

    path = tmp_path / "per_class.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    # Tolerance 999 is not in the published tolerances.
    bad_row = _per_class_row("1", "kick").replace(",50,raw,", ",999,raw,")
    path.write_text("\n".join([lines[0], lines[1], bad_row, *lines[2:]]) + "\n", encoding="utf-8")

    with pytest.raises(ReportIntegrityError, match="unexpected score combination"):
        read_cohort_reports(tmp_path, expected_identity=_identity())


def test_read_cohort_reports_rejects_inconsistent_per_song_class_set(
    tmp_path: Path,
) -> None:
    write_cohort_reports(_result(), tmp_path)
    _add_snare_to_aggregate_per_class(tmp_path)

    # The fixture already has both raw and aligned modes.  Add snare for
    # song "1" under raw only — the class set for song "1" becomes
    # {kick, snare} for raw but {kick} for aligned, which is inconsistent.
    path = tmp_path / "per_class.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    snare_raw = _per_class_row("1", "snare")
    path.write_text(
        "\n".join([lines[0], lines[1], snare_raw, *lines[2:]]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ReportIntegrityError, match="inconsistent"):
        read_cohort_reports(tmp_path, expected_identity=_identity())
