from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    canonical_json_bytes,
)
from src.benchmark.cohort_scoring import SCORING_VERSION
from src.benchmark.muscriptor_comparison import (
    ComparisonIntegrityError,
    ComparisonOutcome,
    ComparisonRequest,
    _ClassRow,
    _csv_decimal,
    _hash,
    _load_evidence,
    _markdown_metric,
    _metric_delta,
    _pairable_ids,
    _paired_class_rows,
    _paired_song_rows,
    _parse_class_rows,
    _parse_decimal,
    _parse_int,
    _parse_items,
    _parse_metric,
    _parse_run,
    _parse_run_identity,
    _parse_simfile_id,
    _parse_song_rows,
    _read_csv,
    _report_identity_from_snapshot,
    _Reports,
    _RunEvidence,
    _RunIdentity,
    _RunItem,
    _SongRow,
    _subset_ids,
    _text,
    _validate_manifest_lineage,
    _validate_pair_run_identity,
    compare_oaf_muscriptor,
    load_reviewed_subset_manifest,
)
from src.benchmark.muscriptor_corpus_run import MUSCRIPTOR_CORPUS_RUN_SCHEMA
from src.benchmark.oaf_corpus_run import OAF_CORPUS_RUN_SCHEMA
from src.benchmark.published_comparison import (
    PublishedRunEvidence,
    comparison_summary,
    pairable_success_ids,
    paired_song_rows,
    write_markdown,
)
from tests.benchmark.test_muscriptor_comparison import (
    _CLASS_FIELDS,
    _ITEM_FIELDS,
    _SONG_FIELDS,
    _reports,
    _write_csv,
    _write_run,
)

_REF_SHA = "a" * 64
_TIMING_SHA = "b" * 64
_REF_VERSION = "hpa324-v1"
_TIMING_VERSION = "hpa323-v1"
_MODEL_LOCK = "e" * 64
_COHORT = "oaf-run"
_MODEL = "oaf-model"
_MAP = "oaf-map"
_VIEW = "crux.full-mix/v1"


@pytest.fixture
def manifest_loaders(monkeypatch: pytest.MonkeyPatch):
    import src.benchmark.muscriptor_comparison as comparison

    monkeypatch.setattr(
        comparison,
        "load_reference_set_manifest",
        lambda _path: SimpleNamespace(manifest_sha256=_REF_SHA, corpus_version=_REF_VERSION),
    )
    monkeypatch.setattr(
        comparison,
        "load_reference_timing_manifest",
        lambda _path: SimpleNamespace(manifest_sha256=_TIMING_SHA, corpus_version=_TIMING_VERSION),
    )


def _identity(**overrides: object) -> _RunIdentity:
    base: dict[str, object] = {
        "cohort_id": _COHORT,
        "model_id": _MODEL,
        "backend_id": OAF_BACKEND_ID,
        "model_lock_sha256": _MODEL_LOCK,
        "prediction_map_version": _MAP,
        "input_view_id": _VIEW,
        "reference_manifest_sha256": _REF_SHA,
        "reference_manifest_version": _REF_VERSION,
        "reference_timing_manifest_sha256": _TIMING_SHA,
        "reference_timing_version": _TIMING_VERSION,
        "scoring_version": SCORING_VERSION,
    }
    base.update(overrides)
    return _RunIdentity(**base)  # type: ignore[arg-type]


def _report_identity() -> dict[str, str]:
    return {
        "cohort_id": _COHORT,
        "model_id": _MODEL,
        "model_lock_sha256": _MODEL_LOCK,
        "prediction_map_version": _MAP,
        "input_view_id": _VIEW,
        "scoring_version": SCORING_VERSION,
    }


def _items_row(
    *,
    cohort: str = _COHORT,
    simfile_id: str = "1",
    status: str = "success",
    **overrides: str,
) -> dict[str, str]:
    row = {field: "" for field in _ITEM_FIELDS}
    row.update(
        {
            "cohort_id": cohort,
            "simfile_id": simfile_id,
            "status": status,
            "reference_native_event_count": "1",
            "reference_common_event_count": "1",
            "reference_ignored_event_count": "0",
            "reference_unmapped_event_count": "0",
            "reference_duplicate_collapsed_count": "0",
        }
    )
    if status == "success":
        row.update(
            {
                "prediction_native_event_count": "1",
                "prediction_mapped_event_count": "1",
                "prediction_unmapped_event_count": "0",
                "prediction_mapping_coverage": "1",
                "prediction_native_class_counts": "kick=1",
            }
        )
    elif status == "failed":
        row["failure_reason"] = "prediction_missing"
    elif status == "skipped":
        row["failure_reason"] = "explicitly_skipped"
    elif status == "quarantined":
        row["failure_reason"] = "reference_quarantined"
    row.update(overrides)
    return row


def _song_row(
    *,
    simfile_id: str = "1",
    mode: str = "raw",
    **overrides: str,
) -> dict[str, str]:
    row = {field: "" for field in _SONG_FIELDS}
    row.update(
        {
            **_report_identity(),
            "simfile_id": simfile_id,
            "tolerance_ms": "50",
            "mode": mode,
            "tp": "1",
            "fp": "1",
            "fn": "1",
            "precision": "0.5",
            "recall": "0.4",
            "f1": "0.4",
            "prediction_to_reference_ratio": "1",
            "offset_ms": "0",
        }
    )
    row.update(overrides)
    return row


def _class_row(
    *,
    simfile_id: str = "1",
    mode: str = "raw",
    common_class: str = "kick",
    **overrides: str,
) -> dict[str, str]:
    row = {field: "" for field in _CLASS_FIELDS}
    row.update(
        {
            **_report_identity(),
            "simfile_id": simfile_id,
            "tolerance_ms": "50",
            "mode": mode,
            "common_class": common_class,
            "tp": "1",
            "fp": "0",
            "fn": "1",
            "reference_support": "2",
            "prediction_support": "1",
            "precision": "0.5",
            "recall": "0.5",
            "f1": "0.5",
        }
    )
    row.update(overrides)
    return row


def _write_items(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "items.csv"
    _write_csv(path, _ITEM_FIELDS, rows)
    return path


def _write_songs(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "per_song.csv"
    _write_csv(path, _SONG_FIELDS, rows)
    return path


def _write_classes(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "per_class.csv"
    _write_csv(path, _CLASS_FIELDS, rows)
    return path


def _write_raw_run(path: Path, schema: str = OAF_CORPUS_RUN_SCHEMA) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes({"schema": schema}))


def _patch_oaf_parser(monkeypatch: pytest.MonkeyPatch, snapshot: dict[str, object]) -> None:
    import src.benchmark.oaf_corpus_run as oaf_run

    monkeypatch.setattr(oaf_run, "parse_oaf_corpus_run", lambda _content: snapshot)


def _base_snapshot(**overrides: object) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "run_id": _COHORT,
        "backend_descriptor": {"backend_id": OAF_BACKEND_ID},
        "model_id": _MODEL,
        "model_lock_sha256": _MODEL_LOCK,
        "prediction_map_version": _MAP,
        "input_view_id": _VIEW,
        "reference_manifest_sha256": _REF_SHA,
        "reference_manifest_version": _REF_VERSION,
        "reference_timing_manifest_sha256": _TIMING_SHA,
        "reference_timing_version": _TIMING_VERSION,
        "items": [],
    }
    snapshot.update(overrides)
    return snapshot


def test_load_reviewed_subset_manifest_delegates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = SimpleNamespace(marker="loaded")
    import src.benchmark.reviewed_subset as reviewed_subset

    monkeypatch.setattr(reviewed_subset, "load_reviewed_subset_manifest", lambda _path: sentinel)
    path = tmp_path / "subset.json"
    assert load_reviewed_subset_manifest(path) is sentinel


def test_comparison_request_rejects_non_path_fields(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="oaf_run_path must be a Path"):
        ComparisonRequest(
            oaf_run_path="not-a-path",  # type: ignore[arg-type]
            muscriptor_run_path=tmp_path,
            reference_manifest_path=tmp_path,
            timing_manifest_path=tmp_path,
            output_dir=tmp_path,
        )


def test_comparison_request_rejects_non_path_subset(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="subset_manifest_path must be a Path or None"):
        ComparisonRequest(
            oaf_run_path=tmp_path,
            muscriptor_run_path=tmp_path,
            reference_manifest_path=tmp_path,
            timing_manifest_path=tmp_path,
            output_dir=tmp_path,
            subset_manifest_path="not-a-path",  # type: ignore[arg-type]
        )


def test_comparison_outcome_rejects_nonzero_exit_code() -> None:
    with pytest.raises(ValueError, match="exit_code must be 0"):
        ComparisonOutcome(exit_code=1)  # type: ignore[arg-type]


def test_comparison_outcome_rejects_non_path_output_dir() -> None:
    with pytest.raises(TypeError, match="output_dir must be a Path"):
        ComparisonOutcome(output_dir="not-a-path")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field", ["pairable_success_count", "paired_song_count", "paired_class_count"]
)
def test_comparison_outcome_rejects_negative_counts(field: str) -> None:
    with pytest.raises(ValueError, match="must be a nonnegative integer"):
        ComparisonOutcome(**{field: -1})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field", ["pairable_success_count", "paired_song_count", "paired_class_count"]
)
def test_comparison_outcome_rejects_bool_counts(field: str) -> None:
    with pytest.raises(ValueError, match="must be a nonnegative integer"):
        ComparisonOutcome(**{field: True})  # type: ignore[arg-type]


def test_text_rejects_empty_and_non_string() -> None:
    with pytest.raises(ComparisonIntegrityError, match="nonempty string"):
        _text("", "run_id")
    with pytest.raises(ComparisonIntegrityError, match="nonempty string"):
        _text(123, "run_id")  # type: ignore[arg-type]


def test_hash_rejects_bad_sha256() -> None:
    with pytest.raises(ComparisonIntegrityError, match="model_lock_sha256"):
        _hash("not-a-hash", "model_lock_sha256")
    with pytest.raises(ComparisonIntegrityError, match="model_lock_sha256"):
        _hash(None, "model_lock_sha256")


def test_parse_simfile_id_rejects_non_digit() -> None:
    with pytest.raises(ComparisonIntegrityError, match="malformed"):
        _parse_simfile_id("abc")


def test_parse_simfile_id_rejects_zero_and_leading_zeros() -> None:
    with pytest.raises(ComparisonIntegrityError, match="malformed"):
        _parse_simfile_id("0")
    with pytest.raises(ComparisonIntegrityError, match="malformed"):
        _parse_simfile_id("01")


def test_parse_int_rejects_non_digit() -> None:
    with pytest.raises(ComparisonIntegrityError, match="malformed"):
        _parse_int("abc", "tp")


def test_parse_int_rejects_nonpositive_when_positive_required() -> None:
    with pytest.raises(ComparisonIntegrityError, match="malformed"):
        _parse_int("0", "tolerance_ms", positive=True)


def test_parse_decimal_rejects_whitespace_and_non_string() -> None:
    with pytest.raises(ComparisonIntegrityError, match="malformed"):
        _parse_decimal(" 1", "precision")
    with pytest.raises(ComparisonIntegrityError, match="malformed"):
        _parse_decimal(123, "precision")  # type: ignore[arg-type]


def test_parse_decimal_rejects_non_finite() -> None:
    with pytest.raises(ComparisonIntegrityError, match="malformed"):
        _parse_decimal("Infinity", "precision")


def test_parse_metric_rejects_out_of_range_precision() -> None:
    with pytest.raises(ComparisonIntegrityError, match="out of range"):
        _parse_metric("2", "precision")


def test_parse_metric_rejects_negative_ratio() -> None:
    with pytest.raises(ComparisonIntegrityError, match="out of range"):
        _parse_metric("-1", "prediction_to_reference_ratio")


def test_read_csv_rejects_unreadable_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(ComparisonIntegrityError, match="cannot read report"):
        _read_csv(missing, _ITEM_FIELDS)


def test_read_csv_rejects_invalid_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    _write_csv(path, ("wrong", "columns"), [{"wrong": "x", "columns": "y"}])
    with pytest.raises(ComparisonIntegrityError, match="invalid column schema"):
        _read_csv(path, _ITEM_FIELDS)


def test_read_csv_rejects_row_with_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "extra.csv"
    path.write_text("a,b\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ComparisonIntegrityError, match="malformed row"):
        _read_csv(path, ("a", "b"))


def test_read_csv_rejects_malformed_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ok.csv"
    _write_csv(path, ("a",), [{"a": "x"}])

    class _ErrorReader:
        fieldnames = ("a",)

        def __iter__(self):
            raise csv.Error("boom")

    monkeypatch.setattr(csv, "DictReader", lambda *args, **kwargs: _ErrorReader())
    with pytest.raises(ComparisonIntegrityError, match="malformed CSV"):
        _read_csv(path, ("a",))


def test_parse_items_rejects_cohort_mismatch(tmp_path: Path) -> None:
    path = _write_items(tmp_path, [_items_row(cohort="other-cohort")])
    with pytest.raises(ComparisonIntegrityError, match="identity mismatch for cohort_id"):
        _parse_items(path, _identity())


def test_parse_items_rejects_duplicate_simfile_id(tmp_path: Path) -> None:
    path = _write_items(tmp_path, [_items_row(simfile_id="1"), _items_row(simfile_id="1")])
    with pytest.raises(ComparisonIntegrityError, match="duplicate simfile_id"):
        _parse_items(path, _identity())


def test_parse_items_rejects_invalid_status(tmp_path: Path) -> None:
    path = _write_items(tmp_path, [_items_row(status="bogus")])
    with pytest.raises(ComparisonIntegrityError, match="invalid status"):
        _parse_items(path, _identity())


def test_parse_items_parses_nonempty_prediction_counts(tmp_path: Path) -> None:
    path = _write_items(
        tmp_path,
        [
            _items_row(
                prediction_native_event_count="5",
                prediction_mapped_event_count="3",
                prediction_unmapped_event_count="1",
            )
        ],
    )
    assert _parse_items(path, _identity()) == {"1": "success"}


def test_parse_items_parses_native_class_counts(tmp_path: Path) -> None:
    path = _write_items(
        tmp_path,
        [_items_row(prediction_native_class_counts="kick=1|snare=2")],
    )
    assert _parse_items(path, _identity()) == {"1": "success"}


def test_parse_items_rejects_class_counts_without_equals(tmp_path: Path) -> None:
    path = _write_items(tmp_path, [_items_row(prediction_native_class_counts="noequals")])
    with pytest.raises(ComparisonIntegrityError, match="malformed"):
        _parse_items(path, _identity())


def test_parse_items_rejects_class_counts_with_empty_class(tmp_path: Path) -> None:
    path = _write_items(tmp_path, [_items_row(prediction_native_class_counts="=1")])
    with pytest.raises(ComparisonIntegrityError, match="malformed"):
        _parse_items(path, _identity())


def test_parse_song_rows_rejects_non_success_item(tmp_path: Path) -> None:
    path = _write_songs(tmp_path, [_song_row()])
    with pytest.raises(ComparisonIntegrityError, match="non-success item"):
        _parse_song_rows(path, _identity(), successful_ids=set())


def test_parse_song_rows_rejects_invalid_mode(tmp_path: Path) -> None:
    path = _write_songs(tmp_path, [_song_row(mode="bogus")])
    with pytest.raises(ComparisonIntegrityError, match="invalid mode"):
        _parse_song_rows(path, _identity(), successful_ids={"1"})


def test_parse_class_rows_rejects_non_success_item(tmp_path: Path) -> None:
    path = _write_classes(tmp_path, [_class_row()])
    with pytest.raises(ComparisonIntegrityError, match="non-success item"):
        _parse_class_rows(path, _identity(), successful_ids=set())


def test_parse_class_rows_rejects_invalid_mode(tmp_path: Path) -> None:
    path = _write_classes(tmp_path, [_class_row(mode="bogus")])
    with pytest.raises(ComparisonIntegrityError, match="invalid mode"):
        _parse_class_rows(path, _identity(), successful_ids={"1"})


def test_parse_class_rows_rejects_empty_common_class(tmp_path: Path) -> None:
    path = _write_classes(tmp_path, [_class_row(common_class="")])
    with pytest.raises(ComparisonIntegrityError, match="invalid common_class"):
        _parse_class_rows(path, _identity(), successful_ids={"1"})


def test_parse_class_rows_rejects_duplicate_key(tmp_path: Path) -> None:
    path = _write_classes(tmp_path, [_class_row(), _class_row()])
    with pytest.raises(ComparisonIntegrityError, match="duplicate score key"):
        _parse_class_rows(path, _identity(), successful_ids={"1"})


def test_parse_run_identity_rejects_inconsistent_prediction_map() -> None:
    snapshot = _base_snapshot(inference_config={"prediction_map_version": "other-map"})
    with pytest.raises(ComparisonIntegrityError, match="internally inconsistent"):
        _parse_run_identity(snapshot)


def test_parse_run_identity_rejects_inconsistent_input_view() -> None:
    snapshot = _base_snapshot(inference_config={"input_view_id": "other-view"})
    with pytest.raises(ComparisonIntegrityError, match="internally inconsistent"):
        _parse_run_identity(snapshot)


def test_parse_run_identity_rejects_invalid_scoring_version() -> None:
    snapshot = _base_snapshot(scoring_version="bad-version")
    with pytest.raises(ComparisonIntegrityError, match="scoring_version is invalid"):
        _parse_run_identity(snapshot)


def test_parse_run_identity_rejects_non_string_reference_manifest_version() -> None:
    snapshot = _base_snapshot(reference_manifest_version=123)
    with pytest.raises(
        ComparisonIntegrityError, match="reference_manifest_version must be a string"
    ):
        _parse_run_identity(snapshot)


def test_parse_run_rejects_non_object_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_bytes(canonical_json_bytes([1, 2, 3]))
    with pytest.raises(ComparisonIntegrityError, match="run snapshot must be an object"):
        _parse_run(path)


def test_parse_run_rejects_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_bytes(canonical_json_bytes({"schema": "crux.unknown/v1"}))
    with pytest.raises(ComparisonIntegrityError, match="schema is unsupported"):
        _parse_run(path)


def test_parse_run_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_bytes(b"not valid json")
    with pytest.raises(ComparisonIntegrityError, match="invalid run snapshot"):
        _parse_run(path)


def test_parse_run_rejects_items_not_a_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    _write_raw_run(path)
    _patch_oaf_parser(monkeypatch, _base_snapshot(items="not-a-list"))
    with pytest.raises(ComparisonIntegrityError, match="items must be an array"):
        _parse_run(path)


def test_parse_run_rejects_non_object_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "run.json"
    _write_raw_run(path)
    _patch_oaf_parser(monkeypatch, _base_snapshot(items=["not-an-object"]))
    with pytest.raises(ComparisonIntegrityError, match="item must be an object"):
        _parse_run(path)


def test_parse_run_rejects_malformed_simfile_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    _write_raw_run(path)
    _patch_oaf_parser(
        monkeypatch,
        _base_snapshot(items=[{"simfile_id": 0, "execution_disposition": "inferred"}]),
    )
    with pytest.raises(ComparisonIntegrityError, match="simfile_id is malformed"):
        _parse_run(path)


def test_parse_run_rejects_duplicate_simfile_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    _write_raw_run(path)
    _patch_oaf_parser(
        monkeypatch,
        _base_snapshot(
            items=[
                {
                    "simfile_id": 1,
                    "execution_disposition": "inferred",
                    "source_audio_sha256": _REF_SHA,
                    "input_audio_sha256": _REF_SHA,
                },
                {
                    "simfile_id": 1,
                    "execution_disposition": "inferred",
                    "source_audio_sha256": _REF_SHA,
                    "input_audio_sha256": _REF_SHA,
                },
            ]
        ),
    )
    with pytest.raises(ComparisonIntegrityError, match="duplicate simfile_id"):
        _parse_run(path)


def test_parse_run_rejects_invalid_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    _write_raw_run(path)
    _patch_oaf_parser(
        monkeypatch,
        _base_snapshot(items=[{"simfile_id": 1, "execution_disposition": "bogus"}]),
    )
    with pytest.raises(ComparisonIntegrityError, match="execution disposition"):
        _parse_run(path)


def test_parse_run_rejects_non_string_source_hash_on_failed_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    _write_raw_run(path)
    _patch_oaf_parser(
        monkeypatch,
        _base_snapshot(
            items=[
                {
                    "simfile_id": 1,
                    "execution_disposition": "failed",
                    "source_audio_sha256": 123,
                }
            ]
        ),
    )
    with pytest.raises(ComparisonIntegrityError, match="source_audio_sha256 is malformed"):
        _parse_run(path)


def test_parse_run_rejects_non_string_input_hash_on_failed_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    _write_raw_run(path)
    _patch_oaf_parser(
        monkeypatch,
        _base_snapshot(
            items=[
                {
                    "simfile_id": 1,
                    "execution_disposition": "failed",
                    "source_audio_sha256": None,
                    "input_audio_sha256": 123,
                }
            ]
        ),
    )
    with pytest.raises(ComparisonIntegrityError, match="input_audio_sha256 is malformed"):
        _parse_run(path)


def test_load_evidence_rejects_population_mismatch(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "oaf",
        model=_MODEL,
        run_id=_COHORT,
        schema=OAF_CORPUS_RUN_SCHEMA,
        lock=_MODEL_LOCK,
        prediction_map=_MAP,
        item_ids=(1, 2),
    )
    _reports(run.parent, _COHORT, _MODEL, _MODEL_LOCK, _MAP, precision="0.5", item_ids=(1, 2))
    _write_items(run.parent / "reports", [_items_row(simfile_id="1")])
    with pytest.raises(
        ComparisonIntegrityError, match="summary population does not match items report"
    ):
        _load_evidence(run)


def test_load_evidence_rejects_live_report_without_summary(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "oaf",
        model=_MODEL,
        run_id=_COHORT,
        schema=OAF_CORPUS_RUN_SCHEMA,
        lock=_MODEL_LOCK,
        prediction_map=_MAP,
    )
    _reports(run.parent, _COHORT, _MODEL, _MODEL_LOCK, _MAP, precision="0.5")
    (run.parent / "reports" / "summary.json").unlink(missing_ok=True)

    with pytest.raises(ComparisonIntegrityError, match="summary.json"):
        _load_evidence(run)


def test_load_evidence_rejects_status_mismatch(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "oaf",
        model=_MODEL,
        run_id=_COHORT,
        schema=OAF_CORPUS_RUN_SCHEMA,
        lock=_MODEL_LOCK,
        prediction_map=_MAP,
        item_ids=(1,),
        item_dispositions={1: "inferred"},
    )
    _reports(
        run.parent,
        _COHORT,
        _MODEL,
        _MODEL_LOCK,
        _MAP,
        precision="0.5",
        failed_item_ids=frozenset({1}),
    )
    with pytest.raises(ComparisonIntegrityError, match="status does not match"):
        _load_evidence(run)


def _evidence(**identity_overrides: object) -> _RunEvidence:
    return _RunEvidence(
        identity=_identity(**identity_overrides),
        items={},
        reports=_Reports({}, {}, {}),
        snapshot={},
    )


def test_validate_manifest_lineage_rejects_reference_sha_mismatch() -> None:
    reference = SimpleNamespace(manifest_sha256="z" * 64, corpus_version=_REF_VERSION)
    timing = SimpleNamespace(manifest_sha256=_TIMING_SHA, corpus_version=_TIMING_VERSION)
    with pytest.raises(ComparisonIntegrityError, match="reference manifest identity"):
        _validate_manifest_lineage(_evidence(), reference, timing)


def test_validate_manifest_lineage_rejects_reference_version_mismatch() -> None:
    reference = SimpleNamespace(manifest_sha256=_REF_SHA, corpus_version="other-version")
    timing = SimpleNamespace(manifest_sha256=_TIMING_SHA, corpus_version=_TIMING_VERSION)
    with pytest.raises(ComparisonIntegrityError, match="reference manifest identity"):
        _validate_manifest_lineage(_evidence(), reference, timing)


def test_validate_manifest_lineage_rejects_timing_mismatch() -> None:
    reference = SimpleNamespace(manifest_sha256=_REF_SHA, corpus_version=_REF_VERSION)
    timing = SimpleNamespace(manifest_sha256="z" * 64, corpus_version=_TIMING_VERSION)
    with pytest.raises(ComparisonIntegrityError, match="reference timing identity"):
        _validate_manifest_lineage(_evidence(), reference, timing)


def test_validate_pair_run_identity_rejects_mismatch() -> None:
    oaf = _identity(input_view_id="crux.other/v1")
    muscriptor = _identity()
    with pytest.raises(ComparisonIntegrityError, match="run identity mismatch for input_view_id"):
        _validate_pair_run_identity(oaf, muscriptor)


def test_subset_ids_rejects_load_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.benchmark.muscriptor_comparison as comparison

    monkeypatch.setattr(
        comparison,
        "load_reviewed_subset_manifest",
        lambda _path: (_ for _ in ()).throw(ValueError("boom")),
    )
    reference = SimpleNamespace(manifest_sha256=_REF_SHA, corpus_version=_REF_VERSION)
    timing = SimpleNamespace(manifest_sha256=_TIMING_SHA, corpus_version=_TIMING_VERSION)
    with pytest.raises(ComparisonIntegrityError, match="invalid subset manifest"):
        _subset_ids(tmp_path / "subset.json", reference, timing, [])


def test_subset_ids_rejects_lineage_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.benchmark.muscriptor_comparison as comparison

    monkeypatch.setattr(
        comparison,
        "load_reviewed_subset_manifest",
        lambda _path: SimpleNamespace(
            source_reference_manifest_sha256="z" * 64,
            source_reference_manifest_version=_REF_VERSION,
            source_timing_manifest_sha256=_TIMING_SHA,
            source_timing_manifest_version=_TIMING_VERSION,
            rows=(),
        ),
    )
    reference = SimpleNamespace(manifest_sha256=_REF_SHA, corpus_version=_REF_VERSION)
    timing = SimpleNamespace(manifest_sha256=_TIMING_SHA, corpus_version=_TIMING_VERSION)
    with pytest.raises(ComparisonIntegrityError, match="lineage does not match"):
        _subset_ids(tmp_path / "subset.json", reference, timing, [_evidence()])


def test_subset_ids_rejects_absent_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.benchmark.muscriptor_comparison as comparison

    monkeypatch.setattr(
        comparison,
        "load_reviewed_subset_manifest",
        lambda _path: SimpleNamespace(
            source_reference_manifest_sha256=_REF_SHA,
            source_reference_manifest_version=_REF_VERSION,
            source_timing_manifest_sha256=_TIMING_SHA,
            source_timing_manifest_version=_TIMING_VERSION,
            rows=(SimpleNamespace(view=SimpleNamespace(simfile_id=99)),),
        ),
    )
    reference = SimpleNamespace(manifest_sha256=_REF_SHA, corpus_version=_REF_VERSION)
    timing = SimpleNamespace(manifest_sha256=_TIMING_SHA, corpus_version=_TIMING_VERSION)
    evidence = _RunEvidence(
        identity=_identity(),
        items={"1": _RunItem("1", "success", _REF_SHA, _REF_SHA)},
        reports=_Reports({}, {}, {}),
        snapshot={},
    )
    with pytest.raises(ComparisonIntegrityError, match="member is absent"):
        _subset_ids(tmp_path / "subset.json", reference, timing, [evidence])


def test_pairable_ids_counts_source_audio_mismatch() -> None:
    oaf = _RunEvidence(
        identity=_identity(),
        items={"1": _RunItem("1", "success", "a" * 64, "c" * 64)},
        reports=_Reports({}, {}, {}),
        snapshot={},
    )
    muscriptor = _RunEvidence(
        identity=_identity(),
        items={"1": _RunItem("1", "success", "b" * 64, "c" * 64)},
        reports=_Reports({}, {}, {}),
        snapshot={},
    )
    pairable, exclusions = _pairable_ids(oaf, muscriptor, None)
    assert pairable == set()
    assert exclusions["source_audio_mismatch"] == 1


def test_shared_pairing_can_allow_distinct_derived_input_hashes() -> None:
    left = PublishedRunEvidence(
        identity=_identity(),
        items={"1": _RunItem("1", "success", "a" * 64, "c" * 64)},
        reports=SimpleNamespace(),  # type: ignore[arg-type]
    )
    right = PublishedRunEvidence(
        identity=_identity(),
        items={"1": _RunItem("1", "success", "a" * 64, "d" * 64)},
        reports=SimpleNamespace(),  # type: ignore[arg-type]
    )

    with pytest.raises(ComparisonIntegrityError, match="canonical-input"):
        pairable_success_ids(left, right, None)

    pairable, exclusions = pairable_success_ids(
        left,
        right,
        None,
        require_identical_input_hash=False,
        left_label="full_mix",
        right_label="spleeter",
    )
    assert pairable == {"1"}
    assert exclusions == {
        "full_mix_only_success": 0,
        "spleeter_only_success": 0,
        "source_audio_mismatch": 0,
    }


@pytest.mark.parametrize("source_hash", [None, "", "not-a-sha256"])
def test_shared_pairing_rejects_missing_or_invalid_source_hash(source_hash: str | None) -> None:
    left = PublishedRunEvidence(
        identity=_identity(),
        items={"1": _RunItem("1", "success", source_hash, "c" * 64)},
        reports=SimpleNamespace(),  # type: ignore[arg-type]
    )
    right = PublishedRunEvidence(
        identity=_identity(),
        items={"1": _RunItem("1", "success", source_hash, "d" * 64)},
        reports=SimpleNamespace(),  # type: ignore[arg-type]
    )

    with pytest.raises(ComparisonIntegrityError, match="source_audio_sha256"):
        pairable_success_ids(left, right, None, require_identical_input_hash=False)


def test_shared_song_join_parameterizes_only_output_labels() -> None:
    key = ("1", 50, "raw")
    left = {key: _SongRow("1", 50, "raw", Decimal("0.5"), Decimal("0.4"), Decimal("0.4"))}
    right = {key: _SongRow("1", 50, "raw", Decimal("0.8"), Decimal("0.4"), Decimal("0.4"))}

    rows = paired_song_rows(
        left,
        right,
        {"1"},
        left_label="full_mix",
        right_label="spleeter",
    )

    assert rows[0]["full_mix_precision"] == "0.5"
    assert rows[0]["spleeter_precision"] == "0.8"
    assert rows[0]["delta_precision"] == "0.3"


def test_shared_summary_parameterizes_schema_identity_and_labels() -> None:
    left = PublishedRunEvidence(
        identity=_identity(),
        items={"1": _RunItem("1", "success", "a" * 64, "c" * 64)},
        reports=SimpleNamespace(),  # type: ignore[arg-type]
    )
    right = PublishedRunEvidence(
        identity=_identity(),
        items={"1": _RunItem("1", "success", "a" * 64, "d" * 64)},
        reports=SimpleNamespace(),  # type: ignore[arg-type]
    )

    summary = comparison_summary(
        left,
        right,
        {"1"},
        {"source_audio_mismatch": 0},
        [],
        [],
        SimpleNamespace(manifest_sha256="r" * 64, corpus_version="ref-v1"),
        SimpleNamespace(manifest_sha256="t" * 64, corpus_version="timing-v1"),
        None,
        None,
        schema="crux.oaf-separation-comparison/v1",
        identity={"input_views": {"full_mix": "crux.full-mix/v1", "spleeter": "crux.spleeter/v1"}},
        left_label="full_mix",
        right_label="spleeter",
    )

    assert summary["schema"] == "crux.oaf-separation-comparison/v1"
    assert summary["identity"] == {
        "input_views": {"full_mix": "crux.full-mix/v1", "spleeter": "crux.spleeter/v1"}
    }
    assert set(summary["models"]) == {"full_mix", "spleeter"}


def test_shared_markdown_parameterizes_heading_and_labels(tmp_path: Path) -> None:
    population = {
        "total_count": 0,
        "eligible_count": 0,
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "quarantined_count": 0,
    }
    summary = {
        "identity": {"input_views": "full-mix-vs-spleeter"},
        "models": {"full_mix": {"population": population}, "spleeter": {"population": population}},
        "pairing": {
            "pairable_success_intersection": 0,
            "paired_song_row_count": 0,
            "paired_class_row_count": 0,
            "exclusions": {},
        },
        "aggregates": {"song": [], "class": []},
    }
    path = tmp_path / "summary.md"

    write_markdown(
        path,
        summary,
        title="Full Mix vs Spleeter",
        left_label="full_mix",
        right_label="spleeter",
    )

    rendered = path.read_text(encoding="utf-8")
    assert rendered.startswith("# Full Mix vs Spleeter\n")
    assert "| full_mix |" in rendered
    assert "| spleeter |" in rendered


def test_metric_delta_returns_none_for_missing_side() -> None:
    from decimal import Decimal

    assert _metric_delta(None, Decimal("0.5")) is None
    assert _metric_delta(Decimal("0.5"), None) is None


def test_csv_decimal_renders_none_as_empty() -> None:
    assert _csv_decimal(None) == ""


def test_markdown_metric_renders_none_and_non_decimal() -> None:
    assert _markdown_metric(None) == "N/A"
    assert _markdown_metric(5) == "5"


def test_paired_song_rows_skips_non_pairable_ids() -> None:
    from decimal import Decimal

    key = ("1", 50, "raw")
    oaf = {key: _SongRow("1", 50, "raw", Decimal("0.5"), Decimal("0.4"), Decimal("0.4"))}
    muscriptor = {key: _SongRow("1", 50, "raw", Decimal("0.8"), Decimal("0.4"), Decimal("0.4"))}
    assert _paired_song_rows(oaf, muscriptor, pairable_ids=set()) == []


def test_paired_class_rows_skips_non_pairable_ids() -> None:
    from decimal import Decimal

    key = ("1", 50, "raw", "kick")
    oaf = {
        key: _ClassRow(
            "1", 50, "raw", "kick", 2, 1, 1, 0, 1, Decimal("0.5"), Decimal("0.5"), Decimal("0.5")
        )
    }
    muscriptor = {
        key: _ClassRow(
            "1", 50, "raw", "kick", 2, 1, 1, 0, 1, Decimal("0.8"), Decimal("0.5"), Decimal("0.5")
        )
    }
    rows, exclusions = _paired_class_rows(oaf, muscriptor, pairable_ids=set())
    assert rows == []
    assert exclusions == {
        "oaf_only_prediction_class": 0,
        "muscriptor_only_prediction_class": 0,
    }


def test_paired_class_rows_rejects_asymmetric_pairable_key_grid() -> None:
    from decimal import Decimal

    key = ("1", 50, "raw", "kick")
    extra_key = ("1", 50, "raw", "snare")
    row = _ClassRow(
        "1", 50, "raw", "kick", 2, 1, 1, 0, 1, Decimal("0.5"), Decimal("0.5"), Decimal("0.5")
    )
    extra_row = _ClassRow(
        "1", 50, "raw", "snare", 1, 0, 0, 0, 1, Decimal("1"), Decimal("1"), Decimal("1")
    )

    with pytest.raises(ComparisonIntegrityError, match="reference-supported class row is missing"):
        _paired_class_rows(
            {key: row, extra_key: extra_row},
            {key: row},
            pairable_ids={"1"},
        )


def test_paired_class_rows_accepts_muscriptor_one_sided_prediction_row() -> None:
    from decimal import Decimal

    # Regression for the MuScriptor _ClassRow adapter: a legitimate prediction-only
    # one-sided row must be accepted and counted, not rejected as a missing
    # true_positives field by _validate_one_sided_support_identity.
    key = ("241", 50, "raw", "ride")
    row = _ClassRow(
        "241",
        50,
        "raw",
        "ride",
        0,
        2,
        0,
        2,
        0,
        Decimal("0.5"),
        Decimal("0.5"),
        Decimal("0.5"),
    )
    rows, exclusions = _paired_class_rows(
        {key: row},
        {},
        pairable_ids={"241"},
    )
    assert rows == []
    assert exclusions == {
        "oaf_only_prediction_class": 1,
        "muscriptor_only_prediction_class": 0,
    }


def test_compare_rejects_non_comparison_request() -> None:
    with pytest.raises(TypeError, match="request must be ComparisonRequest"):
        compare_oaf_muscriptor({"not": "a request"})  # type: ignore[arg-type]


def test_compare_wraps_os_error_as_integrity_error(tmp_path: Path, manifest_loaders) -> None:
    oaf_root = tmp_path / "oaf"
    muscriptor_root = tmp_path / "muscriptor"
    oaf = _write_run(
        oaf_root,
        model=_MODEL,
        run_id=_COHORT,
        schema=OAF_CORPUS_RUN_SCHEMA,
        lock=_MODEL_LOCK,
        prediction_map=_MAP,
    )
    muscriptor = _write_run(
        muscriptor_root,
        model="muscriptor-model",
        run_id="muscriptor-run",
        schema=MUSCRIPTOR_CORPUS_RUN_SCHEMA,
        lock="f" * 64,
        prediction_map="muscriptor-map",
    )
    _reports(oaf_root, _COHORT, _MODEL, _MODEL_LOCK, _MAP, precision="0.5")
    _reports(
        muscriptor_root,
        "muscriptor-run",
        "muscriptor-model",
        "f" * 64,
        "muscriptor-map",
        precision="0.8",
    )
    blocker = tmp_path / "blocker"
    blocker.write_text("blocks", encoding="utf-8")
    request = ComparisonRequest(
        oaf_run_path=oaf,
        muscriptor_run_path=muscriptor,
        reference_manifest_path=tmp_path / "hpa324.jsonl",
        timing_manifest_path=tmp_path / "hpa323.jsonl",
        output_dir=blocker / "comparison",
    )
    with pytest.raises(ComparisonIntegrityError):
        compare_oaf_muscriptor(request)


def test_parse_metric_returns_parsed_value_for_non_bounded_field() -> None:
    """_parse_metric returns the parsed Decimal for fields without range bounds."""
    assert _parse_metric("0", "offset_ms") == Decimal("0")


def test_parse_song_rows_succeeds_for_valid_csv(tmp_path: Path) -> None:
    """_parse_song_rows returns a dict of score-keyed rows for a well-formed report."""
    path = _write_songs(tmp_path, [_song_row()])
    rows = _parse_song_rows(path, _identity(), {"1"})
    key = ("1", 50, "raw")
    assert key in rows
    assert rows[key].precision == Decimal("0.5")


def test_parse_class_rows_succeeds_for_valid_csv(tmp_path: Path) -> None:
    """_parse_class_rows returns a dict of score-keyed rows for a well-formed report."""
    path = _write_classes(tmp_path, [_class_row()])
    rows = _parse_class_rows(path, _identity(), {"1"})
    key = ("1", 50, "raw", "kick")
    assert key in rows
    assert rows[key].precision == Decimal("0.5")


def test_report_identity_from_snapshot_rejects_unsupported_schema() -> None:
    """An unknown schema triggers ComparisonIntegrityError."""
    with pytest.raises(ComparisonIntegrityError, match="schema is unsupported"):
        _report_identity_from_snapshot({"schema": "crux.unknown/v1"})


def test_report_identity_from_snapshot_wraps_invalid_identity_as_integrity_error() -> None:
    """A valid schema with missing identity fields is wrapped as ComparisonIntegrityError."""
    with pytest.raises(ComparisonIntegrityError, match="cohort identity is invalid"):
        _report_identity_from_snapshot({"schema": "crux.oaf-corpus-run/v1"})
