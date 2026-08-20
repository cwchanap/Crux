"""Focused coverage for src/benchmark/published_comparison.py patch lines."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.benchmark.published_comparison import (
    ComparisonIntegrityError,
    PublishedRunEvidence,
    _identity_attr,
    _identity_values,
    _item_hash,
    _item_status,
    _items,
    _label,
    _manifest_attr,
    aggregate_delta_rows,
    comparison_summary,
    pairable_success_ids,
    paired_song_rows,
    write_comparison_artifacts,
    write_markdown,
)

_SHA = "a" * 64


def _evidence(items: dict | None = None) -> PublishedRunEvidence:
    return PublishedRunEvidence(
        identity=SimpleNamespace(),
        items=items if items is not None else {},
        reports=SimpleNamespace(),  # type: ignore[arg-type]
    )


def test_label_rejects_empty_and_non_string() -> None:
    with pytest.raises(ValueError, match="must be a nonempty string"):
        _label("", "field")
    with pytest.raises(ValueError, match="must be a nonempty string"):
        _label(123, "field")  # type: ignore[arg-type]


def test_items_rejects_non_mapping() -> None:
    evidence = SimpleNamespace(items="not-a-mapping")
    with pytest.raises(ComparisonIntegrityError, match="items must be a mapping"):
        _items(evidence)


def test_item_status_rejects_non_string() -> None:
    item = SimpleNamespace(status=123)
    with pytest.raises(ComparisonIntegrityError, match="status is malformed"):
        _item_status(item)


def test_item_hash_rejects_non_string_non_none() -> None:
    item = SimpleNamespace(input_audio_sha256=123)
    with pytest.raises(ComparisonIntegrityError, match="is malformed"):
        _item_hash(item, "input_audio_sha256")


def test_pairable_success_ids_rejects_non_bool_require_identical_input_hash() -> None:
    left = _evidence()
    right = _evidence()
    with pytest.raises(TypeError, match="require_identical_input_hash must be a bool"):
        pairable_success_ids(
            left,
            right,
            None,
            require_identical_input_hash="yes",  # type: ignore[arg-type]
        )


def test_paired_song_rows_rejects_key_grid_mismatch() -> None:
    key = ("1", 50, "raw")
    extra_key = ("2", 50, "raw")
    row = SimpleNamespace(precision=Decimal("0.5"), recall=Decimal("0.5"), f1=Decimal("0.5"))
    with pytest.raises(ComparisonIntegrityError, match="per_song score key grid mismatch"):
        paired_song_rows(
            {key: row},
            {key: row, extra_key: row},
            {"1", "2"},
        )


def test_manifest_attr_rejects_missing_field() -> None:
    manifest = SimpleNamespace(manifest_sha256=_SHA)
    with pytest.raises(ComparisonIntegrityError, match="manifest is missing required field"):
        _manifest_attr(manifest, "corpus_version")


def test_identity_attr_rejects_missing_field() -> None:
    identity = SimpleNamespace(input_view_id="view")
    with pytest.raises(ComparisonIntegrityError, match="identity is missing required field"):
        _identity_attr(identity, "model_id")


def test_identity_values_uses_field_fallback_when_report_values_not_callable() -> None:
    identity = SimpleNamespace(
        cohort_id="cohort",
        model_id="model",
        model_lock_sha256=_SHA,
        prediction_map_version="map",
        input_view_id="view",
        scoring_version="scoring",
    )
    values = _identity_values(identity)
    assert values["cohort_id"] == "cohort"
    assert values["model_id"] == "model"
    assert values["scoring_version"] == "scoring"


def test_identity_values_fails_when_required_field_missing() -> None:
    identity = SimpleNamespace(
        cohort_id="cohort",
        model_id="model",
    )
    with pytest.raises(ComparisonIntegrityError, match="identity is missing required field"):
        _identity_values(identity)


def test_comparison_summary_rejects_non_mapping_identity() -> None:
    left = _evidence()
    right = _evidence()
    with pytest.raises(TypeError, match="identity must be a mapping"):
        comparison_summary(
            left,
            right,
            set(),
            {},
            [],
            [],
            SimpleNamespace(manifest_sha256=_SHA, corpus_version="v1"),
            SimpleNamespace(manifest_sha256=_SHA, corpus_version="v1"),
            None,
            None,
            identity="not-a-mapping",  # type: ignore[arg-type]
        )


def test_write_markdown_rejects_non_mapping_identity(tmp_path: Path) -> None:
    summary = {"identity": "not-a-mapping", "models": {}, "pairing": {}, "aggregates": {}}
    with pytest.raises(ComparisonIntegrityError, match="identity must be a mapping"):
        write_markdown(tmp_path / "out.md", summary)


def test_write_markdown_rejects_non_mapping_models(tmp_path: Path) -> None:
    summary = {"identity": {}, "models": "not-a-mapping", "pairing": {}, "aggregates": {}}
    with pytest.raises(ComparisonIntegrityError, match="models must be a mapping"):
        write_markdown(tmp_path / "out.md", summary)


def test_write_markdown_rejects_non_mapping_pairing(tmp_path: Path) -> None:
    summary = {"identity": {}, "models": {}, "pairing": "not-a-mapping", "aggregates": {}}
    with pytest.raises(ComparisonIntegrityError, match="pairing must be a mapping"):
        write_markdown(tmp_path / "out.md", summary)


def test_write_markdown_rejects_non_mapping_aggregates(tmp_path: Path) -> None:
    summary = {"identity": {}, "models": {}, "pairing": {}, "aggregates": "not-a-mapping"}
    with pytest.raises(ComparisonIntegrityError, match="aggregates must be a mapping"):
        write_markdown(tmp_path / "out.md", summary)


def test_write_markdown_renders_reason_counts_column(tmp_path: Path) -> None:
    pop = {
        "total_count": 1,
        "eligible_count": 1,
        "success_count": 1,
        "failed_count": 0,
        "skipped_count": 0,
        "quarantined_count": 0,
    }
    summary = {
        "identity": {"field": "value"},
        "models": {
            "oaf": {"population": pop, "reason_counts": {"error": 1}},
            "muscriptor": {"population": pop},
        },
        "pairing": {
            "pairable_success_intersection": 0,
            "paired_song_row_count": 0,
            "paired_class_row_count": 0,
            "exclusions": {},
        },
        "aggregates": {"song": [], "class": []},
    }
    path = tmp_path / "summary.md"
    write_markdown(path, summary)
    rendered = path.read_text(encoding="utf-8")
    assert "reason_counts" in rendered
    assert "error" in rendered


def test_write_comparison_artifacts_writes_all_files(tmp_path: Path) -> None:
    song_rows = [
        {
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
    ]
    class_rows = [
        {
            "simfile_id": "1",
            "tolerance_ms": "50",
            "mode": "raw",
            "common_class": "kick",
            "oaf_reference_support": "2",
            "muscriptor_reference_support": "2",
            "oaf_prediction_support": "1",
            "muscriptor_prediction_support": "1",
            "oaf_precision": "0.5",
            "muscriptor_precision": "0.8",
            "delta_precision": "0.3",
            "oaf_recall": "0.5",
            "muscriptor_recall": "0.5",
            "delta_recall": "0",
            "oaf_f1": "0.5",
            "muscriptor_f1": "0.5",
            "delta_f1": "0",
        }
    ]
    pop = {
        "total_count": 1,
        "eligible_count": 1,
        "success_count": 1,
        "failed_count": 0,
        "skipped_count": 0,
        "quarantined_count": 0,
    }
    summary = {
        "schema": "crux.oaf-muscriptor-comparison/v1",
        "identity": {"field": "value"},
        "subset_manifest": None,
        "models": {
            "oaf": {"population": pop, "runtime": {}},
            "muscriptor": {"population": pop, "runtime": {}},
        },
        "pairing": {
            "pairable_success_intersection": 1,
            "paired_song_row_count": 1,
            "paired_class_row_count": 1,
            "exclusions": {
                "oaf_only_success": 0,
                "muscriptor_only_success": 0,
                "source_audio_mismatch": 0,
            },
        },
        "aggregates": {
            "song": aggregate_delta_rows(song_rows),
            "class": aggregate_delta_rows(class_rows),
        },
    }
    output_dir = tmp_path / "comparison"
    write_comparison_artifacts(output_dir, song_rows, class_rows, summary)
    assert (output_dir / "paired_per_song.csv").is_file()
    assert (output_dir / "paired_per_class.csv").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "summary.md").is_file()
