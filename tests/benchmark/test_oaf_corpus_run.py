from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from runtime.oaf_tf1.model import load_model_config
from src.benchmark.backend_identity import (
    BackendDescriptor,
    StrictJsonError,
    canonical_json_bytes,
    strict_json_loads,
)
from src.benchmark.backends.oaf import OAF_ADAPTER_REVISION
from src.benchmark.cohort_scoring import COHORT_FAILURE_REASONS
from src.benchmark.oaf_corpus_run import (
    OAF_BACKEND_ERROR_POLICY,
    OAF_CANONICALIZATION_REVISION,
    OAF_CORPUS_REQUEST_TIMEOUT_SECONDS,
    OAF_CORPUS_RUN_SCHEMA,
    OAF_FULL_MIX_INPUT_VIEW_ID,
    OAF_INFERENCE_CONFIG_SCHEMA,
    OAF_WORKER_CLOSE_TIMEOUT_SECONDS,
    RUNNER_FAILURE_TO_COHORT_REASON,
    OafCorpusRunOutcome,
    OafCorpusRunRequest,
    _validate_scope,
    build_inference_config,
    build_run_id,
    classify_oaf_backend_error,
    compute_model_lock_sha256,
    inference_config_sha256,
    parse_oaf_corpus_run,
    prediction_path,
    render_oaf_corpus_run,
    write_oaf_corpus_run,
)
from src.benchmark.taxonomy import OAF_PREDICTION_MAP_ID

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
COMMIT = "d" * 40


def test_constants_and_public_dataclasses_are_frozen_contracts() -> None:
    assert OAF_CORPUS_RUN_SCHEMA == "crux.oaf-corpus-run/v1"
    assert OAF_INFERENCE_CONFIG_SCHEMA == "crux.oaf-inference-config/v1"
    assert OAF_FULL_MIX_INPUT_VIEW_ID == "crux.oaf-full-mix-mono44k1-pcm16/v1"
    assert OAF_CANONICALIZATION_REVISION == "librosa-soxr-hq-mono44k1-soundfile-pcm16/v1"
    assert OAF_CORPUS_REQUEST_TIMEOUT_SECONDS == 3600.0
    assert OAF_WORKER_CLOSE_TIMEOUT_SECONDS == 30.0

    request = OafCorpusRunRequest(
        reference_manifest_path=Path("reference.jsonl"),
        timing_manifest_path=Path("timing.jsonl"),
        cache_dir=Path("cache"),
        output_dir=Path("output"),
        include_simfile_ids=(30, 10, 10),
        exclude_simfile_ids=(40,),
        crux_commit=COMMIT,
    )
    assert request.include_simfile_ids == (10, 30)
    assert request.exclude_simfile_ids == (40,)
    outcome = OafCorpusRunOutcome(
        overall_status="complete",
        exit_code=0,
        run_id="oaf-" + SHA_A[:16],
        run_path=Path("run.json"),
        reports_path=None,
        success_count=1,
        failed_count=0,
        skipped_count=0,
        quarantined_count=0,
        aggregate_rtf=1.25,
        projected_full_wall_time_sec=3600.0,
    )
    assert outcome.success_count == 1


def test_model_lock_hash_is_exact_file_sha256(tmp_path: Path) -> None:
    model_lock = tmp_path / "model.json"
    model_lock.write_bytes(b'{"schema":"test"}\n')

    assert compute_model_lock_sha256(model_lock) == (
        "b3c34c00a38e16700dcf90b345fbd4820bac96cbb0137d459d7856bf88fbcc3e"
    )


def test_inference_config_has_exact_keys_and_deterministic_hash() -> None:
    config = load_model_config()
    descriptor = BackendDescriptor(payload={}, sha256=SHA_A)
    payload = build_inference_config(config, descriptor, SHA_B)

    assert set(payload) == {
        "schema",
        "backend_descriptor_sha256",
        "model_lock_sha256",
        "checkpoint_archive_sha256",
        "adapter_revision",
        "prediction_map_version",
        "input_view_id",
        "canonicalization_revision",
    }
    assert payload == {
        "schema": OAF_INFERENCE_CONFIG_SCHEMA,
        "backend_descriptor_sha256": SHA_A,
        "model_lock_sha256": SHA_B,
        "checkpoint_archive_sha256": config.checkpoint.archive_sha256,
        "adapter_revision": OAF_ADAPTER_REVISION,
        "prediction_map_version": OAF_PREDICTION_MAP_ID,
        "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
        "canonicalization_revision": OAF_CANONICALIZATION_REVISION,
    }
    first_hash = inference_config_sha256(payload)
    assert first_hash == inference_config_sha256(dict(payload))
    assert first_hash != inference_config_sha256({**payload, "input_view_id": "other/v1"})
    assert first_hash != inference_config_sha256({**payload, "adapter_revision": "other/v1"})
    assert first_hash != inference_config_sha256({**payload, "checkpoint_archive_sha256": SHA_C})
    assert first_hash != inference_config_sha256({**payload, "backend_descriptor_sha256": SHA_C})


def test_run_id_is_deterministic_and_binds_reference_and_scope() -> None:
    common = dict(
        reference_manifest_sha256=SHA_A,
        reference_timing_manifest_sha256=SHA_B,
        backend_descriptor_sha256=SHA_C,
        model_lock_sha256=SHA_A,
        checkpoint_archive_sha256=SHA_B,
        inference_config_sha256=SHA_C,
        include_simfile_ids=(30, 10, 10),
        exclude_simfile_ids=(40,),
    )
    first = build_run_id(**common)
    assert first == build_run_id(**common)
    assert first.startswith("oaf-")
    assert first != build_run_id(**{**common, "reference_manifest_sha256": SHA_B})
    assert first != build_run_id(**{**common, "include_simfile_ids": (10, 20)})
    assert first != build_run_id(**{**common, "exclude_simfile_ids": (41,)})


def test_prediction_path_is_source_keyed_and_reference_independent(tmp_path: Path) -> None:
    path = prediction_path(
        tmp_path,
        simfile_id=10,
        source_audio_sha256=SHA_A,
        backend_descriptor_sha256=SHA_B,
        inference_config_sha256=SHA_C,
    )

    assert path == (tmp_path / "predictions" / "10" / SHA_A / SHA_B / f"{SHA_C}.jsonl")
    assert "input_audio_sha256" not in str(path)
    assert "reference_manifest" not in str(path)


def test_scope_preflight_normalizes_ids_and_rejects_unknown_or_overlap() -> None:
    assert _validate_scope((10, 10), (), {10, 20, 30}) == ((10,), ())
    assert _validate_scope((10,), (), {10, 20, 30}) == ((10,), ())
    with pytest.raises(ValueError, match="unknown include"):
        _validate_scope((99,), (), {10, 20, 30})
    with pytest.raises(ValueError, match="unknown exclude"):
        _validate_scope((), (99,), {10, 20, 30})
    with pytest.raises(ValueError, match="overlap"):
        _validate_scope((10,), (10,), {10, 20, 30})


def test_invalid_identity_and_scope_values_fail_closed() -> None:
    config = load_model_config()
    with pytest.raises(ValueError):
        build_inference_config(config, BackendDescriptor(payload={}, sha256="bad"), SHA_B)
    with pytest.raises(ValueError):
        build_run_id(
            reference_manifest_sha256="bad",
            reference_timing_manifest_sha256=SHA_B,
            backend_descriptor_sha256=SHA_C,
            model_lock_sha256=SHA_A,
            checkpoint_archive_sha256=SHA_B,
            inference_config_sha256=SHA_C,
        )
    with pytest.raises(ValueError, match="commit"):
        OafCorpusRunRequest(
            reference_manifest_path=Path("reference.jsonl"),
            timing_manifest_path=Path("timing.jsonl"),
            cache_dir=Path("cache"),
            output_dir=Path("output"),
            crux_commit="not-a-commit",
        )
    with pytest.raises(ValueError):
        _validate_scope((True,), (), {1})


def test_backend_error_policy_is_closed_and_unknown_errors_poison() -> None:
    assert OAF_BACKEND_ERROR_POLICY == {
        "inference_failed": ("inference_failed", "item_local"),
        "invalid_request": ("inference_failed", "item_local"),
        "input_path_invalid": ("canonical_input_failed", "item_local"),
        "native_event_invalid": ("inference_failed", "item_local"),
        "worker_error": ("worker_protocol_failed", "poison"),
        "worker_start_failed": ("backend_unavailable", "poison"),
        "worker_ready_invalid": ("backend_unavailable", "poison"),
        "worker_identity_invalid": ("backend_unavailable", "poison"),
        "worker_response_invalid": ("worker_protocol_failed", "poison"),
        "backend_closed": ("worker_protocol_failed", "poison"),
        "descriptor_invalid": (None, "fatal_preflight"),
        "worker_close_failed": (None, "finalization"),
    }
    assert classify_oaf_backend_error("future_code") == (
        "worker_protocol_failed",
        "poison",
    )
    assert set(RUNNER_FAILURE_TO_COHORT_REASON.values()) <= COHORT_FAILURE_REASONS
    assert RUNNER_FAILURE_TO_COHORT_REASON == {
        "source_audio_unavailable": "inference_failed",
        "source_audio_decode_failed": "inference_failed",
        "canonical_input_failed": "inference_failed",
        "backend_unavailable": "backend_unavailable",
        "worker_protocol_failed": "backend_unavailable",
        "inference_failed": "inference_failed",
        "prediction_artifact_invalid": "prediction_artifact_invalid",
        "prediction_output_conflict": "prediction_artifact_invalid",
        "prediction_publish_failed": "prediction_artifact_invalid",
        "prediction_missing": "prediction_missing",
    }


def _snapshot() -> dict[str, object]:
    return {
        "schema": OAF_CORPUS_RUN_SCHEMA,
        "run_id": "oaf-" + SHA_A[:16],
        "reference_manifest_sha256": SHA_A,
        "request_timeout_seconds": 3600.0,
        "exclude_simfile_ids": [20],
        "overall_status": "complete",
        "success_count": 1,
        "failed_count": 0,
        "skipped_count": 1,
        "quarantined_count": 0,
        "completed_at": "2026-08-14T00:00:00+00:00",
        "items": [
            {
                "simfile_id": 20,
                "execution_disposition": "skipped",
            },
            {
                "simfile_id": 10,
                "execution_disposition": "inferred",
                "wall_time_sec": 1.25,
                "rtf": 0.5,
            },
        ],
    }


def test_render_normalizes_floats_sorts_items_and_round_trips() -> None:
    content = render_oaf_corpus_run(_snapshot())
    parsed = parse_oaf_corpus_run(content)

    assert parsed["request_timeout_seconds"] == Decimal("3600")
    assert parsed["items"][0]["wall_time_sec"] == Decimal("1.25")
    assert [item["simfile_id"] for item in parsed["items"]] == [10, 20]
    assert render_oaf_corpus_run(parsed) == content


def test_parse_rejects_canonical_but_semantically_unsorted_items() -> None:
    rendered = render_oaf_corpus_run(_snapshot())
    canonical_snapshot = strict_json_loads(rendered)
    assert isinstance(canonical_snapshot, dict)
    items = canonical_snapshot["items"]
    assert isinstance(items, list)
    canonical_snapshot["items"] = list(reversed(items))
    semantically_unsorted = canonical_json_bytes(canonical_snapshot)

    with pytest.raises(StrictJsonError, match="semantically canonical"):
        parse_oaf_corpus_run(semantically_unsorted)


def test_canonical_json_rejects_direct_float_injection() -> None:
    with pytest.raises(StrictJsonError, match="unsupported JSON value"):
        canonical_json_bytes({"wall_time_sec": 1.25})


def test_snapshot_validates_schema_counts_skips_and_completion() -> None:
    invalid_schema = _snapshot()
    invalid_schema["schema"] = "other/v1"
    with pytest.raises(ValueError, match="schema"):
        render_oaf_corpus_run(invalid_schema)

    invalid_counts = _snapshot()
    invalid_counts["success_count"] = 2
    with pytest.raises(ValueError, match="count"):
        render_oaf_corpus_run(invalid_counts)

    invalid_skip = _snapshot()
    invalid_skip["exclude_simfile_ids"] = []
    with pytest.raises(ValueError, match="skipped"):
        render_oaf_corpus_run(invalid_skip)

    incomplete = _snapshot()
    incomplete["overall_status"] = "partial"
    incomplete["completed_at"] = "2026-08-14T00:00:00+00:00"
    with pytest.raises(ValueError, match="completed"):
        render_oaf_corpus_run(incomplete)


@pytest.mark.parametrize("missing_field", ["execution_disposition", "failed_count"])
def test_complete_snapshot_requires_complete_item_accounting(missing_field: str) -> None:
    incomplete = _snapshot()
    if missing_field == "execution_disposition":
        del incomplete["items"][0][missing_field]
    else:
        del incomplete[missing_field]

    with pytest.raises(ValueError, match="complete|count|disposition"):
        render_oaf_corpus_run(incomplete)


def test_complete_snapshot_rejects_null_item_disposition() -> None:
    incomplete = _snapshot()
    incomplete["exclude_simfile_ids"] = []
    incomplete["items"] = [{"simfile_id": 10, "execution_disposition": None}]
    incomplete["success_count"] = 0
    incomplete["failed_count"] = 0
    incomplete["skipped_count"] = 0
    incomplete["quarantined_count"] = 0

    with pytest.raises(ValueError, match="disposition"):
        render_oaf_corpus_run(incomplete)


def test_snapshot_writer_delegates_to_atomic_replace(tmp_path: Path, monkeypatch) -> None:
    import src.benchmark.oaf_corpus_run as run_module

    run_path = tmp_path / "runs" / "run.json"
    run_path.parent.mkdir()
    captured: dict[str, object] = {}

    def fake_atomic_replace(path: Path, content: bytes) -> None:
        captured["path"] = path
        captured["content"] = content

    monkeypatch.setattr(run_module, "atomic_replace_bytes", fake_atomic_replace)
    write_oaf_corpus_run(run_path, _snapshot())

    assert captured == {"path": run_path, "content": render_oaf_corpus_run(_snapshot())}
