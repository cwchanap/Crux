"""Targeted coverage for validation and edge branches in muscriptor_corpus_run."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.benchmark.backend_identity import (
    BackendDescriptor,
    StrictJsonError,
)
from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.cohort_scoring import CohortIdentity
from src.benchmark.corpus_cache import ResolvedSourceAudio
from src.benchmark.muscriptor_corpus_run import (
    MUSCRIPTOR_CORPUS_RUN_SCHEMA,
    MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
    MUSCRIPTOR_PREDICTION_MAP_ID,
    RUNNER_FAILURE_TO_COHORT_REASON,
    MuscriptorCorpusRunOutcome,
    MuscriptorCorpusRunRequest,
    _bounded_close_error,
    _bounded_error,
    _cohort_identity_from_snapshot,
    _cohort_item_from_run_row,
    _cohort_item_without_prediction,
    _device_peak_memory_bytes,
    _expected_muscriptor_descriptor,
    _finite_positive,
    _normalize_scope,
    _normalize_snapshot_value,
    _prediction_artifact_matches,
    _prior_source_matches,
    _process_peak_rss_bytes,
    _project_runtime,
    _read_existing_prediction,
    _remove_temporary_input,
    _require_commit,
    _require_hash,
    _require_revision,
    _snapshot_counts,
    _source_failure_code,
    _timestamp,
    _validate_scope,
    build_inference_config,
    build_muscriptor_cohort_from_snapshot,
    build_run_id,
    classify_muscriptor_backend_error,
    compute_model_lock_sha256,
    inference_config_sha256,
    parse_muscriptor_corpus_run,
    render_muscriptor_corpus_run,
    run_muscriptor_corpus,
    write_muscriptor_corpus_run,
)
from src.benchmark.taxonomy import (
    DTX_LANE_MAP_VERSION,
    TAXONOMY_VERSION,
)
from tests.benchmark.muscriptor_run_fixtures import (
    SHA_A,
    SHA_B,
    SHA_C,
    _install_seams,
    _lock,
    _mapping,
    _prediction,
    _request,
)

SHA = "a" * 64


# ---------------------------------------------------------------------------
# MuscriptorCorpusRunRequest validation
# ---------------------------------------------------------------------------


def test_request_rejects_non_path_field() -> None:
    with pytest.raises(TypeError, match="must be a Path"):
        MuscriptorCorpusRunRequest(
            reference_manifest_path="reference.jsonl",  # type: ignore[arg-type]
            timing_manifest_path=Path("timing.jsonl"),
            cache_dir=Path("cache"),
            output_dir=Path("output"),
        )


def test_request_rejects_non_bool_resume() -> None:
    with pytest.raises(TypeError, match="resume must be a bool"):
        MuscriptorCorpusRunRequest(
            reference_manifest_path=Path("reference.jsonl"),
            timing_manifest_path=Path("timing.jsonl"),
            cache_dir=Path("cache"),
            output_dir=Path("output"),
            resume="yes",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# MuscriptorCorpusRunOutcome validation
# ---------------------------------------------------------------------------


def _valid_outcome_kwargs() -> dict[str, object]:
    return dict(
        overall_status="complete",
        exit_code=0,
        run_id="muscriptor-abc",
        run_path=Path("run.json"),
        reports_path=Path("reports"),
        success_count=1,
        failed_count=0,
        skipped_count=0,
        quarantined_count=0,
        aggregate_rtf=0.5,
        projected_full_wall_time_sec=10.0,
    )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"overall_status": "bogus"}, "overall_status"),
        ({"exit_code": 3}, "exit_code"),
        ({"exit_code": True}, "exit_code"),
        ({"run_id": ""}, "run_id"),
        ({"run_path": "x"}, "run_path"),
        ({"reports_path": "x"}, "reports_path"),
        ({"success_count": -1}, "success_count"),
        ({"failed_count": True}, "failed_count"),
        ({"aggregate_rtf": "x"}, "aggregate_rtf"),
        ({"aggregate_rtf": float("inf")}, "aggregate_rtf"),
        ({"peak_process_rss_bytes": -1}, "peak_process_rss_bytes"),
        ({"peak_process_rss_bytes": True}, "peak_process_rss_bytes"),
        ({"fatal_reason": ""}, "fatal_reason"),
    ],
)
def test_outcome_rejects_invalid_fields(override: dict[str, object], match: str) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        MuscriptorCorpusRunOutcome(**{**_valid_outcome_kwargs(), **override})


# ---------------------------------------------------------------------------
# _require_commit / _require_hash / _require_revision
# ---------------------------------------------------------------------------


def test_require_commit_rejects_non_matching() -> None:
    with pytest.raises(StrictJsonError, match="crux_commit"):
        _require_commit("not-a-commit")
    with pytest.raises(StrictJsonError, match="crux_commit"):
        _require_commit(123)  # type: ignore[arg-type]


def test_require_hash_rejects_non_str() -> None:
    with pytest.raises(StrictJsonError, match="SHA-256"):
        _require_hash(None, "field")  # type: ignore[arg-type]


def test_require_revision_rejects_invalid() -> None:
    with pytest.raises(StrictJsonError, match="hex"):
        _require_revision("x" * 40)
    with pytest.raises(StrictJsonError, match="hex"):
        _require_revision(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_model_lock_sha256
# ---------------------------------------------------------------------------


def test_compute_model_lock_sha256_rejects_non_path(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Path"):
        compute_model_lock_sha256("not-a-path")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_inference_config type checks
# ---------------------------------------------------------------------------


def test_build_inference_config_rejects_wrong_types() -> None:
    lock = _lock()
    descriptor = _expected_muscriptor_descriptor(lock)
    with pytest.raises(TypeError, match="lock must be"):
        build_inference_config("not-a-lock", descriptor, SHA_A)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="descriptor must be"):
        build_inference_config(lock, "not-a-descriptor", SHA_A)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# inference_config_sha256 validation
# ---------------------------------------------------------------------------


def _valid_inference_payload() -> dict[str, object]:
    lock = _lock()
    descriptor = _expected_muscriptor_descriptor(lock)
    return build_inference_config(lock, descriptor, SHA_A)


def test_inference_config_sha256_rejects_wrong_key_set() -> None:
    with pytest.raises(StrictJsonError, match="exact key set"):
        inference_config_sha256({})


def test_inference_config_sha256_rejects_wrong_schema() -> None:
    payload = _valid_inference_payload()
    payload["schema"] = "wrong"
    with pytest.raises(StrictJsonError, match="schema"):
        inference_config_sha256(payload)


@pytest.mark.parametrize(
    "field",
    [
        "backend_descriptor_sha256",
        "model_lock_sha256",
        "checkpoint_sha256",
        "inference_config_sha256",
    ],
)
def test_inference_config_sha256_rejects_bad_hash(field: str) -> None:
    payload = _valid_inference_payload()
    payload[field] = "not-a-hash"
    with pytest.raises(StrictJsonError):
        inference_config_sha256(payload)


def test_inference_config_sha256_rejects_bad_revision() -> None:
    payload = _valid_inference_payload()
    payload["checkpoint_revision"] = "x" * 40
    with pytest.raises(StrictJsonError):
        inference_config_sha256(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adapter_revision", ""),
        ("prediction_map_version", ""),
        ("input_view_id", ""),
        ("canonicalization_revision", ""),
        ("device", ""),
        ("dtype", ""),
    ],
)
def test_inference_config_rejects_empty_string_fields(field: str, value: str) -> None:
    payload = _valid_inference_payload()
    payload[field] = value
    with pytest.raises(StrictJsonError):
        inference_config_sha256(payload)


@pytest.mark.parametrize(
    ("field", "bad_value", "expected"),
    [
        ("adapter_revision", "wrong", "adapter_revision"),
        ("prediction_map_version", "wrong", "prediction_map_version"),
        ("input_view_id", "wrong", "input_view_id"),
        ("canonicalization_revision", "wrong", "canonicalization_revision"),
    ],
)
def test_inference_config_rejects_wrong_identity_strings(
    field: str, bad_value: str, expected: str
) -> None:
    payload = _valid_inference_payload()
    payload[field] = bad_value
    with pytest.raises(StrictJsonError, match=expected):
        inference_config_sha256(payload)


def test_inference_config_rejects_bad_instruments() -> None:
    payload = _valid_inference_payload()
    payload["instruments"] = ["piano"]
    with pytest.raises(StrictJsonError, match="instruments"):
        inference_config_sha256(payload)


def test_inference_config_rejects_non_int_sample_rate() -> None:
    payload = _valid_inference_payload()
    payload["input_sample_rate_hz"] = True
    with pytest.raises(StrictJsonError, match="input_sample_rate_hz"):
        inference_config_sha256(payload)


@pytest.mark.parametrize("field", ["chunk_duration_sec", "temperature", "cfg_coef"])
def test_inference_config_rejects_non_numeric(field: str) -> None:
    payload = _valid_inference_payload()
    payload[field] = "x"
    with pytest.raises(StrictJsonError, match="numeric"):
        inference_config_sha256(payload)


@pytest.mark.parametrize("field", ["chunk_duration_sec", "temperature", "cfg_coef"])
def test_inference_config_rejects_non_finite(field: str) -> None:
    payload = _valid_inference_payload()
    payload[field] = Decimal("Infinity")
    with pytest.raises(StrictJsonError, match="finite"):
        inference_config_sha256(payload)


@pytest.mark.parametrize("field", ["use_sampling", "no_eos_is_ok", "prelude_forcing"])
def test_inference_config_rejects_non_bool_flags(field: str) -> None:
    payload = _valid_inference_payload()
    payload[field] = "yes"
    with pytest.raises(StrictJsonError, match="boolean"):
        inference_config_sha256(payload)


@pytest.mark.parametrize("field", ["batch_size", "beam_size"])
def test_inference_config_rejects_non_int_sizes(field: str) -> None:
    payload = _valid_inference_payload()
    payload[field] = True
    with pytest.raises(StrictJsonError, match="integer"):
        inference_config_sha256(payload)


# ---------------------------------------------------------------------------
# _normalize_scope / _validate_scope
# ---------------------------------------------------------------------------


def test_normalize_scope_rejects_str_input() -> None:
    with pytest.raises(ValueError, match="simfile IDs"):
        _normalize_scope("10", ())  # type: ignore[arg-type]


def test_normalize_scope_rejects_non_iterable() -> None:
    with pytest.raises(ValueError, match="simfile IDs"):
        _normalize_scope(10, ())  # type: ignore[arg-type]


def test_normalize_scope_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _normalize_scope((0,), ())


def test_normalize_scope_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _normalize_scope((10, 20), (20,))


def test_validate_scope_rejects_non_iterable_loaded() -> None:
    with pytest.raises(ValueError, match="loaded manifest IDs"):
        _validate_scope((), (), 10)  # type: ignore[arg-type]


def test_validate_scope_rejects_bad_loaded_ids() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        _validate_scope((), (), (0,))


def test_validate_scope_rejects_unknown_exclude() -> None:
    with pytest.raises(ValueError, match="unknown exclude"):
        _validate_scope((), (99,), (10, 20))


# ---------------------------------------------------------------------------
# build_run_id
# ---------------------------------------------------------------------------


def test_build_run_id_rejects_empty_identity_string() -> None:
    with pytest.raises(StrictJsonError, match="nonempty string"):
        build_run_id(
            SHA_A,
            SHA_B,
            SHA_B,
            SHA_C,
            "d" * 40,
            SHA_A,
            SHA_C,
            prediction_map_version="",
        )


# ---------------------------------------------------------------------------
# _normalize_snapshot_value
# ---------------------------------------------------------------------------


def test_normalize_snapshot_value_rejects_non_str_key() -> None:
    with pytest.raises(StrictJsonError, match="keys must be strings"):
        _normalize_snapshot_value({1: "x"})  # type: ignore[dict-item]


def test_normalize_snapshot_value_rejects_unsupported_type() -> None:
    with pytest.raises(StrictJsonError, match="unsupported"):
        _normalize_snapshot_value(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _validate_snapshot via render/parse
# ---------------------------------------------------------------------------


def _valid_snapshot() -> dict[str, object]:
    return {
        "schema": MUSCRIPTOR_CORPUS_RUN_SCHEMA,
        "run_id": "muscriptor-" + SHA_A[:16],
        "items": [
            {"simfile_id": 10, "execution_disposition": "inferred", "rtf": 0.5},
        ],
        "overall_status": "complete",
        "success_count": 1,
        "failed_count": 0,
        "skipped_count": 0,
        "quarantined_count": 0,
        "completed_at": "2026-08-14T00:00:00+00:00",
    }


def test_render_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        render_muscriptor_corpus_run("not-a-mapping")  # type: ignore[arg-type]


def test_validate_rejects_wrong_schema() -> None:
    snapshot = _valid_snapshot()
    snapshot["schema"] = "wrong"
    with pytest.raises(StrictJsonError, match="schema"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_empty_run_id() -> None:
    snapshot = _valid_snapshot()
    snapshot["run_id"] = ""
    with pytest.raises(StrictJsonError, match="run_id"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_non_array_scope() -> None:
    snapshot = _valid_snapshot()
    snapshot["include_simfile_ids"] = "x"
    with pytest.raises(StrictJsonError, match="scope"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_non_array_items() -> None:
    snapshot = _valid_snapshot()
    snapshot["items"] = "x"
    with pytest.raises(StrictJsonError, match="items"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_non_object_item() -> None:
    snapshot = _valid_snapshot()
    snapshot["items"] = ["x"]
    with pytest.raises(StrictJsonError, match="item must be an object"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_invalid_simfile_id() -> None:
    snapshot = _valid_snapshot()
    snapshot["items"] = [{"simfile_id": 0, "execution_disposition": "inferred"}]
    with pytest.raises(StrictJsonError, match="simfile_id"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_duplicate_simfile_ids() -> None:
    snapshot = _valid_snapshot()
    snapshot["items"] = [
        {"simfile_id": 10, "execution_disposition": "inferred"},
        {"simfile_id": 10, "execution_disposition": "inferred"},
    ]
    with pytest.raises(StrictJsonError, match="unique"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_invalid_disposition() -> None:
    snapshot = _valid_snapshot()
    snapshot["items"] = [{"simfile_id": 10, "execution_disposition": "bogus"}]
    with pytest.raises(StrictJsonError, match="disposition"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_skipped_without_exclusion() -> None:
    snapshot = _valid_snapshot()
    snapshot["items"] = [{"simfile_id": 10, "execution_disposition": "skipped"}]
    with pytest.raises(StrictJsonError, match="skipped"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_invalid_overall_status() -> None:
    snapshot = _valid_snapshot()
    snapshot["overall_status"] = "bogus"
    with pytest.raises(StrictJsonError, match="overall_status"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_complete_requires_counts() -> None:
    snapshot = _valid_snapshot()
    snapshot.pop("success_count")
    with pytest.raises(StrictJsonError, match="counts"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_complete_requires_item_dispositions() -> None:
    snapshot = _valid_snapshot()
    snapshot["items"] = [{"simfile_id": 10}]
    with pytest.raises(StrictJsonError, match="dispositions"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_negative_counts() -> None:
    snapshot = _valid_snapshot()
    snapshot["success_count"] = -1
    with pytest.raises(StrictJsonError, match="nonnegative"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_unreconciled_counts() -> None:
    snapshot = _valid_snapshot()
    snapshot["success_count"] = 5
    with pytest.raises(StrictJsonError, match="reconcile"):
        render_muscriptor_corpus_run(snapshot)


def test_validate_rejects_completed_at_on_non_complete() -> None:
    snapshot = _valid_snapshot()
    snapshot["overall_status"] = "partial"
    snapshot["completed_at"] = "2026-08-14T00:00:00+00:00"
    with pytest.raises(StrictJsonError, match="completed"):
        render_muscriptor_corpus_run(snapshot)


def test_parse_rejects_non_object() -> None:
    with pytest.raises(StrictJsonError, match="object|canonical"):
        parse_muscriptor_corpus_run(b"[]\n")


def test_parse_rejects_non_canonical() -> None:
    valid = render_muscriptor_corpus_run(_valid_snapshot())
    # Re-serialize with different key order / spacing to break canonicity.
    import json

    parsed = json.loads(valid)
    non_canonical = (json.dumps(parsed, separators=(",", ":")) + "\n").encode("utf-8")
    with pytest.raises(StrictJsonError, match="canonical"):
        parse_muscriptor_corpus_run(non_canonical)


def test_parse_rejects_run_id_mismatch() -> None:
    valid = render_muscriptor_corpus_run(_valid_snapshot())
    with pytest.raises(StrictJsonError, match="run_id"):
        parse_muscriptor_corpus_run(valid, expected_run_id="muscriptor-different")


def test_write_rejects_non_path(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Path"):
        write_muscriptor_corpus_run("not-a-path", _valid_snapshot())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _expected_muscriptor_descriptor
# ---------------------------------------------------------------------------


def test_expected_descriptor_rejects_non_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.benchmark.muscriptor_corpus_run as run_module

    lock = _lock()
    monkeypatch.setattr(run_module, "descriptor_for_lock", lambda _lock: "not-a-descriptor")
    with pytest.raises(StrictJsonError, match="descriptor is invalid"):
        _expected_muscriptor_descriptor(lock)


def test_expected_descriptor_rejects_non_normalizable(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.benchmark.muscriptor_corpus_run as run_module

    lock = _lock()

    class BadDescriptor:
        sha256 = "x"
        payload = {"model_id": "wrong"}

    monkeypatch.setattr(run_module, "descriptor_for_lock", lambda _lock: BadDescriptor())
    with pytest.raises(StrictJsonError, match="descriptor is invalid"):
        _expected_muscriptor_descriptor(lock)


def test_expected_descriptor_rejects_payload_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.benchmark.muscriptor_corpus_run as run_module

    lock = _lock()
    real = _expected_muscriptor_descriptor(lock)
    altered_payload = dict(real.payload)
    altered_payload["model_id"] = "muscriptor-medium-aaaaaaaaaaaa-bbbbbbbbbbbb"

    class MismatchedDescriptor:
        sha256 = real.sha256
        payload = altered_payload

    monkeypatch.setattr(run_module, "descriptor_for_lock", lambda _lock: MismatchedDescriptor())
    with pytest.raises(StrictJsonError, match="descriptor is invalid"):
        _expected_muscriptor_descriptor(lock)


def test_expected_descriptor_rejects_model_id_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.benchmark.muscriptor_corpus_run as run_module

    lock = _lock()
    real = _expected_muscriptor_descriptor(lock)
    altered_payload = dict(real.payload)
    altered_payload["model_id"] = "wrong-model-id"

    class MismatchedDescriptor:
        sha256 = real.sha256
        payload = altered_payload

    monkeypatch.setattr(run_module, "descriptor_for_lock", lambda _lock: MismatchedDescriptor())
    with pytest.raises(StrictJsonError, match="descriptor is invalid"):
        _expected_muscriptor_descriptor(lock)


# ---------------------------------------------------------------------------
# _timestamp / _bounded_error / _bounded_close_error
# ---------------------------------------------------------------------------


def test_timestamp_rejects_non_datetime() -> None:
    with pytest.raises(TypeError, match="datetime"):
        _timestamp("not-a-datetime")  # type: ignore[arg-type]


def test_timestamp_accepts_naive_datetime() -> None:
    result = _timestamp(datetime(2026, 8, 14))
    assert result.endswith("+00:00")


def test_bounded_error_uses_type_name_for_empty_message() -> None:
    assert _bounded_error(ValueError("")) == "ValueError"
    assert _bounded_error(ValueError("x\x00y")) == "x y"


def test_bounded_close_error_falls_back_for_non_str_code() -> None:
    error = RuntimeError("boom")
    error.code = 123  # type: ignore[attr-defined]
    result = _bounded_close_error(error)
    assert result["code"] == "worker_close_failed"
    assert result["message"] == "boom"


# ---------------------------------------------------------------------------
# _snapshot_counts / _finite_positive / _project_runtime
# ---------------------------------------------------------------------------


def test_snapshot_counts_classifies_all_dispositions() -> None:
    counts = _snapshot_counts(
        [
            {"execution_disposition": "inferred"},
            {"execution_disposition": "resumed"},
            {"execution_disposition": "failed"},
            {"execution_disposition": "skipped"},
            {"execution_disposition": "quarantined"},
            {"execution_disposition": None},
        ]
    )
    assert counts == {
        "success_count": 2,
        "failed_count": 1,
        "skipped_count": 1,
        "quarantined_count": 1,
    }


def test_finite_positive_rejects_zero_without_allow() -> None:
    assert _finite_positive(0) is None
    assert _finite_positive(0, allow_zero=True) == 0.0
    assert _finite_positive(-1) is None
    assert _finite_positive(True) is None
    assert _finite_positive("x") is None


def test_project_runtime_skips_rows_missing_timing() -> None:
    runtime = _project_runtime(
        (
            {
                "execution_disposition": "inferred",
                "wall_time_sec": None,
                "source_duration_sec": 1.0,
            },
            {
                "execution_disposition": "inferred",
                "wall_time_sec": 1.0,
                "source_duration_sec": None,
            },
            {"execution_disposition": "failed", "wall_time_sec": 1.0, "source_duration_sec": 1.0},
        ),
        eligible_audio_durations=(1.0, 2.0, 3.0),
    )
    assert runtime["aggregate_rtf"] is None
    assert runtime["projected_full_wall_time_sec"] is None
    assert runtime["eligible_audio_duration_coverage_count"] == 3


# ---------------------------------------------------------------------------
# _process_peak_rss_bytes / _device_peak_memory_bytes
# ---------------------------------------------------------------------------


def test_process_peak_rss_returns_int_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    import resource as resource_module
    import sys as sys_module

    class FakeRusage:
        ru_maxrss = 123456

    monkeypatch.setattr(resource_module, "getrusage", lambda _who: FakeRusage())
    monkeypatch.setattr(sys_module, "platform", "darwin")
    assert _process_peak_rss_bytes() == 123456


def test_process_peak_rss_returns_none_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import resource as resource_module

    def raise_error(_who):
        raise OSError("nope")

    monkeypatch.setattr(resource_module, "getrusage", raise_error)
    assert _process_peak_rss_bytes() is None


def test_process_peak_rss_returns_none_for_bad_value(monkeypatch: pytest.MonkeyPatch) -> None:
    import resource as resource_module

    class FakeRusage:
        ru_maxrss = -1

    monkeypatch.setattr(resource_module, "getrusage", lambda _who: FakeRusage())
    assert _process_peak_rss_bytes() is None


def test_device_peak_memory_returns_none_for_no_backend() -> None:
    assert _device_peak_memory_bytes(None) is None


def test_device_peak_memory_reads_callable_attribute() -> None:
    class Backend:
        def device_peak_memory_bytes(self) -> int:
            return 4096

    assert _device_peak_memory_bytes(Backend()) == 4096


def test_device_peak_memory_returns_none_when_callable_raises() -> None:
    class Backend:
        def device_peak_memory_bytes(self) -> int:
            raise RuntimeError("boom")

    assert _device_peak_memory_bytes(Backend()) is None


def test_device_peak_memory_skips_invalid_values() -> None:
    class Backend:
        device_peak_memory_bytes = -1

        def peak_memory_bytes(self) -> int:
            return 8192

    assert _device_peak_memory_bytes(Backend()) == 8192


# ---------------------------------------------------------------------------
# _source_failure_code / _set_skipped / _set_quarantined / _remove_temporary_input
# ---------------------------------------------------------------------------


def test_source_failure_code_classifies_decode_errors() -> None:
    assert _source_failure_code(OSError("invalid wav header")) == "source_audio_decode_failed"
    assert _source_failure_code(RuntimeError("decode failed")) == "source_audio_decode_failed"
    assert _source_failure_code(ValueError("unreadable bytes")) == "source_audio_decode_failed"
    assert _source_failure_code(ValueError("network down")) == "source_audio_unavailable"


def test_set_skipped_and_quarantined_set_disposition() -> None:
    item: dict[str, object] = {}
    from src.benchmark.muscriptor_corpus_run import _set_quarantined, _set_skipped

    _set_skipped(item)
    assert item["execution_disposition"] == "skipped"
    assert item["runner_failure_code"] == "explicitly_skipped"
    _set_quarantined(item)
    assert item["execution_disposition"] == "quarantined"
    assert item["runner_failure_code"] == "reference_quarantined"


def test_remove_temporary_input_ignores_paths_outside_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "file.wav"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"x")
    _remove_temporary_input(outside, tmp_path / "input_root")
    assert outside.exists()


def test_remove_temporary_input_swallows_unlink_errors(tmp_path: Path) -> None:
    input_root = tmp_path / "input_root"
    input_root.mkdir()
    target = input_root / "file.wav"
    target.write_bytes(b"x")
    # Make the parent directory read-only so unlink fails, then restore.
    input_root.chmod(0o555)
    try:
        _remove_temporary_input(target, input_root)
    finally:
        input_root.chmod(0o755)


# ---------------------------------------------------------------------------
# _read_existing_prediction
# ---------------------------------------------------------------------------


def test_read_existing_prediction_handles_missing_and_unreadable(tmp_path: Path) -> None:
    exists, content = _read_existing_prediction(tmp_path / "missing.jsonl")
    assert (exists, content) == (False, None)

    target = tmp_path / "pred.jsonl"
    target.write_bytes(b"payload")
    exists, content = _read_existing_prediction(target)
    assert (exists, content) == (True, b"payload")


def test_read_existing_prediction_returns_none_on_unreadable(tmp_path: Path) -> None:
    target = tmp_path / "pred.jsonl"
    target.mkdir()  # a directory is unreadable as a regular file
    exists, content = _read_existing_prediction(target)
    assert exists is True
    assert content is None


# ---------------------------------------------------------------------------
# _prediction_artifact_matches
# ---------------------------------------------------------------------------


def _cohort_identity() -> CohortIdentity:
    return CohortIdentity(
        cohort_id="muscriptor-test",
        reference_manifest_sha256=SHA_A,
        reference_timing_version="timing-v1",
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id="muscriptor-v0.3.0-drums-v1",
        model_id="muscriptor-medium-aaaaaaaaaaaa-bbbbbbbbbbbb",
        model_lock_sha256=SHA_C,
        backend_descriptor_sha256=SHA_B,
        prediction_map_version=MUSCRIPTOR_PREDICTION_MAP_ID,
        input_view_id=MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
    )


def test_prediction_artifact_matches_detects_each_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)
    source = ResolvedSourceAudio(
        path=tmp_path / "s.wav",
        source_audio_id="10/audio.wav",
        source_audio_sha256=SHA_A,
        duration_sec=1.0,
        content=None,
    )
    audio = CanonicalAudio(
        path=tmp_path / "a.wav",
        source_audio_id="10/audio.wav",
        source_audio_sha256=SHA_A,
        input_view_id=MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
        input_audio_sha256=SHA_B,
        byte_length=46,
        sample_rate=44100,
        channel_count=1,
        sample_width_bytes=2,
        audio_frame_count=1,
    )
    artifact = _publish_artifact(tmp_path, audio, descriptor)

    assert _prediction_artifact_matches(artifact, source=source, audio=audio, descriptor=descriptor)

    # descriptor sha256 mismatch
    other_descriptor = replace(descriptor, sha256="c" * 64)
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=audio, descriptor=other_descriptor
    )

    # descriptor payload mismatch
    other_descriptor = replace(descriptor, payload={**dict(descriptor.payload), "model_id": "x"})
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=audio, descriptor=other_descriptor
    )

    # source_audio_id mismatch
    other_source = replace(source, source_audio_id="99/audio.wav")
    assert not _prediction_artifact_matches(
        artifact, source=other_source, audio=audio, descriptor=descriptor
    )

    # source_audio_sha256 mismatch
    other_source = replace(source, source_audio_sha256="z" * 64)
    assert not _prediction_artifact_matches(
        artifact, source=other_source, audio=audio, descriptor=descriptor
    )

    # input_view_id mismatch on prediction
    other_audio = replace(audio, input_view_id="other-view")
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=other_audio, descriptor=descriptor
    )

    # input_audio_sha256 mismatch
    other_audio = replace(audio, input_audio_sha256="q" * 64)
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=other_audio, descriptor=descriptor
    )

    # source_audio_id mismatch on audio
    other_audio = replace(audio, source_audio_id="99/audio.wav")
    assert not _prediction_artifact_matches(
        artifact, source=source, audio=other_audio, descriptor=descriptor
    )


def _publish_artifact(tmp_path: Path, audio: CanonicalAudio, descriptor: BackendDescriptor):
    from src.benchmark.mapping import map_muscriptor_prediction
    from src.benchmark.prediction_artifact import (
        prediction_path,
        publish_prediction_artifact,
        read_prediction_artifact,
    )

    native = NativePrediction(
        audio=audio,
        descriptor=descriptor,
        events=(
            NativeEvent(
                time_sec=0.25,
                native_class_id="drums:midi_36",
                model_output_bin=None,
                native_midi_note=36,
                native_metadata={"instrument_group": "drums"},
                confidence=None,
                velocity_midi=None,
            ),
        ),
    )
    mapped, _ = map_muscriptor_prediction(native)
    target = prediction_path(
        tmp_path,
        simfile_id=10,
        source_audio_sha256=audio.source_audio_sha256,
        backend_descriptor_sha256=descriptor.sha256,
        inference_config_sha256=SHA_C,
    )
    publish_prediction_artifact(target, mapped)
    return read_prediction_artifact(target.read_bytes())


# ---------------------------------------------------------------------------
# _cohort_item_without_prediction / _cohort_item_from_run_row
# ---------------------------------------------------------------------------


def test_cohort_item_without_prediction_handles_missing_mapping() -> None:
    identity = _cohort_identity()
    item = _cohort_item_without_prediction(
        identity,
        "10",
        mapping=None,
        status="failed",
        failure_reason="inference_failed",
    )
    assert item.status == "failed"
    assert item.failure_reason == "inference_failed"
    assert item.reference_events == ()


def test_cohort_item_from_run_row_rejects_bad_types() -> None:
    identity = _cohort_identity()
    with pytest.raises(TypeError, match="identity"):
        _cohort_item_from_run_row(  # type: ignore[arg-type]
            "not-identity", {"simfile_id": 10}, None, output_dir=Path("out")
        )
    with pytest.raises(TypeError, match="run row"):
        _cohort_item_from_run_row(identity, "not-a-row", None, output_dir=Path("out"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="output_dir"):
        _cohort_item_from_run_row(identity, {"simfile_id": 10}, None, output_dir="out")  # type: ignore[arg-type]


def test_cohort_item_from_run_row_rejects_bad_simfile_id() -> None:
    identity = _cohort_identity()
    with pytest.raises(ValueError, match="simfile_id"):
        _cohort_item_from_run_row(identity, {"simfile_id": True}, None, output_dir=Path("out"))
    with pytest.raises(ValueError, match="simfile_id"):
        _cohort_item_from_run_row(identity, {"simfile_id": ""}, None, output_dir=Path("out"))


def test_cohort_item_from_run_row_inferred_missing_prediction_path() -> None:
    identity = _cohort_identity()
    item = _cohort_item_from_run_row(
        identity,
        {"simfile_id": 10, "execution_disposition": "inferred"},
        None,
        output_dir=Path("out"),
    )
    assert item.status == "failed"
    assert item.failure_reason == "prediction_missing"


def test_cohort_item_from_run_row_inferred_missing_file(tmp_path: Path) -> None:
    identity = _cohort_identity()
    item = _cohort_item_from_run_row(
        identity,
        {"simfile_id": 10, "execution_disposition": "inferred", "prediction_path": "missing.jsonl"},
        None,
        output_dir=tmp_path,
    )
    assert item.failure_reason == "prediction_missing"


def test_cohort_item_from_run_row_inferred_unreadable_file(tmp_path: Path) -> None:
    identity = _cohort_identity()
    target = tmp_path / "pred.jsonl"
    target.mkdir()  # unreadable as a file
    item = _cohort_item_from_run_row(
        identity,
        {"simfile_id": 10, "execution_disposition": "inferred", "prediction_path": "pred.jsonl"},
        None,
        output_dir=tmp_path,
    )
    assert item.failure_reason == "prediction_artifact_invalid"


def test_cohort_item_from_run_row_inferred_no_mapping(tmp_path: Path) -> None:
    identity = _cohort_identity()
    target = tmp_path / "pred.jsonl"
    target.write_bytes(b"{}")
    item = _cohort_item_from_run_row(
        identity,
        {"simfile_id": 10, "execution_disposition": "inferred", "prediction_path": "pred.jsonl"},
        None,
        output_dir=tmp_path,
    )
    assert item.failure_reason == "prediction_artifact_invalid"


def test_cohort_item_from_run_row_quarantined_disposition() -> None:
    identity = _cohort_identity()
    item = _cohort_item_from_run_row(
        identity,
        {"simfile_id": 10, "execution_disposition": "quarantined"},
        None,
        output_dir=Path("out"),
    )
    assert item.status == "quarantined"


def test_cohort_item_from_run_row_skipped_disposition_uses_runner_code() -> None:
    identity = _cohort_identity()
    item = _cohort_item_from_run_row(
        identity,
        {
            "simfile_id": 10,
            "execution_disposition": "skipped",
            "runner_failure_code": "explicitly_skipped",
        },
        None,
        output_dir=Path("out"),
    )
    assert item.status == "skipped"
    assert item.failure_reason == "explicitly_skipped"


def test_cohort_item_from_run_row_failed_disposition_defaults_runner_code() -> None:
    identity = _cohort_identity()
    item = _cohort_item_from_run_row(
        identity,
        {
            "simfile_id": 10,
            "execution_disposition": "failed",
            "runner_failure_code": "inference_failed",
        },
        None,
        output_dir=Path("out"),
    )
    assert item.status == "failed"
    assert item.failure_reason == "inference_failed"

    item_no_code = _cohort_item_from_run_row(
        identity,
        {"simfile_id": 10, "execution_disposition": "failed"},
        None,
        output_dir=Path("out"),
    )
    assert item_no_code.failure_reason == "backend_unavailable"


# ---------------------------------------------------------------------------
# build_muscriptor_cohort_from_snapshot / _cohort_identity_from_snapshot
# ---------------------------------------------------------------------------


def test_build_cohort_rejects_bad_types() -> None:
    with pytest.raises(TypeError, match="snapshot"):
        build_muscriptor_cohort_from_snapshot("not-a-mapping", mappings={}, output_dir=Path("out"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="output_dir"):
        build_muscriptor_cohort_from_snapshot({"items": []}, mappings={}, output_dir="out")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="items"):
        build_muscriptor_cohort_from_snapshot({"items": "x"}, mappings={}, output_dir=Path("out"))


def test_cohort_identity_from_snapshot_rejects_missing_descriptor() -> None:
    with pytest.raises(ValueError, match="descriptor"):
        _cohort_identity_from_snapshot({"run_id": "x"})


def test_cohort_identity_from_snapshot_rejects_incomplete_identity() -> None:
    snapshot = {
        "backend_descriptor": {"backend_id": "muscriptor-v0.3.0-drums-v1"},
        "run_id": "muscriptor-x",
        "reference_manifest_sha256": SHA_A,
        "reference_timing_version": "timing-v1",
        "model_id": "model",
        "model_lock_sha256": SHA_C,
        "backend_descriptor_sha256": SHA_B,
        "prediction_map_version": MUSCRIPTOR_PREDICTION_MAP_ID,
        "input_view_id": MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
    }
    # Remove one required field to make identity incomplete.
    del snapshot["model_lock_sha256"]
    with pytest.raises(ValueError, match="cohort identity"):
        _cohort_identity_from_snapshot(snapshot)


# ---------------------------------------------------------------------------
# classify_muscriptor_backend_error / _prior_source_matches
# ---------------------------------------------------------------------------


def test_classify_backend_error_handles_non_str_code() -> None:
    assert classify_muscriptor_backend_error(123) == ("worker_protocol_failed", "poison")  # type: ignore[arg-type]
    assert classify_muscriptor_backend_error("unknown_code") == ("worker_protocol_failed", "poison")


def test_prior_source_matches_detects_mismatches() -> None:
    source = ResolvedSourceAudio(
        path=Path("s.wav"),
        source_audio_id="10/audio.wav",
        source_audio_sha256=SHA_A,
        duration_sec=1.0,
        content=None,
    )
    assert _prior_source_matches({}, source) is True
    assert not _prior_source_matches({"source_audio_id": "99/audio.wav"}, source)
    assert not _prior_source_matches({"source_audio_sha256": "z" * 64}, source)
    assert not _prior_source_matches({"source_duration_sec": "bad"}, source)
    assert not _prior_source_matches({"source_duration_sec": 2.0}, source)
    assert _prior_source_matches({"source_duration_sec": 1.0}, source)


# ---------------------------------------------------------------------------
# Runner integration: resume and error branches
# ---------------------------------------------------------------------------


def test_run_rejects_non_request() -> None:
    with pytest.raises(TypeError, match="request must be"):
        run_muscriptor_corpus("not-a-request")  # type: ignore[arg-type]


def test_run_fatal_when_manifests_unloadable(tmp_path: Path) -> None:
    outcome = run_muscriptor_corpus(_request(tmp_path))
    assert outcome.overall_status == "failed"
    assert outcome.exit_code == 2


def test_run_fatal_when_clock_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_seams(monkeypatch, tmp_path)

    def bad_clock():
        raise OSError("clock broken")

    outcome = run_muscriptor_corpus(_request(tmp_path), clock=bad_clock)
    assert outcome.exit_code == 2


def test_run_fatal_when_existing_run_without_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    first = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0

    # Re-run without resume: existing run.json must be fatal.
    second = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert second.exit_code == 2


def test_run_resume_succeeds_when_evidence_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    first = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0

    resumed = run_muscriptor_corpus(
        _request(tmp_path, resume=True),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert resumed.exit_code == 0
    rows = parse_muscriptor_corpus_run(resumed.run_path.read_bytes())["items"]
    assert all(row["execution_disposition"] == "resumed" for row in rows)


def test_run_resume_fatal_when_prior_snapshot_header_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    first = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0
    assert first.run_path is not None

    # Corrupt the prior snapshot so resume parsing fails.
    first.run_path.write_bytes(b"{not valid json}\n")
    resumed = run_muscriptor_corpus(
        _request(tmp_path, resume=True),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert resumed.exit_code == 2


def test_run_source_resolution_failure_marks_item_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)

    def failing_resolve(source, *_args, **kwargs):
        simfile_id = int(source["source_audio_key"].split("/", 1)[0])
        if simfile_id == 20:
            raise OSError("source unavailable")
        return ResolvedSourceAudio(
            path=tmp_path / f"{simfile_id}.wav",
            source_audio_id=source["source_audio_key"],
            source_audio_sha256=SHA_A,
            duration_sec=1.0,
            content=b"source bytes" if kwargs["load_body"] else None,
        )

    monkeypatch.setattr(
        __import__("src.benchmark.muscriptor_corpus_run", fromlist=["resolve_source_audio"]),
        "resolve_source_audio",
        failing_resolve,
    )

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    outcome = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    rows = {
        row["simfile_id"]: row
        for row in parse_muscriptor_corpus_run(outcome.run_path.read_bytes())["items"]
    }
    assert rows[20]["runner_failure_code"] == "source_audio_decode_failed"


def test_run_resume_rejects_prediction_with_changed_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    first = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0

    # Change the resolved source sha on the second pass for one item.
    import src.benchmark.muscriptor_corpus_run as run_module

    original_resolve = run_module.resolve_source_audio
    call_count = {"n": 0}

    def shifting_resolve(source, *_args, **kwargs):
        call_count["n"] += 1
        result = original_resolve(source, *_args, **kwargs)
        if kwargs["load_body"] and int(source["source_audio_key"].split("/", 1)[0]) == 10:
            return replace(result, source_audio_sha256="z" * 64)
        return result

    monkeypatch.setattr(run_module, "resolve_source_audio", shifting_resolve)
    resumed = run_muscriptor_corpus(
        _request(tmp_path, resume=True),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    rows = {
        row["simfile_id"]: row
        for row in parse_muscriptor_corpus_run(resumed.run_path.read_bytes())["items"]
    }
    assert rows[10]["runner_failure_code"] == "source_audio_unavailable"


def test_run_native_prediction_identity_change_is_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)

    class ShiftingBackend:
        def __init__(self) -> None:
            self._call = False

        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            self._call = not self._call
            pred = _prediction(audio, descriptor)
            if not self._call:
                return replace(pred, audio=replace(audio, source_audio_id="changed"))
            return pred

        def close(self) -> None:
            return None

    outcome = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: ShiftingBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.exit_code == 1


def test_run_backend_descriptor_identity_change_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)
    altered = replace(descriptor, sha256="z" * 64)

    class BadDescriptorBackend:
        def descriptor(self) -> BackendDescriptor:
            return altered

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, altered)

        def close(self) -> None:
            return None

    outcome = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: BadDescriptorBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert outcome.exit_code == 2


def test_run_backend_close_error_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)

    class CloseFailingBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            raise OSError("close failed")

    outcome = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: CloseFailingBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    snapshot = parse_muscriptor_corpus_run(outcome.run_path.read_bytes())
    assert snapshot.get("close_error") is not None


def test_run_device_peak_memory_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)

    class MemoryBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

        def device_peak_memory_bytes(self) -> int:
            return 1048576

    outcome = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: MemoryBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    snapshot = parse_muscriptor_corpus_run(outcome.run_path.read_bytes())
    assert snapshot["device_peak_memory_bytes"] == 1048576


def test_run_excludes_and_skips_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    outcome = run_muscriptor_corpus(
        replace(_request(tmp_path), exclude_simfile_ids=(20,)),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    rows = {
        row["simfile_id"]: row
        for row in parse_muscriptor_corpus_run(outcome.run_path.read_bytes())["items"]
    }
    assert rows[20]["execution_disposition"] == "skipped"
    assert rows[10]["execution_disposition"] == "inferred"


def test_run_quarantines_ineligible_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.benchmark.muscriptor_corpus_run as run_module

    lock, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)
    reference_manifest, timing_manifest = _ineligible_manifests()
    monkeypatch.setattr(run_module, "load_reference_set_manifest", lambda _: reference_manifest)
    monkeypatch.setattr(run_module, "load_reference_timing_manifest", lambda _: timing_manifest)
    monkeypatch.setattr(
        run_module,
        "preflight_reference_mappings",
        lambda *_a, **_k: {10: _mapping(10)},
    )

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    outcome = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    rows = {
        row["simfile_id"]: row
        for row in parse_muscriptor_corpus_run(outcome.run_path.read_bytes())["items"]
    }
    assert rows[20]["execution_disposition"] == "quarantined"


def _ineligible_manifests():
    from src.benchmark.reference_set_manifest import (
        LoadedReferenceSetManifest,
        LoadedReferenceSetRow,
        ReferenceSetRowView,
    )
    from src.benchmark.reference_timing_manifest import LoadedReferenceTimingManifest

    rows = (
        LoadedReferenceSetRow(
            source_row={
                "selected_chart_key": "10/chart.dtx",
                "selected_chart_content_hash": SHA_A,
                "source_audio_key": "10/audio.wav",
                "source_audio_content_hash": SHA_A,
                "source_endpoint_sha256": SHA_A,
                "source_bucket": "simfile-dtx",
            },
            view=ReferenceSetRowView(
                simfile_id=10,
                eligibility_status="eligible",
                eligibility_reason_codes=(),
                eligibility_warnings=(),
                mapped_event_count=1,
                common_scored_event_count=1,
                ignored_event_count=0,
                unmapped_event_count=0,
                duplicate_common_event_count=0,
            ),
        ),
        LoadedReferenceSetRow(
            source_row={
                "selected_chart_key": "20/chart.dtx",
                "selected_chart_content_hash": SHA_A,
                "source_audio_key": "20/audio.wav",
                "source_audio_content_hash": SHA_A,
                "source_endpoint_sha256": SHA_A,
                "source_bucket": "simfile-dtx",
            },
            view=ReferenceSetRowView(
                simfile_id=20,
                eligibility_status="ineligible",
                eligibility_reason_codes=("no_dtx_lanes",),
                eligibility_warnings=(),
                mapped_event_count=0,
                common_scored_event_count=0,
                ignored_event_count=0,
                unmapped_event_count=0,
                duplicate_common_event_count=0,
            ),
        ),
    )
    return (
        LoadedReferenceSetManifest(
            manifest_sha256=SHA_A,
            corpus_version="reference-v1",
            source_reference_timing_manifest_sha256=SHA_B,
            source_reference_timing_version="timing-v1",
            rows=rows,
        ),
        LoadedReferenceTimingManifest(
            manifest_sha256=SHA_B,
            corpus_version="timing-v1",
            rows=(),
        ),
    )


def test_run_prediction_output_conflict_marks_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, descriptor, _, _ = _install_seams(monkeypatch, tmp_path)

    class HealthyBackend:
        def descriptor(self) -> BackendDescriptor:
            return descriptor

        def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
            return _prediction(audio, descriptor)

        def close(self) -> None:
            return None

    first = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    assert first.exit_code == 0
    assert first.run_path is not None
    first.run_path.unlink()  # force a fresh non-resume run but keep predictions

    second = run_muscriptor_corpus(
        _request(tmp_path),
        backend_factory=lambda **_: HealthyBackend(),
        perf_counter=lambda: 0.0,
        clock=lambda: datetime(2026, 8, 14, tzinfo=timezone.utc),
    )
    rows = {
        row["simfile_id"]: row
        for row in parse_muscriptor_corpus_run(second.run_path.read_bytes())["items"]
    }
    assert all(row["runner_failure_code"] == "prediction_output_conflict" for row in rows.values())


def test_runner_failure_mapping_covers_all_reasons() -> None:
    assert set(RUNNER_FAILURE_TO_COHORT_REASON.values()) <= {
        "reference_quarantined",
        "backend_unavailable",
        "inference_failed",
        "prediction_artifact_invalid",
        "prediction_missing",
        "explicitly_skipped",
    }
