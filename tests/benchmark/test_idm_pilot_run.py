from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from src.benchmark.cohort_scoring import COHORT_FAILURE_REASONS
from src.benchmark.idm_pilot_run import (
    IDM_FAILURE_TO_COHORT_REASON,
    IDM_PILOT_RUN_SCHEMA,
    IDM_STEM_INPUT_VIEW_ID,
    IdmPilotRunRequest,
    build_idm_inference_config,
    build_run_id,
    classify_idm_backend_error,
    idm_inference_config_sha256,
)


def test_fixed_failure_mapping_is_closed_and_preserves_upstream_stem_semantics() -> None:
    assert set(IDM_FAILURE_TO_COHORT_REASON.values()) <= COHORT_FAILURE_REASONS
    assert IDM_FAILURE_TO_COHORT_REASON["upstream_stem_unavailable"] == "inference_failed"


def test_idm_pilot_request_has_only_explicit_handoff_and_runtime_inputs() -> None:
    names = {field.name for field in fields(IdmPilotRunRequest)}
    assert "source_cache_dir" not in names
    request = IdmPilotRunRequest(
        separation_handoff_path=Path("handoff.jsonl"),
        reference_manifest_path=Path("reference.jsonl"),
        timing_manifest_path=Path("timing.jsonl"),
        separation_artifact_root=Path("separation"),
        stem_cache_root=Path("stems"),
        output_dir=Path("output"),
        model_lock_path=Path("model.json"),
        model_root=Path("model"),
        runtime_python=Path("python"),
    )
    assert request.resume is False


def test_run_identity_binds_schema_lineage_backend_config_view_and_commit() -> None:
    kwargs = {
        "handoff_manifest_sha256": "a" * 64,
        "handoff_manifest_version": "sha256:" + "b" * 64,
        "reference_manifest_sha256": "c" * 64,
        "reference_manifest_version": "sha256:" + "d" * 64,
        "timing_manifest_sha256": "e" * 64,
        "timing_manifest_version": "sha256:" + "f" * 64,
        "backend_descriptor_sha256": "1" * 64,
        "model_lock_sha256": "2" * 64,
        "inference_config_sha256": "3" * 64,
        "input_view_id": IDM_STEM_INPUT_VIEW_ID,
        "crux_commit": "4" * 40,
    }
    first = build_run_id(**kwargs)
    second = build_run_id(**{**kwargs, "inference_config_sha256": "5" * 64})
    assert first.startswith("idm-")
    assert first != second
    assert IDM_PILOT_RUN_SCHEMA == "crux.idm-stem-pilot-run/v1"


def test_timeout_is_part_of_the_idm_inference_identity() -> None:
    default = build_idm_inference_config("1" * 64, "2" * 64)
    changed = build_idm_inference_config("1" * 64, "2" * 64, timeout_seconds=1799)
    assert idm_inference_config_sha256(default) != idm_inference_config_sha256(changed)


def test_unknown_backend_codes_poison_the_persistent_worker() -> None:
    assert classify_idm_backend_error("future_worker_code") == (
        "worker_protocol_failed",
        "poison",
    )


@pytest.mark.parametrize(
    "field",
    [
        "separation_handoff_path",
        "reference_manifest_path",
        "timing_manifest_path",
        "separation_artifact_root",
        "stem_cache_root",
        "output_dir",
        "model_lock_path",
        "model_root",
        "runtime_python",
    ],
)
def test_idm_pilot_request_rejects_non_path_values(field: str) -> None:
    kwargs = {
        "separation_handoff_path": Path("handoff.jsonl"),
        "reference_manifest_path": Path("reference.jsonl"),
        "timing_manifest_path": Path("timing.jsonl"),
        "separation_artifact_root": Path("separation"),
        "stem_cache_root": Path("stems"),
        "output_dir": Path("output"),
        "model_lock_path": Path("model.json"),
        "model_root": Path("model"),
        "runtime_python": Path("python"),
    }
    kwargs[field] = "not-a-path"
    with pytest.raises(TypeError):
        IdmPilotRunRequest(**kwargs)  # type: ignore[arg-type]
