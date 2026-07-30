from __future__ import annotations

import os
import shutil
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from src.benchmark.backend_identity import JsonValue, strict_json_loads
from src.benchmark.backend_lock import (
    load_backend_lock,
    load_conversion_audit,
    load_runtime_lock,
    load_seal_evidence,
    validate_oaf_lock_set,
)
from src.benchmark.backends import NativeEvent, NativePrediction
from src.benchmark.backends.oaf_tf1 import OafBackendConfig, OafTf1Backend
from src.benchmark.input_view import load_direct_audio
from src.benchmark.prediction_artifact import render_prediction_artifact
from tools.hpa320.seal_oaf_backend import load_native_host_evidence

# The native acceptance path intentionally keeps all three runs and their identities together.
# pylint: disable=too-many-locals

BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"
NATIVE_TEST_FLAG = "HPA320_RUN_NATIVE_OAF_REAL_CHECKPOINT"
NATIVE_HOST_EVIDENCE = Path("/workspace/hpa320/native-host-evidence.json")

pytestmark = pytest.mark.skipif(
    os.environ.get(NATIVE_TEST_FLAG) != "1",
    reason=f"set {NATIVE_TEST_FLAG}=1 only on the accepted native seal worker",
)


def _strict_oracle(path: Path) -> dict[str, JsonValue]:
    content = path.read_bytes()
    assert content.endswith(b"\n")
    value = strict_json_loads(content[:-1], require_canonical=True)
    assert isinstance(value, dict)
    assert value["schema"] == "crux.oaf-smoke-oracle/v1"
    assert isinstance(value["native_events"], list) and value["native_events"]
    return value


def _oracle_event(value: JsonValue) -> NativeEvent:
    assert isinstance(value, dict)
    return NativeEvent(
        time_sec=float(Decimal(cast(str, value["time_sec_raw"]))),
        native_class_id=cast(str, value["native_class_id"]),
        model_output_bin=cast(int, value["model_output_bin"]),
        native_midi_note=cast(int, value["native_midi_note"]),
        native_metadata={"upstream_8hit_group_id": cast(str | None, value["upstream_group_id"])},
        confidence=float(Decimal(cast(str, value["confidence_raw"]))),
        velocity_midi=cast(int, value["velocity"]),
    )


def test_real_checkpoint_is_exact_twice_in_process_and_once_fresh(
    tmp_path: Path,
) -> None:
    repository = Path.cwd().resolve(strict=True)
    config_root = repository / "config/benchmark/backends"
    backend_lock = load_backend_lock(config_root / f"{BACKEND_ID}.backend-lock.json")
    runtime_lock = load_runtime_lock(config_root / f"{BACKEND_ID}.runtime-lock.json")
    seal = load_seal_evidence(config_root / f"{BACKEND_ID}.seal-evidence.json")
    audit = load_conversion_audit(
        repository / "docs/superpowers/evidence/hpa-320/legacy-tf2-conversion-coverage.json"
    )
    validate_oaf_lock_set(backend_lock, runtime_lock, seal, audit)
    host_evidence = load_native_host_evidence(NATIVE_HOST_EVIDENCE)

    checked_fixture = repository / "tests/fixtures/oaf_tf1_smoke"
    oracle = _strict_oracle(checked_fixture / "smoke-oracle.json")
    input_root = tmp_path / "inputs"
    smoke_root = input_root / "smoke"
    smoke_root.mkdir(parents=True)
    shutil.copyfile(checked_fixture / "canonical.wav", smoke_root / "canonical.wav")
    shutil.copyfile(checked_fixture / "smoke-oracle.json", smoke_root / "smoke-oracle.json")

    model_identity = cast(str, backend_lock.descriptor.payload["model_artifact_set_sha256"])
    model_cache = repository / "artifacts/benchmark/model-cache/sha256" / model_identity
    config = OafBackendConfig(
        backend_lock_path=backend_lock.path,
        runtime_lock_path=runtime_lock.path,
        seal_evidence_path=seal.path,
        conversion_audit_path=audit.path,
        host_adapter_source_manifest_path=(
            repository / "runtime/oaf_tf1/host-adapter-source-manifest.json"
        ),
        model_cache_root=model_cache,
        input_root=input_root,
        native_host_evidence=host_evidence,
        allow_emulated_diagnostics=False,
        strict_checkout=True,
    )
    audio = load_direct_audio(
        smoke_root / "canonical.wav",
        source_audio_id=cast(str, oracle["source_audio_id"]),
        input_view_id=cast(str, oracle["input_view_id"]),
        max_input_audio_frames=backend_lock.max_input_audio_frames,
    )
    events = tuple(_oracle_event(value) for value in cast(list[JsonValue], oracle["native_events"]))
    expected = render_prediction_artifact(
        NativePrediction(
            audio=audio,
            descriptor=backend_lock.descriptor,
            events=events,
            backend_lock_sha256=backend_lock.sha256,
            runtime_lock_sha256=runtime_lock.sha256,
            parameter_lock_sha256=None,
            model_artifact_set_sha256=model_identity,
            upstream_source_commit=cast(
                str,
                backend_lock.payload["upstream_source_commit"],
            ),
            training_data_map_id=cast(
                str,
                backend_lock.payload["training_data_map_id"],
            ),
        )
    )

    shared = OafTf1Backend(config)
    try:
        assert shared.verify().status == "verified"
        first = render_prediction_artifact(shared.transcribe(audio))
        second = render_prediction_artifact(shared.transcribe(audio))
    finally:
        shared.close()

    fresh = OafTf1Backend(config)
    try:
        assert fresh.verify().status == "verified"
        third = render_prediction_artifact(fresh.transcribe(audio))
    finally:
        fresh.close()

    assert first == expected
    assert second == expected
    assert third == expected
