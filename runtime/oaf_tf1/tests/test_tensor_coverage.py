from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.oaf_tf1 import oaf_backend
from runtime.oaf_tf1.oaf_backend import (
    ModelIntegrityFailure,
    TensorCoverage,
    assert_no_reachable_stochastic_ops,
    validate_tensor_coverage,
)
from runtime.oaf_tf1.protocol import (
    AuthenticatedObject,
    ProtocolFailure,
    VerifiedWav,
    load_authenticated_object,
)


def _entry(index: int) -> dict[str, object]:
    return {
        "dtype": "float32",
        "name": f"tensor_{index:03d}",
        "shape": [index + 1],
    }


def _inventories():
    checkpoint = [_entry(index) for index in range(130)]
    required = checkpoint[:78]
    non_inference = [
        {**entry, "reason": f"locked non-inference state {index:03d}"}
        for index, entry in enumerate(checkpoint[78:])
    ]
    graph = required
    return checkpoint, required, non_inference, graph


def test_tensor_coverage_accepts_exact_130_78_52_partition() -> None:
    checkpoint, required, non_inference, graph = _inventories()

    coverage = validate_tensor_coverage(
        checkpoint_inventory=checkpoint,
        required_inventory=required,
        non_inference_inventory=non_inference,
        graph_inventory=graph,
        uninitialized_required=(),
    )

    assert isinstance(coverage, TensorCoverage)
    assert coverage.checkpoint_count == 130
    assert coverage.required_count == 78
    assert coverage.restored_count == 78
    assert coverage.non_inference_count == 52
    assert len(coverage.required_inventory_sha256) == 64
    assert len(coverage.non_inference_inventory_sha256) == 64


def test_runner_lock_key_sets_match_host_and_reject_legacy_runtime_fields(tmp_path: Path) -> None:
    from src.benchmark import backend_lock

    assert oaf_backend.BACKEND_LOCK_KEYS == backend_lock.BACKEND_LOCK_KEYS
    assert oaf_backend.RUNTIME_LOCK_KEYS == backend_lock.RUNTIME_LOCK_KEYS
    assert oaf_backend.SEAL_EVIDENCE_KEYS == backend_lock.SEAL_EVIDENCE_KEYS

    payload = {key: "fixture" for key in oaf_backend.RUNTIME_LOCK_KEYS}
    payload["schema"] = "crux.transcription-runtime-lock/v1"
    path = tmp_path / "runtime-lock.json"
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    loaded = load_authenticated_object(
        path,
        label="runtime_lock",
        exact_keys=oaf_backend.RUNTIME_LOCK_KEYS,
        expected_schema="crux.transcription-runtime-lock/v1",
    )
    assert loaded.payload["schema"] == "crux.transcription-runtime-lock/v1"

    legacy = dict(payload)
    legacy["debian_snapshot_repository"] = "legacy"
    path.write_bytes(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    with pytest.raises(ProtocolFailure, match="mounted runner identity"):
        load_authenticated_object(
            path,
            label="runtime_lock",
            exact_keys=oaf_backend.RUNTIME_LOCK_KEYS,
            expected_schema="crux.transcription-runtime-lock/v1",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda values: values[0].pop(), "checkpoint count"),
        (lambda values: values[1].pop(), "required count"),
        (lambda values: values[2].pop(), "non-inference count"),
        (
            lambda values: values[3].__setitem__(0, {**values[3][0], "shape": [999]}),
            "graph inventory",
        ),
        (
            lambda values: values[0].__setitem__(0, {**values[0][0], "dtype": "float64"}),
            "checkpoint inventory",
        ),
    ],
)
def test_tensor_coverage_rejects_count_shape_dtype_and_partition_drift(mutation, match) -> None:
    values = [list(value) for value in _inventories()]
    mutation(values)

    with pytest.raises(ModelIntegrityFailure, match=match):
        validate_tensor_coverage(
            checkpoint_inventory=values[0],
            required_inventory=values[1],
            non_inference_inventory=values[2],
            graph_inventory=values[3],
            uninitialized_required=(),
        )


def test_tensor_coverage_rejects_uninitialized_required_variable() -> None:
    checkpoint, required, non_inference, graph = _inventories()

    with pytest.raises(ModelIntegrityFailure, match="uninitialized"):
        validate_tensor_coverage(
            checkpoint_inventory=checkpoint,
            required_inventory=required,
            non_inference_inventory=non_inference,
            graph_inventory=graph,
            uninitialized_required=("tensor_003",),
        )


def test_ready_carries_authenticated_smoke_prediction_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor_fields = {
        "architecture_id": "magenta-oaf-model-tpu-drums-v1",
        "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
        "descriptor_schema": "crux.transcription-backend-descriptor/v1",
        "model_id": "magenta-egmd-ckpt-569400-v1",
        "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
        "native_output_space_id": "magenta-oaf-midi88-a0-v1",
        "prediction_schema": "crux.drum-prediction-events/v1",
        "protocol_schema": "crux.transcription-runner/v1",
        "runtime_image_manifest_digest": f"sha256:{'4' * 64}",
        "runtime_lock_sha256": "b" * 64,
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
        "upstream_source_commit": "9" * 40,
    }
    objects = {
        "backend_lock": AuthenticatedObject(
            payload={
                **descriptor_fields,
                "checkpoint_components": [],
                "checkpoint_inventory": [],
                "max_input_audio_frames": 44100,
                "non_inference_inventory": [],
                "required_inference_inventory": [],
                "seal_evidence_sha256": "c" * 64,
                "smoke_audio_sha256": "5" * 64,
                "smoke_oracle_sha256": "6" * 64,
                "upstream_source_manifest_sha256": "7" * 64,
            },
            sha256="a" * 64,
        ),
        "runtime_lock": AuthenticatedObject(
            payload={
                "runner_source_manifest_sha256": "8" * 64,
                "runtime_image_manifest_digest": f"sha256:{'4' * 64}",
                "stdout_max_line_bytes": 4096,
                "tensorflow_abi": "cp37-cp37m-manylinux2010_x86_64",
                "tensorflow_build": "v1.15.5",
                "upstream_source_manifest_sha256": "7" * 64,
            },
            sha256="b" * 64,
        ),
        "seal_evidence": AuthenticatedObject(
            payload={
                "checkpoint_components": [],
                "checkpoint_inventory": [],
                "non_inference_inventory": [],
                "required_inference_inventory": [],
                "runner_source_manifest_sha256": "8" * 64,
                "smoke_audio_sha256": "5" * 64,
                "smoke_oracle_sha256": "6" * 64,
                "smoke_prediction_sha256": "d" * 64,
            },
            sha256="c" * 64,
        ),
        "runner_source_manifest": AuthenticatedObject(payload={}, sha256="8" * 64),
        "upstream_source_manifest": AuthenticatedObject(payload={}, sha256="7" * 64),
        "smoke_oracle": AuthenticatedObject(
            payload={"native_events": [{"native_class_id": "midi_36"}]},
            sha256="6" * 64,
        ),
    }
    monkeypatch.setattr(
        oaf_backend,
        "load_authenticated_object",
        lambda _path, *, label, **_kwargs: objects[label],
    )
    monkeypatch.setattr(
        oaf_backend,
        "build_and_restore_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            coverage=TensorCoverage(
                checkpoint_count=130,
                required_count=78,
                restored_count=78,
                non_inference_count=52,
                checkpoint_inventory_sha256="1" * 64,
                required_inventory_sha256="2" * 64,
                non_inference_inventory_sha256="3" * 64,
            )
        ),
    )
    monkeypatch.setattr(
        oaf_backend,
        "read_verified_canonical_wav",
        lambda *_args, **_kwargs: VerifiedWav(
            relative_path="smoke/canonical.wav",
            content=b"smoke",
            sha256="5" * 64,
            audio_frame_count=44100,
        ),
    )
    monkeypatch.setattr(
        oaf_backend,
        "transcribe_canonical_wav",
        lambda *_args, **_kwargs: [{"native_class_id": "midi_36"}],
    )

    startup = oaf_backend.authenticate_startup()

    assert startup.ready_payload["smoke_prediction_sha256"] == "d" * 64


class _Operation:
    def __init__(self, name: str, operation_type: str, inputs=(), controls=()) -> None:
        self.name = name
        self.type = operation_type
        self.inputs = tuple(_Tensor(operation) for operation in inputs)
        self.control_inputs = tuple(controls)


class _Tensor:
    def __init__(self, operation: _Operation) -> None:
        self.op = operation


def test_predict_fetch_reachability_rejects_active_randomness() -> None:
    random = _Operation("dropout/random_uniform", "RandomUniform")
    multiply = _Operation("dropout/mul", "Mul", inputs=(random,))
    fetch = _Operation("predictions", "Identity", inputs=(multiply,))

    with pytest.raises(ModelIntegrityFailure, match="stochastic"):
        assert_no_reachable_stochastic_ops((fetch,))


def test_predict_fetch_reachability_rejects_active_stateless_randomness() -> None:
    random = _Operation("dropout/stateless_random_uniform", "StatelessRandomUniform")
    fetch = _Operation("predictions", "Identity", inputs=(random,))

    with pytest.raises(ModelIntegrityFailure, match="stochastic"):
        assert_no_reachable_stochastic_ops((fetch,))


def test_disconnected_randomness_does_not_fail_predict_reachability() -> None:
    deterministic = _Operation("logits", "MatMul")
    fetch = _Operation("predictions", "Identity", inputs=(deterministic,))
    _Operation("training/random_uniform", "RandomUniform")

    assert_no_reachable_stochastic_ops((fetch,))


def test_actual_train_util_estimator_uses_explicit_prediction_session_config(
    tmp_path: Path,
) -> None:
    tf = pytest.importorskip("tensorflow.compat.v1")
    patched_vendor = Path("/opt/crux/vendor")
    if not patched_vendor.is_dir():
        pytest.skip("TensorFlow 1 estimator integration runs in the runtime image")
    train_util_path = (
        patched_vendor / "magenta" / "models" / "onsets_frames_transcription" / "train_util.py"
    )
    spec = importlib.util.spec_from_file_location("_crux_test_train_util", str(train_util_path))
    assert spec is not None and spec.loader is not None
    train_util = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_util)
    from tensorflow.contrib import training as contrib_training

    hparams = contrib_training.HParams(
        batch_size=1,
        eval_batch_size=1,
        predict_batch_size=1,
    )

    def model_fn(features, labels, mode, params, config):
        del features, labels, params, config
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"value": tf.constant([1])},
        )

    estimator = train_util.create_estimator(
        model_fn,
        str(tmp_path / "model"),
        hparams,
    )

    configured = oaf_backend.configure_prediction_estimator_session(estimator, tf)

    assert configured is estimator
    assert configured.config.session_config.inter_op_parallelism_threads == 1
    assert configured.config.session_config.intra_op_parallelism_threads == 1
    assert configured._session_config == configured.config.session_config
    assert configured._session_config is configured._config._session_config
    assert configured._session_config.inter_op_parallelism_threads == 1
    assert configured._session_config.intra_op_parallelism_threads == 1
