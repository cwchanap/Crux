from __future__ import annotations

import hashlib
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


def _final_stage_copy_mappings(repository: Path) -> list[tuple[str, str]]:
    dockerfile = (repository / "runtime/oaf_tf1/Dockerfile").read_text()
    marker = "FROM runtime-build AS runtime\n"
    assert marker in dockerfile
    final_stage = dockerfile.split(marker, maxsplit=1)[1]
    mappings: list[tuple[str, str]] = []
    instruction = ""
    for line in final_stage.splitlines():
        stripped = line.strip()
        instruction += stripped[:-1] if stripped.endswith("\\") else stripped
        if line.rstrip().endswith("\\"):
            instruction += " "
            continue
        fields = instruction.split()
        instruction = ""
        if not fields or fields[0] != "COPY" or fields[1].startswith("--from="):
            continue
        assert len(fields) == 3
        mappings.append((fields[1], fields[2]))
    assert not instruction
    return mappings


def _final_stage_destinations(mappings: list[tuple[str, str]], source_path: str) -> set[str]:
    destinations = set()
    for copied_source, destination in mappings:
        if source_path == copied_source:
            destinations.add(
                destination + source_path.rsplit("/", maxsplit=1)[-1]
                if destination.endswith("/")
                else destination
            )
        if copied_source.endswith("/") and source_path.startswith(copied_source):
            destinations.add(destination + source_path[len(copied_source) :])
    return destinations


def _stage_final_image_file(
    *,
    repository: Path,
    staged_root: Path,
    mappings: list[tuple[str, str]],
    source_path: str,
    destination: str | None = None,
) -> Path:
    final_destination = destination or f"/opt/crux/{source_path}"
    assert final_destination in _final_stage_destinations(mappings, source_path)
    prefix = "/opt/crux/"
    assert final_destination.startswith(prefix)
    staged = staged_root / final_destination[len(prefix) :]
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes((repository / source_path).read_bytes())
    return staged


def test_mounted_source_manifest_accepts_roots_top_level_and_package_ancestors(
    tmp_path: Path,
) -> None:
    (tmp_path / "magenta" / "music").mkdir(parents=True)
    (tmp_path / "magenta" / "common").mkdir()
    (tmp_path / "magenta" / "models" / "deep").mkdir(parents=True)
    (tmp_path / "LICENSE").write_bytes(b"license\n")
    source = tmp_path / "magenta" / "music" / "source.py"
    source.write_bytes(b"source\n")
    package = tmp_path / "magenta" / "__init__.py"
    package.write_bytes(b"package\n")
    models_package = tmp_path / "magenta" / "models" / "__init__.py"
    models_package.write_bytes(b"models package\n")
    oaf_backend._validate_mounted_source_manifest(
        {
            "covered_roots": ["magenta/models/deep", "magenta/music", "magenta/common"],
            "files": [
                {"path": "LICENSE", "sha256": hashlib.sha256(b"license\n").hexdigest()},
                {
                    "path": "magenta/__init__.py",
                    "sha256": hashlib.sha256(b"package\n").hexdigest(),
                },
                {
                    "path": "magenta/models/__init__.py",
                    "sha256": hashlib.sha256(b"models package\n").hexdigest(),
                },
                {
                    "path": "magenta/music/source.py",
                    "sha256": hashlib.sha256(b"source\n").hexdigest(),
                },
            ],
        },
        tmp_path,
    )


def test_mounted_source_manifest_rejects_out_of_contract_sibling(tmp_path: Path) -> None:
    (tmp_path / "magenta" / "models" / "deep").mkdir(parents=True)
    sibling = tmp_path / "magenta" / "unrelated" / "source.py"
    sibling.parent.mkdir()
    sibling.write_bytes(b"sibling\n")

    with pytest.raises(ProtocolFailure, match="Mounted source row is invalid"):
        oaf_backend._validate_mounted_source_manifest(
            {
                "covered_roots": ["magenta/models/deep"],
                "files": [
                    {
                        "path": "magenta/unrelated/source.py",
                        "sha256": hashlib.sha256(b"sibling\n").hexdigest(),
                    }
                ],
            },
            tmp_path,
        )


def test_mounted_source_manifest_accepts_checked_in_upstream_vendor_tree() -> None:
    repository = Path(__file__).parents[3]
    manifest = json.loads((repository / "runtime/oaf_tf1/source-manifest.json").read_text())

    oaf_backend._validate_mounted_source_manifest(
        manifest,
        repository / "runtime/oaf_tf1/vendor",
    )


@pytest.mark.parametrize("mutation", ["empty", "missing", "hash", "outside"])
def test_mounted_source_manifest_fails_closed(tmp_path: Path, mutation: str) -> None:
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "source.py").write_bytes(b"source\n")
    payload = {
        "covered_roots": ["root"],
        "files": [{"path": "root/source.py", "sha256": hashlib.sha256(b"source\n").hexdigest()}],
    }
    if mutation == "empty":
        payload["files"] = []
    if mutation == "missing":
        payload["files"][0]["path"] = "root/missing.py"
    if mutation == "hash":
        payload["files"][0]["sha256"] = "0" * 64
    if mutation == "outside":
        payload["files"][0]["path"] = "other/source.py"
    with pytest.raises(ProtocolFailure):
        oaf_backend._validate_mounted_source_manifest(payload, tmp_path)


def test_mounted_source_manifest_requires_each_declared_root(tmp_path: Path) -> None:
    (tmp_path / "present").mkdir()
    source = tmp_path / "present" / "source.py"
    source.write_bytes(b"source\n")
    payload = {
        "covered_roots": ["missing", "present"],
        "files": [{"path": "present/source.py", "sha256": hashlib.sha256(b"source\n").hexdigest()}],
    }
    with pytest.raises(ProtocolFailure, match="root is missing"):
        oaf_backend._validate_mounted_source_manifest(payload, tmp_path)


def test_final_image_preserves_every_runner_manifest_path() -> None:
    repository = Path(__file__).parents[3]
    manifest = json.loads((repository / "runtime/oaf_tf1/runner-source-manifest.json").read_text())
    mappings = _final_stage_copy_mappings(repository)
    for row in manifest["files"]:
        source_path = row["path"]
        assert (repository / source_path).is_file()
        assert f"/opt/crux/{source_path}" in _final_stage_destinations(mappings, source_path)


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
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[3]
    staged_root = tmp_path / "crux"
    mappings = _final_stage_copy_mappings(repository)
    runner_manifest_content = (
        repository / "runtime/oaf_tf1/runner-source-manifest.json"
    ).read_bytes()
    runner_manifest_payload = json.loads(runner_manifest_content)
    runner_manifest = _stage_final_image_file(
        repository=repository,
        staged_root=staged_root,
        mappings=mappings,
        source_path="runtime/oaf_tf1/runner-source-manifest.json",
        destination="/opt/crux/runtime/runner-source-manifest.json",
    )
    for row in runner_manifest_payload["files"]:
        staged = _stage_final_image_file(
            repository=repository,
            staged_root=staged_root,
            mappings=mappings,
            source_path=row["path"],
        )
        assert staged == staged_root / row["path"]
    upstream_manifest_content = (repository / "runtime/oaf_tf1/source-manifest.json").read_bytes()
    upstream_manifest_payload = json.loads(upstream_manifest_content)
    upstream_manifest = _stage_final_image_file(
        repository=repository,
        staged_root=staged_root,
        mappings=mappings,
        source_path="runtime/oaf_tf1/source-manifest.json",
        destination="/opt/crux/vendor/source-manifest.json",
    )
    for row in upstream_manifest_payload["files"]:
        source = repository / "runtime/oaf_tf1/vendor" / row["path"]
        destination = staged_root / "upstream" / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    runner_manifest_sha256 = hashlib.sha256(runner_manifest_content).hexdigest()
    upstream_manifest_sha256 = hashlib.sha256(upstream_manifest_content).hexdigest()
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
                "upstream_source_manifest_sha256": upstream_manifest_sha256,
            },
            sha256="a" * 64,
        ),
        "runtime_lock": AuthenticatedObject(
            payload={
                "environment": {
                    "CUDA_VISIBLE_DEVICES": "-1",
                    "MKL_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "PYTHONHASHSEED": "0",
                    "TF_NUM_INTEROP_THREADS": "1",
                    "TF_NUM_INTRAOP_THREADS": "1",
                },
                "runner_source_manifest_sha256": runner_manifest_sha256,
                "runtime_image_manifest_digest": f"sha256:{'4' * 64}",
                "seal_evidence_sha256": "c" * 64,
                "stdout_max_line_bytes": 4096,
                "tensorflow_abi": "cp37-cp37m-manylinux2010_x86_64",
                "tensorflow_build": "v1.15.5",
                "upstream_source_manifest_sha256": upstream_manifest_sha256,
            },
            sha256="b" * 64,
        ),
        "seal_evidence": AuthenticatedObject(
            payload={
                "checkpoint_components": [],
                "checkpoint_inventory": [],
                "max_input_audio_frames": 44100,
                "non_inference_inventory": [],
                "required_inference_inventory": [],
                "runner_source_manifest_sha256": runner_manifest_sha256,
                "runtime_image_manifest_digest": f"sha256:{'4' * 64}",
                "smoke_audio_sha256": "5" * 64,
                "smoke_oracle_sha256": "6" * 64,
                "smoke_prediction_sha256": "d" * 64,
                "stdout_max_line_bytes": 4096,
                "tensorflow_abi": "cp37-cp37m-manylinux2010_x86_64",
                "tensorflow_build": "v1.15.5",
                "upstream_source_manifest_sha256": upstream_manifest_sha256,
            },
            sha256="c" * 64,
        ),
        "smoke_oracle": AuthenticatedObject(
            payload={"native_events": [{"native_class_id": "midi_36"}]},
            sha256="6" * 64,
        ),
    }
    load_authenticated_object = oaf_backend.load_authenticated_object

    def load_fixture_or_source_manifest(
        path: Path, *, label: str, **kwargs: object
    ) -> AuthenticatedObject:
        if label in {"runner_source_manifest", "upstream_source_manifest"}:
            return load_authenticated_object(path, label=label, **kwargs)
        return objects[label]

    monkeypatch.setattr(oaf_backend, "load_authenticated_object", load_fixture_or_source_manifest)
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
    startup = oaf_backend.authenticate_startup(
        runner_source_manifest_path=runner_manifest,
        upstream_source_manifest_path=upstream_manifest,
    )

    assert startup.ready_payload["smoke_prediction_sha256"] == "d" * 64
    assert startup.ready_payload["runner_source_manifest_sha256"] == runner_manifest_sha256
    assert startup.ready_payload["upstream_source_manifest_sha256"] == upstream_manifest_sha256


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
