# The task plan keeps the complete audit contract and its regressions in one module.
# pylint: disable=too-many-lines

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.benchmark.backend_lock as backend_lock_module
import tools.hpa320.audit_legacy_tf2_conversion as audit_module
from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.backend_lock import load_conversion_audit
from tools.hpa320.audit_legacy_tf2_conversion import (
    CANDIDATE_MANIFEST_NAME,
    CANDIDATE_MANIFEST_SCHEMA,
    AuditError,
    ConversionCoverage,
    ConverterLayerSpec,
    ResolvedCheckpoint,
    VariableSpec,
    audit_conversion_coverage,
    build_evidence,
    converter_layer_specs,
    enumerate_checkpoint_variables,
    enumerate_hdf5_weights,
    load_required_inventory,
    main,
    resolve_candidate_checkpoint,
    resolve_explicit_checkpoint,
    stage_verified_inputs,
    write_evidence_atomic,
)


def _write_canonical(path: Path, value: Any) -> Path:
    path.write_bytes(canonical_json_bytes(value, trailing_newline=True))
    return path


def _variable(
    name: str,
    shape: tuple[int, ...] = (2, 3),
    dtype: str = "float32",
) -> VariableSpec:
    return VariableSpec(name=name, shape=shape, dtype=dtype)


def _layer(
    name: str,
    kind: str,
    kernel_shape: tuple[int, ...] = (2, 3),
    dtype: str = "float32",
) -> ConverterLayerSpec:
    return ConverterLayerSpec(
        name=name,
        kind=kind,
        kernel=_variable(f"{name}/kernel", kernel_shape, dtype),
        bias=_variable(f"{name}/bias", (kernel_shape[-1],), dtype),
    )


# The six fields are the complete Task 1 candidate-relation schema.
# pylint: disable-next=too-many-arguments
def _candidate_match(
    *,
    required_name: str,
    candidate_name: str,
    match_kind: str,
    dtype_compatible: bool,
    shape_compatible: bool,
    assigned: bool,
) -> dict[str, Any]:
    return {
        "assigned": assigned,
        "candidate_name": candidate_name,
        "dtype_compatible": dtype_compatible,
        "match_kind": match_kind,
        "required_name": required_name,
        "shape_compatible": shape_compatible,
    }


def _checkpoint_components(cache_directory: Path) -> list[dict[str, Any]]:
    contents = {
        "model.ckpt-569400.data-00000-of-00001": b"data",
        "model.ckpt-569400.index": b"index",
        "model.ckpt-569400.meta": b"meta",
    }
    components: list[dict[str, Any]] = []
    for name, content in sorted(contents.items()):
        (cache_directory / name).write_bytes(content)
        components.append(
            {
                "name": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        )
    return components


def _candidate_fixture(
    tmp_path: Path,
    *,
    required_inventory_sha256: str = "a" * 64,
) -> tuple[Path, Path, dict[str, Any]]:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    cache_root = tmp_path / "model-cache"
    cache_root.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    components = _checkpoint_components(staging)
    artifact_sha256 = hashlib.sha256(canonical_json_bytes(components)).hexdigest()
    cache_directory = cache_root / "sha256" / artifact_sha256
    cache_directory.mkdir(parents=True)
    for component in components:
        source = staging / component["name"]
        source.replace(cache_directory / component["name"])
    payload = {
        "checkpoint_components": components,
        "checkpoint_prefix": f"sha256/{artifact_sha256}/model.ckpt-569400",
        "model_artifact_set_sha256": artifact_sha256,
        "required_inference_inventory_sha256": required_inventory_sha256,
        "schema": CANDIDATE_MANIFEST_SCHEMA,
    }
    _write_canonical(candidate / CANDIDATE_MANIFEST_NAME, payload)
    return candidate, cache_root, payload


def test_audit_reports_zero_restored_required_tensors() -> None:
    required = _variable("onsets/conv0/weights", (3, 3, 1, 16))

    result = audit_conversion_coverage(
        checkpoint_variables=(required,),
        required_variables=(required,),
        converter_layers=(_layer("conv_0", "Conv2D", (3, 3, 229, 32)),),
    )

    assert not result.restored_required
    assert result.restored_required_count == 0
    assert result.unmatched_required == ("onsets/conv0/weights",)


def test_loose_conv_candidate_is_not_an_assignment_when_shape_is_incompatible() -> None:
    required = _variable("onsets/conv0/weights", (3, 3, 1, 16))

    result = audit_conversion_coverage(
        checkpoint_variables=(required,),
        required_variables=(required,),
        converter_layers=(_layer("conv_0", "Conv2D", (3, 3, 229, 32)),),
    )

    assert result.candidate_matches == (
        _candidate_match(
            required_name=required.name,
            candidate_name="conv_0/kernel",
            match_kind="loose_substring",
            dtype_compatible=True,
            shape_compatible=False,
            assigned=False,
        ),
    )


def test_dense_transpose_assignment_is_recorded_separately() -> None:
    required = _variable("onsets/onset_probs/weights", (2, 3))

    result = audit_conversion_coverage(
        checkpoint_variables=(required,),
        required_variables=(required,),
        converter_layers=(_layer("onset_probs", "Dense", (3, 2)),),
    )

    assert result.restored_required == (required.name,)
    assert result.candidate_matches == (
        _candidate_match(
            required_name=required.name,
            candidate_name="onset_probs/kernel",
            match_kind="dense_transpose",
            dtype_compatible=True,
            shape_compatible=True,
            assigned=True,
        ),
    )


def test_dense_direct_assignment_uses_exact_name_category() -> None:
    required = _variable("onsets/onset_probs/weights", (2, 3))

    result = audit_conversion_coverage(
        checkpoint_variables=(required,),
        required_variables=(required,),
        converter_layers=(_layer("onset_probs", "Dense", (2, 3)),),
    )

    assert result.candidate_matches[0]["match_kind"] == "exact_name"
    assert result.candidate_matches[0]["assigned"] is True


def test_dense_name_matching_preserves_legacy_case_sensitivity() -> None:
    required = _variable("Onsets/Onset_Probs/weights", (2, 3))

    result = audit_conversion_coverage(
        checkpoint_variables=(required,),
        required_variables=(required,),
        converter_layers=(_layer("onset_probs", "Dense"),),
    )

    assert not result.candidate_matches
    assert result.unmatched_required == (required.name,)


def test_only_actual_top_level_conv2d_and_dense_layers_are_eligible() -> None:
    class FakeWeight:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self.dtype = "float32"

    class FakeConv2D:
        name = "nested_conv"
        kernel = FakeWeight((3, 3, 1, 16))
        bias = FakeWeight((16,))

    class FakeDense:
        name = "onset_probs"
        kernel = FakeWeight((2, 3))
        bias = FakeWeight((3,))

    nested_submodel = SimpleNamespace(layers=[FakeConv2D()])
    noneligible_top_level = SimpleNamespace(
        name="looks_dense",
        kernel=FakeWeight((2, 3)),
        bias=FakeWeight((3,)),
    )
    model = SimpleNamespace(layers=[nested_submodel, FakeDense(), noneligible_top_level])

    layers = converter_layer_specs(
        model,
        conv2d_type=FakeConv2D,
        dense_type=FakeDense,
    )

    assert layers == (_layer("onset_probs", "Dense"),)


def test_arbitrary_nested_hdf5_dataset_never_becomes_a_candidate() -> None:
    required = _variable("onsets/conv0/weights", (3, 3, 1, 16))

    result = audit_conversion_coverage(
        checkpoint_variables=(required,),
        required_variables=(required,),
        converter_layers=(),
    )

    assert not result.candidate_matches
    assert result.unmatched_required == (required.name,)


@pytest.mark.parametrize(
    ("keras_shape", "keras_dtype", "shape_compatible", "dtype_compatible"),
    [
        ((7, 11), "float32", False, True),
        ((2, 3), "float64", True, False),
    ],
)
def test_shape_or_dtype_mismatch_cannot_be_reported_as_assigned(
    keras_shape: tuple[int, ...],
    keras_dtype: str,
    shape_compatible: bool,
    dtype_compatible: bool,
) -> None:
    required = _variable("onsets/onset_probs/weights", (2, 3))

    result = audit_conversion_coverage(
        checkpoint_variables=(required,),
        required_variables=(required,),
        converter_layers=(_layer("onset_probs", "Dense", keras_shape, keras_dtype),),
    )

    assert not result.restored_required
    assert result.candidate_matches[0]["assigned"] is False
    assert result.candidate_matches[0]["shape_compatible"] is shape_compatible
    assert result.candidate_matches[0]["dtype_compatible"] is dtype_compatible


def test_results_use_deterministic_utf8_semantic_order() -> None:
    required = (
        _variable("é/onset_probs/weights"),
        _variable("z/frame_probs/weights"),
        _variable("a/conv0/weights"),
    )
    layers = (
        _layer("frame_probs", "Dense"),
        _layer("conv_9", "Conv2D", (9, 9)),
        _layer("onset_probs", "Dense"),
    )

    forward = audit_conversion_coverage(
        checkpoint_variables=required,
        required_variables=required,
        converter_layers=layers,
    )
    reverse = audit_conversion_coverage(
        checkpoint_variables=tuple(reversed(required)),
        required_variables=tuple(reversed(required)),
        converter_layers=tuple(reversed(layers)),
    )

    assert forward == reverse
    assert forward.restored_required == (
        "z/frame_probs/weights",
        "é/onset_probs/weights",
    )
    assert forward.unmatched_required == ("a/conv0/weights",)
    semantic_keys = [
        (
            row["required_name"].encode("utf-8"),
            row["candidate_name"].encode("utf-8"),
            row["match_kind"].encode("utf-8"),
        )
        for row in forward.candidate_matches
    ]
    assert semantic_keys == sorted(semantic_keys)


@pytest.mark.parametrize(
    "field",
    ["checkpoint_variables", "required_variables", "converter_layers"],
)
def test_duplicate_variable_names_are_rejected(field: str) -> None:
    required = _variable("onsets/onset_probs/weights")
    arguments = {
        "checkpoint_variables": (required,),
        "required_variables": (required,),
        "converter_layers": (_layer("onset_probs", "Dense"),),
    }
    arguments[field] = (arguments[field][0], arguments[field][0])

    with pytest.raises(AuditError, match="unique"):
        audit_conversion_coverage(**arguments)


def test_required_inventory_must_match_checkpoint_shape_and_dtype() -> None:
    checkpoint = _variable("onsets/onset_probs/weights", (2, 3), "float32")

    with pytest.raises(AuditError, match="checkpoint"):
        audit_conversion_coverage(
            checkpoint_variables=(checkpoint,),
            required_variables=(_variable("onsets/onset_probs/weights", (3, 2), "float32"),),
            converter_layers=(),
        )


@pytest.mark.parametrize(
    "match",
    [
        _candidate_match(
            required_name="tensor",
            candidate_name="layer/kernel",
            match_kind="exact_name",
            dtype_compatible=False,
            shape_compatible=True,
            assigned=True,
        ),
        _candidate_match(
            required_name="tensor",
            candidate_name="layer/kernel",
            match_kind="exact_name",
            dtype_compatible=True,
            shape_compatible=True,
            assigned=False,
        ),
    ],
)
def test_conversion_coverage_rejects_dishonest_assigned_flags(
    match: dict[str, Any],
) -> None:
    with pytest.raises(AuditError, match="assigned"):
        ConversionCoverage(
            restored_required=("tensor",),
            unmatched_required=(),
            candidate_matches=(match,),
        )


def test_scalar_shapes_are_valid_and_records_are_immutable() -> None:
    scalar = _variable("global_step", ())
    result = audit_conversion_coverage(
        checkpoint_variables=(scalar,),
        required_variables=(scalar,),
        converter_layers=(),
    )

    assert result.unmatched_required == ("global_step",)
    with pytest.raises(FrozenInstanceError):
        scalar.dtype = "float64"  # type: ignore[misc]


@pytest.mark.parametrize("shape", [(-1,), (True,)])
def test_invalid_shape_dimensions_are_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(AuditError, match="shape"):
        VariableSpec("invalid", shape, "float32")


def test_candidate_matches_are_deeply_immutable() -> None:
    required = _variable("onsets/onset_probs/weights")
    result = audit_conversion_coverage(
        checkpoint_variables=(required,),
        required_variables=(required,),
        converter_layers=(_layer("onset_probs", "Dense"),),
    )

    with pytest.raises(TypeError):
        result.candidate_matches[0]["assigned"] = False  # type: ignore[index]


def test_required_inventory_strict_loads_canonical_rows_and_scalar_shapes(
    tmp_path: Path,
) -> None:
    inventory = [
        {"dtype": "int64", "name": "global_step", "shape": []},
        {"dtype": "float32", "name": "tensor", "shape": [2, 3]},
    ]
    path = _write_canonical(tmp_path / "required.json", inventory)

    loaded, digest = load_required_inventory(path)

    assert loaded == (
        VariableSpec("global_step", (), "int64"),
        VariableSpec("tensor", (2, 3), "float32"),
    )
    assert digest == hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()


@pytest.mark.parametrize(
    "content",
    [
        b'[{"dtype": "float32", "name": "tensor", "shape": [2]}]\n',
        b'[{"dtype":"float32","extra":true,"name":"tensor","shape":[2]}]\n',
        b'[{"dtype":"float32","name":"tensor","shape":[2]},'
        b'{"dtype":"float32","name":"tensor","shape":[2]}]\n',
    ],
)
def test_required_inventory_rejects_malformed_or_duplicate_rows(
    tmp_path: Path,
    content: bytes,
) -> None:
    path = tmp_path / "required.json"
    path.write_bytes(content)

    with pytest.raises(AuditError):
        load_required_inventory(path)


def test_candidate_manifest_resolves_verified_content_addressed_prefix(
    tmp_path: Path,
) -> None:
    candidate, cache_root, payload = _candidate_fixture(tmp_path)

    resolved = resolve_candidate_checkpoint(
        candidate,
        cache_root,
        expected_required_inventory_sha256="a" * 64,
    )

    assert resolved.prefix == (
        cache_root / "sha256" / payload["model_artifact_set_sha256"] / "model.ckpt-569400"
    )
    assert resolved.model_artifact_set_sha256 == payload["model_artifact_set_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"unknown": True}), "fields"),
        (
            lambda payload: payload.update({"checkpoint_prefix": "../outside/model.ckpt-569400"}),
            "checkpoint prefix",
        ),
        (
            lambda payload: payload.update(
                {
                    "checkpoint_prefix": (
                        f"sha256/{payload['model_artifact_set_sha256']}//model.ckpt-569400"
                    )
                }
            ),
            "checkpoint prefix",
        ),
        (
            lambda payload: payload.update({"model_artifact_set_sha256": "f" * 64}),
            "artifact",
        ),
        (
            lambda payload: payload.update({"required_inference_inventory_sha256": "f" * 64}),
            "required",
        ),
    ],
)
def test_candidate_manifest_rejects_malformed_identity_or_traversal(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    candidate, cache_root, payload = _candidate_fixture(tmp_path)
    mutation(payload)
    _write_canonical(candidate / CANDIDATE_MANIFEST_NAME, payload)

    with pytest.raises(AuditError, match=message):
        resolve_candidate_checkpoint(
            candidate,
            cache_root,
            expected_required_inventory_sha256="a" * 64,
        )


def test_candidate_and_cache_symlink_boundaries_are_rejected(tmp_path: Path) -> None:
    candidate, cache_root, payload = _candidate_fixture(tmp_path)
    candidate_link = tmp_path / "candidate-link"
    candidate_link.symlink_to(candidate, target_is_directory=True)

    with pytest.raises(AuditError, match="symlink"):
        resolve_candidate_checkpoint(
            candidate_link,
            cache_root,
            expected_required_inventory_sha256="a" * 64,
        )

    component = (
        cache_root
        / "sha256"
        / payload["model_artifact_set_sha256"]
        / payload["checkpoint_components"][0]["name"]
    )
    replacement = tmp_path / "replacement"
    replacement.write_bytes(component.read_bytes())
    component.unlink()
    component.symlink_to(replacement)

    with pytest.raises(AuditError, match="regular file"):
        resolve_candidate_checkpoint(
            candidate,
            cache_root,
            expected_required_inventory_sha256="a" * 64,
        )


def test_candidate_cache_hash_disagreement_is_rejected(tmp_path: Path) -> None:
    candidate, cache_root, payload = _candidate_fixture(tmp_path)
    component = (
        cache_root
        / "sha256"
        / payload["model_artifact_set_sha256"]
        / payload["checkpoint_components"][0]["name"]
    )
    component.write_bytes(b"changed")

    with pytest.raises(AuditError, match="component"):
        resolve_candidate_checkpoint(
            candidate,
            cache_root,
            expected_required_inventory_sha256="a" * 64,
        )


def test_candidate_cache_rejects_unlisted_entries(tmp_path: Path) -> None:
    candidate, cache_root, payload = _candidate_fixture(tmp_path)
    cache_directory = cache_root / "sha256" / payload["model_artifact_set_sha256"]
    (cache_directory / "unexpected").write_bytes(b"extra")

    with pytest.raises(AuditError, match="exactly"):
        resolve_candidate_checkpoint(
            candidate,
            cache_root,
            expected_required_inventory_sha256="a" * 64,
        )


def test_staging_binds_hdf5_hash_to_the_copy_later_enumerated(
    tmp_path: Path,
) -> None:
    checkpoint_directory = tmp_path / "checkpoint"
    checkpoint_directory.mkdir()
    _checkpoint_components(checkpoint_directory)
    resolved = resolve_explicit_checkpoint(checkpoint_directory / "model.ckpt-569400")
    hdf5 = tmp_path / "weights.h5"
    hdf5.write_bytes(b"reviewed-hdf5")
    converter_source = tmp_path / "convert.py"
    converter_source.write_bytes(b"converter")
    model_source = tmp_path / "model.py"
    model_source.write_bytes(b"model")

    with stage_verified_inputs(
        resolved=resolved,
        hdf5=hdf5,
        converter_source=converter_source,
        tf2_model_source=model_source,
        staging_parent=tmp_path,
    ) as staged:
        replacement = tmp_path / "replacement.h5"
        replacement.write_bytes(b"replacement")
        replacement.replace(hdf5)

        assert staged.hdf5.read_bytes() == b"reviewed-hdf5"
        assert staged.observed_hdf5_sha256 == hashlib.sha256(b"reviewed-hdf5").hexdigest()
        staged.verify()


def test_staging_rejects_checkpoint_mutated_after_resolution(
    tmp_path: Path,
) -> None:
    checkpoint_directory = tmp_path / "checkpoint"
    checkpoint_directory.mkdir()
    _checkpoint_components(checkpoint_directory)
    resolved = resolve_explicit_checkpoint(checkpoint_directory / "model.ckpt-569400")
    (checkpoint_directory / "model.ckpt-569400.index").write_bytes(b"changed")
    hdf5 = tmp_path / "weights.h5"
    hdf5.write_bytes(b"hdf5")
    converter_source = tmp_path / "convert.py"
    converter_source.write_bytes(b"converter")
    model_source = tmp_path / "model.py"
    model_source.write_bytes(b"model")

    with pytest.raises(AuditError, match="checkpoint component"):
        with stage_verified_inputs(
            resolved=resolved,
            hdf5=hdf5,
            converter_source=converter_source,
            tf2_model_source=model_source,
            staging_parent=tmp_path,
        ):
            pass


def test_checkpoint_enumeration_uses_lazy_reader_shape_and_dtype_maps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeDType:
        name = "float32"

    class FakeReader:
        @staticmethod
        def get_variable_to_shape_map() -> dict[str, list[int]]:
            return {"scalar": [], "weights": [2, 3]}

        @staticmethod
        def get_variable_to_dtype_map() -> dict[str, FakeDType]:
            return {"scalar": FakeDType(), "weights": FakeDType()}

    fake_tensorflow = SimpleNamespace(train=SimpleNamespace(load_checkpoint=lambda _: FakeReader()))
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)

    assert enumerate_checkpoint_variables(tmp_path / "model.ckpt") == (
        VariableSpec("scalar", (), "float32"),
        VariableSpec("weights", (2, 3), "float32"),
    )


def test_hdf5_enumeration_uses_lazy_dataset_walk_and_keras_variable_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeDataset:
        def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
            self.shape = shape
            self.dtype = dtype

    class FakeFile:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        @staticmethod
        def visititems(visitor: Any) -> None:
            visitor("layers/conv_0/vars/0", FakeDataset((3, 3, 1, 16), "float32"))
            visitor("layers/conv_0/vars/1", FakeDataset((16,), "float32"))

    fake_h5py = SimpleNamespace(
        Dataset=FakeDataset,
        File=lambda *_args, **_kwargs: FakeFile(),
    )
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)

    assert enumerate_hdf5_weights(tmp_path / "weights.h5") == (
        VariableSpec("layers/conv_0/bias", (16,), "float32"),
        VariableSpec("layers/conv_0/kernel", (3, 3, 1, 16), "float32"),
    )


def test_tool_import_does_not_load_tensorflow_or_h5py() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import tools.hpa320.audit_legacy_tf2_conversion; "
            "assert 'tensorflow' not in sys.modules; "
            "assert 'h5py' not in sys.modules"
        ),
    ]

    completed = subprocess.run(
        command,
        check=False,
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_build_evidence_has_exact_schema_and_does_not_mutate_coverage() -> None:
    coverage = ConversionCoverage(
        restored_required=(),
        unmatched_required=("tensor",),
        candidate_matches=(),
    )

    evidence = build_evidence(
        coverage=coverage,
        converter_source_manifest_sha256="1" * 64,
        model_artifact_set_sha256="2" * 64,
        observed_hdf5_sha256="3" * 64,
        required_inference_inventory_sha256="4" * 64,
        tf2_model_source_manifest_sha256="5" * 64,
    )

    assert evidence == {
        "candidate_matches": [],
        "converter_source_manifest_sha256": "1" * 64,
        "matching_algorithm": "exact_assignment_trace",
        "matching_algorithm_version": "v1",
        "model_artifact_set_sha256": "2" * 64,
        "observed_hdf5_sha256": "3" * 64,
        "required_inference_inventory_sha256": "4" * 64,
        "restored_required": [],
        "restored_required_count": 0,
        "schema": "crux.legacy-tf2-conversion-coverage/v1",
        "tf2_model_source_manifest_sha256": "5" * 64,
        "unmatched_required": ["tensor"],
    }
    assert coverage.unmatched_required == ("tensor",)


def test_atomic_output_is_canonical_and_replaces_only_after_complete_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit.json"
    output.write_bytes(b"prior\n")
    payload = {"schema": "example", "values": [2, 1]}

    write_evidence_atomic(output, payload)

    assert output.read_bytes() == canonical_json_bytes(payload, trailing_newline=True)

    output.write_bytes(b"prior\n")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(AuditError, match="publish"):
        write_evidence_atomic(output, payload)

    assert output.read_bytes() == b"prior\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["audit.json"]


def test_cli_publishes_only_evidence_accepted_by_the_strict_v1_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    required = tuple(_variable(f"required_{index:03d}", (index + 1,)) for index in range(78))
    inventory = [
        {"dtype": variable.dtype, "name": variable.name, "shape": list(variable.shape)}
        for variable in required
    ]
    required_path = _write_canonical(tmp_path / "required.json", inventory)
    hdf5_path = tmp_path / "weights.h5"
    hdf5_path.write_bytes(b"hdf5")
    converter_path = tmp_path / "convert.py"
    converter_path.write_bytes(b"converter")
    model_path = tmp_path / "model.py"
    model_path.write_bytes(b"model")
    output = tmp_path / "audit.json"
    checkpoint_directory = tmp_path / "checkpoint"
    checkpoint_directory.mkdir()
    _checkpoint_components(checkpoint_directory)
    checkpoint_prefix = checkpoint_directory / "model.ckpt-569400"
    monkeypatch.setattr(audit_module, "enumerate_checkpoint_variables", lambda _: required)
    monkeypatch.setattr(audit_module, "enumerate_hdf5_weights", lambda _: ())
    monkeypatch.setattr(audit_module, "enumerate_converter_layers", lambda _: ())
    observed_hdf5_sha256 = hashlib.sha256(b"hdf5").hexdigest()
    monkeypatch.setattr(
        backend_lock_module,
        "_OBSERVED_HDF5_SHA256",
        observed_hdf5_sha256,
    )

    exit_code = main(
        [
            "--checkpoint-prefix",
            str(checkpoint_prefix),
            "--hdf5",
            str(hdf5_path),
            "--expected-hdf5-sha256",
            observed_hdf5_sha256,
            "--required-inventory",
            str(required_path),
            "--converter-source",
            str(converter_path),
            "--tf2-model-source",
            str(model_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    content = output.read_bytes()
    assert content.endswith(b"\n")
    assert not content.endswith(b"\n\n")
    assert b" " not in content
    assert load_conversion_audit(output).payload["unmatched_required"] == tuple(
        variable.name for variable in required
    )


# The end-to-end fixture keeps every identity-bound input visible at the call boundary.
# pylint: disable-next=too-many-locals
def test_cli_cleanup_failure_preserves_prior_output_and_returns_audit_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    required = tuple(_variable(f"required_{index:03d}", (index + 1,)) for index in range(78))
    inventory = [
        {"dtype": variable.dtype, "name": variable.name, "shape": list(variable.shape)}
        for variable in required
    ]
    required_path = _write_canonical(tmp_path / "required.json", inventory)
    hdf5_path = tmp_path / "weights.h5"
    hdf5_path.write_bytes(b"hdf5")
    converter_path = tmp_path / "convert.py"
    converter_path.write_bytes(b"converter")
    model_path = tmp_path / "model.py"
    model_path.write_bytes(b"model")
    output = tmp_path / "audit.json"
    output.write_bytes(b"prior\n")
    checkpoint_directory = tmp_path / "checkpoint"
    checkpoint_directory.mkdir()
    _checkpoint_components(checkpoint_directory)
    checkpoint_prefix = checkpoint_directory / "model.ckpt-569400"
    monkeypatch.setattr(audit_module, "enumerate_checkpoint_variables", lambda _: required)
    monkeypatch.setattr(audit_module, "enumerate_hdf5_weights", lambda _: ())
    monkeypatch.setattr(audit_module, "enumerate_converter_layers", lambda _: ())
    observed_hdf5_sha256 = hashlib.sha256(b"hdf5").hexdigest()
    monkeypatch.setattr(
        backend_lock_module,
        "_OBSERVED_HDF5_SHA256",
        observed_hdf5_sha256,
    )
    original_temporary_directory = audit_module.tempfile.TemporaryDirectory

    class CleanupFailureTemporaryDirectory(original_temporary_directory):
        """A real temporary directory whose completed cleanup reports failure."""

        def cleanup(self) -> None:
            super().cleanup()
            raise OSError("forced cleanup failure")

    monkeypatch.setattr(
        audit_module.tempfile,
        "TemporaryDirectory",
        CleanupFailureTemporaryDirectory,
    )

    exit_code = main(
        [
            "--checkpoint-prefix",
            str(checkpoint_prefix),
            "--hdf5",
            str(hdf5_path),
            "--expected-hdf5-sha256",
            observed_hdf5_sha256,
            "--required-inventory",
            str(required_path),
            "--converter-source",
            str(converter_path),
            "--tf2-model-source",
            str(model_path),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "legacy conversion audit failed:" in captured.err
    assert "staging cleanup" in captured.err
    assert output.read_bytes() == b"prior\n"


def test_cli_rejects_zero_restored_when_required_inventory_is_not_78(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    required = _variable("onsets/conv0/weights", (3, 3, 1, 16))
    inventory = [{"dtype": required.dtype, "name": required.name, "shape": [3, 3, 1, 16]}]
    required_path = _write_canonical(tmp_path / "required.json", inventory)
    hdf5_path = tmp_path / "weights.h5"
    hdf5_path.write_bytes(b"hdf5")
    converter_path = tmp_path / "convert.py"
    converter_path.write_bytes(b"converter")
    model_path = tmp_path / "model.py"
    model_path.write_bytes(b"model")
    output = tmp_path / "audit.json"
    output.write_bytes(b"prior\n")
    monkeypatch.setattr(
        audit_module,
        "resolve_explicit_checkpoint",
        lambda _: ResolvedCheckpoint(tmp_path / "model.ckpt", "6" * 64),
    )
    monkeypatch.setattr(
        audit_module,
        "enumerate_checkpoint_variables",
        lambda _: (required,),
    )
    monkeypatch.setattr(audit_module, "enumerate_hdf5_weights", lambda _: ())

    exit_code = main(
        [
            "--checkpoint-prefix",
            str(tmp_path / "model.ckpt"),
            "--hdf5",
            str(hdf5_path),
            "--expected-hdf5-sha256",
            hashlib.sha256(b"hdf5").hexdigest(),
            "--required-inventory",
            str(required_path),
            "--converter-source",
            str(converter_path),
            "--tf2-model-source",
            str(model_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert output.read_bytes() == b"prior\n"


def test_cli_failure_leaves_prior_output_intact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    required = _variable("onsets/onset_probs/weights")
    inventory = [{"dtype": required.dtype, "name": required.name, "shape": list(required.shape)}]
    required_path = _write_canonical(tmp_path / "required.json", inventory)
    hdf5_path = tmp_path / "weights.h5"
    hdf5_path.write_bytes(b"hdf5")
    converter_path = tmp_path / "convert.py"
    converter_path.write_bytes(b"converter")
    model_path = tmp_path / "model.py"
    model_path.write_bytes(b"model")
    output = tmp_path / "audit.json"
    output.write_bytes(b"prior\n")
    monkeypatch.setattr(
        audit_module,
        "resolve_explicit_checkpoint",
        lambda _: ResolvedCheckpoint(tmp_path / "model.ckpt", "6" * 64),
    )
    monkeypatch.setattr(
        audit_module,
        "enumerate_checkpoint_variables",
        lambda _: (required,),
    )
    monkeypatch.setattr(
        audit_module,
        "enumerate_hdf5_weights",
        lambda _: (_variable("onset_probs/kernel"),),
    )

    exit_code = main(
        [
            "--checkpoint-prefix",
            str(tmp_path / "model.ckpt"),
            "--hdf5",
            str(hdf5_path),
            "--expected-hdf5-sha256",
            hashlib.sha256(b"hdf5").hexdigest(),
            "--required-inventory",
            str(required_path),
            "--converter-source",
            str(converter_path),
            "--tf2-model-source",
            str(model_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert output.read_bytes() == b"prior\n"
