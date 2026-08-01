#!/usr/bin/env python3
"""Audit the legacy TF1-to-TF2 converter without assigning or saving weights."""

# Candidate schema tables and atomic publication intentionally mirror the strict loaders.
# pylint: disable=duplicate-code,too-many-lines

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import cast

from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backend_lock import (
    CONVERSION_AUDIT_SCHEMA,
    candidate_match_sort_key,
    load_conversion_audit,
)

CANDIDATE_MANIFEST_NAME = "candidate-manifest.json"
CANDIDATE_MANIFEST_SCHEMA = "crux.oaf-seal-candidate/v2"
AUDIT_CANDIDATE_MANIFEST_SCHEMA = "crux.oaf-audit-candidate-manifest/v1"
MATCHING_ALGORITHM = "exact_assignment_trace"
MATCHING_ALGORITHM_VERSION = "v1"

_VARIABLE_KEYS = frozenset({"dtype", "name", "shape"})
_CANDIDATE_MATCH_KEYS = frozenset(
    {
        "assigned",
        "candidate_name",
        "dtype_compatible",
        "match_kind",
        "required_name",
        "shape_compatible",
    }
)
_CANDIDATE_MATCH_KINDS = frozenset(
    {
        "dense_transpose",
        "exact_name",
        "loose_substring",
    }
)
_CANDIDATE_MANIFEST_KEYS = frozenset(
    {
        "checkpoint_components",
        "checkpoint_prefix",
        "model_artifact_set_sha256",
        "required_inference_inventory_sha256",
        "schema",
    }
)
_CHECKPOINT_COMPONENT_SUFFIXES = (
    ".data-00000-of-00001",
    ".index",
    ".meta",
)
_READ_CHUNK_BYTES = 1024 * 1024


class AuditError(ValueError):
    """The requested legacy-conversion audit is invalid or did not pass."""


@dataclass(frozen=True)
class VariableSpec:
    """Immutable name, shape, and dtype identity for one model variable."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        _require_nonempty_utf8(self.name, "variable name")
        _require_nonempty_utf8(self.dtype, "variable dtype")
        if not isinstance(self.shape, tuple):
            raise AuditError("variable shape must be a tuple")
        for dimension in self.shape:
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                raise AuditError("variable shape dimensions must be positive integers")


@dataclass(frozen=True)
class ConverterLayerSpec:
    """Metadata for one converter-eligible top-level Keras layer."""

    name: str
    kind: str
    kernel: VariableSpec
    bias: VariableSpec | None

    def __post_init__(self) -> None:
        _require_nonempty_utf8(self.name, "converter layer name")
        if self.kind not in {"Conv2D", "Dense"}:
            raise AuditError("converter layer kind must be Conv2D or Dense")
        if self.kernel.name != f"{self.name}/kernel":
            raise AuditError("converter layer kernel metadata name is invalid")
        if self.bias is not None and self.bias.name != f"{self.name}/bias":
            raise AuditError("converter layer bias metadata name is invalid")


@dataclass(frozen=True)
class ConversionCoverage:
    """Immutable exact assignment trace for the required checkpoint variables."""

    restored_required: tuple[str, ...]
    unmatched_required: tuple[str, ...]
    candidate_matches: tuple[Mapping[str, JsonValue], ...]

    def __post_init__(self) -> None:
        restored = _validate_name_tuple(self.restored_required, "restored required")
        unmatched = _validate_name_tuple(self.unmatched_required, "unmatched required")
        if set(restored) & set(unmatched):
            raise AuditError("restored and unmatched required names must be disjoint")
        required_names = set(restored) | set(unmatched)
        frozen_matches, assigned_names = _freeze_candidate_matches(
            self.candidate_matches,
            required_names,
        )
        if assigned_names != set(restored):
            raise AuditError("candidate assignments must equal restored required names")

        object.__setattr__(self, "restored_required", restored)
        object.__setattr__(self, "unmatched_required", unmatched)
        object.__setattr__(self, "candidate_matches", tuple(frozen_matches))

    @property
    def restored_required_count(self) -> int:
        return len(self.restored_required)


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """Verified checkpoint prefix and its content-addressed component identity."""

    prefix: Path
    model_artifact_set_sha256: str
    checkpoint_components: tuple[Mapping[str, JsonValue], ...] = ()


@dataclass(frozen=True)
# The staged record keeps each enumerated path adjacent to its verified identity.
# pylint: disable-next=too-many-instance-attributes
class StagedAuditInputs:
    """Private verified copies used for every optional-library enumeration."""

    checkpoint_prefix: Path
    checkpoint_components: tuple[Mapping[str, JsonValue], ...]
    hdf5: Path
    observed_hdf5_sha256: str
    converter_source: Path
    converter_source_sha256: str
    tf2_model_source: Path
    tf2_model_source_sha256: str

    def verify(self) -> None:
        """Reverify every staged byte after metadata enumeration."""

        _require_exact_checkpoint_directory(self.checkpoint_prefix)
        if tuple(_checkpoint_component_inventory(self.checkpoint_prefix)) != tuple(
            dict(component) for component in self.checkpoint_components
        ):
            raise AuditError("staged checkpoint component identity changed")
        for path, expected, label in (
            (self.hdf5, self.observed_hdf5_sha256, "staged HDF5"),
            (
                self.converter_source,
                self.converter_source_sha256,
                "staged converter source",
            ),
            (
                self.tf2_model_source,
                self.tf2_model_source_sha256,
                "staged TF2 model source",
            ),
        ):
            actual, _ = _hash_regular_file(path, label)
            if actual != expected:
                raise AuditError(f"{label} identity changed")


def audit_conversion_coverage(
    *,
    checkpoint_variables: Sequence[VariableSpec],
    required_variables: Sequence[VariableSpec],
    converter_layers: Sequence[ConverterLayerSpec],
) -> ConversionCoverage:
    """Trace the converter's name/shape/dtype assignment rules without mutation."""

    checkpoint = _validate_variable_specs(checkpoint_variables, "checkpoint variables")
    required = _validate_variable_specs(required_variables, "required variables")
    layers = _validate_converter_layers(converter_layers)
    checkpoint_by_name = {variable.name: variable for variable in checkpoint}
    for required_variable in required:
        if checkpoint_by_name.get(required_variable.name) != required_variable:
            raise AuditError("required variable identity does not exactly match the checkpoint")

    matches: list[dict[str, JsonValue]] = []
    restored: set[str] = set()
    for required_variable in required:
        for layer in layers:
            relation = _candidate_relation(required_variable, layer)
            if relation is None:
                continue
            matches.append(relation)
            if cast(bool, relation["assigned"]):
                restored.add(required_variable.name)
    matches.sort(key=candidate_match_sort_key)

    required_names = tuple(variable.name for variable in required)
    return ConversionCoverage(
        restored_required=tuple(name for name in required_names if name in restored),
        unmatched_required=tuple(name for name in required_names if name not in restored),
        candidate_matches=tuple(matches),
    )


def load_required_inventory(path: Path) -> tuple[tuple[VariableSpec, ...], str]:
    """Strict-load a canonical required-variable inventory and reproduce its identity."""

    value, content = _load_canonical_json(Path(path), "required inventory")
    if not isinstance(value, list):
        raise AuditError("required inventory must be an array")
    variables = _variable_specs_from_rows(value, "required inventory")
    return variables, sha256_hex(content[:-1])


def resolve_explicit_checkpoint(checkpoint_prefix: Path) -> ResolvedCheckpoint:
    """Hash and bind an explicitly supplied checkpoint prefix."""

    prefix = Path(checkpoint_prefix)
    _require_directory_without_symlinks(prefix.parent, "checkpoint directory")
    components = _checkpoint_component_inventory(prefix)
    return ResolvedCheckpoint(
        prefix=prefix,
        model_artifact_set_sha256=sha256_hex(canonical_json_bytes(components)),
        checkpoint_components=tuple(MappingProxyType(component) for component in components),
    )


def resolve_candidate_checkpoint(
    candidate: Path,
    model_cache_root: Path,
    *,
    expected_required_inventory_sha256: str,
) -> ResolvedCheckpoint:
    """Resolve a strict seal candidate to its verified content-addressed cache prefix."""

    candidate_directory = _require_directory_without_symlinks(
        Path(candidate), "candidate directory"
    )
    cache_root = _require_directory_without_symlinks(Path(model_cache_root), "model cache root")
    value, _ = _load_canonical_json(
        candidate_directory / CANDIDATE_MANIFEST_NAME,
        "candidate manifest",
    )
    if not isinstance(value, dict) or set(value) != _CANDIDATE_MANIFEST_KEYS:
        raise AuditError("candidate manifest fields must match the exact schema")
    if value["schema"] != AUDIT_CANDIDATE_MANIFEST_SCHEMA:
        raise AuditError("candidate manifest schema is unsupported")

    artifact_sha256 = _require_sha256_value(
        value["model_artifact_set_sha256"],
        "candidate model artifact set",
    )
    required_sha256 = _require_sha256_value(
        value["required_inference_inventory_sha256"],
        "candidate required inventory",
    )
    expected_required_sha256 = _require_sha256_value(
        expected_required_inventory_sha256,
        "expected required inventory",
    )
    if required_sha256 != expected_required_sha256:
        raise AuditError("candidate required inventory SHA-256 disagrees with input")

    components = _validate_component_rows(value["checkpoint_components"])
    reproduced_artifact_sha256 = sha256_hex(canonical_json_bytes(components))
    if artifact_sha256 != reproduced_artifact_sha256:
        raise AuditError("candidate model artifact set SHA-256 disagrees with components")

    expected_relative_prefix = PurePosixPath("sha256", artifact_sha256, "model.ckpt-569400")
    prefix_value = value["checkpoint_prefix"]
    if (
        not isinstance(prefix_value, str)
        or prefix_value != expected_relative_prefix.as_posix()
        or PurePosixPath(prefix_value).is_absolute()
        or ".." in PurePosixPath(prefix_value).parts
    ):
        raise AuditError("candidate checkpoint prefix disagrees with cache identity")
    prefix = cache_root.joinpath(*expected_relative_prefix.parts)
    _require_directory_without_symlinks(prefix.parent, "model cache directory")
    _require_exact_checkpoint_directory(prefix)

    actual_components = _checkpoint_component_inventory(prefix)
    if actual_components != components:
        raise AuditError("cached checkpoint component identity disagrees with candidate")
    return ResolvedCheckpoint(
        prefix=prefix,
        model_artifact_set_sha256=artifact_sha256,
        checkpoint_components=tuple(MappingProxyType(component) for component in components),
    )


@contextmanager
# One staging transaction deliberately binds all four independent input families.
# pylint: disable-next=too-many-locals
def stage_verified_inputs(
    *,
    resolved: ResolvedCheckpoint,
    hdf5: Path,
    converter_source: Path,
    tf2_model_source: Path,
    staging_parent: Path,
):
    """Yield private copies whose hashes are bound to all later enumeration."""

    parent = _require_directory_without_symlinks(staging_parent, "staging directory")
    if not resolved.checkpoint_components:
        raise AuditError("resolved checkpoint component inventory is unavailable")
    temporary_directory = tempfile.TemporaryDirectory(
        dir=parent,
        prefix=".legacy-conversion-inputs-",
    )
    try:
        root = Path(temporary_directory.name)
        os.chmod(root, 0o700)
        checkpoint_directory = root / "checkpoint"
        checkpoint_directory.mkdir(mode=0o700)
        checkpoint_prefix = checkpoint_directory / resolved.prefix.name
        staged_components: list[dict[str, JsonValue]] = []
        for expected_value in resolved.checkpoint_components:
            expected = dict(expected_value)
            name = cast(str, expected["name"])
            digest, size = _copy_verified_regular_file(
                resolved.prefix.parent / name,
                checkpoint_directory / name,
                "checkpoint component",
                expected_sha256=cast(str, expected["sha256"]),
                expected_size=cast(int, expected["size"]),
            )
            staged_components.append({"name": name, "sha256": digest, "size": size})
        staged_components.sort(key=lambda component: _utf8_key(cast(str, component["name"])))
        if (
            sha256_hex(canonical_json_bytes(staged_components))
            != resolved.model_artifact_set_sha256
        ):
            raise AuditError("staged checkpoint artifact-set identity disagrees")

        staged_hdf5 = root / "weights.h5"
        hdf5_sha256, _ = _copy_verified_regular_file(
            Path(hdf5),
            staged_hdf5,
            "HDF5",
        )
        staged_converter = root / "convert.py"
        converter_sha256, _ = _copy_verified_regular_file(
            Path(converter_source),
            staged_converter,
            "converter source",
        )
        staged_model = root / "tf2_magenta_model.py"
        model_sha256, _ = _copy_verified_regular_file(
            Path(tf2_model_source),
            staged_model,
            "TF2 model source",
        )
        staged = StagedAuditInputs(
            checkpoint_prefix=checkpoint_prefix,
            checkpoint_components=tuple(
                MappingProxyType(component) for component in staged_components
            ),
            hdf5=staged_hdf5,
            observed_hdf5_sha256=hdf5_sha256,
            converter_source=staged_converter,
            converter_source_sha256=converter_sha256,
            tf2_model_source=staged_model,
            tf2_model_source_sha256=model_sha256,
        )
        yield staged
    finally:
        active_error = sys.exception()
        try:
            temporary_directory.cleanup()
        except OSError as cleanup_error:
            if active_error is not None:
                active_error.add_note(f"private staging cleanup also failed: {cleanup_error}")
            else:
                raise AuditError("private staging cleanup failed") from cleanup_error


def enumerate_checkpoint_variables(checkpoint_prefix: Path) -> tuple[VariableSpec, ...]:
    """Enumerate checkpoint metadata through a lazily imported TensorFlow reader."""

    try:
        tensorflow = importlib.import_module("tensorflow")
    except ImportError as error:
        raise AuditError("TensorFlow is required to enumerate the checkpoint") from error
    try:
        if hasattr(tensorflow.train, "load_checkpoint"):
            reader = tensorflow.train.load_checkpoint(str(checkpoint_prefix))
        else:
            reader = tensorflow.compat.v1.train.NewCheckpointReader(str(checkpoint_prefix))
        shapes = reader.get_variable_to_shape_map()
        dtypes = reader.get_variable_to_dtype_map()
    except (AttributeError, OSError, RuntimeError, ValueError) as error:
        raise AuditError("TensorFlow could not enumerate the checkpoint") from error

    if set(shapes) != set(dtypes):
        raise AuditError("checkpoint shape and dtype inventories disagree")
    variables = tuple(
        VariableSpec(
            name=name,
            shape=tuple(int(dimension) for dimension in shapes[name]),
            dtype=_dtype_name(dtypes[name]),
        )
        for name in sorted(shapes, key=_utf8_key)
    )
    return _validate_variable_specs(variables, "checkpoint variables")


def enumerate_hdf5_weights(hdf5_path: Path) -> tuple[VariableSpec, ...]:
    """Enumerate HDF5 datasets through a lazily imported h5py dependency."""

    try:
        h5py = importlib.import_module("h5py")
    except ImportError as error:
        raise AuditError("h5py is required to enumerate the HDF5 weights") from error

    variables: list[VariableSpec] = []

    def collect(name: str, value: object) -> None:
        if isinstance(value, h5py.Dataset):
            variables.append(
                VariableSpec(
                    name=_keras_weight_alias(name),
                    shape=tuple(int(dimension) for dimension in value.shape),
                    dtype=_dtype_name(value.dtype),
                )
            )

    try:
        with h5py.File(hdf5_path, "r") as hdf5_file:
            hdf5_file.visititems(collect)
    except (OSError, RuntimeError, ValueError) as error:
        raise AuditError("h5py could not enumerate the HDF5 weights") from error
    return _validate_variable_specs(variables, "Keras weights")


def enumerate_converter_layers(tf2_model_source: Path) -> tuple[ConverterLayerSpec, ...]:
    """Build the audited model and inspect eligible immediate layer metadata only."""

    try:
        tensorflow = importlib.import_module("tensorflow")
        specification = importlib.util.spec_from_file_location(
            "_crux_legacy_tf2_model_audit",
            tf2_model_source,
        )
        if specification is None or specification.loader is None:
            raise AuditError("TF2 model source could not be imported")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        model = module.create_drum_model()
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as error:
        if isinstance(error, AuditError):
            raise
        raise AuditError("TF2 model source could not provide layer metadata") from error
    return converter_layer_specs(
        model,
        conv2d_type=tensorflow.keras.layers.Conv2D,
        dense_type=tensorflow.keras.layers.Dense,
    )


def converter_layer_specs(
    model: object,
    *,
    conv2d_type: type,
    dense_type: type,
) -> tuple[ConverterLayerSpec, ...]:
    """Extract weight metadata from actual top-level Conv2D and Dense layers."""

    layers: list[ConverterLayerSpec] = []
    for layer in model.layers:
        if isinstance(layer, conv2d_type):
            kind = "Conv2D"
        elif isinstance(layer, dense_type):
            kind = "Dense"
        else:
            continue
        name = _require_nonempty_utf8(layer.name, "converter layer name")
        kernel = VariableSpec(
            name=f"{name}/kernel",
            shape=tuple(int(dimension) for dimension in layer.kernel.shape),
            dtype=_dtype_name(layer.kernel.dtype),
        )
        layer_bias = getattr(layer, "bias", None)
        bias = (
            None
            if layer_bias is None
            else VariableSpec(
                name=f"{name}/bias",
                shape=tuple(int(dimension) for dimension in layer_bias.shape),
                dtype=_dtype_name(layer_bias.dtype),
            )
        )
        layers.append(ConverterLayerSpec(name=name, kind=kind, kernel=kernel, bias=bias))
    return _validate_converter_layers(layers)


# The schema has five independent content identities plus the audited coverage.
# pylint: disable-next=too-many-arguments
def build_evidence(
    *,
    coverage: ConversionCoverage,
    converter_source_manifest_sha256: str,
    model_artifact_set_sha256: str,
    observed_hdf5_sha256: str,
    required_inference_inventory_sha256: str,
    tf2_model_source_manifest_sha256: str,
) -> dict[str, JsonValue]:
    """Build the exact v1 evidence object after every locked expectation passes."""

    if coverage.restored_required_count != 0:
        raise AuditError("legacy conversion audit expected zero restored required tensors")
    hashes = {
        "converter_source_manifest_sha256": converter_source_manifest_sha256,
        "model_artifact_set_sha256": model_artifact_set_sha256,
        "observed_hdf5_sha256": observed_hdf5_sha256,
        "required_inference_inventory_sha256": required_inference_inventory_sha256,
        "tf2_model_source_manifest_sha256": tf2_model_source_manifest_sha256,
    }
    for field, value in hashes.items():
        hashes[field] = _require_sha256_value(value, field)

    return {
        "candidate_matches": [dict(match) for match in coverage.candidate_matches],
        "converter_source_manifest_sha256": hashes["converter_source_manifest_sha256"],
        "matching_algorithm": MATCHING_ALGORITHM,
        "matching_algorithm_version": MATCHING_ALGORITHM_VERSION,
        "model_artifact_set_sha256": hashes["model_artifact_set_sha256"],
        "observed_hdf5_sha256": hashes["observed_hdf5_sha256"],
        "required_inference_inventory_sha256": hashes["required_inference_inventory_sha256"],
        "restored_required": list(coverage.restored_required),
        "restored_required_count": coverage.restored_required_count,
        "schema": CONVERSION_AUDIT_SCHEMA,
        "tf2_model_source_manifest_sha256": hashes["tf2_model_source_manifest_sha256"],
        "unmatched_required": list(coverage.unmatched_required),
    }


def write_evidence_atomic(
    output: Path,
    payload: Mapping[str, JsonValue],
    *,
    validate_strict_conversion_audit: bool = False,
) -> None:
    """Atomically publish canonical evidence without following an existing symlink."""

    output_path = Path(output)
    parent = _require_directory_without_symlinks(output_path.parent, "output directory")
    if output_path.exists() or output_path.is_symlink():
        try:
            output_status = os.lstat(output_path)
        except OSError as error:
            raise AuditError("output path could not be inspected") from error
        if stat.S_ISLNK(output_status.st_mode) or not stat.S_ISREG(output_status.st_mode):
            raise AuditError("output path must be a no-follow regular file")

    content = canonical_json_bytes(dict(payload), trailing_newline=True)
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as temporary_file:
            descriptor = -1
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if validate_strict_conversion_audit:
            load_conversion_audit(temporary_path)
        os.replace(temporary_path, output_path)
        temporary_path = None
    except (OSError, StrictJsonError, ValueError) as error:
        raise AuditError("could not atomically publish conversion audit") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _candidate_relation(
    required: VariableSpec,
    layer: ConverterLayerSpec,
) -> dict[str, JsonValue] | None:
    if layer.kind == "Conv2D":
        if "conv" not in required.name.lower() or "conv" not in layer.name.lower():
            return None
    elif not (
        layer.name in required.name or layer.name.replace("_", "") in required.name.replace("_", "")
    ):
        return None

    if "weights" in required.name or "kernel" in required.name:
        candidate = layer.kernel
        required_role = "kernel"
    elif "bias" in required.name and layer.bias is not None:
        candidate = layer.bias
        required_role = "bias"
    else:
        return None

    if layer.kind == "Conv2D":
        match_kind = "loose_substring"
        shape_compatible = required.shape == candidate.shape
    else:
        transpose_compatible = (
            required_role == "kernel"
            and len(required.shape) == 2
            and tuple(reversed(required.shape)) == candidate.shape
        )
        if transpose_compatible:
            match_kind = "dense_transpose"
            shape_compatible = True
        else:
            match_kind = "exact_name"
            shape_compatible = required.shape == candidate.shape
    dtype_compatible = required.dtype == candidate.dtype
    assigned = shape_compatible and dtype_compatible
    return {
        "assigned": assigned,
        "candidate_name": candidate.name,
        "dtype_compatible": dtype_compatible,
        "match_kind": match_kind,
        "required_name": required.name,
        "shape_compatible": shape_compatible,
    }


def _validate_converter_layers(
    layers: Sequence[ConverterLayerSpec],
) -> tuple[ConverterLayerSpec, ...]:
    values = tuple(layers)
    if any(not isinstance(layer, ConverterLayerSpec) for layer in values):
        raise AuditError("converter layers must contain ConverterLayerSpec records")
    names = [layer.name for layer in values]
    if len(set(names)) != len(names):
        raise AuditError("converter layer names must be unique")
    return tuple(sorted(values, key=lambda layer: _utf8_key(layer.name)))


def _freeze_candidate_matches(
    values: Sequence[Mapping[str, JsonValue]],
    required_names: set[str],
) -> tuple[tuple[Mapping[str, JsonValue], ...], set[str]]:
    frozen_matches: list[Mapping[str, JsonValue]] = []
    semantic_keys: list[tuple[bytes, bytes, bytes]] = []
    assigned_names: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping) or set(value) != _CANDIDATE_MATCH_KEYS:
            raise AuditError("candidate match fields must match the exact schema")
        row = dict(value)
        required_name = _require_nonempty_utf8(row["required_name"], "candidate required name")
        _require_nonempty_utf8(row["candidate_name"], "candidate name")
        if row["match_kind"] not in _CANDIDATE_MATCH_KINDS:
            raise AuditError("candidate match kind is unsupported")
        if required_name not in required_names:
            raise AuditError("candidate required name is outside the required inventory")
        flags = _validate_candidate_flags(row)
        if flags["assigned"] != (flags["dtype_compatible"] and flags["shape_compatible"]):
            raise AuditError("candidate assigned flag contradicts exact compatibility")
        if flags["assigned"]:
            assigned_names.add(required_name)
        frozen_row = MappingProxyType(row)
        frozen_matches.append(frozen_row)
        semantic_keys.append(candidate_match_sort_key(frozen_row))

    if len(set(semantic_keys)) != len(semantic_keys):
        raise AuditError("candidate match semantic relations must be unique")
    if semantic_keys != sorted(semantic_keys):
        raise AuditError("candidate matches must follow UTF-8 semantic order")
    return tuple(frozen_matches), assigned_names


def _validate_candidate_flags(row: Mapping[str, JsonValue]) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for field in ("assigned", "dtype_compatible", "shape_compatible"):
        flag = row[field]
        if not isinstance(flag, bool):
            raise AuditError(f"candidate match {field} must be boolean")
        flags[field] = flag
    return flags


def _validate_variable_specs(
    variables: Sequence[VariableSpec],
    label: str,
) -> tuple[VariableSpec, ...]:
    values = tuple(variables)
    if any(not isinstance(variable, VariableSpec) for variable in values):
        raise AuditError(f"{label} must contain VariableSpec records")
    names = [variable.name for variable in values]
    if len(set(names)) != len(names):
        raise AuditError(f"{label} names must be unique")
    return tuple(sorted(values, key=lambda variable: _utf8_key(variable.name)))


def _variable_specs_from_rows(value: list[JsonValue], label: str) -> tuple[VariableSpec, ...]:
    variables: list[VariableSpec] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != _VARIABLE_KEYS:
            raise AuditError(f"{label} row fields must match the exact schema")
        shape = item["shape"]
        if not isinstance(shape, list):
            raise AuditError(f"{label} shape must be an array")
        if not isinstance(item["name"], str) or not isinstance(item["dtype"], str):
            raise AuditError(f"{label} names and dtypes must be strings")
        variables.append(
            VariableSpec(
                name=item["name"],
                shape=tuple(shape),
                dtype=item["dtype"],
            )
        )
    validated = _validate_variable_specs(variables, label)
    if tuple(variables) != validated:
        raise AuditError(f"{label} names must follow UTF-8 lexical order")
    return validated


def _validate_component_rows(value: JsonValue) -> list[dict[str, JsonValue]]:
    if not isinstance(value, list):
        raise AuditError("candidate checkpoint components must be an array")
    expected_names = {f"model.ckpt-569400{suffix}" for suffix in _CHECKPOINT_COMPONENT_SUFFIXES}
    components: list[dict[str, JsonValue]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "sha256", "size"}:
            raise AuditError("candidate checkpoint component fields are invalid")
        name = _require_nonempty_utf8(item["name"], "checkpoint component name")
        digest = _require_sha256_value(item["sha256"], "checkpoint component")
        size = item["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise AuditError("checkpoint component size must be a positive integer")
        components.append({"name": name, "sha256": digest, "size": size})
    names = [cast(str, component["name"]) for component in components]
    if set(names) != expected_names or names != sorted(names, key=_utf8_key):
        raise AuditError("candidate checkpoint component names are invalid")
    return components


def _checkpoint_component_inventory(prefix: Path) -> list[dict[str, JsonValue]]:
    components: list[dict[str, JsonValue]] = []
    for suffix in _CHECKPOINT_COMPONENT_SUFFIXES:
        path = prefix.parent / f"{prefix.name}{suffix}"
        digest, size = _hash_regular_file(path, "checkpoint component")
        components.append({"name": path.name, "sha256": digest, "size": size})
    components.sort(key=lambda component: _utf8_key(cast(str, component["name"])))
    return components


def _require_exact_checkpoint_directory(prefix: Path) -> None:
    expected_names = {f"{prefix.name}{suffix}" for suffix in _CHECKPOINT_COMPONENT_SUFFIXES}
    try:
        actual_names = {entry.name for entry in os.scandir(prefix.parent)}
    except OSError as error:
        raise AuditError("model cache directory could not be enumerated") from error
    if actual_names != expected_names:
        raise AuditError("model cache directory must contain exactly three components")


def _load_canonical_json(path: Path, label: str) -> tuple[JsonValue, bytes]:
    content = _read_regular_file(path, label)
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise AuditError(f"{label} must have exactly one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise AuditError(str(error)) from None
    return value, content


def _read_regular_file(path: Path, label: str) -> bytes:
    descriptor, before = _open_regular_file(Path(path), label)
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        _require_unchanged_descriptor(descriptor, before, total, label)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


# The descriptor-copy loop keeps byte count, digest, and both descriptors in one scope.
# pylint: disable-next=too-many-locals
def _copy_verified_regular_file(
    source: Path,
    destination: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[str, int]:
    source_descriptor, before = _open_regular_file(Path(source), label)
    destination_descriptor = -1
    digest = hashlib.sha256()
    total = 0
    try:
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
        )
        while True:
            chunk = os.read(source_descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        os.fsync(destination_descriptor)
        _require_unchanged_descriptor(source_descriptor, before, total, label)
    except OSError as error:
        raise AuditError(f"{label} could not be copied into private staging") from error
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise AuditError(f"{label} SHA-256 changed before staging")
    if expected_size is not None and total != expected_size:
        raise AuditError(f"{label} size changed before staging")
    return actual_sha256, total


def _hash_regular_file(path: Path, label: str) -> tuple[str, int]:
    descriptor, before = _open_regular_file(Path(path), label)
    digest = hashlib.sha256()
    total = 0
    try:
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        _require_unchanged_descriptor(descriptor, before, total, label)
    finally:
        os.close(descriptor)
    return digest.hexdigest(), total


def _open_regular_file(path: Path, label: str) -> tuple[int, os.stat_result]:
    _require_directory_without_symlinks(path.parent, f"{label} directory")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    non_block = getattr(os, "O_NONBLOCK", None)
    if no_follow is None or non_block is None:
        raise AuditError(f"{label} no-follow reads are unavailable")
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow | non_block)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise OSError
        return descriptor, status
    except OSError:
        if "descriptor" in locals():
            os.close(descriptor)
        raise AuditError(f"{label} must be a no-follow regular file") from None


def _require_unchanged_descriptor(
    descriptor: int,
    before: os.stat_result,
    bytes_read: int,
    label: str,
) -> None:
    after = os.fstat(descriptor)
    if (
        bytes_read != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ino != before.st_ino
        or after.st_dev != before.st_dev
    ):
        raise AuditError(f"{label} changed while being read")


def _require_directory_without_symlinks(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    chain = tuple(reversed(absolute.parents)) + (absolute,)
    try:
        for component in chain:
            status = os.lstat(component)
            if stat.S_ISLNK(status.st_mode):
                raise AuditError(f"{label} must not contain a symlink")
            if not stat.S_ISDIR(status.st_mode):
                raise AuditError(f"{label} must be a directory")
    except FileNotFoundError:
        raise AuditError(f"{label} must be an existing directory") from None
    except OSError as error:
        raise AuditError(f"{label} could not be inspected") from error
    return absolute


def _validate_name_tuple(names: Sequence[str], label: str) -> tuple[str, ...]:
    values = tuple(_require_nonempty_utf8(name, label) for name in names)
    if len(set(values)) != len(values):
        raise AuditError(f"{label} names must be unique")
    if values != tuple(sorted(values, key=_utf8_key)):
        raise AuditError(f"{label} names must follow UTF-8 lexical order")
    return values


def _require_nonempty_utf8(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuditError(f"{label} must be a nonempty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise AuditError(f"{label} must be valid UTF-8") from None
    return value


def _require_sha256_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AuditError(f"{label} must be lowercase SHA-256")
    try:
        return require_sha256(value, label)
    except StrictJsonError as error:
        raise AuditError(str(error)) from None


def _utf8_key(value: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        raise AuditError("names must be valid UTF-8") from None


def _keras_weight_alias(name: str) -> str:
    normalized = name.removesuffix(":0")
    parts = normalized.split("/")
    if len(parts) >= 3 and parts[-2] == "vars" and parts[-1] in {"0", "1"}:
        role = "kernel" if parts[-1] == "0" else "bias"
        return "/".join((*parts[:-2], role))
    return normalized


def _dtype_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    rendered = str(value)
    if not rendered:
        raise AuditError("variable dtype must be available")
    return rendered


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit legacy TF1-to-TF2 conversion coverage without mutation."
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint-prefix", type=Path)
    checkpoint_group.add_argument("--candidate", type=Path)
    parser.add_argument("--model-cache-root", type=Path)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--expected-hdf5-sha256", required=True)
    parser.add_argument("--required-inventory", type=Path, required=True)
    parser.add_argument("--converter-source", type=Path, required=True)
    parser.add_argument("--tf2-model-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the strict audit CLI, publishing only a passing zero-restored result."""

    arguments = _build_argument_parser().parse_args(argv)
    try:
        required_variables, required_sha256 = load_required_inventory(arguments.required_inventory)
        if len(required_variables) != 78:
            raise AuditError("required inventory must contain exactly 78 entries")
        expected_hdf5_sha256 = _require_sha256_value(
            arguments.expected_hdf5_sha256,
            "expected HDF5",
        )
        if arguments.candidate is not None:
            if arguments.model_cache_root is None:
                raise AuditError("--candidate requires --model-cache-root")
            resolved = resolve_candidate_checkpoint(
                arguments.candidate,
                arguments.model_cache_root,
                expected_required_inventory_sha256=required_sha256,
            )
        else:
            if arguments.model_cache_root is not None:
                raise AuditError("--model-cache-root is valid only together with --candidate")
            resolved = resolve_explicit_checkpoint(arguments.checkpoint_prefix)

        with stage_verified_inputs(
            resolved=resolved,
            hdf5=arguments.hdf5,
            converter_source=arguments.converter_source,
            tf2_model_source=arguments.tf2_model_source,
            staging_parent=arguments.output.parent,
        ) as staged:
            if staged.observed_hdf5_sha256 != expected_hdf5_sha256:
                raise AuditError("observed HDF5 SHA-256 does not match the expected hash")
            checkpoint_variables = enumerate_checkpoint_variables(staged.checkpoint_prefix)
            enumerate_hdf5_weights(staged.hdf5)
            converter_layers = enumerate_converter_layers(staged.tf2_model_source)
            coverage = audit_conversion_coverage(
                checkpoint_variables=checkpoint_variables,
                required_variables=required_variables,
                converter_layers=converter_layers,
            )
            staged.verify()
            evidence = build_evidence(
                coverage=coverage,
                converter_source_manifest_sha256=staged.converter_source_sha256,
                model_artifact_set_sha256=resolved.model_artifact_set_sha256,
                observed_hdf5_sha256=staged.observed_hdf5_sha256,
                required_inference_inventory_sha256=required_sha256,
                tf2_model_source_manifest_sha256=staged.tf2_model_source_sha256,
            )
        write_evidence_atomic(
            arguments.output,
            evidence,
            validate_strict_conversion_audit=True,
        )
        return 0
    except AuditError as error:
        print(f"legacy conversion audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
