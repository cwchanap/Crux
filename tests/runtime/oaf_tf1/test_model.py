from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


def test_model_import_does_not_load_tensorflow() -> None:
    sys.modules.pop("tensorflow", None)
    importlib.import_module("runtime.oaf_tf1.model")
    assert "tensorflow" not in sys.modules


def test_required_tensor_inventory_restores_exactly_78_and_names_missing_tensor() -> None:
    from runtime.oaf_tf1.model import OafModelError, validate_restored_tensor_inventory

    required = tuple(f"tensor_{index:03d}" for index in range(78))
    assert validate_restored_tensor_inventory(required) == 78

    missing = required[:-1]
    with pytest.raises(OafModelError, match=required[-1]):
        validate_restored_tensor_inventory(missing, required_names=required)


def test_required_inventory_fixture_has_exact_130_78_52_partition() -> None:
    from runtime.oaf_tf1.model import TensorCoverage, validate_tensor_coverage

    checkpoint = [
        {"dtype": "float32", "name": f"tensor_{index:03d}", "shape": [index + 1]}
        for index in range(130)
    ]
    required = checkpoint[:78]
    non_inference = [
        {**entry, "reason": f"locked non-inference state {index:03d}"}
        for index, entry in enumerate(checkpoint[78:])
    ]
    coverage = validate_tensor_coverage(
        checkpoint_inventory=checkpoint,
        required_inventory=required,
        non_inference_inventory=non_inference,
        graph_inventory=required,
        uninitialized_required=(),
    )
    assert isinstance(coverage, TensorCoverage)
    assert coverage.restored_count == 78


def test_required_inventory_fixture_names_missing_tensor() -> None:
    from runtime.oaf_tf1.model import OafModelError, validate_tensor_coverage

    checkpoint = [
        {"dtype": "float32", "name": f"tensor_{index:03d}", "shape": [index + 1]}
        for index in range(130)
    ]
    required = checkpoint[:78]
    non_inference = [{**entry, "reason": "locked non-inference state"} for entry in checkpoint[78:]]
    missing_name = required[-1]["name"]
    checkpoint = [entry for entry in checkpoint if entry["name"] != missing_name]
    checkpoint.append({"dtype": "float32", "name": "replacement", "shape": [999]})
    checkpoint.sort(key=lambda entry: entry["name"])
    with pytest.raises(OafModelError, match=str(missing_name)):
        validate_tensor_coverage(
            checkpoint_inventory=checkpoint,
            required_inventory=required,
            non_inference_inventory=non_inference,
            graph_inventory=required,
            uninitialized_required=(),
        )


def test_stochastic_graph_gate_rejects_reachable_stochastic_operation() -> None:
    from runtime.oaf_tf1.model import OafModelError, assert_no_reachable_stochastic_ops

    deterministic = SimpleNamespace(type="MatMul", inputs=(), control_inputs=())
    assert_no_reachable_stochastic_ops((deterministic,))

    stochastic = SimpleNamespace(type="RandomUniform", inputs=(), control_inputs=())
    output = SimpleNamespace(op=stochastic)
    with pytest.raises(OafModelError, match="stochastic"):
        assert_no_reachable_stochastic_ops((output,))


@pytest.mark.parametrize(
    ("note", "native_class", "output_bin", "group"),
    [
        (38, "midi_38", 17, "snare"),
        (46, "midi_46", 25, "hihat"),
        (48, "midi_48", 27, "toms"),
    ],
)
def test_native_event_conversion_preserves_capture_values(
    note: int, native_class: str, output_bin: int, group: str
) -> None:
    from runtime.oaf_tf1.model import native_event_from_capture

    event = native_event_from_capture(
        start_frame=64,
        native_midi_note=note,
        raw_velocity=0.8,
        raw_confidence=0.625,
    )

    assert event.native_class_id == native_class
    assert event.model_output_bin == output_bin
    assert event.native_midi_note == note
    assert event.upstream_8hit_group_id == group
    assert event.time_sec == 64 * 512 / 44100
    assert event.confidence == 0.625
    assert event.velocity_midi == int(0.8 * 127)
