from __future__ import annotations

import builtins
import importlib
import sys

import pytest

import src.benchmark.prediction_artifact as prediction_artifact
import src.benchmark.scorer_input as scorer_input
from src.benchmark.prediction_artifact import render_prediction_artifact
from src.benchmark.scorer_input import CanonicalMappingRequired, read_scorer_events
from tests.benchmark.test_prediction_artifact import (
    make_heuristic_event,
    make_heuristic_prediction,
)


def test_native_prediction_requires_canonical_mapping() -> None:
    content = render_prediction_artifact(make_heuristic_prediction())

    with pytest.raises(
        CanonicalMappingRequired,
        match=r"^canonical_mapping_required$",
    ):
        read_scorer_events(content)


def test_scorer_gate_uses_only_the_common_prediction_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = render_prediction_artifact(make_heuristic_prediction())

    def fail_if_reparsed(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("scorer_input must not parse artifact lines directly")

    monkeypatch.setattr(
        scorer_input,
        "strict_json_loads",
        fail_if_reparsed,
        raising=False,
    )

    with pytest.raises(CanonicalMappingRequired, match="canonical_mapping_required"):
        scorer_input.read_scorer_events(content)
    with pytest.raises(prediction_artifact.PredictionArtifactError):
        scorer_input.read_scorer_events(b"not-json\n")


def test_scorer_gate_never_guesses_a_class_from_native_midi() -> None:
    obvious_midi_kick = make_heuristic_event(
        native_class_id="midi_36",
        native_midi_note=36,
        velocity_midi=100,
    )
    content = render_prediction_artifact(make_heuristic_prediction(events=(obvious_midi_kick,)))

    with pytest.raises(CanonicalMappingRequired, match="canonical_mapping_required"):
        read_scorer_events(content)


def test_scorer_gate_import_and_use_are_backend_and_mapping_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = render_prediction_artifact(make_heuristic_prediction())
    real_import = builtins.__import__
    module_name = "src.benchmark.scorer_input"
    original = sys.modules[module_name]
    saved_state = dict(original.__dict__)

    try:

        def guarded_import(
            name: str,
            globals_: dict[str, object] | None = None,
            locals_: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            forbidden_fromlist = {"mapping", "oaf_adapter", "oaf_backend"}
            if (
                "oaf" in name
                or name == "src.benchmark.mapping"
                or forbidden_fromlist.intersection(fromlist)
            ):
                raise AssertionError(f"forbidden scorer import: {name}")
            return real_import(name, globals_, locals_, fromlist, level)

        monkeypatch.delitem(sys.modules, "src.benchmark.mapping", raising=False)
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        module = importlib.reload(original)

        with pytest.raises(module.CanonicalMappingRequired, match="canonical_mapping_required"):
            module.read_scorer_events(content)
    finally:
        sys.modules[module_name] = original
        original.__dict__.clear()
        original.__dict__.update(saved_state)
