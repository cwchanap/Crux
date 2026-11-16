"""Focused checks for the graph-check probe's moving-statistics validation.

TensorFlow 2.15 only executes under the isolated Python 3.11 runtime, so these
tests feed the validator the same kind of NumPy arrays the real Slim session
returns. The executed graph in ``graph_check.py`` calls the same function.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parents[2]
GRAPH_CHECK_PATH = REPO_ROOT / "experiments/oaf_tf215/graph_check.py"
MISSING = object()


def _load_graph_check() -> ModuleType:
    spec = importlib.util.spec_from_file_location("crux_oaf_tf215_graph_check", GRAPH_CHECK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {GRAPH_CHECK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def graph_check() -> ModuleType:
    return _load_graph_check()


def _arrays(output: object = MISSING, mean: object = MISSING, variance: object = MISSING) -> dict:
    """Return finite probe-shaped arrays, optionally replaced by the caller."""
    values = {
        "output": np.full((2, 8, 8, 4), 0.25, dtype=np.float32),
        "mean": np.arange(4, dtype=np.float32),
        "variance": np.full((4,), 0.5, dtype=np.float32),
    }
    for name, supplied in (("output", output), ("mean", mean), ("variance", variance)):
        if supplied is not MISSING:
            values[name] = supplied
    return values


def test_accepts_finite_output_and_updated_moving_statistics(graph_check: ModuleType) -> None:
    values = _arrays()
    graph_check.validate_slim_statistics(np, values["output"], values["mean"], values["variance"])


@pytest.mark.parametrize("field", ["output", "mean", "variance"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_output_or_moving_statistics(
    graph_check: ModuleType, field: str, value: float
) -> None:
    values = _arrays()
    values[field].flat[0] = value
    with pytest.raises(ValueError, match=f"{field} is not finite"):
        graph_check.validate_slim_statistics(
            np, values["output"], values["mean"], values["variance"]
        )


def test_rejects_moving_variance_that_never_changed(graph_check: ModuleType) -> None:
    values = _arrays()
    with pytest.raises(ValueError, match="moving variance was never updated"):
        graph_check.validate_slim_statistics(
            np, values["output"], values["mean"], np.ones_like(values["variance"])
        )


def test_rejects_moving_mean_that_never_changed(graph_check: ModuleType) -> None:
    values = _arrays()
    with pytest.raises(ValueError, match="moving mean was never updated"):
        graph_check.validate_slim_statistics(
            np, values["output"], np.zeros_like(values["mean"]), values["variance"]
        )
