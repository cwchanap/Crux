#!/usr/bin/env python3
"""Execute the TF1-compatibility graph checks for the OaF TF2.15 experiment.

This is a probe, not a production entry point. It runs under the Python 3.11
environment built from ``experiments/oaf_tf215/requirements.lock`` and proves
that TensorFlow 2.15 can build and execute, in graph mode:

* a ``tf_slim`` Conv2D + batch-normalization graph, including the moving
  statistics updates, and
* a legacy ``tf.compat.v1.nn.rnn_cell.LSTMCell`` graph through ``dynamic_rnn``.

Both sessions must execute for the process to exit 0. Imports alone, a built
graph, or a failed stage are failures with the stage named in the result. The
compact JSON result is written to ``--output`` on every path so CI can keep a
redacted record beside the job log.

The process environment is validated before TensorFlow is imported; the
required values (``CUDA_VISIBLE_DEVICES=-1``, ``PYTHONHASHSEED=0``,
``TF_ENABLE_ONEDNN_OPTS=0``, ``NUMBA_CACHE_DIR=/tmp/numba-cache``) have to be
set by the caller because ``PYTHONHASHSEED`` is only read at interpreter
startup. The tf-slim/TF 2.15 compatibility controls are documented in
``experiments/oaf_tf215/README.md``. Usage::

    CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_ENABLE_ONEDNN_OPTS=0 \
    NUMBA_CACHE_DIR=/tmp/numba-cache /tmp/oaf-tf215-venv/bin/python \
        experiments/oaf_tf215/graph_check.py --output graph-check.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA = "oaf-tf215-graph-check/v1"

REQUIRED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "PYTHONHASHSEED": "0",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "NUMBA_CACHE_DIR": "/tmp/numba-cache",
}


class StageFailure(RuntimeError):
    """A named graph stage did not execute."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def validate_environment() -> dict[str, Any]:
    """Check the pre-import process environment and the Numba cache location."""
    observed = {name: os.environ.get(name) for name in REQUIRED_ENVIRONMENT}
    wrong = {name: value for name, value in observed.items() if value != REQUIRED_ENVIRONMENT[name]}
    if wrong:
        raise ValueError(
            f"process environment must be set before launch: "
            f"expected {REQUIRED_ENVIRONMENT}, observed {wrong}"
        )
    if sys.flags.hash_randomization:
        raise ValueError(
            "hash randomization is enabled; PYTHONHASHSEED was not applied at interpreter startup"
        )
    cache_dir = Path(REQUIRED_ENVIRONMENT["NUMBA_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".graph-check-write-probe-") as probe:
        probe.write(b"ok")
    return {
        "flags": observed,
        "hash_randomization": sys.flags.hash_randomization,
        "numba_cache_dir": str(cache_dir),
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _restore_tf_slim_tensor_alias() -> bool:
    """Restore TF 1.x's ``framework_ops.Tensor`` name that tf-slim 1.1 reads.

    TF 2.14 moved the class to ``tensorflow.python.framework.tensor.Tensor`` and
    removed the ``ops.Tensor`` alias, which ``tf_slim.layers.utils.constant_value``
    still dereferences. Adding the alias back is additive and local to this
    process; the scratch runtime needs the same line in its compatibility patch.
    """
    from tensorflow.python.framework import ops as framework_ops
    from tensorflow.python.framework import tensor as framework_tensor

    if hasattr(framework_ops, "Tensor"):
        return False
    framework_ops.Tensor = framework_tensor.Tensor
    return True


def import_stack(stack: dict[str, Any]) -> dict[str, Any]:
    """Import TensorFlow, then switch to TF1 graph semantics before graph use."""
    import numpy as np
    import tensorflow as tf

    tf.compat.v1.disable_v2_behavior()
    alias_restored = _restore_tf_slim_tensor_alias()

    import tf_slim as slim

    stack.update(tf=tf, slim=slim, np=np)
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": _package_version("keras"),
        "tf-slim": _package_version("tf-slim"),
        "numpy": np.__version__,
        "tf_slim_tensor_alias_restored": alias_restored,
    }


def validate_slim_statistics(np: Any, output: Any, moving_mean: Any, moving_variance: Any) -> None:
    """Reject non-finite results, and moving statistics the session never updated.

    The moving variables initialize to 0 (mean) and 1 (variance); an executed
    update has to move both. Variance is compared with its initial ones rather
    than against zero so an unchanged update cannot hide behind a plausible sum.
    """
    if not np.isfinite(output).all():
        raise ValueError("Slim batch-normalization output is not finite")
    if not np.isfinite(moving_mean).all():
        raise ValueError("Slim batch-normalization moving mean is not finite")
    if not np.isfinite(moving_variance).all():
        raise ValueError("Slim batch-normalization moving variance is not finite")
    if not float(np.abs(moving_mean).sum()):
        raise ValueError("Slim batch-normalization moving mean was never updated")
    if np.array_equal(moving_variance, np.ones_like(moving_variance)):
        raise ValueError("Slim batch-normalization moving variance was never updated")


def execute_slim_batch_norm(tf: Any, slim: Any, np: Any) -> dict[str, Any]:
    """Run a real Slim Conv2D + batch-normalization session, updates included."""
    rng = np.random.RandomState(0)
    inputs_value = rng.uniform(-0.5, 0.5, size=(2, 8, 8, 3)).astype(np.float32)

    with tf.Graph().as_default() as graph:
        inputs = tf.compat.v1.placeholder(tf.float32, [2, 8, 8, 3], name="probe_inputs")
        net = slim.conv2d(inputs, 4, [3, 3], scope="probe_conv")
        # tf-slim 1.1's default batch_norm routes through ``tf_keras`` legacy
        # layers, which no published tf-keras wheel exposes at the path TF 2.15
        # asks for. fused=False with updates_collections=None keeps tf-slim's own
        # TF1 normalization implementation: same variables and inference math,
        # with the moving-average updates executed in the forward pass.
        net = slim.batch_norm(
            net,
            is_training=True,
            fused=False,
            updates_collections=None,
            scope="probe_bn",
        )
        variables = graph.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        moving_mean = next(v for v in variables if v.op.name.endswith("/moving_mean"))
        moving_variance = next(v for v in variables if v.op.name.endswith("/moving_variance"))
        with tf.compat.v1.Session(graph=graph) as session:
            session.run(tf.compat.v1.global_variables_initializer())
            output = session.run(net, feed_dict={inputs: inputs_value})
            mean_value, variance_value = session.run([moving_mean, moving_variance])

    validate_slim_statistics(np, output, mean_value, variance_value)
    return {
        "output_shape": list(output.shape),
        "output_sum": round(float(output.sum()), 6),
        "moving_mean_sum": round(float(mean_value.sum()), 6),
        "moving_variance_sum": round(float(variance_value.sum()), 6),
        "variable_count": len(graph.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)),
    }


def execute_legacy_lstm(tf: Any, np: Any) -> dict[str, Any]:
    """Run a legacy TF1 RNN cell graph through ``dynamic_rnn``."""
    rng = np.random.RandomState(1)
    inputs_value = rng.uniform(-1.0, 1.0, size=(2, 5, 3)).astype(np.float32)

    with tf.Graph().as_default() as graph:
        inputs = tf.compat.v1.placeholder(tf.float32, [2, 5, 3], name="probe_sequence")
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(6, name="probe_lstm_cell")
        outputs, final_state = tf.compat.v1.nn.dynamic_rnn(
            cell, inputs, dtype=tf.float32, scope="probe_rnn"
        )
        with tf.compat.v1.Session(graph=graph) as session:
            session.run(tf.compat.v1.global_variables_initializer())
            output, state = session.run([outputs, final_state], feed_dict={inputs: inputs_value})

    if not np.isfinite(output).all() or not np.isfinite(state.c).all():
        raise ValueError("legacy LSTM output or state is not finite")
    return {
        "output_shape": list(output.shape),
        "output_sum": round(float(output.sum()), 6),
        "final_cell_state_sum": round(float(state.c.sum()), 6),
        "variable_count": len(graph.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)),
    }


def run_stage(record: dict[str, Any], name: str, action: Callable[[], Any]) -> Any:
    """Run one named stage, recording duration, details and the failure."""
    started = time.monotonic()
    try:
        details = action()
    except Exception as error:  # noqa: BLE001 - recorded as the failed stage
        record["stages"].append(
            {
                "name": name,
                "status": "failed",
                "duration_ms": round((time.monotonic() - started) * 1000),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        raise StageFailure(name) from error
    record["stages"].append(
        {
            "name": name,
            "status": "passed",
            "duration_ms": round((time.monotonic() - started) * 1000),
            "details": details,
        }
    )
    return details


def build_record() -> dict[str, Any]:
    """Create the result skeleton, including non-secret provenance only."""
    return {
        "schema": SCHEMA,
        "status": "failed",
        "failed_stage": None,
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "ci": {
            key: os.environ[key]
            for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "RUNNER_OS", "RUNNER_ARCH")
            if key in os.environ
        },
        "versions": {},
        "stages": [],
    }


def write_result(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute real TF1-compatibility Slim and legacy LSTM graph sessions "
            "under TensorFlow 2.15."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="path for the compact redacted JSON result",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = build_record()

    try:
        run_stage(record, "environment", validate_environment)
        stack: dict[str, Any] = {}
        record["versions"] = run_stage(record, "import", lambda: import_stack(stack))
        tf, slim, np = stack["tf"], stack["slim"], stack["np"]
        run_stage(record, "slim_batchnorm", lambda: execute_slim_batch_norm(tf, slim, np))
        run_stage(record, "legacy_lstm", lambda: execute_legacy_lstm(tf, np))
    except StageFailure as failure:
        record["failed_stage"] = failure.stage
    else:
        record["status"] = "passed"

    write_result(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
