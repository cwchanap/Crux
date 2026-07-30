"""CPython entrypoint for the frozen OaF runner.

Only standard-library modules are imported before the process environment is
validated. Numeric and vendored imports occur inside ``main`` afterward.
"""

# Numeric imports must remain below the exact environment check.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import os
import sys

EXPECTED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
}


def discard_interpreter_bootstrap_environment():
    """Remove the sole CPython bootstrap control before exact validation."""
    if os.environ.pop("PYTHONCOERCECLOCALE", None) != "0":
        os.write(2, b"code=process_environment_invalid count=1\n")
        raise SystemExit(2)


def validate_process_environment(expected_environment):
    """Exit before runner imports when any locked process value differs."""
    if dict(os.environ) != dict(expected_environment):
        os.write(2, b"code=process_environment_invalid count=1\n")
        raise SystemExit(2)


def _protocol_failure_type():
    try:
        from protocol import ProtocolFailure
    except ImportError:
        from runtime.oaf_tf1.protocol import ProtocolFailure
    return ProtocolFailure


def _authenticate_runtime_environment() -> None:
    try:
        from oaf_backend import authenticate_runtime_environment
    except ImportError:
        from runtime.oaf_tf1.oaf_backend import authenticate_runtime_environment
    authenticate_runtime_environment()


def _import_numeric_modules():
    import random

    import numpy as np
    import tensorflow.compat.v1 as tf

    try:
        from oaf_backend import FrozenOafBackend, authenticate_startup
    except ImportError:
        from runtime.oaf_tf1.oaf_backend import FrozenOafBackend, authenticate_startup
    try:
        from protocol import canonical_json_line, serve_requests
    except ImportError:
        from runtime.oaf_tf1.protocol import canonical_json_line, serve_requests
    return (
        random,
        np,
        tf,
        FrozenOafBackend,
        authenticate_startup,
        canonical_json_line,
        serve_requests,
    )


def main() -> int:
    discard_interpreter_bootstrap_environment()
    validate_process_environment(EXPECTED_ENVIRONMENT)
    vendor_root = "/opt/crux/vendor"
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)

    try:
        protocol_failure = _protocol_failure_type()
        _authenticate_runtime_environment()
    except protocol_failure as error:
        os.write(2, ("code=" + error.code + " count=1\n").encode("ascii", errors="strict"))
        return 2
    except BaseException:  # pylint: disable=broad-exception-caught
        os.write(2, b"code=runner_dependency_import_failed count=1\n")
        return 2

    try:
        (
            random,
            np,
            tf,
            FrozenOafBackend,
            authenticate_startup,
            canonical_json_line,
            serve_requests,
        ) = _import_numeric_modules()
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)
        startup = authenticate_startup()
        backend = FrozenOafBackend.from_startup(startup)
        sys.stdout.buffer.write(
            canonical_json_line(startup.ready_payload, startup.stdout_max_line_bytes)
        )
        sys.stdout.buffer.flush()
        serve_requests(
            stdin=sys.stdin.buffer,
            stdout=sys.stdout.buffer,
            backend=backend,
            input_root=startup.input_root,
            descriptor_sha256=startup.descriptor_sha256,
            max_input_audio_frames=startup.max_input_audio_frames,
            stdout_max_line_bytes=startup.stdout_max_line_bytes,
        )
        return 0
    except protocol_failure as error:
        os.write(2, ("code=" + error.code + " count=1\n").encode("ascii", errors="strict"))
        return 2
    except BaseException:  # pylint: disable=broad-exception-caught
        os.write(2, b"code=runner_internal_failure count=1\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
