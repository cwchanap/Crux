"""Prepare and fingerprint the isolated TF2.15 OaF worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runtime.oaf_tf1.model import OafModelConfig
    from src.benchmark.backend_identity import BackendDescriptor, JsonValue

PROBE_SCHEMA = "crux.oaf-tf215-probe/v1"
RUNTIME_SCHEMA = "crux.oaf-tf215-runtime/v1"
EXPERIMENT = Path("experiments/oaf_tf215")
FROZEN = Path("runtime/oaf_tf1")
PRISTINE_CONVERTER = Path("upstream/magenta/music/sequences_lib.py")
WORKER_ENVIRONMENT = MappingProxyType(
    {
        "CUDA_VISIBLE_DEVICES": "-1",
        "PYTHONHASHSEED": "0",
        "TF_ENABLE_ONEDNN_OPTS": "0",
    }
)
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES PYTHONHASHSEED TF_ENABLE_ONEDNN_OPTS NUMBA_CACHE_DIR "
    "OMP_NUM_THREADS OMP_THREAD_LIMIT OMP_DYNAMIC OMP_PROC_BIND OMP_PLACES "
    "KMP_AFFINITY KMP_BLOCKTIME MKL_NUM_THREADS MKL_DYNAMIC MKL_CBWR "
    "OPENBLAS_NUM_THREADS BLIS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS "
    "TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS TF_XLA_FLAGS XLA_FLAGS "
    "TF_DETERMINISTIC_OPS TF_CUDNN_DETERMINISTIC TF_USE_LEGACY_KERAS "
    "ONEDNN_MAX_CPU_ISA DNNL_MAX_CPU_ISA LD_PRELOAD LD_LIBRARY_PATH"
).split()


@dataclass(frozen=True)
class OafTf215Runtime:
    python311: Path
    runtime_root: Path
    checkpoint_dir: Path
    worker_environment: Mapping[str, str]
    descriptor: BackendDescriptor
    identity: dict[str, JsonValue]
    sha256: str


def build_tf215_worker_environment() -> dict[str, str]:
    """Build the worker environment with a stable owner-only Numba cache."""
    from src.benchmark.backends.idm import _isolated_worker_environment

    if not hasattr(os, "geteuid"):
        raise ValueError("TF2.15 worker cache requires POSIX user identity")
    from pwd import getpwuid

    uid = os.geteuid()
    try:
        home = Path(getpwuid(uid).pw_dir).resolve(strict=True)
    except KeyError as error:
        raise ValueError("TF2.15 worker cache requires a local account home") from error
    home_metadata = home.lstat()
    if (
        not stat.S_ISDIR(home_metadata.st_mode)
        or home_metadata.st_uid != uid
        or home_metadata.st_mode & 0o022
    ):
        raise ValueError(f"expected a private Numba cache parent owned by the current user: {home}")
    cache = home / ".crux-oaf-tf215-numba"
    try:
        cache.mkdir(mode=0o700)
    except FileExistsError:
        pass
    metadata = cache.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != uid
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError(f"expected a private Numba cache owned by the current user: {cache}")
    return {**_isolated_worker_environment(), **WORKER_ENVIRONMENT, "NUMBA_CACHE_DIR": str(cache)}


tf215_worker_environment = build_tf215_worker_environment


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cpu_fields() -> dict[str, Any]:
    fields: dict[str, Any] = {}
    flags: set[str] = set()
    names = {
        "vendor_id": "cpu_vendor",
        "model name": "cpu_model",
        "CPU implementer": "cpu_implementer",
        "CPU part": "cpu_part",
        "CPU revision": "cpu_revision",
    }
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition(":")
        name, value = name.strip(), value.strip()
        if name in {"flags", "Features"}:
            flags.update(value.split())
        elif value and name in names:
            fields.setdefault(names[name], value)
    fields["cpu_flags"] = sorted(flags)
    return fields


def _version_inventory() -> list[dict[str, str]]:
    packages = sorted(
        (package.metadata["Name"], package.version)
        for package in importlib.metadata.distributions()
    )
    return [{"name": name, "version": version} for name, version in packages]


def probe_runtime() -> dict[str, Any]:
    """Collect worker facts without importing Crux or TensorFlow."""
    reasons: list[str] = []
    system, machine = platform.system(), platform.machine()
    if system != "Linux" or machine != "x86_64":
        reasons.append("unsupported_platform")
    if sys.version_info[:2] != (3, 11):
        reasons.append("unsupported_interpreter")
    libc_name, libc_version = platform.libc_ver()
    if not libc_name or not libc_version:
        reasons.append("unavailable_libc")
    cpu_count = os.cpu_count()
    if not cpu_count:
        reasons.append("unavailable_cpu_count")
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = None
    if not affinity_count:
        reasons.append("unavailable_affinity")
    try:
        cpu_fields = _cpu_fields()
    except (OSError, UnicodeError):
        cpu_fields = {}
    if not cpu_fields.get("cpu_flags") or (
        machine == "x86_64" and not all(cpu_fields.get(k) for k in ("cpu_model", "cpu_vendor"))
    ):
        reasons.append("unavailable_cpu_fields")
    try:
        interpreter_sha = _file_sha256(Path(sys.executable).resolve(strict=True))
    except OSError:
        interpreter_sha = None
        reasons.append("unavailable_interpreter_digest")
    try:
        distributions = _version_inventory()
    except (OSError, ValueError, TypeError):
        distributions = []
        reasons.append("unavailable_distribution_inventory")
    return {
        "schema": PROBE_SCHEMA,
        "status": "blocked" if reasons else "ok",
        "reasons": reasons,
        "host": {
            "system": system,
            "machine": machine,
            "release": platform.release(),
            "libc_name": libc_name,
            "libc_version": libc_version,
            "cpu_count": cpu_count,
            "affinity_cpu_count": affinity_count,
            **cpu_fields,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "sha256": interpreter_sha,
        },
        "distributions": distributions,
        "worker_environment": {key: os.environ.get(key) for key in ENV_KEYS},
    }


def _require_probe_identity(record: Mapping[str, Any], environment: Mapping[str, str]) -> None:
    host = record.get("host")
    if (
        not isinstance(host, dict)
        or host.get("system") != "Linux"
        or host.get("machine") != "x86_64"
        or not all(
            isinstance(host.get(field), str) and host[field]
            for field in ("release", "libc_name", "libc_version", "cpu_model", "cpu_vendor")
        )
        or not all(
            type(host.get(field)) is int and host[field] > 0
            for field in ("cpu_count", "affinity_cpu_count")
        )
        or not isinstance(host.get("cpu_flags"), list)
        or not host["cpu_flags"]
        or not all(isinstance(flag, str) and flag for flag in host["cpu_flags"])
    ):
        raise ValueError("runtime probe host identity is invalid")
    python = record.get("python")
    if (
        not isinstance(python, dict)
        or not isinstance(python.get("implementation"), str)
        or not python["implementation"]
        or not isinstance(python.get("version"), str)
        or not re.fullmatch(r"3\.11\.\d+.*", python["version"])
        or not isinstance(python.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", python["sha256"])
    ):
        raise ValueError("runtime probe interpreter identity is invalid")
    if record.get("worker_environment") != {key: environment.get(key) for key in ENV_KEYS}:
        raise ValueError("runtime probe environment differs from the worker launch environment")


def collect_probe(python311: Path, *, worker_environment: Mapping[str, str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [os.fspath(python311.absolute()), "-s", os.fspath(Path(__file__).resolve()), "--probe"],
            capture_output=True,
            text=True,
            env=dict(worker_environment),
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("runtime probe timed out") from error
    except OSError:
        return {
            "schema": PROBE_SCHEMA,
            "status": "blocked",
            "reasons": ["interpreter_start_unavailable"],
        }
    try:
        record = json.loads(completed.stdout)
    except ValueError as error:
        raise ValueError("runtime probe did not produce a JSON diagnostic") from error
    if (
        not isinstance(record, dict)
        or record.get("schema") != PROBE_SCHEMA
        or not isinstance(record.get("status"), str)
        or completed.returncode != {"ok": 0, "blocked": 2}.get(record.get("status"))
    ):
        raise ValueError("runtime probe returned an invalid diagnostic")
    if record["status"] == "ok":
        _require_probe_identity(record, worker_environment)
    return record


def build_candidate_descriptor(config: OafModelConfig) -> BackendDescriptor:
    from src.benchmark.backend_identity import (
        OAF_DESCRIPTOR_KEYS,
        OAF_DESCRIPTOR_SCHEMA,
        OAF_TF215_BACKEND_ID,
        build_descriptor,
    )
    from src.benchmark.oaf_corpus_run import _expected_oaf_descriptor

    payload = {**_expected_oaf_descriptor(config).payload, "backend_id": OAF_TF215_BACKEND_ID}
    return build_descriptor(payload, OAF_DESCRIPTOR_KEYS, OAF_DESCRIPTOR_SCHEMA)


def _source_digest(root: Path, relative: Path) -> str:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"invalid source path: {relative}")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"source path is a symlink: {relative}")
    if not path.is_file():
        raise ValueError(f"source file is unavailable: {relative}")
    return _file_sha256(path)


def _scratch_inventory(root: Path) -> dict[str, str]:
    def entries(directory: Path):
        for path in sorted(directory.iterdir()):
            if path.is_symlink():
                raise ValueError(f"scratch path is a symlink: {path.relative_to(root)}")
            if path.is_dir():
                if path.name != "__pycache__" and path != root / "model-scratch":
                    yield from entries(path)
            elif path.is_file():
                yield path.relative_to(root).as_posix(), _file_sha256(path)
            else:
                raise ValueError(f"unsupported scratch entry: {path.relative_to(root)}")

    if root.is_symlink() or not root.is_dir():
        raise ValueError("scratch root is unavailable or is a symlink")
    return dict(entries(root))


def build_runtime_identity(
    repo_root: Path,
    runtime_root: Path,
    probe: Mapping[str, Any],
    *,
    driver_files: Sequence[Path],
) -> dict[str, Any]:
    from runtime.oaf_tf1.model import load_model_config

    mappings = ("host", "python", "worker_environment")
    if probe.get("schema") != PROBE_SCHEMA:
        raise ValueError("runtime probe schema is invalid")
    if probe.get("status") == "blocked":
        reasons = probe.get("reasons")
        if (
            isinstance(reasons, list)
            and reasons
            and all(isinstance(reason, str) for reason in reasons)
        ):
            raise ValueError(f"runtime probe is blocked: {', '.join(reasons)[:512]}")
        raise ValueError("runtime probe is blocked")
    if probe.get("status") != "ok" or not all(isinstance(probe.get(key), dict) for key in mappings):
        raise ValueError("a complete successful runtime probe is required")
    packages = probe.get("distributions")
    if not isinstance(packages, list) or not all(
        isinstance(package, dict)
        and all(isinstance(package.get(key), str) and package[key] for key in ("name", "version"))
        for package in packages
    ):
        raise ValueError("runtime probe package inventory is invalid")
    tensorflow_versions = [
        package["version"]
        for package in packages
        if package["name"].lower().replace("_", "-") == "tensorflow-cpu"
    ]
    if tensorflow_versions != ["2.15.1"]:
        raise ValueError("runtime requires exactly one tensorflow-cpu==2.15.1 distribution")
    config_path = FROZEN / "model.json"
    config_sha = _source_digest(repo_root, config_path)
    scratch = _scratch_inventory(runtime_root)
    if "worker.py" not in scratch:
        raise ValueError("scratch worker is unavailable")
    if scratch.get("model.json") != config_sha:
        raise ValueError("scratch model configuration differs from frozen model.json")
    return {
        "schema": RUNTIME_SCHEMA,
        **{field: probe[field] for field in mappings},
        "distributions": packages,
        "requirements_lock_sha256": _source_digest(repo_root, EXPERIMENT / "requirements.lock"),
        "model_config_sha256": config_sha,
        "checkpoint_archive_sha256": load_model_config(
            repo_root / config_path
        ).checkpoint.archive_sha256,
        "instrumentation_patch_sha256": _source_digest(
            repo_root, FROZEN / "patches/capture-emitted-frame.patch"
        ),
        "compatibility_patch_sha256": _source_digest(
            repo_root, EXPERIMENT / "patches/tf215-compat.patch"
        ),
        "pristine_converter_sha256": _source_digest(runtime_root.parent, PRISTINE_CONVERTER),
        "scratch_files": scratch,
        "driver_files": {
            path.as_posix(): _source_digest(repo_root, path) for path in sorted(set(driver_files))
        },
    }


def prepare_tf215_runtime(
    repo_root: Path,
    *,
    python311: Path,
    runtime_root: Path,
    checkpoint_cache: Path,
) -> OafTf215Runtime:
    """Verify local runtime inputs and capture the environment used by the worker."""
    from runtime.oaf_tf1.model import load_model_config
    from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex
    from src.benchmark.checkpoint_acquisition import prepare_oaf_checkpoint

    # The executable symlink is part of virtualenv discovery. Resolve its target
    # only inside the probe when hashing the interpreter binary.
    python311 = python311.absolute()
    runtime_root = runtime_root.absolute()
    environment = build_tf215_worker_environment()
    probe = collect_probe(python311, worker_environment=environment)
    identity = build_runtime_identity(repo_root, runtime_root, probe, driver_files=())
    config = load_model_config(repo_root / FROZEN / "model.json")
    checkpoint_dir = prepare_oaf_checkpoint(config, checkpoint_cache.absolute(), download=False)
    return OafTf215Runtime(
        python311=python311,
        runtime_root=runtime_root,
        checkpoint_dir=checkpoint_dir,
        worker_environment=MappingProxyType(environment),
        descriptor=build_candidate_descriptor(config),
        identity=identity,
        sha256=sha256_hex(canonical_json_bytes(identity)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", required=True)
    parser.parse_args(argv)
    result = probe_runtime()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
