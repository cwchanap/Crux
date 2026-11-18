"""The experimental identity accounts for the worker and the host driver."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.oaf_tf215 import runtime_identity as identity
from experiments.oaf_tf215.compare import MODEL_CONFIG_PATH, _worker_environment
from runtime.oaf_tf1.model import load_model_config
from src.benchmark.backend_identity import (
    StrictJsonError,
    canonical_json_bytes,
    normalize_known_backend_descriptor,
)
from src.benchmark.oaf_corpus_run import _expected_oaf_descriptor


def test_candidate_changes_only_backend_identity():
    config = load_model_config(MODEL_CONFIG_PATH)
    baseline = _expected_oaf_descriptor(config)
    descriptor = identity.build_candidate_descriptor(config)
    assert descriptor.payload == {
        **baseline.payload,
        "backend_id": "magenta-egmd-tf215-94529798-8hit-experimental-v1",
    }
    assert descriptor.sha256 != baseline.sha256
    assert normalize_known_backend_descriptor(baseline.payload) == baseline.payload
    with pytest.raises(StrictJsonError, match="unknown"):
        normalize_known_backend_descriptor(descriptor.payload)


def test_arm_probe_describes_platform_without_importing_project_code():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            r"""
import importlib.abc, platform, runpy, sys
from pathlib import Path

class CheckImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in sys.stdlib_module_names:
            raise AssertionError(fullname)

sys.meta_path.insert(0, CheckImports())
platform.machine = lambda: 'aarch64'
platform.system = lambda: 'Linux'
read_text = Path.read_text
Path.read_text = lambda self, *a, **k: (
    'processor: 0\nFeatures: fp asimd\nCPU implementer: 0x41\nCPU part: 0xd0c\n'
    if str(self) == '/proc/cpuinfo' else read_text(self, *a, **k)
)
sys.argv = [sys.argv[1], '--probe']
runpy.run_path(sys.argv[0], run_name='__main__')
""",
            os.fspath(Path(identity.__file__).resolve()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    record = json.loads(result.stdout)
    assert record["status"] == "blocked"
    assert "unsupported_platform" in record["reasons"]
    assert record["host"]["cpu_flags"] == ["asimd", "fp"]
    assert record["host"]["cpu_implementer"] == "0x41"


def test_probe_launch_uses_worker_arguments_and_environment(tmp_path):
    shim = tmp_path / "python"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "assert sys.argv[1] == '-s' and sys.argv[-1] == '--probe'\n"
        "assert os.environ['OPENBLAS_NUM_THREADS'] == '3'\n"
        "assert os.environ['TF_ENABLE_ONEDNN_OPTS'] == '0'\n"
        "assert 'PYTHONPATH' not in os.environ\n"
        "print(json.dumps({'schema': 'crux.oaf-tf215-probe/v1', 'status': 'blocked'}))\n"
        "sys.exit(2)\n"
    )
    shim.chmod(0o755)
    result = identity.collect_probe(
        shim, worker_environment={**_worker_environment(), "OPENBLAS_NUM_THREADS": "3"}
    )
    assert result["status"] == "blocked"


def test_x86_probe_requires_affinity(monkeypatch):
    monkeypatch.setattr(identity.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(identity.platform, "system", lambda: "Linux")
    monkeypatch.setattr(identity.sys, "version_info", (3, 11, 9))
    monkeypatch.setattr(identity.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        identity,
        "_cpu_fields",
        lambda: {
            "cpu_vendor": "GenuineIntel",
            "cpu_model": "example",
            "cpu_flags": ["sse2"],
        },
    )
    monkeypatch.setattr(identity.platform, "libc_ver", lambda: ("glibc", "2.31"))
    monkeypatch.setattr(identity.os, "sched_getaffinity", lambda _: {0, 1})
    monkeypatch.setattr(identity, "_version_inventory", lambda: [])
    probe = identity.probe_runtime()
    assert probe["status"] == "ok"
    assert probe["host"]["cpu_count"] == 8
    assert probe["host"]["affinity_cpu_count"] == 2
    monkeypatch.setattr(identity.os, "sched_getaffinity", lambda _: set())
    assert "unavailable_affinity" in identity.probe_runtime()["reasons"]


def test_identity_hashes_actual_source_and_worker_environment(tmp_path):
    def fingerprint(root, *, worker_setting=None, core_count=2, tensorflow_version="2.15.1"):
        runtime = root / "scratch"
        driver = Path("src/benchmark/scoring.py")
        fixture_files = [
            driver,
            Path("experiments/oaf_tf215/requirements.lock"),
            Path("experiments/oaf_tf215/patches/tf215-compat.patch"),
            Path("runtime/oaf_tf1/patches/capture-emitted-frame.patch"),
            Path("runtime/oaf_tf1/model.json"),
            Path("scratch/worker.py"),
            Path("scratch/model.json"),
            Path("upstream/magenta/music/sequences_lib.py"),
        ]
        for relative in fixture_files:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(
                    MODEL_CONFIG_PATH.read_bytes() if path.name == "model.json" else b"fixture"
                )
        probe = {
            "schema": "crux.oaf-tf215-probe/v1",
            "status": "ok",
            "python": {"sha256": "2" * 64, "version": "3.11.9", "implementation": "CPython"},
            "distributions": [{"name": "tensorflow-cpu", "version": tensorflow_version}],
            "host": {"cpu_count": core_count, "affinity_cpu_count": 2},
            "worker_environment": {"OPENBLAS_NUM_THREADS": worker_setting},
        }
        value = identity.build_runtime_identity(root, runtime, probe, driver_files=[driver])
        return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

    first = tmp_path / "one"
    second = tmp_path / "two"
    initial = fingerprint(first)
    assert fingerprint(second) == initial
    assert fingerprint(second, worker_setting="") != initial
    assert fingerprint(second, core_count=4) != initial
    with pytest.raises(ValueError, match="tensorflow-cpu==2.15.1"):
        fingerprint(second, tensorflow_version="2.16.0")
    (second / "scratch/model-scratch").mkdir()
    (second / "scratch/model-scratch/estimator-log").write_text("worker-generated state")
    assert fingerprint(second) == initial
    converter = second / "upstream/magenta/music/sequences_lib.py"
    converter.write_bytes(b"changed pristine converter")
    assert fingerprint(second) != initial
    converter.write_bytes(b"fixture")
    (second / "src/benchmark/scoring.py").write_bytes(b"changed host scoring")
    assert fingerprint(second) != initial
    (second / "scratch/model.json").write_bytes(b"changed worker configuration")
    with pytest.raises(ValueError, match="scratch model configuration"):
        fingerprint(second)
    (first / "scratch/link.py").symlink_to(second / "src/benchmark/scoring.py")
    with pytest.raises(ValueError, match="symlink"):
        fingerprint(first)
