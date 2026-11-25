"""The experimental identity accounts for the worker and the host driver."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.oaf_tf215.compare import MODEL_CONFIG_PATH
from runtime.oaf_tf1.model import load_model_config
from src.benchmark import oaf_tf215_runtime as identity
from src.benchmark.backend_identity import (
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
    assert descriptor.sha256 == "938cbcccff5adda03f11344596e21534e178c98929098bf8327feb1e9ed4c890"
    assert normalize_known_backend_descriptor(baseline.payload) == baseline.payload
    assert normalize_known_backend_descriptor(descriptor.payload) == descriptor.payload


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
        shim,
        worker_environment={**identity.tf215_worker_environment(), "OPENBLAS_NUM_THREADS": "3"},
    )
    assert result["status"] == "blocked"


def test_x86_probe_collects_cpu_and_package_identity(tmp_path, monkeypatch):
    cpuinfo = (
        "processor : 0\n"
        "vendor_id : GenuineIntel\n"
        "model name : Test CPU\n"
        "flags : sse2 avx\n\n"
        "processor : 1\n"
        "vendor_id : GenuineIntel\n"
        "model name : Test CPU\n"
        "flags : avx sse2 avx2\n"
    )
    read_text = Path.read_text

    def read_cpuinfo(path, *args, **kwargs):
        if path == Path("/proc/cpuinfo"):
            return cpuinfo
        return read_text(path, *args, **kwargs)

    for name, version in (("tensorflow-cpu", "2.15.1"), ("numpy", "1.26.4")):
        metadata = tmp_path / f"{name.replace('-', '_')}-{version}.dist-info" / "METADATA"
        metadata.parent.mkdir()
        metadata.write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
    distributions = identity.importlib.metadata.distributions
    monkeypatch.setattr(Path, "read_text", read_cpuinfo)
    monkeypatch.setattr(
        identity.importlib.metadata, "distributions", lambda: distributions(path=[str(tmp_path)])
    )
    monkeypatch.setattr(identity.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(identity.platform, "system", lambda: "Linux")
    monkeypatch.setattr(identity.sys, "version_info", (3, 11, 9))
    monkeypatch.setattr(identity.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(identity.platform, "libc_ver", lambda: ("glibc", "2.31"))
    monkeypatch.setattr(identity.os, "sched_getaffinity", lambda _: {0, 1})
    probe = identity.probe_runtime()
    assert probe["status"] == "ok"
    assert probe["host"]["cpu_count"] == 8
    assert probe["host"]["affinity_cpu_count"] == 2
    assert probe["host"]["cpu_vendor"] == "GenuineIntel"
    assert probe["host"]["cpu_model"] == "Test CPU"
    assert probe["host"]["cpu_flags"] == ["avx", "avx2", "sse2"]
    assert probe["distributions"] == [
        {"name": "numpy", "version": "1.26.4"},
        {"name": "tensorflow-cpu", "version": "2.15.1"},
    ]
    monkeypatch.setattr(identity.os, "sched_getaffinity", lambda _: set())
    assert "unavailable_affinity" in identity.probe_runtime()["reasons"]


def test_probe_command_reports_unavailable_runtime_facts(tmp_path, monkeypatch, capsys):
    def unavailable(*_args):
        raise OSError("unavailable")

    monkeypatch.setattr(identity.platform, "system", lambda: "Linux")
    monkeypatch.setattr(identity.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(identity.platform, "libc_ver", lambda: ("", ""))
    monkeypatch.setattr(identity.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(identity.sys, "executable", str(tmp_path / "missing-python"))
    monkeypatch.setattr(identity.os, "cpu_count", lambda: None)
    monkeypatch.setattr(identity.os, "sched_getaffinity", unavailable)
    monkeypatch.setattr(identity, "_cpu_fields", unavailable)
    monkeypatch.setattr(identity, "_version_inventory", unavailable)

    assert identity.main(["--probe"]) == 2
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "blocked"
    assert set(record["reasons"]) == {
        "unsupported_interpreter",
        "unavailable_libc",
        "unavailable_cpu_count",
        "unavailable_affinity",
        "unavailable_cpu_fields",
        "unavailable_interpreter_digest",
        "unavailable_distribution_inventory",
    }
    assert record["host"]["affinity_cpu_count"] is None
    assert record["python"]["sha256"] is None
    assert record["distributions"] == []


def test_collect_probe_reports_missing_interpreter(tmp_path):
    record = identity.collect_probe(tmp_path / "missing-python", worker_environment={})

    assert record == {
        "schema": "crux.oaf-tf215-probe/v1",
        "status": "blocked",
        "reasons": ["interpreter_start_unavailable"],
    }


@pytest.mark.parametrize(
    ("returncode", "stdout", "message"),
    [
        (0, "unexpected startup log", "did not produce a JSON diagnostic"),
        (0, '{"schema":"foreign","status":"ok"}', "invalid diagnostic"),
        (0, '{"schema":"crux.oaf-tf215-probe/v1","status":{}}', "invalid diagnostic"),
        (0, '{"schema":"crux.oaf-tf215-probe/v1","status":"blocked"}', "invalid diagnostic"),
    ],
)
def test_collect_probe_rejects_invalid_diagnostic(monkeypatch, returncode, stdout, message):
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(command, returncode, stdout=stdout)

    monkeypatch.setattr(identity.subprocess, "run", run)

    with pytest.raises(ValueError, match=message):
        identity.collect_probe(Path(sys.executable), worker_environment={})


def _runtime_tree(root: Path) -> Path:
    relative_files = (
        "src/cli/benchmark.py",
        "experiments/oaf_tf215/requirements.lock",
        "experiments/oaf_tf215/patches/tf215-compat.patch",
        "runtime/oaf_tf1/patches/capture-emitted-frame.patch",
        "runtime/oaf_tf1/model.json",
        "scratch/worker.py",
        "scratch/model.json",
        "upstream/magenta/music/sequences_lib.py",
    )
    for relative in relative_files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(
                MODEL_CONFIG_PATH.read_bytes() if path.name == "model.json" else b"fixture"
            )
    return root / "scratch"


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    from src.benchmark import checkpoint_acquisition

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "repo"
    scratch = _runtime_tree(root)
    python = tmp_path / "venv/python"
    python.parent.mkdir()
    python.symlink_to(sys.executable)
    environment = identity.tf215_worker_environment()
    probe = {
        "schema": identity.PROBE_SCHEMA,
        "status": "ok",
        "host": {
            "system": "Linux",
            "machine": "x86_64",
            "release": "6.1.0",
            "libc_name": "glibc",
            "libc_version": "2.31",
            "cpu_count": 2,
            "affinity_cpu_count": 2,
            "cpu_model": "example",
            "cpu_vendor": "GenuineIntel",
            "cpu_flags": ["sse2"],
        },
        "python": {"sha256": "2" * 64, "version": "3.11.9", "implementation": "CPython"},
        "worker_environment": {key: environment.get(key) for key in identity.ENV_KEYS},
        "distributions": [{"name": "tensorflow-cpu", "version": "2.15.1"}],
    }
    seen = []

    def run(command, **kwargs):
        assert command == [str(python), "-s", str(Path(identity.__file__).resolve()), "--probe"]
        seen.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(probe))

    def verify(config, cache_root, *, download):
        assert config == load_model_config(MODEL_CONFIG_PATH)
        assert not download
        return cache_root / "sha256" / config.checkpoint.archive_sha256

    monkeypatch.setattr(identity.subprocess, "run", run)
    monkeypatch.setattr(checkpoint_acquisition, "prepare_oaf_checkpoint", verify)

    def prepare():
        return identity.prepare_tf215_runtime(
            root,
            python311=python.relative_to(tmp_path),
            runtime_root=scratch.relative_to(tmp_path),
            checkpoint_cache=Path("cache"),
        )

    return root, scratch, python, probe, seen, prepare


def test_candidate_preparation_fingerprints_worker_and_invokes_venv_path(preparation, monkeypatch):
    root, scratch, python, probe, seen, prepare = preparation
    monkeypatch.setenv("PYTHONPATH", "/unrelated/imports")
    first = prepare()
    assert first.python311 == python
    assert first.runtime_root == scratch
    assert first.checkpoint_dir.is_absolute()
    assert seen == [dict(first.worker_environment)]
    assert "PYTHONPATH" not in first.worker_environment
    assert first.worker_environment["TF_ENABLE_ONEDNN_OPTS"] == "0"
    assert first.sha256 == hashlib.sha256(canonical_json_bytes(first.identity)).hexdigest()

    (root / "src/cli/benchmark.py").write_text("changed host driver")
    assert prepare().sha256 == first.sha256
    probe["host"]["cpu_count"] = 4
    assert prepare().sha256 != first.sha256
    probe["host"]["cpu_count"] = 2
    (scratch / "worker.py").write_text("changed worker")
    assert prepare().sha256 != first.sha256


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("host", "runtime probe host identity is invalid"),
        ("python", "runtime probe interpreter identity is invalid"),
        ("worker_environment", "runtime probe environment differs"),
    ],
)
def test_candidate_preparation_rejects_empty_success_fields(preparation, field, message):
    _, _, _, probe, _, prepare = preparation
    probe[field].clear()
    with pytest.raises(ValueError, match=message):
        prepare()


def test_candidate_preparation_rejects_environment_mismatch(preparation):
    _, _, _, probe, _, prepare = preparation
    probe["worker_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    with pytest.raises(ValueError, match="runtime probe environment differs"):
        prepare()


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("blocked", "unsupported_platform"),
        ("timeout", "runtime probe timed out"),
        ("malformed", "runtime probe package inventory is invalid"),
        ("wrong_tensorflow", "tensorflow-cpu==2.15.1"),
        ("changed_config", "scratch model configuration differs"),
        ("missing_worker", "scratch worker is unavailable"),
        ("symlink_worker", "symlink"),
        ("symlink_scratch", "scratch root is unavailable or is a symlink"),
    ],
)
def test_candidate_preparation_rejects_unavailable_runtime(
    preparation, monkeypatch, problem, message
):
    _, scratch, _, probe, _, prepare = preparation
    if problem == "blocked":
        monkeypatch.setattr(
            identity.subprocess,
            "run",
            lambda command, **_: subprocess.CompletedProcess(
                command,
                2,
                stdout=json.dumps(
                    {
                        "schema": identity.PROBE_SCHEMA,
                        "status": "blocked",
                        "reasons": ["unsupported_platform"],
                    }
                ),
            ),
        )
    elif problem == "timeout":

        def timeout(command, **_):
            raise subprocess.TimeoutExpired(command, 60)

        monkeypatch.setattr(identity.subprocess, "run", timeout)
    elif problem == "malformed":
        probe["distributions"] = [{}]
    elif problem == "wrong_tensorflow":
        probe["distributions"][0]["version"] = "2.16.0"
    elif problem == "changed_config":
        (scratch / "model.json").write_text("changed model configuration")
    elif problem in {"missing_worker", "symlink_worker"}:
        worker = scratch / "worker.py"
        worker.unlink()
        if problem == "symlink_worker":
            worker.symlink_to(MODEL_CONFIG_PATH)
    elif problem == "symlink_scratch":
        target = scratch.rename(scratch.with_name("scratch-target"))
        scratch.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match=message):
        prepare()


def test_identity_hashes_actual_source_and_worker_environment(tmp_path):
    def fingerprint(root, *, worker_setting=None, core_count=2, tensorflow_version="2.15.1"):
        runtime = _runtime_tree(root)
        probe = {
            "schema": "crux.oaf-tf215-probe/v1",
            "status": "ok",
            "python": {"sha256": "2" * 64, "version": "3.11.9", "implementation": "CPython"},
            "distributions": [{"name": "tensorflow-cpu", "version": tensorflow_version}],
            "host": {"cpu_count": core_count, "affinity_cpu_count": 2},
            "worker_environment": {"OPENBLAS_NUM_THREADS": worker_setting},
        }
        value = identity.build_runtime_identity(
            root, runtime, probe, driver_files=[Path("src/cli/benchmark.py")]
        )
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
    (second / "src/cli/benchmark.py").write_bytes(b"changed host driver")
    assert fingerprint(second) != initial
    (second / "scratch/model.json").write_bytes(b"changed worker configuration")
    with pytest.raises(ValueError, match="scratch model configuration"):
        fingerprint(second)
    (first / "scratch/link.py").symlink_to(second / "src/cli/benchmark.py")
    with pytest.raises(ValueError, match="symlink"):
        fingerprint(first)
