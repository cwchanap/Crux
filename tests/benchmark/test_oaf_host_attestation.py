from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.benchmark import backend_publication
from src.benchmark.backend_attestation import HostNumericFingerprint
from tools.hpa320 import oaf_host_attestation as host_module
from tools.hpa320.oaf_host_attestation import (
    HostAttestationError,
    load_native_host_attestation_bundle,
    publish_github_host_attestation,
)


def set_same_job_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: str = "bootstrap",
    runner_environment: str = "github-hosted",
) -> None:
    job = f"native-{phase}"
    workflow = f"hpa320-native-{phase}.yml"
    values = {
        "COMMIT_SHA": "a" * 40,
        "GITHUB_JOB": job,
        "GITHUB_REPOSITORY": "cwchanap/Crux",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_WORKFLOW_REF": (f"cwchanap/Crux/.github/workflows/{workflow}@refs/tags/native-v2"),
        "RUNNER_ARCH": "X64",
        "RUNNER_ARCH_CONTEXT": "X64",
        "RUNNER_ENVIRONMENT": runner_environment,
        "RUNNER_ENVIRONMENT_CONTEXT": runner_environment,
        "RUNNER_OS": "Linux",
        "RUNNER_OS_CONTEXT": "Linux",
        "WORKFLOW_SOURCE_SHA": "a" * 40,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def stub_native_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    def run_text(command: tuple[str, ...], _label: str) -> str:
        if command == ("git", "rev-parse", "HEAD"):
            return "a" * 40
        if command == ("uname", "-m"):
            return "x86_64"
        raise AssertionError(f"unexpected native command: {command!r}")

    docker_values = {
        "OSType": "linux",
        "Architecture": "x86_64",
        "ServerVersion": "28.3.2",
    }
    monkeypatch.setattr(host_module, "_run_text", run_text)
    monkeypatch.setattr(host_module, "_docker_info", docker_values.__getitem__)
    monkeypatch.setattr(
        host_module,
        "collect_host_numeric_fingerprint",
        lambda: HostNumericFingerprint(
            architecture="x86_64",
            cpu_vendor_id="GenuineIntel",
            cpu_family="6",
            cpu_model="143",
            cpu_stepping="8",
        ),
    )


def test_publish_github_host_attestation_binds_the_current_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_same_job_environment(monkeypatch)
    stub_native_commands(monkeypatch)
    output = tmp_path / "bootstrap-host-attestation"

    bundle = publish_github_host_attestation(phase="bootstrap", output_directory=output)

    assert {path.name for path in output.iterdir()} == {
        "attestation-bundle.json",
        "native-host-evidence.json",
        "native-host-observation.json",
    }
    assert bundle.observation["github_job"] == "native-bootstrap"
    assert bundle.observation["github_workflow_sha"] == "a" * 40
    assert bundle.observation["runner_environment"] == "github-hosted"
    assert bundle.evidence.payload["run_url"].endswith("/actions/runs/123456789")
    assert (
        load_native_host_attestation_bundle(
            output / "attestation-bundle.json",
            expected_phase="bootstrap",
        )
        == bundle
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RUNNER_ENVIRONMENT", "self-hosted"),
        ("RUNNER_ENVIRONMENT_CONTEXT", "self-hosted"),
        ("RUNNER_OS_CONTEXT", "Windows"),
        ("RUNNER_ARCH_CONTEXT", "ARM64"),
        ("GITHUB_JOB", "observe-native-host"),
        ("WORKFLOW_SOURCE_SHA", "b" * 40),
    ],
)
def test_publish_github_host_attestation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    set_same_job_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    stub_native_commands(monkeypatch)
    output = tmp_path / "bootstrap-host-attestation"

    with pytest.raises(HostAttestationError):
        publish_github_host_attestation(phase="bootstrap", output_directory=output)

    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    [
        "COMMIT_SHA",
        "GITHUB_JOB",
        "GITHUB_REPOSITORY",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_RUN_ID",
        "GITHUB_SERVER_URL",
        "GITHUB_SHA",
        "GITHUB_WORKFLOW_REF",
        "RUNNER_ARCH",
        "RUNNER_ARCH_CONTEXT",
        "RUNNER_ENVIRONMENT",
        "RUNNER_ENVIRONMENT_CONTEXT",
        "RUNNER_OS",
        "RUNNER_OS_CONTEXT",
        "WORKFLOW_SOURCE_SHA",
    ],
)
@pytest.mark.parametrize("missing", [True, False])
def test_publish_github_host_attestation_requires_every_environment_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    missing: bool,
) -> None:
    set_same_job_environment(monkeypatch)
    if missing:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, "")
    stub_native_commands(monkeypatch)

    with pytest.raises(HostAttestationError):
        publish_github_host_attestation(
            phase="bootstrap",
            output_directory=tmp_path / "bootstrap-host-attestation",
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RUNNER_ENVIRONMENT", "self-hosted"),
        ("RUNNER_ENVIRONMENT_CONTEXT", "self-hosted"),
        ("RUNNER_OS", "Windows"),
        ("RUNNER_OS_CONTEXT", "Windows"),
        ("RUNNER_ARCH", "ARM64"),
        ("RUNNER_ARCH_CONTEXT", "ARM64"),
    ],
)
def test_publish_github_host_attestation_rejects_runner_context_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    set_same_job_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    stub_native_commands(monkeypatch)

    with pytest.raises(HostAttestationError):
        publish_github_host_attestation(
            phase="bootstrap",
            output_directory=tmp_path / "bootstrap-host-attestation",
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GITHUB_REPOSITORY", "other/Crux"),
        (
            "GITHUB_WORKFLOW_REF",
            "cwchanap/Crux/.github/workflows/hpa320-native-measurement.yml@refs/tags/native-v2",
        ),
        ("GITHUB_RUN_ID", "0"),
        ("GITHUB_RUN_ATTEMPT", "0"),
        ("GITHUB_SERVER_URL", "https://github.example"),
        ("GITHUB_SHA", "b" * 40),
        ("COMMIT_SHA", "b" * 40),
    ],
)
def test_publish_github_host_attestation_rejects_wrong_execution_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    set_same_job_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    stub_native_commands(monkeypatch)

    with pytest.raises(HostAttestationError):
        publish_github_host_attestation(
            phase="bootstrap",
            output_directory=tmp_path / "bootstrap-host-attestation",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("OSType", "darwin"), ("Architecture", "arm64"), ("ServerVersion", "")],
)
def test_publish_github_host_attestation_rejects_wrong_docker_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    set_same_job_environment(monkeypatch)
    stub_native_commands(monkeypatch)
    original_docker_info = host_module._docker_info

    def docker_info(name: str) -> str:
        return value if name == field else original_docker_info(name)

    monkeypatch.setattr(host_module, "_docker_info", docker_info)

    with pytest.raises(HostAttestationError):
        publish_github_host_attestation(
            phase="bootstrap",
            output_directory=tmp_path / "bootstrap-host-attestation",
        )


def test_publish_github_host_attestation_never_replaces_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_same_job_environment(monkeypatch)
    stub_native_commands(monkeypatch)
    output = tmp_path / "bootstrap-host-attestation"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(HostAttestationError):
        publish_github_host_attestation(phase="bootstrap", output_directory=output)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert {path.name for path in output.iterdir()} == {"sentinel"}


def test_publish_github_host_attestation_cleans_unpublished_staging_on_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_same_job_environment(monkeypatch)
    stub_native_commands(monkeypatch)
    output = tmp_path / "bootstrap-host-attestation"
    original_write = host_module._write_new_regular_file
    writes = 0

    def fail_second_write(path: Path, content: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected write failure")
        original_write(path, content)

    monkeypatch.setattr(host_module, "_write_new_regular_file", fail_second_write)

    with pytest.raises(HostAttestationError):
        publish_github_host_attestation(phase="bootstrap", output_directory=output)

    assert not output.exists()
    assert not list(tmp_path.glob(".bootstrap-host-attestation.*"))


def test_publish_github_host_attestation_rolls_back_after_post_rename_sync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_same_job_environment(monkeypatch)
    stub_native_commands(monkeypatch)
    output = tmp_path / "bootstrap-host-attestation"
    real_rename = backend_publication._rename_no_replace_syscall
    real_fsync = os.fsync

    def rename_then_fail_sync(*args: object, **kwargs: object) -> None:
        real_rename(*args, **kwargs)  # type: ignore[arg-type]

        def fail_once(_descriptor: int) -> None:
            monkeypatch.setattr(backend_publication.os, "fsync", real_fsync)
            raise OSError("injected parent sync failure")

        monkeypatch.setattr(backend_publication.os, "fsync", fail_once)

    monkeypatch.setattr(backend_publication, "_rename_no_replace_syscall", rename_then_fail_sync)

    with pytest.raises(HostAttestationError):
        publish_github_host_attestation(phase="bootstrap", output_directory=output)

    assert not output.exists()


def test_host_attestation_loader_never_uses_an_unbounded_file_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_same_job_environment(monkeypatch)
    stub_native_commands(monkeypatch)
    output = tmp_path / "bootstrap-host-attestation"
    publish_github_host_attestation(phase="bootstrap", output_directory=output)
    real_reader = host_module.read_regular_file_no_follow

    def bounded_reader(path: Path, *, max_bytes: int | None = None) -> bytes:
        if max_bytes is None:
            raise OSError("unbounded test read")
        return real_reader(path, max_bytes=max_bytes)

    monkeypatch.setattr(host_module, "read_regular_file_no_follow", bounded_reader)

    bundle = load_native_host_attestation_bundle(
        output / "attestation-bundle.json",
        expected_phase="bootstrap",
    )

    assert bundle.phase == "bootstrap"


def test_host_attestation_cli_only_accepts_same_job_publication() -> None:
    parser = host_module._parser()

    arguments = parser.parse_args(["publish-github", "--phase", "bootstrap", "--output", "bundle"])

    assert arguments.command == "publish-github"
    assert arguments.phase == "bootstrap"
    assert arguments.output == Path("bundle")
    with pytest.raises(SystemExit):
        parser.parse_args(["observe-github", "--output", "bundle"])
