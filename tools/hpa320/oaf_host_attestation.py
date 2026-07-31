#!/usr/bin/env python3
"""Strictly publish same-job OaF native-host attestation bundles."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from src.benchmark.backend_attestation import parse_host_numeric_fingerprint
from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    strict_json_loads,
)
from src.benchmark.backend_process import NativeHostEvidence
from src.benchmark.backend_publication import (
    ArtifactPublicationError,
    DirectoryPublicationError,
    read_regular_file_no_follow,
    rename_directory_no_replace,
    rollback_published_directory,
)
from src.benchmark.checkpoint_acquisition import CheckpointIdentity
from tools.hpa320.github_host_evidence import (
    build_github_hosted_evidence,
    collect_host_numeric_fingerprint,
)

BUNDLE_SCHEMA = "crux.oaf-native-host-attestation-bundle/v2"
PHASE_WORKFLOWS = MappingProxyType(
    {
        "bootstrap": ("native-bootstrap", "hpa320-native-bootstrap.yml"),
        "measurement": ("native-measurement", "hpa320-native-measurement.yml"),
        "candidate": ("native-candidate", "hpa320-native-candidate.yml"),
    }
)
_IDENTITY_NAMES = MappingProxyType(
    {
        "native_host_evidence": "native-host-evidence.json",
        "native_host_observation": "native-host-observation.json",
    }
)
_BUNDLE_KEYS = frozenset({"native_host_evidence", "native_host_observation", "phase", "schema"})
_IDENTITY_KEYS = frozenset({"name", "sha256", "size"})
_EVIDENCE_KEYS = frozenset({"kind", "official_execution_allowed", "payload", "sha256"})
_OBSERVATION_KEYS = frozenset(
    {
        "docker_architecture",
        "docker_os_type",
        "docker_server_version",
        "github_job",
        "github_repository",
        "github_run_attempt",
        "github_run_id",
        "github_run_url",
        "github_sha",
        "github_workflow_ref",
        "github_workflow_sha",
        "host_numeric_fingerprint",
        "runner_arch",
        "runner_environment",
        "runner_os",
        "uname_architecture",
    }
)
_GITHUB_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_LOWERCASE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_BUNDLE_MAX_BYTES = 64 * 1024
_SIBLING_MAX_BYTES = 64 * 1024


class HostAttestationError(ValueError):
    """The native-host bundle does not prove one accepted GitHub execution."""


@dataclass(frozen=True)
class NativeHostAttestationBundle:
    native_host_evidence: CheckpointIdentity
    native_host_observation: CheckpointIdentity
    phase: str
    sha256: str
    evidence: NativeHostEvidence
    observation: Mapping[str, JsonValue]


def bundle_phase(path: Path) -> str:
    """Return the strictly parsed phase before loading a phase-owned bundle."""

    content = _read_file(
        Path(path),
        "native-host attestation bundle",
        max_bytes=_BUNDLE_MAX_BYTES,
    )
    payload = _canonical_object(content, "native-host attestation bundle")
    phase = payload.get("phase")
    if phase not in PHASE_WORKFLOWS:
        raise HostAttestationError("native-host attestation phase is invalid")
    return cast(str, phase)


def load_native_host_attestation_bundle(
    path: Path,
    *,
    expected_phase: str,
) -> NativeHostAttestationBundle:
    """Authenticate one canonical v2 bundle and its exact two sibling files."""

    if expected_phase not in PHASE_WORKFLOWS:
        raise HostAttestationError("expected native-host attestation phase is invalid")
    bundle_path = Path(path)
    content = _read_file(
        bundle_path,
        "native-host attestation bundle",
        max_bytes=_BUNDLE_MAX_BYTES,
    )
    payload = _canonical_object(content, "native-host attestation bundle")
    if (
        set(payload) != _BUNDLE_KEYS
        or payload["schema"] != BUNDLE_SCHEMA
        or payload["phase"] != expected_phase
    ):
        raise HostAttestationError("native-host attestation bundle phase or fields are invalid")
    identities: dict[str, CheckpointIdentity] = {}
    sibling_contents: dict[str, bytes] = {}
    for field, expected_name in _IDENTITY_NAMES.items():
        identity = _load_identity(
            payload[field],
            expected_name,
            field,
            maximum_size=_SIBLING_MAX_BYTES,
        )
        sibling_content = _read_file(
            bundle_path.parent / expected_name,
            f"native-host attestation {field}",
            max_bytes=identity.size,
        )
        if (
            len(sibling_content) != identity.size
            or hashlib.sha256(sibling_content).hexdigest() != identity.sha256
        ):
            raise HostAttestationError(f"native-host attestation {field} identity mismatch")
        identities[field] = identity
        sibling_contents[field] = sibling_content
    evidence = _load_native_evidence(
        _canonical_object(sibling_contents["native_host_evidence"], "native host evidence")
    )
    observation = _canonical_object(
        sibling_contents["native_host_observation"],
        "native host observation",
    )
    _cross_validate(evidence=evidence, observation=observation, phase=expected_phase)
    return NativeHostAttestationBundle(
        native_host_evidence=identities["native_host_evidence"],
        native_host_observation=identities["native_host_observation"],
        phase=expected_phase,
        sha256=hashlib.sha256(content).hexdigest(),
        evidence=evidence,
        observation=MappingProxyType(observation),
    )


def publish_github_host_attestation(
    *,
    phase: str,
    output_directory: Path,
) -> NativeHostAttestationBundle:
    """Validate this exact work job and atomically publish its v2 host bundle."""

    try:
        observation = _same_job_observation(phase)
        fingerprint = parse_host_numeric_fingerprint(observation["host_numeric_fingerprint"])
        evidence_record = build_github_hosted_evidence(
            github_job=cast(str, observation["github_job"]),
            github_repository=cast(str, observation["github_repository"]),
            github_run_attempt=cast(int, observation["github_run_attempt"]),
            github_run_id=cast(int, observation["github_run_id"]),
            github_workflow_ref=cast(str, observation["github_workflow_ref"]),
            github_workflow_sha=cast(str, observation["github_workflow_sha"]),
            host_numeric_fingerprint=fingerprint,
            run_url=cast(str, observation["github_run_url"]),
            runner_arch=cast(str, observation["runner_arch"]),
            runner_environment=cast(str, observation["runner_environment"]),
            runner_os=cast(str, observation["runner_os"]),
            workflow_commit=cast(str, observation["github_sha"]),
        )
        output = Path(output_directory)
        parent = output.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
        publication = None
        try:
            observation_path = staging / "native-host-observation.json"
            evidence_path = staging / "native-host-evidence.json"
            observation_content = canonical_json_bytes(observation, trailing_newline=True)
            evidence_content = canonical_json_bytes(evidence_record, trailing_newline=True)
            _write_new_regular_file(observation_path, observation_content)
            _write_new_regular_file(evidence_path, evidence_content)
            bundle_payload: JsonValue = {
                "native_host_evidence": _identity_payload(evidence_path.name, evidence_content),
                "native_host_observation": _identity_payload(
                    observation_path.name,
                    observation_content,
                ),
                "phase": phase,
                "schema": BUNDLE_SCHEMA,
            }
            bundle_path = staging / "attestation-bundle.json"
            bundle_content = canonical_json_bytes(bundle_payload, trailing_newline=True)
            _write_new_regular_file(bundle_path, bundle_content)
            load_native_host_attestation_bundle(bundle_path, expected_phase=phase)
            publication = rename_directory_no_replace(staging, output)
            return load_native_host_attestation_bundle(
                output / "attestation-bundle.json",
                expected_phase=phase,
            )
        except DirectoryPublicationError as error:
            rollback_published_directory(error.publication)
            raise
        except Exception:
            if publication is None:
                shutil.rmtree(staging, ignore_errors=True)
            else:
                rollback_published_directory(publication)
            raise
    except (
        ArtifactPublicationError,
        OSError,
        StrictJsonError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, HostAttestationError):
            raise
        raise HostAttestationError("native-host attestation publication failed") from error


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Validate the isolated exact v2 bundle schema without sibling evidence."""

    try:
        if schema != BUNDLE_SCHEMA:
            raise HostAttestationError("native-host attestation schema is unsupported")
        payload = _canonical_object(content, "native-host attestation bundle golden")
        if (
            set(payload) != _BUNDLE_KEYS
            or payload["schema"] != BUNDLE_SCHEMA
            or payload["phase"] not in PHASE_WORKFLOWS
        ):
            raise HostAttestationError("native-host attestation bundle fields are invalid")
        for field, expected_name in _IDENTITY_NAMES.items():
            _load_identity(payload[field], expected_name, field)
    except HostAttestationError as error:
        raise ValueError(str(error)) from None


def _same_job_observation(phase: str) -> dict[str, JsonValue]:
    if phase not in PHASE_WORKFLOWS:
        raise HostAttestationError("native-host attestation phase is invalid")
    runner_environment = _matching_runner_value(
        context_name="RUNNER_ENVIRONMENT_CONTEXT",
        default_name="RUNNER_ENVIRONMENT",
        expected="github-hosted",
    )
    runner_os = _matching_runner_value(
        context_name="RUNNER_OS_CONTEXT",
        default_name="RUNNER_OS",
        expected="Linux",
    )
    runner_arch = _matching_runner_value(
        context_name="RUNNER_ARCH_CONTEXT",
        default_name="RUNNER_ARCH",
        expected="X64",
    )
    job, workflow_file = PHASE_WORKFLOWS[phase]
    if _required_environment("GITHUB_JOB") != job:
        raise HostAttestationError("current GitHub job does not own this native phase")
    commit = _required_environment("COMMIT_SHA")
    checked_out_commit = _run_text(("git", "rev-parse", "HEAD"), "checked-out commit")
    if _LOWERCASE_COMMIT.fullmatch(commit) is None or {
        commit,
        _required_environment("GITHUB_SHA"),
        _required_environment("WORKFLOW_SOURCE_SHA"),
        checked_out_commit,
    } != {commit}:
        raise HostAttestationError("workflow commit does not match the exact checkout")
    repository = _required_environment("GITHUB_REPOSITORY")
    if _GITHUB_REPOSITORY.fullmatch(repository) is None:
        raise HostAttestationError("GITHUB_REPOSITORY is not a canonical owner/name value")
    run_id = _positive_environment_integer("GITHUB_RUN_ID")
    run_attempt = _positive_environment_integer("GITHUB_RUN_ATTEMPT")
    server_url = _required_environment("GITHUB_SERVER_URL").rstrip("/")
    if server_url != "https://github.com":
        raise HostAttestationError("GITHUB_SERVER_URL must be https://github.com")
    workflow_ref = _required_environment("GITHUB_WORKFLOW_REF")
    workflow_prefix = f"{repository}/.github/workflows/{workflow_file}@"
    if not workflow_ref.startswith(workflow_prefix) or workflow_ref == workflow_prefix:
        raise HostAttestationError("current GitHub workflow does not own this native phase")
    uname_architecture = _run_text(("uname", "-m"), "host architecture")
    docker_os_type = _docker_info("OSType")
    docker_architecture = _docker_info("Architecture")
    docker_server_version = _docker_info("ServerVersion")
    if (
        uname_architecture != "x86_64"
        or docker_os_type != "linux"
        or docker_architecture != "x86_64"
        or not docker_server_version
    ):
        raise HostAttestationError("workflow host is not native Linux X64")
    return {
        "docker_architecture": docker_architecture,
        "docker_os_type": docker_os_type,
        "docker_server_version": docker_server_version,
        "github_job": job,
        "github_repository": repository,
        "github_run_attempt": run_attempt,
        "github_run_id": run_id,
        "github_run_url": f"{server_url}/{repository}/actions/runs/{run_id}",
        "github_sha": commit,
        "github_workflow_ref": workflow_ref,
        "github_workflow_sha": commit,
        "host_numeric_fingerprint": collect_host_numeric_fingerprint().as_json(),
        "runner_arch": runner_arch,
        "runner_environment": runner_environment,
        "runner_os": runner_os,
        "uname_architecture": uname_architecture,
    }


def _cross_validate(
    *,
    evidence: NativeHostEvidence,
    observation: dict[str, JsonValue],
    phase: str,
) -> None:
    if set(observation) != _OBSERVATION_KEYS:
        raise HostAttestationError("native host observation fields are invalid")
    if phase not in PHASE_WORKFLOWS:
        raise HostAttestationError("native-host attestation phase is invalid")
    job, workflow_file = PHASE_WORKFLOWS[phase]
    integer_fields = ("github_run_attempt", "github_run_id")
    if any(
        not isinstance(observation[field], int)
        or isinstance(observation[field], bool)
        or cast(int, observation[field]) <= 0
        for field in integer_fields
    ):
        raise HostAttestationError("native host observation numeric fields are invalid")
    repository = observation["github_repository"]
    commit = observation["github_sha"]
    run_id = observation["github_run_id"]
    workflow_ref = observation["github_workflow_ref"]
    if (
        not isinstance(repository, str)
        or _GITHUB_REPOSITORY.fullmatch(repository) is None
        or not isinstance(commit, str)
        or _LOWERCASE_COMMIT.fullmatch(commit) is None
        or observation["github_workflow_sha"] != commit
        or observation["github_run_url"] != f"https://github.com/{repository}/actions/runs/{run_id}"
        or not isinstance(workflow_ref, str)
        or not workflow_ref.startswith(f"{repository}/.github/workflows/{workflow_file}@")
        or workflow_ref.endswith("@")
    ):
        raise HostAttestationError("native host observation workflow reference is invalid")
    if (
        observation["github_job"] != job
        or observation["docker_architecture"] != "x86_64"
        or observation["docker_os_type"] != "linux"
        or not isinstance(observation["docker_server_version"], str)
        or not observation["docker_server_version"]
        or observation["runner_arch"] != "X64"
        or observation["runner_environment"] != "github-hosted"
        or observation["runner_os"] != "Linux"
        or observation["uname_architecture"] != "x86_64"
    ):
        raise HostAttestationError("native host observation is not GitHub-hosted Linux X64")
    try:
        parse_host_numeric_fingerprint(observation["host_numeric_fingerprint"])
    except (TypeError, ValueError) as error:
        raise HostAttestationError("native host observation fingerprint is invalid") from error
    host_payload = evidence.payload
    expected_fields = {
        "github_job": observation["github_job"],
        "github_repository": observation["github_repository"],
        "github_run_attempt": observation["github_run_attempt"],
        "github_run_id": observation["github_run_id"],
        "github_workflow_ref": observation["github_workflow_ref"],
        "github_workflow_sha": observation["github_workflow_sha"],
        "host_numeric_fingerprint": observation["host_numeric_fingerprint"],
        "run_url": observation["github_run_url"],
        "runner_arch": observation["runner_arch"],
        "runner_environment": observation["runner_environment"],
        "runner_os": observation["runner_os"],
        "workflow_commit": observation["github_sha"],
    }
    if evidence.kind != "github_hosted" or any(
        host_payload[field] != value for field, value in expected_fields.items()
    ):
        raise HostAttestationError("native host evidence disagrees with observation")


def _load_native_evidence(value: dict[str, JsonValue]) -> NativeHostEvidence:
    if set(value) != _EVIDENCE_KEYS:
        raise HostAttestationError("native host evidence fields are invalid")
    try:
        return NativeHostEvidence(
            kind=cast(Any, value["kind"]),
            payload=cast(Any, value["payload"]),
            sha256=cast(Any, value["sha256"]),
            official_execution_allowed=cast(Any, value["official_execution_allowed"]),
        )
    except (TypeError, ValueError) as error:
        raise HostAttestationError(f"native host evidence is invalid: {error}") from None


def _load_identity(
    value: JsonValue,
    expected_name: str,
    label: str,
    *,
    maximum_size: int | None = None,
) -> CheckpointIdentity:
    if not isinstance(value, dict) or set(value) != _IDENTITY_KEYS:
        raise HostAttestationError(f"native-host attestation {label} identity fields are invalid")
    name = value["name"]
    digest = value["sha256"]
    size = value["size"]
    if (
        name != expected_name
        or not _is_sha256(digest)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or maximum_size is not None
        and size > maximum_size
    ):
        raise HostAttestationError(f"native-host attestation {label} name or identity is invalid")
    return CheckpointIdentity(name=expected_name, sha256=cast(str, digest), size=size)


def _canonical_object(content: bytes, label: str) -> dict[str, JsonValue]:
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise HostAttestationError(f"{label} must have one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise HostAttestationError(f"{label} is invalid: {error}") from None
    if not isinstance(value, dict):
        raise HostAttestationError(f"{label} must be an object")
    return value


def _read_file(path: Path, label: str, *, max_bytes: int) -> bytes:
    try:
        return read_regular_file_no_follow(path, max_bytes=max_bytes)
    except OSError as error:
        raise HostAttestationError(f"{label} is missing or unsafe") from error


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity_payload(name: str, content: bytes) -> dict[str, JsonValue]:
    return {"name": name, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}


def _write_new_regular_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)


def _matching_runner_value(*, context_name: str, default_name: str, expected: str) -> str:
    context_value = _required_environment(context_name)
    default_value = _required_environment(default_name)
    if context_value != expected or default_value != expected or context_value != default_value:
        raise HostAttestationError(f"{context_name} and {default_name} do not prove {expected}")
    return context_value


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise HostAttestationError(f"{name} is missing")
    return value


def _positive_environment_integer(name: str) -> int:
    value = _required_environment(name)
    if not value.isdecimal() or int(value) <= 0:
        raise HostAttestationError(f"{name} is not a positive integer")
    return int(value)


def _run_text(command: tuple[str, ...], label: str) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise HostAttestationError(f"{label} could not be observed") from error
    value = result.stdout.strip()
    if not value:
        raise HostAttestationError(f"{label} is empty")
    return value


def _docker_info(field: str) -> str:
    return _run_text(("docker", "info", "--format", f"{{{{.{field}}}}}"), f"Docker {field}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish-github")
    publish.add_argument("--phase", required=True, choices=tuple(PHASE_WORKFLOWS))
    publish.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        publish_github_host_attestation(
            phase=arguments.phase,
            output_directory=arguments.output,
        )
    except HostAttestationError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
