#!/usr/bin/env python3
"""Strictly authenticate durable phase-specific OaF native-host bundles."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.benchmark.backend_attestation import parse_host_numeric_fingerprint
from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    strict_json_loads,
)
from src.benchmark.backend_process import NativeHostEvidence
from src.benchmark.backend_publication import read_regular_file_no_follow
from src.benchmark.checkpoint_acquisition import CheckpointIdentity
from tools.hpa320.github_host_evidence import (
    build_github_hosted_evidence,
    collect_host_numeric_fingerprint,
)

BUNDLE_SCHEMA = "crux.oaf-native-host-attestation-bundle/v1"
PHASES = frozenset({"bootstrap", "measurement", "candidate"})

_BUNDLE_KEYS = frozenset(
    {
        "api_record",
        "native_host_evidence",
        "native_host_observation",
        "phase",
        "schema",
    }
)
_IDENTITY_KEYS = frozenset({"name", "sha256", "size"})
_EVIDENCE_KEYS = frozenset({"kind", "official_execution_allowed", "payload", "sha256"})
_OBSERVATION_KEYS = frozenset(
    {
        "docker_architecture",
        "docker_os_type",
        "docker_server_version",
        "github_repository",
        "github_run_attempt",
        "github_run_id",
        "github_run_url",
        "github_sha",
        "github_workflow_ref",
        "host_numeric_fingerprint",
        "runner_arch",
        "runner_os",
        "uname_architecture",
    }
)
_IDENTITY_NAMES = {
    "api_record": "github-job-api-record.json.hex",
    "native_host_evidence": "native-host-evidence.json",
    "native_host_observation": "native-host-observation.json",
}
_GITHUB_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


class HostAttestationError(ValueError):
    """The native-host bundle does not prove one accepted GitHub execution."""


@dataclass(frozen=True)
class NativeHostAttestationBundle:
    api_record: CheckpointIdentity
    native_host_evidence: CheckpointIdentity
    native_host_observation: CheckpointIdentity
    phase: str
    sha256: str


def bundle_phase(path: Path) -> str:
    """Return the strictly parsed phase before loading a phase-owned bundle."""

    content = _read_file(Path(path), "native-host attestation bundle")
    payload = _canonical_object(content, "native-host attestation bundle")
    phase = payload.get("phase")
    if phase not in PHASES:
        raise HostAttestationError("native-host attestation phase is invalid")
    return cast(str, phase)


def load_native_host_attestation_bundle(
    path: Path,
    *,
    expected_phase: str,
) -> NativeHostAttestationBundle:
    """Authenticate one canonical bundle and its exact three sibling files."""

    if expected_phase not in PHASES:
        raise HostAttestationError("expected native-host attestation phase is invalid")
    bundle_path = Path(path)
    content = _read_file(bundle_path, "native-host attestation bundle")
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
        identity = _load_identity(payload[field], expected_name, field)
        sibling = bundle_path.parent / expected_name
        sibling_content = _read_file(sibling, f"native-host attestation {field}")
        if (
            len(sibling_content) != identity.size
            or hashlib.sha256(sibling_content).hexdigest() != identity.sha256
        ):
            raise HostAttestationError(f"native-host attestation {field} identity mismatch")
        identities[field] = identity
        sibling_contents[field] = sibling_content
    api_bytes = _decode_api_hex(sibling_contents["api_record"])
    api_record = _raw_object(api_bytes, "GitHub job API record")
    evidence_record = _canonical_object(
        sibling_contents["native_host_evidence"],
        "native host evidence",
    )
    observation = _canonical_object(
        sibling_contents["native_host_observation"],
        "native host observation",
    )
    evidence = _load_native_evidence(evidence_record)
    _cross_validate(
        api_record=api_record,
        api_bytes=api_bytes,
        evidence=evidence,
        observation=observation,
        phase=expected_phase,
    )
    return NativeHostAttestationBundle(
        api_record=identities["api_record"],
        native_host_evidence=identities["native_host_evidence"],
        native_host_observation=identities["native_host_observation"],
        phase=expected_phase,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def observe_github_host(*, output_path: Path, github_output_path: Path) -> bytes:
    """Record one native Linux X64 observation for the exact checked-out commit."""

    runner_os = _required_environment("RUNNER_OS")
    runner_arch = _required_environment("RUNNER_ARCH")
    commit = _required_environment("COMMIT_SHA")
    github_sha = _required_environment("GITHUB_SHA")
    checked_out_commit = _run_text(("git", "rev-parse", "HEAD"), "checked-out commit")
    uname_architecture = _run_text(("uname", "-m"), "host architecture")
    docker_os_type = _docker_info("OSType")
    docker_architecture = _docker_info("Architecture")
    docker_server_version = _docker_info("ServerVersion")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", commit)
        or checked_out_commit != commit
        or github_sha != commit
    ):
        raise HostAttestationError("workflow commit does not match the exact checkout")
    if (
        runner_os != "Linux"
        or runner_arch != "X64"
        or uname_architecture != "x86_64"
        or docker_os_type != "linux"
        or docker_architecture != "x86_64"
    ):
        raise HostAttestationError("workflow host is not native Linux X64")
    repository = _required_environment("GITHUB_REPOSITORY")
    run_id = _positive_environment_integer("GITHUB_RUN_ID")
    run_attempt = _positive_environment_integer("GITHUB_RUN_ATTEMPT")
    server_url = _required_environment("GITHUB_SERVER_URL").rstrip("/")
    if server_url != "https://github.com":
        raise HostAttestationError(
            "GITHUB_SERVER_URL must be the canonical https://github.com value"
        )
    observation: JsonValue = {
        "docker_architecture": docker_architecture,
        "docker_os_type": docker_os_type,
        "docker_server_version": docker_server_version,
        "github_repository": repository,
        "github_run_attempt": run_attempt,
        "github_run_id": run_id,
        "github_run_url": f"{server_url}/{repository}/actions/runs/{run_id}",
        "github_sha": commit,
        "github_workflow_ref": _required_environment("GITHUB_WORKFLOW_REF"),
        "host_numeric_fingerprint": collect_host_numeric_fingerprint().as_json(),
        "runner_arch": runner_arch,
        "runner_os": runner_os,
        "uname_architecture": uname_architecture,
    }
    content = canonical_json_bytes(observation, trailing_newline=True)
    Path(output_path).write_bytes(content)
    with Path(github_output_path).open("a", encoding="utf-8") as output:
        output.write(f"observation={content[:-1].decode('utf-8')}\n")
    return content


def finalize_github_host_attestation(
    *,
    phase: str,
    observation_path: Path,
    output_directory: Path,
) -> NativeHostAttestationBundle:
    """Fetch completed-job bytes and publish one exact phase-specific bundle."""

    if phase not in PHASES:
        raise HostAttestationError("native-host attestation phase is invalid")
    token = _required_environment("GH_TOKEN")
    repository = _required_environment("GITHUB_REPOSITORY")
    if not _GITHUB_REPOSITORY.fullmatch(repository):
        raise HostAttestationError("GITHUB_REPOSITORY is not a canonical owner/name value")
    run_id = _positive_environment_integer("GITHUB_RUN_ID")
    commit = _required_environment("COMMIT_SHA")
    observation_content = _read_file(Path(observation_path), "native host observation")
    observation = _canonical_object(observation_content, "native host observation")
    if (
        observation.get("github_repository") != repository
        or observation.get("github_run_id") != run_id
        or observation.get("github_sha") != commit
    ):
        raise HostAttestationError("native host observation belongs to another execution")
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    jobs = _github_json(
        (
            f"https://api.github.com/repos/{repository}/actions/runs/"
            f"{run_id}/jobs?filter=latest&per_page=100"
        ),
        headers,
        "completed GitHub jobs",
    )
    job_rows = jobs.get("jobs")
    if not isinstance(job_rows, list):
        raise HostAttestationError("completed GitHub jobs payload is invalid")
    matches = [
        row
        for row in job_rows
        if isinstance(row, dict) and row.get("name") == "observe-native-host"
    ]
    if len(matches) != 1:
        raise HostAttestationError("completed observation job is not unique")
    job_id = matches[0].get("id")
    if not isinstance(job_id, int) or isinstance(job_id, bool) or job_id <= 0:
        raise HostAttestationError("completed observation job ID is invalid")
    api_bytes = _github_bytes(
        f"https://api.github.com/repos/{repository}/actions/jobs/{job_id}",
        headers,
        "completed GitHub job",
    )
    api_record = _raw_object(api_bytes, "GitHub job API record")
    job_url = api_record.get("html_url")
    if not isinstance(job_url, str):
        raise HostAttestationError("completed observation job URL is invalid")
    try:
        fingerprint = parse_host_numeric_fingerprint(observation.get("host_numeric_fingerprint"))
        evidence = build_github_hosted_evidence(
            api_record_bytes=api_bytes,
            host_numeric_fingerprint=fingerprint,
            job_id=job_id,
            run_url=job_url,
            workflow_commit=commit,
        )
    except (StrictJsonError, TypeError, ValueError) as error:
        raise HostAttestationError("native host evidence could not be derived") from error
    api_hex = api_bytes.hex().encode("ascii") + b"\n"
    evidence_content = canonical_json_bytes(evidence, trailing_newline=True)
    output = Path(output_directory)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise HostAttestationError("native-host attestation output must be absent") from error
    files = {
        _IDENTITY_NAMES["api_record"]: api_hex,
        _IDENTITY_NAMES["native_host_evidence"]: evidence_content,
        _IDENTITY_NAMES["native_host_observation"]: observation_content,
    }
    try:
        for name, content in files.items():
            (output / name).write_bytes(content)
        bundle: JsonValue = {
            "api_record": _identity_payload(_IDENTITY_NAMES["api_record"], api_hex),
            "native_host_evidence": _identity_payload(
                _IDENTITY_NAMES["native_host_evidence"],
                evidence_content,
            ),
            "native_host_observation": _identity_payload(
                _IDENTITY_NAMES["native_host_observation"],
                observation_content,
            ),
            "phase": phase,
            "schema": BUNDLE_SCHEMA,
        }
        bundle_path = output / "attestation-bundle.json"
        bundle_path.write_bytes(canonical_json_bytes(bundle, trailing_newline=True))
        return load_native_host_attestation_bundle(bundle_path, expected_phase=phase)
    except (OSError, ValueError) as error:
        raise HostAttestationError("native-host attestation publication failed") from error


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Validate the isolated exact bundle schema without inventing sibling evidence."""

    try:
        if schema != BUNDLE_SCHEMA:
            raise HostAttestationError("native-host attestation schema is unsupported")
        payload = _canonical_object(content, "native-host attestation bundle golden")
        if (
            set(payload) != _BUNDLE_KEYS
            or payload["schema"] != BUNDLE_SCHEMA
            or payload["phase"] not in PHASES
        ):
            raise HostAttestationError("native-host attestation bundle fields are invalid")
        for field, expected_name in _IDENTITY_NAMES.items():
            _load_identity(payload[field], expected_name, field)
    except HostAttestationError as error:
        raise ValueError(str(error)) from None


def _cross_validate(
    *,
    api_record: dict[str, JsonValue],
    api_bytes: bytes,
    evidence: NativeHostEvidence,
    observation: dict[str, JsonValue],
    phase: str,
) -> None:
    if set(observation) != _OBSERVATION_KEYS:
        raise HostAttestationError("native host observation fields are invalid")
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
    run_url = observation["github_run_url"]
    workflow_ref = observation["github_workflow_ref"]
    if (
        not isinstance(repository, str)
        or not _GITHUB_REPOSITORY.fullmatch(repository)
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or run_url != f"https://github.com/{repository}/actions/runs/{run_id}"
        or not isinstance(workflow_ref, str)
        or not workflow_ref.startswith(f"{repository}/.github/workflows/hpa320-native-{phase}.yml@")
        or workflow_ref.endswith("@")
    ):
        raise HostAttestationError("native host observation workflow reference is invalid")
    if (
        observation["docker_architecture"] != "x86_64"
        or observation["docker_os_type"] != "linux"
        or not isinstance(observation["docker_server_version"], str)
        or not observation["docker_server_version"]
        or observation["runner_arch"] != "X64"
        or observation["runner_os"] != "Linux"
        or observation["uname_architecture"] != "x86_64"
    ):
        raise HostAttestationError("native host observation is not Linux X64")
    required_api = {
        "conclusion": "success",
        "head_sha": commit,
        "name": "observe-native-host",
        "run_id": run_id,
        "status": "completed",
    }
    if any(api_record.get(field) != value for field, value in required_api.items()):
        raise HostAttestationError("GitHub job API record is not a completed successful job")
    if api_record.get("labels") != ["ubuntu-24.04"]:
        raise HostAttestationError("GitHub job API runner labels are invalid")
    job_id = api_record.get("id")
    job_url = api_record.get("html_url")
    if (
        not isinstance(job_id, int)
        or isinstance(job_id, bool)
        or job_id <= 0
        or job_url != f"{run_url}/job/{job_id}"
    ):
        raise HostAttestationError("GitHub job API identity is invalid")
    host_payload = evidence.payload
    if (
        evidence.kind != "github_hosted"
        or host_payload["api_record_sha256"] != hashlib.sha256(api_bytes).hexdigest()
        or host_payload["job_id"] != job_id
        or host_payload["run_url"] != job_url
        or host_payload["workflow_commit"] != commit
        or host_payload["runner_arch"] != observation["runner_arch"]
        or host_payload["runner_os"] != observation["runner_os"]
        or evidence.host_numeric_fingerprint.as_json() != observation["host_numeric_fingerprint"]
    ):
        raise HostAttestationError("native host evidence disagrees with observation or API record")


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


def _load_identity(value: JsonValue, expected_name: str, label: str) -> CheckpointIdentity:
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
    ):
        raise HostAttestationError(f"native-host attestation {label} name or identity is invalid")
    return CheckpointIdentity(name=expected_name, sha256=cast(str, digest), size=size)


def _decode_api_hex(content: bytes) -> bytes:
    if (
        not content.endswith(b"\n")
        or content.endswith(b"\n\n")
        or b"\n" in content[:-1]
        or not content[:-1]
        or len(content[:-1]) % 2
        or any(character not in b"0123456789abcdef" for character in content[:-1])
    ):
        raise HostAttestationError(
            "GitHub job API record hex must be lowercase pairs plus one final newline"
        )
    try:
        return bytes.fromhex(content[:-1].decode("ascii"))
    except (UnicodeError, ValueError):
        raise HostAttestationError("GitHub job API record hex is invalid") from None


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


def _raw_object(content: bytes, label: str) -> dict[str, JsonValue]:
    try:
        value = strict_json_loads(content)
    except StrictJsonError as error:
        raise HostAttestationError(f"{label} is invalid: {error}") from None
    if not isinstance(value, dict):
        raise HostAttestationError(f"{label} must be an object")
    return value


def _read_file(path: Path, label: str) -> bytes:
    try:
        return read_regular_file_no_follow(path)
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


def _github_bytes(url: str, headers: dict[str, str], label: str) -> bytes:
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
    except OSError as error:
        raise HostAttestationError(f"{label} could not be fetched") from error
    if not content:
        raise HostAttestationError(f"{label} is empty")
    return content


def _github_json(url: str, headers: dict[str, str], label: str) -> dict[str, JsonValue]:
    return _raw_object(_github_bytes(url, headers, label), label)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    observe = commands.add_parser("observe-github")
    observe.add_argument("--output", type=Path, required=True)
    observe.add_argument("--github-output", type=Path, required=True)
    finalize = commands.add_parser("finalize-github")
    finalize.add_argument("--phase", choices=sorted(PHASES), required=True)
    finalize.add_argument("--observation", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "observe-github":
            observe_github_host(
                output_path=arguments.output,
                github_output_path=arguments.github_output,
            )
        else:
            finalize_github_host_attestation(
                phase=arguments.phase,
                observation_path=arguments.observation,
                output_directory=arguments.output,
            )
    except HostAttestationError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
