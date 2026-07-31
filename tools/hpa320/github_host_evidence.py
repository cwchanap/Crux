"""Build GitHub-hosted native evidence against the current evidence schema."""

from __future__ import annotations

from src.benchmark.backend_attestation import HostNumericFingerprint, _host_numeric_fingerprint
from src.benchmark.backend_identity import JsonValue, canonical_json_bytes, sha256_hex
from src.benchmark.backend_process import NativeHostEvidence


def collect_host_numeric_fingerprint() -> HostNumericFingerprint:
    """Collect and normalize the stable five-field fingerprint for the current host."""

    return _host_numeric_fingerprint()


def build_github_hosted_evidence(
    *,
    github_job: str,
    github_repository: str,
    github_run_attempt: int,
    github_run_id: int,
    github_workflow_ref: str,
    github_workflow_sha: str,
    host_numeric_fingerprint: HostNumericFingerprint,
    run_url: str,
    runner_arch: str,
    runner_environment: str,
    runner_os: str,
    workflow_commit: str,
) -> dict[str, JsonValue]:
    """Build and self-validate one strict v2 GitHub-hosted evidence record."""

    payload: dict[str, JsonValue] = {
        "schema": "crux.github-hosted-native-evidence/v2",
        "github_job": github_job,
        "github_repository": github_repository,
        "github_run_attempt": github_run_attempt,
        "github_run_id": github_run_id,
        "github_workflow_ref": github_workflow_ref,
        "github_workflow_sha": github_workflow_sha,
        "host_numeric_fingerprint": host_numeric_fingerprint.as_json(),
        "run_url": run_url,
        "runner_arch": runner_arch,
        "runner_environment": runner_environment,
        "runner_os": runner_os,
        "workflow_commit": workflow_commit,
    }
    digest = sha256_hex(canonical_json_bytes(payload))
    NativeHostEvidence(
        kind="github_hosted",
        payload=payload,
        sha256=digest,
        official_execution_allowed=True,
    )
    return {
        "kind": "github_hosted",
        "official_execution_allowed": True,
        "payload": payload,
        "sha256": digest,
    }
