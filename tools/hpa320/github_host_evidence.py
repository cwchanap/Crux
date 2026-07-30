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
    api_record_bytes: bytes,
    job_id: int,
    run_url: str,
    workflow_commit: str,
    host_numeric_fingerprint: HostNumericFingerprint | None = None,
) -> dict[str, JsonValue]:
    """Build one payload and validate it with the current NativeHostEvidence reader."""

    fingerprint = (
        collect_host_numeric_fingerprint()
        if host_numeric_fingerprint is None
        else host_numeric_fingerprint
    )
    payload: dict[str, JsonValue] = {
        "api_record_sha256": sha256_hex(api_record_bytes),
        "approved_labels": ["Linux", "X64"],
        "host_numeric_fingerprint": fingerprint.as_json(),
        "job_id": job_id,
        "run_url": run_url,
        "runner_arch": "X64",
        "runner_os": "Linux",
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
