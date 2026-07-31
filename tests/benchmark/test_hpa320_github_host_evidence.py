from __future__ import annotations

from src.benchmark.backend_attestation import HostNumericFingerprint
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex
from tools.hpa320.github_host_evidence import build_github_hosted_evidence

FINGERPRINT = HostNumericFingerprint(
    architecture="x86_64",
    cpu_vendor_id="GenuineIntel",
    cpu_family="6",
    cpu_model="143",
    cpu_stepping="8",
)


def test_builder_emits_the_exact_v2_record() -> None:
    evidence = build_github_hosted_evidence(
        github_job="native-bootstrap",
        github_repository="cwchanap/Crux",
        github_run_attempt=2,
        github_run_id=123456789,
        github_workflow_ref=(
            "cwchanap/Crux/.github/workflows/hpa320-native-bootstrap.yml@refs/tags/native-v2"
        ),
        github_workflow_sha="a" * 40,
        host_numeric_fingerprint=FINGERPRINT,
        run_url="https://github.com/cwchanap/Crux/actions/runs/123456789",
        runner_arch="X64",
        runner_environment="github-hosted",
        runner_os="Linux",
        workflow_commit="a" * 40,
    )

    assert set(evidence) == {
        "kind",
        "official_execution_allowed",
        "payload",
        "sha256",
    }
    assert set(evidence["payload"]) == {
        "schema",
        "github_job",
        "github_repository",
        "github_run_attempt",
        "github_run_id",
        "github_workflow_ref",
        "github_workflow_sha",
        "host_numeric_fingerprint",
        "run_url",
        "runner_arch",
        "runner_environment",
        "runner_os",
        "workflow_commit",
    }
    assert evidence["payload"]["schema"] == "crux.github-hosted-native-evidence/v2"
    assert evidence["sha256"] == sha256_hex(canonical_json_bytes(evidence["payload"]))
