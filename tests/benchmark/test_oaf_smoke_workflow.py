from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github/workflows/oaf-smoke.yml"


def test_oaf_smoke_workflow_is_manual_single_job_and_minimal() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "on:\n  workflow_dispatch:" in content
    assert "  push:" not in content
    assert "  pull_request:" not in content
    assert "inputs:" not in content
    assert content.count("runs-on: ubuntu-24.04") == 1
    assert "permissions:\n  contents: read" in content
    assert content.count("prepare-backend --backend oaf --download") == 1
    assert content.count("docker build -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1:local .") == 1
    assert content.count("smoke-backend --backend oaf") == 1
    assert "artifacts/benchmark/oaf-smoke/prediction.jsonl" in content
    assert "actions/upload-artifact@v4" in content
    assert "GITHUB_STEP_SUMMARY" in content
    assert "GITHUB_SHA" in content
    assert "GITHUB_REF" in content


def test_oaf_smoke_workflow_has_no_attestation_or_measurement_steps() -> None:
    content = WORKFLOW.read_text(encoding="utf-8").lower()

    for forbidden in (
        "oidc",
        "sigstore",
        "attestation",
        "cosign",
        "calibration",
        "measurement",
        "fingerprint",
        "second image",
    ):
        assert forbidden not in content
