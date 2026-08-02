from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github/workflows/hpa423-native-host-diagnostic.yml"
CHECKOUT = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
SETUP_UV = "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b"


def test_native_host_diagnostic_workflow_is_fail_only() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    assert isinstance(workflow, dict)
    assert workflow["permissions"] == {"contents": "read"}

    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {"diagnose-native-host"}
    job = jobs["diagnose-native-host"]
    assert job == {
        "name": "diagnose-native-host",
        "runs-on": "ubuntu-24.04",
        "timeout-minutes": 20,
        "env": {"COMMIT_SHA": "${{ inputs.commit_sha }}"},
        "steps": job["steps"],
    }

    steps = job["steps"]
    assert isinstance(steps, list)
    assert len(steps) == 3
    assert steps[0] == {
        "name": "Check out exact diagnostic commit",
        "uses": CHECKOUT,
        "with": {
            "persist-credentials": False,
            "ref": "${{ inputs.commit_sha }}",
        },
    }
    assert steps[1] == {
        "name": "Install exact UV",
        "uses": SETUP_UV,
        "with": {
            "version": "0.11.8",
            "python-version": "3.12",
            "enable-cache": False,
        },
    }
    assert steps[2] == {
        "name": "Diagnose the current native host",
        "env": {
            "RUNNER_ENVIRONMENT_CONTEXT": "${{ runner.environment }}",
            "RUNNER_OS_CONTEXT": "${{ runner.os }}",
            "RUNNER_ARCH_CONTEXT": "${{ runner.arch }}",
            "WORKFLOW_SOURCE_SHA": "${{ github.workflow_sha }}",
        },
        "run": (
            "uv run python -m tools.hpa320.oaf_host_attestation_diagnostic "
            "--phase bootstrap --output $RUNNER_TEMP/hpa423-bootstrap-host-attestation"
        ),
    }

    for forbidden in (
        "actions/attest",
        "upload-artifact",
        "prepare-backend",
        "bootstrap-image",
        "attest-base-system",
        "id-token",
        "attestations",
        "continue-on-error",
    ):
        assert forbidden not in text
