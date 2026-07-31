from pathlib import Path

import yaml

from tools.hpa320.oaf_native_artifacts import PHASE_FILES

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPOSITORY_ROOT / ".github/workflows/hpa320-native-measurement.yml"
CHECKOUT = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
SETUP_UV = "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b"
ATTEST = "actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d"
UPLOAD = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"


def _load_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _load() -> dict[str, object]:
    workflow = yaml.safe_load(_load_text())
    assert isinstance(workflow, dict)
    return workflow


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    steps = job["steps"]
    assert isinstance(steps, list)
    assert all(isinstance(step, dict) for step in steps)
    return steps


def _step_named(steps: list[dict[str, object]], name: str) -> dict[str, object]:
    matches = [step for step in steps if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _only_step_using(steps: list[dict[str, object]], action: str) -> dict[str, object]:
    matches = [step for step in steps if step.get("uses") == action]
    assert len(matches) == 1
    return matches[0]


def _only_success_upload(steps: list[dict[str, object]]) -> dict[str, object]:
    matches = [step for step in steps if step.get("uses") == UPLOAD and "if" not in step]
    assert len(matches) == 1
    return matches[0]


def _assert_native_workflow_contract(
    workflow: dict[str, object],
    *,
    phase: str,
    timeout_minutes: int,
    native_work_steps: tuple[str, ...],
) -> None:
    job_id = f"native-{phase}"
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {job_id}
    job = jobs[job_id]
    assert isinstance(job, dict)
    assert job["name"] == job_id
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == timeout_minutes
    assert "needs" not in job
    assert "outputs" not in job
    assert job["env"] == {"COMMIT_SHA": "${{ inputs.commit_sha }}"}
    assert workflow["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
    }

    serialized = yaml.safe_dump(workflow)
    for forbidden in (
        "observe-native-host",
        "observe-github",
        "finalize-github",
        "--github-output",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GITHUB_ENV",
        "GITHUB_OUTPUT",
        "github.token",
        "actions: read",
        "github-job-api-record",
        "artifact-sha256s.txt",
        "api.github.com",
    ):
        assert forbidden not in serialized
    assert serialized.count("publish-github") == 1
    assert serialized.count(ATTEST) == 1
    assert "rtk " not in serialized
    assert "docker/setup-buildx-action" not in serialized
    assert " seal \\" not in serialized
    assert "git add" not in serialized
    assert "git commit" not in serialized

    steps = _steps(job)
    checkout = _only_step_using(steps, CHECKOUT)
    assert checkout["with"] == {
        "persist-credentials": False,
        "ref": "${{ inputs.commit_sha }}",
    }
    setup_uv = _only_step_using(steps, SETUP_UV)
    assert setup_uv["with"] == {
        "version": "0.11.8",
        "python-version": "3.12",
        "enable-cache": False,
    }

    payload_root = f"artifacts/benchmark/backends/hpa320-{phase}"
    archive = f"artifacts/benchmark/backends/hpa320-native-{phase}-${{{{ inputs.commit_sha }}}}.tar"
    bundle = f"artifacts/benchmark/backends/hpa320-native-{phase}-${{{{ inputs.commit_sha }}}}.sigstore.json"
    preflight = _step_named(steps, "Observe and validate the current native work job")
    assert preflight["env"] == {
        "RUNNER_ENVIRONMENT_CONTEXT": "${{ runner.environment }}",
        "RUNNER_OS_CONTEXT": "${{ runner.os }}",
        "RUNNER_ARCH_CONTEXT": "${{ runner.arch }}",
        "WORKFLOW_SOURCE_SHA": "${{ github.workflow_sha }}",
    }
    assert {**job["env"], **preflight["env"]} == {
        "COMMIT_SHA": "${{ inputs.commit_sha }}",
        "RUNNER_ENVIRONMENT_CONTEXT": "${{ runner.environment }}",
        "RUNNER_OS_CONTEXT": "${{ runner.os }}",
        "RUNNER_ARCH_CONTEXT": "${{ runner.arch }}",
        "WORKFLOW_SOURCE_SHA": "${{ github.workflow_sha }}",
    }
    assert preflight["run"] == (
        "uv run python -m tools.hpa320.oaf_host_attestation publish-github "
        f"--phase {phase} --output {payload_root}/{phase}-host-attestation"
    )

    publish = _step_named(steps, "Publish canonical native work subjects")
    assert publish["run"] == (
        "uv run python -m tools.hpa320.oaf_native_artifacts publish "
        f"--phase {phase} --payload-root {payload_root} --host-bundle "
        f"{payload_root}/{phase}-host-attestation/attestation-bundle.json --archive {archive}"
    )
    verify_before_attestation = _step_named(steps, "Verify native work subjects before attestation")
    assert verify_before_attestation["run"] == (
        "uv run python -m tools.hpa320.oaf_native_artifacts verify "
        f"--phase {phase} --payload-root {payload_root} --archive {archive}"
    )
    attest = _only_step_using(steps, ATTEST)
    assert attest["id"] == "attest"
    assert attest["with"] == {
        "subject-path": (
            f"artifacts/benchmark/backends/hpa320-native-{phase}-"
            "${{ inputs.commit_sha }}.tar\n"
            f"artifacts/benchmark/backends/hpa320-{phase}/artifact-manifest.json\n"
        )
    }
    copy_bundle = _step_named(steps, "Preserve the local Sigstore bundle")
    assert copy_bundle["run"] == (
        "uv run python -m tools.hpa320.oaf_native_artifacts copy-bundle "
        "--source '${{ steps.attest.outputs.bundle-path }}' --destination "
        f"{bundle}"
    )
    reverify = _step_named(steps, "Reverify the complete upload set")
    assert reverify["run"] == (
        "uv run python -m tools.hpa320.oaf_native_artifacts verify "
        f"--phase {phase} --payload-root {payload_root} --archive {archive} --bundle {bundle}"
    )

    success_upload = _only_success_upload(steps)
    uploads = [step for step in steps if step.get("uses") == UPLOAD]
    assert uploads == [success_upload]
    assert success_upload["with"] == {
        "name": f"hpa320-native-{phase}-${{{{ inputs.commit_sha }}}}",
        "path": f"{payload_root}/\n{archive}\n{bundle}\n",
        "if-no-files-found": "error",
        "retention-days": 30,
    }
    assert archive.startswith("artifacts/benchmark/backends/")
    assert bundle.startswith("artifacts/benchmark/backends/")
    assert not archive.startswith(f"{payload_root}/")
    assert not bundle.startswith(f"{payload_root}/")
    assert "*" not in archive
    assert "*" not in bundle
    for manifest_mapping in PHASE_FILES.values():
        assert archive not in manifest_mapping
        assert bundle not in manifest_mapping
        assert archive not in manifest_mapping.values()
        assert bundle not in manifest_mapping.values()
    assert steps.index(preflight) < min(
        steps.index(_step_named(steps, name)) for name in native_work_steps
    )
    assert max(steps.index(_step_named(steps, name)) for name in native_work_steps) < steps.index(
        publish
    )
    assert steps.index(publish) < steps.index(verify_before_attestation)
    assert steps.index(verify_before_attestation) < steps.index(attest)
    assert steps.index(attest) < steps.index(copy_bundle)
    assert steps.index(copy_bundle) < steps.index(reverify)
    assert steps.index(reverify) < steps.index(success_upload)
    assert steps.index(success_upload) == len(steps) - 1


def test_measurement_workflow_has_one_signed_native_work_job() -> None:
    _assert_native_workflow_contract(
        _load(),
        phase="measurement",
        timeout_minutes=180,
        native_work_steps=(
            "Reacquire authenticated checkpoint",
            "Rebuild and import the accepted image identity",
            "Measure the accepted authority",
        ),
    )
