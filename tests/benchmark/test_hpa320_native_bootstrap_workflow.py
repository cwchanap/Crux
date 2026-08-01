from pathlib import Path

from tests.benchmark._native_workflow_contract import _assert_native_workflow_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPOSITORY_ROOT / ".github/workflows/hpa320-native-bootstrap.yml"


def test_bootstrap_workflow_has_one_signed_native_work_job() -> None:
    _assert_native_workflow_contract(
        WORKFLOW,
        phase="bootstrap",
        timeout_minutes=120,
        native_work_steps=(
            "Acquire authenticated checkpoint",
            "Build the authenticated image twice",
            "Attest the imported base system",
        ),
    )


def test_legacy_native_host_collector_is_absent() -> None:
    assert not (REPOSITORY_ROOT / ".github/workflows/hpa320-native-host-evidence.yml").exists()
