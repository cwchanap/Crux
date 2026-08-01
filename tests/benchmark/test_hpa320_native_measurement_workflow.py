from pathlib import Path

from tests.benchmark._native_workflow_contract import _assert_native_workflow_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPOSITORY_ROOT / ".github/workflows/hpa320-native-measurement.yml"


def test_measurement_workflow_has_one_signed_native_work_job() -> None:
    _assert_native_workflow_contract(
        WORKFLOW,
        phase="measurement",
        timeout_minutes=180,
        native_work_steps=(
            "Reacquire authenticated checkpoint",
            "Rebuild and import the accepted image identity",
            "Measure the accepted authority",
        ),
    )
