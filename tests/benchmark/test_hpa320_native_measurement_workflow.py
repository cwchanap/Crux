from pathlib import Path

WORKFLOW = Path(".github/workflows/hpa320-native-measurement.yml")
CHECKOUT = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
SETUP_UV = "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b"
UPLOAD = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"


def _load() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_measurement_workflow_is_exact_commit_native_linux_x64() -> None:
    text = _load()
    dispatch = text.split("  workflow_dispatch:\n", 1)[1].split("\npermissions:\n", 1)[0]
    assert dispatch.count("      commit_sha:\n") == 1
    assert text.count(CHECKOUT) == 2
    assert text.count("ref: ${{ inputs.commit_sha }}") == 2
    assert text.count("runs-on: ubuntu-24.04") == 2
    assert "observe-github" in text
    assert "finalize-github" in text
    assert "--phase measurement" in text


def test_measurement_workflow_rebuilds_before_measuring_accepted_authority() -> None:
    text = _load()
    assert '\nenv:\n  UV_FROZEN: "1"\n\njobs:\n' in text
    assert text.count(SETUP_UV) == 2
    assert text.count('version: "0.11.8"') == 2
    assert text.count('python-version: "3.12"') == 2
    assert text.count("enable-cache: false") == 2
    assert text.count(UPLOAD) == 1
    assert "docker/setup-buildx-action" not in text
    assert "rtk " not in text
    assert "prepare-backend" in text
    assert text.index("bootstrap-image") < text.index("seal_oaf_backend measure")
    assert "calibration-measurement-request.json" in text
    assert "docs/superpowers/evidence/hpa-320/native/calibration-bootstrap-evidence.json" in text
    assert " seal \\" not in text
    assert "git add" not in text
    assert "git commit" not in text
