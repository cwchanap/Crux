from __future__ import annotations

from pathlib import Path

from src.benchmark.backend_identity import sha256_hex, strict_json_loads
from src.benchmark.backend_process import NativeHostEvidence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPOSITORY_ROOT / ".github/hpa320-native-evidence/job-90511044879"
WORKFLOW_COMMIT = "60be7d9055aa3826ea34fec571512e7d6c66579d"
PRECONTRACT_JOB_ID = 90_329_445_692


def _read_canonical_object(name: str) -> dict[str, object]:
    content = (EVIDENCE_ROOT / name).read_bytes()
    assert content.endswith(b"\n")
    content = content[:-1]
    value = strict_json_loads(content, require_canonical=True)
    assert isinstance(value, dict)
    return value


def _read_api_record() -> tuple[bytes, dict[str, object]]:
    hex_content = (EVIDENCE_ROOT / "github-job-api-record.json.hex").read_text(encoding="ascii")
    assert hex_content.endswith("\n")
    hex_lines = hex_content.splitlines()
    assert hex_content == "\n".join(hex_lines) + "\n"
    hex_record = "".join(hex_lines)
    assert hex_record == hex_record.lower()
    assert hex_record and all(character in "0123456789abcdef" for character in hex_record)
    content = bytes.fromhex(hex_record)
    value = strict_json_loads(content)
    assert isinstance(value, dict)
    return content, value


def test_preserved_native_host_evidence_authenticates_the_accepted_observation() -> None:
    evidence_content = (EVIDENCE_ROOT / "native-host-evidence.json").read_bytes()
    observation_content = (EVIDENCE_ROOT / "native-host-observation.json").read_bytes()
    api_record_content, api_record = _read_api_record()
    evidence_record = _read_canonical_object("native-host-evidence.json")
    observation = _read_canonical_object("native-host-observation.json")

    assert set(evidence_record) == {
        "kind",
        "official_execution_allowed",
        "payload",
        "sha256",
    }
    assert evidence_record["kind"] == "github_hosted"
    assert evidence_record["official_execution_allowed"] is True
    assert isinstance(evidence_record["payload"], dict)
    assert isinstance(evidence_record["sha256"], str)
    evidence = NativeHostEvidence(
        kind="github_hosted",
        payload=evidence_record["payload"],
        sha256=evidence_record["sha256"],
        official_execution_allowed=True,
    )

    payload = evidence.payload
    assert set(payload) == {
        "api_record_sha256",
        "approved_labels",
        "host_numeric_fingerprint",
        "job_id",
        "run_url",
        "runner_arch",
        "runner_os",
        "workflow_commit",
    }
    assert evidence.host_numeric_fingerprint.as_json() == {
        "architecture": "x86_64",
        "cpu_vendor_id": "AuthenticAMD",
        "cpu_family": "25",
        "cpu_model": "1",
        "cpu_stepping": "1",
    }
    assert payload["approved_labels"] == ("Linux", "X64")
    assert payload["runner_os"] == "Linux"
    assert payload["runner_arch"] == "X64"
    assert payload["workflow_commit"] == WORKFLOW_COMMIT
    assert payload["api_record_sha256"] == sha256_hex(api_record_content)
    assert payload["job_id"] == api_record["id"]
    assert payload["job_id"] != PRECONTRACT_JOB_ID
    assert payload["run_url"] == api_record["html_url"]

    assert api_record["run_id"] == observation["github_run_id"]
    assert api_record["head_sha"] == WORKFLOW_COMMIT
    assert observation["github_sha"] == WORKFLOW_COMMIT
    assert observation["github_run_url"] == (
        "https://github.com/cwchanap/Crux/actions/runs/" + str(api_record["run_id"])
    )
    assert observation["host_numeric_fingerprint"] == dict(
        evidence.host_numeric_fingerprint.as_json()
    )
    assert observation["runner_os"] == payload["runner_os"]
    assert observation["runner_arch"] == payload["runner_arch"]
    assert api_record["name"] == "observe-native-host"
    assert api_record["status"] == "completed"
    assert api_record["conclusion"] == "success"
    assert (
        sha256_hex(evidence_content)
        == "1491e7c426e6ee41f2a883ee0a11e71014d0eb58e50801ec4656b5e33326b61d"
    )
    assert (
        sha256_hex(api_record_content)
        == "d7581ded6bb6dbf5bcd3e9c9e6ea8c9d727bd453f80acaa9c12fee7acf5a99d2"
    )
    assert (
        sha256_hex(observation_content)
        == "da55d74f661e287dc68ff2b6524d687cce8721c43056fadba6b686e0c52804cc"
    )
