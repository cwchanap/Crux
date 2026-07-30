from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmark.backend_attestation import HostNumericFingerprint
from src.benchmark.backend_identity import canonical_json_bytes
from tools.hpa320 import seal_oaf_backend
from tools.hpa320.github_host_evidence import build_github_hosted_evidence


def test_github_producer_emits_evidence_accepted_by_current_validator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fingerprint = HostNumericFingerprint(
        architecture="x86_64",
        cpu_vendor_id="GenuineIntel",
        cpu_family="6",
        cpu_model="143",
        cpu_stepping="8",
    )
    monkeypatch.setattr(
        "tools.hpa320.github_host_evidence.collect_host_numeric_fingerprint",
        lambda: fingerprint,
    )

    evidence_payload = build_github_hosted_evidence(
        api_record_bytes=b'{"id":123}\n',
        job_id=123,
        run_url="https://github.com/acme/crux/actions/runs/456/job/123",
        workflow_commit="b" * 40,
    )
    evidence_path = tmp_path / "native-host-evidence.json"
    evidence_path.write_bytes(canonical_json_bytes(evidence_payload) + b"\n")
    monkeypatch.setattr(
        seal_oaf_backend,
        "_require_current_native_worker",
        lambda _evidence: None,
    )
    evidence = seal_oaf_backend.load_native_host_evidence(evidence_path)

    assert evidence.host_numeric_fingerprint == fingerprint
    assert evidence.payload["host_numeric_fingerprint"] == {
        "architecture": "x86_64",
        "cpu_vendor_id": "GenuineIntel",
        "cpu_family": "6",
        "cpu_model": "143",
        "cpu_stepping": "8",
    }
    assert evidence.sha256 == evidence_payload["sha256"]
