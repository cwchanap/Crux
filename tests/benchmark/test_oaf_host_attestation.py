from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex
from tools.hpa320.oaf_host_attestation import (
    HostAttestationError,
    load_native_host_attestation_bundle,
)


def _write_canonical(path: Path, payload: object) -> bytes:
    content = canonical_json_bytes(payload, trailing_newline=True)
    path.write_bytes(content)
    return content


def _bundle_fixture(tmp_path: Path, *, phase: str = "bootstrap") -> tuple[Path, dict[str, Path]]:
    directory = tmp_path / "bundle"
    directory.mkdir()
    commit = "b" * 40
    run_id = 456
    job_id = 123
    repository = "acme/crux"
    run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    fingerprint = {
        "architecture": "x86_64",
        "cpu_family": "6",
        "cpu_model": "143",
        "cpu_stepping": "8",
        "cpu_vendor_id": "GenuineIntel",
    }
    api_record = {
        "conclusion": "success",
        "head_sha": commit,
        "html_url": f"{run_url}/job/{job_id}",
        "id": job_id,
        "labels": ["ubuntu-24.04"],
        "name": "observe-native-host",
        "run_id": run_id,
        "status": "completed",
    }
    api_bytes = json.dumps(api_record, separators=(", ", ": ")).encode()
    api_path = directory / "github-job-api-record.json.hex"
    api_content = api_bytes.hex().encode("ascii") + b"\n"
    api_path.write_bytes(api_content)
    evidence_payload = {
        "api_record_sha256": sha256_hex(api_bytes),
        "approved_labels": ["Linux", "X64"],
        "host_numeric_fingerprint": fingerprint,
        "job_id": job_id,
        "run_url": api_record["html_url"],
        "runner_arch": "X64",
        "runner_os": "Linux",
        "workflow_commit": commit,
    }
    evidence_content = _write_canonical(
        directory / "native-host-evidence.json",
        {
            "kind": "github_hosted",
            "official_execution_allowed": True,
            "payload": evidence_payload,
            "sha256": sha256_hex(canonical_json_bytes(evidence_payload)),
        },
    )
    observation_content = _write_canonical(
        directory / "native-host-observation.json",
        {
            "docker_architecture": "x86_64",
            "docker_os_type": "linux",
            "docker_server_version": "28.0.4",
            "github_repository": repository,
            "github_run_attempt": 1,
            "github_run_id": run_id,
            "github_run_url": run_url,
            "github_sha": commit,
            "github_workflow_ref": (
                f"{repository}/.github/workflows/hpa320-native-{phase}.yml@refs/heads/test"
            ),
            "host_numeric_fingerprint": fingerprint,
            "runner_arch": "X64",
            "runner_os": "Linux",
            "uname_architecture": "x86_64",
        },
    )
    identities = {
        "api_record": {
            "name": api_path.name,
            "sha256": hashlib.sha256(api_content).hexdigest(),
            "size": len(api_content),
        },
        "native_host_evidence": {
            "name": "native-host-evidence.json",
            "sha256": hashlib.sha256(evidence_content).hexdigest(),
            "size": len(evidence_content),
        },
        "native_host_observation": {
            "name": "native-host-observation.json",
            "sha256": hashlib.sha256(observation_content).hexdigest(),
            "size": len(observation_content),
        },
    }
    bundle_path = directory / "attestation-bundle.json"
    _write_canonical(
        bundle_path,
        {
            **identities,
            "phase": phase,
            "schema": "crux.oaf-native-host-attestation-bundle/v1",
        },
    )
    return bundle_path, {
        "api": api_path,
        "evidence": directory / "native-host-evidence.json",
        "observation": directory / "native-host-observation.json",
    }


def test_host_attestation_bundle_authenticates_raw_api_and_siblings(tmp_path: Path) -> None:
    bundle_path, paths = _bundle_fixture(tmp_path)

    bundle = load_native_host_attestation_bundle(bundle_path, expected_phase="bootstrap")

    assert bundle.phase == "bootstrap"
    assert bundle.api_record.name == paths["api"].name
    assert bundle.native_host_evidence.name == paths["evidence"].name
    assert bundle.sha256 == hashlib.sha256(bundle_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    ("odd-hex", "uppercase-hex", "internal-newline", "wrong-hash", "wrong-name"),
)
def test_host_attestation_bundle_rejects_corrupt_siblings(
    tmp_path: Path,
    mutation: str,
) -> None:
    bundle_path, paths = _bundle_fixture(tmp_path)
    bundle = json.loads(bundle_path.read_bytes())
    if mutation == "odd-hex":
        paths["api"].write_bytes(paths["api"].read_bytes()[:-2] + b"\n")
    elif mutation == "uppercase-hex":
        paths["api"].write_bytes(paths["api"].read_bytes().upper())
    elif mutation == "internal-newline":
        content = paths["api"].read_bytes()
        paths["api"].write_bytes(content[:10] + b"\n" + content[10:])
    elif mutation == "wrong-hash":
        bundle["native_host_evidence"]["sha256"] = "0" * 64
        _write_canonical(bundle_path, bundle)
    elif mutation == "wrong-name":
        bundle["native_host_observation"]["name"] = "observation.json"
        _write_canonical(bundle_path, bundle)

    with pytest.raises(HostAttestationError, match="hex|hash|name|identity"):
        load_native_host_attestation_bundle(bundle_path, expected_phase="bootstrap")


def test_host_attestation_bundle_rejects_phase_and_api_disagreement(tmp_path: Path) -> None:
    bundle_path, paths = _bundle_fixture(tmp_path)

    with pytest.raises(HostAttestationError, match="phase"):
        load_native_host_attestation_bundle(bundle_path, expected_phase="measurement")

    api_bytes = bytes.fromhex(paths["api"].read_text(encoding="ascii").strip())
    api_record = json.loads(api_bytes)
    api_record["conclusion"] = "failure"
    changed = json.dumps(api_record, separators=(", ", ": ")).encode()
    paths["api"].write_bytes(changed.hex().encode() + b"\n")
    bundle = json.loads(bundle_path.read_bytes())
    api_content = paths["api"].read_bytes()
    bundle["api_record"]["sha256"] = hashlib.sha256(api_content).hexdigest()
    bundle["api_record"]["size"] = len(api_content)
    _write_canonical(bundle_path, bundle)

    with pytest.raises(HostAttestationError, match="successful|completed"):
        load_native_host_attestation_bundle(bundle_path, expected_phase="bootstrap")
