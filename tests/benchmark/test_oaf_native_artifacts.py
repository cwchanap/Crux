from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.benchmark.backend_attestation import HostNumericFingerprint
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex, strict_json_loads
from src.benchmark.backend_publication import read_regular_file_no_follow
from tools.hpa320 import oaf_host_attestation as host_module
from tools.hpa320 import oaf_native_artifacts as artifacts_module
from tools.hpa320.oaf_host_attestation import publish_github_host_attestation

BOOTSTRAP_FILES = {
    "bootstrap-host-attestation/attestation-bundle.json": "native_host_attestation_bundle",
    "bootstrap-host-attestation/native-host-evidence.json": "native_host_evidence",
    "bootstrap-host-attestation/native-host-observation.json": "native_host_observation",
    "checkpoint-acquisition-evidence.json": "checkpoint_acquisition_evidence",
    "calibration-image/calibration-bootstrap-evidence.json": "calibration_bootstrap_evidence",
    "calibration-image/oci-layout-manifest.json": "oci_layout_manifest",
    "calibration-image/runtime.oci.tar": "oci_layout_archive",
    "base-system-package-evidence.json": "base_system_package_evidence",
}
MEASUREMENT_FILES = {
    "measurement-host-attestation/attestation-bundle.json": "native_host_attestation_bundle",
    "measurement-host-attestation/native-host-evidence.json": "native_host_evidence",
    "measurement-host-attestation/native-host-observation.json": "native_host_observation",
    "operational-checkpoint-acquisition-evidence.json": "checkpoint_acquisition_evidence",
    "operational-image/calibration-bootstrap-evidence.json": "calibration_bootstrap_evidence",
    "operational-image/oci-layout-manifest.json": "oci_layout_manifest",
    "operational-image/runtime.oci.tar": "oci_layout_archive",
    "calibration-measurement-evidence.json": "calibration_measurement_evidence",
}
CANDIDATE_FILES = {
    "candidate-host-attestation/attestation-bundle.json": "native_host_attestation_bundle",
    "candidate-host-attestation/native-host-evidence.json": "native_host_evidence",
    "candidate-host-attestation/native-host-observation.json": "native_host_observation",
    "operational-checkpoint-acquisition-evidence.json": "checkpoint_acquisition_evidence",
    "operational-image/calibration-bootstrap-evidence.json": "calibration_bootstrap_evidence",
    "operational-image/oci-layout-manifest.json": "oci_layout_manifest",
    "operational-image/runtime.oci.tar": "oci_layout_archive",
    "seal-candidate/candidate-manifest.json": "seal_candidate_manifest",
    "seal-candidate/docs/superpowers/evidence/hpa-320/legacy-conversion-audit.json": (
        "conversion_audit"
    ),
    "seal-candidate/docs/superpowers/evidence/hpa-320/native/"
    "candidate-host-attestation/attestation-bundle.json": "native_host_attestation_bundle",
    "seal-candidate/docs/superpowers/evidence/hpa-320/native/"
    "candidate-host-attestation/native-host-evidence.json": "native_host_evidence",
    "seal-candidate/docs/superpowers/evidence/hpa-320/native/"
    "candidate-host-attestation/native-host-observation.json": "native_host_observation",
    "seal-candidate/runtime/oaf_tf1/host-adapter-source-manifest.json": (
        "host_adapter_source_manifest"
    ),
    "seal-candidate/docs/superpowers/evidence/hpa-320/oaf-tensor-coverage.json": (
        "tensor_coverage"
    ),
    "seal-candidate/docs/superpowers/evidence/hpa-320/oaf-advisory-snapshot.json": (
        "advisory_snapshot"
    ),
    "seal-candidate/docs/superpowers/evidence/hpa-320/oaf-security-scan.json": "security_scan",
    "seal-candidate/artifacts/benchmark/backends/oaf-tf1/runtime.oci.tar": "oci_layout_archive",
    "seal-candidate/docs/superpowers/evidence/hpa-320/oaf-oci-layout-manifest.json": (
        "oci_layout_manifest"
    ),
    "seal-candidate/tests/fixtures/oaf_tf1_smoke/canonical.wav": "smoke_audio",
    "seal-candidate/docs/superpowers/evidence/hpa-320/oaf-smoke-prediction.jsonl": (
        "smoke_prediction"
    ),
    "seal-candidate/tests/fixtures/oaf_tf1_smoke/smoke-oracle.json": "smoke_oracle",
    "seal-candidate/config/benchmark/backends/"
    "magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json": "seal_evidence",
    "seal-candidate/config/benchmark/backends/"
    "magenta-egmd-tf1-94529798-8hit-v1.runtime-lock.json": "runtime_lock",
    "seal-candidate/config/benchmark/backends/"
    "magenta-egmd-tf1-94529798-8hit-v1.backend-lock.json": "backend_lock",
}
PHASE_HOST_BUNDLE_PATHS = {
    "bootstrap": "bootstrap-host-attestation/attestation-bundle.json",
    "measurement": "measurement-host-attestation/attestation-bundle.json",
    "candidate": "candidate-host-attestation/attestation-bundle.json",
}


def _set_bootstrap_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "a" * 40
    environment = {
        "COMMIT_SHA": commit,
        "GITHUB_JOB": "native-bootstrap",
        "GITHUB_REPOSITORY": "cwchanap/Crux",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": commit,
        "GITHUB_WORKFLOW_REF": (
            "cwchanap/Crux/.github/workflows/hpa320-native-bootstrap.yml@refs/tags/native-v2"
        ),
        "RUNNER_ARCH": "X64",
        "RUNNER_ARCH_CONTEXT": "X64",
        "RUNNER_ENVIRONMENT": "github-hosted",
        "RUNNER_ENVIRONMENT_CONTEXT": "github-hosted",
        "RUNNER_OS": "Linux",
        "RUNNER_OS_CONTEXT": "Linux",
        "WORKFLOW_SOURCE_SHA": commit,
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        host_module,
        "_run_text",
        lambda command, _label: commit if command[0] == "git" else "x86_64",
    )
    monkeypatch.setattr(
        host_module,
        "_docker_info",
        {
            "OSType": "linux",
            "Architecture": "x86_64",
            "ServerVersion": "28.3.2",
        }.__getitem__,
    )
    monkeypatch.setattr(
        host_module,
        "collect_host_numeric_fingerprint",
        lambda: HostNumericFingerprint(
            architecture="x86_64",
            cpu_vendor_id="GenuineIntel",
            cpu_family="6",
            cpu_model="143",
            cpu_stepping="8",
        ),
    )


@pytest.fixture
def bootstrap_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    _set_bootstrap_environment(monkeypatch)
    root = tmp_path / "hpa320-bootstrap"
    host_directory = root / "bootstrap-host-attestation"
    publish_github_host_attestation(
        phase="bootstrap",
        output_directory=host_directory,
    )
    for relative in BOOTSTRAP_FILES:
        path = root / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes({"fixture_path": relative}, trailing_newline=True))
    return SimpleNamespace(
        root=root,
        host_bundle=host_directory / "attestation-bundle.json",
    )


def _publish(payload: Any) -> Any:
    return artifacts_module.publish_native_work_manifest(
        phase="bootstrap",
        payload_root=payload.root,
        host_bundle_path=payload.host_bundle,
    )


def _manifest_payload(path: Path) -> dict[str, Any]:
    content = read_regular_file_no_follow(path)
    assert content.endswith(b"\n")
    payload = strict_json_loads(content[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    return payload


def _rewrite_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))


def test_phase_mappings_are_the_exact_immutable_inventories() -> None:
    assert dict(artifacts_module.BOOTSTRAP_FILES) == BOOTSTRAP_FILES
    assert dict(artifacts_module.MEASUREMENT_FILES) == MEASUREMENT_FILES
    assert dict(artifacts_module.CANDIDATE_FILES) == CANDIDATE_FILES
    assert {phase: dict(files) for phase, files in artifacts_module.PHASE_FILES.items()} == {
        "bootstrap": BOOTSTRAP_FILES,
        "measurement": MEASUREMENT_FILES,
        "candidate": CANDIDATE_FILES,
        "diagnostic": {},
    }
    assert dict(artifacts_module.PHASE_HOST_BUNDLE_PATHS) == PHASE_HOST_BUNDLE_PATHS
    assert all("github-job-api-record" not in path for path in artifacts_module.CANDIDATE_FILES)


def test_publish_manifest_hashes_every_allowlisted_payload_file(
    bootstrap_payload: Any,
) -> None:
    manifest = _publish(bootstrap_payload)
    payload = _manifest_payload(bootstrap_payload.root / "artifact-manifest.json")

    assert payload["schema"] == artifacts_module.MANIFEST_SCHEMA
    assert payload["status"] == "success"
    assert [(row["role"], row["path"]) for row in payload["files"]] == sorted(
        [(role, path) for path, role in BOOTSTRAP_FILES.items()],
        key=lambda item: (item[0].encode(), item[1].encode()),
    )
    assert manifest.sha256 == sha256_hex(
        read_regular_file_no_follow(bootstrap_payload.root / "artifact-manifest.json")
    )
    artifacts_module.verify_native_work_payload(
        payload_root=bootstrap_payload.root,
        manifest=manifest,
    )


def test_manifest_publisher_rejects_an_unlisted_payload_file(
    bootstrap_payload: Any,
) -> None:
    (bootstrap_payload.root / "unexpected.json").write_bytes(b"{}\n")

    with pytest.raises(artifacts_module.NativeArtifactError):
        _publish(bootstrap_payload)


def test_manifest_publisher_rejects_a_missing_allowlisted_file(
    bootstrap_payload: Any,
) -> None:
    (bootstrap_payload.root / "checkpoint-acquisition-evidence.json").unlink()

    with pytest.raises(artifacts_module.NativeArtifactError):
        _publish(bootstrap_payload)


def test_manifest_publisher_rejects_an_empty_file_without_publishing_a_manifest(
    bootstrap_payload: Any,
) -> None:
    (bootstrap_payload.root / "checkpoint-acquisition-evidence.json").write_bytes(b"")

    with pytest.raises(artifacts_module.NativeArtifactError):
        _publish(bootstrap_payload)
    assert not (bootstrap_payload.root / "artifact-manifest.json").exists()


def test_manifest_publisher_rejects_a_symlinked_payload_file(
    bootstrap_payload: Any,
) -> None:
    target = bootstrap_payload.root / "checkpoint-acquisition-evidence.json"
    outside = bootstrap_payload.root.parent / "outside.json"
    outside.write_bytes(b"{}\n")
    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(artifacts_module.NativeArtifactError):
        _publish(bootstrap_payload)


def test_manifest_publisher_rejects_a_symlinked_payload_directory(
    bootstrap_payload: Any,
) -> None:
    directory = bootstrap_payload.root / "calibration-image"
    outside = bootstrap_payload.root.parent / "outside-directory"
    directory.rename(outside)
    directory.symlink_to(outside, target_is_directory=True)

    with pytest.raises(artifacts_module.NativeArtifactError):
        _publish(bootstrap_payload)


def test_manifest_loader_rejects_a_valid_role_at_the_wrong_path(
    bootstrap_payload: Any,
) -> None:
    published = _publish(bootstrap_payload)
    payload = _manifest_payload(published.path)
    payload["files"][0]["role"] = "security_scan"
    _rewrite_manifest(published.path, payload)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.load_native_work_manifest(published.path, expected_phase="bootstrap")


def test_manifest_loader_rejects_an_unknown_role(
    bootstrap_payload: Any,
) -> None:
    published = _publish(bootstrap_payload)
    payload = _manifest_payload(published.path)
    payload["files"][0]["role"] = "unknown_role"
    _rewrite_manifest(published.path, payload)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.load_native_work_manifest(published.path, expected_phase="bootstrap")


def test_manifest_loader_requires_the_host_bundle_row_to_match_its_top_level_digest(
    bootstrap_payload: Any,
) -> None:
    published = _publish(bootstrap_payload)
    payload = _manifest_payload(published.path)
    host_bundle = next(
        row for row in payload["files"] if row["role"] == "native_host_attestation_bundle"
    )
    host_bundle["sha256"] = "b" * 64
    _rewrite_manifest(published.path, payload)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.load_native_work_manifest(published.path, expected_phase="bootstrap")


def test_manifest_loader_rejects_a_duplicate_path(
    bootstrap_payload: Any,
) -> None:
    published = _publish(bootstrap_payload)
    payload = _manifest_payload(published.path)
    payload["files"][1]["path"] = payload["files"][0]["path"]
    _rewrite_manifest(published.path, payload)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.load_native_work_manifest(published.path, expected_phase="bootstrap")


def test_manifest_loader_rejects_rows_out_of_utf8_byte_order(
    bootstrap_payload: Any,
) -> None:
    published = _publish(bootstrap_payload)
    payload = _manifest_payload(published.path)
    payload["files"][0], payload["files"][1] = payload["files"][1], payload["files"][0]
    _rewrite_manifest(published.path, payload)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.load_native_work_manifest(published.path, expected_phase="bootstrap")


@pytest.mark.parametrize(
    "path",
    [
        "artifact-manifest.json",
        "hpa320-native-bootstrap.tar",
        "hpa320-native-bootstrap.sigstore.json",
    ],
    ids=["manifest", "archive", "sigstore"],
)
def test_manifest_loader_rejects_self_references(
    bootstrap_payload: Any,
    path: str,
) -> None:
    published = _publish(bootstrap_payload)
    payload = _manifest_payload(published.path)
    payload["files"][0]["path"] = path
    _rewrite_manifest(published.path, payload)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.load_native_work_manifest(published.path, expected_phase="bootstrap")


@pytest.mark.parametrize(
    "path",
    [
        "./checkpoint-acquisition-evidence.json",
        "../checkpoint-acquisition-evidence.json",
        "/x",
        "x\\y",
    ],
    ids=["dot", "parent", "absolute", "backslash"],
)
def test_manifest_loader_rejects_unsafe_reference_paths(
    bootstrap_payload: Any,
    path: str,
) -> None:
    published = _publish(bootstrap_payload)
    payload = _manifest_payload(published.path)
    payload["files"][0]["path"] = path
    _rewrite_manifest(published.path, payload)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.load_native_work_manifest(published.path, expected_phase="bootstrap")


def test_manifest_payload_verification_rejects_a_changed_file(
    bootstrap_payload: Any,
) -> None:
    published = _publish(bootstrap_payload)
    (bootstrap_payload.root / "checkpoint-acquisition-evidence.json").write_bytes(b"changed\n")

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.verify_native_work_payload(
            payload_root=bootstrap_payload.root,
            manifest=published,
        )


def test_manifest_publisher_rejects_inode_drift_during_a_chunked_read(
    bootstrap_payload: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = bootstrap_payload.root / "checkpoint-acquisition-evidence.json"
    target.write_bytes(b"x" * (2 * 1024 * 1024))
    replaced = False

    def replace_after_first_chunk(path: Path) -> None:
        nonlocal replaced
        if path == target and not replaced:
            replaced = True
            target.write_bytes(b"drifted\n")

    monkeypatch.setattr(artifacts_module, "_SCAN_CHUNK_HOOK", replace_after_first_chunk)

    with pytest.raises(artifacts_module.NativeArtifactError):
        _publish(bootstrap_payload)
    assert not (bootstrap_payload.root / "artifact-manifest.json").exists()


def test_manifest_publisher_rejects_an_aba_host_bundle_replacement(
    bootstrap_payload: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_bundle = read_regular_file_no_follow(bootstrap_payload.host_bundle)
    scanned_non_host = bootstrap_payload.root / "base-system-package-evidence.json"
    host_evidence = bootstrap_payload.root / "bootstrap-host-attestation/native-host-evidence.json"
    alternate_bundle = b"alternate-host-bundle\n"
    replaced = False
    bundle_scanned = False

    def replace_then_restore_host_bundle(path: Path) -> None:
        nonlocal bundle_scanned, replaced
        if path == scanned_non_host:
            bootstrap_payload.host_bundle.write_bytes(alternate_bundle)
            replaced = True
        elif path == bootstrap_payload.host_bundle and replaced:
            bundle_scanned = True
        elif path == host_evidence and bundle_scanned:
            bootstrap_payload.host_bundle.write_bytes(original_bundle)

    monkeypatch.setattr(
        artifacts_module,
        "_SCAN_CHUNK_HOOK",
        replace_then_restore_host_bundle,
    )

    with pytest.raises(artifacts_module.NativeArtifactError):
        _publish(bootstrap_payload)
    assert not (bootstrap_payload.root / "artifact-manifest.json").exists()
