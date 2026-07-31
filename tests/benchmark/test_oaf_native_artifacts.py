from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
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


def test_packer_reproduces_byte_identical_ustar_archives(
    bootstrap_payload: Any,
    tmp_path: Path,
) -> None:
    manifest = artifacts_module.publish_native_work_manifest(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        host_bundle_path=bootstrap_payload.host_bundle,
    )
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"

    artifacts_module.pack_native_work_archive(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        manifest_path=manifest.path,
        archive_path=first,
    )
    artifacts_module.pack_native_work_archive(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        manifest_path=manifest.path,
        archive_path=second,
    )

    content = first.read_bytes()
    assert content == second.read_bytes()
    assert content.endswith(b"\0" * 1024)

    root = f"hpa320-native-bootstrap-{'a' * 40}/"
    directories = sorted(
        {
            f"{root}{directory}/"
            for path in BOOTSTRAP_FILES
            for directory in (
                "/".join(path.split("/")[:index]) for index in range(1, len(path.split("/")))
            )
        },
        key=str.encode,
    )
    expected_members = [
        root,
        *directories,
        *sorted(
            [f"{root}artifact-manifest.json", *(f"{root}{path}" for path in BOOTSTRAP_FILES)],
            key=str.encode,
        ),
    ]
    with tarfile.open(first, mode="r:") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == [
        member.rstrip("/") for member in expected_members
    ]
    raw_member_names = []
    for start, _ in _tar_member_spans(content):
        header = content[start : start + 512]
        name = header[:100].split(b"\0", 1)[0]
        prefix = header[345:500].split(b"\0", 1)[0]
        raw_member_names.append((prefix + (b"/" if prefix else b"") + name).decode("utf-8"))
    assert raw_member_names == expected_members
    assert all(member.uid == member.gid == member.mtime == 0 for member in members)
    assert all(member.uname == member.gname == "" for member in members)
    assert all(member.pax_headers == {} for member in members)
    assert all(member.mode == (0o755 if member.isdir() else 0o644) for member in members)
    assert all(member.isdir() or member.isfile() for member in members)


@pytest.fixture
def packed_bootstrap(bootstrap_payload: Any, tmp_path: Path) -> Any:
    published = _publish(bootstrap_payload)
    archive = tmp_path / f"hpa320-native-bootstrap-{'a' * 40}.tar"
    artifacts_module.pack_native_work_archive(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        manifest_path=published.path,
        archive_path=archive,
    )
    return artifacts_module.PackedBootstrap(
        payload_root=bootstrap_payload.root,
        manifest=published.path,
        archive=archive,
    )


@dataclass(frozen=True)
class SignedBootstrap:
    manifest: Path
    archive: Path
    bundle: Path
    trusted_root: Path
    verification_json: bytes


@pytest.fixture
def signed_bootstrap(packed_bootstrap: Any, tmp_path: Path) -> SignedBootstrap:
    bundle = tmp_path / "bootstrap.sigstore.json"
    trusted_root = tmp_path / "trusted-root.json"
    bundle.write_bytes(b'{"mediaType":"application/vnd.dev.sigstore.bundle+json;version=0.3"}\n')
    trusted_root.write_bytes(
        b'{"mediaType":"application/vnd.dev.sigstore.trustedroot+json;version=0.1"}\n'
    )
    subjects = [
        {
            "name": "artifacts/benchmark/backends/hpa320-bootstrap/artifact-manifest.json",
            "digest": {
                "sha256": sha256_hex(read_regular_file_no_follow(packed_bootstrap.manifest))
            },
        },
        {
            "name": (f"artifacts/benchmark/backends/hpa320-native-bootstrap-{'a' * 40}.tar"),
            "digest": {"sha256": sha256_hex(read_regular_file_no_follow(packed_bootstrap.archive))},
        },
    ]
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicate": {},
        "predicateType": "https://slsa.dev/provenance/v1",
        "subject": subjects,
    }
    verification_json = json.dumps(
        [{"verificationResult": {"statement": statement}}],
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return SignedBootstrap(
        manifest=packed_bootstrap.manifest,
        archive=packed_bootstrap.archive,
        bundle=bundle,
        trusted_root=trusted_root,
        verification_json=verification_json,
    )


def test_github_verifier_runs_the_exact_policy_for_both_subjects(
    signed_bootstrap: SignedBootstrap,
) -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...]) -> bytes:
        commands.append(command)
        if command[:2] == ("gh", "version"):
            return b"gh version 2.68.1 (2026-01-01)\n"
        return signed_bootstrap.verification_json

    verified = artifacts_module.verify_github_attestations(
        phase="bootstrap",
        workflow_commit="a" * 40,
        manifest_path=signed_bootstrap.manifest,
        archive_path=signed_bootstrap.archive,
        sigstore_bundle_path=signed_bootstrap.bundle,
        trusted_root_path=signed_bootstrap.trusted_root,
        command_runner=run,
    )

    verify_commands = [
        command for command in commands if command[:3] == ("gh", "attestation", "verify")
    ]
    assert [command[3] for command in verify_commands] == [
        str(signed_bootstrap.manifest),
        str(signed_bootstrap.archive),
    ]
    for command in verify_commands:
        assert command[command.index("--repo") + 1] == "cwchanap/Crux"
        assert command[command.index("--signer-workflow") + 1] == (
            "cwchanap/Crux/.github/workflows/hpa320-native-bootstrap.yml"
        )
        assert command[command.index("--source-digest") + 1] == "a" * 40
        assert command[command.index("--signer-digest") + 1] == "a" * 40
        assert "--deny-self-hosted-runners" in command
        assert command[command.index("--digest-alg") + 1] == "sha256"
        assert command[command.index("--predicate-type") + 1] == ("https://slsa.dev/provenance/v1")
        assert command[command.index("--cert-oidc-issuer") + 1] == (
            "https://token.actions.githubusercontent.com"
        )
        assert command[command.index("--bundle") + 1] == str(signed_bootstrap.bundle)
        assert command[command.index("--custom-trusted-root") + 1] == str(
            signed_bootstrap.trusted_root
        )
        assert command[-2:] == ("--format", "json")
    assert verified.gh_version == "2.68.1"


@pytest.mark.parametrize(
    "version",
    [
        b"gh version 2.67.0 (2025-01-01)\n",
        b"gh version 3.0.0 (2026-01-01)\n",
        b"GitHub CLI 2.68.1\n",
    ],
)
def test_github_verifier_rejects_unsupported_or_malformed_cli_versions(
    signed_bootstrap: SignedBootstrap,
    version: bytes,
) -> None:
    def run(command: tuple[str, ...]) -> bytes:
        if command[:2] == ("gh", "version"):
            return version
        raise AssertionError("verification must not run with an unsupported gh CLI")

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.verify_github_attestations(
            phase="bootstrap",
            workflow_commit="a" * 40,
            manifest_path=signed_bootstrap.manifest,
            archive_path=signed_bootstrap.archive,
            sigstore_bundle_path=signed_bootstrap.bundle,
            trusted_root_path=signed_bootstrap.trusted_root,
            command_runner=run,
        )


def test_github_verifier_rejects_different_verified_statements(
    signed_bootstrap: SignedBootstrap,
) -> None:
    invocation = 0

    def run(command: tuple[str, ...]) -> bytes:
        nonlocal invocation
        if command[:2] == ("gh", "version"):
            return b"gh version 2.68.1 (2026-01-01)\n"
        invocation += 1
        if invocation == 1:
            return signed_bootstrap.verification_json
        value = json.loads(signed_bootstrap.verification_json)
        value[0]["verificationResult"]["statement"]["predicate"]["changed"] = True
        return json.dumps(value).encode("utf-8")

    with pytest.raises(artifacts_module.NativeArtifactError, match="disagree"):
        artifacts_module.verify_github_attestations(
            phase="bootstrap",
            workflow_commit="a" * 40,
            manifest_path=signed_bootstrap.manifest,
            archive_path=signed_bootstrap.archive,
            sigstore_bundle_path=signed_bootstrap.bundle,
            trusted_root_path=signed_bootstrap.trusted_root,
            command_runner=run,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("not-an-array", "one entry"),
        ("missing-statement", "statement is missing"),
        ("missing-manifest", "subjects are invalid"),
        ("missing-archive", "subjects are invalid"),
        ("wrong-manifest-digest", "do not match"),
        ("wrong-archive-digest", "do not match"),
        ("extra-subject", "subjects are invalid"),
    ],
)
def test_github_verifier_rejects_invalid_statement_authority(
    signed_bootstrap: SignedBootstrap,
    mutation: str,
    error: str,
) -> None:
    value: object = json.loads(signed_bootstrap.verification_json)
    if mutation == "not-an-array":
        value = {"verificationResult": {}}
    elif mutation == "missing-statement":
        assert isinstance(value, list)
        value[0]["verificationResult"] = {}
    else:
        assert isinstance(value, list)
        subjects = value[0]["verificationResult"]["statement"]["subject"]
        if mutation == "missing-manifest":
            subjects.pop(0)
        elif mutation == "missing-archive":
            subjects.pop(1)
        elif mutation == "wrong-manifest-digest":
            subjects[0]["digest"]["sha256"] = "0" * 64
        elif mutation == "wrong-archive-digest":
            subjects[1]["digest"]["sha256"] = "0" * 64
        else:
            subjects.append({"name": "diagnostic.json", "digest": {"sha256": "f" * 64}})
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")

    def run(command: tuple[str, ...]) -> bytes:
        if command[:2] == ("gh", "version"):
            return b"gh version 2.68.1 (2026-01-01)\n"
        return encoded

    with pytest.raises(artifacts_module.NativeArtifactError, match=error):
        artifacts_module.verify_github_attestations(
            phase="bootstrap",
            workflow_commit="a" * 40,
            manifest_path=signed_bootstrap.manifest,
            archive_path=signed_bootstrap.archive,
            sigstore_bundle_path=signed_bootstrap.bundle,
            trusted_root_path=signed_bootstrap.trusted_root,
            command_runner=run,
        )


def test_github_verifier_rejects_command_failure(
    signed_bootstrap: SignedBootstrap,
) -> None:
    def run(command: tuple[str, ...]) -> bytes:
        if command[:2] == ("gh", "version"):
            return b"gh version 2.68.1 (2026-01-01)\n"
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(artifacts_module.NativeArtifactError, match="verification failed"):
        artifacts_module.verify_github_attestations(
            phase="bootstrap",
            workflow_commit="a" * 40,
            manifest_path=signed_bootstrap.manifest,
            archive_path=signed_bootstrap.archive,
            sigstore_bundle_path=signed_bootstrap.bundle,
            trusted_root_path=signed_bootstrap.trusted_root,
            command_runner=run,
        )


@pytest.mark.parametrize("rejection", ["wrong-workflow", "wrong-commit", "self-hosted"])
def test_github_verifier_rejects_gh_policy_failures(
    signed_bootstrap: SignedBootstrap,
    rejection: str,
) -> None:
    def run(command: tuple[str, ...]) -> bytes:
        if command[:2] == ("gh", "version"):
            return b"gh version 2.68.1 (2026-01-01)\n"
        raise subprocess.CalledProcessError(1, command, stderr=rejection.encode("utf-8"))

    with pytest.raises(artifacts_module.NativeArtifactError, match="verification failed"):
        artifacts_module.verify_github_attestations(
            phase="bootstrap",
            workflow_commit="a" * 40,
            manifest_path=signed_bootstrap.manifest,
            archive_path=signed_bootstrap.archive,
            sigstore_bundle_path=signed_bootstrap.bundle,
            trusted_root_path=signed_bootstrap.trusted_root,
            command_runner=run,
        )


def test_github_verifier_rejects_signed_failed_result_under_success_subjects(
    signed_bootstrap: SignedBootstrap,
) -> None:
    value = json.loads(signed_bootstrap.verification_json)
    value[0]["verificationResult"]["statement"]["predicate"] = {"status": "failed"}
    failed_result = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")

    def run(command: tuple[str, ...]) -> bytes:
        if command[:2] == ("gh", "version"):
            return b"gh version 2.68.1 (2026-01-01)\n"
        return failed_result

    with pytest.raises(artifacts_module.NativeArtifactError, match="failed result"):
        artifacts_module.verify_github_attestations(
            phase="bootstrap",
            workflow_commit="a" * 40,
            manifest_path=signed_bootstrap.manifest,
            archive_path=signed_bootstrap.archive,
            sigstore_bundle_path=signed_bootstrap.bundle,
            trusted_root_path=signed_bootstrap.trusted_root,
            command_runner=run,
        )


def test_verify_attestation_cli_uses_one_immutable_snapshot_across_gates(
    packed_bootstrap: artifacts_module.PackedBootstrap,
    signed_bootstrap: SignedBootstrap,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_before = signed_bootstrap.manifest.read_bytes()
    archive_before = signed_bootstrap.archive.read_bytes()
    payload_member = packed_bootstrap.payload_root / "checkpoint-acquisition-evidence.json"
    payload_before = payload_member.read_bytes()
    calls: list[str] = []

    def verify(**kwargs: object) -> artifacts_module.VerifiedGitHubAttestation:
        manifest = Path(str(kwargs["manifest_path"]))
        archive = Path(str(kwargs["archive_path"]))
        assert manifest != signed_bootstrap.manifest
        assert archive != signed_bootstrap.archive
        assert manifest.read_bytes() == manifest_before
        assert archive.read_bytes() == archive_before
        signed_bootstrap.manifest.write_bytes(b"coherent replacement manifest\n")
        signed_bootstrap.archive.write_bytes(b"coherent replacement archive\n")
        payload_member.write_bytes(b"coherent replacement payload\n")
        calls.append("github")
        return artifacts_module.VerifiedGitHubAttestation(
            gh_version="2.68.1",
            statement={"subject": []},
            statement_sha256="a" * 64,
            subjects=(
                artifacts_module.CheckpointIdentity(
                    name=("artifacts/benchmark/backends/hpa320-bootstrap/artifact-manifest.json"),
                    sha256=sha256_hex(manifest_before),
                    size=len(manifest_before),
                ),
                artifacts_module.CheckpointIdentity(
                    name=(f"artifacts/benchmark/backends/hpa320-native-bootstrap-{'a' * 40}.tar"),
                    sha256=sha256_hex(archive_before),
                    size=len(archive_before),
                ),
            ),
        )

    def load(path: Path, **_kwargs: object) -> object:
        assert path.read_bytes() == manifest_before
        calls.append("manifest")
        return object()

    def archive(**kwargs: object) -> artifacts_module.CheckpointIdentity:
        assert Path(str(kwargs["archive_path"])).read_bytes() == archive_before
        assert (
            Path(str(kwargs["payload_root"])) / "checkpoint-acquisition-evidence.json"
        ).read_bytes() == payload_before
        calls.append("archive")
        return artifacts_module.CheckpointIdentity(
            "archive", sha256_hex(archive_before), len(archive_before)
        )

    def phase(**kwargs: object) -> None:
        snapshot_member = Path(str(kwargs["payload_root"])) / "checkpoint-acquisition-evidence.json"
        assert snapshot_member.read_bytes() == payload_before
        signed_bootstrap.manifest.write_bytes(manifest_before)
        signed_bootstrap.archive.write_bytes(archive_before)
        payload_member.write_bytes(payload_before)
        calls.append("phase")

    monkeypatch.setattr(artifacts_module, "verify_github_attestations", verify)
    monkeypatch.setattr(artifacts_module, "load_native_work_manifest", load)
    monkeypatch.setattr(artifacts_module, "verify_native_work_archive", archive)
    monkeypatch.setattr(
        artifacts_module,
        "verify_native_work_payload",
        lambda **_kwargs: calls.append("payload"),
    )
    monkeypatch.setattr(artifacts_module, "_validate_native_work_phase", phase)

    assert (
        artifacts_module.main(
            [
                "verify-attestation",
                "--phase",
                "bootstrap",
                "--workflow-commit",
                "a" * 40,
                "--payload-root",
                str(packed_bootstrap.payload_root),
                "--archive",
                str(signed_bootstrap.archive),
                "--bundle",
                str(signed_bootstrap.bundle),
                "--trusted-root",
                str(signed_bootstrap.trusted_root),
                "--repository-root",
                str(tmp_path / "repository"),
            ]
        )
        == 0
    )
    assert calls == ["github", "manifest", "archive", "phase", "manifest", "payload", "archive"]


@pytest.mark.parametrize("entry_kind", ["regular", "directory", "symlink", "diagnostic"])
def test_attestation_snapshot_rejects_unallowlisted_source_entries(
    packed_bootstrap: artifacts_module.PackedBootstrap,
    entry_kind: str,
) -> None:
    root = packed_bootstrap.payload_root
    if entry_kind == "regular":
        (root / "unallowlisted.json").write_bytes(b"unexpected\n")
    elif entry_kind == "directory":
        (root / "unallowlisted").mkdir()
    elif entry_kind == "symlink":
        (root / "unallowlisted-link").symlink_to(root / "checkpoint-acquisition-evidence.json")
    else:
        (root / "diagnostic.json").write_bytes(b"diagnostic\n")

    with pytest.raises(artifacts_module.NativeArtifactError):
        with artifacts_module._attestation_snapshot(
            phase="bootstrap",
            payload_root=root,
            archive_path=packed_bootstrap.archive,
        ):
            pass


def test_verify_attestation_cli_rejects_coherent_snapshot_replacement_after_archive_verification(
    packed_bootstrap: artifacts_module.PackedBootstrap,
    signed_bootstrap: SignedBootstrap,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_archive = signed_bootstrap.archive.read_bytes()
    replacement = tmp_path / "coherent-replacement"
    shutil.copytree(packed_bootstrap.payload_root, replacement)
    (replacement / "artifact-manifest.json").unlink()
    (replacement / "checkpoint-acquisition-evidence.json").write_bytes(b"replacement payload\n")
    replacement_manifest = artifacts_module.publish_native_work_manifest(
        phase="bootstrap",
        payload_root=replacement,
        host_bundle_path=replacement / "bootstrap-host-attestation/attestation-bundle.json",
    )
    replacement_archive = tmp_path / "coherent-replacement.tar"
    artifacts_module.pack_native_work_archive(
        phase="bootstrap",
        payload_root=replacement,
        manifest_path=replacement_manifest.path,
        archive_path=replacement_archive,
    )
    snapshot_archive: Path | None = None

    def verify(**kwargs: object) -> artifacts_module.VerifiedGitHubAttestation:
        nonlocal snapshot_archive
        manifest = Path(str(kwargs["manifest_path"]))
        snapshot_archive = Path(str(kwargs["archive_path"]))
        return artifacts_module.VerifiedGitHubAttestation(
            gh_version="2.68.1",
            statement={"subject": []},
            statement_sha256="a" * 64,
            subjects=(
                artifacts_module.CheckpointIdentity(
                    name="artifacts/benchmark/backends/hpa320-bootstrap/artifact-manifest.json",
                    sha256=sha256_hex(manifest.read_bytes()),
                    size=manifest.stat().st_size,
                ),
                artifacts_module.CheckpointIdentity(
                    name=(f"artifacts/benchmark/backends/hpa320-native-bootstrap-{'a' * 40}.tar"),
                    sha256=sha256_hex(snapshot_archive.read_bytes()),
                    size=snapshot_archive.stat().st_size,
                ),
            ),
        )

    def phase(**kwargs: object) -> None:
        snapshot_root = Path(str(kwargs["payload_root"]))
        assert snapshot_archive is not None
        for directory, _directories, files in os.walk(snapshot_root):
            os.chmod(directory, 0o700)
            for name in files:
                os.chmod(Path(directory) / name, 0o600)
        for source in replacement.rglob("*"):
            if source.is_file():
                target = snapshot_root / source.relative_to(replacement)
                shutil.copyfile(source, target)
                os.chmod(target, 0o400)
        os.chmod(snapshot_archive, 0o600)
        shutil.copyfile(replacement_archive, snapshot_archive)
        os.chmod(snapshot_archive, 0o400)

    monkeypatch.setattr(artifacts_module, "verify_github_attestations", verify)
    monkeypatch.setattr(
        artifacts_module, "load_native_work_manifest", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(artifacts_module, "verify_native_work_payload", lambda **_kwargs: None)
    monkeypatch.setattr(
        artifacts_module,
        "verify_native_work_archive",
        lambda **_kwargs: artifacts_module.CheckpointIdentity(
            "archive", sha256_hex(original_archive), len(original_archive)
        ),
    )
    monkeypatch.setattr(artifacts_module, "_validate_native_work_phase", phase)

    assert (
        artifacts_module.main(
            [
                "verify-attestation",
                "--phase",
                "bootstrap",
                "--workflow-commit",
                "a" * 40,
                "--payload-root",
                str(packed_bootstrap.payload_root),
                "--archive",
                str(signed_bootstrap.archive),
                "--bundle",
                str(signed_bootstrap.bundle),
                "--trusted-root",
                str(signed_bootstrap.trusted_root),
                "--repository-root",
                str(tmp_path / "repository"),
            ]
        )
        == 2
    )
    assert capsys.readouterr().out == ""


def test_verify_attestation_cli_rejects_snapshot_aba_after_archive_verification(
    packed_bootstrap: artifacts_module.PackedBootstrap,
    signed_bootstrap: SignedBootstrap,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_archive = signed_bootstrap.archive.read_bytes()

    def verify(**kwargs: object) -> artifacts_module.VerifiedGitHubAttestation:
        manifest = Path(str(kwargs["manifest_path"]))
        archive = Path(str(kwargs["archive_path"]))
        return artifacts_module.VerifiedGitHubAttestation(
            gh_version="2.68.1",
            statement={"subject": []},
            statement_sha256="a" * 64,
            subjects=(
                artifacts_module.CheckpointIdentity(
                    name="artifacts/benchmark/backends/hpa320-bootstrap/artifact-manifest.json",
                    sha256=sha256_hex(manifest.read_bytes()),
                    size=manifest.stat().st_size,
                ),
                artifacts_module.CheckpointIdentity(
                    name=(f"artifacts/benchmark/backends/hpa320-native-bootstrap-{'a' * 40}.tar"),
                    sha256=sha256_hex(archive.read_bytes()),
                    size=archive.stat().st_size,
                ),
            ),
        )

    def phase(**kwargs: object) -> None:
        member = Path(str(kwargs["payload_root"])) / "checkpoint-acquisition-evidence.json"
        before = member.stat()
        original = member.read_bytes()
        os.chmod(member, 0o600)
        member.write_bytes(b"transient mutation\n")
        member.write_bytes(original)
        os.utime(member, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.chmod(member, 0o400)

    monkeypatch.setattr(artifacts_module, "verify_github_attestations", verify)
    monkeypatch.setattr(
        artifacts_module, "load_native_work_manifest", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(artifacts_module, "verify_native_work_payload", lambda **_kwargs: None)
    monkeypatch.setattr(
        artifacts_module,
        "verify_native_work_archive",
        lambda **_kwargs: artifacts_module.CheckpointIdentity(
            "archive", sha256_hex(original_archive), len(original_archive)
        ),
    )
    monkeypatch.setattr(artifacts_module, "_validate_native_work_phase", phase)

    assert (
        artifacts_module.main(
            [
                "verify-attestation",
                "--phase",
                "bootstrap",
                "--workflow-commit",
                "a" * 40,
                "--payload-root",
                str(packed_bootstrap.payload_root),
                "--archive",
                str(signed_bootstrap.archive),
                "--bundle",
                str(signed_bootstrap.bundle),
                "--trusted-root",
                str(signed_bootstrap.trusted_root),
                "--repository-root",
                str(tmp_path / "repository"),
            ]
        )
        == 2
    )
    assert capsys.readouterr().out == ""


def test_verify_attestation_cli_emits_no_report_when_snapshot_cleanup_fails(
    packed_bootstrap: artifacts_module.PackedBootstrap,
    signed_bootstrap: SignedBootstrap,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingTemporaryDirectory:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.name = str(tmp_path / "failing-cleanup")
            Path(self.name).mkdir()

        def __enter__(self) -> str:
            return self.name

        def __exit__(self, *_args: object) -> None:
            self.cleanup()

        def cleanup(self) -> None:
            raise OSError("cleanup failed")

    def verify(**kwargs: object) -> artifacts_module.VerifiedGitHubAttestation:
        manifest = Path(str(kwargs["manifest_path"]))
        archive = Path(str(kwargs["archive_path"]))
        return artifacts_module.VerifiedGitHubAttestation(
            gh_version="2.68.1",
            statement={"subject": []},
            statement_sha256="a" * 64,
            subjects=(
                artifacts_module.CheckpointIdentity(
                    name="artifacts/benchmark/backends/hpa320-bootstrap/artifact-manifest.json",
                    sha256=sha256_hex(manifest.read_bytes()),
                    size=manifest.stat().st_size,
                ),
                artifacts_module.CheckpointIdentity(
                    name=(f"artifacts/benchmark/backends/hpa320-native-bootstrap-{'a' * 40}.tar"),
                    sha256=sha256_hex(archive.read_bytes()),
                    size=archive.stat().st_size,
                ),
            ),
        )

    monkeypatch.setattr(artifacts_module.tempfile, "TemporaryDirectory", FailingTemporaryDirectory)
    monkeypatch.setattr(artifacts_module, "verify_github_attestations", verify)
    monkeypatch.setattr(artifacts_module, "_validate_native_work_phase", lambda **_kwargs: None)

    assert (
        artifacts_module.main(
            [
                "verify-attestation",
                "--phase",
                "bootstrap",
                "--workflow-commit",
                "a" * 40,
                "--payload-root",
                str(packed_bootstrap.payload_root),
                "--archive",
                str(signed_bootstrap.archive),
                "--bundle",
                str(signed_bootstrap.bundle),
                "--trusted-root",
                str(signed_bootstrap.trusted_root),
                "--repository-root",
                str(tmp_path / "repository"),
            ]
        )
        == 2
    )
    assert capsys.readouterr().out == ""


def test_verify_attestation_cli_runs_the_outside_in_acceptance_order(
    monkeypatch: pytest.MonkeyPatch,
    packed_bootstrap: artifacts_module.PackedBootstrap,
    signed_bootstrap: SignedBootstrap,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    manifest_identity = artifacts_module.CheckpointIdentity(
        name="artifacts/benchmark/backends/hpa320-bootstrap/artifact-manifest.json",
        sha256=sha256_hex(read_regular_file_no_follow(signed_bootstrap.manifest)),
        size=signed_bootstrap.manifest.stat().st_size,
    )
    archive_identity = artifacts_module.CheckpointIdentity(
        name=(f"artifacts/benchmark/backends/hpa320-native-bootstrap-{'a' * 40}.tar"),
        sha256=sha256_hex(read_regular_file_no_follow(signed_bootstrap.archive)),
        size=signed_bootstrap.archive.stat().st_size,
    )
    verified = artifacts_module.VerifiedGitHubAttestation(
        gh_version="2.68.1",
        statement={"subject": []},
        statement_sha256="a" * 64,
        subjects=(manifest_identity, archive_identity),
    )

    def verify(**_kwargs: object) -> artifacts_module.VerifiedGitHubAttestation:
        calls.append("github")
        return verified

    def load(*_args: object, **_kwargs: object) -> object:
        calls.append("manifest")
        return object()

    def archive(**_kwargs: object) -> artifacts_module.CheckpointIdentity:
        calls.append("archive")
        return archive_identity

    def phase(**_kwargs: object) -> None:
        calls.append("phase")

    monkeypatch.setattr(artifacts_module, "verify_github_attestations", verify)
    monkeypatch.setattr(artifacts_module, "load_native_work_manifest", load)
    monkeypatch.setattr(artifacts_module, "verify_native_work_archive", archive)
    monkeypatch.setattr(
        artifacts_module,
        "verify_native_work_payload",
        lambda **_kwargs: calls.append("payload"),
    )
    monkeypatch.setattr(artifacts_module, "_validate_native_work_phase", phase)

    assert (
        artifacts_module.main(
            [
                "verify-attestation",
                "--phase",
                "bootstrap",
                "--workflow-commit",
                "a" * 40,
                "--payload-root",
                str(packed_bootstrap.payload_root),
                "--archive",
                str(signed_bootstrap.archive),
                "--bundle",
                str(signed_bootstrap.bundle),
                "--trusted-root",
                str(signed_bootstrap.trusted_root),
                "--repository-root",
                str(packed_bootstrap.payload_root.parent / "repository"),
            ]
        )
        == 0
    )
    assert calls == ["github", "manifest", "archive", "phase", "manifest", "payload", "archive"]
    report = strict_json_loads(capsys.readouterr().out.encode("utf-8")[:-1], require_canonical=True)
    assert report == {
        "arguments": {
            "archive": str(signed_bootstrap.archive),
            "bundle": str(signed_bootstrap.bundle),
            "payload_root": str(packed_bootstrap.payload_root),
            "phase": "bootstrap",
            "repository_root": str(packed_bootstrap.payload_root.parent / "repository"),
            "trusted_root": str(signed_bootstrap.trusted_root),
            "workflow_commit": "a" * 40,
        },
        "gh_version": "2.68.1",
        "statement_sha256": "a" * 64,
        "subjects": [
            {
                "name": manifest_identity.name,
                "sha256": manifest_identity.sha256,
                "size": manifest_identity.size,
            },
            {
                "name": archive_identity.name,
                "sha256": archive_identity.sha256,
                "size": archive_identity.size,
            },
        ],
    }


def _tar_member_spans(content: bytes) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    while content[offset : offset + 512] != b"\0" * 512:
        header = content[offset : offset + 512]
        assert len(header) == 512
        size = int(header[124:136].split(b"\0", 1)[0], 8)
        end = offset + 512 + ((size + 511) // 512) * 512
        spans.append((offset, end))
        offset = end
    assert content[offset:] == b"\0" * 1024
    return spans


def _rewrite_tar_checksum(header: bytearray) -> None:
    header[148:156] = b" " * 8
    header[148:156] = f"{sum(header):06o}\0 ".encode("ascii")


def _set_tar_octal(header: bytearray, offset: int, width: int, value: int) -> None:
    header[offset : offset + width] = f"{value:0{width - 1}o}\0".encode("ascii")


def _mutate_tar_member(
    content: bytes,
    index: int,
    mutate: Any,
) -> bytes:
    start, _ = _tar_member_spans(content)[index]
    result = bytearray(content)
    header = result[start : start + 512]
    mutate(header)
    _rewrite_tar_checksum(header)
    result[start : start + 512] = header
    return bytes(result)


def _verify_packed_archive(packed_bootstrap: Any, archive: Path) -> None:
    artifacts_module.verify_native_work_archive(
        phase="bootstrap",
        payload_root=packed_bootstrap.payload_root,
        manifest_path=packed_bootstrap.manifest,
        archive_path=archive,
    )


@pytest.mark.parametrize(
    "name",
    [
        b"x" * 100,
        b"p" * 155 + b"/" + b"leaf",
    ],
    ids=["full-name-field", "full-prefix-field"],
)
def test_ustar_parser_accepts_exactly_full_name_or_prefix_fields(name: bytes) -> None:
    header = artifacts_module._ustar_header(
        name=name,
        mode=0o644,
        size=1,
        typeflag=b"0",
    )

    artifacts_module._parse_ustar_header(header)


def test_archive_verifier_rejects_changed_archive_bytes(
    packed_bootstrap: Any,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "changed.tar"
    content = bytearray(read_regular_file_no_follow(packed_bootstrap.archive))
    content[520] ^= 1
    archive.write_bytes(content)

    with pytest.raises(artifacts_module.NativeArtifactError):
        _verify_packed_archive(packed_bootstrap, archive)


def test_archive_verifier_rejects_a_missing_member(
    packed_bootstrap: Any,
    tmp_path: Path,
) -> None:
    content = read_regular_file_no_follow(packed_bootstrap.archive)
    start, end = _tar_member_spans(content)[-1]
    archive = tmp_path / "missing.tar"
    archive.write_bytes(content[:start] + content[end:])

    with pytest.raises(artifacts_module.NativeArtifactError):
        _verify_packed_archive(packed_bootstrap, archive)


def test_archive_verifier_rejects_a_duplicate_member(
    packed_bootstrap: Any,
    tmp_path: Path,
) -> None:
    content = read_regular_file_no_follow(packed_bootstrap.archive)
    start, end = _tar_member_spans(content)[-1]
    archive = tmp_path / "duplicate.tar"
    archive.write_bytes(content[:-1024] + content[start:end] + content[-1024:])

    with pytest.raises(artifacts_module.NativeArtifactError):
        _verify_packed_archive(packed_bootstrap, archive)


@pytest.mark.parametrize(
    ("label", "index", "mutate"),
    [
        (
            "wrong-root",
            0,
            lambda header: header.__setitem__(slice(0, 100), b"wrong-root/".ljust(100, b"\0")),
        ),
        ("wrong-mode", 0, lambda header: _set_tar_octal(header, 100, 8, 0o700)),
        ("wrong-uid", 0, lambda header: _set_tar_octal(header, 108, 8, 1)),
        ("wrong-gid", 0, lambda header: _set_tar_octal(header, 116, 8, 1)),
        ("wrong-mtime", 0, lambda header: _set_tar_octal(header, 136, 12, 1)),
        (
            "nonempty-owner-group",
            0,
            lambda header: (
                header.__setitem__(slice(265, 297), b"owner".ljust(32, b"\0")),
                header.__setitem__(slice(297, 329), b"group".ljust(32, b"\0")),
            ),
        ),
        ("pax-member", -1, lambda header: header.__setitem__(156, ord("x"))),
        ("symlink", -1, lambda header: header.__setitem__(156, ord("2"))),
        ("hard-link", -1, lambda header: header.__setitem__(156, ord("1"))),
        ("device", -1, lambda header: header.__setitem__(156, ord("3"))),
        ("fifo", -1, lambda header: header.__setitem__(156, ord("6"))),
    ],
)
def test_archive_verifier_rejects_malformed_member_metadata(
    packed_bootstrap: Any,
    tmp_path: Path,
    label: str,
    index: int,
    mutate: Any,
) -> None:
    archive = tmp_path / f"{label}.tar"
    archive.write_bytes(
        _mutate_tar_member(read_regular_file_no_follow(packed_bootstrap.archive), index, mutate)
    )

    with pytest.raises(artifacts_module.NativeArtifactError):
        _verify_packed_archive(packed_bootstrap, archive)


def test_archive_verifier_rejects_members_in_the_wrong_order(
    packed_bootstrap: Any,
    tmp_path: Path,
) -> None:
    content = read_regular_file_no_follow(packed_bootstrap.archive)
    spans = _tar_member_spans(content)
    before = content[: spans[-2][0]]
    penultimate = content[spans[-2][0] : spans[-2][1]]
    final = content[spans[-1][0] : spans[-1][1]]
    archive = tmp_path / "wrong-order.tar"
    archive.write_bytes(before + final + penultimate + content[-1024:])

    with pytest.raises(artifacts_module.NativeArtifactError):
        _verify_packed_archive(packed_bootstrap, archive)


def test_archive_verifier_rejects_trailing_bytes(
    packed_bootstrap: Any,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "trailing.tar"
    archive.write_bytes(read_regular_file_no_follow(packed_bootstrap.archive) + b"trailing")

    with pytest.raises(artifacts_module.NativeArtifactError):
        _verify_packed_archive(packed_bootstrap, archive)


def test_packer_removes_temporary_archive_after_a_payload_substitution_race(
    bootstrap_payload: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = bootstrap_payload.root / "calibration-image/runtime.oci.tar"
    target.write_bytes(b"x" * (2 * 1024 * 1024))
    manifest = _publish(bootstrap_payload)
    archive = tmp_path / "substitution.tar"
    replaced = False

    def replace_after_first_chunk(path: Path) -> None:
        nonlocal replaced
        if path == target and not replaced:
            replaced = True
            target.unlink()
            target.write_bytes(b"replacement\n")

    monkeypatch.setattr(artifacts_module, "_ARCHIVE_CHUNK_HOOK", replace_after_first_chunk)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.pack_native_work_archive(
            phase="bootstrap",
            payload_root=bootstrap_payload.root,
            manifest_path=manifest.path,
            archive_path=archive,
        )
    assert replaced
    assert not archive.exists()
    assert not list(tmp_path.glob(".substitution.tar.*.tmp"))


def test_packer_rejects_temp_replacement_after_strict_verification(
    bootstrap_payload: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _publish(bootstrap_payload)
    archive = tmp_path / "verified.tar"
    real_verify = artifacts_module.verify_native_work_archive
    replaced = False

    def verify_then_replace(**kwargs: Any) -> Any:
        nonlocal replaced
        identity = real_verify(**kwargs)
        temporary = kwargs["archive_path"]
        temporary.unlink()
        temporary.write_bytes(b"unverified-after-strict-read\n")
        replaced = True
        return identity

    monkeypatch.setattr(artifacts_module, "verify_native_work_archive", verify_then_replace)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.pack_native_work_archive(
            phase="bootstrap",
            payload_root=bootstrap_payload.root,
            manifest_path=manifest.path,
            archive_path=archive,
        )
    assert replaced
    assert not archive.exists()
    assert not list(tmp_path.glob(".verified.tar.*.tmp"))


def test_packer_removes_replaced_source_after_rename_boundary(
    bootstrap_payload: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _publish(bootstrap_payload)
    archive = tmp_path / "rename-boundary.tar"
    real_rename = artifacts_module._rename_no_replace
    replaced = False

    def replace_then_rename(*, source: str, destination: str, parent_descriptor: int) -> None:
        nonlocal replaced
        os.unlink(source, dir_fd=parent_descriptor)
        replacement_descriptor = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            os.write(replacement_descriptor, b"unverified-at-rename-boundary\\n")
            os.fsync(replacement_descriptor)
        finally:
            os.close(replacement_descriptor)
        replaced = True
        real_rename(
            source=source,
            destination=destination,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(artifacts_module, "_rename_no_replace", replace_then_rename)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.pack_native_work_archive(
            phase="bootstrap",
            payload_root=bootstrap_payload.root,
            manifest_path=manifest.path,
            archive_path=archive,
        )
    assert replaced
    assert not archive.exists()
    assert not list(tmp_path.glob(".rename-boundary.tar.*.tmp"))


def test_packer_removes_symlink_replaced_source_after_rename_boundary(
    bootstrap_payload: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _publish(bootstrap_payload)
    archive = tmp_path / "rename-boundary-symlink.tar"
    target = tmp_path / "renamed-source-target"
    target.write_bytes(b"unverified-at-rename-boundary\\n")
    real_rename = artifacts_module._rename_no_replace
    replaced = False

    def replace_then_rename(*, source: str, destination: str, parent_descriptor: int) -> None:
        nonlocal replaced
        os.unlink(source, dir_fd=parent_descriptor)
        os.symlink(target, source, dir_fd=parent_descriptor)
        replaced = True
        real_rename(
            source=source,
            destination=destination,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(artifacts_module, "_rename_no_replace", replace_then_rename)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.pack_native_work_archive(
            phase="bootstrap",
            payload_root=bootstrap_payload.root,
            manifest_path=manifest.path,
            archive_path=archive,
        )
    assert replaced
    assert not archive.exists()
    assert not archive.is_symlink()


def test_packer_removes_nonempty_directory_after_rename_boundary(
    bootstrap_payload: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _publish(bootstrap_payload)
    archive = tmp_path / "rename-boundary-directory.tar"
    real_rename = artifacts_module._rename_no_replace
    replaced = False

    def replace_then_rename(*, source: str, destination: str, parent_descriptor: int) -> None:
        nonlocal replaced
        if not replaced:
            os.unlink(source, dir_fd=parent_descriptor)
            os.mkdir(source, dir_fd=parent_descriptor)
            source_descriptor = os.open(
                source,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            try:
                os.mkdir("nested", dir_fd=source_descriptor)
                nested_descriptor = os.open(
                    "nested",
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=source_descriptor,
                )
                try:
                    file_descriptor = os.open(
                        "unverified",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=nested_descriptor,
                    )
                    try:
                        os.write(file_descriptor, b"unverified-at-rename-boundary\n")
                    finally:
                        os.close(file_descriptor)
                finally:
                    os.close(nested_descriptor)
            finally:
                os.close(source_descriptor)
            replaced = True
        real_rename(
            source=source,
            destination=destination,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(artifacts_module, "_rename_no_replace", replace_then_rename)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.pack_native_work_archive(
            phase="bootstrap",
            payload_root=bootstrap_payload.root,
            manifest_path=manifest.path,
            archive_path=archive,
        )
    assert replaced
    assert not archive.exists()

    identity = artifacts_module.pack_native_work_archive(
        phase="bootstrap",
        payload_root=bootstrap_payload.root,
        manifest_path=manifest.path,
        archive_path=archive,
    )
    assert identity.size == archive.stat().st_size


def test_copy_attestation_bundle_is_no_replace_and_byte_exact(tmp_path: Path) -> None:
    source = tmp_path / "action-bundle.json"
    destination = tmp_path / "stable.sigstore.json"
    source.write_bytes(b'{"mediaType":"application/vnd.dev.sigstore.bundle+json;version=0.3"}\n')

    identity = artifacts_module.copy_attestation_bundle(source=source, destination=destination)

    assert destination.read_bytes() == source.read_bytes()
    assert identity.sha256 == sha256_hex(source.read_bytes())
    assert identity.size == source.stat().st_size
    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.copy_attestation_bundle(source=source, destination=destination)


def test_copy_attestation_bundle_rejects_a_symlinked_source(tmp_path: Path) -> None:
    source = tmp_path / "action-bundle.json"
    target = tmp_path / "outside.json"
    destination = tmp_path / "stable.sigstore.json"
    target.write_bytes(b"bundle\n")
    source.symlink_to(target)

    with pytest.raises(artifacts_module.NativeArtifactError):
        artifacts_module.copy_attestation_bundle(source=source, destination=destination)
    assert not destination.exists()


def test_cli_orders_structural_artifact_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_root = tmp_path / "payload"
    host_bundle = payload_root / "bootstrap-host-attestation/attestation-bundle.json"
    archive = tmp_path / "archive.tar"
    calls: list[str] = []
    manifest = SimpleNamespace(path=payload_root / "artifact-manifest.json")

    monkeypatch.setattr(
        artifacts_module,
        "publish_native_work_manifest",
        lambda **_kwargs: calls.append("publish-manifest") or manifest,
    )
    monkeypatch.setattr(
        artifacts_module,
        "pack_native_work_archive",
        lambda **_kwargs: calls.append("pack"),
    )
    monkeypatch.setattr(
        artifacts_module,
        "verify_native_work_archive",
        lambda **_kwargs: calls.append("verify-archive"),
    )

    assert (
        artifacts_module.main(
            [
                "publish",
                "--phase",
                "bootstrap",
                "--payload-root",
                str(payload_root),
                "--host-bundle",
                str(host_bundle),
                "--archive",
                str(archive),
            ]
        )
        == 0
    )
    assert calls == ["publish-manifest", "pack", "verify-archive"]

    calls.clear()
    monkeypatch.setattr(
        artifacts_module,
        "load_native_work_manifest",
        lambda *_args, **_kwargs: calls.append("load-manifest") or manifest,
    )
    monkeypatch.setattr(
        artifacts_module,
        "verify_native_work_payload",
        lambda **_kwargs: calls.append("verify-payload"),
    )
    monkeypatch.setattr(
        artifacts_module,
        "_stream_regular_file_identity",
        lambda _path: calls.append("verify-bundle"),
    )
    assert (
        artifacts_module.main(
            [
                "verify",
                "--phase",
                "bootstrap",
                "--payload-root",
                str(payload_root),
                "--archive",
                str(archive),
                "--bundle",
                str(host_bundle),
            ]
        )
        == 0
    )
    assert calls == ["load-manifest", "verify-payload", "verify-archive", "verify-bundle"]

    calls.clear()
    monkeypatch.setattr(
        artifacts_module,
        "copy_attestation_bundle",
        lambda **_kwargs: calls.append("copy-bundle"),
    )
    assert (
        artifacts_module.main(
            ["copy-bundle", "--source", str(host_bundle), "--destination", str(archive)]
        )
        == 0
    )
    assert calls == ["copy-bundle"]


def test_cli_verify_reports_an_unsafe_optional_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_root = tmp_path / "payload"
    archive = tmp_path / "archive.tar"
    manifest = SimpleNamespace(path=payload_root / "artifact-manifest.json")
    monkeypatch.setattr(
        artifacts_module, "load_native_work_manifest", lambda *_args, **_kwargs: manifest
    )
    monkeypatch.setattr(artifacts_module, "verify_native_work_payload", lambda **_kwargs: None)
    monkeypatch.setattr(artifacts_module, "verify_native_work_archive", lambda **_kwargs: None)

    result = artifacts_module.main(
        [
            "verify",
            "--phase",
            "bootstrap",
            "--payload-root",
            str(payload_root),
            "--archive",
            str(archive),
            "--bundle",
            str(tmp_path / "missing.sigstore.json"),
        ]
    )

    assert result == 2
    assert capsys.readouterr().err == "native work attestation bundle is missing or unsafe\n"


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
