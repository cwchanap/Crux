from __future__ import annotations

# The sibling lock suite supplies hand-written, independently validated frozen payloads.
# pylint: disable=duplicate-code,too-many-arguments,too-many-lines,too-many-locals
# pylint: disable=too-many-positional-arguments,too-many-statements
import hashlib
import importlib.util
import json
import os
import platform
import stat
import subprocess
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest

import tools.hpa320.seal_oaf_backend as seal_module
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex, strict_json_loads
from src.benchmark.checkpoint_acquisition import CheckpointIdentity
from tools.hpa320 import oaf_native_calibration
from tools.hpa320.oaf_native_artifacts import CANDIDATE_ARTIFACT_PATHS, CANDIDATE_ARTIFACTS
from tools.hpa320.oaf_oci import OciLayoutIdentity
from tools.hpa320.seal_oaf_backend import SealError

HOST_PATHS = (
    "src/benchmark/backend_attestation.py",
    "src/benchmark/backend_identity.py",
    "src/benchmark/backend_lock.py",
    "src/benchmark/backend_process.py",
    "src/benchmark/backend_publication.py",
    "src/benchmark/backends/base.py",
    "src/benchmark/backends/oaf_tf1.py",
    "src/benchmark/input_view.py",
    "src/benchmark/prediction_artifact.py",
)


def _lock_fixtures() -> ModuleType:
    path = Path(__file__).with_name("test_backend_lock.py")
    spec = importlib.util.spec_from_file_location("_oaf_lock_fixture_payloads", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOCKS = _lock_fixtures()
V2_SCHEMA_REPLACEMENTS = {
    "crux.backend-seal-evidence/v1": "crux.backend-seal-evidence/v2",
    "crux.oaf-base-system-package-evidence/v1": "crux.oaf-base-system-package-evidence/v2",
    "crux.oaf-calibration-bootstrap-evidence/v1": ("crux.oaf-calibration-bootstrap-evidence/v2"),
    "crux.oaf-calibration-measurement-evidence/v1": (
        "crux.oaf-calibration-measurement-evidence/v2"
    ),
    "crux.oaf-seal-candidate/v1": "crux.oaf-seal-candidate/v2",
}


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    return path


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_manifest_validator_allows_package_ancestors_only() -> None:
    payload = {
        "covered_roots": ["magenta/models/deep"],
        "files": [
            {"path": "LICENSE", "sha256": "a" * 64},
            {"path": "magenta/__init__.py", "sha256": "b" * 64},
            {"path": "magenta/models/__init__.py", "sha256": "c" * 64},
            {"path": "magenta/models/deep/model.py", "sha256": "d" * 64},
        ],
        "schema": "crux.oaf-host-adapter-source-manifest/v1",
    }

    seal_module._validate_source_manifest_payload(
        payload,
        "crux.oaf-host-adapter-source-manifest/v1",
    )

    payload["files"][2]["path"] = "magenta/unrelated/source.py"
    with pytest.raises(SealError, match="outside covered roots"):
        seal_module._validate_source_manifest_payload(
            payload,
            "crux.oaf-host-adapter-source-manifest/v1",
        )


def _native_host_payload() -> dict[str, object]:
    return {
        "github_job": "native-bootstrap",
        "github_repository": "acme/crux",
        "github_run_attempt": 1,
        "github_run_id": 456,
        "github_workflow_ref": (
            "acme/crux/.github/workflows/hpa320-native-bootstrap.yml@refs/heads/test"
        ),
        "github_workflow_sha": "b" * 40,
        "run_url": "https://github.com/acme/crux/actions/runs/456",
        "runner_arch": "X64",
        "runner_environment": "github-hosted",
        "runner_os": "Linux",
        "workflow_commit": "b" * 40,
        "schema": "crux.github-hosted-native-evidence/v2",
        "host_numeric_fingerprint": {
            "architecture": "x86_64",
            "cpu_vendor_id": "GenuineIntel",
            "cpu_family": "6",
            "cpu_model": "143",
            "cpu_stepping": "8",
        },
    }


def _native_host_record(
    *,
    kind: str = "github_hosted",
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    evidence_payload = _native_host_payload() if payload is None else payload
    return {
        "kind": kind,
        "official_execution_allowed": True,
        "payload": evidence_payload,
        "sha256": sha256_hex(canonical_json_bytes(evidence_payload)),
    }


def _set_native_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("RUNNER_ARCH", "X64")


def _host_bundle_fixture(
    tmp_path: Path,
    *,
    phase: str,
) -> tuple[Path, Path, dict[str, object]]:
    directory = tmp_path / f"{phase}-host-attestation"
    directory.mkdir()
    commit = "b" * 40
    run_id = {"bootstrap": 456, "measurement": 457, "candidate": 458}[phase]
    repository = "acme/crux"
    run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    fingerprint = {
        "architecture": "x86_64",
        "cpu_family": "6",
        "cpu_model": "143",
        "cpu_stepping": "8",
        "cpu_vendor_id": "GenuineIntel",
    }
    evidence_payload = {
        "github_job": f"native-{phase}",
        "github_repository": repository,
        "github_run_attempt": 1,
        "github_run_id": run_id,
        "github_workflow_ref": (
            f"{repository}/.github/workflows/hpa320-native-{phase}.yml@refs/heads/test"
        ),
        "github_workflow_sha": commit,
        "host_numeric_fingerprint": fingerprint,
        "run_url": run_url,
        "runner_arch": "X64",
        "runner_environment": "github-hosted",
        "runner_os": "Linux",
        "workflow_commit": commit,
        "schema": "crux.github-hosted-native-evidence/v2",
    }
    record = _native_host_record(payload=evidence_payload)
    host_path = _write_json(directory / "native-host-evidence.json", record)
    observation_path = _write_json(
        directory / "native-host-observation.json",
        {
            "docker_architecture": "x86_64",
            "docker_os_type": "linux",
            "docker_server_version": "28.0.4",
            "github_job": f"native-{phase}",
            "github_repository": repository,
            "github_run_attempt": 1,
            "github_run_id": run_id,
            "github_run_url": run_url,
            "github_sha": commit,
            "github_workflow_ref": (
                f"{repository}/.github/workflows/hpa320-native-{phase}.yml@refs/heads/test"
            ),
            "github_workflow_sha": commit,
            "host_numeric_fingerprint": fingerprint,
            "runner_arch": "X64",
            "runner_environment": "github-hosted",
            "runner_os": "Linux",
            "uname_architecture": "x86_64",
        },
    )
    bundle_path = _write_json(
        directory / "attestation-bundle.json",
        {
            "native_host_evidence": {
                "name": host_path.name,
                "sha256": _content_hash(host_path),
                "size": host_path.stat().st_size,
            },
            "native_host_observation": {
                "name": observation_path.name,
                "sha256": _content_hash(observation_path),
                "size": observation_path.stat().st_size,
            },
            "phase": phase,
            "schema": "crux.oaf-native-host-attestation-bundle/v2",
        },
    )
    return bundle_path, host_path, record


def test_seal_host_binding_requires_its_authenticated_reference_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_native_host(monkeypatch)
    host_record = _native_host_record()
    host_path = _write_json(tmp_path / "native-host-evidence.json", host_record)
    host = seal_module.load_native_host_evidence(host_path)
    seal = SimpleNamespace(
        payload={
            "native_host_evidence": host_record,
            "reference_host_numeric_fingerprint": host.host_numeric_fingerprint.as_json(),
        }
    )

    seal_module._validate_host_binding(host, seal)

    seal.payload["reference_host_numeric_fingerprint"]["cpu_model"] = "999"
    with pytest.raises(SealError, match="reference host numeric fingerprint"):
        seal_module._validate_host_binding(host, seal)


def _host_manifest_payload(repository: Path) -> dict[str, object]:
    return {
        "covered_roots": list(HOST_PATHS),
        "files": [
            {
                "path": relative,
                "sha256": hashlib.sha256((repository / relative).read_bytes()).hexdigest(),
            }
            for relative in HOST_PATHS
        ],
        "schema": "crux.oaf-host-adapter-source-manifest/v1",
    }


def _make_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    for index, relative in enumerate(HOST_PATHS):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"host source {index}\n".encode())
    _write_json(
        repository / "runtime/oaf_tf1/source-manifest.json",
        {
            "covered_roots": ["runtime/oaf_tf1/vendor/magenta"],
            "files": [],
            "schema": "crux.oaf-upstream-source-manifest/v1",
            "upstream_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
            "upstream_repository": "https://github.com/magenta/magenta.git",
        },
    )
    _write_json(
        repository / "runtime/oaf_tf1/runner-source-manifest.json",
        {
            "covered_roots": ["runtime/oaf_tf1"],
            "files": [],
            "schema": "crux.oaf-runner-source-manifest/v1",
        },
    )
    _write_json(
        repository / "runtime/oaf_tf1/distribution-build-manifest.json",
        {"schema": "crux.distribution-build-manifest/v2"},
    )
    patch = repository / "runtime/oaf_tf1/patches/capture-emitted-frame.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_bytes(b"reviewed instrumentation patch\n")
    return repository


def _bootstrap_request_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    repository = tmp_path / "repository"
    paths = {
        "runtime/oaf_tf1/base-system-package-request.json": b"base request\n",
        "runtime/oaf_tf1/build-context-manifest.json": b"context manifest\n",
        "runtime/oaf_tf1/distribution-build-manifest.json": b"distribution manifest\n",
        "runtime/oaf_tf1/patches/capture-emitted-frame.patch": b"patch\n",
        "runtime/oaf_tf1/runner-source-manifest.json": b"runner manifest\n",
        "runtime/oaf_tf1/source-manifest.json": b"upstream manifest\n",
        (
            "config/benchmark/backends/"
            "magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json"
        ): b"checkpoint request\n",
    }
    for relative, content in paths.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    payload: dict[str, object] = {
        "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
        "base_image_manifest_digest": (
            "sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673"
        ),
        "base_system_package_request_sha256": hashlib.sha256(
            paths["runtime/oaf_tf1/base-system-package-request.json"]
        ).hexdigest(),
        "build_context_manifest_sha256": hashlib.sha256(
            paths["runtime/oaf_tf1/build-context-manifest.json"]
        ).hexdigest(),
        "checkpoint_acquisition_request_sha256": hashlib.sha256(
            paths[
                "config/benchmark/backends/"
                "magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json"
            ]
        ).hexdigest(),
        "container_restrictions": {
            "drop_capabilities": ["ALL"],
            "network": "none",
            "no_new_privileges": True,
            "platform": "linux/amd64",
            "read_only_root": True,
        },
        "distribution_build_manifest_sha256": hashlib.sha256(
            paths["runtime/oaf_tf1/distribution-build-manifest.json"]
        ).hexdigest(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": "-1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "TF_NUM_INTEROP_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
        },
        "image_build": {
            "annotations": [],
            "buildkit_image": (
                "moby/buildkit@sha256:"
                "63db51c9b30208a7c2b1c40392c7ebb9ce2f85ba238a18a85420f8f5ea2d4684"
            ),
            "buildkit_version": "v0.31.2",
            "buildx_binary_sha256": (
                "d41ece72044243b4f58b343441ae37446d9c29a7d6b5e11c61847bbcf8f7dfda"
            ),
            "buildx_binary_size": 65_265_826,
            "buildx_binary_url": (
                "https://github.com/docker/buildx/releases/download/"
                "v0.35.0/buildx-v0.35.0.linux-amd64"
            ),
            "buildx_version": "v0.35.0",
            "compression": "gzip",
            "compression_level": 6,
            "dockerfile_frontend": (
                "docker/dockerfile-upstream@sha256:"
                "3d6d54b33351b396a910d33248754b86b1d7dd838b4eeb9575d8903a209f6516"
            ),
            "dockerfile_frontend_version": "1.25.0",
            "exporter": "oci",
            "exporter_tar": False,
            "force_compression": False,
            "inline_cache": False,
            "multi_platform_deterministic": True,
            "oci_archive": {
                "compression": "none",
                "final_zero_blocks": 2,
                "format": "posix-ustar",
                "gid": 0,
                "gname": "",
                "member_mode": 420,
                "member_types": "regular-files-only",
                "mtime": 0,
                "path_order": "utf8-byte",
                "uid": 0,
                "uname": "",
            },
            "oci_media_types": True,
            "platform": "linux/amd64",
            "provenance": False,
            "rewrite_timestamp": True,
            "sbom": False,
            "source_date_epoch": 0,
        },
        "instrumentation_patch_sha256": hashlib.sha256(
            paths["runtime/oaf_tf1/patches/capture-emitted-frame.patch"]
        ).hexdigest(),
        "python_coerce_c_locale": "0",
        "resource_ceiling": {
            "cpu_limit_millis": 2000,
            "memory_limit_bytes": 4_294_967_296,
            "monitor_interval_millis": 10,
            "pid_limit": 256,
            "request_deadline_seconds": 1800,
            "shm_bytes": 1_073_741_824,
            "startup_deadline_seconds": 300,
            "stderr_max_line_bytes": 65_536,
            "stderr_read_chunk_bytes": 65_536,
            "stderr_ring_buffer_bytes": 1_048_576,
            "stdout_max_line_bytes": 134_217_728,
            "tmp_bytes": 1_073_741_824,
        },
        "runner_source_manifest_sha256": hashlib.sha256(
            paths["runtime/oaf_tf1/runner-source-manifest.json"]
        ).hexdigest(),
        "runtime_gid": 65_532,
        "runtime_uid": 65_532,
        "schema": "crux.oaf-calibration-bootstrap-request/v1",
        "upstream_source_manifest_sha256": hashlib.sha256(
            paths["runtime/oaf_tf1/source-manifest.json"]
        ).hexdigest(),
    }
    request_path = (
        repository / "config/benchmark/backends/"
        "magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json"
    )
    _write_json(request_path, payload)
    return request_path, payload


def _bootstrap_evidence_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    request_path, request = _bootstrap_request_fixture(tmp_path)
    bundle_path, host_path, host_record = _host_bundle_fixture(
        tmp_path,
        phase="bootstrap",
    )
    payload: dict[str, object] = {
        "base_image_config_digest": LOCKS.BASE_IMAGE_CONFIG_DIGEST,
        "base_image_layer_diff_ids": deepcopy(LOCKS.BASE_IMAGE_LAYER_DIFF_IDS),
        "base_image_layer_digests": deepcopy(LOCKS.BASE_IMAGE_LAYER_DIGESTS),
        "build_context_manifest_sha256": request["build_context_manifest_sha256"],
        "calibration_bootstrap_request_sha256": _content_hash(request_path),
        "image_build": deepcopy(request["image_build"]),
        "native_host_attestation_bundle_sha256": _content_hash(bundle_path),
        "native_host_evidence": host_record,
        "oci_layout_archive": {
            "name": "runtime.oci.tar",
            "sha256": "f" * 64,
            "size": 123,
        },
        "oci_layout_manifest_sha256": "1" * 64,
        "runtime_image_config_digest": LOCKS.RUNTIME_IMAGE_CONFIG_DIGEST,
        "runtime_image_index_digest": LOCKS.RUNTIME_IMAGE_INDEX_DIGEST,
        "runtime_image_layer_diff_ids": deepcopy(LOCKS.RUNTIME_IMAGE_LAYER_DIFF_IDS),
        "runtime_image_layer_digests": deepcopy(LOCKS.RUNTIME_IMAGE_LAYER_DIGESTS),
        "runtime_image_manifest_digest": f"sha256:{'4' * 64}",
        "schema": V2_SCHEMA_REPLACEMENTS["crux.oaf-calibration-bootstrap-evidence/v1"],
    }
    evidence_path = _write_json(tmp_path / "calibration-bootstrap-evidence.json", payload)
    return request_path, evidence_path, bundle_path, payload


def test_calibration_bootstrap_request_loads_exact_recipe_and_cross_hashes(
    tmp_path: Path,
) -> None:
    request_path, payload = _bootstrap_request_fixture(tmp_path)

    request = seal_module.load_calibration_bootstrap_request(request_path)

    assert request.runtime_uid == 65_532
    assert request.runtime_gid == 65_532
    assert request.build_context_manifest_sha256 == payload["build_context_manifest_sha256"]
    assert request.image_build.buildkit_version == "v0.31.2"
    assert request.image_build.oci_archive.final_zero_blocks == 2
    assert request.sha256 == hashlib.sha256(request_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.__setitem__("unknown", True), "fields"),
        (
            lambda payload: payload["image_build"].__setitem__("buildkit_version", "v0.31.3"),
            "image build",
        ),
        (
            lambda payload: payload["image_build"].__setitem__("exporter_tar", 0),
            "image build",
        ),
        (
            lambda payload: payload.__setitem__("build_context_manifest_sha256", "0" * 64),
            "hash",
        ),
    ),
)
def test_calibration_bootstrap_request_rejects_contract_drift(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    request_path, payload = _bootstrap_request_fixture(tmp_path)
    mutation(payload)
    _write_json(request_path, payload)

    with pytest.raises(SealError, match=message):
        seal_module.load_calibration_bootstrap_request(request_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload.__setitem__("calibration_bootstrap_request_sha256", "0" * 64),
            "does not reproduce",
        ),
        (
            lambda payload: payload["image_build"].__setitem__("buildkit_version", "v0.31.3"),
            "does not reproduce",
        ),
        (
            lambda payload: payload.__setitem__("build_context_manifest_sha256", "0" * 64),
            "does not reproduce",
        ),
        (
            lambda payload: payload["runtime_image_layer_digests"].reverse(),
            "prefix or order",
        ),
        (
            lambda payload: payload["runtime_image_layer_diff_ids"].reverse(),
            "prefix or order",
        ),
    ),
)
def test_calibration_bootstrap_evidence_rejects_identity_drift(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    request, evidence, _bundle, _payload = _bootstrap_evidence_fixture(tmp_path)
    changed = json.loads(evidence.read_bytes())
    mutation(changed)
    _write_json(evidence, changed)

    with pytest.raises(SealError, match=message):
        seal_module.load_calibration_bootstrap_evidence(request, evidence)


def test_calibration_bootstrap_evidence_accepts_v2_and_rejects_former_v1_schema(
    tmp_path: Path,
) -> None:
    request, evidence, _bundle, payload = _bootstrap_evidence_fixture(tmp_path)
    assert payload["schema"] == V2_SCHEMA_REPLACEMENTS["crux.oaf-calibration-bootstrap-evidence/v1"]
    seal_module.load_calibration_bootstrap_evidence(request, evidence)

    payload["schema"] = "crux.oaf-calibration-bootstrap-evidence/v1"
    _write_json(evidence, payload)
    with pytest.raises(SealError, match="calibration bootstrap evidence fields"):
        seal_module.load_calibration_bootstrap_evidence(request, evidence)


def test_bootstrap_phase_rejects_config_import_and_bundle_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_native_host(monkeypatch)
    request, evidence, bootstrap_bundle, _payload = _bootstrap_evidence_fixture(tmp_path)
    host = bootstrap_bundle.parent / "native-host-evidence.json"
    monkeypatch.setattr(
        seal_module,
        "_docker_capture",
        lambda _command, _label: f"sha256:{'0' * 64}\n".encode(),
    )

    with pytest.raises(SealError, match="config digest"):
        seal_module._authenticate_bootstrap_for_phase(
            bootstrap_request_path=request,
            bootstrap_evidence_path=evidence,
            host_attestation_bundle_path=bootstrap_bundle,
            host_evidence_path=host,
            phase="bootstrap",
        )

    measurement_bundle, measurement_host, _record = _host_bundle_fixture(
        tmp_path,
        phase="measurement",
    )
    monkeypatch.setattr(seal_module, "_require_imported_runtime_image", lambda _digest: None)
    with pytest.raises(SealError, match="phase"):
        seal_module._authenticate_bootstrap_for_phase(
            bootstrap_request_path=request,
            bootstrap_evidence_path=evidence,
            host_attestation_bundle_path=measurement_bundle,
            host_evidence_path=measurement_host,
            phase="bootstrap",
        )


def test_native_producer_cli_has_no_mutable_image_or_candidate_authority() -> None:
    parser = seal_module._parser()
    help_text = parser.format_help()
    assert "--image" not in help_text
    assert "--candidate-authority" not in help_text
    seal = parser.parse_args(["seal", "--candidate", "candidate", "--repository-root", "."])
    assert vars(seal) == {
        "candidate": Path("candidate"),
        "command": "seal",
        "repository_root": Path("."),
    }


def test_calibration_bootstrap_dockerfile_has_one_pinned_syntax_directive() -> None:
    lines = Path("runtime/oaf_tf1/Dockerfile").read_text(encoding="utf-8").splitlines()

    assert lines[0] == (
        "# syntax=docker/dockerfile-upstream@sha256:"
        "3d6d54b33351b396a910d33248754b86b1d7dd838b4eeb9575d8903a209f6516"
    )
    assert sum(line.startswith("# syntax=") for line in lines) == 1


def test_calibration_bootstrap_buildx_command_has_no_host_passthrough(tmp_path: Path) -> None:
    buildx = tmp_path / "buildx"
    output = tmp_path / "output"

    command = oaf_native_calibration._buildx_build_command(buildx, "builder-one", output)

    assert command == (
        str(buildx),
        "build",
        "--builder",
        "builder-one",
        "--file",
        "runtime/oaf_tf1/Dockerfile",
        "--platform",
        "linux/amd64",
        "--pull",
        "--no-cache",
        "--provenance=false",
        "--sbom=false",
        "--build-arg",
        "BUILDKIT_MULTI_PLATFORM=1",
        "--build-arg",
        "SOURCE_DATE_EPOCH=0",
        "--build-arg",
        "RUNTIME_UID=65532",
        "--build-arg",
        "RUNTIME_GID=65532",
        "--annotation",
        "index:org.opencontainers.image.created=1970-01-01T00:00:00Z",
        "--output",
        (
            "type=oci,tar=false,oci-mediatypes=true,compression=gzip,"
            "compression-level=6,force-compression=false,"
            f"rewrite-timestamp=true,dest={output}"
        ),
        ".",
    )
    assert str(Path.cwd()) not in command


def _oci_identity(archive: Path) -> OciLayoutIdentity:
    digest = "sha256:" + "a" * 64
    return OciLayoutIdentity(
        archive=CheckpointIdentity(
            name=archive.name,
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            size=archive.stat().st_size,
        ),
        base_image_config_digest="sha256:" + "b" * 64,
        base_image_layer_digests=("sha256:" + "c" * 64,),
        base_image_layer_diff_ids=("sha256:" + "d" * 64,),
        index_digest="sha256:" + "e" * 64,
        image_manifest_digest="sha256:" + "f" * 64,
        config_digest=digest,
        layer_digests=("sha256:" + "c" * 64, "sha256:" + "1" * 64),
        layer_diff_ids=("sha256:" + "d" * 64, "sha256:" + "2" * 64),
    )


def test_calibration_bootstrap_import_reinspects_exact_config_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "runtime.oci.tar"
    archive.write_bytes(b"archive")
    expected = _oci_identity(archive)
    calls: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[1:3] == ("image", "inspect"):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{expected.config_digest}\namd64\nlinux\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(oaf_native_calibration.subprocess, "run", run)

    locator = oaf_native_calibration.import_authenticated_oci_archive(
        archive,
        expected,
    )

    assert locator == expected.config_digest
    assert calls[0][1:3] == ("image", "load")
    assert calls[1][1:3] == ("image", "inspect")

    def drifted(
        command: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if command[1:3] == ("image", "inspect"):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"sha256:{'0' * 64}\namd64\nlinux\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(oaf_native_calibration.subprocess, "run", drifted)
    with pytest.raises(SealError, match="config digest"):
        oaf_native_calibration.import_authenticated_oci_archive(archive, expected)


def test_calibration_bootstrap_rejects_non_native_host_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _payload = _bootstrap_request_fixture(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(oaf_native_calibration.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(oaf_native_calibration.platform, "machine", lambda: "arm64")

    with pytest.raises(SealError, match="native linux/amd64"):
        seal_module.bootstrap_image(
            request_path=request,
            host_attestation_bundle_path=tmp_path / "missing-bundle.json",
            host_evidence_path=tmp_path / "missing-evidence.json",
            output=output,
            repository_root=request.parents[3],
        )

    assert not output.exists()


def test_calibration_bootstrap_cli_rejects_non_native_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request, _payload = _bootstrap_request_fixture(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(oaf_native_calibration.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(oaf_native_calibration.platform, "machine", lambda: "arm64")

    exit_code = seal_module.main(
        [
            "bootstrap-image",
            "--request",
            os.fspath(request),
            "--host-attestation-bundle",
            os.fspath(tmp_path / "missing-bundle.json"),
            "--host-evidence",
            os.fspath(tmp_path / "missing-evidence.json"),
            "--output",
            os.fspath(output),
            "--repository-root",
            os.fspath(request.parents[3]),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert json.loads(captured.out) == {
        "exit_code": 1,
        "report_path": None,
        "report_sha256": None,
        "status": "failed",
    }
    assert "native linux/amd64" in captured.err
    assert not output.exists()


def test_calibration_bootstrap_publication_is_idempotent_and_no_replace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    contents = {
        oaf_native_calibration.BOOTSTRAP_EVIDENCE_NAME: b"evidence\n",
        oaf_native_calibration.OCI_LAYOUT_MANIFEST_NAME: b"manifest\n",
        oaf_native_calibration.OCI_ARCHIVE_NAME: b"archive",
    }
    for name, content in contents.items():
        (source / name).write_bytes(content)

    oaf_native_calibration._publish_directory_immutable(source, target)
    oaf_native_calibration._publish_directory_immutable(source, target)

    assert {path.name: path.read_bytes() for path in target.iterdir()} == contents
    (target / oaf_native_calibration.BOOTSTRAP_EVIDENCE_NAME).write_bytes(b"different\n")
    fresh_source = tmp_path / "fresh-source"
    fresh_source.mkdir()
    for name, content in contents.items():
        (fresh_source / name).write_bytes(content)
    with pytest.raises(SealError, match="differs"):
        oaf_native_calibration._publish_directory_immutable(fresh_source, target)
    assert (target / oaf_native_calibration.BOOTSTRAP_EVIDENCE_NAME).read_bytes() == b"different\n"


def _tensor_payload(seal: dict[str, Any]) -> dict[str, object]:
    return {
        "active_predict_dropout": False,
        "checkpoint_inventory": deepcopy(seal["checkpoint_inventory"]),
        "non_inference_inventory": deepcopy(seal["non_inference_inventory"]),
        "note_sequence_byte_parity": True,
        "required_inference_inventory": deepcopy(seal["required_inference_inventory"]),
        "schema": "crux.oaf-tensor-coverage/v1",
        "uninitialized_required": [],
    }


def _output_paths(repository: Path) -> dict[str, Path]:
    return {role: repository / relative for role, relative in CANDIDATE_ARTIFACTS}


def _candidate_artifact_path(candidate: Path, role: str) -> Path:
    return candidate / CANDIDATE_ARTIFACT_PATHS[role]


def test_v2_candidate_inventory_removes_the_jobs_api_record() -> None:
    roles = tuple(role for role, _path in CANDIDATE_ARTIFACTS)
    assert "native_host_api_record" not in roles
    assert all(
        not path.endswith("github-job-api-record.json.hex")
        for path in CANDIDATE_ARTIFACT_PATHS.values()
    )
    assert set(roles) == {
        "conversion_audit",
        "native_host_attestation_bundle",
        "native_host_evidence",
        "native_host_observation",
        "host_adapter_source_manifest",
        "tensor_coverage",
        "advisory_snapshot",
        "security_scan",
        "oci_layout_archive",
        "oci_layout_manifest",
        "smoke_audio",
        "smoke_prediction",
        "smoke_oracle",
        "seal_evidence",
        "runtime_lock",
        "backend_lock",
    }


def _build_candidate(
    repository: Path,
    *,
    audit_mutation: Callable[[dict[str, Any]], None] | None = None,
    seal_mutation: Callable[[dict[str, Any]], None] | None = None,
    tensor_payload: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    candidate = repository / "candidate"
    candidate.mkdir(parents=True)
    bundle, host_path, host_record = _host_bundle_fixture(
        repository,
        phase="candidate",
    )
    for source, role in (
        (bundle, "native_host_attestation_bundle"),
        (host_path, "native_host_evidence"),
        (bundle.parent / "native-host-observation.json", "native_host_observation"),
    ):
        destination = _candidate_artifact_path(candidate, role)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    audit = LOCKS.audit_payload()
    if audit_mutation is not None:
        audit_mutation(audit)
    audit_path = _write_json(_candidate_artifact_path(candidate, "conversion_audit"), audit)

    seal = LOCKS.seal_payload(audit_sha256=_content_hash(audit_path))
    seal["native_host_attestation_bundle_sha256"] = _content_hash(bundle)
    seal["native_host_evidence"] = deepcopy(host_record)
    seal["reference_host_numeric_fingerprint"] = deepcopy(
        host_record["payload"]["host_numeric_fingerprint"]
    )

    upstream = repository / "runtime/oaf_tf1/source-manifest.json"
    runner = repository / "runtime/oaf_tf1/runner-source-manifest.json"
    distribution = repository / "runtime/oaf_tf1/distribution-build-manifest.json"
    instrumentation = repository / "runtime/oaf_tf1/patches/capture-emitted-frame.patch"
    host_manifest_content = (
        json.dumps(
            _host_manifest_payload(repository),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    seal["upstream_source_manifest_sha256"] = _content_hash(upstream)
    seal["runner_source_manifest_sha256"] = _content_hash(runner)
    seal["distribution_build_manifest_sha256"] = _content_hash(distribution)
    seal["instrumentation_patch_sha256"] = _content_hash(instrumentation)
    seal["host_adapter_source_manifest_sha256"] = hashlib.sha256(host_manifest_content).hexdigest()
    host_manifest_path = _candidate_artifact_path(
        candidate,
        "host_adapter_source_manifest",
    )
    host_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    host_manifest_path.write_bytes(host_manifest_content)

    advisory_path = _write_json(
        _candidate_artifact_path(candidate, "advisory_snapshot"),
        {"advisories": [], "schema": "crux.test-advisory-snapshot/v1"},
    )
    security_path = _write_json(
        _candidate_artifact_path(candidate, "security_scan"),
        {
            "advisory_snapshot_sha256": _content_hash(advisory_path),
            "findings": [],
            "schema": "crux.oaf-security-scan/v1",
        },
    )
    seal["advisory_snapshot_sha256"] = _content_hash(advisory_path)
    seal["security_scan_sha256"] = _content_hash(security_path)

    audio = _candidate_artifact_path(candidate, "smoke_audio")
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"RIFF deterministic smoke bytes")
    smoke_prediction = _candidate_artifact_path(candidate, "smoke_prediction")
    smoke_prediction.parent.mkdir(parents=True, exist_ok=True)
    smoke_prediction.write_bytes(b'{"record_type":"header"}\n{"record_type":"terminal"}\n')
    oracle_path = _write_json(
        _candidate_artifact_path(candidate, "smoke_oracle"),
        {
            "input_audio_frame_count": 64,
            "input_audio_sha256": _content_hash(audio),
            "input_view_id": "oaf-smoke-canonical-v1",
            "native_events": [
                {
                    "confidence_binary64": "3fe8000000000000",
                    "frame_index": 12,
                    "model_output_bin": 15,
                    "native_class_id": "midi_36",
                    "native_midi_note": 36,
                    "time_sec_binary64": "3fc1d53a957d7519",
                    "upstream_8hit_group_id": "kick",
                    "velocity_midi": 100,
                }
            ],
            "schema": "crux.oaf-smoke-oracle/v1",
            "source_audio_id": "oaf-smoke-source-v1",
            "source_audio_sha256": _content_hash(audio),
        },
    )
    seal["smoke_audio_sha256"] = _content_hash(audio)
    seal["smoke_oracle_sha256"] = _content_hash(oracle_path)
    seal["smoke_prediction_sha256"] = _content_hash(smoke_prediction)

    archive = _candidate_artifact_path(candidate, "oci_layout_archive")
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"complete deterministic OCI archive bytes")
    seal["oci_layout_archive"] = {
        "name": archive.name,
        "sha256": _content_hash(archive),
        "size": archive.stat().st_size,
    }
    oci_path = _write_json(
        _candidate_artifact_path(candidate, "oci_layout_manifest"),
        {
            "archive": deepcopy(seal["oci_layout_archive"]),
            "base_image_config_digest": seal["base_image_config_digest"],
            "base_image_layer_diff_ids": deepcopy(seal["base_image_layer_diff_ids"]),
            "base_image_layer_digests": deepcopy(seal["base_image_layer_digests"]),
            "config_digest": seal["runtime_image_config_digest"],
            "image_manifest_digest": seal["runtime_image_manifest_digest"],
            "index_digest": seal["runtime_image_index_digest"],
            "layer_diff_ids": deepcopy(seal["runtime_image_layer_diff_ids"]),
            "layer_digests": deepcopy(seal["runtime_image_layer_digests"]),
            "schema": "crux.oaf-oci-layout-manifest/v1",
        },
    )
    seal["oci_layout_manifest_sha256"] = _content_hash(oci_path)

    tensor = _tensor_payload(seal) if tensor_payload is None else tensor_payload
    tensor_path = _write_json(_candidate_artifact_path(candidate, "tensor_coverage"), tensor)
    seal["tensor_coverage_sha256"] = _content_hash(tensor_path)
    if seal_mutation is not None:
        seal_mutation(seal)
    seal_path = _write_json(_candidate_artifact_path(candidate, "seal_evidence"), seal)

    runtime = LOCKS.runtime_payload(seal_sha256=_content_hash(seal_path))
    for field in (
        "additional_system_packages",
        "base_image_archive_keyring_sha256",
        "base_image_config_digest",
        "base_image_layer_diff_ids",
        "base_image_layer_digests",
        "base_image_manifest_digest",
        "base_system_package_evidence_sha256",
        "base_system_package_inventory",
        "base_system_package_inventory_sha256",
        "base_system_package_request_sha256",
        "build_context_manifest_sha256",
        "calibration_bootstrap_evidence_sha256",
        "calibration_bootstrap_request_sha256",
        "distribution_build_manifest_sha256",
        "oci_layout_manifest_sha256",
        "python_distributions",
        "runner_source_manifest_sha256",
        "runtime_image_config_digest",
        "runtime_image_manifest_digest",
        "tensorflow_abi",
        "tensorflow_build",
        "upstream_source_manifest_sha256",
    ):
        runtime[field] = deepcopy(seal[field])
    runtime_path = _write_json(_candidate_artifact_path(candidate, "runtime_lock"), runtime)

    backend = LOCKS.backend_payload(
        runtime_sha256=_content_hash(runtime_path),
        seal_sha256=_content_hash(seal_path),
        audit_sha256=_content_hash(audit_path),
    )
    for field in (
        "checkpoint_archive",
        "checkpoint_components",
        "checkpoint_inventory",
        "host_adapter_source_manifest_sha256",
        "max_input_audio_frames",
        "non_inference_inventory",
        "required_inference_inventory",
        "runtime_image_manifest_digest",
        "smoke_audio_sha256",
        "smoke_oracle_sha256",
        "upstream_source_manifest_sha256",
    ):
        backend[field] = deepcopy(seal[field])
    backend_path = _write_json(_candidate_artifact_path(candidate, "backend_lock"), backend)

    model_identity = LOCKS.identity_sha256(backend["checkpoint_components"])
    artifacts = [
        {
            "path": relative,
            "role": role,
            "sha256": _content_hash(candidate / relative),
        }
        for role, relative in CANDIDATE_ARTIFACTS
    ]
    _write_json(
        candidate / "candidate-manifest.json",
        {
            "artifacts": artifacts,
            "backend_lock_payload_sha256": _content_hash(backend_path),
            "calibration_bootstrap_evidence_sha256": seal["calibration_bootstrap_evidence_sha256"],
            "calibration_measurement_evidence_sha256": seal[
                "calibration_measurement_evidence_sha256"
            ],
            "checkpoint_components": deepcopy(backend["checkpoint_components"]),
            "checkpoint_prefix": f"sha256/{model_identity}/model.ckpt-569400",
            "model_artifact_set_sha256": model_identity,
            "native_host_attestation_bundle_sha256": _content_hash(bundle),
            "required_inference_inventory_sha256": LOCKS.identity_sha256(
                backend["required_inference_inventory"]
            ),
            "runtime_lock_payload_sha256": _content_hash(runtime_path),
            "schema": V2_SCHEMA_REPLACEMENTS["crux.oaf-seal-candidate/v1"],
            "seal_evidence_payload_sha256": _content_hash(seal_path),
            "seal_profile_request_sha256": seal["seal_profile_request_sha256"],
        },
    )
    return candidate, audit_path


def test_seal_candidate_accepts_v2_and_rejects_former_v1_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, _audit = _build_candidate(repository)
    seal_module._load_candidate(candidate=candidate, repository_root=repository)

    manifest_path = candidate / "candidate-manifest.json"
    payload = json.loads(manifest_path.read_bytes())
    assert payload["schema"] == V2_SCHEMA_REPLACEMENTS["crux.oaf-seal-candidate/v1"]
    payload["schema"] = "crux.oaf-seal-candidate/v1"
    _write_json(manifest_path, payload)
    with pytest.raises(SealError, match="candidate manifest fields"):
        seal_module._load_candidate(candidate=candidate, repository_root=repository)


def _seal(
    repository: Path,
    candidate: Path,
    _audit: Path,
) -> tuple[seal_module.PublishedSeal, dict[str, Path]]:
    paths = _output_paths(repository)
    published = seal_module.seal_candidate(
        candidate=candidate,
        repository_root=repository,
    )
    return published, paths


def test_validate_host_strict_reads_existing_native_evidence_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_native_host(monkeypatch)
    path = _write_json(tmp_path / "native-host-evidence.json", _native_host_record())
    original = path.read_bytes()

    assert seal_module.main(["validate-host", "--evidence", os.fspath(path)]) == 0

    assert path.read_bytes() == original
    summary = strict_json_loads(capsys.readouterr().out.encode().rstrip(b"\n"))
    assert summary == {
        "exit_code": 0,
        "report_path": None,
        "report_sha256": None,
        "status": "validated",
    }


@pytest.mark.parametrize("mutation", ["duplicate", "extra", "noncanonical"])
def test_validate_host_rejects_non_strict_wrapper_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _set_native_host(monkeypatch)
    record = _native_host_record()
    path = tmp_path / "native-host-evidence.json"
    if mutation == "duplicate":
        path.write_bytes(
            canonical_json_bytes(record)[:-1]
            + b',"sha256":"'
            + str(record["sha256"]).encode()
            + b'"}\n'
        )
    elif mutation == "extra":
        record["schema"] = "invented-wrapper-schema"
        _write_json(path, record)
    else:
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(SealError, match="host evidence"):
        seal_module.load_native_host_evidence(path)


@pytest.mark.parametrize(
    ("system", "machine", "runner_arch", "kind"),
    [
        ("Darwin", "arm64", "X64", "github_hosted"),
        ("Linux", "aarch64", "X64", "github_hosted"),
        ("Linux", "x86_64", "ARM64", "github_hosted"),
        ("Linux", "x86_64", "X64", "emulated"),
    ],
)
def test_validate_host_rejects_non_native_or_emulated_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    runner_arch: str,
    kind: str,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("RUNNER_ARCH", runner_arch)
    path = _write_json(
        tmp_path / "native-host-evidence.json",
        _native_host_record(kind=kind),
    )

    with pytest.raises(SealError):
        seal_module.load_native_host_evidence(path)


def test_unspecified_native_producers_fail_closed_without_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_native_host(monkeypatch)
    evidence = _write_json(tmp_path / "native-host-evidence.json", _native_host_record())
    model_cache = tmp_path / "model-cache"
    model_cache.mkdir()
    bundle = tmp_path / "system-packages"
    build_args = tmp_path / "build-args.env"
    candidate = tmp_path / "candidate"

    assert (
        seal_module.main(
            [
                "materialize-system-packages",
                "--host-evidence",
                os.fspath(evidence),
                "--bundle",
                os.fspath(bundle),
                "--build-args-output",
                os.fspath(build_args),
            ]
        )
        == 1
    )
    assert not bundle.exists()
    assert not build_args.exists()
    capsys.readouterr()

    assert (
        seal_module.main(
            [
                "calibrate",
                "--request",
                os.fspath(tmp_path / "seal-profile-request.json"),
                "--measurement-evidence",
                os.fspath(tmp_path / "measurements.json"),
                "--bootstrap-request",
                os.fspath(tmp_path / "bootstrap-request.json"),
                "--bootstrap-evidence",
                os.fspath(tmp_path / "bootstrap-evidence.json"),
                "--host-attestation-bundle",
                os.fspath(tmp_path / "host-attestation-bundle.json"),
                "--checkpoint-evidence",
                os.fspath(tmp_path / "checkpoint-evidence.json"),
                "--base-system-evidence",
                os.fspath(tmp_path / "base-evidence.json"),
                "--host-evidence",
                os.fspath(evidence),
                "--model-cache",
                os.fspath(model_cache),
                "--output",
                os.fspath(candidate),
            ]
        )
        == 1
    )
    assert not candidate.exists()
    assert json.loads(capsys.readouterr().out) == {
        "exit_code": 1,
        "report_path": None,
        "report_sha256": None,
        "status": "failed",
    }


def _calibration_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    """Build only explicit fake authorities; no native measurement is implied."""

    _set_native_host(monkeypatch)
    monkeypatch.setattr(seal_module, "_require_imported_runtime_image", lambda _digest: None)
    bootstrap_request, bootstrap_evidence, _bootstrap_bundle, _bootstrap = (
        _bootstrap_evidence_fixture(tmp_path)
    )
    bundle, host, _host_record = _host_bundle_fixture(tmp_path, phase="measurement")
    request_path = tmp_path / "calibration-measurement-request.json"
    _write_json(
        request_path,
        {
            "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
            "calibration_bootstrap_evidence_sha256": _content_hash(bootstrap_evidence),
            "calibration_bootstrap_request_sha256": _content_hash(bootstrap_request),
            "fixture_derivation": {
                "algorithm": "repeat-pcm-s16le-v1",
                "canonical_header_bytes": 44,
                "channel_count": 1,
                "sample_rate": 44_100,
                "sample_width_bytes": 2,
                "source_path": "tests/fixtures/oaf_tf1_smoke/canonical.wav",
                "source_sha256": "b" * 64,
            },
            "fixtures": [
                {
                    "audio_frame_count": 10,
                    "input_audio_sha256": "a" * 64,
                    "input_view_id": "fake-input-10",
                    "source_audio_id": "fake-source-v1",
                    "source_audio_sha256": "b" * 64,
                    "wav_byte_length": 64,
                },
                {
                    "audio_frame_count": 20,
                    "input_audio_sha256": "c" * 64,
                    "input_view_id": "fake-input-20",
                    "source_audio_id": "fake-source-v1",
                    "source_audio_sha256": "b" * 64,
                    "wav_byte_length": 84,
                },
            ],
            "output_schemas": [
                V2_SCHEMA_REPLACEMENTS["crux.oaf-calibration-measurement-evidence/v1"]
            ],
            "repetition_count": 3,
            "required_metrics": ["measurement_row"],
            "schema": "crux.oaf-calibration-measurement-request/v1",
        },
    )
    cache = tmp_path / "model-cache"
    cache.mkdir()
    acquisition = importlib.import_module("src.benchmark.checkpoint_acquisition")
    checked_request = acquisition.load_checkpoint_acquisition_request(
        Path(
            "config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json"
        )
    )
    _, checkpoint_content = acquisition.render_checkpoint_acquisition_evidence(
        checked_request,
        acquisition_mode="cache_verify",
        model_artifact_set_sha256="c" * 64,
        cache_path=acquisition.PurePosixPath("sha256", "c" * 64),
    )
    checkpoint = tmp_path / "checkpoint-evidence.json"
    checkpoint.write_bytes(checkpoint_content)
    packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    base_request_path = Path("runtime/oaf_tf1/base-system-package-request.json")
    base_request = packages.load_base_system_package_request(base_request_path)
    base_payload = _base_system_evidence_payload(base_request.sha256)
    base_payload["base_image_archive_keyring_sha256"] = (
        base_request.base_image_archive_keyring_sha256
    )
    base_payload["package_inventory_sha256"] = packages.inventory_sha256(
        base_payload["package_inventory"]
    )
    base = _write_json(tmp_path / "base-evidence.json", base_payload)
    return (
        request_path,
        bootstrap_request,
        bootstrap_evidence,
        bundle,
        host,
        cache,
        checkpoint,
        base,
        tmp_path / "measurements.json",
    )


def _fake_measurement_row(
    frame_count: int, repetition: int, process: str | None = None
) -> dict[str, object]:
    return {
        "exit_code": 0,
        "inference_call_count_after": 1,
        "inference_call_count_before": 0,
        "input_audio_sha256": ("a" if frame_count == 10 else "c") * 64,
        "input_frame_count": frame_count,
        "oom_killed": False,
        "peak_cpu_millis": 40,
        "peak_pid_count": 3,
        "peak_rss_bytes": 400,
        "peak_shm_bytes": 40,
        "peak_tmp_bytes": 30,
        "prediction_sha256": "d" * 64,
        "process_instance_id": process or f"fake-{frame_count}-{repetition}",
        "repetition": repetition,
        "request_millis": 20,
        "signal": None,
        "startup_millis": 10,
        "stderr_max_line_bytes": 9,
        "stdout_max_line_bytes": 8,
    }


def test_measure_publishes_only_diagnostic_evidence_with_fake_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        request,
        bootstrap_request,
        bootstrap_evidence,
        bundle,
        host,
        cache,
        checkpoint,
        base,
        output,
    ) = _calibration_inputs(tmp_path, monkeypatch)

    artifact = seal_module.measure(
        request_path=request,
        bootstrap_request_path=bootstrap_request,
        bootstrap_evidence_path=bootstrap_evidence,
        host_attestation_bundle_path=bundle,
        host_evidence_path=host,
        model_cache=cache,
        checkpoint_evidence_path=checkpoint,
        base_system_evidence_path=base,
        output_path=output,
        runner=lambda _request, frame, repetition: _fake_measurement_row(frame, repetition),
    )

    payload = json.loads(output.read_bytes())
    assert artifact.path == output
    assert (
        payload["schema"] == V2_SCHEMA_REPLACEMENTS["crux.oaf-calibration-measurement-evidence/v1"]
    )
    assert not (output.parent / "candidate-manifest.json").exists()
    assert [
        (row["input_frame_count"], row["repetition"]) for row in payload["measurement_rows"]
    ] == [(10, 1), (10, 2), (10, 3), (20, 1), (20, 2), (20, 3)]


def test_measurement_evidence_accepts_v2_and_rejects_former_v1_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        request,
        bootstrap_request,
        bootstrap_evidence,
        bundle,
        host,
        cache,
        checkpoint,
        base,
        measurement,
    ) = _calibration_inputs(tmp_path, monkeypatch)
    seal_module.measure(
        request_path=request,
        bootstrap_request_path=bootstrap_request,
        bootstrap_evidence_path=bootstrap_evidence,
        host_attestation_bundle_path=bundle,
        host_evidence_path=host,
        model_cache=cache,
        checkpoint_evidence_path=checkpoint,
        base_system_evidence_path=base,
        output_path=measurement,
        runner=lambda _request, frame, repetition: _fake_measurement_row(frame, repetition),
    )
    payload = json.loads(measurement.read_bytes())
    assert (
        payload["schema"] == V2_SCHEMA_REPLACEMENTS["crux.oaf-calibration-measurement-evidence/v1"]
    )
    request_value = seal_module.load_calibration_measurement_request(request)
    seal_module._load_measurement_evidence(measurement, request_value)

    payload["schema"] = "crux.oaf-calibration-measurement-evidence/v1"
    _write_json(measurement, payload)
    with pytest.raises(SealError, match="calibration measurement evidence fields"):
        seal_module._load_measurement_evidence(measurement, request_value)


def test_measure_rejects_bootstrap_hash_and_phase_bundle_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        request,
        bootstrap_request,
        bootstrap_evidence,
        measurement_bundle,
        measurement_host,
        cache,
        checkpoint,
        base,
        output,
    ) = _calibration_inputs(tmp_path, monkeypatch)
    payload = json.loads(request.read_bytes())
    payload["calibration_bootstrap_evidence_sha256"] = "0" * 64
    _write_json(request, payload)
    with pytest.raises(SealError, match="bind the accepted bootstrap"):
        seal_module.measure(
            request_path=request,
            bootstrap_request_path=bootstrap_request,
            bootstrap_evidence_path=bootstrap_evidence,
            host_attestation_bundle_path=measurement_bundle,
            host_evidence_path=measurement_host,
            model_cache=cache,
            checkpoint_evidence_path=checkpoint,
            base_system_evidence_path=base,
            output_path=output,
            runner=lambda _request, frame, repetition: _fake_measurement_row(frame, repetition),
        )

    payload["calibration_bootstrap_evidence_sha256"] = _content_hash(bootstrap_evidence)
    _write_json(request, payload)
    bootstrap_bundle = bootstrap_evidence.parent / "bootstrap-host-attestation"
    with pytest.raises(SealError, match="phase"):
        seal_module.measure(
            request_path=request,
            bootstrap_request_path=bootstrap_request,
            bootstrap_evidence_path=bootstrap_evidence,
            host_attestation_bundle_path=bootstrap_bundle / "attestation-bundle.json",
            host_evidence_path=bootstrap_bundle / "native-host-evidence.json",
            model_cache=cache,
            checkpoint_evidence_path=checkpoint,
            base_system_evidence_path=base,
            output_path=output,
            runner=lambda _request, frame, repetition: _fake_measurement_row(frame, repetition),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(deepcopy(rows[0])),
        lambda rows: rows.append({**deepcopy(rows[0]), "process_instance_id": "other"}),
    ],
)
def test_measurement_evidence_rejects_incomplete_or_duplicate_probe_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[list[dict[str, object]]], None],
) -> None:
    (
        request,
        bootstrap_request,
        bootstrap_evidence,
        bundle,
        host,
        cache,
        checkpoint,
        base,
        measurement,
    ) = _calibration_inputs(tmp_path, monkeypatch)
    seal_module.measure(
        request_path=request,
        bootstrap_request_path=bootstrap_request,
        bootstrap_evidence_path=bootstrap_evidence,
        host_attestation_bundle_path=bundle,
        host_evidence_path=host,
        model_cache=cache,
        checkpoint_evidence_path=checkpoint,
        base_system_evidence_path=base,
        output_path=measurement,
        runner=lambda _request, frame, repetition: _fake_measurement_row(frame, repetition),
    )
    payload = json.loads(measurement.read_text(encoding="utf-8"))
    mutation(payload["measurement_rows"])
    payload["measurement_rows"].sort(
        key=lambda row: (row["input_frame_count"], row["process_instance_id"], row["repetition"])
    )
    _write_json(measurement, payload)

    with pytest.raises(SealError, match="matrix"):
        seal_module._load_measurement_evidence(
            measurement, seal_module.load_calibration_measurement_request(request)
        )


def _profile_payload(
    *, measurement_request: Path, measurement_evidence: Path, checkpoint: Path, base: Path
) -> dict[str, object]:
    acquisition = importlib.import_module("src.benchmark.checkpoint_acquisition")
    packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    checkpoint_request = acquisition.load_checkpoint_acquisition_request(
        Path(
            "config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json"
        )
    )
    checkpoint_evidence = acquisition.load_checkpoint_acquisition_evidence(
        checkpoint, request=checkpoint_request
    )
    base_request = packages.load_base_system_package_request(
        Path("runtime/oaf_tf1/base-system-package-request.json")
    )
    base_evidence = packages.load_base_system_package_evidence(base, request=base_request)
    measurement_request_payload = json.loads(measurement_request.read_bytes())
    return {
        "base_system_package_evidence_sha256": base_evidence.sha256,
        "base_system_package_request_sha256": base_evidence.request_sha256,
        "calibration_bootstrap_evidence_sha256": measurement_request_payload[
            "calibration_bootstrap_evidence_sha256"
        ],
        "calibration_bootstrap_request_sha256": measurement_request_payload[
            "calibration_bootstrap_request_sha256"
        ],
        "calibration_measurement_evidence_sha256": _content_hash(measurement_evidence),
        "calibration_measurement_request_sha256": _content_hash(measurement_request),
        "checkpoint_acquisition_evidence_sha256": checkpoint_evidence.sha256,
        "checkpoint_acquisition_request_sha256": checkpoint_evidence.request_sha256,
        "cpu_limit_millis": 41,
        "max_input_audio_frames": 20,
        "memory_limit_bytes": 401,
        "pid_limit": 4,
        "request_deadline_seconds": 21,
        "runtime_gid": 65_532,
        "runtime_uid": 65_532,
        "schema": "crux.oaf-seal-profile-request/v1",
        "shm_bytes": 41,
        "startup_deadline_seconds": 11,
        "stderr_max_line_bytes": 10,
        "stderr_read_chunk_bytes": 11,
        "stderr_ring_buffer_bytes": 12,
        "stdout_max_line_bytes": 9,
        "tmp_bytes": 31,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("memory_limit_bytes"),
        lambda payload: payload.__setitem__("memory_limit_bytes", 0),
        lambda payload: payload.__setitem__("calibration_bootstrap_evidence_sha256", "f" * 64),
        lambda payload: payload.__setitem__("calibration_measurement_evidence_sha256", "f" * 64),
        lambda payload: payload.__setitem__("memory_limit_bytes", 400),
    ],
)
def test_calibrate_rejects_missing_unrelated_sentinel_or_underprovisioned_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: Callable[[dict[str, object]], None]
) -> None:
    (
        request,
        bootstrap_request,
        bootstrap_evidence,
        bundle,
        host,
        cache,
        checkpoint,
        base,
        measurement,
    ) = _calibration_inputs(tmp_path, monkeypatch)
    seal_module.measure(
        request_path=request,
        bootstrap_request_path=bootstrap_request,
        bootstrap_evidence_path=bootstrap_evidence,
        host_attestation_bundle_path=bundle,
        host_evidence_path=host,
        model_cache=cache,
        checkpoint_evidence_path=checkpoint,
        base_system_evidence_path=base,
        output_path=measurement,
        runner=lambda _request, frame, repetition: _fake_measurement_row(frame, repetition),
    )
    profile = _profile_payload(
        measurement_request=request,
        measurement_evidence=measurement,
        checkpoint=checkpoint,
        base=base,
    )
    mutation(profile)
    profile_path = tmp_path / "seal-profile-request.json"
    _write_json(profile_path, profile)
    candidate_bundle, candidate_host, _candidate_record = _host_bundle_fixture(
        tmp_path,
        phase="candidate",
    )

    with pytest.raises(SealError):
        seal_module.calibrate(
            request_path=profile_path,
            measurement_evidence_path=measurement,
            bootstrap_request_path=bootstrap_request,
            bootstrap_evidence_path=bootstrap_evidence,
            host_attestation_bundle_path=candidate_bundle,
            checkpoint_evidence_path=checkpoint,
            base_system_evidence_path=base,
            host_evidence=candidate_host,
            model_cache=cache,
            output=tmp_path / "candidate",
            runner=lambda frame, _persistent, _ordinal: (
                _fake_measurement_row(frame, 1),
                frame <= 20,
            ),
        )


def test_calibrate_exercises_bound_process_cases_with_explicit_fake_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        request,
        bootstrap_request,
        bootstrap_evidence,
        _measurement_bundle,
        _measurement_host,
        cache,
        checkpoint,
        base,
        measurement,
    ) = _calibration_inputs(tmp_path, monkeypatch)
    seal_module.measure(
        request_path=request,
        bootstrap_request_path=bootstrap_request,
        bootstrap_evidence_path=bootstrap_evidence,
        host_attestation_bundle_path=_measurement_bundle,
        host_evidence_path=_measurement_host,
        model_cache=cache,
        checkpoint_evidence_path=checkpoint,
        base_system_evidence_path=base,
        output_path=measurement,
        runner=lambda _request, frame, repetition: _fake_measurement_row(frame, repetition),
    )
    bundle, host, _host_record = _host_bundle_fixture(tmp_path, phase="candidate")
    profile_path = tmp_path / "seal-profile-request.json"
    _write_json(
        profile_path,
        _profile_payload(
            measurement_request=request,
            measurement_evidence=measurement,
            checkpoint=checkpoint,
            base=base,
        ),
    )
    calls: list[tuple[int, bool, int]] = []

    def fake_runner(frame: int, persistent: bool, ordinal: int) -> dict[str, object]:
        calls.append((frame, persistent, ordinal))
        row = _fake_measurement_row(frame, 1, "persistent" if persistent else "fresh")
        if frame > 20:
            row["inference_call_count_after"] = row["inference_call_count_before"]
            row["prediction_sha256"] = None
            return {"rejected_before_inference": True, "row": row}
        return {"rejected_before_inference": False, "row": row}

    with pytest.raises(SealError, match="generator is unavailable"):
        seal_module.calibrate(
            request_path=profile_path,
            measurement_evidence_path=measurement,
            bootstrap_request_path=bootstrap_request,
            bootstrap_evidence_path=bootstrap_evidence,
            host_attestation_bundle_path=bundle,
            checkpoint_evidence_path=checkpoint,
            base_system_evidence_path=base,
            host_evidence=host,
            model_cache=cache,
            output=tmp_path / "candidate.json",
            runner=fake_runner,
        )

    assert calls == [
        (19, True, 1),
        (20, True, 2),
        (21, True, 3),
        (20, False, 1),
    ]
    assert not (tmp_path / "candidate.json").exists()


def test_measure_constructs_native_runner_when_no_test_callback_is_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        request,
        bootstrap_request,
        bootstrap_evidence,
        bundle,
        host,
        cache,
        checkpoint,
        base,
        measurement,
    ) = _calibration_inputs(tmp_path, monkeypatch)
    closed: list[bool] = []

    class FakeNativeRunner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        @staticmethod
        def measure(_request: object, frame: int, repetition: int) -> dict[str, object]:
            return _fake_measurement_row(frame, repetition)

        def close(self) -> None:
            closed.append(True)

    native_module = importlib.import_module("tools.hpa320.oaf_native_runner")
    monkeypatch.setattr(native_module, "NativeCalibrationRunner", FakeNativeRunner)

    published = seal_module.measure(
        request_path=request,
        bootstrap_request_path=bootstrap_request,
        bootstrap_evidence_path=bootstrap_evidence,
        host_attestation_bundle_path=bundle,
        host_evidence_path=host,
        model_cache=cache,
        checkpoint_evidence_path=checkpoint,
        base_system_evidence_path=base,
        output_path=measurement,
    )

    assert published.path == measurement
    assert closed == [True]


def test_calibrate_constructs_native_runner_and_internal_candidate_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        request,
        bootstrap_request,
        bootstrap_evidence,
        measurement_bundle,
        measurement_host,
        cache,
        checkpoint,
        base,
        measurement,
    ) = _calibration_inputs(tmp_path, monkeypatch)
    seal_module.measure(
        request_path=request,
        bootstrap_request_path=bootstrap_request,
        bootstrap_evidence_path=bootstrap_evidence,
        host_attestation_bundle_path=measurement_bundle,
        host_evidence_path=measurement_host,
        model_cache=cache,
        checkpoint_evidence_path=checkpoint,
        base_system_evidence_path=base,
        output_path=measurement,
        runner=lambda _request, frame, repetition: _fake_measurement_row(frame, repetition),
    )
    profile_path = tmp_path / "seal-profile-request.json"
    _write_json(
        profile_path,
        _profile_payload(
            measurement_request=request,
            measurement_evidence=measurement,
            checkpoint=checkpoint,
            base=base,
        ),
    )
    candidate_bundle, candidate_host, _record = _host_bundle_fixture(tmp_path, phase="candidate")
    reached: list[str] = []

    class FakeNativeRunner:
        smoke_native_events = ({"native_midi_note": 36},)

        def __init__(self, **kwargs: object) -> None:
            evidence_root = Path(str(kwargs["candidate_evidence_root"]))
            assert stat.S_IMODE(evidence_root.stat().st_mode) == 0o733
            _write_json(
                evidence_root / "tensor-coverage.json",
                {"schema": "crux.oaf-tensor-coverage/v1"},
            )

        @staticmethod
        def probe(frame: int, persistent: bool, ordinal: int) -> dict[str, object]:
            row = _fake_measurement_row(frame, ordinal, "persistent" if persistent else "fresh")
            if frame > 20:
                row["inference_call_count_after"] = row["inference_call_count_before"]
                row["prediction_sha256"] = None
                return {"rejected_before_inference": True, "row": row}
            return {"rejected_before_inference": False, "row": row}

        def close(self) -> None:
            pass

        def smoke(self) -> tuple[dict[str, int], ...]:
            reached.append("smoke")
            return self.smoke_native_events

    def fake_builder(**kwargs: object) -> None:
        staging = Path(str(kwargs["staging"]))
        assert (staging / CANDIDATE_ARTIFACT_PATHS["native_host_attestation_bundle"]).is_file()
        assert kwargs["native_events"] == FakeNativeRunner.smoke_native_events
        reached.append("builder")
        raise SealError("native builder reached")

    native_module = importlib.import_module("tools.hpa320.oaf_native_runner")
    builder_module = importlib.import_module("tools.hpa320.oaf_candidate_builder")
    monkeypatch.setattr(native_module, "NativeCalibrationRunner", FakeNativeRunner)
    monkeypatch.setattr(builder_module, "build_native_candidate", fake_builder)

    with pytest.raises(SealError, match="native builder reached"):
        seal_module.calibrate(
            request_path=profile_path,
            measurement_evidence_path=measurement,
            bootstrap_request_path=bootstrap_request,
            bootstrap_evidence_path=bootstrap_evidence,
            host_attestation_bundle_path=candidate_bundle,
            checkpoint_evidence_path=checkpoint,
            base_system_evidence_path=base,
            host_evidence=candidate_host,
            model_cache=cache,
            output=tmp_path / "candidate",
        )

    assert reached == ["smoke", "builder"]
    assert not (tmp_path / "candidate").exists()


def _base_system_request_payload(
    *, additional_system_packages: list[str] | None = None
) -> dict[str, object]:
    return {
        "additional_system_packages": (
            [] if additional_system_packages is None else additional_system_packages
        ),
        "base_image": "python:3.7.17-slim-bullseye",
        "base_image_archive_keyring_sha256": "a" * 64,
        "base_image_manifest_digest": (
            "sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673"
        ),
        "platform": "linux/amd64",
        "required_probes": [
            "base_python_version",
            "runtime_python_version",
            "runtime_tensorflow_version",
            "runtime_smoke",
        ],
        "schema": "crux.oaf-base-system-package-request/v1",
    }


def _base_system_evidence_payload(
    request_sha256: str,
    *,
    inventory: list[dict[str, str]] | None = None,
    probes: list[dict[str, str]] | None = None,
    manifest_digest: str = "sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673",
) -> dict[str, object]:
    package_inventory = (
        [{"architecture": "amd64", "name": "base-files", "version": "11.1+deb11u11"}]
        if inventory is None
        else inventory
    )
    probe_rows = (
        [
            {"name": "base_python_version", "value": "Python 3.7.17"},
            {"name": "runtime_python_version", "value": "Python 3.7.17"},
            {"name": "runtime_tensorflow_version", "value": "1.15.5"},
            {"name": "runtime_smoke", "value": "passed"},
        ]
        if probes is None
        else probes
    )
    return {
        "additional_system_packages": [],
        "base_image_archive_keyring_sha256": "a" * 64,
        "base_image_manifest_digest": manifest_digest,
        "native_host_evidence": _native_host_record(),
        "package_inventory": package_inventory,
        "package_inventory_sha256": "0" * 64,
        "probes": probe_rows,
        "request_sha256": request_sha256,
        "schema": V2_SCHEMA_REPLACEMENTS["crux.oaf-base-system-package-evidence/v1"],
    }


def test_base_system_request_rejects_additional_packages_and_unapproved_probes(
    tmp_path: Path,
) -> None:
    system_packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    request = _base_system_request_payload(additional_system_packages=["libgomp1"])
    request_path = _write_json(tmp_path / "request.json", request)

    with pytest.raises(system_packages.SystemPackageError, match="additional system packages"):
        system_packages.load_base_system_package_request(request_path)

    request = _base_system_request_payload()
    request["required_probes"] = ["runtime_shell"]
    request_path = _write_json(tmp_path / "unapproved-request.json", request)
    with pytest.raises(system_packages.SystemPackageError, match="required probe"):
        system_packages.load_base_system_package_request(request_path)


@pytest.mark.parametrize(
    ("inventory", "probes", "manifest_digest", "error"),
    [
        (
            [
                {"architecture": "amd64", "name": "zlib1g", "version": "1"},
                {"architecture": "amd64", "name": "base-files", "version": "1"},
            ],
            None,
            "sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673",
            "sorted",
        ),
        (
            [
                {"architecture": "amd64", "name": "base-files", "version": "1"},
                {"architecture": "amd64", "name": "base-files", "version": "1"},
            ],
            None,
            "sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673",
            "duplicate",
        ),
        (
            None,
            [{"name": "runtime_shell", "value": "passed"}],
            "sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673",
            "probes",
        ),
        (None, None, "sha256:" + "b" * 64, "base image manifest"),
    ],
)
def test_base_system_evidence_rejects_untrusted_rows(
    tmp_path: Path,
    inventory: list[dict[str, str]] | None,
    probes: list[dict[str, str]] | None,
    manifest_digest: str,
    error: str,
) -> None:
    system_packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    request_path = _write_json(tmp_path / "request.json", _base_system_request_payload())
    request = system_packages.load_base_system_package_request(request_path)
    evidence = _base_system_evidence_payload(
        request.sha256,
        inventory=inventory,
        probes=probes,
        manifest_digest=manifest_digest,
    )
    if error not in {"duplicate", "sorted"}:
        evidence["package_inventory_sha256"] = system_packages.inventory_sha256(
            evidence["package_inventory"]
        )
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)

    with pytest.raises(system_packages.SystemPackageError, match=error):
        system_packages.load_base_system_package_evidence(evidence_path, request=request)


def test_base_system_evidence_reproduces_exact_inventory(tmp_path: Path) -> None:
    system_packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    request_path = _write_json(tmp_path / "request.json", _base_system_request_payload())
    request = system_packages.load_base_system_package_request(request_path)
    evidence = _base_system_evidence_payload(request.sha256)
    evidence["package_inventory_sha256"] = system_packages.inventory_sha256(
        evidence["package_inventory"]
    )
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)

    loaded = system_packages.load_base_system_package_evidence(evidence_path, request=request)

    assert loaded.request_sha256 == request.sha256
    assert loaded.base_image_manifest_digest == request.base_image_manifest_digest
    assert loaded.package_inventory_sha256 == system_packages.inventory_sha256(
        loaded.package_inventory
    )
    assert request.additional_system_packages == ()


def test_base_system_evidence_accepts_v2_and_rejects_former_v1_schema(tmp_path: Path) -> None:
    system_packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    request_path = _write_json(tmp_path / "request.json", _base_system_request_payload())
    request = system_packages.load_base_system_package_request(request_path)
    evidence = _base_system_evidence_payload(request.sha256)
    evidence["package_inventory_sha256"] = system_packages.inventory_sha256(
        evidence["package_inventory"]
    )
    assert evidence["schema"] == V2_SCHEMA_REPLACEMENTS["crux.oaf-base-system-package-evidence/v1"]
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    system_packages.load_base_system_package_evidence(evidence_path, request=request)

    evidence["schema"] = "crux.oaf-base-system-package-evidence/v1"
    _write_json(evidence_path, evidence)
    with pytest.raises(system_packages.SystemPackageError, match="base-system evidence fields"):
        system_packages.load_base_system_package_evidence(evidence_path, request=request)


def _mock_base_system_attestation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[bytes]:
        rendered = " ".join(command)
        if "image inspect" in rendered:
            output = (
                b"python:3.7.17-slim-bullseye@sha256:"
                b"ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673\n"
            )
        elif "dpkg-query" in rendered:
            output = b"base-files\t11.1+deb11u11\tamd64\n"
        elif "sha256sum" in rendered:
            output = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  keyring\n"
        elif "tensorflow" in rendered:
            output = b"1.15.5\n"
        elif "entrypoint" in rendered:
            output = b"passed\n"
        else:
            output = b"Python 3.7.17\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(seal_module.subprocess, "run", fake_run)
    monkeypatch.setattr(seal_module, "_require_imported_runtime_image", lambda _digest: None)


def _base_attestation_authority(
    tmp_path: Path,
    request_payload: dict[str, object],
) -> tuple[Path, Path, Path, Path]:
    bootstrap_request, bootstrap_evidence, bundle, _evidence = _bootstrap_evidence_fixture(tmp_path)
    base_request_path = (
        bootstrap_request.parents[3] / "runtime/oaf_tf1/base-system-package-request.json"
    )
    _write_json(base_request_path, request_payload)
    bootstrap_payload = json.loads(bootstrap_request.read_bytes())
    bootstrap_payload["base_system_package_request_sha256"] = _content_hash(base_request_path)
    _write_json(bootstrap_request, bootstrap_payload)
    evidence_payload = json.loads(bootstrap_evidence.read_bytes())
    evidence_payload["calibration_bootstrap_request_sha256"] = _content_hash(bootstrap_request)
    _write_json(bootstrap_evidence, evidence_payload)
    return (
        base_request_path,
        bootstrap_request,
        bootstrap_evidence,
        bundle,
    )


def test_attest_base_system_publishes_immutable_native_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_native_host(monkeypatch)
    request_path, bootstrap_request, bootstrap_evidence, bundle = _base_attestation_authority(
        tmp_path, _base_system_request_payload()
    )
    host_path = bundle.parent / "native-host-evidence.json"
    output_path = tmp_path / "base-system-package-evidence.json"
    _mock_base_system_attestation(monkeypatch)

    published = seal_module.attest_base_system(
        request_path=request_path,
        bootstrap_request_path=bootstrap_request,
        bootstrap_evidence_path=bootstrap_evidence,
        host_attestation_bundle_path=bundle,
        host_evidence_path=host_path,
        output_path=output_path,
    )
    request = importlib.import_module(
        "tools.hpa320.oaf_system_packages"
    ).load_base_system_package_request(request_path)
    evidence = importlib.import_module(
        "tools.hpa320.oaf_system_packages"
    ).load_base_system_package_evidence(output_path, request=request)

    assert published.path == output_path
    assert published.sha256 == evidence.sha256
    assert evidence.request_sha256 == request.sha256
    assert evidence.package_inventory[0].name == "base-files"
    rerun = seal_module.attest_base_system(
        request_path=request_path,
        bootstrap_request_path=bootstrap_request,
        bootstrap_evidence_path=bootstrap_evidence,
        host_attestation_bundle_path=bundle,
        host_evidence_path=host_path,
        output_path=output_path,
    )

    assert rerun == published


def test_attest_base_system_rejects_different_existing_evidence_without_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_native_host(monkeypatch)
    request_path, bootstrap_request, bootstrap_evidence, bundle = _base_attestation_authority(
        tmp_path, _base_system_request_payload()
    )
    host_path = bundle.parent / "native-host-evidence.json"
    output_path = tmp_path / "base-system-package-evidence.json"
    _mock_base_system_attestation(monkeypatch)
    output_path.write_bytes(b"different immutable evidence\n")
    original = output_path.read_bytes()

    with pytest.raises(SealError, match="publication failed"):
        seal_module.attest_base_system(
            request_path=request_path,
            bootstrap_request_path=bootstrap_request,
            bootstrap_evidence_path=bootstrap_evidence,
            host_attestation_bundle_path=bundle,
            host_evidence_path=host_path,
            output_path=output_path,
        )

    assert output_path.read_bytes() == original


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("memory_limit_bytes"),
        lambda payload: payload.__setitem__("memory_limit_bytes", "auto"),
        lambda payload: payload.__setitem__("max_input_audio_frames", 0),
    ],
)
def test_seal_rejects_missing_sentinel_or_zero_calibration_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(repository, seal_mutation=mutation)
    paths = _output_paths(repository)

    with pytest.raises(SealError, match="seal evidence|backend lock"):
        _seal(repository, candidate, audit)

    assert not any(path.exists() for path in paths.values())


def test_seal_rejects_count_only_tensor_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(
        repository,
        tensor_payload={
            "checkpoint_count": 130,
            "non_inference_count": 52,
            "required_count": 78,
            "restored_count": 78,
            "schema": "crux.oaf-tensor-coverage/v1",
        },
    )

    with pytest.raises(SealError, match="exact tensor inventories"):
        _seal(repository, candidate, audit)


def test_seal_rejects_oci_archive_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(repository)
    archive = _candidate_artifact_path(candidate, "oci_layout_archive")
    archive.write_bytes(archive.read_bytes() + b"drift")

    with pytest.raises(SealError, match="OCI archive"):
        _seal(repository, candidate, audit)


def test_seal_rejects_observed_hdf5_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(
        repository,
        audit_mutation=lambda payload: payload.__setitem__("observed_hdf5_sha256", "0" * 64),
    )

    with pytest.raises(SealError, match="HDF5"):
        _seal(repository, candidate, audit)


def test_seal_rejects_final_lock_hash_cycle_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(
        repository,
        seal_mutation=lambda payload: payload.__setitem__("runtime_lock_sha256", "f" * 64),
    )

    with pytest.raises(SealError, match="final lock hashes"):
        _seal(repository, candidate, audit)

    assert not any(path.exists() for path in _output_paths(repository).values())


def test_runtime_publication_failure_never_publishes_backend_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(repository)
    paths = _output_paths(repository)
    paths["runtime_lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["runtime_lock"].write_bytes(b"prior immutable runtime\n")

    with pytest.raises(SealError, match="publication"):
        _seal(repository, candidate, audit)

    assert paths["runtime_lock"].read_bytes() == b"prior immutable runtime\n"
    assert not paths["backend_lock"].exists()


def test_seal_is_deterministic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_native_host(monkeypatch)
    repository = _make_repository(tmp_path)
    candidate, audit = _build_candidate(repository)

    first, paths = _seal(repository, candidate, audit)
    first_bytes = {name: path.read_bytes() for name, path in paths.items()}
    second, _ = _seal(repository, candidate, audit)

    assert first == second
    assert {name: path.read_bytes() for name, path in paths.items()} == first_bytes
    assert list(first.publication_order)[-2:] == ["runtime_lock", "backend_lock"]
    host_manifest = strict_json_loads(
        paths["host_adapter_source_manifest"].read_bytes()[:-1],
        require_canonical=True,
    )
    assert [row["path"] for row in host_manifest["files"]] == list(  # type: ignore[index]
        HOST_PATHS
    )
    assert host_manifest["covered_roots"] == list(HOST_PATHS)  # type: ignore[index]
