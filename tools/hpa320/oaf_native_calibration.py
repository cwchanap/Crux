#!/usr/bin/env python3
"""Native-only deterministic OaF image bootstrap orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from src.benchmark.backend_identity import JsonValue, canonical_json_bytes, strict_json_loads
from src.benchmark.backend_publication import read_regular_file_no_follow
from src.benchmark.backends import PublishedArtifact
from tools.hpa320.oaf_build_context import (
    load_build_context_manifest,
    materialize_build_context,
)
from tools.hpa320.oaf_host_attestation import (
    bundle_phase,
    load_native_host_attestation_bundle,
)
from tools.hpa320.oaf_oci import (
    BaseImageIdentity,
    ImageBuildRecipe,
    OciLayoutIdentity,
    authenticate_base_image,
    canonical_pack_oci_layout,
    inspect_oci_layout,
    require_identical_oci_builds,
)
from tools.hpa320.seal_oaf_backend import SealError, load_calibration_bootstrap_request

BOOTSTRAP_EVIDENCE_NAME = "calibration-bootstrap-evidence.json"
OCI_LAYOUT_MANIFEST_NAME = "oci-layout-manifest.json"
OCI_ARCHIVE_NAME = "runtime.oci.tar"
BOOTSTRAP_EVIDENCE_SCHEMA = "crux.oaf-calibration-bootstrap-evidence/v2"
OCI_LAYOUT_MANIFEST_SCHEMA = "crux.oaf-oci-layout-manifest/v1"


def _buildx_build_command(
    buildx: Path,
    builder: str,
    output_directory: Path,
) -> tuple[str, ...]:
    return (
        os.fspath(buildx),
        "build",
        "--builder",
        builder,
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
            f"rewrite-timestamp=true,dest={output_directory}"
        ),
        ".",
    )


def import_authenticated_oci_archive(
    archive_path: Path,
    expected: OciLayoutIdentity,
    docker_executable: str = "docker",
) -> str:
    """Import exact archive bytes and return the re-inspected immutable config digest."""

    archive = Path(archive_path)
    content = _read_file(archive, "authenticated OCI archive")
    if (
        archive.name != expected.archive.name
        or len(content) != expected.archive.size
        or hashlib.sha256(content).hexdigest() != expected.archive.sha256
    ):
        raise SealError("authenticated OCI archive identity does not match")
    _run_checked(
        (docker_executable, "image", "load", "--input", os.fspath(archive)),
        "authenticated OCI archive import failed",
    )
    result = _run_checked(
        (
            docker_executable,
            "image",
            "inspect",
            expected.config_digest,
            "--format",
            "{{.Id}}\n{{.Architecture}}\n{{.Os}}",
        ),
        "imported OCI image inspection failed",
    )
    if result.stdout.strip().splitlines() != [expected.config_digest, "amd64", "linux"]:
        raise SealError("imported OCI image config digest or platform differs")
    return expected.config_digest


def bootstrap_image(
    *,
    request_path: Path,
    host_attestation_bundle_path: Path,
    host_evidence_path: Path,
    output: Path,
    repository_root: Path,
) -> PublishedArtifact:
    """Build twice on native Linux X64 and publish only identical authenticated output."""

    _require_native_linux_amd64()
    request = load_calibration_bootstrap_request(Path(request_path))
    repository = Path(repository_root)
    expected_repository = Path(request_path).parents[3]
    if repository.resolve() != expected_repository.resolve():
        raise SealError("calibration bootstrap repository root does not own the request")
    bundle_path = Path(host_attestation_bundle_path)
    bundle = load_native_host_attestation_bundle(
        bundle_path,
        expected_phase=bundle_phase(bundle_path),
    )
    host_content = _read_file(Path(host_evidence_path), "native host evidence")
    bundled_host = _read_file(
        Path(host_attestation_bundle_path).parent / bundle.native_host_evidence.name,
        "bundled native host evidence",
    )
    if (
        host_content != bundled_host
        or len(host_content) != bundle.native_host_evidence.size
        or hashlib.sha256(host_content).hexdigest() != bundle.native_host_evidence.sha256
    ):
        raise SealError("native host evidence does not match the attestation bundle")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".hpa320-bootstrap-", dir=os.fspath(target.parent)))
    try:
        published = _bootstrap_in_staging(
            request=request,
            repository=repository,
            bundle_sha256=bundle.sha256,
            host_content=host_content,
            staging_root=staging_root,
        )
        publication = staging_root / "publication"
        _publish_directory_immutable(publication, target)
        return PublishedArtifact(
            role=published.role,
            path=target / published.path.name,
            sha256=published.sha256,
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _bootstrap_in_staging(
    *,
    request: Any,
    repository: Path,
    bundle_sha256: str,
    host_content: bytes,
    staging_root: Path,
) -> PublishedArtifact:
    manifest_path = repository / "runtime/oaf_tf1/build-context-manifest.json"
    manifest = load_build_context_manifest(manifest_path)
    if manifest.sha256 != request.build_context_manifest_sha256:
        raise SealError("build-context manifest hash does not match bootstrap request")
    buildx = _download_verified_buildx(request.image_build, staging_root / "tool")
    _require_frontend_source(repository, request.image_build)
    _observe_buildx_version(buildx, request.image_build)
    base = _fetch_base_image(request.payload["base_image_manifest_digest"])
    identities: list[OciLayoutIdentity] = []
    archives: list[Path] = []
    for ordinal in (1, 2):
        context = staging_root / f"context-{ordinal}"
        layout = staging_root / f"layout-{ordinal}"
        archive = staging_root / f"runtime-{ordinal}.oci.tar"
        state = staging_root / f"buildx-state-{ordinal}"
        layout.mkdir()
        state.mkdir()
        materialize_build_context(
            repository_root=repository,
            wheelhouse_root=repository / "runtime/oaf_tf1/wheelhouse",
            manifest_path=manifest_path,
            destination=context,
        )
        builder = f"hpa320-{ordinal}-{uuid4().hex}"
        _run_builder(
            buildx=buildx,
            builder=builder,
            state=state,
            context=context,
            output=layout,
            recipe=request.image_build,
        )
        canonical_pack_oci_layout(layout, archive, request.image_build.oci_archive)
        identities.append(inspect_oci_layout(layout, archive, request.image_build, base))
        archives.append(archive)
    identity = require_identical_oci_builds(
        identities[0],
        identities[1],
        archives[0],
        archives[1],
    )
    publication = staging_root / "publication"
    publication.mkdir()
    final_archive = publication / OCI_ARCHIVE_NAME
    shutil.copyfile(archives[0], final_archive)
    final_identity = OciLayoutIdentity(
        archive=type(identity.archive)(
            name=OCI_ARCHIVE_NAME,
            sha256=identity.archive.sha256,
            size=identity.archive.size,
        ),
        base_image_config_digest=identity.base_image_config_digest,
        base_image_layer_digests=identity.base_image_layer_digests,
        base_image_layer_diff_ids=identity.base_image_layer_diff_ids,
        index_digest=identity.index_digest,
        image_manifest_digest=identity.image_manifest_digest,
        config_digest=identity.config_digest,
        layer_digests=identity.layer_digests,
        layer_diff_ids=identity.layer_diff_ids,
    )
    import_authenticated_oci_archive(final_archive, final_identity)
    oci_content = _render_oci_layout_manifest(final_identity)
    (publication / OCI_LAYOUT_MANIFEST_NAME).write_bytes(oci_content)
    evidence_content = _render_bootstrap_evidence(
        request=request,
        identity=final_identity,
        oci_manifest_sha256=hashlib.sha256(oci_content).hexdigest(),
        bundle_sha256=bundle_sha256,
        host_content=host_content,
    )
    evidence_path = publication / BOOTSTRAP_EVIDENCE_NAME
    evidence_path.write_bytes(evidence_content)
    if {path.name for path in publication.iterdir()} != {
        BOOTSTRAP_EVIDENCE_NAME,
        OCI_LAYOUT_MANIFEST_NAME,
        OCI_ARCHIVE_NAME,
    }:
        raise SealError("calibration bootstrap publication contains unexpected files")
    return PublishedArtifact(
        role="calibration_bootstrap_evidence",
        path=evidence_path,
        sha256=hashlib.sha256(evidence_content).hexdigest(),
    )


def _download_verified_buildx(recipe: ImageBuildRecipe, directory: Path) -> Path:
    directory.mkdir()
    output = directory / "buildx"
    try:
        with urllib.request.urlopen(recipe.buildx_binary_url, timeout=120) as response:
            content = response.read()
    except OSError as error:
        raise SealError("Buildx binary download failed") from error
    if (
        len(content) != recipe.buildx_binary_size
        or hashlib.sha256(content).hexdigest() != recipe.buildx_binary_sha256
    ):
        raise SealError("Buildx binary byte length or hash differs")
    output.write_bytes(content)
    output.chmod(stat.S_IRUSR | stat.S_IXUSR)
    return output


def _run_builder(
    *,
    buildx: Path,
    builder: str,
    state: Path,
    context: Path,
    output: Path,
    recipe: ImageBuildRecipe,
) -> None:
    environment = {"BUILDX_CONFIG": os.fspath(state)}
    create = (
        os.fspath(buildx),
        "create",
        "--name",
        builder,
        "--driver",
        "docker-container",
        "--driver-opt",
        f"image={recipe.buildkit_image}",
        "--bootstrap",
    )
    try:
        _run_checked(create, "Buildx builder creation failed", environment=environment)
        inspect = _run_checked(
            (os.fspath(buildx), "inspect", "--builder", builder),
            "BuildKit version inspection failed",
            environment=environment,
        )
        if recipe.buildkit_version not in inspect.stdout.split():
            raise SealError("observed BuildKit version differs from bootstrap request")
        _run_checked(
            _buildx_build_command(buildx, builder, output),
            "deterministic OCI image build failed",
            cwd=context,
            environment=environment,
        )
    finally:
        _run_checked(
            (os.fspath(buildx), "rm", "--force", builder),
            "Buildx builder cleanup failed",
            environment=environment,
            tolerate_failure=True,
        )


def _observe_buildx_version(buildx: Path, recipe: ImageBuildRecipe) -> None:
    result = _run_checked((os.fspath(buildx), "version"), "Buildx version inspection failed")
    if recipe.buildx_version not in result.stdout.split():
        raise SealError("observed Buildx version differs from bootstrap request")


def _require_frontend_source(repository: Path, recipe: ImageBuildRecipe) -> None:
    dockerfile = _read_file(repository / "runtime/oaf_tf1/Dockerfile", "OaF Dockerfile")
    try:
        lines = dockerfile.decode("utf-8", errors="strict").splitlines()
    except UnicodeError:
        raise SealError("OaF Dockerfile is not UTF-8") from None
    directive = f"# syntax={recipe.dockerfile_frontend}"
    if (
        not lines
        or lines[0] != directive
        or sum(line.startswith("# syntax=") for line in lines) != 1
    ):
        raise SealError("observed Dockerfile frontend differs from bootstrap request")


def _fetch_base_image(expected_digest: object) -> BaseImageIdentity:
    if not isinstance(expected_digest, str) or not expected_digest.startswith("sha256:"):
        raise SealError("base image manifest reference is mutable")
    token_url = "https://auth.docker.io/token?" + urllib.parse.urlencode(
        {"service": "registry.docker.io", "scope": "repository:library/python:pull"}
    )
    try:
        with urllib.request.urlopen(token_url, timeout=30) as response:
            token_payload = json.loads(response.read())
        token = token_payload["token"]
        headers = {
            "Accept": (
                "application/vnd.oci.image.manifest.v1+json,"
                "application/vnd.docker.distribution.manifest.v2+json"
            ),
            "Authorization": f"Bearer {token}",
        }
        manifest_url = "https://registry-1.docker.io/v2/library/python/manifests/" + expected_digest
        with urllib.request.urlopen(
            urllib.request.Request(manifest_url, headers=headers),
            timeout=60,
        ) as response:
            manifest_bytes = response.read()
        manifest = strict_json_loads(manifest_bytes)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
            raise SealError("base image manifest config descriptor is invalid")
        config_digest = manifest["config"].get("digest")
        if not isinstance(config_digest, str) or not config_digest.startswith("sha256:"):
            raise SealError("base image config descriptor is invalid")
        config_url = "https://registry-1.docker.io/v2/library/python/blobs/" + config_digest
        with urllib.request.urlopen(
            urllib.request.Request(config_url, headers=headers),
            timeout=60,
        ) as response:
            config_bytes = response.read()
    except (KeyError, OSError, ValueError) as error:
        if isinstance(error, SealError):
            raise
        raise SealError("base image registry authentication failed") from error
    return authenticate_base_image(
        manifest_bytes=manifest_bytes,
        config_bytes=config_bytes,
        expected_manifest_digest=expected_digest,
    )


def _render_oci_layout_manifest(identity: OciLayoutIdentity) -> bytes:
    return canonical_json_bytes(
        {
            "archive": _checkpoint_payload(identity.archive),
            "base_image_config_digest": identity.base_image_config_digest,
            "base_image_layer_diff_ids": list(identity.base_image_layer_diff_ids),
            "base_image_layer_digests": list(identity.base_image_layer_digests),
            "config_digest": identity.config_digest,
            "image_manifest_digest": identity.image_manifest_digest,
            "index_digest": identity.index_digest,
            "layer_diff_ids": list(identity.layer_diff_ids),
            "layer_digests": list(identity.layer_digests),
            "schema": OCI_LAYOUT_MANIFEST_SCHEMA,
        },
        trailing_newline=True,
    )


def _render_bootstrap_evidence(
    *,
    request: Any,
    identity: OciLayoutIdentity,
    oci_manifest_sha256: str,
    bundle_sha256: str,
    host_content: bytes,
) -> bytes:
    host = strict_json_loads(host_content[:-1], require_canonical=True)
    payload: JsonValue = {
        "base_image_config_digest": identity.base_image_config_digest,
        "base_image_layer_diff_ids": list(identity.base_image_layer_diff_ids),
        "base_image_layer_digests": list(identity.base_image_layer_digests),
        "build_context_manifest_sha256": request.build_context_manifest_sha256,
        "calibration_bootstrap_request_sha256": request.sha256,
        "image_build": cast(JsonValue, request.payload["image_build"]),
        "native_host_attestation_bundle_sha256": bundle_sha256,
        "native_host_evidence": cast(JsonValue, host),
        "oci_layout_archive": _checkpoint_payload(identity.archive),
        "oci_layout_manifest_sha256": oci_manifest_sha256,
        "runtime_image_config_digest": identity.config_digest,
        "runtime_image_index_digest": identity.index_digest,
        "runtime_image_layer_diff_ids": list(identity.layer_diff_ids),
        "runtime_image_layer_digests": list(identity.layer_digests),
        "runtime_image_manifest_digest": identity.image_manifest_digest,
        "schema": BOOTSTRAP_EVIDENCE_SCHEMA,
    }
    return canonical_json_bytes(payload, trailing_newline=True)


def _checkpoint_payload(identity: Any) -> dict[str, JsonValue]:
    return {"name": identity.name, "sha256": identity.sha256, "size": identity.size}


def _publish_directory_immutable(source: Path, target: Path) -> None:
    """Publish three staged files without replacing or deleting existing bytes."""

    names = (BOOTSTRAP_EVIDENCE_NAME, OCI_LAYOUT_MANIFEST_NAME, OCI_ARCHIVE_NAME)
    if _directory_matches(source, target, names):
        return
    created = False
    linked: list[str] = []
    try:
        target.mkdir()
        created = True
        for name in names:
            os.link(source / name, target / name, follow_symlinks=False)
            linked.append(name)
        _fsync_directory(target)
        _fsync_directory(target.parent)
        if not _directory_matches(source, target, names):
            raise SealError("published calibration bootstrap bytes differ")
    except FileExistsError:
        if _directory_matches(source, target, names):
            return
        raise SealError("calibration bootstrap output already differs") from None
    except Exception:
        if created:
            for name in linked:
                try:
                    (target / name).unlink()
                except OSError:
                    pass
            try:
                target.rmdir()
            except OSError:
                pass
        raise


def _directory_matches(source: Path, target: Path, names: tuple[str, ...]) -> bool:
    try:
        metadata = target.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            return False
        entries = tuple(
            sorted((entry.name for entry in os.scandir(target)), key=lambda name: name.encode())
        )
        if entries != tuple(sorted(names, key=lambda name: name.encode())):
            return False
        return all(
            _read_file(source / name, "staged bootstrap file")
            == _read_file(target / name, "published bootstrap file")
            for name in names
        )
    except (OSError, SealError):
        return False


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        os.fspath(path),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_native_linux_amd64() -> None:
    if platform.system().lower() != "linux" or platform.machine().lower() not in {
        "amd64",
        "x86_64",
    }:
        raise SealError("calibration bootstrap requires a native linux/amd64 host")


def _run_checked(
    command: tuple[str, ...],
    label: str,
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    tolerate_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        if tolerate_failure:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
        raise SealError(label) from error
    if result.returncode != 0 and not tolerate_failure:
        raise SealError(label)
    return result


def _read_file(path: Path, label: str) -> bytes:
    try:
        return read_regular_file_no_follow(path)
    except OSError as error:
        raise SealError(f"{label} is missing or unsafe") from error
