#!/usr/bin/env python3
"""Authenticate and canonically preserve deterministic OaF OCI layouts."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import stat
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from src.benchmark.backend_identity import JsonValue, StrictJsonError, strict_json_loads
from src.benchmark.backend_publication import read_regular_file_no_follow
from src.benchmark.checkpoint_acquisition import CheckpointIdentity

OCI_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG_MEDIA_TYPE = "application/vnd.oci.image.config.v1+json"
OCI_LAYER_MEDIA_TYPE = "application/vnd.oci.image.layer.v1.tar+gzip"
OCI_INDEX_MEDIA_TYPE = "application/vnd.oci.image.index.v1+json"
DOCKER_MANIFEST_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
DOCKER_CONFIG_MEDIA_TYPE = "application/vnd.docker.container.image.v1+json"
DOCKER_LAYER_MEDIA_TYPE = "application/vnd.docker.image.rootfs.diff.tar.gzip"
EPOCH_TIMESTAMP = "1970-01-01T00:00:00Z"

_DESCRIPTOR_KEYS = frozenset({"digest", "mediaType", "size"})
_INDEX_DESCRIPTOR_KEYS = frozenset({"annotations", "digest", "mediaType", "platform", "size"})
_MANIFEST_KEYS = frozenset({"config", "layers", "mediaType", "schemaVersion"})
_INDEX_KEYS = frozenset({"manifests", "mediaType", "schemaVersion"})
_CONFIG_ROOTFS_KEYS = frozenset({"diff_ids", "type"})
_OCI_LAYOUT_KEYS = frozenset({"imageLayoutVersion"})


class OciArchiveError(ValueError):
    """The OCI layout or its canonical preservation archive is invalid."""


@dataclass(frozen=True)
class OciArchiveRecipe:
    compression: str
    final_zero_blocks: int
    format: str
    gid: int
    gname: str
    member_mode: int
    member_types: str
    mtime: int
    path_order: str
    uid: int
    uname: str


@dataclass(frozen=True)
class ImageBuildRecipe:
    annotations: tuple[str, ...]
    buildkit_image: str
    buildkit_version: str
    buildx_binary_sha256: str
    buildx_binary_size: int
    buildx_binary_url: str
    buildx_version: str
    compression: str
    compression_level: int
    dockerfile_frontend: str
    dockerfile_frontend_version: str
    exporter: str
    exporter_tar: bool
    force_compression: bool
    inline_cache: bool
    multi_platform_deterministic: bool
    oci_archive: OciArchiveRecipe
    oci_media_types: bool
    platform: str
    provenance: bool
    rewrite_timestamp: bool
    sbom: bool
    source_date_epoch: int


@dataclass(frozen=True)
class BaseImageIdentity:
    manifest_digest: str
    config_digest: str
    layer_digests: tuple[str, ...]
    layer_diff_ids: tuple[str, ...]


@dataclass(frozen=True)
class OciLayoutIdentity:
    archive: CheckpointIdentity
    base_image_config_digest: str
    base_image_layer_digests: tuple[str, ...]
    base_image_layer_diff_ids: tuple[str, ...]
    index_digest: str
    image_manifest_digest: str
    config_digest: str
    layer_digests: tuple[str, ...]
    layer_diff_ids: tuple[str, ...]


def authenticate_base_image(
    *,
    manifest_bytes: bytes,
    config_bytes: bytes,
    expected_manifest_digest: str,
) -> BaseImageIdentity:
    """Authenticate raw registry manifest/config bytes without reserializing either."""

    _require_digest(expected_manifest_digest, "base image manifest digest")
    actual_manifest_digest = _digest(manifest_bytes)
    if actual_manifest_digest != expected_manifest_digest:
        raise OciArchiveError("base image manifest digest does not match raw bytes")
    manifest = _strict_object(manifest_bytes, "base image manifest")
    _validate_manifest_header(
        manifest,
        allowed_media_types={OCI_MANIFEST_MEDIA_TYPE, DOCKER_MANIFEST_MEDIA_TYPE},
        label="base image manifest",
    )
    config_descriptor = _parse_descriptor(
        manifest["config"],
        label="base image config descriptor",
        allowed_media_types={OCI_CONFIG_MEDIA_TYPE, DOCKER_CONFIG_MEDIA_TYPE},
    )
    if config_descriptor["digest"] != _digest(config_bytes) or config_descriptor["size"] != len(
        config_bytes
    ):
        raise OciArchiveError("base image config descriptor does not match raw bytes")
    layer_descriptors = _parse_layer_descriptors(
        manifest["layers"],
        allowed_media_types={OCI_LAYER_MEDIA_TYPE, DOCKER_LAYER_MEDIA_TYPE},
        label="base image",
    )
    config = _strict_object(config_bytes, "base image config")
    diff_ids = _parse_config_diff_ids(config, "base image config")
    if config.get("architecture") != "amd64" or config.get("os") != "linux":
        raise OciArchiveError("base image config platform is not linux/amd64")
    if len(layer_descriptors) != len(diff_ids):
        raise OciArchiveError("base image layer digest and DiffID arrays are inconsistent")
    return BaseImageIdentity(
        manifest_digest=actual_manifest_digest,
        config_digest=cast(str, config_descriptor["digest"]),
        layer_digests=tuple(cast(str, descriptor["digest"]) for descriptor in layer_descriptors),
        layer_diff_ids=diff_ids,
    )


def canonical_pack_oci_layout(
    directory: Path,
    output: Path,
    recipe: OciArchiveRecipe,
) -> CheckpointIdentity:
    """Write the exact uncompressed POSIX-ustar serialization of one layout."""

    _validate_archive_recipe(recipe)
    root = _require_directory(Path(directory), "OCI layout directory")
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise OciArchiveError("canonical OCI archive output must be absent")
    try:
        target.relative_to(root)
    except ValueError:
        pass
    else:
        raise OciArchiveError("canonical OCI archive output must be outside the layout")
    content = _canonical_archive_bytes(root, recipe)
    _write_exclusive(target, content)
    return CheckpointIdentity(
        name=target.name,
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )


def inspect_oci_layout(
    directory: Path,
    archive: Path,
    recipe: ImageBuildRecipe,
    base_image: BaseImageIdentity,
) -> OciLayoutIdentity:
    """Strict-inspect raw OCI documents, blobs, DiffIDs, base prefix, and archive."""

    _validate_image_recipe_shape(recipe)
    root = _require_directory(Path(directory), "OCI layout directory")
    actual_files = _enumerate_layout_files(root)
    for required in ("index.json", "oci-layout"):
        if required not in actual_files:
            raise OciArchiveError(f"OCI layout is missing {required}")
    layout_bytes = _read_layout_file(root, "oci-layout")
    layout = _strict_object(layout_bytes, "OCI layout marker")
    if set(layout) != _OCI_LAYOUT_KEYS or layout["imageLayoutVersion"] != "1.0.0":
        raise OciArchiveError("OCI layout marker is invalid")
    index_bytes = _read_layout_file(root, "index.json")
    index = _strict_object(index_bytes, "OCI index")
    if (
        set(index) != _INDEX_KEYS
        or index["schemaVersion"] != 2
        or type(index["schemaVersion"]) is not int
        or index["mediaType"] != OCI_INDEX_MEDIA_TYPE
        or not isinstance(index["manifests"], list)
    ):
        raise OciArchiveError("OCI index fields are invalid")
    selected = _select_platform_descriptor(cast(list[JsonValue], index["manifests"]))
    manifest_bytes = _read_descriptor_blob(root, selected, "OCI image manifest")
    manifest = _strict_object(manifest_bytes, "OCI image manifest")
    _validate_manifest_header(
        manifest,
        allowed_media_types={OCI_MANIFEST_MEDIA_TYPE},
        label="OCI image manifest",
    )
    config_descriptor = _parse_descriptor(
        manifest["config"],
        label="OCI image config descriptor",
        allowed_media_types={OCI_CONFIG_MEDIA_TYPE},
    )
    config_bytes = _read_descriptor_blob(root, config_descriptor, "OCI image config")
    config = _strict_object(config_bytes, "OCI image config")
    if config.get("architecture") != "amd64" or config.get("os") != "linux":
        raise OciArchiveError("OCI image config platform is not linux/amd64")
    if config.get("created") != EPOCH_TIMESTAMP:
        raise OciArchiveError("OCI image config created timestamp is not epoch zero")
    layer_descriptors = _parse_layer_descriptors(
        manifest["layers"],
        allowed_media_types={OCI_LAYER_MEDIA_TYPE},
        label="OCI image",
    )
    diff_ids = _parse_config_diff_ids(config, "OCI image config")
    if len(layer_descriptors) != len(diff_ids):
        raise OciArchiveError("OCI layer digest and DiffID arrays are inconsistent")
    layer_digests = tuple(cast(str, descriptor["digest"]) for descriptor in layer_descriptors)
    prefix_length = len(base_image.layer_digests)
    if (
        prefix_length == 0
        or len(base_image.layer_diff_ids) != prefix_length
        or layer_digests[:prefix_length] != base_image.layer_digests
        or diff_ids[:prefix_length] != base_image.layer_diff_ids
    ):
        raise OciArchiveError("OCI image does not preserve the exact base layer prefix")
    expected_files = {
        "index.json",
        "oci-layout",
        _blob_path(cast(str, selected["digest"])),
        _blob_path(cast(str, config_descriptor["digest"])),
        *(_blob_path(digest) for digest in layer_digests),
    }
    extra_blobs = sorted(
        (path for path in actual_files if path.startswith("blobs/") and path not in expected_files),
        key=lambda item: item.encode("utf-8"),
    )
    if extra_blobs:
        raise OciArchiveError("OCI layout contains an extra blob")
    if actual_files != expected_files:
        raise OciArchiveError("OCI layout contains missing or extra files")
    for index_value, (descriptor, expected_diff_id) in enumerate(
        zip(layer_descriptors, diff_ids, strict=True)
    ):
        compressed = _read_descriptor_blob(root, descriptor, "OCI image layer")
        try:
            uncompressed = gzip.decompress(compressed)
        except (EOFError, OSError):
            raise OciArchiveError("OCI image layer gzip stream is invalid") from None
        if _digest(uncompressed) != expected_diff_id:
            raise OciArchiveError("OCI image layer DiffID does not match uncompressed bytes")
        if index_value >= prefix_length:
            _require_epoch_layer(uncompressed)
    archive_path = Path(archive)
    archive_bytes = _read_file(archive_path, "canonical OCI archive")
    canonical_bytes = _canonical_archive_bytes(root, recipe.oci_archive)
    if archive_bytes != canonical_bytes:
        raise OciArchiveError("OCI archive is not the canonical layout serialization")
    return OciLayoutIdentity(
        archive=CheckpointIdentity(
            name=archive_path.name,
            sha256=hashlib.sha256(archive_bytes).hexdigest(),
            size=len(archive_bytes),
        ),
        base_image_config_digest=base_image.config_digest,
        base_image_layer_digests=base_image.layer_digests,
        base_image_layer_diff_ids=base_image.layer_diff_ids,
        index_digest=_digest(index_bytes),
        image_manifest_digest=_digest(manifest_bytes),
        config_digest=_digest(config_bytes),
        layer_digests=layer_digests,
        layer_diff_ids=diff_ids,
    )


def require_identical_oci_builds(
    first: OciLayoutIdentity,
    second: OciLayoutIdentity,
    first_archive: Path,
    second_archive: Path,
) -> OciLayoutIdentity:
    """Reject the first distinct OCI identity class across two fresh builds."""

    comparisons = (
        ("index", first.index_digest, second.index_digest),
        ("manifest", first.image_manifest_digest, second.image_manifest_digest),
        ("config", first.config_digest, second.config_digest),
        ("layer digest", first.layer_digests, second.layer_digests),
        ("layer DiffID", first.layer_diff_ids, second.layer_diff_ids),
        ("archive identity", first.archive, second.archive),
    )
    for label, first_value, second_value in comparisons:
        if first_value != second_value:
            raise OciArchiveError(f"fresh OCI builds have different {label} values")
    first_bytes = _read_file(Path(first_archive), "first canonical OCI archive")
    second_bytes = _read_file(Path(second_archive), "second canonical OCI archive")
    if first_bytes != second_bytes:
        raise OciArchiveError("fresh OCI builds have different canonical archive bytes")
    return first


def _select_platform_descriptor(values: list[JsonValue]) -> dict[str, JsonValue]:
    parsed: list[dict[str, JsonValue]] = []
    matching: list[dict[str, JsonValue]] = []
    for value in values:
        if not isinstance(value, dict) or set(value) != _INDEX_DESCRIPTOR_KEYS:
            raise OciArchiveError("OCI index descriptor fields are invalid")
        descriptor = _parse_descriptor(
            {key: value[key] for key in _DESCRIPTOR_KEYS},
            label="OCI index descriptor",
            allowed_media_types={OCI_MANIFEST_MEDIA_TYPE},
        )
        platform = value["platform"]
        annotations = value["annotations"]
        if (
            not isinstance(platform, dict)
            or set(platform) != {"architecture", "os"}
            or not isinstance(annotations, dict)
            or annotations != {"org.opencontainers.image.created": EPOCH_TIMESTAMP}
        ):
            raise OciArchiveError("OCI index platform or annotations are invalid")
        combined = {**descriptor, "platform": platform, "annotations": annotations}
        parsed.append(combined)
        if platform == {"architecture": "amd64", "os": "linux"}:
            matching.append(combined)
    if len(matching) != 1 or len(parsed) != 1:
        raise OciArchiveError("OCI index must select exactly one linux/amd64 manifest")
    return matching[0]


def _validate_manifest_header(
    manifest: dict[str, JsonValue],
    *,
    allowed_media_types: set[str],
    label: str,
) -> None:
    if (
        set(manifest) != _MANIFEST_KEYS
        or manifest["schemaVersion"] != 2
        or type(manifest["schemaVersion"]) is not int
        or manifest["mediaType"] not in allowed_media_types
        or not isinstance(manifest["layers"], list)
    ):
        raise OciArchiveError(f"{label} fields are invalid")


def _parse_layer_descriptors(
    value: JsonValue,
    *,
    allowed_media_types: set[str],
    label: str,
) -> tuple[dict[str, JsonValue], ...]:
    if not isinstance(value, list) or not value:
        raise OciArchiveError(f"{label} layer descriptors are invalid")
    return tuple(
        _parse_descriptor(
            descriptor,
            label=f"{label} layer descriptor",
            allowed_media_types=allowed_media_types,
        )
        for descriptor in value
    )


def _parse_descriptor(
    value: JsonValue,
    *,
    label: str,
    allowed_media_types: set[str],
) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or set(value) != _DESCRIPTOR_KEYS:
        raise OciArchiveError(f"{label} fields are invalid")
    digest = value["digest"]
    size = value["size"]
    media_type = value["mediaType"]
    _require_digest(digest, f"{label} digest")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or media_type not in allowed_media_types
    ):
        raise OciArchiveError(f"{label} types are invalid")
    return value


def _parse_config_diff_ids(config: dict[str, JsonValue], label: str) -> tuple[str, ...]:
    rootfs = config.get("rootfs")
    if (
        not isinstance(rootfs, dict)
        or set(rootfs) != _CONFIG_ROOTFS_KEYS
        or rootfs["type"] != "layers"
        or not isinstance(rootfs["diff_ids"], list)
        or not rootfs["diff_ids"]
    ):
        raise OciArchiveError(f"{label} rootfs is invalid")
    diff_ids: list[str] = []
    for value in rootfs["diff_ids"]:
        diff_ids.append(_require_digest(value, f"{label} DiffID"))
    return tuple(diff_ids)


def _read_descriptor_blob(
    root: Path,
    descriptor: dict[str, JsonValue],
    label: str,
) -> bytes:
    digest = cast(str, descriptor["digest"])
    content = _read_layout_file(root, _blob_path(digest))
    if len(content) != descriptor["size"] or _digest(content) != digest:
        raise OciArchiveError(f"{label} descriptor size or digest drift")
    return content


def _require_epoch_layer(content: bytes) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError):
        raise OciArchiveError("generated OCI layer tar is invalid") from None
    if any(member.mtime != 0 for member in members):
        raise OciArchiveError("generated OCI layer member mtime is not epoch zero")


def _validate_image_recipe_shape(recipe: ImageBuildRecipe) -> None:
    expected = ImageBuildRecipe(
        annotations=(),
        buildkit_image=(
            "moby/buildkit@sha256:63db51c9b30208a7c2b1c40392c7ebb9ce2f85ba238a18a85420f8f5ea2d4684"
        ),
        buildkit_version="v0.31.2",
        buildx_binary_sha256=("d41ece72044243b4f58b343441ae37446d9c29a7d6b5e11c61847bbcf8f7dfda"),
        buildx_binary_size=65_265_826,
        buildx_binary_url=(
            "https://github.com/docker/buildx/releases/download/v0.35.0/buildx-v0.35.0.linux-amd64"
        ),
        buildx_version="v0.35.0",
        compression="gzip",
        compression_level=6,
        dockerfile_frontend=(
            "docker/dockerfile-upstream@sha256:"
            "3d6d54b33351b396a910d33248754b86b1d7dd838b4eeb9575d8903a209f6516"
        ),
        dockerfile_frontend_version="1.25.0",
        exporter="oci",
        exporter_tar=False,
        force_compression=False,
        inline_cache=False,
        multi_platform_deterministic=True,
        oci_archive=OciArchiveRecipe(
            compression="none",
            final_zero_blocks=2,
            format="posix-ustar",
            gid=0,
            gname="",
            member_mode=420,
            member_types="regular-files-only",
            mtime=0,
            path_order="utf8-byte",
            uid=0,
            uname="",
        ),
        oci_media_types=True,
        platform="linux/amd64",
        provenance=False,
        rewrite_timestamp=True,
        sbom=False,
        source_date_epoch=0,
    )
    boolean_values = (
        recipe.exporter_tar,
        recipe.force_compression,
        recipe.inline_cache,
        recipe.multi_platform_deterministic,
        recipe.oci_media_types,
        recipe.provenance,
        recipe.rewrite_timestamp,
        recipe.sbom,
    )
    integer_values = (
        recipe.buildx_binary_size,
        recipe.compression_level,
        recipe.source_date_epoch,
    )
    if (
        recipe != expected
        or any(type(value) is not bool for value in boolean_values)
        or any(type(value) is not int for value in integer_values)
    ):
        raise OciArchiveError("OCI image build recipe is invalid")
    _validate_archive_recipe(recipe.oci_archive)


def _validate_archive_recipe(recipe: OciArchiveRecipe) -> None:
    if recipe != OciArchiveRecipe(
        compression="none",
        final_zero_blocks=2,
        format="posix-ustar",
        gid=0,
        gname="",
        member_mode=420,
        member_types="regular-files-only",
        mtime=0,
        path_order="utf8-byte",
        uid=0,
        uname="",
    ):
        raise OciArchiveError("OCI archive recipe is invalid")


def _canonical_archive_bytes(root: Path, recipe: OciArchiveRecipe) -> bytes:
    _validate_archive_recipe(recipe)
    files = _enumerate_layout_files(root)
    output = bytearray()
    for relative in sorted(files, key=lambda item: item.encode("utf-8")):
        content = _read_layout_file(root, relative)
        info = tarfile.TarInfo(relative)
        info.mode = recipe.member_mode
        info.uid = recipe.uid
        info.gid = recipe.gid
        info.uname = recipe.uname
        info.gname = recipe.gname
        info.mtime = recipe.mtime
        info.size = len(content)
        info.type = tarfile.REGTYPE
        try:
            output.extend(
                info.tobuf(
                    format=tarfile.USTAR_FORMAT,
                    encoding="utf-8",
                    errors="strict",
                )
            )
        except (UnicodeError, ValueError):
            raise OciArchiveError("OCI archive member path does not fit POSIX ustar") from None
        output.extend(content)
        output.extend(b"\0" * (-len(content) % tarfile.BLOCKSIZE))
    output.extend(b"\0" * tarfile.BLOCKSIZE * recipe.final_zero_blocks)
    return bytes(output)


def _enumerate_layout_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        metadata = _lstat(path, "OCI layout entry")
        if stat.S_ISLNK(metadata.st_mode):
            raise OciArchiveError("OCI layout contains a link")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise OciArchiveError("OCI layout contains a non-regular entry")
        if metadata.st_nlink != 1:
            raise OciArchiveError("OCI layout contains a hard-linked file")
        relative = path.relative_to(root).as_posix()
        if not _safe_relative_path(relative) or relative in files:
            raise OciArchiveError("OCI layout contains an aliased or duplicate path")
        files.add(relative)
    if not files:
        raise OciArchiveError("OCI layout is empty")
    return files


def _read_layout_file(root: Path, relative: str) -> bytes:
    if not _safe_relative_path(relative):
        raise OciArchiveError("OCI layout path is invalid")
    path = root.joinpath(*PurePosixPath(relative).parts)
    return _read_file(path, "OCI layout file")


def _read_file(path: Path, label: str) -> bytes:
    metadata = _lstat(path, label)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise OciArchiveError(f"{label} is not a stable single-link regular file")
    try:
        return read_regular_file_no_follow(path)
    except OSError as error:
        raise OciArchiveError(f"{label} is unreadable") from error


def _strict_object(content: bytes, label: str) -> dict[str, JsonValue]:
    try:
        value = strict_json_loads(content)
    except StrictJsonError as error:
        raise OciArchiveError(f"{label} JSON is invalid: {error}") from None
    if not isinstance(value, dict):
        raise OciArchiveError(f"{label} JSON must be an object")
    return value


def _require_directory(path: Path, label: str) -> Path:
    metadata = _lstat(path, label)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise OciArchiveError(f"{label} is not a stable directory")
    return path


def _lstat(path: Path, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise OciArchiveError(f"{label} is missing or unreadable") from error


def _write_exclusive(path: Path, content: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
    except OSError as error:
        raise OciArchiveError("canonical OCI archive could not be written") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _blob_path(digest: str) -> str:
    _require_digest(digest, "OCI blob digest")
    return f"blobs/sha256/{digest.removeprefix('sha256:')}"


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise OciArchiveError(f"{label} is invalid or mutable")
    return value


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and str(path) == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )
