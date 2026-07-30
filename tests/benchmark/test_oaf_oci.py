from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from tools.hpa320.oaf_oci import (
    BaseImageIdentity,
    ImageBuildRecipe,
    OciArchiveError,
    OciArchiveRecipe,
    authenticate_base_image,
    canonical_pack_oci_layout,
    inspect_oci_layout,
    require_identical_oci_builds,
)


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _raw_json(payload: object) -> bytes:
    return json.dumps(payload, indent=1, sort_keys=False).encode("utf-8")


def _layer(member_mtime: int, content: bytes) -> tuple[bytes, bytes]:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo("payload.txt")
        info.mode = 0o644
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = member_mtime
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    tar_bytes = stream.getvalue()
    return gzip.compress(tar_bytes, compresslevel=6, mtime=0), tar_bytes


def _archive_recipe() -> OciArchiveRecipe:
    return OciArchiveRecipe(
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
    )


def _image_recipe() -> ImageBuildRecipe:
    return ImageBuildRecipe(
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
        oci_archive=_archive_recipe(),
        oci_media_types=True,
        platform="linux/amd64",
        provenance=False,
        rewrite_timestamp=True,
        sbom=False,
        source_date_epoch=0,
    )


def _descriptor(content: bytes, media_type: str) -> dict[str, object]:
    return {
        "mediaType": media_type,
        "digest": _digest(content),
        "size": len(content),
    }


def _write_blob(directory: Path, content: bytes) -> None:
    path = directory / "blobs/sha256" / hashlib.sha256(content).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_oci_layout(
    tmp_path: Path,
    *,
    generated_mtime: int = 0,
    extra_blob: bool = False,
) -> tuple[Path, Path, ImageBuildRecipe, BaseImageIdentity]:
    directory = tmp_path / "layout"
    directory.mkdir(parents=True)
    base_layer, base_tar = _layer(1_234_567_890, b"base")
    generated_layer, generated_tar = _layer(generated_mtime, b"generated")
    base_config = _raw_json(
        {
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {"type": "layers", "diff_ids": [_digest(base_tar)]},
        }
    )
    base_manifest = _raw_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": _descriptor(base_config, "application/vnd.oci.image.config.v1+json"),
            "layers": [_descriptor(base_layer, "application/vnd.oci.image.layer.v1.tar+gzip")],
        }
    )
    base = authenticate_base_image(
        manifest_bytes=base_manifest,
        config_bytes=base_config,
        expected_manifest_digest=_digest(base_manifest),
    )
    config = _raw_json(
        {
            "created": "1970-01-01T00:00:00Z",
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {
                "type": "layers",
                "diff_ids": [_digest(base_tar), _digest(generated_tar)],
            },
            "history": [{"created": "1970-01-01T00:00:00Z", "created_by": "generated"}],
        }
    )
    manifest = _raw_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": _descriptor(config, "application/vnd.oci.image.config.v1+json"),
            "layers": [
                _descriptor(base_layer, "application/vnd.oci.image.layer.v1.tar+gzip"),
                _descriptor(generated_layer, "application/vnd.oci.image.layer.v1.tar+gzip"),
            ],
        }
    )
    index = _raw_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    **_descriptor(
                        manifest,
                        "application/vnd.oci.image.manifest.v1+json",
                    ),
                    "annotations": {"org.opencontainers.image.created": "1970-01-01T00:00:00Z"},
                    "platform": {"architecture": "amd64", "os": "linux"},
                }
            ],
        }
    )
    (directory / "oci-layout").write_bytes(_raw_json({"imageLayoutVersion": "1.0.0"}))
    (directory / "index.json").write_bytes(index)
    for content in (manifest, config, base_layer, generated_layer):
        _write_blob(directory, content)
    if extra_blob:
        _write_blob(directory, b"extra")
    archive = tmp_path / "layout.tar"
    canonical_pack_oci_layout(directory, archive, _archive_recipe())
    return directory, archive, _image_recipe(), base


def test_oci_base_authentication_hashes_raw_noncanonical_json_bytes() -> None:
    layer, layer_tar = _layer(99, b"base")
    config = _raw_json(
        {
            "architecture": "amd64",
            "os": "linux",
            "rootfs": {"type": "layers", "diff_ids": [_digest(layer_tar)]},
        }
    )
    manifest = _raw_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": _descriptor(config, "application/vnd.oci.image.config.v1+json"),
            "layers": [_descriptor(layer, "application/vnd.oci.image.layer.v1.tar+gzip")],
        }
    )

    identity = authenticate_base_image(
        manifest_bytes=manifest,
        config_bytes=config,
        expected_manifest_digest=_digest(manifest),
    )

    assert identity.manifest_digest == _digest(manifest)
    assert identity.config_digest == _digest(config)
    assert identity.layer_digests == (_digest(layer),)
    assert identity.layer_diff_ids == (_digest(layer_tar),)

    duplicate_key = manifest.replace(
        b'"schemaVersion": 2,', b'"schemaVersion": 2,"schemaVersion":2,'
    )
    with pytest.raises(OciArchiveError, match="duplicate|JSON"):
        authenticate_base_image(
            manifest_bytes=duplicate_key,
            config_bytes=config,
            expected_manifest_digest=_digest(duplicate_key),
        )


def test_oci_inspector_accepts_non_epoch_exact_base_prefix_and_raw_json(tmp_path: Path) -> None:
    directory, archive, recipe, base = _make_oci_layout(tmp_path)

    identity = inspect_oci_layout(directory, archive, recipe, base)

    assert identity.base_image_layer_digests == base.layer_digests
    assert identity.base_image_layer_diff_ids == base.layer_diff_ids
    assert identity.layer_digests[:1] == base.layer_digests
    assert identity.layer_diff_ids[:1] == base.layer_diff_ids
    assert identity.archive.sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert identity.index_digest == _digest((directory / "index.json").read_bytes())


def test_oci_inspector_rejects_non_epoch_generated_layer(tmp_path: Path) -> None:
    directory, archive, recipe, base = _make_oci_layout(tmp_path, generated_mtime=123)

    with pytest.raises(OciArchiveError, match="epoch"):
        inspect_oci_layout(directory, archive, recipe, base)


def test_oci_inspector_rejects_base_prefix_mismatch(tmp_path: Path) -> None:
    directory, archive, recipe, base = _make_oci_layout(tmp_path)
    mismatched = replace(base, layer_digests=("sha256:" + "0" * 64,))

    with pytest.raises(OciArchiveError, match="base.*prefix"):
        inspect_oci_layout(directory, archive, recipe, mismatched)


def test_oci_inspector_rejects_image_recipe_drift(tmp_path: Path) -> None:
    directory, archive, recipe, base = _make_oci_layout(tmp_path)

    with pytest.raises(OciArchiveError, match="image build recipe"):
        inspect_oci_layout(
            directory,
            archive,
            replace(recipe, force_compression=True),
            base,
        )


def test_oci_inspector_rejects_extra_blob_and_noncanonical_archive(tmp_path: Path) -> None:
    directory, archive, recipe, base = _make_oci_layout(tmp_path, extra_blob=True)
    with pytest.raises(OciArchiveError, match="extra blob"):
        inspect_oci_layout(directory, archive, recipe, base)

    directory, archive, recipe, base = _make_oci_layout(tmp_path / "second")
    archive.write_bytes(archive.read_bytes() + b"\0" * 512)
    with pytest.raises(OciArchiveError, match="canonical"):
        inspect_oci_layout(directory, archive, recipe, base)


def test_oci_canonical_packer_is_mode_time_and_creation_order_independent(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory, paths in (
        (first, (("z.txt", b"z"), ("a/n.txt", b"a"))),
        (second, (("a/n.txt", b"a"), ("z.txt", b"z"))),
    ):
        for relative, content in paths:
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(0o600 if directory == first else 0o755)
            os.utime(path, ns=(123, 456))

    first_archive = tmp_path / "first.tar"
    second_archive = tmp_path / "second.tar"
    canonical_pack_oci_layout(first, first_archive, _archive_recipe())
    canonical_pack_oci_layout(second, second_archive, _archive_recipe())

    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first_archive.read_bytes().endswith(b"\0" * 1024)
    assert not first_archive.read_bytes().endswith(b"\0" * 1536)
    with tarfile.open(first_archive, mode="r:") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == ["a/n.txt", "z.txt"]
    assert all(member.mtime == 0 and member.mode == 0o644 for member in members)
    assert all(member.uid == member.gid == 0 for member in members)
    assert all(member.uname == member.gname == "" for member in members)


def test_oci_canonical_packer_rejects_links_and_path_overflow(tmp_path: Path) -> None:
    directory = tmp_path / "layout"
    directory.mkdir()
    target = directory / "target"
    target.write_bytes(b"target")
    (directory / "link").symlink_to(target)

    with pytest.raises(OciArchiveError, match="link|regular"):
        canonical_pack_oci_layout(directory, tmp_path / "linked.tar", _archive_recipe())

    (directory / "link").unlink()
    too_long = directory / ("a" * 101)
    too_long.write_bytes(b"long")
    with pytest.raises(OciArchiveError, match="ustar|path"):
        canonical_pack_oci_layout(directory, tmp_path / "long.tar", _archive_recipe())


def test_oci_double_build_rejects_first_identity_class_drift(tmp_path: Path) -> None:
    _directory, archive, _recipe, _base = _make_oci_layout(tmp_path)
    identity = inspect_oci_layout(*_make_oci_layout(tmp_path / "second"))

    require_identical_oci_builds(identity, identity, archive, archive)

    with pytest.raises(OciArchiveError, match="config"):
        require_identical_oci_builds(
            identity,
            replace(identity, config_digest="sha256:" + "0" * 64),
            archive,
            archive,
        )
