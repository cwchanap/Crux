"""Model-driven OaF checkpoint acquisition and immutable cache preparation."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from src.benchmark.artifact_io import read_regular_file_no_follow

if TYPE_CHECKING:
    from runtime.oaf_tf1.model import OafModelConfig


class CheckpointAcquisitionError(ValueError):
    pass


_CHECKPOINT_DIRECTORY_MODE = 0o755
_CHECKPOINT_COMPONENT_MODE = 0o644


def prepare_oaf_checkpoint(
    config: "OafModelConfig",
    cache_root: Path,
    *,
    download: bool,
    archive_path: Path | None = None,
) -> Path:
    """Verify and publish the model components described by ``model.json``."""
    if not isinstance(cache_root, Path):
        raise TypeError("cache_root must be a Path")
    if download and archive_path is not None:
        raise CheckpointAcquisitionError("checkpoint acquisition modes are mutually exclusive")
    checkpoint = config.checkpoint
    components = dict(checkpoint.components)
    if not components:
        raise CheckpointAcquisitionError("checkpoint components are missing")

    target = cache_root / "sha256" / checkpoint.archive_sha256
    if not download and archive_path is None:
        if not _is_directory(target):
            raise CheckpointAcquisitionError("checkpoint cache is missing")
        _verify_cached_components(target, components)
        return target

    try:
        archive = (
            _download_checkpoint_archive(checkpoint.url)
            if download
            else read_regular_file_no_follow(archive_path)
        )  # type: ignore[arg-type]
    except CheckpointAcquisitionError:
        raise
    except (OSError, TypeError):
        raise CheckpointAcquisitionError("checkpoint archive is unavailable") from None
    if hashlib.sha256(archive).hexdigest() != checkpoint.archive_sha256:
        raise CheckpointAcquisitionError("checkpoint archive hash differs")
    extracted = _read_checkpoint_components(archive, components)
    try:
        (cache_root / "sha256").mkdir(parents=True, exist_ok=True)
    except OSError:
        raise CheckpointAcquisitionError("checkpoint cache directory is unavailable") from None
    if _is_directory(target):
        _verify_cached_components(target, components)
        return target
    if target.exists() or target.is_symlink():
        raise CheckpointAcquisitionError("checkpoint cache destination is unsafe")

    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{checkpoint.archive_sha256}.", dir=target.parent))
        for name, content in extracted.items():
            destination = staging / name
            with destination.open("xb") as output:
                os.fchmod(output.fileno(), _CHECKPOINT_COMPONENT_MODE)
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        _normalize_mode(staging, _CHECKPOINT_DIRECTORY_MODE, directory=True)
        _fsync_directory(staging)
        try:
            os.rename(staging, target)
        except FileExistsError:
            _verify_cached_components(target, components)
        else:
            staging = None
            _fsync_directory(target.parent)
        _verify_cached_components(target, components)
        return target
    except CheckpointAcquisitionError:
        raise
    except OSError:
        raise CheckpointAcquisitionError("checkpoint cache publication failed") from None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _download_checkpoint_archive(url: str) -> bytes:
    if not isinstance(url, str) or not url.startswith("https://"):
        raise CheckpointAcquisitionError("checkpoint URL is invalid")

    class _RejectRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
            del req, fp, code, msg, headers, newurl
            raise OSError("checkpoint redirect rejected")

    try:
        with urllib.request.build_opener(_RejectRedirects()).open(url) as response:
            if response.geturl() != url:
                raise OSError("checkpoint URL changed")
            return response.read()
    except (OSError, ValueError):
        raise CheckpointAcquisitionError("checkpoint download failed") from None


def _read_checkpoint_components(archive: bytes, expected: Mapping[str, str]) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            infos = source.infolist()
            names = [info.filename for info in infos]
            expected_names = set(expected) | {"checkpoint"}
            if len(names) != len(set(names)) or set(names) != expected_names:
                raise CheckpointAcquisitionError("checkpoint archive members differ")
            components: dict[str, bytes] = {}
            for info in infos:
                if not _safe_zip_name(info.filename) or info.is_dir():
                    raise CheckpointAcquisitionError("checkpoint archive member is unsafe")
                content = source.read(info)
                if info.filename == "checkpoint":
                    expected_pointer = (
                        b'model_checkpoint_path: "model.ckpt-569400"\n'
                        b'all_model_checkpoint_paths: "model.ckpt-569400"\n'
                    )
                    if content != expected_pointer:
                        raise CheckpointAcquisitionError("checkpoint pointer differs")
                    continue
                if hashlib.sha256(content).hexdigest() != expected[info.filename]:
                    raise CheckpointAcquisitionError("checkpoint component hash differs")
                components[info.filename] = content
            return components
    except CheckpointAcquisitionError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
        raise CheckpointAcquisitionError("checkpoint archive is invalid") from None


def _verify_cached_components(path: Path, expected: Mapping[str, str]) -> None:
    try:
        if set(entry.name for entry in path.iterdir()) != set(expected):
            raise CheckpointAcquisitionError("checkpoint cache entries differ")
        for name, digest in expected.items():
            content = read_regular_file_no_follow(path / name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise CheckpointAcquisitionError("checkpoint component hash differs")
        _normalize_mode(path, _CHECKPOINT_DIRECTORY_MODE, directory=True)
        for name in expected:
            _normalize_mode(path / name, _CHECKPOINT_COMPONENT_MODE, directory=False)
    except CheckpointAcquisitionError:
        raise
    except OSError:
        raise CheckpointAcquisitionError("checkpoint cache component is unavailable") from None


def _is_directory(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        raise CheckpointAcquisitionError("checkpoint cache destination is unavailable") from None
    return stat.S_ISDIR(metadata.st_mode)


def _safe_zip_name(name: str) -> bool:
    return (
        bool(name)
        and name == Path(name).name
        and name not in {".", ".."}
        and all(character not in name for character in ("/", "\\", ":", "\x00"))
    )


def _normalize_mode(path: Path, mode: int, *, directory: bool) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if directory and not stat.S_ISDIR(metadata.st_mode):
            raise OSError("checkpoint cache path is not a directory")
        if not directory and not stat.S_ISREG(metadata.st_mode):
            raise OSError("checkpoint cache component is not a regular file")
        if stat.S_IMODE(metadata.st_mode) != mode:
            os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["CheckpointAcquisitionError", "prepare_oaf_checkpoint"]
