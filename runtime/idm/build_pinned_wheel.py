#!/usr/bin/env python3
"""Build and attest a local wheel for the pinned upstream IDM checkout.

The upstream repository intentionally remains untouched.  A temporary copy of files read from
the pinned commit receives only the pinned Poetry PEP 517 build-system declaration; the resulting
wheel is then checked against the source commit before it is copied into this runtime directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

UPSTREAM_REPOSITORY = "https://github.com/bernardo-torres/inverse-drum-machine"
UPSTREAM_COMMIT = "456656868538205ef756912c7cf5b0fd936de8af"
POETRY_CORE_REQUIREMENT = "poetry-core==2.2.1"
EXPECTED_PACKAGE = "inverse-drum-machine"
EXPECTED_VERSION = "0.1.0"
SHIM_BUILD_SYSTEM = (
    "[build-system]\n"
    f'requires = ["{POETRY_CORE_REQUIREMENT}"]\n'
    'build-backend = "poetry.core.masonry.api"\n'
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _commit_files(source: Path, commit: str) -> tuple[PurePosixPath, ...]:
    result = subprocess.run(
        ["git", "-C", str(source), "ls-tree", "-r", "--name-only", commit],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return tuple(PurePosixPath(path) for path in result.stdout.splitlines() if path)


def _commit_blob(source: Path, commit: str, relative: PurePosixPath) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(source), "show", f"{commit}:{relative.as_posix()}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _copy_commit_tree(source: Path, commit: str, destination: Path) -> tuple[PurePosixPath, ...]:
    tracked = _commit_files(source, commit)
    for relative in tracked:
        destination_path = destination.joinpath(*relative.parts)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(_commit_blob(source, commit, relative))
    return tracked


def _source_manifest(
    source: Path, commit: str, tracked: tuple[PurePosixPath, ...]
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for relative in tracked:
        content = _commit_blob(source, commit, relative)
        manifest.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_bytes(content),
                "byte_length": len(content),
            }
        )
    return manifest


def _verify_wheel_sources(
    wheel_path: Path,
    source: Path,
    commit: str,
    tracked: tuple[PurePosixPath, ...],
) -> list[dict[str, Any]]:
    expected = {relative for relative in tracked if relative.parts[0] == "idm"}
    with zipfile.ZipFile(wheel_path) as wheel:
        names = {PurePosixPath(name) for name in wheel.namelist()}
        packaged = {name for name in names if name.parts and name.parts[0] == "idm"}
        if packaged != expected:
            missing = sorted(path.as_posix() for path in expected - packaged)
            extra = sorted(path.as_posix() for path in packaged - expected)
            raise RuntimeError(f"wheel idm file set mismatch: missing={missing}, extra={extra}")

        verified: list[dict[str, Any]] = []
        for relative in sorted(expected):
            source_bytes = _commit_blob(source, commit, relative)
            wheel_bytes = wheel.read(relative.as_posix())
            if wheel_bytes != source_bytes:
                raise RuntimeError(f"wheel source bytes differ from pinned checkout: {relative}")
            verified.append(
                {
                    "path": relative.as_posix(),
                    "sha256": _sha256_bytes(source_bytes),
                    "byte_length": len(source_bytes),
                }
            )
    return verified


def _build_wheel(source: Path, commit: str, wheel_output: Path) -> tuple[Path, str]:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to build the pinned wheel")

    with tempfile.TemporaryDirectory(prefix="crux-idm-wheel-") as temporary:
        temporary_path = Path(temporary)
        build_source = temporary_path / "source"
        build_output = temporary_path / "dist"
        build_source.mkdir()
        build_output.mkdir()
        _copy_commit_tree(source, commit, build_source)

        original_pyproject = (build_source / "pyproject.toml").read_bytes()
        if b"[build-system]" in original_pyproject:
            raise RuntimeError("pinned upstream pyproject unexpectedly already has build-system")
        (build_source / "pyproject.toml").write_bytes(
            original_pyproject.rstrip() + b"\n\n" + SHIM_BUILD_SYSTEM.encode("utf-8")
        )

        environment = os.environ.copy()
        environment["SOURCE_DATE_EPOCH"] = "0"
        environment["PYTHONHASHSEED"] = "0"
        subprocess.run(
            [
                uv,
                "build",
                "--python",
                "3.11",
                "--wheel",
                "--out-dir",
                str(build_output),
                str(build_source),
            ],
            check=True,
            env=environment,
        )
        wheels = sorted(build_output.glob("*.whl"))
        expected_name = "inverse_drum_machine-0.1.0-py3-none-any.whl"
        if [path.name for path in wheels] != [expected_name]:
            raise RuntimeError(f"unexpected wheel output: {[path.name for path in wheels]}")
        wheel_path = wheels[0]
        wheel_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wheel_path, wheel_output)
        return wheel_output, _sha256_file(wheel_output)


def _uv_version() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to build the pinned wheel")
    return subprocess.check_output([uv, "--version"], text=True).strip()


def build(source: Path, wheel_output: Path, provenance_output: Path) -> None:
    source = source.resolve()
    wheel_output = wheel_output.resolve()
    provenance_output = provenance_output.resolve()

    actual_commit = _git(source, "rev-parse", "HEAD")
    if actual_commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"source commit mismatch: expected {UPSTREAM_COMMIT}, got {actual_commit}"
        )

    original_pyproject = _commit_blob(source, actual_commit, PurePosixPath("pyproject.toml"))
    if b"[build-system]" in original_pyproject:
        raise RuntimeError("pinned upstream pyproject unexpectedly already has build-system")
    tracked = _commit_files(source, actual_commit)
    source_manifest = _source_manifest(source, actual_commit, tracked)

    built_wheel, wheel_sha256 = _build_wheel(source, actual_commit, wheel_output)
    packaged_idm_files = _verify_wheel_sources(built_wheel, source, actual_commit, tracked)

    provenance = {
        "schema": "crux.idm-wheel-provenance/v1",
        "repository": UPSTREAM_REPOSITORY,
        "source_commit": actual_commit,
        "package_name": EXPECTED_PACKAGE,
        "package_version": EXPECTED_VERSION,
        "original_pyproject_sha256": _sha256_bytes(original_pyproject),
        "original_pyproject_byte_length": len(original_pyproject),
        "shim": {
            "build_system": {
                "requires": [POETRY_CORE_REQUIREMENT],
                "build_backend": "poetry.core.masonry.api",
            },
            "metadata_sha256": _sha256_bytes(
                original_pyproject.rstrip() + b"\n\n" + SHIM_BUILD_SYSTEM.encode("utf-8")
            ),
        },
        "builder": {
            "uv": _uv_version(),
            "python": "3.11",
            "source_date_epoch": 0,
        },
        "wheel": {
            "path": wheel_output.name,
            "sha256": wheel_sha256,
            "byte_length": built_wheel.stat().st_size,
            "tag": "py3-none-any",
        },
        "source_manifest": source_manifest,
        "packaged_idm_files": packaged_idm_files,
        "license_basis": {
            "repository_code": "Apache-2.0",
            "checkpoint_provenance": "repository-level Apache-2.0",
            "checkpoint_notice_found": False,
            "note": (
                "No separate checkpoint notice exists in the pinned repository; this records "
                "repository-level provenance and is not an independent legal conclusion."
            ),
        },
    }
    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    provenance_output.write_text(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="pinned upstream checkout")
    parser.add_argument("--wheel-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.source, arguments.wheel_output, arguments.provenance_output)


if __name__ == "__main__":
    main()
