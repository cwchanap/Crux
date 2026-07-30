#!/usr/bin/env python3
"""Resolve and verify the exact CPython 3.7 Linux/amd64 OaF wheel closure."""

# This single-purpose verifier keeps the full identity schema and explicit fail-closed
# validation branches together so the acquisition contract remains directly auditable.
# pylint: disable=duplicate-code,too-many-arguments,too-many-boolean-expressions
# pylint: disable=too-many-branches,too-many-instance-attributes,too-many-lines
# pylint: disable=too-many-locals,too-many-statements,unidiomatic-typecheck
from __future__ import annotations

import argparse
import base64
import csv
import email.parser
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.backend_publication import atomic_replace_bytes

TARGET_PYTHON_VERSION = "3.7"
TARGET_PYTHON_FULL_VERSION = "3.7.17"
TARGET_IMPLEMENTATION = "cp"
TARGET_ABI = "cp37m"
TARGET_PLATFORM = "manylinux2010_x86_64"
TENSORFLOW_FILENAME = "tensorflow-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl"
TENSORFLOW_SHA256 = "29831dda98d668067de75403b2fca0d06a2f026ef6f217fa2ca873c20b4ee4d3"
DISTRIBUTION_BUILD_SCHEMA = "crux.distribution-build-manifest/v2"
APPROVED_SDIST_FILENAME = "pretty_midi-0.2.10.tar.gz"
APPROVED_SDIST_URL = (
    "https://files.pythonhosted.org/packages/c2/81/"
    "f3e47efdf65eab347d1b78e8279222f6e3122d722cc9d68d443503a3c0e7/"
    "pretty_midi-0.2.10.tar.gz"
)
APPROVED_SDIST_BYTE_LENGTH = 5_592_108
APPROVED_SDIST_SHA256 = "ea6e192f94044674e833336ea1f415318ddf28e320302ac7b109edff0d4534bd"
BUILT_PRETTY_MIDI_FILENAME = "pretty_midi-0.2.10-py3-none-any.whl"
SOURCE_DATE_EPOCH = 1_677_868_454
GAST_SDIST_FILENAME = "gast-0.2.2.tar.gz"
GAST_SDIST_URL = (
    "https://files.pythonhosted.org/packages/4e/35/"
    "11749bf99b2d4e3cceb4d55ca22590b0d7c2c62b9de38ac4a4a7f4687421/"
    "gast-0.2.2.tar.gz"
)
GAST_SDIST_BYTE_LENGTH = 10_294
GAST_SDIST_SHA256 = "fe939df4583692f0512161ec1c880e0a10e71e6a232da045ab8edd3756fbadf0"
GAST_SOURCE_DATE_EPOCH = 1_547_387_031
BUILT_GAST_FILENAME = "gast-0.2.2-py3-none-any.whl"
BASE_IMAGE = "python@sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673"
CONTAINER_PLATFORM = "linux/amd64"
BUILD_ENVIRONMENT = {
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PIP_NO_INDEX": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONPATH": (
        "/inputs/build/setuptools-67.8.0-py3-none-any.whl:"
        "/inputs/build/wheel-0.41.3-py3-none-any.whl"
    ),
    "TZ": "UTC",
}
BUILD_COMMAND = (
    "/bin/sh",
    "-ec",
    "python -c 'import os,tarfile; "
    'tarfile.open("/inputs/sdist/" + os.environ["SDIST_FILENAME"], "r:gz")'
    '.extractall("/work")\'; '
    'cd "/work/${SOURCE_ROOT}"; '
    "python setup.py bdist_wheel --dist-dir /output --bdist-dir /work/bdist",
)
BUILD_CONTAINER_ARGUMENTS = (
    "--rm",
    "--platform",
    CONTAINER_PLATFORM,
    "--network=none",
    "--read-only",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
    "--user",
    "65534:65534",
    "--tmpfs",
    "/tmp:rw,nosuid,nodev,noexec,mode=1777",
    "--tmpfs",
    "/work:rw,nosuid,nodev,noexec,mode=700,uid=65534,gid=65534",
)
BUILD_MOUNTS = (
    {"container_path": "/inputs/sdist", "input": "source_distribution", "mode": "ro"},
    {"container_path": "/inputs/build", "input": "build_tool_wheelhouse", "mode": "ro"},
    {"container_path": "/output", "input": "fresh_output_directory", "mode": "rw"},
)
BUILD_TOOL_DISTRIBUTIONS = (
    {
        "byte_length": 1_093_916,
        "filename": "setuptools-67.8.0-py3-none-any.whl",
        "name": "setuptools",
        "sha256": "5df61bf30bb10c6f756eb19e7c9f3b473051f48db77fddbe06ff2ca307df9a6f",
        "version": "67.8.0",
    },
    {
        "byte_length": 65_801,
        "filename": "wheel-0.41.3-py3-none-any.whl",
        "name": "wheel",
        "sha256": "488609bc63a29322326e05560731bf7bfea8e48ad646e1f5e40d366607de0942",
        "version": "0.41.3",
    },
)
_BUILD_TOP_LEVEL_KEYS = frozenset(
    {
        "allowlist",
        "build",
        "diagnostic_execution",
        "native_reproduction_required",
        "schema",
        "target",
    }
)
_BUILD_KEYS = frozenset(
    {
        "base_image",
        "command",
        "container_arguments",
        "container_platform",
        "environment",
        "mounts",
        "tool_distributions",
    }
)
_DIAGNOSTIC_KEYS = frozenset({"docker_server_version", "emulated", "host_platform"})
_SOURCE_DISTRIBUTION_KEYS = frozenset(
    {
        "byte_length",
        "filename",
        "name",
        "sha256",
        "source_date_epoch",
        "url",
        "version",
    }
)
_ALLOWLIST_ENTRY_KEYS = frozenset(
    {
        "build_environment",
        "compatible_published_wheel_count",
        "fresh_builds",
        "pypi_release",
        "required_by",
        "source_distribution",
        "wheel",
    }
)
_PYPI_RELEASE_KEYS = frozenset({"files", "project", "version"})
_PYPI_RELEASE_FILE_KEYS = frozenset(
    {
        "byte_length",
        "filename",
        "packagetype",
        "python_version",
        "sha256",
        "url",
        "wheel_tags",
        "yanked",
    }
)
_TARGET_KEYS = frozenset({"abi", "implementation", "platform", "python_version"})
_WHEEL_KEYS = frozenset(
    {
        "byte_length",
        "filename",
        "name",
        "root_is_purelib",
        "sha256",
        "tag",
        "version",
    }
)
_DISTRIBUTION_IDENTITY_KEYS = frozenset({"byte_length", "filename", "name", "sha256", "version"})
_LOCK_HEADER = (
    b"--no-index\n--require-hashes\n--only-binary=:all:\n--find-links=/opt/crux/wheelhouse\n"
)
_PIN_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)"
)
_LOCK_COMMENT_PATTERN = re.compile(
    r"# filename=(?P<filename>[^/\\\s]+) byte_length=(?P<byte_length>[1-9][0-9]*)"
)
_LOCK_PIN_PATTERN = re.compile(
    r"(?P<name>[a-z0-9][a-z0-9.-]*)==(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*) "
    r"--hash=sha256:(?P<sha256>[0-9a-f]{64})"
)
_COMPATIBLE_TAGS = frozenset(
    (tag.interpreter, tag.abi, tag.platform)
    for tag in (
        *cpython_tags(
            python_version=(3, 7),
            abis=(TARGET_ABI,),
            platforms=(TARGET_PLATFORM, "manylinux1_x86_64"),
        ),
        *compatible_tags(
            python_version=(3, 7),
            interpreter="cp37",
            platforms=(TARGET_PLATFORM, "manylinux1_x86_64"),
        ),
    )
)
_TARGET_MARKER_ENVIRONMENT = {
    "extra": "",
    "implementation_name": "cpython",
    "implementation_version": TARGET_PYTHON_FULL_VERSION,
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_system": "Linux",
    "python_full_version": TARGET_PYTHON_FULL_VERSION,
    "python_version": TARGET_PYTHON_VERSION,
    "sys_platform": "linux",
}
_UNFROZEN_MARKER_VARIABLES = frozenset({"platform_release", "platform_version"})


class ResolutionError(ValueError):
    """The wheel closure is incompatible, incomplete, or non-reproducible."""


class DistributionBuildError(ValueError):
    """The exceptional source distribution build is not reproducible or pure."""


@dataclass(frozen=True)
class DistributionRecord:
    name: str
    version: str
    filename: str
    byte_length: int
    sha256: str
    requirements: tuple[str, ...]


@dataclass(frozen=True)
class SdistExceptionSpec:
    name: str
    version: str
    sdist_filename: str
    sdist_url: str
    sdist_byte_length: int
    sdist_sha256: str
    source_date_epoch: int
    source_root: str
    wheel_filename: str
    compatible_published_wheel_count: int = 0


@dataclass(frozen=True)
class SdistBuildRecord:
    spec: SdistExceptionSpec
    wheel: DistributionRecord


SDIST_ALLOWLIST = (
    SdistExceptionSpec(
        name="gast",
        version="0.2.2",
        sdist_filename=GAST_SDIST_FILENAME,
        sdist_url=GAST_SDIST_URL,
        sdist_byte_length=GAST_SDIST_BYTE_LENGTH,
        sdist_sha256=GAST_SDIST_SHA256,
        source_date_epoch=GAST_SOURCE_DATE_EPOCH,
        source_root="gast-0.2.2",
        wheel_filename=BUILT_GAST_FILENAME,
    ),
    SdistExceptionSpec(
        name="pretty-midi",
        version="0.2.10",
        sdist_filename=APPROVED_SDIST_FILENAME,
        sdist_url=APPROVED_SDIST_URL,
        sdist_byte_length=APPROVED_SDIST_BYTE_LENGTH,
        sdist_sha256=APPROVED_SDIST_SHA256,
        source_date_epoch=SOURCE_DATE_EPOCH,
        source_root="pretty_midi-0.2.10",
        wheel_filename=BUILT_PRETTY_MIDI_FILENAME,
    ),
)
_SDIST_ALLOWLIST_BY_NAME = {entry.name: entry for entry in SDIST_ALLOWLIST}
_SDIST_ALLOWLIST_BY_FILENAME = {entry.sdist_filename: entry for entry in SDIST_ALLOWLIST}


def canonicalize_pypi_release(
    project_name: str,
    project_version: str,
    payload: object,
) -> dict[str, object]:
    """Return the exact canonical file evidence for one PyPI release."""

    normalized_name = canonicalize_name(project_name)
    try:
        normalized_version = str(Version(project_version))
    except ValueError:
        raise ResolutionError("PyPI release version is invalid") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("urls"), list):
        raise ResolutionError("PyPI release metadata is incomplete")
    files: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in payload["urls"]:
        if not isinstance(item, dict):
            raise ResolutionError("PyPI release file metadata is invalid")
        filename = item.get("filename")
        packagetype = item.get("packagetype")
        python_version = item.get("python_version")
        url = item.get("url")
        byte_length = item.get("size")
        sha256 = item.get("digests", {}).get("sha256")
        yanked = item.get("yanked")
        if (
            not isinstance(filename, str)
            or not filename
            or filename in seen
            or not isinstance(packagetype, str)
            or not packagetype
            or not isinstance(python_version, str)
            or not python_version
            or type(byte_length) is not int
            or byte_length <= 0
            or not isinstance(sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", sha256)
            or type(yanked) is not bool
            or not isinstance(url, str)
        ):
            raise ResolutionError("PyPI release file metadata is invalid or ambiguous")
        parsed_url = urllib.parse.urlsplit(url)
        if (
            parsed_url.scheme != "https"
            or parsed_url.netloc != "files.pythonhosted.org"
            or not parsed_url.path.endswith(f"/{filename}")
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ResolutionError("PyPI release file URL is mutable or mismatched")
        wheel_tags: list[str] = []
        if packagetype == "bdist_wheel":
            try:
                wheel_name, wheel_version, _build, tags = parse_wheel_filename(filename)
            except (TypeError, ValueError):
                raise ResolutionError("PyPI release wheel filename is invalid") from None
            if (
                canonicalize_name(wheel_name) != normalized_name
                or str(wheel_version) != normalized_version
            ):
                raise ResolutionError("PyPI release wheel identity is mismatched")
            wheel_tags = sorted((str(tag) for tag in tags), key=lambda value: value.encode("utf-8"))
        seen.add(filename)
        files.append(
            {
                "byte_length": byte_length,
                "filename": filename,
                "packagetype": packagetype,
                "python_version": python_version,
                "sha256": sha256,
                "url": url,
                "wheel_tags": wheel_tags,
                "yanked": yanked,
            }
        )
    if not files:
        raise ResolutionError("PyPI release metadata contains no files")
    files.sort(key=lambda item: item["filename"].encode("utf-8"))
    return {"files": files, "project": normalized_name, "version": normalized_version}


def count_target_compatible_wheels(release: dict[str, object]) -> int:
    """Count every published wheel compatible with the frozen target."""

    count = 0
    for item in release.get("files", []):
        if item.get("packagetype") != "bdist_wheel":
            continue
        tags = item.get("wheel_tags")
        if not isinstance(tags, list):
            raise ResolutionError("PyPI release wheel tags are invalid")
        if any(tuple(tag.split("-", 2)) in _COMPATIBLE_TAGS for tag in tags):
            count += 1
    return count


def _validate_recorded_pypi_release(
    value: object,
    spec: SdistExceptionSpec,
) -> dict[str, object]:
    release = _require_exact_object(value, _PYPI_RELEASE_KEYS, "PyPI release evidence")
    files = release["files"]
    if not isinstance(files, list):
        raise DistributionBuildError("PyPI release file evidence is invalid")
    raw_files: list[dict[str, object]] = []
    for value_item in files:
        item = _require_exact_object(
            value_item,
            _PYPI_RELEASE_FILE_KEYS,
            "PyPI release file evidence",
        )
        raw_files.append(
            {
                "digests": {"sha256": item["sha256"]},
                "filename": item["filename"],
                "packagetype": item["packagetype"],
                "python_version": item["python_version"],
                "size": item["byte_length"],
                "url": item["url"],
                "yanked": item["yanked"],
            }
        )
    try:
        canonical = canonicalize_pypi_release(
            spec.name,
            spec.version,
            {"urls": raw_files},
        )
    except ResolutionError:
        raise DistributionBuildError("PyPI release evidence is invalid") from None
    if canonical != release:
        raise DistributionBuildError("PyPI release evidence is not canonical")
    source_matches = [
        item
        for item in files
        if item["filename"] == spec.sdist_filename and item["packagetype"] == "sdist"
    ]
    if len(source_matches) != 1:
        raise DistributionBuildError("approved source is absent from PyPI release evidence")
    source = source_matches[0]
    if (
        source["byte_length"] != spec.sdist_byte_length
        or source["sha256"] != spec.sdist_sha256
        or source["url"] != spec.sdist_url
        or source["yanked"] is not False
    ):
        raise DistributionBuildError("approved source does not match PyPI release evidence")
    return release


def derive_allowlist_required_by(
    requirements_path: Path,
    distributions: tuple[DistributionRecord, ...],
) -> dict[str, tuple[str, ...]]:
    """Derive every allowlist exception parent from the selected dependency graph."""

    direct = _load_direct_requirements(Path(requirements_path))
    parents: dict[str, set[str]] = {name: set() for name in _SDIST_ALLOWLIST_BY_NAME}
    for requirement in direct:
        name = canonicalize_name(requirement.name)
        spec = _SDIST_ALLOWLIST_BY_NAME.get(name)
        if spec is not None and spec.version in requirement.specifier:
            parents[name].add("runtime requirements.in")
    for distribution in distributions:
        for dependency_text in distribution.requirements:
            try:
                requirement = Requirement(dependency_text)
            except ValueError:
                raise ResolutionError(
                    f"wheel contains an invalid dependency: {distribution.filename}"
                ) from None
            if not _marker_applies(requirement):
                continue
            name = canonicalize_name(requirement.name)
            spec = _SDIST_ALLOWLIST_BY_NAME.get(name)
            if spec is not None and spec.version in requirement.specifier:
                parents[name].add(f"{distribution.name}=={distribution.version}")
    if any(not values for values in parents.values()):
        raise DistributionBuildError("allowlist entry is not required by the selected graph")
    return {
        name: tuple(sorted(values, key=lambda value: value.encode("utf-8")))
        for name, values in sorted(parents.items(), key=lambda item: item[0].encode("utf-8"))
    }


def validate_approved_sdist(path: Path) -> tuple[int, str]:
    """Require one exact source archive from the explicit reviewed allowlist."""

    source = Path(path)
    spec = _SDIST_ALLOWLIST_BY_FILENAME.get(source.name)
    if spec is None:
        raise DistributionBuildError("source is not in the approved sdist allowlist")
    try:
        content, _metadata = _read_stable_file(source)
    except ResolutionError:
        raise DistributionBuildError("source is not an approved sdist") from None
    digest = hashlib.sha256(content).hexdigest()
    if len(content) != spec.sdist_byte_length or digest != spec.sdist_sha256:
        raise DistributionBuildError("source is not an approved sdist")
    return len(content), digest


def require_reproducible_wheels(first: Path, second: Path) -> tuple[int, str]:
    """Require two fresh exceptional builds to produce byte-identical wheels."""

    try:
        first_content, _ = _read_stable_file(Path(first))
        second_content, _ = _read_stable_file(Path(second))
    except ResolutionError:
        raise DistributionBuildError("exceptional build wheels are unreadable") from None
    if first_content != second_content:
        raise DistributionBuildError("fresh exceptional builds must be byte-identical")
    return len(first_content), hashlib.sha256(first_content).hexdigest()


def validate_built_pure_wheel(
    path: Path,
    spec: SdistExceptionSpec | None = None,
) -> DistributionRecord:
    """Validate exact metadata, purity, and RECORD coverage of the built wheel."""

    wheel_path = Path(path)
    if spec is None:
        matches = [entry for entry in SDIST_ALLOWLIST if entry.wheel_filename == wheel_path.name]
        if len(matches) != 1:
            raise DistributionBuildError("built wheel must name one unambiguous allowlist entry")
        spec = matches[0]
    if wheel_path.name != spec.wheel_filename:
        raise DistributionBuildError("built wheel filename is not approved")
    try:
        record = _inspect_wheel(wheel_path)
    except ResolutionError as error:
        raise DistributionBuildError(str(error)) from None
    if record.name != spec.name or record.version != spec.version:
        raise DistributionBuildError("built wheel name/version is not approved")
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            infos = archive.infolist()
            if len(infos) != len({info.filename for info in infos}):
                raise DistributionBuildError("built wheel contains duplicate paths")
            members = {info.filename: archive.read(info) for info in infos if not info.is_dir()}
    except (OSError, KeyError, zipfile.BadZipFile):
        raise DistributionBuildError("built wheel is unreadable") from None
    record_paths = [name for name in members if name.endswith(".dist-info/RECORD")]
    if len(record_paths) != 1:
        raise DistributionBuildError("built wheel has no complete RECORD")
    record_path = record_paths[0]
    wheel_metadata_paths = [name for name in members if name.endswith(".dist-info/WHEEL")]
    if len(wheel_metadata_paths) != 1:
        raise DistributionBuildError("built wheel has ambiguous WHEEL metadata")
    wheel_message = email.parser.BytesParser().parsebytes(members[wheel_metadata_paths[0]])
    if wheel_message.get("Root-Is-Purelib") != "true":
        raise DistributionBuildError("built wheel is not Root-Is-Purelib: true")
    for info in infos:
        name = info.filename
        if (
            name.startswith("/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))
        ):
            raise DistributionBuildError("built wheel contains a noncanonical path")
        if not info.is_dir() and (
            name.lower().endswith((".so", ".dylib", ".dll", ".pyd", ".exe"))
            or ((info.external_attr >> 16) & 0o111)
        ):
            raise DistributionBuildError("built wheel contains a native or executable file")
    _validate_wheel_record(members, record_path)
    return record


def render_distribution_build_manifest(
    builds: tuple[SdistBuildRecord, ...],
    *,
    diagnostic_execution: dict[str, object],
    pypi_releases: dict[str, dict[str, object]],
    required_by: dict[str, tuple[str, ...]],
) -> bytes:
    """Render the canonical identity for every allowlisted reproducible build."""

    ordered = tuple(sorted(builds, key=lambda item: item.spec.name.encode("utf-8")))
    if tuple(item.spec for item in ordered) != SDIST_ALLOWLIST:
        raise DistributionBuildError("sdist build allowlist is incomplete or ambiguous")
    allowlist: list[dict[str, object]] = []
    for build_record in ordered:
        spec = build_record.spec
        wheel = build_record.wheel
        _validate_record(wheel)
        if (
            wheel.name != spec.name
            or wheel.version != spec.version
            or wheel.filename != spec.wheel_filename
        ):
            raise DistributionBuildError("final wheel identity is not approved")
        try:
            release = pypi_releases[spec.name]
            parents = required_by[spec.name]
        except KeyError:
            raise DistributionBuildError("allowlist eligibility evidence is incomplete") from None
        allowlist.append(_build_manifest_entry(build_record, release, parents))
    if set(pypi_releases) != set(_SDIST_ALLOWLIST_BY_NAME) or set(required_by) != set(
        _SDIST_ALLOWLIST_BY_NAME
    ):
        raise DistributionBuildError("allowlist eligibility evidence is ambiguous")
    payload: dict[str, object] = {
        "allowlist": allowlist,
        "build": {
            "base_image": BASE_IMAGE,
            "command": list(BUILD_COMMAND),
            "container_arguments": list(BUILD_CONTAINER_ARGUMENTS),
            "container_platform": CONTAINER_PLATFORM,
            "environment": BUILD_ENVIRONMENT,
            "mounts": [dict(item) for item in BUILD_MOUNTS],
            "tool_distributions": [dict(item) for item in BUILD_TOOL_DISTRIBUTIONS],
        },
        "diagnostic_execution": diagnostic_execution,
        "native_reproduction_required": True,
        "schema": DISTRIBUTION_BUILD_SCHEMA,
        "target": {
            "abi": TARGET_ABI,
            "implementation": "cpython",
            "platform": TARGET_PLATFORM,
            "python_version": TARGET_PYTHON_FULL_VERSION,
        },
    }
    rendered = canonical_json_bytes(payload, trailing_newline=True)
    validate_distribution_build_manifest(rendered)
    return rendered


def validate_distribution_build_manifest(
    content: bytes,
    *,
    wheelhouse_path: Path | None = None,
    requirements_path: Path | None = None,
    runtime_distributions: tuple[DistributionRecord, ...] | None = None,
) -> dict[str, object]:
    """Strictly validate the allowlist recipe and optionally every built wheel."""

    try:
        payload = strict_json_loads(content)
    except (UnicodeError, ValueError, TypeError):
        raise DistributionBuildError("distribution build manifest is invalid JSON") from None
    if not isinstance(payload, dict) or set(payload) != _BUILD_TOP_LEVEL_KEYS:
        raise DistributionBuildError("distribution build manifest has unknown or missing fields")
    if canonical_json_bytes(payload, trailing_newline=True) != content:
        raise DistributionBuildError("distribution build manifest is not canonical JSON")
    if payload["schema"] != DISTRIBUTION_BUILD_SCHEMA:
        raise DistributionBuildError("distribution build manifest schema is invalid")
    if payload["native_reproduction_required"] is not True:
        raise DistributionBuildError("native reproduction must remain required")

    target = _require_exact_object(payload["target"], _TARGET_KEYS, "target")
    if target != {
        "abi": TARGET_ABI,
        "implementation": "cpython",
        "platform": TARGET_PLATFORM,
        "python_version": TARGET_PYTHON_FULL_VERSION,
    }:
        raise DistributionBuildError("target interpreter or platform is not approved")

    build = _require_exact_object(payload["build"], _BUILD_KEYS, "build recipe")
    if (
        build["base_image"] != BASE_IMAGE
        or build["container_platform"] != CONTAINER_PLATFORM
        or build["command"] != list(BUILD_COMMAND)
        or build["container_arguments"] != list(BUILD_CONTAINER_ARGUMENTS)
        or build["environment"] != BUILD_ENVIRONMENT
        or build["mounts"] != [dict(item) for item in BUILD_MOUNTS]
    ):
        raise DistributionBuildError("build recipe is not approved")
    if build["tool_distributions"] != [dict(item) for item in BUILD_TOOL_DISTRIBUTIONS]:
        raise DistributionBuildError("build toolchain identity is not approved")

    diagnostic = _require_exact_object(
        payload["diagnostic_execution"],
        _DIAGNOSTIC_KEYS,
        "diagnostic execution",
    )
    if (
        not isinstance(diagnostic["docker_server_version"], str)
        or not diagnostic["docker_server_version"]
        or not isinstance(diagnostic["host_platform"], str)
        or not diagnostic["host_platform"]
        or type(diagnostic["emulated"]) is not bool
    ):
        raise DistributionBuildError("diagnostic execution metadata is invalid")

    entries = payload["allowlist"]
    if not isinstance(entries, list) or len(entries) != len(SDIST_ALLOWLIST):
        raise DistributionBuildError("distribution build allowlist is incomplete or ambiguous")
    names: list[str] = []
    recorded_parents: dict[str, tuple[str, ...]] = {}
    for entry, spec in zip(entries, SDIST_ALLOWLIST, strict=True):
        name, identity, parents = _validate_build_manifest_entry(entry, spec)
        names.append(name)
        recorded_parents[name] = parents
        if wheelhouse_path is not None:
            actual = validate_built_pure_wheel(
                Path(wheelhouse_path) / spec.wheel_filename,
                spec,
            )
            if _distribution_identity(actual) != identity:
                raise DistributionBuildError("final wheel bytes do not match the manifest")
    if names != sorted(names, key=lambda value: value.encode("utf-8")) or len(names) != len(
        set(names)
    ):
        raise DistributionBuildError("distribution build allowlist is ambiguous")
    if (requirements_path is None) != (runtime_distributions is None):
        raise DistributionBuildError("dependency graph validation inputs are incomplete")
    if requirements_path is not None and runtime_distributions is not None:
        derived = derive_allowlist_required_by(requirements_path, runtime_distributions)
        if derived != recorded_parents:
            raise DistributionBuildError("allowlist required_by does not match the selected graph")
    return payload


def _build_manifest_entry(
    build_record: SdistBuildRecord,
    pypi_release: dict[str, object],
    required_by: tuple[str, ...],
) -> dict[str, object]:
    spec = build_record.spec
    wheel = build_record.wheel
    return {
        "build_environment": {
            "SDIST_FILENAME": spec.sdist_filename,
            "SOURCE_DATE_EPOCH": str(spec.source_date_epoch),
            "SOURCE_ROOT": spec.source_root,
        },
        "compatible_published_wheel_count": count_target_compatible_wheels(pypi_release),
        "fresh_builds": [
            {"byte_length": wheel.byte_length, "sha256": wheel.sha256},
            {"byte_length": wheel.byte_length, "sha256": wheel.sha256},
        ],
        "pypi_release": pypi_release,
        "required_by": list(required_by),
        "source_distribution": {
            "byte_length": spec.sdist_byte_length,
            "filename": spec.sdist_filename,
            "name": spec.name,
            "sha256": spec.sdist_sha256,
            "source_date_epoch": spec.source_date_epoch,
            "url": spec.sdist_url,
            "version": spec.version,
        },
        "wheel": {
            **_distribution_identity(wheel),
            "root_is_purelib": True,
            "tag": "py3-none-any",
        },
    }


def _validate_build_manifest_entry(
    value: object,
    spec: SdistExceptionSpec,
) -> tuple[str, dict[str, object], tuple[str, ...]]:
    entry = _require_exact_object(value, _ALLOWLIST_ENTRY_KEYS, "allowlist entry")
    source = _require_exact_object(
        entry["source_distribution"],
        _SOURCE_DISTRIBUTION_KEYS,
        "source distribution",
    )
    expected_source = {
        "byte_length": spec.sdist_byte_length,
        "filename": spec.sdist_filename,
        "name": spec.name,
        "sha256": spec.sdist_sha256,
        "source_date_epoch": spec.source_date_epoch,
        "url": spec.sdist_url,
        "version": spec.version,
    }
    if source != expected_source:
        raise DistributionBuildError("source distribution identity is not approved")
    expected_environment = {
        "SDIST_FILENAME": spec.sdist_filename,
        "SOURCE_DATE_EPOCH": str(spec.source_date_epoch),
        "SOURCE_ROOT": spec.source_root,
    }
    parents = entry["required_by"]
    if (
        entry["build_environment"] != expected_environment
        or not isinstance(parents, list)
        or not parents
        or any(not isinstance(item, str) or not item for item in parents)
        or parents != sorted(set(parents), key=lambda item: item.encode("utf-8"))
    ):
        raise DistributionBuildError("allowlist eligibility or recipe is not approved")
    release = _validate_recorded_pypi_release(entry["pypi_release"], spec)
    compatible_count = count_target_compatible_wheels(release)
    if compatible_count != 0 or entry["compatible_published_wheel_count"] != compatible_count:
        raise DistributionBuildError("allowlist has a compatible published wheel")
    wheel = _require_exact_object(entry["wheel"], _WHEEL_KEYS, "final wheel")
    identity = {key: wheel[key] for key in _DISTRIBUTION_IDENTITY_KEYS}
    _validate_manifest_distribution_identity(identity, "final wheel")
    if (
        wheel["name"] != spec.name
        or wheel["version"] != spec.version
        or wheel["filename"] != spec.wheel_filename
        or wheel["root_is_purelib"] is not True
        or wheel["tag"] != "py3-none-any"
    ):
        raise DistributionBuildError("final wheel metadata is not approved")
    builds = entry["fresh_builds"]
    expected_build = {
        "byte_length": identity["byte_length"],
        "sha256": identity["sha256"],
    }
    if (
        not isinstance(builds, list)
        or len(builds) != 2
        or any(item != expected_build for item in builds)
    ):
        raise DistributionBuildError("fresh build outputs are not reproducible")
    return spec.name, identity, tuple(parents)


def build_allowlisted_sdists(
    *,
    sdist_paths: tuple[Path, ...],
    build_tool_wheelhouse: Path,
    build_requirements_path: Path,
    build_lock_path: Path,
    output_wheelhouse_path: Path,
    manifest_path: Path,
    runtime_requirements_path: Path | None = None,
    runtime_wheelhouse_path: Path | None = None,
    docker_executable: str = "docker",
) -> tuple[tuple[SdistBuildRecord, ...], bytes]:
    """Build every allowlisted sdist twice in fresh exact offline containers."""

    source_names = [Path(path).name for path in sdist_paths]
    if len(source_names) != len(set(source_names)):
        raise DistributionBuildError("sdist build inputs contain duplicate basenames")
    sources = {Path(path).name: Path(path) for path in sdist_paths}
    if set(sources) != set(_SDIST_ALLOWLIST_BY_FILENAME):
        raise DistributionBuildError("sdist build inputs do not match the exact allowlist")
    if runtime_requirements_path is None or runtime_wheelhouse_path is None:
        raise DistributionBuildError("runtime dependency graph inputs are required")
    runtime_distributions = verify_requirement_closure(
        Path(runtime_requirements_path),
        inspect_wheelhouse(Path(runtime_wheelhouse_path)),
    )
    required_by = derive_allowlist_required_by(
        Path(runtime_requirements_path),
        runtime_distributions,
    )
    pypi_releases = {
        spec.name: _fetch_pypi_release(spec.name, spec.version) for spec in SDIST_ALLOWLIST
    }
    build_tools = verify_offline_resolution(
        requirements_path=Path(build_requirements_path),
        wheelhouse=Path(build_tool_wheelhouse),
        lock_path=Path(build_lock_path),
    )
    if [_distribution_identity(item) for item in build_tools] != [
        dict(item) for item in BUILD_TOOL_DISTRIBUTIONS
    ]:
        raise DistributionBuildError("build toolchain bytes do not match the approved lock")
    _verify_build_container(docker_executable)
    docker_version = _docker_server_version(docker_executable)

    output_wheelhouse = Path(output_wheelhouse_path)
    manifest = Path(manifest_path)
    output_wheelhouse.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".allowlisted-sdist-build.", dir=output_wheelhouse.parent)
    )
    try:
        input_tools = staging / "build-tools"
        input_tools.mkdir(mode=0o700)
        for tool in build_tools:
            content, _ = _read_stable_file(Path(build_tool_wheelhouse) / tool.filename)
            (input_tools / tool.filename).write_bytes(content)
        build_records: list[SdistBuildRecord] = []
        wheel_contents: dict[str, bytes] = {}
        for spec in SDIST_ALLOWLIST:
            source_content, _ = _read_stable_file(sources[spec.sdist_filename])
            validate_approved_sdist(sources[spec.sdist_filename])
            _validate_approved_sdist_archive(source_content, spec)
            input_sdist = staging / f"sdist-{spec.name}"
            input_sdist.mkdir(mode=0o700)
            (input_sdist / spec.sdist_filename).write_bytes(source_content)
            built_paths: list[Path] = []
            for build_number in (1, 2):
                output = staging / f"output-{spec.name}-{build_number}"
                output.mkdir(mode=0o777)
                os.chmod(output, 0o777)
                _run_exceptional_build(
                    docker_executable=docker_executable,
                    sdist_directory=input_sdist,
                    build_tool_directory=input_tools,
                    output_directory=output,
                    spec=spec,
                )
                entries = sorted(output.iterdir(), key=lambda item: item.name.encode("utf-8"))
                if len(entries) != 1 or entries[0].name != spec.wheel_filename:
                    raise DistributionBuildError(
                        "exceptional build did not produce exactly the approved wheel"
                    )
                validate_built_pure_wheel(entries[0], spec)
                built_paths.append(entries[0])
            require_reproducible_wheels(built_paths[0], built_paths[1])
            record = validate_built_pure_wheel(built_paths[0], spec)
            build_records.append(SdistBuildRecord(spec=spec, wheel=record))
            wheel_contents[spec.wheel_filename] = _read_stable_file(built_paths[0])[0]
        diagnostic = {
            "docker_server_version": docker_version,
            "emulated": (
                platform.system().lower() != "linux"
                or platform.machine().lower() not in {"amd64", "x86_64"}
            ),
            "host_platform": (f"{platform.system().lower()}/{platform.machine().lower()}"),
        }
        manifest_content = render_distribution_build_manifest(
            tuple(build_records),
            diagnostic_execution=diagnostic,
            pypi_releases=pypi_releases,
            required_by=required_by,
        )
        _publish_allowlisted_builds(
            output_wheelhouse=output_wheelhouse,
            manifest=manifest,
            wheel_contents=wheel_contents,
            manifest_content=manifest_content,
        )
        return tuple(build_records), manifest_content
    except (OSError, ResolutionError) as error:
        if isinstance(error, ResolutionError):
            raise DistributionBuildError(str(error)) from None
        raise DistributionBuildError("exceptional build staging failed") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _validate_approved_sdist_archive(
    content: bytes,
    spec: SdistExceptionSpec,
) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError):
        raise DistributionBuildError("approved source distribution is unreadable") from None
    if not members:
        raise DistributionBuildError("approved source distribution is empty")
    names: set[str] = set()
    for member in members:
        name = member.name
        path = Path(name)
        if (
            not name
            or name.startswith("/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.parts[0] != spec.source_root
            or name in names
            or not (member.isfile() or member.isdir())
        ):
            raise DistributionBuildError("approved source distribution contains an unsafe member")
        names.add(name)
    if f"{spec.source_root}/setup.py" not in names:
        raise DistributionBuildError("approved source distribution has no setup.py")


def _verify_build_container(docker_executable: str) -> None:
    command = (
        docker_executable,
        "image",
        "inspect",
        "--format",
        "{{.Architecture}}\n{{.Os}}\n{{json .RepoDigests}}",
        BASE_IMAGE,
    )
    result = _run_checked(command, "could not inspect the exact build image")
    lines = result.stdout.strip().splitlines()
    if len(lines) != 3 or lines[0] != "amd64" or lines[1] != "linux":
        raise DistributionBuildError("build image platform is not linux/amd64")
    try:
        repo_digests = json.loads(lines[2])
    except (TypeError, ValueError):
        raise DistributionBuildError("build image digest metadata is invalid") from None
    if not isinstance(repo_digests, list) or BASE_IMAGE not in repo_digests:
        raise DistributionBuildError("build image does not match the approved digest")


def _docker_server_version(docker_executable: str) -> str:
    result = _run_checked(
        (docker_executable, "version", "--format", "{{.Server.Version}}"),
        "could not inspect Docker server version",
    )
    version = result.stdout.strip()
    if not version or "\n" in version:
        raise DistributionBuildError("Docker server version metadata is invalid")
    return version


def _run_exceptional_build(
    *,
    docker_executable: str,
    sdist_directory: Path,
    build_tool_directory: Path,
    output_directory: Path,
    spec: SdistExceptionSpec,
) -> None:
    command = [
        docker_executable,
        "run",
        *BUILD_CONTAINER_ARGUMENTS,
        "--mount",
        f"type=bind,src={sdist_directory.resolve()},dst=/inputs/sdist,readonly",
        "--mount",
        f"type=bind,src={build_tool_directory.resolve()},dst=/inputs/build,readonly",
        "--mount",
        f"type=bind,src={output_directory.resolve()},dst=/output",
    ]
    for name, value in BUILD_ENVIRONMENT.items():
        command.extend(("--env", f"{name}={value}"))
    for name, value in (
        ("SDIST_FILENAME", spec.sdist_filename),
        ("SOURCE_DATE_EPOCH", str(spec.source_date_epoch)),
        ("SOURCE_ROOT", spec.source_root),
    ):
        command.extend(("--env", f"{name}={value}"))
    command.extend((BASE_IMAGE, *BUILD_COMMAND))
    _run_checked(tuple(command), "exceptional source build failed")


def _run_checked(
    command: tuple[str, ...],
    failure_message: str,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        raise DistributionBuildError(failure_message) from None
    if result.returncode != 0:
        diagnostic = result.stderr.strip().splitlines()[-1:] or ["unknown failure"]
        raise DistributionBuildError(f"{failure_message}: {diagnostic[0]}")
    return result


def _publish_allowlisted_builds(
    *,
    output_wheelhouse: Path,
    manifest: Path,
    wheel_contents: dict[str, bytes],
    manifest_content: bytes,
) -> None:
    output_exists = output_wheelhouse.exists() or output_wheelhouse.is_symlink()
    manifest_exists = manifest.exists() or manifest.is_symlink()
    if output_exists or manifest_exists:
        if not output_exists or not manifest_exists:
            raise DistributionBuildError("exceptional build publication is incomplete")
        try:
            actual_names = {
                path.name
                for path in output_wheelhouse.iterdir()
                if path.is_file() and not path.is_symlink()
            }
            if actual_names != set(wheel_contents):
                raise DistributionBuildError("existing exceptional wheelhouse is stale")
            for filename, content in wheel_contents.items():
                if _read_stable_file(output_wheelhouse / filename)[0] != content:
                    raise DistributionBuildError("existing exceptional wheel is stale")
            if manifest.read_bytes() != manifest_content:
                raise DistributionBuildError("existing distribution build manifest is stale")
        except (OSError, ResolutionError):
            raise DistributionBuildError(
                "existing exceptional build publication is unreadable"
            ) from None
        return
    publication = Path(
        tempfile.mkdtemp(
            prefix=".allowlisted-wheelhouse.",
            dir=output_wheelhouse.parent,
        )
    )
    published_wheelhouse = False
    try:
        for filename, content in sorted(
            wheel_contents.items(), key=lambda item: item[0].encode("utf-8")
        ):
            output = publication / filename
            with output.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(publication, output_wheelhouse)
        published_wheelhouse = True
        atomic_replace_bytes(manifest, manifest_content)
    except (OSError, ValueError) as error:
        if published_wheelhouse:
            shutil.rmtree(output_wheelhouse, ignore_errors=True)
        raise DistributionBuildError("could not publish exceptional build atomically") from error
    finally:
        shutil.rmtree(publication, ignore_errors=True)


def inspect_wheelhouse(wheelhouse: Path) -> tuple[DistributionRecord, ...]:
    """Read and validate every wheel without importing or installing it."""

    root = Path(wheelhouse)
    try:
        root_metadata = os.lstat(root)
    except OSError:
        raise ResolutionError("wheelhouse is unavailable") from None
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ResolutionError("wheelhouse must be a no-follow directory")
    records: list[DistributionRecord] = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name.encode("utf-8")):
        metadata = os.lstat(entry)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ResolutionError("wheelhouse contains a symlink or special file")
        if entry.suffix != ".whl":
            raise ResolutionError("wheelhouse contains a source distribution or non-wheel file")
        records.append(_inspect_wheel(entry))
    if not records:
        raise ResolutionError("wheelhouse is empty")
    names = [record.name for record in records]
    if len(names) != len(set(names)):
        raise ResolutionError("wheelhouse has duplicate or ambiguous normalized names")
    return tuple(records)


def verify_requirement_closure(
    requirements_path: Path,
    distributions: tuple[DistributionRecord, ...],
) -> tuple[DistributionRecord, ...]:
    """Return the exact dependency closure and reject missing or extra wheels."""

    direct = _load_direct_requirements(Path(requirements_path))
    by_name = {distribution.name: distribution for distribution in distributions}
    pending = deque(direct)
    selected: set[str] = set()
    while pending:
        requirement = pending.popleft()
        if not _marker_applies(requirement):
            continue
        normalized_name = canonicalize_name(requirement.name)
        distribution = by_name.get(normalized_name)
        if distribution is None:
            raise ResolutionError(f"dependency is missing from wheelhouse: {normalized_name}")
        if requirement.extras:
            raise ResolutionError("dependency extras must be expanded by the resolver")
        if distribution.version not in requirement.specifier:
            raise ResolutionError(
                f"dependency version does not satisfy metadata: {normalized_name}"
            )
        if normalized_name in selected:
            continue
        selected.add(normalized_name)
        for dependency_text in distribution.requirements:
            try:
                pending.append(Requirement(dependency_text))
            except ValueError:
                raise ResolutionError(
                    f"wheel contains an invalid dependency: {distribution.filename}"
                ) from None
    extras = set(by_name) - selected
    if extras:
        raise ResolutionError(f"wheelhouse contains extra distributions: {sorted(extras)!r}")
    return tuple(sorted((by_name[name] for name in selected), key=_record_sort_key))


def render_requirements_lock(distributions: tuple[DistributionRecord, ...]) -> bytes:
    """Render deterministic pip-compatible hash-required lock bytes."""

    ordered = sorted(distributions, key=_record_sort_key)
    names = [record.name for record in ordered]
    if len(names) != len(set(names)):
        raise ResolutionError("cannot render duplicate normalized distribution names")
    body = bytearray(_LOCK_HEADER)
    for record in ordered:
        _validate_record(record)
        body.extend(
            (
                f"# filename={record.filename} byte_length={record.byte_length}\n"
                f"{record.name}=={record.version} --hash=sha256:{record.sha256}\n"
            ).encode("utf-8")
        )
    return bytes(body)


def _parse_requirements_lock(content: bytes) -> tuple[DistributionRecord, ...]:
    if not content.startswith(_LOCK_HEADER):
        raise ResolutionError("requirements lock header is invalid")
    try:
        lines = content[len(_LOCK_HEADER) :].decode("utf-8").splitlines()
    except UnicodeError:
        raise ResolutionError("requirements lock is not UTF-8") from None
    if len(lines) % 2:
        raise ResolutionError("requirements lock entries are incomplete")
    records: list[DistributionRecord] = []
    for index in range(0, len(lines), 2):
        comment = _LOCK_COMMENT_PATTERN.fullmatch(lines[index])
        pin = _LOCK_PIN_PATTERN.fullmatch(lines[index + 1])
        if comment is None or pin is None:
            raise ResolutionError("requirements lock entry is invalid")
        record = DistributionRecord(
            name=pin["name"],
            version=pin["version"],
            filename=comment["filename"],
            byte_length=int(comment["byte_length"]),
            sha256=pin["sha256"],
            requirements=(),
        )
        _validate_record(record)
        records.append(record)
    if not records:
        raise ResolutionError("requirements lock is empty")
    ordered = tuple(sorted(records, key=_record_sort_key))
    if tuple(records) != ordered or len({record.name for record in records}) != len(records):
        raise ResolutionError("requirements lock entries are not canonical")
    return ordered


def _verify_materialized_wheelhouse(
    root: Path,
    locked: tuple[DistributionRecord, ...],
) -> tuple[DistributionRecord, ...]:
    try:
        metadata = os.lstat(root)
    except OSError:
        raise ResolutionError("materialized wheelhouse is unavailable") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ResolutionError("materialized wheelhouse must be a no-follow directory")
    expected = {record.filename for record in locked}
    actual: set[str] = set()
    for entry in root.iterdir():
        entry_metadata = os.lstat(entry)
        if stat.S_ISLNK(entry_metadata.st_mode) or not stat.S_ISREG(entry_metadata.st_mode):
            raise ResolutionError("materialized wheelhouse contains a symlink or special file")
        actual.add(entry.name)
    if actual - expected:
        raise ResolutionError("materialized wheelhouse contains extra files")
    if expected - actual:
        raise ResolutionError("materialized wheelhouse is missing locked files")
    inspected = inspect_wheelhouse(root)
    if render_requirements_lock(inspected) != render_requirements_lock(locked):
        raise ResolutionError("materialized wheelhouse bytes do not match the lock")
    return inspected


def _download_locked_wheel(record: DistributionRecord) -> bytes:
    release = _fetch_pypi_release(record.name, record.version)
    matches = [
        item
        for item in release["files"]
        if item["filename"] == record.filename
        and item["packagetype"] == "bdist_wheel"
        and item["byte_length"] == record.byte_length
        and item["sha256"] == record.sha256
        and item["yanked"] is False
    ]
    if len(matches) != 1:
        raise ResolutionError(f"locked wheel has no exact immutable PyPI file: {record.filename}")
    try:
        with urllib.request.urlopen(matches[0]["url"], timeout=60) as response:
            content = response.read()
    except (OSError, urllib.error.URLError):
        raise ResolutionError(f"could not download locked wheel: {record.filename}") from None
    if len(content) != record.byte_length or hashlib.sha256(content).hexdigest() != record.sha256:
        raise ResolutionError(f"downloaded locked wheel bytes drifted: {record.filename}")
    return content


def materialize_locked_wheelhouse(
    *,
    lock_path: Path,
    destination: Path,
    offline_cache: Path | None = None,
) -> tuple[DistributionRecord, ...]:
    """Materialize exactly one lock into a no-follow wheelhouse."""

    lock_content, _ = _read_stable_file(Path(lock_path))
    locked = _parse_requirements_lock(lock_content)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        return _verify_materialized_wheelhouse(destination, locked)
    cache = Path(offline_cache) if offline_cache is not None else None
    if cache is not None:
        try:
            cache_metadata = os.lstat(cache)
        except OSError:
            raise ResolutionError("offline wheel cache is unavailable") from None
        if stat.S_ISLNK(cache_metadata.st_mode) or not stat.S_ISDIR(cache_metadata.st_mode):
            raise ResolutionError("offline wheel cache must be a no-follow directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".locked-wheelhouse.", dir=destination.parent))
    published = False
    try:
        for record in locked:
            if cache is None:
                content = _download_locked_wheel(record)
            else:
                candidate = cache / record.filename
                try:
                    content, _ = _read_stable_file(candidate)
                except ResolutionError:
                    raise ResolutionError(
                        f"offline wheel cache is missing a locked file: {record.filename}"
                    ) from None
                if (
                    len(content) != record.byte_length
                    or hashlib.sha256(content).hexdigest() != record.sha256
                ):
                    raise ResolutionError(f"offline wheel cache bytes drifted: {record.filename}")
            (staging / record.filename).write_bytes(content)
        inspected = _verify_materialized_wheelhouse(staging, locked)
        os.replace(staging, destination)
        published = True
        return inspected
    except OSError as error:
        raise ResolutionError("could not publish the materialized wheelhouse") from error
    finally:
        if staging.exists() and not published:
            shutil.rmtree(staging, ignore_errors=True)


def verify_offline_resolution(
    *,
    requirements_path: Path,
    wheelhouse: Path,
    lock_path: Path,
    distribution_build_manifest_path: Path | None = None,
) -> tuple[DistributionRecord, ...]:
    """Revalidate local wheels, closure, and exact checked-in lock bytes."""

    distributions = verify_requirement_closure(
        requirements_path,
        inspect_wheelhouse(wheelhouse),
    )
    rendered = render_requirements_lock(distributions)
    try:
        current = Path(lock_path).read_bytes()
    except OSError:
        raise ResolutionError("requirements lock is unavailable") from None
    if current != rendered:
        raise ResolutionError("requirements lock does not match the verified wheelhouse")
    _verify_tensorflow_identity(distributions, Path(requirements_path))
    _verify_exceptional_distribution(
        distributions=distributions,
        requirements_path=Path(requirements_path),
        wheelhouse=Path(wheelhouse),
        manifest_path=distribution_build_manifest_path,
    )
    return distributions


def _host_pip_download_command(
    pip_executable: str,
    *,
    requirements_path: Path,
    staging: Path,
) -> tuple[str, ...]:
    """Return the explicit diagnostic host-pip target command."""

    return (
        pip_executable,
        "download",
        "--disable-pip-version-check",
        "--dest",
        os.fspath(staging),
        "--requirement",
        os.fspath(requirements_path),
        "--implementation",
        TARGET_IMPLEMENTATION,
        "--python-version",
        TARGET_PYTHON_FULL_VERSION,
        "--abi",
        TARGET_ABI,
        "--platform",
        TARGET_PLATFORM,
        "--platform",
        "manylinux1_x86_64",
        "--find-links",
        os.fspath(staging),
        "--only-binary",
        ":all:",
    )


def _download_in_target_container(
    *,
    docker_executable: str,
    requirements_path: Path,
    staging: Path,
) -> None:
    """Resolve markers under the exact CPython 3.7 Linux/amd64 interpreter."""

    _verify_build_container(docker_executable)
    os.chmod(staging, 0o777)
    command = (
        docker_executable,
        "run",
        "--rm",
        "--platform",
        CONTAINER_PLATFORM,
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user",
        "65534:65534",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,mode=1777",
        "--mount",
        (f"type=bind,src={requirements_path.resolve()},dst=/inputs/requirements.in,readonly"),
        "--mount",
        f"type=bind,src={staging.resolve()},dst=/output",
        "--env",
        "HOME=/tmp",
        "--env",
        "PIP_NO_CACHE_DIR=1",
        BASE_IMAGE,
        "python",
        "-m",
        "pip",
        "download",
        "--disable-pip-version-check",
        "--dest",
        "/output",
        "--requirement",
        "/inputs/requirements.in",
        "--implementation",
        TARGET_IMPLEMENTATION,
        "--python-version",
        TARGET_PYTHON_FULL_VERSION,
        "--abi",
        TARGET_ABI,
        "--platform",
        TARGET_PLATFORM,
        "--platform",
        "manylinux1_x86_64",
        "--find-links",
        "/output",
        "--only-binary=:all:",
    )
    try:
        _run_checked(command, "wheel resolution failed")
    except DistributionBuildError as error:
        raise ResolutionError(str(error)) from None


def resolve_runtime(
    *,
    requirements_path: Path,
    wheelhouse: Path,
    lock_path: Path,
    offline: bool,
    pip_executable: str | None = None,
    docker_executable: str = "docker",
    exceptional_wheelhouse_path: Path | None = None,
    distribution_build_manifest_path: Path | None = None,
) -> tuple[DistributionRecord, ...]:
    """Resolve online into private staging or verify an existing offline set."""

    if offline:
        return verify_offline_resolution(
            requirements_path=requirements_path,
            wheelhouse=wheelhouse,
            lock_path=lock_path,
            distribution_build_manifest_path=distribution_build_manifest_path,
        )
    wheelhouse_path = Path(wheelhouse)
    if wheelhouse_path.exists() or wheelhouse_path.is_symlink():
        raise ResolutionError("online resolution requires an absent wheelhouse destination")
    wheelhouse_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".oaf-wheelhouse.", dir=wheelhouse_path.parent))
    published = False
    try:
        if _requires_allowlisted_builds(Path(requirements_path)):
            if exceptional_wheelhouse_path is None or distribution_build_manifest_path is None:
                raise ResolutionError(
                    "runtime requires its allowlisted wheels and distribution build manifest"
                )
            try:
                manifest_content, _ = _read_stable_file(Path(distribution_build_manifest_path))
                validate_distribution_build_manifest(
                    manifest_content,
                    wheelhouse_path=Path(exceptional_wheelhouse_path),
                )
            except DistributionBuildError as error:
                raise ResolutionError(f"allowlisted wheels are invalid: {error}") from None
            for spec in SDIST_ALLOWLIST:
                content, _ = _read_stable_file(
                    Path(exceptional_wheelhouse_path) / spec.wheel_filename
                )
                with (staging / spec.wheel_filename).open("xb") as stream:
                    stream.write(content)
        command = (
            _host_pip_download_command(
                pip_executable,
                requirements_path=Path(requirements_path),
                staging=staging,
            )
            if pip_executable is not None
            else None
        )
        if command is None:
            _download_in_target_container(
                docker_executable=docker_executable,
                requirements_path=Path(requirements_path),
                staging=staging,
            )
        else:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                diagnostic = result.stderr.strip().splitlines()[-1:] or ["unknown pip failure"]
                raise ResolutionError(f"wheel resolution failed: {diagnostic[0]}")
        distributions = verify_requirement_closure(
            requirements_path,
            inspect_wheelhouse(staging),
        )
        _verify_tensorflow_identity(distributions, requirements_path)
        build_manifest = _verify_exceptional_distribution(
            distributions=distributions,
            requirements_path=Path(requirements_path),
            wheelhouse=staging,
            manifest_path=distribution_build_manifest_path,
        )
        _verify_pypi_identities(distributions, build_manifest=build_manifest)
        rendered = render_requirements_lock(distributions)
        os.replace(staging, wheelhouse_path)
        published = True
        try:
            atomic_replace_bytes(lock_path, rendered)
        except OSError:
            shutil.rmtree(wheelhouse_path, ignore_errors=True)
            published = False
            raise
        return distributions
    except (OSError, ResolutionError) as error:
        if published:
            shutil.rmtree(wheelhouse_path, ignore_errors=True)
        if isinstance(error, ResolutionError):
            raise
        raise ResolutionError("could not publish resolved wheel closure") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _inspect_wheel(path: Path) -> DistributionRecord:
    try:
        parsed_name, parsed_version, _build, filename_tags = parse_wheel_filename(path.name)
    except (ValueError, TypeError):
        raise ResolutionError(f"invalid wheel filename: {path.name}") from None
    if not any(
        (tag.interpreter, tag.abi, tag.platform) in _COMPATIBLE_TAGS for tag in filename_tags
    ):
        raise ResolutionError(
            f"wheel is not compatible with CPython 3.7 manylinux2010: {path.name}"
        )
    content, metadata_before = _read_stable_file(path)
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                raise ResolutionError(f"wheel metadata is incomplete or ambiguous: {path.name}")
            metadata_content = archive.read(metadata_names[0])
            wheel_content = archive.read(wheel_names[0])
    except (OSError, KeyError, zipfile.BadZipFile):
        raise ResolutionError(f"wheel archive is unreadable: {path.name}") from None
    metadata_after = os.stat(path, follow_symlinks=False)
    if _stable_stat_identity(metadata_after) != _stable_stat_identity(metadata_before):
        raise ResolutionError(f"wheel changed while being inspected: {path.name}")
    parser = email.parser.BytesParser()
    message = parser.parsebytes(metadata_content)
    metadata_name = message.get("Name")
    metadata_version = message.get("Version")
    if not metadata_name or not metadata_version:
        raise ResolutionError(f"wheel metadata lacks name/version: {path.name}")
    canonical_name = canonicalize_name(metadata_name)
    canonical_version = str(Version(metadata_version))
    if canonical_name != canonicalize_name(parsed_name) or canonical_version != str(parsed_version):
        raise ResolutionError(f"wheel filename conflicts with metadata: {path.name}")
    dist_info_prefix = metadata_names[0].split("/", 1)[0]
    dist_info_stem = dist_info_prefix.removesuffix(".dist-info")
    dist_info_name, separator, dist_info_version = dist_info_stem.rpartition("-")
    if not separator or canonicalize_name(dist_info_name) != canonical_name:
        raise ResolutionError(f"wheel dist-info conflicts with metadata: {path.name}")
    if str(Version(dist_info_version)) != canonical_version:
        raise ResolutionError(f"wheel dist-info version conflicts with metadata: {path.name}")
    wheel_message = parser.parsebytes(wheel_content)
    metadata_tags = {
        tuple(value.split("-", 2))
        for value in wheel_message.get_all("Tag", [])
        if len(value.split("-", 2)) == 3
    }
    filename_tag_values = {(tag.interpreter, tag.abi, tag.platform) for tag in filename_tags}
    if not metadata_tags or metadata_tags != filename_tag_values:
        raise ResolutionError(f"wheel tags conflict with metadata: {path.name}")
    requirements = tuple(message.get_all("Requires-Dist", []))
    return DistributionRecord(
        name=canonical_name,
        version=canonical_version,
        filename=path.name,
        byte_length=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        requirements=requirements,
    )


def _read_stable_file(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ResolutionError("wheel is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stable_stat_identity(before) != _stable_stat_identity(after):
            raise ResolutionError("wheel changed while being read")
        content = b"".join(chunks)
        if len(content) != before.st_size:
            raise ResolutionError("wheel changed while being read")
        return content, before
    except OSError:
        raise ResolutionError(f"wheel is unreadable: {path.name}") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _stable_stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_wheel_record(members: dict[str, bytes], record_path: str) -> None:
    try:
        text = members[record_path].decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeError, csv.Error):
        raise DistributionBuildError("built wheel RECORD is invalid") from None
    if not rows or any(len(row) != 3 for row in rows):
        raise DistributionBuildError("built wheel RECORD is invalid")
    records: dict[str, tuple[str, str]] = {}
    for name, digest, size in rows:
        if not name or name in records:
            raise DistributionBuildError("built wheel RECORD has duplicate or empty paths")
        records[name] = (digest, size)
    if set(records) != set(members):
        raise DistributionBuildError("built wheel RECORD is incomplete")
    for name, content in members.items():
        digest, size = records[name]
        if name == record_path:
            if digest or size:
                raise DistributionBuildError("built wheel RECORD self-row must be unhashed")
            continue
        encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        if digest != f"sha256={encoded}" or size != str(len(content)):
            raise DistributionBuildError("built wheel RECORD hash or size mismatch")


def _load_direct_requirements(path: Path) -> tuple[Requirement, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ResolutionError("requirements input is unreadable") from None
    requirements: list[Requirement] = []
    seen: set[str] = set()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if not _PIN_PATTERN.fullmatch(line):
            raise ResolutionError("direct requirements must be exact top-level pins")
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        if name in seen:
            raise ResolutionError("direct requirements contain duplicate normalized names")
        seen.add(name)
        requirements.append(requirement)
    if not requirements:
        raise ResolutionError("requirements input is empty")
    return tuple(requirements)


def _marker_applies(requirement: Requirement) -> bool:
    if requirement.marker is None:
        return True
    variables: set[str] = set()

    def collect(value: object) -> None:
        if type(value).__name__ == "Variable" and isinstance(getattr(value, "value", None), str):
            variables.add(value.value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(requirement.marker._markers)  # pylint: disable=protected-access
    unsupported = variables & _UNFROZEN_MARKER_VARIABLES
    if unsupported:
        raise ResolutionError(
            f"unsupported marker variable for the frozen target: {sorted(unsupported)!r}"
        )
    return requirement.marker.evaluate(_TARGET_MARKER_ENVIRONMENT)


def _verify_tensorflow_identity(
    distributions: tuple[DistributionRecord, ...],
    requirements_path: Path,
) -> None:
    direct = _load_direct_requirements(requirements_path)
    requires_tensorflow = any(canonicalize_name(item.name) == "tensorflow" for item in direct)
    tensorflow = [item for item in distributions if item.name == "tensorflow"]
    if not requires_tensorflow:
        if tensorflow:
            raise ResolutionError("test dependency closure leaked TensorFlow")
        return
    if len(tensorflow) != 1:
        raise ResolutionError("runtime closure must contain exactly one TensorFlow wheel")
    record = tensorflow[0]
    if record.filename != TENSORFLOW_FILENAME or record.sha256 != TENSORFLOW_SHA256:
        raise ResolutionError("TensorFlow wheel filename or SHA-256 is not frozen")


def _directly_requires(requirements_path: Path, name: str) -> bool:
    expected = canonicalize_name(name)
    return any(
        canonicalize_name(requirement.name) == expected
        for requirement in _load_direct_requirements(Path(requirements_path))
    )


def _requires_allowlisted_builds(requirements_path: Path) -> bool:
    return any(
        _directly_requires(requirements_path, name) for name in ("pretty-midi", "tensorflow")
    )


def _verify_exceptional_distribution(
    *,
    distributions: tuple[DistributionRecord, ...],
    requirements_path: Path,
    wheelhouse: Path,
    manifest_path: Path | None,
) -> dict[str, object] | None:
    requires_exception = _requires_allowlisted_builds(requirements_path)
    selected = [record for record in distributions if record.name in _SDIST_ALLOWLIST_BY_NAME]
    if not requires_exception:
        if selected:
            raise ResolutionError("test dependency closure leaked allowlisted runtime wheels")
        return None
    if {record.name for record in selected} != set(
        _SDIST_ALLOWLIST_BY_NAME
    ) or manifest_path is None:
        raise ResolutionError("runtime closure must contain every manifest-bound allowlisted wheel")
    try:
        content, _ = _read_stable_file(Path(manifest_path))
        payload = validate_distribution_build_manifest(
            content,
            wheelhouse_path=Path(wheelhouse),
            requirements_path=Path(requirements_path),
            runtime_distributions=distributions,
        )
    except (DistributionBuildError, ResolutionError):
        raise ResolutionError(
            "allowlisted wheel does not match the distribution build manifest"
        ) from None
    return payload


def _fetch_pypi_release(name: str, version: str) -> dict[str, object]:
    quoted_name = urllib.parse.quote(name, safe="")
    quoted_version = urllib.parse.quote(version, safe="")
    url = f"https://pypi.org/pypi/{quoted_name}/{quoted_version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        raise ResolutionError(f"could not verify immutable package URL: {name}") from None
    return canonicalize_pypi_release(name, version, payload)


def _verify_pypi_identities(
    distributions: tuple[DistributionRecord, ...],
    *,
    build_manifest: dict[str, object] | None,
) -> None:
    recorded_releases = {
        entry["pypi_release"]["project"]: entry["pypi_release"]
        for entry in (build_manifest or {}).get("allowlist", [])
    }
    for record in distributions:
        release = _fetch_pypi_release(record.name, record.version)
        if record.name in _SDIST_ALLOWLIST_BY_NAME:
            if recorded_releases.get(record.name) != release:
                raise ResolutionError(f"allowlisted PyPI release evidence drifted: {record.name}")
            if count_target_compatible_wheels(release) != 0:
                raise ResolutionError(
                    f"allowlisted release has a compatible published wheel: {record.name}"
                )
            continue
        matches = [
            item
            for item in release["files"]
            if item.get("filename") == record.filename and item.get("packagetype") == "bdist_wheel"
        ]
        if len(matches) != 1:
            raise ResolutionError(f"package URL identity is missing or ambiguous: {record.name}")
        item = matches[0]
        if item.get("yanked"):
            raise ResolutionError(f"resolved wheel is yanked: {record.filename}")
        if item["sha256"] != record.sha256 or item["byte_length"] != record.byte_length:
            raise ResolutionError(f"package URL identity is mutable or mismatched: {record.name}")


def _record_sort_key(record: DistributionRecord) -> bytes:
    return record.name.encode("utf-8")


def _distribution_identity(record: DistributionRecord) -> dict[str, object]:
    return {
        "byte_length": record.byte_length,
        "filename": record.filename,
        "name": record.name,
        "sha256": record.sha256,
        "version": record.version,
    }


def _require_exact_object(
    value: object,
    keys: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DistributionBuildError(f"{label} has unknown or missing fields")
    return value


def _validate_manifest_distribution_identity(
    identity: dict[str, object],
    label: str,
) -> None:
    if set(identity) != _DISTRIBUTION_IDENTITY_KEYS:
        raise DistributionBuildError(f"{label} identity has unknown or missing fields")
    if (
        not isinstance(identity["name"], str)
        or canonicalize_name(identity["name"]) != identity["name"]
        or not isinstance(identity["version"], str)
        or str(Version(identity["version"])) != identity["version"]
        or not isinstance(identity["filename"], str)
        or "/" in identity["filename"]
        or "\\" in identity["filename"]
        or not identity["filename"].endswith(".whl")
        or type(identity["byte_length"]) is not int
        or identity["byte_length"] <= 0
        or not isinstance(identity["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"])
    ):
        raise DistributionBuildError(f"{label} identity is invalid")


def _validate_record(record: DistributionRecord) -> None:
    if canonicalize_name(record.name) != record.name:
        raise ResolutionError("distribution name is not canonical")
    if str(Version(record.version)) != record.version:
        raise ResolutionError("distribution version is not canonical")
    if "/" in record.filename or "\\" in record.filename or not record.filename.endswith(".whl"):
        raise ResolutionError("distribution filename is invalid")
    if type(record.byte_length) is not int or record.byte_length <= 0:
        raise ResolutionError("distribution byte length must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", record.sha256):
        raise ResolutionError("distribution SHA-256 is invalid")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--materialize-wheelhouse", action="store_true")
    parser.add_argument("--offline-cache", type=Path)
    parser.add_argument(
        "--pip-executable",
        help="diagnostic override; default resolution runs in the exact target container",
    )
    parser.add_argument("--exceptional-wheelhouse", type=Path)
    parser.add_argument("--distribution-build-manifest", type=Path)
    parser.add_argument("--build-allowlisted-sdists", action="store_true")
    parser.add_argument("--sdist", action="append", type=Path)
    parser.add_argument("--build-tool-wheelhouse", type=Path)
    parser.add_argument("--build-requirements", type=Path)
    parser.add_argument("--build-lock", type=Path)
    parser.add_argument("--output-wheelhouse", type=Path)
    parser.add_argument("--runtime-requirements", type=Path)
    parser.add_argument("--runtime-wheelhouse", type=Path)
    parser.add_argument("--docker-executable", default="docker")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pylint: disable=too-many-return-statements
    args = _parse_args(argv)
    if args.materialize_wheelhouse:
        if args.lock is None or args.wheelhouse is None:
            print("OaF wheelhouse materialization failed: --lock and --wheelhouse are required")
            return 1
        try:
            records = materialize_locked_wheelhouse(
                lock_path=args.lock,
                destination=args.wheelhouse,
                offline_cache=args.offline_cache,
            )
        except (OSError, ResolutionError) as error:
            print(f"OaF wheelhouse materialization failed: {error}")
            return 1
        print(
            json.dumps(
                {
                    "distribution_count": len(records),
                    "lock_sha256": hashlib.sha256(Path(args.lock).read_bytes()).hexdigest(),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    if args.build_allowlisted_sdists:
        required = {
            "--sdist": args.sdist,
            "--build-tool-wheelhouse": args.build_tool_wheelhouse,
            "--build-requirements": args.build_requirements,
            "--build-lock": args.build_lock,
            "--output-wheelhouse": args.output_wheelhouse,
            "--distribution-build-manifest": args.distribution_build_manifest,
            "--runtime-requirements": args.runtime_requirements,
            "--runtime-wheelhouse": args.runtime_wheelhouse,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            print(f"OaF runtime resolution failed: missing {' '.join(missing)}")
            return 1
        try:
            records, manifest_content = build_allowlisted_sdists(
                sdist_paths=tuple(args.sdist),
                build_tool_wheelhouse=args.build_tool_wheelhouse,
                build_requirements_path=args.build_requirements,
                build_lock_path=args.build_lock,
                output_wheelhouse_path=args.output_wheelhouse,
                manifest_path=args.distribution_build_manifest,
                runtime_requirements_path=args.runtime_requirements,
                runtime_wheelhouse_path=args.runtime_wheelhouse,
                docker_executable=args.docker_executable,
            )
        except (OSError, DistributionBuildError) as error:
            print(f"OaF exceptional distribution build failed: {error}")
            return 1
        print(
            json.dumps(
                {
                    "distribution_build_manifest_sha256": hashlib.sha256(
                        manifest_content
                    ).hexdigest(),
                    "wheel_sha256": {record.spec.name: record.wheel.sha256 for record in records},
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    if args.requirements is None or args.wheelhouse is None or args.lock is None:
        print("OaF runtime resolution failed: --requirements --wheelhouse and --lock are required")
        return 1
    try:
        distributions = resolve_runtime(
            requirements_path=args.requirements,
            wheelhouse=args.wheelhouse,
            lock_path=args.lock,
            offline=args.offline,
            pip_executable=args.pip_executable,
            docker_executable=args.docker_executable,
            exceptional_wheelhouse_path=args.exceptional_wheelhouse,
            distribution_build_manifest_path=args.distribution_build_manifest,
        )
    except (OSError, ResolutionError) as error:
        print(f"OaF runtime resolution failed: {error}")
        return 1
    print(
        json.dumps(
            {
                "distribution_count": len(distributions),
                "lock_sha256": hashlib.sha256(render_requirements_lock(distributions)).hexdigest(),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
