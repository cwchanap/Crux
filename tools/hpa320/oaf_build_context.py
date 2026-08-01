#!/usr/bin/env python3
"""Generate, validate, and materialize the exact OaF image build context."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from src.benchmark.backend_identity import (
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    strict_json_loads,
)
from src.benchmark.backend_publication import read_regular_file_no_follow
from tools.hpa320._fsync import _fsync_directory

BUILD_CONTEXT_SCHEMA = "crux.oaf-build-context-manifest/v1"
BUILD_CONTEXT_MANIFEST_PATH = "runtime/oaf_tf1/build-context-manifest.json"
DIRECTORY_MODE = 0o755
FILE_MODE = 0o644
WHEELHOUSE_PREFIX = "runtime/oaf_tf1/wheelhouse/"

# These are reviewed Dockerfile inputs. Task-specific entrypoints are added here when
# their COPY instructions land; the repository root is never supplied as the context.
REVIEWED_REPOSITORY_PATHS = (
    ".github/workflows/hpa320-native-bootstrap.yml",
    "tools/hpa320/audit_legacy_tf2_conversion.py",
    "tools/hpa320/generate_runner_source_manifest.py",
    "tools/hpa320/github_host_evidence.py",
    "tools/hpa320/oaf_build_context.py",
    "tools/hpa320/oaf_candidate_builder.py",
    "tools/hpa320/oaf_host_attestation.py",
    "tools/hpa320/oaf_native_calibration.py",
    "tools/hpa320/oaf_native_artifacts.py",
    "tools/hpa320/oaf_native_runner.py",
    "tools/hpa320/oaf_oci.py",
    "tools/hpa320/oaf_system_packages.py",
    "tools/hpa320/seal_oaf_backend.py",
)
REVIEWED_REPOSITORY_ROOTS = ("runtime/oaf_tf1",)
REVIEWED_WHEELHOUSE_ROOTS = ("runtime", "test")

_MANIFEST_KEYS = frozenset({"directory_mode", "file_mode", "files", "manifest_path", "schema"})
_SOURCE_FILE_KEYS = frozenset({"byte_length", "path", "sha256"})


class BuildContextError(ValueError):
    """The reviewed build context is unsafe, incomplete, or byte-inconsistent."""


@dataclass(frozen=True)
class SourceFile:
    byte_length: int
    path: str
    sha256: str


@dataclass(frozen=True)
class BuildContextManifest:
    directory_mode: int
    file_mode: int
    files: tuple[SourceFile, ...]
    manifest_path: str
    sha256: str


def load_build_context_manifest(path: Path) -> BuildContextManifest:
    """Strict-load one canonical self-excluding build-context manifest."""

    content = _read_regular(Path(path), "build-context manifest")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise BuildContextError("build-context manifest must have one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
    except StrictJsonError as error:
        raise BuildContextError(f"build-context manifest JSON is invalid: {error}") from None
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise BuildContextError("build-context manifest fields are invalid")
    if (
        value["schema"] != BUILD_CONTEXT_SCHEMA
        or value["directory_mode"] != DIRECTORY_MODE
        or type(value["directory_mode"]) is not int
        or value["file_mode"] != FILE_MODE
        or type(value["file_mode"]) is not int
        or value["manifest_path"] != BUILD_CONTEXT_MANIFEST_PATH
    ):
        raise BuildContextError("build-context manifest authority is invalid")
    rows = value["files"]
    if not isinstance(rows, list) or not rows:
        raise BuildContextError("build-context manifest files are invalid")
    parsed: list[SourceFile] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != _SOURCE_FILE_KEYS:
            raise BuildContextError("build-context source file fields are invalid")
        relative = row["path"]
        size = row["byte_length"]
        digest = row["sha256"]
        if (
            not _safe_relative_path(relative)
            or relative == BUILD_CONTEXT_MANIFEST_PATH
            or _is_ignore_file(relative)
            or _is_generated_cache_path(relative)
        ):
            raise BuildContextError("build-context source file path is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BuildContextError("build-context source file size is invalid")
        if not _is_sha256(digest):
            raise BuildContextError("build-context source file hash is invalid")
        parsed.append(SourceFile(size, relative, digest))
    paths = [row.path for row in parsed]
    if paths != sorted(paths, key=lambda item: item.encode("utf-8")) or len(paths) != len(
        set(paths)
    ):
        raise BuildContextError("build-context source files are not uniquely sorted")
    return BuildContextManifest(
        directory_mode=DIRECTORY_MODE,
        file_mode=FILE_MODE,
        files=tuple(parsed),
        manifest_path=BUILD_CONTEXT_MANIFEST_PATH,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def generate_build_context_manifest(
    *,
    repository_root: Path,
    wheelhouse_root: Path,
) -> bytes:
    """Render canonical bytes for only the source-reviewed Dockerfile inputs."""

    repository = _require_root(Path(repository_root), "repository root")
    wheelhouse = _require_root(Path(wheelhouse_root), "wheelhouse root")
    sources: dict[str, bytes] = {}
    for relative in REVIEWED_REPOSITORY_PATHS:
        _add_source(sources, relative, repository, relative)
    for relative_root in REVIEWED_REPOSITORY_ROOTS:
        for relative in _enumerate_regular_files(repository, relative_root):
            if relative == BUILD_CONTEXT_MANIFEST_PATH or relative.startswith(WHEELHOUSE_PREFIX):
                continue
            _add_source(sources, relative, repository, relative)
    for relative_root in REVIEWED_WHEELHOUSE_ROOTS:
        for wheel_relative in _enumerate_regular_files(wheelhouse, relative_root):
            logical = WHEELHOUSE_PREFIX + wheel_relative
            _add_source(sources, logical, wheelhouse, wheel_relative)
    if not sources:
        raise BuildContextError("reviewed build context is empty")
    files: list[JsonValue] = []
    for relative in sorted(sources, key=lambda item: item.encode("utf-8")):
        content = sources[relative]
        files.append(
            {
                "byte_length": len(content),
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    payload: JsonValue = {
        "directory_mode": DIRECTORY_MODE,
        "file_mode": FILE_MODE,
        "files": files,
        "manifest_path": BUILD_CONTEXT_MANIFEST_PATH,
        "schema": BUILD_CONTEXT_SCHEMA,
    }
    return canonical_json_bytes(payload, trailing_newline=True)


def materialize_build_context(
    *,
    repository_root: Path,
    wheelhouse_root: Path,
    manifest_path: Path,
    destination: Path,
) -> Path:
    """Create a fresh normalized directory containing exactly the authenticated files."""

    repository = _require_root(Path(repository_root), "repository root")
    wheelhouse = _require_root(Path(wheelhouse_root), "wheelhouse root")
    manifest_source = Path(manifest_path)
    manifest = load_build_context_manifest(manifest_source)
    manifest_content = _read_source_path(manifest_source, "build-context manifest")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise BuildContextError("build-context destination must be absent")
    try:
        target.mkdir(mode=DIRECTORY_MODE, parents=False)
    except OSError as error:
        raise BuildContextError("build-context destination could not be created") from error
    expected_files = {row.path for row in manifest.files} | {manifest.manifest_path}
    try:
        for row in manifest.files:
            if row.path.startswith(WHEELHOUSE_PREFIX):
                source_root = wheelhouse
                source_relative = row.path.removeprefix(WHEELHOUSE_PREFIX)
            else:
                source_root = repository
                source_relative = row.path
            content = _read_source(source_root, source_relative)
            if len(content) != row.byte_length:
                raise BuildContextError(f"build-context source size drift: {row.path}")
            if hashlib.sha256(content).hexdigest() != row.sha256:
                raise BuildContextError(f"build-context source hash drift: {row.path}")
            _write_normalized_file(target, row.path, content)
        _write_normalized_file(target, manifest.manifest_path, manifest_content)
        actual_files = {
            path.relative_to(target).as_posix()
            for path in target.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if actual_files != expected_files:
            raise BuildContextError("materialized build context has missing or extra entries")
        for directory in sorted(
            (path for path in target.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _normalize_directory(directory)
        _normalize_directory(target)
        return target
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Validate one isolated build-context manifest through the production loader."""

    if schema != BUILD_CONTEXT_SCHEMA:
        raise ValueError("build-context schema golden is unsupported")
    import tempfile

    try:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "build-context-manifest.json"
            path.write_bytes(content)
            load_build_context_manifest(path)
    except (BuildContextError, OSError) as error:
        raise ValueError(str(error)) from None


def _add_source(
    sources: dict[str, bytes],
    logical_path: str,
    source_root: Path,
    source_relative: str,
) -> None:
    if (
        not _safe_relative_path(logical_path)
        or logical_path == BUILD_CONTEXT_MANIFEST_PATH
        or _is_ignore_file(logical_path)
        or logical_path in sources
    ):
        raise BuildContextError("reviewed build-context path is invalid or duplicated")
    sources[logical_path] = _read_source(source_root, source_relative)


def _enumerate_regular_files(root: Path, relative_root: str) -> tuple[str, ...]:
    if not _safe_relative_path(relative_root):
        raise BuildContextError("reviewed build-context root is invalid")
    start = root / Path(*PurePosixPath(relative_root).parts)
    metadata = _lstat(start, "reviewed build-context root")
    if not stat.S_ISDIR(metadata.st_mode):
        raise BuildContextError("reviewed build-context root is not a directory")
    paths: list[str] = []
    for path in start.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if _is_generated_cache_path(relative):
            continue
        metadata = _lstat(path, "reviewed build-context input")
        if stat.S_ISLNK(metadata.st_mode):
            raise BuildContextError("reviewed build-context input contains a link")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise BuildContextError("reviewed build-context input is not regular")
        paths.append(relative)
    return tuple(sorted(paths, key=lambda item: item.encode("utf-8")))


def _read_source(root: Path, relative: str) -> bytes:
    if not _safe_relative_path(relative):
        raise BuildContextError("build-context source path is invalid")
    current = root
    parts = PurePosixPath(relative).parts
    for part in parts[:-1]:
        current /= part
        metadata = _lstat(current, "build-context source ancestor")
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise BuildContextError("build-context source ancestor is not a stable directory")
    path = root.joinpath(*parts)
    metadata = _lstat(path, "build-context source")
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise BuildContextError("build-context source is not a regular file")
    if metadata.st_nlink != 1:
        raise BuildContextError("build-context source is multiply linked")
    return _read_source_path(path, "build-context source")


def _read_source_path(path: Path, label: str) -> bytes:
    metadata = _lstat(path, label)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise BuildContextError(f"{label} is not a stable single-link regular file")
    try:
        return read_regular_file_no_follow(path)
    except OSError as error:
        raise BuildContextError(f"{label} is not a stable regular file") from error


def _write_normalized_file(root: Path, relative: str, content: bytes) -> None:
    path = root.joinpath(*PurePosixPath(relative).parts)
    path.parent.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            FILE_MODE,
        )
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fchmod(descriptor, FILE_MODE)
            os.utime(descriptor, ns=(0, 0))
        finally:
            os.close(descriptor)
    except OSError as error:
        raise BuildContextError("materialized build-context file could not be written") from error


def _normalize_directory(path: Path) -> None:
    try:
        path.chmod(DIRECTORY_MODE, follow_symlinks=False)
        os.utime(path, ns=(0, 0), follow_symlinks=False)
    except OSError as error:
        raise BuildContextError("materialized build-context directory is unstable") from error


def _read_regular(path: Path, label: str) -> bytes:
    return _read_source_path(path, label)


def _require_root(path: Path, label: str) -> Path:
    metadata = _lstat(path, label)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise BuildContextError(f"{label} must be a stable directory")
    return path


def _lstat(path: Path, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise BuildContextError(f"{label} is missing or unreadable") from error


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and str(path) == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _is_ignore_file(relative: str) -> bool:
    return PurePosixPath(relative).name == ".dockerignore" or relative.endswith(".dockerignore")


def _is_generated_cache_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        any(
            part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
            for part in path.parts
        )
        or path.name == ".DS_Store"
        or path.suffix in {".pyc", ".pyo"}
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_manifest_content(path: Path, content: bytes, *, replace: bool) -> None:
    output = Path(path)
    parent = _require_root(output.parent, "build-context manifest parent")
    if not replace:
        if output.exists() or output.is_symlink():
            raise BuildContextError("build-context manifest output must be absent")
        published_path: str | None = None
        try:
            descriptor = os.open(
                output,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                FILE_MODE,
            )
            published_path = os.fspath(output)
            with os.fdopen(descriptor, "wb") as handle:
                os.fchmod(handle.fileno(), FILE_MODE)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            load_build_context_manifest(output)
            published_path = None
            _fsync_directory(parent)
        except (BuildContextError, OSError):
            if published_path is not None:
                try:
                    os.unlink(published_path)
                except OSError:
                    pass
            raise
        return

    descriptor, temporary = tempfile.mkstemp(
        prefix=".build-context-manifest-",
        dir=parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), FILE_MODE)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        load_build_context_manifest(Path(temporary))
        os.replace(temporary, output)
        _fsync_directory(parent)
    except (BuildContextError, OSError):
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--repository-root", type=Path, required=True)
    generate.add_argument("--wheelhouse-root", type=Path, required=True)
    generate.add_argument("--output", type=Path)
    mode = generate.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--replace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        content = generate_build_context_manifest(
            repository_root=args.repository_root,
            wheelhouse_root=args.wheelhouse_root,
        )
        if args.output is None:
            if args.check or args.replace:
                raise BuildContextError("build-context manifest mode requires --output")
            sys.stdout.buffer.write(content)
        else:
            output = Path(args.output)
            if args.check:
                checked = _read_regular(output, "build-context manifest output")
                if checked != content:
                    raise BuildContextError(
                        "build-context manifest does not match generated canonical bytes"
                    )
            else:
                _write_manifest_content(output, content, replace=args.replace)
    except (BuildContextError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
