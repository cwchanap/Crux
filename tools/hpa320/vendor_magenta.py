#!/usr/bin/env python3
"""Vendor the reviewed Magenta OaF source closure without executing it."""

# CLI serialization and subprocess capture intentionally mirror the sibling generators.
# pylint: disable=duplicate-code
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
from collections import deque
from pathlib import Path, PurePosixPath

from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.backend_publication import atomic_replace_bytes

EXPECTED_UPSTREAM_COMMIT = "94529798dfbbb14c27ddfd76f23027dc8e2ce185"
EXPECTED_UPSTREAM_REPOSITORY = "https://github.com/magenta/magenta.git"
SOURCE_MANIFEST_SCHEMA = "crux.oaf-upstream-source-manifest/v1"
SOURCE_LICENSE = "Apache-2.0"
_ENTRYPOINT_ROOT = PurePosixPath("magenta/models/onsets_frames_transcription")
_ALLOWED_ROOTS = (
    _ENTRYPOINT_ROOT,
    PurePosixPath("magenta/common"),
    PurePosixPath("magenta/music"),
    PurePosixPath("magenta/pipelines"),
)
_ALLOWED_FILES = frozenset(
    {
        PurePosixPath("LICENSE"),
        PurePosixPath("NOTICE"),
        PurePosixPath("magenta/__init__.py"),
        PurePosixPath("magenta/models/__init__.py"),
        PurePosixPath("magenta/version.py"),
    }
)
_READ_CHUNK_BYTES = 1024 * 1024


class VendoringError(ValueError):
    """The source checkout or requested publication is not safe to vendor."""


def vendor_magenta(
    *,
    source: Path,
    destination: Path,
    manifest_path: Path,
    expected_commit: str = EXPECTED_UPSTREAM_COMMIT,
    expected_repository: str = EXPECTED_UPSTREAM_REPOSITORY,
) -> dict[str, object]:
    """Validate, stage, and publish the frozen source subset and manifest."""

    source_root = _verify_checkout(Path(source), expected_commit, expected_repository)
    selected = _discover_source_closure(source_root)
    rows: list[dict[str, str]] = []
    staged_bytes: dict[PurePosixPath, bytes] = {}
    for relative_path in sorted(selected, key=lambda item: item.as_posix().encode("utf-8")):
        content = _read_head_verified_file(source_root, relative_path, expected_commit)
        staged_bytes[relative_path] = content
        rows.append(
            {
                "license": SOURCE_LICENSE,
                "path": relative_path.as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )

    payload: dict[str, object] = {
        "covered_roots": [root.as_posix() for root in _ALLOWED_ROOTS],
        "files": rows,
        "schema": SOURCE_MANIFEST_SCHEMA,
        "upstream_commit": expected_commit,
        "upstream_repository": expected_repository,
    }
    if _verify_checkout(source_root, expected_commit, expected_repository) != source_root:
        raise VendoringError("source checkout identity changed during vendoring")
    manifest_bytes = canonical_json_bytes(payload, trailing_newline=True)
    _publish_vendor_tree(
        Path(destination),
        Path(manifest_path),
        staged_bytes,
        manifest_bytes,
    )
    return payload


def _verify_checkout(source: Path, expected_commit: str, expected_repository: str) -> Path:
    if not isinstance(expected_commit, str) or len(expected_commit) != 40:
        raise VendoringError("expected source commit must be a full Git object ID")
    try:
        int(expected_commit, 16)
    except ValueError:
        raise VendoringError("expected source commit must be lowercase hexadecimal") from None
    try:
        source_root = source.resolve(strict=True)
    except OSError:
        raise VendoringError("source checkout is unavailable") from None
    if not source_root.is_dir():
        raise VendoringError("source checkout must be a directory")

    top_level = _git(source_root, "rev-parse", "--show-toplevel", error_label="commit")
    try:
        top_level_path = Path(top_level).resolve(strict=True)
    except OSError:
        raise VendoringError("source checkout root is invalid") from None
    if top_level_path != source_root:
        raise VendoringError("source must name the exact Git worktree root")
    head = _git(source_root, "rev-parse", "HEAD", error_label="commit")
    if head != expected_commit:
        raise VendoringError("source commit does not match the frozen commit")
    repository = _git(source_root, "remote", "get-url", "origin", error_label="repository")
    if repository != expected_repository:
        raise VendoringError("source repository does not match the frozen repository")
    if _git(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        error_label="dirty",
    ):
        raise VendoringError("source checkout is dirty")
    submodules = _git(
        source_root,
        "submodule",
        "status",
        "--recursive",
        error_label="submodule",
    )
    for line in submodules.splitlines():
        if line and line[0] in "-+U":
            raise VendoringError("source checkout has mismatched submodules")
    return source_root


def _git(source: Path, *args: str, error_label: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", os.fspath(source), *args),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, UnicodeError):
        raise VendoringError(f"source {error_label} metadata is unavailable") from None
    if result.returncode != 0:
        raise VendoringError(f"source {error_label} metadata is unavailable")
    return result.stdout.rstrip("\n")


def _git_bytes(source: Path, *args: str, error_label: str) -> bytes:
    try:
        result = subprocess.run(
            ("git", "-C", os.fspath(source), *args),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        raise VendoringError(f"source {error_label} metadata is unavailable") from None
    if result.returncode != 0:
        raise VendoringError(f"source {error_label} metadata is unavailable")
    return result.stdout


def _read_head_verified_file(  # pylint: disable=too-many-locals
    source_root: Path,
    relative_path: PurePosixPath,
    expected_commit: str,
) -> bytes:
    """Read one selected file and prove worktree, index, and HEAD-tree identity."""

    relative = _checked_relative_path(relative_path.as_posix()).as_posix()
    encoded_relative = relative.encode("utf-8")
    flags = _git_bytes(
        source_root,
        "ls-files",
        "-v",
        "-z",
        "--",
        relative,
        error_label="index",
    )
    if flags != b"H " + encoded_relative + b"\0":
        raise VendoringError(f"selected source has an unsupported index flag: {relative}")

    tree = _git_bytes(
        source_root,
        "ls-tree",
        "-z",
        expected_commit,
        "--",
        relative,
        error_label="tree",
    )
    index = _git_bytes(
        source_root,
        "ls-files",
        "--stage",
        "-z",
        "--",
        relative,
        error_label="index",
    )
    try:
        tree_header, tree_path = tree.removesuffix(b"\0").split(b"\t", 1)
        tree_mode, object_type, tree_object = tree_header.split(b" ", 2)
        index_header, index_path = index.removesuffix(b"\0").split(b"\t", 1)
        index_mode, index_object, index_stage = index_header.split(b" ", 2)
    except ValueError:
        raise VendoringError(f"selected source is absent from the HEAD tree: {relative}") from None
    if (  # pylint: disable=too-many-boolean-expressions
        tree_path != encoded_relative
        or index_path != encoded_relative
        or object_type != b"blob"
        or tree_mode not in {b"100644", b"100755"}
        or index_mode != tree_mode
        or index_object != tree_object
        or index_stage != b"0"
    ):
        raise VendoringError(f"selected source index does not match the HEAD tree: {relative}")
    head_content = _git_bytes(
        source_root,
        "cat-file",
        "blob",
        tree_object.decode("ascii"),
        error_label="tree",
    )
    worktree_content = _read_stable_regular_file(source_root, relative_path)
    if worktree_content != head_content:
        raise VendoringError(f"selected source bytes do not match the HEAD tree: {relative}")
    return worktree_content


def _discover_source_closure(source_root: Path) -> set[PurePosixPath]:
    selected = _walk_reviewed_root(source_root, _ENTRYPOINT_ROOT)
    for required in _ALLOWED_FILES:
        candidate = source_root / required.as_posix()
        if candidate.exists():
            selected.add(required)
    queue = deque(path for path in selected if path.suffix == ".py")
    parsed: set[PurePosixPath] = set()
    while queue:
        relative_path = queue.popleft()
        if relative_path in parsed:
            continue
        parsed.add(relative_path)
        content = _read_stable_regular_file(source_root, relative_path)
        try:
            tree = ast.parse(content, filename=relative_path.as_posix())
        except (SyntaxError, ValueError):
            raise VendoringError(
                f"could not parse vendored import source: {relative_path}"
            ) from None
        for module_name in _magenta_imports(tree, relative_path):
            for dependency in _resolve_module_files(source_root, module_name):
                if not _is_allowed(dependency):
                    raise VendoringError(
                        f"import closure leaves the reviewed allowlist: {dependency}"
                    )
                if dependency not in selected:
                    selected.add(dependency)
                    if dependency.suffix == ".py":
                        queue.append(dependency)
    _validate_selected_paths(source_root, selected)
    return selected


def _walk_reviewed_root(source_root: Path, relative_root: PurePosixPath) -> set[PurePosixPath]:
    absolute_root = source_root / relative_root.as_posix()
    try:
        root_metadata = os.lstat(absolute_root)
    except OSError:
        raise VendoringError(f"required source root is missing: {relative_root}") from None
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise VendoringError(f"required source root is not a regular directory: {relative_root}")
    selected: set[PurePosixPath] = set()
    for current_root, directory_names, file_names in os.walk(
        absolute_root,
        topdown=True,
        followlinks=False,
    ):
        directory_names.sort(key=lambda value: value.encode("utf-8"))
        file_names.sort(key=lambda value: value.encode("utf-8"))
        current = Path(current_root)
        for directory_name in directory_names:
            metadata = os.lstat(current / directory_name)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise VendoringError("source tree contains a symlink or special directory")
        for file_name in file_names:
            absolute = current / file_name
            metadata = os.lstat(absolute)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise VendoringError("source tree contains a symlink or special file")
            selected.add(_relative_posix(source_root, absolute))
    if not selected:
        raise VendoringError(f"required source root is empty: {relative_root}")
    return selected


def _magenta_imports(tree: ast.AST, source_path: PurePosixPath) -> set[str]:
    modules: set[str] = set()
    package = ".".join(source_path.with_suffix("").parts[:-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names if alias.name.startswith("magenta"))
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_import_module(package, node.module, node.level)
            if not module.startswith("magenta"):
                continue
            modules.add(module)
            for alias in node.names:
                if alias.name != "*":
                    modules.add(f"{module}.{alias.name}")
    return modules


def _absolute_import_module(package: str, module: str | None, level: int) -> str:
    if level == 0:
        return module or ""
    package_parts = package.split(".")
    if level > len(package_parts):
        return ""
    prefix = package_parts[: len(package_parts) - level + 1]
    if module:
        prefix.extend(module.split("."))
    return ".".join(prefix)


def _resolve_module_files(source_root: Path, module_name: str) -> set[PurePosixPath]:
    parts = module_name.split(".")
    if not parts or parts[0] != "magenta" or any(not part for part in parts):
        return set()
    selected: set[PurePosixPath] = set()
    for index in range(1, len(parts) + 1):
        initializer = PurePosixPath(*parts[:index], "__init__.py")
        if (source_root / initializer.as_posix()).is_file():
            selected.add(initializer)
    module_file = PurePosixPath(*parts).with_suffix(".py")
    package_file = PurePosixPath(*parts, "__init__.py")
    if (source_root / module_file.as_posix()).is_file():
        selected.add(module_file)
    elif (source_root / package_file.as_posix()).is_file():
        selected.add(package_file)
    return selected


def _validate_selected_paths(source_root: Path, selected: set[PurePosixPath]) -> None:
    seen: set[str] = set()
    for relative_path in selected:
        normalized = unicodedata.normalize("NFC", relative_path.as_posix())
        if normalized != relative_path.as_posix():
            raise VendoringError("source path is not Unicode NFC")
        checked = _checked_relative_path(normalized)
        if checked.as_posix() in seen:
            raise VendoringError("source selection contains duplicate paths")
        seen.add(checked.as_posix())
        _read_stable_regular_file(source_root, checked)


def _is_allowed(path: PurePosixPath) -> bool:
    return path in _ALLOWED_FILES or any(
        path == root or root in path.parents for root in _ALLOWED_ROOTS
    )


def _relative_posix(root: Path, path: Path) -> PurePosixPath:
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise VendoringError("source path escapes the checkout") from None
    return _checked_relative_path(relative.as_posix())


def _checked_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise VendoringError("source path is not a canonical POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise VendoringError("source path escapes or aliases the checkout")
    if path.as_posix() != value:
        raise VendoringError("source path is not canonical")
    return path


def _read_stable_regular_file(root: Path, relative_path: PurePosixPath) -> bytes:
    relative_path = _checked_relative_path(relative_path.as_posix())
    descriptors: list[int] = []
    try:
        root_metadata = os.lstat(root)
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
            raise VendoringError("source root is not a no-follow directory")
        directory_descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptors.append(directory_descriptor)
        for component in relative_path.parts[:-1]:
            directory_descriptor = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            descriptors.append(directory_descriptor)
        file_descriptor = os.open(
            relative_path.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VendoringError("source path is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
        if (
            (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise VendoringError("source bytes changed while being read")
        content = b"".join(chunks)
        if len(content) != before.st_size:
            raise VendoringError("source bytes changed while being read")
        return content
    except OSError:
        raise VendoringError(f"source file is unreadable: {relative_path}") from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _publish_vendor_tree(
    destination: Path,
    manifest_path: Path,
    files: dict[PurePosixPath, bytes],
    manifest_bytes: bytes,
) -> None:
    if destination.exists() or destination.is_symlink():
        _verify_existing_publication(destination, manifest_path, files, manifest_bytes)
        return
    if manifest_path.exists() or manifest_path.is_symlink():
        raise VendoringError("existing manifest has no matching vendored destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    published_destination = False
    try:
        os.chmod(staging_path, 0o700)
        for relative_path, content in files.items():
            output = staging_path / relative_path.as_posix()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        _verify_staged_tree(staging_path, files)
        os.replace(staging_path, destination)
        published_destination = True
        atomic_replace_bytes(manifest_path, manifest_bytes)
    except (OSError, ValueError) as error:
        if published_destination:
            shutil.rmtree(destination, ignore_errors=True)
        raise VendoringError("could not publish vendored source atomically") from error
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path, ignore_errors=True)


def _verify_existing_publication(
    destination: Path,
    manifest_path: Path,
    files: dict[PurePosixPath, bytes],
    manifest_bytes: bytes,
) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise VendoringError("existing vendor destination is not a regular directory")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise VendoringError("existing source manifest is missing or unsafe")
    try:
        if manifest_path.read_bytes() != manifest_bytes:
            raise VendoringError("existing source manifest is stale")
    except OSError:
        raise VendoringError("existing source manifest is unreadable") from None
    _verify_staged_tree(destination, files)


def _verify_staged_tree(root: Path, files: dict[PurePosixPath, bytes]) -> None:
    actual: set[PurePosixPath] = set()
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        current = Path(current_root)
        for directory_name in directory_names:
            metadata = os.lstat(current / directory_name)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise VendoringError("vendored destination contains a symlink or special directory")
        for file_name in file_names:
            absolute = current / file_name
            relative = _relative_posix(root, absolute)
            actual.add(relative)
            if relative not in files:
                raise VendoringError("vendored destination contains a stale unlisted file")
            if _read_stable_regular_file(root, relative) != files[relative]:
                raise VendoringError("vendored destination bytes do not match the manifest")
    if actual != set(files):
        raise VendoringError("vendored destination is missing listed files")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = vendor_magenta(
            source=args.source,
            destination=args.destination,
            manifest_path=args.manifest,
        )
    except (OSError, VendoringError) as error:
        print(f"Magenta vendoring failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "file_count": len(payload["files"]),
                "manifest_sha256": hashlib.sha256(
                    canonical_json_bytes(payload, trailing_newline=True)
                ).hexdigest(),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
