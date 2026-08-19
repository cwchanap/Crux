"""Inventory one isolated separator interpreter using only the standard library."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import stat
import sys
import sysconfig
from pathlib import Path, PurePosixPath, PureWindowsPath

_SCHEMA = "crux.separator-environment/v1"
_ROOT_TAGS = (
    "stdlib",
    "platstdlib",
    "purelib",
    "platlib",
    "include",
    "platinclude",
    "scripts",
    "data",
)
_ROOT_TAG_PRIORITY = (
    "stdlib",
    "purelib",
    "include",
    "scripts",
    "data",
    "platstdlib",
    "platlib",
    "platinclude",
)
_SEPARATOR_TARGETS = {
    "spleeter": ("spleeter4-drums-v1", "spleeter"),
    "demucs": ("htdemucs-drums-v1", "demucs"),
}
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
_DIRECTORY_FLAGS = _READ_FLAGS | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CHUNK_SIZE = 1024 * 1024


class _ProbeError(RuntimeError):
    """An environment cannot be represented by the fixed portable manifest."""


def canonical_json_bytes(value: object, *, trailing_newline: bool = False) -> bytes:
    try:
        content = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise _ProbeError("manifest JSON is not canonical") from error
    return content + (b"\n" if trailing_newline else b"")


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_regular(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise _ProbeError("expected a regular file")


def _require_directory(metadata: os.stat_result) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise _ProbeError("expected a directory")


def _read_hash_fd(
    file_descriptor: int, *, capture: bool = False
) -> tuple[bytes, int, str, tuple[int, ...]]:
    before = os.fstat(file_descriptor)
    _require_regular(before)
    digest = hashlib.sha256()
    captured: list[bytes] = []
    byte_length = 0
    while True:
        chunk = os.read(file_descriptor, _CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        byte_length += len(chunk)
        if capture:
            captured.append(chunk)
    after = os.fstat(file_descriptor)
    if _identity(before) != _identity(after):
        raise _ProbeError("file identity changed while reading")
    return b"".join(captured), byte_length, digest.hexdigest(), _identity(before)


def _open_root(path: Path) -> tuple[int, tuple[int, ...]]:
    try:
        file_descriptor = os.open(os.fspath(path), _DIRECTORY_FLAGS | _NOFOLLOW)
        metadata = os.fstat(file_descriptor)
        _require_directory(metadata)
    except (OSError, _ProbeError) as error:
        try:
            os.close(file_descriptor)
        except (UnboundLocalError, OSError):
            pass
        raise _ProbeError("installation root is unavailable") from error
    return file_descriptor, _identity(metadata)


def _open_relative(root_descriptor: int, parts: tuple[str, ...]) -> int:
    if not parts:
        raise _ProbeError("relative file path is empty")
    current = os.dup(root_descriptor)
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(
                part,
                _DIRECTORY_FLAGS | _NOFOLLOW,
                dir_fd=current,
            )
            try:
                _require_directory(os.fstat(next_descriptor))
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(current)
            current = next_descriptor
        return os.open(parts[-1], _READ_FLAGS | _NOFOLLOW, dir_fd=current)
    except (OSError, _ProbeError) as error:
        raise _ProbeError("relative file is unavailable") from error
    finally:
        os.close(current)


def _validate_relative_path(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise _ProbeError("record path is invalid")
    if (
        value.startswith(("/", "\\"))
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or "\\" in value
    ):
        raise _ProbeError("record path is not relative")
    parts = tuple(value.split("/"))
    if any(not part or part in (".", "..") for part in parts):
        raise _ProbeError("record path is not normalized")
    return parts


def _relative_name(parts: tuple[str, ...]) -> str:
    return "/".join(parts)


def _is_bytecode(parts: tuple[str, ...]) -> bool:
    return "__pycache__" in parts or parts[-1].endswith(".pyc")


def _normalize_distribution_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _ProbeError("distribution name is invalid")
    normalized: list[str] = []
    pending_separator = False
    for character in value:
        if character in "-_.":
            pending_separator = True
            continue
        if "A" <= character <= "Z":
            character = character.lower()
        if not ("a" <= character <= "z" or "0" <= character <= "9"):
            raise _ProbeError("distribution name is invalid")
        if pending_separator:
            if not normalized:
                raise _ProbeError("distribution name is invalid")
            normalized.append("-")
            pending_separator = False
        normalized.append(character)
    if pending_separator or not normalized or normalized[-1] == "-":
        raise _ProbeError("distribution name is invalid")
    return "".join(normalized)


def _resolved_virtual_environment(interpreter: Path) -> Path | None:
    candidate = interpreter.parent.parent
    configuration = candidate / "pyvenv.cfg"
    try:
        metadata = os.stat(os.fspath(configuration), follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        return None
    return candidate


def _configured_paths(interpreter: Path) -> dict[str, str]:
    try:
        configured = dict(sysconfig.get_paths())
    except (KeyError, OSError, TypeError) as error:
        raise _ProbeError("installation roots are unavailable") from error
    virtual_environment = _resolved_virtual_environment(interpreter)
    if virtual_environment is not None:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        if os.name == "nt":
            library = virtual_environment / "Lib"
            scripts = virtual_environment / "Scripts"
        else:
            library = virtual_environment / "lib" / version
            scripts = virtual_environment / "bin"
        configured.update(
            {
                "platstdlib": os.fspath(library),
                "purelib": os.fspath(library / "site-packages"),
                "platlib": os.fspath(library / "site-packages"),
                "scripts": os.fspath(scripts),
                "data": os.fspath(virtual_environment),
            }
        )
    return configured


def _canonical_root_paths(interpreter: Path) -> dict[str, Path]:
    configured = _configured_paths(interpreter)
    roots: dict[str, Path] = {}
    seen_paths: set[str] = set()
    for tag in _ROOT_TAG_PRIORITY:
        value = configured.get(tag)
        if not isinstance(value, str) or not value:
            raise _ProbeError("installation root is missing")
        try:
            path = Path(value).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise _ProbeError("installation root is unavailable") from error
        if not path.is_absolute():
            raise _ProbeError("installation root is not absolute")
        path_key = os.path.normcase(os.fspath(path))
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        roots[tag] = path
    if not roots:
        raise _ProbeError("installation roots are empty")
    for tag in _ROOT_TAGS:
        if tag not in configured:
            raise _ProbeError("installation root tag is missing")
    return roots


def _root_for_path(path: Path, roots: dict[str, Path]) -> tuple[str, str, tuple[str, ...]]:
    matches: list[tuple[int, str, Path]] = []
    for tag, root in roots.items():
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        matches.append((len(root.parts), tag, relative))
    if not matches:
        raise _ProbeError("record file is outside installation roots")
    deepest = max(item[0] for item in matches)
    deepest_matches = [item for item in matches if item[0] == deepest]
    if len(deepest_matches) != 1:
        raise _ProbeError("record file has an ambiguous installation root")
    _, tag, relative = deepest_matches[0]
    parts = _validate_relative_path(relative.as_posix())
    return tag, _relative_name(parts), parts


def _hash_relative_file(
    root_descriptor: int,
    parts: tuple[str, ...],
    *,
    capture: bool = False,
) -> tuple[bytes, int, str, tuple[int, ...]]:
    try:
        file_descriptor = _open_relative(root_descriptor, parts)
    except (OSError, _ProbeError) as error:
        raise _ProbeError("record file is unavailable") from error
    try:
        return _read_hash_fd(file_descriptor, capture=capture)
    finally:
        os.close(file_descriptor)


def _hash_absolute_file(path: Path) -> str:
    try:
        file_descriptor = os.open(os.fspath(path), _READ_FLAGS | _NOFOLLOW)
    except OSError as error:
        raise _ProbeError("interpreter is unavailable") from error
    try:
        _, _, digest, _ = _read_hash_fd(file_descriptor)
        return digest
    except (OSError, _ProbeError) as error:
        raise _ProbeError("interpreter is unavailable") from error
    finally:
        os.close(file_descriptor)


def _parse_record(content: bytes) -> list[str]:
    try:
        text = content.decode("utf-8", errors="strict")
        rows = list(csv.reader(text.splitlines(), strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise _ProbeError("distribution RECORD is malformed") from error
    if not rows:
        raise _ProbeError("distribution RECORD is empty")
    parsed: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise _ProbeError("distribution RECORD row is malformed")
        path, _, _ = row
        parts = _validate_relative_path(path)
        normalized = _relative_name(parts)
        if normalized in seen:
            raise _ProbeError("distribution RECORD contains duplicate paths")
        seen.add(normalized)
        parsed.append(normalized)
    return parsed


def _record_path_for_distribution(
    distribution: importlib.metadata.Distribution,
) -> tuple[Path, Path]:
    try:
        distribution_path = getattr(distribution, "_path", None)
        if isinstance(distribution_path, Path):
            record_path = distribution_path / "RECORD"
        else:
            candidates = [
                file
                for file in (distribution.files or ())
                if file.as_posix().endswith(".dist-info/RECORD")
            ]
            if len(candidates) != 1:
                raise _ProbeError("distribution RECORD is unavailable")
            record_path = Path(distribution.locate_file(candidates[0]))
    except (OSError, TypeError, ValueError) as error:
        raise _ProbeError("distribution RECORD is unavailable") from error
    if record_path.parent.name == "" or not record_path.parent.name.endswith(".dist-info"):
        raise _ProbeError("distribution RECORD is not dist-info metadata")
    return record_path, record_path.parent.parent


def _inventory_distribution(
    distribution: importlib.metadata.Distribution,
    roots: dict[str, Path],
    root_descriptors: dict[str, int],
) -> tuple[dict[str, object], set[tuple[str, str]]]:
    try:
        name = _normalize_distribution_name(distribution.metadata.get("Name", distribution.name))
        version = distribution.version
    except (AttributeError, TypeError, ValueError, _ProbeError) as error:
        raise _ProbeError("distribution metadata is malformed") from error
    if not isinstance(version, str) or not version:
        raise _ProbeError("distribution version is malformed")

    record_path, distribution_root = _record_path_for_distribution(distribution)
    record_tag, record_name, record_parts = _root_for_path(record_path, roots)
    record_bytes, record_length, record_sha256, record_identity = _hash_relative_file(
        root_descriptors[record_tag],
        record_parts,
        capture=True,
    )
    rows = _parse_record(record_bytes)
    files: dict[tuple[str, str], dict[str, object]] = {}
    expected: set[tuple[str, str]] = set()
    for relative_name in rows:
        parts = _validate_relative_path(relative_name)
        if _is_bytecode(parts):
            continue
        file_path = distribution_root.joinpath(*parts)
        root_tag, portable_path, portable_parts = _root_for_path(file_path, roots)
        _, byte_length, sha256, _ = _hash_relative_file(
            root_descriptors[root_tag],
            portable_parts,
        )
        key = (root_tag, portable_path)
        if key in files:
            raise _ProbeError("distribution contains duplicate files")
        files[key] = {
            "root": root_tag,
            "path": portable_path,
            "byte_length": byte_length,
            "sha256": sha256,
        }
        expected.add(key)

    record_key = (record_tag, record_name)
    if record_key not in files:
        files[record_key] = {
            "root": record_tag,
            "path": record_name,
            "byte_length": record_length,
            "sha256": record_sha256,
        }
        expected.add(record_key)
    elif (
        files[record_key]["sha256"] != record_sha256
        or files[record_key]["byte_length"] != record_length
    ):
        raise _ProbeError("distribution RECORD changed while reading")
    del record_identity

    return (
        {
            "name": name,
            "version": version,
            "files": [files[key] for key in sorted(files)],
        },
        expected,
    )


def _same_expected_identity(actual: os.stat_result, expected: tuple[int, ...]) -> bool:
    return _identity(actual) == expected


def _walk_tree(
    root_descriptor: int,
    root_identity: tuple[int, ...],
    root_tag: str,
    expected: set[tuple[str, str]],
    expected_identities: dict[tuple[str, str], tuple[int, ...]],
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, str]]:
    try:
        before = os.fstat(root_descriptor)
        _require_directory(before)
        if _identity(before) != root_identity:
            raise _ProbeError("installation tree directory changed")
        observed: set[tuple[str, str]] = set()
        with os.scandir(root_descriptor) as entries:
            for entry in entries:
                name = entry.name
                if (
                    not isinstance(name, str)
                    or not name
                    or name in (".", "..")
                    or "/" in name
                    or "\\" in name
                ):
                    raise _ProbeError("installation tree path is invalid")
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise _ProbeError("installation tree contains a symlink")
                child_parts = prefix + (name,)
                if stat.S_ISDIR(metadata.st_mode):
                    if name == "__pycache__":
                        continue
                    child_descriptor = os.open(
                        name,
                        _DIRECTORY_FLAGS | _NOFOLLOW,
                        dir_fd=root_descriptor,
                    )
                    try:
                        observed.update(
                            _walk_tree(
                                child_descriptor,
                                _identity(metadata),
                                root_tag,
                                expected,
                                expected_identities,
                                child_parts,
                            )
                        )
                    finally:
                        os.close(child_descriptor)
                elif stat.S_ISREG(metadata.st_mode):
                    portable_path = _relative_name(child_parts)
                    key = (root_tag, portable_path)
                    if name.endswith(".pyc"):
                        continue
                    if key not in expected:
                        raise _ProbeError("installation tree contains an unexpected file")
                    if key in expected_identities and not _same_expected_identity(
                        metadata,
                        expected_identities[key],
                    ):
                        raise _ProbeError("installation tree file changed")
                    observed.add(key)
                else:
                    raise _ProbeError("installation tree contains a special file")
        after = os.fstat(root_descriptor)
        if _identity(before) != _identity(after):
            raise _ProbeError("installation tree changed while walking")
        return observed
    except (OSError, _ProbeError) as error:
        if isinstance(error, _ProbeError):
            raise
        raise _ProbeError("installation tree is unavailable") from error


def _python_version() -> str:
    version = sys.version_info
    return f"{version.major}.{version.minor}.{version.micro}"


def _python_implementation() -> str:
    name = getattr(sys.implementation, "name", "")
    if name == "cpython":
        return "CPython"
    if name == "pypy":
        return "PyPy"
    if not isinstance(name, str) or not name:
        raise _ProbeError("Python implementation is unavailable")
    return name


def _python_abi() -> str:
    value = sysconfig.get_config_var("SOABI")
    if not isinstance(value, str) or not value:
        value = getattr(sys.implementation, "cache_tag", None)
    if not isinstance(value, str) or not value:
        raise _ProbeError("Python ABI is unavailable")
    return value


def _distribution_search_paths(interpreter: Path, roots: dict[str, Path]) -> list[str] | None:
    if _resolved_virtual_environment(interpreter) is None:
        return None
    return [os.fspath(path) for tag, path in roots.items() if tag in ("purelib", "platlib")]


def build_environment_manifest() -> dict[str, object]:
    interpreter = Path(sys.executable).resolve(strict=True)
    interpreter_sha256 = _hash_absolute_file(interpreter)
    roots = _canonical_root_paths(interpreter)
    root_descriptors: dict[str, int] = {}
    root_identities: dict[str, tuple[int, ...]] = {}
    try:
        for tag, path in roots.items():
            descriptor, identity = _open_root(path)
            root_descriptors[tag] = descriptor
            root_identities[tag] = identity

        search_paths = _distribution_search_paths(interpreter, roots)
        if search_paths is None:
            distributions = list(importlib.metadata.distributions())
        else:
            distributions = list(importlib.metadata.distributions(path=search_paths))
        inventory: list[tuple[str, dict[str, object]]] = []
        expected_by_root: dict[str, set[tuple[str, str]]] = {}
        for distribution in distributions:
            item, expected = _inventory_distribution(distribution, roots, root_descriptors)
            name = item["name"]
            if not isinstance(name, str):
                raise _ProbeError("distribution name is malformed")
            if any(existing_name == name for existing_name, _ in inventory):
                raise _ProbeError("duplicate normalized distribution name")
            inventory.append((name, item))
            for key in expected:
                expected_by_root.setdefault(key[0], set()).add(key)

        if not inventory:
            raise _ProbeError("no installed separator distribution")
        inventory.sort(key=lambda item: item[0])
        targets = [item for item in inventory if item[0] in _SEPARATOR_TARGETS]
        if len(targets) != 1:
            raise _ProbeError("separator distribution is ambiguous")
        target_name, target_distribution = targets[0]
        separator_id, package_name = _SEPARATOR_TARGETS[target_name]
        package_version = target_distribution["version"]
        if not isinstance(package_version, str) or not package_version:
            raise _ProbeError("separator package version is malformed")

        expected_identities: dict[tuple[str, str], tuple[int, ...]] = {}
        for item in inventory:
            for file in item[1]["files"]:
                if not isinstance(file, dict):
                    raise _ProbeError("distribution inventory is malformed")
                key = (file["root"], file["path"])
                expected_identities[key] = expected_identities.get(key, ())
        for tag, expected in expected_by_root.items():
            for key in expected:
                root_tag, portable_path = key
                parts = _validate_relative_path(portable_path)
                _, _, _, identity = _hash_relative_file(
                    root_descriptors[root_tag],
                    parts,
                )
                expected_identities[(root_tag, portable_path)] = identity
            observed = _walk_tree(
                root_descriptors[tag],
                root_identities[tag],
                tag,
                expected,
                expected_identities,
            )
            if observed != expected:
                raise _ProbeError("installation tree membership changed")

        return {
            "schema": _SCHEMA,
            "separator_id": separator_id,
            "package_name": package_name,
            "package_version": package_version,
            "python_implementation": _python_implementation(),
            "python_version": _python_version(),
            "python_abi": _python_abi(),
            "platform": sys.platform,
            "interpreter_sha256": interpreter_sha256,
            "distributions": [item[1] for item in inventory],
        }
    finally:
        for descriptor in root_descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass


def main() -> int:
    try:
        payload = build_environment_manifest()
        sys.stdout.buffer.write(canonical_json_bytes(payload, trailing_newline=True))
        return 0
    except Exception:
        sys.stderr.write("separator_environment_probe_failed\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
