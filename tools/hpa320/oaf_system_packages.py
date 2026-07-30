#!/usr/bin/env python3
"""Authenticate and install one exact offline Debian package bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath

SYSTEM_PACKAGE_BUNDLE_SCHEMA = "crux.oaf-system-package-bundle/v2"
BASE_IMAGE_ARCHIVE_KEYRING = Path("/usr/share/keyrings/debian-archive-keyring.gpg")
GPGV_EMPTY_HOMEDIR = Path("/nonexistent")
MAX_COMPRESSED_INDEX_BYTES = 64 * 1024 * 1024
MAX_DECOMPRESSED_INDEX_BYTES = 256 * 1024 * 1024
_TOP_LEVEL_KEYS = frozenset(
    {
        "architecture",
        "codename",
        "expected_dpkg_inventory",
        "inrelease",
        "package_indexes",
        "packages",
        "schema",
        "signing_fingerprint",
        "snapshot_url",
    }
)
_IDENTITY_KEYS = frozenset({"byte_length", "filename", "sha256"})
_INDEX_IDENTITY_KEYS = frozenset({"byte_length", "release_path", "sha256"})
_PACKAGE_KEYS = frozenset({"architecture", "filename", "package", "sha256", "size", "version"})
_EXPECTED_ROOT_ENTRIES = frozenset(
    {
        "InRelease",
        "bundle-manifest.json",
        "expected-dpkg-inventory.txt",
        "indexes",
        "packages",
    }
)
_REQUIRED_PACKAGE_FIELDS = frozenset(
    {"Architecture", "Filename", "Package", "SHA256", "Size", "Version"}
)


class SystemPackageError(ValueError):
    """The offline Debian input bundle is invalid or cannot be installed exactly."""


def _strict_json_loads(content: bytes) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            content.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeError, ValueError, TypeError):
        raise SystemPackageError("system package manifest is invalid JSON") from None


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError):
        raise SystemPackageError("system package manifest is not canonical JSON") from None


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SystemPackageError(f"{label} must be a lowercase SHA-256")
    return value


def _require_fingerprint(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9A-F]{40,64}", value):
        raise SystemPackageError(f"{label} is invalid")
    return value


def _checked_relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SystemPackageError(f"{label} is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SystemPackageError(f"{label} contains traversal or is not canonical")
    return path


def _read_stable_file(path: Path, *, maximum_bytes: int | None = None) -> bytes:
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise SystemPackageError("system package bundle contains a symlink or special file")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise SystemPackageError("system package bundle file exceeds its size bound")
        with path.open("rb") as stream:
            content = stream.read()
        after = os.lstat(path)
    except OSError:
        raise SystemPackageError("system package bundle file is unreadable") from None

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if identity(before) != identity(after) or len(content) != before.st_size:
        raise SystemPackageError("system package bundle file changed while being read")
    return content


def _require_directory(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError:
        raise SystemPackageError(f"{label} is unavailable") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SystemPackageError(f"{label} must be a no-follow directory")


def _walk_regular_files(root: Path) -> set[PurePosixPath]:
    result: set[PurePosixPath] = set()

    def visit(directory: Path, relative: PurePosixPath) -> None:
        _require_directory(directory, "system package index directory")
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name.encode("utf-8"))
        except OSError:
            raise SystemPackageError("system package index directory is unreadable") from None
        for entry in entries:
            child = relative / entry.name
            try:
                metadata = os.lstat(entry)
            except OSError:
                raise SystemPackageError("system package index entry is unreadable") from None
            if stat.S_ISLNK(metadata.st_mode):
                raise SystemPackageError("system package index tree contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                visit(entry, child)
            elif stat.S_ISREG(metadata.st_mode):
                result.add(child)
            else:
                raise SystemPackageError("system package index tree contains a special file")

    visit(root, PurePosixPath())
    return result


def _verify_root_identity(
    bundle: Path,
    value: object,
    *,
    expected_filename: str,
) -> tuple[dict[str, object], bytes]:
    if not isinstance(value, dict) or set(value) != _IDENTITY_KEYS:
        raise SystemPackageError("system package file identity is invalid")
    if value["filename"] != expected_filename:
        raise SystemPackageError("system package file identity is invalid")
    byte_length = value["byte_length"]
    if not isinstance(byte_length, int) or isinstance(byte_length, bool) or byte_length <= 0:
        raise SystemPackageError("system package file identity is invalid")
    sha256 = _require_hash(value["sha256"], "system package file hash")
    content = _read_stable_file(bundle / expected_filename)
    if len(content) != byte_length or hashlib.sha256(content).hexdigest() != sha256:
        raise SystemPackageError("bundle file byte length or hash does not match the manifest")
    return value, content


def _verify_gpgv(
    *,
    inrelease_path: Path,
    expected_fingerprint: str,
    expected_keyring_sha256: str | None,
    gpgv_executable: str,
) -> None:
    keyring = _read_stable_file(BASE_IMAGE_ARCHIVE_KEYRING)
    if (
        expected_keyring_sha256 is not None
        and hashlib.sha256(keyring).hexdigest() != expected_keyring_sha256
    ):
        raise SystemPackageError("base-image archive keyring hash does not match")
    result = subprocess.run(
        (
            gpgv_executable,
            "--status-fd",
            "1",
            "--homedir",
            os.fspath(GPGV_EMPTY_HOMEDIR),
            "--keyring",
            os.fspath(BASE_IMAGE_ARCHIVE_KEYRING),
            os.fspath(inrelease_path),
        ),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemPackageError("Debian InRelease signature verification failed")
    try:
        status_lines = result.stdout.decode("ascii").splitlines()
    except UnicodeError:
        raise SystemPackageError("gpgv status output is invalid") from None
    fingerprints: list[str] = []
    for line in status_lines:
        if line.startswith("[GNUPG:] VALIDSIG "):
            fields = line.split()
            if len(fields) < 4:
                raise SystemPackageError("gpgv VALIDSIG status is malformed")
            fingerprints.append(fields[2])
    if fingerprints != [expected_fingerprint]:
        raise SystemPackageError("gpgv signing fingerprint does not match the reviewed release")


def _extract_clearsigned_payload(content: bytes) -> str:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeError:
        raise SystemPackageError("InRelease is not UTF-8") from None
    if not lines or lines[0] != "-----BEGIN PGP SIGNED MESSAGE-----":
        raise SystemPackageError("InRelease is not a clear-signed document")
    try:
        header_end = lines.index("", 1)
        signature_start = lines.index("-----BEGIN PGP SIGNATURE-----", header_end + 1)
    except ValueError:
        raise SystemPackageError("InRelease clear-sign framing is invalid") from None
    headers = lines[1:header_end]
    if not headers or any(not line.startswith("Hash: ") for line in headers):
        raise SystemPackageError("InRelease clear-sign headers are invalid")
    if "SHA256" not in " ".join(headers):
        raise SystemPackageError("InRelease does not declare SHA256")
    body: list[str] = []
    for line in lines[header_end + 1 : signature_start]:
        body.append(line[2:] if line.startswith("- ") else line)
    return "\n".join(body).rstrip("\n") + "\n"


def _parse_control_stanza(text: str, label: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith((" ", "\t")):
            if current is None:
                raise SystemPackageError(f"{label} has an orphan continuation")
            fields[current] += ("\n" if fields[current] else "") + line[1:]
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9-]*):(?: (.*))?", line)
        if match is None:
            raise SystemPackageError(f"{label} contains a malformed field")
        name, value = match.group(1), match.group(2) or ""
        if name in fields:
            raise SystemPackageError(f"{label} contains a duplicate field")
        fields[name] = value
        current = name
    return fields


def _parse_release(content: bytes) -> tuple[dict[str, str], dict[str, tuple[int, str]]]:
    fields = _parse_control_stanza(_extract_clearsigned_payload(content).rstrip("\n"), "InRelease")
    if not {"Architectures", "Codename", "SHA256"}.issubset(fields):
        raise SystemPackageError("InRelease is missing required fields")
    indexes: dict[str, tuple[int, str]] = {}
    for line in fields["SHA256"].splitlines():
        parts = line.split()
        if len(parts) != 3:
            raise SystemPackageError("InRelease SHA256 entry is malformed")
        sha256, size_text, path_text = parts
        _require_hash(sha256, "InRelease index hash")
        path = _checked_relative_path(path_text, "InRelease index path").as_posix()
        if path in indexes:
            raise SystemPackageError("InRelease index path is duplicate")
        try:
            size = int(size_text)
        except ValueError:
            raise SystemPackageError("InRelease index size is invalid") from None
        if size <= 0:
            raise SystemPackageError("InRelease index size is invalid")
        indexes[path] = (size, sha256)
    if not any(path.endswith(("Packages", "Packages.xz")) for path in indexes):
        raise SystemPackageError("InRelease has no authenticated Packages indexes")
    return fields, indexes


def _decompress_packages_index(path: PurePosixPath, content: bytes) -> bytes:
    if path.name == "Packages":
        if len(content) > MAX_DECOMPRESSED_INDEX_BYTES:
            raise SystemPackageError("Packages index exceeds its decompressed size bound")
        return content
    if path.name != "Packages.xz":
        raise SystemPackageError("package index compression is unsupported")
    try:
        decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
        output = decompressor.decompress(content, max_length=MAX_DECOMPRESSED_INDEX_BYTES + 1)
    except lzma.LZMAError:
        raise SystemPackageError("Packages.xz index is invalid") from None
    if (
        len(output) > MAX_DECOMPRESSED_INDEX_BYTES
        or not decompressor.eof
        or decompressor.unused_data
    ):
        raise SystemPackageError("Packages.xz index exceeds its bound or is not exact")
    return output


def _parse_packages_index(content: bytes) -> list[dict[str, object]]:
    try:
        text = content.decode("utf-8")
    except UnicodeError:
        raise SystemPackageError("Packages index is not UTF-8") from None
    if "\r" in text or "\x00" in text or not text.endswith("\n"):
        raise SystemPackageError("Packages index encoding is not canonical")
    stanzas = [stanza for stanza in text.rstrip("\n").split("\n\n") if stanza]
    if not stanzas:
        raise SystemPackageError("Packages index contains no stanzas")
    result: list[dict[str, object]] = []
    for stanza_text in stanzas:
        fields = _parse_control_stanza(stanza_text, "Packages stanza")
        if not _REQUIRED_PACKAGE_FIELDS.issubset(fields):
            raise SystemPackageError("Packages stanza is missing an authenticated field")
        package = fields["Package"]
        version = fields["Version"]
        architecture = fields["Architecture"]
        filename = _checked_relative_path(
            fields["Filename"], "authenticated package Filename"
        ).as_posix()
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", package)
            or not version
            or any(character.isspace() for character in version)
            or architecture not in {"all", "amd64"}
            or not filename.endswith(".deb")
        ):
            raise SystemPackageError("Packages stanza identity is invalid")
        try:
            size = int(fields["Size"])
        except ValueError:
            raise SystemPackageError("Packages stanza Size is invalid") from None
        if size <= 0:
            raise SystemPackageError("Packages stanza Size is invalid")
        result.append(
            {
                "architecture": architecture,
                "filename": filename,
                "package": package,
                "sha256": _require_hash(fields["SHA256"], "Packages stanza SHA256"),
                "size": size,
                "version": version,
            }
        )
    return result


def _parse_inventory(content: bytes) -> set[tuple[str, str, str]]:
    try:
        text = content.decode("utf-8")
    except UnicodeError:
        raise SystemPackageError("expected dpkg inventory is not UTF-8") from None
    if "\r" in text or not text.endswith("\n"):
        raise SystemPackageError("expected dpkg inventory is not canonical")
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        columns = line.split("\t")
        if len(columns) != 3:
            raise SystemPackageError("expected dpkg inventory row is invalid")
        package, version, architecture = columns
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", package)
            or not version
            or architecture not in {"all", "amd64"}
        ):
            raise SystemPackageError("expected dpkg inventory row is invalid")
        rows.append((package, version, architecture))
    if rows != sorted(set(rows), key=lambda row: tuple(value.encode("utf-8") for value in row)):
        raise SystemPackageError("expected dpkg inventory is duplicate or unsorted")
    return set(rows)


def _validate_manifest_package(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _PACKAGE_KEYS:
        raise SystemPackageError("system package manifest package record is invalid")
    filename = _checked_relative_path(value["filename"], "manifest package Filename")
    if (  # pylint: disable=too-many-boolean-expressions
        not filename.name.endswith(".deb")
        or value["architecture"] not in {"all", "amd64"}
        or not isinstance(value["package"], str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", value["package"])
        or not isinstance(value["version"], str)
        or not value["version"]
        or not isinstance(value["size"], int)
        or isinstance(value["size"], bool)
        or value["size"] <= 0
    ):
        raise SystemPackageError("system package manifest package record is invalid")
    _require_hash(value["sha256"], "manifest package hash")
    return value


def _verify_authenticated_indexes(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    *,
    bundle: Path,
    payload: dict[str, object],
    release_indexes: dict[str, tuple[int, str]],
    inventory: set[tuple[str, str, str]],
) -> list[dict[str, object]]:
    entries = payload["package_indexes"]
    if not isinstance(entries, list) or not entries:
        raise SystemPackageError("system package manifest has no package indexes")
    paths: list[str] = []
    authenticated: dict[str, dict[str, object]] = {}
    identities: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _INDEX_IDENTITY_KEYS:
            raise SystemPackageError("system package index identity is invalid")
        path = _checked_relative_path(entry["release_path"], "manifest index path")
        path_text = path.as_posix()
        size = entry["byte_length"]
        sha256 = _require_hash(entry["sha256"], "manifest index hash")
        if (  # pylint: disable=too-many-boolean-expressions
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or size > MAX_COMPRESSED_INDEX_BYTES
            or path_text not in release_indexes
            or release_indexes[path_text] != (size, sha256)
        ):
            raise SystemPackageError("package index is not exactly authenticated by InRelease")
        content = _read_stable_file(
            bundle / "indexes" / Path(*path.parts),
            maximum_bytes=MAX_COMPRESSED_INDEX_BYTES,
        )
        if len(content) != size or hashlib.sha256(content).hexdigest() != sha256:
            raise SystemPackageError("package index byte length or hash does not match")
        for record in _parse_packages_index(_decompress_packages_index(path, content)):
            filename = record["filename"]
            identity = (record["package"], record["version"], record["architecture"])
            if filename in authenticated or identity in identities:
                raise SystemPackageError("authenticated package index is ambiguous or duplicate")
            authenticated[filename] = record
            identities.add(identity)
        paths.append(path_text)
    if paths != sorted(set(paths), key=lambda value: value.encode("utf-8")):
        raise SystemPackageError("system package indexes are duplicate or unsorted")
    actual_indexes = {path.as_posix() for path in _walk_regular_files(bundle / "indexes")}
    if actual_indexes != set(paths):
        raise SystemPackageError("system package index tree has missing or extra files")

    manifest_packages = payload["packages"]
    if not isinstance(manifest_packages, list) or not manifest_packages:
        raise SystemPackageError("system package manifest has no package closure")
    selected: list[dict[str, object]] = []
    local_names: set[str] = set()
    filenames: list[str] = []
    for value in manifest_packages:
        record = _validate_manifest_package(value)
        filename = record["filename"]
        if authenticated.get(filename) != record:
            raise SystemPackageError(
                "manifest package does not match an authenticated package index"
            )
        local_name = PurePosixPath(filename).name
        if local_name in local_names:
            raise SystemPackageError("authenticated package local basename is ambiguous")
        local_names.add(local_name)
        content = _read_stable_file(bundle / "packages" / local_name)
        if (
            len(content) != record["size"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]
        ):
            raise SystemPackageError("authenticated package byte length or hash does not match")
        if (record["package"], record["version"], record["architecture"]) not in inventory:
            raise SystemPackageError(
                "expected dpkg inventory does not contain authenticated package"
            )
        filenames.append(filename)
        selected.append(record)
    if filenames != sorted(set(filenames), key=lambda value: value.encode("utf-8")):
        raise SystemPackageError("system package manifest is ambiguous or unsorted")
    _require_directory(bundle / "packages", "system package directory")
    actual_packages = {
        entry.name
        for entry in (bundle / "packages").iterdir()
        if entry.is_file() and not entry.is_symlink()
    }
    if actual_packages != local_names or len(list((bundle / "packages").iterdir())) != len(
        actual_packages
    ):
        raise SystemPackageError("system package directory has missing, extra, or unsafe files")
    return selected


def verify_system_package_bundle(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    *,
    bundle: Path,
    expected_snapshot_url: str,
    expected_manifest_sha256: str,
    expected_inrelease_sha256: str,
    expected_inventory_sha256: str,
    expected_signing_fingerprint: str,
    expected_codename: str,
    expected_architecture: str,
    expected_keyring_sha256: str | None = None,
    gpgv_executable: str = "gpgv",
) -> dict[str, object]:
    """Verify the fixed trust anchor, signed indexes, local packages, and inventory."""

    _require_hash(expected_manifest_sha256, "bundle manifest hash")
    _require_hash(expected_inrelease_sha256, "InRelease hash")
    _require_hash(expected_inventory_sha256, "inventory hash")
    if expected_keyring_sha256 is not None:
        _require_hash(expected_keyring_sha256, "base-image archive keyring hash")
    _require_fingerprint(expected_signing_fingerprint, "reviewed signing fingerprint")
    if not isinstance(expected_snapshot_url, str) or not expected_snapshot_url.startswith(
        "https://snapshot.debian.org/"
    ):
        raise SystemPackageError("Debian snapshot URL is invalid")
    if (
        not isinstance(expected_codename, str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", expected_codename)
        or expected_architecture != "amd64"
    ):
        raise SystemPackageError("reviewed Debian codename or architecture is invalid")

    root = Path(bundle)
    _require_directory(root, "system package bundle")
    try:
        entries = {entry.name for entry in root.iterdir()}
    except OSError:
        raise SystemPackageError("system package bundle is unavailable") from None
    if entries != _EXPECTED_ROOT_ENTRIES:
        raise SystemPackageError("system package bundle has missing or extra root entries")
    _require_directory(root / "indexes", "system package index directory")
    _require_directory(root / "packages", "system package directory")

    manifest_content = _read_stable_file(root / "bundle-manifest.json")
    if hashlib.sha256(manifest_content).hexdigest() != expected_manifest_sha256:
        raise SystemPackageError("bundle manifest hash does not match the build input")
    payload = _strict_json_loads(manifest_content)
    if (
        not isinstance(payload, dict)
        or set(payload) != _TOP_LEVEL_KEYS
        or _canonical_json_bytes(payload) != manifest_content
        or payload["schema"] != SYSTEM_PACKAGE_BUNDLE_SCHEMA
        or payload["snapshot_url"] != expected_snapshot_url
    ):
        raise SystemPackageError("system package manifest is not canonical or exact")
    if payload["codename"] != expected_codename:
        raise SystemPackageError("manifest codename does not match the reviewed release")
    if payload["architecture"] != expected_architecture:
        raise SystemPackageError("manifest architecture does not match the reviewed target")
    if payload["signing_fingerprint"] != expected_signing_fingerprint:
        raise SystemPackageError("manifest signing fingerprint does not match the reviewed key")

    inrelease, inrelease_content = _verify_root_identity(
        root, payload["inrelease"], expected_filename="InRelease"
    )
    inventory_identity, inventory_content = _verify_root_identity(
        root,
        payload["expected_dpkg_inventory"],
        expected_filename="expected-dpkg-inventory.txt",
    )
    if (
        inrelease["sha256"] != expected_inrelease_sha256
        or inventory_identity["sha256"] != expected_inventory_sha256
    ):
        raise SystemPackageError("system package evidence hash does not match the build input")
    _verify_gpgv(
        inrelease_path=root / "InRelease",
        expected_fingerprint=expected_signing_fingerprint,
        expected_keyring_sha256=expected_keyring_sha256,
        gpgv_executable=gpgv_executable,
    )
    release, release_indexes = _parse_release(inrelease_content)
    if release["Codename"] != expected_codename:
        raise SystemPackageError("signed InRelease codename does not match")
    if expected_architecture not in release["Architectures"].split():
        raise SystemPackageError("signed InRelease architecture does not match")
    inventory = _parse_inventory(inventory_content)
    authenticated_packages = _verify_authenticated_indexes(
        bundle=root,
        payload=payload,
        release_indexes=release_indexes,
        inventory=inventory,
    )
    result = dict(payload)
    result["packages"] = authenticated_packages
    return result


def install_system_package_bundle(
    *,
    bundle: Path,
    payload: dict[str, object],
    dpkg_executable: str = "dpkg",
    dpkg_query_executable: str = "dpkg-query",
) -> None:
    """Install only authenticated local packages and prove the complete inventory."""

    root = Path(bundle)
    package_paths: list[str] = []
    for value in payload["packages"]:
        record = _validate_manifest_package(value)
        path = root / "packages" / PurePosixPath(record["filename"]).name
        content = _read_stable_file(path)
        if (
            len(content) != record["size"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]
        ):
            raise SystemPackageError("authenticated package changed before installation")
        package_paths.append(os.fspath(path))
    install = subprocess.run(
        (dpkg_executable, "--install", *package_paths),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if install.returncode != 0:
        raise SystemPackageError("exact authenticated Debian package installation failed")
    query = subprocess.run(
        (
            dpkg_query_executable,
            "-W",
            "-f=${Package}\\t${Version}\\t${Architecture}\\n",
        ),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if query.returncode != 0:
        raise SystemPackageError("final dpkg inventory query failed")
    expected = _read_stable_file(root / payload["expected_dpkg_inventory"]["filename"])
    if query.stdout != expected:
        raise SystemPackageError("final dpkg inventory does not match the canonical inventory")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--snapshot-url", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--inrelease-sha256", required=True)
    parser.add_argument("--keyring-sha256", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--signing-fingerprint", required=True)
    parser.add_argument("--codename", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--install", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = verify_system_package_bundle(
            bundle=args.bundle,
            expected_snapshot_url=args.snapshot_url,
            expected_manifest_sha256=args.manifest_sha256,
            expected_inrelease_sha256=args.inrelease_sha256,
            expected_keyring_sha256=args.keyring_sha256,
            expected_inventory_sha256=args.inventory_sha256,
            expected_signing_fingerprint=args.signing_fingerprint,
            expected_codename=args.codename,
            expected_architecture=args.architecture,
        )
        if args.install:
            install_system_package_bundle(bundle=args.bundle, payload=payload)
    except (OSError, SystemPackageError) as error:
        print(f"OaF system package verification failed: {error}")
        return 1
    print("OaF system package verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
