"""Apply the sole reviewed OaF instrumentation patch to a private source copy."""

# The strict patch parser and no-follow tree copier deliberately keep all
# validation branches local so publication cannot bypass an earlier check.
# pylint: disable=too-many-boolean-expressions,too-many-branches,too-many-locals
# pylint: disable=too-many-nested-blocks,too-many-statements

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

PATCH_SHA256 = "5d1cf487bd1600495ad8830d7fd5c2158b51ff4de7b4a8065846627839c48103"
SOURCE_PREIMAGES = {
    "magenta/models/onsets_frames_transcription/infer_util.py": (
        "e8a32decc4aed541de98385a329a22c28f60ab3c276e5b5a4d298cda158dcea0"
    ),
    "magenta/music/sequences_lib.py": (
        "486f29b7c40e3c602c5e8e543ff4e8bab278a670238e2edefa2daf3c54e9683a"
    ),
}
MANIFEST_SCHEMA = "crux.oaf-instrumented-source-manifest/v1"
_HUNK = re.compile(
    rb"@@ -(?P<old>[1-9][0-9]*)(?:,(?P<old_count>[0-9]+))? "
    rb"\+(?P<new>[1-9][0-9]*)(?:,(?P<new_count>[0-9]+))? @@(?: .*)?\n\Z"
)


class InstrumentationPatchError(ValueError):
    pass


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _path_from_header(value: bytes, prefix: bytes) -> str:
    if not value.startswith(prefix) or not value.endswith(b"\n"):
        raise InstrumentationPatchError("patch path header is invalid")
    try:
        path = value[len(prefix) : -1].decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise InstrumentationPatchError("patch path is not UTF-8") from None
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or "\x00" in path
        or unicodedata.normalize("NFC", path) != path
        or any(component in {"", ".", ".."} for component in path.split("/"))
    ):
        raise InstrumentationPatchError("patch path traversal is invalid")
    return path


def _parse_patch(content: bytes) -> Mapping[str, Sequence[Tuple[int, int, Sequence[bytes]]]]:
    if b"\r" in content or not content.endswith(b"\n"):
        raise InstrumentationPatchError("patch line endings are invalid")
    lines = content.splitlines(keepends=True)
    index = 0
    parsed: Dict[str, List[Tuple[int, int, Sequence[bytes]]]] = {}
    while index < len(lines):
        line = lines[index]
        if not line.startswith(b"diff --git a/"):
            raise InstrumentationPatchError("patch file header is invalid")
        pieces = line[:-1].split(b" ")
        if len(pieces) != 4:
            raise InstrumentationPatchError("patch file header is invalid")
        old_path = _path_from_header(pieces[2] + b"\n", b"a/")
        new_path = _path_from_header(pieces[3] + b"\n", b"b/")
        if old_path != new_path or old_path in parsed:
            raise InstrumentationPatchError("patch target is duplicate or renamed")
        index += 1
        if index + 1 >= len(lines):
            raise InstrumentationPatchError("patch target header is incomplete")
        if _path_from_header(lines[index], b"--- a/") != old_path:
            raise InstrumentationPatchError("patch old target differs")
        index += 1
        if _path_from_header(lines[index], b"+++ b/") != old_path:
            raise InstrumentationPatchError("patch new target differs")
        index += 1
        hunks: List[Tuple[int, int, Sequence[bytes]]] = []
        previous_end = 0
        while index < len(lines) and lines[index].startswith(b"@@ "):
            match = _HUNK.fullmatch(lines[index])
            if match is None:
                raise InstrumentationPatchError("patch hunk header is invalid")
            old_start = int(match.group("old"))
            old_count = int(match.group("old_count") or b"1")
            new_count = int(match.group("new_count") or b"1")
            if old_start - 1 < previous_end:
                raise InstrumentationPatchError("patch hunks overlap")
            index += 1
            body = []
            observed_old = 0
            observed_new = 0
            while index < len(lines):
                item = lines[index]
                if item.startswith(b"diff --git ") or item.startswith(b"@@ "):
                    break
                if not item or item[:1] not in (b" ", b"+", b"-"):
                    raise InstrumentationPatchError("patch hunk content is invalid")
                body.append(item)
                if item[:1] in (b" ", b"-"):
                    observed_old += 1
                if item[:1] in (b" ", b"+"):
                    observed_new += 1
                index += 1
            if observed_old != old_count or observed_new != new_count:
                raise InstrumentationPatchError("patch hunk counts are invalid")
            previous_end = old_start - 1 + old_count
            hunks.append((old_start, old_count, tuple(body)))
        if not hunks:
            raise InstrumentationPatchError("patch target has no hunks")
        parsed[old_path] = hunks
    if set(parsed) != set(SOURCE_PREIMAGES):
        raise InstrumentationPatchError("patch target set is invalid")
    return parsed


def _validate_directory(path: Path, label: str) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError:
        raise InstrumentationPatchError(label + " directory is unavailable") from None
    if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
        raise InstrumentationPatchError(label + " must be a regular no-follow directory")
    return status


def _copy_regular_tree(source: Path, destination: Path) -> None:
    _validate_directory(source, "source")
    destination.mkdir(mode=0o755)
    pending = [(source, destination)]
    while pending:
        source_dir, destination_dir = pending.pop()
        try:
            entries = sorted(os.scandir(str(source_dir)), key=lambda entry: os.fsencode(entry.name))
        except OSError:
            raise InstrumentationPatchError("source tree could not be enumerated") from None
        for entry in entries:
            source_path = source_dir / entry.name
            destination_path = destination_dir / entry.name
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError:
                raise InstrumentationPatchError("source entry is unstable") from None
            if stat.S_ISDIR(status.st_mode):
                destination_path.mkdir(mode=stat.S_IMODE(status.st_mode))
                pending.append((source_path, destination_path))
                continue
            if not stat.S_ISREG(status.st_mode) or entry.is_symlink():
                raise InstrumentationPatchError("source entries must be regular no-follow files")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                source_fd = os.open(str(source_path), flags)
                try:
                    opened = os.fstat(source_fd)
                    if (
                        opened.st_dev,
                        opened.st_ino,
                        stat.S_IFMT(opened.st_mode),
                    ) != (status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode)):
                        raise OSError("source identity changed")
                    output_fd = os.open(
                        str(destination_path),
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        stat.S_IMODE(status.st_mode),
                    )
                    try:
                        while True:
                            chunk = os.read(source_fd, 65536)
                            if not chunk:
                                break
                            view = memoryview(chunk)
                            while view:
                                written = os.write(output_fd, view)
                                view = view[written:]
                    finally:
                        os.close(output_fd)
                finally:
                    os.close(source_fd)
            except OSError:
                raise InstrumentationPatchError("source copy failed closed") from None


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode):
                raise OSError("not regular")
            chunks = []
            while True:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            final_status = os.fstat(descriptor)
            if (status.st_dev, status.st_ino, status.st_size) != (
                final_status.st_dev,
                final_status.st_ino,
                final_status.st_size,
            ):
                raise OSError("identity changed")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError:
        raise InstrumentationPatchError(
            "source file is not a stable regular no-follow file"
        ) from None


def _apply_hunks(original: bytes, hunks: Sequence[Tuple[int, int, Sequence[bytes]]]) -> bytes:
    if b"\r" in original or (original and not original.endswith(b"\n")):
        raise InstrumentationPatchError("source preimage line endings are invalid")
    original_lines = original.splitlines(keepends=True)
    result = []
    cursor = 0
    for old_start, old_count, body in hunks:
        start = old_start - 1
        if start < cursor or start + old_count > len(original_lines):
            raise InstrumentationPatchError("patch hunk lies outside source preimage")
        expected = [line[1:] for line in body if line[:1] in (b" ", b"-")]
        if original_lines[start : start + old_count] != expected:
            raise InstrumentationPatchError("patch source preimage context drifted")
        result.extend(original_lines[cursor:start])
        result.extend(line[1:] for line in body if line[:1] in (b" ", b"+"))
        cursor = start + old_count
    result.extend(original_lines[cursor:])
    return b"".join(result)


def _atomic_replace(path: Path, content: bytes) -> None:
    try:
        original_mode = stat.S_IMODE(path.lstat().st_mode)
    except OSError:
        raise InstrumentationPatchError("patch destination preimage is unavailable") from None
    descriptor, temporary = tempfile.mkstemp(prefix=".instrumented-", dir=str(path.parent))
    try:
        os.fchmod(descriptor, original_mode)
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _manifest(root: Path, patch_sha256: str) -> Mapping[str, object]:
    files = []
    for directory, directory_names, file_names in os.walk(str(root), followlinks=False):
        directory_names.sort(key=os.fsencode)
        file_names.sort(key=os.fsencode)
        base = Path(directory)
        for name in file_names:
            path = base / name
            content = _read_regular(path)
            files.append(
                {
                    "byte_length": len(content),
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(content),
                }
            )
    files.sort(key=lambda entry: entry["path"].encode("utf-8"))
    return {
        "covered_roots": ["magenta"],
        "files": files,
        "patch_sha256": patch_sha256,
        "schema": MANIFEST_SCHEMA,
    }


def apply_reviewed_patch(
    source_root: Path, destination_root: Path, patch_path: Path
) -> Mapping[str, object]:
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    patch_path = Path(patch_path)
    patch_content = _read_regular(patch_path)
    patch_sha256 = _sha256(patch_content)
    if patch_sha256 != PATCH_SHA256:
        raise InstrumentationPatchError("instrumentation patch identity is unreviewed")
    parsed = _parse_patch(patch_content)
    _validate_directory(source_root, "source")
    if destination_root.exists() or destination_root.is_symlink():
        raise InstrumentationPatchError("destination must not already exist")
    destination_parent = destination_root.parent
    _validate_directory(destination_parent, "destination parent")

    validated_outputs = {}
    for relative_path, expected_sha256 in SOURCE_PREIMAGES.items():
        original = _read_regular(source_root / relative_path)
        if _sha256(original) != expected_sha256:
            raise InstrumentationPatchError("instrumentation source preimage does not match")
        validated_outputs[relative_path] = _apply_hunks(original, parsed[relative_path])

    stage = Path(tempfile.mkdtemp(prefix=".oaf-instrumented-", dir=str(destination_parent)))
    shutil.rmtree(str(stage))
    try:
        _copy_regular_tree(source_root, stage)
        for relative_path, content in validated_outputs.items():
            _atomic_replace(stage / relative_path, content)
        manifest = _manifest(stage, patch_sha256)
        os.rename(str(stage), str(destination_root))
        return manifest
    except (InstrumentationPatchError, OSError):
        if stage.exists():
            shutil.rmtree(str(stage))
        if isinstance(os.sys.exc_info()[1], InstrumentationPatchError):
            raise
        raise InstrumentationPatchError("instrumented destination publication failed") from None


def _canonical_manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _write_manifest(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise InstrumentationPatchError("applied-output manifest destination already exists")
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = apply_reviewed_patch(args.source, args.destination, args.patch)
        _write_manifest(args.manifest, _canonical_manifest_bytes(manifest))
    except InstrumentationPatchError:
        os.write(2, b"code=instrumentation_patch_invalid count=1\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
