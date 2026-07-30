"""Generate the strict source manifest for the frozen OaF runner."""

# The manifest intentionally mirrors the runner's no-follow/canonical helpers.
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

SCHEMA = "crux.oaf-runner-source-manifest/v1"
MANIFEST_PATH = "runtime/oaf_tf1/runner-source-manifest.json"
SOURCE_PATHS = (
    "runtime/oaf_tf1/Dockerfile",
    "runtime/oaf_tf1/apply_instrumentation_patch.py",
    "runtime/oaf_tf1/distribution-build-manifest.json",
    "runtime/oaf_tf1/entrypoint.py",
    "runtime/oaf_tf1/oaf_backend.py",
    "runtime/oaf_tf1/patches/capture-emitted-frame.patch",
    "runtime/oaf_tf1/protocol.py",
    "runtime/oaf_tf1/requirements-build.in",
    "runtime/oaf_tf1/requirements-build.lock",
    "runtime/oaf_tf1/requirements-test.in",
    "runtime/oaf_tf1/requirements-test.lock",
    "runtime/oaf_tf1/requirements.in",
    "runtime/oaf_tf1/requirements.lock",
    "runtime/oaf_tf1/tests/test_instrumentation.py",
    "runtime/oaf_tf1/tests/test_protocol.py",
    "runtime/oaf_tf1/tests/test_tensor_coverage.py",
    "tools/hpa320/generate_runner_source_manifest.py",
    "tools/hpa320/oaf_system_packages.py",
)
COVERED_ROOTS = (
    "runtime/oaf_tf1",
    "tools/hpa320",
)


class RunnerManifestError(ValueError):
    pass


def _read_regular_no_follow(path: Path) -> bytes:
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
        raise RunnerManifestError("runner source is not a stable regular file") from None


def build_runner_source_manifest(repository_root: Path) -> Mapping[str, object]:
    repository_root = Path(repository_root)
    files = []
    for relative_path in SOURCE_PATHS:
        content = _read_regular_no_follow(repository_root / relative_path)
        files.append(
            {
                "byte_length": len(content),
                "path": relative_path,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    files.sort(key=lambda entry: entry["path"].encode("utf-8"))
    if len({entry["path"] for entry in files}) != len(files):
        raise RunnerManifestError("runner source paths are duplicate")
    if MANIFEST_PATH in {entry["path"] for entry in files}:
        raise RunnerManifestError("runner source manifest cannot hash itself")
    return {
        "covered_roots": list(COVERED_ROOTS),
        "files": files,
        "schema": SCHEMA,
    }


def canonical_manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def write_runner_source_manifest(repository_root: Path, output_path: Path) -> str:
    content = canonical_manifest_bytes(build_runner_source_manifest(repository_root))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".runner-source-manifest-", dir=str(output_path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o644)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, str(output_path))
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return hashlib.sha256(content).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.repository_root / MANIFEST_PATH
    try:
        digest = write_runner_source_manifest(args.repository_root, output)
    except RunnerManifestError:
        os.write(2, b"code=runner_source_manifest_invalid count=1\n")
        return 2
    os.write(1, ("runner_source_manifest_sha256=" + digest + "\n").encode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
