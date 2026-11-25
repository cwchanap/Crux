"""Build one throwaway TF 2.15 OaF runtime from the frozen TF1 source.

The frozen ``runtime/oaf_tf1`` tree is only read. The scratch copy is created in
the caller's output directory, instrumented with the reviewed emitted-frame
patch first, and only then adapted by ``patches/tf215-compat.patch``.
"""

# The builder keeps explicit validation at every filesystem and patch boundary.
# pylint: disable=too-many-branches

from __future__ import annotations

import argparse
import hashlib
import locale
import os
import shutil
import subprocess
import types
from pathlib import Path
from typing import Sequence

FROZEN_RUNTIME = Path("runtime") / "oaf_tf1"
SCRATCH_FILES = ("model.py", "model.json", "worker.py")
INSTRUMENTATION_PATCH = FROZEN_RUNTIME / "patches" / "capture-emitted-frame.patch"
COMPATIBILITY_PATCH = Path("experiments") / "oaf_tf215" / "patches" / "tf215-compat.patch"
COMPATIBILITY_PATCH_SHA256 = "554527246744662d914ea45973b93dd23aa6a31e5270a36b729589327da4e041"
# Inherited values force Git into repository mode and override the discovery
# ceiling, which makes `git apply` rebase and skip the patch paths.
_GIT_REPOSITORY_SELECTION = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_WORK_TREE",
    }
)


class ScratchRuntimeError(RuntimeError):
    """The scratch runtime could not be built."""


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ScratchRuntimeError(f"source file is unavailable: {path}")


def _copy_regular_file(source: Path, destination: Path) -> None:
    _require_regular_file(source)
    shutil.copyfile(source, destination)


def _copy_regular_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ScratchRuntimeError(f"source directory is unavailable: {source}")
    for entry in sorted(source.rglob("*")):
        target = destination / entry.relative_to(source)
        if entry.is_symlink():
            raise ScratchRuntimeError(f"source tree contains a symlink: {entry}")
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif entry.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry, target)
        else:
            raise ScratchRuntimeError(f"source tree entry is unsupported: {entry}")


def _load_instrumentation_module(path: Path) -> types.ModuleType:
    """Load the frozen patcher without writing bytecode into the frozen tree."""
    _require_regular_file(path)
    module = types.ModuleType("crux_oaf_instrumentation")
    module.__file__ = os.fspath(path)
    exec(compile(path.read_bytes(), os.fspath(path), "exec"), module.__dict__)
    return module


def _apply_instrumentation(repo_root: Path, upstream: Path, vendor: Path, patch_path: Path) -> None:
    module = _load_instrumentation_module(
        repo_root / "runtime" / "oaf_tf1" / "apply_instrumentation_patch.py"
    )
    module.apply_reviewed_patch(upstream, vendor, patch_path)


def _run_git(root: Path, *arguments: str, patch: bytes) -> None:
    # A scratch root inside a Git checkout would make `git apply` resolve patch
    # paths against the enclosing repository and silently skip them. Drop
    # inherited repository selection and set a ceiling so Git treats `root` as
    # the top level.
    environment = {
        key: value for key, value in os.environ.items() if key not in _GIT_REPOSITORY_SELECTION
    }
    environment["GIT_CEILING_DIRECTORIES"] = os.fspath(root.parent)
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        input=patch,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if result.returncode != 0:
        output = result.stderr.strip() or result.stdout.strip()
        detail = output.decode(locale.getpreferredencoding(False), errors="replace") or "git failed"
        raise ScratchRuntimeError(f"compatibility patch rejected: {detail}")


def build_runtime(repo_root: Path, output_root: Path) -> Path:
    """Create a new scratch runtime and return its ``runtime/`` directory."""
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root)
    # Check the supplied path before resolve(): resolve() follows a dangling
    # symlink to a nonexistent target, which would then pass both checks.
    if output_root.is_symlink() or output_root.exists():
        raise ScratchRuntimeError("scratch runtime destination already exists or is a symlink")
    output_root = output_root.resolve()
    frozen_runtime = repo_root / FROZEN_RUNTIME
    if output_root.is_relative_to(frozen_runtime.resolve()):
        raise ScratchRuntimeError("scratch runtime destination is inside the frozen runtime tree")

    compatibility_patch = repo_root / COMPATIBILITY_PATCH
    _require_regular_file(compatibility_patch)
    patch = compatibility_patch.read_bytes()
    if hashlib.sha256(patch).hexdigest() != COMPATIBILITY_PATCH_SHA256:
        raise ScratchRuntimeError("compatibility patch is not the reviewed revision")

    runtime_dir = output_root / "runtime"
    upstream_dir = output_root / "upstream"
    vendor_dir = runtime_dir / "vendor"
    output_root.mkdir(parents=True)
    runtime_dir.mkdir()
    upstream_dir.mkdir()
    for name in SCRATCH_FILES:
        _copy_regular_file(frozen_runtime / name, runtime_dir / name)
    _copy_regular_tree(frozen_runtime / "vendor" / "magenta", upstream_dir / "magenta")
    _copy_regular_file(frozen_runtime / "vendor" / "LICENSE", upstream_dir / "LICENSE")
    if vendor_dir.exists() or vendor_dir.is_symlink():
        raise ScratchRuntimeError("runtime/vendor must not exist before instrumentation")

    _apply_instrumentation(
        repo_root,
        upstream_dir,
        vendor_dir,
        repo_root / INSTRUMENTATION_PATCH,
    )
    _run_git(output_root, "apply", "--check", "-", patch=patch)
    _run_git(output_root, "apply", "-", patch=patch)
    return runtime_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(build_runtime(arguments.repo_root, arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
