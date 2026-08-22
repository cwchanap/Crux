from __future__ import annotations

import importlib.util
import subprocess
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

BUILDER_PATH = Path(__file__).parents[2] / "runtime" / "idm" / "build_pinned_wheel.py"


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("crux_idm_wheel_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BUILDER_PATH}")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def run_git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _fixture_repository(repo: Path) -> str:
    run_git(repo, "init", "--quiet")
    run_git(repo, "config", "user.email", "test@example.invalid")
    run_git(repo, "config", "user.name", "IDM wheel test")
    (repo / "idm").mkdir()
    (repo / "idm" / "__init__.py").write_text("VALUE = 'committed'\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'inverse-drum-machine'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )
    run_git(repo, "add", "idm", "pyproject.toml")
    run_git(repo, "commit", "--quiet", "-m", "initial")
    return run_git(repo, "rev-parse", "HEAD")


def _write_idm_wheel(builder: ModuleType, path: Path, repo: Path, commit: str) -> None:
    tracked = builder._commit_files(repo, commit)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for relative in tracked:
            if relative.parts[0] == "idm":
                wheel.writestr(relative.as_posix(), builder._commit_blob(repo, commit, relative))


def test_commit_blobs_ignore_staged_and_unstaged_source_edits(tmp_path: Path) -> None:
    builder = _load_builder()
    repo = tmp_path / "upstream"
    repo.mkdir()
    commit = _fixture_repository(repo)
    tracked = builder._commit_files(repo, commit)
    original_manifest = builder._source_manifest(repo, commit, tracked)

    first_tree = tmp_path / "first-tree"
    first_tree.mkdir()
    builder._copy_commit_tree(repo, commit, first_tree)
    first_wheel = tmp_path / "first.whl"
    _write_idm_wheel(builder, first_wheel, repo, commit)
    first_verified = builder._verify_wheel_sources(first_wheel, repo, commit, tracked)

    module = repo / "idm" / "__init__.py"
    module.write_text("VALUE = 'staged'\n", encoding="utf-8")
    run_git(repo, "add", "idm/__init__.py")
    module.write_text("VALUE = 'unstaged'\n", encoding="utf-8")

    second_tree = tmp_path / "second-tree"
    second_tree.mkdir()
    builder._copy_commit_tree(repo, commit, second_tree)
    second_wheel = tmp_path / "second.whl"
    _write_idm_wheel(builder, second_wheel, repo, commit)
    second_manifest = builder._source_manifest(repo, commit, tracked)
    second_verified = builder._verify_wheel_sources(second_wheel, repo, commit, tracked)

    assert second_tree.joinpath("idm", "__init__.py").read_text() == "VALUE = 'committed'\n"
    assert second_manifest == original_manifest
    assert second_verified == first_verified
    assert second_wheel.read_bytes() == first_wheel.read_bytes()

    tampered_wheel = tmp_path / "tampered.whl"
    with zipfile.ZipFile(tampered_wheel, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        wheel.writestr("idm/__init__.py", module.read_bytes())
    with pytest.raises(RuntimeError, match="wheel source bytes differ"):
        builder._verify_wheel_sources(tampered_wheel, repo, commit, tracked)
