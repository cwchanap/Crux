from __future__ import annotations

# Contract fixtures intentionally repeat exact public record shapes.
# pylint: disable=duplicate-code
import os
import subprocess
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from src.benchmark import backend_attestation
from src.benchmark.backend_attestation import (
    AttestationError,
    ChangedFile,
    ExecutionConditions,
    build_changed_file_manifest,
    publish_execution_attestation,
)
from src.benchmark.backend_identity import (
    canonical_json_bytes,
    sha256_hex,
    strict_json_loads,
)

FIXED_UTC = datetime(2026, 7, 27, 1, 2, 3, 456789, tzinfo=UTC)
FIXED_UUID = UUID("12345678-1234-4678-9234-567812345678")
OAF_CONDITIONS = ExecutionConditions(
    cpu_limit="2",
    memory_bytes=4_294_967_296,
    pid_limit=128,
    tmp_bytes=1_073_741_824,
    shm_bytes=67_108_864,
    startup_deadline_seconds=120,
    request_deadline_seconds=300,
)


def git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-c", "user.name=Crux Tests", "-c", "user.email=crux@example.invalid", *arguments],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def create_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init", "-q")
    (repository / "src" / "inference").mkdir(parents=True)
    (repository / "docs").mkdir()
    (repository / "src" / "inference" / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
    (repository / "src" / "inference" / "adapter.py").write_text(
        "ADAPTER = 1\n",
        encoding="utf-8",
    )
    (repository / "docs" / "notes.md").write_text("clean\n", encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "fixture")
    return repository


def write_source_manifest(
    repository: Path,
    name: str = "source-manifest.json",
    *,
    files: list[object] | None = None,
    covered_roots: list[object] | None = None,
) -> Path:
    path = repository / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "covered_roots": ["src/inference"] if covered_roots is None else covered_roots,
        "files": (
            [
                {
                    "path": "src/inference/model.py",
                    "sha256": sha256_hex(
                        (repository / "src" / "inference" / "model.py").read_bytes()
                    ),
                }
            ]
            if files is None
            else files
        ),
        "schema": "crux.source-manifest/v1",
    }
    path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))
    return path


def test_clean_inference_scope_has_no_changed_files(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)

    changed = build_changed_file_manifest(repository, (manifest,))

    assert not changed


def test_scope_is_union_of_every_manifest_path_and_covered_root(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    first = write_source_manifest(repository)
    second = write_source_manifest(
        repository,
        "runner-source-manifest.json",
        files=[{"path": "src/inference/adapter.py", "sha256": "a" * 64}],
        covered_roots=["runtime/runner"],
    )
    (repository / "src" / "inference" / "adapter.py").write_text(
        "ADAPTER = 2\n",
        encoding="utf-8",
    )
    (repository / "runtime" / "runner").mkdir(parents=True)
    (repository / "runtime" / "runner" / "new.py").write_text("NEW = 1\n", encoding="utf-8")

    changed = build_changed_file_manifest(repository, (first, second))

    assert changed == (
        ChangedFile(
            path="runtime/runner/new.py",
            status="untracked",
            sha256=sha256_hex(b"NEW = 1\n"),
        ),
        ChangedFile(
            path="src/inference/adapter.py",
            status="modified",
            sha256=sha256_hex(b"ADAPTER = 2\n"),
        ),
    )


def test_modified_deleted_and_untracked_files_are_sorted_and_hashed(
    tmp_path: Path,
) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(
        repository,
        files=[
            {"path": "src/inference/model.py", "sha256": "a" * 64},
            {"path": "src/inference/adapter.py", "sha256": "b" * 64},
        ],
    )
    (repository / "src" / "inference" / "model.py").write_text("MODEL = 2\n", encoding="utf-8")
    (repository / "src" / "inference" / "adapter.py").unlink()
    (repository / "src" / "inference" / "added.py").write_text("ADDED = 1\n", encoding="utf-8")

    changed = build_changed_file_manifest(repository, (manifest,))

    assert changed == (
        ChangedFile(
            path="src/inference/adapter.py",
            status="deleted",
            sha256=None,
        ),
        ChangedFile(
            path="src/inference/added.py",
            status="untracked",
            sha256=sha256_hex(b"ADDED = 1\n"),
        ),
        ChangedFile(
            path="src/inference/model.py",
            status="modified",
            sha256=sha256_hex(b"MODEL = 2\n"),
        ),
    )


def test_nul_git_status_preserves_paths_containing_spaces(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)
    spaced = repository / "src" / "inference" / "new adapter.py"
    spaced.write_text("SPACE = 1\n", encoding="utf-8")

    changed = build_changed_file_manifest(repository, (manifest,))

    assert changed == (
        ChangedFile(
            path="src/inference/new adapter.py",
            status="untracked",
            sha256=sha256_hex(b"SPACE = 1\n"),
        ),
    )


def test_rename_is_deleted_plus_untracked(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)
    git(
        repository,
        "mv",
        "src/inference/model.py",
        "src/inference/renamed.py",
    )

    changed = build_changed_file_manifest(repository, (manifest,))

    assert changed == (
        ChangedFile("src/inference/model.py", "deleted", None),
        ChangedFile(
            "src/inference/renamed.py",
            "untracked",
            sha256_hex(b"MODEL = 1\n"),
        ),
    )


def test_unrelated_dirty_path_is_excluded_from_changed_file_manifest(
    tmp_path: Path,
) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)
    (repository / "docs" / "notes.md").write_text("dirty docs\n", encoding="utf-8")

    changed = build_changed_file_manifest(repository, (manifest,))

    assert not changed


def test_source_manifest_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    manifest = repository / "source-manifest.json"
    manifest.write_bytes(
        b'{"covered_roots":[],"files":[],"files":[],"schema":"crux.source-manifest/v1"}\n'
    )

    with pytest.raises(AttestationError, match="source manifest"):
        build_changed_file_manifest(repository, (manifest,))


def test_source_manifest_must_be_inside_repository_root(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    outside = tmp_path / "outside-source-manifest.json"
    outside.write_bytes(
        canonical_json_bytes(
            {
                "covered_roots": ["src/inference"],
                "files": [],
                "schema": "crux.source-manifest/v1",
            },
            trailing_newline=True,
        )
    )

    with pytest.raises(AttestationError, match="source manifest"):
        build_changed_file_manifest(repository, (outside,))


@pytest.mark.parametrize(
    "invalid_path",
    [
        "",
        "/absolute.py",
        "src//model.py",
        "src/./model.py",
        "src/../model.py",
        r"src\model.py",
        unicodedata.normalize("NFD", "src/inférence.py"),
        "src/inference/",
    ],
)
def test_source_manifest_rejects_invalid_repository_paths(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(
        repository,
        files=[{"path": invalid_path, "sha256": "a" * 64}],
    )

    with pytest.raises(AttestationError, match="path"):
        build_changed_file_manifest(repository, (manifest,))


def test_source_manifest_rejects_duplicate_enumerated_path(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    row = {"path": "src/inference/model.py", "sha256": "a" * 64}
    manifest = write_source_manifest(repository, files=[row, row])

    with pytest.raises(AttestationError, match="duplicate"):
        build_changed_file_manifest(repository, (manifest,))


def test_present_relevant_symlink_is_rejected(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)
    model = repository / "src" / "inference" / "model.py"
    model.unlink()
    model.symlink_to(repository / "docs" / "notes.md")

    with pytest.raises(AttestationError, match="regular"):
        build_changed_file_manifest(repository, (manifest,))


def test_enumerated_untracked_path_remains_classified_untracked(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    path = repository / "src" / "inference" / "new.py"
    manifest = write_source_manifest(
        repository,
        files=[{"path": "src/inference/new.py", "sha256": "a" * 64}],
    )
    path.write_text("NEW = 1\n", encoding="utf-8")

    assert build_changed_file_manifest(repository, (manifest,)) == (
        ChangedFile(
            path="src/inference/new.py",
            status="untracked",
            sha256=sha256_hex(b"NEW = 1\n"),
        ),
    )


@pytest.mark.parametrize("swap_kind", ["ancestor", "leaf"])
def test_source_manifest_read_is_anchored_against_path_swaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_kind: str,
) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository, "manifests/source.json")
    (repository / "src" / "inference" / "model.py").write_text(
        "MODEL = 2\n",
        encoding="utf-8",
    )
    attacker = tmp_path / "attacker-manifests"
    attacker.mkdir()
    attacker_manifest = attacker / "source.json"
    attacker_manifest.write_bytes(
        canonical_json_bytes(
            {
                "covered_roots": ["docs"],
                "files": [],
                "schema": "crux.source-manifest/v1",
            },
            trailing_newline=True,
        )
    )
    real_open = os.open
    swapped = False

    def swap_during_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        rendered = os.fspath(path)
        is_old_open = rendered == str(manifest)
        is_anchored_ancestor = rendered == "manifests" and dir_fd is not None
        is_anchored_leaf = rendered == "source.json" and dir_fd is not None
        if not swapped and swap_kind == "ancestor" and (is_old_open or is_anchored_ancestor):
            if is_anchored_ancestor:
                descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            pinned = repository / "manifests-pinned"
            manifest.parent.rename(pinned)
            manifest.parent.symlink_to(attacker, target_is_directory=True)
            swapped = True
            if is_anchored_ancestor:
                return descriptor
        if not swapped and swap_kind == "leaf" and (is_old_open or is_anchored_leaf):
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            saved = manifest.with_name("source-original.json")
            manifest.rename(saved)
            manifest.write_bytes(attacker_manifest.read_bytes())
            swapped = True
            return descriptor
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_during_open)

    changed = build_changed_file_manifest(repository, (manifest,))

    assert swapped
    assert changed == (
        ChangedFile(
            "src/inference/model.py",
            "modified",
            sha256_hex(b"MODEL = 2\n"),
        ),
    )


@pytest.mark.parametrize("swap_kind", ["ancestor", "leaf"])
def test_changed_file_hash_is_anchored_against_path_swaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_kind: str,
) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)
    changed_path = repository / "src" / "inference" / "model.py"
    changed_path.write_text("MODEL = 2\n", encoding="utf-8")
    attacker = tmp_path / "attacker-inference"
    attacker.mkdir()
    (attacker / "model.py").write_text("ATTACKER = 1\n", encoding="utf-8")
    real_open = os.open
    swapped = False

    def swap_during_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        rendered = os.fspath(path)
        is_old_open = rendered == str(changed_path)
        is_anchored_ancestor = rendered == "inference" and dir_fd is not None
        is_anchored_leaf = rendered == "model.py" and dir_fd is not None
        if not swapped and swap_kind == "ancestor" and (is_old_open or is_anchored_ancestor):
            if is_anchored_ancestor:
                descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            pinned = repository / "src" / "inference-pinned"
            changed_path.parent.rename(pinned)
            changed_path.parent.symlink_to(attacker, target_is_directory=True)
            swapped = True
            if is_anchored_ancestor:
                return descriptor
        if not swapped and swap_kind == "leaf" and (is_old_open or is_anchored_leaf):
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
            saved = changed_path.with_name("model-original.py")
            changed_path.rename(saved)
            changed_path.write_bytes(b"ATTACKER = 1\n")
            swapped = True
            return descriptor
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_during_open)

    changed = build_changed_file_manifest(repository, (manifest,))

    assert swapped
    assert changed == (
        ChangedFile(
            "src/inference/model.py",
            "modified",
            sha256_hex(b"MODEL = 2\n"),
        ),
    )


def test_non_utf8_git_status_path_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)
    real_run = subprocess.run

    def invalid_status_run(arguments: list[str], **kwargs: object) -> object:
        if "status" in arguments:
            return SimpleNamespace(stdout=b"?? src/inference/\xff.py\0")
        return real_run(arguments, **kwargs)  # pylint: disable=subprocess-run-check

    monkeypatch.setattr(backend_attestation.subprocess, "run", invalid_status_run)

    with pytest.raises(AttestationError, match="UTF-8"):
        build_changed_file_manifest(repository, (manifest,))


def test_strict_mode_rejects_inference_relevant_change(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)
    (repository / "src" / "inference" / "model.py").write_text("MODEL = 2\n", encoding="utf-8")

    with pytest.raises(AttestationError, match="strict"):
        publish_execution_attestation(
            repository,
            repository / "artifacts" / "backend",
            backend_id="magenta-egmd-tf1-94529798-8hit-v1",
            descriptor_sha256="a" * 64,
            source_manifests=(manifest,),
            strict_mode=True,
            conditions=OAF_CONDITIONS,
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert not (repository / "artifacts").exists()


def test_attestation_tracks_whole_checkout_dirty_but_null_relevant_manifest(
    tmp_path: Path,
) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)
    (repository / "docs" / "notes.md").write_text("dirty docs\n", encoding="utf-8")

    published = publish_execution_attestation(
        repository,
        repository / "artifacts" / "backend",
        backend_id="magenta-egmd-tf1-94529798-8hit-v1",
        descriptor_sha256="a" * 64,
        source_manifests=(manifest,),
        strict_mode=False,
        conditions=OAF_CONDITIONS,
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )
    payload = strict_json_loads(published.path.read_bytes()[:-1], require_canonical=True)

    assert isinstance(payload, dict)
    assert payload["git_commit"] == git(repository, "rev-parse", "HEAD").decode().strip()
    assert payload["checkout_dirty"] is True
    assert payload["changed_files_manifest"] is None


def test_changed_file_and_attestation_are_both_immutable_canonical_artifacts(
    tmp_path: Path,
) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)
    changed_path = repository / "src" / "inference" / "model.py"
    changed_path.write_text("MODEL = 2\n", encoding="utf-8")
    backend_root = repository / "artifacts" / "backend"

    published = publish_execution_attestation(
        repository,
        backend_root,
        backend_id="magenta-egmd-tf1-94529798-8hit-v1",
        descriptor_sha256="a" * 64,
        source_manifests=(manifest,),
        strict_mode=False,
        conditions=OAF_CONDITIONS,
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )
    payload = strict_json_loads(published.path.read_bytes()[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    changed_reference = payload["changed_files_manifest"]
    assert isinstance(changed_reference, dict)
    changed_artifact = repository / str(changed_reference["path"])
    expected_changed = canonical_json_bytes(
        [
            {
                "path": "src/inference/model.py",
                "sha256": sha256_hex(b"MODEL = 2\n"),
                "status": "modified",
            }
        ],
        trailing_newline=True,
    )

    assert changed_artifact.read_bytes() == expected_changed
    assert changed_reference["sha256"] == sha256_hex(expected_changed)
    assert changed_artifact == (
        backend_root
        / "attestations"
        / "changed-files"
        / "sha256"
        / f"{sha256_hex(expected_changed)}.json"
    )
    assert published.path == (
        backend_root
        / "attestations"
        / "20260727T010203456789Z-12345678-1234-4678-9234-567812345678.json"
    )
    assert set(payload) == {
        "schema",
        "backend_id",
        "descriptor_sha256",
        "git_commit",
        "checkout_dirty",
        "strict_mode",
        "changed_files_manifest",
        "cpu_limit",
        "memory_bytes",
        "pid_limit",
        "tmp_bytes",
        "shm_bytes",
        "startup_deadline_seconds",
        "request_deadline_seconds",
    }
    assert published.sha256 == sha256_hex(published.path.read_bytes())


def test_heuristic_conditions_allow_null_container_resources(tmp_path: Path) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)
    conditions = ExecutionConditions(
        cpu_limit=None,
        memory_bytes=None,
        pid_limit=None,
        tmp_bytes=None,
        shm_bytes=None,
        startup_deadline_seconds=30,
        request_deadline_seconds=90,
    )

    published = publish_execution_attestation(
        repository,
        repository / "artifacts" / "heuristic",
        backend_id="heuristic-onset-v1",
        descriptor_sha256="b" * 64,
        source_manifests=(manifest,),
        strict_mode=True,
        conditions=conditions,
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )
    payload = strict_json_loads(published.path.read_bytes()[:-1])
    assert isinstance(payload, dict)
    assert [
        payload[field]
        for field in (
            "cpu_limit",
            "memory_bytes",
            "pid_limit",
            "tmp_bytes",
            "shm_bytes",
        )
    ] == [None, None, None, None, None]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpu_limit", None),
        ("cpu_limit", "0"),
        ("memory_bytes", None),
        ("memory_bytes", 0),
        ("pid_limit", None),
        ("pid_limit", -1),
        ("tmp_bytes", None),
        ("tmp_bytes", 0),
        ("shm_bytes", None),
        ("shm_bytes", 0),
        ("startup_deadline_seconds", 0),
        ("request_deadline_seconds", -1),
    ],
)
def test_oaf_conditions_require_positive_resources_and_deadlines(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    repository = create_repository(tmp_path)
    manifest = write_source_manifest(repository)
    values = dict(OAF_CONDITIONS.__dict__)
    values[field] = value

    with pytest.raises((AttestationError, ValueError), match="positive|required"):
        publish_execution_attestation(
            repository,
            repository / "artifacts" / "backend",
            backend_id="magenta-egmd-tf1-94529798-8hit-v1",
            descriptor_sha256="a" * 64,
            source_manifests=(manifest,),
            strict_mode=False,
            conditions=ExecutionConditions(**values),  # type: ignore[arg-type]
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )
