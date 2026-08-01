from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import stat
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

import tools.hpa320.generate_runner_source_manifest as runner_manifest_module
import tools.hpa320.oaf_build_context as context_module
import tools.hpa320.seal_oaf_backend as seal_module
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex, strict_json_loads
from tools.hpa320.generate_runner_source_manifest import SOURCE_PATHS
from tools.hpa320.oaf_build_context import (
    BUILD_CONTEXT_MANIFEST_PATH,
    REVIEWED_REPOSITORY_PATHS,
    BuildContextError,
    generate_build_context_manifest,
    load_build_context_manifest,
    materialize_build_context,
)


def _write_manifest(repository: Path, files: dict[str, bytes]) -> Path:
    rows = [
        {
            "byte_length": len(content),
            "path": path,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for path, content in sorted(files.items(), key=lambda item: item[0].encode("utf-8"))
    ]
    path = repository / BUILD_CONTEXT_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        canonical_json_bytes(
            {
                "directory_mode": 493,
                "file_mode": 420,
                "files": rows,
                "manifest_path": BUILD_CONTEXT_MANIFEST_PATH,
                "schema": "crux.oaf-build-context-manifest/v1",
            },
            trailing_newline=True,
        )
    )
    return path


def _write_context_sources(
    repository: Path,
    wheelhouse: Path,
) -> tuple[dict[str, bytes], Path]:
    files = {
        "runtime/oaf_tf1/Dockerfile": b"FROM scratch\n",
        "runtime/oaf_tf1/entrypoint.py": b"print('runner')\n",
        "runtime/oaf_tf1/wheelhouse/runtime/package.whl": b"wheel bytes\n",
        "tools/hpa320/generate_runner_source_manifest.py": b"print('manifest')\n",
    }
    for relative, content in files.items():
        if relative.startswith("runtime/oaf_tf1/wheelhouse/"):
            path = wheelhouse / relative.removeprefix("runtime/oaf_tf1/wheelhouse/")
        else:
            path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return files, _write_manifest(repository, files)


def test_build_context_loader_is_strict_and_hashes_exact_manifest_bytes(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    files = {"runtime/oaf_tf1/Dockerfile": b"FROM scratch\n"}
    source = repository / "runtime/oaf_tf1/Dockerfile"
    source.parent.mkdir(parents=True)
    source.write_bytes(files[source.relative_to(repository).as_posix()])
    path = _write_manifest(repository, files)

    manifest = load_build_context_manifest(path)

    assert manifest.directory_mode == 0o755
    assert manifest.file_mode == 0o644
    assert manifest.manifest_path == BUILD_CONTEXT_MANIFEST_PATH
    assert manifest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert tuple(file.path for file in manifest.files) == ("runtime/oaf_tf1/Dockerfile",)

    payload = json.loads(path.read_bytes())
    payload["unknown"] = True
    path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))
    with pytest.raises(BuildContextError, match="fields"):
        load_build_context_manifest(path)


@pytest.mark.parametrize(
    "bad_path",
    (
        "../Dockerfile",
        "runtime/oaf_tf1/../Dockerfile",
        "/runtime/oaf_tf1/Dockerfile",
        "runtime\\oaf_tf1\\Dockerfile",
        "runtime/oaf_tf1/__pycache__/oaf_backend.cpython-312.pyc",
        BUILD_CONTEXT_MANIFEST_PATH,
    ),
)
def test_build_context_loader_rejects_aliases_escapes_and_self_reference(
    tmp_path: Path,
    bad_path: str,
) -> None:
    repository = tmp_path / "repository"
    path = _write_manifest(repository, {"runtime/oaf_tf1/Dockerfile": b"FROM scratch\n"})
    payload = json.loads(path.read_bytes())
    payload["files"][0]["path"] = bad_path
    path.write_bytes(canonical_json_bytes(payload, trailing_newline=True))

    with pytest.raises(BuildContextError, match="path|manifest"):
        load_build_context_manifest(path)


def test_build_context_materializer_is_minimal_and_normalized(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    wheelhouse = tmp_path / "wheelhouse"
    files, manifest_path = _write_context_sources(repository, wheelhouse)
    ignored = repository / ".dockerignore"
    ignored.write_text("**\n", encoding="utf-8")
    dockerfile_ignore = repository / "runtime/oaf_tf1/Dockerfile.dockerignore"
    dockerfile_ignore.write_text("ignored\n", encoding="utf-8")
    unlisted = repository / "unlisted.txt"
    unlisted.write_text("unlisted\n", encoding="utf-8")
    for relative in files:
        source = (
            wheelhouse / relative.removeprefix("runtime/oaf_tf1/wheelhouse/")
            if relative.startswith("runtime/oaf_tf1/wheelhouse/")
            else repository / relative
        )
        source.chmod(0o600 if source.name.endswith(".py") else 0o755)
        os.utime(source, ns=(1_000_000_000, 1_000_000_000))

    destination = tmp_path / "staged"
    result = materialize_build_context(
        repository_root=repository,
        wheelhouse_root=wheelhouse,
        manifest_path=manifest_path,
        destination=destination,
    )

    assert result == destination
    staged_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert staged_files == {*files, BUILD_CONTEXT_MANIFEST_PATH}
    assert ".dockerignore" not in staged_files
    assert "runtime/oaf_tf1/Dockerfile.dockerignore" not in staged_files
    for path in destination.rglob("*"):
        metadata = path.stat()
        assert metadata.st_mtime_ns == 0
        assert stat.S_IMODE(metadata.st_mode) == (0o755 if path.is_dir() else 0o644)
    assert destination.stat().st_mtime_ns == 0
    assert stat.S_IMODE(destination.stat().st_mode) == 0o755


def test_build_context_materializer_rejects_linked_or_drifted_inputs(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    wheelhouse = tmp_path / "wheelhouse"
    files, manifest_path = _write_context_sources(repository, wheelhouse)
    source = repository / "runtime/oaf_tf1/entrypoint.py"
    linked = repository / "runtime/oaf_tf1/linked.py"
    os.link(source, linked)

    with pytest.raises(BuildContextError, match="multiply linked"):
        materialize_build_context(
            repository_root=repository,
            wheelhouse_root=wheelhouse,
            manifest_path=manifest_path,
            destination=tmp_path / "linked-context",
        )

    linked.unlink()
    source.write_bytes(files["runtime/oaf_tf1/entrypoint.py"] + b"drift")
    with pytest.raises(BuildContextError, match="hash|size"):
        materialize_build_context(
            repository_root=repository,
            wheelhouse_root=wheelhouse,
            manifest_path=manifest_path,
            destination=tmp_path / "drifted-context",
        )


def test_build_context_materializer_rejects_symlink_and_special_file(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    wheelhouse = tmp_path / "wheelhouse"
    files, manifest_path = _write_context_sources(repository, wheelhouse)
    source = repository / "runtime/oaf_tf1/entrypoint.py"
    source.unlink()
    source.symlink_to(repository / "runtime/oaf_tf1/Dockerfile")

    with pytest.raises(BuildContextError, match="regular|link"):
        materialize_build_context(
            repository_root=repository,
            wheelhouse_root=wheelhouse,
            manifest_path=manifest_path,
            destination=tmp_path / "symlink-context",
        )

    source.unlink()
    os.mkfifo(source)
    with pytest.raises(BuildContextError, match="regular"):
        materialize_build_context(
            repository_root=repository,
            wheelhouse_root=wheelhouse,
            manifest_path=manifest_path,
            destination=tmp_path / "fifo-context",
        )


def test_generate_build_context_ignores_source_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.hpa320.oaf_build_context as context_module

    repository = tmp_path / "repository"
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    source = repository / "runtime/oaf_tf1/Dockerfile"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"FROM scratch\n")
    monkeypatch.setattr(
        context_module,
        "REVIEWED_REPOSITORY_PATHS",
        ("runtime/oaf_tf1/Dockerfile",),
    )
    monkeypatch.setattr(context_module, "REVIEWED_REPOSITORY_ROOTS", ())
    monkeypatch.setattr(context_module, "REVIEWED_WHEELHOUSE_ROOTS", ())

    source.chmod(0o600)
    first = generate_build_context_manifest(
        repository_root=repository,
        wheelhouse_root=wheelhouse,
    )
    source.chmod(0o755)
    second = generate_build_context_manifest(
        repository_root=repository,
        wheelhouse_root=wheelhouse,
    )

    assert first == second


def test_generate_build_context_excludes_generated_python_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.hpa320.oaf_build_context as context_module

    repository = tmp_path / "repository"
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    runtime = repository / "runtime/oaf_tf1"
    runtime.mkdir(parents=True)
    (runtime / "Dockerfile").write_bytes(b"FROM scratch\n")
    cache = runtime / "__pycache__"
    cache.mkdir()
    (cache / "oaf_backend.cpython-312.pyc").write_bytes(b"host-specific cache\n")
    monkeypatch.setattr(context_module, "REVIEWED_REPOSITORY_PATHS", ())
    monkeypatch.setattr(context_module, "REVIEWED_REPOSITORY_ROOTS", ("runtime/oaf_tf1",))
    monkeypatch.setattr(context_module, "REVIEWED_WHEELHOUSE_ROOTS", ())

    content = generate_build_context_manifest(
        repository_root=repository,
        wheelhouse_root=wheelhouse,
    )

    paths = {row["path"] for row in json.loads(content)["files"]}
    assert paths == {"runtime/oaf_tf1/Dockerfile"}


def _configure_minimal_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    source = repository / "runtime/oaf_tf1/Dockerfile"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"FROM scratch\n")
    monkeypatch.setattr(
        context_module,
        "REVIEWED_REPOSITORY_PATHS",
        ("runtime/oaf_tf1/Dockerfile",),
    )
    monkeypatch.setattr(context_module, "REVIEWED_REPOSITORY_ROOTS", ())
    monkeypatch.setattr(context_module, "REVIEWED_WHEELHOUSE_ROOTS", ())
    return repository, wheelhouse, source


def _generate_arguments(repository: Path, wheelhouse: Path, output: Path) -> list[str]:
    return [
        "generate",
        "--repository-root",
        os.fspath(repository),
        "--wheelhouse-root",
        os.fspath(wheelhouse),
        "--output",
        os.fspath(output),
    ]


def test_build_context_generate_default_requires_absent_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, wheelhouse, _source = _configure_minimal_generation(tmp_path, monkeypatch)
    output = tmp_path / "build-context-manifest.json"
    arguments = _generate_arguments(repository, wheelhouse, output)

    assert context_module.main(arguments) == 0
    original = output.read_bytes()
    assert context_module.main(arguments) == 2
    assert output.read_bytes() == original


def test_build_context_generate_cleans_output_when_parent_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, wheelhouse, _source = _configure_minimal_generation(tmp_path, monkeypatch)
    output = tmp_path / "build-context-manifest.json"
    arguments = _generate_arguments(repository, wheelhouse, output)
    real_fsync = context_module._fsync_directory
    sync_calls = 0

    def fail_first_parent_sync(path: Path) -> None:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise OSError("injected parent directory sync failure")
        real_fsync(path)

    monkeypatch.setattr(context_module, "_fsync_directory", fail_first_parent_sync)

    assert context_module.main(arguments) == 2
    # The still-owned output must be rolled back so a retry does not collide
    # with a leftover file whose directory entry was never synchronized.
    assert not output.exists()

    monkeypatch.setattr(context_module, "_fsync_directory", real_fsync)
    assert context_module.main(arguments) == 0
    assert output.exists()


def test_build_context_generate_check_is_exact_and_non_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, wheelhouse, source = _configure_minimal_generation(tmp_path, monkeypatch)
    output = tmp_path / "build-context-manifest.json"
    arguments = _generate_arguments(repository, wheelhouse, output)
    assert context_module.main(arguments) == 0
    original = output.read_bytes()
    original_mtime_ns = 1_234_567_890
    os.utime(output, ns=(original_mtime_ns, original_mtime_ns))

    assert context_module.main([*arguments, "--check"]) == 0
    assert output.read_bytes() == original
    assert output.stat().st_mtime_ns == original_mtime_ns

    source.write_bytes(b"FROM changed\n")
    assert context_module.main([*arguments, "--check"]) == 2
    assert output.read_bytes() == original
    assert output.stat().st_mtime_ns == original_mtime_ns


def test_build_context_generate_replace_updates_only_named_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, wheelhouse, source = _configure_minimal_generation(tmp_path, monkeypatch)
    output = tmp_path / "build-context-manifest.json"
    sibling = tmp_path / "keep.txt"
    sibling.write_bytes(b"keep\n")
    arguments = _generate_arguments(repository, wheelhouse, output)
    assert context_module.main(arguments) == 0
    original_inode = output.stat().st_ino
    source.write_bytes(b"FROM changed\n")

    assert context_module.main([*arguments, "--replace"]) == 0
    assert output.stat().st_ino != original_inode
    assert output.read_bytes() == generate_build_context_manifest(
        repository_root=repository,
        wheelhouse_root=wheelhouse,
    )
    assert sibling.read_bytes() == b"keep\n"
    assert tuple(tmp_path.glob(".build-context-manifest-*")) == ()


def test_runner_source_manifest_check_is_exact_and_non_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    source = repository / "source.py"
    source.parent.mkdir()
    source.write_bytes(b"first\n")
    output = tmp_path / "runner-source-manifest.json"
    monkeypatch.setattr(runner_manifest_module, "SOURCE_PATHS", ("source.py",))
    arguments = [
        "--repository-root",
        os.fspath(repository),
        "--output",
        os.fspath(output),
    ]
    assert runner_manifest_module.main(arguments) == 0
    original = output.read_bytes()
    original_mtime_ns = 1_234_567_890
    os.utime(output, ns=(original_mtime_ns, original_mtime_ns))

    assert runner_manifest_module.main([*arguments, "--check"]) == 0
    assert output.read_bytes() == original
    assert output.stat().st_mtime_ns == original_mtime_ns

    source.write_bytes(b"second\n")
    assert runner_manifest_module.main([*arguments, "--check"]) == 2
    assert output.read_bytes() == original
    assert output.stat().st_mtime_ns == original_mtime_ns


def test_build_context_reviews_native_bootstrap_host_tools() -> None:
    assert {
        ".github/workflows/hpa320-native-bootstrap.yml",
        "tools/hpa320/audit_legacy_tf2_conversion.py",
        "tools/hpa320/github_host_evidence.py",
        "tools/hpa320/oaf_build_context.py",
        "tools/hpa320/oaf_candidate_builder.py",
        "tools/hpa320/oaf_host_attestation.py",
        "tools/hpa320/oaf_native_calibration.py",
        "tools/hpa320/oaf_native_runner.py",
        "tools/hpa320/oaf_oci.py",
        "tools/hpa320/seal_oaf_backend.py",
    }.issubset(REVIEWED_REPOSITORY_PATHS)


def test_native_artifact_tool_is_covered_once_by_each_freeze_manifest() -> None:
    path = "tools/hpa320/oaf_native_artifacts.py"
    assert SOURCE_PATHS.count(path) == 1
    assert REVIEWED_REPOSITORY_PATHS.count(path) == 1


def _copy_bootstrap_request_authority(tmp_path: Path) -> tuple[Path, Path]:
    source_root = Path(__file__).resolve().parents[2]
    repository = tmp_path / "repository"
    request_relative = (
        "config/benchmark/backends/"
        "magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json"
    )
    for relative in (
        request_relative,
        *(relative for relative, _field in seal_module._BOOTSTRAP_HASH_FIELDS),
    ):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, target)
    return repository, repository / request_relative


def test_reissue_bootstrap_request_changes_only_current_cross_hashes(
    tmp_path: Path,
) -> None:
    repository, request_path = _copy_bootstrap_request_authority(tmp_path)
    original = strict_json_loads(request_path.read_bytes()[:-1], require_canonical=True)
    assert isinstance(original, dict)
    sibling = request_path.with_name("keep.json")
    sibling.write_bytes(b"keep\n")
    original_inode = request_path.stat().st_ino
    hash_fields = {field for _relative, field in seal_module._BOOTSTRAP_HASH_FIELDS}
    for relative, field in seal_module._BOOTSTRAP_HASH_FIELDS:
        (repository / relative).write_bytes(f"current {field}\n".encode())

    digest = seal_module.reissue_calibration_bootstrap_request(
        request_path=request_path,
        repository_root=repository,
    )

    reissued = strict_json_loads(request_path.read_bytes()[:-1], require_canonical=True)
    assert isinstance(reissued, dict)
    assert {key: value for key, value in reissued.items() if key not in hash_fields} == {
        key: value for key, value in original.items() if key not in hash_fields
    }
    for relative, field in seal_module._BOOTSTRAP_HASH_FIELDS:
        assert reissued[field] == sha256_hex((repository / relative).read_bytes())
    assert digest == sha256_hex(request_path.read_bytes())
    assert request_path.stat().st_ino != original_inode
    assert stat.S_IMODE(request_path.stat().st_mode) == 0o644
    assert sibling.read_bytes() == b"keep\n"
    assert tuple(request_path.parent.glob(f".{request_path.name}.reissue-*")) == ()


def test_reissue_bootstrap_request_cli_reissues_named_request(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository, request_path = _copy_bootstrap_request_authority(tmp_path)
    relative, field = seal_module._BOOTSTRAP_HASH_FIELDS[0]
    (repository / relative).write_bytes(b"new authority\n")

    assert (
        seal_module.main(
            [
                "reissue-bootstrap-request",
                "--repository-root",
                os.fspath(repository),
                "--request",
                os.fspath(request_path),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "exit_code": 0,
        "report_path": os.fspath(request_path),
        "report_sha256": sha256_hex(request_path.read_bytes()),
        "status": "reissued",
    }
    assert json.loads(request_path.read_bytes())[field] == sha256_hex(
        (repository / relative).read_bytes()
    )


def _stage_copy_sources(dockerfile: str, stage: str, next_stage: str | None) -> tuple[str, ...]:
    body = dockerfile.split(f" AS {stage}", 1)[1]
    if next_stage is not None:
        body = body.split(f" AS {next_stage}", 1)[0]
    logical_lines = body.replace("\\\n", " ").splitlines()
    sources: list[str] = []
    for line in logical_lines:
        if not line.startswith("COPY ") or line.startswith("COPY --from="):
            continue
        arguments = [
            argument for argument in shlex.split(line)[1:] if not argument.startswith("--")
        ]
        sources.extend(arguments[:-1])
    return tuple(sources)


def _copy_source_covers(source: str, path: str) -> bool:
    if source.endswith("/"):
        return path.startswith(source)
    return source == path or fnmatchcase(path, source)


def test_dockerfile_copy_sources_are_manifested_and_runner_sources_reach_both_stages() -> None:
    dockerfile = Path("runtime/oaf_tf1/Dockerfile").read_text(encoding="utf-8")
    manifest = load_build_context_manifest(Path(BUILD_CONTEXT_MANIFEST_PATH))
    manifested = {row.path for row in manifest.files}
    all_sources = _stage_copy_sources(dockerfile, "runtime-build", None)
    for source in all_sources:
        assert any(_copy_source_covers(source, path) for path in manifested), source
    for stage, next_stage in (("test", "runtime"), ("runtime", None)):
        stage_sources = _stage_copy_sources(dockerfile, stage, next_stage)
        for path in SOURCE_PATHS:
            assert any(_copy_source_covers(source, path) for source in stage_sources), (
                stage,
                path,
            )
