"""Coverage for the authenticated IDM model freeze CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.benchmark.backend_identity import IDM_RELEASE_COMMIT
from src.cli.freeze_idm_model import (
    FreezeError,
    _build_parser,
    _publish_model_inputs,
    _read_git_blob,
    _read_regular,
    _require_path,
    _resolve_device,
    _resolve_dtype,
    _runtime_python_version,
    _verify_license_evidence,
    _verify_model_config,
    _verify_source_revision,
    freeze_model,
    main,
)

_LICENSE_NOTE = (
    "No separate checkpoint notice exists in the pinned repository; this records repository-level "
    "provenance and is not an independent legal conclusion."
)

VALID_CONFIG_YAML = b"""\
sampling_rate: 44100
encoder:
  sampling_rate: 44100
  transform:
    sample_rate: 44100
    n_fft: 1024
    hop_length: 256
    n_mels: 128
  transcription_head:
    onset_activation: none
    velocity_activation: exp_sigmoid
decoder:
  sampling_rate: 44100
train_classes: [CY_CR, CY_RD, HH_CHH, HH_OHH, KD, SD, TT_HFT, TT_HMT, TT_LMT]
"""

LICENSE_BYTES = b"Apache License\nVersion 2.0\n"
CHECKPOINT_BYTES = b"synthetic-checkpoint-bytes"


def _write_provenance(source_commit: str = IDM_RELEASE_COMMIT) -> bytes:
    return (
        json.dumps(
            {
                "source_commit": source_commit,
                "package_name": "inverse-drum-machine",
                "package_version": "0.1.0",
                "license_basis": {
                    "repository_code": "Apache-2.0",
                    "checkpoint_provenance": "repository-level Apache-2.0",
                    "checkpoint_notice_found": False,
                    "note": _LICENSE_NOTE,
                },
            }
        ).encode("utf-8")
        + b"\n"
    )


def _make_fake_python(tmp_path: Path, version: str = "Python 3.11.0") -> Path:
    script = tmp_path / "fake-python"
    script.write_text(
        f'#!/bin/sh\nif [ "$1" = "--version" ]; then\n  echo "{version}"\nfi\n',
        encoding="utf-8",
    )
    os.chmod(script, 0o755)
    return script


def _git_env() -> dict[str, str]:
    """Isolate test git from user and system configuration."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }


def _init_git_repo(source_root: Path) -> str:
    """Create a git repo with the required IDM files and return the commit SHA."""
    config_dir = source_root / "pretrained" / "idm-44-train-kits" / "checkpoints"
    config_dir.mkdir(parents=True)
    (config_dir / "model.yaml").write_bytes(VALID_CONFIG_YAML)
    (config_dir / "val-epoch=518-global_step=0.ckpt").write_bytes(CHECKPOINT_BYTES)
    (source_root / "LICENSE").write_bytes(LICENSE_BYTES)

    env = _git_env()
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True, env=env)
    subprocess.run(["git", "add", "."], cwd=source_root, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "test"], cwd=source_root, check=True, env=env)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    return sha


# --- _require_path ---


def test_require_path_returns_path() -> None:
    path = Path("/tmp/test")
    assert _require_path(path, "field") is path


def test_require_path_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="field must be a Path"):
        _require_path("not-a-path", "field")  # type: ignore[arg-type]


# --- _resolve_dtype ---


def test_resolve_dtype_accepts_auto() -> None:
    assert _resolve_dtype("auto") == "float32"


def test_resolve_dtype_accepts_float32() -> None:
    assert _resolve_dtype("float32") == "float32"


def test_resolve_dtype_rejects_invalid() -> None:
    with pytest.raises(FreezeError, match="dtype must be auto or float32"):
        _resolve_dtype("float16")


# --- _resolve_device ---


def test_resolve_device_accepts_cpu() -> None:
    assert _resolve_device("cpu", Path("/usr/bin/python3")) == "cpu"


def test_resolve_device_rejects_non_cpu_non_auto() -> None:
    with pytest.raises(FreezeError, match="device must be auto or cpu"):
        _resolve_device("cuda", Path("/usr/bin/python3"))


def test_resolve_device_auto_probes_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_python = _make_fake_python(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: "cpu\n",
    )
    assert _resolve_device("auto", fake_python) == "cpu"


def test_resolve_device_auto_rejects_non_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_python = _make_fake_python(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: "cuda\n",
    )
    with pytest.raises(FreezeError, match="IDM KISS runtime supports only CPU"):
        _resolve_device("auto", fake_python)


def test_resolve_device_auto_rejects_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_python = _make_fake_python(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["python"])),
    )
    with pytest.raises(FreezeError, match="runtime device feasibility probe failed"):
        _resolve_device("auto", fake_python)


# --- _runtime_python_version ---


def test_runtime_python_version_extracts_311(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_python = _make_fake_python(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: "Python 3.11.12\n",
    )
    assert _runtime_python_version(fake_python) == "3.11.12"


def test_runtime_python_version_rejects_non_311(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_python = _make_fake_python(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: "Python 3.12.0\n",
    )
    with pytest.raises(FreezeError, match="isolated runtime Python must be 3.11.x"):
        _runtime_python_version(fake_python)


def test_runtime_python_version_rejects_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_python = _make_fake_python(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["python"])),
    )
    with pytest.raises(FreezeError, match="isolated runtime Python is unavailable"):
        _runtime_python_version(fake_python)


# --- _read_regular ---


def test_read_regular_returns_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"hello\n")
    assert _read_regular(path, "test file") == b"hello\n"


def test_read_regular_wraps_os_error(tmp_path: Path) -> None:
    with pytest.raises(FreezeError, match="missing file is unavailable"):
        _read_regular(tmp_path / "nonexistent", "missing file")


# --- _read_git_blob ---


def test_read_git_blob_returns_blob_bytes(tmp_path: Path) -> None:
    sha = _init_git_repo(tmp_path)
    blob = _read_git_blob(tmp_path, sha, "LICENSE")
    assert blob == LICENSE_BYTES


def test_read_git_blob_wraps_failure(tmp_path: Path) -> None:
    with pytest.raises(FreezeError, match="pinned source file is unavailable"):
        _read_git_blob(tmp_path, "0" * 40, "LICENSE")


# --- _verify_source_revision ---


def test_verify_source_revision_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = _init_git_repo(tmp_path)
    monkeypatch.setattr("src.cli.freeze_idm_model.IDM_RELEASE_COMMIT", sha)
    assert _verify_source_revision(tmp_path) == sha


def test_verify_source_revision_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(FreezeError, match="pinned source root is unavailable"):
        _verify_source_revision(link)


def test_verify_source_revision_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file"
    file_path.write_text("data", encoding="utf-8")
    with pytest.raises(FreezeError, match="pinned source root is unavailable"):
        _verify_source_revision(file_path)


def test_verify_source_revision_rejects_non_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.cli.freeze_idm_model.IDM_RELEASE_COMMIT", "a" * 40)
    with pytest.raises(FreezeError, match="pinned source revision is unavailable"):
        _verify_source_revision(tmp_path)


def test_verify_source_revision_rejects_revision_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_git_repo(tmp_path)
    monkeypatch.setattr("src.cli.freeze_idm_model.IDM_RELEASE_COMMIT", "b" * 40)
    with pytest.raises(FreezeError, match="source revision mismatch"):
        _verify_source_revision(tmp_path)


def test_git_env_isolates_user_and_system_git_config() -> None:
    env = _git_env()
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"


def test_freeze_subprocess_calls_are_bounded_and_wrap_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every freeze subprocess needs a timeout and TimeoutExpired becomes FreezeError."""
    fake_python = _make_fake_python(tmp_path)
    seen: list[dict[str, object]] = []

    def fake_check_output(cmd, **kwargs):
        seen.append(kwargs)
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    cases = [
        (lambda: _verify_source_revision(tmp_path), "pinned source revision is unavailable"),
        (lambda: _resolve_device("auto", fake_python), "runtime device feasibility probe failed"),
        (lambda: _runtime_python_version(fake_python), "isolated runtime Python is unavailable"),
        (
            lambda: _read_git_blob(tmp_path, "0" * 40, "LICENSE"),
            "pinned source file is unavailable: LICENSE",
        ),
    ]
    for caller, match in cases:
        with pytest.raises(FreezeError, match=match):
            caller()
    assert len(seen) == 4
    assert all(kwargs.get("timeout") is not None for kwargs in seen)


# --- _verify_model_config ---


def test_verify_model_config_accepts_valid() -> None:
    _verify_model_config(VALID_CONFIG_YAML)


def test_verify_model_config_rejects_non_dict() -> None:
    with pytest.raises(FreezeError, match="model config must be a mapping"):
        _verify_model_config(b"- item\n- item2\n")


def test_verify_model_config_rejects_missing_keys() -> None:
    with pytest.raises(FreezeError, match="model config facts differ"):
        _verify_model_config(b"sampling_rate: 44100\n")


def test_verify_model_config_rejects_wrong_sampling_rate() -> None:
    config = VALID_CONFIG_YAML.replace(b"sampling_rate: 44100", b"sampling_rate: 22050", 1)
    with pytest.raises(FreezeError, match="model config facts differ"):
        _verify_model_config(config)


def test_verify_model_config_rejects_wrong_train_classes() -> None:
    config = VALID_CONFIG_YAML.replace(
        b"train_classes: [CY_CR, CY_RD, HH_CHH, HH_OHH, KD, SD, TT_HFT, TT_HMT, TT_LMT]",
        b"train_classes: [KD, SD]",
    )
    with pytest.raises(FreezeError, match="model config facts differ"):
        _verify_model_config(config)


def test_verify_model_config_rejects_wrong_activation() -> None:
    config = VALID_CONFIG_YAML.replace(b"onset_activation: none", b"onset_activation: sigmoid")
    with pytest.raises(FreezeError, match="model config facts differ"):
        _verify_model_config(config)


def test_verify_model_config_rejects_invalid_yaml() -> None:
    with pytest.raises(FreezeError, match="model config YAML is unavailable"):
        _verify_model_config(b"{{invalid yaml")


def test_verify_model_config_rejects_missing_pyyaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(FreezeError, match="PyYAML is unavailable while verifying model config"):
        _verify_model_config(VALID_CONFIG_YAML)


# --- _verify_license_evidence ---


def test_verify_license_evidence_accepts_valid(tmp_path: Path) -> None:
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_bytes(_write_provenance())
    _verify_license_evidence(LICENSE_BYTES, provenance_path)


def test_verify_license_evidence_rejects_non_apache_license(tmp_path: Path) -> None:
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_bytes(_write_provenance())
    with pytest.raises(FreezeError, match="source code license evidence is not Apache-2.0"):
        _verify_license_evidence(b"MIT License\n", provenance_path)


def test_verify_license_evidence_rejects_missing_version(tmp_path: Path) -> None:
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_bytes(_write_provenance())
    with pytest.raises(FreezeError, match="source code license evidence is not Apache-2.0"):
        _verify_license_evidence(b"Apache License\n", provenance_path)


def test_verify_license_evidence_rejects_missing_provenance(tmp_path: Path) -> None:
    with pytest.raises(FreezeError, match="license provenance is unavailable"):
        _verify_license_evidence(LICENSE_BYTES, tmp_path / "nonexistent.json")


def test_verify_license_evidence_rejects_invalid_json(tmp_path: Path) -> None:
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_bytes(b"not json\n")
    with pytest.raises(FreezeError, match="IDM checkpoint license provenance is unavailable"):
        _verify_license_evidence(LICENSE_BYTES, provenance_path)


def test_verify_license_evidence_rejects_non_dict_provenance(tmp_path: Path) -> None:
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_bytes(b"[]\n")
    with pytest.raises(FreezeError, match="IDM checkpoint license provenance is unavailable"):
        _verify_license_evidence(LICENSE_BYTES, provenance_path)


def test_verify_license_evidence_rejects_non_dict_basis(tmp_path: Path) -> None:
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_bytes(
        json.dumps({"source_commit": IDM_RELEASE_COMMIT, "license_basis": "not-a-dict"}).encode()
        + b"\n"
    )
    with pytest.raises(FreezeError, match="IDM checkpoint license provenance is unavailable"):
        _verify_license_evidence(LICENSE_BYTES, provenance_path)


def test_verify_license_evidence_rejects_wrong_source_commit(tmp_path: Path) -> None:
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_bytes(_write_provenance("b" * 40))
    with pytest.raises(FreezeError, match="IDM checkpoint license evidence is contradictory"):
        _verify_license_evidence(LICENSE_BYTES, provenance_path)


def test_verify_license_evidence_rejects_wrong_note(tmp_path: Path) -> None:
    provenance = json.loads(_write_provenance()[:-1])
    provenance["license_basis"]["note"] = "wrong note"
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_bytes((json.dumps(provenance) + "\n").encode())
    with pytest.raises(FreezeError, match="IDM checkpoint license evidence is contradictory"):
        _verify_license_evidence(LICENSE_BYTES, provenance_path)


def test_verify_license_evidence_rejects_checkpoint_notice_found_true(
    tmp_path: Path,
) -> None:
    provenance = json.loads(_write_provenance()[:-1])
    provenance["license_basis"]["checkpoint_notice_found"] = True
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_bytes((json.dumps(provenance) + "\n").encode())
    with pytest.raises(FreezeError, match="IDM checkpoint license evidence is contradictory"):
        _verify_license_evidence(LICENSE_BYTES, provenance_path)


# --- _build_parser ---


def test_build_parser_requires_source_root() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_parses_all_args() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--source-root",
            "/tmp/source",
            "--runtime-lock",
            "/tmp/runtime/uv.lock",
            "--runtime-python",
            "/tmp/python",
            "--model-root",
            "/tmp/model",
            "--license-provenance",
            "/tmp/provenance.json",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--output",
            "/tmp/model.json",
        ]
    )
    assert args.source_root == Path("/tmp/source")
    assert args.runtime_lock == Path("/tmp/runtime/uv.lock")
    assert args.runtime_python == Path("/tmp/python")
    assert args.model_root == Path("/tmp/model")
    assert args.license_provenance == Path("/tmp/provenance.json")
    assert args.device == "cpu"
    assert args.dtype == "float32"
    assert args.output == Path("/tmp/model.json")


def test_build_parser_uses_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--source-root", "/tmp/source"])
    assert args.runtime_lock == Path("runtime/idm/uv.lock")
    assert args.runtime_python is None
    assert args.model_root is None
    assert args.license_provenance is None
    assert args.device == "auto"
    assert args.dtype == "auto"
    assert args.output == Path("runtime/idm/model.json")


# --- freeze_model integration tests ---


def _setup_freeze_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_root_separate: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    """Set up a full freeze environment.

    Returns (source_root, runtime_lock_path, output, model_root, runtime_python).
    """
    source_root = tmp_path / "source"
    source_root.mkdir()
    sha = _init_git_repo(source_root)

    # Monkeypatch IDM_RELEASE_COMMIT in both modules that imported it
    monkeypatch.setattr("src.cli.freeze_idm_model.IDM_RELEASE_COMMIT", sha)
    monkeypatch.setattr("src.benchmark.idm_model.IDM_RELEASE_COMMIT", sha)

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    runtime_lock_path = runtime_dir / "uv.lock"
    runtime_lock_path.write_bytes(b"fake uv lock content\n")
    provenance_path = runtime_dir / "idm-wheel-provenance.json"
    provenance_path.write_bytes(_write_provenance(sha))

    fake_python = _make_fake_python(tmp_path)
    output = tmp_path / "model.json"

    if model_root_separate:
        model_root = tmp_path / "model-root"
        model_root.mkdir()
    else:
        model_root = source_root

    return source_root, runtime_lock_path, output, model_root, fake_python


def test_freeze_model_succeeds_with_model_root_equal_to_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    lock = freeze_model(
        source_root=source_root,
        runtime_lock_path=runtime_lock_path,
        output=output,
        model_root=model_root,
        runtime_python=fake_python,
        license_provenance_path=runtime_lock_path.parent / "idm-wheel-provenance.json",
        device="cpu",
        dtype="float32",
    )
    assert output.exists()
    assert lock.model_id.startswith("idm-44-train-kits-")
    # The output file should be loadable
    from src.benchmark.idm_model import load_idm_model_lock

    reloaded = load_idm_model_lock(output)
    assert reloaded.model_id == lock.model_id


def test_freeze_model_succeeds_with_separate_model_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch, model_root_separate=True
    )
    lock = freeze_model(
        source_root=source_root,
        runtime_lock_path=runtime_lock_path,
        output=output,
        model_root=model_root,
        runtime_python=fake_python,
        license_provenance_path=runtime_lock_path.parent / "idm-wheel-provenance.json",
        device="cpu",
        dtype="float32",
    )
    assert lock.model_id.startswith("idm-44-train-kits-")
    # The model files should have been published to model_root
    config_path = model_root / "pretrained" / "idm-44-train-kits" / "checkpoints" / "model.yaml"
    assert config_path.exists()
    assert config_path.read_bytes() == VALID_CONFIG_YAML


def test_freeze_model_rejects_symlink_source_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    link = tmp_path / "link"
    link.symlink_to(source_root)
    with pytest.raises(FreezeError, match="pinned source root is unavailable"):
        freeze_model(
            source_root=link,
            runtime_lock_path=runtime_lock_path,
            output=output,
            model_root=model_root,
            runtime_python=fake_python,
            device="cpu",
            dtype="float32",
        )


def test_freeze_model_rejects_symlink_runtime_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    link = tmp_path / "link.lock"
    link.symlink_to(runtime_lock_path)
    with pytest.raises(FreezeError, match="runtime lock is unavailable"):
        freeze_model(
            source_root=source_root,
            runtime_lock_path=link,
            output=output,
            model_root=model_root,
            runtime_python=fake_python,
            device="cpu",
            dtype="float32",
        )


def test_freeze_model_rejects_revision_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    # Override to a wrong commit
    monkeypatch.setattr("src.cli.freeze_idm_model.IDM_RELEASE_COMMIT", "c" * 40)
    with pytest.raises(FreezeError, match="source revision mismatch"):
        freeze_model(
            source_root=source_root,
            runtime_lock_path=runtime_lock_path,
            output=output,
            model_root=model_root,
            runtime_python=fake_python,
            device="cpu",
            dtype="float32",
        )


def test_freeze_model_rejects_invalid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    # Tamper with the config file in the working tree (git blob stays valid)
    config_path = source_root / "pretrained" / "idm-44-train-kits" / "checkpoints" / "model.yaml"
    config_path.write_bytes(b"invalid: true\n")
    # Re-commit so the git blob matches the tampered content
    env = _git_env()
    subprocess.run(["git", "add", "."], cwd=source_root, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "tamper"], cwd=source_root, check=True, env=env)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    monkeypatch.setattr("src.cli.freeze_idm_model.IDM_RELEASE_COMMIT", sha)
    monkeypatch.setattr("src.benchmark.idm_model.IDM_RELEASE_COMMIT", sha)
    # Update provenance to match new commit
    provenance_path = runtime_lock_path.parent / "idm-wheel-provenance.json"
    provenance_path.write_bytes(_write_provenance(sha))

    with pytest.raises(FreezeError, match="model config facts differ"):
        freeze_model(
            source_root=source_root,
            runtime_lock_path=runtime_lock_path,
            output=output,
            model_root=model_root,
            runtime_python=fake_python,
            license_provenance_path=provenance_path,
            device="cpu",
            dtype="float32",
        )


def test_freeze_model_rejects_non_apache_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    # Replace LICENSE with non-Apache and re-commit
    (source_root / "LICENSE").write_bytes(b"MIT License\n")
    env = _git_env()
    subprocess.run(["git", "add", "."], cwd=source_root, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "mit"], cwd=source_root, check=True, env=env)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    monkeypatch.setattr("src.cli.freeze_idm_model.IDM_RELEASE_COMMIT", sha)
    monkeypatch.setattr("src.benchmark.idm_model.IDM_RELEASE_COMMIT", sha)
    provenance_path = runtime_lock_path.parent / "idm-wheel-provenance.json"
    provenance_path.write_bytes(_write_provenance(sha))

    with pytest.raises(FreezeError, match="source code license evidence is not Apache-2.0"):
        freeze_model(
            source_root=source_root,
            runtime_lock_path=runtime_lock_path,
            output=output,
            model_root=model_root,
            runtime_python=fake_python,
            license_provenance_path=provenance_path,
            device="cpu",
            dtype="float32",
        )


def test_freeze_model_rejects_invalid_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    with pytest.raises(FreezeError, match="device must be auto or cpu"):
        freeze_model(
            source_root=source_root,
            runtime_lock_path=runtime_lock_path,
            output=output,
            model_root=model_root,
            runtime_python=fake_python,
            device="cuda",
            dtype="float32",
        )


def test_freeze_model_rejects_invalid_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    with pytest.raises(FreezeError, match="dtype must be auto or float32"):
        freeze_model(
            source_root=source_root,
            runtime_lock_path=runtime_lock_path,
            output=output,
            model_root=model_root,
            runtime_python=fake_python,
            device="cpu",
            dtype="float16",
        )


def test_freeze_model_rejects_wrong_python_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, _ = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    fake_python_312 = _make_fake_python(tmp_path, version="Python 3.12.0")
    with pytest.raises(FreezeError, match="isolated runtime Python must be 3.11.x"):
        freeze_model(
            source_root=source_root,
            runtime_lock_path=runtime_lock_path,
            output=output,
            model_root=model_root,
            runtime_python=fake_python_312,
            device="cpu",
            dtype="float32",
        )


def test_freeze_model_rejects_non_path_args(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="source_root must be a Path"):
        freeze_model(
            source_root="not-a-path",  # type: ignore[arg-type]
            runtime_lock_path=tmp_path / "uv.lock",
            output=tmp_path / "out.json",
        )


def test_freeze_model_auto_device_probes_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    original_check_output = subprocess.check_output

    def mock_check_output(cmd, **kwargs):
        if isinstance(cmd, list) and cmd and str(cmd[0]) == str(fake_python):
            if len(cmd) > 1 and cmd[1] == "--version":
                return "Python 3.11.0\n"
            return "cpu\n"
        return original_check_output(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)
    lock = freeze_model(
        source_root=source_root,
        runtime_lock_path=runtime_lock_path,
        output=output,
        model_root=model_root,
        runtime_python=fake_python,
        license_provenance_path=runtime_lock_path.parent / "idm-wheel-provenance.json",
        device="auto",
        dtype="float32",
    )
    assert lock.device == "cpu"


def test_freeze_model_rejects_symlink_model_root(tmp_path: Path) -> None:
    real_model_root = tmp_path / "real-model-root"
    real_model_root.mkdir()
    symlink_model_root = tmp_path / "symlink-model-root"
    symlink_model_root.symlink_to(real_model_root)
    with pytest.raises(FreezeError, match="model root is unavailable"):
        _publish_model_inputs(symlink_model_root, VALID_CONFIG_YAML, CHECKPOINT_BYTES)


def test_freeze_model_rejects_derived_model_id_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        "src.cli.freeze_idm_model.derive_idm_model_id",
        lambda lock: "wrong-id",
    )
    with pytest.raises(FreezeError, match="derived model ID does not match the frozen lock"):
        freeze_model(
            source_root=source_root,
            runtime_lock_path=runtime_lock_path,
            output=output,
            model_root=model_root,
            runtime_python=fake_python,
            license_provenance_path=runtime_lock_path.parent / "idm-wheel-provenance.json",
            device="cpu",
            dtype="float32",
        )


# --- main() ---


def test_main_succeeds_and_prints_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    source_root, runtime_lock_path, output, model_root, fake_python = _setup_freeze_environment(
        tmp_path, monkeypatch
    )
    provenance_path = runtime_lock_path.parent / "idm-wheel-provenance.json"
    rc = main(
        [
            "--source-root",
            str(source_root),
            "--runtime-lock",
            str(runtime_lock_path),
            "--runtime-python",
            str(fake_python),
            "--model-root",
            str(model_root),
            "--license-provenance",
            str(provenance_path),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out.strip().startswith("idm-44-train-kits-")


def test_main_returns_1_on_freeze_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _, runtime_lock_path, output, _, _ = _setup_freeze_environment(tmp_path, monkeypatch)
    # Point to a non-existent source root to trigger a FreezeError
    rc = main(
        [
            "--source-root",
            str(tmp_path / "nonexistent"),
            "--runtime-lock",
            str(runtime_lock_path),
            "--output",
            str(output),
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "freeze failed:" in captured.err


def test_main_returns_1_on_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    source_root, runtime_lock_path, output, _, _ = _setup_freeze_environment(tmp_path, monkeypatch)

    def raise_os_error(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("src.cli.freeze_idm_model.freeze_model", raise_os_error)
    rc = main(
        [
            "--source-root",
            str(source_root),
            "--runtime-lock",
            str(runtime_lock_path),
            "--output",
            str(output),
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "freeze failed:" in captured.err
