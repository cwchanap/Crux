from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import venv
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPO_ROOT / "src" / "benchmark" / "separator_environment_probe.py"


def canonical_json_bytes(value: object, *, trailing_newline: bool = False) -> bytes:
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return content + (b"\n" if trailing_newline else b"")


def strict_json_loads(content: bytes, *, require_canonical: bool = False) -> object:
    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        content.decode("utf-8"),
        object_pairs_hook=object_from_pairs,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant: {constant}")
        ),
    )
    if require_canonical and canonical_json_bytes(value) != content:
        raise ValueError("JSON bytes are not canonical")
    return value


def _venv_interpreter(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _distribution_file_digest(content: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")


def _write_distribution(
    purelib: Path,
    package_name: str,
    package_version: str,
) -> None:
    package_dir = purelib / package_name
    package_dir.mkdir()
    init_path = package_dir / "__init__.py"
    init_path.write_text("__version__ = '2.4.2'\n", encoding="utf-8")

    distribution_dir = purelib / f"{package_name}-{package_version}.dist-info"
    distribution_dir.mkdir()
    metadata_path = distribution_dir / "METADATA"
    metadata_path.write_text(
        f"Metadata-Version: 2.1\nName: {package_name}\nVersion: {package_version}\n",
        encoding="utf-8",
    )
    record_path = distribution_dir / "RECORD"
    files = (init_path, metadata_path)
    rows = []
    for file_path in files:
        relative_path = file_path.relative_to(purelib).as_posix()
        content = file_path.read_bytes()
        rows.append(
            [
                relative_path,
                f"sha256={_distribution_file_digest(content)}",
                str(len(content)),
            ]
        )
    rows.append([record_path.relative_to(purelib).as_posix(), "", ""])
    with record_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def _synthetic_environment(
    tmp_path: Path,
    *,
    package_name: str = "spleeter",
    package_version: str = "2.4.2",
) -> tuple[Path, Path]:
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=False).create(environment)
    interpreter = _venv_interpreter(environment)
    purelib = Path(
        subprocess.check_output(
            [
                str(interpreter),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            text=True,
        ).strip()
    )
    _write_distribution(purelib, package_name, package_version)
    return interpreter, purelib


def _run_probe(interpreter: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(interpreter), str(PROBE_PATH)],
        check=False,
        capture_output=True,
    )


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("separator_environment_probe_test", PROBE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in _iter_strings(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _iter_strings(child)]
    return []


def _assert_no_absolute_paths(value: object) -> None:
    assert all(
        not (
            Path(item).is_absolute()
            or PurePosixPath(item).is_absolute()
            or PureWindowsPath(item).is_absolute()
            or bool(PureWindowsPath(item).drive)
        )
        for item in _iter_strings(value)
    )


def test_probe_emits_canonical_manifest_with_synthetic_distribution(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)

    result = _run_probe(interpreter)

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.endswith(b"\n")
    payload = strict_json_loads(result.stdout[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    assert canonical_json_bytes(payload, trailing_newline=True) == result.stdout
    _assert_no_absolute_paths(payload)

    assert payload["schema"] == "crux.separator-environment/v1"
    assert payload["separator_id"] == "spleeter4-drums-v1"
    assert payload["package_name"] == "spleeter"
    assert payload["package_version"] == "2.4.2"
    assert (
        payload["interpreter_sha256"]
        == hashlib.sha256(interpreter.resolve(strict=True).read_bytes()).hexdigest()
    )

    distributions = payload["distributions"]
    assert isinstance(distributions, list)
    synthetic = next(item for item in distributions if item["name"] == "spleeter")
    assert synthetic["version"] == "2.4.2"
    files = {item["path"]: item for item in synthetic["files"]}
    for relative_path in (
        "spleeter/__init__.py",
        "spleeter-2.4.2.dist-info/METADATA",
        "spleeter-2.4.2.dist-info/RECORD",
    ):
        content = (purelib / relative_path).read_bytes()
        assert files[relative_path] == {
            "root": "purelib",
            "path": relative_path,
            "byte_length": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }


def test_probe_ignores_stale_record_digest_and_size(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    record_path = purelib / "spleeter-2.4.2.dist-info" / "RECORD"
    rows = list(csv.reader(record_path.read_text(encoding="utf-8").splitlines()))
    assert rows[0][0] == "spleeter/__init__.py"
    rows[0][1] = "sha256=stale"
    rows[0][2] = "0"
    with record_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)

    result = _run_probe(interpreter)

    assert result.returncode == 0, result.stderr.decode()
    payload = strict_json_loads(result.stdout[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    distribution = next(item for item in payload["distributions"] if item["name"] == "spleeter")
    file = next(item for item in distribution["files"] if item["path"] == "spleeter/__init__.py")
    actual_bytes = (purelib / "spleeter" / "__init__.py").read_bytes()
    assert file["byte_length"] == len(actual_bytes)
    assert file["sha256"] == hashlib.sha256(actual_bytes).hexdigest()
    assert canonical_json_bytes(payload, trailing_newline=True) == result.stdout


def test_probe_reports_changed_declared_member_bytes(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    baseline_result = _run_probe(interpreter)
    assert baseline_result.returncode == 0, baseline_result.stderr.decode()
    baseline = strict_json_loads(baseline_result.stdout[:-1], require_canonical=True)
    assert isinstance(baseline, dict)

    _mutate_distribution_tree(purelib, "record_content_changed")
    result = _run_probe(interpreter)

    assert result.returncode == 0, result.stderr.decode()
    payload = strict_json_loads(result.stdout[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    assert canonical_json_bytes(payload, trailing_newline=True) == result.stdout
    baseline_distribution = next(
        item for item in baseline["distributions"] if item["name"] == "spleeter"
    )
    distribution = next(item for item in payload["distributions"] if item["name"] == "spleeter")
    baseline_file = next(
        item for item in baseline_distribution["files"] if item["path"] == "spleeter/__init__.py"
    )
    file = next(item for item in distribution["files"] if item["path"] == "spleeter/__init__.py")
    actual_bytes = (purelib / "spleeter" / "__init__.py").read_bytes()
    assert file["byte_length"] == len(actual_bytes)
    assert file["sha256"] == hashlib.sha256(actual_bytes).hexdigest()
    assert (file["byte_length"], file["sha256"]) != (
        baseline_file["byte_length"],
        baseline_file["sha256"],
    )


def _mutate_distribution_tree(purelib: Path, mutation: str) -> None:
    mutations = {
        "missing_record_member": lambda: (purelib / "spleeter" / "__init__.py").unlink(),
        "record_content_changed": lambda: (purelib / "spleeter" / "__init__.py").write_text(
            "changed\n", encoding="utf-8"
        ),
        "record_self_changed": lambda: (purelib / "spleeter-2.4.2.dist-info" / "RECORD").write_text(
            "changed\n", encoding="utf-8"
        ),
        "extra_python": lambda: (purelib / "injected.py").write_text("x = 1\n", encoding="utf-8"),
        "extra_pth": lambda: (purelib / "injected.pth").write_text(
            "/tmp/escape\n", encoding="utf-8"
        ),
        "sitecustomize": lambda: (purelib / "sitecustomize.py").write_text(
            "pass\n", encoding="utf-8"
        ),
    }
    if mutation in mutations:
        mutations[mutation]()
        return
    if mutation == "leaf_symlink":
        target = purelib / "missing-target"
        (purelib / "spleeter" / "__init__.py").unlink()
        (purelib / "spleeter" / "__init__.py").symlink_to(target)
        return
    if mutation == "parent_symlink":
        package_dir = purelib / "spleeter"
        shutil.rmtree(package_dir)
        package_dir.symlink_to(purelib / "missing-package", target_is_directory=True)
        return
    raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_record_member",
        "record_self_changed",
        "extra_python",
        "extra_pth",
        "sitecustomize",
        "leaf_symlink",
        "parent_symlink",
    ),
)
def test_probe_rejects_distribution_tree_drift(tmp_path: Path, mutation: str) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    _mutate_distribution_tree(purelib, mutation)

    result = _run_probe(interpreter)

    assert result.returncode != 0
    assert result.stdout == b""


def test_probe_ignores_generated_bytecode(tmp_path: Path) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    baseline = _run_probe(interpreter)
    assert baseline.returncode == 0, baseline.stderr.decode()

    pycache = purelib / "spleeter" / "__pycache__"
    pycache.mkdir()
    (pycache / "module.cpython-313.pyc").write_bytes(b"synthetic bytecode")

    result = _run_probe(interpreter)

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == baseline.stdout


def test_probe_hashes_resolved_interpreter_target(tmp_path: Path) -> None:
    interpreter, _ = _synthetic_environment(tmp_path)
    linked_interpreter = tmp_path / "linked-python"
    linked_interpreter.symlink_to(interpreter)

    result = _run_probe(linked_interpreter)

    assert result.returncode == 0, result.stderr.decode()
    payload = strict_json_loads(result.stdout[:-1], require_canonical=True)
    assert isinstance(payload, dict)
    final_target = interpreter.resolve(strict=True)
    assert payload["interpreter_sha256"] == hashlib.sha256(final_target.read_bytes()).hexdigest()


def _synthetic_root_configuration(tmp_path: Path) -> dict[str, str]:
    tags = (
        "stdlib",
        "platstdlib",
        "purelib",
        "platlib",
        "include",
        "platinclude",
        "scripts",
        "data",
    )
    paths = {tag: tmp_path / tag for tag in tags}
    for path in paths.values():
        path.mkdir(parents=True)
    configuration = {tag: str(path) for tag, path in paths.items()}
    configuration["platstdlib"] = configuration["stdlib"]
    configuration["platlib"] = configuration["purelib"]
    configuration["platinclude"] = configuration["include"]
    return configuration


def test_root_aliases_are_primary_by_role_and_cross_role_collisions_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    configuration = _synthetic_root_configuration(tmp_path)
    monkeypatch.setattr(probe, "_configured_paths", lambda _: configuration)

    roots = probe._canonical_root_paths(Path("/synthetic/python"))

    assert tuple(roots) == ("stdlib", "purelib", "include", "scripts", "data")

    ambiguous = dict(configuration)
    ambiguous["platlib"] = ambiguous["stdlib"]
    monkeypatch.setattr(probe, "_configured_paths", lambda _: ambiguous)
    with pytest.raises(probe._ProbeError):
        probe._canonical_root_paths(Path("/synthetic/python"))


def test_configured_symlink_root_and_missing_no_follow_support_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _load_probe_module()
    configuration = _synthetic_root_configuration(tmp_path / "configured")
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    configuration["purelib"] = str(link)
    monkeypatch.setattr(probe, "_configured_paths", lambda _: configuration)

    with pytest.raises(probe._ProbeError):
        probe._canonical_root_paths(Path("/synthetic/python"))

    monkeypatch.setattr(probe, "_NOFOLLOW", 0)
    with pytest.raises(probe._ProbeError):
        probe._canonical_root_paths(Path("/synthetic/python"))


def test_inventory_carries_first_hash_identity_to_tree_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter, purelib = _synthetic_environment(tmp_path)
    probe = _load_probe_module()
    roots = probe._canonical_root_paths(interpreter)
    descriptors = {}
    identities = {}
    try:
        for tag, path in roots.items():
            descriptor, identity = probe._open_root(path)
            descriptors[tag] = descriptor
            identities[tag] = identity
        distributions = list(probe.importlib.metadata.distributions(path=[str(purelib)]))
        distribution = next(item for item in distributions if item.metadata["Name"] == "spleeter")
        original_hash = probe._hash_relative_file
        mutated = False

        def racing_hash(root_descriptor, parts, *, capture=False):
            nonlocal mutated
            result = original_hash(root_descriptor, parts, capture=capture)
            if not capture and parts == ("spleeter", "__init__.py") and not mutated:
                (purelib / "spleeter" / "__init__.py").write_text("raced\n", encoding="utf-8")
                mutated = True
            return result

        monkeypatch.setattr(probe, "_hash_relative_file", racing_hash)
        item, expected, expected_identities = probe._inventory_distribution(
            distribution, roots, descriptors
        )
        assert item["name"] == "spleeter"
        assert expected_identities
        with pytest.raises(probe._ProbeError):
            probe._walk_tree(
                descriptors["purelib"],
                identities["purelib"],
                "purelib",
                expected,
                expected_identities,
            )
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
