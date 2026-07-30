from __future__ import annotations

import hashlib
import importlib
import json
import lzma
import shlex
import struct
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from src.benchmark.input_view import parse_canonical_wav
from tools.hpa320 import resolve_oaf_runtime as resolver
from tools.hpa320.generate_smoke_fixture import (
    FIXED_PARAMETERS,
    SmokeFixtureError,
    generate_smoke_wav,
    lcg_step,
    saturating_add,
    write_smoke_fixture,
)
from tools.hpa320.resolve_oaf_runtime import (
    APPROVED_SDIST_SHA256,
    DISTRIBUTION_BUILD_SCHEMA,
    SDIST_ALLOWLIST,
    DistributionBuildError,
    ResolutionError,
    SdistBuildRecord,
    inspect_wheelhouse,
    render_distribution_build_manifest,
    render_requirements_lock,
    require_reproducible_wheels,
    validate_approved_sdist,
    validate_built_pure_wheel,
    validate_distribution_build_manifest,
    verify_offline_resolution,
    verify_requirement_closure,
)
from tools.hpa320.vendor_magenta import (
    EXPECTED_UPSTREAM_COMMIT,
    VendoringError,
    vendor_magenta,
)


def test_smoke_generator_is_byte_deterministic() -> None:
    first = generate_smoke_wav(FIXED_PARAMETERS)
    second = generate_smoke_wav(FIXED_PARAMETERS)

    assert first == second
    parsed = parse_canonical_wav(first, max_input_audio_frames=44100)
    assert parsed.audio_frame_count == 44100
    assert len(first) == 44 + 44100 * 2


def test_smoke_wav_has_exact_canonical_riff_header() -> None:
    content = generate_smoke_wav(FIXED_PARAMETERS)

    assert content[:44] == (
        b"RIFF"
        + struct.pack("<I", 36 + 44100 * 2)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 44100, 88200, 2, 16)
        + b"data"
        + struct.pack("<I", 44100 * 2)
    )
    assert content[44:] != b"\x00" * (44100 * 2)


def test_smoke_parameters_are_strict_canonical_json() -> None:
    fixture = Path("tests/fixtures/oaf_tf1_smoke/generator-parameters.json")
    raw = fixture.read_bytes()

    assert raw.endswith(b"\n")
    assert raw == (
        json.dumps(
            FIXED_PARAMETERS,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    assert set(FIXED_PARAMETERS) == {
        "bits_per_sample",
        "channel_count",
        "cymbal",
        "frame_count",
        "kick",
        "lcg",
        "sample_rate",
        "schema",
        "snare",
    }


def test_checked_in_smoke_fixture_matches_generator() -> None:
    fixture = Path("tests/fixtures/oaf_tf1_smoke/canonical.wav")
    generated = generate_smoke_wav(FIXED_PARAMETERS)

    assert fixture.read_bytes() == generated
    assert hashlib.sha256(generated).hexdigest() == hashlib.sha256(fixture.read_bytes()).hexdigest()


def test_smoke_impulses_and_integer_envelopes_have_exact_samples() -> None:
    content = generate_smoke_wav(FIXED_PARAMETERS)
    samples = struct.unpack("<44100h", content[44:])

    assert [
        (index, samples[index])
        for index in (
            4409,
            4410,
            4411,
            4455,
            4589,
            6614,
            6615,
            17639,
            17640,
            17641,
            22049,
            22050,
            30869,
            30870,
            30871,
            37484,
            37485,
        )
    ] == [
        (4409, 0),
        (4410, 28000),
        (4411, 27987),
        (4455, 27428),
        (4589, -25726),
        (6614, 12),
        (6615, 0),
        (17639, 0),
        (17640, -10558),
        (17641, -8856),
        (22049, -1),
        (22050, 0),
        (30869, 0),
        (30870, 3435),
        (30871, 11325),
        (37484, -2),
        (37485, 0),
    ]


def test_smoke_integer_helpers_wrap_and_saturate_exactly() -> None:
    assert lcg_step(0xFFFFFFFF, 1664525, 1013904223) == 1012239698
    assert saturating_add(32760, 10) == 32767
    assert saturating_add(-32760, -10) == -32768
    assert saturating_add(12, -4) == 8


def test_smoke_parameters_reject_unknown_fields_and_noninteger_values() -> None:
    unknown = deepcopy(FIXED_PARAMETERS)
    unknown["kick"]["host_default"] = 1
    with pytest.raises(SmokeFixtureError, match="unknown or missing"):
        generate_smoke_wav(unknown)

    noninteger = deepcopy(FIXED_PARAMETERS)
    noninteger["snare"]["amplitude"] = 20000.0
    with pytest.raises(SmokeFixtureError, match="positive integer"):
        generate_smoke_wav(noninteger)


def test_smoke_publication_is_identical_on_second_run(tmp_path: Path) -> None:
    parameters = tmp_path / "parameters.json"
    wav = tmp_path / "canonical.wav"

    first = write_smoke_fixture(parameters, wav)
    first_parameter_bytes = parameters.read_bytes()
    first_wav_bytes = wav.read_bytes()
    second = write_smoke_fixture(parameters, wav)

    assert second == first
    assert parameters.read_bytes() == first_parameter_bytes
    assert wav.read_bytes() == first_wav_bytes
    assert (
        parse_canonical_wav(wav.read_bytes(), max_input_audio_frames=44100).audio_frame_count
        == 44100
    )


def test_vendor_rejects_wrong_commit_without_changing_prior_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "vendor"
    destination.mkdir()
    prior = destination / "prior"
    prior.write_bytes(b"prior")
    manifest = tmp_path / "source-manifest.json"
    manifest.write_bytes(b"prior manifest\n")

    with pytest.raises(VendoringError, match="commit"):
        vendor_magenta(
            source=source,
            destination=destination,
            manifest_path=manifest,
            expected_commit=EXPECTED_UPSTREAM_COMMIT,
        )

    assert prior.read_bytes() == b"prior"
    assert manifest.read_bytes() == b"prior manifest\n"


def test_vendor_rejects_dirty_checkout_and_symlink(tmp_path: Path) -> None:
    source, commit, repository = _write_fake_magenta_checkout(tmp_path)
    (source / "dirty.txt").write_text("dirty", encoding="utf-8")

    with pytest.raises(VendoringError, match="dirty"):
        vendor_magenta(
            source=source,
            destination=tmp_path / "vendor-dirty",
            manifest_path=tmp_path / "dirty.json",
            expected_commit=commit,
            expected_repository=repository,
        )

    (source / "dirty.txt").unlink()
    (source / "magenta/models/onsets_frames_transcription/link.py").symlink_to(
        source / "magenta/__init__.py"
    )
    _git(source, "add", ".")
    _git(source, "commit", "-m", "symlink")
    symlink_commit = _git(source, "rev-parse", "HEAD")
    with pytest.raises(VendoringError, match="symlink"):
        vendor_magenta(
            source=source,
            destination=tmp_path / "vendor-link",
            manifest_path=tmp_path / "link.json",
            expected_commit=symlink_commit,
            expected_repository=repository,
        )


@pytest.mark.parametrize("index_flag", ("--assume-unchanged", "--skip-worktree"))
def test_vendor_rejects_selected_file_hidden_by_index_flag(
    tmp_path: Path,
    index_flag: str,
) -> None:
    source, commit, repository = _write_fake_magenta_checkout(tmp_path)
    selected = "magenta/models/onsets_frames_transcription/model.py"
    _git(source, "update-index", index_flag, selected)
    (source / selected).write_bytes(b"TAMPERED = True\n")
    assert _git(source, "status", "--porcelain=v1", "--untracked-files=all") == ""

    with pytest.raises(VendoringError, match="index flag|HEAD tree"):
        vendor_magenta(
            source=source,
            destination=tmp_path / "vendor",
            manifest_path=tmp_path / "manifest.json",
            expected_commit=commit,
            expected_repository=repository,
        )


def test_vendor_rejects_import_outside_reviewed_closure(tmp_path: Path) -> None:
    source, _commit, repository = _write_fake_magenta_checkout(
        tmp_path,
        model_source="import magenta.unreviewed\n",
        extra_files={"magenta/unreviewed.py": b"VALUE = 1\n"},
    )
    commit = _git(source, "rev-parse", "HEAD")

    with pytest.raises(VendoringError, match="allowlist"):
        vendor_magenta(
            source=source,
            destination=tmp_path / "vendor",
            manifest_path=tmp_path / "manifest.json",
            expected_commit=commit,
            expected_repository=repository,
        )


def test_vendor_manifest_is_deterministic_and_rejects_stale_destination(
    tmp_path: Path,
) -> None:
    source, commit, repository = _write_fake_magenta_checkout(tmp_path)
    first_destination = tmp_path / "vendor-1"
    first_manifest = tmp_path / "manifest-1.json"
    second_destination = tmp_path / "vendor-2"
    second_manifest = tmp_path / "manifest-2.json"

    vendor_magenta(
        source=source,
        destination=first_destination,
        manifest_path=first_manifest,
        expected_commit=commit,
        expected_repository=repository,
    )
    vendor_magenta(
        source=source,
        destination=second_destination,
        manifest_path=second_manifest,
        expected_commit=commit,
        expected_repository=repository,
    )
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert _tree_bytes(first_destination) == _tree_bytes(second_destination)

    (first_destination / "stale.py").write_bytes(b"stale")
    with pytest.raises(VendoringError, match="stale"):
        vendor_magenta(
            source=source,
            destination=first_destination,
            manifest_path=first_manifest,
            expected_commit=commit,
            expected_repository=repository,
        )


def test_runtime_lock_rendering_is_deterministic_and_hash_required(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _write_wheel(
        wheelhouse / "demo_pkg-1.2.3-py3-none-any.whl",
        name="demo-pkg",
        version="1.2.3",
    )
    distributions = inspect_wheelhouse(wheelhouse)

    first = render_requirements_lock(distributions)
    second = render_requirements_lock(tuple(reversed(distributions)))

    assert first == second
    assert first.startswith(b"--no-index\n--require-hashes\n")
    assert b"demo-pkg==1.2.3 --hash=sha256:" in first


def test_runtime_resolver_rejects_wrong_python_wheel(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _write_wheel(
        wheelhouse / "demo_pkg-1.2.3-cp38-cp38-manylinux2010_x86_64.whl",
        name="demo-pkg",
        version="1.2.3",
    )

    with pytest.raises(ResolutionError, match="compatible"):
        inspect_wheelhouse(wheelhouse)


def test_runtime_resolver_rejects_missing_extra_and_duplicate_dependencies(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.in"
    requirements.write_text("root==1.0\n", encoding="utf-8")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _write_wheel(
        wheelhouse / "root-1.0-py3-none-any.whl",
        name="root",
        version="1.0",
        requirements=("child>=2",),
    )
    with pytest.raises(ResolutionError, match="missing"):
        verify_requirement_closure(requirements, inspect_wheelhouse(wheelhouse))

    _write_wheel(
        wheelhouse / "child-2.0-py3-none-any.whl",
        name="child",
        version="2.0",
    )
    _write_wheel(
        wheelhouse / "extra-1.0-py3-none-any.whl",
        name="extra",
        version="1.0",
    )
    with pytest.raises(ResolutionError, match="extra"):
        verify_requirement_closure(requirements, inspect_wheelhouse(wheelhouse))

    (wheelhouse / "extra-1.0-py3-none-any.whl").unlink()
    _write_wheel(
        wheelhouse / "child-2.1-py3-none-any.whl",
        name="child",
        version="2.1",
    )
    with pytest.raises(ResolutionError, match="duplicate"):
        inspect_wheelhouse(wheelhouse)


def test_requirement_markers_use_frozen_linux_target_on_synthetic_darwin_host() -> None:
    code = """
import json
import platform
platform.system = lambda: "Darwin"
platform.release = lambda: "synthetic-darwin-release"
platform.version = lambda: "synthetic-darwin-version"
platform.machine = lambda: "arm64"
platform.python_implementation = lambda: "PyPy"
from packaging.requirements import Requirement
from tools.hpa320.resolve_oaf_runtime import _marker_applies
markers = [
    'platform_system == "Linux"',
    'platform_system == "Darwin"',
    'os_name == "posix"',
    'platform_machine == "x86_64"',
    'python_full_version == "3.7.17" and python_version == "3.7"',
    'implementation_name == "cpython" and implementation_version == "3.7.17"',
    'platform_python_implementation == "CPython"',
    'sys_platform == "linux"',
    'extra == "test"',
]
print(json.dumps([_marker_applies(Requirement("dep; " + marker)) for marker in markers]))
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    assert result.stdout == "[true, false, true, true, true, true, true, true, false]\n"


@pytest.mark.parametrize("marker_name", ("platform_release", "platform_version"))
def test_requirement_markers_reject_unfrozen_platform_values(marker_name: str) -> None:
    requirement = resolver.Requirement(f'dep; {marker_name} == "synthetic"')

    with pytest.raises(ResolutionError, match="unsupported marker variable"):
        resolver._marker_applies(requirement)  # pylint: disable=protected-access


def test_runtime_resolver_rejects_tensorflow_filename_hash_substitution(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.in"
    requirements.write_text("tensorflow==1.15.5\n", encoding="utf-8")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _write_wheel(
        wheelhouse / "tensorflow-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl",
        name="tensorflow",
        version="1.15.5",
        tag="cp37-cp37m-manylinux2010_x86_64",
    )
    records = inspect_wheelhouse(wheelhouse)
    lock = tmp_path / "requirements.lock"
    lock.write_bytes(render_requirements_lock(records))

    with pytest.raises(ResolutionError, match="TensorFlow wheel"):
        verify_offline_resolution(
            requirements_path=requirements,
            wheelhouse=wheelhouse,
            lock_path=lock,
        )


def test_runtime_resolver_rejects_unapproved_sdist(tmp_path: Path) -> None:
    sdist = tmp_path / "pretty_midi-0.2.10.tar.gz"
    sdist.write_bytes(b"not the approved source archive")

    with pytest.raises(DistributionBuildError, match="approved sdist"):
        validate_approved_sdist(sdist)


def test_sdist_allowlist_is_explicit_complete_and_utf8_sorted() -> None:
    names = [entry.name for entry in SDIST_ALLOWLIST]

    assert names == ["gast", "pretty-midi"]
    assert names == sorted(names, key=lambda value: value.encode("utf-8"))
    assert len({(entry.name, entry.version) for entry in SDIST_ALLOWLIST}) == len(SDIST_ALLOWLIST)
    assert all(entry.compatible_published_wheel_count == 0 for entry in SDIST_ALLOWLIST)


def test_allowlist_release_evidence_records_every_file_and_proves_target_wheels() -> None:
    payload = {
        "urls": [
            {
                "digests": {"sha256": "1" * 64},
                "filename": "gast-0.2.2.tar.gz",
                "packagetype": "sdist",
                "python_version": "source",
                "size": 10,
                "url": "https://files.pythonhosted.org/packages/a/gast-0.2.2.tar.gz",
                "yanked": False,
            },
            {
                "digests": {"sha256": "2" * 64},
                "filename": "gast-0.2.2-py2-none-any.whl",
                "packagetype": "bdist_wheel",
                "python_version": "py2",
                "size": 20,
                "url": "https://files.pythonhosted.org/packages/b/gast-0.2.2-py2-none-any.whl",
                "yanked": True,
            },
        ]
    }

    evidence = resolver.canonicalize_pypi_release("gast", "0.2.2", payload)

    assert evidence == {
        "files": [
            {
                "byte_length": 20,
                "filename": "gast-0.2.2-py2-none-any.whl",
                "packagetype": "bdist_wheel",
                "python_version": "py2",
                "sha256": "2" * 64,
                "url": "https://files.pythonhosted.org/packages/b/gast-0.2.2-py2-none-any.whl",
                "wheel_tags": ["py2-none-any"],
                "yanked": True,
            },
            {
                "byte_length": 10,
                "filename": "gast-0.2.2.tar.gz",
                "packagetype": "sdist",
                "python_version": "source",
                "sha256": "1" * 64,
                "url": "https://files.pythonhosted.org/packages/a/gast-0.2.2.tar.gz",
                "wheel_tags": [],
                "yanked": False,
            },
        ],
        "project": "gast",
        "version": "0.2.2",
    }
    assert resolver.count_target_compatible_wheels(evidence) == 0


def test_allowlist_required_by_is_derived_from_direct_and_wheel_metadata(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.in"
    requirements.write_text(
        "pretty-midi==0.2.10\ntensorflow==1.15.5\n",
        encoding="utf-8",
    )
    distributions = (
        resolver.DistributionRecord(
            name="gast",
            version="0.2.2",
            filename="gast-0.2.2-py3-none-any.whl",
            byte_length=1,
            sha256="1" * 64,
            requirements=(),
        ),
        resolver.DistributionRecord(
            name="pretty-midi",
            version="0.2.10",
            filename="pretty_midi-0.2.10-py3-none-any.whl",
            byte_length=1,
            sha256="2" * 64,
            requirements=(),
        ),
        resolver.DistributionRecord(
            name="tensorflow",
            version="1.15.5",
            filename="tensorflow-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl",
            byte_length=1,
            sha256="3" * 64,
            requirements=("gast == 0.2.2",),
        ),
    )

    assert resolver.derive_allowlist_required_by(requirements, distributions) == {
        "gast": ("tensorflow==1.15.5",),
        "pretty-midi": ("runtime requirements.in",),
    }


def test_allowlisted_sdist_inputs_reject_duplicate_basenames(tmp_path: Path) -> None:
    first = tmp_path / "first" / "gast-0.2.2.tar.gz"
    duplicate = tmp_path / "duplicate" / "gast-0.2.2.tar.gz"
    pretty_midi = tmp_path / "pretty_midi-0.2.10.tar.gz"
    first.parent.mkdir()
    duplicate.parent.mkdir()
    first.write_bytes(b"first")
    duplicate.write_bytes(b"duplicate")
    pretty_midi.write_bytes(b"pretty-midi")

    with pytest.raises(DistributionBuildError, match="duplicate"):
        resolver.build_allowlisted_sdists(
            sdist_paths=(first, duplicate, pretty_midi),
            build_tool_wheelhouse=tmp_path / "build-tools",
            build_requirements_path=tmp_path / "requirements-build.in",
            build_lock_path=tmp_path / "requirements-build.lock",
            output_wheelhouse_path=tmp_path / "output",
            manifest_path=tmp_path / "manifest.json",
        )


def test_locked_wheelhouse_materializer_is_exact_idempotent_and_offline(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    wheel = cache / "demo_pkg-1.2.3-py3-none-any.whl"
    _write_wheel(wheel, name="demo-pkg", version="1.2.3")
    records = inspect_wheelhouse(cache)
    lock = tmp_path / "requirements.lock"
    lock.write_bytes(render_requirements_lock(records))
    destination = tmp_path / "wheelhouse"

    first = resolver.materialize_locked_wheelhouse(
        lock_path=lock,
        destination=destination,
        offline_cache=cache,
    )
    second = resolver.materialize_locked_wheelhouse(
        lock_path=lock,
        destination=destination,
        offline_cache=cache,
    )

    assert first == second == records
    assert _tree_bytes(destination) == {wheel.name: wheel.read_bytes()}
    (destination / "extra.whl").write_bytes(b"extra")
    with pytest.raises(ResolutionError, match="extra"):
        resolver.materialize_locked_wheelhouse(
            lock_path=lock,
            destination=destination,
            offline_cache=cache,
        )


def test_distribution_build_manifest_rejects_ambiguous_allowlist_entries(
    tmp_path: Path,
) -> None:
    manifest, wheels = _render_complete_test_build_manifest(tmp_path)
    payload = json.loads(manifest)
    payload["allowlist"].append(deepcopy(payload["allowlist"][0]))
    content = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n"

    with pytest.raises(DistributionBuildError, match="ambiguous"):
        validate_distribution_build_manifest(content, wheelhouse_path=wheels)


def test_distribution_build_manifest_binds_release_evidence_and_actual_dependency_graph(
    tmp_path: Path,
) -> None:
    content, wheels = _render_complete_test_build_manifest(tmp_path)
    payload = json.loads(content)
    assert [entry["pypi_release"]["project"] for entry in payload["allowlist"]] == [
        "gast",
        "pretty-midi",
    ]
    requirements = tmp_path / "requirements.in"
    requirements.write_text(
        "pretty-midi==0.2.10\ntensorflow==1.15.5\n",
        encoding="utf-8",
    )
    distributions = (
        *inspect_wheelhouse(wheels),
        resolver.DistributionRecord(
            name="tensorflow",
            version="1.15.5",
            filename="tensorflow-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl",
            byte_length=1,
            sha256="3" * 64,
            requirements=("gast == 0.2.2",),
        ),
    )

    validate_distribution_build_manifest(
        content,
        wheelhouse_path=wheels,
        requirements_path=requirements,
        runtime_distributions=distributions,
    )
    without_parent = tuple(
        resolver.DistributionRecord(
            name=item.name,
            version=item.version,
            filename=item.filename,
            byte_length=item.byte_length,
            sha256=item.sha256,
            requirements=(),
        )
        for item in distributions
    )
    with pytest.raises(DistributionBuildError, match="required|graph"):
        validate_distribution_build_manifest(
            content,
            wheelhouse_path=wheels,
            requirements_path=requirements,
            runtime_distributions=without_parent,
        )


def test_distribution_build_manifest_rejects_native_allowlisted_wheel(
    tmp_path: Path,
) -> None:
    manifest, wheels = _render_complete_test_build_manifest(tmp_path)
    gast_wheel = wheels / "gast-0.2.2-py3-none-any.whl"
    _write_package_wheel(
        gast_wheel,
        name="gast",
        version="0.2.2",
        extra_member="gast/native.so",
    )

    with pytest.raises(DistributionBuildError, match="native"):
        validate_distribution_build_manifest(manifest, wheelhouse_path=wheels)


def test_exceptional_distribution_build_rejects_nonreproducible_wheels(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    with pytest.raises(DistributionBuildError, match="byte-identical"):
        require_reproducible_wheels(first, second)


def test_exceptional_distribution_rejects_native_wheel_member(tmp_path: Path) -> None:
    wheel = tmp_path / "pretty_midi-0.2.10-py3-none-any.whl"
    _write_pure_wheel(wheel, extra_member="pretty_midi/native.so")

    with pytest.raises(DistributionBuildError, match="native"):
        validate_built_pure_wheel(wheel)


def test_exceptional_distribution_requires_complete_record(tmp_path: Path) -> None:
    wheel = tmp_path / "pretty_midi-0.2.10-py3-none-any.whl"
    _write_pure_wheel(wheel, omit_package_record=True)

    with pytest.raises(DistributionBuildError, match="RECORD"):
        validate_built_pure_wheel(wheel)


def test_distribution_build_constants_are_explicit_and_hash_bound() -> None:
    assert DISTRIBUTION_BUILD_SCHEMA == "crux.distribution-build-manifest/v2"
    assert (
        APPROVED_SDIST_SHA256 == "ea6e192f94044674e833336ea1f415318ddf28e320302ac7b109edff0d4534bd"
    )


def test_distribution_build_manifest_rejects_recipe_drift(tmp_path: Path) -> None:
    content, wheels = _render_complete_test_build_manifest(tmp_path)
    manifest = json.loads(content)
    manifest["build"]["command"] = ["/bin/sh", "-ec", "python setup.py bdist_wheel"]

    with pytest.raises(DistributionBuildError, match="recipe"):
        validate_distribution_build_manifest(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode() + b"\n",
            wheelhouse_path=wheels,
        )


def test_distribution_build_manifest_rejects_source_or_toolchain_drift(
    tmp_path: Path,
) -> None:
    content, wheels = _render_complete_test_build_manifest(tmp_path)
    manifest = json.loads(content)
    manifest["allowlist"][0]["source_distribution"]["sha256"] = "0" * 64

    with pytest.raises(DistributionBuildError, match="source distribution"):
        validate_distribution_build_manifest(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode() + b"\n",
            wheelhouse_path=wheels,
        )

    manifest = json.loads(content)
    manifest["build"]["tool_distributions"][0]["sha256"] = "0" * 64
    with pytest.raises(DistributionBuildError, match="toolchain"):
        validate_distribution_build_manifest(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode() + b"\n",
            wheelhouse_path=wheels,
        )


def test_distribution_build_manifest_rejects_final_wheel_byte_drift(
    tmp_path: Path,
) -> None:
    manifest, wheels = _render_complete_test_build_manifest(tmp_path)
    wheel = wheels / "pretty_midi-0.2.10-py3-none-any.whl"
    wheel.write_bytes(wheel.read_bytes() + b"drift")

    with pytest.raises(DistributionBuildError, match="final wheel"):
        validate_distribution_build_manifest(manifest, wheelhouse_path=wheels)


def test_provisional_dockerfile_has_clean_test_and_fail_closed_runtime_inputs() -> None:
    dockerfile = Path("runtime/oaf_tf1/Dockerfile").read_text(encoding="utf-8")

    assert (
        "FROM python@sha256:"
        "ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673" in dockerfile
    )
    assert "AS test" in dockerfile
    assert dockerfile.index("requirements-test.lock") < dockerfile.rindex("AS runtime")
    assert "pip install --no-index --require-hashes --only-binary=:all:" in dockerfile
    test_stage = dockerfile[: dockerfile.index("FROM runtime-build AS test")]
    for argument in ("RUNTIME_UID", "RUNTIME_GID"):
        assert f"ARG {argument}\n" in dockerfile
        assert f"ARG {argument}=" not in dockerfile
    assert "DEBIAN_INRELEASE_SHA256" not in test_stage
    assert "COPY runtime/oaf_tf1/vendor/magenta/" in dockerfile
    assert "COPY runtime/oaf_tf1/ /opt/crux/runtime/" in dockerfile
    assert "COPY runtime/oaf_tf1/system-packages/" in dockerfile
    for argument in (
        "DEBIAN_SNAPSHOT_URL",
        "DEBIAN_INRELEASE_SHA256",
        "DEBIAN_ARCHIVE_KEYRING_SHA256",
        "DEBIAN_SIGNING_FINGERPRINT",
        "DEBIAN_CODENAME",
        "DEBIAN_ARCHITECTURE",
        "SYSTEM_PACKAGE_MANIFEST_SHA256",
        "SYSTEM_PACKAGE_INVENTORY_SHA256",
    ):
        assert f"ARG {argument}\n" in dockerfile
        assert f"ARG {argument}=" not in dockerfile
    assert 'USER "${RUNTIME_UID}:${RUNTIME_GID}"' in dockerfile
    assert "backend-lock.json" not in dockerfile
    assert "runtime-lock.json" not in dockerfile


def test_provisional_dockerfile_test_stage_runs_as_explicit_nonroot_user() -> None:
    dockerfile = Path("runtime/oaf_tf1/Dockerfile").read_text(encoding="utf-8")
    test_stage = dockerfile.split("FROM runtime-build AS test", 1)[1].split(
        "FROM runtime-base AS system-runtime-base", 1
    )[0]

    assert 'USER "${RUNTIME_UID}:${RUNTIME_GID}"' in test_stage


_TEST_DEBIAN_FINGERPRINT = "0123456789ABCDEF0123456789ABCDEF01234567"


def _canonical_test_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode() + b"\n"


def _file_identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "byte_length": len(content),
        "filename": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _write_fake_gpgv(path: Path, *, fingerprint: str, argument_log: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {shlex.quote(str(argument_log))}\n"
        f"printf '%s\\n' '[GNUPG:] VALIDSIG {fingerprint} 20200101 0 4 0 1 10 00 "
        f"{fingerprint}'\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_authenticated_system_package_bundle(
    tmp_path: Path,
    *,
    indexed_version: str = "1.0-1",
    inventory_version: str = "1.0-1",
    release_codename: str = "buster",
    release_architecture: str = "amd64",
    signing_fingerprint: str = _TEST_DEBIAN_FINGERPRINT,
) -> tuple[Path, Path, Path, dict[str, object], dict[str, object]]:
    bundle = tmp_path / "authenticated-bundle"
    indexes = bundle / "indexes" / "main" / "binary-amd64"
    packages = bundle / "packages"
    indexes.mkdir(parents=True)
    packages.mkdir()
    base_keyring = tmp_path / "base-image-debian-archive-keyring.gpg"
    base_keyring.write_bytes(b"immutable base image archive keys")
    deb = packages / "libexample_1.0-1_amd64.deb"
    deb.write_bytes(b"authenticated deb bytes")
    pool_path = "pool/main/libe/libexample/libexample_1.0-1_amd64.deb"
    deb_sha256 = hashlib.sha256(deb.read_bytes()).hexdigest()
    stanza = (
        "Package: libexample\n"
        f"Version: {indexed_version}\n"
        "Architecture: amd64\n"
        f"Filename: {pool_path}\n"
        f"Size: {len(deb.read_bytes())}\n"
        f"SHA256: {deb_sha256}\n"
        "Description: fixture\n\n"
    ).encode()
    index = indexes / "Packages.xz"
    index.write_bytes(lzma.compress(stanza, format=lzma.FORMAT_XZ))
    release_path = "main/binary-amd64/Packages.xz"
    release = (
        "Origin: Debian\n"
        f"Codename: {release_codename}\n"
        f"Architectures: all {release_architecture}\n"
        "SHA256:\n"
        f" {hashlib.sha256(index.read_bytes()).hexdigest()} "
        f"{len(index.read_bytes())} {release_path}\n"
    )
    inrelease = bundle / "InRelease"
    inrelease.write_text(
        "-----BEGIN PGP SIGNED MESSAGE-----\n"
        "Hash: SHA256\n\n"
        f"{release}"
        "-----BEGIN PGP SIGNATURE-----\n\n"
        "fixture\n"
        "-----END PGP SIGNATURE-----\n",
        encoding="utf-8",
    )
    inventory = bundle / "expected-dpkg-inventory.txt"
    inventory.write_text(
        f"base-files\t1.0\tamd64\nlibexample\t{inventory_version}\tamd64\n",
        encoding="utf-8",
    )
    package_record = {
        "architecture": "amd64",
        "filename": pool_path,
        "package": "libexample",
        "sha256": deb_sha256,
        "size": len(deb.read_bytes()),
        "version": "1.0-1",
    }
    manifest_payload = {
        "architecture": "amd64",
        "codename": "buster",
        "expected_dpkg_inventory": _file_identity(inventory),
        "inrelease": _file_identity(inrelease),
        "package_indexes": [
            {
                "byte_length": len(index.read_bytes()),
                "release_path": release_path,
                "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            }
        ],
        "packages": [package_record],
        "schema": "crux.oaf-system-package-bundle/v2",
        "signing_fingerprint": signing_fingerprint,
        "snapshot_url": "https://snapshot.debian.org/archive/debian/20200101T000000Z",
    }
    manifest = bundle / "bundle-manifest.json"
    manifest.write_bytes(_canonical_test_json(manifest_payload))
    argument_log = tmp_path / "gpgv-arguments.txt"
    gpgv = tmp_path / "gpgv"
    _write_fake_gpgv(gpgv, fingerprint=signing_fingerprint, argument_log=argument_log)
    expected = {
        "expected_architecture": "amd64",
        "expected_codename": "buster",
        "expected_inrelease_sha256": manifest_payload["inrelease"]["sha256"],
        "expected_inventory_sha256": manifest_payload["expected_dpkg_inventory"]["sha256"],
        "expected_keyring_sha256": hashlib.sha256(base_keyring.read_bytes()).hexdigest(),
        "expected_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "expected_signing_fingerprint": _TEST_DEBIAN_FINGERPRINT,
        "expected_snapshot_url": manifest_payload["snapshot_url"],
        "gpgv_executable": str(gpgv),
    }
    return bundle, base_keyring, argument_log, manifest_payload, expected


def _verify_authenticated_test_bundle(
    monkeypatch: pytest.MonkeyPatch,
    system_packages: object,
    bundle: Path,
    base_keyring: Path,
    expected: dict[str, object],
) -> dict[str, object]:
    monkeypatch.setattr(system_packages, "BASE_IMAGE_ARCHIVE_KEYRING", base_keyring, raising=False)
    return system_packages.verify_system_package_bundle(bundle=bundle, **expected)


def test_system_package_bundle_anchors_gpgv_to_base_image_keyring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    bundle, base_keyring, argument_log, _, expected = _write_authenticated_system_package_bundle(
        tmp_path
    )

    _verify_authenticated_test_bundle(monkeypatch, system_packages, bundle, base_keyring, expected)

    arguments = argument_log.read_text(encoding="utf-8").splitlines()
    assert str(base_keyring) in arguments
    assert "/nonexistent" in arguments
    assert not (bundle / "debian-archive-keyring.gpg").exists()


def test_signed_release_cannot_authenticate_arbitrary_manifest_only_deb(
    tmp_path: Path,
) -> None:
    system_packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    bundle = tmp_path / "old-bundle"
    packages = bundle / "packages"
    packages.mkdir(parents=True)
    inrelease = bundle / "InRelease"
    keyring = bundle / "debian-archive-keyring.gpg"
    inventory = bundle / "expected-dpkg-inventory.txt"
    arbitrary_deb = packages / "arbitrary_9.9_amd64.deb"
    inrelease.write_bytes(b"legitimately signed release without this package")
    keyring.write_bytes(b"attacker-selected signing key")
    inventory.write_bytes(b"arbitrary\t9.9\n")
    arbitrary_deb.write_bytes(b"arbitrary unsigned package")
    payload = {
        "archive_keyring": _file_identity(keyring),
        "expected_dpkg_inventory": _file_identity(inventory),
        "inrelease": _file_identity(inrelease),
        "packages": [_file_identity(arbitrary_deb)],
        "schema": "crux.oaf-system-package-bundle/v1",
        "snapshot_url": "https://snapshot.debian.org/archive/debian/20200101T000000Z",
    }
    manifest = bundle / "bundle-manifest.json"
    manifest.write_bytes(_canonical_test_json(payload))
    gpgv = tmp_path / "gpgv"
    _write_fake_gpgv(
        gpgv,
        fingerprint=_TEST_DEBIAN_FINGERPRINT,
        argument_log=tmp_path / "arguments.txt",
    )

    with pytest.raises(system_packages.SystemPackageError):
        system_packages.verify_system_package_bundle(
            bundle=bundle,
            expected_snapshot_url=payload["snapshot_url"],
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            expected_inrelease_sha256=payload["inrelease"]["sha256"],
            expected_keyring_sha256=payload["archive_keyring"]["sha256"],
            expected_inventory_sha256=payload["expected_dpkg_inventory"]["sha256"],
            expected_signing_fingerprint=_TEST_DEBIAN_FINGERPRINT,
            expected_codename="buster",
            expected_architecture="amd64",
            gpgv_executable=str(gpgv),
        )


@pytest.mark.parametrize(
    ("fixture_override", "error_pattern"),
    [
        ({"indexed_version": "2.0-1"}, "authenticated package"),
        ({"inventory_version": "2.0-1"}, "inventory"),
        ({"release_codename": "sid"}, "codename"),
        ({"release_architecture": "arm64"}, "architecture"),
        ({"signing_fingerprint": "F" * 40}, "fingerprint"),
    ],
)
def test_system_package_bundle_rejects_signed_metadata_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_override: dict[str, str],
    error_pattern: str,
) -> None:
    system_packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    bundle, base_keyring, _, _, expected = _write_authenticated_system_package_bundle(
        tmp_path, **fixture_override
    )

    with pytest.raises(system_packages.SystemPackageError, match=error_pattern):
        _verify_authenticated_test_bundle(
            monkeypatch, system_packages, bundle, base_keyring, expected
        )


@pytest.mark.parametrize("mutation", ["hash", "path", "stanza", "duplicate-field"])
def test_system_package_bundle_rejects_index_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    system_packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    bundle, base_keyring, _, payload, expected = _write_authenticated_system_package_bundle(
        tmp_path
    )
    manifest = bundle / "bundle-manifest.json"
    index = bundle / "indexes" / "main" / "binary-amd64" / "Packages.xz"
    if mutation == "hash":
        index.write_bytes(index.read_bytes() + b"drift")
    elif mutation == "path":
        payload["package_indexes"][0]["release_path"] = "../Packages.xz"
        manifest.write_bytes(_canonical_test_json(payload))
        expected["expected_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    else:
        raw = lzma.decompress(index.read_bytes())
        if mutation == "stanza":
            raw = raw.replace(b"Package: libexample", b"Package: other")
        else:
            raw = raw.replace(
                b"Package: libexample\n",
                b"Package: libexample\nPackage: duplicate\n",
            )
        old_size = payload["package_indexes"][0]["byte_length"]
        old_sha256 = payload["package_indexes"][0]["sha256"]
        index.write_bytes(lzma.compress(raw, format=lzma.FORMAT_XZ))
        payload["package_indexes"][0]["byte_length"] = len(index.read_bytes())
        payload["package_indexes"][0]["sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
        inrelease = bundle / "InRelease"
        inrelease.write_bytes(
            inrelease.read_bytes().replace(
                f"{old_sha256} {old_size} ".encode(),
                (
                    f"{payload['package_indexes'][0]['sha256']} "
                    f"{payload['package_indexes'][0]['byte_length']} "
                ).encode(),
            )
        )
        payload["inrelease"] = _file_identity(inrelease)
        expected["expected_inrelease_sha256"] = payload["inrelease"]["sha256"]
        manifest.write_bytes(_canonical_test_json(payload))
        expected["expected_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()

    with pytest.raises(system_packages.SystemPackageError):
        _verify_authenticated_test_bundle(
            monkeypatch, system_packages, bundle, base_keyring, expected
        )


def test_system_package_install_uses_only_authenticated_debs_and_exact_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    bundle, base_keyring, _, _, expected = _write_authenticated_system_package_bundle(tmp_path)
    payload = _verify_authenticated_test_bundle(
        monkeypatch, system_packages, bundle, base_keyring, expected
    )
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        stdout = (
            (bundle / "expected-dpkg-inventory.txt").read_bytes()
            if command[0] == "dpkg-query"
            else b""
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(system_packages.subprocess, "run", fake_run)
    system_packages.install_system_package_bundle(bundle=bundle, payload=payload)

    assert calls[0] == (
        "dpkg",
        "--install",
        str(bundle / "packages" / "libexample_1.0-1_amd64.deb"),
    )
    assert calls[1] == (
        "dpkg-query",
        "-W",
        "-f=${Package}\\t${Version}\\t${Architecture}\\n",
    )


def test_system_package_bundle_rejects_deb_hash_drift_before_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_packages = importlib.import_module("tools.hpa320.oaf_system_packages")
    bundle, base_keyring, _, _, expected = _write_authenticated_system_package_bundle(tmp_path)
    deb = bundle / "packages" / "libexample_1.0-1_amd64.deb"
    deb.write_bytes(b"tampered deb")

    with pytest.raises(system_packages.SystemPackageError, match="authenticated package.*hash"):
        _verify_authenticated_test_bundle(
            monkeypatch, system_packages, bundle, base_keyring, expected
        )


def _write_wheel(
    path: Path,
    *,
    name: str,
    version: str,
    requirements: tuple[str, ...] = (),
    tag: str = "py3-none-any",
) -> None:
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        requirement_lines = "".join(
            f"Requires-Dist: {requirement}\n" for requirement in requirements
        )
        archive.writestr(
            f"{dist_info}/METADATA",
            (f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{requirement_lines}"),
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            (f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: {tag}\n"),
        )


def _write_pure_wheel(
    path: Path,
    *,
    extra_member: str | None = None,
    omit_package_record: bool = False,
) -> None:
    _write_package_wheel(
        path,
        name="pretty-midi",
        version="0.2.10",
        package_path="pretty_midi",
        extra_member=extra_member,
        omit_package_record=omit_package_record,
    )


def _write_package_wheel(
    path: Path,
    *,
    name: str,
    version: str,
    package_path: str | None = None,
    extra_member: str | None = None,
    omit_package_record: bool = False,
) -> None:
    import base64

    package = package_path or name.replace("-", "_")
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    members = {
        f"{package}/__init__.py": f"VERSION = {version!r}\n".encode(),
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    if extra_member is not None:
        members[extra_member] = b"native"
    record_path = f"{dist_info}/RECORD"
    rows = []
    for name, content in sorted(members.items()):
        if omit_package_record and name == f"{package}/__init__.py":
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        rows.append(f"{name},sha256={digest},{len(content)}")
    rows.append(f"{record_path},,")
    members[record_path] = ("\n".join(rows) + "\n").encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _render_complete_test_build_manifest(tmp_path: Path) -> tuple[bytes, Path]:
    wheelhouse = tmp_path / "allowlisted-wheels"
    wheelhouse.mkdir()
    builds = []
    for spec in SDIST_ALLOWLIST:
        path = wheelhouse / spec.wheel_filename
        _write_package_wheel(path, name=spec.name, version=spec.version)
        builds.append(
            SdistBuildRecord(
                spec=spec,
                wheel=validate_built_pure_wheel(path, spec),
            )
        )
    manifest = render_distribution_build_manifest(
        tuple(builds),
        diagnostic_execution={
            "docker_server_version": "test",
            "emulated": True,
            "host_platform": "test/host",
        },
        pypi_releases={
            spec.name: {
                "files": [
                    {
                        "byte_length": spec.sdist_byte_length,
                        "filename": spec.sdist_filename,
                        "packagetype": "sdist",
                        "python_version": "source",
                        "sha256": spec.sdist_sha256,
                        "url": spec.sdist_url,
                        "wheel_tags": [],
                        "yanked": False,
                    }
                ],
                "project": spec.name,
                "version": spec.version,
            }
            for spec in SDIST_ALLOWLIST
        },
        required_by={
            "gast": ("tensorflow==1.15.5",),
            "pretty-midi": ("runtime requirements.in",),
        },
    )
    return manifest, wheelhouse


def _write_fake_magenta_checkout(
    tmp_path: Path,
    *,
    model_source: str = "from magenta import music\n",
    extra_files: dict[str, bytes] | None = None,
) -> tuple[Path, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    repository = "https://example.invalid/magenta.git"
    files = {
        "LICENSE": b"license\n",
        "magenta/__init__.py": b"",
        "magenta/models/__init__.py": b"",
        "magenta/models/onsets_frames_transcription/__init__.py": b"",
        "magenta/models/onsets_frames_transcription/model.py": model_source.encode(),
        "magenta/music/__init__.py": b"",
    }
    files.update(extra_files or {})
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _git(source, "init")
    _git(source, "config", "user.name", "HPA Test")
    _git(source, "config", "user.email", "hpa@example.invalid")
    _git(source, "remote", "add", "origin", repository)
    _git(source, "add", ".")
    _git(source, "commit", "-m", "fixture")
    return source, _git(source, "rev-parse", "HEAD"), repository


def _git(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(source), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.rstrip("\n")


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
