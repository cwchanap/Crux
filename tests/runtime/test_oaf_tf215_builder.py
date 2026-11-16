"""Stdlib-only tests for the scratch TF 2.15 OaF runtime builder."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).parents[2]
BUILDER_PATH = REPO_ROOT / "experiments" / "oaf_tf215" / "build_runtime.py"
PATCH_PATH = REPO_ROOT / "experiments" / "oaf_tf215" / "patches" / "tf215-compat.patch"
FROZEN_RUNTIME = REPO_ROOT / "runtime" / "oaf_tf1"

SCRATCH_PATH_CONSTANTS = (
    "MODEL_SCRATCH_DIRECTORY",
    "DEFAULT_UNINSTRUMENTED_SEQUENCE_LIB",
)
TF_COMPAT_FUNCTION = "_initialize_tf_compat"

EXPECTED_COMPAT_PATHS = {
    "runtime/model.py",
    "runtime/vendor/magenta/__init__.py",
    "runtime/vendor/magenta/common/__init__.py",
    "runtime/vendor/magenta/common/tf_utils.py",
    "runtime/vendor/magenta/contrib/__init__.py",
    "runtime/vendor/magenta/contrib/rnn.py",
    "runtime/vendor/magenta/contrib/training.py",
    "runtime/vendor/magenta/music/__init__.py",
    "runtime/vendor/magenta/models/onsets_frames_transcription/audio_transform.py",
    "runtime/vendor/magenta/models/onsets_frames_transcription/configs.py",
    "runtime/vendor/magenta/models/onsets_frames_transcription/estimator_spec_util.py",
    "runtime/vendor/magenta/models/onsets_frames_transcription/model_tpu.py",
    "runtime/vendor/magenta/models/onsets_frames_transcription/train_util.py",
}

LOCKED_IMPLEMENTATIONS = (
    "validate_restored_tensor_inventory",
    "validate_tensor_coverage",
    "assert_no_reachable_stochastic_ops",
    "_serialized_example",
    "_load_uninstrumented_sequences_module",
    "_build_coverage_graph",
    "transcribe",
)


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("crux_oaf_tf215_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BUILDER_PATH}")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


@pytest.fixture(scope="module")
def scratch_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    builder = _load_builder()
    output = tmp_path_factory.mktemp("oaf-tf215") / "scratch"
    runtime = builder.build_runtime(REPO_ROOT, output)
    assert runtime == output.resolve() / "runtime"
    return output.resolve()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root).as_posix().encode()
        if path.is_symlink():
            digest.update(b"link:" + relative)
        elif path.is_file():
            digest.update(b"file:" + relative + path.read_bytes())
        else:
            digest.update(b"dir:" + relative)
    return digest.hexdigest()


_FROZEN_DIGEST_BEFORE = _tree_digest(FROZEN_RUNTIME)


def _patch_paths() -> set[str]:
    paths: set[str] = set()
    for line in PATCH_PATH.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", line)
        if match:
            assert match.group(1) == match.group(2), line
            paths.add(match.group(1))
    return paths


def _normalized_model_module(source: bytes) -> str:
    tree = ast.parse(source)
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == TF_COMPAT_FUNCTION:
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in SCRATCH_PATH_CONSTANTS
        ):
            node.value = ast.Constant(value="<scratch path>")
        body.append(node)
    tree.body = body
    removed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load":
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Call)
                and isinstance(first.value.func, ast.Name)
                and first.value.func.id == TF_COMPAT_FUNCTION
            ):
                removed += 1
                node.body = node.body[1:]
    assert removed in (0, 1)
    return ast.dump(tree, include_attributes=False)


def test_build_copies_frozen_inputs_and_only_scratch_sources(scratch_root: Path) -> None:
    runtime = scratch_root / "runtime"
    assert runtime.is_dir()
    assert _tree_digest(FROZEN_RUNTIME) == _FROZEN_DIGEST_BEFORE
    assert (runtime / "model.json").read_bytes() == (FROZEN_RUNTIME / "model.json").read_bytes()
    assert (runtime / "worker.py").read_bytes() == (FROZEN_RUNTIME / "worker.py").read_bytes()
    scratch_upstream = scratch_root / "upstream"
    assert (scratch_upstream / "LICENSE").read_bytes() == (
        FROZEN_RUNTIME / "vendor" / "LICENSE"
    ).read_bytes()
    assert (runtime / "vendor" / "LICENSE").read_bytes() == (
        FROZEN_RUNTIME / "vendor" / "LICENSE"
    ).read_bytes()
    pristine = FROZEN_RUNTIME / "vendor" / "magenta" / "music" / "sequences_lib.py"
    assert (scratch_upstream / "magenta" / "music" / "sequences_lib.py").read_bytes() == (
        pristine.read_bytes()
    )
    instrumented = runtime / "vendor" / "magenta" / "music" / "sequences_lib.py"
    assert instrumented.read_bytes() != pristine.read_bytes()
    assert b"capture_emitted_frames" in instrumented.read_bytes()
    infer_util = runtime / "vendor" / "magenta" / "models" / "onsets_frames_transcription"
    assert b"pair_emitted_frame_confidence" in (infer_util / "infer_util.py").read_bytes()


def test_builder_refuses_an_existing_destination(tmp_path: Path) -> None:
    builder = _load_builder()
    destination = tmp_path / "scratch"
    destination.mkdir()
    with pytest.raises(builder.ScratchRuntimeError, match="already exists"):
        builder.build_runtime(REPO_ROOT, destination)
    assert builder.build_runtime(REPO_ROOT, tmp_path / "another").is_dir()


def test_instrumentation_runs_before_the_compatibility_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _load_builder()
    events: list[tuple[str, bool]] = []
    git_calls: list[tuple[str, ...]] = []
    original_instrumentation = builder._apply_instrumentation
    original_git = builder._run_git

    def spy_instrumentation(
        repo_root: Path, upstream: Path, vendor: Path, patch_path: Path
    ) -> object:
        events.append(("instrumentation", vendor.exists()))
        result = original_instrumentation(repo_root, upstream, vendor, patch_path)
        events.append(("instrumentation-done", vendor.is_dir()))
        return result

    def spy_git(root: Path, *arguments: str) -> None:
        git_calls.append(tuple(arguments))
        events.append(("compat", (root / "runtime" / "vendor").is_dir()))
        return original_git(root, *arguments)

    monkeypatch.setattr(builder, "_apply_instrumentation", spy_instrumentation)
    monkeypatch.setattr(builder, "_run_git", spy_git)

    builder.build_runtime(REPO_ROOT, tmp_path / "scratch")

    assert events[0] == ("instrumentation", False)
    assert events[1] == ("instrumentation-done", True)
    assert events[2:] == [("compat", True), ("compat", True)]
    assert git_calls[0][:2] == ("apply", "--check")
    assert git_calls[1][0] == "apply"
    assert git_calls[0][-1] == os.fspath(PATCH_PATH)


def test_compatibility_patch_is_scoped_to_the_reviewed_paths() -> None:
    builder = _load_builder()
    assert builder.COMPATIBILITY_PATCH_SHA256 == hashlib.sha256(PATCH_PATH.read_bytes()).hexdigest()
    assert _patch_paths() == EXPECTED_COMPAT_PATHS


def test_builder_rejects_unreviewed_patch_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = _load_builder()
    monkeypatch.setattr(builder, "COMPATIBILITY_PATCH_SHA256", "0" * 64)
    with pytest.raises(builder.ScratchRuntimeError, match="reviewed revision"):
        builder.build_runtime(REPO_ROOT, tmp_path / "scratch")
    assert not (tmp_path / "scratch").exists()


def test_copied_model_allows_only_compat_initialization_and_paths(
    scratch_root: Path,
) -> None:
    frozen_source = (FROZEN_RUNTIME / "model.py").read_bytes()
    patched_source = (scratch_root / "runtime" / "model.py").read_bytes()
    assert _normalized_model_module(patched_source) == _normalized_model_module(frozen_source)
    patched = patched_source.decode("utf-8")
    for name in LOCKED_IMPLEMENTATIONS:
        assert f"def {name}(" in patched
    tree = ast.parse(patched_source)
    compat_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TF_COMPAT_FUNCTION
    ]
    assert len(compat_calls) == 1
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == TF_COMPAT_FUNCTION
    )
    function_text = ast.unparse(function)
    assert "ops.Tensor = tensor.Tensor" in function_text
    assert "disable_v2_behavior()" in function_text


def test_package_initializers_and_configs_drop_only_unused_imports(
    scratch_root: Path,
) -> None:
    vendor = scratch_root / "runtime" / "vendor" / "magenta"
    common_init = (vendor / "common" / "__init__.py").read_text(encoding="utf-8")
    assert "nade" not in common_init.lower()
    assert "from .beam_search import beam_search" in common_init
    assert "from .tf_utils import merge_hparams" in common_init

    music_init = (vendor / "music" / "__init__.py").read_text(encoding="utf-8")
    assert "notebook_utils" not in music_init
    assert "from magenta.music.drums_encoder_decoder import MultiDrumOneHotEncoding" in (music_init)
    assert "from magenta.music.sequences_lib import quantize_note_sequence" in music_init

    magenta_init = (vendor / "__init__.py").read_text(encoding="utf-8")
    for eager in ("magenta.common.nade", "notebook_utils", "import magenta.music."):
        assert eager not in magenta_init
    assert "from magenta.version import __version__" in magenta_init

    configs = (vendor / "models" / "onsets_frames_transcription" / "configs.py").read_text(
        encoding="utf-8"
    )
    assert "import model\n" not in configs
    assert "CONFIG_MAP['onsets_frames']" not in configs
    assert "CONFIG_MAP['drums'] = Config(" in configs
    assert "model_fn=model_tpu.model_fn" in configs
    assert "spec_hop_length=512" in configs
    assert "hop_length=441" in configs
    for relative in (
        "common/tf_utils.py",
        "models/onsets_frames_transcription/audio_transform.py",
        "models/onsets_frames_transcription/configs.py",
        "models/onsets_frames_transcription/estimator_spec_util.py",
        "models/onsets_frames_transcription/model_tpu.py",
        "models/onsets_frames_transcription/train_util.py",
    ):
        assert "tensorflow.contrib" not in (vendor / relative).read_text(encoding="utf-8")
