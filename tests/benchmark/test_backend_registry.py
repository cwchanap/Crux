from __future__ import annotations

import builtins
import sys
from types import ModuleType

import pytest

from src.benchmark.backend_identity import BackendDescriptor
from src.benchmark.backend_registry import (
    HEURISTIC_BACKEND_ID,
    LEGACY_TF2_BACKEND_ID,
    OFFICIAL_BACKEND_ID,
    BackendLockUnavailable,
    BackendRegistry,
    BackendUnavailable,
    default_backend_registry,
)
from src.benchmark.backend_reports import UNAVAILABLE_BACKEND_REPORT_ID


class DescriptorBackend:
    def __init__(self, backend_id: str) -> None:
        self._descriptor = BackendDescriptor(
            payload={"backend_id": backend_id},
            sha256="a" * 64,
        )

    def descriptor(self) -> BackendDescriptor:
        return self._descriptor


def test_registry_uses_official_backend_only_when_backend_is_omitted() -> None:
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        factories={OFFICIAL_BACKEND_ID: lambda: DescriptorBackend(OFFICIAL_BACKEND_ID)},
    )

    backend = registry.create(None)

    assert backend.descriptor().payload["backend_id"] == OFFICIAL_BACKEND_ID


def test_registry_selects_heuristic_only_when_explicitly_requested() -> None:
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        factories={
            OFFICIAL_BACKEND_ID: lambda: DescriptorBackend(OFFICIAL_BACKEND_ID),
            HEURISTIC_BACKEND_ID: lambda: DescriptorBackend(HEURISTIC_BACKEND_ID),
        },
    )

    backend = registry.create(HEURISTIC_BACKEND_ID)

    assert backend.descriptor().payload["backend_id"] == HEURISTIC_BACKEND_ID


def test_unknown_backend_uses_stable_report_namespace_without_retaining_input() -> None:
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        factories={OFFICIAL_BACKEND_ID: lambda: DescriptorBackend(OFFICIAL_BACKEND_ID)},
    )

    with pytest.raises(BackendUnavailable, match="unknown backend") as captured:
        registry.create("../../arbitrary-user-input")

    assert captured.value.report_backend_id == UNAVAILABLE_BACKEND_REPORT_ID
    assert captured.value.unknown_backend is True
    assert "../../arbitrary-user-input" not in str(captured.value)


def test_default_registry_does_not_import_backend_implementations_until_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_names = {
        "src.benchmark.backends.oaf_tf1",
        "src.benchmark.backends.heuristic",
    }
    for module_name in module_names:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    registry = default_backend_registry()

    assert module_names.isdisjoint(sys.modules)
    assert set(registry.factories) == {
        OFFICIAL_BACKEND_ID,
        HEURISTIC_BACKEND_ID,
        LEGACY_TF2_BACKEND_ID,
    }


def test_default_registry_imports_only_the_explicit_heuristic_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DescriptorBackend(HEURISTIC_BACKEND_ID)
    heuristic_module = ModuleType("src.benchmark.backends.heuristic")
    heuristic_module.create_backend = lambda: backend  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "src.benchmark.backends.heuristic",
        heuristic_module,
    )
    monkeypatch.delitem(sys.modules, "src.benchmark.backends.oaf_tf1", raising=False)

    selected = default_backend_registry().create(HEURISTIC_BACKEND_ID)

    assert selected is backend
    assert "src.benchmark.backends.oaf_tf1" not in sys.modules


@pytest.mark.parametrize("backend_id", [OFFICIAL_BACKEND_ID, HEURISTIC_BACKEND_ID])
def test_missing_backend_module_maps_to_known_backend_unavailable_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    backend_id: str,
) -> None:
    selected_module = {
        OFFICIAL_BACKEND_ID: "src.benchmark.backends.oaf_tf1",
        HEURISTIC_BACKEND_ID: "src.benchmark.backends.heuristic",
    }[backend_id]
    original_import = builtins.__import__
    imported: list[str] = []

    def deny_selected_import(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        imported.append(name)
        if name == selected_module:
            raise ImportError("dependency details must not escape")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", deny_selected_import)

    with pytest.raises(
        BackendUnavailable,
        match="backend implementation is unavailable",
    ) as captured:
        default_backend_registry().create(backend_id)

    assert captured.value.report_backend_id == backend_id
    assert captured.value.unknown_backend is False
    unselected_module = (
        "src.benchmark.backends.heuristic"
        if backend_id == OFFICIAL_BACKEND_ID
        else "src.benchmark.backends.oaf_tf1"
    )
    assert unselected_module not in imported


def test_missing_checked_in_lock_maps_to_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("src.benchmark.backends.oaf_tf1")

    def missing_lock() -> object:
        raise BackendLockUnavailable("private/checkpoint/path")

    module.create_backend = missing_lock  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.benchmark.backends.oaf_tf1", module)

    with pytest.raises(
        BackendUnavailable,
        match="backend implementation is unavailable",
    ) as captured:
        default_backend_registry().create(OFFICIAL_BACKEND_ID)

    assert captured.value.report_backend_id == OFFICIAL_BACKEND_ID
    assert "private/checkpoint/path" not in str(captured.value)


def test_registry_does_not_catch_unclassified_file_not_found() -> None:
    def unexpected_failure() -> object:
        raise FileNotFoundError("unrelated/input/path")

    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        factories={OFFICIAL_BACKEND_ID: unexpected_failure},
    )

    with pytest.raises(FileNotFoundError, match="unrelated/input/path"):
        registry.create(None)


def test_registry_does_not_catch_unclassified_factory_failure() -> None:
    def unexpected_failure() -> object:
        raise RuntimeError("unexpected")

    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        factories={OFFICIAL_BACKEND_ID: unexpected_failure},
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        registry.create(None)


def test_legacy_tf2_registry_entry_is_rejected_without_importing_tensorflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    original_import = builtins.__import__

    def record_import(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        imported.append(name)
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", record_import)

    with pytest.raises(BackendUnavailable, match="not a transcription backend") as captured:
        default_backend_registry().create(LEGACY_TF2_BACKEND_ID)

    assert captured.value.report_backend_id == LEGACY_TF2_BACKEND_ID
    assert not any(name == "tensorflow" or name.startswith("tensorflow.") for name in imported)
