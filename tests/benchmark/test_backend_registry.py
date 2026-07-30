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
    BackendIntegrityUnavailable,
    BackendLockUnavailable,
    BackendNotSealed,
    BackendRegistration,
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


class CountingFactory:
    def __init__(self, result: DescriptorBackend | BaseException) -> None:
        self.calls = 0
        self._result = result

    def __call__(self, **_kwargs: object) -> DescriptorBackend:
        self.calls += 1
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


def test_preseal_official_backend_returns_typed_not_sealed() -> None:
    unexpected_factory = CountingFactory(DescriptorBackend(OFFICIAL_BACKEND_ID))
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="preseal",
                factory=unexpected_factory,
            )
        },
    )

    with pytest.raises(BackendNotSealed) as captured:
        registry.create(None)

    assert captured.value.backend_id == OFFICIAL_BACKEND_ID
    assert unexpected_factory.calls == 0


def test_sealed_backend_invokes_its_registered_factory() -> None:
    backend = DescriptorBackend(OFFICIAL_BACKEND_ID)
    factory = CountingFactory(backend)
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="sealed",
                factory=factory,
            )
        },
    )

    assert registry.create(None) is backend
    assert factory.calls == 1


def test_sealed_missing_lock_returns_typed_integrity_unavailable() -> None:
    factory = CountingFactory(BackendLockUnavailable("missing lock"))
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="sealed",
                factory=factory,
            )
        },
    )

    with pytest.raises(BackendIntegrityUnavailable) as captured:
        registry.create(None)

    assert captured.value.backend_id == OFFICIAL_BACKEND_ID
    assert factory.calls == 1


@pytest.mark.parametrize(
    ("seal_state", "official_result"),
    [
        ("preseal", DescriptorBackend(OFFICIAL_BACKEND_ID)),
        ("sealed", BackendLockUnavailable("missing lock")),
    ],
)
def test_registry_state_failures_do_not_invoke_heuristic(
    seal_state: str,
    official_result: DescriptorBackend | BaseException,
) -> None:
    official_factory = CountingFactory(official_result)
    heuristic_factory = CountingFactory(DescriptorBackend(HEURISTIC_BACKEND_ID))
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state=seal_state,  # type: ignore[arg-type]
                factory=official_factory,
            ),
            HEURISTIC_BACKEND_ID: BackendRegistration(
                backend_id=HEURISTIC_BACKEND_ID,
                seal_state="sealed",
                factory=heuristic_factory,
            ),
        },
    )

    expected = BackendNotSealed if seal_state == "preseal" else BackendIntegrityUnavailable
    with pytest.raises(expected):
        registry.create(None)

    assert heuristic_factory.calls == 0


def test_registry_uses_official_backend_only_when_backend_is_omitted() -> None:
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="sealed",
                factory=lambda: DescriptorBackend(OFFICIAL_BACKEND_ID),
            )
        },
    )

    backend = registry.create(None)

    assert backend.descriptor().payload["backend_id"] == OFFICIAL_BACKEND_ID


def test_registry_selects_heuristic_only_when_explicitly_requested() -> None:
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="sealed",
                factory=lambda: DescriptorBackend(OFFICIAL_BACKEND_ID),
            ),
            HEURISTIC_BACKEND_ID: BackendRegistration(
                backend_id=HEURISTIC_BACKEND_ID,
                seal_state="sealed",
                factory=lambda: DescriptorBackend(HEURISTIC_BACKEND_ID),
            ),
        },
    )

    backend = registry.create(HEURISTIC_BACKEND_ID)

    assert backend.descriptor().payload["backend_id"] == HEURISTIC_BACKEND_ID


def test_unknown_backend_uses_stable_report_namespace_without_retaining_input() -> None:
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="sealed",
                factory=lambda: DescriptorBackend(OFFICIAL_BACKEND_ID),
            )
        },
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
    assert set(registry.registrations) == {
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


@pytest.mark.parametrize("backend_id", [HEURISTIC_BACKEND_ID])
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


def test_sealed_checked_in_lock_maps_to_backend_integrity_unavailable() -> None:
    def missing_lock() -> object:
        raise BackendLockUnavailable("private/checkpoint/path")

    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="sealed",
                factory=missing_lock,  # type: ignore[arg-type]
            )
        },
    )

    with pytest.raises(BackendIntegrityUnavailable) as captured:
        registry.create(OFFICIAL_BACKEND_ID)

    assert captured.value.backend_id == OFFICIAL_BACKEND_ID
    assert "private/checkpoint/path" not in str(captured.value)


def test_registry_does_not_catch_unclassified_file_not_found() -> None:
    def unexpected_failure() -> object:
        raise FileNotFoundError("unrelated/input/path")

    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="sealed",
                factory=unexpected_failure,  # type: ignore[arg-type]
            )
        },
    )

    with pytest.raises(FileNotFoundError, match="unrelated/input/path"):
        registry.create(None)


def test_registry_does_not_catch_unclassified_factory_failure() -> None:
    def unexpected_failure() -> object:
        raise RuntimeError("unexpected")

    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="sealed",
                factory=unexpected_failure,  # type: ignore[arg-type]
            )
        },
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
