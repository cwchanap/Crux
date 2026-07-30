from __future__ import annotations

# Factories deliberately import adapters lazily. Phase B defines the registry
# before the Phase A/C implementation modules exist.
# pylint: disable=import-outside-toplevel,import-error,no-name-in-module
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from src.benchmark.backend_reports import UNAVAILABLE_BACKEND_REPORT_ID
from src.benchmark.backends import TranscriptionBackend

OFFICIAL_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"
HEURISTIC_BACKEND_ID = "heuristic-onset-v1"
LEGACY_TF2_BACKEND_ID = "legacy-tf2-h5-v0"


class BackendUnavailable(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        report_backend_id: str,
        unknown_backend: bool,
    ) -> None:
        self.report_backend_id = report_backend_id
        self.unknown_backend = unknown_backend
        super().__init__(message)


class BackendLockUnavailable(RuntimeError):
    """A factory cannot start because a required backend lock is absent."""


@dataclass(frozen=True)
class BackendRegistry:
    default_backend_id: str
    factories: Mapping[str, Callable[[], TranscriptionBackend]]

    def create(
        self,
        backend_id: str | None,
        *,
        allow_emulated_diagnostics: bool = False,
    ) -> TranscriptionBackend:
        selected = self.default_backend_id if backend_id is None else backend_id
        factory = self.factories.get(selected)
        if factory is None:
            raise BackendUnavailable(
                "unknown backend",
                report_backend_id=UNAVAILABLE_BACKEND_REPORT_ID,
                unknown_backend=True,
            )
        try:
            if allow_emulated_diagnostics:
                return factory(allow_emulated_diagnostics=True)  # type: ignore[call-arg]
            return factory()
        except (ImportError, BackendLockUnavailable):
            raise BackendUnavailable(
                "backend implementation is unavailable",
                report_backend_id=selected,
                unknown_backend=False,
            ) from None


def _create_official_backend(
    *,
    allow_emulated_diagnostics: bool = False,
) -> TranscriptionBackend:
    from src.benchmark.backends.oaf_tf1 import create_backend

    if allow_emulated_diagnostics:
        return create_backend(allow_emulated_diagnostics=True)
    return create_backend()


def _create_heuristic_backend(
    *,
    allow_emulated_diagnostics: bool = False,
) -> TranscriptionBackend:
    del allow_emulated_diagnostics
    from src.benchmark.backends.heuristic import create_backend

    return create_backend()


def _reject_legacy_tf2_backend(
    *,
    allow_emulated_diagnostics: bool = False,
) -> TranscriptionBackend:
    del allow_emulated_diagnostics
    raise BackendUnavailable(
        "legacy TF2 backend is not a transcription backend",
        report_backend_id=LEGACY_TF2_BACKEND_ID,
        unknown_backend=False,
    )


def default_backend_registry() -> BackendRegistry:
    return BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        factories={
            OFFICIAL_BACKEND_ID: _create_official_backend,
            HEURISTIC_BACKEND_ID: _create_heuristic_backend,
            LEGACY_TF2_BACKEND_ID: _reject_legacy_tf2_backend,
        },
    )
