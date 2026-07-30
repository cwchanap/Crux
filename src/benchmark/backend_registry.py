from __future__ import annotations

# Factories deliberately import adapters lazily. Phase B defines the registry
# before the Phase A/C implementation modules exist.
# pylint: disable=import-outside-toplevel,import-error,no-name-in-module
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from src.benchmark.backend_reports import UNAVAILABLE_BACKEND_REPORT_ID
from src.benchmark.backends import TranscriptionBackend

OFFICIAL_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"
HEURISTIC_BACKEND_ID = "heuristic-onset-v1"
LEGACY_TF2_BACKEND_ID = "legacy-tf2-h5-v0"
SealState = Literal["preseal", "sealed"]


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


class BackendNotSealed(RuntimeError):
    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        super().__init__("backend is not sealed")


class BackendIntegrityUnavailable(RuntimeError):
    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        super().__init__("backend integrity is unavailable")


@dataclass(frozen=True)
class BackendRegistration:
    backend_id: str
    seal_state: SealState
    factory: Callable[..., TranscriptionBackend]


@dataclass(frozen=True)
class BackendRegistry:
    default_backend_id: str
    registrations: Mapping[str, BackendRegistration]

    def create(
        self,
        backend_id: str | None,
        *,
        allow_emulated_diagnostics: bool = False,
    ) -> TranscriptionBackend:
        selected = self.default_backend_id if backend_id is None else backend_id
        registration = self.registrations.get(selected)
        if registration is None:
            raise BackendUnavailable(
                "unknown backend",
                report_backend_id=UNAVAILABLE_BACKEND_REPORT_ID,
                unknown_backend=True,
            )
        if registration.seal_state == "preseal":
            raise BackendNotSealed(registration.backend_id)
        try:
            if allow_emulated_diagnostics:
                return registration.factory(allow_emulated_diagnostics=True)
            return registration.factory()
        except BackendLockUnavailable:
            raise BackendIntegrityUnavailable(registration.backend_id) from None
        except ImportError:
            raise BackendUnavailable(
                "backend implementation is unavailable",
                report_backend_id=registration.backend_id,
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
        registrations={
            OFFICIAL_BACKEND_ID: BackendRegistration(
                backend_id=OFFICIAL_BACKEND_ID,
                seal_state="preseal",
                factory=_create_official_backend,
            ),
            HEURISTIC_BACKEND_ID: BackendRegistration(
                backend_id=HEURISTIC_BACKEND_ID,
                seal_state="sealed",
                factory=_create_heuristic_backend,
            ),
            LEGACY_TF2_BACKEND_ID: BackendRegistration(
                backend_id=LEGACY_TF2_BACKEND_ID,
                seal_state="sealed",
                factory=_reject_legacy_tf2_backend,
            ),
        },
    )
