"""Small backend selector for the single supported OaF implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from src.benchmark.backends import TranscriptionBackend


class BackendUnavailable(RuntimeError):
    """The requested backend selector is not registered."""


BackendFactory = Callable[..., TranscriptionBackend]


@dataclass(frozen=True)
class BackendRegistry:
    default_backend_id: str
    factories: Mapping[str, BackendFactory]

    def __post_init__(self) -> None:
        if self.default_backend_id not in self.factories:
            raise ValueError("default backend must have a factory")

    def create(self, backend_id: str | None = None, **kwargs: object) -> TranscriptionBackend:
        selected = self.default_backend_id if backend_id is None else backend_id
        factory = self.factories.get(selected)
        if factory is None:
            raise BackendUnavailable("unknown backend")
        try:
            return factory(**kwargs)
        except BackendUnavailable:
            raise
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise BackendUnavailable(str(error)) from error


def _create_oaf(**kwargs: object) -> TranscriptionBackend:
    from src.benchmark.backends.oaf import create_backend

    return create_backend(**kwargs)


def default_backend_registry() -> BackendRegistry:
    return BackendRegistry(default_backend_id="oaf", factories={"oaf": _create_oaf})


__all__ = ["BackendRegistry", "BackendUnavailable", "default_backend_registry"]
