from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from src.benchmark.r2_corpus_models import RemoteObject

ObjectKeyResolutionStatus = Literal[
    "exact",
    "casefold",
    "missing",
    "ambiguous",
    "invalid_path",
]


@dataclass(frozen=True)
class ResolvedObjectKey:
    status: ObjectKeyResolutionStatus
    normalized_key: str | None
    remote: RemoteObject | None


def resolve_inventory_object_key(
    relative_path: str,
    *,
    base_object_key_dir: str,
    object_prefix: str,
    objects: tuple[RemoteObject, ...],
) -> ResolvedObjectKey:
    prefix_parts = _object_key_parts(object_prefix, require_trailing_separator=True)
    base_parts = _object_key_parts(base_object_key_dir, require_trailing_separator=False)
    relative_parts = _relative_path_parts(relative_path)
    if (
        prefix_parts is None
        or base_parts is None
        or relative_parts is None
        or not _is_under_prefix(base_parts, prefix_parts)
    ):
        return ResolvedObjectKey("invalid_path", None, None)
    assert prefix_parts is not None
    assert base_parts is not None
    assert relative_parts is not None

    resolved_parts = list(base_parts)
    for part in relative_parts or ():
        if part == "..":
            if len(resolved_parts) <= len(prefix_parts):
                return ResolvedObjectKey("invalid_path", None, None)
            resolved_parts.pop()
        else:
            resolved_parts.append(part)
    if not _is_under_prefix(tuple(resolved_parts), prefix_parts):
        return ResolvedObjectKey("invalid_path", None, None)

    normalized_key = "/".join(resolved_parts)
    candidates = tuple(
        remote
        for remote in objects
        if isinstance(remote.key, str) and _remote_is_under_prefix(remote.key, prefix_parts)
    )
    exact_matches = tuple(remote for remote in candidates if remote.key == normalized_key)
    if exact_matches:
        return ResolvedObjectKey(
            "exact" if len(exact_matches) == 1 else "ambiguous",
            normalized_key,
            exact_matches[0] if len(exact_matches) == 1 else None,
        )

    folded_key = normalized_key.casefold()
    casefold_matches = tuple(remote for remote in candidates if remote.key.casefold() == folded_key)
    if casefold_matches:
        return ResolvedObjectKey(
            "casefold" if len(casefold_matches) == 1 else "ambiguous",
            normalized_key,
            casefold_matches[0] if len(casefold_matches) == 1 else None,
        )
    return ResolvedObjectKey("missing", normalized_key, None)


def _object_key_parts(value: object, *, require_trailing_separator: bool) -> tuple[str, ...] | None:
    normalized = _slash_normalized(value)
    if normalized is None or require_trailing_separator and not normalized.endswith("/"):
        return None
    parts: list[str] = []
    for part in PurePosixPath(normalized).parts:
        if part == ".":
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    return tuple(parts) if parts else None


def _relative_path_parts(value: object) -> tuple[str, ...] | None:
    normalized = _slash_normalized(value)
    if normalized is None:
        return None
    parts: list[str] = []
    for part in PurePosixPath(normalized).parts:
        if part == ".":
            continue
        if part == "..":
            parts.append(part)
        else:
            parts.append(part)
    return tuple(parts)


def _slash_normalized(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    normalized = value.replace("\\", "/")
    if (
        normalized.startswith("/")
        or _has_drive_prefix(normalized)
        or PurePosixPath(normalized).is_absolute()
    ):
        return None
    return normalized


def _has_drive_prefix(value: str) -> bool:
    return len(value) >= 2 and value[0].isalpha() and value[1] == ":"


def _is_under_prefix(parts: tuple[str, ...], prefix_parts: tuple[str, ...]) -> bool:
    return len(parts) >= len(prefix_parts) and parts[: len(prefix_parts)] == prefix_parts


def _remote_is_under_prefix(key: str, prefix_parts: tuple[str, ...]) -> bool:
    remote_parts = _object_key_parts(key, require_trailing_separator=False)
    return remote_parts is not None and _is_under_prefix(remote_parts, prefix_parts)
