from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from types import MappingProxyType
from typing import TypeAlias

JsonScalar: TypeAlias = None | bool | int | str | Decimal
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
SIX_PLACES = Decimal("0.000001")


class StrictJsonError(ValueError):
    pass


@dataclass(frozen=True)
class BackendDescriptor:
    payload: Mapping[str, str]
    sha256: str


def quantize_six(value: float) -> Decimal:
    if not math.isfinite(value):
        raise StrictJsonError("nonfinite binary float")
    quantized = Decimal.from_float(value).quantize(SIX_PLACES, rounding=ROUND_HALF_EVEN)
    return Decimal(0) if quantized.is_zero() else quantized


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def require_sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise StrictJsonError(f"{field} must be lowercase SHA-256")
    return value


def canonical_json_bytes(value: JsonValue, *, trailing_newline: bool = False) -> bytes:
    try:
        content = _render_json(value).encode("utf-8")
    except UnicodeEncodeError:
        raise StrictJsonError("JSON strings must be valid UTF-8") from None
    return content + (b"\n" if trailing_newline else b"")


def strict_json_loads(content: bytes, *, require_canonical: bool = False) -> JsonValue:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise StrictJsonError("JSON content must be valid UTF-8") from None

    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_float=Decimal,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise StrictJsonError(f"invalid JSON: {error.msg}") from None

    canonical = canonical_json_bytes(value)
    if require_canonical and content != canonical:
        raise StrictJsonError("JSON bytes are not canonical")
    return value


def build_descriptor(
    payload: Mapping[str, object],
    allowed_keys: Collection[str],
    schema: str,
) -> BackendDescriptor:
    if set(payload) != set(allowed_keys):
        raise StrictJsonError("descriptor must contain the exact key set")
    if any(not isinstance(value, str) for value in payload.values()):
        raise StrictJsonError("descriptor fields must be strings")
    if payload.get("descriptor_schema") != schema:
        raise StrictJsonError(f"descriptor_schema must be {schema}")

    descriptor_payload = {key: value for key, value in payload.items() if isinstance(value, str)}
    content = canonical_json_bytes(descriptor_payload)
    return BackendDescriptor(
        payload=MappingProxyType(descriptor_payload),
        sha256=sha256_hex(content),
    )


# The closed JSON union is clearest as one return per supported value kind.
# pylint: disable-next=too-many-return-statements
def _render_json(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Decimal):
        return _render_decimal(value)
    if isinstance(value, list):
        return "[" + ",".join(_render_json(item) for item in value) + "]"
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise StrictJsonError("JSON object keys must be strings")
        return (
            "{"
            + ",".join(f"{_render_json(key)}:{_render_json(value[key])}" for key in sorted(value))
            + "}"
        )
    raise StrictJsonError(f"unsupported JSON value: {type(value).__name__}")


def _render_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise StrictJsonError("nonfinite decimal")
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _object_from_pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_constant(constant: str) -> None:
    raise StrictJsonError(f"nonfinite JSON constant: {constant}")
