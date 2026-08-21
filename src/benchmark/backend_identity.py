from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from types import MappingProxyType
from typing import TypeAlias

JsonScalar: TypeAlias = None | bool | int | str | Decimal
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
SIX_PLACES = Decimal("0.000001")
OAF_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"
OAF_DESCRIPTOR_SCHEMA = "crux.transcription-backend-descriptor/v2"
OAF_DESCRIPTOR_KEYS = frozenset(
    {
        "architecture_id",
        "backend_id",
        "descriptor_schema",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "prediction_schema",
        "training_data_map_id",
        "upstream_source_commit",
    }
)
OAF_DESCRIPTOR_IDENTITIES = {
    "architecture_id": "magenta-oaf-model-tpu-drums-v1",
    "backend_id": OAF_BACKEND_ID,
    "descriptor_schema": OAF_DESCRIPTOR_SCHEMA,
    "model_id": "magenta-egmd-ckpt-569400-v1",
    "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
    "native_output_space_id": "magenta-oaf-midi88-a0-v1",
    "prediction_schema": "crux.drum-prediction-events/v2",
    "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
}
MUSCRIPTOR_BACKEND_ID = "muscriptor-v0.3.0-drums-v1"
MUSCRIPTOR_DESCRIPTOR_SCHEMA = "crux.transcription-backend-descriptor/v2"
MUSCRIPTOR_MODEL_ID_RE = re.compile(r"muscriptor-(medium|small)-[0-9a-f]{12}-[0-9a-f]{12}\Z")
MUSCRIPTOR_DESCRIPTOR_KEYS = frozenset(OAF_DESCRIPTOR_KEYS)
MUSCRIPTOR_DESCRIPTOR_IDENTITIES = {
    "architecture_id": "muscriptor-transformer-v0.3.0",
    "backend_id": MUSCRIPTOR_BACKEND_ID,
    "descriptor_schema": MUSCRIPTOR_DESCRIPTOR_SCHEMA,
    "native_metadata_schema_id": "muscriptor-note-start-metadata-v1",
    "native_output_space_id": "muscriptor-drums-midi128-v1",
    "prediction_schema": "crux.drum-prediction-events/v2",
    "training_data_map_id": "muscriptor-training-data-v0.3.0",
}
IDM_BACKEND_ID = "idm-44-train-kits-v1"
IDM_DESCRIPTOR_SCHEMA = "crux.transcription-backend-descriptor/v2"
IDM_NATIVE_METADATA_SCHEMA_ID = "idm-peak-event-metadata-v1"
IDM_MODEL_ID_RE = re.compile(r"idm-44-train-kits-[0-9a-f]{12}-[0-9a-f]{12}\Z")
IDM_DESCRIPTOR_KEYS = frozenset(OAF_DESCRIPTOR_KEYS)
IDM_DESCRIPTOR_IDENTITIES = {
    "architecture_id": "inverse-drum-machine-v0.1.0",
    "backend_id": IDM_BACKEND_ID,
    "descriptor_schema": IDM_DESCRIPTOR_SCHEMA,
    "native_metadata_schema_id": IDM_NATIVE_METADATA_SCHEMA_ID,
    "native_output_space_id": "idm-44-train-kits-9class-v1",
    "prediction_schema": "crux.drum-prediction-events/v2",
    "training_data_map_id": "idm-training-contract-44-train-kits-v1",
    "upstream_source_commit": "456656868538205ef756912c7cf5b0fd936de8af",
}


_DESCRIPTOR_POLICIES = {
    OAF_BACKEND_ID: (OAF_DESCRIPTOR_KEYS, OAF_DESCRIPTOR_IDENTITIES, {}),
    MUSCRIPTOR_BACKEND_ID: (
        MUSCRIPTOR_DESCRIPTOR_KEYS,
        MUSCRIPTOR_DESCRIPTOR_IDENTITIES,
        {"model_id": MUSCRIPTOR_MODEL_ID_RE},
    ),
    IDM_BACKEND_ID: (
        IDM_DESCRIPTOR_KEYS,
        IDM_DESCRIPTOR_IDENTITIES,
        {"model_id": IDM_MODEL_ID_RE},
    ),
}


class StrictJsonError(ValueError):
    pass


@dataclass(frozen=True)
class BackendDescriptor:
    payload: Mapping[str, str]
    sha256: str


def quantize_six(value: float) -> Decimal:
    if not math.isfinite(value):
        raise StrictJsonError("nonfinite binary float")
    try:
        quantized = Decimal.from_float(value).quantize(SIX_PLACES, rounding=ROUND_HALF_EVEN)
    except InvalidOperation as error:
        raise StrictJsonError("binary float quantization failed") from error
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


# The frozen descriptor families keep all identity checks in one shared validator.
# pylint: disable-next=too-many-branches
def normalize_known_backend_descriptor(value: Mapping[str, object]) -> dict[str, str]:
    backend_id = value.get("backend_id")
    policy = _DESCRIPTOR_POLICIES.get(backend_id) if isinstance(backend_id, str) else None
    if policy is None:
        raise StrictJsonError("descriptor backend_id is unknown")
    expected_keys, expected_identities, pattern_fields = policy
    if set(value) != set(expected_keys):
        raise StrictJsonError("descriptor must contain the exact key set")
    descriptor: dict[str, str] = {}
    for field, field_value in value.items():
        if not isinstance(field_value, str) or not field_value:
            raise StrictJsonError(f"descriptor {field} must be a nonempty string")
        descriptor[field] = field_value
    for field, expected in expected_identities.items():
        if descriptor[field] != expected:
            raise StrictJsonError(f"descriptor {field} does not match frozen identity")
    for field, pattern in pattern_fields.items():
        if pattern.fullmatch(descriptor[field]) is None:
            raise StrictJsonError(f"descriptor {field} does not match frozen pattern")
    commit = descriptor["upstream_source_commit"]
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise StrictJsonError("descriptor upstream_source_commit must be lowercase Git identity")
    return descriptor


def expected_muscriptor_model_id(lock: object) -> str:
    """Derive the exact MuScriptor descriptor model ID from its frozen lock."""
    from src.benchmark.muscriptor_model import derive_muscriptor_model_id

    return derive_muscriptor_model_id(lock)


def validate_schema_golden(schema: str, content: bytes) -> None:
    if schema != OAF_DESCRIPTOR_SCHEMA:
        raise ValueError("unsupported schema golden")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise StrictJsonError("schema golden must have one final newline")
    value = strict_json_loads(content[:-1], require_canonical=True)
    if not isinstance(value, dict):
        raise StrictJsonError("descriptor schema golden must be an object")
    descriptor = normalize_known_backend_descriptor(value)
    if descriptor["descriptor_schema"] != schema:
        raise StrictJsonError("descriptor schema golden does not match requested schema")


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
