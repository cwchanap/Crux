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
OAF_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"
HEURISTIC_BACKEND_ID = "heuristic-onset-v1"
OAF_DESCRIPTOR_SCHEMA = "crux.transcription-backend-descriptor/v1"
HEURISTIC_DESCRIPTOR_SCHEMA = "crux.heuristic-backend-descriptor/v1"
OAF_DESCRIPTOR_KEYS = frozenset(
    {
        "architecture_id",
        "backend_id",
        "backend_lock_sha256",
        "descriptor_schema",
        "model_artifact_set_sha256",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "prediction_schema",
        "protocol_schema",
        "runtime_image_manifest_digest",
        "runtime_lock_sha256",
        "training_data_map_id",
        "upstream_source_commit",
    }
)
HEURISTIC_DESCRIPTOR_KEYS = frozenset(
    {
        "adapter_source_manifest_sha256",
        "architecture_id",
        "backend_id",
        "descriptor_schema",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "parameter_lock_sha256",
        "prediction_schema",
    }
)
OAF_DESCRIPTOR_IDENTITIES = {
    "architecture_id": "magenta-oaf-model-tpu-drums-v1",
    "backend_id": OAF_BACKEND_ID,
    "descriptor_schema": OAF_DESCRIPTOR_SCHEMA,
    "model_id": "magenta-egmd-ckpt-569400-v1",
    "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
    "native_output_space_id": "magenta-oaf-midi88-a0-v1",
    "prediction_schema": "crux.drum-prediction-events/v1",
    "protocol_schema": "crux.transcription-runner/v1",
    "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
}
HEURISTIC_DESCRIPTOR_IDENTITIES = {
    "architecture_id": "librosa-onset-centroid-zcr-v1",
    "backend_id": HEURISTIC_BACKEND_ID,
    "descriptor_schema": HEURISTIC_DESCRIPTOR_SCHEMA,
    "model_id": "crux-heuristic-onset-nonmodel-v1",
    "native_metadata_schema_id": "crux-empty-native-metadata-v1",
    "native_output_space_id": "crux-heuristic-midi7-v1",
    "prediction_schema": "crux.drum-prediction-events/v1",
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


# The two frozen descriptor shapes keep all identity checks in one shared validator.
# pylint: disable-next=too-many-branches
def normalize_known_backend_descriptor(value: Mapping[str, object]) -> dict[str, str]:
    backend_id = value.get("backend_id")
    if backend_id == OAF_BACKEND_ID:
        expected_schema = OAF_DESCRIPTOR_SCHEMA
        expected_keys = OAF_DESCRIPTOR_KEYS
        expected_identities = OAF_DESCRIPTOR_IDENTITIES
    elif backend_id == HEURISTIC_BACKEND_ID:
        expected_schema = HEURISTIC_DESCRIPTOR_SCHEMA
        expected_keys = HEURISTIC_DESCRIPTOR_KEYS
        expected_identities = HEURISTIC_DESCRIPTOR_IDENTITIES
    else:
        raise StrictJsonError("descriptor backend_id is unknown")
    if set(value) != set(expected_keys):
        raise StrictJsonError("descriptor must contain the exact key set")
    if value.get("descriptor_schema") != expected_schema:
        raise StrictJsonError(f"descriptor_schema must be {expected_schema}")
    descriptor: dict[str, str] = {}
    for field, field_value in value.items():
        if not isinstance(field_value, str) or not field_value:
            raise StrictJsonError(f"descriptor {field} must be a nonempty string")
        descriptor[field] = field_value
    for field, expected in expected_identities.items():
        if descriptor[field] != expected:
            raise StrictJsonError(f"descriptor {field} does not match frozen identity")
    if backend_id == OAF_BACKEND_ID:
        for field in (
            "backend_lock_sha256",
            "model_artifact_set_sha256",
            "runtime_lock_sha256",
        ):
            require_sha256(descriptor[field], f"descriptor {field}")
        commit = descriptor["upstream_source_commit"]
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise StrictJsonError(
                "descriptor upstream_source_commit must be lowercase Git identity"
            )
        image_digest = descriptor["runtime_image_manifest_digest"]
        if not image_digest.startswith("sha256:") or len(image_digest) != 71:
            raise StrictJsonError(
                "descriptor runtime_image_manifest_digest must be sha256 identity"
            )
        require_sha256(image_digest[7:], "descriptor runtime_image_manifest_digest")
    else:
        for field in ("adapter_source_manifest_sha256", "parameter_lock_sha256"):
            require_sha256(descriptor[field], f"descriptor {field}")
    return descriptor


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
