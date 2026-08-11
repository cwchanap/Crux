"""Exact post-adapter oracle for the OaF smoke fixture."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    require_sha256,
    strict_json_loads,
)
from src.benchmark.prediction_artifact import MappedPrediction

SMOKE_ORACLE_SCHEMA = "crux.oaf-smoke-oracle/v2"
_TOP_LEVEL_KEYS = frozenset({"schema", "backend_id", "input_audio_sha256", "native_events"})
_EVENT_KEYS = frozenset(
    {
        "time_sec_binary64",
        "native_class_id",
        "model_output_bin",
        "native_midi_note",
        "upstream_8hit_group_id",
        "confidence_binary64",
        "velocity_midi",
    }
)
_BINARY64_HEX_LENGTH = struct.calcsize(">d") * 2


@dataclass(frozen=True)
class SmokeOracleEvent:
    time_sec_binary64: str
    native_class_id: str
    model_output_bin: int | None
    native_midi_note: int | None
    upstream_8hit_group_id: str | None
    confidence_binary64: str | None
    velocity_midi: int | None


@dataclass(frozen=True)
class SmokeOracle:
    schema: Literal["crux.oaf-smoke-oracle/v2"]
    backend_id: str
    input_audio_sha256: str
    native_events: tuple[SmokeOracleEvent, ...]


def render_smoke_oracle(prediction: MappedPrediction) -> bytes:
    """Render the retained native fields in one canonical JSON document."""
    if not isinstance(prediction, MappedPrediction):
        raise TypeError("prediction must be MappedPrediction")
    backend_id = prediction.descriptor.payload.get("backend_id")
    if backend_id != OAF_BACKEND_ID:
        raise ValueError("prediction descriptor backend_id is invalid")
    require_sha256(prediction.audio.input_audio_sha256, "input_audio_sha256")
    events = tuple(_event_from_native(event.native) for event in prediction.events)
    document: dict[str, JsonValue] = {
        "backend_id": backend_id,
        "input_audio_sha256": prediction.audio.input_audio_sha256,
        "native_events": [_event_payload(event) for event in events],
        "schema": SMOKE_ORACLE_SCHEMA,
    }
    return canonical_json_bytes(document, trailing_newline=True)


def read_smoke_oracle(content: bytes) -> SmokeOracle:
    """Read and validate one canonical smoke-oracle document."""
    if not isinstance(content, bytes):
        raise TypeError("oracle content must be bytes")
    if not content.endswith(b"\n") or content.endswith(b"\n\n"):
        raise ValueError("oracle must have one final newline")
    try:
        value = strict_json_loads(content[:-1], require_canonical=True)
        return _oracle_from_value(value)
    except (StrictJsonError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and not isinstance(error, StrictJsonError):
            raise
        raise ValueError(str(error)) from None


def assert_smoke_oracle_matches(prediction: MappedPrediction, oracle: SmokeOracle) -> None:
    """Assert exact identity and native-event equality against an oracle."""
    if not isinstance(prediction, MappedPrediction):
        raise AssertionError("prediction must be MappedPrediction")
    if not isinstance(oracle, SmokeOracle):
        raise AssertionError("oracle must be SmokeOracle")
    backend_id = prediction.descriptor.payload.get("backend_id")
    if oracle.schema != SMOKE_ORACLE_SCHEMA:
        raise AssertionError("oracle schema mismatch")
    if oracle.backend_id != backend_id:
        raise AssertionError("oracle backend_id mismatch")
    if oracle.input_audio_sha256 != prediction.audio.input_audio_sha256:
        raise AssertionError("oracle input_audio_sha256 mismatch")
    expected = tuple(_event_from_native(event.native) for event in prediction.events)
    if oracle.native_events != expected:
        for index, (actual, wanted) in enumerate(zip(oracle.native_events, expected, strict=False)):
            if actual != wanted:
                raise AssertionError(f"oracle native event mismatch at index {index}")
        raise AssertionError("oracle native event count mismatch")


def validate_schema_golden(schema: str, content: bytes) -> None:
    if schema != SMOKE_ORACLE_SCHEMA:
        raise ValueError("unsupported schema golden")
    read_smoke_oracle(content)


def _oracle_from_value(value: object) -> SmokeOracle:
    if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_KEYS:
        raise ValueError("oracle top-level keys are invalid")
    if value["schema"] != SMOKE_ORACLE_SCHEMA:
        raise ValueError("oracle schema is invalid")
    backend_id = value["backend_id"]
    if backend_id != OAF_BACKEND_ID:
        raise ValueError("oracle backend_id is invalid")
    input_audio_sha256 = value["input_audio_sha256"]
    if not isinstance(input_audio_sha256, str):
        raise ValueError("oracle input_audio_sha256 is invalid")
    try:
        require_sha256(input_audio_sha256, "input_audio_sha256")
    except StrictJsonError as error:
        raise ValueError(str(error)) from None
    raw_events = value["native_events"]
    if not isinstance(raw_events, list):
        raise ValueError("oracle native_events is invalid")
    return SmokeOracle(
        schema=SMOKE_ORACLE_SCHEMA,
        backend_id=backend_id,
        input_audio_sha256=input_audio_sha256,
        native_events=tuple(_event_from_value(raw_event) for raw_event in raw_events),
    )


def _event_from_value(value: object) -> SmokeOracleEvent:
    if not isinstance(value, Mapping) or set(value) != _EVENT_KEYS:
        raise ValueError("oracle event keys are invalid")
    time_sec_binary64 = _binary64_hex(value["time_sec_binary64"], "time_sec_binary64")
    confidence_value = value["confidence_binary64"]
    confidence_binary64 = (
        None if confidence_value is None else _binary64_hex(confidence_value, "confidence_binary64")
    )
    native_class_id = value["native_class_id"]
    group = value["upstream_8hit_group_id"]
    if not isinstance(native_class_id, str) or not native_class_id:
        raise ValueError("oracle native_class_id is invalid")
    if group is not None and (not isinstance(group, str) or not group):
        raise ValueError("oracle upstream_8hit_group_id is invalid")
    return SmokeOracleEvent(
        time_sec_binary64=time_sec_binary64,
        native_class_id=native_class_id,
        model_output_bin=_int_or_none(value["model_output_bin"], "model_output_bin"),
        native_midi_note=_int_or_none(value["native_midi_note"], "native_midi_note"),
        upstream_8hit_group_id=group,
        confidence_binary64=confidence_binary64,
        velocity_midi=_int_or_none(value["velocity_midi"], "velocity_midi"),
    )


def _event_from_native(native: object) -> SmokeOracleEvent:
    if not hasattr(native, "native_metadata"):
        raise ValueError("native event is invalid")
    metadata = native.native_metadata
    if not isinstance(metadata, Mapping) or set(metadata) != {"upstream_8hit_group_id"}:
        raise ValueError("native event metadata is invalid")
    group = metadata["upstream_8hit_group_id"]
    if group is not None and (not isinstance(group, str) or not group):
        raise ValueError("native event group is invalid")
    return SmokeOracleEvent(
        time_sec_binary64=_float_binary64(native.time_sec, "time_sec"),
        native_class_id=_required_string(native.native_class_id, "native_class_id"),
        model_output_bin=_int_or_none(native.model_output_bin, "model_output_bin"),
        native_midi_note=_int_or_none(native.native_midi_note, "native_midi_note"),
        upstream_8hit_group_id=group,
        confidence_binary64=(
            None if native.confidence is None else _float_binary64(native.confidence, "confidence")
        ),
        velocity_midi=_int_or_none(native.velocity_midi, "velocity_midi"),
    )


def _event_payload(event: SmokeOracleEvent) -> dict[str, JsonValue]:
    return {
        "confidence_binary64": event.confidence_binary64,
        "model_output_bin": event.model_output_bin,
        "native_class_id": event.native_class_id,
        "native_midi_note": event.native_midi_note,
        "time_sec_binary64": event.time_sec_binary64,
        "upstream_8hit_group_id": event.upstream_8hit_group_id,
        "velocity_midi": event.velocity_midi,
    }


def _binary64_hex(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != _BINARY64_HEX_LENGTH:
        raise ValueError(f"oracle {field} is invalid")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"oracle {field} is invalid")
    try:
        struct.unpack(">d", bytes.fromhex(value))
    except (ValueError, struct.error):
        raise ValueError(f"oracle {field} is invalid") from None
    return value


def _float_binary64(value: object, field: str) -> str:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"native event {field} is invalid")
    return struct.pack(">d", value).hex()


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"native event {field} is invalid")
    return value


def _int_or_none(value: object, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"oracle {field} is invalid")
    return value


__all__ = [
    "SMOKE_ORACLE_SCHEMA",
    "SmokeOracle",
    "SmokeOracleEvent",
    "assert_smoke_oracle_matches",
    "read_smoke_oracle",
    "render_smoke_oracle",
    "validate_schema_golden",
]
