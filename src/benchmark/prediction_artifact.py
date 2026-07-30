from __future__ import annotations

# Schema validation uses exact integer checks to reject booleans and necessarily
# branches across the fixed header/event/terminal record variants.
# pylint: disable=too-many-branches,unidiomatic-typecheck
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import cast

from src.benchmark.backend_identity import (
    BackendDescriptor,
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    quantize_six,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backend_publication import publish_immutable_bytes, read_regular_file_no_follow
from src.benchmark.backends import CanonicalAudio, NativeEvent, NativePrediction, PublishedArtifact

PREDICTION_SCHEMA = "crux.drum-prediction-events/v1"
OAF_METADATA_SCHEMA = "magenta-oaf-native-metadata-v1"
EMPTY_METADATA_SCHEMA = "crux-empty-native-metadata-v1"
OAF_GROUP_IDS = frozenset(
    {"kick", "snare", "toms", "hihat", "ride", "ride_bell", "crash", "sticks"}
)
NATIVE_METADATA_SCHEMAS = {
    OAF_METADATA_SCHEMA: {
        "upstream_8hit_group_id": OAF_GROUP_IDS | {None},
    },
    EMPTY_METADATA_SCHEMA: {},
}
HEADER_KEYS = frozenset(
    {
        "architecture_id",
        "artifact_role",
        "audio_frame_count",
        "backend_descriptor",
        "backend_descriptor_sha256",
        "backend_lock_sha256",
        "byte_length",
        "channel_count",
        "input_audio_sha256",
        "input_view_id",
        "model_artifact_set_sha256",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "parameter_lock_sha256",
        "record_type",
        "runtime_lock_sha256",
        "sample_rate",
        "sample_width_bytes",
        "schema",
        "source_audio_id",
        "source_audio_sha256",
        "training_data_map_id",
        "upstream_source_commit",
    }
)
EVENT_KEYS = frozenset(
    {
        "canonical_class",
        "confidence",
        "event_index",
        "mapping_status",
        "model_output_bin",
        "native_class_id",
        "native_metadata",
        "native_midi_note",
        "prediction_map_version",
        "record_type",
        "time_sec",
        "velocity_midi",
    }
)
TERMINAL_KEYS = frozenset({"event_count", "prefix_sha256", "record_type"})


class PredictionArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class PredictionArtifact:
    prediction: NativePrediction
    event_count: int
    prefix_sha256: str
    artifact_sha256: str
    content: bytes


@dataclass(frozen=True)
class _NormalizedEvent:
    time_sec: Decimal
    native_class_id: str
    model_output_bin: int | None
    native_midi_note: int | None
    native_metadata: dict[str, str | None]
    confidence: Decimal | None
    velocity_midi: int | None

    @property
    def sort_key(self) -> tuple[Decimal, str, int, int, int, Decimal]:
        return (
            self.time_sec,
            self.native_class_id,
            self.model_output_bin if self.model_output_bin is not None else -1,
            self.native_midi_note if self.native_midi_note is not None else -1,
            self.velocity_midi if self.velocity_midi is not None else -1,
            self.confidence if self.confidence is not None else Decimal(-1),
        )


def render_prediction_artifact(prediction: NativePrediction) -> bytes:
    try:
        header = _build_header(prediction)
        metadata_schema = cast(str, header["native_metadata_schema_id"])
        events = sorted(
            (_normalize_event(event, metadata_schema) for event in prediction.events),
            key=lambda event: event.sort_key,
        )
        for first, second in zip(events, events[1:], strict=False):
            if first.sort_key == second.sort_key:
                raise PredictionArtifactError("duplicate_native_event")

        prefix = canonical_json_bytes(header, trailing_newline=True)
        for event_index, event in enumerate(events):
            prefix += canonical_json_bytes(
                _event_record(event, event_index),
                trailing_newline=True,
            )
        terminal = {
            "event_count": len(events),
            "prefix_sha256": sha256_hex(prefix),
            "record_type": "terminal",
        }
        return prefix + canonical_json_bytes(terminal, trailing_newline=True)
    except PredictionArtifactError:
        raise
    except (StrictJsonError, TypeError, ValueError) as error:
        raise PredictionArtifactError(str(error)) from None


def read_prediction_artifact(content: bytes) -> PredictionArtifact:
    try:
        return _read_prediction_artifact(content)
    except PredictionArtifactError:
        raise
    except (StrictJsonError, TypeError, ValueError) as error:
        raise PredictionArtifactError(str(error)) from None


def publish_prediction_artifact(
    path: Path,
    prediction: NativePrediction,
) -> PublishedArtifact:
    content = render_prediction_artifact(prediction)
    expected_sha256 = sha256_hex(content)
    published = publish_immutable_bytes(
        path,
        content,
        expected_sha256,
        role="prediction",
    )
    persisted = read_regular_file_no_follow(path)
    artifact = read_prediction_artifact(persisted)
    if artifact.content != content or artifact.artifact_sha256 != expected_sha256:
        raise PredictionArtifactError("published prediction bytes changed")
    return published


def _read_prediction_artifact(content: bytes) -> PredictionArtifact:
    if not isinstance(content, bytes):
        raise PredictionArtifactError("prediction artifact must be bytes")
    if not content or not content.endswith(b"\n"):
        raise PredictionArtifactError("prediction artifact must end with a newline")

    physical_lines = content.splitlines(keepends=True)
    if len(physical_lines) < 2:
        raise PredictionArtifactError("prediction artifact requires header and terminal")
    if any(not line.endswith(b"\n") or line == b"\n" for line in physical_lines):
        raise PredictionArtifactError("prediction artifact contains a blank or partial line")

    records: list[dict[str, JsonValue]] = []
    for line in physical_lines:
        parsed = strict_json_loads(line[:-1], require_canonical=True)
        if not isinstance(parsed, dict):
            raise PredictionArtifactError("prediction records must be objects")
        records.append(parsed)

    header = records[0]
    terminal = records[-1]
    event_records = records[1:-1]
    _require_exact_keys(header, HEADER_KEYS, "header")
    _require_exact_keys(terminal, TERMINAL_KEYS, "terminal")
    for event in event_records:
        _require_exact_keys(event, EVENT_KEYS, "event")

    if header["record_type"] != "header":
        raise PredictionArtifactError("first record must be header")
    if terminal["record_type"] != "terminal":
        raise PredictionArtifactError("last record must be terminal")
    if any(event["record_type"] != "event" for event in event_records):
        raise PredictionArtifactError("records between header and terminal must be events")

    prediction = _prediction_from_records(header, event_records)
    for expected_index, event in enumerate(event_records):
        if type(event["event_index"]) is not int or event["event_index"] != expected_index:
            raise PredictionArtifactError("event_index must be contiguous from zero")
        _require_native_mapping_fields(event)

    event_count = terminal["event_count"]
    if type(event_count) is not int or event_count != len(event_records):
        raise PredictionArtifactError("terminal event_count mismatch")
    prefix_sha256 = terminal["prefix_sha256"]
    if not isinstance(prefix_sha256, str):
        raise PredictionArtifactError("prefix_sha256 must be a string")
    require_sha256(prefix_sha256, "prefix_sha256")
    actual_prefix_sha256 = sha256_hex(b"".join(physical_lines[:-1]))
    if prefix_sha256 != actual_prefix_sha256:
        raise PredictionArtifactError("terminal prefix_sha256 mismatch")

    rendered = render_prediction_artifact(prediction)
    if rendered != content:
        raise PredictionArtifactError("prediction artifact does not match canonical domain data")
    return PredictionArtifact(
        prediction=prediction,
        event_count=event_count,
        prefix_sha256=prefix_sha256,
        artifact_sha256=sha256_hex(content),
        content=content,
    )


def _build_header(prediction: NativePrediction) -> dict[str, JsonValue]:
    descriptor = prediction.descriptor
    descriptor_payload = dict(descriptor.payload)
    if any(not isinstance(key, str) for key in descriptor_payload):
        raise PredictionArtifactError("backend descriptor keys must be strings")
    if any(not isinstance(value, str) for value in descriptor_payload.values()):
        raise PredictionArtifactError("backend descriptor values must be strings")
    calculated_descriptor_sha256 = sha256_hex(canonical_json_bytes(descriptor_payload))
    require_sha256(descriptor.sha256, "backend_descriptor_sha256")
    if descriptor.sha256 != calculated_descriptor_sha256:
        raise PredictionArtifactError("backend_descriptor_sha256 mismatch")

    required_descriptor_ids = {
        "prediction_schema",
        "model_id",
        "architecture_id",
        "native_output_space_id",
        "native_metadata_schema_id",
    }
    if not required_descriptor_ids.issubset(descriptor_payload):
        raise PredictionArtifactError("backend descriptor is missing prediction identity fields")
    if descriptor_payload["prediction_schema"] != PREDICTION_SCHEMA:
        raise PredictionArtifactError(f"prediction_schema must be {PREDICTION_SCHEMA}")
    for field in required_descriptor_ids - {"prediction_schema"}:
        _require_nonempty_string(descriptor_payload[field], field)
    metadata_schema = descriptor_payload["native_metadata_schema_id"]
    if metadata_schema not in NATIVE_METADATA_SCHEMAS:
        raise PredictionArtifactError("unknown native_metadata_schema_id")

    audio = prediction.audio
    _require_nonempty_string(audio.source_audio_id, "source_audio_id")
    _require_nonempty_string(audio.input_view_id, "input_view_id")
    require_sha256(audio.source_audio_sha256, "source_audio_sha256")
    require_sha256(audio.input_audio_sha256, "input_audio_sha256")
    for field in (
        "byte_length",
        "sample_rate",
        "channel_count",
        "sample_width_bytes",
        "audio_frame_count",
    ):
        _require_positive_int(getattr(audio, field), field)

    _validate_header_nullability(prediction, metadata_schema)
    for field in (
        "backend_lock_sha256",
        "runtime_lock_sha256",
        "parameter_lock_sha256",
        "model_artifact_set_sha256",
    ):
        value = getattr(prediction, field)
        if value is not None:
            require_sha256(value, field)

    return {
        "architecture_id": descriptor_payload["architecture_id"],
        "artifact_role": "native",
        "audio_frame_count": audio.audio_frame_count,
        "backend_descriptor": descriptor_payload,
        "backend_descriptor_sha256": descriptor.sha256,
        "backend_lock_sha256": prediction.backend_lock_sha256,
        "byte_length": audio.byte_length,
        "channel_count": audio.channel_count,
        "input_audio_sha256": audio.input_audio_sha256,
        "input_view_id": audio.input_view_id,
        "model_artifact_set_sha256": prediction.model_artifact_set_sha256,
        "model_id": descriptor_payload["model_id"],
        "native_metadata_schema_id": metadata_schema,
        "native_output_space_id": descriptor_payload["native_output_space_id"],
        "parameter_lock_sha256": prediction.parameter_lock_sha256,
        "record_type": "header",
        "runtime_lock_sha256": prediction.runtime_lock_sha256,
        "sample_rate": audio.sample_rate,
        "sample_width_bytes": audio.sample_width_bytes,
        "schema": PREDICTION_SCHEMA,
        "source_audio_id": audio.source_audio_id,
        "source_audio_sha256": audio.source_audio_sha256,
        "training_data_map_id": prediction.training_data_map_id,
        "upstream_source_commit": prediction.upstream_source_commit,
    }


def _validate_header_nullability(
    prediction: NativePrediction,
    metadata_schema: str,
) -> None:
    if metadata_schema == OAF_METADATA_SCHEMA:
        required = (
            prediction.backend_lock_sha256,
            prediction.runtime_lock_sha256,
            prediction.model_artifact_set_sha256,
            prediction.upstream_source_commit,
            prediction.training_data_map_id,
        )
        if any(value is None for value in required) or prediction.parameter_lock_sha256 is not None:
            raise PredictionArtifactError("oaf_header_nullability")
        for field in ("upstream_source_commit", "training_data_map_id"):
            _require_nonempty_string(cast(str, getattr(prediction, field)), field)
        return

    forbidden = (
        prediction.backend_lock_sha256,
        prediction.runtime_lock_sha256,
        prediction.model_artifact_set_sha256,
        prediction.upstream_source_commit,
        prediction.training_data_map_id,
    )
    if any(value is not None for value in forbidden) or prediction.parameter_lock_sha256 is None:
        raise PredictionArtifactError("heuristic_header_nullability")

    descriptor_parameter_sha256 = prediction.descriptor.payload.get("parameter_lock_sha256")
    if descriptor_parameter_sha256 != prediction.parameter_lock_sha256:
        raise PredictionArtifactError("parameter_lock_sha256 must match backend descriptor")


def _normalize_event(event: NativeEvent, metadata_schema: str) -> _NormalizedEvent:
    if type(event.time_sec) is not float:
        raise PredictionArtifactError("time_sec must be a binary float")
    if event.time_sec < 0:
        raise PredictionArtifactError("time_sec must be nonnegative")
    try:
        time_sec = quantize_six(event.time_sec)
    except StrictJsonError as error:
        raise PredictionArtifactError(f"time_sec: {error}") from None

    _require_nonempty_string(event.native_class_id, "native_class_id")
    _require_optional_int_range(event.model_output_bin, "model_output_bin", 0, 87)
    _require_optional_int_range(event.native_midi_note, "native_midi_note", 0, 127)
    _require_optional_int_range(event.velocity_midi, "velocity_midi", 0, 127)
    confidence = _normalize_confidence(event.confidence)
    metadata = _validate_metadata(event.native_metadata, metadata_schema)

    if metadata_schema == OAF_METADATA_SCHEMA:
        if (
            event.model_output_bin is None
            or event.native_midi_note is None
            or confidence is None
            or event.velocity_midi is None
        ):
            raise PredictionArtifactError("oaf_event_nullability")
        expected_midi_note = event.model_output_bin + 21
        if (
            event.native_midi_note != expected_midi_note
            or event.native_class_id != f"midi_{expected_midi_note}"
        ):
            raise PredictionArtifactError("oaf_native_identity")

    return _NormalizedEvent(
        time_sec=time_sec,
        native_class_id=event.native_class_id,
        model_output_bin=event.model_output_bin,
        native_midi_note=event.native_midi_note,
        native_metadata=metadata,
        confidence=confidence,
        velocity_midi=event.velocity_midi,
    )


def _normalize_confidence(value: float | None) -> Decimal | None:
    if value is None:
        return None
    if type(value) is not float:
        raise PredictionArtifactError("confidence must be a binary float or null")
    try:
        confidence = quantize_six(value)
    except StrictJsonError as error:
        raise PredictionArtifactError(f"confidence: {error}") from None
    if value < 0 or value > 1:
        raise PredictionArtifactError("confidence must be in 0..1")
    return confidence


def _validate_metadata(
    metadata: Mapping[str, str | None],
    metadata_schema: str,
) -> dict[str, str | None]:
    if not isinstance(metadata, Mapping):
        raise PredictionArtifactError("native_metadata must be an object")
    policy = NATIVE_METADATA_SCHEMAS[metadata_schema]
    if set(metadata) != set(policy):
        raise PredictionArtifactError("native_metadata must contain the exact schema keys")
    result: dict[str, str | None] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or value not in policy[key]:
            raise PredictionArtifactError("native_metadata contains an invalid value")
        result[key] = value
    return result


def _event_record(event: _NormalizedEvent, event_index: int) -> dict[str, JsonValue]:
    return {
        "canonical_class": None,
        "confidence": event.confidence,
        "event_index": event_index,
        "mapping_status": "not_applied",
        "model_output_bin": event.model_output_bin,
        "native_class_id": event.native_class_id,
        "native_metadata": event.native_metadata,
        "native_midi_note": event.native_midi_note,
        "prediction_map_version": None,
        "record_type": "event",
        "time_sec": event.time_sec,
        "velocity_midi": event.velocity_midi,
    }


def _prediction_from_records(
    header: dict[str, JsonValue],
    events: list[dict[str, JsonValue]],
) -> NativePrediction:
    _require_native_header_fields(header)
    descriptor_payload_value = header["backend_descriptor"]
    if not isinstance(descriptor_payload_value, dict):
        raise PredictionArtifactError("backend_descriptor must be an object")
    if any(not isinstance(value, str) for value in descriptor_payload_value.values()):
        raise PredictionArtifactError("backend_descriptor values must be strings")
    descriptor_sha256 = header["backend_descriptor_sha256"]
    if not isinstance(descriptor_sha256, str):
        raise PredictionArtifactError("backend_descriptor_sha256 must be a string")

    audio = CanonicalAudio(
        path=Path(),
        source_audio_id=cast(str, header["source_audio_id"]),
        source_audio_sha256=cast(str, header["source_audio_sha256"]),
        input_view_id=cast(str, header["input_view_id"]),
        input_audio_sha256=cast(str, header["input_audio_sha256"]),
        byte_length=cast(int, header["byte_length"]),
        sample_rate=cast(int, header["sample_rate"]),
        channel_count=cast(int, header["channel_count"]),
        sample_width_bytes=cast(int, header["sample_width_bytes"]),
        audio_frame_count=cast(int, header["audio_frame_count"]),
    )
    descriptor = BackendDescriptor(
        payload=cast(dict[str, str], descriptor_payload_value),
        sha256=descriptor_sha256,
    )
    native_events = tuple(_native_event_from_record(event) for event in events)
    return NativePrediction(
        audio=audio,
        descriptor=descriptor,
        events=native_events,
        backend_lock_sha256=cast(str | None, header["backend_lock_sha256"]),
        runtime_lock_sha256=cast(str | None, header["runtime_lock_sha256"]),
        parameter_lock_sha256=cast(str | None, header["parameter_lock_sha256"]),
        model_artifact_set_sha256=cast(
            str | None,
            header["model_artifact_set_sha256"],
        ),
        upstream_source_commit=cast(str | None, header["upstream_source_commit"]),
        training_data_map_id=cast(str | None, header["training_data_map_id"]),
    )


def _require_native_header_fields(header: dict[str, JsonValue]) -> None:
    if header["schema"] != PREDICTION_SCHEMA:
        raise PredictionArtifactError(f"schema must be {PREDICTION_SCHEMA}")
    if header["artifact_role"] != "native":
        raise PredictionArtifactError("artifact_role must be native")
    string_fields = (
        "architecture_id",
        "backend_descriptor_sha256",
        "input_audio_sha256",
        "input_view_id",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "source_audio_id",
        "source_audio_sha256",
    )
    for field in string_fields:
        if not isinstance(header[field], str):
            raise PredictionArtifactError(f"{field} must be a string")
    optional_string_fields = (
        "backend_lock_sha256",
        "model_artifact_set_sha256",
        "parameter_lock_sha256",
        "runtime_lock_sha256",
        "training_data_map_id",
        "upstream_source_commit",
    )
    for field in optional_string_fields:
        if header[field] is not None and not isinstance(header[field], str):
            raise PredictionArtifactError(f"{field} must be a string or null")
    for field in (
        "audio_frame_count",
        "byte_length",
        "channel_count",
        "sample_rate",
        "sample_width_bytes",
    ):
        if type(header[field]) is not int:
            raise PredictionArtifactError(f"{field} must be an integer")

    descriptor = header["backend_descriptor"]
    if isinstance(descriptor, dict):
        for field in (
            "architecture_id",
            "model_id",
            "native_metadata_schema_id",
            "native_output_space_id",
        ):
            if descriptor.get(field) != header[field]:
                raise PredictionArtifactError(f"{field} must match backend_descriptor")


def _native_event_from_record(event: dict[str, JsonValue]) -> NativeEvent:
    _require_native_mapping_fields(event)
    time_sec = _json_number_to_float(event["time_sec"], "time_sec")
    confidence_value = event["confidence"]
    confidence = (
        None if confidence_value is None else _json_number_to_float(confidence_value, "confidence")
    )
    metadata = event["native_metadata"]
    if not isinstance(metadata, dict):
        raise PredictionArtifactError("native_metadata must be an object")
    if any(
        not isinstance(key, str) or (value is not None and not isinstance(value, str))
        for key, value in metadata.items()
    ):
        raise PredictionArtifactError("native_metadata values must be strings or null")
    native_class_id = event["native_class_id"]
    if not isinstance(native_class_id, str):
        raise PredictionArtifactError("native_class_id must be a string")
    return NativeEvent(
        time_sec=time_sec,
        native_class_id=native_class_id,
        model_output_bin=_json_optional_int(event["model_output_bin"], "model_output_bin"),
        native_midi_note=_json_optional_int(event["native_midi_note"], "native_midi_note"),
        native_metadata=cast(dict[str, str | None], metadata),
        confidence=confidence,
        velocity_midi=_json_optional_int(event["velocity_midi"], "velocity_midi"),
    )


def _require_native_mapping_fields(event: dict[str, JsonValue]) -> None:
    if event["mapping_status"] != "not_applied":
        raise PredictionArtifactError("mapping_status must be not_applied")
    if event["prediction_map_version"] is not None:
        raise PredictionArtifactError("prediction_map_version must be null")
    if event["canonical_class"] is not None:
        raise PredictionArtifactError("canonical_class must be null")


def _json_number_to_float(value: JsonValue, field: str) -> float:
    if type(value) is int:
        return float(cast(int, value))
    if isinstance(value, Decimal):
        return float(value)
    raise PredictionArtifactError(f"{field} must be a JSON number")


def _json_optional_int(value: JsonValue, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise PredictionArtifactError(f"{field} must be an integer or null")
    return cast(int, value)


def _require_exact_keys(
    record: dict[str, JsonValue],
    keys: frozenset[str],
    record_type: str,
) -> None:
    if set(record) != set(keys):
        raise PredictionArtifactError(f"{record_type} must contain the exact key set")


def _require_nonempty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise PredictionArtifactError(f"{field} must be a nonempty string")


def _require_positive_int(value: object, field: str) -> None:
    if type(value) is not int or value <= 0:
        raise PredictionArtifactError(f"{field} must be a positive integer")


def _require_optional_int_range(
    value: object,
    field: str,
    minimum: int,
    maximum: int,
) -> None:
    if value is None:
        return
    if type(value) is not int or not minimum <= value <= maximum:
        raise PredictionArtifactError(f"{field} must be an integer in {minimum}..{maximum}")
