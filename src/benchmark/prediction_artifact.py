"""Canonical JSONL persistence for mapped drum predictions."""

from __future__ import annotations

# The fixed JSONL union intentionally validates each record branch explicitly.
# pylint: disable=too-many-branches,too-many-locals,too-many-statements
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from src.benchmark.artifact_io import (
    PublishedArtifact,
    publish_immutable_file,
    read_regular_file_no_follow,
)
from src.benchmark.backend_identity import (
    IDM_BACKEND_ID,
    IDM_NATIVE_METADATA_SCHEMA_ID,
    MUSCRIPTOR_BACKEND_ID,
    OAF_BACKEND_ID,
    BackendDescriptor,
    JsonValue,
    StrictJsonError,
    canonical_json_bytes,
    normalize_known_backend_descriptor,
    quantize_six,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backends import CanonicalAudio, NativeEvent
from src.benchmark.idm_model import IDM_TRAIN_CLASSES

PREDICTION_SCHEMA = "crux.drum-prediction-events/v2"
OAF_METADATA_SCHEMA = "magenta-oaf-native-metadata-v1"
MUSCRIPTOR_METADATA_SCHEMA = "muscriptor-note-start-metadata-v1"
OAF_GROUP_IDS = frozenset(
    {"kick", "snare", "toms", "hihat", "ride", "ride_bell", "crash", "sticks"}
)
NATIVE_METADATA_SCHEMAS = {
    OAF_METADATA_SCHEMA: {"upstream_8hit_group_id": OAF_GROUP_IDS | {None}},
    MUSCRIPTOR_METADATA_SCHEMA: {"instrument_group": {"drums"}},
    IDM_NATIVE_METADATA_SCHEMA_ID: {},
}
_IDM_FRAME_INDEX_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_IDM_NATIVE_VELOCITY_QUANTUM = Decimal("0.000001")
_IDM_NATIVE_VELOCITY_MAX = Decimal("2.000001")
HEADER_KEYS = frozenset(
    {
        "architecture_id",
        "artifact_role",
        "audio_frame_count",
        "backend_descriptor",
        "backend_descriptor_sha256",
        "byte_length",
        "channel_count",
        "input_audio_sha256",
        "input_view_id",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "record_type",
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
        "common_class",
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


def _require_hash(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise StrictJsonError(f"{field} must be lowercase SHA-256")
    return require_sha256(value, field)


def prediction_path(
    output_dir: Path,
    *,
    simfile_id: int,
    source_audio_sha256: str,
    backend_descriptor_sha256: str,
    inference_config_sha256: str,
) -> Path:
    """Return the source-keyed immutable prediction-v2 location."""
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if isinstance(simfile_id, bool) or not isinstance(simfile_id, int) or simfile_id <= 0:
        raise ValueError("simfile_id must be a positive integer")
    source_sha = _require_hash(source_audio_sha256, "source_audio_sha256")
    descriptor_sha = _require_hash(backend_descriptor_sha256, "backend_descriptor_sha256")
    config_sha = _require_hash(inference_config_sha256, "inference_config_sha256")
    return (
        output_dir
        / "predictions"
        / str(simfile_id)
        / source_sha
        / descriptor_sha
        / f"{config_sha}.jsonl"
    )


@dataclass(frozen=True)
class MappedPredictionEvent:
    native: NativeEvent
    canonical_class: str | None
    common_class: str | None
    mapping_status: Literal["mapped", "unmapped"]
    prediction_map_version: str


@dataclass(frozen=True)
class MappedPrediction:
    audio: CanonicalAudio
    descriptor: BackendDescriptor
    events: tuple[MappedPredictionEvent, ...]


@dataclass(frozen=True)
class PredictionArtifact:
    prediction: MappedPrediction
    event_count: int
    prefix_sha256: str
    artifact_sha256: str
    content: bytes


@dataclass(frozen=True)
class _NormalizedEvent:
    mapped: MappedPredictionEvent
    time_sec: Decimal
    confidence: Decimal | None
    native_metadata: dict[str, str | None]

    @property
    def sort_key(self) -> tuple[Decimal, str, int, int, int, Decimal]:
        native = self.mapped.native
        return (
            self.time_sec,
            native.native_class_id,
            native.model_output_bin if native.model_output_bin is not None else -1,
            native.native_midi_note if native.native_midi_note is not None else -1,
            native.velocity_midi if native.velocity_midi is not None else -1,
            self.confidence if self.confidence is not None else Decimal(-1),
        )


def render_prediction_artifact(prediction: MappedPrediction) -> bytes:
    if not isinstance(prediction, MappedPrediction):
        raise PredictionArtifactError("prediction must be MappedPrediction")
    try:
        header = _build_header(prediction)
        metadata_schema = cast(str, header["native_metadata_schema_id"])
        backend_id = cast(str, prediction.descriptor.payload["backend_id"])
        events = sorted(
            (_normalize_event(event, metadata_schema, backend_id) for event in prediction.events),
            key=lambda event: event.sort_key,
        )
        for first, second in zip(events, events[1:], strict=False):
            if first.sort_key == second.sort_key:
                raise PredictionArtifactError("duplicate_native_event")
        prefix = canonical_json_bytes(header, trailing_newline=True)
        for index, event in enumerate(events):
            prefix += canonical_json_bytes(_event_record(event, index), trailing_newline=True)
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


def validate_schema_golden(schema: str, content: bytes) -> None:
    if schema != PREDICTION_SCHEMA:
        raise ValueError("unsupported schema golden")
    read_prediction_artifact(content)


def publish_prediction_artifact(
    path: Path,
    prediction: MappedPrediction,
) -> PublishedArtifact:
    content = render_prediction_artifact(prediction)
    published = publish_immutable_file(path, content)
    persisted = read_regular_file_no_follow(path)
    artifact = read_prediction_artifact(persisted)
    if artifact.content != content or artifact.artifact_sha256 != published.sha256:
        raise PredictionArtifactError("published prediction bytes changed")
    return published


def prediction_artifact_matches_audio(
    artifact: PredictionArtifact,
    *,
    source_audio_id: str,
    source_audio_sha256: str,
    audio: CanonicalAudio,
    descriptor: BackendDescriptor,
    prediction_map_version: str,
) -> bool:
    """Check persisted prediction identity against one canonical audio input."""
    if not isinstance(artifact, PredictionArtifact):
        return False
    if not isinstance(audio, CanonicalAudio) or not isinstance(descriptor, BackendDescriptor):
        return False
    if not all(
        isinstance(value, str) and value
        for value in (source_audio_id, source_audio_sha256, prediction_map_version)
    ):
        return False
    prediction = artifact.prediction
    return (
        prediction.descriptor.sha256 == descriptor.sha256
        and dict(prediction.descriptor.payload) == dict(descriptor.payload)
        and prediction.audio.source_audio_id == source_audio_id
        and prediction.audio.source_audio_sha256 == source_audio_sha256
        and prediction.audio.input_view_id == audio.input_view_id
        and prediction.audio.input_audio_sha256 == audio.input_audio_sha256
        and prediction.audio.source_audio_id == audio.source_audio_id
        and prediction.audio.source_audio_sha256 == audio.source_audio_sha256
        and all(
            event.prediction_map_version == prediction_map_version for event in prediction.events
        )
    )


def prediction_artifact_matches_run_row(
    artifact: PredictionArtifact,
    row: Mapping[str, object],
    *,
    expected_input_view_id: str,
) -> bool:
    """Bind raw persisted prediction bytes to persisted row evidence and view policy."""
    if not isinstance(artifact, PredictionArtifact) or not isinstance(row, Mapping):
        return False
    if not isinstance(expected_input_view_id, str) or not expected_input_view_id:
        return False
    values = (
        row.get("prediction_artifact_sha256"),
        row.get("source_audio_id"),
        row.get("source_audio_sha256"),
        row.get("input_audio_sha256"),
    )
    if not all(isinstance(value, str) and value for value in values):
        return False
    row_input_view_id = row.get("input_view_id")
    if row_input_view_id is not None and row_input_view_id != expected_input_view_id:
        return False
    prediction = artifact.prediction
    return (
        artifact.artifact_sha256 == values[0]
        and prediction.audio.source_audio_id == values[1]
        and prediction.audio.source_audio_sha256 == values[2]
        and prediction.audio.input_view_id == expected_input_view_id
        and prediction.audio.input_audio_sha256 == values[3]
    )


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
    header, terminal, event_records = records[0], records[-1], records[1:-1]
    _require_exact_keys(header, HEADER_KEYS, "header")
    _require_exact_keys(terminal, TERMINAL_KEYS, "terminal")
    for event in event_records:
        _require_exact_keys(event, EVENT_KEYS, "event")
    if header["record_type"] != "header" or terminal["record_type"] != "terminal":
        raise PredictionArtifactError("header and terminal record types are invalid")
    if any(event["record_type"] != "event" for event in event_records):
        raise PredictionArtifactError("records between header and terminal must be events")

    prediction = _prediction_from_records(header, event_records)
    for expected_index, event in enumerate(event_records):
        if type(event["event_index"]) is not int or event["event_index"] != expected_index:
            raise PredictionArtifactError("event_index must be contiguous from zero")
        _validate_mapping_fields(event)
    event_count = terminal["event_count"]
    if type(event_count) is not int or event_count != len(event_records):
        raise PredictionArtifactError("terminal event_count mismatch")
    prefix_sha256 = terminal["prefix_sha256"]
    if not isinstance(prefix_sha256, str):
        raise PredictionArtifactError("prefix_sha256 must be a string")
    require_sha256(prefix_sha256, "prefix_sha256")
    if prefix_sha256 != sha256_hex(b"".join(physical_lines[:-1])):
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


def _build_header(prediction: MappedPrediction) -> dict[str, JsonValue]:
    descriptor_payload = dict(prediction.descriptor.payload)
    if set(descriptor_payload) != {
        "architecture_id",
        "backend_id",
        "descriptor_schema",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "prediction_schema",
        "training_data_map_id",
        "upstream_source_commit",
    }:
        raise PredictionArtifactError("backend descriptor key set is invalid")
    if any(not isinstance(value, str) or not value for value in descriptor_payload.values()):
        raise PredictionArtifactError("backend descriptor values must be nonempty strings")
    calculated_sha256 = sha256_hex(canonical_json_bytes(descriptor_payload))
    require_sha256(prediction.descriptor.sha256, "backend_descriptor_sha256")
    if prediction.descriptor.sha256 != calculated_sha256:
        raise PredictionArtifactError("backend_descriptor_sha256 mismatch")
    try:
        normalize_known_backend_descriptor(descriptor_payload)
    except StrictJsonError as error:
        raise PredictionArtifactError(str(error)) from None
    if descriptor_payload["prediction_schema"] != PREDICTION_SCHEMA:
        raise PredictionArtifactError(f"prediction_schema must be {PREDICTION_SCHEMA}")
    metadata_schema = descriptor_payload["native_metadata_schema_id"]
    if metadata_schema not in NATIVE_METADATA_SCHEMAS:
        raise PredictionArtifactError("unknown native_metadata_schema_id")

    audio = prediction.audio
    for field in ("source_audio_id", "input_view_id"):
        _require_nonempty_string(getattr(audio, field), field)
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
    _validate_canonical_audio(audio)
    return {
        "architecture_id": descriptor_payload["architecture_id"],
        "artifact_role": "native",
        "audio_frame_count": audio.audio_frame_count,
        "backend_descriptor": descriptor_payload,
        "backend_descriptor_sha256": prediction.descriptor.sha256,
        "byte_length": audio.byte_length,
        "channel_count": audio.channel_count,
        "input_audio_sha256": audio.input_audio_sha256,
        "input_view_id": audio.input_view_id,
        "model_id": descriptor_payload["model_id"],
        "native_metadata_schema_id": metadata_schema,
        "native_output_space_id": descriptor_payload["native_output_space_id"],
        "record_type": "header",
        "sample_rate": audio.sample_rate,
        "sample_width_bytes": audio.sample_width_bytes,
        "schema": PREDICTION_SCHEMA,
        "source_audio_id": audio.source_audio_id,
        "source_audio_sha256": audio.source_audio_sha256,
        "training_data_map_id": descriptor_payload["training_data_map_id"],
        "upstream_source_commit": descriptor_payload["upstream_source_commit"],
    }


def _normalize_event(
    event: MappedPredictionEvent,
    metadata_schema: str,
    backend_id: str,
) -> _NormalizedEvent:
    native = event.native
    if type(native.time_sec) is not float or native.time_sec < 0:
        raise PredictionArtifactError("time_sec must be a nonnegative binary float")
    try:
        time_sec = quantize_six(native.time_sec)
    except StrictJsonError as error:
        raise PredictionArtifactError(f"time_sec: {error}") from None
    _require_nonempty_string(native.native_class_id, "native_class_id")
    _require_optional_int_range(native.model_output_bin, "model_output_bin", 0, 87)
    _require_optional_int_range(native.native_midi_note, "native_midi_note", 0, 127)
    _require_optional_int_range(native.velocity_midi, "velocity_midi", 0, 127)
    confidence = _normalize_confidence(native.confidence)
    metadata = _validate_metadata(native.native_metadata, metadata_schema)
    if backend_id == OAF_BACKEND_ID:
        if (
            native.model_output_bin is None
            or native.native_midi_note is None
            or confidence is None
            or native.velocity_midi is None
        ):
            raise PredictionArtifactError("oaf_event_nullability")
        if native.native_midi_note != native.model_output_bin + 21:
            raise PredictionArtifactError("oaf_native_identity")
        if native.native_class_id != f"midi_{native.native_midi_note}":
            raise PredictionArtifactError("oaf_native_identity")
    elif backend_id == MUSCRIPTOR_BACKEND_ID:
        if (
            native.model_output_bin is not None
            or native.native_midi_note is None
            or confidence is not None
            or native.velocity_midi is not None
            or native.native_metadata != {"instrument_group": "drums"}
            or native.native_class_id != f"drums:midi_{native.native_midi_note}"
        ):
            raise PredictionArtifactError("muscriptor_native_identity")
    elif backend_id == IDM_BACKEND_ID:
        if native.model_output_bin is None or native.model_output_bin >= len(IDM_TRAIN_CLASSES):
            raise PredictionArtifactError("idm_model_output_bin")
        if native.native_class_id != IDM_TRAIN_CLASSES[native.model_output_bin]:
            raise PredictionArtifactError("idm_native_identity")
        if native.native_midi_note is not None:
            raise PredictionArtifactError("idm_native_identity")
        if confidence is None or native.velocity_midi is None:
            raise PredictionArtifactError("idm_event_nullability")
    else:  # pragma: no cover - descriptor normalization rejects unknown families.
        raise PredictionArtifactError("unknown_backend")
    _validate_mapping_values(event)
    return _NormalizedEvent(event, time_sec, confidence, metadata)


def _normalize_confidence(value: float | None) -> Decimal | None:
    if value is None:
        return None
    if type(value) is not float:
        raise PredictionArtifactError("confidence must be a binary float or null")
    if value < 0 or value > 1:
        raise PredictionArtifactError("confidence must be in 0..1")
    try:
        return quantize_six(value)
    except StrictJsonError as error:
        raise PredictionArtifactError(f"confidence: {error}") from None


def _validate_metadata(
    metadata: Mapping[str, str | None], metadata_schema: str
) -> dict[str, str | None]:
    if not isinstance(metadata, Mapping):
        raise PredictionArtifactError("native_metadata must be an object")
    if metadata_schema == IDM_NATIVE_METADATA_SCHEMA_ID:
        return _validate_idm_metadata(metadata)
    policy = NATIVE_METADATA_SCHEMAS[metadata_schema]
    if set(metadata) != set(policy):
        raise PredictionArtifactError("native_metadata must contain the exact schema keys")
    result: dict[str, str | None] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or value not in policy[key]:
            raise PredictionArtifactError("native_metadata contains an invalid value")
        result[key] = value
    return result


def _validate_idm_metadata(metadata: Mapping[str, str | None]) -> dict[str, str | None]:
    if set(metadata) != {"frame_index", "native_velocity"}:
        raise PredictionArtifactError("idm_native_metadata keys are invalid")
    frame_index = metadata["frame_index"]
    if not isinstance(frame_index, str) or _IDM_FRAME_INDEX_RE.fullmatch(frame_index) is None:
        raise PredictionArtifactError("idm_native_metadata frame_index is invalid")

    native_velocity = metadata["native_velocity"]
    if not isinstance(native_velocity, str):
        raise PredictionArtifactError("idm_native_metadata native_velocity is invalid")
    try:
        value = Decimal(native_velocity)
        quantized = value.quantize(_IDM_NATIVE_VELOCITY_QUANTUM)
    except InvalidOperation:
        raise PredictionArtifactError("idm_native_metadata native_velocity is invalid") from None
    if not value.is_finite() or value < 0 or value > _IDM_NATIVE_VELOCITY_MAX:
        raise PredictionArtifactError("idm_native_metadata native_velocity is invalid")
    canonical = format(abs(quantized), "f").rstrip("0").rstrip(".") or "0"
    if value != quantized or native_velocity != canonical:
        raise PredictionArtifactError("idm_native_metadata native_velocity is invalid")
    return {"frame_index": frame_index, "native_velocity": native_velocity}


def _event_record(event: _NormalizedEvent, event_index: int) -> dict[str, JsonValue]:
    mapped = event.mapped
    native = mapped.native
    return {
        "canonical_class": mapped.canonical_class,
        "common_class": mapped.common_class,
        "confidence": event.confidence,
        "event_index": event_index,
        "mapping_status": mapped.mapping_status,
        "model_output_bin": native.model_output_bin,
        "native_class_id": native.native_class_id,
        "native_metadata": event.native_metadata,
        "native_midi_note": native.native_midi_note,
        "prediction_map_version": mapped.prediction_map_version,
        "record_type": "event",
        "time_sec": event.time_sec,
        "velocity_midi": native.velocity_midi,
    }


def _prediction_from_records(
    header: dict[str, JsonValue], events: list[dict[str, JsonValue]]
) -> MappedPrediction:
    _validate_header(header)
    descriptor_payload = cast(dict[str, str], header["backend_descriptor"])
    descriptor_sha256 = cast(str, header["backend_descriptor_sha256"])
    descriptor = BackendDescriptor(
        payload=MappingProxyType(dict(descriptor_payload)),
        sha256=descriptor_sha256,
    )
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
    return MappedPrediction(
        audio=audio,
        descriptor=descriptor,
        events=tuple(_mapped_event_from_record(event) for event in events),
    )


def _validate_header(header: dict[str, JsonValue]) -> None:
    if header["schema"] != PREDICTION_SCHEMA or header["artifact_role"] != "native":
        raise PredictionArtifactError("prediction header identity is invalid")
    for field in (
        "architecture_id",
        "backend_descriptor_sha256",
        "input_audio_sha256",
        "input_view_id",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "source_audio_id",
        "source_audio_sha256",
        "training_data_map_id",
        "upstream_source_commit",
    ):
        _require_nonempty_string(header[field], field)
    for field in (
        "audio_frame_count",
        "byte_length",
        "channel_count",
        "sample_rate",
        "sample_width_bytes",
    ):
        _require_positive_int(header[field], field)
    descriptor = header["backend_descriptor"]
    if not isinstance(descriptor, dict):
        raise PredictionArtifactError("backend_descriptor must be an object")
    try:
        normalize_known_backend_descriptor(descriptor)
    except StrictJsonError as error:
        raise PredictionArtifactError(str(error)) from None
    for field in (
        "architecture_id",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "training_data_map_id",
        "upstream_source_commit",
    ):
        if descriptor.get(field) != header[field]:
            raise PredictionArtifactError(f"{field} must match backend_descriptor")
    _validate_canonical_audio_header(header)


def _mapped_event_from_record(event: dict[str, JsonValue]) -> MappedPredictionEvent:
    _validate_mapping_fields(event)
    metadata = event["native_metadata"]
    if not isinstance(metadata, dict):
        raise PredictionArtifactError("native_metadata must be an object")
    native_class_id = event["native_class_id"]
    if not isinstance(native_class_id, str):
        raise PredictionArtifactError("native_class_id must be a string")
    return MappedPredictionEvent(
        native=NativeEvent(
            time_sec=_json_number_to_float(event["time_sec"], "time_sec"),
            native_class_id=native_class_id,
            model_output_bin=_json_optional_int(event["model_output_bin"], "model_output_bin"),
            native_midi_note=_json_optional_int(event["native_midi_note"], "native_midi_note"),
            native_metadata=MappingProxyType(
                {cast(str, key): cast(str | None, value) for key, value in metadata.items()}
            ),
            confidence=(
                None
                if event["confidence"] is None
                else _json_number_to_float(event["confidence"], "confidence")
            ),
            velocity_midi=_json_optional_int(event["velocity_midi"], "velocity_midi"),
        ),
        canonical_class=cast(str | None, event["canonical_class"]),
        common_class=cast(str | None, event["common_class"]),
        mapping_status=cast(Literal["mapped", "unmapped"], event["mapping_status"]),
        prediction_map_version=cast(str, event["prediction_map_version"]),
    )


def _validate_mapping_values(event: MappedPredictionEvent) -> None:
    if not isinstance(event.prediction_map_version, str) or not event.prediction_map_version:
        raise PredictionArtifactError("prediction_map_version must be non-null")
    if event.mapping_status not in {"mapped", "unmapped"}:
        raise PredictionArtifactError("mapping_status is invalid")
    if event.mapping_status == "mapped":
        if not isinstance(event.common_class, str) or not event.common_class:
            raise PredictionArtifactError("mapped event common_class must be non-null")
        if event.canonical_class is not None and not isinstance(event.canonical_class, str):
            raise PredictionArtifactError("mapped event canonical_class must be a string or null")
    elif event.common_class is not None or event.canonical_class is not None:
        raise PredictionArtifactError("unmapped event classes must be null")


def _validate_mapping_fields(event: dict[str, JsonValue]) -> None:
    status = event["mapping_status"]
    version = event["prediction_map_version"]
    canonical = event["canonical_class"]
    common = event["common_class"]
    if status not in {"mapped", "unmapped"}:
        raise PredictionArtifactError("mapping_status is invalid")
    if not isinstance(version, str) or not version:
        raise PredictionArtifactError("prediction_map_version must be non-null")
    if status == "mapped":
        if not isinstance(common, str) or not common:
            raise PredictionArtifactError("mapped event common_class must be non-null")
        if canonical is not None and not isinstance(canonical, str):
            raise PredictionArtifactError("mapped event canonical_class must be a string or null")
    elif canonical is not None or common is not None:
        raise PredictionArtifactError("unmapped event classes must be null")


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
    record: dict[str, JsonValue], keys: frozenset[str], record_type: str
) -> None:
    if set(record) != set(keys):
        raise PredictionArtifactError(f"{record_type} must contain the exact key set")


def _require_nonempty_string(value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise PredictionArtifactError(f"{field} must be a nonempty string")


def _require_positive_int(value: object, field: str) -> None:
    if type(value) is not int or value <= 0:
        raise PredictionArtifactError(f"{field} must be a positive integer")


def _require_optional_int_range(value: object, field: str, minimum: int, maximum: int) -> None:
    if value is None:
        return
    if type(value) is not int or not minimum <= value <= maximum:
        raise PredictionArtifactError(f"{field} must be an integer in {minimum}..{maximum}")


def _validate_canonical_audio(audio: CanonicalAudio) -> None:
    if audio.sample_rate != 44100 or audio.channel_count != 1 or audio.sample_width_bytes != 2:
        raise PredictionArtifactError("audio format is not canonical")
    if audio.byte_length != 44 + audio.audio_frame_count * 2:
        raise PredictionArtifactError("byte_length must match canonical WAV frame data")


def _validate_canonical_audio_header(header: dict[str, JsonValue]) -> None:
    if header["sample_rate"] != 44100 or header["channel_count"] != 1:
        raise PredictionArtifactError("audio format is not canonical")
    if header["sample_width_bytes"] != 2:
        raise PredictionArtifactError("sample_width_bytes must be 2")
    frame_count = header["audio_frame_count"]
    if type(frame_count) is not int or frame_count <= 0:
        raise PredictionArtifactError("audio_frame_count must be positive")
    if header["byte_length"] != 44 + frame_count * 2:
        raise PredictionArtifactError("byte_length must match canonical WAV frame data")


__all__ = [
    "MappedPrediction",
    "MappedPredictionEvent",
    "PredictionArtifact",
    "PredictionArtifactError",
    "PREDICTION_SCHEMA",
    "prediction_artifact_matches_audio",
    "prediction_artifact_matches_run_row",
    "prediction_path",
    "publish_prediction_artifact",
    "read_prediction_artifact",
    "render_prediction_artifact",
    "validate_schema_golden",
]
