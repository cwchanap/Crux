from __future__ import annotations

# Exact key-set fixtures intentionally mirror the external prediction schema.
# pylint: disable=duplicate-code
import builtins
import hashlib
import importlib
import json
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from src.benchmark.backend_identity import (
    BackendDescriptor,
    canonical_json_bytes,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backends import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.prediction_artifact import (
    OAF_GROUP_IDS,
    PredictionArtifactError,
    publish_prediction_artifact,
    read_prediction_artifact,
    render_prediction_artifact,
)

SCHEMA = "crux.drum-prediction-events/v1"
OAF_METADATA_SCHEMA = "magenta-oaf-native-metadata-v1"
EMPTY_METADATA_SCHEMA = "crux-empty-native-metadata-v1"
EXPECTED_OAF_GROUP_IDS = frozenset(
    {"kick", "snare", "toms", "hihat", "ride", "ride_bell", "crash", "sticks"}
)
OAF_DESCRIPTOR_PAYLOAD = {
    "architecture_id": "magenta-oaf-model-tpu-drums-v1",
    "backend_id": "test-oaf-envelope-v1",
    "descriptor_schema": "crux.test-oaf-backend-descriptor/v1",
    "model_id": "magenta-egmd-ckpt-569400-v1",
    "native_metadata_schema_id": OAF_METADATA_SCHEMA,
    "native_output_space_id": "magenta-oaf-midi88-a0-v1",
    "prediction_schema": SCHEMA,
}
OAF_DESCRIPTOR = BackendDescriptor(
    payload=OAF_DESCRIPTOR_PAYLOAD,
    sha256="0874c2cd8ad07e976007b6647e0d29b681d7e7f05c83a434d66b4cedfdc7cbbb",
)
HEURISTIC_DESCRIPTOR_PAYLOAD = {
    "adapter_source_manifest_sha256": "f" * 64,
    "architecture_id": "librosa-onset-centroid-zcr-v1",
    "backend_id": "heuristic-onset-v1",
    "descriptor_schema": "crux.heuristic-backend-descriptor/v1",
    "model_id": "crux-heuristic-onset-nonmodel-v1",
    "native_metadata_schema_id": EMPTY_METADATA_SCHEMA,
    "native_output_space_id": "crux-heuristic-midi7-v1",
    "parameter_lock_sha256": "1" * 64,
    "prediction_schema": SCHEMA,
}
HEURISTIC_DESCRIPTOR = BackendDescriptor(
    payload=HEURISTIC_DESCRIPTOR_PAYLOAD,
    sha256="dd737f02e57202bf5c5b7b08511e9657d315b9e8e245ce50a71de61a02ae9296",
)
FROZEN_OAF_DESCRIPTOR_PAYLOAD = {
    "architecture_id": "magenta-oaf-model-tpu-drums-v1",
    "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
    "backend_lock_sha256": "c" * 64,
    "descriptor_schema": "crux.transcription-backend-descriptor/v1",
    "model_artifact_set_sha256": "e" * 64,
    "model_id": "magenta-egmd-ckpt-569400-v1",
    "native_metadata_schema_id": OAF_METADATA_SCHEMA,
    "native_output_space_id": "magenta-oaf-midi88-a0-v1",
    "prediction_schema": SCHEMA,
    "protocol_schema": "crux.transcription-runner/v1",
    "runtime_image_manifest_digest": f"sha256:{'f' * 64}",
    "runtime_lock_sha256": "d" * 64,
    "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
    "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
}
FROZEN_OAF_DESCRIPTOR = BackendDescriptor(
    payload=FROZEN_OAF_DESCRIPTOR_PAYLOAD,
    sha256=hashlib.sha256(
        json.dumps(
            FROZEN_OAF_DESCRIPTOR_PAYLOAD,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest(),
)
HEADER_KEYS = {
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
EVENT_KEYS = {
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
TERMINAL_KEYS = {"event_count", "prefix_sha256", "record_type"}
EXPECTED_HEADER_LINE = (
    b'{"architecture_id":"magenta-oaf-model-tpu-drums-v1","artifact_role":"native",'
    b'"audio_frame_count":44100,"backend_descriptor":{"architecture_id":'
    b'"magenta-oaf-model-tpu-drums-v1","backend_id":'
    b'"test-oaf-envelope-v1","descriptor_schema":'
    b'"crux.test-oaf-backend-descriptor/v1","model_id":"magenta-egmd-ckpt-569400-v1",'
    b'"native_metadata_schema_id":"magenta-oaf-native-metadata-v1",'
    b'"native_output_space_id":"magenta-oaf-midi88-a0-v1","prediction_schema":'
    b'"crux.drum-prediction-events/v1"},"backend_descriptor_sha256":'
    b'"0874c2cd8ad07e976007b6647e0d29b681d7e7f05c83a434d66b4cedfdc7cbbb",'
    b'"backend_lock_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",'
    b'"byte_length":88244,"channel_count":1,"input_audio_sha256":'
    b'"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
    b'"input_view_id":"direct","model_artifact_set_sha256":'
    b'"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",'
    b'"model_id":"magenta-egmd-ckpt-569400-v1","native_metadata_schema_id":'
    b'"magenta-oaf-native-metadata-v1","native_output_space_id":'
    b'"magenta-oaf-midi88-a0-v1","parameter_lock_sha256":null,"record_type":"header",'
    b'"runtime_lock_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",'
    b'"sample_rate":44100,"sample_width_bytes":2,"schema":'
    b'"crux.drum-prediction-events/v1","source_audio_id":"song-1","source_audio_sha256":'
    b'"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    b'"training_data_map_id":"magenta-egmd-data-8hit-94529798-v1",'
    b'"upstream_source_commit":"94529798dfbbb14c27ddfd76f23027dc8e2ce185"}\n'
)
EXPECTED_EVENT_LINE = (
    b'{"canonical_class":null,"confidence":0.75,"event_index":0,"mapping_status":'
    b'"not_applied","model_output_bin":15,"native_class_id":"midi_36",'
    b'"native_metadata":{"upstream_8hit_group_id":"kick"},"native_midi_note":36,'
    b'"prediction_map_version":null,"record_type":"event","time_sec":0.5,'
    b'"velocity_midi":100}\n'
)
EXPECTED_TERMINAL_LINE = (
    b'{"event_count":1,"prefix_sha256":'
    b'"96e224e6cd6b229a894ad6fdba4527329913b83d4ab8614d81b550a453cfc8de",'
    b'"record_type":"terminal"}\n'
)
EXPECTED_ARTIFACT = EXPECTED_HEADER_LINE + EXPECTED_EVENT_LINE + EXPECTED_TERMINAL_LINE


def make_audio() -> CanonicalAudio:
    return CanonicalAudio(
        path=Path("audio/input.wav"),
        source_audio_id="song-1",
        source_audio_sha256="a" * 64,
        input_view_id="direct",
        input_audio_sha256="b" * 64,
        byte_length=88244,
        sample_rate=44100,
        channel_count=1,
        sample_width_bytes=2,
        audio_frame_count=44100,
    )


def make_oaf_event(**changes: object) -> NativeEvent:
    values: dict[str, object] = {
        "time_sec": 0.5,
        "native_class_id": "midi_36",
        "model_output_bin": 15,
        "native_midi_note": 36,
        "native_metadata": {"upstream_8hit_group_id": "kick"},
        "confidence": 0.75,
        "velocity_midi": 100,
    }
    values.update(changes)
    return NativeEvent(**values)  # type: ignore[arg-type]


def make_heuristic_event(**changes: object) -> NativeEvent:
    values: dict[str, object] = {
        "time_sec": 0.5,
        "native_class_id": "heuristic_kick",
        "model_output_bin": None,
        "native_midi_note": None,
        "native_metadata": {},
        "confidence": None,
        "velocity_midi": None,
    }
    values.update(changes)
    return NativeEvent(**values)  # type: ignore[arg-type]


def make_oaf_prediction(
    events: tuple[NativeEvent, ...] | None = None,
    **changes: object,
) -> NativePrediction:
    values: dict[str, object] = {
        "audio": make_audio(),
        "descriptor": OAF_DESCRIPTOR,
        "events": (make_oaf_event(),) if events is None else events,
        "backend_lock_sha256": "c" * 64,
        "runtime_lock_sha256": "d" * 64,
        "parameter_lock_sha256": None,
        "model_artifact_set_sha256": "e" * 64,
        "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
    }
    values.update(changes)
    return NativePrediction(**values)  # type: ignore[arg-type]


def make_heuristic_prediction(
    events: tuple[NativeEvent, ...] | None = None,
    **changes: object,
) -> NativePrediction:
    values: dict[str, object] = {
        "audio": make_audio(),
        "descriptor": HEURISTIC_DESCRIPTOR,
        "events": (make_heuristic_event(),) if events is None else events,
        "backend_lock_sha256": None,
        "runtime_lock_sha256": None,
        "parameter_lock_sha256": "1" * 64,
        "model_artifact_set_sha256": None,
        "upstream_source_commit": None,
        "training_data_map_id": None,
    }
    values.update(changes)
    return NativePrediction(**values)  # type: ignore[arg-type]


def parsed_records(content: bytes) -> list[dict[str, object]]:
    records = [strict_json_loads(line) for line in content.splitlines()]
    assert all(isinstance(record, dict) for record in records)
    return records  # type: ignore[return-value]


def canonical_records(records: list[dict[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(record, trailing_newline=True) for record in records)


def canonical_records_with_prefix(records: list[dict[str, object]]) -> bytes:
    prefix = b"".join(
        canonical_json_bytes(record, trailing_newline=True) for record in records[:-1]
    )
    records[-1]["prefix_sha256"] = hashlib.sha256(prefix).hexdigest()
    return prefix + canonical_json_bytes(records[-1], trailing_newline=True)


def test_prediction_has_exact_canonical_records_and_hashes() -> None:
    content = render_prediction_artifact(make_oaf_prediction())
    artifact = read_prediction_artifact(content)
    records = parsed_records(content)

    assert content == EXPECTED_ARTIFACT
    assert set(records[0]) == HEADER_KEYS
    assert set(records[1]) == EVENT_KEYS
    assert set(records[2]) == TERMINAL_KEYS
    assert artifact.prefix_sha256 == (
        "96e224e6cd6b229a894ad6fdba4527329913b83d4ab8614d81b550a453cfc8de"
    )
    assert artifact.artifact_sha256 == (
        "1c99eb5a9fa729919aa620ef05ce0eebdac8974d536f446f70c5f3462c98ce96"
    )
    assert content.endswith(b"\n")
    assert content.count(b"\n") == 3


def test_prediction_quantizes_then_sorts_and_assigns_indexes() -> None:
    prediction = make_oaf_prediction(
        events=(
            make_oaf_event(
                time_sec=0.5000004,
                native_class_id="midi_38",
                model_output_bin=17,
                native_midi_note=38,
                native_metadata={"upstream_8hit_group_id": "snare"},
                confidence=0.7500004,
            ),
            make_oaf_event(
                time_sec=0.4999996,
                velocity_midi=90,
                confidence=0.2499996,
            ),
        )
    )

    content = render_prediction_artifact(prediction)
    records = parsed_records(content)

    expected_event_lines = (
        b'{"canonical_class":null,"confidence":0.25,"event_index":0,"mapping_status":'
        b'"not_applied","model_output_bin":15,"native_class_id":"midi_36",'
        b'"native_metadata":{"upstream_8hit_group_id":"kick"},"native_midi_note":36,'
        b'"prediction_map_version":null,"record_type":"event","time_sec":0.5,'
        b'"velocity_midi":90}\n'
        b'{"canonical_class":null,"confidence":0.75,"event_index":1,"mapping_status":'
        b'"not_applied","model_output_bin":17,"native_class_id":"midi_38",'
        b'"native_metadata":{"upstream_8hit_group_id":"snare"},"native_midi_note":38,'
        b'"prediction_map_version":null,"record_type":"event","time_sec":0.5,'
        b'"velocity_midi":100}\n'
    )
    expected_terminal = (
        b'{"event_count":2,"prefix_sha256":'
        b'"b7df20255e2711c8856a3c58a00801981f8ca482e597115eaa78b49469570a15",'
        b'"record_type":"terminal"}\n'
    )
    assert content == EXPECTED_HEADER_LINE + expected_event_lines + expected_terminal
    assert records[1]["event_index"] == 0
    assert records[1]["native_class_id"] == "midi_36"
    assert records[1]["time_sec"] == Decimal("0.5")
    assert records[-1]["event_count"] == 2
    assert read_prediction_artifact(content).artifact_sha256 == (
        "c16a4465e5e1580853cc578a24ed714ff5034064fad82746eeefe216f47e5438"
    )


def test_prediction_sorts_on_all_six_persisted_fields() -> None:
    events = (
        make_heuristic_event(
            time_sec=0.5,
            native_class_id="b",
            model_output_bin=0,
            native_midi_note=0,
            velocity_midi=0,
            confidence=0.0,
        ),
        make_heuristic_event(
            time_sec=0.5,
            native_class_id="b",
            model_output_bin=0,
            native_midi_note=0,
            velocity_midi=0,
        ),
        make_heuristic_event(
            time_sec=0.5,
            native_class_id="b",
            model_output_bin=0,
            native_midi_note=0,
        ),
        make_heuristic_event(
            time_sec=0.5,
            native_class_id="b",
            model_output_bin=0,
        ),
        make_heuristic_event(time_sec=0.5, native_class_id="b"),
        make_heuristic_event(time_sec=0.5, native_class_id="a", model_output_bin=10),
        make_heuristic_event(time_sec=0.4, native_class_id="z"),
    )

    records = parsed_records(render_prediction_artifact(make_heuristic_prediction(events=events)))
    observed = [
        (
            record["time_sec"],
            record["native_class_id"],
            record["model_output_bin"],
            record["native_midi_note"],
            record["velocity_midi"],
            record["confidence"],
        )
        for record in records[1:-1]
    ]

    assert observed == [
        (Decimal("0.4"), "z", None, None, None, None),
        (Decimal("0.5"), "a", 10, None, None, None),
        (Decimal("0.5"), "b", None, None, None, None),
        (Decimal("0.5"), "b", 0, None, None, None),
        (Decimal("0.5"), "b", 0, 0, None, None),
        (Decimal("0.5"), "b", 0, 0, 0, None),
        (Decimal("0.5"), "b", 0, 0, 0, Decimal("0")),
    ]


def test_prediction_rejects_full_sort_key_tie_even_when_metadata_differs() -> None:
    first = make_oaf_event()
    second = make_oaf_event(native_metadata={"upstream_8hit_group_id": None})

    with pytest.raises(PredictionArtifactError, match="duplicate_native_event"):
        render_prediction_artifact(make_oaf_prediction(events=(first, second)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_lock_sha256", None),
        ("runtime_lock_sha256", None),
        ("model_artifact_set_sha256", None),
        ("upstream_source_commit", None),
        ("training_data_map_id", None),
        ("parameter_lock_sha256", "2" * 64),
    ],
)
def test_oaf_header_requires_exact_lock_nullability(field: str, value: object) -> None:
    with pytest.raises(PredictionArtifactError, match="oaf_header_nullability"):
        render_prediction_artifact(make_oaf_prediction(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_lock_sha256", "2" * 64),
        ("runtime_lock_sha256", "2" * 64),
        ("model_artifact_set_sha256", "2" * 64),
        ("upstream_source_commit", "commit"),
        ("training_data_map_id", "map"),
        ("parameter_lock_sha256", None),
    ],
)
def test_heuristic_header_requires_exact_lock_nullability(field: str, value: object) -> None:
    with pytest.raises(PredictionArtifactError, match="heuristic_header_nullability"):
        render_prediction_artifact(make_heuristic_prediction(**{field: value}))


@pytest.mark.parametrize(
    ("descriptor_field", "replacement", "message"),
    [
        ("prediction_schema", "wrong-schema", "prediction_schema"),
        ("model_id", "", "model_id"),
        ("architecture_id", "", "architecture_id"),
        ("native_output_space_id", "", "native_output_space_id"),
        (
            "native_metadata_schema_id",
            "unknown-metadata-v1",
            "unknown native_metadata_schema_id",
        ),
    ],
)
def test_header_rejects_unknown_or_invalid_descriptor_ids(
    descriptor_field: str,
    replacement: str,
    message: str,
) -> None:
    payload = {**OAF_DESCRIPTOR_PAYLOAD, descriptor_field: replacement}
    descriptor_bytes = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    descriptor = BackendDescriptor(
        payload=payload,
        sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
    )

    with pytest.raises(PredictionArtifactError, match=message):
        render_prediction_artifact(make_oaf_prediction(descriptor=descriptor))


@pytest.mark.parametrize("field", ["source_audio_id", "input_view_id"])
def test_header_rejects_empty_audio_identity(field: str) -> None:
    audio = replace(make_audio(), **{field: ""})

    with pytest.raises(PredictionArtifactError, match=field):
        render_prediction_artifact(make_oaf_prediction(audio=audio))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_audio_sha256", "A" * 64),
        ("input_audio_sha256", "b" * 63),
        ("byte_length", 0),
        ("sample_rate", True),
        ("channel_count", 0),
        ("sample_width_bytes", 0),
        ("audio_frame_count", 0),
    ],
)
def test_header_rejects_invalid_audio_provenance(field: str, value: object) -> None:
    audio = replace(make_audio(), **{field: value})

    with pytest.raises(PredictionArtifactError, match=field):
        render_prediction_artifact(make_oaf_prediction(audio=audio))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_output_bin", None),
        ("native_midi_note", None),
        ("confidence", None),
        ("velocity_midi", None),
    ],
)
def test_oaf_events_require_all_native_numeric_fields(field: str, value: object) -> None:
    with pytest.raises(PredictionArtifactError, match="oaf_event_nullability"):
        render_prediction_artifact(make_oaf_prediction(events=(make_oaf_event(**{field: value}),)))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_output_bin", -1, "model_output_bin"),
        ("model_output_bin", 88, "model_output_bin"),
        ("native_midi_note", 35, "oaf_native_identity"),
        ("native_class_id", "midi_035", "oaf_native_identity"),
        ("native_class_id", "kick", "oaf_native_identity"),
    ],
)
def test_oaf_events_require_bin_pitch_and_class_agreement(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(PredictionArtifactError, match=message):
        render_prediction_artifact(make_oaf_prediction(events=(make_oaf_event(**{field: value}),)))


def test_oaf_metadata_policy_has_exact_eight_hit_group_ids() -> None:
    assert OAF_GROUP_IDS == EXPECTED_OAF_GROUP_IDS


@pytest.mark.parametrize("group_id", sorted(EXPECTED_OAF_GROUP_IDS) + [None])
def test_oaf_metadata_accepts_exact_eight_hit_group_ids(group_id: str | None) -> None:
    content = render_prediction_artifact(
        make_oaf_prediction(
            events=(make_oaf_event(native_metadata={"upstream_8hit_group_id": group_id}),)
        )
    )

    assert parsed_records(content)[1]["native_metadata"] == {"upstream_8hit_group_id": group_id}


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"upstream_8hit_group_id": "kick", "extra": None},
        {"upstream_8hit_group_id": "unknown"},
        {"upstream_8hit_group_id": 1},
    ],
)
def test_oaf_metadata_rejects_missing_unknown_or_wrong_values(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(PredictionArtifactError, match="native_metadata"):
        render_prediction_artifact(
            make_oaf_prediction(events=(make_oaf_event(native_metadata=metadata),))
        )


def test_heuristic_metadata_requires_an_empty_object() -> None:
    assert (
        parsed_records(render_prediction_artifact(make_heuristic_prediction()))[1][
            "native_metadata"
        ]
        == {}
    )

    with pytest.raises(PredictionArtifactError, match="native_metadata"):
        render_prediction_artifact(
            make_heuristic_prediction(
                events=(make_heuristic_event(native_metadata={"source": "guessed"}),)
            )
        )


@pytest.mark.parametrize("time_sec", [-0.0000001, float("nan"), float("inf"), float("-inf")])
def test_prediction_rejects_negative_or_nonfinite_time(time_sec: float) -> None:
    with pytest.raises(PredictionArtifactError, match="time_sec"):
        render_prediction_artifact(
            make_heuristic_prediction(events=(make_heuristic_event(time_sec=time_sec),))
        )


@pytest.mark.parametrize("confidence", [-0.000001, 1.000001, float("nan"), float("inf")])
def test_prediction_rejects_out_of_range_or_nonfinite_confidence(confidence: float) -> None:
    with pytest.raises(PredictionArtifactError, match="confidence"):
        render_prediction_artifact(
            make_heuristic_prediction(events=(make_heuristic_event(confidence=confidence),))
        )


@pytest.mark.parametrize("velocity", [-1, 128, True, 1.5])
def test_prediction_rejects_invalid_velocity(velocity: object) -> None:
    with pytest.raises(PredictionArtifactError, match="velocity_midi"):
        render_prediction_artifact(
            make_heuristic_prediction(events=(make_heuristic_event(velocity_midi=velocity),))
        )


@pytest.mark.parametrize("value", [-1, 88, True, 1.5])
def test_prediction_rejects_invalid_optional_output_bin(value: object) -> None:
    with pytest.raises(PredictionArtifactError, match="model_output_bin"):
        render_prediction_artifact(
            make_heuristic_prediction(events=(make_heuristic_event(model_output_bin=value),))
        )


def test_zero_event_prediction_contains_only_header_and_terminal() -> None:
    content = render_prediction_artifact(make_oaf_prediction(events=()))
    records = parsed_records(content)

    assert [record["record_type"] for record in records] == ["header", "terminal"]
    assert records[-1]["event_count"] == 0
    assert records[-1]["prefix_sha256"] == (
        "e85c2281a2f0a544cd1aa2f6e3233458972b375c44b6f3f46fa41ecb9e62663e"
    )


def test_native_mapping_fields_are_fixed_and_unmapped() -> None:
    records = parsed_records(render_prediction_artifact(make_oaf_prediction()))

    assert records[0]["artifact_role"] == "native"
    assert records[1]["mapping_status"] == "not_applied"
    assert records[1]["prediction_map_version"] is None
    assert records[1]["canonical_class"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mapping_status", "mapped"),
        ("prediction_map_version", "map-v1"),
        ("canonical_class", "kick"),
    ],
)
def test_reader_rejects_non_native_mapping_fields(field: str, value: object) -> None:
    records = parsed_records(EXPECTED_ARTIFACT)
    records[1][field] = value

    with pytest.raises(PredictionArtifactError, match=field):
        read_prediction_artifact(canonical_records(records))


@pytest.mark.parametrize(
    ("record_index", "field"),
    [(0, "unknown_header"), (1, "unknown_event"), (2, "unknown_terminal")],
)
def test_reader_rejects_unknown_record_keys(record_index: int, field: str) -> None:
    records = parsed_records(EXPECTED_ARTIFACT)
    records[record_index][field] = "unexpected"

    with pytest.raises(PredictionArtifactError, match="exact key set"):
        read_prediction_artifact(canonical_records(records))


@pytest.mark.parametrize(
    ("record_index", "field"),
    [(0, "schema"), (1, "time_sec"), (2, "event_count")],
)
def test_reader_rejects_missing_record_keys(record_index: int, field: str) -> None:
    records = parsed_records(EXPECTED_ARTIFACT)
    del records[record_index][field]

    with pytest.raises(PredictionArtifactError, match="exact key set"):
        read_prediction_artifact(canonical_records(records))


@pytest.mark.parametrize(
    "content",
    [
        EXPECTED_ARTIFACT.replace(
            b'"artifact_role":"native"',
            b'"artifact_role":"native","artifact_role":"native"',
            1,
        ),
        EXPECTED_ARTIFACT.replace(
            b'"event_index":0',
            b'"event_index":0,"event_index":0',
            1,
        ),
        EXPECTED_ARTIFACT.replace(
            b'"event_count":1',
            b'"event_count":1,"event_count":1',
            1,
        ),
    ],
)
def test_reader_rejects_duplicate_keys_in_every_record_kind(content: bytes) -> None:
    with pytest.raises(PredictionArtifactError, match="duplicate key"):
        read_prediction_artifact(content)


@pytest.mark.parametrize(
    "content",
    [
        EXPECTED_ARTIFACT.replace(
            b'{"architecture_id":"magenta-oaf-model-tpu-drums-v1","artifact_role":"native"',
            b'{"artifact_role":"native","architecture_id":"magenta-oaf-model-tpu-drums-v1"',
            1,
        ),
        EXPECTED_ARTIFACT.replace(b'"time_sec":0.5', b'"time_sec":5e-1', 1),
        EXPECTED_ARTIFACT.replace(b'"time_sec":0.5', b'"time_sec":-0', 1),
    ],
)
def test_reader_rejects_reordered_exponent_and_negative_zero_bytes(content: bytes) -> None:
    with pytest.raises(PredictionArtifactError, match="canonical"):
        read_prediction_artifact(content)


@pytest.mark.parametrize(
    "content",
    [
        EXPECTED_ARTIFACT[:-1],
        EXPECTED_HEADER_LINE + b"\n" + EXPECTED_EVENT_LINE + EXPECTED_TERMINAL_LINE,
        EXPECTED_ARTIFACT + EXPECTED_EVENT_LINE,
        EXPECTED_HEADER_LINE,
        EXPECTED_TERMINAL_LINE,
        EXPECTED_EVENT_LINE + EXPECTED_TERMINAL_LINE,
        EXPECTED_HEADER_LINE + EXPECTED_TERMINAL_LINE + EXPECTED_EVENT_LINE,
    ],
)
def test_reader_rejects_bad_physical_record_layout(content: bytes) -> None:
    with pytest.raises(PredictionArtifactError):
        read_prediction_artifact(content)


@pytest.mark.parametrize(
    ("record_index", "field", "value"),
    [
        (1, "event_index", 4),
        (2, "event_count", 2),
        (2, "prefix_sha256", "0" * 64),
    ],
)
def test_reader_rejects_index_count_and_prefix_mismatches(
    record_index: int,
    field: str,
    value: object,
) -> None:
    records = parsed_records(EXPECTED_ARTIFACT)
    records[record_index][field] = value

    with pytest.raises(PredictionArtifactError):
        read_prediction_artifact(canonical_records(records))


def test_reader_round_trip_is_byte_identical() -> None:
    artifact = read_prediction_artifact(EXPECTED_ARTIFACT)

    assert artifact.content == EXPECTED_ARTIFACT
    assert artifact.event_count == 1
    assert artifact.prediction.events == (
        NativeEvent(
            time_sec=0.5,
            native_class_id="midi_36",
            model_output_bin=15,
            native_midi_note=36,
            native_metadata={"upstream_8hit_group_id": "kick"},
            confidence=0.75,
            velocity_midi=100,
        ),
    )
    assert render_prediction_artifact(artifact.prediction) == EXPECTED_ARTIFACT


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate", 48000),
        ("channel_count", 2),
        ("sample_width_bytes", 3),
        ("audio_frame_count", 0),
        ("byte_length", 88243),
    ],
)
def test_reader_rejects_noncanonical_audio_contract(field: str, value: object) -> None:
    records = parsed_records(EXPECTED_ARTIFACT)
    records[0][field] = value

    with pytest.raises(PredictionArtifactError, match=field):
        read_prediction_artifact(canonical_records_with_prefix(records))


@pytest.mark.parametrize(
    ("header_field", "descriptor_field", "replacement"),
    [
        ("backend_lock_sha256", "backend_lock_sha256", "1" * 64),
        ("runtime_lock_sha256", "runtime_lock_sha256", "2" * 64),
        ("model_artifact_set_sha256", "model_artifact_set_sha256", "3" * 64),
        ("upstream_source_commit", "upstream_source_commit", "4" * 40),
        ("training_data_map_id", "training_data_map_id", "different-map-v1"),
    ],
)
def test_reader_binds_every_oaf_header_identity_to_frozen_descriptor(
    header_field: str,
    descriptor_field: str,
    replacement: str,
) -> None:
    content = render_prediction_artifact(make_oaf_prediction(descriptor=FROZEN_OAF_DESCRIPTOR))
    records = parsed_records(content)
    records[0][header_field] = replacement

    with pytest.raises(PredictionArtifactError, match=header_field):
        read_prediction_artifact(canonical_records_with_prefix(records))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("backend_lock_sha256", "1" * 64),
        ("runtime_lock_sha256", "2" * 64),
        ("model_artifact_set_sha256", "3" * 64),
        ("upstream_source_commit", "4" * 40),
        ("training_data_map_id", "different-map-v1"),
    ],
)
def test_writer_binds_every_oaf_prediction_identity_to_frozen_descriptor(
    field: str,
    replacement: str,
) -> None:
    prediction = make_oaf_prediction(
        descriptor=FROZEN_OAF_DESCRIPTOR,
        **{field: replacement},
    )

    with pytest.raises(PredictionArtifactError, match=field):
        render_prediction_artifact(prediction)


def test_frozen_oaf_writer_output_is_accepted_by_the_strict_reader() -> None:
    content = render_prediction_artifact(make_oaf_prediction(descriptor=FROZEN_OAF_DESCRIPTOR))

    assert read_prediction_artifact(content).content == content


def test_known_oaf_backend_cannot_bypass_frozen_descriptor_rules() -> None:
    payload = {
        **OAF_DESCRIPTOR_PAYLOAD,
        "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
    }
    descriptor = BackendDescriptor(
        payload=payload,
        sha256=sha256_hex(canonical_json_bytes(payload)),
    )

    with pytest.raises(PredictionArtifactError, match="descriptor"):
        render_prediction_artifact(make_oaf_prediction(descriptor=descriptor))


def test_reader_returns_detached_immutable_nested_identity_mappings() -> None:
    artifact = read_prediction_artifact(EXPECTED_ARTIFACT)
    descriptor = artifact.prediction.descriptor.payload
    metadata = artifact.prediction.events[0].native_metadata

    with pytest.raises(TypeError):
        descriptor["model_id"] = "tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        metadata["upstream_8hit_group_id"] = "snare"  # type: ignore[index]
    assert artifact.content == EXPECTED_ARTIFACT


def test_generic_reader_does_not_import_oaf_or_mapping_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        forbidden_fromlist = {"mapping", "oaf_adapter", "oaf_backend"}
        if (
            "oaf" in name
            or name == "src.benchmark.mapping"
            or forbidden_fromlist.intersection(fromlist)
        ):
            raise AssertionError(f"forbidden backend-specific import: {name}")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.delitem(sys.modules, "src.benchmark.mapping", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.reload(sys.modules["src.benchmark.prediction_artifact"])

    assert module.read_prediction_artifact(EXPECTED_ARTIFACT).event_count == 1


def test_publish_prediction_artifact_returns_strict_read_immutable_bytes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "requested" / "prediction.jsonl"

    published = publish_prediction_artifact(destination, make_oaf_prediction())

    assert published.role == "prediction"
    assert published.path == destination
    assert published.sha256 == hashlib.sha256(EXPECTED_ARTIFACT).hexdigest()
    assert destination.read_bytes() == EXPECTED_ARTIFACT
    assert read_prediction_artifact(destination.read_bytes()).content == EXPECTED_ARTIFACT


def test_publish_prediction_artifact_preserves_prior_valid_prediction_on_collision(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "prediction.jsonl"
    destination.write_bytes(EXPECTED_ARTIFACT)

    with pytest.raises(Exception):
        publish_prediction_artifact(destination, make_oaf_prediction(events=()))

    assert destination.read_bytes() == EXPECTED_ARTIFACT
