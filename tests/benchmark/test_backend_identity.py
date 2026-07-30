from decimal import Decimal

import pytest

from src.benchmark.backend_identity import (
    StrictJsonError,
    build_descriptor,
    canonical_json_bytes,
    quantize_six,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)

HEURISTIC_DESCRIPTOR_KEYS = frozenset(
    {
        "descriptor_schema",
        "backend_id",
        "prediction_schema",
        "adapter_source_manifest_sha256",
        "parameter_lock_sha256",
        "model_id",
        "architecture_id",
        "native_output_space_id",
        "native_metadata_schema_id",
    }
)
HEURISTIC_DESCRIPTOR_SCHEMA = "crux.heuristic-backend-descriptor/v1"
HEURISTIC_DESCRIPTOR = {
    "descriptor_schema": HEURISTIC_DESCRIPTOR_SCHEMA,
    "backend_id": "heuristic-onset-v1",
    "prediction_schema": "crux.drum-prediction-events/v1",
    "adapter_source_manifest_sha256": "a" * 64,
    "parameter_lock_sha256": "b" * 64,
    "model_id": "crux-heuristic-onset-nonmodel-v1",
    "architecture_id": "librosa-onset-centroid-zcr-v1",
    "native_output_space_id": "crux-heuristic-midi7-v1",
    "native_metadata_schema_id": "crux-empty-native-metadata-v1",
}
HEURISTIC_DESCRIPTOR_BYTES = (
    b'{"adapter_source_manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    b'aaaaaaaaaaaaaaaa","architecture_id":"librosa-onset-centroid-zcr-v1","backend_id":'
    b'"heuristic-onset-v1","descriptor_schema":"crux.heuristic-backend-descriptor/v1",'
    b'"model_id":"crux-heuristic-onset-nonmodel-v1","native_metadata_schema_id":'
    b'"crux-empty-native-metadata-v1","native_output_space_id":"crux-heuristic-midi7-v1",'
    b'"parameter_lock_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
    b'bbbbbbbb","prediction_schema":"crux.drum-prediction-events/v1"}'
)


def test_canonical_json_sorts_keys_and_preserves_decimal_number() -> None:
    payload = {"z": Decimal("1.250000"), "a": "音楽"}

    assert canonical_json_bytes(payload, trailing_newline=True) == (
        b'{"a":"\xe9\x9f\xb3\xe6\xa5\xbd","z":1.25}\n'
    )


@pytest.mark.parametrize(
    "value",
    [
        0.5,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_canonical_json_rejects_binary_floats(value: float) -> None:
    with pytest.raises(StrictJsonError, match="unsupported JSON value"):
        canonical_json_bytes({"value": value})


def test_canonical_json_rejects_surrogate_strings() -> None:
    with pytest.raises(StrictJsonError, match="UTF-8"):
        canonical_json_bytes({"unsafe": "\ud800"})


def test_canonical_json_rejects_non_string_object_keys() -> None:
    with pytest.raises(StrictJsonError, match="object keys must be strings"):
        canonical_json_bytes({1: "value"})  # type: ignore[dict-item]


@pytest.mark.parametrize("value", [(1, 2), {1, 2}, b"bytes"])
def test_canonical_json_rejects_unsupported_containers(value: object) -> None:
    with pytest.raises(StrictJsonError, match="unsupported JSON value"):
        canonical_json_bytes({"value": value})  # type: ignore[dict-item]


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity")])
def test_canonical_json_rejects_nonfinite_decimals(value: Decimal) -> None:
    with pytest.raises(StrictJsonError, match="nonfinite decimal"):
        canonical_json_bytes({"value": value})


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(StrictJsonError, match="duplicate key"):
        strict_json_loads(b'{"schema":"a","schema":"b"}')


def test_strict_json_rejects_non_utf8_bytes() -> None:
    with pytest.raises(StrictJsonError, match="UTF-8"):
        strict_json_loads(b'{"value":"\xff"}')


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_strict_json_rejects_nonfinite_constants(constant: bytes) -> None:
    with pytest.raises(StrictJsonError, match="nonfinite JSON constant"):
        strict_json_loads(b'{"value":' + constant + b"}")


def test_strict_json_preserves_decimal_and_can_require_canonical_bytes() -> None:
    content = b'{"a":[null,true,2,1.25],"z":"music"}'

    assert strict_json_loads(content, require_canonical=True) == {
        "a": [None, True, 2, Decimal("1.25")],
        "z": "music",
    }
    with pytest.raises(StrictJsonError, match="not canonical"):
        strict_json_loads(b'{"z":"music", "a":[null,true,2,1.25]}', require_canonical=True)


def test_quantize_six_uses_binary64_then_half_even() -> None:
    assert quantize_six(0.1234565) == Decimal.from_float(0.1234565).quantize(Decimal("0.000001"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_quantize_six_rejects_nonfinite_binary_float(value: float) -> None:
    with pytest.raises(StrictJsonError, match="nonfinite binary float"):
        quantize_six(value)


def test_quantize_six_normalizes_negative_zero() -> None:
    assert quantize_six(-0.0).as_tuple().sign == 0


def test_sha256_helpers_require_lowercase_hex() -> None:
    assert sha256_hex(b"crux") == "3e2819ddf33ea664304986228bafca1c7af4cbb514400c45c85b8c909cdff723"
    assert require_sha256("a" * 64, "artifact_sha256") == "a" * 64

    with pytest.raises(StrictJsonError, match="artifact_sha256 must be lowercase SHA-256"):
        require_sha256("A" * 64, "artifact_sha256")


def test_build_descriptor_has_exact_canonical_bytes_and_hash() -> None:
    descriptor = build_descriptor(
        HEURISTIC_DESCRIPTOR,
        allowed_keys=HEURISTIC_DESCRIPTOR_KEYS,
        schema=HEURISTIC_DESCRIPTOR_SCHEMA,
    )

    assert canonical_json_bytes(dict(descriptor.payload)) == HEURISTIC_DESCRIPTOR_BYTES
    assert descriptor.sha256 == "56df815e050d1b49923de40ddf64fb3d1fec79dc0a603f05d8a4c18dbbd44487"
    assert descriptor.sha256 != sha256_hex(HEURISTIC_DESCRIPTOR_BYTES + b"\n")


@pytest.mark.parametrize(
    "field",
    ["timestamp", "request_id", "path", "dirty", "verification_status"],
)
def test_build_descriptor_rejects_operational_fields(field: str) -> None:
    payload = {**HEURISTIC_DESCRIPTOR, field: "excluded"}

    with pytest.raises(StrictJsonError, match="exact key set"):
        build_descriptor(
            payload,
            allowed_keys=HEURISTIC_DESCRIPTOR_KEYS,
            schema=HEURISTIC_DESCRIPTOR_SCHEMA,
        )


def test_build_descriptor_rejects_missing_fields() -> None:
    payload = {
        key: value for key, value in HEURISTIC_DESCRIPTOR.items() if key != "parameter_lock_sha256"
    }

    with pytest.raises(StrictJsonError, match="exact key set"):
        build_descriptor(
            payload,
            allowed_keys=HEURISTIC_DESCRIPTOR_KEYS,
            schema=HEURISTIC_DESCRIPTOR_SCHEMA,
        )


def test_build_descriptor_rejects_non_string_values() -> None:
    payload = {**HEURISTIC_DESCRIPTOR, "backend_id": 1}

    with pytest.raises(StrictJsonError, match="descriptor fields must be strings"):
        build_descriptor(
            payload,
            allowed_keys=HEURISTIC_DESCRIPTOR_KEYS,
            schema=HEURISTIC_DESCRIPTOR_SCHEMA,
        )


def test_build_descriptor_requires_matching_schema() -> None:
    payload = {**HEURISTIC_DESCRIPTOR, "descriptor_schema": "wrong-schema"}

    with pytest.raises(StrictJsonError, match="descriptor_schema"):
        build_descriptor(
            payload,
            allowed_keys=HEURISTIC_DESCRIPTOR_KEYS,
            schema=HEURISTIC_DESCRIPTOR_SCHEMA,
        )
