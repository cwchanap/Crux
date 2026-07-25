import json
from pathlib import Path

import pytest

from src.benchmark.corpus_provenance import load_provenance, provenance_for

MAX_SAFE_SIMFILE_ID = 9_007_199_254_740_991


def write_mapping(
    path: Path,
    simfiles: dict[str, object],
    schema: object = "crux.corpus-provenance/v1",
) -> None:
    path.write_text(
        json.dumps({"schema_version": schema, "simfiles": simfiles}),
        encoding="utf-8",
    )


def write_raw_mapping(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_loads_known_record_and_supplies_explicit_unknown_default(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    write_mapping(
        path,
        {
            "42": {
                "source_origin": "personal",
                "source_author_or_pack": "Example Pack",
                "source_reference": "private archive",
                "rights_status": "privately_authorized",
                "redistribution_allowed": False,
                "provenance_notes": "Local benchmark use.",
            }
        },
    )

    records = load_provenance(path)

    assert records[42].rights_status == "privately_authorized"
    assert records[42].redistribution_allowed is False
    assert provenance_for(records, 99).rights_status == "unknown"
    assert provenance_for(records, 99).redistribution_allowed is None


def test_absent_provenance_mapping_loads_no_records() -> None:
    assert load_provenance(None) == {}


def test_loads_checked_in_empty_provenance_mapping() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    assert load_provenance(repository_root / "config/corpus-provenance.json") == {}


@pytest.mark.parametrize(
    ("schema", "unsafe_value"),
    [
        ("", None),
        ("crux.corpus-provenance/v2", "crux.corpus-provenance/v2"),
        ("other", "other"),
        (None, None),
    ],
)
def test_rejects_unknown_schema_without_echoing_value(
    schema: object, unsafe_value: str | None, tmp_path: Path
) -> None:
    path = tmp_path / "provenance.json"
    write_mapping(path, {}, schema=schema)

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == "unsupported provenance schema_version"
    if unsafe_value is not None:
        assert unsafe_value not in str(error.value)


def test_rejects_malformed_json_with_a_sanitized_error(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    path.write_text('{"schema_version":"secret-token"', encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == "invalid provenance JSON"
    assert "secret-token" not in str(error.value)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "unsupported provenance schema_version"),
        (
            {"schema_version": "crux.corpus-provenance/v1", "simfiles": []},
            "provenance simfiles must be an object",
        ),
    ],
)
def test_rejects_non_object_document_components(
    payload: object, message: str, tmp_path: Path
) -> None:
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == message


@pytest.mark.parametrize(
    ("field_name", "unsafe_value"),
    [
        ("credentials", "AKIA-not-a-real-key"),
        ("secret", "private-secret-value"),
        ("signed_url", "https://example.invalid/object?X-Amz-Signature=secret"),
        ("body", "untrusted-body-content"),
        ("unexpected_metadata", "unexpected-value"),
    ],
)
def test_rejects_unknown_root_fields_without_echoing_keys_or_values(
    field_name: str, unsafe_value: str, tmp_path: Path
) -> None:
    path = tmp_path / "provenance.json"
    write_raw_mapping(
        path,
        "{"
        + f'"schema_version":"crux.corpus-provenance/v1","simfiles":{{}},"{field_name}":'
        + json.dumps(unsafe_value)
        + "}",
    )

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == "provenance document contains unsupported field"
    assert field_name not in str(error.value)
    assert unsafe_value not in str(error.value)


def test_rejects_duplicate_root_field_names_from_raw_json(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    write_raw_mapping(
        path,
        '{"schema_version":"crux.corpus-provenance/v1","schema_version":"secret-schema","simfiles":{}}',
    )

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == "duplicate provenance document field"
    assert "secret-schema" not in str(error.value)


def test_rejects_duplicate_record_field_names_from_raw_json(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    write_raw_mapping(
        path,
        '{"schema_version":"crux.corpus-provenance/v1","simfiles":{"42":'
        '{"source_origin":"first","source_origin":"secret-origin"}}}',
    )

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == "duplicate provenance record field"
    assert "secret-origin" not in str(error.value)


def test_rejects_duplicate_exact_simfile_ids_from_raw_json(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    write_raw_mapping(
        path,
        '{"schema_version":"crux.corpus-provenance/v1","simfiles":{"42":{},"42":'
        '{"source_origin":"secret-origin"}}}',
    )

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == "duplicate simfile ID after numeric normalization"
    assert "secret-origin" not in str(error.value)


def test_accepts_inclusive_simfile_id_bounds(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    write_mapping(path, {"0": {}, str(MAX_SAFE_SIMFILE_ID): {}})

    assert set(load_provenance(path)) == {0, MAX_SAFE_SIMFILE_ID}


@pytest.mark.parametrize("raw_id", [str(MAX_SAFE_SIMFILE_ID + 1), "9" * 5_000])
def test_rejects_out_of_range_simfile_ids_without_integer_conversion_leaks(
    raw_id: str, tmp_path: Path
) -> None:
    path = tmp_path / "provenance.json"
    write_mapping(path, {raw_id: {}})

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == "simfile ID must be a decimal integer in supported range"
    assert raw_id not in str(error.value)


def test_accepts_thousands_of_zeroes_before_an_in_range_simfile_id(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    write_mapping(path, {"0" * 5_000 + "42": {}})

    assert set(load_provenance(path)) == {42}


def test_detects_duplicates_after_large_zero_padded_id_normalization(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    write_mapping(path, {"0" * 5_000 + "42": {}, "42": {}})

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == "duplicate simfile ID after numeric normalization"


@pytest.mark.parametrize("raw_id", ["-1", "9007199254740992", "1.5", "not-an-id"])
def test_rejects_out_of_range_or_non_decimal_simfile_ids_without_echoing_value(
    raw_id: str, tmp_path: Path
) -> None:
    path = tmp_path / "provenance.json"
    write_mapping(path, {raw_id: {}})

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == "simfile ID must be a decimal integer in supported range"
    assert raw_id not in str(error.value)


def test_rejects_duplicate_ids_after_numeric_normalization(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    write_mapping(path, {"1": {}, "01": {}})

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == "duplicate simfile ID after numeric normalization"


@pytest.mark.parametrize(
    ("record", "message", "unsafe_value"),
    [
        ([], "provenance record must be an object", "[]"),
        (
            {"__unsafe_field__": "secret-value"},
            "provenance record contains unsupported field",
            "secret-value",
        ),
        (
            {"source_origin": ["secret-value"]},
            "provenance string fields must be strings or null",
            "secret-value",
        ),
        (
            {"rights_status": ""},
            "rights_status must be a non-empty string",
            "",
        ),
        (
            {"rights_status": None},
            "rights_status must be a non-empty string",
            "None",
        ),
        (
            {"redistribution_allowed": 1},
            "redistribution_allowed must be a boolean or null",
            "1",
        ),
        (
            {"redistribution_allowed": "secret-value"},
            "redistribution_allowed must be a boolean or null",
            "secret-value",
        ),
    ],
)
def test_rejects_invalid_record_fields_without_echoing_values(
    record: object, message: str, unsafe_value: str, tmp_path: Path
) -> None:
    path = tmp_path / "provenance.json"
    write_mapping(path, {"42": record})

    with pytest.raises(ValueError) as error:
        load_provenance(path)

    assert str(error.value) == message
    if unsafe_value:
        assert unsafe_value not in str(error.value)
