from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.r2_corpus_models import (
    MAX_SIMFILE_ID,
    PROVENANCE_SCHEMA,
    ProvenanceRecord,
)

ALLOWED_FIELDS = {
    "source_origin",
    "source_author_or_pack",
    "source_reference",
    "rights_status",
    "redistribution_allowed",
    "provenance_notes",
}
_ROOT_FIELDS = {"schema_version", "simfiles"}
_MAX_SIMFILE_ID_TEXT = str(MAX_SIMFILE_ID)

_NULLABLE_STRING_FIELDS = (
    "source_origin",
    "source_author_or_pack",
    "source_reference",
    "provenance_notes",
)


class _JSONObject(list[tuple[str, object]]):
    """Preserves JSON object member order and duplicates during validation."""


def load_provenance(path: Path | None) -> dict[int, ProvenanceRecord]:
    if path is None:
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_JSONObject)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("invalid provenance JSON") from None

    document = _parse_object(
        payload,
        invalid_message="unsupported provenance schema_version",
        duplicate_message="duplicate provenance document field",
    )
    if document.keys() - _ROOT_FIELDS:
        raise ValueError("provenance document contains unsupported field")
    if document.get("schema_version") != PROVENANCE_SCHEMA:
        raise ValueError("unsupported provenance schema_version")

    raw_simfiles = _parse_object(
        document.get("simfiles"),
        invalid_message="provenance simfiles must be an object",
        duplicate_message="duplicate simfile ID after numeric normalization",
    )

    records: dict[int, ProvenanceRecord] = {}
    for raw_id, raw_record in raw_simfiles.items():
        simfile_id = _parse_id(raw_id)
        if simfile_id in records:
            raise ValueError("duplicate simfile ID after numeric normalization")
        records[simfile_id] = _parse_record(raw_record)
    return records


def provenance_for(
    mapping: dict[int, ProvenanceRecord],
    simfile_id: int,
) -> ProvenanceRecord:
    return mapping.get(simfile_id, ProvenanceRecord())


def _parse_id(raw_id: object) -> int:
    if not isinstance(raw_id, str) or not raw_id.isascii() or not raw_id.isdecimal():
        raise ValueError("simfile ID must be a decimal integer in supported range")

    normalized_id = raw_id.lstrip("0") or "0"
    if len(normalized_id) > len(_MAX_SIMFILE_ID_TEXT) or (
        len(normalized_id) == len(_MAX_SIMFILE_ID_TEXT) and normalized_id > _MAX_SIMFILE_ID_TEXT
    ):
        raise ValueError("simfile ID must be a decimal integer in supported range")
    return int(normalized_id)


def _parse_record(raw_record: object) -> ProvenanceRecord:
    record = _parse_object(
        raw_record,
        invalid_message="provenance record must be an object",
        duplicate_message="duplicate provenance record field",
    )
    if record.keys() - ALLOWED_FIELDS:
        raise ValueError("provenance record contains unsupported field")

    for field in _NULLABLE_STRING_FIELDS:
        value = record.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError("provenance string fields must be strings or null")

    rights_status = record.get("rights_status", "unknown")
    if not isinstance(rights_status, str) or not rights_status:
        raise ValueError("rights_status must be a non-empty string")

    redistribution_allowed = record.get("redistribution_allowed")
    if redistribution_allowed is not None and type(redistribution_allowed) is not bool:
        raise ValueError("redistribution_allowed must be a boolean or null")

    return ProvenanceRecord(
        source_origin=record.get("source_origin"),
        source_author_or_pack=record.get("source_author_or_pack"),
        source_reference=record.get("source_reference"),
        rights_status=rights_status,
        redistribution_allowed=redistribution_allowed,
        provenance_notes=record.get("provenance_notes"),
    )


def _parse_object(
    value: object,
    *,
    invalid_message: str,
    duplicate_message: str,
) -> dict[str, object]:
    if not isinstance(value, _JSONObject):
        raise ValueError(invalid_message)

    parsed: dict[str, object] = {}
    for key, member_value in value:
        if key in parsed:
            raise ValueError(duplicate_message)
        parsed[key] = member_value
    return parsed
