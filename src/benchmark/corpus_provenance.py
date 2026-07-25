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

_NULLABLE_STRING_FIELDS = (
    "source_origin",
    "source_author_or_pack",
    "source_reference",
    "provenance_notes",
)


def load_provenance(path: Path | None) -> dict[int, ProvenanceRecord]:
    if path is None:
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("invalid provenance JSON") from None

    if not isinstance(payload, dict) or payload.get("schema_version") != PROVENANCE_SCHEMA:
        raise ValueError("unsupported provenance schema_version")

    raw_simfiles = payload.get("simfiles")
    if not isinstance(raw_simfiles, dict):
        raise ValueError("provenance simfiles must be an object")

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

    simfile_id = int(raw_id)
    if not 0 <= simfile_id <= MAX_SIMFILE_ID:
        raise ValueError("simfile ID must be a decimal integer in supported range")
    return simfile_id


def _parse_record(raw_record: object) -> ProvenanceRecord:
    if not isinstance(raw_record, dict):
        raise ValueError("provenance record must be an object")
    if raw_record.keys() - ALLOWED_FIELDS:
        raise ValueError("provenance record contains unsupported field")

    for field in _NULLABLE_STRING_FIELDS:
        value = raw_record.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError("provenance string fields must be strings or null")

    rights_status = raw_record.get("rights_status", "unknown")
    if not isinstance(rights_status, str) or not rights_status:
        raise ValueError("rights_status must be a non-empty string")

    redistribution_allowed = raw_record.get("redistribution_allowed")
    if redistribution_allowed is not None and type(redistribution_allowed) is not bool:
        raise ValueError("redistribution_allowed must be a boolean or null")

    return ProvenanceRecord(
        source_origin=raw_record.get("source_origin"),
        source_author_or_pack=raw_record.get("source_author_or_pack"),
        source_reference=raw_record.get("source_reference"),
        rights_status=rights_status,
        redistribution_allowed=redistribution_allowed,
        provenance_notes=raw_record.get("provenance_notes"),
    )
