from __future__ import annotations

import re
from dataclasses import dataclass

from src.benchmark.dtx_text import decode_dtxmania_text

_SLOT_DIRECTIVE_RE = re.compile(
    r"^[#*]\s*L(?P<level>[1-5])(?P<field>LABEL|FILE)(?:\s*:\s*|\s+|(?=;)|$)(?P<value>.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SetDefSlot:
    level: int
    label: str | None
    file: str | None


@dataclass(frozen=True)
class ParsedSetDef:
    slots: tuple[SetDefSlot, ...]
    warnings: tuple[str, ...]


def parse_set_def_bytes(raw: bytes, *, source_name: str) -> ParsedSetDef:
    text = decode_dtxmania_text(raw, source_name=source_name, kind="set_def")
    return parse_set_def_text(text)


def parse_set_def_text(text: str) -> ParsedSetDef:
    text = text.removeprefix("\ufeff")
    values: dict[int, dict[str, str | None]] = {level: {} for level in range(1, 6)}
    seen_fields: set[tuple[int, str]] = set()
    warnings: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue

        match = _SLOT_DIRECTIVE_RE.match(line)
        if match is None:
            continue

        level = int(match.group("level"))
        field = match.group("field").upper()
        field_key = (level, field)
        if field_key in seen_fields:
            warnings.append(f"duplicate L{level}{field}; last value wins")
        seen_fields.add(field_key)
        values[level][field.lower()] = _parse_value(match.group("value"))

    return ParsedSetDef(
        slots=tuple(
            SetDefSlot(
                level=level, label=values[level].get("label"), file=values[level].get("file")
            )
            for level in range(1, 6)
        ),
        warnings=tuple(warnings),
    )


def _parse_value(value: str) -> str | None:
    value = _strip_semicolon_comment(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value or None


def _strip_semicolon_comment(value: str) -> str:
    quote: str | None = None
    for index, character in enumerate(value):
        if quote is not None:
            if character == quote:
                quote = None
        elif character in {'"', "'"}:
            quote = character
        elif character == ";":
            return value[:index]
    return value
