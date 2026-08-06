from __future__ import annotations

import codecs
import re
from typing import Literal

DtxTextKind = Literal["dtx", "set_def"]

_BOM_ENCODINGS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16le"),
    (codecs.BOM_UTF16_BE, "utf-16be"),
)
_FALLBACK_ENCODINGS = ("utf-8", "cp932", "shift-jis", "utf-16le", "utf-16be")
_DTX_DIRECTIVE_RE = re.compile(
    r"^[#*]\s*(?:\d{3}[0-9A-Za-z]{2}\s*:|[A-Za-z][A-Za-z0-9_]*(?=\s|:|;|$))",
    re.MULTILINE,
)
_SET_DEF_DIRECTIVE_RE = re.compile(
    r"^[#*]\s*L[1-5](?:LABEL|FILE)(?=\s|:|;|$)", re.IGNORECASE | re.MULTILINE
)


def decode_dtxmania_text(
    raw: bytes,
    *,
    source_name: str,
    kind: DtxTextKind,
) -> str:
    """Decode a DTXMania text object only when it contains a valid directive."""
    predicate = _acceptance_predicate(kind, source_name)
    encodings = [encoding for bom, encoding in _BOM_ENCODINGS if raw.startswith(bom)] + list(
        _FALLBACK_ENCODINGS
    )

    for encoding in encodings:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        text = text.removeprefix("\ufeff")
        if predicate.search(text):
            return text

    raise ValueError(f"could not decode DTXMania {kind} text: {source_name}")


def _acceptance_predicate(kind: DtxTextKind, source_name: str) -> re.Pattern[str]:
    if kind == "dtx":
        return _DTX_DIRECTIVE_RE
    if kind == "set_def":
        return _SET_DEF_DIRECTIVE_RE
    raise ValueError(f"could not decode DTXMania {kind} text: {source_name}")
