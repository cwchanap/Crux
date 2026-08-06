import codecs

import pytest

from src.benchmark.dtx_text import decode_dtxmania_text

DTX_TEXT = "#TITLE: Decoded Title\n#BPM: 120\n#00011: 0100\n"


@pytest.mark.parametrize(
    ("encoding_name", "raw"),
    [
        ("utf8-bom", codecs.BOM_UTF8 + DTX_TEXT.encode("utf-8")),
        ("utf16le-bom", codecs.BOM_UTF16_LE + DTX_TEXT.encode("utf-16le")),
        ("utf16be-bom", codecs.BOM_UTF16_BE + DTX_TEXT.encode("utf-16be")),
        ("utf16le", DTX_TEXT.encode("utf-16le")),
        ("utf16be", DTX_TEXT.encode("utf-16be")),
        ("utf8", DTX_TEXT.encode("utf-8")),
    ],
)
def test_decode_dtxmania_text_supports_dtx_encodings(encoding_name: str, raw: bytes) -> None:
    assert decode_dtxmania_text(raw, source_name=f"{encoding_name}.dtx", kind="dtx") == DTX_TEXT


def test_decode_dtxmania_text_supports_existing_shift_jis_chart_fixture() -> None:
    text = "#TITLE: テスト\r\n#BPM: 120\r\n"

    assert (
        decode_dtxmania_text(text.encode("shift-jis"), source_name="sjis.dtx", kind="dtx") == text
    )


def test_decode_dtxmania_text_supports_cp932_only_title() -> None:
    text = "#TITLE: CP932 ①\n#BPM: 120\n"

    assert decode_dtxmania_text(text.encode("cp932"), source_name="cp932.dtx", kind="dtx") == text


@pytest.mark.parametrize(
    "text",
    [
        "#TITLE: Header Directive\n",
        "*BPM: 180\n",
        "#00011: 0100\n",
    ],
)
def test_decode_dtxmania_text_accepts_dtx_directives(text: str) -> None:
    assert decode_dtxmania_text(text.encode(), source_name="directive.dtx", kind="dtx") == text


@pytest.mark.parametrize(
    "text",
    [
        "#L1LABEL: Beginner\n",
        "*L5FILE: real.dtx\n",
    ],
)
def test_decode_dtxmania_text_accepts_authored_set_def_directives(text: str) -> None:
    assert decode_dtxmania_text(text.encode(), source_name="set.def", kind="set_def") == text


@pytest.mark.parametrize(
    "text",
    [
        "#L0LABEL: Invalid\n",
        "#L6FILE: Invalid.dtx\n",
        "#TITLE: Not a slot\n",
        "#00011: 0100\n",
    ],
)
def test_decode_dtxmania_text_rejects_non_slot_set_def_directives(text: str) -> None:
    with pytest.raises(ValueError, match="set.def"):
        decode_dtxmania_text(text.encode(), source_name="set.def", kind="set_def")


@pytest.mark.parametrize(
    "raw",
    [
        b"this is valid utf-8 but not DTXMania text",
        b"\x00\x01\x02\x03",
    ],
)
def test_decode_dtxmania_text_rejects_gibberish_without_leaking_source_bytes(raw: bytes) -> None:
    source_name = "nested/garbage.dtx"

    with pytest.raises(ValueError) as error:
        decode_dtxmania_text(raw, source_name=source_name, kind="dtx")

    message = str(error.value)
    assert source_name in message
    assert raw.decode("latin-1") not in message
    assert repr(raw) not in message
