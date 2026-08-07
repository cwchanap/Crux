from importlib import import_module
from types import ModuleType

import pytest


def _load_parser() -> ModuleType:
    try:
        return import_module("src.benchmark.set_def_parser")
    except ModuleNotFoundError:
        pytest.fail("the requested src.benchmark.set_def_parser module is absent")


def _slot_values(parsed: object) -> list[tuple[int, str | None, str | None]]:
    return [(slot.level, slot.label, slot.file) for slot in parsed.slots]  # type: ignore[attr-defined]


def test_parse_set_def_text_accepts_markers_separators_spaces_case_and_quotes() -> None:
    parser = _load_parser()

    parsed = parser.parse_set_def_text(
        "\n".join(
            [
                '\ufeff# L1LABEL : "Beginner"',
                "*l1file 'custom chart name.dtx'",
                "#L2LABEL Intermediate",
                "* L2FILE :  advanced.TXT",
            ]
        )
    )

    assert isinstance(parsed, parser.ParsedSetDef)
    assert _slot_values(parsed) == [
        (1, "Beginner", "custom chart name.dtx"),
        (2, "Intermediate", "advanced.TXT"),
        (3, None, None),
        (4, None, None),
        (5, None, None),
    ]
    assert parsed.warnings == ()


def test_parse_set_def_text_ignores_comments_and_unrelated_directives() -> None:
    parser = _load_parser()

    parsed = parser.parse_set_def_text(
        "\n".join(
            [
                "; #L1LABEL: commented out",
                "#TITLE: ignored metadata",
                '#L3LABEL: "Expert; Edition" ; author comment',
                "#L3FILE: finale.dtx ; chart comment",
                "*L4LABEL: Ignored? no, this is a slot",
            ]
        )
    )

    assert _slot_values(parsed) == [
        (1, None, None),
        (2, None, None),
        (3, "Expert; Edition", "finale.dtx"),
        (4, "Ignored? no, this is a slot", None),
        (5, None, None),
    ]
    assert parsed.warnings == ()


def test_parse_set_def_text_uses_last_duplicate_field_and_warns() -> None:
    parser = _load_parser()

    parsed = parser.parse_set_def_text(
        "\n".join(
            [
                "#L2LABEL: Initial",
                "*l2label: Final",
                "#L2FILE: original.dtx",
                "* L2FILE : selected.txt",
            ]
        )
    )

    assert _slot_values(parsed) == [
        (1, None, None),
        (2, "Final", "selected.txt"),
        (3, None, None),
        (4, None, None),
        (5, None, None),
    ]
    assert parsed.warnings == (
        "duplicate L2LABEL; last value wins",
        "duplicate L2FILE; last value wins",
    )


def test_parse_set_def_text_returns_five_ordered_slots_for_empty_values() -> None:
    parser = _load_parser()

    parsed = parser.parse_set_def_text("#L1LABEL:\n* L3FILE: \n")

    assert _slot_values(parsed) == [
        (1, None, None),
        (2, None, None),
        (3, None, None),
        (4, None, None),
        (5, None, None),
    ]
    assert parsed.warnings == ()


def test_parse_set_def_text_preserves_custom_dtx_and_txt_file_values() -> None:
    parser = _load_parser()

    parsed = parser.parse_set_def_text(
        "\n".join(
            [
                "#L4FILE: alternate-chart.DTX",
                "#L5FILE: score-export.txt",
            ]
        )
    )

    assert _slot_values(parsed) == [
        (1, None, None),
        (2, None, None),
        (3, None, None),
        (4, None, "alternate-chart.DTX"),
        (5, None, "score-export.txt"),
    ]


def test_parse_set_def_bytes_decodes_cp932_labels() -> None:
    parser = _load_parser()
    raw = "#L4LABEL: ①中級\n#L4FILE: stage.dtx\n".encode("cp932")

    parsed = parser.parse_set_def_bytes(raw, source_name="nested/SET.DEF")

    assert _slot_values(parsed) == [
        (1, None, None),
        (2, None, None),
        (3, None, None),
        (4, "①中級", "stage.dtx"),
        (5, None, None),
    ]


def test_parse_set_def_bytes_accepts_whitespace_after_marker() -> None:
    parser = _load_parser()
    raw = "# L5FILE: real.dtx\n* L5LABEL: Real\n".encode("utf-8")

    parsed = parser.parse_set_def_bytes(raw, source_name="nested/SET.DEF")

    assert _slot_values(parsed) == [
        (1, None, None),
        (2, None, None),
        (3, None, None),
        (4, None, None),
        (5, "Real", "real.dtx"),
    ]
