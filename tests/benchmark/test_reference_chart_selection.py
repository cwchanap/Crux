from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

import src.benchmark.corpus_cache as corpus_cache
from src.benchmark.chart_names import CHART_FILENAME_PRIORITY
from src.benchmark.corpus_manifest import ManifestRowView
from src.benchmark.r2_corpus_models import (
    ProvenanceRecord,
    RemoteObject,
    SimfileInventory,
    SyncError,
)
from src.benchmark.reference_chart_selection import (
    ChartSelection,
    LoadedOverrides,
    SelectionOverride,
    load_selection_overrides,
    select_reference_chart,
)

_FIXED_TIME = datetime(2026, 8, 5, tzinfo=timezone.utc)
_SOURCE_ENDPOINT_SHA256 = "a" * 64
_BUCKET = "simfile-dtx"
_OVERRIDE_SCHEMA = "crux.reference-chart-overrides/v1"
_EMPTY_OVERRIDE_DOCUMENT = (
    b'{"overrides":{},"schema_version":"crux.reference-chart-overrides/v1"}\n'
)


def _chart_body(*, dlevel: str, note_evidence: bool = True) -> bytes:
    lines = [
        "#TITLE: Shared Rank",
        "#ARTIST: Test Artist",
        f"#DLEVEL: {dlevel}",
    ]
    if note_evidence:
        lines.append("#00011: 01")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _remote(key: str, body: bytes) -> RemoteObject:
    digest = sha256(body).hexdigest()
    return RemoteObject(
        key=key,
        size=len(body),
        etag=f"etag-{key}",
        etag_is_weak=False,
        last_modified=_FIXED_TIME,
        content_type="text/plain",
        cache_status="verified",
        sha256=digest,
        cache_path=f"sha256/{digest[:2]}/{digest}",
    )


def _row(
    objects: tuple[RemoteObject, ...],
    *,
    sync_status: str = "complete",
    sync_errors: tuple[SyncError, ...] = (),
) -> ManifestRowView:
    return ManifestRowView(
        inventory=SimfileInventory(
            simfile_id=42,
            object_prefix="42/",
            objects=objects,
            sync_status=sync_status,
            sync_errors=sync_errors,
        ),
        provenance=ProvenanceRecord(),
        corpus_version="sha256:" + "b" * 64,
        cache_profile="setdef_dtx_txt_v1",
        source_endpoint_sha256=_SOURCE_ENDPOINT_SHA256,
        source_bucket=_BUCKET,
        source_discovery_method="r2_list_objects_v2",
    )


def _install_cached_bodies(
    cache_dir: Path,
    fixtures: tuple[tuple[RemoteObject, bytes | None], ...],
) -> None:
    for remote, body in fixtures:
        if body is None:
            continue
        assert remote.sha256 is not None
        cache_path = cache_dir / "sha256" / remote.sha256[:2] / remote.sha256
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(body)


def _empty_overrides() -> LoadedOverrides:
    return LoadedOverrides(
        document_sha256=sha256(_EMPTY_OVERRIDE_DOCUMENT).hexdigest(),
        by_simfile_id={},
    )


def _select(
    tmp_path: Path,
    fixtures: tuple[tuple[RemoteObject, bytes | None], ...],
    *,
    sync_status: str = "complete",
    sync_errors: tuple[SyncError, ...] = (),
    overrides: LoadedOverrides | None = None,
):
    cache_dir = tmp_path / "cache"
    _install_cached_bodies(cache_dir, fixtures)
    return select_reference_chart(
        _row(
            tuple(remote for remote, _ in fixtures),
            sync_status=sync_status,
            sync_errors=sync_errors,
        ),
        cache_dir=cache_dir,
        overrides=overrides or _empty_overrides(),
    )


def _ranked_chart_fixtures() -> tuple[tuple[RemoteObject, bytes], ...]:
    fixtures: list[tuple[RemoteObject, bytes]] = []
    for name in CHART_FILENAME_PRIORITY:
        body = _chart_body(dlevel="50")
        fixtures.append((_remote(f"42/{name}.dtx", body), body))
    return tuple(fixtures)


def _set_def_body(file_name: str) -> bytes:
    return f"#L5FILE: {file_name}\n".encode("utf-8")


def _overrides_for(chart_key: str) -> LoadedOverrides:
    return LoadedOverrides(
        document_sha256=sha256(_EMPTY_OVERRIDE_DOCUMENT).hexdigest(),
        by_simfile_id={42: SelectionOverride(chart_key=chart_key, reason="manual audit")},
    )


def test_shared_rank_selects_real_for_equal_dlevel_fallback(tmp_path: Path) -> None:
    selection = _select(tmp_path, _ranked_chart_fixtures())

    assert selection.status == "selected"
    assert selection.method == "filename_tiebreak_fallback"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/real.dtx"


def test_authored_order_wins_over_shared_rank(tmp_path: Path) -> None:
    set_def_body = b"#L5FILE: bas.dtx\n"
    set_def = _remote("42/set.def", set_def_body)
    selection = _select(tmp_path, ((set_def, set_def_body), *_ranked_chart_fixtures()))

    assert selection.status == "selected"
    assert selection.method == "set_def_slot"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/bas.dtx"
    assert selection.selected_level_slot == "L5"


def test_load_selection_overrides_accepts_canonical_decimal_id_and_hashes_exact_bytes(
    tmp_path: Path,
) -> None:
    content = (
        b'{"overrides":{"42":{"chart_key":"42/real.dtx","reason":"manual audit"}},'
        b'"schema_version":"crux.reference-chart-overrides/v1"}\n'
    )
    path = tmp_path / "overrides.json"
    path.write_bytes(content)

    loaded = load_selection_overrides(path, missing_ok=False)

    assert loaded.document_sha256 == sha256(content).hexdigest()
    assert loaded.by_simfile_id == {
        42: SelectionOverride(chart_key="42/real.dtx", reason="manual audit")
    }


@pytest.mark.parametrize("raw_id", ["042", "+42", "42.0", "４２"])
def test_load_selection_overrides_rejects_noncanonical_decimal_ids(
    tmp_path: Path, raw_id: str
) -> None:
    content = (
        b'{"overrides":{"'
        + raw_id.encode("utf-8")
        + b'":{"chart_key":"42/real.dtx","reason":"manual audit"}},'
        + b'"schema_version":"crux.reference-chart-overrides/v1"}\n'
    )
    path = tmp_path / "overrides.json"
    path.write_bytes(content)

    with pytest.raises(ValueError):
        load_selection_overrides(path, missing_ok=False)


def test_load_selection_overrides_rejects_duplicate_ids_after_numeric_normalization(
    tmp_path: Path,
) -> None:
    content = (
        b'{"overrides":{"01":{"chart_key":"42/full.dtx","reason":"first"},'
        b'"1":{"chart_key":"42/real.dtx","reason":"second"}},'
        b'"schema_version":"crux.reference-chart-overrides/v1"}\n'
    )
    path = tmp_path / "overrides.json"
    path.write_bytes(content)

    with pytest.raises(ValueError, match="duplicate"):
        load_selection_overrides(path, missing_ok=False)


@pytest.mark.parametrize(
    "entry",
    [
        b'{"chart_key":"42/real.dtx"}',
        b'{"chart_key":"42/real.dtx","reason":"manual audit","unused":"no"}',
    ],
)
def test_load_selection_overrides_requires_exact_entry_keys(tmp_path: Path, entry: bytes) -> None:
    content = (
        b'{"overrides":{"42":'
        + entry
        + b'},"schema_version":"crux.reference-chart-overrides/v1"}\n'
    )
    path = tmp_path / "overrides.json"
    path.write_bytes(content)

    with pytest.raises(ValueError):
        load_selection_overrides(path, missing_ok=False)


@pytest.mark.parametrize("field", ["chart_key", "reason"])
def test_load_selection_overrides_requires_nonempty_entry_strings(
    tmp_path: Path, field: str
) -> None:
    chart_key = b'""' if field == "chart_key" else b'"42/real.dtx"'
    reason = b'""' if field == "reason" else b'"manual audit"'
    content = (
        b'{"overrides":{"42":{"chart_key":'
        + chart_key
        + b',"reason":'
        + reason
        + b'}},"schema_version":"crux.reference-chart-overrides/v1"}\n'
    )
    path = tmp_path / "overrides.json"
    path.write_bytes(content)

    with pytest.raises(ValueError):
        load_selection_overrides(path, missing_ok=False)


def test_load_selection_overrides_requires_canonical_json_bytes(tmp_path: Path) -> None:
    path = tmp_path / "overrides.json"
    path.write_bytes(b'{"schema_version":"crux.reference-chart-overrides/v1", "overrides":{}}\n')

    with pytest.raises(ValueError):
        load_selection_overrides(path, missing_ok=False)


def test_load_selection_overrides_requires_one_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "overrides.json"
    path.write_bytes(_EMPTY_OVERRIDE_DOCUMENT.rstrip(b"\n"))

    with pytest.raises(ValueError):
        load_selection_overrides(path, missing_ok=False)


def test_load_selection_overrides_uses_canonical_empty_document_when_missing_is_allowed() -> None:
    loaded = load_selection_overrides(None, missing_ok=True)

    assert loaded.document_sha256 == sha256(_EMPTY_OVERRIDE_DOCUMENT).hexdigest()
    assert loaded.by_simfile_id == {}


def test_repository_default_override_document_is_canonical_empty_bytes() -> None:
    path = (
        Path(__file__).resolve().parents[2] / "config" / "benchmark-reference-chart-overrides.json"
    )

    assert path.read_bytes() == _EMPTY_OVERRIDE_DOCUMENT


@pytest.mark.parametrize("path", [None, Path("missing-overrides.json")])
def test_load_selection_overrides_rejects_missing_document_when_required(path: Path | None) -> None:
    with pytest.raises(ValueError):
        load_selection_overrides(path, missing_ok=False)


def test_source_inventory_empty_quarantines_without_selection(tmp_path: Path) -> None:
    selection = _select(tmp_path, (), sync_status="empty")

    assert selection.status == "quarantined"
    assert selection.reason_codes == ("source_inventory_unusable",)
    assert selection.selected_chart is None


def test_source_inventory_with_no_verified_objects_quarantines(tmp_path: Path) -> None:
    body = _chart_body(dlevel="50")
    failed_chart = replace(_remote("42/real.dtx", body), cache_status="failed")
    selection = _select(tmp_path, ((failed_chart, body),))

    assert selection.status == "quarantined"
    assert selection.reason_codes == ("source_inventory_unusable",)


@pytest.mark.parametrize("sync_status", ["partial", "failed"])
def test_source_inventory_noncomplete_status_warns_but_keeps_verified_authored_chart(
    tmp_path: Path,
    sync_status: str,
) -> None:
    set_def_body = _set_def_body("author.dtx")
    chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", set_def_body), set_def_body),
            (_remote("42/author.dtx", chart_body), chart_body),
        ),
        sync_status=sync_status,
    )

    assert selection.status == "selected"
    assert selection.warnings == ("partial_inventory",)
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/author.dtx"


def test_source_inventory_sync_errors_warn_but_do_not_block_verified_authored_chart(
    tmp_path: Path,
) -> None:
    set_def_body = _set_def_body("author.dtx")
    chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", set_def_body), set_def_body),
            (_remote("42/author.dtx", chart_body), chart_body),
        ),
        sync_errors=(SyncError("object", "object_get_failed", "unrelated fetch failed"),),
    )

    assert selection.status == "selected"
    assert selection.warnings == ("partial_inventory",)
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/author.dtx"


def test_set_def_discovery_prefers_root_copy_over_nested_copy(tmp_path: Path) -> None:
    root_set_def_body = _set_def_body("root.dtx")
    nested_set_def_body = _set_def_body("nested.dtx")
    root_chart_body = _chart_body(dlevel="50")
    nested_chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", root_set_def_body), root_set_def_body),
            (_remote("42/meta/SET.DEF", nested_set_def_body), nested_set_def_body),
            (_remote("42/root.dtx", root_chart_body), root_chart_body),
            (_remote("42/meta/nested.dtx", nested_chart_body), nested_chart_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.set_def is not None
    assert selection.set_def.key == "42/set.def"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/root.dtx"


def test_set_def_discovery_uses_unique_nested_copy(tmp_path: Path) -> None:
    set_def_body = _set_def_body("author.dtx")
    chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/meta/SET.DEF", set_def_body), set_def_body),
            (_remote("42/meta/author.dtx", chart_body), chart_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.set_def is not None
    assert selection.set_def.key == "42/meta/SET.DEF"


def test_set_def_discovery_prefers_unique_exact_lowercase_root_name(tmp_path: Path) -> None:
    lower_set_def_body = _set_def_body("lower.dtx")
    upper_set_def_body = _set_def_body("upper.dtx")
    lower_chart_body = _chart_body(dlevel="50")
    upper_chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", lower_set_def_body), lower_set_def_body),
            (_remote("42/SET.DEF", upper_set_def_body), upper_set_def_body),
            (_remote("42/lower.dtx", lower_chart_body), lower_chart_body),
            (_remote("42/upper.dtx", upper_chart_body), upper_chart_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.set_def is not None
    assert selection.set_def.key == "42/set.def"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/lower.dtx"


def test_set_def_discovery_quarantines_same_depth_nested_copies(tmp_path: Path) -> None:
    left_body = _set_def_body("left.dtx")
    right_body = _set_def_body("right.dtx")
    selection = _select(
        tmp_path,
        (
            (_remote("42/left/set.def", left_body), left_body),
            (_remote("42/right/SET.DEF", right_body), right_body),
        ),
    )

    assert selection.status == "quarantined"
    assert selection.reason_codes == ("ambiguous_set_def",)


@pytest.mark.parametrize(
    ("root_body", "expected_reason"),
    [
        (None, "cached_body_unavailable"),
        (b"\xff", "invalid_set_def"),
    ],
)
def test_set_def_discovery_quarantines_unusable_canonical_copy_without_fallback(
    tmp_path: Path,
    root_body: bytes | None,
    expected_reason: str,
) -> None:
    canonical_content = b"#L5FILE: root.dtx\n" if root_body is None else root_body
    nested_set_def_body = _set_def_body("nested.dtx")
    nested_chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", canonical_content), root_body),
            (_remote("42/meta/set.def", nested_set_def_body), nested_set_def_body),
            (_remote("42/meta/nested.dtx", nested_chart_body), nested_chart_body),
        ),
    )

    assert selection.status == "quarantined"
    assert selection.reason_codes == (expected_reason,)
    assert selection.selected_chart is None


def test_authored_slots_prefer_l5_over_l4(tmp_path: Path) -> None:
    set_def_body = b"#L5FILE: l5.dtx\n#L4FILE: l4.dtx\n"
    l5_body = _chart_body(dlevel="20")
    l4_body = _chart_body(dlevel="99")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", set_def_body), set_def_body),
            (_remote("42/l5.dtx", l5_body), l5_body),
            (_remote("42/l4.dtx", l4_body), l4_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.selected_level_slot == "L5"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/l5.dtx"


@pytest.mark.parametrize("file_name", ["custom-name.dtx", "custom-name.txt"])
def test_authored_slots_accept_custom_dtx_and_txt_names(tmp_path: Path, file_name: str) -> None:
    set_def_body = _set_def_body(file_name)
    chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", set_def_body), set_def_body),
            (_remote(f"42/{file_name}", chart_body), chart_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == f"42/{file_name}"


def test_authored_slots_resolve_nested_relative_path(tmp_path: Path) -> None:
    set_def_body = _set_def_body("charts/lead.dtx")
    chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/meta/SET.DEF", set_def_body), set_def_body),
            (_remote("42/meta/charts/lead.dtx", chart_body), chart_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/meta/charts/lead.dtx"


def test_authored_slots_prefer_exact_object_key_over_casefold_match(tmp_path: Path) -> None:
    set_def_body = _set_def_body("stage.dtx")
    exact_body = _chart_body(dlevel="50")
    other_body = _chart_body(dlevel="60")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", set_def_body), set_def_body),
            (_remote("42/stage.dtx", exact_body), exact_body),
            (_remote("42/Stage.dtx", other_body), other_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/stage.dtx"


def test_authored_slots_accept_unique_casefold_object_key(tmp_path: Path) -> None:
    set_def_body = _set_def_body("STAGE.DTX")
    chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", set_def_body), set_def_body),
            (_remote("42/stage.dtx", chart_body), chart_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/stage.dtx"


def test_authored_slots_allow_contained_parent_path(tmp_path: Path) -> None:
    set_def_body = _set_def_body("../root.dtx")
    chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/meta/set.def", set_def_body), set_def_body),
            (_remote("42/root.dtx", chart_body), chart_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/root.dtx"


def test_authored_slots_retry_simfile_root_after_nested_relative_miss(tmp_path: Path) -> None:
    set_def_body = _set_def_body("root.dtx")
    chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/meta/set.def", set_def_body), set_def_body),
            (_remote("42/root.dtx", chart_body), chart_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.warnings == ("set_def_root_fallback",)
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/root.dtx"


def test_authored_slots_continue_to_l4_when_l5_is_missing(tmp_path: Path) -> None:
    set_def_body = b"#L5FILE: absent.dtx\n#L4FILE: l4.dtx\n"
    chart_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", set_def_body), set_def_body),
            (_remote("42/l4.dtx", chart_body), chart_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.selected_level_slot == "L4"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/l4.dtx"


@pytest.mark.parametrize(
    ("set_def_body", "remotes", "expected_reason"),
    [
        (
            _set_def_body("../../escape.dtx"),
            (),
            "invalid_chart_reference",
        ),
        (
            _set_def_body("stage.dtx"),
            (
                ("42/Stage.dtx", _chart_body(dlevel="50")),
                ("42/STAGE.dtx", _chart_body(dlevel="50")),
            ),
            "ambiguous_chart_key",
        ),
    ],
)
def test_authored_slots_quarantine_invalid_or_ambiguous_references(
    tmp_path: Path,
    set_def_body: bytes,
    remotes: tuple[tuple[str, bytes], ...],
    expected_reason: str,
) -> None:
    fixtures: list[tuple[RemoteObject, bytes]] = [
        (_remote("42/set.def", set_def_body), set_def_body)
    ]
    fixtures.extend((_remote(key, body), body) for key, body in remotes)
    selection = _select(tmp_path, tuple(fixtures))

    assert selection.status == "quarantined"
    assert selection.reason_codes == (expected_reason,)


def test_authored_slots_do_not_downgrade_after_existing_l5_chart_parse_failure(
    tmp_path: Path,
) -> None:
    set_def_body = b"#L5FILE: broken.dtx\n#L4FILE: working.dtx\n"
    working_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", set_def_body), set_def_body),
            (_remote("42/broken.dtx", b"\xff"), b"\xff"),
            (_remote("42/working.dtx", working_body), working_body),
        ),
    )

    assert selection.status == "quarantined"
    assert selection.reason_codes == ("selected_chart_parse_failed",)
    assert selection.selected_chart is None


@pytest.mark.parametrize("replacement_target", ["set_def", "chart"])
def test_selection_consumes_the_same_cache_descriptor_that_was_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_target: str,
) -> None:
    set_def_body = _set_def_body("authored.dtx")
    chart_body = _chart_body(dlevel="50")
    set_def = _remote("42/set.def", set_def_body)
    chart = _remote("42/authored.dtx", chart_body)
    target = set_def if replacement_target == "set_def" else chart
    assert target.sha256 is not None
    cache_path = tmp_path / "cache" / "sha256" / target.sha256[:2] / target.sha256
    replacement_path = tmp_path / f"replacement-{replacement_target}"
    replacement_path.write_bytes(b"not DTXMania text")
    original_verify = corpus_cache._verify_regular_file_binding
    replaced = False

    def replace_after_descriptor_verification(parent_fd: int, name: str, descriptor: int) -> None:
        nonlocal replaced
        original_verify(parent_fd, name, descriptor)
        if not replaced and name == target.sha256:
            os.replace(replacement_path, cache_path)
            replaced = True

    monkeypatch.setattr(
        corpus_cache,
        "_verify_regular_file_binding",
        replace_after_descriptor_verification,
    )

    selection = _select(
        tmp_path,
        (
            (set_def, set_def_body),
            (chart, chart_body),
        ),
    )

    assert replaced
    assert selection.status == "selected"
    assert selection.method == "set_def_slot"
    assert selection.selected_chart == chart


def test_override_application_precedes_authored_set_def(tmp_path: Path) -> None:
    set_def_body = _set_def_body("authored.dtx")
    authored_body = _chart_body(dlevel="50")
    override_body = _chart_body(dlevel="10")
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", set_def_body), set_def_body),
            (_remote("42/authored.dtx", authored_body), authored_body),
            (_remote("42/override.dtx", override_body), override_body),
        ),
        overrides=_overrides_for("42/override.dtx"),
    )

    assert selection.status == "selected"
    assert selection.method == "override"
    assert selection.set_def is None
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/override.dtx"


@pytest.mark.parametrize(
    ("override_key", "remote_key", "cache_status", "expected_body"),
    [
        ("42/STAGE.dtx", "42/stage.dtx", "verified", _chart_body(dlevel="50")),
        ("42/failed.dtx", "42/failed.dtx", "failed", _chart_body(dlevel="50")),
        ("42/audio.mp3", "42/audio.mp3", "verified", b"audio"),
        ("42/broken.dtx", "42/broken.dtx", "verified", b"\xff"),
    ],
)
def test_override_application_quarantines_invalid_row_entries_without_fallback(
    tmp_path: Path,
    override_key: str,
    remote_key: str,
    cache_status: str,
    expected_body: bytes,
) -> None:
    authored_set_def_body = _set_def_body("authored.dtx")
    authored_body = _chart_body(dlevel="50")
    override_remote = replace(
        _remote(remote_key, expected_body),
        cache_status=cache_status,
    )
    selection = _select(
        tmp_path,
        (
            (_remote("42/set.def", authored_set_def_body), authored_set_def_body),
            (_remote("42/authored.dtx", authored_body), authored_body),
            (override_remote, expected_body),
        ),
        overrides=_overrides_for(override_key),
    )

    assert selection.status == "quarantined"
    assert selection.reason_codes == ("override_invalid",)
    assert selection.override == SelectionOverride(chart_key=override_key, reason="manual audit")
    assert selection.selected_chart is None


@pytest.mark.parametrize(
    "body",
    [
        b"#TITLE: Header only\n#ARTIST: Artist\n",
        b"#BPM: 120\n",
        b"#00002: 1.000\n",
        b"#00001: 01\n",
    ],
)
def test_fallback_rejects_files_without_noncontrol_note_evidence(
    tmp_path: Path,
    body: bytes,
) -> None:
    selection = _select(tmp_path, ((_remote("42/candidate.txt", body), body),))

    assert selection.status == "quarantined"
    assert selection.reason_codes == ("no_verified_chart",)


def test_fallback_considers_only_verified_chart_objects(tmp_path: Path) -> None:
    chart_body = _chart_body(dlevel="50")
    unverified_chart = replace(_remote("42/real.dtx", chart_body), cache_status="failed")
    nonchart_body = b"not a chart"
    selection = _select(
        tmp_path,
        (
            (unverified_chart, chart_body),
            (_remote("42/readme.md", nonchart_body), nonchart_body),
        ),
    )

    assert selection.status == "quarantined"
    assert selection.reason_codes == ("no_verified_chart",)


@pytest.mark.parametrize("invalid_first", [True, False], ids=["invalid-first", "invalid-last"])
def test_fallback_skips_verified_nonchart_txt_regardless_of_input_order(
    tmp_path: Path,
    invalid_first: bool,
) -> None:
    readme_body = b"This package contains one authored chart.\n"
    chart_body = _chart_body(dlevel="50")
    readme = (_remote("42/readme.txt", readme_body), readme_body)
    chart = (_remote("42/real.dtx", chart_body), chart_body)
    fixtures = (readme, chart) if invalid_first else (chart, readme)

    selection = _select(tmp_path, fixtures)

    assert selection.status == "selected"
    assert selection.method == "single_candidate_fallback"
    assert selection.selected_chart == chart[0]


def test_fallback_retains_cache_unavailability_quarantine(tmp_path: Path) -> None:
    chart_body = _chart_body(dlevel="50")
    chart = _remote("42/real.dtx", chart_body)

    selection = _select(tmp_path, ((chart, None),))

    assert selection.status == "quarantined"
    assert selection.reason_codes == ("cached_body_unavailable",)


def test_fallback_selects_one_evidence_bearing_candidate(tmp_path: Path) -> None:
    chart_body = _chart_body(dlevel="50")
    selection = _select(tmp_path, ((_remote("42/any-name.dtx", chart_body), chart_body),))

    assert selection.status == "selected"
    assert selection.method == "single_candidate_fallback"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/any-name.dtx"
    assert selection.title == "Shared Rank"
    assert selection.artist == "Test Artist"


def test_fallback_selects_unique_highest_numeric_dlevel(tmp_path: Path) -> None:
    lower_body = _chart_body(dlevel="50")
    higher_body = _chart_body(dlevel="60")
    selection = _select(
        tmp_path,
        (
            (_remote("42/real.dtx", lower_body), lower_body),
            (_remote("42/custom.dtx", higher_body), higher_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.method == "dlevel_fallback"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/custom.dtx"


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (("real", "full", "mas", "ext", "adv", "bas"), "real"),
        (("full", "mas", "ext", "adv", "bas"), "full"),
        (("mas", "ext", "adv", "bas"), "mas"),
        (("ext", "adv", "bas"), "ext"),
        (("adv", "bas"), "adv"),
        (("bas", "unrecognized"), "bas"),
    ],
)
def test_fallback_uses_shared_filename_rank_for_equal_highest_dlevel(
    tmp_path: Path,
    available: tuple[str, ...],
    expected: str,
) -> None:
    fixtures: list[tuple[RemoteObject, bytes]] = []
    for name in available:
        body = _chart_body(dlevel="50")
        fixtures.append((_remote(f"42/{name}.dtx", body), body))
    selection = _select(tmp_path, tuple(fixtures))

    assert selection.status == "selected"
    assert selection.method == "filename_tiebreak_fallback"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == f"42/{expected}.dtx"


def test_fallback_higher_dlevel_outranks_better_filename_rank(tmp_path: Path) -> None:
    real_body = _chart_body(dlevel="50")
    bas_body = _chart_body(dlevel="60")
    selection = _select(
        tmp_path,
        (
            (_remote("42/real.dtx", real_body), real_body),
            (_remote("42/bas.dtx", bas_body), bas_body),
        ),
    )

    assert selection.status == "selected"
    assert selection.method == "dlevel_fallback"
    assert selection.selected_chart is not None
    assert selection.selected_chart.key == "42/bas.dtx"


def test_fallback_quarantines_duplicate_recognized_basenames(tmp_path: Path) -> None:
    left_body = _chart_body(dlevel="50")
    right_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/left/mas.dtx", left_body), left_body),
            (_remote("42/right/MAS.txt", right_body), right_body),
        ),
    )

    assert selection.status == "quarantined"
    assert selection.reason_codes == ("ambiguous_fallback",)


def test_fallback_never_uses_alphabetical_order_for_unrecognized_tie(tmp_path: Path) -> None:
    alpha_body = _chart_body(dlevel="50")
    zulu_body = _chart_body(dlevel="50")
    selection = _select(
        tmp_path,
        (
            (_remote("42/alpha.dtx", alpha_body), alpha_body),
            (_remote("42/zulu.dtx", zulu_body), zulu_body),
        ),
    )

    assert selection.status == "quarantined"
    assert selection.reason_codes == ("ambiguous_fallback",)


@pytest.mark.parametrize(
    ("status", "method", "reason_codes", "selected_chart", "title", "artist"),
    [
        ("selected", None, (), None, "Title", "Artist"),
        ("selected", "override", ("override_invalid",), None, "Title", "Artist"),
        ("selected", "override", (), None, None, "Artist"),
        ("quarantined", "override", ("override_invalid",), None, None, None),
        ("quarantined", None, (), None, None, None),
    ],
)
def test_fallback_chart_selection_enforces_status_invariants(
    status: str,
    method: str | None,
    reason_codes: tuple[str, ...],
    selected_chart: RemoteObject | None,
    title: str | None,
    artist: str | None,
) -> None:
    with pytest.raises(ValueError):
        ChartSelection(
            status=status,  # type: ignore[arg-type]
            method=method,  # type: ignore[arg-type]
            reason_codes=reason_codes,  # type: ignore[arg-type]
            warnings=(),
            set_def=None,
            selected_chart=selected_chart,
            selected_level_slot=None,
            selected_level_label=None,
            dlevel_raw=None,
            dlevel_normalized=None,
            title=title,
            artist=artist,
            override=None,
        )


@pytest.mark.parametrize(("title", "artist"), [(None, "Artist"), ("Title", None)])
def test_chart_selection_rejects_nonstring_selected_metadata(
    title: str | None,
    artist: str | None,
) -> None:
    body = _chart_body(dlevel="50")
    remote = _remote("42/chart.dtx", body)

    with pytest.raises(ValueError):
        ChartSelection(
            status="selected",
            method="override",
            reason_codes=(),
            warnings=(),
            set_def=None,
            selected_chart=remote,
            selected_level_slot=None,
            selected_level_label=None,
            dlevel_raw="50",
            dlevel_normalized=50,
            title=title,
            artist=artist,
            override=None,
        )
