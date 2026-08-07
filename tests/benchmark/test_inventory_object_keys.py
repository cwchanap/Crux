from datetime import datetime, timezone

import pytest

from src.benchmark.inventory_object_keys import (
    ResolvedObjectKey,
    resolve_inventory_object_key,
)
from src.benchmark.r2_corpus_models import RemoteObject

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def remote(key: str) -> RemoteObject:
    return RemoteObject(
        key=key,
        size=1,
        etag="etag",
        etag_is_weak=False,
        last_modified=FIXED_TIME,
        content_type="text/plain",
    )


def test_resolve_inventory_object_key_normalizes_backslashes() -> None:
    target = remote("42/Charts/REAL.DTX")

    result = resolve_inventory_object_key(
        r"Charts\REAL.DTX",
        base_object_key_dir="42",
        object_prefix="42/",
        objects=(target,),
    )

    assert result == ResolvedObjectKey("exact", "42/Charts/REAL.DTX", target)


def test_resolve_inventory_object_key_normalizes_dot_and_parent_inside_prefix() -> None:
    target = remote("42/Charts/REAL.DTX")

    result = resolve_inventory_object_key(
        r".\..\Charts\REAL.DTX",
        base_object_key_dir="42/config",
        object_prefix="42/",
        objects=(target,),
    )

    assert result == ResolvedObjectKey("exact", "42/Charts/REAL.DTX", target)


def test_resolve_inventory_object_key_resolves_dot_to_the_base_directory() -> None:
    target = remote("42/config")

    result = resolve_inventory_object_key(
        ".",
        base_object_key_dir="42/config",
        object_prefix="42/",
        objects=(target,),
    )

    assert result == ResolvedObjectKey("exact", "42/config", target)


def test_resolve_inventory_object_key_prefers_an_exact_key_over_casefold_matches() -> None:
    exact = remote("42/Charts/REAL.DTX")
    casefold_match = remote("42/charts/real.dtx")

    result = resolve_inventory_object_key(
        "Charts/REAL.DTX",
        base_object_key_dir="42",
        object_prefix="42/",
        objects=(casefold_match, exact),
    )

    assert result == ResolvedObjectKey("exact", "42/Charts/REAL.DTX", exact)


def test_resolve_inventory_object_key_returns_a_unique_casefold_match() -> None:
    target = remote("42/charts/real.dtx")

    result = resolve_inventory_object_key(
        r"..\Charts\REAL.DTX",
        base_object_key_dir="42/config",
        object_prefix="42/",
        objects=(target,),
    )

    assert result == ResolvedObjectKey("casefold", "42/Charts/REAL.DTX", target)


def test_resolve_inventory_object_key_reports_a_missing_key() -> None:
    result = resolve_inventory_object_key(
        "Charts/MISSING.DTX",
        base_object_key_dir="42",
        object_prefix="42/",
        objects=(remote("42/Charts/REAL.DTX"),),
    )

    assert result == ResolvedObjectKey("missing", "42/Charts/MISSING.DTX", None)


def test_resolve_inventory_object_key_reports_duplicate_casefold_matches() -> None:
    first = remote("42/CHARTS/REAL.DTX")
    second = remote("42/charts/real.dtx")

    result = resolve_inventory_object_key(
        "Charts/Real.Dtx",
        base_object_key_dir="42",
        object_prefix="42/",
        objects=(first, second),
    )

    assert result == ResolvedObjectKey("ambiguous", "42/Charts/Real.Dtx", None)


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "\x00",
        "/absolute",
        "//unc/share",
        r"C:\drive",
        "../../escape",
    ],
)
def test_resolve_inventory_object_key_rejects_invalid_or_escaping_paths(relative_path: str) -> None:
    result = resolve_inventory_object_key(
        relative_path,
        base_object_key_dir="42/config",
        object_prefix="42/",
        objects=(remote("42/Charts/REAL.DTX"),),
    )

    assert result == ResolvedObjectKey("invalid_path", None, None)


def test_resolve_inventory_object_key_never_matches_a_sibling_prefix() -> None:
    result = resolve_inventory_object_key(
        "Charts/REAL.DTX",
        base_object_key_dir="42",
        object_prefix="42/",
        objects=(remote("420/Charts/REAL.DTX"),),
    )

    assert result == ResolvedObjectKey("missing", "42/Charts/REAL.DTX", None)


def test_resolve_inventory_object_key_rejects_object_prefix_without_trailing_slash() -> None:
    result = resolve_inventory_object_key(
        "REAL.DTX",
        base_object_key_dir="42",
        object_prefix="42",
        objects=(remote("42/REAL.DTX"),),
    )

    assert result == ResolvedObjectKey("invalid_path", None, None)


def test_resolve_inventory_object_key_rejects_base_dir_starting_above_prefix() -> None:
    result = resolve_inventory_object_key(
        "REAL.DTX",
        base_object_key_dir="../42",
        object_prefix="42/",
        objects=(remote("42/REAL.DTX"),),
    )

    assert result == ResolvedObjectKey("invalid_path", None, None)


def test_resolve_inventory_object_key_normalizes_dot_in_object_prefix() -> None:
    """A prefix like './42/' contains a '.' component that pathlib preserves
    at the start.  The resolver normalizes it away (line 89)."""
    result = resolve_inventory_object_key(
        "REAL.DTX",
        base_object_key_dir="42",
        object_prefix="./42/",
        objects=(remote("42/REAL.DTX"),),
    )

    assert result.status == "exact"
    assert result.normalized_key == "42/REAL.DTX"


def test_resolve_inventory_object_key_normalizes_dot_dot_in_object_prefix() -> None:
    """A prefix like '42/sub/../' contains a '..' that pathlib preserves.
    The resolver pops the preceding component (line 93) and normalizes
    the prefix to '42/'."""
    result = resolve_inventory_object_key(
        "REAL.DTX",
        base_object_key_dir="42",
        object_prefix="42/sub/../",
        objects=(remote("42/REAL.DTX"),),
    )

    assert result.status == "exact"
    assert result.normalized_key == "42/REAL.DTX"


def test_resolve_inventory_object_key_normalizes_dot_in_base_dir() -> None:
    """A base dir like './42' contains a '.' component that pathlib preserves
    at the start.  The resolver normalizes it away (line 106)."""
    result = resolve_inventory_object_key(
        "REAL.DTX",
        base_object_key_dir="./42",
        object_prefix="42/",
        objects=(remote("42/REAL.DTX"),),
    )

    assert result.status == "exact"
    assert result.normalized_key == "42/REAL.DTX"
