from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest
from click.testing import CliRunner

import src.benchmark.reference_chart_manifest as reference_chart_manifest
from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.corpus_manifest import (
    ManifestPublicationError,
    build_manifest_rows,
    render_manifest,
)
from src.benchmark.r2_corpus_models import RemoteObject, SimfileInventory, SyncError
from src.cli import benchmark as benchmark_cli
from src.cli.main import main

_FIXED_TIME = datetime(2026, 8, 5, tzinfo=timezone.utc)
_ENDPOINT_SHA256 = "a" * 64
_BUCKET = "simfile-dtx"
_OVERRIDE_SCHEMA = "crux.reference-chart-overrides/v1"

runner = CliRunner()


def _chart_body(*, dlevel: int = 50, has_note_evidence: bool = True) -> bytes:
    lines = [
        "#TITLE: Offline Fixture",
        "#ARTIST: Acceptance Test",
        f"#DLEVEL: {dlevel}",
    ]
    if has_note_evidence:
        lines.append("#00011: 01")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _remote(simfile_id: int, relative_key: str, body: bytes) -> RemoteObject:
    digest = sha256(body).hexdigest()
    return RemoteObject(
        key=f"{simfile_id}/{relative_key}",
        size=len(body),
        etag=f"etag-{simfile_id}-{relative_key}",
        etag_is_weak=False,
        last_modified=_FIXED_TIME,
        content_type="text/plain",
        cache_status="verified",
        sha256=digest,
        cache_path=f"sha256/{digest[:2]}/{digest}",
    )


def _write_cached_bodies(
    cache_dir: Path,
    fixtures: tuple[tuple[RemoteObject, bytes], ...],
) -> None:
    for remote, body in fixtures:
        assert remote.sha256 is not None
        cache_path = cache_dir / "sha256" / remote.sha256[:2] / remote.sha256
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(body)


def _write_source_manifest(
    root: Path,
    inventories: tuple[SimfileInventory, ...],
) -> Path:
    source_path = root / "manifests" / "source.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_manifest(build_manifest_rows(inventories, {}, _ENDPOINT_SHA256, _BUCKET))
    source_path.write_bytes(rendered.content)
    return source_path


def _read_manifest_rows(summary: dict[str, object]) -> dict[int, dict[str, object]]:
    manifest_path = summary["manifest_path"]
    assert isinstance(manifest_path, str)
    return {
        row["simfile_id"]: row
        for row in (
            json.loads(line)
            for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        )
    }


def _invoke_selection(
    manifest_path: Path,
    output_dir: Path,
    overrides_path: Path | None = None,
) -> tuple[object, dict[str, object]]:
    arguments = [
        "benchmark",
        "select-reference-charts",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
    ]
    if overrides_path is not None:
        arguments.extend(("--overrides-file", str(overrides_path)))
    result = runner.invoke(main, arguments)
    return result, {} if not result.stdout else json.loads(result.stdout)


def _offline_fixture(tmp_path: Path) -> tuple[Path, Path]:
    corpus_root = tmp_path / "r2-corpus"
    cache_dir = corpus_root / "cache"
    fixtures: list[tuple[RemoteObject, bytes]] = []

    def row(simfile_id: int, *entries: tuple[str, bytes]) -> SimfileInventory:
        remotes = tuple(_remote(simfile_id, key, body) for key, body in entries)
        fixtures.extend(zip(remotes, (body for _, body in entries), strict=True))
        return SimfileInventory(simfile_id, f"{simfile_id}/", remotes, "complete")

    authored_l5 = row(
        101,
        ("set.def", b"#L5FILE: real.dtx\n"),
        ("real.dtx", _chart_body(dlevel=95)),
    )
    custom_txt = row(
        102,
        ("set.def", b"#L5FILE: custom.txt\n"),
        ("custom.txt", _chart_body(dlevel=80)),
    )
    nested = row(
        103,
        ("meta/SET.DEF", b"#L5FILE: charts/lead.dtx\n"),
        ("meta/charts/lead.dtx", _chart_body(dlevel=75)),
    )
    casefold = row(
        104,
        ("set.def", b"#L5FILE: STAGE.DTX\n"),
        ("stage.dtx", _chart_body(dlevel=70)),
    )
    root_fallback = row(
        105,
        ("meta/set.def", b"#L5FILE: root.dtx\n"),
        ("root.dtx", _chart_body(dlevel=65)),
    )
    overridden = row(
        106,
        ("alternate.dtx", _chart_body(dlevel=30)),
        ("approved.txt", _chart_body(dlevel=40)),
    )
    evidence_fallback = row(
        107,
        ("header-only.txt", _chart_body(dlevel=99, has_note_evidence=False)),
        ("fallback.dtx", _chart_body(dlevel=60)),
    )
    ambiguity = row(
        108,
        ("left.dtx", _chart_body(dlevel=55)),
        ("right.dtx", _chart_body(dlevel=55)),
    )

    _write_cached_bodies(cache_dir, tuple(fixtures))
    manifest_path = _write_source_manifest(
        corpus_root,
        (
            authored_l5,
            custom_txt,
            nested,
            casefold,
            root_fallback,
            overridden,
            evidence_fallback,
            ambiguity,
        ),
    )
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_bytes(
        canonical_json_bytes(
            {
                "overrides": {
                    "106": {
                        "chart_key": "106/approved.txt",
                        "reason": "offline audit",
                    }
                },
                "schema_version": _OVERRIDE_SCHEMA,
            },
            trailing_newline=True,
        )
    )
    return manifest_path, overrides_path


def test_reference_chart_acceptance_selects_the_offline_fixture_without_r2_or_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path, overrides_path = _offline_fixture(tmp_path)

    def unexpected_r2_sync(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reference chart selection must not invoke R2 synchronization")

    def unexpected_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reference chart selection must not open a network connection")

    monkeypatch.setattr(benchmark_cli, "sync_r2_corpus", unexpected_r2_sync)
    monkeypatch.setattr(socket, "create_connection", unexpected_network)
    monkeypatch.setattr(socket.socket, "connect", unexpected_network)

    first, first_summary = _invoke_selection(
        manifest_path,
        tmp_path / "first-output",
        overrides_path,
    )

    assert first.exit_code == 1
    assert first.stderr_bytes == b""
    assert set(first_summary) == {
        "corpus_version",
        "exit_code",
        "manifest_path",
        "manifest_sha256",
        "quarantined_count",
        "selected_count",
        "status",
    }
    assert "report_path" not in first_summary
    assert first_summary["status"] == "partial"
    assert first_summary["exit_code"] == 1
    assert first_summary["selected_count"] == 7
    assert first_summary["quarantined_count"] == 1
    assert isinstance(first_summary["corpus_version"], str)
    assert isinstance(first_summary["manifest_sha256"], str)

    first_rows = _read_manifest_rows(first_summary)
    assert first_rows[101]["selected_chart_key"] == "101/real.dtx"
    assert first_rows[101]["selected_level_slot"] == "L5"
    assert first_rows[102]["selected_chart_key"] == "102/custom.txt"
    assert first_rows[103]["selected_chart_key"] == "103/meta/charts/lead.dtx"
    assert first_rows[104]["selected_chart_key"] == "104/stage.dtx"
    assert first_rows[105]["selected_chart_key"] == "105/root.dtx"
    assert first_rows[105]["selection_warnings"] == ["set_def_root_fallback"]
    assert first_rows[106]["selection_method"] == "override"
    assert first_rows[106]["selected_chart_key"] == "106/approved.txt"
    assert first_rows[106]["selection_override"] == {
        "chart_key": "106/approved.txt",
        "reason": "offline audit",
    }
    assert first_rows[107]["selected_chart_key"] == "107/fallback.dtx"
    assert first_rows[108]["selection_status"] == "quarantined"
    assert first_rows[108]["selection_reason_codes"] == ["ambiguous_fallback"]

    second, second_summary = _invoke_selection(
        manifest_path,
        tmp_path / "second-output",
        overrides_path,
    )

    assert second.exit_code == 1
    assert second_summary["manifest_sha256"] == first_summary["manifest_sha256"]
    assert second_summary["corpus_version"] == first_summary["corpus_version"]
    second_manifest_path = second_summary["manifest_path"]
    first_manifest_path = first_summary["manifest_path"]
    assert isinstance(second_manifest_path, str)
    assert isinstance(first_manifest_path, str)
    assert Path(second_manifest_path).read_bytes() == Path(first_manifest_path).read_bytes()

    overrides_path.write_bytes(
        canonical_json_bytes(
            {
                "overrides": {
                    "106": {
                        "chart_key": "106/approved.txt",
                        "reason": "changed offline audit",
                    }
                },
                "schema_version": _OVERRIDE_SCHEMA,
            },
            trailing_newline=True,
        )
    )
    third, third_summary = _invoke_selection(
        manifest_path,
        tmp_path / "third-output",
        overrides_path,
    )

    assert third.exit_code == 1
    assert third_summary["manifest_sha256"] != first_summary["manifest_sha256"]
    assert third_summary["corpus_version"] != first_summary["corpus_version"]
    third_manifest_path = third_summary["manifest_path"]
    assert isinstance(third_manifest_path, str)
    assert Path(third_manifest_path).read_bytes() != Path(first_manifest_path).read_bytes()


def test_reference_chart_acceptance_publishes_an_all_selected_manifest(tmp_path: Path) -> None:
    source_body = _chart_body()
    source = _remote(200, "real.dtx", source_body)
    corpus_root = tmp_path / "r2-corpus"
    _write_cached_bodies(corpus_root / "cache", ((source, source_body),))
    manifest_path = _write_source_manifest(
        corpus_root,
        (SimfileInventory(200, "200/", (source,), "complete"),),
    )

    result, summary = _invoke_selection(manifest_path, tmp_path / "output")

    assert result.exit_code == 0
    assert summary["status"] == "complete"
    assert summary["exit_code"] == 0
    assert summary["selected_count"] == 1
    assert summary["quarantined_count"] == 0
    assert _read_manifest_rows(summary)[200]["selection_status"] == "selected"


def test_reference_chart_acceptance_publishes_an_all_empty_source_manifest(tmp_path: Path) -> None:
    corpus_root = tmp_path / "r2-corpus"
    manifest_path = _write_source_manifest(
        corpus_root,
        (
            SimfileInventory(201, "201/", (), "empty"),
            SimfileInventory(202, "202/", (), "empty"),
        ),
    )

    result, summary = _invoke_selection(manifest_path, tmp_path / "output")

    assert result.exit_code == 1
    assert summary["status"] == "partial"
    assert summary["exit_code"] == 1
    assert summary["selected_count"] == 0
    assert summary["quarantined_count"] == 2
    rows = _read_manifest_rows(summary)
    assert all(row["selection_status"] == "quarantined" for row in rows.values())
    assert rows[201]["selection_reason_codes"] == ["source_inventory_unusable"]


@pytest.mark.parametrize("content", (b"", b"{}\n"), ids=("empty", "malformed"))
def test_reference_chart_acceptance_rejects_invalid_source_rows(
    content: bytes,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "source.jsonl"
    manifest_path.write_bytes(content)

    result, summary = _invoke_selection(manifest_path, tmp_path / "output")

    assert result.exit_code == 2
    assert summary == {
        "corpus_version": None,
        "exit_code": 2,
        "manifest_path": None,
        "manifest_sha256": None,
        "quarantined_count": 0,
        "selected_count": 0,
        "status": "failed",
    }


def test_reference_chart_acceptance_returns_fatal_for_publication_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_body = _chart_body()
    source = _remote(301, "real.dtx", source_body)
    corpus_root = tmp_path / "r2-corpus"
    _write_cached_bodies(corpus_root / "cache", ((source, source_body),))
    manifest_path = _write_source_manifest(
        corpus_root,
        (SimfileInventory(301, "301/", (source,), "complete"),),
    )

    def fail_publication(*_args: object, **_kwargs: object) -> None:
        raise ManifestPublicationError(
            SyncError("artifact", "artifact_write_failed", "offline publication failed")
        )

    monkeypatch.setattr(reference_chart_manifest, "publish_manifest", fail_publication)

    result, summary = _invoke_selection(manifest_path, tmp_path / "output")

    assert result.exit_code == 2
    assert summary["status"] == "failed"
    assert summary["exit_code"] == 2
    assert summary["manifest_path"] is None
    assert summary["manifest_sha256"] is None
