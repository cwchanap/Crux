"""Offline unit tests for the HPA-323 reference-timing corpus diagnostic.

The diagnostic is exercised against fixture rows built through the *real* HPA-322
selection machinery, with only the disk/R2 I/O seams replaced by fakes.  These
tests assert the tool delegates to each production seam and never reaches for a
private HPA-322 validator or path/casefold helper.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

import tools.hpa323.analyze_reference_timing as analyze_reference_timing
from src.benchmark.backend_identity import canonical_json_bytes
from src.benchmark.corpus_cache import CacheIndexStore
from src.benchmark.corpus_manifest import build_manifest_rows, render_manifest
from src.benchmark.dtx_parser import parse_dtx_bytes
from src.benchmark.r2_corpus_models import R2Config, RemoteObject, SimfileInventory
from src.benchmark.reference_chart_manifest import (
    SelectionOutcome,
    SelectionRequest,
    reference_chart_row_view_from_row,
    select_reference_manifest,
)
from src.benchmark.reference_timing import resolve_bgm_reference_groups
from src.benchmark.timing import build_dtx_timing_map
from tools.hpa323.analyze_reference_timing import (
    AnalysisConfig,
    AnalysisDeps,
    AudioProbeOutcome,
    run_reference_timing_analysis,
)

_FIXED_TIME = datetime(2026, 8, 5, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixture helpers (mirror the validated HPA-322 selection path)
# ---------------------------------------------------------------------------


def _remote(simfile_id: int, key: str, body: bytes) -> RemoteObject:
    digest = sha256(body).hexdigest()
    return RemoteObject(
        key=f"{simfile_id}/{key}",
        size=len(body),
        etag=f"etag-{simfile_id}-{key}",
        etag_is_weak=False,
        last_modified=_FIXED_TIME,
        content_type="application/octet-stream",
        cache_status="verified",
        sha256=digest,
        cache_path=f"sha256/{digest[:2]}/{digest}",
    )


def _install_cached_bodies(
    cache_dir: Path,
    fixtures: tuple[tuple[RemoteObject, bytes], ...],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for remote, body in fixtures:
        assert remote.sha256 is not None
        cache_path = cache_dir / "sha256" / remote.sha256[:2] / remote.sha256
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(body)


def _source_rows(
    inventories: tuple[SimfileInventory, ...],
) -> tuple[dict[str, object], ...]:
    return render_manifest(build_manifest_rows(inventories, {}, "f" * 64, "simfile-dtx")).rows


def _write_manifest(tmp_path: Path, rows: tuple[dict[str, object], ...]) -> Path:
    path = tmp_path / "source.jsonl"
    path.write_bytes(b"".join(canonical_json_bytes(row, trailing_newline=True) for row in rows))
    return path


def _publish_reference_manifest(
    tmp_path: Path,
    inventories: tuple[SimfileInventory, ...],
    chart_fixtures: tuple[tuple[RemoteObject, bytes], ...],
) -> Path:
    """Run the real HPA-322 selection and return the published manifest path."""
    cache_dir = tmp_path / "cache"
    _install_cached_bodies(cache_dir, chart_fixtures)
    source_path = _write_manifest(tmp_path, _source_rows(inventories))
    outcome = select_reference_manifest(
        SelectionRequest(
            manifest_path=source_path,
            cache_dir=cache_dir,
            overrides_file=None,
            output_dir=tmp_path / "out",
            default_overrides_missing_ok=True,
        )
    )
    assert isinstance(outcome, SelectionOutcome)
    assert outcome.manifest is not None
    return outcome.manifest.path


# ---------------------------------------------------------------------------
# Fake I/O seams + call recorder
# ---------------------------------------------------------------------------


class _Spy:
    """Wrap a callable, count calls, and delegate to the real implementation."""

    def __init__(self, func):
        self._func = func
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self._func(*args, **kwargs)


class _FakeStore:
    """Minimal R2ObjectStore stand-in; never contacted because sync is faked."""


def _build_deps(
    tmp_path: Path,
    *,
    chart_body: bytes,
    audio_key: str,
    audio_present_first: bool = True,
    audio_decodable: bool = True,
) -> tuple[AnalysisDeps, list[tuple[str, str]], list[str]]:
    """Build injectable deps with offline I/O fakes and a shared call log.

    Returns ``(deps, calls, sync_keys)`` where ``calls`` records
    ``(seam_name, remote_key)`` in invocation order so ordering can be asserted.
    """
    calls: list[tuple[str, str]] = []
    sync_keys: list[str] = []
    audio_resolve_calls: dict[str, int] = {}

    index = CacheIndexStore.load(tmp_path / "cache")
    r2_config = R2Config(
        endpoint_url="https://example.com",
        source_endpoint_sha256="f" * 64,
        bucket="simfile-dtx",
    )

    def read_chart_body(cache_dir, remote, *, source_endpoint_sha256, bucket, expected_sha256=None):
        calls.append(("read_chart_body", remote.key))
        assert remote.key.endswith(".dtx")
        return chart_body

    def resolve_audio_body(
        cache_dir, remote, *, source_endpoint_sha256, bucket, expected_sha256=None
    ):
        calls.append(("resolve_audio_body", remote.key))
        if remote.key != audio_key:
            raise ValueError("verified cache body unavailable")
        seen = audio_resolve_calls.get(remote.key, 0)
        audio_resolve_calls[remote.key] = seen + 1
        if not audio_present_first and seen == 0:
            raise ValueError("verified cache body unavailable")
        return tmp_path / "bodies" / remote.key.replace("/", "_")

    def probe_audio(path):
        calls.append(("probe_audio", str(path)))
        return AudioProbeOutcome(
            decodable=audio_decodable, error=None if audio_decodable else "LibsndfileError"
        )

    def sync_explicit(simfiles, store, idx, config, selected_keys, item_progress=None):
        sync_keys.extend(sorted(selected_keys))
        calls.append(("sync_explicit", ",".join(sorted(selected_keys))))
        from src.benchmark.r2_corpus_models import CacheSyncResult

        return CacheSyncResult(simfiles=simfiles, actions=())

    deps = AnalysisDeps(
        r2_config=r2_config,
        store=_FakeStore(),
        index=index,
        build_row_view=_Spy(reference_chart_row_view_from_row),
        read_chart_body=read_chart_body,
        parse_chart=_Spy(parse_dtx_bytes),
        resolve_bgm_groups=_Spy(resolve_bgm_reference_groups),
        build_timing_map=_Spy(build_dtx_timing_map),
        resolve_audio_body=resolve_audio_body,
        probe_audio=probe_audio,
        sync_explicit=sync_explicit,
    )
    return deps, calls, sync_keys


def _run(config: AnalysisConfig, deps: AnalysisDeps) -> dict[str, object]:
    return run_reference_timing_analysis(config, deps)


def _make_config(
    tmp_path: Path, manifest_path: Path, *, audio_sample_limit: int = 50
) -> AnalysisConfig:
    return AnalysisConfig(
        manifest_path=manifest_path,
        cache_dir=tmp_path / "cache",
        output_path=tmp_path / "report.json",
        audio_sample_limit=audio_sample_limit,
    )


# ---------------------------------------------------------------------------
# Chart fixtures
# ---------------------------------------------------------------------------

# Channel-02 length at measure 1 (length 2.0); a playable note at measure 2,
# position 0.5; a single BGM token at measure 0.  Under sticky timing the note
# lands at 8.0s; under diagnostic-legacy (per-measure reset) at 7.0s -> 1.0s delta.
_CH02_CHART = (
    b"#TITLE: Channel Two\n"
    b"#ARTIST: Tester\n"
    b"#DLEVEL: 50\n"
    b"#BPM: 120\n"
    b"#WAV01: bgm.ogg\n"
    b"#00102: 2\n"
    b"#00001: 01\n"
    b"#00211: 0001\n"
)

# Two BGM tokens at different measures -> two BGM groups (multi-group).  A
# drum-lane note (channel 11) is required so fallback selection admits the chart.
_MULTI_GROUP_CHART = (
    b"#TITLE: Multi Group\n"
    b"#ARTIST: Tester\n"
    b"#DLEVEL: 50\n"
    b"#WAV01: bgm.ogg\n"
    b"#00001: 01\n"
    b"#00101: 01\n"
    b"#00011: 01\n"
)


def _selected_fixture(
    tmp_path: Path,
    chart_body: bytes,
    *,
    audio_name: str = "bgm.ogg",
    simfile_id: int = 42,
    audio_verified: bool = True,
) -> tuple[Path, str, str]:
    """Build a published manifest with one selected chart + one quarantined row."""
    chart = _remote(simfile_id, "real.dtx", chart_body)
    audio = _remote(simfile_id, audio_name, b"audio-body")
    if not audio_verified:
        audio = replace(audio, cache_status="not_selected", sha256=None, cache_path=None)
    selected_inventory = SimfileInventory(simfile_id, f"{simfile_id}/", (chart, audio), "complete")
    quarantined_inventory = SimfileInventory(simfile_id + 1, f"{simfile_id + 1}/", (), "empty")
    manifest_path = _publish_reference_manifest(
        tmp_path,
        (selected_inventory, quarantined_inventory),
        ((chart, chart_body),),
    )
    return manifest_path, chart.key, audio.key


# ---------------------------------------------------------------------------
# Default seam contract: only public production seams are reachable
# ---------------------------------------------------------------------------


def test_default_deps_bind_to_public_production_seams_only() -> None:
    defaults = {field.name: field.default for field in dataclass_fields(AnalysisDeps)}
    assert defaults["build_row_view"] is reference_chart_row_view_from_row
    assert defaults["read_chart_body"].__name__ == "read_verified_cache_body"
    assert defaults["parse_chart"] is parse_dtx_bytes
    assert defaults["resolve_bgm_groups"] is resolve_bgm_reference_groups
    assert defaults["build_timing_map"] is build_dtx_timing_map
    assert defaults["resolve_audio_body"].__name__ == "resolve_verified_cache_body"
    assert defaults["sync_explicit"].__name__ == "sync_explicit_cache_keys"


def test_diagnostic_uses_the_shared_production_manifest_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _CH02_CHART)
    deps, _calls, _sync_keys = _build_deps(
        tmp_path,
        chart_body=_CH02_CHART,
        audio_key="42/bgm.ogg",
    )
    real_loader = analyze_reference_timing.load_reference_chart_manifest
    loaded_paths: list[Path] = []

    def loader(path: Path, **kwargs):
        loaded_paths.append(path)
        return real_loader(path, **kwargs)

    monkeypatch.setattr(analyze_reference_timing, "load_reference_chart_manifest", loader)

    _run(_make_config(tmp_path, manifest_path), deps)

    assert loaded_paths == [manifest_path]


def test_diagnostic_does_not_require_r2_dependencies_without_a_fill(tmp_path: Path) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _CH02_CHART)
    deps, _calls, _sync_keys = _build_deps(
        tmp_path,
        chart_body=_CH02_CHART,
        audio_key="42/bgm.ogg",
    )
    deps = replace(deps, r2_config=None, store=None, index=None)

    report = _run(_make_config(tmp_path, manifest_path), deps)

    assert report["sampled_audio_decodable_count"] == 1


def test_diagnostic_rejects_r2_identity_mismatch_before_filling(tmp_path: Path) -> None:
    manifest_path, _chart_key, audio_key = _selected_fixture(
        tmp_path,
        _CH02_CHART,
        audio_verified=False,
    )
    deps, _calls, _sync_keys = _build_deps(
        tmp_path,
        chart_body=_CH02_CHART,
        audio_key=audio_key,
        audio_present_first=False,
    )
    assert deps.r2_config is not None
    deps = replace(
        deps,
        r2_config=replace(deps.r2_config, source_endpoint_sha256="0" * 64),
    )

    with pytest.raises(ValueError, match="identity"):
        _run(_make_config(tmp_path, manifest_path), deps)


# ---------------------------------------------------------------------------
# Production-seam delegation + ordering
# ---------------------------------------------------------------------------


def test_tool_calls_each_production_seam_for_selected_charts(tmp_path: Path) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _CH02_CHART)
    deps, _calls, _sync_keys = _build_deps(tmp_path, chart_body=_CH02_CHART, audio_key="42/bgm.ogg")

    report = _run(_make_config(tmp_path, manifest_path), deps)

    # Pure seams (spies) are invoked for the single selected chart.
    assert deps.build_row_view.call_count == 2  # one selected + one quarantined row
    assert deps.parse_chart.call_count == 1
    assert deps.resolve_bgm_groups.call_count == 1
    # build_timing_map runs twice per chart (corrected + diagnostic-legacy).
    assert deps.build_timing_map.call_count == 2
    # The reported row split reflects the fixture.
    assert report["selected_rows"] == 1
    assert report["upstream_quarantined_rows"] == 1


def test_resolve_verified_cache_body_precedes_sf_info(tmp_path: Path) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _CH02_CHART)
    deps, calls, _sync_keys = _build_deps(tmp_path, chart_body=_CH02_CHART, audio_key="42/bgm.ogg")

    _run(_make_config(tmp_path, manifest_path), deps)

    audio_key = "42/bgm.ogg"
    resolve_indices = [
        i
        for i, (seam, key) in enumerate(calls)
        if seam == "resolve_audio_body" and key == audio_key
    ]
    probe_indices = [i for i, (seam, key) in enumerate(calls) if seam == "probe_audio"]
    assert resolve_indices, "resolve_verified_cache_body must be called for sampled audio"
    assert probe_indices, "soundfile.info (probe_audio) must be called for sampled audio"
    assert min(resolve_indices) < min(probe_indices)


def test_missing_audio_is_filled_via_sync_explicit_cache_keys_before_probe(tmp_path: Path) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _CH02_CHART)
    deps, calls, sync_keys = _build_deps(
        tmp_path,
        chart_body=_CH02_CHART,
        audio_key="42/bgm.ogg",
        audio_present_first=False,  # body unavailable on first resolve
    )

    report = _run(_make_config(tmp_path, manifest_path), deps)

    # The exact audio key was filled through sync_explicit_cache_keys.
    assert sync_keys == ["42/bgm.ogg"]
    sync_index = next(i for i, (seam, key) in enumerate(calls) if seam == "sync_explicit")
    probe_index = next(i for i, (seam, _key) in enumerate(calls) if seam == "probe_audio")
    assert sync_index < probe_index
    assert report["sampled_audio_count"] == 1
    assert report["sampled_audio_decodable_count"] == 1


def test_missing_audio_uses_the_rebuilt_remote_and_separates_decoder_failures(
    tmp_path: Path,
) -> None:
    manifest_path, _chart_key, audio_key = _selected_fixture(
        tmp_path,
        _CH02_CHART,
        audio_verified=False,
    )
    deps, calls, _sync_keys = _build_deps(
        tmp_path,
        chart_body=_CH02_CHART,
        audio_key=audio_key,
        audio_present_first=False,
    )
    rebuilt_digest = sha256(b"audio-body").hexdigest()
    rebuilt_remote = _remote(42, "bgm.ogg", b"audio-body")
    seen_remotes: list[RemoteObject] = []

    def resolve_audio_body(cache_dir, remote, **kwargs):
        del cache_dir, kwargs
        seen_remotes.append(remote)
        if remote.sha256 != rebuilt_digest or remote.cache_status != "verified":
            raise ValueError("verified cache body unavailable")
        return tmp_path / "bodies" / "42_bgm.ogg"

    def sync_explicit(simfiles, store, index, config, selected_keys, item_progress=None):
        del store, index, config, selected_keys, item_progress
        from src.benchmark.r2_corpus_models import CacheSyncResult

        rebuilt = tuple(
            replace(
                simfile,
                objects=tuple(
                    rebuilt_remote if remote.key == audio_key else remote
                    for remote in simfile.objects
                ),
            )
            for simfile in simfiles
        )
        return CacheSyncResult(simfiles=rebuilt, actions=())

    deps = replace(
        deps,
        resolve_audio_body=resolve_audio_body,
        sync_explicit=sync_explicit,
    )

    report = _run(_make_config(tmp_path, manifest_path), deps)

    assert [remote.cache_status for remote in seen_remotes] == ["not_selected", "verified"]
    assert [remote.sha256 for remote in seen_remotes] == [None, rebuilt_digest]
    assert report["sampled_audio_count"] == 1
    assert report["sampled_audio_decodable_count"] == 1
    assert report["sampled_audio_cache_failure_count"] == 0
    assert report["sampled_audio_decoder_failure_count"] == 0
    assert report["sampled_audio_undecodable_count"] == 0
    assert any(seam == "probe_audio" for seam, _key in calls)


def test_missing_audio_cache_failure_is_not_reported_as_decoder_failure(tmp_path: Path) -> None:
    manifest_path, _chart_key, audio_key = _selected_fixture(
        tmp_path,
        _CH02_CHART,
        audio_verified=False,
    )
    deps, _calls, _sync_keys = _build_deps(
        tmp_path,
        chart_body=_CH02_CHART,
        audio_key=audio_key,
        audio_present_first=False,
    )

    def unavailable(*_args, **_kwargs):
        raise ValueError("verified cache body unavailable")

    def no_rebuild(simfiles, *_args, **_kwargs):
        from src.benchmark.r2_corpus_models import CacheSyncResult

        return CacheSyncResult(simfiles=simfiles, actions=())

    deps = replace(deps, resolve_audio_body=unavailable, sync_explicit=no_rebuild)

    report = _run(_make_config(tmp_path, manifest_path), deps)

    assert report["sampled_audio_count"] == 1
    assert report["sampled_audio_cache_failure_count"] == 1
    assert report["sampled_audio_decoder_failure_count"] == 0
    assert report["sampled_audio_decodable_count"] == 0
    assert report["sampled_audio_undecodable_count"] == 0


# ---------------------------------------------------------------------------
# BGM group distribution + multi-group examples
# ---------------------------------------------------------------------------


def test_single_bgm_group_is_counted_as_one(tmp_path: Path) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _CH02_CHART)
    deps, _calls, _sync_keys = _build_deps(tmp_path, chart_body=_CH02_CHART, audio_key="42/bgm.ogg")

    report = _run(_make_config(tmp_path, manifest_path), deps)

    assert report["rows_with_1_bgm_group"] == 1
    assert report["rows_with_0_bgm_groups"] == 0
    assert report["rows_with_multiple_bgm_groups"] == 0
    assert report["multi_group_examples"] == []


def test_multiple_bgm_groups_are_captured_with_a_bounded_example(tmp_path: Path) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _MULTI_GROUP_CHART)
    deps, _calls, _sync_keys = _build_deps(
        tmp_path, chart_body=_MULTI_GROUP_CHART, audio_key="42/bgm.ogg"
    )

    report = _run(_make_config(tmp_path, manifest_path), deps)

    assert report["rows_with_multiple_bgm_groups"] == 1
    (example,) = report["multi_group_examples"]
    assert example["simfile_id"] == 42
    assert example["chart_key"] == "42/real.dtx"
    assert len(example["groups"]) == 2
    assert {group["object_key"] for group in example["groups"]} == {"42/bgm.ogg"}
    assert {group["measure"] for group in example["groups"]} == {0, 1}
    assert all("note_ids" in group and "position" in group for group in example["groups"])


# ---------------------------------------------------------------------------
# Channel-02 blast radius
# ---------------------------------------------------------------------------


def test_channel_02_timing_delta_is_measured_through_the_single_engine(tmp_path: Path) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _CH02_CHART)
    deps, _calls, _sync_keys = _build_deps(tmp_path, chart_body=_CH02_CHART, audio_key="42/bgm.ogg")

    report = _run(_make_config(tmp_path, manifest_path), deps)

    assert report["charts_with_channel_02"] == 1
    assert report["charts_with_multiple_channel_02_changes"] == 0
    # Sticky: 8.0s, diagnostic-legacy: 7.0s -> 1.0s delta.
    assert report["max_channel_02_time_delta_sec"] == 1
    (example,) = report["channel_02_delta_examples"]
    assert set(example) == {
        "chart_key",
        "corrected_sec",
        "delta_sec",
        "legacy_sec",
        "simfile_id",
        "source_order",
    }
    assert example["corrected_sec"] == 8
    assert example["legacy_sec"] == 7
    assert example["delta_sec"] == 1
    assert example["chart_key"] == "42/real.dtx"


def test_chart_without_channel_02_reports_zero_delta(tmp_path: Path) -> None:
    chart = (
        b"#TITLE: Plain\n#ARTIST: Tester\n#DLEVEL: 50\n#BPM: 120\n"
        b"#WAV01: bgm.ogg\n#00001: 01\n#00011: 01\n"
    )
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, chart)
    deps, _calls, _sync_keys = _build_deps(tmp_path, chart_body=chart, audio_key="42/bgm.ogg")

    report = _run(_make_config(tmp_path, manifest_path), deps)

    assert report["charts_with_channel_02"] == 0
    assert report["max_channel_02_time_delta_sec"] == 0
    assert report["channel_02_delta_examples"] == []


# ---------------------------------------------------------------------------
# Authored extension distribution + decodability
# ---------------------------------------------------------------------------


def test_authored_extension_counts_and_decodability_are_reported_separately(tmp_path: Path) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _CH02_CHART)
    deps, _calls, _sync_keys = _build_deps(
        tmp_path, chart_body=_CH02_CHART, audio_key="42/bgm.ogg", audio_decodable=True
    )

    report = _run(_make_config(tmp_path, manifest_path), deps)

    assert report["bgm_extension_counts"] == {".ogg": 1}
    assert report["sampled_audio_count"] == 1
    assert report["sampled_audio_decodable_count"] == 1
    assert report["sampled_audio_undecodable_count"] == 0
    assert report["sampled_audio_undecodable_by_extension"] == {}


def test_undecodable_audio_is_counted_by_authored_extension(tmp_path: Path) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _CH02_CHART)
    deps, _calls, _sync_keys = _build_deps(
        tmp_path, chart_body=_CH02_CHART, audio_key="42/bgm.ogg", audio_decodable=False
    )

    report = _run(_make_config(tmp_path, manifest_path), deps)

    assert report["sampled_audio_undecodable_count"] == 1
    assert report["sampled_audio_undecodable_by_extension"] == {".ogg": 1}
    assert report["sampled_audio_decoder_failure_count"] == 1
    assert report["sampled_audio_decoder_failure_by_extension"] == {".ogg": 1}


# ---------------------------------------------------------------------------
# Path-resolution distribution
# ---------------------------------------------------------------------------


def test_case_insensitive_match_is_reported(tmp_path: Path) -> None:
    # Authored path is lowercase but the inventory key is mixed-case -> casefold.
    manifest_path, _chart_key, _audio_key = _selected_fixture(
        tmp_path, _CH02_CHART, audio_name="Bgm.Ogg"
    )
    deps, _calls, _sync_keys = _build_deps(tmp_path, chart_body=_CH02_CHART, audio_key="42/Bgm.Ogg")

    report = _run(_make_config(tmp_path, manifest_path), deps)

    assert report["rows_needing_case_insensitive_match"] == 1
    assert report["rows_needing_simfile_root_fallback"] == 0
    assert report["rows_with_unresolved_wav"] == 0


# ---------------------------------------------------------------------------
# Determinism (Step 6): byte-identical report on re-run
# ---------------------------------------------------------------------------


def test_report_is_byte_identical_on_re_run(tmp_path: Path) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _CH02_CHART)

    deps_a, _calls_a, _sync_a = _build_deps(
        tmp_path, chart_body=_CH02_CHART, audio_key="42/bgm.ogg"
    )
    deps_b, _calls_b, _sync_b = _build_deps(
        tmp_path, chart_body=_CH02_CHART, audio_key="42/bgm.ogg"
    )

    from tools.hpa323.analyze_reference_timing import render_report

    config = _make_config(tmp_path, manifest_path)
    bytes_a = render_report(_run(config, deps_a))
    bytes_b = render_report(_run(config, deps_b))

    assert bytes_a == bytes_b
    assert sha256(bytes_a).hexdigest() == sha256(bytes_b).hexdigest()
    # Canonical form ends with exactly one final newline.
    assert bytes_a.endswith(b"\n")
    assert not bytes_a.endswith(b"\n\n")


def test_audio_sample_limit_caps_the_sampled_set(tmp_path: Path) -> None:
    manifest_path, _chart_key, _audio_key = _selected_fixture(tmp_path, _CH02_CHART)
    deps, _calls, _sync_keys = _build_deps(tmp_path, chart_body=_CH02_CHART, audio_key="42/bgm.ogg")

    report = _run(_make_config(tmp_path, manifest_path, audio_sample_limit=0), deps)

    assert report["sampled_audio_count"] == 0
    assert report["sampled_audio_decodable_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
