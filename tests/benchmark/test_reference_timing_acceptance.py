"""End-to-end offline acceptance for ``run_reference_timing`` (HPA-323 Task 6b).

These fixtures exercise the full cache / R2-orchestration / event-publication
pipeline against a fake store and real local cache directories.  No network or
``boto3`` dependency is required: the complete-cache path is proven to never
touch the optional R2 store (factory and dependency spies stay uncalled), and
the targeted-fill path is driven through an in-memory fake store.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

import pytest

from src.benchmark.reference_timing_manifest import (
    run_reference_timing,
)

# Reuse the orchestration fixture helpers (chart/audio bodies, manifest
# builder, request builder, fake store) from the sibling manifest test module
# rather than duplicating the HPA-322 row machinery.


def _load_manifest_helpers() -> object:
    spec = importlib.util.spec_from_file_location(
        "_reference_timing_manifest_helpers",
        Path(__file__).parent / "test_reference_timing_manifest.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before execution so module-level ``@dataclass`` decorators (which
    # look up ``sys.modules``) resolve correctly.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_helpers = _load_manifest_helpers()
_READY_CHART_BODY = _helpers._READY_CHART_BODY
_NO_BGM_CHART_BODY = _helpers._NO_BGM_CHART_BODY
_wav_bytes = _helpers._wav_bytes
_SelectedSimfile = _helpers._SelectedSimfile
_publish_timing_manifest = _helpers._publish_timing_manifest
_timing_request = _helpers._timing_request
_ready_audio_spec = _helpers._ready_audio_spec
_AudioFakeStore = _helpers._AudioFakeStore
_RecordingCall = _helpers._RecordingCall
_FILL_ENDPOINT = _helpers._FILL_ENDPOINT
_FILL_ENDPOINT_HASH = _helpers._FILL_ENDPOINT_HASH
_ready_rows = _helpers._ready_rows

# A chart whose single BGM file lands at two measures -> two BGM groups ->
# ``ambiguous_bgm_start``.  A playable note (channel 11) is included so HPA-322
# still selects the chart; only the timing layer observes the ambiguity.
_AMBIGUOUS_BGM_CHART_BODY = (
    b"#TITLE: Example Song\n#ARTIST: Example Artist\n#DLEVEL: 99\n"
    b"#WAV01: bgm.wav\n#00001: 01\n#01001: 01\n#00011: 01\n"
)


@contextmanager
def _fatal_factory() -> Iterator[list[object]]:
    calls: list[object] = []

    def factory(config: object) -> object:
        calls.append(config)
        raise AssertionError("store factory must not be called on a complete cache")

    yield calls, factory


def _read_events(output_dir: Path, relative_path: object) -> bytes:
    assert isinstance(relative_path, str)
    return (output_dir / relative_path).read_bytes()


# ---------------------------------------------------------------------------
# Scenario 1: selected chart + already-cached source audio -> ready (no R2)
# ---------------------------------------------------------------------------


def test_acceptance_ready_row_with_complete_cache_never_touches_r2(tmp_path):
    fixture = _publish_timing_manifest(tmp_path, selected=(_ready_audio_spec(),))
    dependency_calls = _RecordingCall()

    with _fatal_factory() as (factory_calls, factory):
        outcome = run_reference_timing(
            _timing_request(fixture),
            environ={},
            dependency_check=lambda: dependency_calls.calls.append(1),
            store_factory=factory,
        )

    assert outcome.exit_code == 0
    assert outcome.status == "complete"
    assert outcome.ready_count == 1
    assert outcome.events_published == 1
    assert not dependency_calls
    assert not factory_calls
    # The manifest row carries the full source-audio + events identity.
    (row,) = _ready_rows(outcome)
    assert row["timing_status"] == "ready"
    assert row["source_audio_key"] == "42/bgm.wav"
    assert row["reference_events_cache_path"] is not None
    assert _read_events(fixture.output_dir, row["reference_events_cache_path"]).endswith(b"\n")


# ---------------------------------------------------------------------------
# Scenario 2: selected chart + exact-key audio fill through fake store -> ready
# ---------------------------------------------------------------------------


def test_acceptance_exact_key_fill_through_fake_store_reaches_ready(tmp_path):
    spec = _SelectedSimfile(42, _READY_CHART_BODY, _wav_bytes(), audio_verified=False)
    fixture = _publish_timing_manifest(
        tmp_path,
        selected=(spec,),
        endpoint_sha256=_FILL_ENDPOINT_HASH,
    )
    store = _AudioFakeStore({"42/bgm.wav": spec.audio_body})

    outcome = run_reference_timing(
        _timing_request(fixture),
        environ={"CRUX_R2_ENDPOINT_URL": _FILL_ENDPOINT, "CRUX_R2_BUCKET": "simfile-dtx"},
        dependency_check=lambda: None,
        store_factory=lambda config: store,
    )

    assert outcome.exit_code == 0
    assert outcome.ready_count == 1
    assert store.validate_calls == 1
    assert [call[0] for call in store.open_calls] == ["42/bgm.wav"]
    audio_digest = sha256(spec.audio_body).hexdigest()
    assert (
        fixture.cache_dir / "sha256" / audio_digest[:2] / audio_digest
    ).read_bytes() == spec.audio_body
    (row,) = _ready_rows(outcome)
    assert row["source_audio_content_hash"] == audio_digest


# ---------------------------------------------------------------------------
# Scenario 3: upstream HPA-322 quarantine -> preserved timing quarantine
# ---------------------------------------------------------------------------


def test_acceptance_upstream_quarantine_is_preserved(tmp_path):
    fixture = _publish_timing_manifest(
        tmp_path,
        selected=(_ready_audio_spec(42),),
        empty_simfile_ids=(43,),
    )

    outcome = run_reference_timing(_timing_request(fixture))

    assert outcome.exit_code == 1
    assert outcome.ready_count == 1
    assert outcome.quarantined_count == 1
    assert outcome.upstream_quarantined_count == 1
    rows = {row["simfile_id"]: row for row in _ready_rows(outcome)}
    assert rows[43]["timing_status"] == "quarantined"
    assert rows[43]["timing_reason_codes"] == ["upstream_chart_selection_unavailable"]
    assert rows[43]["source_audio_key"] is None
    assert rows[43]["reference_events_cache_path"] is None


# ---------------------------------------------------------------------------
# Scenario 4: unresolved / ambiguous BGM row -> HPA-323 quarantine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chart_body", "expected_reason"),
    [
        (_NO_BGM_CHART_BODY, "bgm_event_missing"),
        (_AMBIGUOUS_BGM_CHART_BODY, "ambiguous_bgm_start"),
    ],
    ids=["no_bgm_event", "ambiguous_bgm_start"],
)
def test_acceptance_bgm_failure_row_is_hpa323_quarantined(
    tmp_path, chart_body: bytes, expected_reason: str
):
    spec = _SelectedSimfile(42, chart_body, _wav_bytes(), audio_verified=True)
    fixture = _publish_timing_manifest(tmp_path, selected=(spec,))

    outcome = run_reference_timing(_timing_request(fixture))

    assert outcome.exit_code == 1
    assert outcome.ready_count == 0
    assert outcome.quarantined_count == 1
    assert outcome.upstream_quarantined_count == 0
    assert outcome.quarantined_count - outcome.upstream_quarantined_count == 1
    (row,) = _ready_rows(outcome)
    assert row["timing_status"] == "quarantined"
    assert row["timing_reason_codes"] == [expected_reason]


# ---------------------------------------------------------------------------
# Scenario 5: unreadable source audio -> source_audio_decode_failed
# ---------------------------------------------------------------------------


def test_acceptance_unreadable_source_audio_quarantines(tmp_path):
    spec = _SelectedSimfile(42, _READY_CHART_BODY, b"definitely not audio", audio_verified=True)
    fixture = _publish_timing_manifest(tmp_path, selected=(spec,))

    outcome = run_reference_timing(_timing_request(fixture))

    assert outcome.exit_code == 1
    assert outcome.ready_count == 0
    assert outcome.quarantined_count == 1
    (row,) = _ready_rows(outcome)
    assert row["timing_reason_codes"] == ["source_audio_decode_failed"]


# ---------------------------------------------------------------------------
# Scenario 6: repeated second run -> deterministic manifest/event identities
# ---------------------------------------------------------------------------


def test_acceptance_second_run_is_byte_identical_and_reuses_events(tmp_path):
    fixture = _publish_timing_manifest(
        tmp_path,
        selected=(_ready_audio_spec(42), _ready_audio_spec(43)),
    )

    first = run_reference_timing(_timing_request(fixture))
    assert first.exit_code == 0
    assert first.manifest is not None
    first_manifest_bytes = first.manifest.path.read_bytes()
    first_manifest_sha = first.manifest.manifest_sha256
    first_events = {
        row["simfile_id"]: _read_events(fixture.output_dir, row["reference_events_cache_path"])
        for row in _ready_rows(first)
    }

    second = run_reference_timing(_timing_request(fixture))
    assert second.exit_code == 0
    assert second.manifest is not None
    assert second.manifest.manifest_sha256 == first_manifest_sha
    assert second.manifest.path.read_bytes() == first_manifest_bytes
    second_events = {
        row["simfile_id"]: _read_events(fixture.output_dir, row["reference_events_cache_path"])
        for row in _ready_rows(second)
    }
    assert first_events == second_events


# ---------------------------------------------------------------------------
# Combined mixed corpus: ready + fill + upstream quarantine + BGM failure
# ---------------------------------------------------------------------------


def test_acceptance_mixed_corpus_balances_accounting_and_publishes_only_ready(
    tmp_path,
):
    ready_verified = _ready_audio_spec(42, audio_verified=True)
    ready_fill = _SelectedSimfile(43, _READY_CHART_BODY, _wav_bytes(), audio_verified=False)
    no_bgm = _SelectedSimfile(44, _NO_BGM_CHART_BODY, _wav_bytes(), audio_verified=True)
    fixture = _publish_timing_manifest(
        tmp_path,
        selected=(ready_verified, ready_fill, no_bgm),
        empty_simfile_ids=(45,),
        endpoint_sha256=_FILL_ENDPOINT_HASH,
    )
    store = _AudioFakeStore({"43/bgm.wav": ready_fill.audio_body})

    outcome = run_reference_timing(
        _timing_request(fixture),
        environ={"CRUX_R2_ENDPOINT_URL": _FILL_ENDPOINT, "CRUX_R2_BUCKET": "simfile-dtx"},
        dependency_check=lambda: None,
        store_factory=lambda config: store,
    )

    assert outcome.exit_code == 1
    assert outcome.ready_count == 2
    assert outcome.quarantined_count == 2
    assert outcome.upstream_quarantined_count == 1
    assert outcome.events_published == 2
    rows = {row["simfile_id"]: row for row in _ready_rows(outcome)}
    assert rows[42]["timing_status"] == "ready"
    assert rows[43]["timing_status"] == "ready"
    assert rows[44]["timing_reason_codes"] == ["bgm_event_missing"]
    assert rows[45]["timing_reason_codes"] == ["upstream_chart_selection_unavailable"]
    # The fill store only served the missing audio; the verified row never hit it.
    assert [call[0] for call in store.open_calls] == ["43/bgm.wav"]


# ---------------------------------------------------------------------------
# Fatal fill failure: a download error quarantines only the affected row.
# ---------------------------------------------------------------------------


def test_acceptance_failed_download_is_mapped_and_does_not_abort_siblings(tmp_path):
    ready = _ready_audio_spec(42, audio_verified=True)
    fill_fails = _SelectedSimfile(43, _READY_CHART_BODY, _wav_bytes(), audio_verified=False)
    fixture = _publish_timing_manifest(
        tmp_path,
        selected=(ready, fill_fails),
        endpoint_sha256=_FILL_ENDPOINT_HASH,
    )
    store = _AudioFakeStore({})  # no body served -> object_get_failed

    outcome = run_reference_timing(
        _timing_request(fixture),
        environ={"CRUX_R2_ENDPOINT_URL": _FILL_ENDPOINT, "CRUX_R2_BUCKET": "simfile-dtx"},
        dependency_check=lambda: None,
        store_factory=lambda config: store,
    )

    assert outcome.exit_code == 1
    assert outcome.ready_count == 1
    assert outcome.quarantined_count == 1
    rows = {row["simfile_id"]: row for row in _ready_rows(outcome)}
    assert rows[42]["timing_status"] == "ready"
    assert rows[43]["timing_reason_codes"] == ["source_audio_download_failed"]
