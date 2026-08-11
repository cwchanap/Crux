from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.reference_set_manifest import (
    BENCHMARK_REFERENCE_MANIFEST_SCHEMA,
    ReferenceSetRequest,
    failed_reference_set_outcome,
    run_reference_set,
    validate_schema_golden,
)
from src.benchmark.reference_timing import (
    NativeReferenceEvent,
    read_reference_events,
    render_reference_events,
)
from src.benchmark.reference_timing_manifest import load_reference_timing_manifest

_GOLDEN = Path(__file__).parent / "schema_goldens/crux.reference-timing-manifest-v1.jsonl"
_AUDIT = (
    Path(__file__).parents[2]
    / "docs/superpowers/evidence/2026-08-09-hpa-324-reference-lane-audit.json"
)


def _timing_rows() -> tuple[dict[str, object], dict[str, object]]:
    return tuple(json.loads(line) for line in _GOLDEN.read_text().splitlines())  # type: ignore[return-value]


def _native_event(
    lane_id: str, *, audio_time_sec: float, source_order: int = 0
) -> NativeReferenceEvent:
    return NativeReferenceEvent(
        simfile_id=42,
        selected_chart_key="42/real.dtx",
        selected_chart_content_hash="5dbd7639514e76fbd11cc8d6c518adae8b168ea944461c555236dadb1a3a4809",
        source_audio_key="42/bgm.wav",
        source_audio_content_hash="c" * 64,
        source_order=source_order,
        measure=1,
        position=0.0,
        lane_id=lane_id,
        note_id=f"{source_order + 1:02X}",
        chart_time_sec=audio_time_sec,
        audio_time_sec=audio_time_sec,
    )


def _write_timing_manifest(
    tmp_path: Path,
    events: tuple[NativeReferenceEvent, ...] | None,
    *,
    upstream: bool = False,
) -> Path:
    ready, quarantined = _timing_rows()
    row = copy.deepcopy(quarantined if upstream else ready)
    if events is not None:
        event_content = render_reference_events(events)
        event_hash = hashlib.sha256(event_content).hexdigest()
        row["reference_events_cache_path"] = f"events/{event_hash}.jsonl"
    else:
        event_content = None

    rendered = render_manifest(
        ({key: value for key, value in row.items() if key != "corpus_version"},)
    )
    # RenderedManifest intentionally has no relative path; publish the fixture
    # in the same layout as HPA-323 so the event reader resolves its sibling.
    manifest_path = tmp_path / "timing" / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(rendered.content)
    if event_content is not None:
        event_path = manifest_path.parent.parent / str(row["reference_events_cache_path"])
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_bytes(event_content)
    return manifest_path


def _published_rows(outcome: object) -> list[dict[str, object]]:
    manifest = getattr(outcome, "manifest")
    assert manifest is not None
    return [json.loads(line) for line in manifest.path.read_text().splitlines()]


def test_upstream_quarantine_preserves_timing_reasons_and_zeroes_mapping_counts(
    tmp_path: Path,
) -> None:
    manifest_path = _write_timing_manifest(tmp_path, None, upstream=True)

    outcome = run_reference_set(ReferenceSetRequest(manifest_path, tmp_path / "reference-set"))

    assert outcome.exit_code == 1
    assert outcome.eligible_count == 0
    assert outcome.quarantined_count == 1
    row = _published_rows(outcome)[0]
    assert row["reference_eligibility_status"] == "quarantined"
    assert row["reference_eligibility_reason_codes"] == ["upstream_reference_unavailable"]
    assert row["timing_reason_codes"] == ["upstream_chart_selection_unavailable"]
    assert row["mapped_event_count"] == 0
    assert row["common_scored_event_count"] == 0
    assert row["ignored_event_count"] == 0
    assert row["unmapped_event_count"] == 0
    assert row["duplicate_common_event_count"] == 0


@pytest.mark.parametrize("artifact", ["missing", "hash-mismatch", "invalid-json"])
def test_ready_row_invalid_event_artifact_is_row_local_quarantine(
    tmp_path: Path,
    artifact: str,
) -> None:
    event = _native_event("13", audio_time_sec=1.0)
    manifest_path = _write_timing_manifest(tmp_path, (event,))
    row = json.loads(manifest_path.read_text())
    relative = row["reference_events_cache_path"]
    event_path = manifest_path.parent.parent / str(relative)
    if artifact == "missing":
        event_path.unlink()
    elif artifact == "hash-mismatch":
        event_path.write_bytes(event_path.read_bytes() + b"tampered")
    else:
        invalid = b'{"not":"an event"}\n'
        digest = hashlib.sha256(invalid).hexdigest()
        event_path.unlink()
        new_row = copy.deepcopy(row)
        new_row["reference_events_cache_path"] = f"events/{digest}.jsonl"
        rendered = render_manifest(
            ({key: value for key, value in new_row.items() if key != "corpus_version"},)
        )
        manifest_path.write_bytes(rendered.content)
        event_path = manifest_path.parent.parent / "events" / f"{digest}.jsonl"
        event_path.write_bytes(invalid)

    outcome = run_reference_set(ReferenceSetRequest(manifest_path, tmp_path / "reference-set"))

    assert outcome.exit_code == 1
    row = _published_rows(outcome)[0]
    assert row["reference_eligibility_status"] == "quarantined"
    assert row["reference_eligibility_reason_codes"] == ["reference_event_artifact_invalid"]
    assert all(
        row[field] == 0
        for field in (
            "mapped_event_count",
            "common_scored_event_count",
            "ignored_event_count",
            "unmapped_event_count",
            "duplicate_common_event_count",
        )
    )


def test_safe_looking_symlink_event_artifact_is_row_local_quarantine(tmp_path: Path) -> None:
    event = _native_event("13", audio_time_sec=1.0)
    manifest_path = _write_timing_manifest(tmp_path, (event,))
    row = json.loads(manifest_path.read_text())
    event_path = manifest_path.parent.parent / str(row["reference_events_cache_path"])
    external_path = tmp_path / "external-events.jsonl"
    external_path.write_bytes(render_reference_events((event,)))
    event_path.unlink()
    event_path.symlink_to(external_path)
    assert load_reference_timing_manifest(manifest_path).rows[0].view.timing_status == "ready"
    assert event_path.is_symlink()

    outcome = run_reference_set(ReferenceSetRequest(manifest_path, tmp_path / "reference-set"))

    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert outcome.manifest is not None
    assert outcome.eligible_count == 0
    assert outcome.quarantined_count == 1
    published_row = _published_rows(outcome)[0]
    assert published_row["reference_eligibility_status"] == "quarantined"
    assert published_row["reference_eligibility_reason_codes"] == [
        "reference_event_artifact_invalid"
    ]


@pytest.mark.parametrize(
    ("identity_field", "event_updates"),
    [
        (
            "simfile_id",
            {
                "simfile_id": 43,
                "selected_chart_key": "43/real.dtx",
                "source_audio_key": "43/bgm.wav",
            },
        ),
        ("selected_chart_key", {"selected_chart_key": "42/alternate.dtx"}),
        (
            "selected_chart_content_hash",
            {"selected_chart_content_hash": "d" * 64},
        ),
        ("source_audio_key", {"source_audio_key": "42/alternate.wav"}),
        (
            "source_audio_content_hash",
            {"source_audio_content_hash": "e" * 64},
        ),
    ],
)
def test_canonical_rehashed_event_identity_mismatch_is_row_local_quarantine(
    tmp_path: Path,
    identity_field: str,
    event_updates: dict[str, object],
) -> None:
    event = replace(_native_event("13", audio_time_sec=1.0), **event_updates)
    manifest_path = _write_timing_manifest(tmp_path, (event,))

    loaded = load_reference_timing_manifest(manifest_path)
    assert loaded.rows[0].view.timing_status == "ready"
    assert getattr(event, identity_field) != loaded.rows[0].source_row[identity_field]
    row = json.loads(manifest_path.read_text())
    relative = row["reference_events_cache_path"]
    assert isinstance(relative, str)
    event_path = manifest_path.parent.parent / relative
    content = event_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == event_path.stem
    assert read_reference_events(content) == (event,)

    outcome = run_reference_set(ReferenceSetRequest(manifest_path, tmp_path / "reference-set"))

    assert outcome.status == "partial"
    assert outcome.exit_code == 1
    assert outcome.eligible_count == 0
    assert outcome.quarantined_count == 1
    published_row = _published_rows(outcome)[0]
    assert published_row["reference_eligibility_status"] == "quarantined"
    assert published_row["reference_eligibility_reason_codes"] == [
        "reference_event_artifact_invalid"
    ]
    assert all(
        published_row[field] == 0
        for field in (
            "mapped_event_count",
            "common_scored_event_count",
            "ignored_event_count",
            "unmapped_event_count",
            "duplicate_common_event_count",
        )
    )


def test_mapped_events_without_diagnostics_are_eligible(tmp_path: Path) -> None:
    manifest_path = _write_timing_manifest(tmp_path, (_native_event("13", audio_time_sec=1.0),))

    outcome = run_reference_set(ReferenceSetRequest(manifest_path, tmp_path / "reference-set"))

    assert outcome.exit_code == 0
    row = _published_rows(outcome)[0]
    assert row["reference_eligibility_status"] == "eligible"
    assert row["reference_eligibility_reason_codes"] == []
    assert row["reference_eligibility_warnings"] == []
    assert row["mapped_event_count"] == 1
    assert row["common_scored_event_count"] == 1
    assert row["ignored_event_count"] == 0
    assert row["unmapped_event_count"] == 0
    assert row["duplicate_common_event_count"] == 0


def test_reviewed_non_drum_lanes_are_eligible_with_deterministic_warnings(tmp_path: Path) -> None:
    evidence = json.loads(_AUDIT.read_text())
    ignored_lanes = evidence["ignored_non_drum_lanes"]
    assert ignored_lanes
    events = tuple(
        [_native_event("13", audio_time_sec=1.0)]
        + [
            _native_event(lane, audio_time_sec=2.0, source_order=index + 1)
            for index, lane in enumerate(ignored_lanes)
        ]
    )
    manifest_path = _write_timing_manifest(tmp_path, events)

    outcome = run_reference_set(ReferenceSetRequest(manifest_path, tmp_path / "reference-set"))

    assert outcome.exit_code == 0
    row = _published_rows(outcome)[0]
    assert row["reference_eligibility_status"] == "eligible"
    assert row["reference_eligibility_warnings"] == [
        f"ignored_reference_lane:{lane}:count=1" for lane in sorted(ignored_lanes)
    ]
    assert row["mapped_event_count"] == 1
    assert row["common_scored_event_count"] == 1
    assert row["ignored_event_count"] == len(ignored_lanes)


def test_exact_common_collapse_is_eligible_with_duplicate_warning(tmp_path: Path) -> None:
    events = (
        _native_event("14", audio_time_sec=3.0),
        _native_event("15", audio_time_sec=3.0, source_order=1),
    )
    manifest_path = _write_timing_manifest(tmp_path, events)

    outcome = run_reference_set(ReferenceSetRequest(manifest_path, tmp_path / "reference-set"))

    row = _published_rows(outcome)[0]
    assert outcome.exit_code == 0
    assert row["reference_eligibility_status"] == "eligible"
    assert row["reference_eligibility_warnings"] == ["duplicate_common_projection:count=1"]
    assert row["mapped_event_count"] == 2
    assert row["common_scored_event_count"] == 1
    assert row["duplicate_common_event_count"] == 1


def test_unclassified_lane_quarantines_even_when_other_events_map(tmp_path: Path) -> None:
    manifest_path = _write_timing_manifest(
        tmp_path,
        (
            _native_event("13", audio_time_sec=1.0),
            _native_event("2A", audio_time_sec=2.0, source_order=1),
        ),
    )

    outcome = run_reference_set(ReferenceSetRequest(manifest_path, tmp_path / "reference-set"))

    row = _published_rows(outcome)[0]
    assert outcome.exit_code == 1
    assert row["reference_eligibility_status"] == "quarantined"
    assert row["reference_eligibility_reason_codes"] == ["unclassified_reference_lane"]
    assert row["mapped_event_count"] == 1
    assert row["unmapped_event_count"] == 1
    assert row["reference_eligibility_warnings"] == []


def test_no_scored_events_after_reviewed_ignores_quarantines(tmp_path: Path) -> None:
    manifest_path = _write_timing_manifest(tmp_path, (_native_event("54", audio_time_sec=1.0),))

    outcome = run_reference_set(ReferenceSetRequest(manifest_path, tmp_path / "reference-set"))

    row = _published_rows(outcome)[0]
    assert outcome.exit_code == 1
    assert row["reference_eligibility_status"] == "quarantined"
    assert row["reference_eligibility_reason_codes"] == ["no_scored_drum_events"]
    assert row["mapped_event_count"] == 0
    assert row["common_scored_event_count"] == 0
    assert row["ignored_event_count"] == 1
    assert row["reference_eligibility_warnings"] == []


def test_accounting_and_fatal_outcome(tmp_path: Path) -> None:
    ready, quarantined = _timing_rows()
    ready_events = render_reference_events((_native_event("13", audio_time_sec=1.0),))
    ready_hash = hashlib.sha256(ready_events).hexdigest()
    ready["reference_events_cache_path"] = f"events/{ready_hash}.jsonl"
    rendered = render_manifest(
        tuple(
            {key: value for key, value in row.items() if key != "corpus_version"}
            for row in (ready, quarantined)
        )
    )
    manifest_path = tmp_path / "timing" / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(rendered.content)
    event_path = manifest_path.parent.parent / "events" / f"{ready_hash}.jsonl"
    event_path.parent.mkdir()
    event_path.write_bytes(ready_events)

    outcome = run_reference_set(ReferenceSetRequest(manifest_path, tmp_path / "reference-set"))

    assert outcome.eligible_count + outcome.quarantined_count == 2
    assert outcome.exit_code == 1
    assert failed_reference_set_outcome().manifest is None
    assert failed_reference_set_outcome().exit_code == 2


def test_fatal_manifest_or_publication_failure_returns_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = run_reference_set(
        ReferenceSetRequest(tmp_path / "missing.jsonl", tmp_path / "missing-output")
    )
    assert missing.status == "failed"
    assert missing.exit_code == 2
    assert missing.manifest is None

    manifest_path = _write_timing_manifest(
        tmp_path / "publication", (_native_event("13", audio_time_sec=1.0),)
    )

    def fail_publish(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated publication failure")

    monkeypatch.setattr("src.benchmark.reference_set_manifest.publish_manifest", fail_publish)
    published = run_reference_set(
        ReferenceSetRequest(manifest_path, tmp_path / "publication-output")
    )
    assert published.status == "failed"
    assert published.exit_code == 2
    assert published.manifest is None


def test_lineage_and_no_mapped_artifact(tmp_path: Path) -> None:
    event = _native_event("13", audio_time_sec=1.0)
    manifest_path = _write_timing_manifest(tmp_path, (event,))
    outcome = run_reference_set(ReferenceSetRequest(manifest_path, tmp_path / "reference-set"))
    row = _published_rows(outcome)[0]
    assert (
        row["source_reference_timing_manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert row["source_reference_timing_version"].startswith("sha256:")
    assert row["taxonomy_version"] == "crux.drum-taxonomy/v1"
    assert row["lane_map_version"] == "crux.dtx-lane-map/v1"
    assert "mapped_reference_events_path" not in row
    assert not list((tmp_path / "reference-set").rglob("*mapped*"))


def test_schema_golden_validator_accepts_registered_shape(tmp_path: Path) -> None:
    ready, quarantined = _timing_rows()
    ready["reference_events_cache_path"] = "events/" + "a" * 64 + ".jsonl"
    for row in (ready, quarantined):
        row.pop("corpus_version", None)
        row["schema_version"] = BENCHMARK_REFERENCE_MANIFEST_SCHEMA
        row["source_reference_timing_manifest_sha256"] = "b" * 64
        row["source_reference_timing_version"] = "sha256:" + "c" * 64
        row["taxonomy_version"] = "crux.drum-taxonomy/v1"
        row["lane_map_version"] = "crux.dtx-lane-map/v1"
    ready.update(
        {
            "reference_eligibility_status": "eligible",
            "reference_eligibility_reason_codes": [],
            "reference_eligibility_warnings": ["ignored_reference_lane:54:count=1"],
            "mapped_event_count": 1,
            "common_scored_event_count": 1,
            "ignored_event_count": 1,
            "unmapped_event_count": 0,
            "duplicate_common_event_count": 0,
        }
    )
    quarantined.update(
        {
            "reference_eligibility_status": "quarantined",
            "reference_eligibility_reason_codes": ["upstream_reference_unavailable"],
            "reference_eligibility_warnings": [],
            "mapped_event_count": 0,
            "common_scored_event_count": 0,
            "ignored_event_count": 0,
            "unmapped_event_count": 0,
            "duplicate_common_event_count": 0,
        }
    )
    rendered = render_manifest(tuple((ready, quarantined)))

    validate_schema_golden(BENCHMARK_REFERENCE_MANIFEST_SCHEMA, rendered.content)
