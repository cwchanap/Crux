from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.reference_set import map_reference_events, project_common_reference_events
from src.benchmark.reference_set_manifest import ReferenceSetRequest, run_reference_set
from src.benchmark.reference_timing import (
    NativeReferenceEvent,
    read_reference_events,
    render_reference_events,
)
from src.benchmark.taxonomy import ClassMapping

_TIMING_GOLDEN = Path(__file__).parent / "schema_goldens/crux.reference-timing-manifest-v1.jsonl"


def _timing_rows() -> tuple[dict[str, object], dict[str, object]]:
    return tuple(
        json.loads(line) for line in _TIMING_GOLDEN.read_text(encoding="utf-8").splitlines()
    )  # type: ignore[return-value]


def _native_event(
    simfile_id: int,
    lane_id: str,
    *,
    audio_time_sec: float,
    source_order: int,
) -> NativeReferenceEvent:
    return NativeReferenceEvent(
        simfile_id=simfile_id,
        selected_chart_key=f"{simfile_id}/real.dtx",
        selected_chart_content_hash="5dbd7639514e76fbd11cc8d6c518adae8b168ea944461c555236dadb1a3a4809",
        source_audio_key=f"{simfile_id}/bgm.wav",
        source_audio_content_hash="c" * 64,
        source_order=source_order,
        measure=1,
        position=0.0,
        lane_id=lane_id,
        note_id=f"{source_order + 1:02X}",
        chart_time_sec=audio_time_sec,
        audio_time_sec=audio_time_sec,
    )


def _ready_row(
    template: dict[str, object],
    *,
    simfile_id: int,
    event_path: str,
) -> dict[str, object]:
    row = copy.deepcopy(template)
    row["simfile_id"] = simfile_id
    row["object_prefix"] = f"{simfile_id}/"
    objects = row["objects"]
    assert isinstance(objects, list) and len(objects) == 1
    object_row = copy.deepcopy(objects[0])
    assert isinstance(object_row, dict)
    object_row["key"] = f"{simfile_id}/real.dtx"
    row["objects"] = [object_row]
    row["selected_chart_key"] = f"{simfile_id}/real.dtx"
    row["source_audio_key"] = f"{simfile_id}/bgm.wav"
    row["reference_events_cache_path"] = event_path
    return row


def _upstream_row(template: dict[str, object], *, simfile_id: int) -> dict[str, object]:
    row = copy.deepcopy(template)
    row["simfile_id"] = simfile_id
    row["object_prefix"] = f"{simfile_id}/"
    return row


def _write_hpa323_fixture(tmp_path: Path) -> tuple[Path, Path, dict[int, bytes]]:
    ready_template, quarantined_template = _timing_rows()
    events_by_id = {
        42: (
            _native_event(42, "13", audio_time_sec=1.0, source_order=0),
            _native_event(42, "12", audio_time_sec=2.0, source_order=1),
        ),
        44: (
            _native_event(44, "14", audio_time_sec=3.0, source_order=0),
            _native_event(44, "15", audio_time_sec=3.0, source_order=1),
        ),
    }
    event_contents: dict[int, bytes] = {
        simfile_id: render_reference_events(events) for simfile_id, events in events_by_id.items()
    }
    rows = []
    for simfile_id, content in event_contents.items():
        event_hash = hashlib.sha256(content).hexdigest()
        rows.append(
            _ready_row(
                ready_template,
                simfile_id=simfile_id,
                event_path=f"events/{event_hash}.jsonl",
            )
        )
    rows.append(_upstream_row(quarantined_template, simfile_id=46))
    rendered = render_manifest(
        tuple({key: value for key, value in row.items() if key != "corpus_version"} for row in rows)
    )

    timing_root = tmp_path / "reference-timing"
    manifest_path = timing_root / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(rendered.content)
    for simfile_id, content in event_contents.items():
        event_hash = hashlib.sha256(content).hexdigest()
        event_path = timing_root / "events" / f"{event_hash}.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_bytes(content)
    return manifest_path, timing_root, event_contents


def _published_rows(manifest_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]


def test_offline_reference_set_acceptance_reconstructs_and_remaps_native_events(
    tmp_path: Path,
) -> None:
    timing_manifest, timing_root, native_contents = _write_hpa323_fixture(tmp_path)
    output_dir = tmp_path / "reference-set"

    outcome = run_reference_set(ReferenceSetRequest(timing_manifest, output_dir))

    assert outcome.eligible_count == 2
    assert outcome.quarantined_count == 1
    assert outcome.exit_code == 1
    assert outcome.manifest is not None
    rows = _published_rows(outcome.manifest.path)
    eligible_rows = [row for row in rows if row["reference_eligibility_status"] == "eligible"]
    assert len(eligible_rows) == 2
    duplicate_row = next(row for row in eligible_rows if row["simfile_id"] == 44)
    assert duplicate_row["reference_eligibility_warnings"] == [
        "duplicate_common_projection:count=1"
    ]
    assert duplicate_row["duplicate_common_event_count"] == 1

    # HPA-325 reconstruction consumes only the inherited native event artifact;
    # no HPA-324 mapped event artifact is persisted or required.
    for row in eligible_rows:
        relative = row["reference_events_cache_path"]
        assert isinstance(relative, str)
        native_path = timing_root.joinpath(*Path(relative).parts)
        native_bytes = native_path.read_bytes()
        native_hash = hashlib.sha256(native_bytes).hexdigest()
        events = read_reference_events(native_bytes)
        mapped = map_reference_events(events)
        common = project_common_reference_events(mapped.mapped_events)
        assert len(mapped.mapped_events) == row["mapped_event_count"]
        assert len(common) == row["common_scored_event_count"]
        assert native_bytes == native_contents[row["simfile_id"]]
        assert native_hash == hashlib.sha256(native_contents[row["simfile_id"]]).hexdigest()
        assert not list(output_dir.rglob("*mapped*"))

    # The native bytes remain immutable while a test-only lane map changes the
    # derived classes; remapping is pure and needs no audio/model inference.
    row = next(row for row in eligible_rows if row["simfile_id"] == 42)
    relative = row["reference_events_cache_path"]
    assert isinstance(relative, str)
    native_path = timing_root.joinpath(*Path(relative).parts)
    before = native_path.read_bytes()
    assert before == native_contents[42]
    alternate_map = {
        "13": ClassMapping("snare", "snare"),
        "12": ClassMapping("kick", "kick"),
    }
    default_result = map_reference_events(read_reference_events(before))
    alternate_result = map_reference_events(
        read_reference_events(before), lane_map=alternate_map, ignored_lanes=frozenset()
    )
    assert default_result.mapped_events != alternate_result.mapped_events
    assert native_path.read_bytes() == before
    assert (
        hashlib.sha256(native_path.read_bytes()).hexdigest() == hashlib.sha256(before).hexdigest()
    )
