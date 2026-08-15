"""Reusable synthetic HPA-323/HPA-324 reference fixture for reviewed-subset tests.

Factors the smallest already-working canonical timing/reference/event builders
from the HPA-323/HPA-324/HPA-326 tests: one ready timing row is cloned from the
``crux.reference-timing-manifest/v1`` schema golden per distinct simfile ID, its
event artifact is re-keyed and written under the timing output root, and the
HPA-324 reference manifest is derived through the real ``run_reference_set``
pipeline so every row is eligible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.reference_set_manifest import ReferenceSetRequest, run_reference_set
from src.benchmark.reference_timing import NativeReferenceEvent, render_reference_events

_TIMING_GOLDEN = Path(__file__).parent / "schema_goldens/crux.reference-timing-manifest-v1.jsonl"
_FIRST_SIMFILE_ID = 100


@dataclass(frozen=True)
class ReviewedSubsetReferenceFixture:
    reference_manifest_path: Path
    timing_manifest_path: Path
    timing_output_root: Path


def build_reviewed_subset_reference_fixture(
    tmp_path: Path,
    *,
    eligible_count: int = 36,
    reverse_rows: bool = False,
) -> ReviewedSubsetReferenceFixture:
    """Build an all-eligible HPA-323 timing + HPA-324 reference population.

    ``eligible_count`` controls the population size: one ready timing row per
    distinct simfile ID, each backed by a valid events artifact.  With
    ``reverse_rows=True`` the timing rows are published in reverse input order
    (and the derived reference manifest inherits that order) so consumers can
    prove candidate selection is order-independent.
    """
    ready_row = json.loads(_TIMING_GOLDEN.read_text(encoding="utf-8").splitlines()[0])
    timing_rows: list[dict[str, object]] = []
    event_artifacts: list[tuple[Path, bytes]] = []
    for offset in range(eligible_count):
        simfile_id = _FIRST_SIMFILE_ID + offset
        event = NativeReferenceEvent(
            simfile_id=simfile_id,
            selected_chart_key=f"{simfile_id}/real.dtx",
            selected_chart_content_hash=ready_row["selected_chart_content_hash"],
            source_audio_key=f"{simfile_id}/bgm.wav",
            source_audio_content_hash=ready_row["source_audio_content_hash"],
            source_order=0,
            measure=1,
            position=0.0,
            lane_id="13",
            note_id="01",
            chart_time_sec=1.0,
            audio_time_sec=0.5,
        )
        event_content = render_reference_events((event,))
        event_hash = sha256(event_content).hexdigest()
        row = dict(ready_row)
        row.pop("corpus_version", None)
        row["simfile_id"] = simfile_id
        row["object_prefix"] = f"{simfile_id}/"
        row["objects"] = [
            {**remote, "key": f"{simfile_id}/real.dtx"} for remote in ready_row["objects"]
        ]
        row["selected_chart_key"] = f"{simfile_id}/real.dtx"
        row["source_audio_key"] = f"{simfile_id}/bgm.wav"
        row["reference_events_cache_path"] = f"events/{event_hash}.jsonl"
        timing_rows.append(row)
        event_artifacts.append((Path("events") / f"{event_hash}.jsonl", event_content))

    source_rows = tuple(reversed(timing_rows) if reverse_rows else timing_rows)
    rendered = render_manifest(source_rows)
    timing_root = tmp_path / "timing"
    timing_path = timing_root / "manifests" / f"{rendered.manifest_sha256}.jsonl"
    timing_path.parent.mkdir(parents=True)
    timing_path.write_bytes(rendered.content)
    for relative, content in event_artifacts:
        artifact_path = timing_root / relative
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(content)

    outcome = run_reference_set(ReferenceSetRequest(timing_path, tmp_path / "reference-set"))
    if outcome.manifest is None or outcome.exit_code != 0:
        raise ValueError("reviewed subset fixture must produce an all-eligible reference manifest")
    return ReviewedSubsetReferenceFixture(
        reference_manifest_path=outcome.manifest.path,
        timing_manifest_path=timing_path,
        timing_output_root=timing_root,
    )
