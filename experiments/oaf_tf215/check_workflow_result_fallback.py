#!/usr/bin/env python3
"""Focused check for the OaF TF2.15 workflow's always-on result fallback.

The workflow's "Record result in job summary" step must synthesize a minimal
failed record when an earlier stage failed before ``graph_check.py`` wrote one,
so the compact artifact and job summary never disappear behind a failed
install step. This script extracts that step's shell block from the committed
workflow and runs it for the setup, sync, check, graph and success paths.

Local check only; it is not executed by CI (the workflow itself is the e2e
check). Standard library only::

    python3 experiments/oaf_tf215/check_workflow_result_fallback.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/oaf-tf215-probe.yml"
RESULT_PATH = Path("artifacts/experiments/oaf-tf215/graph-check.json")
STEP_NAME = "Record result in job summary"
STEP_MARKER = f"      - name: {STEP_NAME}"
SCHEMA = "oaf-tf215-graph-check/v1"


def extract_run_block(text: str) -> str:
    """Return the shell script of the fallback step without a YAML parser."""
    lines = text.splitlines()
    try:
        start = lines.index(STEP_MARKER)
    except ValueError as error:
        raise AssertionError(f"step {STEP_NAME!r} not found") from error
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("      - name:"):
            end = index
            break
    for index in range(start + 1, end):
        if lines[index].strip() == "run: |":
            block: list[str] = []
            indent = 0
            for line in lines[index + 1 : end]:
                if not line.strip():
                    block.append("")
                    continue
                line_indent = len(line) - len(line.lstrip())
                if not block:
                    indent = line_indent
                if line_indent < indent:
                    break
                block.append(line[indent:])
            return "\n".join(block).rstrip("\n") + "\n"
    raise AssertionError(f"run block not found in step {STEP_NAME!r}")


def check_workflow_wiring(text: str) -> None:
    """The outcomes are wired, the step always runs, nothing masks failures."""
    assert "continue-on-error" not in text, "a step must not mask its failure"
    for step_id in ("sync", "check", "graph"):
        assert f"        id: {step_id}" in text, f"missing id for {step_id}"
        assert f"${{{{ steps.{step_id}.outcome }}}}" in text, f"missing outcome for {step_id}"
    assert f"      - name: {STEP_NAME}\n        if: always()" in text
    assert "if-no-files-found: warn" in text


def run_case(
    script: str,
    outcomes: tuple[str, str, str],
    existing: str | None = None,
) -> tuple[dict[str, object], str, str]:
    """Run the extracted step in a temp workspace and return (result, summary, raw)."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        summary_path = workdir / "summary.md"
        summary_path.write_text("", encoding="utf-8")
        result_path = workdir / RESULT_PATH
        if existing is not None:
            result_path.parent.mkdir(parents=True)
            result_path.write_text(existing, encoding="utf-8")
        env = {
            "PATH": os.environ["PATH"],
            "SYNC_OUTCOME": outcomes[0],
            "CHECK_OUTCOME": outcomes[1],
            "GRAPH_OUTCOME": outcomes[2],
            "GITHUB_STEP_SUMMARY": str(summary_path),
            "GITHUB_SHA": "abc123",
            "GITHUB_RUN_ID": "42",
            "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64",
        }
        completed = subprocess.run(
            ["bash", "-c", script],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            f"record step failed for outcomes={outcomes}: {completed.stderr}"
        )
        assert result_path.is_file(), f"no compact result for outcomes={outcomes}"
        raw = result_path.read_text(encoding="utf-8")
        return json.loads(raw), summary_path.read_text(encoding="utf-8"), raw


def assert_synthesized(
    record: dict, summary: str, stage: str, outcomes: tuple[str, str, str]
) -> None:
    """A synthesized record must name the failed stage and never look successful."""
    assert record["schema"] == SCHEMA
    assert record["status"] == "failed", record
    assert record["failed_stage"] == stage, record
    assert record["ci"]["GITHUB_SHA"] == "abc123"
    assert record["ci"]["GITHUB_RUN_ID"] == "42"
    (stage_record,) = record["stages"]
    assert stage_record["name"] == stage
    assert stage_record["status"] == "failed"
    sync, check, graph = outcomes
    assert f"sync={sync} check={check} graph={graph}" in stage_record["error"]
    assert f'"failed_stage": "{stage}"' in summary
    assert '"status": "failed"' in summary
    assert "## OaF TF2.15 graph check" in summary


def main() -> int:
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    check_workflow_wiring(workflow_text)
    print("ok - workflow wires sync/check/graph outcomes and an always-on step")

    script = extract_run_block(workflow_text)

    record, summary, _ = run_case(script, ("failure", "skipped", "skipped"))
    assert_synthesized(record, summary, "sync", ("failure", "skipped", "skipped"))
    print("ok - sync failure writes a failed sync record and summary")

    record, summary, _ = run_case(script, ("success", "failure", "skipped"))
    assert_synthesized(record, summary, "check", ("success", "failure", "skipped"))
    print("ok - check failure writes a failed check record and summary")

    record, summary, _ = run_case(script, ("success", "success", "failure"))
    assert_synthesized(record, summary, "graph", ("success", "success", "failure"))
    print("ok - graph crash without output writes a failed graph record")

    record, summary, _ = run_case(script, ("skipped", "skipped", "skipped"))
    assert_synthesized(record, summary, "environment", ("skipped", "skipped", "skipped"))
    print("ok - setup failure writes a failed environment record")

    stage_record = (
        '{\n  "schema": "oaf-tf215-graph-check/v1",\n  "status": "failed",\n'
        '  "failed_stage": "slim_batchnorm",\n  "stages": []\n}\n'
    )
    _, summary, raw = run_case(script, ("success", "success", "failure"), stage_record)
    assert raw == stage_record, "the probe's own stage record must never be overwritten"
    assert "slim_batchnorm" in summary
    print("ok - existing graph_check stage failure record is preserved")

    passed = json.dumps({"schema": SCHEMA, "status": "passed", "stages": []}) + "\n"
    _, summary, raw = run_case(script, ("success", "success", "success"), passed)
    assert raw == passed, "a passed result must stay passed"
    assert '"status": "passed"' in summary
    assert "failed_stage" not in summary
    print("ok - passed record is preserved and not replaced by a failure")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
