from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github/workflows/oaf-tf215-probe.yml"
SMOKE_WORKFLOW = Path(__file__).parents[2] / ".github/workflows/oaf-smoke.yml"
COMPARE_RESULT = "artifacts/experiments/oaf-tf215/compare.json"
COMPARE_LOG = "artifacts/experiments/oaf-tf215/compare.log"
COMPARE_STEP = "Run full-model comparison (model stage)"
SUMMARY_STEP = "Record full-model result in job summary"
PREPARE_CHECKPOINT = "uv run crux benchmark prepare-backend --backend oaf --download"


def _job_text(content: str, name: str) -> str:
    """Return one job block, from its ``  name:`` line to the next job."""
    marker = f"\n  {name}:\n"
    start = content.index(marker) + 1
    end = len(content)
    for index in range(start + len(marker), len(content)):
        if content.startswith("\n  ", index) and not content.startswith("\n    ", index):
            end = index + 1
            break
    return content[start:end]


def _run_block(job: str, step_name: str) -> str:
    """Return a step's shell script without a YAML parser."""
    lines = job.splitlines()
    marker = f"      - name: {step_name}"
    start = lines.index(marker)
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("      - name:") or (
            lines[index].startswith("  ") and not lines[index].startswith("    ")
        ):
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
    raise AssertionError(f"no run block in step {step_name!r}")


def _run_script(
    script: str, workdir: Path, env: dict[str, str], uv_body: str
) -> subprocess.CompletedProcess[str]:
    """Run an extracted step with a fake ``uv`` first on PATH."""
    bin_dir = workdir / "bin"
    bin_dir.mkdir(exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text(f"#!/usr/bin/env bash\n{uv_body}\n", encoding="utf-8")
    uv.chmod(0o755)
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=workdir,
        env={"PATH": f"{bin_dir}:{os.environ['PATH']}", **env},
        capture_output=True,
        text=True,
        check=False,
    )
    return completed


def _env(workdir: Path) -> dict[str, str]:
    return {
        "PY311": "/tmp/tf215-venv/bin/python",
        "RUNNER_TEMP": f"{workdir}/runner-temp",
        "GITHUB_WORKSPACE": f"{workdir}",
        "GITHUB_ENV": f"{workdir}/github-env",
        "GITHUB_STEP_SUMMARY": f"{workdir}/summary.md",
        "GITHUB_SHA": "abc123",
        "RUNNER_OS": "Linux",
        "RUNNER_ARCH": "X64",
    }


def _model_job() -> str:
    return _job_text(WORKFLOW.read_text(encoding="utf-8"), "model")


def test_probe_workflow_runs_the_graph_job_on_drafts_and_the_model_job_manually() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")
    assert "on:\n  pull_request:" in content
    assert "\n  workflow_dispatch:\n" in content
    graph = _job_text(content, "graph")
    model = _job_text(content, "model")

    assert "tests/runtime/test_oaf_tf215_builder.py" in graph
    assert "tests/benchmark/test_oaf_tf215_compare.py" in graph
    assert "tests/benchmark/test_oaf_tf215_workflow.py" in graph
    assert "graph_check.py" in graph
    assert "prepare-backend" not in graph
    assert "--download" not in graph

    assert "if: github.event_name == 'workflow_dispatch'" in model
    assert "experiments/oaf_tf215/requirements.lock" in model
    assert "experiments/oaf_tf215/build_runtime.py" in model
    assert "uv run experiments/oaf_tf215/compare.py" in model
    assert '--python311 "$PY311"' in model
    assert '--checkpoint-cache "$GITHUB_WORKSPACE/artifacts/benchmark/model-cache"' in model
    assert "secrets." not in model
    assert "continue-on-error" not in content


def test_model_job_uses_the_public_checkpoint_command_from_the_smoke_workflow() -> None:
    smoke = SMOKE_WORKFLOW.read_text(encoding="utf-8")
    model = _model_job()
    assert smoke.count(PREPARE_CHECKPOINT) == 1
    assert model.count(PREPARE_CHECKPOINT) == 1
    assert model.count("--download") == 1
    assert "actions/upload-artifact@v4" in model
    upload = model.split("      - name: Upload full-model result\n", maxsplit=1)[1]
    assert upload.startswith("        if: always()\n")
    assert COMPARE_RESULT in upload
    assert COMPARE_LOG in upload
    assert "if-no-files-found: warn" in upload


def test_model_job_keeps_and_uploads_output_when_the_driver_is_divergent(tmp_path: Path) -> None:
    script = _run_block(_model_job(), COMPARE_STEP)
    divergent = {"schema": "oaf-tf215-compare/v2", "status": "oracle_divergent"}
    completed = _run_script(
        script,
        tmp_path,
        _env(tmp_path),
        (
            "mkdir -p artifacts/experiments/oaf-tf215\n"
            'printf \'%s\\n\' \'{"schema": "oaf-tf215-compare/v2", '
            '"status": "oracle_divergent"}\' > artifacts/experiments/oaf-tf215/compare.json\n'
            "echo divergent-run\n"
            "exit 1"
        ),
    )
    assert completed.returncode == 1, completed.stderr
    result = tmp_path / COMPARE_RESULT
    assert json.loads(result.read_text(encoding="utf-8")) == divergent
    assert "divergent-run" in (tmp_path / COMPARE_LOG).read_text(encoding="utf-8")
    assert "COMPARE_EXIT=1" in (tmp_path / "github-env").read_text(encoding="utf-8")
    assert "divergent-run" in completed.stdout


def test_model_job_reports_a_zero_exit_and_a_missing_record(tmp_path: Path) -> None:
    script = _run_block(_model_job(), COMPARE_STEP)
    env = _env(tmp_path)
    exact = _run_script(
        script,
        tmp_path,
        env,
        (
            "mkdir -p artifacts/experiments/oaf-tf215\n"
            "printf '%s\\n' '{\"status\": \"smoke_exact\"}' > artifacts/experiments/oaf-tf215/compare.json\n"
            "exit 0"
        ),
    )
    assert exact.returncode == 0, exact.stderr
    assert "COMPARE_EXIT=0" in (tmp_path / "github-env").read_text(encoding="utf-8")
    (tmp_path / COMPARE_RESULT).unlink()

    crash = _run_script(
        script,
        tmp_path,
        env,
        "echo crashed-before-writing\nexit 1",
    )
    assert crash.returncode == 1
    assert not (tmp_path / COMPARE_RESULT).exists()
    assert "crashed-before-writing" in (tmp_path / COMPARE_LOG).read_text(encoding="utf-8")


def test_summary_step_preserves_the_record_and_names_pre_compare_failures(
    tmp_path: Path,
) -> None:
    summary_env = {
        **_env(tmp_path),
        "SYNC_OUTCOME": "success",
        "CHECK_OUTCOME": "success",
        "BUILD_OUTCOME": "success",
        "CHECKPOINT_OUTCOME": "failure",
        "COMPARE_OUTCOME": "skipped",
        "COMPARE_EXIT": "",
    }
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("", encoding="utf-8")
    script = _run_block(_model_job(), SUMMARY_STEP)
    result = tmp_path / COMPARE_RESULT

    completed = _run_script(script, tmp_path, summary_env, "exit 0")
    assert completed.returncode == 0, completed.stderr
    summary = summary_path.read_text(encoding="utf-8")
    assert "## OaF TF2.15 full-model comparison" in summary
    assert "No comparison record was produced" in summary
    assert "sync=success check=success build=success checkpoint=failure compare=skipped" in summary
    assert not result.exists()

    divergent = {"schema": "oaf-tf215-compare/v2", "status": "oracle_divergent"}
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(divergent) + "\n", encoding="utf-8")
    completed = _run_script(script, tmp_path, {**summary_env, "COMPARE_EXIT": "1"}, "exit 0")
    assert completed.returncode == 0, completed.stderr
    assert json.loads(result.read_text(encoding="utf-8")) == divergent
    summary = summary_path.read_text(encoding="utf-8")
    assert "compare exit: 1" in summary
    assert '"oracle_divergent"' in summary
