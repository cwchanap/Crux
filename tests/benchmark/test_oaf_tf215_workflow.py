"""The probe workflow's PR/manual wiring and extracted shell steps.

The compare step is executed from the committed workflow with a fake ``uv``
that records its argv and writes to the requested ``--output``. The recorded
argv is parsed by the real ``compare.py`` parser and compared with the expected
resolved paths, so dropping a required flag or writing elsewhere cannot stay
green.

The graph summary step is executed for early failures and existing results.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/oaf-tf215-probe.yml"
SMOKE_WORKFLOW = REPO_ROOT / ".github/workflows/oaf-smoke.yml"
GRAPH_RESULT = "artifacts/experiments/oaf-tf215/graph-check.json"
GRAPH_SCHEMA = "oaf-tf215-graph-check/v1"
GRAPH_SUMMARY_STEP = "Record result in job summary"
GRAPH_STEPS = ("sync", "check", "build", "driver", "graph")
COMPARE_PATH = REPO_ROOT / "experiments/oaf_tf215/compare.py"
COMPARE_SCRIPT = "experiments/oaf_tf215/compare.py"
COMPARE_RESULT = "artifacts/experiments/oaf-tf215/compare.json"
COMPARE_LOG = "artifacts/experiments/oaf-tf215/compare.log"
BUILD_STEP = "Build scratch runtime and import it (build stage)"
COMPARE_STEP = "Run full-model comparison (model stage)"
SUMMARY_STEP = "Record full-model result in job summary"
PREPARE_CHECKPOINT = "uv run crux benchmark prepare-backend --backend oaf --download"
ARGV_FILE = "uv-argv"


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
        ["bash", "--noprofile", "--norc", "-c", script],
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
        "FAKE_UV_ARGV": f"{workdir}/{ARGV_FILE}",
    }


def _model_job() -> str:
    return _job_text(WORKFLOW.read_text(encoding="utf-8"), "model")


def _graph_job() -> str:
    return _job_text(WORKFLOW.read_text(encoding="utf-8"), "graph")


def _load_compare() -> ModuleType:
    spec = importlib.util.spec_from_file_location("crux_oaf_tf215_compare_workflow", COMPARE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {COMPARE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recorded_arguments(workdir: Path) -> list[str]:
    raw = (workdir / ARGV_FILE).read_bytes()
    return [value.decode("utf-8") for value in raw.split(b"\0") if value]


def _assert_compare_command(workdir: Path) -> None:
    """The extracted command must be the exact, parseable driver invocation."""
    env = _env(workdir)
    workspace = env["GITHUB_WORKSPACE"]
    runner_temp = env["RUNNER_TEMP"]
    expected = [
        "run",
        COMPARE_SCRIPT,
        "--python311",
        env["PY311"],
        "--runtime-root",
        f"{runner_temp}/oaf-tf215-scratch/runtime",
        "--checkpoint-cache",
        f"{workspace}/artifacts/benchmark/model-cache",
        "--fixture",
        f"{workspace}/tests/fixtures/oaf_tf1_smoke/canonical.wav",
        "--oracle",
        f"{workspace}/tests/fixtures/oaf_tf1_smoke/smoke-oracle.json",
        "--output",
        COMPARE_RESULT,
    ]
    recorded = _recorded_arguments(workdir)
    assert recorded == expected, recorded

    parsed = _load_compare().parse_args(recorded[2:])
    assert parsed.python311 == Path(env["PY311"])
    assert parsed.runtime_root == Path(f"{runner_temp}/oaf-tf215-scratch/runtime")
    assert parsed.checkpoint_cache == Path(f"{workspace}/artifacts/benchmark/model-cache")
    assert parsed.fixture == Path(f"{workspace}/tests/fixtures/oaf_tf1_smoke/canonical.wav")
    assert parsed.oracle == Path(f"{workspace}/tests/fixtures/oaf_tf1_smoke/smoke-oracle.json")
    assert parsed.output == Path(COMPARE_RESULT)
    assert all(
        path.is_absolute()
        for path in (
            parsed.python311,
            parsed.runtime_root,
            parsed.checkpoint_cache,
            parsed.fixture,
            parsed.oracle,
        )
    )


def test_probe_workflow_runs_the_graph_job_on_drafts_and_the_model_job_manually() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")
    assert "on:\n  pull_request:" in content
    assert "\n  workflow_dispatch:\n" in content
    graph = _job_text(content, "graph")
    model = _job_text(content, "model")

    assert "tests/runtime/test_oaf_tf215_builder.py" in graph
    assert "tests/benchmark/test_oaf_tf215_compare.py" in graph
    assert "tests/benchmark/test_oaf_tf215_workflow.py" in graph
    for name in (
        "graph_check",
        "runtime_identity",
        "panel_metrics",
        "selected_inputs",
        "selected_metrics",
        "selected_compare",
    ):
        path = f"tests/benchmark/test_oaf_tf215_{name}.py"
        assert path in _run_block(graph, "Run host driver and workflow tests (driver stage)")
        assert path in content.split("\n  workflow_dispatch:", maxsplit=1)[0]
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


def test_probe_workflow_triggers_when_frozen_runtime_changes() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")
    pull_request = content.split("\n  workflow_dispatch:", maxsplit=1)[0]
    assert '      - "runtime/oaf_tf1/**"' in pull_request


def test_graph_job_runs_builder_tests_outside_the_locked_worker(tmp_path: Path) -> None:
    worker = tmp_path / "worker-python"
    worker.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\0\' "$@" >> "$FAKE_WORKER_COMMANDS"\n',
        encoding="utf-8",
    )
    worker.chmod(0o755)
    worker_commands = tmp_path / "worker-commands"
    env = {
        **_env(tmp_path),
        "PY311": str(worker),
        "NUMBA_CACHE_DIR": str(tmp_path / "numba-cache"),
        "FAKE_WORKER_COMMANDS": str(worker_commands),
    }
    completed = _run_script(
        _run_block(_graph_job(), BUILD_STEP),
        tmp_path,
        env,
        'printf \'%s\\0\' "$@" >> "$FAKE_UV_ARGV"',
    )
    assert completed.returncode == 0, completed.stderr
    uv_args = _recorded_arguments(tmp_path)
    assert uv_args[0] == "run"
    assert "--isolated" in uv_args
    assert "--no-project" in uv_args
    assert uv_args[uv_args.index("--python") + 1] == "3.11"
    assert uv_args[uv_args.index("--with") + 1] == "pytest"
    assert uv_args[-1] == "tests/runtime/test_oaf_tf215_builder.py"
    assert env["PY311"] not in uv_args
    worker_args = worker_commands.read_bytes().split(b"\0")
    assert b"experiments/oaf_tf215/build_runtime.py" in worker_args
    assert b"pytest" not in worker_args


def test_graph_job_always_records_and_uploads_the_result_with_stage_outcomes() -> None:
    graph = _graph_job()
    summary = graph.split(f"      - name: {GRAPH_SUMMARY_STEP}\n", maxsplit=1)[1]
    summary, upload = summary.split("      - name: Upload compact result\n", maxsplit=1)
    assert summary.startswith("        if: always()\n")
    for step in GRAPH_STEPS:
        assert f"        id: {step}\n" in graph
        assert f"          {step.upper()}_OUTCOME: ${{{{ steps.{step}.outcome }}}}" in summary
    assert upload.startswith("        if: always()\n")
    assert "actions/upload-artifact@v4" in upload
    assert f"path: {GRAPH_RESULT}" in upload
    assert "if-no-files-found: warn" in upload


def _run_graph_summary(workdir: Path, outcomes: tuple[str, ...]) -> str:
    outcome_env = {
        f"{step.upper()}_OUTCOME": outcome
        for step, outcome in zip(GRAPH_STEPS, outcomes, strict=True)
    }
    completed = _run_script(
        _run_block(_graph_job(), GRAPH_SUMMARY_STEP),
        workdir,
        {**_env(workdir), "GITHUB_RUN_ID": "42", **outcome_env},
        "exit 127",
    )
    assert completed.returncode == 0, completed.stderr
    return (workdir / "summary.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("stage", "outcomes"),
    [
        ("sync", ("failure", "skipped", "skipped", "skipped", "skipped")),
        ("check", ("success", "failure", "skipped", "skipped", "skipped")),
        ("build", ("success", "success", "failure", "skipped", "skipped")),
        ("driver", ("success", "success", "success", "failure", "skipped")),
        ("graph", ("success", "success", "success", "success", "failure")),
        ("environment", ("skipped", "skipped", "skipped", "skipped", "skipped")),
    ],
)
def test_graph_summary_records_failure_when_result_is_missing(
    tmp_path: Path, stage: str, outcomes: tuple[str, ...]
) -> None:
    summary = _run_graph_summary(tmp_path, outcomes)
    raw = (tmp_path / GRAPH_RESULT).read_text(encoding="utf-8")
    record = json.loads(raw)
    assert record["schema"] == GRAPH_SCHEMA
    assert record["status"] == "failed"
    assert record["failed_stage"] == stage
    assert record["ci"] == {
        "GITHUB_SHA": "abc123",
        "GITHUB_RUN_ID": "42",
        "RUNNER_OS": "Linux",
        "RUNNER_ARCH": "X64",
    }
    (stage_record,) = record["stages"]
    assert stage_record["name"] == stage
    assert stage_record["status"] == "failed"
    expected_outcomes = " ".join(
        f"{step}={outcome}" for step, outcome in zip(GRAPH_STEPS, outcomes, strict=True)
    )
    assert expected_outcomes in stage_record["error"]
    assert "## OaF TF2.15 graph check" in summary
    assert f"```json\n{raw}```" in summary


@pytest.mark.parametrize(
    ("graph_outcome", "record"),
    [
        (
            "failure",
            {
                "schema": GRAPH_SCHEMA,
                "status": "failed",
                "failed_stage": "slim_batchnorm",
                "stages": [],
            },
        ),
        ("success", {"schema": GRAPH_SCHEMA, "status": "passed", "stages": []}),
    ],
)
def test_graph_summary_preserves_an_existing_record(
    tmp_path: Path, graph_outcome: str, record: dict[str, object]
) -> None:
    result = tmp_path / GRAPH_RESULT
    result.parent.mkdir(parents=True)
    raw = json.dumps(record, indent=2) + "\n"
    result.write_text(raw, encoding="utf-8")

    summary = _run_graph_summary(
        tmp_path, ("success", "success", "success", "success", graph_outcome)
    )
    assert result.read_text(encoding="utf-8") == raw
    assert f"```json\n{raw}```" in summary


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


def _fake_uv(status: str, exit_code: int, *, write_output: bool = True) -> str:
    """Record argv, then write the requested --output like compare.py does."""
    body = (
        'printf \'%s\\0\' "$@" > "$FAKE_UV_ARGV"\n'
        'out=""\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        '    --output) out="$2"; shift 2 ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
    )
    if write_output:
        record = json.dumps({"schema": "oaf-tf215-compare/v2", "status": status})
        body += f'mkdir -p "$(dirname "$out")"\nprintf \'%s\\n\' \'{record}\' > "$out"\n'
    return body + f"echo {status}-run\nexit {exit_code}"


def test_model_job_keeps_and_uploads_output_when_the_driver_is_divergent(tmp_path: Path) -> None:
    script = _run_block(_model_job(), COMPARE_STEP)
    divergent = {"schema": "oaf-tf215-compare/v2", "status": "oracle_divergent"}
    completed = _run_script(script, tmp_path, _env(tmp_path), _fake_uv("oracle_divergent", 1))
    _assert_compare_command(tmp_path)
    assert completed.returncode == 1, completed.stderr
    result = tmp_path / COMPARE_RESULT
    assert json.loads(result.read_text(encoding="utf-8")) == divergent
    assert "oracle_divergent-run" in (tmp_path / COMPARE_LOG).read_text(encoding="utf-8")
    assert "COMPARE_EXIT=1" in (tmp_path / "github-env").read_text(encoding="utf-8")
    assert "oracle_divergent-run" in completed.stdout


def test_model_job_reports_a_zero_exit_and_a_missing_record(tmp_path: Path) -> None:
    script = _run_block(_model_job(), COMPARE_STEP)
    env = _env(tmp_path)
    exact = _run_script(script, tmp_path, env, _fake_uv("smoke_exact", 0))
    _assert_compare_command(tmp_path)
    assert exact.returncode == 0, exact.stderr
    assert json.loads((tmp_path / COMPARE_RESULT).read_text(encoding="utf-8")) == {
        "schema": "oaf-tf215-compare/v2",
        "status": "smoke_exact",
    }
    assert "COMPARE_EXIT=0" in (tmp_path / "github-env").read_text(encoding="utf-8")
    (tmp_path / COMPARE_RESULT).unlink()

    crash = _run_script(
        script, tmp_path, env, _fake_uv("crashed-before-writing", 1, write_output=False)
    )
    _assert_compare_command(tmp_path)
    assert crash.returncode == 1
    assert not (tmp_path / COMPARE_RESULT).exists()
    assert "crashed-before-writing-run" in (tmp_path / COMPARE_LOG).read_text(encoding="utf-8")


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
