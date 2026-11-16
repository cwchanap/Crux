"""The probe workflow's PR/manual wiring and its extracted model step.

The compare step is executed from the committed workflow with a fake ``uv``
that records its argv and writes to the requested ``--output``. The recorded
argv is parsed by the real ``compare.py`` parser and compared with the expected
resolved paths, so dropping a required flag or writing elsewhere cannot stay
green.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/oaf-tf215-probe.yml"
SMOKE_WORKFLOW = REPO_ROOT / ".github/workflows/oaf-smoke.yml"
COMPARE_PATH = REPO_ROOT / "experiments/oaf_tf215/compare.py"
COMPARE_SCRIPT = "experiments/oaf_tf215/compare.py"
COMPARE_RESULT = "artifacts/experiments/oaf-tf215/compare.json"
COMPARE_LOG = "artifacts/experiments/oaf-tf215/compare.log"
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
        "FAKE_UV_ARGV": f"{workdir}/{ARGV_FILE}",
    }


def _model_job() -> str:
    return _job_text(WORKFLOW.read_text(encoding="utf-8"), "model")


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
