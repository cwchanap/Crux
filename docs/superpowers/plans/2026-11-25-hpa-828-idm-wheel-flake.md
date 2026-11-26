# HPA-828 — Focused revalidation and closeout

[Design and evidence](../specs/2026-11-25-hpa-828-idm-wheel-flake-design.md) · [Linear HPA-828](https://linear.app/cwchanap/issue/HPA-828/revalidate-intermittent-idm-wheel-builder-test-failure)

The timestamp defect already has a fix and regression guard on main. Keep this PR documentation-only unless a new reproducible failure requires a separate decision. The previous repeated/parallel test matrix is superseded.

## 1. Confirm the diagnosis

- [x] Compare the historical `7975d80` helper and byte-equality assertion, the `ef43d75` fix, and current `2825860` guard.
- [x] Verify the two-second ZIP timestamp mechanism with a minimal scratch experiment; consult the reviewer's full historical/current test reproduction.
- [x] Verify both existing test lanes passed in [PR #36 CI](https://github.com/cwchanap/Crux/actions/runs/45673844492). No additional full suite or stress load is needed.

Optional reproducibility recipe (adapted from the linked review; scratch only, not a committed test). Use Python 3.11 or 3.12 with pytest and Git installed, from a clone with the historical revision available:

```sh
S=$(mktemp -d)
for rev in 7975d80 282586003c17923b97d8b0f33ac062b5f5328b90; do
  mkdir -p "$S/$rev/runtime/idm" "$S/$rev/tests/runtime"
  git show "${rev}:runtime/idm/build_pinned_wheel.py" > "$S/$rev/runtime/idm/build_pinned_wheel.py"
  git show "${rev}:tests/runtime/test_idm_wheel_builder.py" > "$S/$rev/tests/runtime/test_idm_wheel_builder.py"
done
python - "$S" <<'PY'
import importlib.util
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
for rev in ("7975d80", "282586003c17923b97d8b0f33ac062b5f5328b90"):
    f = root / rev / "tests/runtime/test_idm_wheel_builder.py"
    spec = importlib.util.spec_from_file_location("test_wheel", f)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real = zipfile.time
    ticks = iter(range(1700000000, 1700000100, 2))
    zipfile.time = types.SimpleNamespace(
        time=lambda: next(ticks), localtime=time.localtime
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            module.test_commit_blobs_ignore_staged_and_unstaged_source_edits(
                Path(directory)
            )
    except AssertionError as error:
        import traceback
        traceback.print_exc()
        if rev != "7975d80":
            raise
        frame = traceback.extract_tb(error.__traceback__)[-1]
        if "second_wheel.read_bytes() == first_wheel.read_bytes()" not in (frame.line or ""):
            raise
        print(rev, "expected historical wheel-byte assertion failure")
    else:
        if rev == "7975d80":
            raise AssertionError("expected historical failure did not reproduce")
        print(rev, "pass")
    finally:
        zipfile.time = real
PY
```

The reviewer's original script and 300-run historical sample are already linked in the design. Repeating that sample or adding the clock-step script to the test suite is unnecessary. The recipe only shifts ZIP's clock, not the host clock; the current fixed-timestamp entries are unaffected.

## 2. Run focused confirmation

- [ ] Verify the latest main still contains the timestamp guard. In a clean checkout use existing Python 3.11 and 3.12 environments with pytest. Run once per version, from repository root:

```sh
python -m pytest -q -o addopts=-ra --confcutdir tests/runtime \
  tests/runtime/test_idm_wheel_builder.py::test_commit_blobs_ignore_staged_and_unstaged_source_edits
```

- [ ] Record revision, Python version, command, exit status and pytest output for each. Use an execution-capable MCP runner for this offline E2E scenario, or existing CI runs on the final candidate revision with both jobs verified through MCP. Retain the actual exit output or run/job links; ordinary CI skips draft PR jobs, so do not treat a skipped check as a pass. No new CI jobs, tools or model dependencies.
- [ ] If a new failure occurs, capture the actual assertion or subprocess stderr and keep the issue open. Do not preemptively add diagnostic infrastructure.

## 3. Close out

- [ ] Add the two focused results (or equivalent verified CI evidence on the final revision) to this PR; differentiate reviewer-reported historical runs, independently confirmed mechanism/CI, and final acceptance results.
- [ ] Link the evidence to HPA-828 and close as a diagnosed timestamp defect already fixed by `ef43d75`. Retain the caveat that the original failure traceback was unavailable. Do not close before confirmation.
- [ ] Verify the PR only changes documentation and `git diff --check` passes. No source change, new regression, UI validation, model inference or human acceptance. Same PR throughout.

