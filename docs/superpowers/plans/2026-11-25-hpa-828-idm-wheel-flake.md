# HPA-828 — Revalidation record and closeout

[Design and evidence](../specs/2026-11-25-hpa-828-idm-wheel-flake-design.md) · [Linear HPA-828](https://linear.app/cwchanap/issue/HPA-828/revalidate-intermittent-idm-wheel-builder-test-failure) · [Round 2 review](https://github.com/cwchanap/Crux/pull/37#pullrequestreview-6068230588)

## Completed verification

- Historical helper, fix and current regression guard inspected. A controlled ZIP timestamp experiment confirmed the mechanism; the reviewer reproduced the full historical assertion failure and current pass.
- Main push CI [45741055726](https://github.com/cwchanap/Crux/actions/runs/45741055726) verified at exact checkout `282586003c17923b97d8b0f33ac062b5f5328b90`: Python 3.11 runtime job [139645873722](https://github.com/cwchanap/Crux/actions/runs/45741055726/job/139645873722), 32 passed; Python 3.12 full suite job [139645874023](https://github.com/cwchanap/Crux/actions/runs/45741055726/job/139645874023), 3,835 passed. Both contain the existing regression; detailed commands/results are recorded in the design.
- No further local execution, stress testing or code changes are needed. The two files in this PR record the evidence only. Normal PR checks apply on the final documentation commit; close the linked issue after merge. The original incident traceback is unavailable, so attribution to the now-fixed timestamp defect is strongly supported rather than proven for that occurrence.

## Reproduction reference

Optional reproducibility recipe (adapted from the linked review; scratch only, not a committed test). Use the existing project environment (Python 3.12 with pytest and Git installed) from a clone with the historical revision available. The recipe is a one-off shell command; no permanent harness is required:

```sh
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
for rev in 7975d80 282586003c17923b97d8b0f33ac062b5f5328b90; do
  mkdir -p "$S/$rev/runtime/idm" "$S/$rev/tests/runtime"
  git show "${rev}:runtime/idm/build_pinned_wheel.py" > "$S/$rev/runtime/idm/build_pinned_wheel.py"
  git show "${rev}:tests/runtime/test_idm_wheel_builder.py" > "$S/$rev/tests/runtime/test_idm_wheel_builder.py"
done
uv run python - "$S" <<'PY'
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
