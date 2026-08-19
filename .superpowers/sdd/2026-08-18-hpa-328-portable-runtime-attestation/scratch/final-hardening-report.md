# Final attestation hardening report

## Scope

This remediation addresses the three P1 findings from the final whole-branch
review:

- isolate both the standalone environment probe and separator `python -m`
  launches from ambient Python import-discovery overrides;
- reject separator-lock replacement between pilot identity loading and runtime
  attestation, and derive the persisted rows/snapshot/run identity from the
  locks carried by the attested runtimes;
- retain a no-follow descriptor for each attested model root and pass that
  bound identity to the subprocess, including postflight validation and
  explicit/finalizer cleanup.

All coverage is synthetic and offline. No production model, native separator,
Task 11 input, or production evidence artifact was executed or changed.

## TDD evidence

The focused red tests were run before the production changes:

```text
rtk uv run --extra r2 pytest tests/benchmark/test_separators.py -k "environment_probe_uses_isolated or launch_stays_bound or passes_runtime_launch_environment or spleeter_launch_environment" -q
4 failed: the probe/launcher retained ambient Python settings, used no isolated
interpreter flag, and the process contract had no bound model-root descriptor.

rtk uv run --extra r2 pytest tests/benchmark/test_separation_pilot.py -k "lock_replacement_between_identity" -q
1 failed: the pilot returned exit 0 instead of fatal
separator_lock_companion_mismatch and did not reject the replaced lock.
```

The focused regressions are green:

```text
rtk uv run --extra r2 pytest tests/benchmark/test_separators.py -k "environment_probe_uses_isolated or launch_stays_bound or passes_runtime_launch_environment or spleeter_launch_environment or attested_runtime_close" -q
5 passed, 58 deselected

rtk uv run --extra r2 pytest tests/benchmark/test_separation_pilot.py -k "lock_replacement_between_identity" -q
1 passed, 32 deselected
```

The model-root swap regression replaces the verified root alias after
attestation and confirms the fake subprocess receives/uses the held root bytes,
not the replacement bytes. The launch-environment regressions inspect the
actual subprocess argv/env seam and assert `-I`, `pass_fds`, and removal of
`PYTHONPATH`, `PYTHONHOME`, `PYTHONUSERBASE`, and `PYTHONNOUSERSITE`.

## Verification

```text
rtk uv run --extra r2 pytest tests/benchmark/test_separators.py tests/benchmark/test_separation_pilot.py -q
96 passed in 65.73s

rtk uv run --extra r2 pytest tests/benchmark/test_separator_environment_probe.py tests/benchmark/test_separators.py tests/benchmark/test_separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py tests/benchmark/test_separation_comparison.py tests/benchmark/test_separation_handoff.py tests/test_cli_benchmark.py -q
163 passed in 108.24s

rtk uv run --extra r2 pytest -q
2406 passed in 151.18s

rtk uv run ruff check src/benchmark/separators.py src/benchmark/separation_pilot.py scripts/freeze_separator_runtime.py tests/benchmark/test_separators.py tests/benchmark/test_separation_pilot.py
passed

rtk uv run black --check src/benchmark/separators.py src/benchmark/separation_pilot.py scripts/freeze_separator_runtime.py tests/benchmark/test_separators.py tests/benchmark/test_separation_pilot.py
passed

rtk uv run pylint src/benchmark/separators.py src/benchmark/separation_pilot.py scripts/freeze_separator_runtime.py --errors-only
passed

rtk python3 -m py_compile src/benchmark/separators.py src/benchmark/separation_pilot.py scripts/freeze_separator_runtime.py
passed

rtk git diff --check
passed
```

## Limitation

Model-root binding intentionally fails closed on hosts without the required
POSIX descriptor-relative/no-follow primitives and a child-visible fd path.
Darwin uses a child `fchdir` to the inherited directory descriptor because
`/dev/fd/<n>/child` is not traversable there; Linux uses `/proc/self/fd` (or
`/dev/fd`) with `pass_fds`. Native subprocesses still do not receive a generic
OS-level network sandbox, which remains outside this remediation's contract.
