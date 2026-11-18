# OaF TF 2.15 experiment — recorded results

Evidence record for the HPA-856 / PR #36 Python 3.11 / TensorFlow 2.15.1
prototype, ending at the Task 6 internal review checkpoint. It contains only
executed evidence: GitHub Actions runs and native Linux x86-64 Colab CPU runs.
Nothing here is a parity claim, and a commit that only changes classification,
tests or documentation is not presented as re-executed model evidence.

- Model evidence was produced at `c3ffbef5c553e9277c2b90bf94722232f2f848a9`.
- `0537cb4f03eff4e601e0c3edc593db19bd31c12e` changed only
  `tests/benchmark/test_oaf_tf215_workflow.py`.
- This Task 6 documentation commit changes no code relative to those commits.

## Status

**Repeatable `oracle_divergent` on both hosts; not exact.** On GitHub Actions
`ubuntu-24.04` and on a fresh native Colab CPU VM, the candidate restores the
frozen checkpoint and produces three run-identical native event runs per host
(two requests through one worker, then one through a fresh worker). Time and
confidence fields differ from the frozen TF1 smoke oracle; event identity
(class, note, bin, 8-hit group, velocity) matches. The two hosts are **not
bit-identical to each other** (event 0 confidence differs), so no cross-host
equality is claimed. No TF1 cache or runtime identity is reused.

| Item | Status |
| --- | --- |
| Linux x86-64 lock sync/check, scratch build, drum/worker imports | Passed (GitHub Actions and Colab) |
| Executed graph sessions (Slim batch-norm, legacy LSTM) | Passed on both hosts |
| Public checkpoint cache verification | Passed on both hosts |
| Restore / frozen 130/78/52 partition checks | Passed, 78 restored inference tensors on both hosts |
| 2+1 inference repeatability | Passed within each host, 3 events in every run |
| Exactness vs frozen binary64 oracle | Failed: `oracle_divergent`, exit 1 on both hosts |
| Manual GitHub `workflow_dispatch` full-model job | Executed (run `43963725603`); failed by design at the comparison step, artifacts uploaded |
| PR CI at `0537cb4` | Passed (run `43964412247`) |
| Full suite on native Linux x86-64 Colab | Passed, 3672 tests at `0537cb4` |
| Production integration | Not implemented |

## Provenance

| Artifact | SHA-256 | Source |
| --- | --- | --- |
| `experiments/oaf_tf215/requirements.lock` | `a860e33c78a4af60af0d5179070adbe12772d3688d9c3d7ab0d6296dbf84bbde` | Python 3.11 / `x86_64-manylinux_2_17` |
| `experiments/oaf_tf215/patches/tf215-compat.patch` | `554527246744662d914ea45973b93dd23aa6a31e5270a36b729589327da4e041` | Reviewed TF 2.15 compatibility patch |
| `tests/fixtures/oaf_tf1_smoke/smoke-oracle.json` | `6262da200dafbabe74f72d5108f9fe0244087f8a31b3a05d6a6c55c1aa615c69` | Frozen `crux.oaf-smoke-oracle/v2` |
| `tests/fixtures/oaf_tf1_smoke/canonical.wav` | `8ec2aed65945b7002e17b51818495ca754a519c39940f739cdcd1403eb661673` | Canonical smoke input (`input_audio_sha256`) |
| OaF checkpoint archive | `09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0` | `https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/e-gmd_checkpoint.zip` |
| `model.ckpt-569400.data-00000-of-00001` | `6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5` | Frozen `runtime/oaf_tf1/model.json` |
| `model.ckpt-569400.index` | `475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a` | Frozen `runtime/oaf_tf1/model.json` |
| `model.ckpt-569400.meta` | `e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422` | Frozen `runtime/oaf_tf1/model.json` |

The four repository files above were re-hashed with `sha256sum` for this record
and match. Backend: `magenta-egmd-tf1-94529798-8hit-v1`; model:
`magenta-egmd-ckpt-569400-v1`; upstream source commit
`94529798dfbbb14c27ddfd76f23027dc8e2ce185`. The compatibility patch's source
reference is Magenta `c15687ebd1c1cc12658cd468c737d3c24c02b918`. The frozen
`runtime/oaf_tf1/`, `src/`, root project files, checkpoint bytes,
instrumentation patch and oracle were not modified by this experiment.

## Hosts

| Host | Detail |
| --- | --- |
| GitHub Actions | `ubuntu-24.04`, Linux x86-64 (`RUNNER_ARCH` X64). Worker/TF environment Python 3.11.18, host driver Python 3.12.16. The PR job downloads no checkpoint; the manual model job downloads the public checkpoint. |
| Native Colab CPU | Fresh VM, Linux `6.6.159+` `x86_64`, glibc 2.39. Worker Python 3.11.17, host driver Python 3.12.3. Notebook kernel left unchanged. |

Both hosts ran the same hash lock, scratch builder, `graph_check.py` and
`compare.py` commands with the four environment controls
(`CUDA_VISIBLE_DEVICES=-1`, `PYTHONHASHSEED=0`, `TF_ENABLE_ONEDNN_OPTS=0`,
`NUMBA_CACHE_DIR=/tmp/numba-cache`).

## Graph check — both hosts

Commit `c3ffbef5c553e9277c2b90bf94722232f2f848a9`. All four stages
(`environment`, `import`, `slim_batchnorm`, `legacy_lstm`) passed on both hosts.

| Version | GitHub Actions | Native Colab CPU |
| --- | --- | --- |
| Python | 3.11.18 | 3.11.17 |
| TensorFlow | 2.15.1 | 2.15.1 |
| Keras | 2.15.0 | 2.15.0 |
| NumPy | 1.23.5 | 1.23.5 |
| tf-slim | 1.1.0 | 1.1.0 |
| `framework_ops.Tensor` alias restored | true | true |

Executed graph details on both hosts: Slim batch-norm output shape `[2, 8, 8, 4]`,
5 variables, finite output and moving statistics; legacy LSTM output shape
`[2, 5, 6]`, 2 variables, finite output and final cell state. Initializer seeds
are unset, so the diagnostic sums vary between runs; pass/fail uses shape and
finiteness, not the sums. The GitHub artifact is `oaf-tf215-graph-check` on run
`43963725603` (`GITHUB_SHA=c3ffbef`); the Colab record is the corresponding
session-local `graph_check` result for the same commit.

## Restore — both hosts

On both hosts the worker loaded the frozen model config, verified the cached
checkpoint archive digest, restored **78** inference tensors and reported
`backend_id=magenta-egmd-tf1-94529798-8hit-v1`. The frozen coverage checks
(130-tensor partition, 52 non-inference tensors and the 78 required
names/shapes/dtypes) execute before the worker sends `ready`. The first native
restore at `4490ab7271df3f22dc8e5ea9b9229b385b45ed0c` established this count; it
was repeated at `bcb9c95a36029d301ada5b85a4bd234fec0a8b3a`, and the `c3ffbef`
GitHub model job and Colab run report the same 78.

## Inference and exactness — both hosts

Commit `c3ffbef` (driver schema `oaf-tf215-compare/v2`). Two requests ran
through one loaded worker, the worker was closed, and a third request ran
through a fresh worker, on each host:

| Host | Runs | Workers | Events per run |
| --- | --- | --- | --- |
| Native Colab CPU | 3 | 1, 1, 2 | 3, 3, 3 |
| GitHub Actions | 3 | 1, 1, 2 | 3, 3, 3 |

- Repeatable: **true within each host** (all three attempts identical).
- Frozen oracle: **not equal**.
- Derived status: **`oracle_divergent`**, process exit **1** on both hosts after
  writing the record. Each artifact records `inference_completed=true`,
  `repeatable=true`, `oracle_exact=false`, no failed stage.
- Fresh worker ready: 78 restored tensors, same backend ID.

All events have identical identity fields in every run and in the oracle:
`native_midi_note=38`, `native_class_id=midi_38`, `model_output_bin=17`,
`upstream_8hit_group_id=snare`, velocities 116, 118, 107. Only
`time_sec_binary64` and `confidence_binary64` differ. Time deltas are identical
on both hosts; confidence deltas are identical except event 0:

| Event | Time (candidate) | Time (oracle) | Δ time (s) | Confidence (Colab) | Confidence (oracle) | Δ confidence (Colab) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.09287981859410431 | 0.09288 | -1.8140589569393128e-07 | 0.6134777069091797 | 0.613478 | -2.93090820280284e-07 |
| 1 | 0.39473922902494329 | 0.394739 | 2.2902494328791434e-07 | 0.8444571495056152 | 0.844457 | 1.4950561522120154e-07 |
| 2 | 0.6965986394557823 | 0.696599 | -3.6054421770348455e-07 | 0.8564286828041077 | 0.856429 | -3.171958923298135e-07 |

GitHub Actions event 0 confidence is `0.6134778261184692`, delta
`-1.7388153072950274e-07`; its events 1 and 2 values and deltas match the Colab
table. The corresponding event arrays are therefore **not bit-identical across
hosts**: event 0 confidence is `3fe3a19c40000000` on GitHub and
`3fe3a19c00000000` on Colab, while the other recorded fields match. Each host is
internally repeatable; cross-host bit equality is not claimed.

The recorded binary64 patterns are shown as-is and no root cause is claimed.
Time patterns are full binary64. Confidence patterns are low-bit-reduced but not
uniformly: Colab events 0 and 1 end in 32 zero low bits
(`3fe3a19c00000000`, `3feb05cb00000000`), Colab event 2 ends in 29
(`3feb67dd20000000`), and GitHub event 0 ends in 30 (`3fe3a19c40000000`). No
blanket claim that all low 32 bits are zero is made. The comparison caps
recorded deltas at 8 entries, so run 3's third event delta is omitted; all three
runs per host are byte-identical for these fields, so this is a record cap, not
a missing measurement.

The earlier `afba447cab749a89c40970dc700bc3e3af317ae6` and
`a6579901158d9e86d71d1743cfbd1fe18f2b043b` Colab runs produced the same Colab
event values. Commit `e270498` changed only failure classification and was not
re-run natively; it is not claimed as executed model evidence.

## Manual GitHub job — executed, failed as designed

A Task 5 preflight cited a contents-API 404 for
`.github/workflows/oaf-tf215-probe.yml` at `main` and an earlier draft of this
document recorded the dispatch as unavailable before merge. That prediction was
wrong: the dispatch command was accepted against the PR branch and executed.
The actual command was

```bash
gh workflow run oaf-tf215-probe.yml \
  --ref docs/hpa-856-python311-tf215-prototype
```

which created `workflow_dispatch` run `43963725603` at `c3ffbef`. GitHub ran
both jobs:

| Job | ID | Result |
| --- | --- | --- |
| `Python 3.11 / TF 2.15 graph check` | `133637558510` | Passed |
| `Manual full-model comparison` | `133637559032` | Failed at the comparison step by design |

The model job passed hash-lock sync/check, scratch build, public checkpoint
acquisition, config/input/worker-start/restore (78 tensors) and all three
inferences. It wrote `compare.json` with `repeatable=true`, `oracle_exact=false`
and `status=oracle_divergent`, set `COMPARE_EXIT=1`, and the step failed. The
always-on summary and upload steps then succeeded; artifacts `oaf-tf215-compare`
(`compare.json`, `compare.log`) and `oaf-tf215-graph-check` are attached to run
`43963725603`. The red job is the designed parity gate, **not a pass**. No secret
or credential is used; the checkpoint is the public archive.

## Verification runs

- PR CI run `43964412247` passed at `0537cb4`: the graph job executed hash-lock
  sync/check, the scratch build and Python 3.11 imports, the scoped driver and
  workflow tests, and both graph sessions. The manual model job was correctly
  skipped for `pull_request`.
- Native Colab full suite at `0537cb4`: `uv run --extra mlflow --extra r2
  pytest` => **3672 passed** in 1060.69s, Python 3.12.3 on Linux x86_64.
- Local full suite on the aarch64 development host with the same command:
  **3671 passed, 1 failed**. The failure is
  `tests/benchmark/test_separator_environment_probe.py::test_probe_hashes_resolved_interpreter_target`:
  this host's standalone Python cannot resolve its stdlib through the extra
  interpreter symlink (`/install/lib/python3.12`, missing `encodings`). Setting
  `PYTHONHOME` makes that test pass but breaks 29 unrelated separator tests, so
  no global workaround is claimed. This is a host limitation, not an experiment
  regression; the same suite passes on native Linux x86_64 Colab.
- `uv run ruff check .`, `uv run ruff format --check src tests` and
  `uv run pylint --errors-only src` passed at `c3ffbef`. The later `0537cb4`
  commit changes one test file only, and this documentation commit changes no
  code.

## Decision — Task 6 internal review checkpoint

A repeatable `oracle_divergent` measurement on two independent Linux x86-64
hosts is useful research evidence, not accepted TF1 equivalence:

- It does not license TF1 parity, model promotion, a broad corpus, separation
  work, Drive resume or MLflow publication.
- Before broad science, the TF2.15 candidate needs its own runtime/output
  descriptor and cache identity so TF2.15 events are never presented as the
  frozen `magenta-egmd-tf1-94529798-8hit-v1` baseline, plus quantified deltas on
  more than this single fixture. The five-song continuation below records the
  implementation and the inputs required to measure it.
- The frozen TF1 sources, checkpoint, oracle, root dependencies and Docker
  runtime are unchanged and remain the scientific baseline.

This is an internal review checkpoint in the single HPA-856 PR #36; the work is
not split and full HPA-856 is **not complete**. No production integration
(CLI flag, backend factory, cache identity or MLflow wiring) was added.
Conditional downstream work — reference identities, broad corpus/reviewed
subset, same-machine Spleeter/HTDemucs separation, frozen v2 separator
companions, Drive resume and the existing MLflow publisher — is not implemented
and stays conditional on a decision after review.

## Five-song continuation

The experimental driver implements the
[design](../../docs/superpowers/specs/2026-11-17-hpa-856-tf215-identity-comparison-design.md)
for simfiles `389`, `391`, `27`, `241`, `207`. The pinned Git reports
were verified by SHA-256. The Python 3.11 probe returned
`unsupported_platform` on the local Linux aarch64 host as expected.
All 124 scoped tests passed, including the original smoke and workflow tests.
Running `prepare` with the three pinned reports and a missing `run.json`
returned blocked with exit status 2 and created no selection.

Real measurement is **blocked on the archived TF1 inputs**. The Git evidence
contains the reports. Local artifacts and the expanded connected Drive archive
did not contain a matching bundle for `oaf-2d67684ad190163e`. The expanded
Drive run is `oaf-f2dc8381e50f447f` and has a different reference digest.
The required bundle consists of the pinned run's `run.json`, five original
prediction-v2 bodies, matching reference/timing manifests and selected timing
event bodies, and verified source audio. Neither fresh Colab VM executed the
selected-song comparison.

## Remaining benchmark work

- Only the canonical single-fixture smoke input was executed. No broad corpus,
  reviewed subset, same-machine separation, R2 or Drive artifact was produced.
- No cross-host bit equality and no TF1 rerun: the frozen binary64 oracle is the
  baseline, and the two TF2.15 hosts differ from each other at event 0
  confidence.
- No canonical prediction, report or production cache was written; experiment
  records stay under `artifacts/experiments/oaf-tf215/` and session-local
  copies.
