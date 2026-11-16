# OaF TF 2.15 experiment — recorded results

Initial evidence record for HPA-856 / PR #36. It contains only executed
evidence: GitHub Actions runs and native Linux x86-64 Colab CPU runs. Nothing
here is a parity claim, and a commit that only changes classification or
documentation is not presented as re-executed model evidence.

## Status

**Repeatable `oracle_divergent`; not exact.** The candidate restores the frozen
checkpoint and produces three identical native event runs on native Linux
x86-64, but the time and confidence fields differ from the frozen TF1 smoke
oracle. Event identity (class, note, bin, 8-hit group, velocity) matches. No
TF1 cache or runtime identity is reused.

| Item | Status |
| --- | --- |
| Linux x86-64 graph sessions, drum config/worker imports | Passed (GitHub Actions and Colab) |
| Public checkpoint cache verification | Passed (Colab native run) |
| Restore / frozen 130/78/52 partition checks | Passed, 78 restored inference tensors |
| 2+1 inference repeatability | Passed on one host, 3 events in every run |
| Exactness vs frozen binary64 oracle | Failed: `oracle_divergent`, exit 1 |
| Manual `workflow_dispatch` job | Unavailable before merge, not passed |
| PR CI for the commit that adds this file | Pending |
| Native Colab rerun of the final driver commit | Pending |

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

Backend: `magenta-egmd-tf1-94529798-8hit-v1`; model:
`magenta-egmd-ckpt-569400-v1`; upstream source commit
`94529798dfbbb14c27ddfd76f23027dc8e2ce185`. The compatibility patch's source
reference is Magenta `c15687ebd1c1cc12658cd468c737d3c24c02b918`. The frozen
`runtime/oaf_tf1/`, `src/`, root project files, checkpoint bytes,
instrumentation patch and oracle were not modified by this experiment.

## Hosts

| Host | Detail |
| --- | --- |
| GitHub Actions | `ubuntu-24.04`, Linux x86-64 (`RUNNER_ARCH` X64), Python 3.11 for the worker/TF environment and Python 3.12 for the host driver/tests. No checkpoint in the PR job. |
| Native Colab CPU | Linux `6.6.159+` `x86_64`, glibc 2.39, Intel Xeon 2.20 GHz, notebook Python 3.13.18 (left unchanged), worker Python 3.11.17, host driver Python 3.12.15. |

## GitHub Actions evidence

These runs executed the checkpoint-free probe (lock sync/check, builder,
Python 3.11 imports and graph sessions; later runs also the fake-worker driver
tests). They are **not** model-inference evidence.

| Run | Commit | Scope | Result |
| --- | --- | --- | --- |
| `43930256637` (job `133520460334`) | `a470ddc` | Initial Python 3.11 / TF 2.15 graph probe, Python 3.11.18 | Passed |
| `43931458159` | `5ff919e` | Probe after fallback-artifact fix | Passed |
| `43940737504` | `33f0435` | Scratch builder, imports, graph, builder tests | Passed |
| `43942104484` | `4daafad` | Builder fix, same scoped job | Passed |
| `43942768233` | `0ae2a33` | Builder fix round 2, same scoped job | Passed |
| `43950409714` | `bcb9c95` | Builder, driver tests, graph | Passed |
| `43955315384` | `afba447` | Builder, host driver tests, graph | Passed |
| `43958322102` | `a657990` | Builder, host driver tests, graph | Passed |
| `43960021479` | `e270498` | Builder, host driver tests, graph | Passed |

`43949509517` (commit `4490ab7`) passed the same PR job before the restore
stage was added to the driver. The PR CI run for the Task 5 commit that adds
this file is **pending**; no result is claimed for it.

## Graph check — native Colab CPU

Commit `4490ab7271df3f22dc8e5ea9b9229b385b45ed0c`. All four stages passed
(`environment`, `import`, `slim_batchnorm`, `legacy_lstm`):

| Version | Value |
| --- | --- |
| Python | 3.11.17 |
| TensorFlow | 2.15.1 |
| Keras | 2.15.0 |
| NumPy | 1.23.5 |
| tf-slim | 1.1.0 |
| `framework_ops.Tensor` alias restored | true |

Executed graph details: Slim batch-norm output shape `[2, 8, 8, 4]`, 5
variables, finite sums (`output_sum=1.8e-05`, `moving_variance_sum=3.996091`);
legacy LSTM output shape `[2, 5, 6]`, 2 variables, finite
`output_sum=1.21449` and `final_cell_state_sum=0.565169`. Initializer seeds are
unset, so these diagnostic sums vary between runs; pass/fail uses shape and
finiteness, not the sums.

## Restore — native Colab CPU

Commit `bcb9c95a36029d301ada5b85a4bd234fec0a8b3a`. `config`, `checkpoint`,
`input`, `worker_start` and `restore` passed. The worker loaded the frozen
model config, verified the cached checkpoint archive digest, restored **78**
inference tensors and reported `backend_id=magenta-egmd-tf1-94529798-8hit-v1`.
The frozen coverage checks (130-tensor partition, 52 non-inference tensors and
the 78 required names/shapes/dtypes) execute before the worker sends `ready`.
This commit predates driver inference, so no events were produced here.

## Inference and exactness — native Colab CPU

Commit `a6579901158d9e86d71d1743cfbd1fe18f2b043b` (driver schema
`oaf-tf215-compare/v2`). Two requests ran through one loaded worker, the worker
was closed, and a third request ran through a fresh worker:

| Run | Worker | Events |
| --- | --- | --- |
| 1 | 1 | 3 |
| 2 | 1 | 3 |
| 3 | 2 | 3 |

- Repeatable: **true** (all three attempts are identical).
- Frozen oracle: **not equal**.
- Derived status: **`oracle_divergent`**, process exit **1** after writing the
  record. The artifact records `inference_completed=true`, `repeatable=true`,
  `oracle_exact=false`, no failed stage.
- Fresh worker ready: 78 restored tensors, same backend ID.

All three events have identical identity fields in every run and in the oracle:
`native_midi_note=38`, `native_class_id=midi_38`, `model_output_bin=17`,
`upstream_8hit_group_id=snare`, velocities 116, 118, 107. Only
`time_sec_binary64` and `confidence_binary64` differ:

| Event | Time (candidate) | Time (oracle) | Δ time (s) | Confidence (candidate) | Confidence (oracle) | Δ confidence |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.09287981859410431 | 0.09288 | -1.8140589569393128e-07 | 0.6134777069091797 | 0.613478 | -2.93090820280284e-07 |
| 1 | 0.39473922902494329 | 0.394739 | 2.2902494328791434e-07 | 0.8444571495056152 | 0.844457 | 1.4950561522120154e-07 |
| 2 | 0.6965986394557823 | 0.696599 | -3.6054421770348455e-07 | 0.8564286828041077 | 0.856429 | -3.171958923298135e-07 |

The candidate time and confidence records differ from the oracle in low-order
bits; the candidate confidence patterns have zero low 32 bits (for example
`3fe3a19c00000000` vs `3fe3a19c9d5a187a`), while the time patterns are full
binary64 (for example `3fb7c6f8c751f177` vs `3fb7c6fbd273d5bb`). The recorded
binary64 patterns are shown as-is; no root cause is claimed beyond the
observation. The comparison caps recorded deltas at 8 entries, so run 3's third
event delta is omitted; all three candidate runs are byte-identical for these
fields, so this is a record cap, not a missing measurement.

The same evidence was reproduced at `afba447cab749a89c40970dc700bc3e3af317ae6`
with the same three events and the same divergent fields. Commit `e270498`
changed only failure classification and was not re-run natively; it is not
claimed as executed model evidence.

## Manual GitHub job — recorded as unavailable

The full-model job is `workflow_dispatch`-gated inside
`.github/workflows/oaf-tf215-probe.yml`. That file is added by PR #36 and is
absent from the default branch: GitHub's contents API returned 404 for
`.github/workflows/oaf-tf215-probe.yml` at `main` during the Task 5 preflight,
and GitHub only registers dispatch for workflows on the default branch.
Dispatching

```bash
gh workflow run oaf-tf215-probe.yml \
  --ref docs/hpa-856-python311-tf215-prototype
```

is therefore **unavailable before merge**. It is recorded as unavailable, not
as passed. No `oaf-tf215-compare` GitHub artifact exists for a pre-merge
commit. The identical command sequence was executed on the native Colab CPU
host instead (see above), using the public checkpoint command from
`oaf-smoke.yml`; no secrets or credentials are part of this experiment.

## Pending and out of scope

- PR CI for the Task 5 commit (this file) and a native Colab rerun of the final
  driver commit are pending; neither is claimed here.
- Task 6's decision review is pending. The evidence is a repeatable
  divergence, so it does not license TF1 equivalence, production promotion,
  cache/identity reuse, corpus runs, separation work, Drive resume or MLflow
  publication. A repeatable `oracle_divergent` result requires a new
  runtime/descriptor/cache identity and quantified deltas before broad science.
- Only the canonical single-fixture smoke input was executed. No broad corpus,
  reviewed subset, same-machine separation, R2 or Drive artifact was produced.
