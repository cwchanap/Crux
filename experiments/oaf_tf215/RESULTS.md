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

The original October comparison remains **blocked on the archived TF1 inputs**.
The Git evidence contains the reports. Local artifacts and the expanded
connected Drive archive did not contain a matching bundle for
`oaf-2d67684ad190163e`. The expanded
Drive run is `oaf-f2dc8381e50f447f` and has a different reference digest.
The required bundle consists of the pinned run's `run.json`, five original
prediction-v2 bodies, matching reference/timing manifests and selected timing
event bodies, and verified source audio.

### Fresh September TF1 baseline

The fresh comparison keeps the October-ranked membership and uses the
September reference and timing manifests whose respective SHA-256 digests are
`4f5d5e859650cb78a9f78697e8f19ab1ac3e7ea3c86dbea0e1f00c876357c7a3` and
`372979d42fff94438fe8d844fca2770d430a3c0898a306a4bc46bd8b2887e38f`.
All 180 eligible timing files and the five original OGGs recovered from Drive
matched their manifest hashes. The frozen TF1 scope completed all five songs
as `oaf-9dce500b62a3e24c` on an x86-64 Colab CPU VM. It used a chroot built
from the pinned OCI image layers and the locked TF1 dependencies. The worker
restored 78 tensors and matched the canonical smoke oracle exactly on that VM.
The runner sealed the fresh TF1 run, prediction bodies and canonical WAV
hashes with `prepare-fresh`.

The Colab CLI ran the experiment from a snapshot of `4754086` with the
fresh-selection code and tests applied. The TF2.15 worker used CPython 3.11.17,
`tensorflow-cpu` 2.15.1 and the hash-locked requirements. The bootstrap ran
`uv pip sync --require-hashes`, `uv pip check`, `build_runtime.py` and
`graph_check.py` before the selected-song driver. It used the existing Python
3.12 project lock for the host driver. The first VM had Linux `6.6.159+`,
glibc 2.39 and two logical CPUs in its affinity mask.

Each TF2.15 measurement completed all 15 song attempts. Each song's three
native event arrays agreed exactly within both VMs, with two requests in
one worker and one in a new worker. Both workers for every song restored
78 tensors. The candidate smoke output again differed from the binary64
oracle in time and confidence.

| Simfile | TF1 events | TF2.15 events | Matched within 1 ms | Max confidence difference |
| ---: | ---: | ---: | ---: | ---: |
| 389 | 81 | 81 | 81 | 0.000001 |
| 391 | 530 | 530 | 530 | 0.000001 |
| 27 | 910 | 910 | 910 | 0.000003 |
| 241 | 820 | 820 | 820 | 0.000009 |
| 207 | 710 | 710 | 710 | 0.000002 |

All 3,051 events had identical onset times at six-decimal artifact precision.
The comparison observed zero one-frame shifts, additions, removals and
rounding collisions. Velocity, model output bin, MIDI note and upstream
group agreed on every match. Confidence differences prevented ordered event
equality on all five songs. Median absolute confidence difference was zero
and p95 was `0.000001` for each song.

The second VM reproduced the seal and matched all 3,051 ordered TF2.15 events
at binary64 across VMs. The execution IDs are
`3387d5ac-c5fc-44dc-8f07-8232d6f63f69` and
`68c5d656-d7db-4cb5-8a32-2d0814a732e9`. Both recorded runtime digest
`865c1ea6360f95699f7b89bff340f2140901547926f3e71b3d1bc1ee02203959`
before and after inference. Both verified seal digest
`be7d82e42658c227f20d652bc5536bed89d6e66d0fdc8ea982d3ef2f5e6acf67`.
The candidate descriptor digest is
`938cbcccff5adda03f11344596e21534e178c98929098bf8327feb1e9ed4c890`.

The GitHub selected-song repeat remains blocked because CI has no authorized
delivery path for the corpus inputs. The Colab repeat measures two fresh VMs
from one provider.

The evidence archives, command logs, verified inputs and launcher scripts are
in `artifacts/experiments/oaf-tf215/fresh-september/` in the experiment
worktree. The archives include the TF1 corpus reports and prediction JSONL,
the seal, the TF2.15 smoke and raw attempts, and runtime provenance. Source
OGGs and canonical WAVs were excluded from the returned archives.

After the second measurement, `uv sync --locked --extra mlflow --extra r2`
and `uv run --no-sync pytest -q -o addopts=` passed on Colab with the Python
3.12.3 project interpreter. All 3,726 tests passed in 470.56 seconds. Ruff
lint and format checks passed for the three changed Python files.

## Twenty-song panel

The [preregistered panel](PANEL.md) used 20 additional songs and 6,237.24
seconds of source audio. All 20 Drive originals matched the September
manifest hashes. The TF1 baseline `oaf-ec9176f5c314b5b6` and the TF2.15
candidate both completed every song using the same sealed WAVs on one
two-vCPU x86-64 Colab VM. The frozen TF1 smoke matched exactly. TF2.15
completed its repeatable smoke with binary64 differences from the oracle.

The panel **passed its preregistered gate**. At raw 50 ms and zero offset,
the two backends agreed on every song's TP, FP and FN counts. Both produced
macro F1 **0.217898** and micro-F1 **0.238997** (exactly `4241/17745`).

| Simfile | TP | FP | FN | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 131 | 204 | 1,051 | 5,284 | 0.060507 |
| 384 | 37 | 34 | 4,080 | 0.017670 |
| 329 | 24 | 10 | 1,504 | 0.030730 |
| 209 | 72 | 224 | 1,129 | 0.096192 |
| 112 | 274 | 654 | 2,924 | 0.132816 |
| 68 | 780 | 702 | 6,422 | 0.179641 |
| 56 | 541 | 373 | 5,172 | 0.163271 |
| 201 | 224 | 1,235 | 962 | 0.169376 |
| 298 | 110 | 141 | 838 | 0.183486 |
| 364 | 220 | 183 | 1,537 | 0.203704 |
| 194 | 1,869 | 3,462 | 7,776 | 0.249599 |
| 216 | 509 | 712 | 2,039 | 0.270098 |
| 138 | 1,278 | 3,088 | 4,673 | 0.247746 |
| 187 | 270 | 52 | 1,503 | 0.257757 |
| 225 | 186 | 102 | 845 | 0.282032 |
| 294 | 3,096 | 6,656 | 4,322 | 0.360629 |
| 154 | 614 | 813 | 1,453 | 0.351460 |
| 262 | 1,456 | 2,778 | 3,148 | 0.329486 |
| 148 | 749 | 542 | 1,965 | 0.374032 |
| 200 | 210 | 73 | 563 | 0.397727 |
| Pooled | 12,723 | 22,885 | 58,139 | 0.238997 |

Every one of the 35,608 native events matched within 1 ms, and onset times
agreed at six-decimal precision. The run observed zero frame shifts,
additions, removals or rounding collisions. Velocity, MIDI note, model
output bin and upstream group agreed on every match. The largest confidence
difference was `0.000097` on simfile 262. Each of the two stress anchors
(194 and 384) produced identical raw events on its primary, same-worker
replay and fresh-worker replay.

Pooled inference RTF including worker startup was 0.189538 for TF1 and
0.180042 for TF2.15. Across the 19 songs after the first scheduled song,
RTF was 0.184232 and 0.178518 respectively. The TF2.15 worker took
14.424165 seconds to start for the primary pass. These are measurements
from one VM. The RTF excludes provisioning, canonicalization, scoring and
anchor replay time.

The experiment ran from `ed1df1a` with the panel changes applied, using
CPython 3.12.3 for the driver and CPython 3.11.17 with TensorFlow CPU 2.15.1
for the candidate. `prepare-panel` sealed the TF1 artifacts and WAVs, and
`run-panel` measured the candidate. The candidate fingerprint
`349a19bf89de0b7349372e5079aa874d95171fd7499a5068f3cdaa17f009fc36`
matched before and after inference. The seal is
`ece2a444fff7c1e33ad3e2dc31f571ae53deab9ffec4b5f08a9db419a2bfb7d6`.
The archive and recomputed verification are in
`artifacts/experiments/oaf-tf215/twenty-song-panel/`. Its archive SHA-256 is
`ba7892ed54a9d5b71bd39873ffee959d54fd9b2ca0fa79991a2250fd72adff1c`.
The local OaF verification suite passed 183 tests.

## Remaining benchmark work

- The measured scope covers the canonical smoke fixture and 25 songs.
  Broad corpus, reviewed subset and separation measurements remain open.
- The GitHub and Colab TF2.15 smoke observations differ at event 0 confidence.
- The TF1 prediction and report artifacts are confined to the experiment output
  under `artifacts/experiments/oaf-tf215/`.
