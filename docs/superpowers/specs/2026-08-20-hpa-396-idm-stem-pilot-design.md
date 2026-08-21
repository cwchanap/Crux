# HPA-396 Frozen IDM Stem Pilot Design

**Status:** Proposed  
**Date:** 2026-08-20  
**Linear:** HPA-396 — Evaluate frozen Inverse Drum Machine on fixed drum-stem benchmark pilot

## Decision summary

Implement Inverse Drum Machine (IDM) as the third **benchmark-only** transcription backend over the already-frozen HPA-328 HTDemucs input population.

The design stays deliberately narrow:

1. pin the official `bernardo-torres/inverse-drum-machine` repository at commit `456656868538205ef756912c7cf5b0fd936de8af` and the committed `idm-44-train-kits` checkpoint/config bytes;
2. keep IDM in one isolated `uv` runtime rather than adding its research dependency stack to Crux's Python 3.12 environment;
3. reuse the existing `WorkerProcess` JSON-line controller so the model loads once per pilot, not once per song;
4. use IDM's encoder transcription activations plus its evaluation peak-picking path directly, with no manual onset override and no required synthesis/separation output;
5. extend the existing backend descriptor, prediction-v2, taxonomy, and mapping unions for exactly one third known backend family;
6. consume `load_separation_pilot_manifest()` and the retained HPA-328 HTDemucs canonical input/prediction artifacts; do not rerun HTDemucs or OaF;
7. rebuild OaF and IDM HPA-325 cohort results from the exact fixed rows and publish paired OaF→IDM deltas using the existing model-neutral comparison primitives;
8. keep direct full-mix IDM inference to one fixed five-song smoke diagnostic and never mix it into the headline stem comparison;
9. stop before scored execution if the official model/runtime/license evidence cannot be frozen reproducibly.

Do **not** add a generic model runtime manager, generic corpus-runner base class, new scorer, prediction schema v3, reconstructed IDM stems as a required artifact, model-selection/tuning loop, full-corpus IDM run, production backend registration, training, fine-tuning, or manual transcription corrections.

## Why HPA-396 is next

The live Linear dependency graph leaves four benchmark-closeout children:

- **HPA-627** is High priority but operationally blocked by the gated MuScriptor Hugging Face checkpoint/license acceptance. It cannot be completed from the currently accessible environment.
- **HPA-396** is Todo and both declared blockers, HPA-325 and HPA-328, are complete. It directly blocks HPA-329 final comparative findings.
- **HPA-562** is technically actionable, but publishing the cross-model matrix before IDM exists would force a second pass immediately afterward. HPA-396 already needs one OaF↔IDM paired comparison.
- **HPA-329** remains blocked by HPA-396.

Therefore HPA-396 is the next implementable task on the critical path. HPA-562 should follow after this pilot so it can consolidate already-published OaF, MuScriptor, and IDM evidence once.

## Existing evidence and seams

### HPA-328 already created the handoff HPA-396 needs

`src/benchmark/separation_handoff.py` explicitly owns the immutable HPA-328 closeout boundary for HPA-396. `load_separation_pilot_manifest()` validates:

- canonical `crux.oaf-separation-pilot/v1` JSONL;
- the exact 20–30-song HPA-327 population;
- sorted unique simfile IDs;
- one consistent reviewed-subset/reference/timing/OaF/run identity;
- full-mix, Spleeter, and HTDemucs per-song status/evidence;
- retained stem, canonical input, and prediction hashes;
- fixed comparison artifact identity.

HPA-396 consumes this immutable handoff. It does not parse HPA-328's mutable `run.json`, infer membership from directories, or invoke a separator.

### The shared backend protocol is already sufficient

`src/benchmark/backends/base.py` already defines:

```python
CanonicalAudio
NativeEvent
NativePrediction
TranscriptionBackend
```

IDM fits that protocol. It does not need a new base class or registry.

### The persistent worker seam is already sufficient

`src/benchmark/worker_process.py` already owns a small persistent process with:

- one `ready` record;
- one JSON request/response per audio path;
- request IDs;
- timeout/poison semantics;
- bounded stderr diagnostics;
- deterministic close/terminate/kill behavior.

HPA-396 becomes its second concrete model caller. Only OaF-specific wording should be neutralized if needed; the protocol itself remains unchanged.

### Prediction v2 is intentionally a closed backend union

`src/benchmark/prediction_artifact.py` currently validates exactly OaF and MuScriptor families while retaining a backend-neutral event shape:

```text
time_sec
native_class_id
model_output_bin
native_midi_note
native_metadata
confidence
velocity_midi
canonical_class
common_class
mapping_status
prediction_map_version
```

That shape is enough for IDM. Add a third family to the closed union; do not create prediction v3.

### HPA-328 already promoted the comparison primitives

`src/benchmark/published_comparison.py` contains model-neutral pairing and report helpers with configurable labels, including:

```python
pairable_success_ids(...)
paired_song_rows(...)
paired_class_rows(...)
metric_delta(...)
aggregate_delta_rows(...)
population(...)
runtime(...)
```

HPA-396 should use those primitives behind one small IDM-specific report driver. It should not generalize HPA-562 early.

## Upstream IDM facts frozen by this design

The official repository's current pinned revision is:

```text
repository: https://github.com/bernardo-torres/inverse-drum-machine
revision: 456656868538205ef756912c7cf5b0fd936de8af
commit message: Adding manual onset override support
```

The repository declares package version `0.1.0`, Python `^3.10`, PyTorch/torchaudio `^2.1.1`, librosa `^0.9.1`, and a larger research dependency set. The repository does not commit a Poetry lock file.

The selected pretrained model is already committed in the official repository:

```text
pretrained/idm-44-train-kits/checkpoints/model.yaml
pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt
```

The checkpoint is about 2.5 MB, so there is no gated-download problem analogous to MuScriptor.

The selected `model.yaml` freezes:

```text
sample rate: 44100 Hz
mel n_fft: 1024
mel hop_length: 256
mel bins: 128
transcription onset activation: none (logits)
transcription velocity activation: exp_sigmoid
classes: 9
decoder evaluation onset activation: sigmoid
decoder evaluation peak picking: PeakPicking defaults
```

The exact ordered `train_classes` are:

```text
CY_CR
CY_RD
HH_CHH
HH_OHH
KD
SD
TT_HFT
TT_HMT
TT_LMT
```

The mel transform exposes frame rate as `sample_rate / hop_length`, therefore the expected activation rate is `44100 / 256 = 172.265625 Hz`. The runtime still verifies the model's observed `encoder.frame_rate` rather than trusting this arithmetic alone.

The repository's root `LICENSE` is Apache-2.0. The real freeze step must also establish the licensing basis for the committed checkpoint bytes. If a separate or contradictory weight notice exists at the pinned revision, that notice wins; if the weight-license basis cannot be established, scored execution stops.

## Approach selection

### Approach A — isolated persistent IDM worker — selected

Create a standalone runtime under `runtime/idm/` and launch a small worker with that runtime's Python. The worker loads the exact model once, then handles all pilot requests through the existing `WorkerProcess` controller.

Why this is the right size:

- Crux's main environment is Python 3.12 with `librosa>=0.10`; upstream IDM asks for the 0.9 line and a research stack. Isolation avoids changing unrelated benchmark/runtime dependencies.
- The official model is small enough that a container is unnecessary.
- A persistent process amortizes PyTorch/model startup across 20–30 songs.
- The existing worker controller already provides the process lifecycle and timeout behavior.
- The isolated runtime is comparator-only and never touches the production backend registry.

### Approach B — add IDM to Crux's main optional dependencies — rejected

This looks smaller but creates dependency-resolution risk around librosa and the research stack, and couples unrelated Crux tests/lock resolution to an experimental comparator. HPA-395 could use an optional dependency because MuScriptor resolved inside the main environment; IDM's upstream metadata makes that a poor default here.

### Approach C — one IDM process per song — rejected

This avoids a persistent protocol but reloads PyTorch and the model 20–30 times, contaminates RTF with repeated startup cost, and makes the pilot slower with no useful simplification because `WorkerProcess` already exists.

### Containers / generalized external-runtime framework — rejected

There is no need for Docker, RPC, a plugin host, or another byte-level runtime-attestation framework. HPA-328 needed unusually strict separator runtime isolation because native separator caches/resolvers affected generated stem identity. IDM is a fixed research comparator with a public tiny checkpoint and an isolated lockable environment. Verify the exact runtime/model identities needed for reproducibility and stop there.

## Isolated runtime and model freeze

### Runtime project

Add a standalone project:

```text
runtime/idm/pyproject.toml
runtime/idm/uv.lock
runtime/idm/worker.py
```

The project has one direct dependency on the exact Git revision of the official IDM package. `uv.lock` freezes the resolved transitive environment. The implementation feasibility task selects one supported Python minor (prefer 3.11 if the upstream dependency set requires it) and then freezes that choice; no scored CLI flag chooses another runtime.

Normal Crux `uv run ...` commands do not install this runtime. Operational setup is explicit:

```bash
uv sync --project runtime/idm --frozen
```

The runtime environment directory remains uncommitted.

### Model cache

The model freeze step acquires only the two official committed model files needed by `load_model('idm-44-train-kits', ...)` and stores them beneath a caller-selected local model cache. Runtime inference is local/offline after that freeze.

Do not commit duplicate checkpoint bytes into Crux unless the implementation discovers a concrete operational need. The committed identity is the lock; the local cache is verified against it before worker startup.

### `crux.idm-model/v1`

Add one strict canonical lock in `runtime/idm/model.json`. It records the execution-semantic evidence, not host-specific absolute paths.

At minimum:

```text
schema
repository_url
repository_revision
package_name
package_version
code_license
weight_license
runtime_lock_sha256
python_version
model_name
model_config_relative_path
model_config_sha256
model_config_byte_length
checkpoint_relative_path
checkpoint_sha256
checkpoint_byte_length
model_id
device
dtype
sample_rate_hz
mel_n_fft
mel_hop_length
mel_n_mels
activation_rate_hz
train_classes
onset_activation
peak_pick_div_max
peak_pick_div_avg
peak_pick_div_wait
peak_pick_div_threshold
peak_pick_normalize
velocity_activation
velocity_max_value
velocity_to_midi_revision
manual_onset_override
reconstructed_stems
masking
chunking_mode
native_output_space_id
native_metadata_schema_id
training_data_map_id
```

The selected v1 contract freezes:

```text
repository_revision = 456656868538205ef756912c7cf5b0fd936de8af
package_name = inverse-drum-machine
package_version = 0.1.0
model_name = idm-44-train-kits
sample_rate_hz = 44100
mel_n_fft = 1024
mel_hop_length = 256
mel_n_mels = 128
train_classes = [CY_CR, CY_RD, HH_CHH, HH_OHH, KD, SD, TT_HFT, TT_HMT, TT_LMT]
onset_activation = sigmoid-over-native-logits-before-upstream-peak-picking
peak_pick_div_max = 20
peak_pick_div_avg = 10
peak_pick_div_wait = 16
peak_pick_div_threshold = 5
peak_pick_normalize = false
velocity_activation = exp_sigmoid(exponent=10,max_value=2,threshold=1e-7)
manual_onset_override = false
reconstructed_stems = false
masking = none
chunking_mode = none
```

`device` and `dtype` are frozen before the scored pilot. Hardware feasibility may select CPU, MPS, or CUDA once; benchmark F1 must not participate in that selection.

### Model ID

Use one digest-derived ID rather than a moving friendly name:

```text
idm-44-train-kits-<revision12>-<checkpoint12>
```

The lock derives and validates it from the exact source commit and checkpoint SHA-256.

## Worker transcription contract

`runtime/idm/worker.py` is deliberately small and contains no Crux imports. The host validates the Crux model lock; the worker validates the actual upstream runtime/model behavior it observes.

### Startup

The host launches roughly:

```text
<runtime-idm-python> runtime/idm/worker.py --model-root <verified-local-root>
```

The worker:

1. imports `idm.inference.load_model` from the exact locked package;
2. loads `idm-44-train-kits` once on the frozen device/dtype;
3. verifies the observed sample rate, frame rate, and ordered `train_classes` against startup arguments/expected constants;
4. emits one `ready` record containing the observed model facts.

The `IdmBackend` rejects any ready payload that disagrees with `runtime/idm/model.json`.

### Request

Each request contains the existing worker `id` + `audio_path` shape. HPA-396 passes only a verified canonical 44.1 kHz mono PCM16 WAV.

The worker verifies the audio format and runs only the transcription path:

```python
encoder_outs = model.encoder(audio.unsqueeze(0))
onset_logits = encoder_outs["activations"]["onset"]
velocity = encoder_outs["activations"]["velocity"]
onset_scores = model.decoder.activation(onset_logits)
onsets = model.decoder.peak_picking_val(
    onset_scores,
    activation_rate=encoder_outs["activation_rate"],
)
```

It does **not**:

- call the manual onset override path;
- synthesize one-shot samples;
- run Wiener masking;
- publish IDM-separated audio;
- change peak-picking thresholds;
- post-edit events.

This is equivalent to the evaluation onset path used by IDM's decoder while avoiding synthesis work that does not affect transcription scoring.

### Raw worker events

For every nonzero peak, the worker returns:

```text
class index
native train class
frame index
time_sec
post-sigmoid onset score at that frame
native exp-sigmoid velocity at that frame
```

All shapes, indexes, times, scores, and velocities must be finite and within the frozen class/frame ranges. A malformed tensor or event is an inference failure; do not silently drop it.

### Host-side `NativeEvent`

`IdmBackend` converts each validated raw worker event into the existing shared value:

```python
NativeEvent(
    time_sec=...,
    native_class_id="KD",  # example
    model_output_bin=4,
    native_midi_note=None,
    native_metadata={},
    confidence=<post-sigmoid onset activation>,
    velocity_midi=<deterministic 0..127 projection>,
)
```

The upstream velocity activation is bounded approximately in `(0, 2]`. Freeze a single diagnostic conversion:

```text
velocity_midi = round(clamp(native_velocity / 2.0, 0.0, 1.0) * 127)
```

This does not make velocity part of HPA-325 onset F1. It only preserves a stable velocity diagnostic in the existing prediction-v2 field. The exact transform gets a version ID in the model lock/inference config.

The post-sigmoid onset score is persisted as `confidence`; it is the exact activation used by upstream peak picking. No confidence threshold is introduced by Crux.

## Frozen backend descriptor and prediction-v2 extension

Add one third descriptor family in `backend_identity.py`:

```text
backend_id: idm-44-train-kits-v1
architecture_id: inverse-drum-machine-v0.1.0
model_id: idm-44-train-kits-<revision12>-<checkpoint12>
native_output_space_id: idm-44-train-kits-9class-v1
native_metadata_schema_id: idm-peak-event-metadata-v1
prediction_schema: crux.drum-prediction-events/v2
training_data_map_id: idm-training-contract-44-train-kits-v1
upstream_source_commit: 456656868538205ef756912c7cf5b0fd936de8af
```

`model_id` is the only patterned descriptor field. The backend separately requires exact equality with the loaded model lock.

### Native event policy

Add `idm-peak-event-metadata-v1` as an empty metadata object policy. Native class identity already lives in `native_class_id`, so duplicating it inside metadata is unnecessary.

Prediction-v2 validation for IDM requires:

- `model_output_bin` is an integer `0..8`;
- `native_midi_note is None`;
- `confidence` is finite in `0..1`;
- `velocity_midi` is integer `0..127`;
- `native_metadata == {}`;
- output bin resolves to the exact frozen ordered train class and equals `native_class_id`.

Keep all existing OaF and MuScriptor invariants unchanged. Keep the prediction-v2 schema version unchanged.

## IDM taxonomy and mapping

Add one frozen map:

```text
IDM_PREDICTION_MAP_ID = crux.prediction-map/idm-44-train-kits-v1
```

Use native train-class strings as the only keys.

| IDM native class | Upstream 9-class meaning | Crux detailed class | Crux common class |
| --- | --- | --- | --- |
| `KD` | kick | `kick` | `kick` |
| `SD` | snare | `snare` | `snare` |
| `HH_CHH` | closed hi-hat | `closed_hihat` | `hihat` |
| `HH_OHH` | open hi-hat | `open_hihat` | `hihat` |
| `CY_CR` | crash | `crash` | `crash` |
| `CY_RD` | ride | `ride` | `ride` |
| `TT_HMT` | hi tom | `high_tom` | `tom` |
| `TT_LMT` | mid tom | `null` | `tom` |
| `TT_HFT` | low tom | `low_or_floor_tom` | `tom` |

The upstream repository contains comments about historical reverse mappings for StemGMD tom sample names. HPA-396 therefore freezes only the semantics required by the model's own `class_mapping_9_classes`; it does not infer a nonexistent Crux mid-tom detailed category. `TT_LMT` remains mapped at common `tom` while its detailed class is null.

`map_idm_prediction()` follows the existing OaF/MuScriptor mapping shape and persists every native event. A future unexpected native class is unmapped evidence, not discarded.

## Primary pilot input contract

### Population

The population is exactly the rows in the finalized HPA-328 `crux.oaf-separation-pilot/v1` handoff. HPA-396 has no sample count, seed, include, exclude, or resampling controls.

### Headline input

For an HPA-328 row with successful/resumed HTDemucs evidence, HPA-396 uses the **exact canonical HTDemucs input WAV recorded in `row["htdemucs"]["input"]`**.

That is intentionally stronger than rerunning HTDemucs or re-canonicalizing its retained native stem:

- OaF already consumed those canonical bytes in HPA-328;
- the handoff records their `input_audio_sha256`;
- the OaF prediction artifact records the same input identity;
- IDM therefore sees byte-identical model input for the paired comparison.

HPA-396 still validates the retained native stem hash/lock lineage from the handoff so the canonical input remains attributable to the frozen HTDemucs run.

The historical input view remains exactly:

```text
crux.oaf-htdemucs-drums-mono44k1-pcm16/v1
```

Do not rename it just because IDM is now a consumer. It identifies the frozen input bytes, not the consumer model.

### Upstream HPA-328 failures remain in the population

If HPA-328 has no valid HTDemucs canonical input for a row, HPA-396 records a failed item such as `upstream_stem_unavailable`. It does not silently shrink the cohort.

### Artifact-root resolution

The handoff persists relative artifact paths and hashes, not machine-specific roots. The HPA-396 request therefore receives explicit HPA-328 artifact/cache roots. Resolution is exact and containment-checked; it does not scan directories or guess owners.

## OaF comparator reconstruction

Do not rerun OaF.

For each successful HTDemucs row, read the exact retained OaF prediction artifact referenced by the handoff and verify:

```text
artifact SHA-256
source_audio_id
source_audio_sha256
input_view_id
input_audio_sha256
backend/model/map identity
```

Then rebuild an HPA-325 OaF cohort item directly from that validated prediction artifact and the same reference mapping used for IDM.

This gives the primary comparison two independently persisted prediction artifacts over exactly the same canonical input bytes:

```text
OaF prediction from HPA-328
IDM prediction from HPA-396
```

Both are scored by the current `score_cohort()` implementation at 30/50/100 ms, raw and aligned. HPA-396 does not copy HPA-328 CSV cells or run a second scorer.

## HPA-396 run and persistence

Add one concrete runner, for example:

```text
src/benchmark/idm_pilot_run.py
```

with a local schema such as:

```text
crux.idm-stem-pilot-run/v1
```

The request contains only fixed roots/manifests/runtime choices needed to resolve persisted evidence. It has no sample/tuning controls.

Run identity binds at least:

```text
HPA-328 handoff SHA/corpus version
reference manifest SHA/version
reference timing SHA/version
IDM backend descriptor SHA
IDM model lock SHA
IDM runtime lock SHA
IDM inference config SHA
HTDemucs input view ID
HPA-325 scoring version
Crux commit
```

Per-item rows retain:

```text
simfile/source identity
upstream HTDemucs status/stem/input identity
OaF retained prediction identity
IDM execution disposition/failure code
IDM prediction identity
wall time/RTF when executed
mapping diagnostics
```

Predictions stay immutable and source/descriptor/config keyed through the existing `prediction_path()` rails. Resume reads and validates the immutable artifact before reuse.

Keep the runner concrete. HPA-396 is not the reason to merge OaF, MuScriptor, HPA-328, and IDM run ledgers into a generic framework.

## Scoring and comparison

### Cohort scoring

Reuse unchanged:

```text
load_reference_set_manifest()
load_reference_timing_manifest()
preflight_reference_mappings()
cohort_item_from_validated_prediction_artifact()
cohort_item_without_prediction()
score_cohort()
write_cohort_reports()
```

Publish complete-population OaF and IDM HPA-325 reports for the HTDemucs input view. Velocity/confidence do not affect onset matching.

### Paired comparison

Add one small IDM driver around `published_comparison.py` primitives. Publish:

```text
summary.json
summary.md
paired_per_song.csv
paired_per_class.csv
```

The pairable set requires:

- same simfile ID;
- both OaF and IDM successful;
- same source audio SHA-256;
- **same input audio SHA-256**;
- identical score-key grid.

Unlike HPA-328's full-mix-vs-stem comparison, HPA-396 does not opt out of input-hash equality. The entire point is model-vs-model on byte-identical HTDemucs input.

Report complete population/failure counts separately from paired-success deltas.

### Required evidence

At minimum publish:

- aggregate micro/macro precision/recall/F1 at 30/50/100 ms, raw and aligned;
- per-class metrics on the existing six common classes;
- paired per-song and per-class OaF→IDM deltas;
- mapped/unmapped native class counts;
- IDM onset-activation and MIDI-projected velocity diagnostics;
- inference wall time and RTF;
- worker startup time separately from item RTF;
- peak memory only if it is available without adding a profiler subsystem;
- failure histogram;
- fixed full-mix smoke section;
- representative largest-regression/largest-improvement rows derived from the complete paired CSV rather than a hand-curated hidden list.

No significance/bootstrap layer is added in HPA-396.

## Full-mix compatibility smoke

Create one fixed pre-score `runtime/idm/smoke.json` with exactly five HPA-327 members selected from reference/source characteristics before IDM F1 is known. The reasons should cover useful operational variety such as short/long/dense/sparse/non-drum-heavy.

The smoke uses the historical canonical full-mix view:

```text
crux.oaf-full-mix-mono44k1-pcm16/v1
```

Use retained full-mix canonical input where available; otherwise rematerialize from the authoritative source through the existing `resolve_source_audio()` + `materialize_full_mix_audio()` path and verify the expected source identity.

The smoke is a compatibility diagnostic only:

- no F1 threshold chooses or rejects the model;
- no tuning occurs after inspecting the five songs;
- no smoke result changes the fixed stem-pilot membership/config;
- its metrics are reported under a separate heading and never merged into headline OaF↔IDM stem numbers.

If full-mix smoke cannot execute reproducibly, record it as unavailable and continue the primary stem pilot only if the frozen IDM runtime itself is valid on HTDemucs inputs.

## Failure semantics

Keep stable native failure codes small and concrete. Suggested classes:

```text
upstream_stem_unavailable
retained_input_invalid
retained_oaf_prediction_invalid
worker_start_failed
worker_protocol_failed
inference_failed
native_event_invalid
prediction_artifact_invalid
prediction_output_conflict
prediction_publish_failed
```

Map them into the existing closed HPA-325 cohort reasons rather than extending the scorer's reason enum just for IDM detail. Preserve the native code in the HPA-396 run snapshot/report histogram.

A runtime/model/descriptor identity failure before the first item is fatal preflight and publishes no misleading partial scored run. Item-local audio/inference failures remain per-song failures and do not abort the whole population.

## Operational sequence

The real execution order is intentionally front-loaded for feasibility:

1. build the isolated runtime from the exact git revision;
2. verify code/weight license evidence;
3. acquire/hash the official committed model config/checkpoint;
4. run the official demo/model load on one upstream example or local known-good WAV;
5. verify observed `train_classes`, sample rate, frame rate, onset/velocity tensor shapes, and peak-picking behavior;
6. run deterministic adapter fixtures and one real HPA-328 HTDemucs input;
7. freeze `runtime/idm/model.json`, runtime lock hash, device, dtype, and event extraction config;
8. freeze the five-song full-mix smoke membership before inspecting its scores;
9. run the smoke diagnostic;
10. run the exact HPA-328 20–30-row HTDemucs population;
11. rebuild OaF/IDM HPA-325 reports and publish paired comparison;
12. review failure/runtime/velocity/error evidence without tuning the frozen config.

If steps 1–6 expose an upstream incompatibility that cannot be resolved without changing the scientific target, HPA-396 should be marked blocked with that exact evidence rather than silently substituting another checkpoint/model.

## Test strategy

### Offline unit/contract tests

CI does not download IDM weights or install the IDM runtime. Use fake worker/model records for:

- strict model lock parse/render/hash/derived model ID;
- runtime lock/model file hash verification;
- descriptor third-family validation;
- prediction-v2 IDM render/read/render and OaF/MuScriptor regression;
- exact output-bin↔train-class native invariants;
- mapping table, including `TT_LMT -> common tom / detailed null`;
- worker ready payload validation;
- raw onset/velocity validation and velocity-MIDI projection boundaries;
- non-finite/malformed tensors/events;
- fixed handoff population reconstruction;
- exact HPA-328 input/prediction hash binding;
- failure preservation for rows without HTDemucs input;
- resume/conflict behavior;
- HPA-325 scoring reuse;
- exact pairable source+input identity;
- separate full-mix smoke reporting;
- CLI argument/result contract.

### Real pre-score gates

Real model execution is operational evidence, not a CI dependency:

- exact upstream package/runtime build;
- official model load;
- one official/demo smoke;
- one real HPA-328 HTDemucs input;
- five fixed full-mix smokes;
- then the frozen 20–30-row pilot.

No scored production run starts until the model/runtime lock is committed and its local bytes pass verification.

## Files expected to change during implementation

The implementation should stay close to:

```text
runtime/idm/pyproject.toml
runtime/idm/uv.lock
runtime/idm/worker.py
runtime/idm/model.json
runtime/idm/smoke.json
scripts/freeze_idm_model.py
src/benchmark/idm_model.py
src/benchmark/backends/idm.py
src/benchmark/backend_identity.py
src/benchmark/prediction_artifact.py
src/benchmark/taxonomy.py
src/benchmark/mapping.py
src/benchmark/idm_pilot_run.py
src/benchmark/idm_comparison.py
src/cli/benchmark.py
tests/benchmark/test_idm_model.py
tests/benchmark/test_idm_backend.py
tests/benchmark/test_backend_identity.py
tests/benchmark/test_prediction_artifact.py
tests/benchmark/test_taxonomy.py
tests/benchmark/test_mapping.py
tests/benchmark/test_idm_pilot_run.py
tests/benchmark/test_idm_comparison.py
tests/test_cli_benchmark.py
```

Some existing acceptance/coverage files will also need focused regressions. Do not create files solely to mirror every HPA-395 module if an existing public seam already covers the need.

## Explicit non-goals

HPA-396 does not:

- run IDM over the full broad corpus;
- run a separator or regenerate HPA-328 stems;
- feed IDM-separated audio into OaF;
- require or score IDM reconstructed stems;
- use manual onset overrides;
- tune peak-picking thresholds, class maps, device, dtype, or checkpoint after seeing pilot F1;
- train or fine-tune IDM;
- add velocity-aware scoring;
- alter HPA-325 matching/alignment/tolerance behavior;
- alter HPA-327 membership;
- publish the final all-model benchmark matrix owned by HPA-562;
- publish final model recommendations owned by HPA-329;
- add IDM to production API/backend registration;
- add a generic external-model runtime/runner/experiment framework.

## Completion criteria

HPA-396 is complete when:

1. the exact official IDM code/runtime/model/config/license identity is frozen and reproducible locally;
2. the worker/adapter produces validated native onset + activation + velocity evidence without manual override or required synthesis;
3. every HPA-328 handoff row is represented in the HPA-396 primary population;
4. successful primary rows use byte-identical HPA-328 HTDemucs canonical inputs and retained OaF prediction evidence;
5. OaF and IDM are scored by the unchanged HPA-325 scorer and paired only on exact source/input identity;
6. aggregate/per-class/per-song paired evidence plus class/velocity/runtime/failure diagnostics are published;
7. the five-song full-mix diagnostic is visibly separate;
8. tests preserve OaF/MuScriptor prediction-v2 behavior; and
9. no generic benchmark framework, full-corpus IDM path, tuning loop, training path, or production backend is introduced.
