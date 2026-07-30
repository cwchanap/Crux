# HPA-320: Frozen OaF Drums Transcription Backend Design

**Status:** Review amendments applied; phased implementation planning may proceed,
while final OaF sealing and integration remain blocked on accepted native-`amd64`
evidence

**Issue:** [HPA-320](https://linear.app/cwchanap/issue/HPA-320/validate-and-freeze-the-existing-oaf-drums-transcription-backend)

**Date:** 2026-07-26

## Context

HPA-320 makes one Crux transcription backend reproducible enough to be a trustworthy
benchmark dependency. It freezes the released Magenta Onsets and Frames (OaF) Drums
model, proves that every inference tensor is restored, gives predictions an immutable
identity, and makes backend failure fatal instead of silently changing algorithms.

The authoritative upstream pairing is:

- Magenta source commit
  `94529798dfbbb14c27ddfd76f23027dc8e2ce185`;
- the upstream `drums` configuration;
- the released `e-gmd_checkpoint.zip` checkpoint; and
- the upstream 8-hit drum training-data map.

The current Crux TensorFlow 2 path is not that backend. It changes feature extraction,
model dimensions, thresholds, postprocessing, and class interpretation. Its converter
also accepts a partial conversion, and an audit of the released checkpoint found that
the converter's current name and layer matching restores zero of the 78
inference-state tensors. The local HDF5 file can be loaded, but it has no upstream
provenance and cannot establish equivalence to the released model.

HPA-320 therefore freezes the original TensorFlow 1 inference behavior in an isolated
runtime. A future native TensorFlow 2 implementation is a different backend identity
until an explicit parity suite proves otherwise.

## Audit Evidence

The implementation must preserve the following audited evidence in the checked-in
backend lock and its review history.

### Upstream source

| Property | Frozen value |
| --- | --- |
| Repository | `https://github.com/magenta/magenta` |
| Commit | `94529798dfbbb14c27ddfd76f23027dc8e2ce185` |
| Configuration | `drums` |
| Model family | Onsets and Frames Drums / E-GMD |

The exact upstream files needed by the runner are vendored from that commit. A source
manifest records each upstream path, its SHA-256, and the Apache-2.0 notice. The
runner never clones a repository or resolves a mutable branch at runtime.

### Released checkpoint

| Artifact | SHA-256 |
| --- | --- |
| `e-gmd_checkpoint.zip` | `09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0` |
| `model.ckpt-569400.index` | `475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a` |
| `model.ckpt-569400.data-00000-of-00001` | `6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5` |
| `model.ckpt-569400.meta` | `e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422` |

The informational acquisition URL is:

```text
https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/e-gmd_checkpoint.zip
```

Content hashes, not the URL, are authoritative. Benchmark execution never downloads
weights. A separate explicit setup step may acquire the archive, verify the archive
hash, extract it to a temporary directory, verify every component, and only then
install the components in a content-addressed model cache.

The only network-capable model setup interface is explicit:

```bash
uv run crux benchmark prepare-backend \
  --backend magenta-egmd-tf1-94529798-8hit-v1 \
  --download

uv run crux benchmark prepare-backend \
  --backend magenta-egmd-tf1-94529798-8hit-v1 \
  --archive /operator/staging/e-gmd_checkpoint.zip
```

`--download` and `--archive` are mutually exclusive. With neither option, the command
only verifies an existing cache. The download form may use only the informational URL
in the checked-in lock. All forms stage outside the final cache, verify the archive
when present, reject archive path traversal, require exactly the three locked
components, and atomically publish only a fully verified component set at:

```text
artifacts/benchmark/model-cache/sha256/<model-artifact-set-sha256>/
```

Exit `0` means the exact component set is installed and reverified, exit `1` means
acquisition or local staging could not complete and the cache was left unchanged, and
exit `2` means supplied or cached bytes contradict the lock, the lock is invalid, or
atomic cache publication failed. The command never substitutes another URL or
checkpoint. Inference commands are always offline and only read this cache.

`prepare-backend` is a setup-only reporting exception: it does not launch a backend
and does not write a verification, execution, or legacy-score report. After Click
parsing succeeds, it writes exactly one canonical one-line JSON summary with
`status`, `exit_code`, `report_path`, and `report_sha256`; both report fields are
null. Status is `ready` for exit `0`, `acquisition_failed` for exit `1`, or
`integrity_failed` for exit `2`. Progress and sanitized diagnostics use standard
error. A Click usage error occurs before this summary exists.

During the design audit, a fresh download from the upstream URL matched the existing
three cached checkpoint components byte-for-byte. The checked-in lock preserves the
component hashes above rather than trusting that observation indefinitely.

The official checkpoint contains 130 TensorFlow variables. The frozen inference graph
requires 78 model-state tensors. The other 52 entries are optimizer, training-step, or
other non-inference state and must be classified explicitly in the coverage report;
they must not be ignored by a broad partial-restore option.

### Current local HDF5 artifact

The existing untracked or ignored `tf2_model.weights.h5` observed during the audit had
SHA-256:

```text
d36ced8b2ee241bc37ad6fbb918ba38e95d666350dd4888bca59a1243bf4d10e
```

This hash is audit evidence only. The HDF5 file is not part of the frozen backend,
must not appear in its lock, and is rejected when benchmark mode requests the frozen
OaF backend.

### Legacy conversion coverage audit

The zero-of-78 conclusion must be preserved as a dedicated checked-in evidence
artifact:

```text
docs/superpowers/evidence/hpa-320/legacy-tf2-conversion-coverage.json
```

under strict schema `crux.legacy-tf2-conversion-coverage/v1`. It records the released
checkpoint artifact-set SHA-256, the observed HDF5 SHA-256, converter and TensorFlow 2
architecture source-manifest hashes, the exact matching algorithm and version, the
required-inference inventory SHA-256, restored and unmatched tensor lists, and the
zero restored count. The native seal evidence references this report's SHA-256.
Neither the prose conclusion nor a count without the exact inventories is sufficient
audit evidence.

## Goals

- Define one immutable, inspectable identity for the released OaF Drums backend.
- Run the exact upstream inference graph and postprocessing in a pinned TensorFlow 1
  CPU runtime.
- Verify all 78 required inference tensors before any benchmark item runs.
- Preserve native OaF Drums output classes, pitches, timing, confidence, and velocity
  in a deterministic structured artifact.
- Define a backend-agnostic persisted event envelope with source-audio, input-view,
  model, native-output, and explicit canonical-mapping state.
- Make the official backend the default for frozen benchmark inference commands.
- Require an explicit backend identifier to use the heuristic transcriber.
- Fail before prediction when the checkpoint, runtime, protocol, or smoke fixture does
  not match the lock.
- Keep ordinary Crux tests fast through a fake runner while exercising the real
  checkpoint in a dedicated integration job.
- Leave a narrow backend interface that a later parity-proven runtime can implement
  without changing benchmark artifacts silently.

## Non-goals

- Retraining, fine-tuning, threshold tuning, or architecture changes.
- Treating the existing HDF5 file as a valid released checkpoint conversion.
- Building a TensorFlow 2 replacement in HPA-320.
- Selecting or preparing the benchmark corpus.
- Defining the shared benchmark drum taxonomy.
- Defining scoring, matching tolerances, or aggregate metrics.
- Running or resuming the complete corpus.
- Comparing Crux accuracy with external transcription products.
- Changing the FastAPI `DrumTranscriber` request path.
- Adding automatic checkpoint downloads or automatic fallback behavior.

## Sibling-Issue Ownership

HPA-320 owns inference identity and native prediction persistence. It intentionally
stops before three downstream contracts:

- **HPA-324** owns the shared canonical drum taxonomy and the immutable mapping from
  a model-native class to that taxonomy. HPA-320 preserves enough native information
  to make that mapping lossless and auditable, but it does not name an OaF native
  class as a benchmark canonical class.
- **HPA-325** owns model-agnostic event matching and scoring. HPA-320 provides a
  deterministic backend-agnostic reader and round-trip tests for native predictions;
  it does not define score semantics.
- **HPA-326** owns full-corpus execution, resumability, and per-song run manifests.
  HPA-320 validates one backend, one item at a time, and supplies the adapter that
  HPA-326 will orchestrate.

The existing `transcribe-and-score` command remains a legacy compatibility workflow,
not a frozen-backend workflow. It requires an explicit `--backend`; its executable
legacy selections are `legacy-tf2-h5-v0` and `heuristic-onset-v1`, with no implicit
transcriber or fallback. Its report identifies
`workflow_mode: "legacy_compatibility"` and the requested backend ID, and it must
never identify its prediction as the frozen OaF backend. The command recognizes
`magenta-egmd-tf1-94529798-8hit-v1` only to return the typed result
`canonical_mapping_required` without scoring until HPA-324 and HPA-325 provide the
mapping and scorer orchestration. HPA-320 does not expand or redefine the command's
existing legacy scoring behavior.

## Chosen Architecture

### Isolated exact-runtime runner

The frozen backend runs as a persistent subprocess in a pinned Linux container:

| Runtime property | Frozen value |
| --- | --- |
| Platform | `linux/amd64` |
| Execution | CPU only; GPU devices disabled |
| Base image | `python:3.7.17-slim-bullseye` |
| Base image digest for `linux/amd64` | `sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673` |
| Python | `3.7.17` |
| TensorFlow | `tensorflow==1.15.5` |
| TensorFlow wheel | `tensorflow-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl` |
| TensorFlow CPython 3.7 Linux wheel SHA-256 | `29831dda98d668067de75403b2fca0d06a2f026ef6f217fa2ca873c20b4ee4d3` |

Every direct and transitive Python package is pinned with a distribution hash in a
checked-in runtime requirements lock. System packages are installed from a
snapshot-addressed Debian source and recorded by exact package version and package
hash in the runtime lock. The built image is referenced by its `linux/amd64` OCI
manifest digest, never by a mutable tag.

The runtime lock records:

- base image name, platform manifest digest, and Python version;
- every Python distribution name, version, filename, and SHA-256;
- the snapshot-addressed Debian repository URL, signed Release-file SHA-256, and
  every installed system package version, `.deb` filename, and SHA-256;
- the vendored upstream source-manifest SHA-256;
- the runner source-manifest SHA-256;
- the exact deterministic process environment listed below;
- exact standard-error drain ring-buffer, read-chunk, and logical-line byte bounds;
- TensorFlow build and ABI information reported at startup; and
- the final runner image manifest digest.

The image build context excludes the final runtime lock, avoiding a self-referential
image digest. The lock can therefore attest to the image it describes.

The host/container launcher constructs a fresh environment containing exactly these
lock-recorded values and supplies it to the Python interpreter before `exec`. In
particular, `PYTHONHASHSEED` must be present before CPython starts; assigning it to
`os.environ` inside the runner is forbidden because that does not reseed the running
interpreter:

| Environment variable | Frozen value |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `-1` |
| `OMP_NUM_THREADS` | `1` |
| `MKL_NUM_THREADS` | `1` |
| `OPENBLAS_NUM_THREADS` | `1` |
| `PYTHONHASHSEED` | `0` |
| `TF_NUM_INTRAOP_THREADS` | `1` |
| `TF_NUM_INTEROP_THREADS` | `1` |

As its first startup step, before importing TensorFlow or any numeric library, the
runner validates the complete environment allowlist and fails the handshake on a
missing, extra, or mismatched value. It then calls `random.seed(0)`,
`numpy.random.seed(0)`, and `tf.set_random_seed(0)`, and does not enable mixed
precision or runtime graph rewrites outside TensorFlow 1.15.5 defaults. A conflicting
inherited value cannot override the locked environment.

The session passes explicit positive `ConfigProto` values of `1` for TensorFlow
inter-op and intra-op parallelism. Those session values are authoritative.
TensorFlow 1.15.5 also reads `TF_NUM_INTEROP_THREADS` and
`TF_NUM_INTRAOP_THREADS` in its fallback paths, so the matching environment values
remain locked defense-in-depth controls rather than being treated as inert. The smoke
test proves the resulting output instead of assuming these settings alone guarantee
determinism. The runtime evidence cites the exact v1.15.5
`tensorflow/core/common_runtime/process_util.cc` and `local_device.cc` source blobs
that establish this precedence.

Official verification and official prediction publication require a native
`linux/amd64` worker. The host launches the image with `--platform linux/amd64` and
must establish from trusted host/orchestrator runtime metadata that the underlying
worker is native `amd64`; `uname` inside an emulated container is not sufficient
evidence. Accepted evidence is exactly one of:

- a GitHub-hosted runner whose API job record has an approved Linux X64 label and
  whose sealed report records `RUNNER_OS=Linux`, `RUNNER_ARCH=X64`, workflow commit,
  job ID, immutable run URL, and the SHA-256 of that API record;
- an approved orchestrator-signed worker attestation that identifies the physical
  worker architecture; or
- an approved native seal-host record that combines host-side `linux/x86_64`
  evidence with container-daemon or worker metadata.

A bare local host may run diagnostics, but it cannot seal a lock or publish an
official verification unless it is registered and attested through the third form.
Apple Silicon or other emulated execution is diagnostic only. When
explicitly allowed through `--allow-emulated-diagnostics`, it emits a verification or
execution report whose strongest possible result is
`status: "environment_unsupported"` with exit `1`, and cannot seal locks, return
`verified`, or publish an official prediction artifact. Without that option, the host
returns the same unsupported status before launching the image. A genuine lock,
checkpoint, or protocol integrity failure observed during diagnostics still returns
`failed` with exit `2`.

Every runner launch uses:

- `--network=none`;
- a read-only root filesystem;
- read-only backend-lock, runtime-lock, model-cache, and input mounts;
- a fresh process environment containing only the lock-enumerated allowlist;
- all Linux capabilities dropped and `no-new-privileges`;
- bounded, execution-profile-locked `tmpfs` mounts for `/tmp` and `/dev/shm`;
- locked CPU, memory, and PID limits; and
- the numeric non-root UID/GID proven by the native seal pass.

If the provisional image cannot pass the complete native smoke and coverage suite
under the proposed non-root UID/GID, sealing stops for design review; it does not
silently run as root. Exact tmpfs sizes, resource limits, UID/GID, startup deadline,
request deadline, `max_input_audio_frames`, Debian snapshot, and installed
system-package set are seal-required values with no code defaults.

### Native-amd64 calibration and sealing prerequisite

Before Phase A can finalize the OaF locks or its real-checkpoint runner integration,
or satisfy HPA-320, a narrowly scoped evidence pass on a reference native
`linux/amd64` host must check in:

```text
config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json
```

under schema `crux.backend-seal-evidence/v1`. It records which accepted native-host
evidence form was used and its immutable reference/signature, base-manifest
verification, Debian snapshot and package hashes, wheel filenames and hashes, numeric
UID/GID, tmpfs sizes, CPU/memory/PID limits, standard-error drain bounds, startup and
request deadlines, the proposed positive integer `max_input_audio_frames`,
measurements at that exact bound, tensor inventories, smoke generation inputs and
outputs, the security scan/advisory snapshot, the preserved OCI-layout archive and
manifest/config/layer hashes, and hashes of every reviewed evidence artifact.
Missing values, sentinel values, and prose such as "auto" or "unlimited" are invalid.

Like the smoke oracle, seal evidence excludes final backend-lock and runtime-lock
hashes so the final locks can reference its SHA-256 without a content cycle.

This pass is a disposable design/calibration spike, not production implementation:
only reviewed evidence, the oracle, and sealed values may flow into the final OaF
locks and inference-relevant runner implementation.

The bootstrap sequence is:

1. Build a provisional image from the pinned source, base manifest, package
   distributions, and runner manifests without either final lock in the image. Resolve
   the mutable base tag only to inspect it, prove that its `linux/amd64` manifest is
   the pinned
   `sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673`,
   and build `FROM` that digest. A registry mirror is acceptable only when it supplies
   the same manifest and layer bytes.
2. On native `linux/amd64`, construct the exact prediction graph, generate the tensor
   coverage report, calibrate and test the complete resource/input profile, and
   generate the smoke oracle without lock hashes.
3. Review the coverage report, oracle, deterministic artifact bytes, resource
   measurements, and seal-evidence record. Final OaF lock publication and
   inference-relevant runner integration remain blocked until this review accepts
   every exact value; host-side planning and fake-runner implementation need not
   wait.
4. Seal the backend and runtime locks, referencing the reviewed evidence hash and the
   already-built image manifest digest.
5. Mount the final locks into that image and rerun the complete native verification
   flow twice in-process and once in a fresh process.

Any change to an inference-relevant sealed value, including the graph, hparams,
checkpoint set, package bytes, postprocessing, `max_input_audio_frames`, or
serialization, requires a new backend ID and a new seal. A changed execution-only
resource profile requires a new attested profile and another native verification; it
does not change prediction identity unless an inference-relevant value also changes.

### Implementation phases and gates

HPA-320 remains one authoritative design and parent issue, but implementation is
split into three independently reviewable phases:

1. **Phase A — sealed OaF runtime:** audit evidence, vendored source and
   instrumentation, provisional and final locks, isolated runner, tensor coverage,
   `verify-backend`, and deterministic smoke verification.
2. **Phase B — common prediction contract:** host protocol, provenance and canonical
   input view, strict common artifact reader/writer, operational reports, and
   `transcribe-one`.
3. **Phase C — compatibility adapters:** isolated `heuristic-onset-v1`, explicit
   `legacy-tf2-h5-v0` rewiring, the typed legacy-score report, and optional MIDI.

Phases B and C may be planned, implemented, and unit-tested against fake runners
before native seal evidence exists. Phase A may likewise implement host scaffolding
and the disposable calibration spike, but it cannot finalize locks, merge its
real-checkpoint integration, or claim sealed behavior until the native seal evidence
and generated oracle are accepted. Cross-phase integration and overall HPA-320
completion remain gated on Phase A.

### Host-side backend interface

Benchmark orchestration depends on a small interface independent of FastAPI:

```python
class TranscriptionBackend(Protocol):
    def descriptor(self) -> BackendDescriptor: ...
    def verify(self) -> BackendVerification: ...
    def transcribe(self, audio: CanonicalAudio) -> NativePrediction: ...
    def close(self) -> None: ...
```

`BackendDescriptor` is a schema-discriminated immutable union. Its frozen OaF variant
contains:

- `descriptor_schema`;
- `backend_id`;
- `protocol_schema` and `prediction_schema`;
- `backend_lock_sha256`;
- `model_id` and `architecture_id`;
- `model_artifact_set_sha256`;
- `upstream_source_commit`;
- `native_output_space_id` and `training_data_map_id`;
- `native_metadata_schema_id`;
- `runtime_lock_sha256`; and
- `runtime_image_manifest_digest`.

The exact descriptor schema is `crux.transcription-backend-descriptor/v1`; no other
keys are permitted. Its identity is the SHA-256 of the descriptor object serialized
as UTF-8 JSON with keys sorted lexicographically, no insignificant whitespace,
duplicate or unknown keys forbidden, and no trailing newline. Every descriptor field
is a string, so JSON numbers are forbidden. Request IDs, timestamps, paths, host
facts, verification status, and other operational values are excluded. The handshake
carries both the complete object and that SHA-256, and the host independently
reproduces the bytes and hash.

The official backend identifier is:

```text
magenta-egmd-tf1-94529798-8hit-v1
```

The model, architecture, and native output-space identifiers are:

```text
model_id: magenta-egmd-ckpt-569400-v1
architecture_id: magenta-oaf-model-tpu-drums-v1
native_output_space_id: magenta-oaf-midi88-a0-v1
training_data_map_id: magenta-egmd-data-8hit-94529798-v1
native_metadata_schema_id: magenta-oaf-native-metadata-v1
```

`tpu` in the upstream architecture ID names the upstream graph family; it does not
authorize TPU placement. This backend remains CPU-only.

The explicit non-model baseline identifier is:

```text
heuristic-onset-v1
```

The current HDF5 compatibility path is named:

```text
legacy-tf2-h5-v0
```

That compatibility ID labels an existing unvalidated workflow; it is not a frozen
`TranscriptionBackend` descriptor and cannot pass backend-lock verification.

No descriptor field may be inferred from a filename or directory name. The host loads
the lock, starts the runner, and compares every handshake field against the lock before
accepting the backend.

The host mounts the exact backend and runtime lock files read-only into the runner.
Before constructing the graph, the runner strict-parses and hashes both mounted files,
checks their runner-visible runtime, vendored-source, model, and checkpoint facts
against its own environment, and emits the resulting lock hashes in the handshake.
The host independently verifies the host-adapter source manifest, hashes its local
lock files, and verifies the selected OCI image manifest digest before accepting those
runner-reported values. The hashes are therefore runtime facts, not constants baked
into the image build context.

The heuristic backend has its own immutable descriptor and parameter lock. It may run
only when the operator passes its complete backend identifier. It is never selected
because a model is missing, corrupt, slow, or incompatible, and its artifacts cannot
claim the OaF backend identifier.

Its exact descriptor schema is `crux.heuristic-backend-descriptor/v1`, with only
`descriptor_schema`, `backend_id`, `prediction_schema`,
`adapter_source_manifest_sha256`, `parameter_lock_sha256`, `model_id`,
`architecture_id`, `native_output_space_id`, and `native_metadata_schema_id`. The
last four identities are respectively
`crux-heuristic-onset-nonmodel-v1`,
`librosa-onset-centroid-zcr-v1`, `crux-heuristic-midi7-v1`, and
`crux-empty-native-metadata-v1`. It uses the same canonical descriptor hashing rules
as OaF. It has no runtime image, upstream model, checkpoint artifact set, or training
data map, and may not fabricate those identities.

The parameter lock uses `crux.heuristic-parameter-lock/v1` and includes the complete
resolved algorithm values, adapter source-manifest hash, and the names, versions,
distribution filenames, and SHA-256 values of Librosa, NumPy, SciPy, Numba,
SoundFile, and every transitive numeric/audio dependency. Verification rejects an
installed distribution mismatch. Changing any locked algorithm or package byte
requires a new heuristic backend ID.

The heuristic executes in a small content-addressed virtual environment keyed by the
parameter-lock SHA-256, separate from Crux's main Python environment. Its lock also
records the exact Python implementation/version and platform-compatible distribution
set. Routine Crux dependency upgrades therefore do not alter or retire the historical
`heuristic-onset-v1` environment. A changed locked interpreter, algorithm, adapter,
or distribution byte still requires a new backend ID; version-only matching is never
accepted as a substitute for package-byte identity.

### Source and repository identity

The checked-in backend lock cannot contain the Git commit that introduces the lock
without becoming self-referential. Instead:

- the backend lock contains content hashes for every adapter, runner, vendored
  upstream, model, runtime, schema, and mapping input that affects inference;
- each HPA-320 execution emits a
  `crux.backend-execution-attestation/v1` record with the exact Crux Git commit and
  whether the checkout was dirty;
- the inference-relevant file set is the exact union of the paths enumerated by the
  host-adapter and runner source manifests transitively referenced by the backend and
  runtime locks; and
- a dirty checkout records a canonical SHA-256 manifest of changes to that set, or is
  rejected when strict benchmark mode is enabled.

Each source manifest also declares the repository-relative directory
`covered_roots` from which its enumerated files were selected. Untracked files below
those roots participate in the dirty check even though they could not have been
listed when the source manifest was sealed.

The changed-file manifest payload is exactly one canonical JSON array. Each entry has
only `path`, `status`, and `sha256`. `path` is a repository-relative, UTF-8, POSIX
path with no empty, `.` or `..` component, backslash, absolute prefix, or
normalization alias. `status` is exactly `modified`, `deleted`, or `untracked`.
`sha256` hashes the exact current bytes for modified and untracked files and is null
for a deletion. Entries are sorted by the UTF-8 bytes of `path`; duplicate paths are
invalid. The array uses lexicographically sorted object keys, no insignificant
whitespace, one final Unix newline, and no byte-order mark. Its SHA-256 includes that
newline.

Strict mode rejects modified or deleted enumerated content and any untracked content
under a covered root. Changes outside the manifest paths and covered roots, such as
unrelated documentation, may make the repository dirty but do not invalidate
inference; they remain visible through the Git commit and dirty-state fields.

Present entries must be regular files rather than symlinks, and paths must already be
Unicode NFC. A rename is represented as one `deleted` entry plus one `untracked`
entry. Any path whose Unicode normalization changes or whose bytes cannot be decoded
as UTF-8 is rejected.

The execution attestation also records the effective container CPU quota or pinning,
memory limit, PID limit, startup deadline, and request deadline. These are execution
conditions rather than static image identity, so they do not belong in the runtime
lock. HPA-326 must freeze them in each run configuration and reference the resulting
attestation.

The backend identity is therefore stable across unrelated Crux documentation changes,
while every execution remains attributable to the exact host revision. HPA-326 run
manifests reference this attestation rather than redefining it.

### Persistent process

The model is built and restored once per runner process. The host reuses that process
for sequential requests, with one request in flight at a time. This removes repeated
model-load cost without introducing concurrent TensorFlow state.

Runner process death, handshake drift, malformed protocol output, or a request timeout
invalidates the process. The host does not restart it invisibly within the same
benchmark attempt.

## Frozen Model Semantics

### Effective upstream hyperparameters

The backend lock records the complete resolved upstream `drums` hparams. The following
values are called out because changing any of them creates a different backend:

| Property | Frozen value |
| --- | --- |
| Input sample rate | `44100` Hz |
| Spectrogram type | `mel` |
| Spectrogram mel bins | `250` |
| HTK mel scale | `true` |
| Log-amplitude spectrogram | `true` |
| Minimum spectrogram frequency | `30.0` Hz |
| Effective spectrogram hop | `spec_hop_length=512` samples |
| Drums config `hop_length` | `441` samples |
| Emitted frame rate | `44100 / 512 = 86.1328125` frames/second |
| Librosa audio loader | disabled |
| Training audio transform | disabled |
| Conv filters | `16, 16, 32` |
| Fully connected size | `256` |
| Onset LSTM units | `64` |
| Offset LSTM units | `256` |
| Velocity / frame LSTM units | `0 / 0` |
| Combined LSTM units | `256` |
| Bidirectional recurrent layers | `true` |
| Offset network / shared conv features | `true / false` |
| Piano-range output bins | `88` |
| `drums_only` | `true` |
| `onset_length` | `0` |
| Velocity scale / bias | `127 / 0` |
| Conv dropout keep amounts | `1.0, 0.25, 0.25` |
| Fully connected dropout keep amount | `0.5` |
| Acoustic / combined RNN dropout keep probability | `0.50 / 1.0` |
| Frame threshold | `0.5` |
| Onset selection | strict `onset_probability > 0.5` |
| Offset threshold | `0.0` |
| Peak picking / minimum gap | none |
| Emitted drum note duration | `0.05` seconds |
| Viterbi decoding | disabled |
| Training drum data map | upstream `8-hit` |
| Inference drum prediction map | empty / disabled |

The upstream drums configuration sets `hop_length=441`, but the mel feature pipeline
at the frozen commit consumes `spec_hop_length`, whose resolved value is `512`.
Implementations must record both values and use `512` for feature extraction. Replacing
it with `441` would be a backend change, even though `441` appears intuitive at
44.1 kHz. The complete resolved-hparams lock and its source note must carry this
distinction; copying only the visually prominent `hop_length` field is invalid.

The backend uses the exact upstream TensorFlow 1 graph construction and inference
postprocessing, except for the identity-bound instrumentation and nonfinite safety
rule specified below. It does not recreate the graph with Keras approximations.

The listed dropout keep amounts are resolved configuration evidence, but upstream
dropout is gated by training mode and is inactive in `PREDICT`. The seal must prove
that no stochastic dropout operation is active in the prediction graph. The values
remain locked and identity-relevant under v1's conservative complete-hparams policy;
their presence does not claim that they numerically affect prediction.

`velocity_lstm_units=0` disables only the recurrent layer in the velocity acoustic
branch. The frozen graph still runs that branch's convolutional features and dense
`onset_velocities` head. OaF velocity is therefore a required, non-null prediction;
missing velocity-head tensors are a fatal tensor-coverage failure rather than a reason
to emit null velocity.

Upstream event time is evaluated in binary64 in this exact order:

```text
frames_per_second = 44100 / 512
frame_length_seconds = 1.0 / frames_per_second
time_sec_raw = frame_index * frame_length_seconds
```

The runner passes `time_sec_raw` to the artifact quantizer without recomputing it as
an alternative algebraic expression. Confidence is the selected TensorFlow
float32 onset activation converted directly to a host binary float before the same
artifact quantizer.

For an emitted onset, velocity uses the upstream `_unscale_velocity` order:

```text
require raw_velocity to be finite
clamped = max(min(raw_velocity, 1.0), 0.0)
unscaled = clamped * 127 + 0
velocity_midi = int(unscaled)
```

`int` truncates toward zero; the implementation must not round. A nonfinite raw
velocity is a per-item validation failure before clamping, rather than inheriting a
library-specific NaN result. This is an intentional adapter safety divergence from
the frozen upstream helper, which converts a NaN intermediate to velocity zero. The
divergence is named in the backend lock, covered by a focused test, and included in
the runner source manifest and seal evidence; it must not be described as exact
upstream behavior.

Upstream `infer_util.predict_sequence` returns a serialized `NoteSequence`, while
`magenta.music.sequences_lib.pianoroll_to_note_sequence` selects onset frames
internally and does not expose them. HPA-320 therefore applies a narrowly scoped,
hash-locked instrumentation patch to the vendored conversion path. For each emitted
note it returns paired metadata containing the original `start_frame`, pitch, and raw
velocity from the same postprocessing loop. The runner joins confidence from
`onset_probs[start_frame, pitch]` and uses that same `start_frame` as `frame_index`
for time evaluation; it never reconstructs the frame from the rounded or binary64
`note.start_time`.

The instrumentation patch is enumerated in the vendored-source manifest and seal
evidence. A parity test runs patched and unmodified upstream conversion on the same
locked tensors and requires byte-identical serialized `NoteSequence` output. Any
instrumentation change or parity failure requires design review and a new backend
identity.

### Native output space and upstream 8-hit data map

The graph emits 88 native MIDI-relative bins:

```text
model_output_bin = 0..87
native_midi_note = model_output_bin + 21
native_class_id = "midi_" + native_midi_note
```

Native class IDs are unpadded decimal strings; representative values are `midi_21`,
`midi_36`, and `midi_108`.

The frozen upstream `drums` config sets `drum_data_map='8-hit'` for labels during
training, but leaves `drum_prediction_map=''` during inference. The official
transcription command therefore thresholds all 88 native bins; it does not apply an
8-hit prediction collapse after inference. HPA-320 must not silently enable that
optional collapse.

The lock still preserves the exact upstream training groups because they explain the
eight intended positive base pitches and are required evidence for HPA-324:

| Upstream group ID | Base MIDI / output bin | Complete upstream member pitches |
| --- | --- | --- |
| `kick` | `36 / 15` | `36` |
| `snare` | `38 / 17` | `38, 40, 37, 39` |
| `toms` | `48 / 27` | `48, 50, 45, 47, 43, 58, 64` |
| `hihat` | `46 / 25` | `46, 26, 42, 22, 44, 54, 70` |
| `ride` | `51 / 30` | `51, 59` |
| `ride_bell` | `53 / 32` | `53, 56` |
| `crash` | `49 / 28` | `49, 55, 57, 52` |
| `sticks` | `75 / 54` | `75` |

Every emitted bin is persisted, including a non-base or unmapped MIDI bin. Each event
carries the matching upstream group as diagnostic metadata, or null when the native
MIDI note is outside every group; the host does not collapse or relabel the event.

These native MIDI bins and upstream groups are not Crux benchmark canonical classes.
In particular, the current Crux empirical TensorFlow 2 bin labels and
`DEFAULT_MIDI_NOTE_MAP` are not authoritative for this backend. HPA-324 will define a
separate immutable prediction-map version from the preserved OaF native output space
to the shared taxonomy.

## Backend Lock

The checked-in lock uses schema:

```text
crux.transcription-backend-lock/v1
```

Its canonical JSON contains:

- backend ID, model ID, architecture ID, and descriptor schema;
- upstream repository, full source commit, and upstream source-manifest hash;
- host-adapter source-manifest hash;
- checkpoint acquisition URL as informational metadata;
- archive SHA-256 and exact byte length, plus the exact filename, size, and SHA-256
  of each component;
- model variable inventory, required inference-variable inventory, and allowed
  non-inference inventory;
- complete resolved hparams and their source;
- native output-space ID, complete 88-bin definition, training data-map ID, and every
  upstream 8-hit group;
- native-metadata schema ID and the exact allowed metadata keys, types, and
  nullability;
- the explicitly disabled inference prediction-map setting;
- a required positive `max_input_audio_frames` bound with no implicit default;
- protocol, prediction-artifact, verification-report, execution-report, and
  legacy-score-report schema versions;
- runtime-lock hash and final image manifest digest;
- smoke-audio SHA-256 and smoke-oracle SHA-256;
- native-`amd64` seal-evidence SHA-256;
- legacy conversion coverage-report SHA-256; and
- deterministic serialization rules.

The model-artifact-set identity is the SHA-256 of canonical JSON containing the three
checkpoint component names, sizes, and SHA-256 values in lexical filename order. It
does not depend on a local directory or modification time.

Lock files are parsed with a strict schema. Unknown fields, duplicate keys, missing
fields, unsupported versions, invalid output-space or data-map rows, or invalid hashes
are fatal. Changing any inference-relevant field requires a new lock digest and
a new backend ID.

### Tensor inventory and restore coverage

Coverage is derived from the actual frozen inference construction, not from name
prefixes or a Keras approximation:

1. Vendor the audited upstream source and resolve the locked `drums` hparams.
2. Construct that source's exact `PREDICT` graph.
3. Enumerate checkpoint names and shapes with `tf.train.list_variables`, and obtain
   checkpoint dtypes from TensorFlow's checkpoint reader.
4. Enumerate the graph variables needed for prediction. This includes every trainable
   model variable and batch-normalization moving mean and variance. Any graph global
   excluded as control or training-only state, such as a created `global_step`, must
   be identified by exact name with an individual reason.
5. Require exact name, shape, and dtype equality for every required graph variable,
   restore all of them, and prove that none remains uninitialized.
6. Enumerate every checkpoint entry not in the required set and classify it by exact
   name and reason in the locked non-inference inventory. Broad prefix, substring,
   partial-restore, or "ignore extras" rules are forbidden.

The reviewed inventory is expected to contain 130 checkpoint entries, 78 required
inference tensors, and 52 individually classified non-inference entries. A count can
summarize the evidence but cannot replace the exact inventories.

## Input Contract

The v1 runner accepts canonical audio only:

- RIFF/WAVE PCM;
- mono;
- signed 16-bit little-endian samples;
- exactly `44100` Hz;
- a positive PCM sample-frame count and nonempty sample bytes;
- an audio sample-frame count no greater than the backend lock's
  `max_input_audio_frames`; and
- no trailing non-audio chunks that the strict WAV reader does not recognize.

`max_input_audio_frames` is measured only in mono PCM sample frames, equivalent to
the strict reader's `wave.getnframes()` result. It is not a spectrogram-frame count.
The number of spectrogram frames is separately derived by the frozen preprocessing
pipeline and is never used as this input-bound unit.

The host validates the audio sample-frame bound and hashes the exact WAV bytes before
submitting the item to a runner. It sends both the relative staged path and expected
SHA-256. The runner resolves paths beneath its read-only input root, re-hashes the
bytes, validates the complete WAV contract again, and rejects a path escape or hash
mismatch.

Validation does not replace upstream preprocessing. After validation, the unchanged
WAV bytes enter the frozen upstream feature pipeline.

### Frozen upstream invocation

The frozen entrypoint is
`magenta/models/onsets_frames_transcription/onsets_frames_transcription_transcribe.py`
at the locked source commit, specifically `create_example`, `run`, and its
`main(..., data_fn=data.provide_batch)` binding. The backend lock records that
vendored file's SHA-256 and the inclusive source-line ranges containing each frozen
call. The complete resolved invocation is:

- `main(..., config_map=configs.CONFIG_MAP, data_fn=data.provide_batch)`;
- configuration `drums`, `FLAGS.hparams=""`, and the complete locked resolved
  hparams;
- `hparams.batch_size = 1` and `hparams.truncated_length_secs = 0`;
- `train_util.create_estimator(config.model_fn, <locked runner scratch model_dir>,
  hparams)`, where the exact scratch path and bounded tmpfs are runtime-lock fields;
- `create_example(filename, sample_rate=44100,
  load_audio_with_librosa=False)`, where `filename` is the normalized UTF-8
  runner-relative canonical WAV path;
- `audio_label_data_utils.process_record` with the exact WAV bytes,
  `sample_rate=44100`, an empty `music_pb2.NoteSequence()`,
  `example_id=six.ensure_text(filename, "utf-8")`, `min_length=0`,
  `max_length=-1`, `allow_empty_notesequence=True`, and
  `load_audio_with_librosa=False`;
- exactly one serialized example, rejecting zero or multiple examples;
- `data.provide_batch` with `preprocess_examples=True`, the resolved hparams,
  `is_training=False`, `shuffle_examples=False`, and
  `skip_n_initial_records=0`; and
- `infer_util.labels_to_features_wrapper` around the one-record dataset, followed by
  `estimator.predict(input_fn, checkpoint_path=<exact locked checkpoint prefix>,
  yield_single_examples=False)` and exactly one prediction batch.

These arguments are lock fields, not inherited defaults. In particular,
`min_length=0`, `max_length=-1`, and `truncated_length_secs=0` establish that this
entrypoint does not split or truncate the canonical record. A changed argument, call
target, or vendored line range requires a new source-manifest hash, backend ID, and
seal. The adapter parses `sequence_predictions` and the paired instrumentation
metadata instead of executing the upstream CLI's final MIDI-file write; that
identity-bound output adaptation is also covered by the source manifest and seal.

`CanonicalAudio` also carries host-supplied provenance:

- stable `source_audio_id`;
- SHA-256 of the source audio before input-view derivation;
- immutable `input_view_id`; and
- SHA-256 of the canonical WAV actually presented to the backend.

The backend does not invent those identities. A direct one-file invocation must
provide stable source and input-view IDs, but it does not accept a caller-supplied
source hash: the host treats `--audio` as both source and canonical input and computes
both hashes from the same bytes.

A derived input requires a strict `crux.input-view-manifest/v1` passed through
`--input-view-manifest`. The manifest names readable source and canonical artifacts
beneath their allowed roots and contains the source audio ID and SHA-256, input-view
ID, and canonical audio SHA-256. The host rejects unknown or duplicate fields and
independently re-hashes both artifacts. The source hash may legitimately differ from
the canonical WAV hash after derivation, but either manifest/hash/path mismatch is a
per-item failure before any runner request or output publication. There is no loose
`--source-audio-sha256` provenance override. The `--audio` path must resolve to the
manifest's canonical artifact. The smoke fixture has fixed source and input-view
identities in its locked smoke oracle.

Codec decoding, stem selection, channel mixing, and resampling belong to upstream
audio preparation. Restricting the frozen backend to canonical WAV removes FFmpeg,
platform codec, and resampler drift from backend identity. The procedural smoke
fixture uses the same contract.

Chunking is disabled in v1. Whole-file OaF behavior is part of the backend semantics;
chunk overlap, hidden-state reset, and event merging would require a separately
identified and locked backend policy. An input over `max_input_audio_frames` fails
before any runner request with per-item exit `1`. If HPA-326 needs longer inputs or
chunking, it must introduce a new locked policy/backend identity rather than silently
relaxing this v1 contract.

## Runner Protocol

The host and isolated process communicate with newline-delimited UTF-8 JSON:

- host requests use standard input;
- runner responses use standard output;
- one physical line is one JSON object;
- standard output contains protocol objects only; and
- diagnostic logs use sanitized standard error.

Before sending the first request, the host starts a dedicated concurrent
standard-error drain and keeps it running through process exit. It consumes bounded
byte chunks independently of standard-input writes and standard-output reads, so a
full standard-error pipe can never block TensorFlow or be misreported as a request
timeout. The runtime lock supplies positive exact values for the read-chunk size,
maximum logical-line bytes, and retained ring-buffer bytes. The drain continues after
the ring buffer fills by discarding the oldest retained diagnostic bytes, records
total bytes and a truncation flag, and rate-limits sanitized forwarding without
slowing pipe consumption. Oversized or unterminated logical lines are chunked and
sanitized under the same bounds.

The protocol schema is:

```text
crux.transcription-runner/v1
```

### Startup handshake

After model construction, checkpoint verification, full restore, and smoke
verification, the runner emits one `ready` object containing:

- protocol version;
- complete backend descriptor;
- TensorFlow and Python runtime facts;
- required/restored/non-inference tensor counts and inventory hashes;
- smoke input and oracle hashes; and
- smoke verification status.

The host compares every identity field, not just `backend_id`. Any mismatch terminates
the attempt before an input is accepted.

### Transcription request

A request contains:

- `type: "transcribe"`;
- an opaque request ID used only for correlation;
- a relative canonical WAV path;
- expected audio SHA-256; and
- the expected backend descriptor SHA-256.

The request ID and local path are operational data and are not persisted in the
prediction artifact.

### Response

A successful response contains the request ID, verified audio SHA-256, backend
descriptor SHA-256, and native events. A domain failure returns a typed error with a
stable machine code and sanitized message. Tracebacks, local absolute paths,
environment values, and checkpoint URLs never appear on standard output.

Standard error may contain stable error codes, tensor names and counts, bounded timing
diagnostics, and source-relative sanitized tracebacks. It must redact credentials and
secrets, environment values, absolute host or container paths, credential-bearing URL
userinfo or query strings, and audio bytes or decoded sample content. Standard error
is diagnostic only; automation consumes the protocol objects and machine-readable
reports.

The host enforces explicit startup and per-request deadlines. Deadline values are
recorded in the execution attestation because they affect operational completion, but
not prediction identity. A timeout kills the runner and is a backend-fatal failure
for that attempt.

## Common Prediction Artifact

Structured native events are the authoritative output. MIDI is an optional
deterministic derivative and must never be parsed to recover benchmark events.

The JSONL schema is:

```text
crux.drum-prediction-events/v1
```

Each artifact contains exactly:

1. one header record;
2. zero or more event records; and
3. one terminal record.

### Header

The header contains:

- schema version and `record_type: "header"`;
- `artifact_role: "native"`;
- source audio ID and SHA-256;
- input-view ID;
- exact canonical input-view audio SHA-256, byte length, sample rate, channel count,
  sample width, and audio sample-frame count;
- complete backend descriptor and its SHA-256;
- explicit model ID and architecture ID;
- nullable backend-lock, runtime-lock, and parameter-lock SHA-256 values;
- nullable model-artifact-set SHA-256;
- nullable upstream source commit; and
- native output-space ID, nullable training data-map ID, and native-metadata schema
  ID.

For OaF, every model/upstream/backend/runtime/training-map field is non-null and
`parameter_lock_sha256` is null. For the heuristic, `parameter_lock_sha256` is
non-null; the model-artifact-set, upstream source, backend-lock, runtime-lock, and
training-data-map fields are null. Other nullability combinations are invalid.

It contains no timestamp, hostname, process ID, absolute path, request ID, or Git dirty
state.

### Event

Each event contains:

- `record_type: "event"`;
- zero-based `event_index`, assigned only after validation, quantization, and sorting;
- `time_sec`, rounded to six decimal places from the emitted upstream frame time;
- required opaque `native_class_id`;
- nullable `model_output_bin`;
- nullable `native_midi_note`;
- `native_metadata`, an object governed by the descriptor's locked
  native-metadata schema;
- `confidence`, the onset activation at the emitted frame rounded to six decimal
  places, or null when a backend does not expose confidence;
- `velocity_midi`, the upstream velocity prediction represented as an integer in
  `0..127`, or null when a backend does not expose velocity;
- `mapping_status: "not_applied"`;
- `prediction_map_version: null`; and
- `canonical_class: null`.

For OaF, `native_class_id`, `model_output_bin`, `native_midi_note`, `confidence`, and
`velocity_midi` are all required and non-null. Schema
`magenta-oaf-native-metadata-v1` allows exactly one key,
`upstream_8hit_group_id`, whose value is a known upstream group ID or null. An unknown
metadata key, missing required key, wrong type, or unknown non-null group ID
invalidates the item. Later model adapters use the same envelope with their own
lock-enumerated metadata schemas, without fabricating a MIDI note, output bin,
confidence, or velocity they do not expose.

All persisted numeric fields are validated and quantized before sorting. Events are
then sorted on the exact values that will be serialized, using:

```text
(time_sec, native_class_id, model_output_bin_or_-1, native_midi_note_or_-1,
 velocity_midi_or_-1, confidence_or_-1)
```

before indexes are assigned. Two events whose complete six-field sort keys are equal
are an item validation failure with stable code `duplicate_native_event`; neither
stable emission order nor hidden source order may break the tie. Simultaneous events
with distinct keys remain distinct. The host does not deduplicate, merge, remap, or
threshold events after the frozen upstream postprocessor. Every OaF sort field is
non-null, so OaF ordering never evaluates the `-1` null sentinels; those sentinels
exist only for other schema-conforming adapters that legitimately omit a native
field.

An event with a nonfinite value, negative time, output bin outside `0..87`,
pitch/class/bin disagreement, confidence outside `0..1`, or velocity outside
`0..127` invalidates the item. A successful inference may legitimately contain zero
events; it still produces header and terminal records. The dedicated smoke fixture is
required to be nonempty so that a disconnected output path cannot pass verification.

HPA-320 defines the common native-and-canonical envelope, but deliberately emits
`mapping_status: "not_applied"` and null canonical fields. HPA-324 creates a separate
derived artifact with `artifact_role: "canonical_mapping"`, the native artifact
SHA-256, a non-null immutable model-specific `prediction_map_version`, and explicit
mapped or unmapped status for every native event. It does not mutate the native
artifact. This separation satisfies downstream generic parsing while preventing a
future taxonomy change from rewriting historical model output.

Before HPA-324 freezes and applies that mapping, an HPA-320 artifact is valid backend
preflight evidence but is not scorer-eligible. A generic consumer must report
`canonical_mapping_required`; it must not guess a class from MIDI or import an OaF
mapping inside the scoring layer.

### Terminal record and canonical bytes

The terminal record contains:

- `record_type: "terminal"`;
- event count; and
- SHA-256 of the exact header and event JSONL bytes, including their newline
  terminators and excluding the terminal record.

Every record is serialized as UTF-8 canonical JSON with:

- keys sorted lexicographically;
- no insignificant whitespace;
- no byte-order mark;
- Unix newline terminators;
- duplicate keys forbidden;
- nonfinite JSON numbers forbidden; and
- each accepted binary float converted with `Decimal.from_float(value)` and then
  quantized with
  `quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)` before serialization.

The conversion must not pass through `str(value)` or `repr(value)`. Quantized values
are emitted as fixed-point JSON numbers with at most six fractional digits, no
trailing fractional zeros, zero represented as `0`, and negative zero normalized to
`0`.

The prediction artifact SHA-256 covers the complete file, including the terminal
record. Writers publish through a temporary file in the destination directory,
flush and fsync it, and atomically rename it only after a strict reader validates the
complete artifact.

### Optional MIDI

MIDI export is derived from the validated event records in sorted order. It uses
`native_midi_note`, `velocity_midi`, and the frozen tick/tempo conversion settings.
The v1 derivative freezes 220 pulses per quarter note, a single tempo of 120.0
quarters per minute at tick zero, MIDI drum channel 10 (zero-based channel 9),
program 0, and the upstream fixed note duration of 0.05 seconds. The writer and its
exact package distribution are runtime-locked, and golden tests cover simultaneous
event ordering and time-to-tick conversion.
The MIDI artifact records the source prediction SHA-256 in its sidecar metadata. MIDI
failure does not corrupt or replace a valid structured prediction, but it makes a
request that explicitly asked for MIDI incomplete. The validated JSONL remains
published and that item returns exit `1`.

## Verification and Execution Flow

`verify-backend` performs the following steps in order:

1. Strictly parse and hash the backend and runtime locks.
2. Locate the three checkpoint components without network access.
3. Verify exact filenames, sizes, and component hashes.
4. Verify the runner image platform and OCI manifest digest.
5. Start the isolated runner with read-only model, input, backend-lock, and
   runtime-lock mounts.
6. Have the runner independently strict-parse and hash both locks and verify their
   runner-visible runtime, vendored-source, model, and checkpoint claims.
7. Build the exact frozen graph and compare graph variables with the locked inventory.
8. Restore all 78 required inference tensors.
9. Classify all 52 non-inference checkpoint entries against the explicit allowlist.
10. Prove that no required graph variable is missing or left uninitialized.
11. Run and validate the deterministic smoke fixture.
12. Compare the handshake and smoke output with the host's locks.
13. Return a machine-readable verification report.

Frozen benchmark transcription automatically runs the same preflight before
processing an item. A successful verification may be reused only by the same live
runner process whose descriptor was checked.

The operator interfaces are:

```bash
uv run crux benchmark verify-backend \
  --backend magenta-egmd-tf1-94529798-8hit-v1

uv run crux benchmark transcribe-one \
  --backend magenta-egmd-tf1-94529798-8hit-v1 \
  --audio canonical-input.wav \
  --source-audio-id corpus-song-42-source-v1 \
  --input-view-id full-mix-canonical-wav-v1 \
  --output prediction.jsonl

uv run crux benchmark transcribe-one \
  --backend magenta-egmd-tf1-94529798-8hit-v1 \
  --audio derived-canonical-input.wav \
  --input-view-manifest derived-canonical-input.view.json \
  --output prediction.jsonl
```

Direct mode uses `--source-audio-id` and `--input-view-id`; derived mode uses
`--input-view-manifest`. The modes are mutually exclusive.

Backend support is explicit:

| Backend | `verify-backend` | `transcribe-one` | `transcribe-and-score` | Lock |
| --- | --- | --- | --- | --- |
| `magenta-egmd-tf1-94529798-8hit-v1` | yes; default | yes; default | refuses with `canonical_mapping_required` | complete backend/runtime locks |
| `heuristic-onset-v1` | yes; explicit | yes; explicit | yes; explicit legacy compatibility | isolated parameter-locked environment |
| `legacy-tf2-h5-v0` | no | no | yes; explicit legacy compatibility | none; compatibility label only |

The heuristic parameter lock freezes its canonical input contract; package versions;
44.1 kHz sample rate; 512-sample hop; `librosa.onset.onset_strength` and
`onset_detect` options including `backtrack=true`; frame-to-time conversion; two-hop
classification window; spectral-centroid thresholds `150`, `350`, `500`, `750`,
`1000`, `2000`, and `3000`; zero-crossing-rate thresholds `0.1` and `0.2`; output
pitches `36`, `38`, `42`, `45`, `47`, `49`, and `50`; branch ordering; and fixed
velocities `80`, `70`, `60`, and `65` as applicable. Defaults consumed from Librosa
must be resolved into the lock rather than inherited implicitly. The HPA-320 adapter
calls this locked heuristic directly; it never reaches it as a failed-model fallback.

The fixed pitch/velocity pairs are exactly `36:80`, `38:70`, `42:60`, `45:65`,
`47:65`, `49:65`, and `50:65`. Heuristic events use unpadded `midi_<pitch>` native
class IDs, non-null native MIDI pitch and velocity, null model-output bin and
confidence, and the exact empty native-metadata object.

The official backend is the default only for frozen inference commands:
`verify-backend`, `transcribe-one`, and the HPA-326 inference orchestration.
`transcribe-and-score` has no backend default. Its executable selections are the two
explicit compatibility choices in the table. If given the frozen OaF ID, it returns
`canonical_mapping_required` with exit `1`, invokes neither legacy mapping/scoring nor
OaF inference, and publishes no score.

### Default paths and operational reports

Following the HPA-321 benchmark layout, backend operational state defaults under:

```text
artifacts/benchmark/backends/<backend-id>/
```

Verification, execution, and legacy-score wrapper reports are immutable files at
`reports/<UTC-basic-timestamp>-<run-id>.json`, where the timestamp is
`YYYYMMDDTHHMMSSffffffZ` and the run ID is a lowercase UUIDv4. Report timestamps use
`YYYY-MM-DDTHH:MM:SS.ffffffZ`. After successfully publishing a report, the host
atomically updates `latest-verification.json`, `latest-execution.json`, or
`latest-legacy-score.json` as a mutable convenience copy; no latest file is an
identity input.
`transcribe-one --output` remains required so a caller chooses the immutable
prediction destination. All persisted paths inside reports are repository-relative
POSIX paths.

`verify-backend` and direct `transcribe-one` use the exact reference execution profile
from the accepted seal evidence. HPA-326 may supply a different exact profile only
through its immutable run configuration and execution attestation.

`crux.backend-verification-report/v1` has exactly these top-level keys:

```text
schema, report_type, run_id, started_at, finished_at, status, exit_code,
descriptor, descriptor_sha256, backend_lock_sha256, runtime_lock_sha256,
parameter_lock_sha256, seal_evidence_sha256, execution_attestation,
tensor_coverage, smoke, artifacts, errors
```

`report_type` is `verification`; `status` is `verified`, `failed`, or
`environment_unsupported`. `execution_attestation` is null before one can be
established or is an object with exactly `path` and `sha256`.
`tensor_coverage` has exactly `status`, `required_count`, `restored_count`,
`non_inference_count`, `required_inventory_sha256`,
`non_inference_inventory_sha256`, `report_path`, and `report_sha256`. `smoke` has
exactly `status`, `audio_sha256`, `oracle_sha256`, `prediction_path`, and
`prediction_sha256`. The two nested status fields are `passed`, `failed`, `not_run`,
or `not_applicable`. Artifact path/hash fields are null when the corresponding
artifact was not produced. For the heuristic, both nested statuses are
`not_applicable`, counts are zero, and other OaF-only nested fields are null.
`artifacts` is an empty array when nothing was published; otherwise each entry has
exactly `role`, `path`, and `sha256`.

`crux.backend-execution-report/v1` has exactly these top-level keys:

```text
schema, report_type, run_id, started_at, finished_at, status, exit_code,
descriptor, descriptor_sha256, backend_lock_sha256, runtime_lock_sha256,
parameter_lock_sha256, seal_evidence_sha256, execution_attestation,
verification_report, items, errors
```

`report_type` is `execution`; `status` is `complete`, `partial`, `failed`, or
`environment_unsupported`. `execution_attestation` and `verification_report` are each
null before establishment or an object with exactly `path` and `sha256`. Items remain
in request order. Each item has exactly `item_id`, `source_audio_id`,
`source_audio_sha256`, `input_view_id`, `input_audio_sha256`, `status`, `prediction`,
`midi`, and `errors`; item status is `complete`, `incomplete`, or `failed`.
`prediction` and `midi` are either null or an object with exactly `path` and
`sha256`.

`item_id` is content-addressed rather than caller-selected. It is the lowercase
string `sha256:<hex>` for the SHA-256 of canonical JSON containing exactly:

```text
schema, source_audio_id, source_audio_sha256, input_view_id, input_audio_sha256
```

with `schema: "crux.backend-execution-item-id/v1"`. The payload uses
lexicographically sorted keys, UTF-8, no insignificant whitespace, and no trailing
newline. Repeated execution of the same source/input-view bytes therefore produces
the same item identity regardless of run ID, output path, or backend.

Every error in the verification or execution schema has exactly `code` and sanitized
`message`; errors are sorted first by the UTF-8 bytes of `code` and then of
`message`. Artifact entries are sorted first by `role` and then by `path`.

Identity, attestation, and artifact fields may be null only when a typed failure
occurred before that identity or artifact could be established. A `verified` report
requires every OaF identity, coverage, smoke, and attestation field to be non-null. A
successful heuristic report requires its descriptor and parameter-lock identity but
keeps the non-applicable OaF runtime, seal, and backend-lock fields null.

The legacy combined scorer writes a separate wrapper under schema
`crux.legacy-score-report/v1` with exactly:

```text
schema, report_type, run_id, started_at, finished_at, workflow_mode, backend_id,
backend_validation_status, status, exit_code, score_report, errors
```

`report_type` is `legacy_score` and `workflow_mode` is `legacy_compatibility`.
`backend_validation_status` is `unvalidated` for `legacy-tf2-h5-v0`,
`parameter_locked` for `heuristic-onset-v1`, and `not_checked` for the recognized OaF
ID because this compatibility command does not preflight or run it.
`status` is `complete`, `failed`, or `canonical_mapping_required`. `score_report` is
null or an object with exactly the repository-relative `path` and `sha256` of the
existing combined-score report. A `canonical_mapping_required` result has a null
score report, runs neither transcription nor scoring, and exits `1`.
`legacy-tf2-h5-v0` may remain executable for compatibility, but every wrapper marks
it `unvalidated`; its output is prohibited from official benchmark comparison.
`errors` uses the same exact `code`/sanitized-`message` objects and canonical ordering
as the verification and execution reports.
Removing that path requires a separately approved migration milestone rather than an
implicit HPA-320 deletion.

All three schemas reject unknown or duplicate fields, use atomic publication, and are
canonical UTF-8 JSON with one final Unix newline. They are operational evidence:
timestamps and run IDs intentionally vary and never enter descriptor or prediction
identity. After Click parsing succeeds, `verify-backend` and `transcribe-one` each
write exactly one canonical one-line JSON summary to host standard output containing
only `status`, `exit_code`, `report_path`, and `report_sha256`; human progress goes to
sanitized standard error. `transcribe-and-score` uses the same one-line summary keys
for its legacy-score wrapper. Scripts distinguish
backend-fatal exit `2` from a Click usage error by parsing this summary and report,
not by inventing another exit code.

## Failure Semantics

Frozen backend mode never:

- downloads a checkpoint;
- tries an alternate model URL;
- accepts a mutable "latest" model;
- initializes missing variables randomly;
- restores a subset of required inference variables;
- substitutes the local HDF5 file;
- retries with different feature parameters; or
- falls back to the heuristic backend.

### Fatal backend failures

The following stop the attempt immediately, publish no new prediction for the active
item, and return exit `2`:

- lock parse or identity failure;
- missing, extra, renamed, or hash-mismatched checkpoint components;
- runtime platform or image-digest mismatch;
- missing or extra required graph variables;
- any of the 78 required tensors not restored;
- unclassified checkpoint state;
- failed or mismatched smoke output;
- startup timeout;
- handshake mismatch;
- runner process death;
- protocol corruption; or
- per-request timeout that leaves runner state unknown.

An out-of-memory kill, process death, or allocator failure while processing an input
within the locked `max_input_audio_frames` bound is also backend-fatal. It does not
authorize a smaller hidden chunk size or fallback.

Previously published immutable predictions are not deleted.

### Per-item failures

An input-contract, decoding, provenance, or validly reported inference failure
produces no prediction artifact for that item. This includes a source or canonical
hash mismatch and a canonical input over the locked `max_input_audio_frames` bound. A
multi-item caller may continue collecting diagnostics with the same verified runner,
but the overall result is partial and exit `1`.

Nonfinite values, native bins outside the locked 88-bin output space, malformed
events, or an artifact round-trip failure are item failures if the runner remains
healthy and its protocol identity is intact. Repeated item failures do not authorize
fallback.

The complete command/status mapping is:

| Command condition | Exit | Machine-readable discriminator |
| --- | --- | --- |
| Verification succeeds or every requested item completes | `0` | verification `verified` or execution `complete` |
| One or more items fail while the backend remains valid | `1` | execution `partial` plus item `failed` |
| Requested MIDI derivation fails after valid JSONL publication | `1` | execution `partial`, item `incomplete`, error `midi_derivation_failed` |
| Official execution is attempted on an unsupported host | `1` | verification/execution `environment_unsupported` |
| Legacy scoring cannot run or complete | `1` | legacy-score `failed` |
| Frozen OaF is requested through the legacy scorer before mapping | `1` | legacy-score `canonical_mapping_required` |
| `prepare-backend` cannot acquire or stage bytes | `1` | setup summary `acquisition_failed` |
| Backend integrity, startup, smoke, process, timeout, or protocol fails | `2` | verification/execution `failed` with stable error code |
| `prepare-backend` detects lock, byte, cache, or publication integrity failure | `2` | setup summary `integrity_failed` |
| Click rejects command syntax or options | `2` | no report or command summary exists |

After Click has successfully parsed a backend command, a backend-fatal exit `2` must
publish a typed machine-readable `crux.backend-verification-report/v1` or
`crux.backend-execution-report/v1` with its failure status and stable code. Click
argument and option usage errors may retain Click's standard exit `2`, but occur
before either report exists. Automation distinguishes these cases by report
presence and typed status rather than by exit code alone. If report publication itself
fails, sanitized standard error must state that no machine-readable report was
published.

## Deterministic Smoke Fixture

HPA-320 adds a short, copyright-free, procedurally generated fixture:

- 44.1 kHz;
- mono;
- signed 16-bit PCM WAV;
- deterministic synthesized kick, snare, and cymbal-like transients; and
- checked-in generator parameters and audio SHA-256.

The sole checked-in expected-output authority is `smoke-oracle.json` under schema
`crux.oaf-smoke-oracle/v1`. It contains the source and input-view fixture identities,
canonical audio hash and input facts, and ordered expected native event payload.

The backend lock references the SHA-256 of `smoke-oracle.json`. The oracle deliberately
excludes backend-lock and runtime-lock hashes, so its identity does not form a cycle
with the backend lock.

No second complete expected JSONL is checked in. Verification and integration tests
generate the full expected `crux.drum-prediction-events/v1` bytes deterministically
from the oracle, the exact mounted locks, and the verified descriptor, then compare
the actual artifact byte-for-byte. A generated expected JSONL may be uploaded as CI
evidence, but it is not a separate source of truth that can drift from the locked
oracle.

Verification runs the fixture:

1. twice in one persistent runner; and
2. once in a fresh runner.

All three prediction artifacts must be byte-identical; request-local fields are already
excluded by the artifact schema. The artifact must be finite and nonempty. Backend
identity fields must match the lock. Native classes, event count, rounded times, MIDI
notes, confidence values, velocity integers, and the complete serialized event
payload must match the smoke oracle exactly after the specified six-decimal
quantization. There is no numeric acceptance tolerance. A verifier may calculate and
report raw pre-quantization confidence deltas only after an exact-match failure for
diagnosis; those diagnostics can never turn a mismatched artifact into a pass.

The smoke oracle and deterministically generated artifact are regression evidence for
execution parity, not accuracy claims. Adjusting the model, hparams, thresholds, or
postprocessor to make the fixture look musically better is forbidden.

## Test Strategy

### Unit tests

- Strict backend-lock and runtime-lock schema validation.
- Strict legacy-conversion coverage schema validation and cross-checking of its HDF5,
  source-manifest, required-inventory, and seal-evidence hashes.
- Canonical model-artifact-set and descriptor identity derivation.
- Complete MIDI-21-to-108 native output-space coverage and class/bin consistency.
- Exact upstream 8-hit training-group membership without inference-time collapsing.
- Canonical JSONL serialization and strict reading.
- Quantize-before-sort ordering, rejection of a complete six-field tie as
  `duplicate_native_event`, and
  `Decimal.from_float(...).quantize(...)` half-even boundary cases, including
  trailing-zero and negative-zero normalization.
- Exact binary64 frame-time evaluation order and upstream velocity clamp/scale/truncate
  behavior, including the explicitly divergent rejection of nonfinite raw velocity.
- Instrumented onset-frame, pitch, velocity, and confidence pairing without
  reconstructing a frame from `NoteSequence.start_time`.
- Exact `crux.backend-execution-item-id/v1` canonical payload and hash derivation.
- Exact descriptor canonical bytes and proof that operational fields and a trailing
  newline change neither its schema nor its accepted hash.
- Header, empty-result, event, and terminal validation.
- Required source-audio, input-view, model, native-metadata schema, and mapping-state
  identity.
- Rejection of unknown native-metadata keys, wrong types, and unknown OaF group IDs.
- Direct-input hash derivation and independent derived-manifest source/canonical hash
  verification.
- `max_input_audio_frames - 1`, exact-boundary, and over-bound input behavior.
- Manifest-defined dirty-checkout scope and canonical changed-file hashing.
- Verification/execution report strict schemas, canonical bytes, stable default paths,
  final CLI summaries, and atomic latest-report updates.
- Execution-attestation CPU, memory, PID, and deadline fields.
- A backend-agnostic consumer can read OaF events without importing an OaF module and
  rejects `mapping_status: "not_applied"` through a generic error.
- Atomic publication and preservation of an existing valid artifact on failure.
- Explicit backend selection and proof that missing model state does not select the
  heuristic.
- Heuristic descriptor and parameter-lock coverage, including all resolved Librosa
  defaults, classifier branch ordering, fixed pitch/velocity outputs, and isolation
  from Crux's main dependency environment.
- Sanitized errors and logs.

### Protocol and failure tests

A fake runner covers:

- correct handshake and successful output;
- wrong backend, mounted-lock hash, runtime, metadata-schema, and mapping identities;
- missing, duplicate, and extra handshake fields;
- corrupt JSON and unexpected standard-output text;
- runner crash before and after readiness;
- in-bound out-of-memory/process failure as backend-fatal;
- startup and request timeouts;
- nonfinite or out-of-range events;
- input hash mismatch;
- path escape;
- explicit item error; and
- proof that none of these conditions invokes another backend.

Container-launch tests assert network disablement, read-only mounts/root filesystem,
environment allowlisting, dropped capabilities, `no-new-privileges`, numeric
non-root UID/GID, tmpfs bounds, and CPU/memory/PID limits. An emulated host can run
diagnostics only and must produce `environment_unsupported` without publishing an
official prediction. Launcher tests prove that `PYTHONHASHSEED` and the full
allowlist are present before Python starts, that startup rejects any mismatch before
TensorFlow import, and that explicit `ConfigProto` thread counts remain `1`. Host
evidence tests accept only the three enumerated evidence forms and keep an unattested
bare local host diagnostic-only.

Log tests prove that standard output remains protocol-only and that standard error
retains allowed diagnostics while redacting secrets, environment values, absolute
paths, credential-bearing URLs, and audio content. A stress runner writes more than a
platform pipe buffer and more than the locked ring-buffer capacity while normal
responses continue; the host must drain concurrently, retain bounded diagnostics,
set its truncation flag, and complete without a spurious timeout.

### Real-checkpoint integration tests

The pinned runtime must:

- enumerate the official checkpoint and reproduce the locked 130-entry inventory;
- classify exactly 78 required inference tensors and 52 allowed non-inference entries;
- restore every required tensor with exact name, shape, and dtype;
- report no missing or uninitialized required variables;
- exercise the exact locked `create_example`, `process_record`, `provide_batch`, and
  `estimator.predict` argument set;
- prove that the prediction graph contains no active stochastic dropout;
- require byte-identical serialized `NoteSequence` output from patched and
  unmodified postprocessing while proving the patch reports the selected onset frame;
- run the smoke fixture twice in-process and once in a fresh process;
- generate the expected native artifact from the locked oracle, mounted locks, and
  descriptor and reproduce it byte-for-byte;
- round-trip the JSONL through the host reader without information loss;
- expose backend-independent prediction records to a scorer-facing consumer without
  importing OaF runner or mapping code; and
- generate optional MIDI whose sidecar references the native artifact hash.

### Command tests

CLI tests prove:

- `prepare-backend` emits only the documented setup summary and never a backend
  report;
- `verify-backend` returns the documented reports and exit codes;
- frozen benchmark commands preflight before opening item outputs;
- `transcribe-one` writes atomically;
- direct and derived `transcribe-one` provenance modes are mutually exclusive;
- the official backend is the default for frozen inference commands only;
- `transcribe-and-score` requires an explicit legacy backend, marks its report as
  legacy compatibility, writes `crux.legacy-score-report/v1`, and returns
  `canonical_mapping_required` for OaF;
- the OaF legacy-score refusal reports
  `backend_validation_status: "not_checked"` without backend preflight;
- every `legacy-tf2-h5-v0` score wrapper reports
  `backend_validation_status: "unvalidated"`;
- invoking `transcribe-and-score` without `--backend` is a Click usage error with exit
  `2`, not an implicit legacy or OaF selection;
- heuristic use requires its complete explicit ID;
- Click usage errors and backend-fatal exit `2` are distinguishable by typed report
  presence;
- requested MIDI derivation failure retains JSONL and returns exit `1`; and
- backend-fatal and per-item failures remain distinguishable.

## Continuous Integration

The normal Python suite uses the fake runner and does not install TensorFlow 1 or
require Docker. It validates host logic on every pull request.

A dedicated native Linux `amd64` integration job runs when backend locks, runner code,
vendored upstream source, smoke fixtures, or prediction schemas change. It also runs
on a scheduled cadence and through manual dispatch. The job:

1. downloads the locked archive as an explicit CI preparation step;
2. verifies its exact locked byte length and SHA-256 before extraction;
3. verifies all three component hashes;
4. builds or pulls the pinned runner by immutable digest;
5. runs `verify-backend`;
6. runs the real-checkpoint integration tests; and
7. uploads the verification report, tensor coverage report, and smoke artifacts.

The benchmark command itself remains offline. A network or acquisition failure in CI
fails the dedicated job; it is not converted into a skipped parity result.
An emulated `amd64` job cannot satisfy this required check.

The dedicated job is required before merging an inference-relevant change. Unrelated
host-only changes may reuse a recent successful scheduled result only if repository
branch protection explicitly permits it.

## Repository Boundaries

Implementation planning should preserve these ownership boundaries:

- `src/benchmark/backends/` owns the host protocol, descriptors, backend selection,
  and the isolated OaF adapter.
- `src/benchmark/backend_lock.py` owns strict locks and identity derivation.
- `src/benchmark/prediction_artifact.py` owns the common JSONL schema, validation,
  backend-agnostic reading, and atomic publication.
- `runtime/oaf_tf1/` owns the pinned runner, container definition, package lock, source
  manifests, and protocol implementation.
- `runtime/heuristic_onset/` owns the content-addressed heuristic environment lock and
  reproducible environment builder.
- `config/benchmark/backends/` owns reviewed backend, runtime, and heuristic-parameter
  locks plus the native seal-evidence record.
- `docs/superpowers/evidence/hpa-320/` owns immutable design-audit evidence, including
  the legacy TF2 conversion coverage report.
- `tests/fixtures/oaf_tf1_smoke/` owns the procedural generator and parameters,
  canonical WAV, and sole checked-in smoke oracle.
- `src/cli/benchmark.py` owns command wiring and exit-code presentation only.

Existing FastAPI modules remain unchanged unless a later issue explicitly migrates
them to the backend interface.

The implementation plan's first task must capture current behavior before rewiring:
`src/cli/benchmark.py` currently exposes `transcribe-and-score` without a backend
option, `src/benchmark/runner.py` constructs `DrumTranscriber` directly, and
`src/app/transcriber.py` can enter its heuristic path after model unavailability.
Task 0 adds characterization tests for that reality and the intentional migration;
subsequent tasks must not treat today's implicit fallback as an already-frozen
backend contract.

## Migration and Compatibility

- `magenta-egmd-tf1-94529798-8hit-v1` becomes the default for frozen benchmark
  inference commands and HPA-326 orchestration, not for the legacy combined scorer.
- The existing TensorFlow 2 HDF5 path remains available to
  `transcribe-and-score` only through explicit compatibility ID
  `legacy-tf2-h5-v0`, and may remain available to non-benchmark service code. Every
  legacy-score wrapper marks it `backend_validation_status: "unvalidated"`; it is
  prohibited from official benchmark comparison, cannot pass frozen backend-lock
  validation, and cannot write an artifact claiming frozen OaF identity.
- Existing heuristic logic remains available only as `heuristic-onset-v1`.
- Existing combined-score reports are explicitly marked
  `workflow_mode: "legacy_compatibility"` and are not reclassified as frozen
  predictions.
- Existing scripts that invoke `transcribe-and-score` without `--backend` must migrate
  to an explicit compatibility ID; the old invocation becomes a Click usage error
  with exit `2`.
- Existing benchmark prediction formats are not silently reinterpreted. New native
  artifacts use the common `crux.drum-prediction-events/v1` envelope with explicit
  not-yet-mapped canonical fields.
- A future TensorFlow 2 port receives a new runtime/backend ID. It may become an alias
  for the frozen backend only after tensor coverage, feature parity, and event-level
  parity are demonstrated against a reviewed fixture suite.
- HPA-324 mapping artifacts, HPA-325 score artifacts, and HPA-326 run manifests must
  reference the immutable native prediction SHA-256 rather than mutating it.

## Acceptance Criteria

HPA-320 is complete when:

1. The official source, checkpoint, runtime, hparams, 88-bin native output space,
   native-metadata schema, upstream 8-hit training map, exact upstream invocation, and
   maximum input-audio-frame bound are represented by strict checked-in locks with the
   identities in this design, backed by an accepted native-`amd64` seal-evidence
   record containing every exact resource, deadline, package, and platform value
   required above.
2. The released checkpoint is verified offline and all 78 required inference tensors
   are restored; the 52 non-inference entries are explicitly classified, and the
   zero-of-78 legacy conversion conclusion is preserved in its hashed audit artifact.
3. The launcher supplies the exact environment before CPython starts, and the
   isolated TensorFlow 1.15.5 runner completes an identity-checked handshake with
   bounded concurrent standard-error draining.
4. The procedural fixture reproduces byte-identical, finite, nonempty native
   prediction artifacts across repeated and fresh processes with no numeric
   acceptance tolerance.
5. Structured native predictions preserve independently verified source-audio and
   input-view identity, time, output bin, MIDI pitch, native class, schema-governed
   upstream group, selected onset-frame confidence, velocity, and explicit canonical
   mapping state and pass strict round-trip validation; instrumentation preserves
   byte-identical upstream `NoteSequence` output.
6. The official backend is the frozen-inference default, the heuristic requires
   explicit selection, and `transcribe-and-score` remains an explicitly selected and
   typed legacy workflow that refuses OaF scoring before HPA-324/HPA-325 and marks the
   HDF5 path unvalidated.
7. Missing or corrupt model state, runtime drift, smoke mismatch, runner failure, and
   protocol errors fail without fallback and produce the documented exit code.
8. Host tests pass without TensorFlow 1, and the dedicated real-checkpoint CI job
   publishes auditable verification and coverage artifacts.
9. Every execution records the Crux Git commit, manifest-defined clean or dirty source
   state, resource limits, and deadlines without making the static backend or runtime
   lock self-referential.
10. The FastAPI transcription path, shared canonical taxonomy, scoring semantics, and
   full-corpus orchestration remain outside the HPA-320 change.
11. A lightweight `heuristic-onset-v1` adapter and content-addressed isolated
    environment preserve the current heuristic only through explicit selection; the
    legacy HDF5 path remains combined-score-only and cannot claim a frozen
    descriptor.
12. `docs/drumery-dtx-midi-benchmarking-reference.md`,
    `tests/test_cli_benchmark.py`, and `tests/benchmark/test_runner.py` document and
    test the backend matrix, new defaults, reports, exit semantics, and migration.

## Risks and Mitigations

### TensorFlow 1 is obsolete

The runtime is isolated, network-free during inference, CPU-only, and digest-pinned.
It is not added to Crux's Python 3.13 environment. CI continuously proves that the
frozen container still starts and reproduces the fixture. It accepts no
operator-supplied GraphDef or checkpoint; the only graph source and model components
are hash-locked, and the audio boundary uses the strict canonical WAV contract.

The Python 3.7/Bullseye snapshot may contain known vulnerabilities. HPA-320 accepts
that residual risk only inside the locked non-root, network-disabled, read-only,
capabilities-dropped boundary and records a security scan/advisory snapshot with the
seal evidence. It does not claim that EOL packages are patched. A vulnerability that
invalidates the isolation boundary retires official execution pending a reviewed new
runtime identity.

### Apple Silicon development hosts differ from CI

The only accepted official runner platform is native `linux/amd64`. Local Apple
Silicon execution may explicitly emulate that image for diagnostics rather than
building a separate ARM image, but it reports `environment_unsupported`, cannot seal
locks or publish official predictions, and cannot satisfy the dedicated native
parity check.

### Whole-file inference has bounded resource use

The backend lock has no implicit or unlimited input length: it freezes
`max_input_audio_frames`, and host and runner boundary tests enforce it. CPU, memory,
PID, tmpfs, and deadline limits are recorded in the execution attestation. Direct
verification/transcription uses the sealed reference profile; HPA-326 freezes its
profile in the run configuration. A supported input that exhausts those resources
fails loudly as a backend-fatal condition; a longer-input or chunked policy requires
a new locked identity.

### Heuristic dependency upgrades could erase historical comparability

The heuristic's interpreter and hashed distributions live in a
parameter-lock-addressed environment separate from Crux's main dependencies.
Routine application upgrades do not mutate that environment. Historical locked
environments remain reconstructable while retained; changing any algorithm-affecting
byte creates a new heuristic backend identity rather than silently changing
`heuristic-onset-v1`.

### A single smoke fixture may miss semantic drift

Tensor inventory, source hashes, resolved hparams, runtime identity, and protocol
checks protect the entire backend before the fixture runs. The smoke fixture then
proves the end-to-end path. A later TF2 parity project must use a broader fixture
suite and does not weaken this lock.

### Confidence extraction could diverge from upstream event selection

The runner takes confidence from the frozen onset activation at the exact frame
selected by upstream postprocessing. It must not rerun thresholding or select a nearby
maximum on the host. Event-selection and confidence-extraction tests use the same
captured tensors.

### Taxonomy work could rewrite model history

Native MIDI class, pitch, output bin, upstream training-group metadata, and native
output-space ID are always retained. Canonical mapping is a separate hash-addressed
derivation owned by HPA-324.

### Container or model acquisition may disappear

The checkpoint archive and image are identified by content digest. The seal process
also exports the complete built runner image as a content-addressed OCI layout,
records its manifest/config/layer digests and archive SHA-256 in seal evidence, and
copies it plus the checkpoint archive to approved durable storage. Authorized
registries, Debian snapshot mirrors, or artifact mirrors may supply the same bytes
without changing backend identity. A new byte sequence, even from the same URL or
tag, is rejected. Preservation of the exact OCI bytes avoids making a future registry
or `snapshot.debian.org` outage an unrecoverable rebuild dependency.

## Alternatives Rejected

### Native TensorFlow 2 parity port now

A native port would be easier to operate in Crux's main environment, but it would
require a correct graph conversion, complete tensor-name mapping, feature parity, and
event-level equivalence before it could represent the released model. The current
conversion restores zero required inference tensors, so using it as the headline
backend would freeze an unproven implementation.

### Importing only the released MetaGraph through TensorFlow 2 compatibility mode

This reduces source reconstruction but still changes the TensorFlow runtime and may
change kernels, preprocessing, or graph compatibility behavior. It remains a useful
future candidate with its own runtime ID and parity proof, not the initial authority.

### Keeping the current service transcriber and documenting caveats

Documentation cannot compensate for partial weight loading, mismatched features,
incorrect native labels, or implicit heuristic fallback. Benchmark artifacts must
identify one validated algorithm, not whichever path happened to return events.
