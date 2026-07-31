# HPA-320: Frozen OaF Drums Transcription Backend Design

**Status:** Review amendments applied; remaining implementation is specified by
`docs/superpowers/plans/2026-07-29-hpa-320-deterministic-oci-seal-closure.md`, while
final OaF sealing and integration remain blocked on accepted native-`amd64`
calibration evidence

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

## Sealed-Value Authority

The freeze is intentionally staged:

1. Before sealing, the checked-in acquisition, base-system,
   calibration-bootstrap, calibration-measurement, and seal-profile request records
   are the only authorities for intended inputs.
2. Producer commands emit immutable acquisition, base-system, bootstrap, and
   measurement evidence, but measurement output is diagnostic and cannot choose or
   publish sealed values.
3. Review makes one explicit seal-profile request authoritative by binding every
   accepted value to the exact request, measurement, and evidence hashes it reviewed.
4. Calibration generates and validates one complete candidate without publishing
   final locks or changing the registry.
5. Seal dependencies, runtime lock, and backend lock are published in that acyclic
   order and reference the accepted requests, evidence, profile, image, checkpoint,
   and smoke-oracle identities.

Code defaults, sentinel values, mutable reports, guessed resource numbers, and prose
such as `auto` or `unlimited` are never sealed-value authorities.

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
Vendoring does not treat `git status` as provenance: every selected no-follow
worktree file must match the exact `HEAD` tree blob and index entry, and selected
paths carrying `assume-unchanged`, `skip-worktree`, nonzero-stage, or other
unsupported index state are rejected.

### Released checkpoint

| Artifact | Role | Byte length | SHA-256 |
| --- | --- | ---: | --- |
| `e-gmd_checkpoint.zip` | Archive | 25,658,703 | `09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0` |
| `checkpoint` | TensorFlow pointer | 91 | `c4afc7e992f63a290fc9b061bc36582eb08db5f4c10f8b79971982217f039a2b` |
| `model.ckpt-569400.index` | Published component | 2,713 | `475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a` |
| `model.ckpt-569400.data-00000-of-00001` | Published component | 27,793,012 | `6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5` |
| `model.ckpt-569400.meta` | Published component | 3,640,417 | `e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422` |

The informational acquisition URL is:

```text
https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/e-gmd_checkpoint.zip
```

Content hashes, not the URL, are authoritative. Benchmark execution never downloads
weights. A separate explicit setup step may acquire the archive, verify the archive
hash, extract it to a temporary directory, verify every component, and only then
install the components in a content-addressed model cache.

The checked-in pre-seal authority is:

```text
config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json
```

under schema `crux.oaf-checkpoint-acquisition-request/v1`. It records the backend ID,
the exact URL, the archive identity, all four archive-member identities and roles,
and the exact three published component names. Archive and member byte lengths are
deterministic properties of the already frozen upstream bytes, not native resource
measurements. The request therefore records them before the final backend lock
exists. Its SHA-256 becomes a seal dependency.

The only network-capable model setup interface is explicit:

```bash
uv run crux benchmark prepare-backend \
  --backend magenta-egmd-tf1-94529798-8hit-v1 \
  --acquisition-request \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json \
  --evidence-output \
    artifacts/benchmark/backends/hpa320-bootstrap/checkpoint-acquisition-evidence.json \
  --download

uv run crux benchmark prepare-backend \
  --backend magenta-egmd-tf1-94529798-8hit-v1 \
  --acquisition-request \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json \
  --evidence-output \
    artifacts/benchmark/backends/hpa320-bootstrap/checkpoint-acquisition-evidence.json \
  --archive /operator/staging/e-gmd_checkpoint.zip
```

`--download` and `--archive` are mutually exclusive. With neither option, the command
only verifies an existing cache. Before the final backend lock exists, both
`--acquisition-request` and `--evidence-output` are required. After sealing, omitting
those options loads the fixed request beside the final lock and exact-compares the
request, final lock, and cache. The download form may use only the informational URL
in the checked-in request. With neither `--download` nor `--archive`, a missing cache
is `model_cache_missing`, exit `1`, and summary status `acquisition_failed`; this form
never opens a network connection.

All forms stage outside the final cache, verify the archive when present, reject
archive path traversal, require exactly the four requested members, and validate the
pointer payload as:

```text
model_checkpoint_path: "model.ckpt-569400"
all_model_checkpoint_paths: "model.ckpt-569400"
```

Only the three members marked `published_component` enter the cache. The pointer is
verified and recorded in acquisition evidence but is not an inference component.
Any extra, missing, duplicate, renamed, overlapping, unsafe, differently encoded, or
hash-mismatched member is an integrity failure. The command atomically publishes
only a fully verified component set at:

```text
artifacts/benchmark/model-cache/sha256/<model-artifact-set-sha256>/
```

Exit `0` means the exact component set is installed and reverified, exit `1` means
acquisition or local staging could not complete and the cache was left unchanged, and
exit `2` means supplied or cached bytes contradict the request or final lock, a
request or lock is invalid, or atomic cache publication failed. The command never
substitutes another URL or checkpoint. Inference commands are always offline and
only read this cache.

`prepare-backend` is a setup-only reporting exception: it does not launch a backend
and does not write a verification, execution, or legacy-score report. After Click
parsing succeeds, it writes exactly one canonical one-line JSON summary with
`status`, `exit_code`, `report_path`, and `report_sha256`; both report fields are
null unless `--evidence-output` was supplied. In that pre-seal mode they identify the
immutable `crux.oaf-checkpoint-acquisition-evidence/v1` record, which reproduces the
request hash, archive identity, all four member identities, the three-component
artifact-set hash, acquisition mode, and cache location. This setup evidence is not
a backend verification report and cannot make a prediction or seal eligible. Status
is `ready` for exit `0`, `acquisition_failed` for exit `1`, or `integrity_failed` for
exit `2`. Progress and sanitized diagnostics use standard error. A Click usage error
occurs before this summary exists.

During the design audit, a fresh download from the upstream URL matched the frozen
archive SHA-256, all four exact member identities above, and the three existing
cached checkpoint components byte-for-byte. The checked-in request and final lock
preserve those identities rather than trusting that observation indefinitely.

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
OaF backend. Enforcement occurs at the authenticated model-cache boundary: the host
accepts only the exact three-component content-addressed cache and mounts only that
directory read-only. The isolated runner validates that mounted set and does not scan
unrelated host filesystem locations for an HDF5 filename.

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
checked-in runtime requirements lock. Published compatible wheels are required by
default. An explicitly reviewed exception may consume an exact hash-pinned sdist only
to build a pure-Python wheel in a network-disabled, pinned CPython/build-toolchain
environment. The sdist, toolchain distributions, canonical build recipe/environment,
and resulting wheel are recorded by byte identity in a checked-in distribution-build
manifest. For each exception that manifest canonically records every file in the
exact PyPI release, including filename, package type, Python/tag identity, yanked
state, immutable URL, byte length, and SHA-256. Online generation re-fetches and
exact-compares the complete release; offline verification independently counts
target-compatible wheel tags and derives every `required_by` edge from direct pins
and selected wheel metadata. Two fresh builds must reproduce the wheel byte-for-byte,
and Task 8 must
repeat that build on native `linux/amd64`; a version match is not a substitute for
reproducing the locked wheel hash. Requirement markers use a complete frozen
CPython-3.7.17/Linux/x86_64 environment independent of the resolver host;
`platform_release` and `platform_version` are not frozen and therefore make a
dependency invalid rather than inheriting host values.

The exceptional-wheel validator reports the reproducibility hazards independently
of the final whole-wheel hash. Each build uses the exact allowlisted sdist, pinned
base manifest, CPython and build-tool wheels, a network-disabled container, a stable
absolute source root, a fresh empty output directory, and the sdist's recorded
`SOURCE_DATE_EPOCH`. It rejects `.pyc` members, absolute build-root bytes in any
uncompressed member, noncanonical or duplicate paths, and native or executable
members. It exact-compares the ordered ZIP member metadata, including member name,
timestamp, compression method, external attributes, CRC, and sizes, and validates
the ordered `RECORD` rows and their complete hashes before accepting the two
byte-identical outputs.

Checked locks are materialized into ignored wheelhouses by a deterministic command
which verifies every filename, byte length, and hash, rejects extras and symlinks,
and supports an explicit offline cache for locally built allowlisted wheels.

The provisional image already imports TensorFlow 1.15.5 and passes its runtime tests
without installing an additional Debian package. Task 8 therefore does not
reconstruct or mutate the base image's package set. The checked-in request:

```text
runtime/oaf_tf1/base-system-package-request.json
```

uses schema `crux.oaf-base-system-package-request/v1` and records the exact base image
manifest, `linux/amd64`, the fixed archive-keyring path, required Python/TensorFlow
and smoke probes, and exactly `additional_system_packages: []`. On the accepted
native host,
`seal_oaf_backend.py attest-base-system` runs the pinned base and provisional images,
records the complete sorted
`Package<TAB>Version<TAB>Architecture` `dpkg` inventory, its SHA-256, the base
archive-keyring SHA-256, Python 3.7.17, TensorFlow 1.15.5, and the probe outcomes
under schema `crux.oaf-base-system-package-evidence/v1`. The base and provisional
inventories must be byte-identical, proving that the build did not mutate the package
set.

The base manifest authenticates the complete base filesystem, including those
package bytes. Reconstructing a separate mutable Debian snapshot would add a second
supply-chain path without changing the runtime and is forbidden for this seal. The
final Dockerfile has no snapshot-fetch or local-`.deb` installation stage. If a
native smoke run proves that another package is required, Task 8 stops for design
review; the producer cannot add it opportunistically. Any nonempty
`additional_system_packages` request requires a new backend identity, an explicit
authenticated package-acquisition design, and a new seal.

The built image is identified by its `linux/amd64` OCI manifest digest, never by a
mutable tag. The preserved OCI layout also authenticates the selected manifest's
config digest and ordered layer digests. After importing that layout into the local
Docker content store, the host launches the image by the authenticated config digest,
which is Docker's local immutable image locator. Before every launch, the host
re-inspects that config digest and exact-compares it with the config selected by the
sealed manifest. The manifest digest remains the descriptor and runtime-lock identity;
the config digest is a separately recorded launch locator and is never substituted
for the manifest identity.

The calibration-bootstrap request is also the sole image-build recipe authority. Its
strict `image_build` object supplies an immutable Dockerfile frontend reference,
immutable BuildKit worker-image reference, exact Buildx and BuildKit versions, and
all exporter controls. Mutable frontend or worker tags, version ranges, a bundled
unspecified frontend, and host-default exporter behavior are invalid. The Dockerfile
starts with the exact request-authorized `# syntax=<name>@sha256:<digest>` directive;
that directive, the manual native bootstrap workflow source, and the canonical OCI
packer are covered by the runner source manifest.

The repository root is never passed to BuildKit as its local context. A checked-in,
request-hashed `crux.oaf-build-context-manifest/v1` record enumerates every regular
file admitted to the build by repository-relative path, exact byte length, and
SHA-256. Its source-manifest-covered materializer creates a fresh context containing
exactly those files plus the manifest itself, writes every directory as mode `0755`
and every file as mode `0644`, sets every staged mtime to epoch zero, and rejects
symlinks, hard links, special files, missing files, extra files, and byte drift. The
staged context contains no
`.dockerignore` or Dockerfile-specific ignore file, so repository ignore rules and
unlisted checkout files cannot affect transmitted bytes. Buildx runs with that fresh
directory as its working directory and `.` as its context. The manifest itself is
excluded from its own file array to avoid a content cycle; the bootstrap request
authenticates its exact bytes separately through
`build_context_manifest_sha256`.

The request fixes image `source_date_epoch: 0`. BuildKit receives that value through
`SOURCE_DATE_EPOCH`, enables deterministic multi-platform output and
`rewrite_timestamp`, exports one `linux/amd64` OCI-layout directory with OCI media
types and gzip compression at level `6`. `force_compression` is disabled so the
exporter does not gratuitously recompress already-compressed base layers. It emits no
provenance, SBOM, inline cache, or operator-supplied annotations. The deterministic
`org.opencontainers.image.created` index annotation, selected image config's
top-level `created` value, and every newly generated history `created` value encode
`1970-01-01T00:00:00Z`. The pinned base manifest is the sole source-policy authority;
before either build, the producer authenticates its raw manifest and config bytes and
derives their ordered layer digests and DiffIDs. BuildKit v0.31.2 deliberately leaves
an unchanged inherited base-layer prefix untouched when those DiffIDs match. The
output must therefore begin with the exact pinned base layer-digest and DiffID
prefix; those inherited blobs retain their original member timestamps and are exempt
from epoch normalization. Every layer after that exact prefix is generated or
rewritten by the frozen recipe and every one of its tar members must use epoch zero.
An inherited-prefix mismatch, unexpected rewrite, non-epoch generated member, or
compressed/uncompressed digest mismatch fails both builds.

BuildKit does not produce the preserved archive directly. It exports a directory,
which the source-manifest-covered packer validates and serializes as an uncompressed
POSIX ustar archive containing only regular files in UTF-8 byte-sorted path order.
Every header uses mode `0644`, UID/GID `0`, empty owner/group names, and mtime `0`;
there are no PAX headers or explicit directory members, and the archive ends with
exactly two zero blocks. The inspector strict-parses each OCI JSON document from its
original bytes with duplicate-key, type, descriptor-size, and descriptor-digest
checks, but it never requires Crux canonical-JSON formatting and never reserializes
OCI JSON. Every identity hashes the exact BuildKit-emitted bytes. The packer hashes
the exact archive bytes and independently hashes every OCI JSON document and blob
before import. Two fresh native builds must produce identical index, manifest,
config, compressed layers, uncompressed layer DiffIDs, canonical archive, and
archive SHA-256. A mismatch fails before calibration or sealing.

The runtime lock records:

- base image name, platform manifest/config digests, ordered inherited layer digests
  and DiffIDs, and Python version;
- every Python distribution name, version, filename, and SHA-256;
- the Python distribution-build-manifest SHA-256, including any exceptional sdist,
  pinned build toolchain, recipe/environment, and reproduced wheel identity;
- the exact build-context-manifest SHA-256 and its normalized file/directory modes;
- the calibration-bootstrap request and evidence SHA-256 values and their exact
  `image_build` object;
- the base-system-package-request and evidence SHA-256 values;
- the base-image archive-keyring SHA-256;
- the complete sorted base-image `Package<TAB>Version<TAB>Architecture` inventory and
  its SHA-256;
- exactly `additional_system_packages: []`;
- the vendored upstream source-manifest SHA-256;
- the runner source-manifest SHA-256;
- the exact deterministic process environment listed below;
- exact standard-error drain ring-buffer, read-chunk, and logical-line byte bounds;
- the exact maximum physical standard-output protocol-line byte length;
- TensorFlow build and ABI information reported at startup; and
- the final runner image manifest digest and authenticated config digest.

The runtime-lock and seal-evidence schemas remove the unused
`debian_snapshot_repository`, `debian_release_sha256`, and legacy package-distribution
records. They add exact
`base_system_package_request_sha256`,
`base_system_package_evidence_sha256`,
`calibration_bootstrap_request_sha256`,
`calibration_bootstrap_evidence_sha256`,
`build_context_manifest_sha256`,
`base_image_config_digest`,
`base_image_layer_digests`,
`base_image_layer_diff_ids`,
`image_build`,
`base_system_package_inventory`,
`base_system_package_inventory_sha256`, and
`additional_system_packages` fields. Each inventory row has exactly `name`,
`version`, and `architecture`; rows are unique and sorted by UTF-8 byte order.
The backend lock adds
`checkpoint_acquisition_request_sha256` and
`checkpoint_acquisition_evidence_sha256`. Seal validation independently reproduces
those hashes and exact-compares the three final checkpoint components with the
four-member request/evidence pair.

The exact build-context manifest excludes the final runtime lock, and the manifest
excludes its own bytes, avoiding both image-digest and manifest cycles. The bootstrap
request separately hashes that manifest, so the final lock can attest to the image
and exact staged context it describes.

The container entrypoint uses `/usr/bin/env -i` to discard every OCI- or
Docker-injected variable, including `PATH`, `HOSTNAME`, and `HOME`, then supplies a
fresh interpreter-start environment containing exactly these lock-recorded values
plus the bootstrap-only control defined below. The image-baked entrypoint constant is
the execution authority for this earliest pre-import check because the mounted
runtime lock cannot be parsed before Python starts. The mounted runtime lock is the
sealed record. The host launcher, image entrypoint, and runner copies must
exact-match; a disagreement is fatal and none overrides another. In particular,
`PYTHONHASHSEED` must be present before CPython starts; assigning it to `os.environ`
inside the runner is forbidden because that does not reseed the running interpreter:

| Environment variable | Frozen value |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `-1` |
| `OMP_NUM_THREADS` | `1` |
| `MKL_NUM_THREADS` | `1` |
| `OPENBLAS_NUM_THREADS` | `1` |
| `PYTHONHASHSEED` | `0` |
| `TF_NUM_INTRAOP_THREADS` | `1` |
| `TF_NUM_INTEROP_THREADS` | `1` |

CPython 3.7 is additionally launched with `PYTHONCOERCECLOCALE=0`. This is an exact,
bootstrap-only interpreter control, not a runtime-lock environment field. Before
validating the lock-recorded allowlist, the entrypoint requires that the control is
present with value `0` and removes it. A missing or mismatched value is a backend-fatal
startup failure. This prevents CPython locale coercion from injecting environment
state while keeping the post-bootstrap runner environment equal to the locked
seven-variable allowlist.

As its first startup step, before importing TensorFlow or any numeric library, the
runner validates the bootstrap control as described above, then validates the complete
lock-recorded environment allowlist and fails the handshake on a missing, extra, or
mismatched value. It then calls `random.seed(0)`,
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

Each accepted evidence form also contains `host_numeric_fingerprint`, a diagnostic
object with exactly `architecture`, `cpu_vendor_id`, `cpu_family`, `cpu_model`, and
`cpu_stepping`. All logical CPUs visible to the worker must report one identical
vendor/family/model/stepping tuple. Mutable `cpu_microcode` and `kernel_release`
values remain permitted in sanitized host diagnostics but are not members of this
stable comparison object. The fingerprint that generates the smoke oracle is copied
exactly into seal-evidence field
`reference_host_numeric_fingerprint`; every official verification or execution
records its current object in execution-attestation field
`host_numeric_fingerprint`. The object is evidence for investigating a mismatch, not
an acceptance partition and not part of prediction identity.

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

A native `linux/amd64` fingerprint different from the reference is not rejected by
name alone. After all authenticated image, lock, package, checkpoint, and tensor
checks pass, it is eligible for official execution only if its smoke artifact is
byte-identical to the sealed oracle. Every authenticated smoke mismatch is a backend
regression: it produces `failed` with exit `2` and publishes no prediction,
regardless of whether the diagnostic fingerprint equals the reference.
`environment_unsupported` is reserved for platform or host-evidence preflight
failure before inference starts; it can never classify a completed smoke comparison.

Every runner launch uses:

- `--network=none`;
- a read-only root filesystem;
- read-only backend-lock, runtime-lock, model-cache, and input mounts;
- a fresh process environment containing only the lock-enumerated allowlist plus the
  required bootstrap-only `PYTHONCOERCECLOCALE=0` control;
- all Linux capabilities dropped and `no-new-privileges`;
- bounded, execution-profile-locked `tmpfs` mounts for `/tmp` and `/dev/shm`;
- locked CPU, memory, and PID limits; and
- the numeric non-root UID/GID proven by the native seal pass.

Official runner launches use the image's default production entrypoint and mount the
final backend lock, runtime lock, and seal evidence read-only. Calibration launches
override the entrypoint with the separately source-addressed calibration entrypoint
and mount only the calibration-bootstrap request, acquisition/base-system evidence,
source manifests, model cache, and canonical inputs. The production entrypoint
rejects calibration inputs; the calibration entrypoint rejects final locks and
ordinary transcription requests. There is no shared mode flag.

The calibration-bootstrap request chooses numeric UID/GID `65532:65532` before the
image is built. Native base-system attestation must prove that neither identifier is
already assigned in the pinned base image. The reviewed seal profile repeats those
same identifiers exactly; it cannot choose replacements after measurement. If the
provisional image cannot pass calibration, smoke, and coverage under those
identifiers, sealing stops for design review and never runs as root. Exact tmpfs
sizes, resource limits, UID/GID, startup deadline, request deadline,
`max_input_audio_frames`, base-system-package evidence, and the empty
additional-system-package set are seal-required values with no code defaults.

### Native-amd64 calibration and sealing prerequisite

> **HPA-481/HPA-482 amendment:** The paragraph beginning “Every newly accepted
> bootstrap, measurement, and candidate execution preserves its own
> `crux.oaf-native-host-attestation-bundle/v1`.” is superseded by the signed
> same-work-job v2 authority in
> [the HPA-481/HPA-482 design](2026-07-30-hpa-481-work-job-artifact-attestation-design.md).
> All unrelated HPA-320 contracts remain in force.

Before Phase A can finalize the OaF locks or its real-checkpoint runner integration,
or satisfy HPA-320, a narrowly scoped evidence pass on a reference native
`linux/amd64` host must check in:

```text
config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json
```

under schema `crux.backend-seal-evidence/v1`. It records which accepted native-host
evidence form was used, its immutable reference/signature, the exact reference host
numeric fingerprint, base-manifest verification, the checkpoint-acquisition request
and evidence hashes, base-system request/evidence and exact package-inventory hashes,
wheel filenames and hashes, distribution-build-manifest and exceptional
sdist/toolchain/reproduction evidence, numeric UID/GID, tmpfs sizes,
CPU/memory/PID limits, standard-error drain bounds, the standard-output protocol-line
bound, startup and request deadlines, exactly
`max_input_audio_frames: 26214378`, measurements at that exact bound, tensor
inventories, smoke generation inputs and outputs, the security scan/advisory
snapshot, the exact build-context-manifest hash, pinned base config and ordered
layer/DiffID prefix, the preserved OCI-layout archive and raw
manifest/config/layer hashes, the candidate host-attestation-bundle hash, and hashes
of every reviewed evidence artifact. Missing values, sentinel values, and prose such
as `auto` or `unlimited` are invalid.

Pre-contract material formerly stored under `.github/hpa320-native-evidence/` is
scratch evidence and cannot become final authority by copying it. A post-contract
host-observation record proves only the host-attestation path described below. The
final native job and its producer must still regenerate authenticated host evidence
at the exact calibration commit.

Like the smoke oracle, seal evidence excludes final backend-lock and runtime-lock
hashes so the final locks can reference its SHA-256 without a content cycle.

The accepted host-observation material under
`.github/hpa320-native-evidence/job-90511044879/` proves that the authenticated
GitHub-hosted Linux X64 producer path exists. Its native-host-evidence payload
SHA-256 is
`f0631c1d7be27b40fecbabae50b39c056564c4c3db8cf2f80c50e23b3f779c8d`.
It is bootstrap evidence only, not seal evidence. The final calibration job must
regenerate native-host evidence at the exact calibration commit; neither this record
nor any earlier scratch job may be copied into final seal authority.

Every newly accepted bootstrap, measurement, and candidate execution preserves its
own `crux.oaf-native-host-attestation-bundle/v1`. A bundle contains the exact raw
GitHub job API response encoded as canonical lowercase hexadecimal, the canonical
host observation, the canonical `native_host_evidence` record, and a canonical
manifest that hashes those three files. Validation decodes the API record, reproduces
the nested `api_record_sha256`, and proves that the observation, completed API job,
workflow commit, phase-specific checked-in workflow reference, job ID, run URL,
runner labels, and numeric fingerprint agree. Relabeling another phase's three files
therefore fails even if the manifest's `phase` string is changed. The bootstrap,
measurement, or seal evidence produced by that execution records the bundle-manifest
SHA-256. Workflow artifacts alone are not durable authority: the accepted bundles
are checked in with their owning evidence, and the candidate bundle is published by
`seal` in the final seal commit.

These three HPA-320 seal-production workflows deliberately use the GitHub-hosted
evidence form, so their bundle schema can preserve and revalidate the raw GitHub job
record. The orchestrator-signed and approved-native-host forms remain valid for
post-seal official verification and prediction under the general execution contract,
but they cannot replace the accepted bootstrap, measurement, or candidate bundle for
this seal.

Task 8 has five checked-in request authorities and the following acyclic production
chain:

1. The checkpoint-acquisition request freezes the released archive and four-member
   layout independently of the not-yet-published backend lock.
2. The base-system-package request freezes the base manifest, native probes, and
   empty additional-package set independently of the not-yet-published runtime lock.
3. A `crux.oaf-calibration-bootstrap-request/v1` record freezes the provisional
   image inputs, exact build-context manifest, calibration entrypoint, non-root
   identity, process environment, container restrictions, diagnostic resource
   ceilings, and metric sampling interval. `bootstrap-image` publishes an
   authenticated OCI layout and `crux.oaf-calibration-bootstrap-evidence/v1`; it
   publishes no final lock.
4. A `crux.oaf-calibration-measurement-request/v1` record freezes the deterministic
   fixture derivation, every exact frame count and derived WAV hash, repetition
   count, required metrics, and output schemas. `measure` publishes diagnostic
   measurement evidence only.
5. After human review, a `crux.oaf-seal-profile-request/v1` record references every
   preceding request/evidence hash, repeats the approved UID/GID, and supplies every
   final CPU/memory/PID/tmpfs bound, startup/request deadline, standard-output and
   standard-error bound, and `max_input_audio_frames`. No field has a code default.
6. `calibrate` exact-compares the complete authority chain, reruns boundary and
   process-identity probes, generates all candidate evidence, validates it in
   memory, and immutably publishes a complete `crux.oaf-seal-candidate/v1`
   directory. It consumes no externally assembled candidate input.
7. After human review, `seal` strict-loads that candidate and publishes its exact
   repository-relative artifact allowlist in dependency order. It cannot regenerate,
   reinterpret, or replace candidate bytes.

### Calibration-bootstrap authority

> **HPA-481/HPA-482 amendment:** The paragraph beginning “`bootstrap-image`
> materializes two fresh copies of the exact request-authenticated build context and
> performs one `linux/amd64` build from each.” is superseded by the signed
> same-work-job bootstrap authority in
> [the HPA-481/HPA-482 design](2026-07-30-hpa-481-work-job-artifact-attestation-design.md).
> All unrelated HPA-320 contracts remain in force.

The calibration-bootstrap request fixes these diagnostic ceilings:

| Field | Exact value |
| --- | ---: |
| `runtime_uid` | `65532` |
| `runtime_gid` | `65532` |
| `cpu_limit_millis` | `2000` |
| `memory_limit_bytes` | `4294967296` |
| `pid_limit` | `256` |
| `tmp_bytes` | `1073741824` |
| `shm_bytes` | `1073741824` |
| `startup_deadline_seconds` | `300` |
| `request_deadline_seconds` | `1800` |
| `stdout_max_line_bytes` | `134217728` |
| `stderr_read_chunk_bytes` | `65536` |
| `stderr_max_line_bytes` | `65536` |
| `stderr_ring_buffer_bytes` | `1048576` |
| `monitor_interval_millis` | `10` |

These are safety ceilings for diagnostic execution, not automatically selected
final values. Reaching a ceiling, deadline, OOM state, signal, output bound, or
monitor failure is an operational failure. The producer publishes no partial
measurement and never raises a value automatically. A different ceiling requires a
reviewed request revision and a new bootstrap-evidence hash.

The provisional image contains two source-manifest-covered entrypoints. The default
production entrypoint accepts only final locks and
`crux.transcription-runner/v1`. The explicitly overridden calibration entrypoint
accepts only the bootstrap request/evidence chain and
`crux.oaf-calibration-runner/v1`. Before importing TensorFlow, it authenticates the
bootstrap request, checkpoint request/evidence and cache, base-system
request/evidence, runner/upstream source manifests, image config digest, and exact
process environment. It rejects final locks and ordinary transcription requests.
The production entrypoint rejects every calibration mount or request. Neither
entrypoint has a mode switch.

`bootstrap-image` materializes two fresh copies of the exact request-authenticated
build context and performs one `linux/amd64` build from each. Before building, it
authenticates the pinned base manifest/config bytes and derives their ordered
inherited layer digest/DiffID prefix. It canonical-packs both OCI-layout directories
and exact-compares their raw index, selected manifest, config, layers, DiffIDs, and
archive bytes. It also proves that both outputs begin with the exact inherited base
prefix and that every member in every later layer has epoch-zero mtime. It publishes
one authenticated archive and evidence only after those comparisons, then imports
the archive into Docker. Calibration launches use the imported config digest as the
immutable local locator. Bootstrap evidence records the build-context hash, base
config and layer-prefix identities, final config digest, and owning manifest, layers,
and DiffIDs; a mutable tag is invalid.

### Exact diagnostic fixture matrix

The sole seed is
`tests/fixtures/oaf_tf1_smoke/canonical.wav`, SHA-256
`8ec2aed65945b7002e17b51818495ca754a519c39940f739cdcd1403eb661673`.
Fixture derivation copies its 44-byte canonical PCM header, repeats the signed
16-bit mono PCM sample bytes cyclically, truncates them to exactly
`audio_frame_count * 2` bytes, and rewrites only the RIFF and data byte-length
fields. The input-view ID includes the derivation-schema ID and frame count.
Implementations must regenerate and exact-compare these rows before inference:

| Frames | Duration | WAV bytes | SHA-256 |
| ---: | ---: | ---: | --- |
| `44100` | 1 second | `88244` | `8ec2aed65945b7002e17b51818495ca754a519c39940f739cdcd1403eb661673` |
| `441000` | 10 seconds | `882044` | `17a326ecfd1789bf2757dd82646326ffaaff9781574fe41077e804ab8cbb555b` |
| `2646000` | 1 minute | `5292044` | `a230a22dc261c19577ec2b8854e59bef4794b9a86c2802180b515cb10b88d5cb` |
| `13230000` | 5 minutes | `26460044` | `8d5061666f280264699633d3368ba800ae4ac7280c2411d3ec36d12938fee0d5` |
| `26214378` | 50 MiB maximum | `52428800` | `0edebae8b719b0ac6778c588572a795d400a0b2730b4ce6ef33e240766c2b94d` |

The repetition count is exactly `3`. Each of the 15 measurement rows uses a fresh
container and therefore a unique Docker container ID as `process_instance_id`.
Rows are sorted by frame count, process ID, and repetition. Every measurement request
sets protocol field `max_input_audio_frames` to the matrix maximum, exactly
`26214378`; the smaller fixtures do not redefine the bound.

The native monitor requires cgroup v2. It samples `cpu.stat` at the locked 10 ms
interval. For each adjacent sample it computes
`ceil(1000 * delta_usage_usec / delta_elapsed_usec)` from `cpu.stat` and
`time.monotonic_ns()` and records the largest result as `peak_cpu_millis`. It records
cgroup `memory.peak` as the historically named `peak_rss_bytes` field and
`pids.peak` as `peak_pid_count`; a missing required cgroup file fails measurement. A
calibration-only monitor thread samples allocated bytes beneath `/tmp` and
`/dev/shm` at the same interval; its overhead is intentionally included. Host
readers record the maximum physical standard-output and logical standard-error line
lengths. Startup and request durations use `time.monotonic_ns()` and round upward to
whole milliseconds. Docker inspection supplies exit code, signal, and OOM state.
The runner supplies the canonical prediction SHA-256 and inference-call counters.

Every healthy in-bound measurement performs exactly one inference call. Each row
records input audio identity, frame count, repetition, process ID, inference-call
counters, peak CPU/RSS/tmpfs/PID values, startup/request duration, output bounds,
exit/signal/OOM state, and prediction SHA-256. The profile gate requires each final
limit to be strictly greater than the corresponding observed in-bound peak after
exact unit conversion and no greater than its bootstrap ceiling.

### Calibration, candidate publication, and seal

> **HPA-481/HPA-482 amendment:** The paragraph beginning “After those probes,
> `calibrate` generates exact 130/78/52 tensor coverage, uninitialized-variable
> evidence, active-dropout evidence, patched/unmodified `NoteSequence` parity, the
> nonempty raw smoke oracle and calibration-native event payload,
> security/advisory evidence, OCI archive/manifest/config/layer evidence, candidate
> runtime/backend/seal payloads, and a manifest of every candidate artifact.” is
> superseded by the signed same-work-job candidate authority in
> [the HPA-481/HPA-482 design](2026-07-30-hpa-481-work-job-artifact-attestation-design.md).
> All unrelated HPA-320 contracts remain in force.

The measurement evidence is diagnostic and cannot by itself be consumed by `seal`,
inference, or a scorer. Only a separately reviewed profile and complete calibrated
candidate can authorize publication. The profile request fixes
`max_input_audio_frames: 26214378`, exactly the measured 50 MiB fixture, and repeats
`runtime_uid: 65532` and `runtime_gid: 65532`. It records explicit reviewed headroom;
no percentage, rounding rule, or code default selects a final limit.

`calibrate` first validates the complete request/evidence/image chain and all 15
measurement rows. It then performs these four probes:

1. `26214377` frames in a persistent process;
2. `26214378` frames in that same process;
3. `26214379` frames in that same process; and
4. `26214378` frames in a fresh process.

The first, second, and fourth probes must each increment the authenticated
inference-call counter by exactly one and succeed. The third must return a typed
over-bound result before inference, leave the counter unchanged, keep the persistent
process healthy, and have no prediction SHA-256. OOM, signal, allocator failure,
process restart, or post-inference rejection is not boundary proof.

After those probes, `calibrate` generates exact 130/78/52 tensor coverage,
uninitialized-variable evidence, active-dropout evidence, patched/unmodified
`NoteSequence` parity, the nonempty raw smoke oracle and calibration-native event
payload, security/advisory evidence, OCI archive/manifest/config/layer evidence,
candidate runtime/backend/seal payloads, and a manifest of every candidate artifact.
It validates the complete directory and absence of final-lock hash cycles in memory,
then publishes the directory by no-replace atomic rename. The calibration-native
event payload excludes final-lock and descriptor fields and cannot be published as
an official prediction. The calibration entrypoint cannot publish official
predictions, locks, or a registry change.

`seal` consumes only that complete candidate. It publishes immutable evidence,
oracle, source/OCI manifests, and other dependencies first, the runtime lock second,
and the backend lock last. Existing different bytes are integrity failure and are
never overwritten. The registry remains `preseal` until the exact sealed image and
locks pass the real-checkpoint test and five-call production verification.

The bootstrap sequence is:

1. Regenerate native-host evidence at the exact manual-bootstrap commit.
2. Acquire the exact checkpoint and publish acquisition evidence.
3. Build and authenticate the provisional OCI image from the bootstrap request.
4. Attest the base/provisional package identity and native probes.
5. Review and commit the acquisition, bootstrap, and base-system evidence.
6. Commit a separate measurement request that binds the accepted bootstrap
   request/evidence hashes; the later measurement evidence binds the accepted
   checkpoint-acquisition and base-system evidence it consumed.
7. Regenerate native-host evidence at the exact measurement-request commit, generate
   the five exact fixtures, and publish all 15 measurements.
8. Review and commit the measurement evidence, then separately commit the
   seal-profile request.
9. Regenerate native-host evidence at that exact profile/calibration commit, run
   calibration, and review the complete candidate.
10. Seal dependencies, runtime lock, and backend lock in acyclic order.
11. Mount the final locks into the unchanged image and run exactly five production
   inference calls: one startup raw-oracle check in each of two processes, two
   post-ready artifact requests in the persistent process, and one in the fresh
   process.

After the manual bootstrap run is reviewed, its exact accepted evidence bytes are
checked in together at:

```text
docs/superpowers/evidence/hpa-320/native/checkpoint-acquisition-evidence.json
docs/superpowers/evidence/hpa-320/native/base-system-package-evidence.json
docs/superpowers/evidence/hpa-320/native/calibration-bootstrap-evidence.json
docs/superpowers/evidence/hpa-320/native/bootstrap-host-attestation/attestation-bundle.json
docs/superpowers/evidence/hpa-320/native/bootstrap-host-attestation/github-job-api-record.json.hex
docs/superpowers/evidence/hpa-320/native/bootstrap-host-attestation/native-host-evidence.json
docs/superpowers/evidence/hpa-320/native/bootstrap-host-attestation/native-host-observation.json
```

Only then can the checked-in calibration-measurement request bind the accepted
bootstrap request/evidence identity. The later measurement evidence also binds the
accepted checkpoint-acquisition and base-system evidence it consumed. After the
separate manual measurement run is reviewed, its exact accepted evidence bytes are
checked in at:

```text
docs/superpowers/evidence/hpa-320/native/calibration-measurement-evidence.json
docs/superpowers/evidence/hpa-320/native/measurement-host-attestation/attestation-bundle.json
docs/superpowers/evidence/hpa-320/native/measurement-host-attestation/github-job-api-record.json.hex
docs/superpowers/evidence/hpa-320/native/measurement-host-attestation/native-host-evidence.json
docs/superpowers/evidence/hpa-320/native/measurement-host-attestation/native-host-observation.json
```

The candidate host-attestation bundle is reviewed with the candidate and published
by `seal` to:

```text
docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/attestation-bundle.json
docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/github-job-api-record.json.hex
docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/native-host-evidence.json
docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/native-host-observation.json
```

It cannot be substituted by bootstrap or measurement evidence. The candidate's
artifact allowlist contains all four files, and its
`native_host_attestation_bundle_sha256` exact-matches both the bundle manifest and the
seal-evidence field. The staging files and checked-in files must be byte-identical.
The OCI archive named by bootstrap evidence is too large for the source tree; the
workflow preserves and uploads it by the exact `oci_layout_archive` identity, and
every later candidate or integration run either downloads those exact bytes or
rebuilds and exact-compares the same manifest/config/layer identities.

The host evidence nested in bootstrap evidence identifies the manual-bootstrap
commit and job. The host evidence nested in measurement evidence identifies the
later measurement-request commit and job. Each producer receives both
`--host-attestation-bundle` and the bundle-named `--host-evidence`, and rejects any
byte or identity mismatch. The candidate-phase pair supplied to `calibrate`
identifies the exact profile/candidate commit and job, becomes the seal evidence's
`native_host_evidence` and reference numeric fingerprint, and must itself be an
accepted native form. Each record's matching durable bundle-manifest hash is part of
its owning bootstrap, measurement, or seal evidence. The job IDs need not match. The
producer instead exact-compares the immutable request, evidence, image, checkpoint,
source, and profile hashes and reruns the final-profile probes on the candidate host.

The native producer commands are:

```bash
uv run python -m tools.hpa320.seal_oaf_backend \
  bootstrap-image \
  --request \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json \
  --host-attestation-bundle /workspace/hpa320/attestation-bundle.json \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --output artifacts/benchmark/backends/hpa320-bootstrap/calibration-image \
  --repository-root .

uv run python -m tools.hpa320.seal_oaf_backend \
  attest-base-system \
  --request runtime/oaf_tf1/base-system-package-request.json \
  --bootstrap-request \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json \
  --bootstrap-evidence \
    artifacts/benchmark/backends/hpa320-bootstrap/calibration-image/calibration-bootstrap-evidence.json \
  --host-attestation-bundle /workspace/hpa320/attestation-bundle.json \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --output \
    artifacts/benchmark/backends/hpa320-bootstrap/base-system-package-evidence.json

uv run python -m tools.hpa320.seal_oaf_backend \
  measure \
  --request \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-measurement-request.json \
  --bootstrap-request \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json \
  --bootstrap-evidence \
    artifacts/benchmark/backends/hpa320-bootstrap/calibration-image/calibration-bootstrap-evidence.json \
  --host-attestation-bundle /workspace/hpa320/attestation-bundle.json \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --model-cache artifacts/benchmark/model-cache \
  --checkpoint-evidence \
    artifacts/benchmark/backends/hpa320-bootstrap/checkpoint-acquisition-evidence.json \
  --base-system-evidence \
    artifacts/benchmark/backends/hpa320-bootstrap/base-system-package-evidence.json \
  --output artifacts/benchmark/backends/hpa320-bootstrap/calibration-measurements.json

uv run python -m tools.hpa320.seal_oaf_backend \
  calibrate \
  --request \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-profile-request.json \
  --measurement-evidence \
    docs/superpowers/evidence/hpa-320/native/calibration-measurement-evidence.json \
  --bootstrap-request \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json \
  --bootstrap-evidence \
    docs/superpowers/evidence/hpa-320/native/calibration-bootstrap-evidence.json \
  --host-attestation-bundle /workspace/hpa320/attestation-bundle.json \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --model-cache artifacts/benchmark/model-cache \
  --checkpoint-evidence \
    docs/superpowers/evidence/hpa-320/native/checkpoint-acquisition-evidence.json \
  --base-system-evidence \
    docs/superpowers/evidence/hpa-320/native/base-system-package-evidence.json \
  --output artifacts/benchmark/backends/hpa320-seal-candidate

uv run python -m tools.hpa320.seal_oaf_backend \
  seal \
  --candidate artifacts/benchmark/backends/hpa320-seal-candidate \
  --repository-root .
```

Each producer strict-validates every input and request/evidence link before creating
its destination. It stages on the destination filesystem, rejects symlinks and
special files, flushes and fsyncs complete bytes, and publishes by immutable atomic
rename. Existing identical bytes are idempotent; existing different bytes are an
integrity failure and are never replaced. A failed producer leaves no authoritative
output. Diagnostics are sanitized and cannot include secrets, environment values,
absolute operator paths, or unbounded tracebacks.

The candidate `artifacts` array is an exact role/path/hash allowlist. `seal` rejects
an unknown, missing, duplicate, absolute, escaping, symlinked, or noncanonical path.
It publishes dependencies first, the runtime lock second, and the backend lock last;
it never changes the registry. The reviewed final commit may include the registry
change only after native final verification succeeds against those exact bytes; the
sealed integration workflow repeats that verification on the commit.

`prepare-backend` and every `seal_oaf_backend` producer use one exit convention:

- exit `0` only after complete immutable success;
- exit `1` for operational inability, including unavailable acquisition,
  unsupported or unattested environment, unavailable cgroup v2 metrics, diagnostic
  ceiling or deadline reached, failed probe, unhealthy measurement, insufficient
  reviewed headroom, or inability to construct a candidate without contradictory
  authenticated bytes; and
- exit `2` for integrity, authentication, immutable-publication contradiction, or
  invalid request/evidence relationships, including a mutable image reference,
  incomplete measurement matrix, process-identity contradiction, inference-counter
  contradiction, or failure to reject over-bound input before inference.

After argument parsing succeeds, each command emits exactly one canonical typed
one-line JSON summary containing `status`, `exit_code`, `report_path`, and
`report_sha256`. The command-specific status distinguishes operational inability
from integrity failure. Argument-parser usage errors retain exit `2`, occur before
any output artifact or typed summary exists, and are therefore distinguishable from
integrity failures.

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

Paths named in this design that are absent from the current checkout are required
implementation deliverables, not claims that final lock, request, evidence, or
configuration artifacts already exist.

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

The checked-in backend registry gives the OaF entry an explicit `seal_state` whose
only values are `preseal` and `sealed`. Before the final locks exist, the entry is
`preseal`; any frozen inference command selecting it returns
`backend_not_sealed`, exit `1`, and publishes no prediction. Once a `sealed` entry is
selected, a missing, malformed, or contradictory backend lock, runtime lock, seal
evidence record, or referenced identity is an integrity failure with exit `2`; it is
never downgraded to `backend_unavailable`. The registry changes to `sealed` in the
same commit that publishes both final locks, so no checked-in state advertises a
sealed backend without its complete authorities.

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
records exact `python_implementation`, `python_version`, `python_abi`, and
`platform: "linux/amd64"` values plus the complete compatible distribution set and
wheel tags. Official `heuristic-onset-v1` verification and prediction publication are
native-`linux/amd64` only. macOS, `arm64`, and other platforms are diagnostic-only,
return `environment_unsupported`, and cannot publish an official heuristic
prediction. Supporting another platform or distribution set requires a new heuristic
backend ID rather than placing platform-dependent bytes behind the existing identity.
Routine Crux dependency upgrades therefore do not alter or retire the historical
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

The runner preserves the resulting `time_sec_raw` binary64 bit pattern without
recomputing it as an alternative algebraic expression. Confidence is the selected
TensorFlow float32 onset activation converted directly to binary64. The wire encoding
below transports both exact bit patterns; the host reconstructs those values and
passes them once to the artifact quantizer.

For an emitted onset, velocity uses the upstream `_unscale_velocity` order:

```text
require raw_velocity to be finite
clamped = max(min(raw_velocity, 1.0), 0.0)
unscaled = clamped * 127 + 0
velocity_midi = int(unscaled)
```

`int` truncates toward zero; the implementation must not round. A nonfinite raw
velocity is a per-item validation failure before clamping. The frozen upstream helper
would produce MIDI velocity `127` for `+inf`, `0` for `-inf`, and `0` for `NaN`;
HPA-320 deliberately rejects all three instead. This is an intentional adapter
safety divergence. It is named in the backend lock, covered by focused tests, and
included in the runner source manifest and seal evidence; it must not be described as
exact upstream behavior.

Upstream `infer_util.predict_sequence` returns a serialized `NoteSequence`, while
`magenta.music.sequences_lib.pianoroll_to_note_sequence` selects onset frames
internally and does not expose them. HPA-320 therefore applies a narrowly scoped,
hash-locked instrumentation patch to the vendored conversion path. For each emitted
note it returns paired metadata containing the original `start_frame`, pitch, and raw
velocity from the same postprocessing loop. The runner joins confidence from
`onset_probs[start_frame, pitch]` and uses that same `start_frame` as `frame_index`
for time evaluation; it never reconstructs the frame from the rounded or binary64
`note.start_time`.

The upstream source manifest continues to identify the unmodified upstream bytes.
The instrumentation patch and the exact build-time patch applier are enumerated in
the runner-source manifest and seal evidence; the image build applies them only after
verifying the upstream preimage. A parity test runs patched and unmodified upstream
conversion on the same locked tensors and requires byte-identical serialized
`NoteSequence` output. Any instrumentation change or parity failure requires design
review and a new backend identity.

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
- exact `max_input_audio_frames: 26214378`, matching the existing 50 MiB canonical
  PCM upload domain, with no implicit default;
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
  `max_input_audio_frames`, which is exactly `26214378` for this backend; and
- no trailing non-audio chunks that the strict WAV reader does not recognize.

`max_input_audio_frames` is measured only in mono PCM sample frames, equivalent to
the strict reader's `wave.getnframes()` result. It is not a spectrogram-frame count.
The number of spectrogram frames is separately derived by the frozen preprocessing
pipeline and is never used as this input-bound unit.

At two bytes per mono PCM frame plus the required 44-byte canonical WAV header,
`26214378` frames produce exactly `52428800` bytes, the existing 50 MiB upload
ceiling. The resulting duration is approximately 594.43 seconds. A file with
`26214379` frames is over-bound even if it reaches the backend through a local
benchmark command rather than the HTTP upload surface.

The bound is conservatively part of backend identity because it changes the accepted
input domain and resource proof. Raising it therefore creates a new backend identity,
even though prediction artifacts for identical inputs at or below the old bound
remain scientifically comparable when all other descriptor fields match.

HPA-321 cannot establish corpus fit because it inventories object metadata and does
not select or decode benchmark audio. HPA-322 must record the canonical mono PCM
frame count for every selected benchmark item. Before HPA-326 can declare a corpus
run eligible, it must exact-compare every selected frame count with this backend
bound and explicitly enumerate any excluded over-bound items. An over-bound item is
either excluded under a reviewed corpus policy or requires a separately identified
backend with a newly sealed input domain; hidden chunking is never permitted.
HPA-320 sealing is not conditional on that later corpus-fit result because the
canonical benchmark item set is outside HPA-320.

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

The runtime lock supplies a positive exact `stdout_max_line_bytes`. The host reads
standard output in bounded chunks under the active startup or request deadline and
rejects a physical line before buffering more than that locked byte count. EOF before
the terminating newline, an oversized line, invalid UTF-8, or malformed JSON is
backend-fatal. The bound is operational and seal-required; it has no code default and
does not enter prediction identity.

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
descriptor SHA-256, and native events. Each OaF native-event protocol object contains
exactly `frame_index`, `time_sec_binary64`, `native_class_id`, `model_output_bin`,
`native_midi_note`, `upstream_8hit_group_id`, `confidence_binary64`, and
`velocity_midi`.

`time_sec_binary64` and `confidence_binary64` are strings containing exactly 16
lowercase hexadecimal digits: the big-endian IEEE-754 binary64 bytes produced by
`struct.pack(">d", value).hex()`, without a prefix. The runner rejects a nonfinite
value before encoding. Raw time and confidence are never JSON numbers, decimal
strings, or pre-quantized fixed-point values on the wire. The host requires the exact
string form, reconstructs the binary64 value with
`struct.unpack(">d", bytes.fromhex(value))[0]`, rejects nonfinite or out-of-range
values, and performs the specified `Decimal.from_float` artifact quantization exactly
once. It independently evaluates the locked frame-time expression from `frame_index`
and requires the reconstructed time bit pattern to match, but never substitutes the
recomputed value in the artifact.

A domain failure returns a typed error with a stable machine code and sanitized
message. Tracebacks, local absolute paths, environment values, and checkpoint URLs
never appear on standard output.

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
Every header field duplicated inside `backend_descriptor` must exact-match the
embedded descriptor, including nullable values. A disagreement invalidates the
artifact; neither the flattened header field nor the descriptor wins.

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

> **HPA-481/HPA-482 amendment:** The paragraph beginning “`verify-backend` performs
> the following steps in order:” retains its 13-step post-seal behavior unchanged.
> Only pre-consumption acceptance of bootstrap, measurement, and candidate artifacts
> now uses the signed outside-in v2 gate specified by
> [the HPA-481/HPA-482 design](2026-07-30-hpa-481-work-job-artifact-attestation-design.md).
> All unrelated HPA-320 contracts remain in force.

`verify-backend` performs the following steps in order:

1. Strictly parse and hash the checkpoint-acquisition request, backend lock, and
   runtime lock, and exact-compare their checkpoint identities.
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
Cross-backend correlation must use `(descriptor_sha256, item_id)` and must never use
`item_id` alone; HPA-326 run manifests and reports inherit that composite key.

Every error in the verification or execution schema has exactly `code` and sanitized
`message`; errors are sorted first by the UTF-8 bytes of `code` and then of
`message`. The array is a canonical set of normalized failure facts, not a causal
event log; causal order and detailed sequence belong in bounded sanitized
diagnostics. Artifact entries are sorted first by `role` and then by `path`.

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

- request, evidence, lock, or cross-identity failure;
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
| Platform or authenticated host-evidence preflight fails before inference | `1` | verification/execution `environment_unsupported` |
| A pre-seal registry entry is selected for frozen inference | `1` | verification/execution `failed`, error `backend_not_sealed` |
| Legacy scoring cannot run or complete | `1` | legacy-score `failed` |
| Frozen OaF is requested through the legacy scorer before mapping | `1` | legacy-score `canonical_mapping_required` |
| `prepare-backend` cannot acquire or stage bytes, or its verify-only form finds no cache | `1` | setup summary `acquisition_failed` |
| Backend integrity, startup, any authenticated smoke comparison, process, timeout, or protocol fails | `2` | verification/execution `failed` with stable error code |
| `prepare-backend` detects request, evidence, lock, byte, cache, or publication integrity failure | `2` | setup summary `integrity_failed` |
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

Each of the two runner processes first performs one raw-oracle startup check before
emitting `ready`. After readiness, verification runs the fixture:

1. twice in one persistent runner; and
2. once in a fresh runner.

The full verification therefore performs exactly five inference calls: two startup
raw-oracle checks, one per process, plus three post-ready artifact requests. Startup
checks are handshake evidence and are not among the three published or directly
compared prediction artifacts.

All three post-ready prediction artifacts must be byte-identical; request-local
fields are already excluded by the artifact schema. The artifact must be finite and
nonempty. Backend identity fields must match the lock. Native classes, event count,
rounded times, MIDI notes, confidence values, velocity integers, and the complete
serialized event payload must match the smoke oracle exactly after the specified
six-decimal quantization. There is no numeric acceptance tolerance. A verifier may
calculate and report raw pre-quantization confidence deltas only after an exact-match
failure for diagnosis; those diagnostics can never turn a mismatched artifact into a
pass.

The smoke oracle and deterministically generated artifact are regression evidence for
execution parity, not accuracy claims. Adjusting the model, hparams, thresholds, or
postprocessor to make the fixture look musically better is forbidden.

## Normative Schema Key Reference

This appendix, not an implementation constant or a representative golden, is the
normative key-set authority for HPA-320 v1. Every object has exactly the keys listed
for its schema or named shape. A key is still present when its value is nullable.
Arrays contain only the named element shape. Unknown, duplicate, or omitted keys are
invalid. Types, value domains, ordering, nullability, and cross-field rules remain as
specified in the preceding sections.

### Shared nested shapes

```text
hash_ref:
  path, sha256
artifact_ref:
  role, path, sha256
error:
  code, message
checkpoint_identity:
  name, sha256, size
archive_member:
  name, role, sha256, size
variable:
  dtype, name, shape
non_inference_variable:
  dtype, name, reason, shape
python_distribution:
  filename, name, sha256, version
system_package:
  architecture, name, version
source_file:
  byte_length, path, sha256
licensed_source_file:
  license, path, sha256
native_output_bin:
  model_output_bin, native_class_id, native_midi_note
training_group:
  base_midi, group_id, member_pitches, output_bin
native_metadata_field:
  name, nullable, type
serialization:
  encoding, final_newline, key_order, whitespace
host_numeric_fingerprint:
  architecture, cpu_family, cpu_model, cpu_stepping, cpu_vendor_id
changed_file:
  path, sha256, status
probe:
  name, value
calibration_fixture:
  audio_frame_count, input_audio_sha256, input_view_id, source_audio_id,
  source_audio_sha256, wav_byte_length
calibration_resource_ceiling:
  cpu_limit_millis, memory_limit_bytes, monitor_interval_millis, pid_limit,
  request_deadline_seconds, shm_bytes, startup_deadline_seconds,
  stderr_max_line_bytes, stderr_read_chunk_bytes, stderr_ring_buffer_bytes,
  stdout_max_line_bytes, tmp_bytes
calibration_probe:
  inference_call_count_after, inference_call_count_before, persistent,
  rejected_before_inference, request_ordinal, row
measurement_row:
  exit_code, inference_call_count_after, inference_call_count_before,
  input_audio_sha256, input_frame_count, oom_killed, peak_cpu_millis,
  peak_pid_count, peak_rss_bytes, peak_shm_bytes, peak_tmp_bytes,
  prediction_sha256, process_instance_id, repetition, request_millis,
  signal, startup_millis, stderr_max_line_bytes, stdout_max_line_bytes
```

`native_host_evidence` has exactly `kind`, `official_execution_allowed`, `payload`,
and `sha256`. Its `payload` is discriminated by `kind` and has exactly one of:

```text
github_hosted_runner:
  api_record_sha256, host_numeric_fingerprint, job_id, runner_arch,
  runner_os, run_url, workflow_commit
orchestrator_attestation:
  attestation_sha256, host_numeric_fingerprint, issuer, key_id,
  worker_architecture
native_seal_host:
  daemon_architecture, evidence_sha256, host_architecture,
  host_numeric_fingerprint
```

The OaF metadata object under schema `magenta-oaf-native-metadata-v1` has exactly
`upstream_8hit_group_id`. The heuristic metadata object under schema
`crux-empty-native-metadata-v1` has no keys. Accordingly, the OaF backend lock's
`native_metadata_fields` contains exactly one `native_metadata_field` row:
`name: "upstream_8hit_group_id"`, `nullable: true`, and `type: "string"`.

The backend lock's `hparams` object has exactly:

```text
acoustic_rnn_dropout_keep_prob, bidirectional,
combined_lstm_dropout_keep_prob, combined_lstm_units,
conv_dropout_keep_amts, conv_filters, drum_data_map, drum_note_duration,
drum_prediction_map, drums_only, fc_dropout_keep_amt, fc_size,
frame_lstm_units, frame_threshold, hop_length, log_amplitude, min_gap,
num_pitches, offset_lstm_units, offset_network, offset_threshold,
onset_length, onset_lstm_units, onset_threshold, peak_picking,
sample_rate, share_conv_features, spec_fmin, spec_hop_length, spec_htk,
spec_mel_bins, spec_type, transform_audio, use_librosa, velocity_bias,
velocity_lstm_units, velocity_scale, viterbi_decoding
```

### Descriptor, lock, and seal schemas

`crux.transcription-backend-descriptor/v1`:

```text
architecture_id, backend_id, backend_lock_sha256, descriptor_schema,
model_artifact_set_sha256, model_id, native_metadata_schema_id,
native_output_space_id, prediction_schema, protocol_schema,
runtime_image_manifest_digest, runtime_lock_sha256, training_data_map_id,
upstream_source_commit
```

`crux.heuristic-backend-descriptor/v1`:

```text
adapter_source_manifest_sha256, architecture_id, backend_id,
descriptor_schema, model_id, native_metadata_schema_id,
native_output_space_id, parameter_lock_sha256, prediction_schema
```

`crux.transcription-backend-lock/v1`:

```text
architecture_id, backend_id, checkpoint_acquisition_evidence_sha256,
checkpoint_acquisition_request_sha256, checkpoint_archive,
checkpoint_components, checkpoint_inventory, checkpoint_url,
descriptor_schema, drum_prediction_map, execution_report_schema,
host_adapter_source_manifest_sha256, hparams, hparams_source,
legacy_conversion_coverage_sha256, legacy_score_report_schema,
max_input_audio_frames, model_id, native_metadata_fields,
native_metadata_schema_id, native_output_bins, native_output_space_id,
non_inference_inventory, prediction_schema, protocol_schema,
required_inference_inventory, runtime_image_manifest_digest,
runtime_lock_sha256, schema, seal_evidence_sha256, serialization,
smoke_audio_sha256, smoke_oracle_sha256, training_data_map_id,
training_groups, upstream_repository, upstream_source_commit,
upstream_source_manifest_sha256, verification_report_schema
```

Here `checkpoint_archive` and every `checkpoint_components` row use
`checkpoint_identity`; checkpoint and required inventories use `variable`;
non-inference inventory uses `non_inference_variable`; output bins, groups, metadata
fields, and serialization use the shared shapes above.

`crux.transcription-runtime-lock/v1`:

```text
additional_system_packages, base_image, base_image_archive_keyring_sha256,
base_image_config_digest, base_image_layer_diff_ids, base_image_layer_digests,
base_image_manifest_digest, base_system_package_evidence_sha256,
base_system_package_inventory, base_system_package_inventory_sha256,
base_system_package_request_sha256, build_context_manifest_sha256,
calibration_bootstrap_evidence_sha256, calibration_bootstrap_request_sha256,
distribution_build_manifest_sha256, environment, image_build,
oci_layout_manifest_sha256, platform, python_distributions, python_version,
runner_source_manifest_sha256, runtime_image_config_digest,
runtime_image_manifest_digest, schema, seal_evidence_sha256,
stderr_max_line_bytes, stderr_read_chunk_bytes, stderr_ring_buffer_bytes,
stdout_max_line_bytes, tensorflow_abi, tensorflow_build,
upstream_source_manifest_sha256
```

`environment` has exactly the seven variable names in the deterministic-runtime
table. `image_build` is the exact calibration-bootstrap request object described
below. Package and distribution arrays use their shared shapes.

`crux.backend-seal-evidence/v1`:

```text
additional_system_packages, advisory_snapshot_sha256,
base_image_archive_keyring_sha256, base_image_config_digest,
base_image_layer_diff_ids, base_image_layer_digests, base_image_manifest_digest,
base_system_package_evidence_sha256, base_system_package_inventory,
base_system_package_inventory_sha256, base_system_package_request_sha256,
boundary_probes, build_context_manifest_sha256,
calibration_bootstrap_evidence_sha256,
calibration_bootstrap_request_sha256, calibration_measurement_evidence_sha256,
calibration_measurement_request_sha256, checkpoint_acquisition_evidence_sha256,
checkpoint_acquisition_request_sha256, checkpoint_archive,
checkpoint_components, checkpoint_inventory, cpu_limit_millis,
distribution_build_manifest_sha256, host_adapter_source_manifest_sha256,
instrumentation_patch_sha256, legacy_conversion_coverage_sha256,
max_input_audio_frames, measurements, memory_limit_bytes,
native_host_attestation_bundle_sha256, native_host_evidence,
non_inference_inventory, oci_layout_archive,
oci_layout_manifest_sha256, pid_limit, python_distributions,
reference_host_numeric_fingerprint, request_deadline_seconds,
required_inference_inventory, runner_source_manifest_sha256, runtime_gid,
runtime_image_config_digest, runtime_image_index_digest,
runtime_image_layer_diff_ids, runtime_image_layer_digests,
runtime_image_manifest_digest, runtime_uid, schema, seal_profile_request_sha256,
security_scan_sha256, shm_bytes,
smoke_audio_sha256, smoke_oracle_sha256, smoke_prediction_sha256,
startup_deadline_seconds, stderr_max_line_bytes, stderr_read_chunk_bytes,
stderr_ring_buffer_bytes, stdout_max_line_bytes, tensor_coverage_sha256,
tensorflow_abi, tensorflow_build, tmp_bytes,
upstream_source_manifest_sha256
```

`measurements` contains `measurement_row`; `boundary_probes` contains
`calibration_probe`, whose `row` uses `measurement_row`; archive, component,
inventory, distribution, package, host-evidence, and fingerprint values use the
shared shapes. The candidate manifest is a review envelope and is deliberately not
referenced from seal evidence; this keeps the candidate-to-payload graph acyclic.
`smoke_prediction_sha256` addresses the calibration protocol's canonical native-event
array described below, not a `crux.drum-prediction-events/v1` artifact containing
final lock identities.

`crux.heuristic-parameter-lock/v1`:

```text
adapter_source_manifest_sha256, algorithm, backend_id, distributions,
platform, python_abi, python_implementation, python_version, schema,
serialization, wheel_tags
```

Its `algorithm` has exactly `classifier_branches`, `fixed_native_midi_notes`,
`fixed_velocity_midi`, `hop_length`, `librosa_calls`, and `sample_rate`. Every
`librosa_calls` row has exactly `arguments` and `call`; every `arguments` row has
exactly `name` and `value`, so resolved defaults are data rather than open object
keys. Every classifier-branch row has exactly `condition`, `native_midi_note`, and
`velocity_midi`. The two fixed-value fields are arrays of integers.

### Bootstrap request and evidence schemas

`crux.oaf-checkpoint-acquisition-request/v1`:

```text
archive, archive_members, backend_id, checkpoint_url,
published_component_names, schema
```

`crux.oaf-checkpoint-acquisition-evidence/v1`:

```text
acquisition_mode, archive, archive_members, cache_path,
model_artifact_set_sha256, published_components, request_sha256, schema
```

The request/evidence `archive` uses `checkpoint_identity`; `archive_members` use
`archive_member`; `published_components` use `checkpoint_identity`.

`crux.oaf-base-system-package-request/v1`:

```text
additional_system_packages, base_image, base_image_archive_keyring_sha256,
base_image_manifest_digest, platform, required_probes, schema
```

`crux.oaf-base-system-package-evidence/v1`:

```text
additional_system_packages, base_image_archive_keyring_sha256,
base_image_manifest_digest, native_host_evidence, package_inventory,
package_inventory_sha256, probes, request_sha256, schema
```

`required_probes` is an array of names; `probes` use `probe`; package arrays use
`system_package`. The evidence's `native_host_evidence` bytes must equal the file
authenticated by the same bootstrap-phase bundle named by calibration-bootstrap
evidence; base-system attestation cannot introduce a second unaudited host record.

`crux.oaf-calibration-bootstrap-request/v1`:

```text
backend_id, base_image_manifest_digest, base_system_package_request_sha256,
build_context_manifest_sha256, checkpoint_acquisition_request_sha256,
container_restrictions,
distribution_build_manifest_sha256, environment, image_build,
instrumentation_patch_sha256, python_coerce_c_locale, resource_ceiling,
runner_source_manifest_sha256, runtime_gid, runtime_uid, schema,
upstream_source_manifest_sha256
```

Its `container_restrictions` has exactly:

```text
drop_capabilities, network, no_new_privileges, platform, read_only_root
```

Its `resource_ceiling` uses `calibration_resource_ceiling`. Its `environment` has
exactly the seven deterministic-runtime variable names, and
`python_coerce_c_locale` is exactly the bootstrap-only
`PYTHONCOERCECLOCALE=0` control.

Its `image_build` has exactly:

```text
annotations, buildkit_image, buildkit_version, buildx_binary_sha256,
buildx_binary_size, buildx_binary_url, buildx_version, compression,
compression_level, dockerfile_frontend, dockerfile_frontend_version,
exporter, exporter_tar, force_compression, inline_cache,
multi_platform_deterministic, oci_archive, oci_media_types, platform,
provenance, rewrite_timestamp, sbom, source_date_epoch
```

The exact toolchain values are:

```text
buildkit_image:
  moby/buildkit@sha256:63db51c9b30208a7c2b1c40392c7ebb9ce2f85ba238a18a85420f8f5ea2d4684
buildkit_version: v0.31.2
buildx_binary_sha256:
  d41ece72044243b4f58b343441ae37446d9c29a7d6b5e11c61847bbcf8f7dfda
buildx_binary_size: 65265826
buildx_binary_url:
  https://github.com/docker/buildx/releases/download/v0.35.0/buildx-v0.35.0.linux-amd64
buildx_version: v0.35.0
dockerfile_frontend:
  docker/dockerfile-upstream@sha256:3d6d54b33351b396a910d33248754b86b1d7dd838b4eeb9575d8903a209f6516
dockerfile_frontend_version: 1.25.0
```

The two image references are exact `linux/amd64` manifest references. Both version
fields are exact releases, not ranges, and the downloaded Buildx binary must
exact-match both its byte length and SHA-256 before execution. `annotations` is
exactly `[]` and means no operator-supplied annotations; `compression` is `gzip`;
`compression_level` is `6`; `exporter` is `oci`; `exporter_tar` and
`force_compression` are `false`; `multi_platform_deterministic`,
`oci_media_types`, and `rewrite_timestamp` are `true`; `inline_cache`,
`provenance`, and `sbom` are `false`; `platform` is `linux/amd64`; and
`source_date_epoch` is `0`.

Its `oci_archive` has exactly:

```text
compression, final_zero_blocks, format, gid, gname, member_mode,
member_types, mtime, path_order, uid, uname
```

Those values are exactly `none`, `2`, `posix-ustar`, `0`, `""`, `420`,
`regular-files-only`, `0`, `utf8-byte`, `0`, and `""`, respectively.

`crux.oaf-calibration-bootstrap-evidence/v1`:

```text
base_image_config_digest, base_image_layer_diff_ids, base_image_layer_digests,
build_context_manifest_sha256, calibration_bootstrap_request_sha256,
image_build, native_host_attestation_bundle_sha256, native_host_evidence,
oci_layout_archive, oci_layout_manifest_sha256,
runtime_image_config_digest, runtime_image_index_digest,
runtime_image_layer_diff_ids, runtime_image_layer_digests,
runtime_image_manifest_digest, schema
```

`oci_layout_archive` uses `checkpoint_identity`; `native_host_evidence` uses its
shared discriminated shape. The evidence `image_build` must exact-match the request
object after the producer independently observes both tool versions and immutable
image references. The config digest is the immutable Docker launch locator after
importing the authenticated OCI archive; the manifest digest remains the sealed
image identity. Its context and base fields exact-match the request-authenticated
manifest, independently authenticated base bytes, and OCI-layout manifest. Its
bundle hash addresses the accepted bootstrap-phase bundle manifest.

`crux.oaf-calibration-measurement-request/v1`:

```text
backend_id, calibration_bootstrap_evidence_sha256,
calibration_bootstrap_request_sha256, fixture_derivation, fixtures,
output_schemas, repetition_count, required_metrics, schema
```

`fixture_derivation` has exactly:

```text
algorithm, canonical_header_bytes, channel_count, sample_rate,
sample_width_bytes, source_path, source_sha256
```

Each `fixtures` row uses `calibration_fixture`. The five rows are exactly the frame
counts, WAV lengths, and hashes in the diagnostic fixture matrix, and
`repetition_count` is exactly `3`. `required_metrics` and `output_schemas` are
arrays of exact string identifiers, not open objects.

The corresponding immutable measurement evidence uses schema
`crux.oaf-calibration-measurement-evidence/v1` and exactly:

```text
base_system_package_evidence_sha256,
calibration_bootstrap_evidence_sha256,
checkpoint_acquisition_evidence_sha256, measurement_rows,
native_host_attestation_bundle_sha256, native_host_evidence, request_sha256,
runtime_image_config_digest, runtime_image_manifest_digest, schema
```

`measurement_rows` contains `measurement_row`. The bundle hash addresses the
accepted measurement-phase bundle manifest; relabeling bootstrap or candidate files
cannot satisfy its phase-specific workflow-reference checks.

`crux.oaf-seal-profile-request/v1`:

```text
base_system_package_evidence_sha256,
base_system_package_request_sha256,
calibration_bootstrap_evidence_sha256,
calibration_bootstrap_request_sha256,
calibration_measurement_evidence_sha256,
calibration_measurement_request_sha256,
checkpoint_acquisition_evidence_sha256,
checkpoint_acquisition_request_sha256, cpu_limit_millis,
max_input_audio_frames, memory_limit_bytes, pid_limit,
request_deadline_seconds, runtime_gid, runtime_uid, schema, shm_bytes,
startup_deadline_seconds, stderr_max_line_bytes, stderr_read_chunk_bytes,
stderr_ring_buffer_bytes, stdout_max_line_bytes, tmp_bytes
```

`crux.oaf-seal-candidate/v1`:

```text
artifacts, backend_lock_payload_sha256,
calibration_bootstrap_evidence_sha256,
calibration_measurement_evidence_sha256, checkpoint_components,
checkpoint_prefix, model_artifact_set_sha256,
native_host_attestation_bundle_sha256, required_inference_inventory_sha256,
runtime_lock_payload_sha256, schema, seal_evidence_payload_sha256,
seal_profile_request_sha256
```

`artifacts` uses `artifact_ref`; `checkpoint_components` uses
`checkpoint_identity`. The three `*_payload_sha256` fields address candidate bytes
before immutable publication. No final payload references the candidate manifest
hash. The candidate bundle hash addresses the candidate-phase bundle manifest, whose
four files must all appear in `artifacts`.

### Protocol and prediction schemas

`crux.oaf-calibration-runner/v1` request:

```text
audio_frame_count, audio_path, audio_sha256, max_input_audio_frames,
request_id, type
```

Ready response:

```text
base_system_package_evidence_sha256,
calibration_bootstrap_request_sha256,
checkpoint_acquisition_evidence_sha256, checkpoint_inventory_sha256,
non_inference_count, non_inference_inventory_sha256, process_instance_id,
protocol_schema, required_inference_count,
required_inference_inventory_sha256, restored_inference_count,
runner_source_manifest_sha256, runtime_image_config_digest, tensorflow_abi,
tensorflow_build, type, upstream_source_manifest_sha256
```

Measurement or boundary-probe response:

```text
audio_sha256, inference_call_count_after, inference_call_count_before,
native_events, prediction_sha256, rejected_before_inference, request_id, type
```

The request `type` is exactly `measure` or `calibration_probe`; the response echoes
that exact type, and the readiness response uses `ready`. An over-bound probe has an
empty `native_events` array, null `prediction_sha256`,
`rejected_before_inference: true`, and identical before/after inference counters.
For an accepted inference, `prediction_sha256` is the SHA-256 of `native_events`
serialized alone as canonical UTF-8 JSON with lexicographically sorted object keys,
no insignificant whitespace, and no trailing newline. It is calibration-native
evidence, contains no final lock or descriptor identity, and is never published as a
benchmark prediction.
The calibration entrypoint alone accepts this protocol. The production entrypoint
rejects it before TensorFlow import.

`crux.transcription-runner/v1` request:

```text
audio_path, audio_sha256, backend_descriptor_sha256, request_id, type
```

Ready response:

```text
backend_descriptor, backend_descriptor_sha256, backend_lock_sha256,
checkpoint_inventory_sha256, non_inference_count,
non_inference_inventory_sha256, protocol_schema, python_version,
required_inference_count, required_inference_inventory_sha256,
restored_inference_count, runner_source_manifest_sha256,
runtime_lock_sha256, smoke_audio_sha256, smoke_oracle_sha256,
smoke_prediction_sha256, smoke_status, tensorflow_abi, tensorflow_build,
type, upstream_source_manifest_sha256
```

Successful result:

```text
audio_sha256, backend_descriptor_sha256, native_events, request_id, type
```

Error response:

```text
code, message, type
```

Each runner `native_events` row has exactly:

```text
confidence_binary64, frame_index, model_output_bin, native_class_id,
native_midi_note, time_sec_binary64, upstream_8hit_group_id, velocity_midi
```

`crux.drum-prediction-events/v1` header:

```text
architecture_id, artifact_role, audio_frame_count, backend_descriptor,
backend_descriptor_sha256, backend_lock_sha256, byte_length, channel_count,
input_audio_sha256, input_view_id, model_artifact_set_sha256, model_id,
native_metadata_schema_id, native_output_space_id, parameter_lock_sha256,
record_type, runtime_lock_sha256, sample_rate, sample_width_bytes, schema,
source_audio_id, source_audio_sha256, training_data_map_id,
upstream_source_commit
```

Event:

```text
canonical_class, confidence, event_index, mapping_status, model_output_bin,
native_class_id, native_metadata, native_midi_note, prediction_map_version,
record_type, time_sec, velocity_midi
```

Terminal:

```text
event_count, prefix_sha256, record_type
```

`crux.oaf-smoke-oracle/v1`:

```text
input_audio_frame_count, input_audio_sha256, input_view_id, native_events,
schema, source_audio_id, source_audio_sha256
```

Its `native_events` rows use the runner native-event key set above.

### Provenance and report schemas

> **HPA-481/HPA-482 amendment:** The paragraph beginning “The host-attestation
> bundle's `phase` is exactly one of `bootstrap`, `measurement`, or `candidate`.” is
> superseded by the strict v2 bundle and containing-evidence schemas in
> [the HPA-481/HPA-482 design](2026-07-30-hpa-481-work-job-artifact-attestation-design.md).
> All unrelated HPA-320 contracts remain in force.

`crux.input-view-manifest/v1`:

```text
input_audio_path, input_audio_sha256, input_view_id, schema,
source_audio_id, source_audio_sha256, source_path
```

`crux.backend-execution-item-id/v1`:

```text
input_audio_sha256, input_view_id, schema, source_audio_id,
source_audio_sha256
```

`crux.backend-execution-attestation/v1`:

```text
backend_id, changed_files_manifest, checkout_dirty, cpu_limit,
descriptor_sha256, git_commit, host_numeric_fingerprint, memory_bytes,
pid_limit, request_deadline_seconds, schema, shm_bytes,
startup_deadline_seconds, strict_mode, tmp_bytes
```

`changed_files_manifest` is `hash_ref` or null for a clean checkout. The referenced
canonical JSON array contains `changed_file` rows.

`crux.backend-verification-report/v1`:

```text
artifacts, backend_lock_sha256, descriptor, descriptor_sha256, errors,
execution_attestation, exit_code, finished_at, parameter_lock_sha256,
report_type, run_id, runtime_lock_sha256, schema, seal_evidence_sha256,
smoke, started_at, status, tensor_coverage
```

Its `tensor_coverage` has exactly `non_inference_count`,
`non_inference_inventory_sha256`, `report_path`, `report_sha256`,
`required_count`, `required_inventory_sha256`, `restored_count`, and `status`.
Its `smoke` has exactly `audio_sha256`, `oracle_sha256`, `prediction_path`,
`prediction_sha256`, and `status`. `artifacts` use `artifact_ref`; `errors` use
`error`; execution attestation uses `hash_ref`.

`crux.backend-execution-report/v1`:

```text
backend_lock_sha256, descriptor, descriptor_sha256, errors,
execution_attestation, exit_code, finished_at, items,
parameter_lock_sha256, report_type, run_id, runtime_lock_sha256, schema,
seal_evidence_sha256, started_at, status, verification_report
```

Each item has exactly `errors`, `input_audio_sha256`, `input_view_id`, `item_id`,
`midi`, `prediction`, `source_audio_id`, `source_audio_sha256`, and `status`.
`prediction`, `midi`, `execution_attestation`, and `verification_report` use
`hash_ref`; errors use `error`.

`crux.legacy-score-report/v1`:

```text
backend_id, backend_validation_status, errors, exit_code, finished_at,
report_type, run_id, schema, score_report, started_at, status, workflow_mode
```

`score_report` uses `hash_ref`; errors use `error`.

`crux.legacy-tf2-conversion-coverage/v1`:

```text
candidate_matches, converter_source_manifest_sha256, matching_algorithm,
matching_algorithm_version, model_artifact_set_sha256,
observed_hdf5_sha256, required_inference_inventory_sha256,
restored_required, restored_required_count, schema,
tf2_model_source_manifest_sha256, unmatched_required
```

`restored_required` and `unmatched_required` use `variable`. Each candidate match has
exactly `assigned`, `candidate_name`, `dtype_compatible`, `match_kind`,
`required_name`, and `shape_compatible`.

Supporting HPA-320 evidence schemas are equally strict:

```text
crux.oaf-tensor-coverage/v1:
  active_predict_dropout, checkpoint_inventory, non_inference_inventory,
  note_sequence_byte_parity, required_inference_inventory, schema,
  uninitialized_required
crux.oaf-build-context-manifest/v1:
  directory_mode, file_mode, files, manifest_path, schema
crux.oaf-native-host-attestation-bundle/v1:
  api_record, native_host_evidence, native_host_observation, phase, schema
crux.oaf-oci-layout-manifest/v1:
  archive, base_image_config_digest, base_image_layer_diff_ids,
  base_image_layer_digests, config_digest, image_manifest_digest,
  index_digest, layer_diff_ids, layer_digests, schema
crux.oaf-upstream-source-manifest/v1:
  covered_roots, files, schema, upstream_commit, upstream_repository
crux.oaf-runner-source-manifest/v1:
  covered_roots, files, schema
crux.oaf-host-adapter-source-manifest/v1:
  covered_roots, files, schema
```

The tensor inventories use the shared variable shapes. The build-context manifest
uses `source_file`; its `directory_mode` and `file_mode` values are the decimal
integers `493` and `420`, and `manifest_path` is exactly
`runtime/oaf_tf1/build-context-manifest.json`. Its `files` array is UTF-8 byte-sorted
and excludes that self-named manifest while enumerating every other context file. The
bootstrap request separately authenticates the manifest's exact bytes.

The host-attestation bundle's `phase` is exactly one of `bootstrap`, `measurement`, or
`candidate`. Its other three keys use `checkpoint_identity`; their `name` values are
exactly `github-job-api-record.json.hex`, `native-host-evidence.json`, and
`native-host-observation.json`, respectively. The hexadecimal record contains only
lowercase ASCII hexadecimal plus one final LF; removing the final LF and decoding
pairs yields the exact raw GitHub API response bytes hashed by the nested
`api_record_sha256`. The bundle manifest does not hash itself.

The OCI `archive` uses `checkpoint_identity`. OCI JSON is strict-parsed directly from
the original BuildKit-emitted bytes, and its descriptors are cross-checked against
those exact bytes; the documents need not use Crux canonical-JSON serialization and
must not be reserialized before hashing or comparison. Upstream manifest files use
`licensed_source_file`; runner and host-adapter manifest files use `source_file`.

## Test Strategy

Every versioned schema introduced by HPA-320, including the `crux.*/v1` schemas and
`magenta-oaf-native-metadata-v1`, is listed in
`tests/benchmark/schema_goldens/manifest.json`. Each manifest entry contains exactly
`schema`, `validator_modules`, and `golden_path`. Its `validator_modules` list contains
unique importable module paths sorted by UTF-8 bytes; schemas deliberately enforced
on both sides of the isolation boundary list both validators. Each golden is checked
in as canonical bytes, must pass every listed validator, and has unknown-key,
missing-key, duplicate-key, and wrong-type rejection tests. This test manifest is a
drift detector against the normative key reference above, not an inference identity,
and is not referenced by a backend or runtime lock. A golden cannot add, remove, or
reinterpret a key from that reference.

### Unit tests

- Strict backend-lock and runtime-lock schema validation.
- Strict bootstrap-request/evidence, measurement-request/evidence, seal-profile,
  seal-evidence, seal-candidate, build-context-manifest, OCI-layout-manifest, and
  native-host-attestation-bundle schemas; canonical Crux-record hashes; and complete
  cross-authority linkage.
- Build-context manifest strictness and fresh exact materialization, including
  normalized modes and rejection of ignored, extra, missing, linked, special, or
  byte-drifted inputs.
- OCI index selection; strict parsing without canonicalizing the raw JSON bytes;
  manifest/config/layer verification; pinned-base manifest/config and inherited
  digest/DiffID-prefix authentication; epoch-zero enforcement only after that prefix;
  immutable config-digest import and launch; and mutable-tag rejection.
- Exact five-row fixture derivation from the canonical WAV, including all frame
  counts, byte lengths, and SHA-256 values.
- Cgroup-v2 metric parsing and unit conversion, monotonic duration rounding,
  10-millisecond sampling, output-line accounting, and unavailable-metric failure.
- Exact 15-row measurement-matrix completeness, fresh-process uniqueness,
  inference-counter increments, stable row ordering, and no partial publication.
- Persistent/fresh boundary-probe ordering and counters, including proof that the
  over-bound request is rejected before inference.
- Calibration exit `0`/`1`/`2` classification, immutable no-replace publication,
  and proof that failure leaves no authoritative output.
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
- Exact binary64 frame-time evaluation order; 16-digit big-endian wire round-trip for
  time and confidence; proof that artifact quantization occurs exactly once; and
  upstream velocity clamp/scale/truncate behavior, including the explicitly divergent
  rejection of nonfinite raw velocity.
- Instrumented onset-frame, pitch, velocity, and confidence pairing without
  reconstructing a frame from `NoteSequence.start_time`.
- Exact `crux.backend-execution-item-id/v1` canonical payload and hash derivation.
- Cross-backend correlation requires the descriptor/item composite key.
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
- Execution-attestation host numeric fingerprint, CPU, memory, PID, and deadline
  fields; every authenticated smoke mismatch is backend-fatal irrespective of
  diagnostic fingerprint, while `environment_unsupported` is limited to pre-inference
  platform/evidence failure.
- A backend-agnostic consumer can read OaF events without importing an OaF module and
  rejects `mapping_status: "not_applied"` through a generic error.
- Atomic publication and preservation of an existing valid artifact on failure.
- Explicit backend selection and proof that missing model state does not select the
  heuristic.
- Heuristic descriptor and parameter-lock coverage, including all resolved Librosa
  defaults, classifier branch ordering, fixed pitch/velocity outputs, and isolation
  from Crux's main dependency environment, plus rejection of official publication
  outside native `linux/amd64`.
- Sanitized errors and logs.

### Protocol and failure tests

A fake runner covers:

- correct handshake and successful output;
- wrong backend, mounted-lock hash, runtime, metadata-schema, and mapping identities;
- missing, duplicate, and extra handshake fields;
- corrupt JSON and unexpected standard-output text;
- malformed, noncanonical, nonfinite, and bit-mismatched binary64 event fields;
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
allowlist are present before Python starts, that `PYTHONCOERCECLOCALE=0` is required
and removed before exact allowlist validation, that startup rejects any mismatch
before TensorFlow import, and that explicit `ConfigProto` thread counts remain `1`.
Container tests also prove that the production and calibration entrypoints are
distinct source-manifest-covered files, reject each other's mounts and protocols,
and authenticate every calibration request/evidence/source/image identity before
TensorFlow import.
Host evidence tests accept only the three enumerated evidence forms and keep an
unattested bare local host diagnostic-only. Per-phase bundle tests decode the
lowercase-hex API record, authenticate all three files, cross-check the completed job
and observation fields, reject phase substitution, and require the owning evidence
and candidate allowlist hashes. Cross-CPU tests require exact smoke identity on a
different fingerprint before official publication and classify every completed
mismatch as backend-fatal `failed`, exit `2`.

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
- perform the two startup raw-oracle checks and three post-ready artifact requests,
  for exactly five fixture inferences across two processes;
- generate the expected native artifact from the locked oracle, mounted locks, and
  descriptor and reproduce it byte-for-byte;
- round-trip the JSONL through the host reader without information loss;
- expose backend-independent prediction records to a scorer-facing consumer without
  importing OaF runner or mapping code; and
- generate optional MIDI whose sidecar references the native artifact hash.

### Command tests

CLI tests prove:

- `prepare-backend` boots from the exact request without a final lock, verifies the
  real four-member archive, publishes only three components, emits only the
  documented setup summary, and never writes a backend report;
- post-seal `prepare-backend` rejects any request/evidence/final-lock disagreement;
- `attest-base-system` accepts only the pinned base manifest, exact native inventory,
  required probes, and empty additional-package set;
- `bootstrap-image` refuses mutable launch references, repository-root contexts,
  context-manifest or host-bundle drift, and base-prefix drift, and publishes only an
  authenticated OCI layout plus bootstrap evidence;
- `measure` requires the exact five-by-three fresh-container matrix and cannot
  publish a seal candidate;
- `calibrate` constructs its own candidate, rejects a missing, unrelated,
  sentinel-bearing, or underprovisioned seal-profile request, and leaves no
  authoritative output on any operational or integrity failure;
- a `preseal` OaF registry entry returns `backend_not_sealed`, exit `1`, without
  opening prediction output, while a `sealed` entry with missing or corrupt locks
  returns integrity failure, exit `2`;
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

> **HPA-481/HPA-482 amendment:** The paragraphs beginning “A manual native Linux
> `amd64` bootstrap workflow accepts one exact commit SHA, checks out that commit,
> and records it in regenerated authenticated native-host evidence.”, “A manual
> native Linux `amd64` measurement workflow accepts that exact measurement-request
> commit, regenerates authenticated native-host evidence, obtains the exact accepted
> OCI archive or rebuilds and exact-compares it, regenerates and exact-compares all
> five fixture rows, runs exactly three fresh-container repetitions per row,
> validates the exact 15 healthy measurement rows, and uploads a canonical artifact
> manifest plus the complete measurement host-attestation bundle and every
> measurement and sanitized diagnostic artifact needed for review.”, and “A
> separate manual candidate phase accepts that exact profile commit, regenerates
> native-host evidence for it, strict-validates the committed evidence/profile
> chain, runs `calibrate`, and uploads the complete candidate, its host-attestation
> bundle, and a canonical artifact manifest for review.” are superseded by the
> signed same-work-job workflows and outside-in acceptance gate in
> [the HPA-481/HPA-482 design](2026-07-30-hpa-481-work-job-artifact-attestation-design.md).
> All unrelated HPA-320 contracts remain in force.

The normal Python suite uses the fake runner and does not install TensorFlow 1 or
require Docker. It validates schemas, identities, host logic, entrypoint separation
with test doubles, matrix and boundary orchestration, failure semantics, and atomic
publication on every pull request.

A manual native Linux `amd64` bootstrap workflow accepts one exact commit SHA,
checks out that commit, and records it in regenerated authenticated native-host
evidence. It then:

1. acquires and verifies the exact checkpoint request;
2. materializes the request-authenticated minimal build context, authenticates the
   pinned base manifest/config and inherited layer prefix, builds the provisional OCI
   archive from the calibration-bootstrap request, authenticates its raw
   manifest/config/layer bytes, and imports it by config digest;
3. attests the pinned base/provisional package inventory, unused UID/GID, empty
   additional-package set, and required native probes; and
4. uploads a canonical artifact manifest plus the complete bootstrap
   host-attestation bundle and every acquisition, bootstrap, base-system, OCI, and
   sanitized diagnostic artifact needed for review.

Acquisition, bootstrap, and base-system evidence are committed together only after
human review of the uploaded bootstrap run. The accepted bootstrap host-attestation
bundle is committed in that same evidence commit and its manifest hash exact-matches
bootstrap evidence. A separate calibration-measurement request then binds the
accepted bootstrap request/evidence hashes in its own commit; the measurement
evidence records the accepted acquisition and base-system evidence hashes it
consumed.

A manual native Linux `amd64` measurement workflow accepts that exact
measurement-request commit, regenerates authenticated native-host evidence, obtains
the exact accepted OCI archive or rebuilds and exact-compares it, regenerates and
exact-compares all five fixture rows, runs exactly three fresh-container repetitions
per row, validates the exact 15 healthy measurement rows, and uploads a canonical
artifact manifest plus the complete measurement host-attestation bundle and every
measurement and sanitized diagnostic artifact needed for review.

Neither bootstrap nor measurement invokes `calibrate` or `seal`. Measurement
evidence and its accepted host-attestation bundle are committed only after human
review and exact hash comparison. The seal-profile request is then reviewed and
committed in a separate commit, so measured facts and the chosen headroom cannot
enter history as one opaque decision.

A separate manual candidate phase accepts that exact profile commit, regenerates
native-host evidence for it, strict-validates the committed evidence/profile chain,
runs `calibrate`, and uploads the complete candidate, its host-attestation bundle,
and a canonical artifact manifest for review. The candidate binds the bundle-manifest
hash and includes all four bundle files in its exact artifact allowlist. No manual
pre-seal phase invokes `seal`, writes final locks, or changes the registry, and no
evidence commit triggers automatic sealing.

Every native workflow upload manifest is canonical UTF-8 JSON consisting of one
array of `artifact_ref` rows, sorted first by `role` and then by repository-relative
POSIX `path`, with lexicographically ordered object keys, no insignificant
whitespace, and one final newline. It includes itself only through the workflow
provider's external artifact digest, never as an array row. The manifest is
operational CI evidence and is not a lock identity input. Each manifest includes the
phase's three host-attestation inputs and bundle manifest as ordinary authenticated
rows; omission of any of the four is fatal.

The sealed native integration workflow is introduced in the same final-seal commit
as the immutable seal dependencies, runtime lock, backend lock, smoke oracle, sealed
registry entry, and workflow configuration. It runs on changes to any request,
evidence, lock, runner or calibration source, vendored upstream source, fixture,
prediction schema, or native workflow; it also runs on a schedule and by manual
dispatch. On a native Linux `amd64` worker it strict-validates the checked-in
authority chain, rederives the candidate payloads in memory, rebuilds the image and
exact-compares the authenticated minimal context, pinned base manifest/config and
inherited layer prefix, and deterministic output manifest/config/layer identities,
reacquires the checkpoint, and reruns the measurement request as fresh operational
evidence. Fresh host, process, and timing fields are expected to differ; the workflow
validates their schemas, identities, exact 15-row shape, health, and compliance with
both bootstrap ceilings and the reviewed final profile instead of byte-comparing
them with the accepted measurement record. It then reruns boundary/process probes,
exact-compares every deterministic sealed artifact, runs the real-checkpoint tests
without skips, and runs the five-call production verification against the final
locks.

This rebuild is a mandatory contemporary regression check for the exact pinned build
recipe, not a promise that the supporting registry, runner, or build toolchain will
remain available forever. A non-identical rebuild fails the workflow and cannot be
substituted for preserved bytes. A passing rebuild proves only that every recorded
deterministic byte was reproduced by an accepted native execution of the supported
recipe at that time.

Every native workflow fails on a network/acquisition failure in a phase that requires
acquisition, unavailable required Docker or cgroup-v2 metrics, an emulated worker, a
missing upload, or a skipped required native test. Measurement and sealed integration
also fail on a non-exact 15-row matrix or an unhealthy row. Candidate and sealed
integration also fail on coverage other than exact 130/78/52, a process/counter
contradiction, or a boundary request not rejected before inference. Any
non-identical deterministic image or sealed byte fails every phase that reproduces
it. The sealed workflow passes only when all three published post-ready smoke
artifacts are byte-identical,
`verify-backend` exits `0` with strict status `verified`, every required native test
ran, and the canonical artifact manifest authenticates every required report and
upload. Exit status alone is insufficient.

The sealed workflow is required before merging an inference-relevant change.
Unrelated host-only changes may reuse a recent successful scheduled result only if
repository branch protection explicitly permits it. The benchmark command itself
remains offline.

## Repository Boundaries

Implementation planning should preserve these ownership boundaries:

- `src/benchmark/backends/` owns the host protocol, descriptors, backend selection,
  and the isolated OaF adapter.
- `src/benchmark/backend_lock.py` owns strict locks and identity derivation.
- `src/benchmark/prediction_artifact.py` owns the common JSONL schema, validation,
  backend-agnostic reading, and atomic publication.
- `runtime/oaf_tf1/` owns the pinned runner, container definition, package lock,
  source manifests, minimal build-context manifest, base-system-package request, and
  protocol implementation.
- `runtime/oaf_tf1/calibration_entrypoint.py` owns only the authenticated
  `crux.oaf-calibration-runner/v1` protocol. The production entrypoint remains a
  separate file and owns only `crux.transcription-runner/v1`.
- `tools/hpa320/oaf_build_context.py` owns only exact build-context-manifest
  generation/validation and fresh normalized staging.
- `tools/hpa320/oaf_host_attestation.py` owns phase-bundle validation and durable raw
  GitHub API byte recovery.
- `tools/hpa320/oaf_oci.py` owns strict parsing of raw OCI JSON and blobs, pinned-base
  prefix validation, and canonical archive packing.
- `tools/hpa320/oaf_native_calibration.py` owns OCI archive selection/import,
  immutable Docker launch, cgroup-v2 monitoring, fixture-matrix execution, boundary
  probes, and in-memory candidate assembly.
- `runtime/heuristic_onset/` owns the content-addressed heuristic environment lock and
  reproducible environment builder.
- `config/benchmark/backends/` owns reviewed checkpoint-acquisition,
  calibration-bootstrap, calibration-measurement, and seal-profile requests;
  backend, runtime, and heuristic-parameter locks; and the native seal-evidence
  record.
- `docs/superpowers/evidence/hpa-320/` owns immutable design-audit and accepted
  native-calibration evidence, including the legacy TF2 conversion coverage report,
  the four reviewed files directly under `native/`, and each phase's complete durable
  host-attestation bundle.
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
   identities in this design. Through the accepted native-`amd64` seal-evidence
   record, the locks cross-reference strict checkpoint, base-system,
   calibration-bootstrap, measurement, and seal-profile request/evidence hashes and
   every exact build-context manifest, pinned-base manifest/config and inherited
   layer prefix, resource, deadline, package, reference host numeric fingerprint, and
   platform value required above.
2. The released checkpoint is verified offline and all 78 required inference tensors
   are restored; the four-member archive and three-component cache are exact; the 52
   non-inference entries are explicitly classified; and the zero-of-78 legacy
   conversion conclusion is preserved in its hashed audit artifact.
3. The production and calibration entrypoints are distinct and reject each other's
   authorities and protocols; each launcher supplies its exact environment before
   CPython starts, and the isolated TensorFlow 1.15.5 runner authenticates every
   mounted identity before TensorFlow import and completes an identity-checked
   handshake with bounded concurrent standard-error draining.
4. The five exact diagnostic fixtures produce 15 healthy fresh-container measurement
   rows; persistent/fresh boundary probes prove the exact input bound and
   pre-inference over-bound rejection; and the smoke fixture reproduces three
   byte-identical, finite, nonempty published prediction artifacts with no numeric
   acceptance tolerance. Every authenticated mismatch is backend-fatal irrespective
   of diagnostic host fingerprint.
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
8. Host tests pass without TensorFlow 1; the manual bootstrap, measurement, and
   candidate workflows publish reviewable evidence but cannot seal; bootstrap
   evidence and its durable host-attestation bundle, its dependent measurement
   request, measurement evidence and its durable host-attestation bundle, and the
   reviewed seal-profile request enter the required separate commits; the candidate
   bundle is published only from the reviewed candidate by `seal`; and the sealed
   native integration workflow runs every real-checkpoint test without skips and
   publishes auditable exact 130/78/52 coverage, boundary, verification, raw OCI, and
   smoke artifacts.
9. Every execution records the Crux Git commit, manifest-defined clean or dirty source
   state, resource limits, and deadlines without making the static backend or runtime
   lock self-referential.
10. The FastAPI transcription path, shared canonical taxonomy, scoring semantics, and
   full-corpus orchestration remain outside the HPA-320 change.
11. A lightweight native-`linux/amd64` `heuristic-onset-v1` adapter and
    content-addressed isolated environment preserve the current heuristic only
    through explicit selection; other platforms are diagnostic-only, and the legacy
    HDF5 path remains combined-score-only and cannot claim a frozen descriptor.
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

The Python 3.7/Bullseye base image may contain known vulnerabilities. HPA-320 accepts
that residual risk only inside the digest-pinned, inventory-attested, non-root,
network-disabled, read-only, capabilities-dropped boundary and records a security
scan/advisory snapshot with the seal evidence. It does not claim that EOL packages
are patched. A vulnerability that invalidates the isolation boundary retires
official execution pending a reviewed new runtime identity.

### Apple Silicon development hosts differ from CI

The only accepted official runner platform is native `linux/amd64`. Local Apple
Silicon execution may explicitly emulate that image for diagnostics rather than
building a separate ARM image, but it reports `environment_unsupported`, cannot seal
locks or publish official predictions, and cannot satisfy the dedicated native
parity check.

### Native amd64 does not guarantee numerical identity

The seal preserves the reference host numeric fingerprint, and every official
execution records its current diagnostic fingerprint. A different native-amd64 CPU
is accepted only after reproducing the sealed smoke artifact byte-for-byte. Every
completed mismatch is a backend regression with `failed`, exit `2`; a fingerprint
difference may explain the failure but cannot reclassify it as unsupported.
Supporting an execution environment that produces different bytes requires a
reviewed new runtime/backend identity. New evidence alone cannot waive the exact-match
rule.

### Whole-file inference has bounded resource use

The backend lock has no implicit or unlimited input length: it freezes
`max_input_audio_frames`, and host and runner boundary tests enforce it. CPU, memory,
PID, tmpfs, and deadline limits are recorded in the execution attestation. Direct
verification/transcription uses the sealed reference profile; HPA-326 freezes its
profile in the run configuration. A supported input that exhausts those resources
fails loudly as a backend-fatal condition; a longer-input or chunked policy requires
a new locked identity. That identity change is conservative domain versioning:
artifacts for identical inputs below the prior bound remain scientifically comparable
when their remaining descriptor fields agree.

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
registries or artifact mirrors may supply the same bytes without changing backend
identity. A new byte sequence, even from the same URL or tag, is rejected.
Preservation of the exact OCI and checkpoint bytes avoids making a future registry or
upstream-storage outage an immediate rebuild dependency and is the authoritative
long-term availability path. HPA-320 does not promise that the exact image can be
reproducibly rebuilt forever from surviving source and package references. If every
approved preserved copy is lost, a rebuild may restore availability only when an
accepted native execution reproduces every recorded index, manifest, config, layer,
DiffID, canonical archive byte, and digest and passes the complete sealed workflow.
A version-equivalent or semantically equivalent image is never recovery. If exact
reproduction is no longer possible, that runtime identity is operationally
unrecoverable and must be retired.

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
