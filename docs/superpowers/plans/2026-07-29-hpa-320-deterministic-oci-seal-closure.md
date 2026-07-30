# HPA-320 Deterministic OCI Seal Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task by task. Use the
> checkboxes as the execution record, stop at every native-evidence review gate, and
> apply `superpowers:verification-before-completion` before claiming a task complete.

**Goal:** Close the remaining HPA-320 gap by making the provisional OaF image
byte-reproducible, authenticating its calibration-only runtime, threading that
bootstrap identity through the existing seal chain, and publishing the final frozen
backend only after accepted native `linux/amd64` evidence.

**Architecture:** Preserve the implementation already landed through commit
`78b2424`. Add a fifth checked-in request authority for image bootstrap, a
source-addressed calibration entrypoint separate from the production entrypoint, and
a deterministic OCI-layout packer. A request-hashed manifest materializes the only
files BuildKit may receive into a fresh normalized context. `bootstrap-image`
authenticates the pinned base manifest/config and inherited layer prefix, then builds
twice with one exact BuildKit/Buildx/frontend recipe and publishes evidence only when
every raw OCI identity and canonical archive byte matches. Bootstrap, measurement,
and candidate phases each preserve a complete host-attestation bundle. Measurement
and calibration consume authenticated bootstrap evidence rather than a
caller-supplied image name. Final locks remain acyclic and are published only from a
reviewed candidate on native `linux/amd64`.

**Tech Stack:** Python 3.12 host tools, Python 3.7.17 isolated runner,
TensorFlow 1.15.5 CPU, Docker Buildx `v0.35.0`, BuildKit `v0.31.2`, Dockerfile
frontend `1.25.0`, OCI image layout, uncompressed POSIX ustar, SHA-256, canonical
JSON/JSONL, Click/argparse, Pytest, Ruff, Black, and Pylint.

## Current-State Boundary

- The historical
  `docs/superpowers/plans/2026-07-28-hpa-320-phase-a-seal-closure.md` implemented
  Tasks 1-7 through `78b2424`. Do not redo those tasks.
- The normative design is
  `docs/superpowers/specs/2026-07-26-hpa-320-freeze-oaf-drums-backend-design.md`
  at or after commit `3dcb0e3`.
- Existing checkpoint acquisition, base-system validation, deterministic environment,
  exact binary64 events, five-call verification, and accepted host-observation tests
  are source truth and must remain passing.
- This plan begins with host-testable code and schemas. It does not treat the local
  Apple Silicon host, QEMU, Rosetta, a Docker Desktop VM, or copied historic output
  as accepted native evidence.
- Final seal values, measurement rows, smoke output, OCI identities, profile
  headroom, and lock hashes must come from the reviewed native workflow. No test
  fixture or local diagnostic result may be promoted into final authority.

## Frozen Build Recipe

The calibration-bootstrap request must contain these exact immutable tool identities:

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

The request also fixes:

```json
{
  "annotations": [],
  "compression": "gzip",
  "compression_level": 6,
  "exporter": "oci",
  "exporter_tar": false,
  "force_compression": false,
  "inline_cache": false,
  "multi_platform_deterministic": true,
  "oci_media_types": true,
  "platform": "linux/amd64",
  "provenance": false,
  "rewrite_timestamp": true,
  "sbom": false,
  "source_date_epoch": 0
}
```

The nested `oci_archive` object must be exactly:

```json
{
  "compression": "none",
  "final_zero_blocks": 2,
  "format": "posix-ustar",
  "gid": 0,
  "gname": "",
  "member_mode": 420,
  "member_types": "regular-files-only",
  "mtime": 0,
  "path_order": "utf8-byte",
  "uid": 0,
  "uname": ""
}
```

The Dockerfile's first line must be:

```dockerfile
# syntax=docker/dockerfile-upstream@sha256:3d6d54b33351b396a910d33248754b86b1d7dd838b4eeb9575d8903a209f6516
```

## Cross-Task Interfaces

Implement the following immutable host-side types. They are the single parsed
representation used by the CLI, OCI inspector, validators, and tests:

```python
@dataclass(frozen=True)
class OciArchiveRecipe:
    compression: str
    final_zero_blocks: int
    format: str
    gid: int
    gname: str
    member_mode: int
    member_types: str
    mtime: int
    path_order: str
    uid: int
    uname: str


@dataclass(frozen=True)
class ImageBuildRecipe:
    annotations: tuple[str, ...]
    buildkit_image: str
    buildkit_version: str
    buildx_binary_sha256: str
    buildx_binary_size: int
    buildx_binary_url: str
    buildx_version: str
    compression: str
    compression_level: int
    dockerfile_frontend: str
    dockerfile_frontend_version: str
    exporter: str
    exporter_tar: bool
    force_compression: bool
    inline_cache: bool
    multi_platform_deterministic: bool
    oci_archive: OciArchiveRecipe
    oci_media_types: bool
    platform: str
    provenance: bool
    rewrite_timestamp: bool
    sbom: bool
    source_date_epoch: int


@dataclass(frozen=True)
class CalibrationBootstrapRequest:
    backend_id: str
    build_context_manifest_sha256: str
    image_build: ImageBuildRecipe
    runtime_uid: int
    runtime_gid: int
    sha256: str
    payload: Mapping[str, JsonValue]


@dataclass(frozen=True)
class SourceFile:
    byte_length: int
    path: str
    sha256: str


@dataclass(frozen=True)
class BuildContextManifest:
    directory_mode: int
    file_mode: int
    files: tuple[SourceFile, ...]
    manifest_path: str
    sha256: str


@dataclass(frozen=True)
class BaseImageIdentity:
    manifest_digest: str
    config_digest: str
    layer_digests: tuple[str, ...]
    layer_diff_ids: tuple[str, ...]


@dataclass(frozen=True)
class NativeHostAttestationBundle:
    api_record: CheckpointIdentity
    native_host_evidence: CheckpointIdentity
    native_host_observation: CheckpointIdentity
    phase: str
    sha256: str


@dataclass(frozen=True)
class OciLayoutIdentity:
    archive: CheckpointIdentity
    base_image_config_digest: str
    base_image_layer_digests: tuple[str, ...]
    base_image_layer_diff_ids: tuple[str, ...]
    index_digest: str
    image_manifest_digest: str
    config_digest: str
    layer_digests: tuple[str, ...]
    layer_diff_ids: tuple[str, ...]
```

Expose these exact functions:

```python
def load_calibration_bootstrap_request(path: Path) -> CalibrationBootstrapRequest: ...

def load_build_context_manifest(path: Path) -> BuildContextManifest: ...

def generate_build_context_manifest(
    *,
    repository_root: Path,
    wheelhouse_root: Path,
) -> bytes: ...

def materialize_build_context(
    *,
    repository_root: Path,
    wheelhouse_root: Path,
    manifest_path: Path,
    destination: Path,
) -> Path: ...

def authenticate_base_image(
    *,
    manifest_bytes: bytes,
    config_bytes: bytes,
    expected_manifest_digest: str,
) -> BaseImageIdentity: ...

def load_native_host_attestation_bundle(
    path: Path,
    *,
    expected_phase: str,
) -> NativeHostAttestationBundle: ...

def inspect_oci_layout(
    directory: Path,
    archive: Path,
    recipe: ImageBuildRecipe,
    base_image: BaseImageIdentity,
) -> OciLayoutIdentity: ...

def canonical_pack_oci_layout(
    directory: Path,
    output: Path,
    recipe: OciArchiveRecipe,
) -> CheckpointIdentity: ...

def require_identical_oci_builds(
    first: OciLayoutIdentity,
    second: OciLayoutIdentity,
    first_archive: Path,
    second_archive: Path,
) -> OciLayoutIdentity: ...
```

`materialize_build_context` creates a fresh directory containing exactly the
manifest-enumerated regular files plus the manifest itself. Logical paths beneath
`runtime/oaf_tf1/wheelhouse/` come from the verified `wheelhouse_root`; all others
come from `repository_root`. It normalizes directories to `0755` and files to `0644`,
sets every staged mtime to epoch zero, does not consult any ignore file, and rejects
links, special files, missing or extra entries, and size/hash drift.

`generate_build_context_manifest` is also exposed through
`python -m tools.hpa320.oaf_build_context generate`. It enumerates only the reviewed
Dockerfile-input paths and roots defined in that source-manifest-covered module,
normalizes logical POSIX paths, and returns canonical bytes without writing through
symlinks. The checked-in manifest is generated only after the verified wheelhouse and
runner source manifest are final.

`authenticate_base_image` strict-parses but does not canonicalize the exact raw
registry manifest/config bytes. It verifies the requested platform-manifest digest,
the config descriptor digest/size, and the ordered layer descriptors, then obtains
the ordered DiffIDs from the authenticated config. `inspect_oci_layout` requires the
output layer-digest and DiffID arrays to begin with that exact base prefix. Only
members of layers after the inherited prefix must have epoch-zero mtime.

`load_native_host_attestation_bundle` accepts only
`crux.oaf-native-host-attestation-bundle/v1` and the expected phase. It authenticates
the three exact sibling basenames and sizes, decodes the lowercase-hex API record,
and proves that its raw-byte hash, completed GitHub job, commit, URLs, labels,
phase-specific checked-in workflow reference, observation, nested native-host
evidence, and numeric fingerprint agree. Producer commands accept both the bundle and
its named native-host-evidence file, then require those bytes to be identical; the
owning evidence records the canonical bundle manifest's SHA-256.

`canonical_pack_oci_layout` runs before `inspect_oci_layout`; the inspector verifies
that the supplied archive is the canonical serialization of the directory before
returning its archive identity. `require_identical_oci_builds` exact-compares index
JSON bytes, selected manifest JSON bytes, config JSON bytes, ordered compressed layer
blobs, ordered uncompressed layer DiffIDs, canonical archive bytes, and archive
SHA-256. OCI JSON receives duplicate-key/type/digest/size validation directly from
the BuildKit-emitted bytes; it is never required to have Crux canonical formatting
and is never reserialized for hashing or comparison. The function reports the first
identity class that differs and publishes nothing.

## Execution Order

- Tasks 1-5 are host-testable and must land before any native workflow is run.
- Task 6 contains separate manual native bootstrap and measurement phases. Bootstrap
  evidence must be reviewed and committed before the measurement request can bind
  it; measurement evidence is reviewed and committed later. Neither phase may run
  `calibrate`, `seal`, or modify the registry.
- Task 7 starts only after accepted measurement evidence is committed. The
  seal-profile request is a separate human-reviewed commit.
- Task 8 starts only after the candidate artifact is reviewed. It is the sole task
  allowed to publish final locks, seal evidence, smoke oracle, OCI manifest, and the
  `sealed` registry entry.
- If native evidence is missing, emulated, incomplete, unhealthy, or does not
  reproduce exact image bytes, stop. Do not widen ceilings, add packages, change the
  toolchain, or derive profile values automatically.

---

### Task 1: Make exceptional wheel hazards independently reproducible

**Files:**

- Modify: `tools/hpa320/resolve_oaf_runtime.py`
- Modify: `tests/benchmark/test_oaf_smoke_generator.py`

**Interfaces:**

Keep the existing public signatures:

```python
def require_reproducible_wheels(first: Path, second: Path) -> tuple[int, str]: ...

def validate_built_pure_wheel(
    path: Path,
    spec: SdistExceptionSpec | None = None,
) -> DistributionRecord: ...
```

Add an internal immutable fingerprint:

```python
@dataclass(frozen=True)
class _WheelMemberFingerprint:
    name: str
    timestamp: tuple[int, int, int, int, int, int]
    compression_method: int
    external_attributes: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
```

- [ ] Add failing tests that create otherwise valid pure wheels containing a `.pyc`
      member and an exact stable build-root byte string in an uncompressed member.
      Assert each hazard has its own `DistributionBuildError`.
- [ ] Add a failing test where two wheels have equal extracted file bytes but
      different ordered ZIP metadata. Assert `require_reproducible_wheels` rejects
      them before the whole-wheel mismatch becomes the only diagnostic.
- [ ] Add failing tests for duplicate, reordered, unhashed, incorrectly hashed, and
      extra `RECORD` rows.
- [ ] Inspect members in archive order. Reject duplicate or noncanonical paths,
      directory ambiguity, `.pyc`, native/executable members, and the stable absolute
      source/build roots used by `_run_exceptional_build`.
- [ ] Compare the ordered `_WheelMemberFingerprint` tuples and ordered canonical
      `RECORD` rows before comparing complete wheel bytes.
- [ ] Keep the final byte-for-byte comparison and returned `(byte_length, sha256)`
      unchanged.

Run the new tests first:

```bash
rtk uv run pytest \
  tests/benchmark/test_oaf_smoke_generator.py \
  -k 'exceptional_distribution or reproducible_wheels'
```

Expected before implementation: FAIL because `.pyc`, build-root leakage, and ZIP
metadata drift are not independently rejected.

Run after implementation:

```bash
rtk uv run pytest tests/benchmark/test_oaf_smoke_generator.py
rtk uv run ruff check tools/hpa320/resolve_oaf_runtime.py \
  tests/benchmark/test_oaf_smoke_generator.py
rtk uv run black --check tools/hpa320/resolve_oaf_runtime.py \
  tests/benchmark/test_oaf_smoke_generator.py
```

- [ ] Commit:

```bash
rtk git add tools/hpa320/resolve_oaf_runtime.py \
  tests/benchmark/test_oaf_smoke_generator.py
rtk git commit -m "fix: diagnose exceptional wheel reproducibility"
```

### Task 2: Add the strict bootstrap authority and deterministic OCI primitives

**Files:**

- Create: `tools/hpa320/oaf_build_context.py`
- Create: `tests/benchmark/test_oaf_build_context.py`
- Create: `tools/hpa320/oaf_oci.py`
- Create: `tests/benchmark/test_oaf_oci.py`
- Modify: `tools/hpa320/seal_oaf_backend.py`
- Modify: `runtime/oaf_tf1/Dockerfile`
- Modify: `tests/benchmark/test_oaf_seal.py`

**Authority contract:**

`load_calibration_bootstrap_request` must require schema
`crux.oaf-calibration-bootstrap-request/v1` and exactly these top-level keys:

```text
backend_id, base_image_manifest_digest, base_system_package_request_sha256,
build_context_manifest_sha256, checkpoint_acquisition_request_sha256,
container_restrictions,
distribution_build_manifest_sha256, environment, image_build,
instrumentation_patch_sha256, python_coerce_c_locale, resource_ceiling,
runner_source_manifest_sha256, runtime_gid, runtime_uid, schema,
upstream_source_manifest_sha256
```

The request uses `runtime_uid: 65532`, `runtime_gid: 65532`, the design's exact
seven-variable deterministic environment, `PYTHONCOERCECLOCALE=0`, and these
diagnostic ceilings:

```json
{
  "cpu_limit_millis": 2000,
  "memory_limit_bytes": 4294967296,
  "monitor_interval_millis": 10,
  "pid_limit": 256,
  "request_deadline_seconds": 1800,
  "shm_bytes": 1073741824,
  "startup_deadline_seconds": 300,
  "stderr_max_line_bytes": 65536,
  "stderr_read_chunk_bytes": 65536,
  "stderr_ring_buffer_bytes": 1048576,
  "stdout_max_line_bytes": 134217728,
  "tmp_bytes": 1073741824
}
```

- [ ] Add strict-loader tests using synthetic canonical request bytes for exact key
      sets, exact tool
      identities, exact booleans, exact platform, exact archive recipe, and
      cross-hashes of every referenced current file.
- [ ] Add a strict loader and materializer for
      `crux.oaf-build-context-manifest/v1`. Require exactly
      `directory_mode`, `file_mode`, `files`, `manifest_path`, and `schema`; require
      decimal modes `493`/`420`; require exact manifest path
      `runtime/oaf_tf1/build-context-manifest.json`; and require a unique UTF-8
      byte-sorted `source_file` array that excludes the manifest itself.
- [ ] Add materializer tests proving the fresh context contains only enumerated files
      plus the manifest, every directory/file mode is normalized to `0755`/`0644`,
      every mtime is epoch zero, `.dockerignore` and Dockerfile-specific ignore files
      are absent and never
      consulted, unlisted checkout files cannot enter, and source mode-only changes
      do not change staged bytes. Reject symlinks, multiply linked files, special
      files, path aliases/escapes, missing inputs, extra staged entries, and
      size/hash drift.
- [ ] Add a test asserting the Dockerfile's first line is the exact digest-pinned
      syntax directive and that no second syntax directive exists.
- [ ] Implement `oaf_oci.py` without importing Docker, TensorFlow, or project runtime
      adapters. All input paths must be stable regular files below the OCI directory;
      reject symlinks, hard links, devices, sockets, path aliases, missing blobs,
      extra blobs, digest drift, size drift, multiple matching platform manifests,
      and mutable image references.
- [ ] Strict-parse the exact raw base-manifest and config bytes obtained from the
      pinned registry digest. Reject duplicate keys, wrong types, wrong
      digest/descriptor size, platform drift, and unordered or inconsistent layer
      arrays. Record its config digest, ordered layer digests, and ordered DiffIDs.
- [ ] Select exactly one `linux/amd64` descriptor from `index.json`. Strict-parse each
      BuildKit-emitted OCI JSON document from its original bytes, reject duplicate
      keys and type/descriptor drift, and hash/compare those raw bytes. Never require
      Crux canonical-JSON formatting and never reserialize OCI JSON for identity.
- [ ] Decompress every gzip layer, hash the exact uncompressed tar bytes, and verify
      ordered DiffIDs. Require the output layer-digest and DiffID arrays to begin
      with the exact authenticated base prefix. Exempt only that exact unchanged
      prefix from epoch normalization; reject nonzero member mtime in every later
      generated or rewritten layer.
- [ ] Implement the canonical uncompressed POSIX ustar packer. Include only regular
      files in UTF-8 byte-sorted relative-path order, use mode `0644`, UID/GID `0`,
      empty owner/group names, mtime `0`, no PAX headers, no explicit directory
      members, and exactly two terminal zero blocks.
- [ ] Add deterministic packer tests with two differently created source directories.
      Assert byte-identical archives and exact rejection of symlinks, unexpected
      files, a base-prefix mismatch, non-epoch generated-layer entries, path overflow,
      bad digests, and a third terminal zero block. Explicitly prove a non-epoch member
      in an exact inherited base layer is accepted and the same member in a later
      layer is rejected.
- [ ] Keep this task's request objects in test fixtures. The checked-in request is
      created in Task 5 only after every source-manifest-covered file is final.

Run the tests before implementation:

```bash
rtk uv run pytest tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_oaf_oci.py \
  tests/benchmark/test_oaf_seal.py -k 'build_context or bootstrap or oci'
```

Expected: collection or assertion failure because the context/OCI modules, request,
and strict loaders do not exist.

Run after implementation:

```bash
rtk uv run pytest tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_oaf_oci.py \
  tests/benchmark/test_oaf_seal.py -k 'build_context or bootstrap or oci'
rtk uv run ruff check tools/hpa320/oaf_build_context.py \
  tools/hpa320/oaf_oci.py tools/hpa320/seal_oaf_backend.py \
  tests/benchmark/test_oaf_build_context.py tests/benchmark/test_oaf_oci.py \
  tests/benchmark/test_oaf_seal.py
rtk uv run black --check tools/hpa320/oaf_build_context.py \
  tools/hpa320/oaf_oci.py tools/hpa320/seal_oaf_backend.py \
  tests/benchmark/test_oaf_build_context.py tests/benchmark/test_oaf_oci.py \
  tests/benchmark/test_oaf_seal.py
```

- [ ] Commit:

```bash
rtk git add \
  runtime/oaf_tf1/Dockerfile tools/hpa320/oaf_build_context.py \
  tools/hpa320/oaf_oci.py tools/hpa320/seal_oaf_backend.py \
  tests/benchmark/test_oaf_build_context.py tests/benchmark/test_oaf_oci.py \
  tests/benchmark/test_oaf_seal.py
rtk git commit -m "feat: freeze OaF calibration image recipe"
```

### Task 3: Build twice and isolate the calibration protocol

**Files:**

- Create: `tools/hpa320/oaf_native_calibration.py`
- Create: `runtime/oaf_tf1/calibration_entrypoint.py`
- Create: `runtime/oaf_tf1/calibration_protocol.py`
- Create: `runtime/oaf_tf1/tests/test_calibration_protocol.py`
- Modify: `runtime/oaf_tf1/Dockerfile`
- Modify: `runtime/oaf_tf1/entrypoint.py`
- Modify: `tools/hpa320/oaf_build_context.py`
- Create: `tools/hpa320/oaf_host_attestation.py`
- Modify: `tools/hpa320/seal_oaf_backend.py`
- Modify: `tests/benchmark/test_oaf_build_context.py`
- Create: `tests/benchmark/test_oaf_host_attestation.py`
- Modify: `tests/benchmark/test_oaf_oci.py`
- Modify: `tests/benchmark/test_oaf_seal.py`

**Host API:**

```python
def bootstrap_image(
    *,
    request_path: Path,
    host_attestation_bundle_path: Path,
    host_evidence_path: Path,
    output: Path,
    repository_root: Path,
) -> PublishedArtifact: ...


def import_authenticated_oci_archive(
    archive_path: Path,
    expected: OciLayoutIdentity,
    docker_executable: str = "docker",
) -> str:
    """Return the verified sha256 config digest used as the local image locator."""
```

**CLI:**

```text
seal_oaf_backend.py bootstrap-image
  --request PATH
  --host-attestation-bundle PATH
  --host-evidence PATH
  --output DIRECTORY
  --repository-root DIRECTORY
```

The output directory contains exactly:

```text
calibration-bootstrap-evidence.json
oci-layout-manifest.json
runtime.oci.tar
```

- [ ] Add failing CLI and API tests that prove `bootstrap-image` rejects a non-native
      host, mutable tool references, a wrong Buildx byte length/hash, observed
      Buildx/BuildKit/frontend version drift, pre-existing different output, any
      build mismatch, an OCI import whose inspected config digest differs, and a
      missing, wrong-phase, corrupt, or host-evidence-mismatched attestation bundle.
- [ ] Implement the strict host-bundle loader independently of GitHub artifact
      retention. Tests must preserve exact raw API bytes via canonical lowercase
      hexadecimal, reject an odd digit, uppercase, whitespace other than the one final
      LF, wrong size/hash/basename, incomplete or unsuccessful jobs, observation/API
      disagreement, nested-evidence disagreement, phase/workflow-ref disagreement,
      and wrong phase.
- [ ] Download the Buildx binary to a fresh staging directory, read it without
      following links, verify byte length `65265826` and SHA-256
      `d41ece72044243b4f58b343441ae37446d9c29a7d6b5e11c61847bbcf8f7dfda`,
      mark only those verified staged bytes executable, and execute those bytes.
- [ ] Create two fresh Buildx builders using the exact BuildKit platform manifest.
      Give each an empty state directory and distinct OCI output directory. Disable
      cache import/export and pass no host-derived build arguments.
- [ ] Fetch the raw pinned base platform-manifest and config bytes through the OCI
      Distribution API, verify every descriptor against the requested immutable
      digest, and derive one `BaseImageIdentity`. Do not trust a mutable registry tag,
      Docker daemon inspection, or an output-image prefix as the base authority.
- [ ] Materialize two independent fresh build contexts from the exact
      request-authenticated manifest. Run Buildx with each staged directory as
      `cwd` and `.` as the context argument. Never pass the repository checkout,
      ignored wheelhouse parent, or any host directory directly to BuildKit.
- [ ] Invoke both builds with `SOURCE_DATE_EPOCH=0`, `linux/amd64`, OCI media types,
      gzip level `6`, `force-compression=false`, provenance/SBOM disabled,
      multi-platform deterministic output, and timestamp rewrite. Do not use the
      Docker daemon exporter or a tag as evidence.
- [ ] Construct the build command from typed request fields, with no passthrough
      flags or environment-derived defaults. Apart from the distinct builder/output
      arguments, its exact argument shape is:

```python
def _buildx_build_command(
    buildx: Path,
    builder: str,
    output_directory: Path,
) -> tuple[str, ...]:
    return (
        os.fspath(buildx),
        "build",
        "--builder",
        builder,
        "--file",
        "runtime/oaf_tf1/Dockerfile",
        "--platform",
        "linux/amd64",
        "--pull",
        "--no-cache",
        "--provenance=false",
        "--sbom=false",
        "--build-arg",
        "BUILDKIT_MULTI_PLATFORM=1",
        "--build-arg",
        "SOURCE_DATE_EPOCH=0",
        "--build-arg",
        "RUNTIME_UID=65532",
        "--build-arg",
        "RUNTIME_GID=65532",
        "--annotation",
        "index:org.opencontainers.image.created=1970-01-01T00:00:00Z",
        "--output",
        (
            "type=oci,tar=false,oci-mediatypes=true,compression=gzip,"
            "compression-level=6,force-compression=false,"
            f"rewrite-timestamp=true,dest={output_directory}"
        ),
        ".",
    )
```

The caller executes this tuple with `cwd` set to the freshly materialized context;
the relative Dockerfile path and final `.` therefore resolve only inside that
directory. A test must fail if the checkout root is supplied as either `cwd` or the
context argument.

- [ ] Observe and exact-compare the executing Buildx, BuildKit, and Dockerfile
      frontend versions/references with the request before accepting output.
- [ ] Canonical-pack both layouts, call `require_identical_oci_builds`, publish only
      the first identical archive plus its strict manifest/evidence, and import that
      authenticated archive. Require both output identities to contain the exact base
      config/layer/DiffID prefix, and epoch-zero mtimes only in subsequent layers.
      Re-inspect the imported image and return only the expected config digest.
- [ ] Record `build_context_manifest_sha256`, `base_image_config_digest`, ordered
      `base_image_layer_digests`, and ordered `base_image_layer_diff_ids` in both the
      bootstrap evidence and OCI-layout manifest. Exact-compare all repeated values
      with the bootstrap request, authenticated base bytes, and output identity.
- [ ] Implement `crux.oaf-calibration-runner/v1` in the two new runtime files. The
      request has exactly `audio_frame_count`, `audio_path`, `audio_sha256`,
      `max_input_audio_frames`, `request_id`, and `type`. Accept only `measure` and
      `calibration_probe`.
- [ ] Before importing TensorFlow, the calibration entrypoint must authenticate the
      bootstrap request, checkpoint acquisition evidence/cache, base-system evidence,
      source manifests, process environment, runtime config digest, and absence of
      final locks or production requests.
- [ ] Before importing TensorFlow, the production entrypoint must reject calibration
      request/evidence mounts and `crux.oaf-calibration-runner/v1`. There is no shared
      mode flag.
- [ ] Add runtime tests that install an import guard for TensorFlow and prove all
      cross-mode, unknown-key, wrong-hash, wrong-environment, and over-bound failures
      occur before that import.

Run before implementation:

```bash
rtk uv run pytest tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_oaf_host_attestation.py \
  tests/benchmark/test_oaf_oci.py \
  tests/benchmark/test_oaf_seal.py \
  runtime/oaf_tf1/tests/test_calibration_protocol.py
```

Expected: collection failure for the new calibration modules, followed by failing
bootstrap orchestration assertions.

Run after implementation:

```bash
rtk uv run pytest tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_oaf_host_attestation.py \
  tests/benchmark/test_oaf_oci.py \
  tests/benchmark/test_oaf_seal.py \
  runtime/oaf_tf1/tests/test_calibration_protocol.py \
  runtime/oaf_tf1/tests/test_protocol.py
rtk uv run ruff check tools/hpa320/oaf_build_context.py \
  tools/hpa320/oaf_host_attestation.py tools/hpa320/oaf_native_calibration.py \
  runtime/oaf_tf1/calibration_entrypoint.py \
  runtime/oaf_tf1/calibration_protocol.py \
  tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_oaf_host_attestation.py tests/benchmark/test_oaf_oci.py
rtk uv run black --check tools/hpa320/oaf_build_context.py \
  tools/hpa320/oaf_host_attestation.py tools/hpa320/oaf_native_calibration.py \
  runtime/oaf_tf1/calibration_entrypoint.py \
  runtime/oaf_tf1/calibration_protocol.py \
  tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_oaf_host_attestation.py tests/benchmark/test_oaf_oci.py
```

- [ ] Commit:

```bash
rtk git add tools/hpa320/oaf_native_calibration.py \
  tools/hpa320/oaf_build_context.py tools/hpa320/oaf_host_attestation.py \
  tools/hpa320/seal_oaf_backend.py \
  runtime/oaf_tf1/Dockerfile \
  runtime/oaf_tf1/entrypoint.py runtime/oaf_tf1/calibration_entrypoint.py \
  runtime/oaf_tf1/calibration_protocol.py \
  runtime/oaf_tf1/tests/test_calibration_protocol.py \
  runtime/oaf_tf1/tests/test_protocol.py \
  tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_oaf_host_attestation.py tests/benchmark/test_oaf_oci.py \
  tests/benchmark/test_oaf_seal.py
rtk git commit -m "feat: authenticate OaF calibration bootstrap"
```

### Task 4: Thread bootstrap identity through measurement, candidate, and final locks

**Files:**

- Modify: `tools/hpa320/oaf_host_attestation.py`
- Modify: `tools/hpa320/seal_oaf_backend.py`
- Modify: `tools/hpa320/oaf_native_calibration.py`
- Modify: `src/benchmark/backend_lock.py`
- Modify: `runtime/oaf_tf1/oaf_backend.py`
- Modify: `tests/benchmark/test_backend_lock.py`
- Modify: `tests/benchmark/test_oaf_backend.py`
- Modify: `tests/benchmark/test_oaf_host_attestation.py`
- Modify: `tests/benchmark/test_oaf_seal.py`

**Producer API changes:**

```python
def attest_base_system(
    *,
    request_path: Path,
    bootstrap_request_path: Path,
    bootstrap_evidence_path: Path,
    host_attestation_bundle_path: Path,
    host_evidence_path: Path,
    output_path: Path,
) -> PublishedArtifact: ...


def measure(
    *,
    request_path: Path,
    bootstrap_request_path: Path,
    bootstrap_evidence_path: Path,
    host_attestation_bundle_path: Path,
    host_evidence_path: Path,
    model_cache: Path,
    checkpoint_evidence_path: Path,
    base_system_evidence_path: Path,
    output_path: Path,
) -> PublishedArtifact: ...


def calibrate(
    *,
    request_path: Path,
    measurement_evidence_path: Path,
    bootstrap_request_path: Path,
    bootstrap_evidence_path: Path,
    host_attestation_bundle_path: Path,
    host_evidence: Path,
    model_cache: Path,
    checkpoint_evidence_path: Path,
    base_system_evidence_path: Path,
    output: Path,
) -> PublishedSealCandidate: ...
```

Remove the free-form `image` arguments from base attestation, measurement, and
calibration, and remove the externally assembled `candidate_authority` argument.
Resolve the already imported image only through the bootstrap evidence's
authenticated config digest.

- [ ] Add failing tests showing measurement and calibration reject a bootstrap
      request/evidence hash mismatch, request/evidence `image_build` drift, manifest
      ownership drift, config-digest import drift, wrong layer order, wrong DiffID
      order, build-context drift, base-prefix drift, host-bundle drift, and a
      caller-supplied mutable image reference.
- [ ] Add the same bootstrap/config ownership checks to `attest_base_system`; its CLI
      replaces `--image` with required `--bootstrap-request` and
      `--bootstrap-evidence` paths and authenticates the same bootstrap-phase host
      bundle used by bootstrap evidence.
- [ ] Update the measurement request strict schema to add
      `calibration_bootstrap_request_sha256` and
      `calibration_bootstrap_evidence_sha256`. Update measurement evidence to carry
      both accepted bootstrap identities plus the runtime image config and manifest
      digests exactly as specified by the design.
- [ ] Update the profile request to bind both bootstrap hashes. Require its UID/GID
      to equal `65532:65532`, its limits to stay within the bootstrap ceilings, and
      every final numeric bound to be strictly above the accepted measured maximum
      where the design requires headroom.
- [ ] Update the candidate manifest to carry the bootstrap evidence hash and the
      complete exact artifact allowlist, including all four candidate-phase
      host-attestation files. Add `native_host_attestation_bundle_sha256` and
      exact-cross-check it with candidate seal evidence and the bundle manifest.
      Candidate payload hashes address bytes before publication; final payloads must
      not reference the candidate hash.
- [ ] Update runtime lock validation with
      `calibration_bootstrap_request_sha256`,
      `calibration_bootstrap_evidence_sha256`, `build_context_manifest_sha256`,
      `base_image_config_digest`, ordered `base_image_layer_digests`, ordered
      `base_image_layer_diff_ids`, and the exact `image_build` object.
- [ ] Update seal evidence validation with both bootstrap hashes,
      `build_context_manifest_sha256`, the complete base identity,
      `runtime_image_index_digest`, ordered `runtime_image_layer_digests`, ordered
      `runtime_image_layer_diff_ids`, the canonical OCI archive identity, its OCI
      layout manifest hash, and `native_host_attestation_bundle_sha256`.
- [ ] Update bootstrap evidence with build-context/base identities and its
      `native_host_attestation_bundle_sha256`; update measurement evidence with its
      own bundle hash. Validate each bundle's exact phase and prohibit reusing a
      bootstrap or measurement bundle as candidate authority.
- [ ] Update OCI evidence validation to require exactly `archive`, `index_digest`,
      `image_manifest_digest`, `config_digest`, `layer_digests`, `layer_diff_ids`,
      `base_image_config_digest`, `base_image_layer_digests`,
      `base_image_layer_diff_ids`, and `schema`.
- [ ] Cross-check every repeated authority value in both
      `src/benchmark/backend_lock.py` and `runtime/oaf_tf1/oaf_backend.py`; neither
      side may merely type-check fields that the other side authenticates.
- [ ] Extend host-bundle tests across producers: each owning evidence record must bind
      the exact phase bundle, and bootstrap/measurement/candidate bundles are not
      interchangeable even when their nested host evidence is otherwise valid.
- [ ] Update CLI parsing to match the normative commands. `seal` accepts only
      `--candidate` and `--repository-root`; the candidate's allowlist determines
      every publication path.

Run the focused failing tests:

```bash
rtk uv run pytest tests/benchmark/test_backend_lock.py \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_oaf_host_attestation.py \
  tests/benchmark/test_oaf_seal.py
```

Expected: FAIL because current schemas omit bootstrap/context/base/bundle identities
and current producers accept `--image`/`--candidate-authority`.

Run after implementation:

```bash
rtk uv run pytest tests/benchmark/test_backend_lock.py \
  tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_oaf_host_attestation.py \
  tests/benchmark/test_oaf_seal.py \
  runtime/oaf_tf1/tests/test_calibration_protocol.py
rtk uv run ruff check src/benchmark/backend_lock.py \
  tools/hpa320/seal_oaf_backend.py tools/hpa320/oaf_native_calibration.py \
  tools/hpa320/oaf_host_attestation.py runtime/oaf_tf1/oaf_backend.py
rtk uv run black --check src/benchmark/backend_lock.py \
  tools/hpa320/seal_oaf_backend.py tools/hpa320/oaf_native_calibration.py \
  tools/hpa320/oaf_host_attestation.py runtime/oaf_tf1/oaf_backend.py
```

- [ ] Commit:

```bash
rtk git add src/benchmark/backend_lock.py runtime/oaf_tf1/oaf_backend.py \
  tools/hpa320/oaf_host_attestation.py tools/hpa320/oaf_native_calibration.py \
  tools/hpa320/seal_oaf_backend.py \
  tests/benchmark/test_backend_lock.py tests/benchmark/test_oaf_backend.py \
  tests/benchmark/test_oaf_host_attestation.py tests/benchmark/test_oaf_seal.py
rtk git commit -m "feat: bind OaF seal to bootstrap image evidence"
```

### Task 5: Regenerate strict goldens and source manifests

**Files:**

- Create: `.github/workflows/hpa320-native-bootstrap.yml`
- Create: `.github/workflows/hpa320-native-measurement.yml`
- Create: `.github/workflows/hpa320-native-candidate.yml`
- Modify: `runtime/oaf_tf1/Dockerfile`
- Modify: `tools/hpa320/generate_runner_source_manifest.py`
- Modify: `tools/hpa320/oaf_build_context.py`
- Modify: `runtime/oaf_tf1/runner-source-manifest.json`
- Create: `runtime/oaf_tf1/build-context-manifest.json`
- Create:
  `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json`
- Modify: `tests/benchmark/schema_goldens/manifest.json`
- Create:
  `tests/benchmark/schema_goldens/crux.oaf-calibration-bootstrap-request-v1.json`
- Create:
  `tests/benchmark/schema_goldens/crux.oaf-calibration-bootstrap-evidence-v1.json`
- Create:
  `tests/benchmark/schema_goldens/crux.oaf-build-context-manifest-v1.json`
- Create:
  `tests/benchmark/schema_goldens/crux.oaf-native-host-attestation-bundle-v1.json`
- Modify:
  `tests/benchmark/schema_goldens/crux.oaf-calibration-measurement-request-v1.json`
- Modify:
  `tests/benchmark/schema_goldens/crux.oaf-calibration-measurement-evidence-v1.json`
- Modify:
  `tests/benchmark/schema_goldens/crux.oaf-seal-profile-request-v1.json`
- Modify:
  `tests/benchmark/schema_goldens/crux.oaf-seal-candidate-v1.json`
- Modify:
  `tests/benchmark/schema_goldens/crux.backend-seal-evidence-v1.json`
- Modify:
  `tests/benchmark/schema_goldens/crux.transcription-runtime-lock-v1.json`
- Modify:
  `tests/benchmark/schema_goldens/crux.oaf-oci-layout-manifest-v1.json`
- Create: `tests/benchmark/test_hpa320_native_bootstrap_workflow.py`
- Create: `tests/benchmark/test_hpa320_native_measurement_workflow.py`
- Create: `tests/benchmark/test_hpa320_native_candidate_workflow.py`
- Modify: `tests/benchmark/test_schema_goldens.py`
- Modify: `tests/benchmark/test_oaf_smoke_generator.py`

- [ ] Characterize the current host-evidence workflow. Add tests that all three new
      workflows accept exactly one 40-character commit SHA, check out that SHA, run
      on a non-emulated GitHub-hosted Linux X64 worker, and regenerate authenticated
      native-host evidence for the checked-out commit using the existing two-job
      observation/completed-job attestation pattern. Do not change or repurpose the
      accepted observation workflow.
- [ ] Make bootstrap acquire, build twice, attest, and upload; make measurement
      require accepted bootstrap evidence and a later measurement request; make
      candidate require the later profile and run `calibrate`. None may run `seal`
      or mutate repository files. Each workflow must generate and upload its own
      canonical phase-specific host-attestation bundle, including the exact raw API
      record bytes encoded as lowercase hexadecimal.
- [ ] Pin every workflow action by full commit. Reuse
      `actions/checkout@11d5960a326750d5838078e36cf38b85af677262` and
      `actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02`;
      install UV with
      `astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b`,
      `version: "0.11.8"`, `python-version: "3.12"`, and
      `enable-cache: false`. Set `UV_FROZEN=1`. Do not use a setup-buildx action
      because `bootstrap-image` verifies and runs the request-authorized Buildx
      binary itself.
- [ ] Workflow `run:` blocks invoke `uv` directly. `rtk` is the workstation command
      wrapper used by plan executors and is not installed on GitHub-hosted runners.
- [ ] Add `runtime/oaf_tf1/calibration_entrypoint.py`,
      `runtime/oaf_tf1/calibration_protocol.py`,
      `tools/hpa320/oaf_build_context.py`,
      `tools/hpa320/oaf_host_attestation.py`, `tools/hpa320/oaf_oci.py`,
      `tools/hpa320/oaf_native_calibration.py`, and
      `tools/hpa320/seal_oaf_backend.py` to `SOURCE_PATHS`. Also add
      `.github/workflows/hpa320-native-bootstrap.yml`, the sole workflow that
      determines image-build execution, and add `.github/workflows` to
      `COVERED_ROOTS`. Measurement and candidate workflow source remains reviewed
      CI orchestration but does not become an image byte. Keep the manifest itself
      excluded and all paths UTF-8 byte-sorted.
- [ ] Generate `crux.oaf-build-context-manifest/v1` from every exact Dockerfile input:
      the Dockerfile, dependency locks, source and distribution manifests, vendored
      and patched sources, runtime/test files, verified runtime/test wheelhouses,
      every explicitly copied host tool, and the native bootstrap workflow. Exclude
      final backend/runtime locks and the context manifest itself from `files`; the
      materializer adds the context manifest as the sole extra control file. Assert
      every Dockerfile `COPY` source is a subset of this manifest and that no
      `.dockerignore`, Dockerfile-specific ignore, symlink, hard link, or special file
      can enter the staged context.
- [ ] Update the Dockerfile's test and runtime stages to copy every
      runner-source-enumerated host tool plus the native bootstrap workflow beneath
      the same `/opt/crux` repository-relative paths used by mounted source
      validation. Broad directory or globbed `COPY` is permitted only because tests
      prove its complete source set is the exact staged manifest, not the repository
      root. Assert each covered root is a real directory and every enumerated byte is
      present in both stages.
- [ ] Add the bootstrap, build-context, and host-attestation-bundle schemas to the
      schema-golden manifest. List both
      `tools.hpa320.seal_oaf_backend` and `runtime.oaf_tf1.oaf_backend` whenever both
      sides validate the same schema.
- [ ] Update every affected golden to the normative exact key sets. Add unknown-key,
      missing-key, duplicate-key, wrong-type, wrong-digest, reordered-layer, and
      cross-validator disagreement tests.
- [ ] Preserve the acyclic generation order. First regenerate the runner source
      manifest after every covered source file is final; it excludes itself and the
      build-context manifest. Second generate the build-context manifest, which
      includes the runner manifest and excludes itself. Third create the
      calibration-bootstrap request with the exact build recipe and both
      `runner_source_manifest_sha256` and `build_context_manifest_sha256`. Compute
      every other referenced hash from exact current bytes and strict-load the
      completed request before committing.
- [ ] Assert the runner source manifest covers the Dockerfile syntax directive,
      native bootstrap workflow, calibration entrypoint/protocol, OCI packer, native
      orchestration, seal producer, manifest generator, and every runtime-side
      validator.
- [ ] Do not create final backend/runtime locks, final smoke oracle, final OCI
      identities, or native evidence in this task. Goldens are synthetic schema
      examples and must use obviously synthetic identities already accepted by the
      golden convention.

Run before regeneration:

```bash
rtk uv run pytest tests/benchmark/test_schema_goldens.py \
  tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_oaf_smoke_generator.py \
  -k 'build_context or schema_golden or runner_source_manifest'
rtk uv run pytest \
  tests/benchmark/test_hpa320_native_bootstrap_workflow.py \
  tests/benchmark/test_hpa320_native_measurement_workflow.py \
  tests/benchmark/test_hpa320_native_candidate_workflow.py
```

Expected: FAIL for missing bootstrap/context/bundle schemas and stale manifest bytes.

Regenerate and verify:

```bash
rtk uv run python tools/hpa320/generate_runner_source_manifest.py \
  --repository-root . \
  --output runtime/oaf_tf1/runner-source-manifest.json
rtk uv run python -m tools.hpa320.oaf_build_context generate \
  --repository-root . \
  --wheelhouse-root runtime/oaf_tf1/wheelhouse \
  --output runtime/oaf_tf1/build-context-manifest.json
rtk uv run pytest tests/benchmark/test_schema_goldens.py \
  tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_oaf_smoke_generator.py \
  tests/benchmark/test_hpa320_native_bootstrap_workflow.py \
  tests/benchmark/test_hpa320_native_measurement_workflow.py \
  tests/benchmark/test_hpa320_native_candidate_workflow.py
rtk uv run ruff check tools/hpa320/generate_runner_source_manifest.py \
  tools/hpa320/oaf_build_context.py \
  tests/benchmark/test_schema_goldens.py
rtk uv run black --check tools/hpa320/generate_runner_source_manifest.py \
  tools/hpa320/oaf_build_context.py \
  tests/benchmark/test_schema_goldens.py
```

- [ ] Commit:

```bash
rtk git add tools/hpa320/generate_runner_source_manifest.py \
  tools/hpa320/oaf_build_context.py \
  .github/workflows/hpa320-native-bootstrap.yml \
  .github/workflows/hpa320-native-measurement.yml \
  .github/workflows/hpa320-native-candidate.yml \
  runtime/oaf_tf1/Dockerfile \
  runtime/oaf_tf1/build-context-manifest.json \
  runtime/oaf_tf1/runner-source-manifest.json \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json \
  tests/benchmark/schema_goldens tests/benchmark/test_oaf_build_context.py \
  tests/benchmark/test_schema_goldens.py \
  tests/benchmark/test_oaf_smoke_generator.py \
  tests/benchmark/test_hpa320_native_bootstrap_workflow.py \
  tests/benchmark/test_hpa320_native_measurement_workflow.py \
  tests/benchmark/test_hpa320_native_candidate_workflow.py
rtk git commit -m "feat: freeze OaF native bootstrap contract"
```

### Task 6: Accept native bootstrap, bind it, then accept measurement evidence

**Files:**

- Create after bootstrap review:
  `docs/superpowers/evidence/hpa-320/native/checkpoint-acquisition-evidence.json`
- Create after bootstrap review:
  `docs/superpowers/evidence/hpa-320/native/base-system-package-evidence.json`
- Create after bootstrap review:
  `docs/superpowers/evidence/hpa-320/native/calibration-bootstrap-evidence.json`
- Create after bootstrap review:
  `docs/superpowers/evidence/hpa-320/native/bootstrap-host-attestation/attestation-bundle.json`
- Create after bootstrap review:
  `docs/superpowers/evidence/hpa-320/native/bootstrap-host-attestation/github-job-api-record.json.hex`
- Create after bootstrap review:
  `docs/superpowers/evidence/hpa-320/native/bootstrap-host-attestation/native-host-evidence.json`
- Create after bootstrap review:
  `docs/superpowers/evidence/hpa-320/native/bootstrap-host-attestation/native-host-observation.json`
- Create after bootstrap evidence is committed:
  `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-measurement-request.json`
- Create after measurement review:
  `docs/superpowers/evidence/hpa-320/native/calibration-measurement-evidence.json`
- Create after measurement review:
  `docs/superpowers/evidence/hpa-320/native/measurement-host-attestation/attestation-bundle.json`
- Create after measurement review:
  `docs/superpowers/evidence/hpa-320/native/measurement-host-attestation/github-job-api-record.json.hex`
- Create after measurement review:
  `docs/superpowers/evidence/hpa-320/native/measurement-host-attestation/native-host-evidence.json`
- Create after measurement review:
  `docs/superpowers/evidence/hpa-320/native/measurement-host-attestation/native-host-observation.json`

The three manual workflow definitions are already committed and authenticated by
Task 5. The preserved host observation remains bootstrap-path evidence only and
cannot be copied into any new evidence record.

**Phase A: run and review native bootstrap**

- [ ] Dispatch the bootstrap workflow at the exact workflow-source commit. It runs:

```bash
uv run python -m tools.hpa320.seal_oaf_backend bootstrap-image \
  --request config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json \
  --host-attestation-bundle /workspace/hpa320/attestation-bundle.json \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --output artifacts/benchmark/backends/hpa320-bootstrap/calibration-image \
  --repository-root .

uv run python -m tools.hpa320.seal_oaf_backend attest-base-system \
  --request runtime/oaf_tf1/base-system-package-request.json \
  --bootstrap-request config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json \
  --bootstrap-evidence artifacts/benchmark/backends/hpa320-bootstrap/calibration-image/calibration-bootstrap-evidence.json \
  --host-attestation-bundle /workspace/hpa320/attestation-bundle.json \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --output artifacts/benchmark/backends/hpa320-bootstrap/base-system-package-evidence.json
```

- [ ] Reject the run on build non-reproduction, wrong observed tool identity,
      package drift, assigned UID/GID, probe failure, archive mismatch, or config
      import drift.
- [ ] Human-review the bootstrap artifact manifest, complete host-attestation bundle,
      decoded raw API job bytes, regenerated host identity, checkpoint acquisition,
      minimal build context, pinned-base prefix, tool observations, OCI
      index/manifest/config/layers/DiffIDs/archive, package inventory, probes, and all
      cross-hashes.
- [ ] Copy the three accepted JSON evidence files and all four bootstrap bundle files
      to the exact checked-in paths. Prove staging and checked-in bytes are identical
      and that bootstrap evidence's bundle hash matches
      `attestation-bundle.json`. Preserve the OCI archive externally by the exact
      identity in bootstrap evidence.
- [ ] Commit those evidence and bundle files together. Do not include a measurement
      request, measurement evidence, or profile in this commit.

```bash
rtk git add \
  docs/superpowers/evidence/hpa-320/native/checkpoint-acquisition-evidence.json \
  docs/superpowers/evidence/hpa-320/native/base-system-package-evidence.json \
  docs/superpowers/evidence/hpa-320/native/calibration-bootstrap-evidence.json \
  docs/superpowers/evidence/hpa-320/native/bootstrap-host-attestation
rtk git commit -m "evidence: accept native OaF bootstrap"
```

**Phase B: bind the accepted bootstrap in a measurement request**

- [ ] Create the strict calibration-measurement request only now, using the exact
      committed bootstrap request/evidence hashes. The later measurement evidence,
      not this request, records the accepted acquisition and base-system evidence
      hashes it consumed. Freeze the five design-specified frame counts, WAV byte
      lengths and SHA-256 values, fixture derivation, required metrics, output
      schemas, and repetition count `3`.
- [ ] Strict-load the request through every validator and assert it has no profile,
      final-lock, mutable image, host-job, or generated measurement value.
- [ ] Commit this request alone so the later measurement evidence can prove which
      pre-existing authority it executed.

```bash
rtk uv run pytest tests/benchmark/test_oaf_seal.py \
  tests/benchmark/test_schema_goldens.py -k 'measurement_request'
rtk git add \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-measurement-request.json
rtk git commit -m "config: freeze OaF measurement request"
```

**Phase C: run and review native measurement**

- [ ] Dispatch the measurement workflow at the exact measurement-request commit.
      Regenerate host evidence for that commit, reacquire the checkpoint, and run
      `bootstrap-image` into a fresh operational output directory. Exact-compare the
      rebuilt index, manifest, config, layers, DiffIDs, archive identity, and
      `image_build` with the accepted bootstrap evidence before using its imported
      config digest. The fresh wrapper evidence is operational output and never
      replaces the accepted bootstrap evidence. Then run:

```bash
uv run python -m tools.hpa320.seal_oaf_backend measure \
  --request config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-measurement-request.json \
  --bootstrap-request config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json \
  --bootstrap-evidence docs/superpowers/evidence/hpa-320/native/calibration-bootstrap-evidence.json \
  --host-attestation-bundle /workspace/hpa320/attestation-bundle.json \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --model-cache artifacts/benchmark/model-cache \
  --checkpoint-evidence docs/superpowers/evidence/hpa-320/native/checkpoint-acquisition-evidence.json \
  --base-system-evidence docs/superpowers/evidence/hpa-320/native/base-system-package-evidence.json \
  --output artifacts/benchmark/backends/hpa320-bootstrap/calibration-measurement-evidence.json
```

- [ ] Require all five exact fixtures and exactly three fresh processes per fixture.
      Reject a non-exact 15-row matrix, duplicate process ID, unhealthy row, ceiling
      contact, deadline, OOM, signal, output-bound contact, unavailable monitor, or
      any bootstrap/checkpoint/base identity drift.
- [ ] Human-review every row, metric, prediction hash, host identity, decoded raw API
      job bytes, complete measurement host-attestation bundle, and artifact manifest.
      Copy the accepted canonical measurement evidence and all four bundle files to
      their checked-in paths; prove byte identity and the owning bundle hash.
- [ ] Commit measurement evidence and its bundle together. Do not include a
      seal-profile request.

```bash
rtk git add \
  docs/superpowers/evidence/hpa-320/native/calibration-measurement-evidence.json \
  docs/superpowers/evidence/hpa-320/native/measurement-host-attestation
rtk git commit -m "evidence: accept native OaF measurements"
```

### Task 7: Review the profile and generate the candidate

**Files:**

- Create in a separate review commit:
  `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-profile-request.json`

- [ ] Confirm the measurement request predates the accepted measurement run and its
      exact hash equals `request_sha256` in measurement evidence. If not, discard the
      run and repeat Task 6 Phase C.
- [ ] Review each of the 15 rows and choose every final CPU, memory, PID, tmpfs,
      deadline, stdout, and stderr bound explicitly. Each chosen value must satisfy
      the design's headroom rules and remain within the bootstrap ceiling.
- [ ] Set `max_input_audio_frames` exactly `26214378` and UID/GID exactly
      `65532:65532`. Bind the profile to all acquisition, base-system, bootstrap, and
      measurement request/evidence hashes.
- [ ] Commit the profile alone after human review. The producer must never generate
      or edit this file.

```bash
rtk uv run pytest tests/benchmark/test_oaf_seal.py \
  -k 'profile or calibrate'
rtk git add \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-profile-request.json
rtk git commit -m "config: approve OaF native seal profile"
```

- [ ] Dispatch the already-committed candidate workflow at that exact profile
      commit. Regenerate native-host evidence for the commit, strict-validate the
      committed authority chain, rerun `bootstrap-image` into a fresh operational
      output directory, exact-compare every deterministic OCI identity with the
      accepted bootstrap evidence, retain the accepted evidence as authority, and
      run:

```bash
uv run python -m tools.hpa320.seal_oaf_backend calibrate \
  --request config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-profile-request.json \
  --measurement-evidence docs/superpowers/evidence/hpa-320/native/calibration-measurement-evidence.json \
  --bootstrap-request config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json \
  --bootstrap-evidence docs/superpowers/evidence/hpa-320/native/calibration-bootstrap-evidence.json \
  --host-attestation-bundle /workspace/hpa320/attestation-bundle.json \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --model-cache artifacts/benchmark/model-cache \
  --checkpoint-evidence docs/superpowers/evidence/hpa-320/native/checkpoint-acquisition-evidence.json \
  --base-system-evidence docs/superpowers/evidence/hpa-320/native/base-system-package-evidence.json \
  --output artifacts/benchmark/backends/hpa320-seal-candidate
```

- [ ] Require exact 130/78/52 tensor coverage, byte-identical patched/unmodified
      `NoteSequence`, finite nonempty calibration-native smoke events, healthy
      boundary/process probes, security/advisory evidence, and a complete candidate
      artifact allowlist. The allowlist must contain the exact candidate
      `attestation-bundle.json`, lowercase-hex raw API record,
      `native-host-evidence.json`, and `native-host-observation.json`; the candidate
      bundle hash must exact-match seal evidence. `calibrate` copies those four input
      files verbatim into the candidate staging tree at their final repository-relative
      paths; it never reconstructs them.
- [ ] Upload the candidate, complete candidate host-attestation bundle, and canonical
      artifact manifest. The workflow must not invoke `seal`, publish final locks, or
      modify the registry.
- [ ] Review candidate bytes, decoded raw API response, all four bundle files, and
      manifest. Any mismatch or missing native result returns to Task 6 or the
      profile-review step; it is not repaired inside the candidate directory.

### Task 8: Publish the seal, prove integration, and document corpus ownership

**Files:**

- Create from the reviewed candidate:
  `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json`
- Create from the reviewed candidate:
  `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.runtime-lock.json`
- Create from the reviewed candidate:
  `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.backend-lock.json`
- Modify: `src/benchmark/backend_registry.py`
- Create/Update: `tests/fixtures/oaf_tf1_smoke/smoke-oracle.json`
- Create/Update: `runtime/oaf_tf1/host-adapter-source-manifest.json`
- Create/Update: `docs/superpowers/evidence/hpa-320/oaf-oci-layout-manifest.json`
- Create/Update: `docs/superpowers/evidence/hpa-320/oaf-security-scan.json`
- Create/Update: `docs/superpowers/evidence/hpa-320/oaf-tensor-coverage.json`
- Create from the reviewed candidate:
  `docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/attestation-bundle.json`
- Create from the reviewed candidate:
  `docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/github-job-api-record.json.hex`
- Create from the reviewed candidate:
  `docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/native-host-evidence.json`
- Create from the reviewed candidate:
  `docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/native-host-observation.json`
- Create: `.github/workflows/hpa320-native-sealed-integration.yml`
- Create: `tests/benchmark/test_hpa320_native_sealed_workflow.py`
- Modify: `tests/benchmark/test_backend_contract_acceptance.py`
- Modify: `tests/benchmark/test_backend_registry.py`
- Modify: `tests/benchmark/test_oaf_real_checkpoint.py`
- Modify: `docs/superpowers/specs/2026-07-26-hpa-320-freeze-oaf-drums-backend-design.md`

- [ ] On the reviewed native candidate, run only:

```bash
rtk uv run python -m tools.hpa320.seal_oaf_backend seal \
  --candidate artifacts/benchmark/backends/hpa320-seal-candidate \
  --repository-root .
```

- [ ] Verify immutable publication order: evidence/oracle/source and OCI manifests
      first, runtime lock second, backend lock last. Existing identical bytes are
      idempotent; existing different bytes are fatal and remain untouched.
- [ ] Require `seal` to publish the four candidate host-attestation files verbatim
      from the candidate allowlist before seal evidence. Recompute the canonical
      bundle-manifest SHA-256 and exact-compare it with candidate and seal-evidence
      fields; never fetch or reconstruct expired workflow evidence during sealing.
- [ ] Preserve the authenticated OCI archive and checkpoint outside the Git tree by
      exact byte length and SHA-256. The repository records their immutable
      identities and recovery location; a future inability to rebuild never licenses
      replacement of preserved bytes.
- [ ] Add the sealed native integration workflow in the same commit as the final
      locks and registry flip. It runs on inference-relevant changes, schedule, and
      manual dispatch on native Linux X64.
- [ ] The workflow exact-rebuilds the image twice, compares index/manifest/config,
      ordered layers, ordered DiffIDs, canonical archive bytes, and archive SHA-256.
      It rematerializes the exact request-hashed context, authenticates the raw pinned
      base manifest/config and inherited prefix, allows historical mtimes only in that
      exact prefix, validates OCI JSON without canonicalizing it, imports by config
      digest, reacquires the checkpoint, and regenerates operational measurement
      evidence.
- [ ] The workflow reruns all real-checkpoint tests without skips and exactly five
      production inference calls: one startup oracle call in each of two processes,
      two post-ready calls in the persistent process, and one post-ready call in the
      fresh process. Require three byte-identical published artifacts and
      `verify-backend` status `verified`, exit `0`.
- [ ] Add acceptance tests proving a missing typed report cannot be mistaken for a
      successful exit, any smoke mismatch is backend-fatal on every host fingerprint,
      and the registry is `sealed` only when all final identities cross-validate.
- [ ] Change only the official OaF registration literal in
      `src/benchmark/backend_registry.py` from `seal_state="preseal"` to
      `seal_state="sealed"` in the same commit as all final lock bytes. Keep the
      heuristic and legacy registrations unchanged.
- [ ] Keep preservation and contemporary rebuild claims distinct in operator-facing
      documentation: preservation guarantees recovery of accepted bytes; a passing
      scheduled rebuild proves the pinned recipe reproduced at that execution time.
- [ ] Preserve the HPA-322/HPA-326 handoff. HPA-322 records canonical mono PCM frame
      counts for selected items. HPA-326 exact-compares each count with
      `26214378`, records every over-bound exclusion in the run manifest, and never
      hides chunking. HPA-320 completion does not depend on a later corpus-fit result.

Run the complete host stack:

```bash
rtk uv run pytest
rtk uv run ruff check src tests tools/hpa320 runtime/oaf_tf1/*.py \
  runtime/oaf_tf1/tests
rtk uv run black --check src tests tools/hpa320 runtime/oaf_tf1/*.py \
  runtime/oaf_tf1/tests
rtk uv run pylint src/app src/cli
rtk git diff --check
```

- [ ] Commit the final seal atomically:

```bash
rtk git add src/benchmark/backend_registry.py \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.runtime-lock.json \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.backend-lock.json \
  runtime/oaf_tf1/host-adapter-source-manifest.json \
  docs/superpowers/evidence/hpa-320/oaf-oci-layout-manifest.json \
  docs/superpowers/evidence/hpa-320/oaf-security-scan.json \
  docs/superpowers/evidence/hpa-320/oaf-tensor-coverage.json \
  docs/superpowers/evidence/hpa-320/native/candidate-host-attestation \
  docs/superpowers/specs/2026-07-26-hpa-320-freeze-oaf-drums-backend-design.md \
  tests/fixtures/oaf_tf1_smoke/smoke-oracle.json \
  .github/workflows/hpa320-native-sealed-integration.yml \
  tests/benchmark/test_hpa320_native_sealed_workflow.py \
  tests/benchmark/test_backend_contract_acceptance.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_oaf_real_checkpoint.py
rtk git commit -m "feat: seal frozen OaF transcription backend"
```

- [ ] Push that exact commit and dispatch the newly committed sealed workflow.
      Require every real-checkpoint test to report executed rather than skipped.
      Download and inspect its canonical artifact manifest; exit status alone is not
      acceptance evidence. HPA-320 remains incomplete until this exact-head run
      passes.

## Completion Evidence

HPA-320 is complete only when the final commit and native workflow jointly prove:

- all five request authorities and their accepted evidence form one acyclic,
  strict-hash chain;
- the request-hashed minimal context is rematerialized with exact bytes and normalized
  modes, without repository ignore rules or unlisted checkout inputs;
- the raw pinned base manifest/config authenticate the exact inherited layer
  digest/DiffID prefix, and epoch-zero mtime is enforced on every later layer without
  falsely requiring the historical base blobs to be rewritten;
- two fresh native builds reproduce every raw OCI identity and canonical archive byte
  without canonicalizing OCI JSON;
- bootstrap, measurement, and candidate evidence each retain a complete checked-in
  host-attestation bundle whose exact raw API response remains verifiable after
  workflow artifact expiry;
- production and calibration entrypoints reject each other's authorities before
  TensorFlow import;
- all 78 inference tensors restore and all 52 non-inference entries remain
  classified;
- the five fixtures produce exactly 15 healthy measurement rows;
- boundary probes reject frame `26214379` before inference while frame `26214378`
  remains within the frozen domain;
- exactly five production inference calls yield three byte-identical published
  artifacts;
- final backend, runtime, seal, OCI, checkpoint, source, smoke, and registry
  identities cross-validate on both host and runner sides; and
- preserved bytes remain recoverable independently of whether a later contemporary
  rebuild service is available.
