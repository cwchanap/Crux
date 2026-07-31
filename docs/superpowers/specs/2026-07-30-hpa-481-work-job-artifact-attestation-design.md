# HPA-481/HPA-482 Work-Job Artifact Attestation Design

**Date:** 2026-07-30
**Status:** Proposed
**Issues:** HPA-481, HPA-482
**Blocks:** HPA-423, then HPA-424 and HPA-425

## Summary

The three HPA-320 native production workflows currently observe one GitHub-hosted
runner and perform the native work on another. GitHub Actions assigns each job a
fresh runner, so the checked host-attestation bundle describes
`observe-native-host`, not the `native-bootstrap`, `native-measurement`, or
`native-candidate` job whose outputs matter.

Each phase will instead use one GitHub-hosted work job. That job will:

1. fail closed unless GitHub reports `runner.environment == "github-hosted"`,
   `runner.os == "Linux"`, and `runner.arch == "X64"`;
2. record the host observation on that same runner;
3. perform all phase work on that runner;
4. produce a canonical manifest and deterministic evidence archive;
5. use GitHub's pinned artifact-attestation action in that same job to sign the
   manifest and archive; and
6. upload the payload, manifest, archive, and local Sigstore bundle together.

Acceptance verifies the signed provenance outside-in before trusting any native
result. The separate observation job, GitHub Jobs REST lookup, raw completed-job API
record, and `actions: read` permission are removed.

This design changes execution provenance only. It does not change the checkpoint,
runtime image, backend lock, inference parameters, smoke oracle, prediction schema,
or the acyclic request/evidence/lock ordering frozen by HPA-320.

## Existing defect

The merged workflows have this shape:

```text
observe-native-host (runner A)
  -> JSON job output
native-{phase} (runner B)
  -> fetch completed API record for runner A's job
  -> perform phase work on runner B
```

`tools/hpa320/oaf_host_attestation.py` then hard-codes the completed job name
`observe-native-host` and accepts its Linux/X64 labels. This proves only that runner A
was suitable. It says nothing about runner B.

The observation also omits GitHub's `runner.environment` context. A self-hosted
runner can expose Linux/X64 labels and environment values, so the current evidence
cannot distinguish the explicitly approved GitHub-hosted producer path.

## Goals

- Bind native host evidence to the exact job that performs the native work.
- Require both the in-job `runner.environment` value and GitHub's signed provenance
  to identify a GitHub-hosted runner.
- Preserve phase, workflow, repository, commit, run, and host-fingerprint evidence.
- Preserve deterministic, hash-addressed artifacts and the existing acyclic seal
  identity model.
- Retain a downloadable local Sigstore bundle for repeatable local-bundle review.
- Reject the old cross-runner evidence format at new acceptance boundaries.
- Keep each phase to one work job; do not add a finalizer or observation job.

## Non-goals

- Running HPA-423 bootstrap, measurement, or candidate work before this change is
  merged.
- Changing the approved native platform from GitHub-hosted Linux X64.
- Making self-hosted runners eligible for these three seal-production phases.
- Redesigning the orchestrator-signed or approved-local evidence forms used by other
  official execution paths.
- Reopening the HPA-320 backend or runtime identity.
- Claiming that a valid provenance signature makes an inference result correct.
  Result schemas, hashes, and phase-specific validation remain independently
  authoritative.

## Considered approaches

### 1. Query the work job after it completes

A later finalizer job could query the completed work-job record. This would identify
the correct job, but it adds another runner and requires a second artifact handoff.
The finalizer would still need a cryptographic binding to prove that the fetched
payload is exactly what the work job produced.

### 2. Keep the observation job and compare labels

This is the current shape. No label or environment comparison can establish that two
jobs ran on the same ephemeral machine. It does not close HPA-481.

### 3. Attest the work artifact in the work job

GitHub artifact attestations bind named subject digests to signed SLSA provenance
containing the repository, workflow, commit, event, and runner class. GitHub CLI
verification can also reject self-hosted signers. This directly authenticates the
outputs of the job that did the work and needs no completed-job lookup.

**Decision:** use approach 3.

## Trust boundary

The phase work job is the producer trust boundary:

```text
reviewed workflow at exact commit
  -> same-job GitHub-hosted/Linux/X64 preflight
  -> same-job host observation and native work
  -> canonical payload manifest
  -> deterministic evidence archive
  -> GitHub OIDC/Sigstore attestation of manifest + archive
  -> verification policy
  -> accepted phase evidence
```

The in-job observation is a signed payload claim, not the cryptographic root. The
Sigstore certificate and verified GitHub provenance are the external root. Acceptance
requires both:

- the signed provenance passes repository, signer-workflow, exact source commit, and
  non-self-hosted policy; and
- the signed payload records literal `github-hosted`, Linux, and X64 values that
  agree with the expected phase workflow.

The workflow source is reviewed at the same exact commit that is dispatched. Because
the attestation step is inside the sole work job, the signed subjects can only be
produced after that job's preflight and native phase commands.

## Workflow architecture

Each of these files becomes a one-job workflow:

- `.github/workflows/hpa320-native-bootstrap.yml`
- `.github/workflows/hpa320-native-measurement.yml`
- `.github/workflows/hpa320-native-candidate.yml`

The job names remain:

- `native-bootstrap`
- `native-measurement`
- `native-candidate`

The `observe-native-host` job, `needs: observe-native-host`,
`NATIVE_HOST_OBSERVATION`, `GH_TOKEN`, and the cross-job JSON output disappear.

Each remaining job performs these ordered stages:

1. Check out the exact lowercase 40-character dispatch commit.
2. Install the existing pinned UV toolchain.
3. On the observation step, export
   `RUNNER_ENVIRONMENT: ${{ runner.environment }}` and
   `WORKFLOW_SOURCE_SHA: ${{ github.workflow_sha }}` without permitting
   caller-provided overrides.
4. Observe and validate the current job before acquisition, Docker build, measurement,
   or calibration.
5. Produce the phase's existing outputs.
6. Generate the canonical payload manifest.
7. Pack the deterministic evidence archive.
8. Generate one SLSA provenance attestation covering both the manifest and archive.
9. Copy the action's local `bundle-path` output to a stable artifact filename.
10. Upload the payload directory, manifest, archive, and local attestation bundle.

The workflow grants only the permissions required for checkout and file provenance:

```yaml
permissions:
  contents: read
  id-token: write
  attestations: write
```

It does not grant `actions: read`. Artifact metadata storage records are not part of
this design; the action is configured not to create one if its selected version would
otherwise attempt that path.

All actions remain immutable-SHA pinned. The artifact-attestation action selected at
design time is:

```text
actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d  # v4.2.1
```

The implementation plan must revalidate the tag-to-commit mapping and action inputs
before editing the workflows. A newer tag does not replace this SHA without review.

## Same-job host observation

`observe-github` no longer writes a value to `$GITHUB_OUTPUT`. It writes one canonical
JSON file directly into the phase payload directory and derives the native-host
evidence beside it.

The strict observation contains the existing repository, run, workflow, commit,
Docker, uname, and numeric-fingerprint fields plus:

```text
github_job
github_workflow_sha
runner_environment
```

The accepted values are phase-specific:

| Phase | `github_job` | workflow file |
| --- | --- | --- |
| bootstrap | `native-bootstrap` | `hpa320-native-bootstrap.yml` |
| measurement | `native-measurement` | `hpa320-native-measurement.yml` |
| candidate | `native-candidate` | `hpa320-native-candidate.yml` |

Every phase requires:

```text
runner_environment = github-hosted
runner_os = Linux
runner_arch = X64
uname_architecture = x86_64
docker_os_type = linux
docker_architecture = x86_64
```

It also requires `GITHUB_SHA`, the checked-out commit, the dispatch `COMMIT_SHA`, and
the step-exported `WORKFLOW_SOURCE_SHA` to agree. `GITHUB_WORKFLOW_REF` is separately
validated as the phase-specific repository/workflow/ref path; its suffix is a Git ref,
not a commit identity. A missing field is a hard error; there is no inferred or
compatibility default.

## Host evidence schema migration

New phase executions use:

```text
crux.oaf-native-host-attestation-bundle/v2
```

The v2 bundle has exactly:

```text
native_host_evidence, native_host_observation, phase, schema
```

It removes `api_record` and the sibling
`github-job-api-record.json.hex`. The raw Jobs API record described another job and
is not replaced by a fabricated same-job record.

The strict `github_hosted` native-evidence payload gains a schema discriminator and
contains:

```text
schema, github_job, github_repository, github_run_attempt, github_run_id,
github_workflow_ref, github_workflow_sha, host_numeric_fingerprint, run_url,
runner_arch, runner_environment, runner_os, workflow_commit
```

It removes:

```text
api_record_sha256, approved_labels, job_id
```

The bundle can be generated before the native work because it contains no output
digest. During production it is a phase-owned host claim embedded by hash in the
existing phase result. It becomes acceptable authority only when the final signed
artifact manifest binds that bundle and the external GitHub attestation passes the
verification policy.

A standalone v2 bundle must therefore never authorize sealing, publication, or
official execution. Consumer paths require the signed manifest and Sigstore bundle in
addition to the internal v2 bundle.

## Canonical artifact manifest

Each successful phase produces:

```text
crux.oaf-native-work-artifact-manifest/v1
```

with exactly:

```text
files, github_job, github_repository, github_run_attempt, github_run_id,
github_workflow_ref, github_workflow_sha,
native_host_attestation_bundle_sha256, phase, schema, status, workflow_commit
```

`status` is exactly `success`. `files` is a nonempty array of strict artifact
references containing:

```text
path, role, sha256, size
```

Rows are sorted by UTF-8 bytes of `(role, path)`. Paths are repository-relative POSIX
paths below the phase payload root. They cannot be absolute, contain `.` or `..`
segments, use backslashes, collide, escape through a symlink, or name the manifest,
archive, or external Sigstore bundle.

The manifest covers every other regular payload file, including:

- `attestation-bundle.json`;
- `native-host-evidence.json`;
- `native-host-observation.json`;
- phase result and validation records;
- the OCI archive and its identity records where that phase emits them; and
- sanitized diagnostics retained by the successful run.

The producer scans the payload after phase work and fails on an unlisted,
multiply-listed, non-regular, or changed file. The consumer reproduces every row from
the extracted payload before reading a phase result.

## Deterministic evidence archive

The archive is an uncompressed POSIX ustar file named:

```text
hpa320-native-<phase>-<workflow_commit>.tar
```

It contains the manifest and all files named by the manifest. Packing uses:

- UTF-8 byte-sorted member paths;
- regular files and directories only;
- file mode `0644` and directory mode `0755`;
- numeric UID/GID `0`, empty owner/group names, and mtime `0`;
- no PAX metadata, symlinks, hard links, devices, sockets, or FIFOs;
- rejection of a path that cannot be represented exactly in ustar; and
- one canonical root directory
  `hpa320-native-<phase>-<workflow_commit>/`.

The packer writes to a new temporary path, verifies the archive by strict re-reading,
and atomically publishes it. Repacking the same payload must reproduce identical
bytes.

## External GitHub artifact attestation

The pinned `actions/attest` invocation uses default SLSA provenance and identifies two
subjects in one attestation:

1. the deterministic evidence archive; and
2. the canonical artifact manifest.

Signing the manifest directly preserves a durable link to every payload digest even
after the convenience archive's GitHub retention window expires. Signing the archive
also proves that the reviewed download is the exact packaged payload.

The local Sigstore bundle returned by `bundle-path` is copied to:

```text
github-artifact-attestation.json
```

That file remains outside the signed manifest and archive. The action also publishes
the attestation to GitHub's attestation API and, for this public repository, the
public Sigstore transparency log.

This ordering is acyclic:

```text
host observation/evidence
  -> v2 host bundle
  -> phase outputs reference host-bundle hash
  -> artifact manifest hashes payload files
  -> archive contains manifest + payload
  -> external attestation signs manifest + archive
```

No payload, manifest, archive, backend lock, runtime lock, or seal-evidence file
contains the external attestation's digest.

## Acceptance and verification

Before a bootstrap, measurement, or candidate result can be checked in or consumed,
the reviewer runs GitHub CLI verification against both signed subjects. The policy
requires:

```text
--repo cwchanap/Crux
--signer-workflow cwchanap/Crux/.github/workflows/hpa320-native-<phase>.yml
--source-digest <workflow_commit>
--signer-digest <workflow_commit>
--deny-self-hosted-runners
```

The saved local bundle is supplied with `--bundle` for durable local-bundle
verification; online lookup may be performed as a second comparison but is not the
sole authority. A fully disconnected verification additionally supplies a trusted
root obtained out of band through `gh attestation trusted-root`; the local bundle
alone is not described as a complete offline trust store. The verifier uses the
default SLSA provenance predicate and GitHub Actions OIDC issuer.

After signature verification, the consumer:

1. requires the manifest and archive to appear as subjects with their exact SHA-256
   digests;
2. strict-parses the manifest;
3. verifies the archive's canonical packing and all manifest rows;
4. loads the v2 host bundle and exact sibling files;
5. cross-validates repository, phase, workflow ref, workflow source SHA, job, run ID,
   run attempt, commit, and host fingerprint across observation, evidence, manifest,
   and phase result;
6. requires `runner_environment == "github-hosted"`, Linux, X64, native uname, and
   native Docker architecture; and
7. applies every existing phase-specific request, evidence, image, checkpoint,
   measurement, and candidate validation.

Attestation verification proves provenance and integrity. Step 7 proves that the
result is acceptable.

The exact verification command and its machine-readable JSON output are recorded in
the review checklist, but that output is not itself treated as signed authority.

## Failure semantics

- A runner-environment, OS, architecture, commit, workflow, or Docker mismatch fails
  before native work starts.
- Native phase failure fails the job and cannot produce an accepted success manifest.
- Partial sanitized diagnostics may still be uploaded under a clearly diagnostic
  artifact name. They are never relabeled as accepted evidence.
- Manifest or deterministic-packing failure fails the job before attestation.
- Attestation or upload failure fails the job; unsigned phase results are not
  accepted.
- A valid attestation with an invalid manifest or phase result is rejected.
- A valid phase result with a missing, invalid, wrong-workflow, wrong-commit, or
  self-hosted attestation is rejected.
- Previously published historical artifacts are preserved for audit. They are not
  rewritten to v2.

## Migration

The migration is atomic across tooling, schemas, consumers, tests, documentation, and
all three manual workflows:

1. Add v2 host-bundle, artifact-manifest, deterministic-packer, and verification
   support.
2. Change all acceptance consumers to reject an unaccompanied v2 bundle.
3. Convert bootstrap, measurement, and candidate workflows to their single-job form.
4. Remove the Jobs REST finalizer, API-record decoder, raw-record schema fields,
   cross-job output handling, and `actions: read`.
5. Update schema goldens and the HPA-320 design's superseded provenance passages.
6. Run the local verification stack.
7. Merge HPA-481 and HPA-482 together.
8. Only then dispatch HPA-423 at an exact merged commit.

The v1 schema remains readable only for explicitly historical evidence and tests.
It is not accepted as newly produced bootstrap, measurement, candidate, final-seal,
or official-publication authority after the migration commit.

No compatibility mode may silently infer `runner_environment`, accept an API record
for another job, or treat v1 and v2 as equivalent.

## Verification strategy

### Unit tests

- Observation accepts the exact GitHub-hosted/Linux/X64 environment.
- Observation rejects missing, empty, self-hosted, wrong-OS, and wrong-architecture
  runner values before Docker work.
- Observation rejects wrong job, workflow ref, workflow source SHA, repository, run
  identity, or commit.
- V2 bundle parsing is strict and rejects v1 substitution at new acceptance
  boundaries.
- Generic GitHub-hosted native evidence requires `runner_environment`.
- Manifest parsing rejects noncanonical JSON, extra or missing keys, path escape,
  duplicates, wrong ordering, missing files, size/hash drift, and self-reference.
- Deterministic packing reproduces byte-identical archives and rejects unsafe member
  types or metadata.
- External verification policy requires repository, phase workflow, source/signing
  commit, SLSA predicate, GitHub issuer, and denial of self-hosted runners.
- Cross-validation rejects a valid signature over internally inconsistent evidence.

### Workflow contract tests

Each phase workflow test requires:

- exactly one job with the expected `native-*` name;
- no `observe-native-host`, `needs`, `GH_TOKEN`, or `actions: read`;
- explicit `runner.environment` propagation;
- host preflight before any native work;
- manifest and archive generation after successful native work;
- immutable-SHA-pinned `actions/attest` in that same job;
- `contents: read`, `id-token: write`, and `attestations: write`;
- attestation of both manifest and archive;
- upload of the local Sigstore bundle; and
- no bootstrap, measurement, or candidate acceptance before attestation succeeds.

### Tamper tests

Fixtures cover:

- changed archive bytes;
- changed signed-manifest bytes;
- changed payload after manifest creation;
- wrong phase bundle;
- wrong workflow or commit;
- self-hosted signer provenance;
- missing attestation subject;
- a signed diagnostic/failed result presented as success; and
- an old v1 raw-API bundle presented as new authority.

### Repository checks

The implementation finishes with:

```text
uv run pytest
uv run ruff check src tests tools
uv run black --check src tests tools
uv run pylint src/app src/cli
```

If repository configuration narrows any command differently at implementation time,
the plan records the exact authoritative command and reason.

## Repository ownership

- `tools/hpa320/oaf_host_attestation.py` owns same-job observation and strict v2 host
  bundle validation.
- A small dedicated HPA-320 artifact module owns canonical manifest generation,
  deterministic archive packing, and structural verification; workflow shell must not
  reimplement these rules.
- `src/benchmark/backend_process.py` owns strict generic
  `github_hosted` native-evidence validation.
- `tools/hpa320/seal_oaf_backend.py` owns phase-specific producer and consumer gates.
- `.github/workflows/hpa320-native-*.yml` owns ordered invocation, least privileges,
  pinned actions, and artifact publication.
- `tests/benchmark/` owns schema, tamper, workflow-contract, and failure-path coverage.
- `docs/superpowers/evidence/hpa-320/native/` owns accepted signed manifests, local
  Sigstore bundles, v2 host bundles, and their owning phase evidence after review.

## Supersession of the HPA-320 design

For bootstrap, measurement, and candidate production after this migration, this
document supersedes the HPA-320 design passages that require:

- a separate `observe-native-host` job;
- a completed `observe-native-host` Jobs API record;
- `github-job-api-record.json.hex`;
- `crux.oaf-native-host-attestation-bundle/v1` as new phase authority; or
- a four-file phase host bundle containing that API record.

All unrelated HPA-320 contracts remain in force. The HPA-320 design should receive a
short amendment pointing to this document rather than silently retaining conflicting
instructions.

## Operational gate

HPA-423 remains blocked until:

1. this design is approved;
2. its implementation plan is approved;
3. implementation and verification complete;
4. HPA-481 and HPA-482 merge; and
5. the exact merged commit is confirmed.

Only then may the native bootstrap workflow be dispatched. Measurement, candidate,
and final sealing retain their existing human-review and request/evidence ordering.

## References

- GitHub artifact attestation concepts:
  <https://docs.github.com/en/actions/concepts/security/artifact-attestations>
- GitHub artifact attestation usage:
  <https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations>
- `actions/attest`:
  <https://github.com/actions/attest>
- `gh attestation verify`:
  <https://cli.github.com/manual/gh_attestation_verify>
- `gh attestation trusted-root`:
  <https://cli.github.com/manual/gh_attestation_trusted-root>
