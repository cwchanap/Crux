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

The changed GitHub-hosted evidence is not inserted into existing `/v1` containers.
The host bundle and all five existing outer evidence schemas that embed the changed
payload or candidate inventory move to `/v2` together. There is no production v1
compatibility reader.

The legacy `.github/workflows/hpa320-native-host-evidence.yml` scratch collector uses
the same cross-runner API-record pattern but does not produce bootstrap, measurement,
or candidate authority. It is deleted rather than migrated. Its trigger branch
`hpa-320-native-seal-evidence` is retired, or the workflow is disabled in GitHub,
before HPA-423 is allowed to dispatch.

This design changes execution provenance and the source-attestation surface, not the
checkpoint, model graph, inference parameters, smoke oracle, prediction schema, or
the acyclic request/evidence/lock ordering frozen by HPA-320. The bootstrap workflow,
changed HPA-320 tools, runtime validators, and runner source manifest are copied into
the provisional image, so its not-yet-published bytes and OCI identity necessarily
change. The migration therefore regenerates the runner source manifest and build
context, then reissues the pre-dispatch calibration-bootstrap request with their new
hashes. No bootstrap evidence, measurement request, seal profile, candidate, backend
lock, or runtime lock has been accepted, so this does not rewrite a published
backend, runtime, request/evidence chain, or prediction identity.

## Ticket ownership and atomic delivery

The live issue split is:

| Issue | Owned requirements |
| --- | --- |
| HPA-481 | Collapse observation and native work into one job; remove the Jobs REST/API-record path; bind the complete work-job output through the canonical manifest, deterministic archive, GitHub attestation, and acceptance policy. |
| HPA-482 | Propagate `runner.environment`; cross-check the context and default environment values; record `runner_environment`; and reject non-`github-hosted` production and consumption. |

The manifest and deterministic packer are HPA-481 mechanisms, not HPA-482 scope.
HPA-482 is folded into HPA-481 because both issues change the same host observation,
evidence payload, outer schemas, workflows, and tests. Implementation may stage
internally by the ownership above, but there is one schema migration, one acceptance
gate, and one atomic merge.

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

`.github/workflows/hpa320-native-host-evidence.yml` independently duplicates the
same defect with inline Python and publishes only scratch host evidence. No current
production or validation workflow consumes its artifact, and retaining it would
preserve the deprecated `actions: read` and Jobs API path.

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
- Reopening the HPA-320 checkpoint, model, inference, or final sealed-identity
  contract. The pre-dispatch provisional image and bootstrap request are deliberately
  reissued because their source-attested inputs change before any evidence accepts
  them.
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

The external runner-class authority is the Fulcio certificate in GitHub's signed
provenance. Acceptance verifies that certificate and applies
`--deny-self-hosted-runners`; this is the independent proof that the signer used a
GitHub-hosted runner. The phase work job is then the producer trust boundary:

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
Sigstore certificate and verified GitHub provenance are the external root. The
`RUNNER_*_CONTEXT == RUNNER_*` checks compare two values exposed to the same runner
process; they are useful consistency and accidental-override guards, but they are not
an independent runner-class authority. Acceptance requires both:

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

The obsolete `.github/workflows/hpa320-native-host-evidence.yml` is deleted. It is
not a fourth phase and receives no v2 replacement.

The job names remain:

- `native-bootstrap`
- `native-measurement`
- `native-candidate`

The `observe-native-host` job, `needs: observe-native-host`,
`NATIVE_HOST_OBSERVATION`, `GH_TOKEN`, and the cross-job JSON output disappear.

The dispatch operator resolves one branch or tag to the intended lowercase
40-character commit immediately before dispatch, records that resolution, and invokes
the workflow with both:

```text
gh workflow run hpa320-native-<phase>.yml \
  --ref <dispatch-branch-or-tag> \
  --raw-field commit_sha=<resolved-workflow-commit>
```

`gh workflow run --ref` is treated as a branch-or-tag selector, not as proof of an
arbitrary commit SHA. An immutable tag is preferred. A branch is permitted only when
its resolved commit is recorded immediately before dispatch; if it moves before
GitHub creates the run, the in-job commit-equality preflight fails before native work.
Passing `commit_sha` without the matching `--ref` is not a valid production dispatch.

Each remaining job performs these ordered stages:

1. Check out the exact lowercase 40-character dispatch commit.
2. Install the existing pinned UV toolchain.
3. On the observation step, export
   `RUNNER_ENVIRONMENT_CONTEXT: ${{ runner.environment }}`,
   `RUNNER_OS_CONTEXT: ${{ runner.os }}`,
   `RUNNER_ARCH_CONTEXT: ${{ runner.arch }}`, and
   `WORKFLOW_SOURCE_SHA: ${{ github.workflow_sha }}` without permitting
   caller-provided overrides.
4. Invoke the single same-job host-bundle command and validate the current job before
   acquisition, Docker build, measurement, or calibration.
5. Produce the phase's existing outputs.
6. Generate the canonical payload manifest.
7. Pack the deterministic evidence archive.
8. Revalidate the exact payload and require the manifest and archive to be no-follow
   regular files.
9. Generate one SLSA provenance attestation covering both the manifest and archive.
10. Copy the action's local `bundle-path` output to a stable artifact filename.
11. Revalidate the exact payload and require the manifest, archive, and copied bundle
    to be present at their literal no-follow paths.
12. In one success-artifact upload, publish the complete payload root, which already
    contains the manifest, plus the sibling archive and local attestation bundle.

The runner and workflow context values are injected only through the observation
step's YAML `env:` mapping:

```yaml
- name: Observe and validate the current native work job
  env:
    RUNNER_ENVIRONMENT_CONTEXT: ${{ runner.environment }}
    RUNNER_OS_CONTEXT: ${{ runner.os }}
    RUNNER_ARCH_CONTEXT: ${{ runner.arch }}
    WORKFLOW_SOURCE_SHA: ${{ github.workflow_sha }}
```

The custom names are deliberate. GitHub's default `RUNNER_*` variables cannot be
overridden, so the workflow must not assign a context expression to
`RUNNER_ENVIRONMENT`, `RUNNER_OS`, or `RUNNER_ARCH`. The host preflight independently
reads each runner-supplied default and its corresponding `*_CONTEXT` value and requires
them to agree before normalizing the accepted value into the observation.

The workflow must not interpolate these contexts inside shell source or write them
through `$GITHUB_ENV` or `$GITHUB_OUTPUT`. The caller cannot supply or override the
step-level context values.

The workflow grants only the permissions required for checkout and file provenance:

```yaml
permissions:
  contents: read
  id-token: write
  attestations: write
```

It does not grant `actions: read`. `push-to-registry` is omitted and therefore keeps
its pinned-action default of `false`. The `create-storage-record` input applies only
when registry push is enabled, so this file-subject workflow neither sets that input
nor grants `artifact-metadata: write`.

All actions remain immutable-SHA pinned. The artifact-attestation action selected at
design time is:

```text
actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d  # v4.2.1
```

The implementation plan must revalidate the tag-to-commit mapping and action inputs
before editing the workflows. Neither a newer nor an older tag or action version
replaces this SHA without review.

## Same-job host observation

The sole production command replacing both `observe-github` and `finalize-github` is:

```text
uv run python -m tools.hpa320.oaf_host_attestation publish-github \
  --phase <bootstrap|measurement|candidate> \
  --output <payload-root>/<phase-host-bundle-directory>
```

The command reads the current job's GitHub defaults and the four reviewed step-level
context values directly. It accepts no `--observation`, `--github-output`, API-record,
job-ID, or token input, performs no GitHub REST request, and writes nothing to
`$GITHUB_OUTPUT`. It fails closed before acquisition, Docker build, measurement, or
calibration.

On success, it atomically publishes a previously absent host-bundle directory
containing exactly:

```text
attestation-bundle.json
native-host-evidence.json
native-host-observation.json
```

The observation, evidence, and bundle are first generated and cross-validated in a
private sibling staging directory. Publication is one no-replace directory rename;
failure removes the staging directory and leaves no partially acceptable output.
The old `observe-github` and `finalize-github` production paths are removed rather
than retained as compatibility aliases.

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

`github_job` is the `jobs.<job_id>` YAML mapping key exposed as `GITHUB_JOB`, not the
optional human-readable job `name:`. The workflow may keep a matching `name:` for
readability, but changing that display name does not change the evidence identity;
changing the YAML key does.

Every phase requires:

```text
RUNNER_ENVIRONMENT_CONTEXT == RUNNER_ENVIRONMENT == github-hosted
RUNNER_OS_CONTEXT == RUNNER_OS == Linux
RUNNER_ARCH_CONTEXT == RUNNER_ARCH == X64
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

## Host and containing-evidence schema migration

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

The strict `github_hosted` native-evidence payload has
`schema = "crux.github-hosted-native-evidence/v2"` and contains:

```text
schema, github_job, github_repository, github_run_attempt, github_run_id,
github_workflow_ref, github_workflow_sha, host_numeric_fingerprint, run_url,
runner_arch, runner_environment, runner_os, workflow_commit
```

It removes:

```text
api_record_sha256, approved_labels, job_id
```

Changing that exact nested payload and removing the candidate API-record row also
changes every existing containing schema. The migration therefore replaces these
five identities:

| Removed production identity | Replacement identity |
| --- | --- |
| `crux.backend-seal-evidence/v1` | `crux.backend-seal-evidence/v2` |
| `crux.oaf-base-system-package-evidence/v1` | `crux.oaf-base-system-package-evidence/v2` |
| `crux.oaf-calibration-bootstrap-evidence/v1` | `crux.oaf-calibration-bootstrap-evidence/v2` |
| `crux.oaf-calibration-measurement-evidence/v1` | `crux.oaf-calibration-measurement-evidence/v2` |
| `crux.oaf-seal-candidate/v1` | `crux.oaf-seal-candidate/v2` |

Their field sets otherwise remain as frozen by HPA-320 except for the changed nested
`native_host_evidence` payload and, for the seal candidate, removal of the
`native_host_api_record` artifact row. Producers, strict loaders, schema registries,
candidate validation, and goldens switch to these v2 identities in the same commit.

The request schema identities do not move, but the checked-in calibration-bootstrap
request cannot remain byte-identical. The dependency chain is:

1. `runtime/oaf_tf1/runner-source-manifest.json` covers the native bootstrap
   workflow, the changed HPA-320 attestation and candidate tools, and the runtime
   validators that enforce the migrated schemas.
2. `runtime/oaf_tf1/build-context-manifest.json` contains those exact source bytes
   and the runner source manifest bytes.
3. The `crux.oaf-calibration-bootstrap-request/v1` record cross-hashes both
   manifests through `runner_source_manifest_sha256` and
   `build_context_manifest_sha256`.

After all source changes are final, the migration regenerates the runner source
manifest, regenerates the build-context manifest using the exact authenticated
wheelhouse, and reissues
`config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json`
under its existing `/v1` request schema with the new manifest hashes. The old request
bytes remain in Git history but are superseded before dispatch and are never accepted
into a bootstrap evidence chain. The vendored upstream source manifest,
checkpoint-acquisition request, base-system-package request, distribution-build
manifest, and instrumentation patch remain byte-identical unless implementation
changes one of their independently frozen inputs.

A newly generated `crux.oaf-calibration-measurement-request/v1` names
`crux.oaf-calibration-measurement-evidence/v2` in its existing `output_schemas`
field, and the later seal-profile request references the resulting v2 evidence
hashes through its existing fields. No bootstrap evidence or downstream request has
been published, so this preserves the acyclic request/evidence order without
rewriting accepted input authority.

For v2, `run_url` is the run-level URL with exact grammar:

```text
https://github.com/<owner>/<repository>/actions/runs/<positive-decimal-github_run_id>
```

Its owner/repository path must equal `github_repository`, and its terminal decimal
component must equal `github_run_id`. It has no job suffix and no attempt suffix;
`github_job` and `github_run_attempt` carry those identities separately. The
v2-only parser rejects the former job-level URL grammar.

`tools/hpa320/github_host_evidence.py` remains the sole builder for the strict
`github_hosted` evidence record. Its v2 builder consumes an already validated
same-job observation and no longer accepts raw API bytes or a numeric Jobs API ID.
It constructs the exact v2 payload, reproduces its canonical SHA-256, and validates
the result through `NativeHostEvidence` before publication.

The bundle can be generated before the native work because it contains no output
digest. During production it is a phase-owned host claim embedded by hash in the
existing phase result. It becomes acceptable authority only when the final signed
artifact manifest binds that bundle and the external GitHub attestation passes the
verification policy.

A standalone v2 bundle must therefore never authorize sealing, publication, or
official execution. Consumer paths require the signed manifest and Sigstore bundle in
addition to the internal v2 bundle.

No durable v1 bootstrap, measurement, candidate, or seal evidence exists in the
repository or its Git history; the existing v1 examples are schema goldens and test
fixtures. The migration does not retain a production v1 parser. For
`kind == "github_hosted"`, `NativeHostEvidence` accepts only the exact v2 payload
after the migration. Existing v1 success goldens are replaced by v2 goldens. Old v1
bytes may remain only as clearly named negative fixtures proving rejection; no
runtime entry point loads them successfully.

## Canonical artifact manifest

The phase payload roots are fixed:

| Phase | Payload root | Host-bundle directory |
| --- | --- | --- |
| bootstrap | `artifacts/benchmark/backends/hpa320-bootstrap/` | `bootstrap-host-attestation/` |
| measurement | `artifacts/benchmark/backends/hpa320-measurement/` | `measurement-host-attestation/` |
| candidate | `artifacts/benchmark/backends/hpa320-candidate/` | `candidate-host-attestation/` |

Existing nested phase outputs remain below their corresponding root. For example,
bootstrap keeps `calibration-image/`, checkpoint-acquisition evidence, and
base-system-package evidence below `hpa320-bootstrap/`; measurement and candidate
retain their existing operational-image and result subtrees.

Each successful phase produces
`<payload-root>/artifact-manifest.json` under schema:

```text
crux.oaf-native-work-artifact-manifest/v1
```

It replaces `<payload-root>/artifact-sha256s.txt`; the legacy text checksum file is
not emitted or accepted after migration. The manifest has exactly:

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

Rows are sorted by UTF-8 bytes of `(role, path)`. Paths are POSIX paths relative to
the phase payload root and preserve nested subdirectories; they do not repeat the
`artifacts/benchmark/backends/hpa320-<phase>/` prefix. They cannot be empty or
absolute, contain `.` or `..` segments, use backslashes, collide, escape through a
symlink, or name the manifest, archive, or external Sigstore bundle.

`role` is not an open string. The v1 manifest role vocabulary is exactly:

```text
advisory_snapshot
backend_lock
base_system_package_evidence
calibration_bootstrap_evidence
calibration_measurement_evidence
checkpoint_acquisition_evidence
conversion_audit
diagnostic
host_adapter_source_manifest
native_host_attestation_bundle
native_host_evidence
native_host_observation
oci_layout_archive
oci_layout_manifest
runtime_lock
seal_candidate_manifest
seal_evidence
security_scan
smoke_audio
smoke_oracle
smoke_prediction
tensor_coverage
```

Unknown roles are rejected. `native_host_api_record` is deliberately absent and
cannot appear in a v2 phase manifest.

The shared vocabulary does not by itself authorize a file. Each phase producer owns
an exact `(payload-root-relative path, role)` allowlist:

- bootstrap permits the three bootstrap host-bundle paths, operational checkpoint
  acquisition evidence, the calibration-image bootstrap/OCI outputs,
  base-system-package evidence, and explicitly named successful diagnostics;
- measurement permits the three measurement host-bundle paths, operational
  checkpoint acquisition evidence, the operational-image bootstrap/OCI outputs,
  calibration-measurement evidence, and explicitly named successful diagnostics; and
- candidate permits the three candidate host-bundle paths, operational checkpoint
  acquisition evidence, the operational-image bootstrap/OCI outputs, the seal
  candidate subtree described below, and explicitly named successful diagnostics.

`calibrate --output <payload-root>/seal-candidate` writes a repository-shaped tree
inside that output directory. `_CANDIDATE_ARTIFACTS` paths are relative to the
seal-candidate directory, not to the phase payload root. The outer candidate phase
mapping therefore uses this exact prefix transformation:

```text
payload path = "seal-candidate/" + candidate repository-relative path
role         = existing candidate role
```

The v2 payload-root-relative seal-candidate mapping is exactly:

| Payload-root-relative path | Role |
| --- | --- |
| `seal-candidate/candidate-manifest.json` | `seal_candidate_manifest` |
| `seal-candidate/docs/superpowers/evidence/hpa-320/legacy-conversion-audit.json` | `conversion_audit` |
| `seal-candidate/docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/attestation-bundle.json` | `native_host_attestation_bundle` |
| `seal-candidate/docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/native-host-evidence.json` | `native_host_evidence` |
| `seal-candidate/docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/native-host-observation.json` | `native_host_observation` |
| `seal-candidate/runtime/oaf_tf1/host-adapter-source-manifest.json` | `host_adapter_source_manifest` |
| `seal-candidate/docs/superpowers/evidence/hpa-320/oaf-tensor-coverage.json` | `tensor_coverage` |
| `seal-candidate/docs/superpowers/evidence/hpa-320/oaf-advisory-snapshot.json` | `advisory_snapshot` |
| `seal-candidate/docs/superpowers/evidence/hpa-320/oaf-security-scan.json` | `security_scan` |
| `seal-candidate/artifacts/benchmark/backends/oaf-tf1/runtime.oci.tar` | `oci_layout_archive` |
| `seal-candidate/docs/superpowers/evidence/hpa-320/oaf-oci-layout-manifest.json` | `oci_layout_manifest` |
| `seal-candidate/tests/fixtures/oaf_tf1_smoke/canonical.wav` | `smoke_audio` |
| `seal-candidate/docs/superpowers/evidence/hpa-320/oaf-smoke-prediction.jsonl` | `smoke_prediction` |
| `seal-candidate/tests/fixtures/oaf_tf1_smoke/smoke-oracle.json` | `smoke_oracle` |
| `seal-candidate/config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json` | `seal_evidence` |
| `seal-candidate/config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.runtime-lock.json` | `runtime_lock` |
| `seal-candidate/config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.backend-lock.json` | `backend_lock` |

The former
`seal-candidate/docs/superpowers/evidence/hpa-320/native/candidate-host-attestation/github-job-api-record.json.hex`
path is absent. The three phase-level `candidate-host-attestation/` rows remain
separate from the repository-shaped copies inside `seal-candidate/`; both locations
must match their exact allowlisted roles and expected bytes.

The concrete mappings live as immutable phase constants in
`tools/hpa320/oaf_native_artifacts.py` and are asserted path-for-path by tests.
Diagnostics are allowed only at deterministic phase-specific paths below
`diagnostics/`; the producer cannot classify an unexpected file as `diagnostic` to
bypass the allowlist. A role valid for one phase or path remains invalid everywhere
else unless that phase's mapping names it.

The manifest covers every other regular payload file, including:

- `attestation-bundle.json`;
- `native-host-evidence.json`;
- `native-host-observation.json`;
- phase result and validation records;
- the phase-emitted OCI layout archive and its identity records where applicable; and
- sanitized diagnostics retained by the successful run.

The producer starts from the selected phase's exact mapping, scans the payload after
phase work, and fails on an unlisted, missing, multiply listed, non-regular, or
changed file. It never infers a role from a filename. The consumer selects the same
phase mapping and reproduces every row from the extracted payload before reading a
phase result.

## Deterministic evidence archive

The archive is an uncompressed POSIX ustar file named:

```text
artifacts/benchmark/backends/hpa320-native-<phase>-<workflow_commit>.tar
```

It is a sibling of, not a member of, the phase payload root. It contains
`artifact-manifest.json` and every file named by that manifest, mapped below the
archive's canonical root without changing their payload-root-relative paths. Packing
uses:

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

Every phase uses this exact input shape. `<phase>` is replaced by the workflow's
hard-coded phase; the workflow commit comes only from the required dispatch input:

```yaml
- name: Attest native phase evidence
  id: attest
  uses: actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d
  with:
    subject-path: |
      artifacts/benchmark/backends/hpa320-native-<phase>-${{ inputs.commit_sha }}.tar
      artifacts/benchmark/backends/hpa320-<phase>/artifact-manifest.json
```

`subject-path` is mandatory. The workflow does not rely on the action's
`$GITHUB_ARTIFACTS_LIST` subject discovery. `sbom-path`, `predicate`, and
`predicate-type` are absent, so the reviewed v4 invocation generates its default
SLSA provenance for those two explicit subjects.

Signing the manifest directly preserves a durable link to every payload digest even
after the convenience archive's GitHub retention window expires. Signing the archive
also proves that the reviewed download is the exact packaged payload.

The local Sigstore bundle returned by `bundle-path` is copied to:

```text
artifacts/benchmark/backends/hpa320-native-<phase>-<workflow_commit>.sigstore.json
```

That file is a sibling of the phase payload root and remains outside the signed
manifest and archive. The workflow uploads the complete payload root, manifest-bearing
archive, and detached Sigstore bundle as three non-overlapping paths. The action also
publishes the attestation to GitHub's attestation API and, for this public repository,
the public Sigstore transparency log.

Each phase uses exactly one `actions/upload-artifact` success invocation named:

```text
hpa320-native-<phase>-<workflow_commit>
```

Its `path` input contains exactly:

```text
artifacts/benchmark/backends/hpa320-<phase>/
artifacts/benchmark/backends/hpa320-native-<phase>-<workflow_commit>.tar
artifacts/benchmark/backends/hpa320-native-<phase>-<workflow_commit>.sigstore.json
```

The manifest is not listed separately because it is already inside the first path.
The three inputs share `artifacts/benchmark/backends/` as their upload root, so a
download contains one `hpa320-<phase>/` payload directory and the two sibling files;
it must not contain an extra nested copy of the payload root, manifest, tar, or
Sigstore bundle. The invocation sets:

```text
if-no-files-found: error
retention-days: 30
```

It uses no wildcard and no second success upload. Workflow-contract tests freeze the
artifact name, the three literal path shapes, the retention value, and the facts that
the tar and Sigstore bundle are outside the payload root and absent from every
manifest row.

`if-no-files-found: error` is only an aggregate final guard: it fails when none of
the configured paths yields a file, not when one of the three inputs is missing. The
real completeness guarantee is ordered before upload. The Python-owned verifier
reproduces the exact manifest scan and canonical archive, rejects a symlinked payload
root, and requires the manifest and archive to be no-follow regular files immediately
before attestation. After the action returns and its bundle is copied, the workflow
runs the same payload verification again, requires the payload root to be a
non-symlink directory, and requires the manifest, archive, and detached bundle to be
no-follow regular files at their literal paths. Any missing or changed input fails
the job before `actions/upload-artifact`.

This exclusion is intentional: the bundle is the detached signature, certificate,
timestamp, and transparency-log material used to verify the two subjects, so placing
it inside either subject would create a cycle. It is not self-authenticating in
isolation. Tampering is detected only when a verifier checks it against trusted roots,
the expected GitHub OIDC issuer and certificate identity, the transparency proof, and
the exact subject digests.

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
the reviewer runs GitHub CLI verification against both signed subjects. The supported
CLI range is:

```text
2.68.0 <= gh < 3.0.0
```

Version 2.68.0 is the first GitHub CLI release containing both
`--source-digest` and `--signer-digest`. The reviewer records `gh version` and
rejects an older or new-major client rather than silently dropping unsupported
policy flags.

`gh attestation verify` accepts one artifact path per invocation. Acceptance
therefore requires these two invocations, with the same phase, commit, local bundle,
trusted root, and policy flags:

```text
gh attestation verify <artifact-manifest-path> \
  --repo cwchanap/Crux \
  --signer-workflow cwchanap/Crux/.github/workflows/hpa320-native-<phase>.yml \
  --source-digest <workflow_commit> \
  --signer-digest <workflow_commit> \
  --deny-self-hosted-runners \
  --digest-alg sha256 \
  --predicate-type https://slsa.dev/provenance/v1 \
  --cert-oidc-issuer https://token.actions.githubusercontent.com \
  --bundle <sigstore-bundle-path> \
  --custom-trusted-root <trusted-root-path> \
  --format json

gh attestation verify <archive-path> \
  --repo cwchanap/Crux \
  --signer-workflow cwchanap/Crux/.github/workflows/hpa320-native-<phase>.yml \
  --source-digest <workflow_commit> \
  --signer-digest <workflow_commit> \
  --deny-self-hosted-runners \
  --digest-alg sha256 \
  --predicate-type https://slsa.dev/provenance/v1 \
  --cert-oidc-issuer https://token.actions.githubusercontent.com \
  --bundle <sigstore-bundle-path> \
  --custom-trusted-root <trusted-root-path> \
  --format json
```

Both commands must succeed and report the same verified attestation statement, whose
subject array contains both expected SHA-256 identities. A pass for only one subject
is insufficient.

Both digest flags deliberately receive the same value. This is valid only because the
producer preflight has already required:

```text
github.workflow_sha == GITHUB_SHA == checked-out HEAD == COMMIT_SHA
```

The equality is an invariant of these three workflows, not a convenient default. If a
future dispatch runs the workflow definition from a different commit than the checked
out source, production fails. Supporting divergent signer and source commits would
require a new reviewed design and separate `--signer-digest` and `--source-digest`
authorities; an implementer must not loosen this policy in place.

The saved local bundle is supplied with `--bundle` for durable local-bundle
verification. The trusted root is obtained out of band through
`gh attestation trusted-root`, transferred independently, and supplied explicitly to
both commands. The local bundle alone is not described as a complete offline trust
store. Online lookup may be performed as a second comparison but is not the sole
authority.

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
- Partial sanitized diagnostics may still be uploaded under exactly
  `hpa320-native-<phase>-diagnostic-<workflow_commit>-run-<github_run_id>-attempt-<github_run_attempt>`.
  That name cannot equal the success artifact name, the diagnostic upload contains no
  `status: success` artifact manifest, archive, or Sigstore bundle, and it is never
  relabeled as accepted evidence. When present, the diagnostic upload contains only
  explicitly allowlisted sanitized files and uses `retention-days: 30`.
- Manifest or deterministic-packing failure fails the job before attestation.
- Attestation or upload failure fails the job; unsigned phase results are not
  accepted.
- A valid attestation with an invalid manifest or phase result is rejected.
- A valid phase result with a missing, invalid, wrong-workflow, wrong-commit, or
  self-hosted attestation is rejected.
- Any retained ephemeral v1 workflow artifacts are preserved as audit material only.
  They are not rewritten to v2 and migrated production readers reject them.

## Migration

The migration is atomic across tooling, schemas, consumers, tests, documentation, and
all three manual workflows:

1. Add the v2 host bundle, exact v2 GitHub-hosted evidence payload, five v2
   containing-evidence identities, artifact manifest, deterministic packer, and
   verification support.
2. Replace the v1 success goldens and schema-registry rows with v2 goldens; retain
   old v1 bytes only where a named negative fixture proves rejection.
3. Change all producers and acceptance consumers to the v2 identities and reject an
   unaccompanied v2 bundle.
4. Convert bootstrap, measurement, and candidate workflows to their single-job form.
5. Remove the Jobs REST finalizer, API-record decoder, raw-record schema fields,
   cross-job output handling, and `actions: read`; delete
   `.github/workflows/hpa320-native-host-evidence.yml`.
6. Update the HPA-320 design's superseded provenance and schema passages.
7. After every covered source byte is final, regenerate
   `runtime/oaf_tf1/runner-source-manifest.json`, then regenerate
   `runtime/oaf_tf1/build-context-manifest.json` from the exact authenticated
   wheelhouse, then reissue the checked-in calibration-bootstrap request with the
   resulting hashes. Strict-load the request and exact-compare every referenced
   current file before committing.
8. Run the local verification stack.
9. Merge HPA-481 and HPA-482 together.
10. Retire `hpa-320-native-seal-evidence` or disable its legacy workflow in GitHub;
   deleting the file from the default branch alone does not disable the workflow copy
   retained on that branch.
11. Only then dispatch HPA-423 at an exact merged commit.

No production or test success path retains v1 compatibility. No compatibility mode
may silently infer `runner_environment`, accept an API record for another job, or
treat v1 and v2 as equivalent.

## Verification strategy

### Unit tests

- Observation accepts the exact GitHub-hosted/Linux/X64 environment.
- Observation rejects missing, empty, self-hosted, wrong-OS, and wrong-architecture
  runner values before Docker work.
- Observation rejects disagreement between each runner context value and its
  corresponding runner-supplied default.
- Observation rejects wrong job, workflow ref, workflow source SHA, repository, run
  identity, or commit.
- V2 evidence accepts only the exact run-level `run_url` grammar and rejects the v1
  job-level form.
- V2 bundle and all five v2 containing-evidence parsers are strict and reject v1
  substitution at every migrated boundary.
- Generic GitHub-hosted native evidence requires `runner_environment`.
- No v1 GitHub-hosted native-evidence success parser remains.
- The checked-in runner source manifest reproduces the final covered workflow,
  tooling, and runtime-validator bytes exactly.
- The checked-in build-context manifest reproduces from the final repository and
  authenticated wheelhouse exactly and contains the regenerated runner source
  manifest bytes.
- The reissued calibration-bootstrap request strict-loads and exact-cross-hashes the
  current runner source manifest, build-context manifest, upstream source manifest,
  checkpoint request, base-system request, distribution manifest, and
  instrumentation patch.
- Manifest parsing rejects noncanonical JSON, extra or missing keys, path escape,
  duplicates, wrong ordering, missing files, size/hash drift, self-reference, an
  unknown role, and a valid role at a phase-disallowed path.
- Candidate manifest tests assert every payload-root-relative row in the explicit
  `seal-candidate/` mapping and reject the removed API-record path.
- Deterministic packing reproduces byte-identical archives and rejects unsafe member
  types or metadata.
- External verification policy requires repository, phase workflow, source/signing
  commit, SLSA predicate, GitHub issuer, denial of self-hosted runners, a supported
  GitHub CLI version, and separate passes for both signed subjects.
- Cross-validation rejects a valid signature over internally inconsistent evidence.

### Workflow contract tests

Each phase workflow test requires:

- exactly one `jobs.<job_id>` YAML key with the expected `native-*` identity;
- `GITHUB_JOB` validation against that key rather than the optional `name:` value;
- no `observe-native-host`, `needs`, `GH_TOKEN`, or `actions: read`;
- exactly one `publish-github` invocation and no `observe-github`,
  `finalize-github`, `--github-output`, or Jobs API input;
- explicit `runner.environment`, `runner.os`, and `runner.arch` propagation through
  the three custom `*_CONTEXT` variables, with no attempted `RUNNER_*` override;
- host preflight before any native work;
- manifest and archive generation after successful native work;
- `actions/attest` pinned to the reviewed v4.2.1 immutable SHA in that same job;
- `contents: read`, `id-token: write`, and `attestations: write`;
- the exact two-line `subject-path` block for archive and manifest, with no implicit
  `$GITHUB_ARTIFACTS_LIST` discovery;
- ordered no-follow subject checks before attestation and exact payload,
  manifest, archive, and bundle checks before upload;
- one success upload with the exact success name, three non-overlapping paths,
  aggregate `if-no-files-found: error`, and `retention-days: 30`;
- the payload-root-relative manifest excludes the sibling tar and Sigstore bundle;
- a diagnostic upload name that cannot collide with the success artifact; and
- no bootstrap, measurement, or candidate acceptance before attestation succeeds.

A repository-wide workflow test also requires that
`.github/workflows/hpa320-native-host-evidence.yml` is absent.

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

The current `.github/workflows/ci.yml` commands are the acceptance contract. The
implementation finishes with their local `uv` equivalents:

```text
uv run pytest
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

The repository's `AGENTS.md` additionally requires:

```text
uv run ruff check src tests
uv run black --check src tests
uv run pylint src/app src/cli
```

Because CI's formatter scope excludes `tools/`, the new and changed HPA-320 tool
modules also receive focused checks:

```text
uv run ruff format --check tools/hpa320/oaf_host_attestation.py tools/hpa320/github_host_evidence.py tools/hpa320/oaf_native_artifacts.py
uv run pylint --errors-only --disable=E1120,E0401 tools/hpa320/oaf_host_attestation.py tools/hpa320/github_host_evidence.py tools/hpa320/oaf_native_artifacts.py
```

These scopes intentionally differ because they mirror their owning CI or repository
contract. They must not be normalized silently. If either contract changes before
implementation, the plan records the exact replacement command and source.

## Repository ownership

- `tools/hpa320/oaf_host_attestation.py` owns same-job observation and strict v2 host
  bundle validation.
- `tools/hpa320/github_host_evidence.py` owns strict v2 `github_hosted` evidence
  construction from a validated same-job observation; it owns no GitHub API access.
- `tools/hpa320/oaf_native_artifacts.py` owns canonical manifest generation,
  deterministic archive packing, and structural verification; workflow shell must
  not reimplement these rules.
- `src/benchmark/backend_process.py` owns strict generic
  v2-only `github_hosted` native-evidence validation; it owns no v1 compatibility
  entry point.
- `src/benchmark/backend_lock.py`, `tools/hpa320/oaf_system_packages.py`, and
  `tools/hpa320/seal_oaf_backend.py` own their exact v2 containing-evidence identities
  and reject the replaced v1 identities.
- `tools/hpa320/seal_oaf_backend.py` also owns phase-specific producer and consumer
  gates.
- `tools/hpa320/oaf_native_calibration.py`,
  `tools/hpa320/oaf_candidate_builder.py`,
  `runtime/oaf_tf1/calibration_entrypoint.py`, and
  `runtime/oaf_tf1/oaf_backend.py` own the corresponding native-runtime production
  and strict-consumption expectations for those v2 identities; they migrate in the
  same commit.
- The bootstrap, measurement, and candidate files under `.github/workflows/` own
  ordered invocation, least privileges, pinned actions, and artifact publication.
- `tests/benchmark/` owns schema, tamper, workflow-contract, and failure-path coverage.
- `docs/superpowers/evidence/hpa-320/native/` owns accepted signed manifests, local
  Sigstore bundles, v2 host bundles, and their owning phase evidence after review.

## Supersession of the HPA-320 design

For bootstrap, measurement, and candidate production after this migration, this
document supersedes the conflicting provenance passages under these exact HPA-320
design headings:

- `Native-amd64 calibration and sealing prerequisite`;
- `Calibration-bootstrap authority`;
- `Calibration, candidate publication, and seal`;
- `Verification and Execution Flow`;
- `Normative Schema Key Reference` > `Provenance and report schemas`; and
- `Continuous Integration`.

Within those passages, the superseded requirements are:

- a separate `observe-native-host` job;
- a completed `observe-native-host` Jobs API record;
- `github-job-api-record.json.hex`;
- `crux.oaf-native-host-attestation-bundle/v1` as new phase authority; or
- a four-file phase host bundle containing that API record;
- `crux.backend-seal-evidence/v1`,
  `crux.oaf-base-system-package-evidence/v1`,
  `crux.oaf-calibration-bootstrap-evidence/v1`,
  `crux.oaf-calibration-measurement-evidence/v1`, or
  `crux.oaf-seal-candidate/v1` as identities for newly produced native phase
  evidence.

All unrelated HPA-320 contracts remain in force. The HPA-320 design should receive a
short amendment under each affected heading naming the replaced paragraph by its
unique opening sentence and pointing to this document. Line numbers are not used
because later editorial changes would make them stale.

## Operational gate

HPA-423 remains blocked until:

1. this design is approved;
2. its implementation plan is approved;
3. implementation and verification complete;
4. HPA-481 and HPA-482 merge;
5. the legacy workflow is disabled or its trigger branch is retired; and
6. the exact merged commit is confirmed.

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
- GitHub CLI 2.68.0 digest-policy release:
  <https://github.com/cli/cli/releases/tag/v2.68.0>
- `gh attestation trusted-root`:
  <https://cli.github.com/manual/gh_attestation_trusted-root>
- `gh workflow run` dispatch-ref contract:
  <https://cli.github.com/manual/gh_workflow_run>
- GitHub Actions default-variable contract:
  <https://docs.github.com/en/actions/reference/workflows-and-actions/variables>
- `actions/upload-artifact` multiple-path contract:
  <https://github.com/actions/upload-artifact/tree/v4#upload-using-multiple-paths-and-exclusions>
