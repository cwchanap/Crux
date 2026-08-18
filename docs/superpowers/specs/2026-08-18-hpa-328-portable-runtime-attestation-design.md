# HPA-328 Portable Separator Runtime Attestation

## Status

Approved design for implementation planning. This extends the HPA-328
freeze-time separator lock into a portable, live runtime-attestation contract.
It does not authorize Task 11 production locks, separator inference, scoring,
or finalization until the immutable upstream inputs are available.

## Goal

Before a Spleeter or HTDemucs stem can be reused from cache or attributed to a
pilot run, prove that all of the following match the committed lock:

1. the resolved isolated Python interpreter;
2. the complete installed-distribution environment used by that interpreter;
3. the explicit, package-specific model root that the separator will load.

The contract must remain portable: committed artifacts contain hashes and
normalized relative paths, never machine-local absolute paths.

## Non-goals

- Do not run real separators, generate Task 11 locks, or alter pilot outputs
  while implementing this feature.
- Do not add a generic plugin, RPC, database, or framework layer.
- Do not change HPA-328 run, comparison, or handoff schemas solely to record
  host-local roots.
- Do not claim that a pip-installed package proves the lock's Git revision.
  `repository_revision` remains freeze-time provenance metadata; verified code
  identity is the package/environment content hash.

## Persisted artifacts

### Separator lock v2

Replace `crux.separator-lock/v1` with `crux.separator-lock/v2`. A v2 lock is a
canonical JSON file and remains the cache/published identity. It adds:

- `interpreter_sha256`: SHA-256 of the resolved executable bytes;
- `environment_manifest_sha256`: SHA-256 of the canonical sibling environment
  manifest;
- `model_root_kind`: one policy-controlled value per separator;
- model files expressed only as normalized paths relative to the supplied model
  root.

The loader intentionally rejects v1 locks. The lock's SHA therefore prevents
reuse of v1 cache entries.

### Sibling environment manifest

Each lock directory contains a canonical `environment.json` with schema
`crux.separator-environment/v1`. It contains no absolute paths and records:

- separator ID, package name/version, Python implementation/version, ABI, and
  platform identity;
- the resolved interpreter SHA-256;
- every installed distribution discoverable from the isolated interpreter;
- for each distribution, a canonical `RECORD` inventory of normalized relative
  file names, byte sizes, and SHA-256 hashes, plus the `RECORD` file hash.

The lock stores the manifest hash; runtime resolves only the exact sibling file
and verifies its canonical bytes and SHA before comparing a freshly computed
manifest.

### Model-root layouts

`model_root_kind` is a closed policy value, not user-controlled metadata.

| Separator | Root kind | Required use |
| --- | --- | --- |
| Spleeter 2.4.2 | `spleeter-model-path-v1` | Root passed as absolute `MODEL_PATH`; fixed `spleeter:4stems` layout includes `4stems/checkpoint`, `.probe`, `model.index`, `model.data-00000-of-00001`, and `model.meta`. |
| Demucs 4.1.0 | `demucs-hf-hub-cache-v1` | Root passed as absolute `HF_HUB_CACHE` with offline mode. It contains the materialized `models--adefossez--HTDemucs` cache, an exact `refs/main`, and the matching snapshot files. |

All attested model files are regular files beneath the supplied root. The
Demucs cache is materialized before freezing; symlinks, cache escapes, legacy
`--repo`, and network/remote fallback are rejected.

## Freeze flow

`scripts/freeze_separator_runtime.py` evolves to require:

- an isolated interpreter;
- an explicit model root;
- repeated root-relative model file names;
- a caller-supplied repository revision for provenance;
- paths for `model.json` and its sibling `environment.json`.

It resolves the interpreter, builds the canonical full-environment manifest
from the isolated interpreter, validates every model file beneath the root,
then writes both immutable artifacts and round-trips them through their
loaders. The freezer never installs packages, downloads models, or runs
inference.

## Runtime flow

`OafSeparationPilotRequest` gains required Spleeter and Demucs model roots.
The pilot CLI gains matching required options while retaining existing
interpreter options.

For every fresh and resumed derived view, before any cached stem is read or
any separator process starts, the runner:

1. loads the v2 lock and exact sibling environment manifest;
2. resolves and hashes the supplied interpreter;
3. invokes a fixed isolated probe to rebuild the environment manifest from
   the installed distribution `RECORD` inventories;
4. verifies the package version, environment manifest, and every root-relative
   model file;
5. validates the separator-specific root layout;
6. launches the fixed separator command with only the approved root
   environment variables.

The environment probe uses only standard-library code and descriptor-relative,
no-follow file traversal. It rejects unexpected missing or unhashed `RECORD`
entries, parent/leaf symlinks, traversal, malformed metadata, and nondeterministic
manifest content.

An attestation failure occurs before cache access, emits a native separator
attestation failure code, and retains no newly attributed stem/input/prediction.
The other fixed derived view may continue under existing partial-run semantics.
Resume paths attest again even if a valid stem already exists.

## Identity and reporting

Model roots and interpreter locations are host-local runtime inputs. They must
not appear in run identity, snapshots, cache keys, comparison reports, or
handoff rows. The v2 lock SHA already binds the verified environment/model
identity and continues to be the persisted provenance field.

`crux.oaf-separation-run/v1`, `crux.oaf-separation-comparison/v1`, and the
handoff schema remain unchanged. Existing source/input identity, full-mix
non-rerun, failure histogram, and resource-reporting contracts remain intact.

## Test plan

Tests use synthetic isolated-environment fixtures and never invoke the retained
production separator runtimes. Required coverage includes:

- canonical v2 lock/environment parsing, v1 rejection, and companion-file
  substitution resistance;
- interpreter, package-tree, `RECORD`, entrypoint, parent/leaf symlink, and
  model-file mutation detection;
- Spleeter fixed root propagation and required layout validation;
- Demucs exact Hugging Face cache/revision validation, offline propagation,
  and rejection of legacy/remote fallback;
- attestation before cache hits and on retained-stem resume paths;
- CLI/freezer option wiring, no host roots in persisted artifacts, and
  continuation/error mapping;
- regression coverage for existing HPA-328 schema, comparison, and handoff
  behavior.

## Task 11 impact

When the upstream HPA-321/323/324/326/327 inputs become available, Task 11
will first materialize/verify the model roots, generate and commit each v2
`model.json` plus `environment.json`, and only then run the fixed pilot. The
later explicit decision/rationale gate for finalization remains unchanged.
