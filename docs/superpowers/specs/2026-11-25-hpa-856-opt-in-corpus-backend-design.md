# HPA-856 opt-in TF2.15 corpus backend

## Evidence and decision

The five-song comparison repeated on two Colab CPU VMs. The subsequent
twenty-song panel passed its preregistered gate with identical raw 50 ms F1
counts and 35,608 strict native matches. The candidate has measurable
confidence differences and its binary64 smoke output differs from the frozen
TF1 oracle. These results support testing the candidate as a separately
identified corpus backend. [Experiment results](../../../experiments/oaf_tf215/RESULTS.md#twenty-song-panel)

The proposed next milestone is an explicit TF2.15 option on the existing
corpus benchmark CLI, followed by a two-song Docker-free Colab run and a
same-runtime resume. This work belongs in
[draft PR #36](https://github.com/cwchanap/Crux/pull/36).

| Approach | What it proves | Tradeoff |
| --- | --- | --- |
| Opt-in backend on the existing corpus runner | The normal CLI can infer, score, write canonical artifacts and resume under the candidate identity. | Requires deliberate changes at the benchmark's descriptor, mapping and cache boundaries. |
| Wider experiment using the existing panel driver | More reference F1 and event-drift data before changing benchmark code. | Leaves the CLI, canonical prediction writer and resume behavior untested. |

## Command and worker

Add an explicit `crux benchmark run-oaf-corpus --backend oaf-tf215` mode.
The default backend remains the frozen Docker worker. The candidate mode
requires paths to the verified Python 3.11 environment and built scratch
runtime. Setup uses the existing hash lock, scratch builder and public
checkpoint verifier. Carry failures through the existing CLI result and
corpus row diagnostics. Expose bounded preflight diagnostics for either
backend.

Provision the pilot environment with a clean hash-checked lock install and
the existing builder, and retain the setup logs. At runtime, reuse the
experiment's interpreter, distribution and scratch-file fingerprint checks.
Those checks identify the environment being used. The clean installation
and build logs supply its provisioning provenance.

Use the current OaF adapter's path checks, request/response decoder and
persistent worker cleanup. Launch the candidate directly with `python -s`
and the verified checkpoint directory. Set the same sanitized environment
used during measurement, and use the canonical input directory as the
worker's working directory. Validate the worker's existing protocol backend
ID and the exact integer restore count of 78 in the shared adapter. The host
publishes the separate TF2.15 candidate descriptor already measured in the
experiment. Keep launch and fingerprint helpers in the installed `src` package,
with the scratch builder and lock supplied as checkout artifacts.

## Artifacts and resume

Register the candidate descriptor and OaF event metadata in the benchmark
prediction reader and writer. Give the candidate its own map ID and reuse
the verified 8-hit taxonomy mapping. Keep the TF1 descriptor, map, inference
configuration schema and serialization stable. A candidate whose normalized
events collide on a canonical sort key receives an explicit row failure.

Fingerprint the worker once during preflight and store the identity in the
candidate run header. The fingerprint covers the interpreter, installed
distribution names and versions, host CPU and platform, captured environment,
requirements lock, frozen model config, instrumentation and compatibility
patches, pristine converter, and scratch inventory. Host behavior is versioned
through the adapter, map and canonicalization identities. Record the checkout
revision separately as provenance.

The candidate-specific inference schema includes the runtime SHA-256, which
feeds the existing run ID and prediction path. Both modes use the existing
resume flow and select the appropriate map. The runner checks prior row
evidence when available and always validates the artifact against the current
canonical WAV, descriptor and map before reuse. A run can recover an artifact
published just before its checkpoint. `--resume` performs inference for a
missing prediction path. A present invalid artifact fails its row and stays
untouched.

The pilot uses a single writer and a runner-owned output root. During an
invocation, keep the venv, scratch tree and checkpoint fixed. The digest-keyed
path namespaces normal lookup. It relies on the runner owning publication
into that directory because prediction-v2 headers carry the descriptor and
audio identity, while the run and path carry the runtime binding.

The runner writes its ordinary reference scores and reports. Separation
handoff stays explicitly bound to its supported frozen backend until a
candidate-aware separation path has been validated. The app backend and
deployment defaults stay as they are.

## Verification

Add coverage for TF1 golden compatibility, candidate descriptor and prediction
round trips, separate runtime-derived cache paths, worker lifecycle,
and the existing resume behavior under the candidate map. Test rejection of
changed artifact bytes and canonical WAVs. A changed runtime must derive a
different run ID and cache path. A failed ready response or invalid canonical
event receives an explicit diagnostic. Run the full suite on supported Linux
x86-64 at each implementation checkpoint and before building the pilot bundle.

On fresh x86-64 Colab CPU, scope the normal CLI to simfiles 200 and 209 with
the existing hash-verified September manifests, all 180 eligible timing
artifacts and the panel audio cache. Expect 283 and 296 native events,
respectively, with raw 50 ms F1 of 0.397727 and 0.096192. Verify both
canonical artifacts through the normal reader and compare their scientific
outputs to the panel. Then resume with the same attested runtime and verify
both rows reuse their artifacts without launching the worker. Preserve the
first run snapshot and reports before resuming so both sets of dispositions,
artifact hashes and scores can be verified. A fake-factory test pins the
zero-worker behavior.
Retrieve reports and provenance and stop the VM.

Review the pilot evidence before choosing the broader corpus, reviewed
subset, separation and Drive continuation. Cross-VM Drive resume needs a
separate decision about which runtime fingerprints can share a revision.
