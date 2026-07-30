# Task 6 report: frozen OaF runner and frame instrumentation

Status: `DONE_WITH_CONCERNS`

Base: `2eda2f7`

## Delivered

- A stdlib-first CPython 3.7 entrypoint that validates the exact locked model
  environment before numeric imports, seeds Python/NumPy/TensorFlow, authenticates
  startup evidence, emits one canonical ready line, and serves requests sequentially.
- Strict canonical JSON, bounded output, stable diagnostics, no-follow mounted reads,
  runner-relative input paths, SHA-256 checks, and canonical PCM WAV validation.
- Exact OaF drums PREDICT construction/invocation, checkpoint component verification,
  the locked `130 = 78 + 52` tensor partition, restore/uninitialized proof, reachable
  stochastic-op rejection, smoke authentication, and raw native event output.
- A fixed-identity pure-Python patch applier. It authenticates the two exact upstream
  preimages and applies `capture-emitted-frame.patch` only to an image-build copy.
  The checked-in Task 5 vendor bytes and upstream source manifest remain unchanged.
- Patched/unpatched serialized `NoteSequence` parity, direct frame/pitch/raw-velocity
  capture, and confidence pairing from the selected onset-probability cell.
- Deterministic runner-source manifest generation. The final manifest was generated
  twice with identical SHA-256
  `a4cd1bec2554b750e7d304354b040c59e7660150d3a97c9d0e2c68e84d21f62c`.
- A provisional non-root `linux/amd64` test image with offline hashed wheels and
  CPython 3.7 execution. Final diagnostic image ID:
  `sha256:8d8ce1232af5592e8c4c74fbdfc0fbb4a294c794f37602db53c824c6ecb40ff4`.

## RED-GREEN evidence

The following failures were witnessed before the corresponding implementation or
correction.

1. Initial required RED:

   ```text
   rtk uv run pytest -q runtime/oaf_tf1/tests/test_protocol.py runtime/oaf_tf1/tests/test_instrumentation.py
   ```

   Collection failed with two `ModuleNotFoundError` errors because the entrypoint
   and instrumentation applier did not exist.

2. Initial protocol/instrumentation GREEN progression:

   ```text
   40 passed, 2 skipped, 5 failed
   44 passed, 2 skipped, 1 failed
   45 passed, 2 skipped
   ```

   The five then one remaining failures exposed WAV error classification, patch hunk
   count handling, and stable symlink diagnostics. The final focused command passed.

3. Tensor stochastic-reachability RED:

   ```text
   1 failed, 9 passed
   Failed: DID NOT RAISE ModelIntegrityFailure
   ```

   After enumerating stateless random operation types:

   ```text
   10 passed
   ```

4. Runner-manifest generator RED/GREEN:

   ```text
   collection error: No module named tools.hpa320.generate_runner_source_manifest
   15 passed, 2 skipped
   ```

5. First complete focused host GREEN:

   ```text
   56 passed, 2 skipped
   ```

6. Container layout and patch-publication REDs:

   - Initial container collection could not import the packaged runner modules.
   - The next run reached `55 passed, 1 skipped, 2 failed`; patched files had mode
     `0600` rather than the authenticated upstream modes.
   - The next run reached `55 passed, 1 skipped, 2 failed`; broad Magenta package
     import triggered an unwritable Numba cache and unavailable `libsndfile` in the
     intentionally minimal test target.

   Packaging the runner under its repository module path, preserving source modes,
   and loading only the two patched modules for parity produced:

   ```text
   57 passed, 1 skipped
   ```

7. Atomic runner-manifest permission RED:

   ```text
   1 failed
   assert 384 == 420
   ```

   After setting the temporary descriptor to `0644` before publication:

   ```text
   1 passed, 17 deselected
   ```

8. Exact final-image environment RED:

   ```text
   1 failed
   /opt/crux/venv/bin/python entrypoint differed from the required env -i bootstrap
   ```

   The structural test passed after adding the shell-free exact environment
   bootstrap.

9. Canonical request JSON RED:

   ```text
   1 failed
   Failed: DID NOT RAISE ProtocolFailure
   ```

   Re-encoding and comparing each physical request line before item validation
   produced:

   ```text
   1 passed, 32 deselected
   ```

10. Task 1-6 compatibility RED:

    ```text
    1 failed, 412 passed, 2 skipped
    ```

    The failure identified a required Docker source-copy boundary. Restoring that
    boundary produced `2 passed, 50 deselected`, followed by a green full suite and
    a final green exact compatibility run.

11. Container source-layout RED:

    ```text
    2 failed, 58 passed, 1 skipped
    ```

    The two failures were host-only Dockerfile/complete-repository checks. The
    permission test now uses a minimal source fixture and the Dockerfile shape test
    skips only when the Dockerfile is intentionally absent in the runtime test
    layout.

12. CPython 3.7 bootstrap RED:

    ```text
    code=process_environment_invalid count=1
    ```

    An empty CPython 3.7 environment coerces the locale and injects
    `LC_CTYPE=C.UTF-8`; `-I` would also ignore the locked `PYTHONHASHSEED`. A focused
    unit RED first failed collection because the bootstrap-removal helper did not
    exist. The image now supplies only the reviewed bootstrap-only
    `PYTHONCOERCECLOCALE=0`, deletes it before exact seven-variable validation, and
    uses `-s` so the hash seed is honored:

    ```text
    2 passed, 32 deselected
    ```

## Final verification

- Exact Task 1-6 compatibility command:

  ```text
  414 passed, 2 skipped in 29.87s
  ```
- Full host suite:

  ```text
  1519 passed in 47.79s
  ```

- Final provisional image build:

  ```text
  linux/amd64, uid=10001, gid=10001
  sha256:8d8ce1232af5592e8c4c74fbdfc0fbb4a294c794f37602db53c824c6ecb40ff4
  ```

- Exact non-root CPython 3.7 image suite:

  ```text
  60 passed, 2 skipped in 3.88s
  ```

  Both skips are explicit host-source checks; all protocol, tensor, patch-applier,
  patched conversion, confidence-pairing, and manifest-permission tests executed.

- CPython 3.7 compilation: passed with bytecode directed to writable `/tmp`.
- Final image checks: `amd64 linux 10001:10001`; manifest readable at mode `0644`;
  two fresh `PYTHONHASHSEED=0` processes produced the same hash; missing startup
  mounts produced no stdout and only
  `code=mounted_identity_invalid count=1` on stderr.
- Ruff: passed.
- Black: 8 files unchanged.
- Targeted Pylint: `10.00/10`; `duplicate-code` was disabled because the reviewed
  no-follow/canonical helpers intentionally mirror one another across the isolated
  image-build scripts.
- `git diff --check`: passed.
- Runtime and test wheelhouses materialized exactly:
  - runtime: 71 distributions, lock SHA-256
    `9e00c42066a72c673051e65404d28c0eb7fe2833b1266db6e451395af3fa1457`
  - test: 9 distributions, lock SHA-256
    `574406e35b7c226b1f4dbd3decd9fe6cf7a2ccd75b98b28923cc466e63fcb193`

## Remaining gates and concern

- Task 8 still owns the final runtime lock, seal evidence, smoke oracle,
  tensor-coverage evidence, final UID/GID/resources, final image/OCI identities,
  security/advisory evidence, real checkpoint inference, and native-amd64 result.
  The emulated provisional image is diagnostic only and is not seal evidence.
- Task 7 integration must make the authenticated seal-evidence file available at
  `/run/crux/seal-evidence.json`. The existing Task 4 `RunnerLaunchProfile` mounts
  backend lock, runtime lock, model cache, and input root only. This Task 6 runner
  correctly fails closed without the seal evidence, but the host integration must
  extend or stage that mount before a real ready handshake can succeed.

## Fix Round 1

Status: `DONE_WITH_CONCERNS`

Base: `1ff0514`

### Review findings resolved

1. Smoke-oracle events now compare canonical JSON bytes, so equivalent nonzero
   binary64 inference floats and strict-JSON `Decimal` values match without
   weakening exact serialized-number comparison.
2. The real `train_util.create_estimator` result now receives a replaced
   `RunConfig` whose prediction `session_config` explicitly locks inter-op and
   intra-op threads to `1`.
3. Dependency, TensorFlow, and runner imports plus deterministic seeding are inside
   a guarded boundary. Any failure emits only
   `code=runner_dependency_import_failed count=1` and exits `2`.
4. Conversion parity independently loads the unmodified
   `/opt/crux/upstream/magenta/music/sequences_lib.py` and the patched
   `/opt/crux/vendor` module. Runtime inference compares the independently generated
   upstream serialization before consuming capture metadata, and the runtime image
   retains both source trees.
5. Checkpoint components are authenticated and copied through the same no-follow
   descriptors into a fresh private directory. Each file and directory is fsynced,
   the completed directory is atomically published, and TensorFlow receives only
   the private prefix. Regression tests cover mutation/path replacement after
   authentication and replacement of a mounted pathname during a multi-chunk read.

### RED-GREEN evidence

The following failures were witnessed before their corresponding implementation.
Fixture-only failures are included because they were also observed during the strict
TDD loop.

1. Canonical smoke comparison:

   ```text
   rtk uv run pytest -q runtime/oaf_tf1/tests/test_protocol.py \
     -k smoke_match_uses_canonical_numbers
   RED: 1 failed, 34 deselected
   AttributeError: module 'runtime.oaf_tf1.oaf_backend' has no attribute
   'smoke_events_match'

   GREEN: 1 passed, 34 deselected
   ```

2. Actual Estimator prediction-session configuration:

   ```text
   rtk docker run ... /opt/crux/venv/bin/python -m pytest -q \
     /opt/crux/runtime/tests/test_tensor_coverage.py \
     -k actual_train_util_estimator
   Fixture RED: 1 failed, 10 deselected
   AttributeError: module 'tensorflow.compat.v1' has no attribute 'contrib'

   Intended RED: 1 failed, 10 deselected
   AttributeError: module 'runtime.oaf_tf1.oaf_backend' has no attribute
   'configure_prediction_estimator_session'

   GREEN: 1 passed, 10 deselected in 2.89s
   ```

3. Stable dependency-import failure:

   ```text
   rtk uv run pytest -q runtime/oaf_tf1/tests/test_protocol.py \
     -k dependency_import_failure_is_one_stable
   Fixture RED: 1 failed, 35 deselected
   Expected dependency failure, received code=process_environment_invalid count=1

   rtk docker run ... /opt/crux/venv/bin/python -m pytest -q \
     /opt/crux/runtime/tests/test_protocol.py \
     -k dependency_import_failure_is_one_stable
   Fixture RED: 1 failed, 35 deselected
   FileNotFoundError: /opt/crux/runtime/entrypoint.py

   Intended RED: 1 failed, 35 deselected
   Dependency exception escaped with return code 1 and a traceback containing the
   injected arbitrary detail.

   GREEN: 1 passed, 35 deselected in 0.22s
   ```

4. Independent upstream/vendor parity:

   ```text
   rtk docker run ... /opt/crux/venv/bin/python -m pytest -q \
     /opt/crux/runtime/tests/test_instrumentation.py \
     -k instrumented_conversion_is_byte_identical
   RED: 1 failed, 17 deselected
   AttributeError: module 'runtime.oaf_tf1.oaf_backend' has no attribute
   'load_uninstrumented_sequences_module'

   GREEN: 1 passed, 17 deselected in 3.16s
   ```

5. Private checkpoint materialization:

   ```text
   rtk uv run pytest -q runtime/oaf_tf1/tests/test_protocol.py \
     -k 'checkpoint_consumers_receive_private or checkpoint_copy_keeps_descriptor'
   RED: 2 failed, 36 deselected
   AttributeError: module 'runtime.oaf_tf1.oaf_backend' has no attribute
   'materialize_authenticated_checkpoint'

   GREEN: 2 passed, 36 deselected in 0.65s
   ```

### Final verification

- Runner source manifest was regenerated twice from the final source bytes. Both
  runs produced SHA-256
  `af835db9902386d71a0e36ad5f4a79997ea2ea3c626bfce07a9ff0adc11aa64d`.
- Focused host Task 6 suite: `63 passed, 4 skipped in 10.15s`. The four skips are
  the image-only dependency boundary, two real patched-source integrations, and the
  real TensorFlow 1 Estimator integration.
- Exact Task 1-6 compatibility suite:
  `417 passed, 4 skipped in 31.41s`.
- Full host suite: `1519 passed in 44.02s`.
- Final exact non-root CPython 3.7 image suite:
  `65 passed, 2 skipped in 4.78s`.
- Final diagnostic image:
  `amd64 linux 10001:10001`,
  `sha256:39e288600478ad9370390c610bcbc1c3d62fa948a926e189e23b359138e9523e`.
- CPython 3.7 compilation: passed.
- Ruff: passed.
- Black: 6 files unchanged.
- Targeted Pylint: `10.00/10`.
- `git diff --check`: passed.

### Remaining gates and concern

- Task 8 still owns the final locks, seal evidence, final smoke oracle and real
  checkpoint prediction, tensor evidence, final UID/GID/resources, OCI identities,
  security/advisory evidence, and verified native-linux/amd64 result. The emulated
  image remains diagnostic only.
- Task 7 must still mount or stage authenticated seal evidence at
  `/run/crux/seal-evidence.json` before a real ready handshake can succeed.

## Fix Round 2

Status: `DONE_WITH_CONCERNS`

Base: `bb5e930`

### Residual findings resolved

1. Smoke comparison now serializes a strict type-tagged tree. Finite inference
   `float` and strict-oracle `Decimal` values share one exact coefficient/exponent
   normalization, while null, boolean, integer, real, string, list, object, key,
   and nesting identity remain distinct. Trailing coefficient zeros are normalized
   without expanding large exponents into unbounded fixed-point strings.
2. The actual TF 1.15 Estimator now installs the single-thread `ConfigProto` in
   both `_config` and constructor-cached `_session_config`. The real
   `train_util.create_estimator` regression verifies the cached object consumed by
   prediction is the same object as the installed internal RunConfig session
   configuration, with both thread counts equal to `1`.

### RED-GREEN evidence

1. Positive and negative exponent-form smoke numbers:

   ```text
   rtk uv run pytest -q runtime/oaf_tf1/tests/test_protocol.py \
     -k smoke_match_normalizes_exponent_numbers
   RED: 1 failed, 38 deselected
   Decimal('1E-7')/Decimal('-1E-7') did not match float 1e-7/-1e-7.

   rtk uv run pytest -q runtime/oaf_tf1/tests/test_protocol.py \
     -k 'smoke_match_uses_canonical_numbers or smoke_match_normalizes_exponent_numbers'
   GREEN: 2 passed, 37 deselected
   ```

   The regression also covers `1E+20` and `-1E+20`. It then changes the oracle to
   `0.00000010000000000000001` and requires inequality, proving normalization does
   not weaken exact numeric comparison.

2. Actual cached TF 1.15 prediction-session configuration:

   ```text
   rtk docker run ... /opt/crux/venv/bin/python -m pytest -q \
     /opt/crux/runtime/tests/test_tensor_coverage.py \
     -k actual_train_util_estimator
   RED: 1 failed, 10 deselected
   estimator.config.session_config had inter/intra=1, but estimator._session_config
   remained the constructor-cached allow-soft-placement configuration.
   ```

   The first implementation attempt set both values but an assertion requiring
   identity with `estimator.config.session_config` still failed:

   ```text
   1 failed, 10 deselected
   ```

   Real TF 1.15 inspection showed the public `config` property returns a copied
   RunConfig, so public-property identity is impossible. The assertion was corrected
   to the prediction-relevant invariant:
   `estimator._session_config is estimator._config._session_config`.

   ```text
   GREEN: 1 passed, 10 deselected in 3.59s
   ```

### Final verification

- Final runner source manifest generated twice with identical SHA-256
  `de9db15b84eff1657885bfb2f8209c3d7db272bd5c177c9f4e41036ab4eb494c`.
- Focused host Task 6 suite: `64 passed, 4 skipped in 9.53s`.
- Exact Task 1-6 compatibility suite:
  `418 passed, 4 skipped in 26.08s`.
- Full host suite: `1519 passed in 42.20s`.
- Exact non-root CPython 3.7 image suite:
  `66 passed, 2 skipped in 3.86s`.
- Final diagnostic image:
  `amd64 linux 10001:10001`,
  `sha256:82a31d4c779023c50ac6f04a3359cfdfb9be36fc911469f803adb5424ba41af2`.
- CPython 3.7 compilation: passed.
- Ruff: passed.
- Black: 6 files unchanged.
- Targeted Pylint: `10.00/10`.
- `git diff --check`: passed.

### Self-review

- Numeric normalization is smoke-only, so it does not change the frozen wire JSON
  serializer or mounted-object canonicality rules.
- Every normalized node carries a type tag. A float cannot equal an integer or
  string, list/object structure cannot alias, keys must remain exact strings, and
  nearby unequal real values retain distinct coefficient/exponent identities.
- Float conversion begins from CPython's finite shortest round-trip `repr`, matching
  the decimal token serialized by inference; strict oracle `Decimal` values retain
  all coefficient digits.
- The Estimator regression uses the actual TF 1.15/Magenta `TPUEstimator`, not a
  fake. It verifies both public values and the private constructor cache used by
  prediction.
- The Task 5 vendored source and upstream source manifest remain unchanged.

### Remaining gates and concern

- Task 8 still owns final locks, seal evidence, final smoke oracle and real
  checkpoint prediction, tensor evidence, final UID/GID/resources, OCI identities,
  security/advisory evidence, and verified native-linux/amd64 execution.
- Task 7 still must mount or stage authenticated seal evidence at
  `/run/crux/seal-evidence.json`.
