# HPA-494 Formatter and Lint Contract Design

## Context

HPA-494 was opened after HPA-481/HPA-482 deliberately left two legacy repository commands outside their native-attestation scope:

- `uv run black --check src tests`
- `uv run pylint src/app src/cli`

The repository has since converged on a different active contract:

- `.github/workflows/ci.yml` runs `ruff check .`, `ruff format --check src tests`, and errors-only Pylint over `src` with the existing `E1120,E0401` exceptions.
- `.pre-commit-config.yaml` runs Ruff lint, Ruff format, and errors-only Pylint for staged Python files.
- recent Crux PRs use the same Ruff / Ruff-format / errors-only-Pylint checks as their repository-wide static gates.

The remaining inconsistency is configuration and documentation drift, not a production-code defect:

- `pyproject.toml` still installs Black in both dev dependency lists and still carries `[tool.black]` configuration;
- `README.md` still tells contributors to run Black and an obsolete `ruff app/` command;
- `CLAUDE.md` still lists full-warning `pylint src/app src/cli`, even though CI intentionally gates only Pylint errors;
- pre-commit omits `E0401` from the two explicit Pylint error exceptions already used by CI.

HPA-627 remains operationally blocked on gated Hugging Face model access, so HPA-494 is the highest-value executable Crux backlog task rather than another attempt to work around that external gate.

## Goals

- Make Ruff format the single authoritative Python formatter.
- Remove Black from active project dependencies and configuration.
- Make the repository's Pylint acceptance contract explicit: errors are blocking; score/style/refactor warnings are advisory.
- Use the same two existing Pylint error exceptions, `E1120` and `E0401`, in CI-facing and pre-commit guidance instead of introducing a baseline file or broader disables.
- Align active contributor documentation with the checks the repository actually enforces.
- Preserve all runtime behavior, benchmark identities, tests, and generated evidence.

## Non-goals

- Do not refactor `src/app`, `src/cli`, `src/benchmark`, or tests merely to improve a Pylint score.
- Do not add a Pylint score threshold, warning allowlist, generated baseline, or suppression framework.
- Do not keep Black as a compatibility formatter or add formatter-cross-check tests.
- Do not upgrade Ruff, Pylint, pre-commit, Python, or unrelated dependencies as part of this task.
- Do not reorganize the duplicated dev-dependency declaration structure in `pyproject.toml`; only remove Black from the existing lists.
- Do not rewrite historical Superpowers plans/reports that mention Black. They are records of the checks used when those changes were produced, not current contributor guidance.
- Do not change CI job structure, test coverage, runtime code, benchmark artifacts, or frozen model/runtime identities.

## Options considered

### A. Retire Black and formalize the existing Ruff + errors-only Pylint contract — selected

Keep the checks already enforced by CI:

```text
Ruff lint             -> repository-wide correctness/style lint
Ruff format           -> sole Python formatter
Pylint --errors-only  -> secondary error detector
pytest                 -> behavior regression gate
```

Remove Black from the active dependency/configuration surface, align pre-commit's explicit error exceptions with CI, and update active contributor docs.

This is the smallest change because the desired contract already exists in CI and has been used by recent PRs. It deletes duplicate machinery rather than normalizing source code around a legacy command.

### B. Keep Black and require Ruff-format compatibility

Reformat the three historically failing files, retain both formatter dependencies/configurations, and require both formatters to remain green.

Rejected because two formatters provide no useful independent correctness signal here. They create two version/configuration surfaces and can diverge later for formatting-only reasons.

### C. Make full-warning Pylint pass

Refactor or suppress every convention/refactor/warning diagnostic under `src/app` and `src/cli` until `uv run pylint src/app src/cli` exits zero.

Rejected because CI and pre-commit do not use full-warning Pylint as an acceptance gate. Spending feature-development time on a 10/10 score would create unrelated source churn and invite broad suppressions or a baseline framework solely to satisfy a legacy command.

## Canonical formatting contract

Ruff format is the only authoritative formatter for active Python source and tests.

The existing repository settings remain authoritative:

- Python target: `py312`
- line length: `100`
- format scope in CI: `src tests`
- lint scope in CI: repository-wide via `ruff check .`

No Ruff upgrade or configuration redesign is needed. `.github/workflows/ci.yml` and `.pre-commit-config.yaml` already pin the CI/pre-commit Ruff integration to `0.12.9`; HPA-494 does not change those versions.

Implementation removes:

- `black>=24.4.0` from `[project.optional-dependencies].dev`;
- `black>=24.4.0` from `[tool.uv].dev-dependencies`;
- the entire `[tool.black]` section.

Then regenerate `uv.lock` through normal `uv` locking. Do not hand-edit transitive lock entries.

Historical planning/report Markdown files may continue to contain old Black commands. The absence check applies only to active tooling/configuration and contributor guidance.

## Canonical Pylint contract

Pylint is a secondary error detector, not a project score gate.

The canonical repository-wide verification command is the command already used by CI:

```bash
uv run pylint --errors-only --disable=E1120,E0401 src
```

The two existing exceptions stay narrow and explicit:

- `E1120` is already excluded by both CI and pre-commit.
- `E0401` is already excluded by CI and is added to pre-commit so local staged-file checks use the same error policy.

Do not add any further Pylint disables in this task. Existing `pyproject.toml` Pylint settings remain untouched.

Full-warning Pylint may still be run manually as advisory analysis, but a nonzero score/warning result is not repository acceptance failure. `CLAUDE.md` should state this directly so future work does not reopen the same baseline debt accidentally.

## Active documentation contract

Update only active contributor-facing instructions:

### `CLAUDE.md`

The command section should list:

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
uv run ruff format src tests
```

The code-style guidance should state that Ruff format is canonical and that Pylint's blocking contract is errors-only. It may note that full-warning Pylint is advisory.

### `README.md`

Replace the stale Black/old-Ruff formatting snippet with the same active check commands plus the Ruff-format apply command. Do not expand HPA-494 into a general README rewrite even though other historical content may also be stale.

### `.pre-commit-config.yaml`

Keep the existing Ruff hooks and local Pylint hook. Change only the local Pylint command so its error exceptions match CI:

```text
uv run pylint --errors-only --disable=E1120,E0401
```

Pre-commit continues to append staged Python filenames automatically; do not replace it with a repository-wide wrapper script.

### `.github/workflows/ci.yml`

No change. It is the current source of truth HPA-494 is aligning other active surfaces to.

## Expected implementation surface

Modify only:

- `pyproject.toml`
- `uv.lock`
- `.pre-commit-config.yaml`
- `CLAUDE.md`
- `README.md`

Planning documents in this PR remain alongside those implementation changes when execution begins.

No `src/`, `tests/`, runtime lock, benchmark artifact, workflow, or generated evidence file should change. If an implementation step appears to require source reformatting or runtime behavior changes, stop and reassess rather than folding that work into HPA-494.

## Verification

### Baseline before edits

Confirm the already-enforced contract is green before changing tooling metadata:

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

Also reproduce the two HPA-494 legacy baselines as characterization only:

```bash
uv run black --check src tests
uv run pylint src/app src/cli
```

Those legacy commands are expected to be nonzero before the change. They are not implementation targets.

### Final gates

After the cleanup:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
uv run pre-commit run --all-files
uv run pytest -q
git diff --check
```

Active-surface drift checks:

```bash
git grep -n -i '\bblack\b' -- \
  pyproject.toml README.md CLAUDE.md .pre-commit-config.yaml .github/workflows/ci.yml

grep -n 'name = "black"' uv.lock
```

Both searches should produce no matches after implementation. Historical planning/report files are intentionally outside the first search.

Finally, the implementation diff relative to `main` should contain no changes under `src/`, `tests/`, `runtime/`, `artifacts/`, or `.github/workflows/`.

## Acceptance mapping

HPA-494's acceptance criteria map directly to this design:

- **Canonical formatter relationship:** Ruff format is authoritative; Black is retired.
- **Black check:** formally retired from the repository contract by removing the dependency/configuration and active documentation references.
- **Pylint baseline:** errors-only Pylint with the two already-reviewed error exceptions is the explicit blocking baseline; full-warning scores are advisory and no broad warning disables are added.
- **Regression gates:** pytest, Ruff lint, and Ruff format remain green; the stronger current repository-wide errors-only Pylint check remains green as well.
- **Isolation from HPA-481/HPA-482 evidence:** no source, runtime, artifact, workflow, or frozen-input file changes are planned.

## Risks and fallback

### Ruff format is unexpectedly not clean at implementation start

Stop before removing Black. HPA-494 assumes the current CI/recent-PR Ruff baseline is green. If `ruff format --check src tests` fails on the implementation base, first determine whether `main` moved or the local environment is stale; do not silently mix an unrelated formatting migration into this task.

### Errors-only Pylint differs between CI and local pre-commit

The policy is the two explicit exceptions already present in CI. If staged-file pre-commit still produces an import/environment-only error after adding `E0401`, diagnose that concrete environment mismatch. Do not add a third disable without separately reviewing the diagnostic.

### Black remains transitively present in `uv.lock`

The goal is to remove Black as an active Crux dependency. If another dependency unexpectedly requires it, do not remove or fork that dependency merely to make the lock search empty. Record the transitive owner, keep the direct/configuration removal, and change the lock verification to assert that Crux no longer declares Black directly.
