# HPA-494 Formatter and Lint Contract Design

## Context

HPA-494 was opened after HPA-481/HPA-482 left two legacy checks outside their scope: Black formatting and full-warning Pylint. Since then, Crux has converged on a simpler active contract:

- CI runs `ruff check .`, `ruff format --check src tests`, and errors-only Pylint over `src` with `E1120,E0401` excluded.
- pre-commit already runs Ruff lint, Ruff format, and errors-only Pylint for staged Python files.
- recent Crux PRs use those same CI-equivalent gates.

The remaining debt is active-tooling drift:

- `pyproject.toml` still declares Black twice and retains `[tool.black]`;
- `README.md` still recommends Black and an obsolete Ruff invocation;
- `CLAUDE.md` still presents full-warning Pylint as a required check;
- pre-commit excludes `E1120` but not CI's existing `E0401` exception.

HPA-627 is still blocked on gated Hugging Face access, so HPA-494 is the next executable Crux backlog task.

## Decision

Use the contract that already ships in CI:

```text
Ruff lint             -> primary lint gate
Ruff format           -> only formatter
Pylint --errors-only  -> secondary error gate
pytest                 -> behavior regression gate
```

Retire Black instead of keeping two formatters, and make full-warning Pylint advisory instead of spending time chasing a score that CI does not enforce.

## Options considered

### A. Ruff-only formatting + errors-only Pylint — selected

Remove Black from active dependencies/configuration, align pre-commit's two Pylint error exceptions with CI, and update contributor guidance. This deletes stale machinery and does not touch production code.

### B. Keep Black alongside Ruff format

Rejected. Two formatters add dependency/configuration/version drift without an independent correctness benefit.

### C. Refactor until full-warning Pylint exits zero

Rejected. It would create unrelated source churn, broad suppressions, or a baseline framework solely to satisfy a legacy non-CI command.

## Contract

### Formatting

Ruff format is authoritative for `src tests` with the existing `py312` target and 100-character line length. HPA-494 does not upgrade Ruff or redesign its configuration.

Remove from `pyproject.toml`:

- both `black>=24.4.0` dev dependency entries;
- the `[tool.black]` section.

Regenerate `uv.lock` with `uv lock`; never hand-edit it. Historical planning/report files may continue to mention Black because they document past verification.

### Pylint

The blocking repository-wide command is the one already used by CI:

```bash
uv run pylint --errors-only --disable=E1120,E0401 src
```

Add `E0401` to the existing pre-commit Pylint hook so staged-file checks use the same error policy. Add no new disables, score threshold, warning baseline, wrapper, or suppression framework.

Full-warning Pylint remains available for manual analysis but is not an acceptance gate.

### Active contributor guidance

Update `CLAUDE.md` and the README Code Formatting section to use:

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
uv run ruff format src tests
```

`CLAUDE.md` should explicitly say Ruff format is canonical and full-warning Pylint is advisory.

Do not turn HPA-494 into a general README cleanup.

## Scope

Implementation modifies only:

- `pyproject.toml`
- `uv.lock`
- `.pre-commit-config.yaml`
- `CLAUDE.md`
- `README.md`

`.github/workflows/ci.yml` is deliberately unchanged because it already expresses the target contract.

Do not modify `src/`, `tests/`, `runtime/`, `artifacts/`, benchmark evidence, or frozen runtime/model identities. Do not reorganize the duplicated dev-dependency structure or upgrade unrelated tools.

## Baseline and verification

Before editing, require the current enforced gates to pass:

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

Also run the legacy Black/full-Pylint commands once as characterization and record their current exit status. Their historical nonzero state is not a gate: if later work happened to make either command green, the duplicate/stale contract still gets retired rather than preserved.

After implementation run:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
uv run pre-commit run --all-files
uv run pytest -q
git diff --check main...HEAD
```

Active config/guidance should no longer contain Black:

```bash
git grep -n -i 'black' -- \
  pyproject.toml README.md CLAUDE.md .pre-commit-config.yaml .github/workflows/ci.yml
```

Normally Black should also disappear from `uv.lock`. If another dependency unexpectedly retains it transitively, record that owner and keep the scope at removing Crux's direct Black contract rather than deleting unrelated dependencies.

## Acceptance mapping

- **Formatter relationship:** Ruff format is authoritative; Black is formally retired.
- **Black check:** no longer part of the repository contract because Black is no longer directly configured/required.
- **Pylint baseline:** errors-only with the two existing CI exceptions is the explicit blocking policy; full-warning scores are advisory.
- **Regression gates:** pytest, Ruff lint, Ruff format, Pylint errors, pre-commit, lock consistency, and diff whitespace stay green.
- **HPA-481/HPA-482 isolation:** no runtime, source, workflow, artifact, or frozen-input change is planned.

## Fallbacks

If Ruff format or errors-only Pylint is not green on the implementation base, stop before changing configuration and diagnose the moved baseline rather than folding source cleanup into HPA-494.

If pre-commit exposes a new Pylint error after matching CI's `E1120,E0401` policy, diagnose that concrete error; do not add another disable without separate review.
