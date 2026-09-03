# HPA-494 Formatter and Lint Contract Design

## Context

HPA-494 is active-tooling cleanup, not a source refactor.

Current `main` has three small inconsistencies:

- Black is still declared twice in `pyproject.toml` and has a `[tool.black]` section even though CI/pre-commit use Ruff format.
- Pylint error exceptions are duplicated in command lines: CI has `E1120,E0401`, while pre-commit has only `E1120`. `pyproject.toml` already has the shared `[tool.pylint.messages_control].disable` table that should own this policy.
- `CLAUDE.md` and `README.md` still describe stale commands; README even points Black/Ruff at non-existent `app/` instead of `src/app/`.

CI currently runs:

```bash
ruff check .
ruff format --check src tests
pylint --errors-only --disable=E1120,E0401 src
```

The lint/format path split is historical CI scope, not a desired architectural distinction. HPA-494 keeps `.github/workflows/ci.yml` unchanged to stay a five-file tooling/docs cleanup; guidance mirrors the current CI scopes without calling the split load-bearing or permanent.

## Decision

Use the existing simple toolchain:

```text
Ruff lint             -> primary lint gate
Ruff format           -> only formatter
Pylint --errors-only  -> secondary error gate
pytest                 -> behavior regression gate
```

Retire Black. Do not chase full-warning Pylint scores.

Centralize `E1120` and `E0401` in the existing Pylint configuration table so local, pre-commit, and future invocations inherit the same exceptions without copying the list again.

## Changes

### 1. Retire Black

In `pyproject.toml`:

- remove `black>=24.4.0` from both existing dev dependency lists;
- delete `[tool.black]`;
- do not reorganize the duplicated dev lists or upgrade Ruff/Pylint/pre-commit.

Regenerate `uv.lock` with `uv lock`; do not hand-edit it.

Historical Superpowers records may continue to mention Black because they record past verification.

### 2. Centralize Pylint error exceptions

Extend the existing table:

```toml
[tool.pylint.messages_control]
disable = [
  "missing-module-docstring",
  "missing-class-docstring",
  "missing-function-docstring",
  "too-few-public-methods",
  "E1120",
  "E0401",
]
```

Then simplify pre-commit to:

```text
uv run pylint --errors-only
```

Contributor guidance uses:

```bash
uv run pylint --errors-only src
```

CI remains unchanged in this ticket. Its explicit `--disable=E1120,E0401` becomes redundant but harmless; a later workflow cleanup may remove that duplication.

Full-warning Pylint remains advisory. Do not add a score threshold, baseline file, wrapper, or additional disables.

Pre-commit keeps its existing staged-Python-file scope. HPA-494 centralizes message policy only; it does not add `files: ^src/` or otherwise align path selection with CI.

### 3. Update active guidance

`CLAUDE.md` and README's Code Formatting subsection use the current CI-equivalent commands:

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
uv run ruff format src tests
```

Add only a short note that the formatter scope mirrors today's historical CI scope (`src tests`) and is not widened in HPA-494. Do not turn either file into a broader documentation cleanup.

### 4. Preserve the agent-guidance symlink

`AGENTS.md` is a Git symlink (`mode 120000`) targeting `CLAUDE.md`. Updating `CLAUDE.md` updates agent guidance automatically.

Do not edit, materialize, or stage `AGENTS.md`.

## Implementation surface

Modify only:

- `pyproject.toml`
- `uv.lock`
- `.pre-commit-config.yaml`
- `CLAUDE.md`
- `README.md`

Planning docs remain in the same PR.

Do not modify `src/`, `tests/`, `runtime/`, `artifacts/`, or `.github/workflows/`.

## Verification

Before editing, require the enforced baseline to pass:

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

Do not rerun legacy Black/full-warning Pylint merely to characterize them; their outcome cannot change this design.

After `uv lock`, inspect dependency/version churn rather than immediately proving the lock matches the input that just generated it:

```bash
git diff -U0 -- uv.lock | rg '^[+-](name|version) = ' | sort -u
```

Only Black and dependencies removed solely because Black is no longer reachable may disappear. Any unrelated package version movement stops the task for investigation.

Final gates:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
uv run pre-commit run --all-files
uv run pytest -q
git diff --check main...HEAD
```

Final scope verification also requires:

- `AGENTS.md` remains mode `120000` and target `CLAUDE.md`;
- `AGENTS.md` is absent from the PR diff;
- no changes exist under `src/`, `tests/`, `runtime/`, `artifacts/`, or `.github/workflows/`;
- active tooling/guidance no longer references Black directly.

## Non-goals

- No source/test refactor for Pylint warnings or score.
- No CI workflow change in HPA-494.
- No Ruff/Pylint/pre-commit/Python upgrade.
- No dev-dependency-list unification.
- No wrapper, Makefile, baseline, or new lint job.
- No second PR.
