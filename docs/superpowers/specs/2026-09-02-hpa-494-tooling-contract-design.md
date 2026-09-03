# HPA-494 Formatter and Lint Contract Design

## Context

HPA-494 is tooling cleanup, not source cleanup.

Current `main` has three active inconsistencies:

- Black is still declared twice in `pyproject.toml` and configured in `[tool.black]`, although CI/pre-commit format with Ruff.
- Pylint error exceptions are duplicated in commands: CI has `E1120,E0401`; pre-commit has only `E1120`, while `pyproject.toml` already has a shared Pylint disable table.
- `CLAUDE.md` and README contain stale commands; README also points Black/Ruff at non-existent `app/`.

## Decision

Keep the existing lean toolchain:

- Ruff format is the only formatter.
- Ruff remains the primary lint gate.
- Pylint blocks errors only; full-warning scores are advisory.
- `[tool.pylint.messages_control].disable` owns `E1120,E0401` so the policy is not copied between invocations.
- CI stays unchanged in HPA-494.

The current CI split (`ruff check .` vs `ruff format --check src tests`) is historical workflow scope, not a desired invariant. Guidance mirrors it only because workflow cleanup is outside this five-file ticket.

## Implementation

### `pyproject.toml`

- Remove both `black>=24.4.0` dev entries.
- Delete `[tool.black]`.
- Add `"E1120"` and `"E0401"` to the existing `[tool.pylint.messages_control].disable` list.
- Do not reorganize dev lists or upgrade tools.

Regenerate `uv.lock` with `uv lock`. Inspect changed package name/version lines and reject unrelated version movement.

### `.pre-commit-config.yaml`

Keep staged-file behavior and Ruff hooks. Change Pylint to:

```text
uv run pylint --errors-only
```

Do not add path filters, wrappers, or more disables.

### `CLAUDE.md` and `README.md`

Use:

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
uv run ruff format src tests
```

Briefly note that `src tests` is today's historical CI formatter scope, not a policy being expanded in HPA-494. Keep full-warning Pylint advisory. Do not perform unrelated doc cleanup.

### `AGENTS.md`

Do not modify it. It is the existing Git symlink (`120000 -> CLAUDE.md`), so `CLAUDE.md` remains the single source of agent guidance.

## Scope

Implementation changes exactly five files:

- `pyproject.toml`
- `uv.lock`
- `.pre-commit-config.yaml`
- `CLAUDE.md`
- `README.md`

No changes under `src/`, `tests/`, `runtime/`, `artifacts/`, `.github/workflows/`, or `AGENTS.md`.

## Verification

Before edits:

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

Do not rerun legacy Black/full-warning Pylint for characterization; their result cannot change the decision.

After `uv lock`:

```bash
git diff -U0 -- uv.lock | rg '^[+-](name|version) = ' | sort -u
```

Only Black and Black-exclusive removals may disappear; unrelated version changes stop the task.

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

Also verify `AGENTS.md` is still `120000 -> CLAUDE.md`, is absent from the diff, and the PR touches no forbidden paths.

## Non-goals

No source refactor, Pylint baseline/score gate, CI change, wrapper/Makefile, tool upgrade, dev-list unification, historical-doc rewrite, or second PR.
