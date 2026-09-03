# HPA-494 Formatter and Lint Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire Black and make Ruff format plus errors-only Pylint the explicit Crux tooling contract without changing runtime code.

**Architecture:** Reuse the policy surfaces already present: Ruff remains formatter/linter, `[tool.pylint.messages_control]` becomes the single owner of Pylint error exceptions, and CI remains unchanged. This is one five-file cleanup commit followed by verification.

**Tech Stack:** Python 3.12, uv, Ruff 0.12.9, Pylint, pre-commit, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-hpa-494-tooling-contract-design.md`

## Global Constraints

- Ruff format is the only formatter.
- Keep CI unchanged in HPA-494; its `ruff check .` / `ruff format --check src tests` split is a historical scope retained only because workflows are outside this ticket.
- Put `E1120,E0401` in `[tool.pylint.messages_control].disable`; do not duplicate them in pre-commit or contributor commands.
- Full-warning Pylint stays advisory. Add no baseline, wrapper, score threshold, or extra disables.
- Preserve `AGENTS.md` as Git mode `120000` targeting `CLAUDE.md`; never stage it.
- Do not modify `src/`, `tests/`, `runtime/`, `artifacts/`, or `.github/workflows/`.
- Do not upgrade tools or reorganize the duplicated dev dependency lists.
- Keep implementation and verification in this PR; no second PR.

## Planned implementation files

- `pyproject.toml`
- `uv.lock`
- `.pre-commit-config.yaml`
- `CLAUDE.md`
- `README.md`

---

### Task 0: Confirm the enforced baseline

**Files:** read only

**Interfaces:**
- Consumes: current HPA-494 branch based on `main`.
- Produces: proof that the active CI-equivalent contract is green before tooling cleanup.

- [ ] **Step 1: Confirm a clean working tree**

```bash
git status --short
git merge-base --is-ancestor main HEAD
```

Expected: no status output; merge-base exits `0`.

- [ ] **Step 2: Run the active static gates**

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

Expected: all exit `0`.

If any fail, stop before editing. Do not fold source formatting or new Pylint suppressions into HPA-494.

Task 0 creates no commit.

---

### Task 1: Normalize the tooling contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.pre-commit-config.yaml`
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Verify unchanged: `AGENTS.md`

**Interfaces:**
- Consumes: Task 0's green baseline.
- Produces: one focused implementation commit with Black retired and Pylint error policy centralized.

- [ ] **Step 1: Update `pyproject.toml`**

Remove Black from both existing dev lists and delete `[tool.black]`:

```diff
-    "black>=24.4.0",
```

from both `[project.optional-dependencies].dev` and `[tool.uv].dev-dependencies`, plus:

```diff
-[tool.black]
-line-length = 100
-target-version = ["py312"]
```

Extend the existing Pylint disable table:

```diff
 [tool.pylint.messages_control]
 disable = [
   "missing-module-docstring",
   "missing-class-docstring",
   "missing-function-docstring",
   "too-few-public-methods",
+  "E1120",
+  "E0401",
 ]
```

Do not alter other Ruff/Pylint settings or unify the two dev lists.

- [ ] **Step 2: Regenerate and inspect `uv.lock`**

Run:

```bash
uv lock
git diff -U0 -- uv.lock | rg '^[+-](name|version) = ' | sort -u
```

Expected: Black and dependencies removed only because Black is no longer reachable may disappear. No unrelated package version may change.

If an unrelated package has a changed/added version line, stop and inspect the lock resolution before continuing. Do not accept unrelated upgrades in HPA-494.

- [ ] **Step 3: Simplify pre-commit Pylint**

Change only the local Pylint entry:

```diff
-        entry: uv run pylint --errors-only --disable=E1120
+        entry: uv run pylint --errors-only
```

Keep staged-file behavior and existing Ruff hooks unchanged. Do not add `files: ^src/` or a wrapper script.

- [ ] **Step 4: Align `CLAUDE.md`**

Replace the stale lint/format commands with:

```bash
# Linting and formatting checks
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src

# Format code before committing
uv run ruff format src tests
```

Keep Code Style concise:

```markdown
- Python 3.12, 4-space indent, 100-character soft line limit.
- Ruff format is the canonical formatter; Ruff `E/F/W/I` rules are the primary lint gate.
- CI currently format-checks only `src tests`; HPA-494 mirrors that historical workflow scope rather than widening CI.
- Pylint is blocking only in `--errors-only` mode; full-warning scores are advisory.
```

Retain the existing naming/import/commit-style bullets. Do not rewrite unrelated guidance.

- [ ] **Step 5: Ensure `AGENTS.md` was not materialized**

```bash
test -z "$(git diff --name-only -- AGENTS.md)"
```

Expected: exits `0`.

Do not add `AGENTS.md` to any `git add` command.

- [ ] **Step 6: Replace README's stale Code Formatting subsection**

Use:

````markdown
### Code Formatting

CI currently format-checks only `src tests`; this section mirrors that existing workflow scope.

```bash
# Check lint and formatting
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src

# Apply formatting
uv run ruff format src tests
```
````

Do not fix unrelated README drift.

- [ ] **Step 7: Inspect the focused diff**

```bash
if git grep -n -i 'black' -- \
  pyproject.toml README.md CLAUDE.md .pre-commit-config.yaml; then
  echo 'Active tooling/guidance still references Black' >&2
  exit 1
fi

git diff --check
git diff -- pyproject.toml .pre-commit-config.yaml CLAUDE.md README.md
```

Expected: no active Black references, no whitespace errors, and only the planned policy/doc changes.

- [ ] **Step 8: Commit the implementation**

```bash
git add pyproject.toml uv.lock .pre-commit-config.yaml CLAUDE.md README.md
git commit -m "chore: normalize Python tooling contract"
```

Expected: one implementation commit containing exactly those five files.

---

### Task 2: Verify behavior and scope

**Files:** verify Task 1 output; no implementation edits expected.

**Interfaces:**
- Consumes: Task 1 implementation commit.
- Produces: merge-readiness evidence for HPA-494.

- [ ] **Step 1: Run final repository gates**

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
uv run pre-commit run --all-files
uv run pytest -q
git diff --check main...HEAD
```

Expected: all exit `0`.

- [ ] **Step 2: Verify the `AGENTS.md` symlink after the commit**

```bash
test "$(git ls-files -s AGENTS.md | awk '{print $1}')" = "120000"
test "$(git show HEAD:AGENTS.md)" = "CLAUDE.md"
test -z "$(git diff --name-only main...HEAD -- AGENTS.md)"
```

Expected: all exit `0`.

- [ ] **Step 3: Verify the closed file set**

```bash
if git diff --name-only main...HEAD | \
  grep -E '^(src/|tests/|runtime/|artifacts/|\.github/workflows/)'; then
  echo 'HPA-494 touched an out-of-scope surface' >&2
  exit 1
fi

git diff --name-only main...HEAD | sort
```

Expected final paths:

```text
.pre-commit-config.yaml
CLAUDE.md
README.md
docs/superpowers/plans/2026-09-02-hpa-494-tooling-contract.md
docs/superpowers/specs/2026-09-02-hpa-494-tooling-contract-design.md
pyproject.toml
uv.lock
```

- [ ] **Step 4: Record verification on PR #33**

Record the Task 2 command results and the `uv.lock` dependency/version diff inspection on the existing PR. Do not create another PR.
