# HPA-494 Formatter and Lint Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire Black and make Crux's already-enforced Ruff-format plus errors-only-Pylint checks the explicit repository tooling contract without changing runtime code.

**Architecture:** Treat `.github/workflows/ci.yml` as the existing target contract. Remove duplicate Black metadata, align pre-commit and active contributor guidance to that contract, and verify the repository without touching production/test/evidence code.

**Tech Stack:** Python 3.12, uv, Ruff 0.12.9, Pylint, pre-commit, pytest, Markdown/YAML/TOML configuration.

**Spec:** `docs/superpowers/specs/2026-09-02-hpa-494-tooling-contract-design.md`

## Global Constraints

- Ruff format is the only authoritative Python formatter; keep `py312` and line length `100`.
- Preserve CI's intentional path split exactly: lint with `ruff check .`; format-check/apply only `src tests`. Do not unify those scopes in HPA-494.
- Blocking Pylint policy is exactly `--errors-only --disable=E1120,E0401`; add no further disables or baseline machinery.
- Align only Pylint flags between CI and pre-commit. Pre-commit keeps its existing staged-Python-file path scope; do not add `files: ^src/`, broaden CI, or move `E1120,E0401` into `pyproject.toml`.
- Preserve `AGENTS.md` as the existing Git symlink (`mode 120000`) targeting `CLAUDE.md`; do not edit, materialize, or stage it.
- Do not upgrade tooling, reorganize dev-dependency groups, or rewrite historical Superpowers records.
- Do not modify `src/`, `tests/`, `runtime/`, `artifacts/`, or `.github/workflows/`.
- If Ruff format or errors-only Pylint is not green at Task 0, stop before editing and diagnose the moved baseline.
- Keep planning, implementation, and verification in this single HPA-494 PR.

## Planned file surface

- `pyproject.toml` — remove Black from both existing dev lists and delete `[tool.black]`.
- `uv.lock` — regenerate through `uv lock`.
- `.pre-commit-config.yaml` — align Pylint's explicit error exceptions with CI while keeping staged-file scope.
- `CLAUDE.md` — document the canonical commands, the lint/format path split, and full-warning Pylint's advisory status.
- `README.md` — replace only the stale Code Formatting commands and name the same lint/format path split.

`.github/workflows/ci.yml` stays unchanged; it already expresses the target contract. `AGENTS.md` also stays unchanged; its `120000 -> CLAUDE.md` symlink is the single-source agent-guidance boundary.

---

### Task 0: Reconfirm the current contract

**Files:** read only

**Interfaces:**
- Consumes: the current HPA-494 branch based on `main`.
- Produces: a green enforced baseline plus current characterization of the two legacy commands.

- [ ] **Step 1: Confirm the working tree is clean**

```bash
git status --short
git merge-base --is-ancestor main HEAD
```

Expected: no status output; merge-base command exits `0`.

- [ ] **Step 2: Require the checks already enforced by CI to pass**

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

Expected: all exit `0`. These scopes are deliberately different because they copy CI: lint uses `.`, while formatting uses `src tests`. If any fail, stop before implementation; do not normalize the scopes, reformat unrelated paths, or add Pylint suppressions in HPA-494.

- [ ] **Step 3: Characterize the legacy commands without gating on their old result**

```bash
set +e
uv run black --check src tests
black_status=$?
uv run pylint src/app src/cli
full_pylint_status=$?
set -e

printf 'black_status=%s\nfull_pylint_status=%s\n' \
  "$black_status" "$full_pylint_status"
```

Expected: record both statuses in the PR. They may still be nonzero as HPA-494 originally recorded, or later changes may have made either command green. Neither outcome changes the selected design: Black is duplicate tooling and full-warning Pylint is not the enforced gate.

- [ ] **Step 4: Confirm characterization changed no files**

```bash
git status --short
```

Expected: no output.

Task 0 creates no commit.

---

### Task 1: Normalize the active formatter/lint contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.pre-commit-config.yaml`
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Verify unchanged symlink: `AGENTS.md`

**Interfaces:**
- Consumes: Task 0's green enforced baseline.
- Produces: one active Ruff-format/errors-only-Pylint contract with no direct Black dependency and no second agent-guidance file.

- [ ] **Step 1: Remove Black from `pyproject.toml`**

Apply only these semantic changes:

```diff
 [project.optional-dependencies]
 dev = [
     "pytest>=7.4.3",
     "pytest-asyncio>=0.21.1",
     "pytest-cov>=4.1.0",
-    "black>=24.4.0",
     "ruff>=0.1.6",
     "pylint>=3.2.0",
     "pre-commit>=3.7.0",
 ]
@@
 [tool.uv]
 dev-dependencies = [
     "pytest>=7.4.3",
     "pytest-asyncio>=0.21.1",
     "pytest-cov>=4.1.0",
-    "black>=24.4.0",
     "ruff>=0.1.6",
     "pylint>=3.2.0",
     "pre-commit>=3.7.0",
 ]
@@
 [tool.ruff.lint]
 select = ["E", "F", "W", "I"]
 # Let the formatter handle line length; avoid blocking commits on E501.
 ignore = ["E501"]
-
-[tool.black]
-line-length = 100
-target-version = ["py312"]
```

Do not change Ruff or Pylint configuration otherwise, and do not unify the duplicated dev lists.

- [ ] **Step 2: Regenerate the lockfile**

```bash
uv lock
uv lock --check
```

Expected: both exit `0`; never hand-edit `uv.lock`.

- [ ] **Step 3: Align pre-commit Pylint flags with CI**

Change only the local Pylint entry:

```diff
-        entry: uv run pylint --errors-only --disable=E1120
+        entry: uv run pylint --errors-only --disable=E1120,E0401
```

Keep pre-commit's existing staged-Python-file behavior and the existing Ruff hooks. Do not add `files: ^src/`, do not move `E1120,E0401` into `[tool.pylint.messages_control]`, and do not add a wrapper script. The staged-file path scope is pre-existing local behavior, not unfinished HPA-494 work.

- [ ] **Step 4: Align `CLAUDE.md`**

Replace its lint/format commands with:

```bash
# Linting and formatting checks
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src

# Format code before committing
uv run ruff format src tests
```

Use these Code Style bullets:

```markdown
- Python 3.12, 4-space indent, 100-character soft line limit.
- Ruff format is the canonical Python formatter; Ruff `E/F/W/I` rules provide the primary lint gate.
- CI intentionally lints `.` but format-checks only `src tests`; keep those path scopes distinct.
- Pylint is blocking only for `--errors-only --disable=E1120,E0401`; full-warning scores are advisory.
- `verb_noun` naming for functions, `PascalCase` for classes.
- Imports sorted by Ruff `I` rules; no eager TF imports.
- Commit style: Conventional Commits (`feat:`, `fix:`, `refactor:`, `chore:`, `test:`), subject under 72 chars.
```

Do not rewrite unrelated architecture/environment guidance.

- [ ] **Step 5: Prove `AGENTS.md` remains the symlink to `CLAUDE.md`**

Run immediately after editing `CLAUDE.md`, before staging anything:

```bash
git ls-files -s AGENTS.md
test "$(git ls-files -s AGENTS.md | awk '{print $1}')" = "120000"
test "$(git show HEAD:AGENTS.md)" = "CLAUDE.md"
test -z "$(git diff --name-only -- AGENTS.md)"
```

Expected: the index entry starts with mode `120000`, the target blob is exactly `CLAUDE.md`, and `AGENTS.md` is absent from the working-tree diff. Do not add `AGENTS.md` to any `git add` command.

- [ ] **Step 6: Align only README's Code Formatting subsection**

Use:

````markdown
### Code Formatting

CI intentionally lints `.` but format-checks only `src tests`; keep those path scopes distinct.

```bash
# Check lint and formatting
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src

# Apply formatting
uv run ruff format src tests
```
````

Do not expand this into a general README cleanup.

- [ ] **Step 7: Run focused contract checks before committing**

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
uv run pre-commit run --all-files
git diff --check
```

Expected: all exit `0` and pre-commit does not modify tracked files. Do not interpret pre-commit's staged/all-file Python path scope as a reason to change CI's `src` scope.

- [ ] **Step 8: Confirm active surfaces no longer mention Black**

```bash
if git grep -n -i 'black' -- \
  pyproject.toml README.md CLAUDE.md .pre-commit-config.yaml .github/workflows/ci.yml; then
  echo 'Active tooling/guidance still references Black' >&2
  exit 1
fi
```

Expected: no matches.

Check whether Black disappeared from the lockfile:

```bash
set +e
grep -n 'name = "black"' uv.lock
lock_black_status=$?
set -e
printf 'lock_black_status=%s\n' "$lock_black_status"
```

Normally this prints `lock_black_status=1`. If it is `0`, identify and record the transitive owner; do not remove an unrelated dependency solely to erase Black from the lockfile.

- [ ] **Step 9: Commit the implementation**

```bash
git add pyproject.toml uv.lock .pre-commit-config.yaml CLAUDE.md README.md
git commit -m "chore: normalize Python tooling contract"
```

Expected: one focused implementation commit containing only those five files. `AGENTS.md` must not be staged or committed separately.

---

### Task 2: Verify behavior and scope isolation

**Files:** verify the five implementation files; all runtime/test/evidence/workflow files and `AGENTS.md` remain untouched.

**Interfaces:**
- Consumes: Task 1's implementation commit.
- Produces: merge-ready evidence satisfying HPA-494 without source churn or contract-scope drift.

- [ ] **Step 1: Run the complete repository gates**

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
uv run pre-commit run --all-files
uv run pytest -q
git diff --check main...HEAD
```

Expected: all exit `0`. HPA-494 adds no unit tests because it changes no runtime behavior. The lint/format path split remains exactly the one CI owns.

- [ ] **Step 2: Prove `AGENTS.md` stayed single-sourced**

```bash
test "$(git ls-files -s AGENTS.md | awk '{print $1}')" = "120000"
test "$(git show HEAD:AGENTS.md)" = "CLAUDE.md"
git diff --exit-code main...HEAD -- AGENTS.md
```

Expected: mode `120000`, target `CLAUDE.md`, and no PR diff for `AGENTS.md`.

- [ ] **Step 3: Prove forbidden surfaces are unchanged**

```bash
git diff --exit-code main...HEAD -- .github/workflows/ci.yml

if git diff --name-only main...HEAD | \
  grep -E '^(src/|tests/|runtime/|artifacts/|\.github/workflows/|AGENTS\.md$)'; then
  echo 'HPA-494 touched a forbidden runtime/test/evidence/workflow/symlink surface' >&2
  exit 1
fi
```

Expected: no CI diff and no forbidden path output.

- [ ] **Step 4: Verify the final file set**

```bash
git diff --name-only main...HEAD | sort
```

Expected paths are limited to:

```text
.pre-commit-config.yaml
CLAUDE.md
README.md
docs/superpowers/plans/2026-09-02-hpa-494-tooling-contract.md
docs/superpowers/specs/2026-09-02-hpa-494-tooling-contract-design.md
pyproject.toml
uv.lock
```

- [ ] **Step 5: Record verification on this PR**

Add the final exit/result summary for the Task 2 commands to the existing HPA-494 PR, plus the Task 0 legacy-command statuses and whether Black disappeared from `uv.lock` or remained only transitively.

Also record that lint remains repository-root scope, format remains `src tests`, pre-commit keeps staged-Python-file Pylint scope, and `AGENTS.md` remains `120000 -> CLAUDE.md`.

Do not create a second implementation PR.
