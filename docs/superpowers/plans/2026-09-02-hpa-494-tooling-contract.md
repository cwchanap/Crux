# HPA-494 Formatter and Lint Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire Black and make Crux's already-enforced Ruff-format plus errors-only-Pylint checks the explicit repository tooling contract without changing runtime code.

**Architecture:** Treat `.github/workflows/ci.yml` as the existing target contract. Remove duplicate Black metadata, align pre-commit and active contributor guidance to that contract, and verify the repository without touching production/test/evidence code.

**Tech Stack:** Python 3.12, uv, Ruff 0.12.9, Pylint, pre-commit, pytest, Markdown/YAML/TOML configuration.

**Spec:** `docs/superpowers/specs/2026-09-02-hpa-494-tooling-contract-design.md`

## Global Constraints

- Ruff format is the only authoritative Python formatter; keep `py312` and line length `100`.
- Blocking Pylint policy is exactly `--errors-only --disable=E1120,E0401`; add no further disables or baseline machinery.
- Do not upgrade tooling, reorganize dev-dependency groups, or rewrite historical Superpowers records.
- Do not modify `src/`, `tests/`, `runtime/`, `artifacts/`, or `.github/workflows/`.
- If Ruff format or errors-only Pylint is not green at Task 0, stop before editing and diagnose the moved baseline.
- Keep planning, implementation, and verification in this single HPA-494 PR.

## Planned file surface

- `pyproject.toml` — remove Black from both existing dev lists and delete `[tool.black]`.
- `uv.lock` — regenerate through `uv lock`.
- `.pre-commit-config.yaml` — align Pylint's explicit error exceptions with CI.
- `CLAUDE.md` — document the canonical commands and full-warning Pylint's advisory status.
- `README.md` — replace only the stale Code Formatting commands.

`.github/workflows/ci.yml` stays unchanged; it already expresses the target contract.

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

Expected: all exit `0`. If any fail, stop before implementation; do not fold source reformatting or extra Pylint suppressions into HPA-494.

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

**Interfaces:**
- Consumes: Task 0's green enforced baseline.
- Produces: one active Ruff-format/errors-only-Pylint contract with no direct Black dependency.

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

Do not change Ruff or Pylint configuration otherwise.

- [ ] **Step 2: Regenerate the lockfile**

```bash
uv lock
uv lock --check
```

Expected: both exit `0`; never hand-edit `uv.lock`.

- [ ] **Step 3: Align pre-commit Pylint with CI**

Change only the local Pylint entry:

```diff
-        entry: uv run pylint --errors-only --disable=E1120
+        entry: uv run pylint --errors-only --disable=E1120,E0401
```

Keep staged-file behavior and the existing Ruff hooks; add no wrapper script.

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
- Pylint is blocking only for `--errors-only --disable=E1120,E0401`; full-warning scores are advisory.
- `verb_noun` naming for functions, `PascalCase` for classes.
- Imports sorted by Ruff `I` rules; no eager TF imports.
- Commit style: Conventional Commits (`feat:`, `fix:`, `refactor:`, `chore:`, `test:`), subject under 72 chars.
```

Do not rewrite unrelated architecture/environment guidance.

- [ ] **Step 5: Align only README's Code Formatting subsection**

Use:

````markdown
### Code Formatting
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

- [ ] **Step 6: Run focused contract checks before committing**

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
uv run pre-commit run --all-files
git diff --check
```

Expected: all exit `0` and pre-commit does not modify tracked files.

- [ ] **Step 7: Confirm active surfaces no longer mention Black**

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

- [ ] **Step 8: Commit the implementation**

```bash
git add pyproject.toml uv.lock .pre-commit-config.yaml CLAUDE.md README.md
git commit -m "chore: normalize Python tooling contract"
```

Expected: one focused implementation commit containing only those five files.

---

### Task 2: Verify behavior and scope isolation

**Files:** verify the five implementation files; all runtime/test/evidence/workflow files remain untouched.

**Interfaces:**
- Consumes: Task 1's implementation commit.
- Produces: merge-ready evidence satisfying HPA-494 without source churn.

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

Expected: all exit `0`. HPA-494 adds no unit tests because it changes no runtime behavior.

- [ ] **Step 2: Prove forbidden surfaces are unchanged**

```bash
git diff --exit-code main...HEAD -- .github/workflows/ci.yml

if git diff --name-only main...HEAD | \
  grep -E '^(src/|tests/|runtime/|artifacts/|\.github/workflows/)'; then
  echo 'HPA-494 touched a forbidden runtime/test/evidence/workflow surface' >&2
  exit 1
fi
```

Expected: no CI diff and no forbidden path output.

- [ ] **Step 3: Verify the final file set**

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

- [ ] **Step 4: Record verification on this PR**

Add the final exit/result summary for the Task 2 commands to the existing HPA-494 PR, plus the Task 0 legacy-command statuses and whether Black disappeared from `uv.lock` or remained only transitively.

Do not create a second implementation PR.
