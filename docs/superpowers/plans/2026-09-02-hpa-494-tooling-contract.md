# HPA-494 Formatter and Lint Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire Black and make Crux's already-enforced Ruff-format plus errors-only-Pylint checks the explicit repository tooling contract without changing runtime code.

**Architecture:** Treat `.github/workflows/ci.yml` as the existing target contract rather than inventing a new lint system. Remove Black from project metadata/lock state, align staged-file Pylint policy and active contributor docs with CI, and prove the change through repository tooling gates instead of production-code edits.

**Tech Stack:** Python 3.12, uv, Ruff 0.12.9, Pylint, pre-commit, pytest, Markdown/YAML/TOML configuration.

**Spec:** `docs/superpowers/specs/2026-09-02-hpa-494-tooling-contract-design.md`

## Global Constraints

- Ruff format is the only authoritative Python formatter.
- Keep the existing Python target `py312` and line length `100`.
- Do not upgrade Ruff, Pylint, pre-commit, Python, or unrelated dependencies.
- The blocking Pylint contract is exactly `--errors-only --disable=E1120,E0401`; add no further disables.
- Do not add a Pylint score threshold, baseline file, warning allowlist, suppression framework, or wrapper script.
- Do not reorganize the duplicated dev dependency lists; remove only Black from them.
- Do not rewrite historical Superpowers plans/reports that mention Black.
- Do not modify `src/`, `tests/`, `runtime/`, `artifacts/`, or `.github/workflows/` in HPA-494.
- If the current Ruff-format or errors-only-Pylint baseline is not green at Task 0, stop before editing and diagnose the moved/stale baseline rather than folding unrelated cleanup into this ticket.
- Keep design, planning, implementation, and verification in this single HPA-494 PR.

---

## File Structure

Implementation modifies only these active surfaces:

- `pyproject.toml` — remove Black from both existing dev dependency declarations and delete `[tool.black]`; keep Ruff/Pylint settings otherwise unchanged.
- `uv.lock` — regenerate from `pyproject.toml`; never hand-edit package entries.
- `.pre-commit-config.yaml` — keep the existing Ruff hooks and align the local errors-only Pylint exception list with CI.
- `CLAUDE.md` — document the canonical repository lint/format commands and the advisory status of full-warning Pylint.
- `README.md` — replace only the stale Code Formatting commands; do not turn this into a general README cleanup.

The existing `.github/workflows/ci.yml` is deliberately unchanged because it already expresses the target repository-wide contract.

---

### Task 0: Reconfirm the active and legacy baselines

**Files:**
- Read only: `.github/workflows/ci.yml`
- Read only: `.pre-commit-config.yaml`
- Read only: `pyproject.toml`
- No repository modifications

**Interfaces:**
- Consumes: current `main`-based HPA-494 branch and the repository's installed dev environment.
- Produces: evidence that the existing CI contract is green and that HPA-494 is still cleanup of two legacy nonzero commands rather than a new runtime defect.

- [ ] **Step 1: Confirm the implementation branch starts clean**

Run:

```bash
git status --short
git merge-base --is-ancestor main HEAD
```

Expected: `git status --short` has no output and the merge-base command exits `0`.

- [ ] **Step 2: Run the formatter/lint contract that already ships in CI**

Run:

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

Expected: all three commands exit `0`.

If either Ruff command or errors-only Pylint is nonzero, stop HPA-494 implementation and inspect the exact moved baseline before editing configuration. Do not reformat or suppress source as part of this plan.

- [ ] **Step 3: Characterize the two legacy HPA-494 commands without treating them as targets**

Run:

```bash
set +e
uv run black --check src tests
black_status=$?
uv run pylint src/app src/cli
full_pylint_status=$?
set -e

printf 'black_status=%s\nfull_pylint_status=%s\n' \
  "$black_status" "$full_pylint_status"

test "$black_status" -ne 0
test "$full_pylint_status" -ne 0
```

Expected: both captured statuses are nonzero, matching HPA-494's legacy-baseline description. The final two `test` commands exit `0`.

- [ ] **Step 4: Confirm characterization did not modify the tree**

Run:

```bash
git status --short
```

Expected: no output.

Task 0 intentionally creates no commit.

---

### Task 1: Retire Black from active project metadata

**Files:**
- Modify: `pyproject.toml`
- Modify (generated): `uv.lock`

**Interfaces:**
- Consumes: Task 0's green Ruff/errors-only-Pylint baseline.
- Produces: project metadata with Ruff as the only declared formatter and a lockfile regenerated from that metadata.

- [ ] **Step 1: Remove the two direct Black dependency declarations and Black configuration**

Edit `pyproject.toml` with exactly these semantic changes:

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
 constraint-dependencies = [
     "numpy<2",
     "torch==2.2.2",
     "torchaudio==2.2.2",
 ]
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

Do not change Ruff or Pylint configuration while performing this edit.

- [ ] **Step 2: Regenerate the lockfile through uv**

Run:

```bash
uv lock
```

Expected: exit `0`; `uv.lock` changes only as a consequence of removing the direct Black dependency and dependencies that are no longer reachable.

Do not hand-edit `uv.lock`.

- [ ] **Step 3: Verify project metadata no longer declares Black**

Run:

```bash
if git grep -n -i '\bblack\b' -- pyproject.toml; then
  echo 'Black is still declared in pyproject.toml' >&2
  exit 1
fi

uv lock --check
```

Expected: the grep produces no matches and `uv lock --check` exits `0`.

- [ ] **Step 4: Check whether Black remains transitively locked**

Run:

```bash
set +e
grep -n 'name = "black"' uv.lock
lock_black_status=$?
set -e
printf 'lock_black_status=%s\n' "$lock_black_status"
```

Expected: normally `lock_black_status=1`, meaning Black disappeared from the lockfile.

If the status is `0`, inspect the surrounding lockfile dependency owner before proceeding. Keep the direct/configuration removal, but do not remove or replace an unrelated dependency solely to eliminate a transitive Black package; record that owner in the PR instead.

- [ ] **Step 5: Re-run Ruff formatting before committing metadata changes**

Run:

```bash
uv run ruff format --check src tests
git diff --check
```

Expected: both exit `0`; no Python source formatting changes are needed.

- [ ] **Step 6: Commit the formatter retirement**

Run:

```bash
git add pyproject.toml uv.lock
git commit -m "chore: retire Black formatter"
```

Expected: commit succeeds and contains only `pyproject.toml` plus `uv.lock` from this task.

---

### Task 2: Align pre-commit and contributor guidance with CI

**Files:**
- Modify: `.pre-commit-config.yaml`
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Read only: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: Task 1's Black-free project metadata and the unchanged CI command contract.
- Produces: one active local/documentation contract for Ruff lint, Ruff format, and errors-only Pylint.

- [ ] **Step 1: Align the pre-commit Pylint exception list with CI**

Change only the local Pylint hook entry:

```diff
   - repo: local
     hooks:
       - id: pylint
         name: pylint (errors-only)
-        entry: uv run pylint --errors-only --disable=E1120
+        entry: uv run pylint --errors-only --disable=E1120,E0401
         language: system
         types: [python]
```

Keep pre-commit's staged-file behavior; do not add a wrapper script or force a repository-wide Pylint invocation from the hook.

- [ ] **Step 2: Replace the stale lint/format command block in `CLAUDE.md`**

Use this command block:

```bash
# Linting and formatting checks
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src

# Format code before committing
uv run ruff format src tests
```

Then make the Code Style section explicitly contain these policy bullets:

```markdown
- Python 3.12, 4-space indent, 100-character soft line limit.
- Ruff format is the canonical Python formatter; Ruff `E/F/W/I` rules provide the primary lint gate.
- Pylint is blocking only for `--errors-only --disable=E1120,E0401`; full-warning scores are advisory.
- `verb_noun` naming for functions, `PascalCase` for classes.
- Imports sorted by Ruff `I` rules; no eager TF imports.
- Commit style: Conventional Commits (`feat:`, `fix:`, `refactor:`, `chore:`, `test:`), subject under 72 chars.
```

Do not rewrite the architecture or environment sections.

- [ ] **Step 3: Replace only README's stale Code Formatting snippet**

The `## Development` / `### Code Formatting` subsection should become:

```markdown
### Code Formatting
```bash
# Check lint and formatting
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src

# Apply formatting
uv run ruff format src tests
```
```

Do not fix unrelated README architecture, deployment, endpoint, or quick-start drift in HPA-494.

- [ ] **Step 4: Verify active surfaces no longer refer to Black**

Run:

```bash
set +e
git grep -n -i '\bblack\b' -- \
  pyproject.toml README.md CLAUDE.md .pre-commit-config.yaml .github/workflows/ci.yml
active_black_status=$?
set -e

test "$active_black_status" -eq 1
```

Expected: the grep prints nothing and the final `test` exits `0`.

Historical files under `docs/superpowers/` are deliberately excluded.

- [ ] **Step 5: Verify local hooks use the same error policy as CI**

Run:

```bash
uv run pre-commit run --all-files
```

Expected: every hook passes without modifying tracked files.

If the local Pylint hook raises an error not raised by the CI-equivalent command, diagnose that concrete environment/file-specific error. Do not add a third Pylint disable as part of HPA-494.

- [ ] **Step 6: Inspect the focused diff**

Run:

```bash
git diff -- .pre-commit-config.yaml CLAUDE.md README.md
git diff --check
```

Expected: only the Pylint exception-list alignment and contributor command/policy wording described above; whitespace check passes.

- [ ] **Step 7: Commit the active-contract alignment**

Run:

```bash
git add .pre-commit-config.yaml CLAUDE.md README.md
git commit -m "chore: align lint and format contract"
```

Expected: commit succeeds with exactly the three files from this task.

---

### Task 3: Run full verification and prove scope isolation

**Files:**
- Verify: `pyproject.toml`
- Verify: `uv.lock`
- Verify: `.pre-commit-config.yaml`
- Verify: `CLAUDE.md`
- Verify: `README.md`
- Verify unchanged: `.github/workflows/ci.yml`
- Verify untouched: `src/`, `tests/`, `runtime/`, `artifacts/`

**Interfaces:**
- Consumes: Tasks 1-2 implementation commits.
- Produces: merge-ready evidence that HPA-494 changed only active tooling/documentation policy and preserved repository behavior.

- [ ] **Step 1: Verify dependency metadata is internally consistent**

Run:

```bash
uv lock --check
```

Expected: exit `0`.

- [ ] **Step 2: Run the canonical repository static gates**

Run:

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

Expected: all commands exit `0`.

- [ ] **Step 3: Run all pre-commit hooks against the repository**

Run:

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass and `git status --short` remains clean.

- [ ] **Step 4: Run the full behavior regression suite**

Run:

```bash
uv run pytest -q
```

Expected: all repository tests pass; HPA-494 adds no tests because it changes no runtime behavior.

- [ ] **Step 5: Re-run active Black-reference checks**

Run:

```bash
if git grep -n -i '\bblack\b' -- \
  pyproject.toml README.md CLAUDE.md .pre-commit-config.yaml .github/workflows/ci.yml; then
  echo 'Active repository guidance/config still references Black' >&2
  exit 1
fi
```

Expected: no matches and exit `0`.

Then check the lockfile:

```bash
set +e
grep -n 'name = "black"' uv.lock
lock_black_status=$?
set -e
printf 'lock_black_status=%s\n' "$lock_black_status"
```

Expected: `1` unless Task 1 documented a real transitive owner.

- [ ] **Step 6: Prove CI and production/evidence surfaces were not changed**

Run:

```bash
git diff --exit-code main...HEAD -- .github/workflows/ci.yml

if git diff --name-only main...HEAD | \
  grep -E '^(src/|tests/|runtime/|artifacts/|\.github/workflows/)'; then
  echo 'HPA-494 touched a forbidden runtime/test/evidence/workflow surface' >&2
  exit 1
fi
```

Expected: CI diff is empty; the scope guard prints no paths and exits successfully.

- [ ] **Step 7: Verify the complete planned file set and whitespace**

Run:

```bash
git diff --name-only main...HEAD | sort
git diff --check main...HEAD
```

Expected implementation/plan paths are limited to:

```text
.pre-commit-config.yaml
CLAUDE.md
README.md
docs/superpowers/plans/2026-09-02-hpa-494-tooling-contract.md
docs/superpowers/specs/2026-09-02-hpa-494-tooling-contract-design.md
pyproject.toml
uv.lock
```

`git diff --check main...HEAD` exits `0`.

- [ ] **Step 8: Record verification on the existing HPA-494 PR**

Update the PR body or add one top-level PR comment with the exact final command results for:

```text
uv lock --check
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
uv run pre-commit run --all-files
uv run pytest -q
git diff --check main...HEAD
```

Also record whether Black disappeared entirely from `uv.lock` or remained only as a documented transitive dependency.

Do not create a second implementation PR; implementation and verification stay on the draft HPA-494 PR created from this plan.
