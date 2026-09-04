# HPA-494 Formatter and Lint Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire Black and centralize the errors-only Pylint policy without changing runtime code.

**Architecture:** Reuse Ruff and the existing `[tool.pylint.messages_control]` table. CI stays unchanged; one five-file implementation commit performs the cleanup.

**Tech Stack:** Python 3.12, uv, Ruff 0.12.9, Pylint, pre-commit, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-hpa-494-tooling-contract-design.md`

## Global Constraints

- Ruff format is the only formatter.
- Put `E1120,E0401` in Pylint config; do not duplicate them in pre-commit/docs.
- Full-warning Pylint stays advisory.
- CI remains unchanged; its formatter scope stays `src tests` only because that is today's workflow.
- Preserve `AGENTS.md` as `120000 -> CLAUDE.md`; never stage it.
- No source/test/runtime/artifact/tool-upgrade work and no second PR.
  (Superseded for Tasks 3–4 by the post-execution addendum below.)

---

### Task 0: Confirm the active baseline

**Files:** read only

- [ ] **Step 1: Confirm a clean branch**

```bash
git status --short
git merge-base --is-ancestor main HEAD
```

Expected: clean status; merge-base exits `0`.

- [ ] **Step 2: Run the actual static gates**

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

Expected: all pass. If not, stop; do not fold source cleanup into HPA-494.

---

### Task 1: Normalize the five active files

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.pre-commit-config.yaml`
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update `pyproject.toml`**

Remove both:

```toml
"black>=24.4.0",
```

Delete:

```toml
[tool.black]
line-length = 100
target-version = ["py312"]
```

Extend the existing Pylint table:

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

Do not change other tool settings or reorganize dev lists.

- [ ] **Step 2: Regenerate and inspect `uv.lock`**

```bash
uv lock
git diff -U0 -- uv.lock | rg '^[+-](name|version) = ' | sort -u
```

Expected: only Black/Black-exclusive removals; no unrelated package version changes. Stop on unrelated churn.

- [ ] **Step 3: Simplify pre-commit Pylint**

```diff
-        entry: uv run pylint --errors-only --disable=E1120
+        entry: uv run pylint --errors-only
```

Keep staged-file behavior and Ruff hooks unchanged.

- [ ] **Step 4: Update `CLAUDE.md` and README formatting guidance**

Use the same commands in both:

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only src
uv run ruff format src tests
```

State briefly that `src tests` mirrors today's historical CI formatter scope; do not describe it as a permanent design constraint. Keep full-warning Pylint advisory. Do not perform unrelated doc cleanup.

- [ ] **Step 5: Inspect the focused diff and symlink boundary**

```bash
test -z "$(git diff --name-only -- AGENTS.md)"

if git grep -n -i 'black' -- \
  pyproject.toml README.md CLAUDE.md .pre-commit-config.yaml; then
  exit 1
fi

git diff --check
git diff -- pyproject.toml .pre-commit-config.yaml CLAUDE.md README.md
```

Expected: `AGENTS.md` unchanged, no active Black reference, no unrelated edits.

- [ ] **Step 6: Commit once**

```bash
git add pyproject.toml uv.lock .pre-commit-config.yaml CLAUDE.md README.md
git commit -m "chore: normalize Python tooling contract"
```

Expected: exactly those five implementation files.

---

### Task 2: Verify and close out

**Files:** verification only

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

Expected: all pass.

- [ ] **Step 2: Verify symlink and closed scope**

```bash
test "$(git ls-files -s AGENTS.md | awk '{print $1}')" = "120000"
test "$(git show HEAD:AGENTS.md)" = "CLAUDE.md"
test -z "$(git diff --name-only main...HEAD -- AGENTS.md)"

if git diff --name-only main...HEAD | \
  grep -E '^(src/|tests/|runtime/|artifacts/|\.github/workflows/)' | \
  grep -vE '^(\.github/workflows/ci\.yml|tests/benchmark/test_r2_inventory\.py)$'; then
  exit 1
fi

git diff --name-only main...HEAD | sort
```

Expected final paths:

```text
.github/workflows/ci.yml
.pre-commit-config.yaml
CLAUDE.md
README.md
docs/superpowers/plans/2026-09-02-hpa-494-tooling-contract.md
docs/superpowers/specs/2026-09-02-hpa-494-tooling-contract-design.md
pyproject.toml
tests/benchmark/test_r2_inventory.py
uv.lock
```

- [ ] **Step 3: Record evidence on PR #33**

Record Task 2 results plus the `uv.lock` name/version diff inspection on this PR. No second PR.

---

## Addendum: post-execution scope expansion

Executing Tasks 0–2 surfaced two follow-ups that were deliberately excluded
from the five-file scope. With the tooling contract centralized and verified,
both land on this same PR (still no second PR):

- CI's Pylint step still passes `--disable=E1120,E0401` — redundant now that
  the `[tool.pylint.messages_control]` table in `pyproject.toml` owns the
  policy.
- `tests/benchmark/test_r2_inventory.py` imports `botocore` at module level;
  with `boto3` under the optional `r2` extra, plain `uv run pytest` fails
  collection. Pre-existing on `main`; CI installs boto3 explicitly, so only
  bare local environments hit it.

Amended constraints (supersede the originals for Tasks 3–4 only):

- Task 3 may modify `.github/workflows/ci.yml` — the Pylint flag removal only.
- Task 4 may modify `tests/benchmark/test_r2_inventory.py` — the skip guard only.

### Task 3: Align CI Pylint with the centralized policy

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Drop redundant disables**

```diff
       - name: Pylint (errors-only)
         run: |
-          pylint --errors-only --disable=E1120,E0401 src
+          pylint --errors-only src
```

No other workflow changes. CI's pip-installed Pylint reads the same
`pyproject.toml` table from the repo root.

### Task 4: Gate r2 inventory tests on boto3 availability

**Files:**
- Modify: `tests/benchmark/test_r2_inventory.py`

- [ ] **Step 1: Skip the module when the `r2` extra is absent**

Insert `pytest.importorskip("boto3")` after the stdlib/pytest imports and
before the `botocore` / `src.benchmark` imports (mark the moved imports
`# noqa: E402`). Gate on `boto3` (the dependency the `r2` extra declares), not
`botocore` (a transitive dependency that can be present without `boto3`).
Tests still run in CI (boto3 installed) and under `uv run --extra r2 pytest`.

### Addendum verification

- [ ] `uv run pytest -q` collects and passes in a bare environment (module skips)
- [ ] `uv run --extra r2 pytest -q tests/benchmark/test_r2_inventory.py` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check src tests` pass
- [ ] `uv run pylint --errors-only src` still passes
