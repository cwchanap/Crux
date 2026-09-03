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
  grep -E '^(src/|tests/|runtime/|artifacts/|\.github/workflows/)'; then
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

- [ ] **Step 3: Record evidence on PR #33**

Record Task 2 results plus the `uv.lock` name/version diff inspection on this PR. No second PR.
