# Benchmark Artifact Defaults Design

## Goal

Make benchmark commands write artifacts into a repo-local directory by default, while still
allowing explicit `--output-dir` overrides for reproducible or custom runs.

## Decisions

- Default benchmark artifact root: `artifacts/benchmark/`
- `--output-dir` remains supported and takes precedence over any default resolution
- Add optional `--run-name` to benchmark commands that write outputs
- If `--output-dir` is omitted:
  - `prepare-corpus` uses the basename of `--raw-dir`
  - `score-midi`, `transcribe-and-score`, and `export-reference-midi` use the basename of
    `--charts-dir`
  - if `--run-name` is provided, it replaces the derived basename
- Ignore the entire repo-local `artifacts/` directory in `.gitignore`

## CLI Behavior

Output path resolution should be shared by benchmark commands rather than duplicated in each
command. The resolved directory shape is:

```text
artifacts/benchmark/<run-name-or-input-dir-name>/
```

Examples:

```text
crux benchmark prepare-corpus --raw-dir raw/Test DTX
-> artifacts/benchmark/Test DTX/

crux benchmark score-midi --charts-dir parsed/charts --predictions-dir parsed/predictions
-> artifacts/benchmark/charts/

crux benchmark score-midi --charts-dir parsed/charts --predictions-dir parsed/predictions --run-name soukyuu-stem-test
-> artifacts/benchmark/soukyuu-stem-test/
```

The benchmark services do not need to change. Only the CLI resolves a default output directory
before passing it into the existing service functions.

## Testing

Add CLI tests that prove:

- `prepare-corpus` uses `artifacts/benchmark/<raw-dir-name>/` when `--output-dir` is omitted
- `score-midi` uses `artifacts/benchmark/<run-name>/` when `--run-name` is provided
- explicit `--output-dir` still works
- `artifacts/` is git-ignored
