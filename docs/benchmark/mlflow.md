# Publishing Benchmark Reports to MLflow

`crux benchmark publish-mlflow-cohort` publishes one canonical HPA-325 cohort
report directory as an idempotent MLflow run so benchmark results can be
visualized, filtered, and compared in an MLflow tracking UI.

## Canonical and disposable data

Crux reports are canonical. The published cohort report directory is the
source of truth; MLflow holds only a disposable projection. Any MLflow run can
be deleted and rebuilt by republishing from the canonical reports — nothing in
the repository or in the reports depends on MLflow state.

## Installation

The publisher is an optional extra; the base environment never imports mlflow.

```bash
uv sync --extra mlflow
```

## Environment setup

The command reads its configuration from the environment:

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | Required. Tracking server URL (see constraints below). |
| `MLFLOW_TRACKING_USERNAME` | Optional. Username for the tracking server. |
| `MLFLOW_TRACKING_PASSWORD` | Optional. Password for the tracking server. |

`MLFLOW_TRACKING_URI` must be an explicit `http://` or `https://` server URL
with a hostname. The command rejects local file/sqlite fallbacks, URLs with
embedded credentials, and URLs with a query or fragment. Keep credentials in
the separate username/password variables, never in the URI. Path-bearing
https URLs (as used by some hosted providers) are accepted.

## Usage

```bash
crux benchmark publish-mlflow-cohort --reports PATH --scope broad
```

- `--reports PATH` (required): one published cohort report directory
  (the directory containing `summary.json` and its sibling report artifacts).
- `--scope broad|reviewed|pilot` (required, no default): the display scope
  published with this run.
- `--experiment NAME` (optional): target MLflow experiment name;
  defaults to `crux-benchmark`.

On success the command prints one canonical JSON object with `created`,
`experiment_id`, `experiment_name`, `projection_sha256`, `run_id`, and
`status` (`published` or `already_published`). Failures exit `2` with a
bounded error code on stdout and a sanitized message on stderr; vendor detail
stays out of the CLI output.

## What is uploaded

Exactly five small report artifacts are attached under the `crux-reports/`
artifact directory:

```text
summary.json
summary.md
items.csv
per_song.csv
per_class.csv
```

Nothing else is uploaded. Diagnostics, audio, stems, predictions, and
checkpoints are explicitly never uploaded to MLflow; they remain in the
canonical report/corpus locations.

## Metrics, tags, and UI filters

Each benchmark view is flattened into MLflow metrics:

- `event_micro.precision|recall|f1.<tolerance>ms.<mode>` — micro event metrics
- `song_macro.f1.<tolerance>ms.<mode>` and `class_macro.f1.<tolerance>ms.<mode>`
- `song_f1.<stat>.<tolerance>ms.<mode>` — per-song F1 distribution
  (`minimum`, `p10`, `p25`, `median`, `p75`, `p90`, `maximum`)
- `class.<class>.precision|recall|f1.<tolerance>ms.<mode>` — per-class metrics
- `class.<class>.reference_support.<tolerance>ms.<mode>` and
  `class.<class>.prediction_support.<tolerance>ms.<mode>`
- `population.total|success|failed|skipped|quarantined` and
  `population.reason.<reason>` — population counts

`<tolerance>` is 30, 50, or 100 ms and `<mode>` is `raw` or `aligned`; the
baseline 30/50/100 raw/aligned views are always required.

Runs carry the projection identity as `crux.*` tags, including
`crux.scope`, `crux.cohort_id`, `crux.backend_id`, `crux.model_id`,
`crux.input_view_id`, and the content fingerprint `crux.projection_sha256`.
Run names follow `<scope>-<model_id>-<input_view_id>-<cohort_id>`.

Useful tracking-UI filters:

```text
tags."crux.scope" = 'broad'
tags."crux.scope" = 'reviewed'
tags."crux.scope" = 'pilot'
tags."crux.backend_id" = '<backend-id>'
```

Filtering by `crux.scope` shows the broad/reviewed/pilot views; runs from
full-mix, Spleeter, and HTDemucs backends share the same tag schema, so they
can be compared side by side in one experiment.

## Scope is part of run identity

`--scope` participates in both the projection fingerprint and the idempotency
search key. Publishing the same reports under a different scope creates a
distinct run; it never conflicts with the original. Correcting a mistyped
scope is therefore just a republish with the right `--scope`.

## Idempotency and retry semantics

Publication is idempotent per search identity
(`crux.cohort_id` + `crux.projection_version` + `crux.scope`):

- Re-publishing an identical projection returns the existing run ID with
  `created=false` and `status=already_published`. No duplicate run is created.
- A matching run in `FAILED` or `KILLED` state is ignored as retriable: a new
  run is created and published.
- Any other conflicting state (a matching run that is still `RUNNING`, more
  than one `FINISHED` match, or a fingerprint mismatch within the search
  identity) fails with the `run_conflict` error code.

## Recovering from a true same-scope conflict

A true conflict means two different projections claim the same scope and
cohort identity. Because Crux reports are canonical and MLflow is disposable,
recovery is:

1. Open the MLflow tracking UI and identify the incorrect run for that scope
   and cohort (check `crux.projection_sha256`).
2. Delete the incorrect MLflow run in the tracking UI.
3. Republish from the canonical Crux reports with the correct `--reports`
   and `--scope`.

## Changing MLflow provider

Switching providers (self-hosted MLflow, DagsHub, or any HTTP tracking
server) requires environment changes only: point `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` at the new server
and rerun the publish commands. No code or report changes are involved, and
the old server's runs can simply be abandoned or rebuilt from canonical
reports.
