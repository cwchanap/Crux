# Benchmark Mapping Calibration Design

## Goal

Add a one-off analysis script that calibrates the current `models/e-gmd/tf2_model.weights.h5`
artifact against the prepared benchmark corpus, without changing production decoding.

## Scope

The script will:

- load prepared corpus items from `charts/` and `audio/`
- run the current TF2 transcription model to obtain raw `88`-bin onset outputs
- build candidate mappings from output bins to canonical drum classes
- score those candidates with the existing benchmark scorer
- write ranked calibration artifacts under `artifacts/benchmark/`

The script will not:

- change `crux benchmark` CLI behavior
- update production transcriber mappings automatically
- attempt to train a new model

## Approach

Use a standalone Python script under `scripts/` that reuses existing modules in
`src/app/transcriber.py` and `src/benchmark/*`. The script should operate on the prepared corpus,
so future runs only need the normalized `charts/` and `audio/` directories produced by
`prepare-corpus`.

The calibration logic should stay simple and empirical:

1. Parse DTX ground truth and map it to canonical benchmark classes.
2. Run the TF2 model once per audio file and extract per-bin onset peaks.
3. Estimate candidate bin-to-class assignments from temporal agreement with ground truth.
4. Score several candidate mapping strategies with the existing benchmark scorer.
5. Emit JSON reports that show the best mappings, scores, and class-distribution diagnostics.

## Outputs

The script should write a run directory under `artifacts/benchmark/` containing:

- `summary.json`
- `best_mapping.json`
- `per_chart.json`

## Success Criteria

- The script runs end to end on `artifacts/benchmark/Test DTX/`.
- It produces a reproducible ranked mapping report for `Soukyuu e no shouka`.
- No production benchmark or transcription behavior changes unless we later decide to promote a
  calibrated mapping.
