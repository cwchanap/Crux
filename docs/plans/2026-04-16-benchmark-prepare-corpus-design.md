# Benchmark Prepare-Corpus Design

## Purpose

Add a preparation step that converts raw DTX song folders into a canonical parsed
benchmark corpus. Future benchmark work should operate only on the parsed corpus, not
on the raw source folders.

## Goals

- Normalize one raw song folder into one benchmark item.
- Select exactly one chart per song using difficulty priority `mas > ext > adv > bas`.
- Select exactly one drum-stem audio file per song using an explicit filename
  allowlist.
- Write a deterministic parsed corpus that can be fed directly into existing
  benchmark commands.
- Emit manifest and invalid-item reports so the preparation step is auditable.

## Non-Goals

- Do not make score/transcription commands parse raw song folders directly.
- Do not try to infer drum stems heuristically from arbitrary filenames.
- Do not silently choose between multiple allowed stem files.
- Do not mutate raw song folders in place.

## Parsed Corpus Layout

Use a split-directory parsed corpus:

```text
parsed/
  charts/
    <song_id>.dtx
  audio/
    <song_id>.<ext>
  manifest.json
  invalid.json
```

`song_id` comes from the raw folder name, not the chosen chart filename. This gives a
stable ID shared by chart, audio, reports, predictions, and future benchmark runs.

## Preparation Workflow

Add a new CLI command:

```text
crux benchmark prepare-corpus --raw-dir <raw_root> --output-dir <parsed_root>
```

The command scans each immediate child directory of `raw_root` as one song folder.
For each folder it:

1. Picks the highest available chart from `mas`, `ext`, `adv`, `bas`.
2. Picks the drum stem from an explicit allowlist.
3. Copies both files into the parsed corpus using the folder name as `song_id`.
4. Records the decision in `manifest.json`.

If a folder cannot be normalized safely, it is skipped and recorded in `invalid.json`.

The parsed corpus becomes the only supported input for later benchmark work:

- `crux benchmark score-midi` uses `parsed/charts` and `parsed/predictions`
- `crux benchmark transcribe-and-score` uses `parsed/charts` and `parsed/audio`

## Audio Selection

Use an explicit filename allowlist rather than heuristic discovery. Define a constant
that is easy to edit later, for example:

```python
DRUM_AUDIO_FILENAMES = ("2 Drums.mp3", "drum.mp3")
```

Matching should be case-insensitive. Rules:

- If exactly one allowed filename exists in the raw song folder, use it.
- If more than one allowed filename exists, mark the folder invalid.
- If none exist, mark the folder invalid.

This keeps preparation deterministic and avoids silently choosing the wrong file in a
folder that also contains `bgm.ogg`, previews, and sample assets.

## Chart Selection

Use fixed difficulty priority:

```text
mas > ext > adv > bas
```

If multiple files for the same selected level exist, the folder should be marked
invalid rather than guessed. If no recognized chart exists, the folder is invalid.

## Outputs

`manifest.json` should contain one entry per valid parsed item with at least:

- `song_id`
- `raw_folder`
- `selected_chart`
- `selected_chart_level`
- `selected_audio`
- `parsed_chart_path`
- `parsed_audio_path`

`invalid.json` should contain one entry per rejected raw folder with:

- `raw_folder`
- `reason`
- optional `details`

## Validation Philosophy

Preparation is intentionally conservative. It should fail loudly on ambiguous raw
folders instead of guessing. The parsed corpus should be safe to reuse for benchmark
runs, sharing, and archiving without carrying the raw-folder ambiguity forward.
