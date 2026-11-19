# Twenty-song TF2.15 comparison

The panel uses the same September corpus as the five-song comparison. Both
workers will receive the same verified, materialized WAV for each song. The
October raw 50 ms F1 reports supply the sampling ranks.

## Membership

Start with October successes that are eligible and timing-ready in September.
Exclude the five songs already measured. Sort the 175 remaining songs by exact
raw F1 and then integer ID. Split the sorted list at indices 43, 87 and 131.

Within each stratum, select the song with the latest audio-aligned reference
onset, then the remaining song with the most common-taxonomy reference events
per second between its first and last aligned onset. Use a one-second minimum
for that denominator. Break ties by ascending ID. From the remaining songs
select the F1 ranks at zero-based indices 10, 20 and 30. The latest reference
onset is a proxy for track length available before audio acquisition.

| October F1 stratum | Latest onset | Highest density | Remaining F1 ranks |
| --- | ---: | ---: | --- |
| 1 | 131 | 384 | 329, 209, 112 |
| 2 | 68 | 56 | 201, 298, 364 |
| 3 | 194 | 216 | 138, 187, 225 |
| 4 | 294 | 154 | 262, 148, 200 |

The twenty IDs above define the scope. The semantic TF1 run ID is
`oaf-ec9176f5c314b5b6`. Verify the existing September manifest digests
`4f5d5e859650cb78a9f78697e8f19ab1ac3e7ea3c86dbea0e1f00c876357c7a3` and
`372979d42fff94438fe8d844fca2770d430a3c0898a306a4bc46bd8b2887e38f`.
Record unavailable or changed source audio by ID. The panel requires the
original audio digest for all twenty members.

## Execution and acceptance

- On one x86-64 Colab CPU VM, require an exact frozen TF1 smoke match and a
  completed, repeatable TF2.15 smoke. Record the candidate runtime fingerprint before and
  after measurement. Run each backend once through the panel in ascending
  ID order with a persistent worker. Compare per-song wall time and pooled
  real-time factor. Record candidate worker startup separately and identify
  the first TF1 call, which includes startup.
- After the primary run, repeat IDs 194 and 384 on the candidate worker and
  in a fresh candidate worker. Require each anchor's native event array to
  match its primary attempt exactly.
- Score both backends against the same common-taxonomy reference at raw 50 ms
  with zero offset and six-decimal prediction times. Report per-song TP, FP,
  FN and F1, macro F1 and pooled micro-F1. Also report native event drift at
  1 ms, possible one-frame shifts, unmatched events and confidence changes.
- A panel supports the next integration review when all twenty input seals
  and paired runs succeed, the smoke and anchor repeatability checks pass, and
  candidate pooled micro-F1 is at least the frozen TF1 result. Compare the two
  F1 fractions from integer counts. The TF2.15 smoke check records its
  binary64 differences from the oracle. Review per-song regressions, drift
  and runtime separately when deciding whether to integrate the candidate.
