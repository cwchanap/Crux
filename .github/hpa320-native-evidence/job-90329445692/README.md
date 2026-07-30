# HPA-320 native-host evidence

This scratch-branch directory preserves the exact artifact downloaded from the
successful GitHub-hosted Linux X64 validation run.

- Evidence source:
  <https://github.com/cwchanap/Crux/actions/runs/30375338017/job/90329445692>
- Evidence-source workflow commit:
  `7a8e7c10849fccfc230f02c074fa49ba5c3b5217`
- Native validation and bundle run:
  <https://github.com/cwchanap/Crux/actions/runs/30375744098/job/90330841676>
- Native validation workflow commit:
  `74758236e0579e56ea26a010fbf79ea04f307474`
- Raw job API record SHA-256:
  `875b6a7bc72848f28516d6ef4aa19a468897f384647999532cc9e56b988b8f6e`

`native-host-evidence.json` passed
`python3 -m tools.hpa320.seal_oaf_backend validate-host` on the native validation
worker. The three JSON files are exact downloaded artifact bytes.

These files and the one-off workflow are scratch evidence. Do not merge them into
the final HPA-320 feature history. When the remaining seal inputs are available,
provide `native-host-evidence.json` to the native seal worker at
`/workspace/hpa320/native-host-evidence.json`.
