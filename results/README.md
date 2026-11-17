# Experiment outputs

No measurements have been produced during Windows authoring.

`just all` on the Ubuntu VM writes a timestamped run directory containing raw
calibration and benchmark CSVs, the frozen policy, provenance manifests, compiler
IR, terminal logs, and generated summaries. `just sweep` creates separate policies
and measurements for batches 1, 4, 8 and 16.

Managed `just run` / `just start` jobs instead use
`results/jobs/<job-id>/experiment/`. Their `console.log` and `state.json` remain
in the parent job directory. `just setup` / `just setup-bg` also use persistent
job logs there. `just status` reports the actual recorded exit status; a
detached SSH/tmux client does not imply successful completion.

Runtime measurements are ignored by Git to avoid presenting unverified or
machine-specific numbers as source. Use `just collect results/<run-directory>`
to package a run for review. The collector includes the recorded environment
and measurements, not SSH credentials or unrelated machine files.
