# Experiment outputs

No measurements have been produced during Windows authoring.

`just all` on the Ubuntu VM writes a timestamped run directory containing raw
calibration and benchmark CSVs, the frozen policy, provenance manifests, compiler
IR, terminal logs, and generated summaries. `just sweep` creates separate policies
and measurements for batches 1, 4, 8 and 16.

Runtime measurements are ignored by Git to avoid presenting unverified or
machine-specific numbers as source. Use `just collect results/<run-directory>`
to package a run for review. The collector includes the recorded environment
and measurements, not SSH credentials or unrelated machine files.
