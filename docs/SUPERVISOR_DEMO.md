# Supervisor demonstration

Run all commands on the Ubuntu VM. Use the existing toolchain configuration
from `mlir-quant-linear`; `./nm` activates the environment automatically.

## Preparation and evidence

```bash
git pull --ff-only
just doctor
just build
just test
just metadata
```

Show the actual CTest output. Do not describe tests as passed unless this run
succeeds. `doctor` records guest-visible CPU features and tools rather than
assuming NVIDIA or host CPU capabilities are available to the guest.

## Calibrate then freeze

```bash
just calibrate
```

Explain the fixed numerical budget and separate calibration/test seeds. Show
the measured BF16/A8 costs, learned clipping scale and whether the risk guard
qualified. Calibration labels use reference outputs; runtime selection does not.

## Show the runtime interaction

```bash
just demo
```

The demonstration supplies ordinary activations twice, a genuine outlier
input, then returns to the same ordinary input. Inspect the actual printed
descriptor, cache status, selected executor, fallback and reason. Cache hits
still check the current predicate. Output error is computed separately for
evaluation. Timings and variant choices depend on the VM and fitted policy.

If all adaptive calls choose BF16, show the reason. The subsequent forced BF16
and A8 runs demonstrate the two numerical implementations separately; they do
not show that the adaptive policy selected A8.

## Show the compiler contribution

```bash
less build/generated/linear.linalg.mlir
less build/generated/executors.prepared.mlir
less build/generated/executors.llvm.mlir
```

Point out torch-mlir's imported projection, packed-nibble loads, INT32 A8
accumulators, BF16 bit conversion, generated `nm_risk_guard`, and native
dispatch interface. Explain that these are ahead-of-time variants and a small
real preparation pass; arbitrary graph quantization and runtime JIT are future
work.

## Compare complete invocation costs

```bash
just bench
just summarize results
```

Compare original FP32 reference, static BF16, static A8, guarded adaptive and
guard-disabled ablation. Show inspection and total overhead alongside kernel
latency; include false-safe/false-fallback results. Do not promise a speedup.

For a self-contained logged run use `just all`. For batch comparisons use
`just sweep`; each shape is independently calibrated. For sampled features use
`just all --sampled` with its own policy and held-out evaluation.

## Research questions to discuss

1. Is the observed risk discrimination sufficient to justify real LLM traces?
2. Does inspection pay for itself at decode-sized batches?
3. Should the next compiler step replace the template emitter with a real
   Linalg rewrite, or first optimize/fuse descriptor computation?
4. What held-out false-safe tolerance and primary metric should gate expansion?

The demonstration establishes at most an operator-level interaction. It does
not establish model quality, end-to-end efficiency, energy savings, GPU benefit
or a completed MDE/JIT framework.
