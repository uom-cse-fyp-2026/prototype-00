# VM task runner. Each recipe delegates to the existing environment-aware CLI.
# Pass arguments positionally so paths containing spaces remain one argument.
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set positional-arguments

# List available commands.
default:
    @just --list

# Show VM hardware, compiler versions and the existing torch-mlir environment.
doctor:
    @bash ./nm doctor

# Reuse an existing toolchain, or bootstrap a fresh native toolchain.
setup *args:
    @bash ./nm setup "$@"

# Build the full MLIR/LLVM prototype in Release mode.
build *args:
    @bash ./nm build "$@"

# Run the automated tests on the VM.
test:
    @bash ./nm test

# Show generated configuration metadata and numerical formats.
metadata:
    @bash ./nm metadata

# Fit activation scales, the risk guard and measured kernel costs.
calibrate *args:
    @bash ./nm calibrate "$@"

# Demonstrate normal -> outlier -> normal inputs and forced variants.
demo *args:
    @bash ./nm demo "$@"

# Run one projection; for example: just run --variant a8 --regime outliers.
run *args:
    @bash ./nm run "$@"

# Benchmark static, adaptive and guard-disabled execution.
bench *args:
    @bash ./nm bench "$@"

# Summarize measured results; defaults to the results directory.
summarize *args:
    @bash ./nm summarize "$@"

# Build, test, calibrate, demonstrate, benchmark and summarize on the VM.
all *args:
    @bash ./nm all "$@"

# Independently calibrate and benchmark batches 1, 4, 8 and 16.
sweep *args:
    @bash ./nm sweep "$@"

# Package a results directory and its logs for review.
collect *args:
    @bash ./nm collect "$@"

# Build the optional Ubuntu 24.04 compiler container on the VM.
setup-container:
    @bash ./nm setup --container

# Run the complete workflow inside the optional compiler container.
all-container *args:
    @bash ./nm all --container "$@"
