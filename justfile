# Project-local VM workflows. Long jobs run in a project-specific tmux server.
# Ctrl+B then D detaches; close SSH and return with just attach.
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set positional-arguments

# List available commands.
default:
    @just --list

# Show VM hardware, compiler versions and this project's environment status.
doctor:
    @bash ./nm doctor

# Install this project's own Python/GCC/LLVM/torch-mlir stack in tmux.
setup *args:
    @/usr/bin/python3 tools/jobs.py start setup "$@"

# Start toolchain setup detached; return immediately so SSH can be closed.
setup-bg *args:
    @/usr/bin/python3 tools/jobs.py start --detach setup "$@"

# Detached setup with more CPU parallelism; customize: just setup-fast 24 2.
setup-fast jobs="16" links="2":
    @JOBS="$1" LINK_JOBS="$2" /usr/bin/python3 tools/jobs.py start --detach setup

# Foreground setup with the command's actual exit code (no tmux).
setup-fg *args:
    @bash ./nm setup "$@"

# Build the full MLIR/LLVM prototype in Release mode.
build *args:
    @bash ./nm build "$@"

# Run the automated tests on the VM.
test:
    @bash ./nm test

# Check job-state/argument/PID safety on the VM before the heavy toolchain build.
test-jobs:
    @/usr/bin/python3 tests/test_jobs.py

# Show generated configuration metadata and numerical formats.
metadata:
    @bash ./nm metadata

# Fit activation scales, the risk guard and measured kernel costs.
calibrate *args:
    @bash ./nm calibrate "$@"

# Demonstrate normal -> outlier -> normal inputs and forced variants.
demo *args:
    @bash ./nm demo "$@"

# Build/test/calibrate/demo/benchmark in tmux, like the quant-sensitivity workflow.
run *args:
    @/usr/bin/python3 tools/jobs.py start all "$@"

# Start the complete workflow detached and return immediately.
start *args:
    @/usr/bin/python3 tools/jobs.py start --detach all "$@"

# Run one projection; for example: just projection --variant a8 --regime outliers.
projection *args:
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

# Start any supported long task detached, e.g. just background build.
background task *args:
    @/usr/bin/python3 tools/jobs.py start --detach "$@"

# Reattach to the latest job's tmux pane; Ctrl+B then D detaches again.
attach:
    @/usr/bin/python3 tools/jobs.py attach

alias fg := attach

# Show actual completion state/exit code and the end of the latest console log.
status:
    @/usr/bin/python3 tools/jobs.py status

# Follow the latest job log without attaching; Ctrl+C only stops this viewer.
logs:
    @/usr/bin/python3 tools/jobs.py logs

# Stop this project's active job and its subprocess group; preserve logs.
stop:
    @/usr/bin/python3 tools/jobs.py stop

# List this project's recorded setup/experiment jobs.
jobs:
    @/usr/bin/python3 tools/jobs.py list

# Update this checkout; refuse while a project job is still running.
pull:
    @/usr/bin/python3 tools/jobs.py guard
    git pull --ff-only

# Package a results directory and its logs for review.
collect *args:
    @bash ./nm collect "$@"

# Build the optional Ubuntu 24.04 compiler container on the VM.
setup-container:
    @bash ./nm setup --container

# Run the complete workflow inside the optional compiler container.
all-container *args:
    @bash ./nm all --container "$@"
