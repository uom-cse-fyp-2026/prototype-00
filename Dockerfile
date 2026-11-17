# Optional compiler environment for an Ubuntu 20.04 VM host.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates build-essential cmake ninja-build clang lld \
    python3 python3-dev python3-venv libzstd-dev zlib1g-dev libedit-dev \
    libxml2-dev libffi-dev && rm -rf /var/lib/apt/lists/*
COPY toolchain.lock.json /opt/neurominds-setup/toolchain.lock.json
COPY scripts/setup_toolchain.sh /opt/neurominds-setup/scripts/setup_toolchain.sh
ENV TORCH_MLIR_SRC=/opt/torch-mlir TORCH_MLIR_BUILD=/opt/torch-mlir/build JOBS=2
RUN bash /opt/neurominds-setup/scripts/setup_toolchain.sh
WORKDIR /work
CMD ["bash", "./nm", "help"]
