# Optional image with this prototype's own stack, independent of other projects.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl bzip2 ca-certificates python3 && rm -rf /var/lib/apt/lists/*
COPY toolchain.lock.json env.sh /opt/neurominds-stack/
COPY configs/host-environment.yml /opt/neurominds-stack/configs/host-environment.yml
COPY scripts/setup_toolchain.sh /opt/neurominds-stack/scripts/setup_toolchain.sh
ENV NM_ENV_ROOT=/opt/neurominds-stack JOBS=2
RUN bash /opt/neurominds-stack/scripts/setup_toolchain.sh
WORKDIR /work
CMD ["bash", "./nm", "help"]
