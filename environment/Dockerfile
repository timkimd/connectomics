# hash:sha256:9a5cfbedd252951fcefeba513dd0abdd75c5ea91407a47674bf9c2d7bcb9c3e9
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf==3.14.6 \
    hdmf-zarr==0.11.3 \
    nwbwidgets==0.11.3 \
    pynwb==3.0.0
