# hash:sha256:a8f446278ada335f294b844a0a743fc43c2dc68fb328e3eed69f0dbc6e7527a6
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf==3.14.6 \
    hdmf-zarr==0.11.3 \
    jupytext \
    matplotlib \
    nwbwidgets==0.11.3 \
    pynwb==3.0.0
