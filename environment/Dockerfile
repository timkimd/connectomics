# hash:sha256:881114493f8ad08f7ed5e4595c14ea3080f5d63f2ba8c8f9b6f4e91a931cb15a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api==1.2.1 \
    caveclient==7.11.0 \
    hdmf==3.14.6 \
    hdmf-zarr==0.11.3 \
    jupytext==1.17.3 \
    matplotlib==3.10.6 \
    nwbwidgets==0.11.3 \
    pandas==2.2.3 \
    pynwb==3.0.0

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
