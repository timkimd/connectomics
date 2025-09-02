# hash:sha256:29571bc80e807d2447ace5418d16d540a25186620c0f3ff9b59257e41f632674
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN conda install -y \
        opencv==4.8.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-access-api==1.2.1 \
    caveclient==7.11.0 \
    hdmf==3.14.6 \
    hdmf-zarr==0.11.3 \
    jupytext==1.17.3 \
    matplotlib==3.10.6 \
    numpy==2.2.6 \
    nwbwidgets==0.11.3 \
    pandas==2.2.3 \
    pynwb==3.0.0 \
    ruff==0.12.11 \
    scipy==1.15.3 \
    seaborn==0.13.2 \
    skeleton-plot==0.1.0 \
    standard-transform==1.4.1 \
    tqdm==4.67.1 \
    typing==3.7.4.3

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
