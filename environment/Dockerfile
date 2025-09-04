# hash:sha256:b097e8a8fe975aae6b16708182c599cc32617363df40b88b46db29fd5dc38b08
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        texlive-fonts-extra \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-access-api==1.2.1 \
    caveclient==7.11.0 \
    hdmf==3.14.6 \
    hdmf-zarr==0.11.3 \
    jupytext==1.17.3 \
    matplotlib==3.10.6 \
    nwbwidgets==0.11.3 \
    pandas==2.2.3 \
    pynwb==3.0.0 \
    scipy==1.15.3 \
    seaborn==0.13.2 \
    skeleton-plot==0.1.0 \
    tqdm==4.67.1 \
    typing==3.7.4.3

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
