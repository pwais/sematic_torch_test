# Copyright 2023 Maintainers of sematic_torch_test
# syntax=docker/dockerfile:1

# Fetch pytorch and submodules in a separate image / cache
FROM curlimages/curl:8.00.1 AS torch-wheel-fetch
RUN \
  cd /tmp && \
  curl -L --retry 3 https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl \
    -o torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl
FROM curlimages/curl:8.00.1 AS torchvision-wheel-fetch
RUN \
  cd /tmp && \
  curl -L --retry 3 https://download.pytorch.org/whl/cpu/torchvision-0.15.2%2Bcpu-cp310-cp310-linux_x86_64.whl \
    -o torchvision-0.15.2+cu118-cp310-cp310-linux_x86_64.whl

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# ENV DEBIAN_FRONTEND noninteractive
# ENV TZ America/Los_Angeles
ENV PYTHONDONTWRITEBYTECODE 1

# Sematic worker image wants python3 available as /usr/bin/python3
# TODO(sematic) allow /usr/bin/python
RUN \
  apt-get update && \
  apt-get install -y \
    python-dev-is-python3 \
    python3-dev \
    python3-pip \
  && \
  pip install --upgrade pip \
  && \
  /usr/bin/python3 --version


# Sematic likes to have some system libraries available
RUN \
  --mount=type=cache,target=/var/cache/apt \
  apt-get update && \
  apt-get install -y \
    git \
    libmagic1 \
  && \
  pip install sematic==0.34.1

# Install torch+CUDA from cached wheels
COPY --from=torch-wheel-fetch /tmp/torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl /tmp/torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl
COPY --from=torchvision-wheel-fetch /tmp/torchvision-0.15.2+cu118-cp310-cp310-linux_x86_64.whl /tmp/torchvision-0.15.2+cu118-cp310-cp310-linux_x86_64.whl
RUN pip install -v \
  /tmp/torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl \
  /tmp/torchvision-0.15.2+cu118-cp310-cp310-linux_x86_64.whl \
    --extra-index-url https://download.pytorch.org/whl/cu118 && \
  rm /tmp/torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl && \
  rm /tmp/torchvision-0.15.2+cu118-cp310-cp310-linux_x86_64.whl

# Install developer tools
RUN \
  --mount=type=cache,target=/var/cache/apt \
  apt-get update && \
  apt-get install -y \
    curl \
    net-tools \
    vim \
  && \
  pip install \
    yapf

# Install anything that our sematic_torch_test module needs
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN rm -f /tmp/requirements.txt

# Ensure in dev container that host-mounted code is on PYTHONPATH
ENV PYTHONPATH $PYTHONPATH:/opt/sematic_torch_test