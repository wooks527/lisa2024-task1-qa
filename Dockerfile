FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-dev \
        curl \
        ca-certificates \
        git \
        && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python3 -m pip install --upgrade pip
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --trusted-host download.pytorch.org -r /tmp/requirements.txt

WORKDIR /workspace
