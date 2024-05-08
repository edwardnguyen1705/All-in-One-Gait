ARG BASE_IMAGE=nvidia/11.6.1-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev python3-pip git wget sudo ninja-build libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install numpy==1.20.0 pandas scikit-image
COPY requirements.txt /opt/tmp/requirements.txt
RUN pip3 install -r /opt/tmp/requirements.txt
WORKDIR /app
