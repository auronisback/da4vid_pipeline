FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV CARBONARA_HOME=/CARBonAra

RUN apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y git software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.10 python3.10-dev python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    git clone https://github.com/LBM-EPFL/CARBonAra.git

RUN pip install ${CARBONARA_HOME} && \
    rm -r ${CARBONARA_HOME}/examples ${CARBONARA_HOME}/results

WORKDIR ${CARBONARA_HOME}