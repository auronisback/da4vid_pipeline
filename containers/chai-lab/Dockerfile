FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  EDITOR=vim \
  MYPY_CACHE_DIR='/tmp/.chai_lab_mypy_cache' \
  PYTHONCACHEPREFIX='/tmp/.chai_lab_pycache' \
  CHAI_HOME='/chai-lab'

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update  \
    && apt-get -qq upgrade \
    && apt-get -qq install -y \
      gnupg ca-certificates wget git curl aria2 lsb-release tzdata rsync \
      tree htop tmux unzip clang psmisc libibverbs1 librdmacm1 nano vim \
      build-essential libstdc++6 python3.10 python3.10-dev python3-pip kalign \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && git clone https://github.com/chaidiscovery/chai-lab.git ${CHAI_HOME}

WORKDIR ${CHAI_HOME}

ENV PYTHONPATH=${CHAI_HOME}

RUN pip3 install -r requirements.in
RUN printf '#!/bin/bash\n\npython ${CHAI_HOME}/chai_lab/main.py $@\n' > /usr/bin/chai-lab \
    && chmod 775 /usr/bin/chai-lab

