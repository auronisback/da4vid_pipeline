FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV OMEGAFOLD_HOME=/Omegafold

RUN apt-get update \
  && apt-get install -y build-essential git wget \
  && apt-get clean \
  && apt-get install python3 -y\
  && apt-get install python3-pip -y\
  && rm -rf /var/lib/apt/lists/* \
  && pip install --upgrade pip \
  && git clone https://github.com/HeliXonProtein/OmegaFold.git ${OMEGAFOLD_HOME} \
  && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR ${OMEGAFOLD_HOME}

RUN pip install numpy==1.26.3 && \
    pip install -r requirements.txt

CMD ["/bin/bash"]



