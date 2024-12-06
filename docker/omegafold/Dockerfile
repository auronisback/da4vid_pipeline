FROM nvidia/cuda:11.8.0-base-ubuntu22.04

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

RUN python setup.py install

CMD ["/bin/bash"]



