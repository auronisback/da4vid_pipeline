FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update \
  && apt-get install -y build-essential git wget\
  && apt-get clean \
  && apt-get install python3 -y\
  && apt-get install python3-pip -y\
  && rm -rf /var/lib/apt/lists/* \
  && pip install --upgrade pip \
  && pip install numpy==1.26.3 torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
    torchaudio==2.0.1  --index-url https://download.pytorch.org/whl/cu118

ENV PYTHONPATH=/home/ProteinMPNN/
ENV PROTEIN_MPNN=/home/ProteinMPNN/

RUN pip3 install biopython \
  && git clone https://github.com/dauparas/ProteinMPNN.git ${PROTEIN_MPNN} \
  && rm -r ${PROTEIN_MPNN}/examples ${PROTEIN_MPNN}/inputs ${PROTEIN_MPNN}/outputs \
      ${PROTEIN_MPNN}/colab_notebooks ${PROTEIN_MPNN}/training \
  && ln -s /usr/bin/python3 /usr/bin/python

CMD ["/bin/bash"]
