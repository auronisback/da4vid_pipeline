FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt update && apt upgrade -y \
    && apt install -y git software-properties-common pip \
    && ln -s $(which python3) /usr/bin/python

RUN git clone https://github.com/LBM-EPFL/PeSTo.git /PeSTo

WORKDIR /PeSTo

RUN pip install -r requirements.txt

COPY resources/run_inference.py .

CMD ["/bin/bash"]