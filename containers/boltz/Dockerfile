FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG WORK_DIRECTORY="/boltz"

RUN apt update && apt upgrade -y && apt install -y git wget gcc software-properties-common \
	&& add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && apt install -y python3.11 python3.11-venv python3.11-dev \
	&& ln -s $(which python3.11) /usr/bin/python && python -m ensurepip && ln -s $(which pip3.11) /usr/bin/pip

RUN git clone https://github.com/jwohlwend/boltz.git ${WORK_DIRECTORY}
WORKDIR ${WORK_DIRECTORY}
RUN pip install -e .

CMD ["/bin/bash"]
