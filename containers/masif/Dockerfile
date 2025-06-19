FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV PATH="/miniconda3/bin:${PATH}"
ARG PATH="/miniconda3/bin:${PATH}"

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget git apbs pdb2pqr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /miniconda3 && \
    rm -f miniconda.sh

COPY ./environment.yml .

ENV APBS_BIN=/usr/bin/apbs
ENV MULTIVALUE_BIN=/usr/share/apbs/tools/bin/bin/multivalue
ENV PDB2PQR_BIN=/usr/bin/pdb2pqr
ENV PATH=$PATH:/miniconda3/envs/masif-env/bin/reduce
ENV REDUCE_HET_DICT=/miniconda3/envs/masif-env/reduce_wwPDB_het_dict.txt
ENV PYMESH_PATH=/path/to/PyMesh
ENV MSMS_BIN=/miniconda3/envs/masif-env/bin/msms
ENV PDB2XYZRN=/path/to/msms/pdb_to_xyzrn


RUN conda init bash
RUN	conda env create --file environment.yml
RUN echo "conda activate masif-env" >> ~/.bashrc

# Cloning masif from github
RUN git clone https://github.com/lpdi-epfl/masif /masif
WORKDIR /masif/data/masif_site

COPY resources/prepare_from_list.sh /masif/data/masif_site
COPY resources/data_prepare_one.sh /masif/data/masif_site
COPY resources/predict_site.sh /masif/data/masif_site
COPY resources/masif_opts /masif/source/default_config/masif_opts.py
COPY resources/custom_params.py /masif/data/masif_site/nn_models/all_feat_3l/custom_params.py
RUN chmod 755 /masif/data/masif_site/prepare_from_list.sh && chmod 755 /masif/data/masif_site/data_prepare_one.sh \
    && chmod 755 /masif/data/masif_site/predict_site.sh

# Updating computation of APBS (mismatch in output files)
COPY computeAPBS.py /masif/source/triangulation