FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS pt25

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        gnutls-bin \
        gnutls-dev \
        libarchive-dev \
        libboost-all-dev \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        rapidjson-dev \
        wget \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

# RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
#     # To activate a particular environment
#     # echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     # echo "conda activate <env_name>" >> ~/.bashrc

# COPY .condarc /opt/conda/

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /media/data_2/opensource/LLMs-from-scratch
