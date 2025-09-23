FROM ubuntu:20.04

# Install apt packages
RUN apt-get update && \
    apt-get install -y \
        wget curl git vim unzip \
        gcc g++ make \
    && \
    apt-get autoclean

# Set timezone
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -yq tzdata && \
    dpkg-reconfigure -f noninteractive tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get autoclean

# Install miniconda
RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Download protoc
RUN mkdir -p /opt/protoc && cd /opt/protoc && \
    curl -LjO https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protoc-31.1-linux-x86_64.zip \
    && unzip protoc-31.1-linux-x86_64.zip \
    && rm -f protoc-31.1-linux-x86_64.zip \
    && chmod +x bin/protoc

# Update in bashrc
RUN echo "source /opt/miniconda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda deactivate" >> /root/.bashrc && \
    ln -s /opt/protoc/bin/protoc /usr/bin/protoc && \
    protoc --version

# Prepare conda env
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda config --add channels conda-forge && \
    conda tos accept && \
    conda create -n speech2motion python=3.10 -y && \
    conda activate speech2motion && \
    pip install toml-to-requirements && \
    pip cache purge && \
    conda clean --all

# COPY pyproject.toml and export requirements
COPY pyproject.toml /opt/pyproject.toml
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate speech2motion && \
    cd /opt && \
    toml-to-req --toml-file pyproject.toml --optional-lists dev && \
    pip install -r requirements.txt && \
    pip cache purge

# COPY code
COPY . /workspace/speech2motion
# Install code
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate speech2motion && \
    cd /workspace/speech2motion && \
    pip install . && \
    pip cache purge

# Set working directory
WORKDIR /workspace/speech2motion

# Set entrypoint
ENTRYPOINT ["/bin/bash", "-c", "source /opt/miniconda/etc/profile.d/conda.sh && conda activate speech2motion && python main.py"]
