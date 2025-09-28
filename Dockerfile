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

# Install Python 3.10 from deadsnakes PPA
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

# Download protoc
RUN mkdir -p /opt/protoc && cd /opt/protoc && \
    curl -LjO https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protoc-31.1-linux-x86_64.zip \
    && unzip protoc-31.1-linux-x86_64.zip \
    && rm -f protoc-31.1-linux-x86_64.zip \
    chmod +x bin/protoc && \
    ln -s /opt/protoc/bin/protoc /usr/bin/protoc

# Create virtual environment
RUN python3.10 -m venv /opt/venv && \
/opt/venv/bin/pip install --upgrade pip setuptools wheel && \
/opt/venv/bin/pip cache purge

# Update PATH to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Prepare conda env
RUN /opt/venv/bin/pip install toml-to-requirements && \
    /opt/venv/bin/pip cache purge && \
    conda clean --all

# COPY pyproject.toml and export requirements
COPY pyproject.toml /opt/pyproject.toml
RUN cd /opt && \
    /opt/venv/bin/pip install toml-to-requirements && \
    toml-to-req --toml-file pyproject.toml --optional-lists dev && \
    /opt/venv/bin/pip install -r requirements.txt && \
    /opt/venv/bin/pip cache purge

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
ENTRYPOINT ["/opt/venv/bin/python", "main.py", "--config_path", "configs/local.py"]
