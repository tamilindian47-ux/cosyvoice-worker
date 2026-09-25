FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

# 1. Install system utilities, audio tools, and compiler
RUN apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    wget \
    ffmpeg \
    unzip \
    sox \
    libsox-dev \
    git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Miniforge for conda package resolution
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:$PATH"

# 3. Create clean Python 3.10 environment and install pynini via conda
RUN conda create -y -n cosyvoice python=3.10 && \
    conda run -n cosyvoice conda install -y -c conda-forge pynini==2.1.5 && \
    conda clean -afy

ENV CONDA_DEFAULT_ENV=cosyvoice
ENV PATH="/opt/conda/envs/cosyvoice/bin:$PATH"

WORKDIR /app

# 4. Clone CosyVoice with submodules
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice

WORKDIR /app/CosyVoice

# 5. Install Python dependencies using trusted mirror & allow safe resolution
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com || \
    pip install --no-cache-dir -r requirements.txt

# 6. Install RunPod serverless SDK and huggingface hub
RUN pip install --no-cache-dir runpod huggingface_hub

# 7. Download CosyVoice2-0.5B model weights into image
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

ENV PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

COPY handler.py /app/CosyVoice/handler.py

CMD ["python", "-u", "/app/CosyVoice/handler.py"]
