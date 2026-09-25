FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system utilities & audio packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    sox \
    libsox-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pynini via conda (required by WeTextProcessing)
RUN conda install -y -c conda-forge pynini==2.1.5 && conda clean -afy

# Clone CosyVoice repository with submodules
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice

WORKDIR /app/CosyVoice

# Install Python requirements and RunPod SDK
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir runpod huggingface_hub

# Download CosyVoice2-0.5B pretrained model weights directly into container image
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

ENV PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

COPY handler.py /app/CosyVoice/handler.py

CMD ["python", "-u", "/app/CosyVoice/handler.py"]
