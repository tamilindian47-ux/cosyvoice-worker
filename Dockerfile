FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. System packages for audio and C++ builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    sox \
    libsox-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Pin setuptools/wheel to prevent legacy build failures
RUN pip install --no-cache-dir "setuptools<70.0.0" "wheel" "pip<24.1"

# 3. Clone CosyVoice with submodules
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice

WORKDIR /app/CosyVoice

# 4. Install pynini & runtime inference dependencies directly
RUN conda install -y -c conda-forge pynini==2.1.5 && conda clean -afy

# 5. Install exact runtime requirements needed for CosyVoice2 inference
RUN pip install --no-cache-dir \
    conformer==0.3.2 \
    diffusers==0.29.0 \
    gdown==5.1.0 \
    hydra-core==1.3.2 \
    HyperPyYAML==1.2.2 \
    inflect==7.3.1 \
    librosa==0.10.2 \
    onnx==1.16.0 \
    onnxruntime-gpu==1.18.0 \
    openai-whisper==20231117 \
    protobuf==4.25.0 \
    pydantic==2.7.0 \
    pyworld==0.3.4 \
    soundfile==0.12.1 \
    transformers==4.40.2 \
    WeTextProcessing==1.0.3 \
    runpod \
    huggingface_hub

# 6. Download CosyVoice2-0.5B model weights into image at build time
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

ENV PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

COPY handler.py /app/CosyVoice/handler.py

CMD ["python", "-u", "/app/CosyVoice/handler.py"]
