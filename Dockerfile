FROM lucferreira/cosyvoice:v1.0

# Install RunPod Serverless SDK
RUN pip install --no-cache-dir runpod huggingface_hub

# Download CosyVoice 2 model weights into the pre-configured environment
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='/opt/CosyVoice/CosyVoice/pretrained_models/CosyVoice2-0.5B')"

WORKDIR /opt/CosyVoice/CosyVoice
COPY handler.py /opt/CosyVoice/CosyVoice/handler.py

CMD ["python3", "-u", "handler.py"]
