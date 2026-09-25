import os
import io
import sys
import base64
import tempfile
import torchaudio
import runpod
from pathlib import Path

# Add third_party paths
sys.path.append("/app/CosyVoice")
sys.path.append("/app/CosyVoice/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

MODEL_PATH = "pretrained_models/CosyVoice2-0.5B"

# Preload model onto CUDA during worker initialization
print("Loading CosyVoice2-0.5B weights onto GPU...")
cosyvoice = CosyVoice2(MODEL_PATH, load_jit=False, load_trt=False, fp16=True)
print("CosyVoice2 model loaded successfully!")

def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text", "").strip()
    reference_b64 = job_input.get("reference_b64")
    speed = float(job_input.get("speed", 1.0))

    if not text:
        return {"error": "Missing 'text' input to synthesize."}
    if not reference_b64:
        return {"error": "Missing 'reference_b64' audio file for voice cloning."}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ref_raw = tmp / "ref.raw"
        ref_wav = tmp / "ref.wav"
        out_mp3 = tmp / "output.mp3"

        # Decode base64 reference audio
        ref_raw.write_bytes(base64.b64decode(reference_b64))

        # Normalize reference audio to 16kHz mono WAV using ffmpeg
        os.system(f"ffmpeg -y -i {ref_raw} -ac 1 -ar 16000 {ref_wav} >/dev/null 2>&1")

        prompt_speech_16k = load_wav(str(ref_wav), 16000)

        # Run cross-lingual zero-shot voice cloning
        audio_segments = []
        for chunk in cosyvoice.inference_cross_lingual(text, prompt_speech_16k, stream=False, speed=speed):
            audio_segments.append(chunk["tts_speech"])

        if not audio_segments:
            return {"error": "CosyVoice did not generate audio output."}

        full_audio = torchaudio.torch.cat(audio_segments, dim=1)

        # Save to MP3 format for low latency and small response payload
        torchaudio.save(str(out_mp3), full_audio, cosyvoice.sample_rate, format="mp3")

        out_b64 = base64.b64encode(out_mp3.read_bytes()).decode("ascii")

        return {
            "audio_b64": out_b64,
            "filename": "cosyvoice_cloned.mp3"
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
