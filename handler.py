import os
import sys
import base64
import tempfile
import torchaudio
import runpod
from pathlib import Path

# Base image root directory
BASE_DIR = "/opt/CosyVoice/CosyVoice"
sys.path.append(BASE_DIR)
sys.path.append(f"{BASE_DIR}/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

MODEL_PATH = f"{BASE_DIR}/pretrained_models/CosyVoice2-0.5B"

print("Loading CosyVoice model...")
try:
    cosyvoice = CosyVoice2(MODEL_PATH, load_jit=False, load_trt=False, fp16=True)
except Exception:
    cosyvoice = CosyVoice(MODEL_PATH, load_jit=False, load_trt=False, fp16=True)
print("CosyVoice successfully loaded into GPU memory!")

def handler(job):
    job_input = job.get("input", {})
    source_b64 = job_input.get("source_b64")
    reference_b64 = job_input.get("reference_b64")
    speed = float(job_input.get("speed", 1.0))

    if not source_b64 or not reference_b64:
        return {"error": "Missing 'source_b64' or 'reference_b64'"}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src_raw = tmp / "src.raw"
        src_wav = tmp / "src.wav"
        ref_raw = tmp / "ref.raw"
        ref_wav = tmp / "ref.wav"
        out_mp3 = tmp / "output.mp3"

        src_raw.write_bytes(base64.b64decode(source_b64))
        ref_raw.write_bytes(base64.b64decode(reference_b64))

        # Standardize both to 16kHz mono WAV
        os.system(f"ffmpeg -y -i {src_raw} -ac 1 -ar 16000 {src_wav} >/dev/null 2>&1")
        os.system(f"ffmpeg -y -i {ref_raw} -ac 1 -ar 16000 {ref_wav} >/dev/null 2>&1")

        source_speech = load_wav(str(src_wav), 16000)
        prompt_speech = load_wav(str(ref_wav), 16000)

        audio_segments = []
        for chunk in cosyvoice.inference_vc(source_speech, prompt_speech, stream=False, speed=speed):
            audio_segments.append(chunk["tts_speech"])

        if not audio_segments:
            return {"error": "Voice conversion failed to generate audio."}

        full_audio = torchaudio.torch.cat(audio_segments, dim=1)
        torchaudio.save(str(out_mp3), full_audio, cosyvoice.sample_rate, format="mp3")

        out_b64 = base64.b64encode(out_mp3.read_bytes()).decode("ascii")
        return {"audio_b64": out_b64, "filename": "converted_voice.mp3"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
