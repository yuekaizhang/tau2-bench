"""Minimal Chatterbox-Turbo TTS server for tau2 voice eval.

POST /tts {"text": "...", "sample_rate": 16000}
  -> raw PCM_S16LE mono bytes at the requested rate
     (response header X-Sample-Rate echoes the rate)

Uses the built-in voice (conds.pt from ResembleAI/chatterbox-turbo); set
CHATTERBOX_REF_WAV to a ~10s wav to clone a different voice instead.
A single model instance is guarded by a lock: generation is GPU-bound and
tau2's max-concurrency handles parallelism upstream.
"""

import os
import threading

import numpy as np
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import perth

# resemble-perth ships PerthImplicitWatermarker=None in this build; watermarking
# is irrelevant for offline eval, so fall back to the no-op watermarker.
if perth.PerthImplicitWatermarker is None:
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

from chatterbox.tts_turbo import ChatterboxTurboTTS

app = FastAPI()
model = None
model_lock = threading.Lock()
ref_wav = os.environ.get("CHATTERBOX_REF_WAV") or None


class TTSRequest(BaseModel):
    text: str
    sample_rate: int = 16000


@app.get("/health")
def health():
    return {"ok": model is not None}


@app.post("/tts")
def tts(req: TTSRequest):
    try:
        with model_lock:
            wav = model.generate(req.text, audio_prompt_path=ref_wav)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if req.sample_rate != model.sr:
            wav = torchaudio.functional.resample(wav, model.sr, req.sample_rate)
        pcm = (
            (wav.squeeze(0).clamp(-1, 1) * 32767.0)
            .to(torch.int16)
            .cpu()
            .numpy()
            .astype("<i2")
            .tobytes()
        )
        return Response(
            content=pcm,
            media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(req.sample_rate)},
        )
    except Exception as e:  # surface the error to the caller for retry logic
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    if ref_wav:
        model.prepare_conditionals(ref_wav)
    # warmup
    _ = model.generate("Hello, this is a warmup sentence.", audio_prompt_path=ref_wav)
    print(f"chatterbox-turbo ready on {device}, sr={model.sr}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8002")))
