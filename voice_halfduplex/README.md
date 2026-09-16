# tau-voice: half-duplex speech-in / text-out evaluation for local models

A custom harness that runs the standard tau2-bench tasks in a **half-duplex
voice setting against locally served (vLLM) omni models**: every user turn is
synthesized to audio (Chatterbox-Turbo TTS, telephony-style μ-law 8 kHz
channel effects), the agent model *listens* to the audio and replies in text,
and scoring is the unmodified tau2 reward. The official τ³ Voice track is
full-duplex realtime-websocket + ElevenLabs only, which leaves no path for
local checkpoints — this harness fills that gap.

```
user simulator (any OpenAI-compatible LLM, text)
        │ user text turn
        ▼
Chatterbox TTS server (chatterbox_server.py, GPU, :8002)
        │ PCM16 wav  →  telephony channel effects (μ-law 8k) → resampled 16k
        ▼
agent model  (vLLM OpenAI server, :8000)  — receives audio_url data-URI parts,
        │                                    replies with text / tool calls
        ▼
standard tau2 environment + reward
```

## What is in this branch

- `src/tau2/config.py` (patched): the NL-assertions LLM judge can be
  redirected via env vars — `TAU2_NL_ASSERTIONS_MODEL` (litellm model id) and
  `TAU2_NL_ASSERTIONS_ARGS` (JSON merged into litellm kwargs, e.g.
  `{"base_url": ..., "api_key": ...}`). Without this, the 40 retail tasks
  that carry `nl_assertions` require direct OpenAI access and otherwise die
  as infrastructure errors.
- `src/tau2/voice/synthesis/synthesize.py` (patched): a `chatterbox` TTS
  provider that calls the local server (`CHATTERBOX_TTS_URL`, default
  `http://localhost:8002`).
- `voice_halfduplex/run_voice_halfduplex.py`: the eval runner. Key pieces:
  - `AudioLLMAgent`: replaces the user's text turn with a WAV `audio_url`
    data-URI part before calling the agent. (Never run the default
    `LLMAgent` here — it would read the text transcript and the "voice" eval
    would be fake.)
  - voice user simulator with the official per-task `control`
    speech-complexity presets, telephony channel effects enabled;
  - audio is converted to PCM16 and resampled to 16 kHz before reaching the
    agent (vLLM rejects 8 kHz WAVs with a misleading "install vllm[audio]"
    error).
- `voice_halfduplex/chatterbox_server.py`: minimal FastAPI wrapper around
  Chatterbox-Turbo. `POST /tts {"text": ..., "sample_rate": 16000}` → raw
  PCM_S16LE; `GET /health`.

## Setup

Two Python environments (keep them separate — chatterbox pins conflict with
tau2):

1. **tau2 venv** (Python 3.13 works):
   ```bash
   python -m venv venv
   venv/bin/pip install -e . audioop-lts   # audioop was removed in Py3.13
   ```
   If `pyaudio` fails to build (no portaudio) a one-line stub module is
   enough — any missing voice import silently unregisters the voice user
   ("Voice dependencies not installed, skipping voice user registration").

2. **chatterbox venv** (Python 3.11):
   ```bash
   python3.11 -m venv chatterbox_venv
   chatterbox_venv/bin/pip install --no-deps chatterbox-tts
   # then install its deps manually; chatterbox's published pins are
   # mutually impossible. torch and torchaudio MUST be the same version.
   chatterbox_venv/bin/pip install torch torchaudio transformers fastapi uvicorn
   ```
   If `perth.PerthImplicitWatermarker` resolves to None the server
   substitutes a dummy watermarker automatically.

## Running

```bash
# 1. serve the agent under test (example: a Nemotron nano-omni checkpoint)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $CKPT --served-model-name $TAG \
  --port 8000 --tensor-parallel-size 4 --max-model-len 65536 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 --mamba-ssm-cache-dtype float32 \
  --trust-remote-code --gpu-memory-utilization 0.9

# 2. TTS server on a spare GPU
CUDA_VISIBLE_DEVICES=4 PORT=8002 chatterbox_venv/bin/python \
  voice_halfduplex/chatterbox_server.py

# 3. judge routing (only needed for retail; any OpenAI-compatible gateway)
export TAU2_NL_ASSERTIONS_MODEL="openai/<judge-model>"
export TAU2_NL_ASSERTIONS_ARGS="{\"base_url\": \"$BASE\", \"api_key\": \"$KEY\"}"

# 4. run one domain
venv/bin/python voice_halfduplex/run_voice_halfduplex.py \
  --domain airline --agent-model $TAG \
  --user-llm "openai/<user-sim-model>" \
  --user-llm-args "{\"temperature\":0.0,\"base_url\":\"$BASE\",\"api_key\":\"$KEY\"}" \
  --agent-max-tokens 20480 \
  --workers 8 --save-to voice_${TAG}_airline
```

Results land in `voice_results/<save-to>/summary.json` (per-task rewards +
per-sim JSON). Domain sizes: airline 50 tasks, retail 114, telecom 114
(1 trial each by default; `--num-trials` to change). Airline ≈ 1 h at
`--workers 8` with a 30B agent on 4×H100.

Reporting convention: quote `avg_reward` together with coverage
(`n_scored / n_jobs`). Unscored tasks are almost always the agent producing
an empty turn (runner error `AssistantMessage must have either content or
tool_calls`) — that is a model failure mode, not harness noise; do not
silently drop it from the story.

## Pitfalls (all encountered in practice)

| Symptom | Cause / fix |
|---|---|
| Agent 500 "Please install vllm[audio]" | vLLM rejects 8 kHz WAVs; the runner already converts to PCM16 @16 kHz — don't remove that path |
| Every retail sim with `nl_assertions` dies at scoring | Judge routing env vars not set (step 3 above) |
| "Voice dependencies not installed, skipping voice user registration" | Some voice import failed (often pyaudio); stub it |
| Reasoning models: empty answers scored 0 / excluded | Raise `--agent-max-tokens` to ≥20480; residual empties are a real model failure mode (repetition loops) — report them |
| Suspiciously fast domain run | Every sim died instantly; read the log for the real error before trusting any number |
| Results overwritten | `--save-to` tags must be unique per run |

## Provenance

Branch base: upstream `c339866` — byte-identical to the checkout used for
all published tau-voice numbers in our experiment notes. The two `src/tau2`
patches and the two scripts under `voice_halfduplex/` are the complete
delta; environments are rebuilt per the Setup section.
