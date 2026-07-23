# nvidia/ — full voice pipeline on one Jetson Nano

Implements [NvidiaRun.md](../NvidiaRun.md): **mic → Whisper STT (GPU) → MarianMT int8
translation (GPU) → Piper TTS (CPU)**, all on a single Jetson Nano Dev Kit, offline.

```
engine.py       Warm both GPU models once + bind the CPU Piper speaker. Timed stages.
jetson_mic.py   Push-to-talk loop, --text mode, and --bench latency mode. Entry point.
requirements.txt  faster-whisper + mic stack on top of requirements/pi.txt.
```

## Design (why it's efficient)

- **Two models loaded once, kept warm.** Cold start (CUDA context + both models) is paid
  once at launch; each utterance is then three quick calls.
- **STT + translation share the one CUDA runtime** — both are CTranslate2, so putting the
  translator on the GPU costs no extra dependency.
- **Piper is a short-lived CPU subprocess per utterance** (reused unchanged from
  [it2edge/serve/speak.py](../it2edge/serve/speak.py)) — its RAM frees between sentences, and
  it stays off the GPU the two neural models are using.
- **Automatic CPU fallback** per model if the CUDA CT2 build isn't present — the pipeline
  always runs, and prints which device each stage used.

## Run it

```bash
# on the Jetson, from the project root, in a Python 3.8+ env with a CUDA-enabled CT2 build
sudo nvpmodel -m 0 && sudo jetson_clocks     # max clocks (see NvidiaRun.md)

# full push-to-talk loop
WHISPER_MODEL=base WHISPER_DEVICE=cuda CT2_DEVICE=cuda python -m nvidia.jetson_mic

# type text instead of speaking (tests translate + speak)
python -m nvidia.jetson_mic --text "Hello, how are you?"

# latency benchmark (no mic needed)
python -m nvidia.jetson_mic --bench            # translation latency
python -m nvidia.jetson_mic --bench --speak    # + TTS latency
```

## Env

| Var | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `base` | `tiny`/`base`/`small` — `base` fits 4 GB alongside the translator |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu`; auto-falls back to cpu |
| `CT2_DEVICE` | `cuda` | translator device; auto-falls back to cpu |
| `CT2_MODEL_DIR` | `model_cache_compact_ct2` | the int8 CT2 package |
| Piper: `PIPER_BIN`, `PIPER_VOICE`, `AUDIO_OUT`, `ALSA_DEVICE` | see [speak.py](../it2edge/serve/speak.py) | TTS binary/voice/audio out |
