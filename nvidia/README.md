# nvidia/ — real-time EN→HI voice pipeline on one Jetson Nano

Full plan: [../jetsonNvidia.md](../jetsonNvidia.md). Runtime/placement companion:
[NvidiaRun.md](NvidiaRun.md). Migration/bring-up: [NvidiaPlan.md](NvidiaPlan.md).

**Pipeline:** `mic → VAD → Whisper STT (GPU) → MarianMT int8 ONNX (GPU) → Piper TTS (CPU) → speaker`,
all on one Jetson Nano, offline. **Always listening, socket-based, sentence-level pipelined** —
it idles when the room is silent and translates in parallel while you keep talking.

```
protocol.py    Socket wire format: length-prefixed audio up, JSON events down.
vad.py         Voice-activity gate (webrtcvad default, Silero optional) + utterance state machine.
engine.py      Warm Whisper + MarianMT + Piper, loaded once. Sentence-level STT. Timed stages.
server.py      UNIX-socket server: bounded queues + STT/MT/TTS worker threads. Entry point.
client.py      Mic → VAD → stream frames on speech, END on pause; prints pushed-back results.
marian_onnx.py MarianMT en→hi over ONNX Runtime (GPU via CUDAExecutionProvider, CPU fallback).
jetson_mic.py  Legacy push-to-talk loop — kept as a --text / --bench debug/baseline tool.
it2-jetson.service   systemd unit: max clocks + launch server on boot (Phase 6).
```

## Architecture (why it's the right shape)

- **One warm server process.** Whisper + MarianMT load once into GPU RAM and stay warm; Piper is
  bound on the CPU. Cold start (CUDA context + both models) is paid once, never per utterance.
- **Sockets, not polling.** A UNIX domain socket streams audio up and *pushes* results down. The
  client never asks "is it done yet?" — the server sends a `partial` event the instant each
  sentence is translated.
- **VAD makes it idle-when-silent.** During silence the client sends nothing; the server's queues
  are empty; all three workers block at 0% CPU. The instant you speak, frames flow and it wakes.
- **Sentence-level pipeline.** STT / MT / TTS are three worker threads joined by bounded queues.
  Utterance N+1 transcribes on the GPU while Piper speaks N on the CPU, and the first Hindi line
  of a 3-line utterance plays before the last line is even transcribed ([§11.3](../jetsonNvidia.md)).
- **Bounded backpressure.** Over-fast speech drops the *oldest* pending item and sends a visible
  `dropped` event — never a silent drop, never unbounded RAM.

## Run it

```bash
# on the Jetson, project root, Python 3.8 env (see ../requirements/jetson.txt)
sudo nvpmodel -m 0 && sudo jetson_clocks          # max clocks — do before benchmarking

# 1. warm-model server (both GPU models + Piper). Leave it running.
WHISPER_MODEL=base WHISPER_DEVICE=cuda CT2_DEVICE=cuda TRANSLATE_BACKEND=onnx \
  python -m nvidia.server

# 2. always-listening mic client (another shell). Speak English; hear Hindi.
python -m nvidia.client

# tune the VAD first without the server:
python -m nvidia.client --vad-test

# quick baseline without VAD/sockets (the old push-to-talk / bench tool):
python -m nvidia.jetson_mic --text "Hello, how are you?"
python -m nvidia.jetson_mic --bench
```

## Env

| Var | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `base` | `tiny`/`base`/`small` — `base` fits 4 GB alongside the translator |
| `WHISPER_DEVICE` | `cuda` | STT device; auto-falls back to cpu |
| `TRANSLATE_BACKEND` | `onnx` | `onnx` (GPU) or `ct2` (CPU int8 fallback) |
| `CT2_DEVICE` | `cuda` | ONNX/CT2 translator device; auto-falls back to cpu |
| `ONNX_MODEL_DIR` | `model_onnx` | the int8 ONNX package (quantize per [§6](../jetsonNvidia.md)) |
| `IT2_SOCKET` | `/run/it2/it2.sock` | UNIX socket path (server + client) |
| `VAD_BACKEND` | `webrtc` | `webrtc` (no torch) or `silero` (needs torch) |
| `VAD_AGGRESSIVENESS` | `2` | webrtcvad 0–3 (higher = filters more) |
| `VAD_HANGOVER_MS` | `500` | trailing silence that ends an utterance |
| `VAD_MAX_UTTERANCE_MS` | `20000` | safety cap (~3+ lines) before a forced utterance end |
| `IT2_UTTERANCE_Q` / `IT2_MT_Q` / `IT2_TTS_Q` | `3`/`8`/`8` | stage queue depths (backpressure) |
| Piper: `PIPER_BIN`, `PIPER_VOICE`, `AUDIO_OUT`, `ALSA_DEVICE` | see [speak.py](../it2edge/serve/speak.py) | TTS binary/voice/audio out |
