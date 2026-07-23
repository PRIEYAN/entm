# JetsonSetup.md — from your dev machine to a running Jetson Nano

Step-by-step setup + run guide for the real-time EN→HI voice pipeline. Companion to
[jetsonNvidia.md](jetsonNvidia.md) (the design) — this file is **just the commands**, in order,
from your laptop all the way to hearing Hindi come out of the Nano's speaker.

**Two machines:**
- **DEV BOX** — your laptop (x86, modern Python). Used once to quantize the model.
- **JETSON** — the Nano Dev Kit (JetPack 4.6 / Python 3.8 / CUDA 10.2). Runs everything.

Follow the phases in order. **Each phase de-risks the next — don't skip ahead.**

---

## Phase 0 — Commit the code (DEV BOX)

You're on branch `jetson-realtime`. Lock the work in first:

```bash
cd ~/weeb/entm
git add -A
git commit -m "Jetson real-time streaming voice pipeline: socket server + VAD client + cleanup"
```

---

## Phase 1 — Quantize the model to int8 (DEV BOX, ~10 min)

`model_onnx/` is currently fp32 (859 MB) — too heavy next to GPU-Whisper on the Nano's 4 GB.
Shrink it to int8 **here**, not on the Jetson (export tooling needs modern Python).

```bash
cd ~/weeb/entm
source venv/bin/activate                 # or your dev env
pip install -r requirements/dev.txt      # includes optimum / onnx / onnxruntime

# 1a. export fresh fp32 ONNX graphs from your fine-tuned checkpoint
optimum-cli export onnx \
  --model model_cache_compact_ft \
  --task text2text-generation-with-past \
  model_onnx_fp32

# 1b. dynamic-quantize each graph to int8 into model_onnx/
python - <<'PY'
from onnxruntime.quantization import quantize_dynamic, QuantType
import glob, os, shutil
os.makedirs("model_onnx", exist_ok=True)
for f in glob.glob("model_onnx_fp32/*.onnx"):
    out = os.path.join("model_onnx", os.path.basename(f))
    quantize_dynamic(f, out, weight_type=QuantType.QInt8)
    print("int8:", out)
# tokenizer + config files travel with the model
for f in glob.glob("model_onnx_fp32/*.spm") + glob.glob("model_onnx_fp32/*.json"):
    shutil.copy(f, "model_onnx/")
print("done -> model_onnx/")
PY

# 1c. optional: drop the unused merged decoder to save space
rm -f model_onnx/decoder_model_merged.onnx model_onnx_fp32/decoder_model_merged.onnx
```

> **Stuck on the export?** Skip this entire phase. You can run the translator on the CPU using
> the 80 MB `model_cache_compact_ct2/` model with `TRANSLATE_BACKEND=ct2` — the pipeline works,
> just with translation on the CPU instead of the GPU. Quantize later.

Quick check on the dev box that the int8 model still translates:

```bash
TRANSLATE_BACKEND=onnx CT2_DEVICE=cpu ONNX_MODEL_DIR=model_onnx \
  python -m nvidia.marian_onnx "Good morning, how are you today?"
# -> सुप्रभात, आज आप कैसे हैं?   (device may print cpu here — that's fine; GPU is on the Jetson)
```

---

## Phase 2 — Copy everything to the Jetson (DEV BOX)

`model_onnx/` and `model_cache_compact_ct2/` are git-ignored, so `git clone` alone won't bring
them — copy the whole working tree with `rsync` (skip the heavy junk):

```bash
# find the Jetson's IP first (on the Jetson: `hostname -I`), then from the dev box:
rsync -av --progress \
  --exclude venv --exclude .git --exclude '__pycache__' \
  --exclude model_cache_compact_ft \
  ~/weeb/entm/  <JETSON_USER>@<JETSON_IP>:~/entm/
```

- We exclude `model_cache_compact_ft/` (1.2 GB training checkpoint — not needed at runtime).
- We keep `model_onnx/` (GPU translator) and `model_cache_compact_ct2/` (CPU fallback).

---

## Phase 3 — First-time Jetson setup (JETSON)

SSH in and do the one-time install.

```bash
ssh <JETSON_USER>@<JETSON_IP>
cd ~/entm
```

### 3a. Confirm the board
```bash
cat /etc/nv_tegra_release        # expect L4T 32.7.x  (tells you the JetPack, needed for the wheel below)
python3 --version                # may be 3.6; you need 3.8 (next step)
free -m                          # 4 GB (or 2 GB model)
```

### 3b. Max performance (do this before ANY benchmark — or the numbers lie)
```bash
sudo nvpmodel -m 0               # MAXN — max CPU + GPU clocks
sudo jetson_clocks               # pin clocks, skip the DVFS ramp
```

### 3c. Python 3.8 venv + Python deps
```bash
# if python3.8 is missing:  sudo apt update && sudo apt install -y python3.8 python3.8-venv
sudo apt install -y libsndfile1 portaudio19-dev alsa-utils   # audio libs for soundfile/sounddevice/aplay

python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements/jetson.txt
```

### 3d. ⚠ Swap in NVIDIA's GPU onnxruntime (the one manual, board-specific step)
The PyPI `onnxruntime` in jetson.txt is **CPU-only** on aarch64. To reach the GPU you need
NVIDIA's Jetson `onnxruntime-gpu` wheel that matches your JetPack (from the Jetson Zoo /
NVIDIA developer forums — search "onnxruntime gpu jetson jetpack 4.6").

```bash
pip uninstall -y onnxruntime
pip install <path-or-url-to onnxruntime_gpu-*-cp38-*-linux_aarch64.whl>
```

### 3e. Install Piper TTS (CPU) + a Hindi voice
Piper is an external binary, not a pip package.

```bash
mkdir -p ~/piper && cd ~/piper
# download the aarch64 Piper release for your board from:
#   https://github.com/rhasspy/piper/releases   (file: piper_linux_aarch64.tar.gz)
tar -xzf piper_linux_aarch64.tar.gz             # gives ~/piper/piper
chmod +x ~/piper/piper

# Hindi voice (.onnx + .onnx.json) from https://huggingface.co/rhasspy/piper-voices (hi_IN)
mkdir -p ~/piper_voices && cd ~/piper_voices
# download hi_IN-pratham-medium.onnx  and  hi_IN-pratham-medium.onnx.json  here
cd ~/entm
```

Defaults expected by [it2edge/serve/speak.py](it2edge/serve/speak.py): `~/piper/piper` and
`~/piper_voices/hi_IN-pratham-medium.onnx`. If yours differ, set `PIPER_BIN` / `PIPER_VOICE`.

### 3f. CUDA libs on the loader path
`engine.py` already adds `/usr/local/cuda/lib64`, but make it explicit for your shell:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

---

## Phase 4 — Bring it up ONE stage at a time (JETSON)

Do these in order. **Getting step 1 working is the whole game** — it proves ONNX Runtime's CUDA
provider loads your int8 model on the Nano's old GPU. Everything after it is easy.

```bash
source venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

### Step 1 — translation on the GPU (the risky one)
```bash
CT2_DEVICE=cuda TRANSLATE_BACKEND=onnx ONNX_MODEL_DIR=model_onnx \
  python -m nvidia.marian_onnx "Hello, how are you?"
```
- Prints `[onnx] translator on cuda ...` → 🎉 GPU works. Continue.
- Prints `cpu` or errors on sm_53 → fall back: use `TRANSLATE_BACKEND=ct2` for the rest of the
  steps (translation on CPU, still fast for a 77M model). The pipeline is designed for this.

### Step 2 — translate + speak (proves Piper + the speaker)
```bash
python -m nvidia.jetson_mic --text "Hello, how are you?"
# should print Hindi AND play it out the speaker
```

### Step 3 — tune the VAD (no server, no models — just the mic gate)
```bash
python -m nvidia.client --vad-test
# speak → "🎤 speech start"; stop → "… silence (utterance end)"
# too jumpy? VAD_AGGRESSIVENESS=3 python -m nvidia.client --vad-test
# cuts you off mid-sentence? VAD_HANGOVER_MS=800 python -m nvidia.client --vad-test
```

### Step 4 — THE REAL THING (two terminals)
```bash
# terminal A — warm-model server (leave running)
sudo nvpmodel -m 0 && sudo jetson_clocks
WHISPER_MODEL=base WHISPER_DEVICE=cuda CT2_DEVICE=cuda TRANSLATE_BACKEND=onnx \
  ONNX_MODEL_DIR=model_onnx python -m nvidia.server

# terminal B — always-listening mic client
python -m nvidia.client
```

Speak English continuously; Hindi plays back sentence-by-sentence. Silence = the box idles.

### Watch it (third terminal)
```bash
tegrastats        # RAM must stay < 4 GB; GR3D_FREQ shows GPU use; watch for throttling
```

---

## Phase 5 — Run on boot (optional)

`nvidia/it2-jetson.service` starts the server automatically. Edit the `User=`, `WorkingDirectory=`,
and paths inside it to match your board, then:

```bash
sudo cp nvidia/it2-jetson.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now it2-jetson
journalctl -u it2-jetson -f          # watch startup + latency logs
```
The server runs on boot; run `python -m nvidia.client` whenever you want to talk to it.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not connect to server ... it2.sock` | Start the server first (Phase 4 step 4, terminal A). |
| Translator prints `cpu`, not `cuda` | Wrong onnxruntime wheel (Phase 3d) or sm_53 unsupported — use `TRANSLATE_BACKEND=ct2`. |
| `Piper binary not found` | Install Piper (Phase 3e) or set `PIPER_BIN` / `PIPER_VOICE`. |
| No audio / `aplay` errors | `sudo apt install -y alsa-utils`; set `ALSA_DEVICE=hw:0,0` (list with `aplay -l`). |
| Mic not captured | Check `arecord -l`; sounddevice uses the default input — set it in ALSA. |
| Whisper CUDA fails at startup | It auto-falls back to CPU and prints so; slower but works. Confirm `LD_LIBRARY_PATH`. |
| RAM near 4 GB in tegrastats | Use `WHISPER_MODEL=tiny`; make sure the int8 ONNX (Phase 1) is in use, not fp32. |
| VAD cuts you off / won't trigger | Tune `VAD_AGGRESSIVENESS` (0–3) and `VAD_HANGOVER_MS` (Phase 4 step 3). |

## Key env vars (full list in [nvidia/README.md](nvidia/README.md))

```
WHISPER_MODEL=base           tiny | base | small   (base fits 4 GB alongside the translator)
WHISPER_DEVICE=cuda          STT device (auto-falls back to cpu)
TRANSLATE_BACKEND=onnx       onnx (GPU) | ct2 (CPU fallback)
CT2_DEVICE=cuda              translator device (auto-falls back to cpu)
ONNX_MODEL_DIR=model_onnx    the int8 ONNX package
IT2_SOCKET=/run/it2/it2.sock server + client socket path (must match)
VAD_BACKEND=webrtc           webrtc (no torch) | silero (needs torch)
VAD_AGGRESSIVENESS=2         0–3, higher = filters more
VAD_HANGOVER_MS=500          trailing silence that ends an utterance
```
