# How the voice pipeline runs on an NVIDIA Jetson Nano Dev Kit

The full chain — **mic → Whisper STT → EN→HI translation → Piper TTS → speaker** — running
**entirely on one Jetson Nano Dev Kit**, offline, with the two neural models on the **GPU**
and TTS on the CPU.

This is the runtime companion to [NvidiaPlan.md](NvidiaPlan.md) (the migration plan) and the
Jetson version of [docs/STT-TTS.md](docs/STT-TTS.md) (the two-machine Pi setup). The focus
here is **how it executes** — process model, device placement, what lives in memory, and the
latency budget.

---

## Device placement — the whole design in one line

**Whisper STT and translation run on the GPU (128 CUDA cores). Piper TTS runs on the CPU.**

```
┌──────────────────────── JETSON NANO DEV KIT (one box) ────────────────────────┐
│                                                                                │
│   🎤 USB mic                                                                    │
│      │  sounddevice / arecord  (16 kHz mono)                                    │
│      ▼                                                                          │
│  ┌── GPU (CUDA, 128 Maxwell cores) ─────────────────────────────────────────┐  │
│  │  Whisper STT   ── faster-whisper / CTranslate2 (cuda) ──▶ English text     │  │
│  │      │                                                                     │  │
│  │      ▼                                                                     │  │
│  │  Translation   ── MarianMT int8 CT2, your model (cuda) ──▶ Hindi text      │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│      │                                                                          │
│      ▼                                                                          │
│  ┌── CPU (4× Cortex-A57) ────────────────────────────────────────────────────┐ │
│  │  Piper TTS   ── ONNX subprocess, one per utterance ──▶ Hindi audio (PCM)    │ │
│  └────────────────────────────────────────────────────────────────────────────┘│
│      │                                                                          │
│      ▼                                                                          │
│   🔊 speaker  (USB / 3.5 mm jack)                                               │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

Both CTranslate2 models (Whisper *and* the translator) share the same CUDA runtime, so once
the GPU is up, both run on it. Piper is deliberately kept off the GPU — see below.

---

## Why this split

| Stage | Runtime | Device | Rationale |
|---|---|---|---|
| **Whisper STT** | faster-whisper (CTranslate2) | **GPU** — `device="cuda"`, `compute_type="int8_float16"` | Heaviest stage; the reason to use a Jetson. Already coded for CUDA in [laptop_mic.py](laptop_mic.py) `_get_whisper()`. |
| **Translation** | MarianMT int8 CT2 (your model) | **GPU** — `device="cuda"`, `compute_type="int8"` | Same CTranslate2 CUDA runtime as Whisper — no extra dependency once CUDA is working. Keeps the two neural stages together on the accelerator. |
| **Piper TTS** | Piper (ONNX Runtime), subprocess | **CPU** | Purpose-built for CPU/edge; short Hindi utterances are cheap. ONNX-GPU on CUDA 10.2 is a separate build with no real payoff, so Piper stays on the CPU and off the GPU that the two CT2 models are using. |

**Result:** the GPU carries both neural models; the CPU is left free to drive Piper and
audio I/O. The two workloads don't contend.

> ⚠️ **Honest caveat on GPU translation.** Running CTranslate2 on the GPU needs a
> **CUDA-enabled CT2 build** (the plain PyPI wheel is CPU-only, and it targets modern CUDA,
> not the Nano's CUDA 10.2). On the Nano you get this from an NVIDIA `l4t`-based container or
> a source build for Maxwell (`sm_53`) — see [NvidiaPlan.md](NvidiaPlan.md) §3.2. If the
> CUDA CT2 build won't load the int8 model on `sm_53`, translation transparently falls back
> to CPU (~tens of ms — it's a 77 M-param model, so this is not a functional loss). **The
> design target is GPU; CPU is the safety net, not the plan.**

---

## Enabling GPU for both models

Whisper already takes the device from `WHISPER_DEVICE` (default `cuda`) in
[laptop_mic.py](laptop_mic.py). The translator currently hardcodes CPU — flip it to CUDA:

```python
# it2edge/serve/marian_ct2.py — load_marian()
translator = ctranslate2.Translator(
    model_dir,
    device="cuda",              # was "cpu"
    compute_type="int8",        # int8 weights run on GPU via the CUDA CT2 build
    inter_threads=inter_threads,
    intra_threads=intra_threads,
)
```

Best done behind an env switch so the same code still runs CPU-only on the laptop/Pi:

```python
device = os.environ.get("CT2_DEVICE", "cuda")   # "cuda" on the Jetson, "cpu" elsewhere
translator = ctranslate2.Translator(model_dir, device=device, compute_type="int8", ...)
```

Both CT2 models need the CUDA libraries on the loader path. `laptop_mic.py`'s
`_add_pip_cuda_libs()` already wires `libcublas`/`libcudnn` into `LD_LIBRARY_PATH`; on the
Jetson those libs come from JetPack (`/usr/local/cuda`), so ensure that's on
`LD_LIBRARY_PATH` before launch.

---

## The runtime process model

### One long-lived process, both GPU models warm

```
jetson_mic.py  (long-lived, started once)
│
├─ Whisper model            ── loaded ONCE at startup → resident in GPU memory
├─ MarianMT CT2 translator  ── loaded ONCE at startup → resident in GPU memory
│
└─ per utterance:
     record mic
       → whisper.transcribe()      (GPU)
       → translate_marian()        (GPU)
       → speak()                   (CPU) ── spawns Piper as a SHORT-LIVED subprocess
                                             that exits & frees its RAM after each sentence
```

**Two load-bearing rules:**

1. **Load both GPU models once; keep them warm.** Cold start (Python + CUDA context +
   CTranslate2 + transformers + two model loads) is **seconds to tens of seconds** on the
   Nano — CUDA context creation alone is not free. Per-utterance work afterward is
   sub-second. Never re-launch a CLI per sentence, or you pay the whole cold start every
   time. This is the single biggest difference between a snappy demo and an unusable one.

2. **Piper stays a short-lived CPU subprocess, one per utterance.**
   [speak.py](it2edge/serve/speak.py) already does this on purpose — Piper's memory is
   released between sentences rather than held by the long-running process, which matters on
   the Nano's shared memory.

### What's resident while it runs

The Nano Dev Kit has **4 GB shared between CPU and GPU** — the tightest runtime constraint.
Both neural models now live in that shared pool as GPU allocations.

| Resident | Approx | Notes |
|---|---|---|
| L4T / Ubuntu base | ~0.6–1.0 GB | run headless, no desktop |
| CUDA context + CT2 runtime | ~0.3–0.6 GB | created once when the first GPU model loads |
| Whisper `base` (GPU) | ~0.7 GB | `tiny` ~0.5 GB, `small` ~1.5 GB — **use `base`** |
| MarianMT int8 CT2 (GPU) | ~0.1–0.3 GB | the 79 MB `model.bin` + GPU buffers |
| Piper (only while speaking) | ~0.15–0.25 GB | transient CPU RAM, per utterance |
| **Working total** | **~2.0–2.9 GB** | fits 4 GB **if** Whisper is `base`, not `small`/`medium` |

> With **both** models on the GPU, memory is tighter than a CPU-Whisper setup. Whisper
> `base` is the safe default; `small` only after you've confirmed headroom in `tegrastats`.
> `medium`+ will OOM.

---

## Step-by-step: one mic press

1. **Record** — `sounddevice`/`arecord` captures ~5 s of 16 kHz mono from the USB mic into a
   temp WAV (same as [laptop_mic.py](laptop_mic.py) `transcribe()`).
2. **STT (GPU)** — the warm Whisper model transcribes to English on CUDA. If CUDA didn't
   initialize at startup you'll have seen `[info] Whisper on CPU` — that's the fallback,
   several× slower.
3. **Translate (GPU)** — the warm MarianMT CT2 translator turns English into Hindi via
   [translate_marian()](it2edge/serve/marian_ct2.py), greedy decode (`beam_size=1`),
   deterministic, sub-second on the GPU.
4. **Speak (CPU)** — [speak()](it2edge/serve/speak.py) spawns Piper as a CPU subprocess,
   pipes the Hindi in, and streams raw PCM to `aplay` — **playback starts before synthesis
   finishes**, so perceived latency is lower than the synth time.
5. Piper exits and frees its RAM; both GPU models stay warm; the loop waits for the next
   press.

---

## Latency budget (measure yours — ballpark)

Per utterance, models already warm. Cold start (once, at launch) is seconds and separate —
which is exactly why the process is long-lived.

| Stage | Device | Ballpark | Note |
|---|---|---|---|
| Record | — | = recording window (~5 s, your choice) | not compute |
| **Whisper STT (`base`)** | **GPU** | **~0.3–1.0 s** | the stage the GPU is for |
| **Translate (int8, greedy)** | **GPU** | **~0.03–0.2 s** | tiny model; negligible on GPU |
| Piper TTS (`medium` voice) | CPU | ~1–3 s, **streams early** | perceived latency lower — audio starts before synth ends |
| **Speak-to-speak total** | | **~1.5–4.5 s** after you stop talking | now bounded by STT + TTS, not translation |

Compared with the Pi build, where translation was the ~20 s bottleneck: on the Jetson both
neural stages are on the GPU and translation is effectively free — the pipeline is
interactive, and TTS is now the largest compute cost.

---

## Keep the clocks up (or the numbers lie)

The Nano ships in a **5 W power mode with DVFS**. With both models on the GPU this matters
more — a throttled GPU hurts the two heaviest stages at once.

```bash
sudo nvpmodel -m 0        # MAXN — max CPU + GPU clocks (10 W mode)
sudo jetson_clocks        # pin clocks to max, skip the DVFS ramp
tegrastats               # live CPU/GPU/RAM/temp — the Nano's `vcgencmd`
```

Watch `tegrastats` during a sustained run — the `GR3D_FREQ` (GPU) and `RAM` fields are the
ones to watch here. If it's throttling, the latency above is pessimistic; add a fan and
re-check.

---

## The runtime prerequisite you can't skip

The Nano is frozen at **JetPack 4.6 / CUDA 10.2 / Ubuntu 18.04 / Python 3.6**, and this
design needs a **CUDA-enabled CTranslate2** (for both models) on that stack — which the plain
PyPI wheel does not provide. Getting there (Python 3.8/3.9, a CUDA CT2 build via an `l4t`
container or source build for `sm_53`, CUDA libs on `LD_LIBRARY_PATH`) is the bulk of the
work and lives in [NvidiaPlan.md](NvidiaPlan.md) §0–§4. This doc assumes that's done and
describes how the pipeline *runs*.

**Bring-up order (each step de-risks the next):**

```bash
# on the Jetson, Python 3.8+ env, CUDA-enabled CT2 installed
cd ~/entm

# 1. translation on GPU — must be byte-identical to the Pi / laptop
CT2_DEVICE=cuda python -m it2edge.serve.translate_ct2 "Hello, how are you?"
#   -> हेलो, आप कैसे हैं?      (if it errors on sm_53, it falls back to CPU — see caveat)

# 2. translate + speak (Piper on CPU — see docs/STT-TTS.md Part 3 for install)
CT2_DEVICE=cuda python -m it2edge.serve.speak "Hello, how are you?"

# 3. full loop: mic → STT(GPU) → translate(GPU) → speak(CPU), all on one box
WHISPER_MODEL=base WHISPER_DEVICE=cuda CT2_DEVICE=cuda python jetson_mic.py
```

Get step 1 working first — it proves the CUDA CT2 build loads your int8 model on the GPU.
Then 2, then the full loop.

---

## Offline guarantee

Every stage is local; nothing leaves the board.

| Stage | Runs on | Internet? |
|---|---|---|
| Mic capture + Whisper STT | Jetson **GPU** | ❌ after one-time Whisper model download |
| Translate EN→HI | Jetson **GPU** (your int8 model) | ❌ self-contained CT2 package |
| Piper TTS | Jetson **CPU** | ❌ voice is a local `.onnx` file |
| Play audio | Jetson speaker | ❌ |

One-time setup needs the internet (Whisper model + Piper voice, `pip install`, the CUDA CT2
build). After that, unplug it — the whole pipeline runs on the board alone.

---

## Runtime checklist

- [ ] Python 3.8+ env with a **CUDA-enabled** CTranslate2 (both models need it)
- [ ] `marian_ct2.py` translator set to `device="cuda"` (via `CT2_DEVICE`)
- [ ] CUDA libs (`/usr/local/cuda/lib64`) on `LD_LIBRARY_PATH` before launch
- [ ] `translate_ct2 "Hello, how are you?"` on GPU prints Hindi byte-identical to the Pi
- [ ] Both models load once at startup and stay warm (not per utterance)
- [ ] Whisper shows `[info] Whisper on CUDA` at startup (not the CPU fallback)
- [ ] Piper installed; `speak.py` plays Hindi from the CPU out the speaker
- [ ] `jetson_mic.py` runs the full mic→STT(GPU)→translate(GPU)→speak(CPU) loop on one box
- [ ] `nvpmodel -m 0` + `jetson_clocks` set; `tegrastats` shows no GPU/thermal throttling
- [ ] Peak RAM under 4 GB with Whisper `base` and both models on GPU (`tegrastats` RAM field)
```
