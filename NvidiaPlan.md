# Plan — Run the full voice pipeline on an NVIDIA Jetson Nano

> **STATUS: NOT STARTED.** The Pi deployment ([pla.md](pla.md)) is done and verified.
> This plan ports the *same three-stage pipeline* to a Jetson Nano (128 CUDA cores,
> Maxwell) and decides — per stage — whether the GPU actually helps.

**Goal:** run the whole chain **on one Jetson Nano**, offline:

```
mic (English speech)
   → Whisper STT           (faster-whisper / CTranslate2)
   → EN→HI translation     (our fine-tuned MarianMT int8 CT2 — model_cache_compact_ct2/)
   → Piper TTS             (Hindi speech out the speaker)
```

On the Pi these stages were split across two machines (laptop did STT over SSH, Pi did
translate+speak — see [laptop_mic.py](laptop_mic.py)). **On the Jetson all three run on
the same box.** That is the whole point of moving to the Jetson: it has a GPU, so STT no
longer needs a separate laptop.

**Nothing here re-trains or re-converts the model.** The int8 CT2 package
([model_cache_compact_ct2/](model_cache_compact_ct2/), 79 MB) is the exact artifact that
already runs on the Pi and the laptop. This is a *deployment + placement* plan.

---

## 0. The one fact that decides everything: which JetPack

**Do this first. Everything downstream depends on it.** On the Jetson:

```bash
cat /etc/nv_tegra_release        # L4T version
sudo apt-cache show nvidia-jetpack | grep Version   # JetPack version, if installed
nvcc --version                   # CUDA toolkit version (may be absent)
python3 --version
uname -m                         # aarch64
free -m                          # Nano is 4 GB (or 2 GB on the 2GB model)
```

The **Jetson Nano (Maxwell, 128 CUDA cores)** is end-of-life for L4T. Its last supported
software stack is:

| Component | Jetson Nano ceiling | Why it matters here |
|---|---|---|
| **JetPack** | **4.6.x** (last for Nano) | pins everything below |
| **L4T** | 32.7.x | |
| **CUDA** | **10.2** | too old for modern CT2 CUDA wheels |
| **cuDNN** | 8.2 | |
| **Ubuntu** | **18.04** (bionic) | ships **Python 3.6** |
| **Python** | **3.6** (system default) | see the trap below |

> ⚠️ **This is the central constraint of the entire plan.** The Jetson Nano is *not* a
> modern CUDA box. It is a 2019 Maxwell GPU frozen at CUDA 10.2 / Python 3.6. Our whole
> stack (CTranslate2 4.8.1, transformers 4.57.6, sentencepiece 0.2.1) was verified on
> **Python 3.11/3.13** and assumes recent aarch64 wheels that **do not exist for CUDA 10.2
> / Python 3.6**. Resolving this mismatch — not wiring the pipeline — is 80% of the work.

**Decision gate — pick a lane before writing any code:**

| If the board has… | Lane | Section |
|---|---|---|
| Stock JetPack 4.6 (Python 3.6, CUDA 10.2) | **Lane A — Python 3.8/3.9 venv, GPU only where wheels exist** | §3 |
| A newer Python you can install (3.8/3.9 via deadsnakes or conda) | **Lane A** (recommended) | §3 |
| You accept **CPU-only** for everything (simplest, proven) | **Lane B — CPU, identical to the Pi** | §4 |
| A community/unofficial newer-L4T image | **Lane C — modern stack (unsupported, risky)** | §8 |

**Recommendation: start with Lane B (CPU-only), then selectively move Whisper to GPU
(Lane A).** Rationale in §2.

---

## 1. What runs where — realistic GPU placement on 128 Maxwell cores

Not every stage benefits from this GPU. Be honest about it up front — a 128-core Maxwell
GPU at CUDA 10.2 is modest, and two of our three stages barely touch it.

| Stage | Runtime | GPU on Jetson? | Verdict |
|---|---|---|---|
| **Whisper STT** | faster-whisper (CTranslate2) | **Yes — this is the win.** `WhisperModel(device="cuda", compute_type="int8_float16")`. STT is the heaviest stage and the one [laptop_mic.py](laptop_mic.py) already runs on CUDA (`float16`). | **GPU** |
| **Translation** | MarianMT int8 CT2 (77 M params) | *Marginal.* Per-sentence CPU inference is already ~37 ms on a laptop / a few hundred ms on ARM. CT2 CUDA needs int8 support on Maxwell/CUDA 10.2 — flaky. The model is tiny. | **CPU** (keep as-is) |
| **Piper TTS** | Piper (ONNX Runtime subprocess) | *Not worth it.* Piper's `hi_IN-pratham-medium` runs fine on CPU; ONNX Runtime GPU on CUDA 10.2 is a separate build headache for little gain on short Hindi utterances. | **CPU** (subprocess, as on the Pi) |

**So the realistic Jetson win is: Whisper on the GPU, translation + TTS on the CPU, all on
one box — no more laptop-over-SSH hop.** That alone is the reason to use the Jetson.

> Do **not** assume CT2 will run our int8 Marian model on this GPU. CTranslate2's CUDA int8
> path targets newer compute capabilities; Nano is `sm_53` (Maxwell). Measure it (§6) — if
> it doesn't load or isn't faster than CPU, leave translation on CPU. It's 37 ms either way.

---

## 2. Why CPU-first (Lane B), then GPU for Whisper

The translation + TTS stages are *proven identical* to the Pi build and need **zero**
Jetson-specific work — the CT2 model and Piper are architecture-portable aarch64 code. The
only stage that wants the GPU is Whisper, and the GPU path is exactly where the CUDA 10.2 /
Python 3.6 pain lives.

So sequence the risk:

1. **Lane B first** — get the *entire* pipeline working CPU-only on the Jetson. This is
   essentially the Pi build (§4) and should "just work." Now you have a working end-to-end
   demo and a fallback.
2. **Then Lane A** — move *only Whisper* to CUDA and measure the speedup. If the GPU wheel
   fight isn't worth it, you still shipped in step 1.

Never let the GPU port block having a working pipeline.

---

## 3. Lane A — GPU for Whisper (the recommended target)

### 3.1 The Python problem

Stock JetPack 4.6 is **Python 3.6**, which is past end-of-life and where modern
`faster-whisper` / `ctranslate2` / `transformers` wheels **do not exist**. Two ways out:

**Option A1 — a newer Python via conda (archiconda / miniforge aarch64), recommended:**

```bash
# miniforge for aarch64 (works on Nano)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
conda create -n voice python=3.9 -y
conda activate voice
```

**Option A2 — deadsnakes-style system Python 3.8/3.9 build.** More fragile on bionic; A1
is cleaner.

> **Why not just use system Python 3.6?** Our verified stack (CT2 4.8.1, transformers
> 4.57.6) has no 3.6 wheels, and 3.6 can't even import [marian_ct2.py](it2edge/serve/marian_ct2.py)
> cleanly on modern transformers. Get to **3.8+** and stay there. The Pi build targets
> 3.11+; 3.9 is the practical floor on the Nano.

### 3.2 The CUDA problem for CTranslate2

This is the crux. faster-whisper's GPU path is CTranslate2's CUDA runtime, and CTranslate2's
**PyPI wheels are built for recent CUDA (11/12), not CUDA 10.2.** On the Nano you have two
realistic options:

**Option G1 — GPU Whisper via NVIDIA's own container (recommended for GPU).**
Use an `l4t`-based image where NVIDIA has already solved the CUDA 10.2 + Python versions:

```bash
# on the Jetson, with nvidia-container-runtime (ships with JetPack)
sudo docker run --runtime nvidia -it --rm \
  --device /dev/snd \
  -v $HOME/entm:/workspace/entm \
  nvcr.io/nvidia/l4t-ml:r32.7.1-py3   # or an l4t-pytorch tag matching your L4T
```

Inside, install a CTranslate2 build compatible with CUDA 10.2, or fall back to CPU Whisper
if none resolves (see G3). Container is the least-painful way to get matching CUDA libs.

**Option G2 — build CTranslate2 from source against CUDA 10.2.** Possible but slow and
finicky on Maxwell (`sm_53`). Only if G1 fails and GPU is mandatory. Expect a long build and
manual cuDNN wiring.

**Option G3 — GPU is not achievable → fall back to CPU Whisper.**
[laptop_mic.py](laptop_mic.py) **already implements this fallback** (`_get_whisper()` catches
a broken CUDA and drops to `device="cpu", compute_type="int8"`). On the Nano's 4 A57 cores,
a `tiny`/`base` Whisper model on CPU is usable. This is the honest floor for STT on the Nano.

### 3.3 Whisper model size on a 4 GB Nano

The Nano has **4 GB shared** between CPU and GPU. Budget accordingly — Whisper `small`
(what [laptop_mic.py](laptop_mic.py) defaults to) may be too big alongside the translator
+ Piper + the OS on 4 GB. Start small:

| Whisper model | ~VRAM/RAM | Recommendation on Nano |
|---|---|---|
| `tiny` | ~0.5 GB | safe, fast, lower accuracy |
| `base` | ~0.7 GB | **good default for the Nano** |
| `small` | ~1.5 GB | only if GPU + memory headroom confirmed |
| `medium`+ | ≥3 GB | **do not** — will OOM with the rest of the pipeline |

Set it explicitly: `export WHISPER_MODEL=base`.

---

## 4. Lane B — CPU-only (the proven fallback, do this first)

This is the Pi build ([pla.md](pla.md) §3–4) running on the Jetson's ARM CPU. It needs no
CUDA and no special Python beyond 3.8+.

### 4.1 Get the artifacts onto the Jetson

The CT2 package is self-contained (weights + both SentencePiece models + tokenizer config
in one directory) — nothing downloads from HuggingFace at runtime.

```bash
# from the laptop, project root
scp -r model_cache_compact_ct2 <user>@<jetson-ip>:~/entm/
scp -r it2edge deploy requirements pyproject.toml <user>@<jetson-ip>:~/entm/

# verify the model transferred intact (a truncated model.bin fails confusingly later)
sha256sum model_cache_compact_ct2/model.bin                    # laptop
ssh <user>@<jetson-ip> sha256sum ~/entm/model_cache_compact_ct2/model.bin   # must match
```

### 4.2 Environment (CPU)

```bash
# on the Jetson (inside conda env or a 3.8+ venv)
cd ~/entm
python -m venv venv && source venv/bin/activate    # or: conda activate voice
pip install --upgrade pip
pip install -r requirements/pi.txt                  # the Pi's floored requirements work as-is
```

> [requirements/pi.txt](requirements/pi.txt) exact-pins only the three model-critical
> packages and floors the rest precisely so it resolves on varied aarch64 Python versions —
> that flexibility helps here too. **Watch the install log:** if anything says
> "Building wheel for ctranslate2" you're on the wrong Python (likely 3.6) — stop and fix
> §3.1, don't wait out an hours-long source build.

### 4.3 Smoke test — translation only

```bash
python -m it2edge.serve.translate_ct2 "Hello, how are you?"
# expected, byte-identical to the Pi and laptop:
#   EN: Hello, how are you?
#   -> हेलो, आप कैसे हैं?
```

If the Hindi differs, the artifact moved wrong — re-check the §4.1 checksum before anything
else. Identical model + tokenizer + greedy decode is deterministic across architectures.

### 4.4 Install Piper (Hindi TTS)

[speak.py](it2edge/serve/speak.py) shells out to the Piper binary — same as on the Pi.

```bash
# Piper aarch64 binary + Hindi voice (paths match speak.py defaults)
mkdir -p ~/piper ~/piper_voices
# download piper_linux_aarch64.tar.gz -> ~/piper/piper
# download hi_IN-pratham-medium.onnx + .onnx.json -> ~/piper_voices/
sudo apt install -y alsa-utils                      # aplay (ALSA_DEVICE default hw:0,0)
python -m it2edge.serve.speak "Hello, how are you?" # translates + speaks Hindi
```

See [docs/STT-TTS.md](docs/STT-TTS.md) for the exact Piper download steps used on the Pi.

---

## 5. Wiring the full on-box pipeline (mic → STT → translate → speak)

On the Pi, STT lived on the laptop and reached the Pi over SSH ([laptop_mic.py](laptop_mic.py)).
**On the Jetson, collapse that into one local process** — no SSH, no second machine.

There is no single-box mic→STT→translate→speak script in the repo yet — write one
(`jetson_mic.py`), adapting [laptop_mic.py](laptop_mic.py) by replacing the `send_to_pi()`
SSH hop with a **direct local call** into the already-loaded translator + Piper:

```python
# jetson_mic.py (new) — all three stages in one process, all local
from it2edge.serve.marian_ct2 import load_marian, translate_marian
from it2edge.serve.speak import speak

# 1. STT: reuse laptop_mic.py's _get_whisper()/transcribe() verbatim.
#    On the Nano: WHISPER_DEVICE=cuda (falls back to cpu), WHISPER_MODEL=base.
# 2. Translate: load the CT2 model ONCE at startup, keep warm (never per-utterance).
tok, tr = load_marian("model_cache_compact_ct2")     # warm, once
# 3. loop: english = transcribe(); hindi = translate_marian([english], tok, tr)[0]; speak(hindi)
```

**Two hard rules, both inherited from the Pi build:**

1. **Load the translator once and keep it warm.** Cold start (Python + CT2 + transformers
   import) is seconds; per-sentence inference is tens of ms. Never pay import cost per
   utterance. This is the single biggest usability factor — same lesson as the Pi.
2. **Piper stays a short-lived subprocess per utterance** ([speak.py](it2edge/serve/speak.py)
   already does this) so its RAM is released between sentences — important on 4 GB shared.

**Alternative — keep the HTTP service.** [it2edge/serve/app.py](it2edge/serve/app.py) already
serves `/translate` and `/speak` with the model warm and TTS in a background thread. You can
run it on the Jetson and have a thin local mic client POST to `localhost:8080`. Use this if
you want the service model; use `jetson_mic.py` if you want one tight local loop for the demo.

---

## 6. Benchmark — measure, don't assume (per stage)

The Nano's per-stage numbers are unknown. Measure each; the GPU claim must be earned.

```bash
# translation latency (same benchmark as the Pi build)
python -m it2edge.evaluate.benchmark_models latency \
  --ct2_dir model_cache_compact_ct2 --data_dir en-indic-exp \
  --inter 1 --intra $(nproc) --num 40 --warmup 5 \
  --out jetson_latency_$(uname -m).json
```

Fill in — and be explicit about which stage used the GPU:

| Stage | Metric | Jetson CPU | Jetson GPU (Whisper only) |
|---|---|---|---|
| Whisper STT (`base`) | wall-clock per utterance | | |
| Translation (int8 CT2) | median / p95 per sentence | | |
| Piper TTS | synth time per sentence | | |
| **End-to-end** | speak-to-speak latency | | |
| Peak RAM (of 4 GB) | | | |

**GPU decision rule:** keep Whisper on CUDA only if it's meaningfully faster than CPU Whisper
*and* leaves enough of the 4 GB for the translator + Piper. Otherwise ship CPU Whisper (G3).

### 6.1 Thermals / power — the Jetson equivalent of the Pi's throttling check

```bash
sudo nvpmodel -q                 # current power mode
sudo nvpmodel -m 0               # MAXN (max clocks) — use 10W mode (-m 0) for the demo
sudo jetson_clocks               # pin clocks to max (skip DVFS ramp)
tegrastats                       # live CPU/GPU/RAM/temp — the Nano's `vcgencmd`
```

Run `tegrastats` during a sustained benchmark. If the Nano is thermal/power throttling, the
numbers are pessimistic — add cooling or set `nvpmodel -m 0` + `jetson_clocks` and re-run.
(This is the direct analogue of the Pi's `vcgencmd get_throttled` gate in [pla.md](pla.md) §6.2.)

---

## 7. Auto-start on boot

Same shape as the Pi's systemd unit ([pla.md](pla.md) §5.4 / §11.3), adapted for the Jetson.
If running the HTTP service, install a user unit; if running `jetson_mic.py` for the demo,
run it interactively (it needs the mic and speaker anyway).

```bash
mkdir -p ~/.config/systemd/user
# adapt deploy/it2-baremetal.service: point ExecStart at the conda/venv python,
# add the CUDA env if GPU Whisper (LD_LIBRARY_PATH for cublas/cudnn — see
# laptop_mic.py:_add_pip_cuda_libs), set WHISPER_MODEL/WHISPER_DEVICE.
loginctl enable-linger $USER
systemctl --user daemon-reload && systemctl --user enable --now it2
```

**Reboot the Jetson and confirm it comes back on its own** — an untested auto-start is not
an auto-start (same rule as the Pi).

---

## 8. Lane C — modern stack via unofficial newer-L4T (only if you must have modern CUDA)

Some community images (e.g. newer Ubuntu + patched L4T) put a Nano on CUDA 11+. This unlocks
modern CT2 CUDA wheels and would let translation run on the GPU too. **Unsupported by
NVIDIA, fragile, and not worth it for this pipeline** — the translation stage is 37 ms on CPU
regardless. Documented only for completeness. Do not start here.

---

## 9. Risks & gotchas (Jetson-specific)

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Python 3.6 on stock JetPack** — no wheels for our stack | Install 3.8/3.9 via miniforge (§3.1). If pip tries to *build* ctranslate2, you're still on 3.6 — stop. |
| 2 | **CUDA 10.2 too old for CT2 PyPI wheels** | GPU Whisper via l4t container (G1), or CPU Whisper fallback (G3, already coded). |
| 3 | **4 GB shared RAM** across CPU+GPU | Whisper `base` not `small`/`medium`; Piper stays subprocess; translator warm-loaded once, single worker. |
| 4 | **CT2 int8 on Maxwell `sm_53`** may not load on GPU | Don't assume — measure (§6). Leave translation on CPU; it's 37 ms either way. |
| 5 | **Thermal / power throttling** | `nvpmodel -m 0` + `jetson_clocks`; watch `tegrastats` during benchmarks (§6.1). |
| 6 | **Truncated `scp` of model.bin** | Checksum both ends (§4.1). |
| 7 | **Mic + speaker on one box** — ALSA device conflicts | `speak.py` serializes playback with a lock; set `ALSA_DEVICE`/`arecord -l`/`aplay -l` to pick the right cards. |
| 8 | **Cold start read as a hang** | Model + Whisper load is seconds. Warm-load once (§5 rule 1); never per-utterance. |
| 9 | **Whisper model download needs network once** | Pre-download on the Jetson while online; it caches to `~/.cache`. Offline after that. |
| 10 | **Wrong power mode silently halves clocks** | `nvpmodel -q` before benchmarking; the Nano ships in 5W mode by default. |

---

## 10. Sequencing

| Milestone | Work | Gate |
|---|---|---|
| **M0 — Recon** | §0 JetPack/CUDA/Python check, RAM, lane choice | JetPack version known, lane chosen |
| **M1 — Python** | §3.1 get to Python 3.8/3.9 (miniforge) | `python --version` ≥ 3.8, pip resolves wheels (no source build) |
| **M2 — CPU pipeline** (Lane B) | §4 transfer + deps + translate smoke + Piper | Hindi output byte-identical to the Pi; Piper speaks |
| **M3 — Full loop, CPU** | §5 `jetson_mic.py`, mic→STT(CPU)→translate→speak on one box | End-to-end voice demo works, all local |
| **M4 — GPU Whisper** (Lane A) | §3.2 CUDA Whisper (container or fallback), §6 benchmark | STT on GPU measurably faster, RAM fits, or documented CPU fallback |
| **M5 — Benchmark + thermals** | §6 per-stage numbers, `tegrastats`, power mode | `jetson_latency_*.json` written; not throttling |
| **M6 — Auto-start** | §7 systemd unit + reboot test | Survives a reboot on its own |

**M2 and M3 are load-bearing** — they give a working end-to-end demo with zero GPU risk.
M4 is the Jetson's reason to exist (GPU STT) but must never block M3.

---

## 11. Definition of done

- [ ] JetPack / CUDA / Python version recorded; lane chosen (§0)
- [ ] Python ≥ 3.8 on the Jetson; `pip install -r requirements/pi.txt` resolves with **no** source build
- [ ] `model_cache_compact_ct2/model.bin` checksum matches the laptop
- [ ] `python -m it2edge.serve.translate_ct2 "Hello, how are you?"` prints the same Hindi as the Pi
- [ ] Piper speaks Hindi on the Jetson's speaker (`speak.py`)
- [ ] `jetson_mic.py` runs the full **mic → STT → translate → speak** loop **on one box**, offline
- [ ] Whisper GPU decision made and justified by §6 numbers (GPU if faster + fits 4 GB, else CPU)
- [ ] `jetson_latency_<arch>.json` written; `tegrastats` shows no throttling during the run
- [ ] Full pipeline auto-starts after a physical reboot
