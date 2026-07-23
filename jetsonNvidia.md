# jetsonNvidia.md — Real-time EN→HI voice translation on ONE Jetson Nano

> **Status: IMPLEMENTED on branch `jetson-realtime`.** The socket-based,
> always-listening, VAD-gated, queue-driven, sentence-level parallel pipeline is built
> (`nvidia/protocol.py`, `vad.py`, `server.py`, `client.py`; `engine.py` extended for
> sentence-level STT). The old push-to-talk `jetson_mic.py` is kept as a debug/baseline
> tool. Legacy Pi/laptop files were removed and ~5.5 GB of dead model dirs cleaned (§5).
> **Remaining to do on the Jetson itself:** install `requirements/jetson.txt` (+ NVIDIA's
> `onnxruntime-gpu` and `webrtcvad`), quantize the ONNX model to int8 (§6), install Piper,
> and measure real latency (§11). Everything above the hardware is code-complete and its
> socket/queue/worker plumbing is integration-tested locally on the CPU fallback backend.
>
> **Pipeline:** `mic → VAD (webrtcvad default) → Whisper STT (GPU) → MarianMT int8 ONNX (GPU) → Piper TTS (CPU) → speaker`
>
> **Board reality:** Jetson Nano Dev Kit = 128 Maxwell cores, **4 GB shared CPU/GPU RAM**,
> frozen at JetPack 4.6 / CUDA 10.2 / Python 3.8. This constraint drives every decision below.

---

## 0. TL;DR — what you're deciding and what you already have

| Thing | Decision | Why |
|---|---|---|
| **Translation model** | ONNX on GPU, **quantized to int8** | fp32 `model_onnx/` is 859 MB — too heavy next to GPU-Whisper on 4 GB. int8 ≈ ¼ the RAM. |
| **Segmentation** | **Silero VAD** streaming | True idle-when-silent; enqueue an utterance the instant the speaker pauses. |
| **Transport** | **UNIX domain socket** | One box → local socket beats TCP/HTTP: no network stack, lowest overhead, no polling. |
| **Process model** | 1 warm **server** process + thin **client(s)** | Models load once, stay warm in GPU RAM. Client just streams audio frames. |
| **Concurrency** | **bounded queue + stage workers** | STT / MT / TTS run as parallel stages; idle when the queue is empty. |

**You already have (verified on disk):**
- `model_onnx/` — complete fp32 ONNX (encoder + decoder + decoder-with-past + tokenizer), 859 MB → **re-quantize to int8**
- `model_cache_compact_ct2/` — CTranslate2 **int8**, 80 MB (CPU fallback, `TRANSLATE_BACKEND=ct2`)
- `model_cache_compact_ft/` — fine-tuned HF checkpoint (the 4-hour train), 1.2 GB → **source for re-export**, keep
- `nvidia/marian_onnx.py` — a working ONNX-Runtime greedy decoder with CUDA provider + CPU fallback ✅
- `nvidia/engine.py` — warm-load-once engine with per-stage timing ✅
- `it2edge/serve/speak.py` — Piper as short-lived CPU subprocess, streams PCM to `aplay` ✅

**So: can you run this on the Jetson today?** Almost. The *translation half* is ready
(your `engine.py` + `marian_onnx.py` load the model once and translate on GPU with CPU fallback).
What's **missing for the architecture you described** is the streaming front end: VAD, the
socket server/client split, and the stage queue. That's what §2–§4 build. And the fp32 ONNX
should be quantized before it shares 4 GB with Whisper (§6). Nothing needs re-training —
your 4-hour fine-tune is the artifact everything derives from.

---

## 1. Why the current push-to-talk design isn't what you want

`nvidia/jetson_mic.py` today is a **blocking loop**:

```
press Enter → record fixed 5 s → STT → translate → speak → repeat
```

Problems for real-time use:
- **You have to press Enter.** Not "always listening."
- **Fixed 5 s window.** Short sentences waste 4 s; long ones get cut.
- **Fully serial.** While Piper speaks utterance N, the mic is dead — utterance N+1 is lost.
- **No idle notion.** It's either blocked on `input()` or blocked recording. Never "listening cheaply."

The design you asked for fixes all four: **VAD decides when speech starts/stops**, an
**utterance is enqueued the moment you pause**, and **STT / MT / TTS run as pipelined stages**
so the machine keeps translating utterance N while you're already saying N+1 — and drops to
near-zero CPU when the room is silent.

---

## 2. Target architecture — socket server + streaming client + stage pipeline

```
┌──────────────────────────── JETSON NANO (one box, offline) ─────────────────────────────┐
│                                                                                          │
│  CLIENT  (thin, cheap)                          SERVER  (warm models, long-lived)        │
│  ┌────────────────────────────┐                 ┌──────────────────────────────────────┐│
│  │ 🎤 mic  16kHz mono frames   │   UNIX socket   │  audio frames in ─┐                    ││
│  │        │                    │  /run/it2.sock  │                   ▼                    ││
│  │        ▼                    │ ───────────────▶│   ┌─── Silero VAD (CPU, tiny) ─────┐   ││
│  │  Silero VAD gate            │   (stream)      │   │ silence → IDLE (no work queued)│   ││
│  │  speech? → send frames      │                 │   │ speech  → accumulate utterance │   ││
│  │  silence(>0.5s) → END mark  │◀─────────────── │   └───────────────┬────────────────┘   ││
│  │        ▲   Hindi text +     │  events back    │        on END mark │ enqueue(utterance)││
│  │        │   status events    │                 │                    ▼                    ││
│  └────────────────────────────┘                 │        ┌── UTTERANCE QUEue (bounded) ──┐││
│                                                  │        └───────────────┬────────────────┘││
│  (VAD can also live server-side — see §3.4;      │  ┌── GPU ──────────────▼──────────────┐ ││
│   client-side VAD keeps the socket quiet         │  │ STT worker   Whisper base int8_f16 │ ││
│   during silence, which is the cheaper default)  │  │      │  English text                │ ││
│                                                  │  │      ▼                              │ ││
│                                                  │  │ MT worker    MarianMT int8 ONNX     │ ││
│                                                  │  │      │  Hindi text ── event ─────────┼─┼─▶ back to client
│                                                  │  └──────┼──────────────────────────────┘ ││
│                                                  │  ┌── CPU ▼──────────────────────────────┐││
│                                                  │  │ TTS worker  Piper subprocess → aplay │││
│                                                  │  └──────────────────────────────────────┘││
│                                                  └──────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 The two processes

**Server** (`nvidia/server.py`, new) — starts once, loads Whisper + MarianMT into GPU RAM,
binds Piper on CPU, listens on a UNIX socket. It owns the queue and the stage workers. This
is your existing `Engine` (from `engine.py`) wrapped in a socket loop + a queue. **Never
restarts per utterance** — cold start (CUDA context + both models) is seconds; you pay it once.

**Client** (`nvidia/client.py`, new) — opens the mic, runs Silero VAD, and streams 16 kHz mono
frames over the socket only while it hears speech. On a pause it sends an `END` marker. It
receives Hindi text + status events back and prints/logs them. It is tiny and holds no models —
you can even run several (e.g. a headless daemon + a debug CLI) against one server.

> **Why a socket and not HTTP/polling?** You explicitly don't want polling. A UNIX socket is a
> persistent, bidirectional, push stream — the server pushes an event the instant a stage
> finishes; the client never asks "is it done yet?" On one box a UNIX socket also skips the
> whole TCP/IP stack, so it's the lowest-latency option available.

### 2.2 The stage pipeline (this is where the parallelism lives)

Three worker threads inside the server, connected by bounded queues:

```
utterance_q ──▶ [STT worker] ──▶ mt_q ──▶ [MT worker] ──▶ tts_q ──▶ [TTS worker]
   (audio)         GPU             (EN)        GPU          (HI)         CPU
```

- Each stage is a thread pulling from its input queue and pushing to the next.
- **STT and MT share the GPU** — they run on the same CUDA runtime but are *different threads*,
  so while MT translates utterance N, STT can already be transcribing utterance N+1's audio.
  (They serialize on the GPU device, but the CPU-side glue and the queue overlap the waits.)
- **TTS is on the CPU**, fully parallel to the GPU stages — Piper speaks N while the GPU works on N+1.
- **Bounded queues** (e.g. maxlen 4) give backpressure: if someone talks faster than the box can
  keep up, the oldest pending utterance is dropped with a logged warning rather than growing RAM
  unbounded. **Never silently drop without logging** — the client shows "⚠ skipped (overloaded)".

### 2.3 Idle-when-silent (the property you asked for)

- **Client side:** Silero VAD runs on ~30 ms frames on the CPU. During silence it sends
  *nothing* over the socket — the socket is quiet, the server's queue is empty, all three
  workers are blocked on empty `queue.get()` (0% CPU). The GPU models sit warm but idle.
- The **only** cost during silence is the VAD forward pass on tiny frames — microseconds on the
  Nano CPU. That is the "machine is idle" state you want, and it's genuinely near-zero.
- The instant VAD flips to speech, frames flow and the pipeline wakes up. No polling anywhere.

---

## 3. Component decisions (all confirmed with you)

### 3.1 Translation: int8 ONNX on GPU
Your `model_onnx/` is fp32 (859 MB). With Whisper `base` also on the GPU that's tight on 4 GB.
**Re-export/quantize to int8** (§6) — same model, ~¼ the memory, faster on the Nano. Your
`nvidia/marian_onnx.py` already drives the encoder/decoder-with-past ONNX sessions and binds
`CUDAExecutionProvider` with a CPU fallback, so it works unchanged on the int8 export.
Keep the CT2 int8 model as the **CPU fallback** (`TRANSLATE_BACKEND=ct2`) — it's 80 MB and
already proven.

### 3.2 Segmentation: Silero VAD
Lightweight neural VAD. Runs on CPU, ~1 MB, robust to noise. Gates the mic so silence = idle,
and marks utterance boundaries by detecting the trailing pause (default: 500 ms of silence ends
an utterance). Emits an utterance to the queue the moment you stop talking — this is what makes
it feel real-time rather than "wait for a fixed 5 s window." (`webrtcvad` is the ultra-light
fallback if Silero's torch dependency is awkward on JetPack 4.6 — see §7.)

### 3.3 Transport: UNIX domain socket
`/run/it2/it2.sock` (or `$XDG_RUNTIME_DIR`). Length-prefixed binary frames for audio in, JSON
lines for events out. Single-box, lowest overhead, no network exposure. (A `--tcp` flag can
expose it on localhost later if you ever split the client onto another device — but that's not
the plan.)

### 3.4 Where VAD lives — client vs server
**Default: client-side VAD.** It keeps the socket silent during silence (no frames cross it),
which is the cheapest idle state and keeps the server's inbox empty. The server still enforces
utterance boundaries defensively (a max-utterance-length guard) so a stuck client can't feed an
infinite stream. If you later want a "dumb mic" client, move VAD server-side — the queue/stage
design doesn't change, only who runs the 1 MB VAD model.

---

## 4. Build plan — incremental, each step de-risks the next

Work on a **new branch** off `main` (you're on `nvidia` now; branch fresh so the cleanup commit
is clean):

```bash
git switch main && git switch -c jetson-realtime
```

### Phase 0 — Confirm the board (do this first, on the Jetson)
```bash
cat /etc/nv_tegra_release            # L4T 32.7.x expected
python3 --version                    # 3.8 expected
free -m                              # 4 GB
sudo nvpmodel -m 0 && sudo jetson_clocks   # max clocks — do before ANY benchmark
tegrastats                           # live RAM/GPU while testing
```

### Phase 1 — Prove the warm engine on GPU (mostly already done)
Your `engine.py` + `marian_onnx.py` already do this. Verify:
```bash
# translation only, GPU
CT2_DEVICE=cuda TRANSLATE_BACKEND=onnx python -m nvidia.marian_onnx "Hello, how are you?"
# full existing push-to-talk loop still works (baseline)
WHISPER_MODEL=base WHISPER_DEVICE=cuda python -m nvidia.jetson_mic --text "Good morning"
```
Gate: translation prints correct Hindi on `cuda`, and `--bench` shows sub-200 ms MT.

### Phase 2 — Quantize the ONNX model to int8 (§6)
Re-export from `model_cache_compact_ft/` with int8, drop into `model_onnx/`, re-run Phase 1
bench, confirm RAM drop in `tegrastats`.

### Phase 3 — VAD + streaming client (`nvidia/vad.py`, `nvidia/client.py`)
- `vad.py`: wrap Silero VAD; API = feed 30 ms frames, get `speech_start` / `speech_end` events.
- `client.py`: mic → VAD → stream frames to socket on speech, `END` on pause. No models.
- Test standalone first: print "🎤 speech" / "… silence" to confirm gating before wiring the socket.

### Phase 4 — Socket server + stage queue (`nvidia/server.py`)
- Wrap the existing `Engine` (don't rewrite it — reuse `engine.py`).
- UNIX socket accept loop → feed audio into `utterance_q`.
- Three worker threads: STT → MT → TTS, connected by bounded queues; push events back to client.
- Idle correctly: workers block on empty queues (verify 0% CPU in `top` during silence).

### Phase 5 — End-to-end + tune
- Run server, run client, speak continuously. Confirm overlap (N+1 transcribes while N speaks).
- Tune: VAD trailing-silence (400–700 ms), queue depth, Whisper `base` vs `tiny` for latency.
- Latency target (models warm): **speak→hear ≈ 1.5–4 s**, bounded by STT + TTS, not translation.

### Phase 6 — systemd service (optional, replaces the Pi's unit)
A `nvidia/it2-jetson.service` that launches the server on boot with `nvpmodel -m 0` +
`jetson_clocks` in an `ExecStartPre`. Client can be a second unit or run on demand.

---

## 5. Files to REMOVE for a clean Jetson-only branch (full repo walked, top to bottom)

I walked **every** file and folder (excluding `.git/`, `venv/`, `__pycache__/`, and the corpus
line-files). The Jetson runtime imports **only** these — nothing else is on the runtime path:
`nvidia/engine.py`, `nvidia/jetson_mic.py`, `nvidia/marian_onnx.py`, `nvidia/__init__.py`,
`it2edge/serve/speak.py`, `it2edge/serve/marian_ct2.py`, `it2edge/paths.py`, `it2edge/__init__.py`,
`it2edge/serve/__init__.py`.

### 5A. DELETE — code/docs/artifacts not needed on a Jetson-only branch

**Root-level files**
| Path | Size | What it is | Why delete |
|---|---|---|---|
| `laptop_mic.py` | 12K | Laptop→Pi push-to-talk over **SSH** | Pi/laptop two-box design; Jetson is one box |
| `translate.py` | 16K | Laptop mic → Pi FastAPI `/speak` **HTTP** client | Pi/laptop; HTTP polling is the opposite of your socket design |
| `tramslate.py` | 4K | 4-line commented `curl` scratch note | Junk |
| `des.txt` | 4K | Raspberry Pi cost/marketing sheet | Pi narrative, no runtime value |
| `pla.md` | 28K | Raspberry Pi deployment plan | Pi narrative; superseded by this doc |
| `ab_report.json` | 4K | Stale A/B eval output (empty/failed run) | Regenerable artifact |
| `ab_report_smoke.json` | 20K | A/B smoke output with Windows `D:\entm` paths | Stale artifact (this is the file open in your IDE) |
| `README.md` | 16K | Pi/laptop "production setup" (Python ≥3.11, FastAPI, Podman) | Describes the old two-box flow; rewrite for Jetson or drop |

**`deploy/` — DELETE THE ENTIRE FOLDER** (32K, all Raspberry Pi deploy plumbing)
| File | What it is |
|---|---|
| `deploy/Containerfile` | arm64 Podman image for the Pi FastAPI service |
| `deploy/build_arm64.sh` | Cross-build that image on an x86 box |
| `deploy/build_on_pi.sh` | Build+run the serving image on the Pi |
| `deploy/it2-baremetal.service` | systemd unit for the Pi uvicorn service |
| `deploy/it2.container` | Podman Quadlet unit for the Pi |
| `deploy/pi_bluetooth_autoconnect.sh` | Pi Bluetooth-speaker reconnect at boot |
| `deploy/push_to_pi.sh` | scp artifacts laptop→Pi over SSH |

**`it2edge/serve/` — delete the two Pi serving files (keep the other three)**
| File | What it is | Verdict |
|---|---|---|
| `it2edge/serve/app.py` | Pi FastAPI HTTP service (`/translate`, `/speak`) | **DELETE** — request/response HTTP, not your socket model |
| `it2edge/serve/translate_ct2.py` | Pi CLI translator the SSH client drove | **DELETE** |
| `it2edge/serve/marian_ct2.py` | CT2 int8 inference | **KEEP** — runtime imports it (CPU fallback) |
| `it2edge/serve/speak.py` | Piper CPU subprocess TTS | **KEEP** — runtime imports it |
| `it2edge/serve/__init__.py` | package init (docstring names app/translate_ct2) | **KEEP**, but trim the docstring after deleting the two above |

**`docs/` — delete the Pi/laptop + archived-IndicTrans2 docs (keep 2 model-lineage docs)**
| File | What it is | Verdict |
|---|---|---|
| `docs/plan.md` | **ARCHIVED** IndicTrans2 fine-tune/deploy plan (says so at top) | **DELETE** |
| `docs/reduceParams.md` | **ARCHIVED** IndicTrans2 200M→5M param note | **DELETE** |
| `docs/runtimeRasp.md` | Raspberry Pi text→text runtime doc | **DELETE** |
| `docs/STT-TTS.md` | laptop-mic → Pi-translate → Pi-TTS two-box voice pipeline | **DELETE** |
| `docs/compact-model.md` | Documents the MarianMT model lineage (your model) | **KEEP** — model provenance |
| `docs/start.md` | End-to-end walkthrough for the MarianMT model | **KEEP** (or fold into a new Jetson README) |

**Build/packaging cruft**
| Path | Size | What it is | Verdict |
|---|---|---|---|
| `it2edge.egg-info/` | 40K | `pip install -e` build metadata; lists files you're deleting | **DELETE** — regenerates on next install |
| `__pycache__/` (root) | 32K | stray root bytecode cache | **DELETE** — already git-ignored |
| `requirements/pi.txt` | — | Raspberry Pi dep pins | **DELETE** |
| `requirements/laptop.txt` | — | Laptop STT+SSH client deps | **DELETE** |
| `requirements/docker.txt` | — | arm64 Pi container deps | **DELETE** |

> ⚠️ **`nvidia/requirements.txt` currently does `-r ../requirements/pi.txt`** — if you delete
> `pi.txt`, fold its real pins into `requirements/jetson.txt` (or into `nvidia/requirements.txt`)
> first, or the Jetson install breaks. `jetson.txt` is the one requirements file the Jetson uses.

### 5B. KEEP — the one-time build/train path (not runtime, but produced your model)
Not imported at runtime, but this is how your 4-hour fine-tune → int8 model was made and how
Phase 2 (int8 ONNX re-export) works. Keep unless you want a strictly runtime-only branch.
| Path | Role |
|---|---|
| `it2edge/train/finetune_marian.py`, `prepare_data.py`, `it2edge/train/__init__.py` | Your fine-tune + CSV→corpus prep |
| `it2edge/convert/convert_ct2.py`, `it2edge/convert/__init__.py` | HF → CT2 int8 conversion |
| `it2edge/download_model.py` | One-time HF base download |
| `it2edge/corpus_utils.py` | Corpus helpers (named in `it2edge/__init__.py __all__`) |
| `it2edge/evaluate/*` | Quality/latency + A/B benchmarking |
| `it2edge/paths.py`, `it2edge/__init__.py` | **KEEP** — runtime + build both import these |
| `final.md` | Model base + fine-tune + int8 lineage (worth keeping) |
| `pyproject.toml`, `.python-version` | Package/interpreter config (has stale ≥3.11 pin + Pi entry points — trim later) |
| `Dataset_English_Hindi.csv` (39M), `en-indic-exp/` (39M), `en-indic-exp-mini/` | Training corpus — keep only if you'll retrain/re-export |

### 5C. KEEP — the Jetson runtime + this plan
`nvidia/` (all of it), the three shared `it2edge` runtime modules (§5, top), `requirements/jetson.txt`,
`requirements/dev.txt` (needed for Phase 2 export on a dev box), `NvidiaPlan.md`, `NvidiaRun.md`,
`jetsonNvidia.md`, `.gitignore`, `.claude/`.

### 5D. Model directories on disk (all git-ignored — disk cleanup, not the repo)
| Dir | Size | Verdict |
|---|---|---|
| `model_onnx/` | 859 MB | **KEEP** — runtime MT. After int8 re-export you can delete the unused `decoder_model_merged.onnx` (234 MB) |
| `model_cache_compact_ct2/` | 80 MB | **KEEP** — CPU fallback (`TRANSLATE_BACKEND=ct2`) |
| `model_cache_compact_ft/` | 1.2 GB | **KEEP root**, **DELETE `checkpoint-24162/`** (~885 MB optimizer/scheduler/rng state — useless for inference/export) |
| `model_cache_compact/` | 1.7 GB | **TRIM/DELETE** — `opus-mt-en-hi/` holds **4 redundant weight formats** (`pytorch_model.bin` + `tf_model.h5` + `rust_model.ot` + `model.safetensors`); delete the whole dir if you keep `model_cache_compact_ft/` |
| `model_cache/` | 2.1 GB | **DELETE** — legacy IndicTrans2 200M (abandoned approach) |
| `model_cache_merged/` | 412 MB | **DELETE** — legacy IndicTrans2 merged |
| `model_cache_ct2/` | 209 MB | **DELETE** — legacy IndicTrans2 CT2 (incomplete — no tokenizer) |
| `lora_adapters/` | 37 MB | **DELETE** — legacy IndicTrans2 LoRA adapter |

**Disk reclaimed:** IndicTrans2 dirs (`model_cache/` + `model_cache_merged/` + `model_cache_ct2/`
+ `lora_adapters/`) ≈ **2.75 GB**; plus `model_cache_compact/` ≈ 1.7 GB and the `checkpoint-24162/`
optimizer state ≈ 0.88 GB → **up to ~5.3 GB freed** on the SD card, model still fully intact.

### 5E. One-shot cleanup commands (run on the new branch, review before executing)
```bash
# --- 5A: legacy code / docs / artifacts ---
git rm -r --cached deploy it2edge.egg-info 2>/dev/null; rm -rf deploy it2edge.egg-info __pycache__
rm -f laptop_mic.py translate.py tramslate.py des.txt pla.md ab_report.json ab_report_smoke.json
rm -f it2edge/serve/app.py it2edge/serve/translate_ct2.py
rm -f docs/plan.md docs/reduceParams.md docs/runtimeRasp.md docs/STT-TTS.md
rm -f requirements/pi.txt requirements/laptop.txt requirements/docker.txt
#   (before the line above: copy pi.txt's real pins into requirements/jetson.txt)

# --- 5D: disk-only, git-ignored model dirs (NOT tracked — plain rm) ---
rm -rf model_cache model_cache_merged model_cache_ct2 lora_adapters   # ~2.75 GB, legacy IndicTrans2
rm -rf model_cache_compact                                            # ~1.7 GB, redundant weight formats
rm -rf model_cache_compact_ft/checkpoint-24162                        # ~0.88 GB, optimizer state
```

> After deleting, `grep -rn "serve.app\|translate_ct2\|requirements/pi" --include=*.py --include=*.txt .`
> to confirm nothing you kept still references a removed file, and trim `pyproject.toml`'s
> `[project.scripts]` (the `it2-*` entry points point at deleted/kept modules) + `it2edge/serve/__init__.py`'s docstring.

---

## 6. Quantizing the ONNX model to int8 (Phase 2 detail)

On a **dev machine** (not the Nano — export tooling wants modern Python), from
`model_cache_compact_ft/`:

```bash
pip install optimum[onnxruntime] onnx onnxruntime

# (a) export int8 directly via optimum quantization, OR
optimum-cli export onnx \
  --model model_cache_compact_ft \
  --task text2text-generation-with-past \
  model_onnx_fp32
# then dynamic-quantize each graph to int8:
python - <<'PY'
from onnxruntime.quantization import quantize_dynamic, QuantType
import glob, os
for f in glob.glob("model_onnx_fp32/*.onnx"):
    out = os.path.join("model_onnx", os.path.basename(f))
    quantize_dynamic(f, out, weight_type=QuantType.QInt8)
PY
# copy tokenizer files (*.spm, vocab.json, *config*.json, special_tokens_map.json) into model_onnx/
```

Then `scp model_onnx/ jetson:~/entm/`. Your `marian_onnx.py` loads it unchanged. Re-run
`python -m nvidia.jetson_mic --bench` and confirm the RAM drop in `tegrastats`.

> If int8 ONNX ever misbehaves on the Nano's old ONNX Runtime build, the CPU CT2 int8 model is
> your safety net — set `TRANSLATE_BACKEND=ct2`. It's already in `engine.py`.

---

## 7. Dependency notes for JetPack 4.6 / Python 3.8

Add to `requirements/jetson.txt` (already floors CT2/transformers for cp38):
- **Silero VAD** needs `torch`. Torch on JetPack 4.6 = NVIDIA's Jetson torch wheel (not PyPI).
  If that's heavy, use **`webrtcvad`** (pure C, trivial, no torch) as the primary VAD instead —
  the client's VAD interface (feed frames → start/end events) is identical, so it's a drop-in.
  **Recommendation:** start with `webrtcvad` for zero torch weight; upgrade to Silero only if
  energy VAD is too noisy in your environment.
- `onnxruntime-gpu` must be **NVIDIA's Jetson wheel** — the PyPI `onnxruntime` is CPU-only on
  aarch64. (Already noted in `requirements/jetson.txt`.)
- Sockets, queues, threads = **standard library** — no new deps for the server/client core.

---

## 8. Alternative architectures — is there something better than what you described?

Your design (VAD → socket server → stage queue) is the right shape for this hardware. Ranked
alternatives, honestly:

### 8.1 ✅ Your design + streaming/partial STT (the one real upgrade)
Instead of waiting for the full utterance to end before STT runs, feed **overlapping chunks**
to Whisper and translate **partial** transcripts, refining as more audio arrives. Lower
perceived latency (Hindi starts before you finish the sentence). **But:** Whisper isn't
natively streaming, chunking hurts accuracy, and MarianMT re-translating partials wastes the
Nano's modest GPU. **Verdict:** nice-to-have, add *after* the utterance-level pipeline works —
don't start here.

### 8.2 ⚖️ asyncio single-process instead of threads + socket
One `asyncio` event loop doing VAD + mic + stages, no socket, no client/server split. Simpler
to write, less IPC. **But:** the GPU calls (Whisper/ONNX) are blocking C — you'd need a thread
pool anyway, and you lose the "multiple clients / headless daemon + debug CLI" flexibility the
socket gives you. **Verdict:** viable if you only ever want one client; the socket split ages
better and matches what you asked for.

### 8.3 ❌ Microservices / gRPC / message broker (Redis, MQTT, ZeroMQ)
Full network transport + broker between stages. **Overkill on one 4 GB board** — every hop
costs RAM and latency you don't have. ZeroMQ is the least-bad of these if you ever go
multi-board, but for one Nano it's pure overhead. **Verdict:** don't.

### 8.4 ❌ Re-launch a CLI per utterance
What the old `jetson_mic.py` almost does. Pays cold start (CUDA context + model load = seconds)
**every utterance**. **Verdict:** the single worst choice on this hardware — the warm long-lived
server exists precisely to avoid this.

### 8.5 🔮 Whisper.cpp / TensorRT for STT
For max STT speed on the Nano, `whisper.cpp` (with CUDA) or a TensorRT-optimized Whisper can beat
faster-whisper. Real gains, but a separate build effort on CUDA 10.2. **Verdict:** a Phase-7
optimization if STT latency is your bottleneck after §4 — measure first.

**Bottom line:** build exactly what you described (§2–§4). The only alternative worth folding in
*later* is 8.1 (partial STT) for lower perceived latency, and 8.5 if STT is the measured
bottleneck. Everything else is more complexity than a single Nano rewards.

---

## 9. New files this plan adds (all under `nvidia/`)

| File | Role |
|---|---|
| `nvidia/vad.py` | VAD wrapper: feed 30 ms frames → `speech_start` / `speech_end` events (webrtcvad first, Silero optional) |
| `nvidia/protocol.py` | Socket wire format: length-prefixed audio frames in, JSON event lines out |
| `nvidia/server.py` | Warm `Engine` + UNIX socket + bounded utterance queue + STT/MT/TTS worker threads |
| `nvidia/client.py` | Mic → VAD → stream frames on speech, `END` on pause; prints Hindi + status events |
| `nvidia/it2-jetson.service` | (optional) systemd unit: `nvpmodel -m 0` + `jetson_clocks` then launch server |

**Reused unchanged:** `nvidia/engine.py`, `nvidia/marian_onnx.py`, `it2edge/serve/speak.py`,
`it2edge/serve/marian_ct2.py`, `it2edge/paths.py`. `nvidia/jetson_mic.py` stays as the
push-to-talk *debug/baseline* tool.

---

## 10. Definition of done

- [ ] New branch off `main`; §5A files + legacy model dirs deleted; branch is Jetson-only
- [ ] ONNX model quantized to int8 in `model_onnx/`; RAM under 4 GB with Whisper `base` (tegrastats)
- [ ] `nvidia/server.py` loads both models once, listens on UNIX socket, 0% CPU when idle
- [ ] `nvidia/client.py` streams only during speech; socket silent during silence
- [ ] Continuous speech: utterance N+1 transcribes while N is still being spoken (overlap proven)
- [ ] Overloaded input drops oldest utterance **with a logged/visible warning**, RAM stays bounded
- [ ] Speak→hear latency ≈ 1.5–4 s, models warm; `nvpmodel -m 0` + `jetson_clocks` set
- [ ] Runs fully offline — unplug the network and it still works

---

## 11. Expected latency for a real utterance (~3 lines, not 1–2 words)

This product is not a word-translator. A real user speaks a **~3-line utterance** — roughly
**40–55 words, ~12–18 seconds of speech, ~3 sentences**. Latency behaves very differently from
a single word, so here are honest numbers for that case, on a Jetson Nano at max clocks
(`nvpmodel -m 0 && jetson_clocks`), models warm, int8 ONNX translation on GPU + Whisper `base`
on GPU + Piper on CPU.

### 11.1 The number that matters: time-to-first-Hindi, not total

There are two very different "latencies" and you must not quote the wrong one:

| Metric | What it measures | This is what the user *feels* |
|---|---|---|
| **Time-to-first-audio** | You stop talking → **first Hindi sentence** starts playing | ✅ **yes** — this is "responsiveness" |
| **Total turn time** | You stop talking → **last** Hindi sentence finishes playing | mostly playback of 3 sentences |

Because the pipeline is **sentence-level** (§11.3), the user hears line 1 while lines 2–3 are
still being translated/synthesized. So the felt latency is time-to-first-audio, not the total.

### 11.2 Where the time goes (per full ~3-line utterance, models warm)

| Stage | Device | ~3-line cost | Note |
|---|---|---|---|
| VAD end-of-speech detect | CPU | 0.4–0.6 s | the trailing-silence window before we call the utterance "done" |
| **Whisper STT `base`** | **GPU** | **~4–7 s** | scales with audio length; ~0.3–0.5× realtime on the Nano for ~15 s of audio. **The dominant cost.** |
| **Translate (int8, greedy)** | **GPU** | **~0.3–0.8 s** | 3 sentences × ~0.1–0.25 s; tiny 77M model, near-free |
| **Piper TTS** | **CPU** | **~3–6 s to synth all 3**, but **streams early** | ~1× realtime; playback of line 1 starts long before line 3 is synthesized |

### 11.3 Two ways to run it — and why sentence-level wins for 3 lines

**(A) Whole-utterance (simple, higher felt latency):** transcribe all 3 lines → translate all 3
→ then speak. First Hindi is heard only after **all** STT + **all** MT finish.

```
felt latency ≈ VAD(0.5) + STT_all(~5.5) + MT_all(~0.6) + TTS_first_sentence(~1.2)
             ≈ 7–9 s  before you hear anything, then ~4–6 s more of playback
```

**(B) Sentence-level pipeline (recommended — build this):** Whisper already returns segments;
emit each **sentence** into the MT→TTS queue as soon as it's transcribed. Then the first Hindi
sentence plays while the GPU is still transcribing/translating the rest.

```
felt latency (time-to-first-audio)
   ≈ VAD(0.5) + STT_sentence1(~2) + MT_sentence1(~0.2) + TTS_start_sentence1(~0.3, streams)
   ≈ 3–4 s  before you hear the first Hindi line
total turn ≈ ~9–13 s (mostly the audio playback of 3 Hindi sentences — unavoidable, it's speech)
```

> **Design consequence:** make the pipeline **sentence-level**, not utterance-level. The stage
> queue in §2.2 should carry *sentences*, and the STT worker should push each Whisper segment
> onward the moment it's ready instead of waiting for the whole utterance. This is the single
> biggest latency lever for multi-line speech and costs almost nothing to implement.

### 11.4 Bottom line (what to promise)

For a **~3-line utterance**, on a warm Jetson Nano at max clocks with the sentence-level pipeline:

- **Time-to-first-Hindi: ~3–4 seconds** after you stop talking. *(This is the headline.)*
- **Total turn (until the last line finishes speaking): ~9–13 seconds**, most of which is simply
  the time it takes to *play* 3 Hindi sentences aloud — not compute.
- **STT is the bottleneck**, not translation. If ~3–4 s isn't fast enough, the levers are, in order:
  Whisper `tiny` instead of `base` (~1.5–2× faster STT, small accuracy cost), then `whisper.cpp`
  or a TensorRT Whisper build (§8.5). Translation and TTS are **not** worth optimizing — they're
  already small next to STT.

> ⚠️ **These are engineering estimates, not measured numbers** — the Nano's real throughput
> depends on thermal throttling, JetPack's ONNX Runtime/CUDA build, and whether `jetson_clocks`
> is actually pinned. Phase 5 (`--bench` + `tegrastats`) replaces every number here with measured
> ones on *your* board. Treat this section as the target, then confirm it.
