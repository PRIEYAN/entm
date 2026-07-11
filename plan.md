# Plan — Fine-tune IndicTrans2 (dist-200M) and deploy to a low-power ARM device with minimal latency

> Goal: take `ai4bharat/indictrans2-en-indic-dist-200M`, **fine-tune it on our own
> data**, package the trained model + full runtime into a **Docker image**, ship
> that image to a **Raspberry-Pi-class device (dual-core, 2 GB RAM, 64-bit)** and
> serve translations locally with **the lowest latency the hardware allows**.

All technical specifics below were verified against primary sources (AI4Bharat
repo, CTranslate2 docs, PyPI, Docker docs) and adversarially fact-checked. Numbers
marked **ESTIMATE** are order-of-magnitude planning figures — measure on the real
device before trusting them.

---

## 0. TL;DR of the whole pipeline

```
                 [ DEV MACHINE (x86-64 + GPU) ]                    [ RASPBERRY PI (aarch64, 2GB) ]
  parallel data ─▶ fine-tune ─▶ merge ─▶ convert to ─▶ build arm64 ──push──▶ pull ─▶ docker run ─▶ warm
   (en → xx)       (LoRA)      one model   CT2 int8      Docker image        (registry)   HTTP service (FastAPI)
                                                                                          intra_threads=2, greedy
```

Latency target on 2 cores @ ~1.5 GHz: **~0.4–1.5 s per sentence (greedy, int8)** — ESTIMATE.
Memory: int8 model + runtime ≈ **250–500 MB resident** — fits 2 GB comfortably.

---

## 1. Target hardware & hard constraints

| Constraint | Value | Consequence |
|---|---|---|
| CPU | ~2 cores, ~1.5 GHz, ARM | CPU-only inference; thread tuning matters a lot |
| RAM | 2 GB | Single model, single worker; no PyTorch-at-runtime luxury |
| OS | **MUST be 64-bit** (aarch64) | `uname -m` must return `aarch64`. 32-bit/armv7 has **no** ctranslate2 wheels → source build → pain. **Reflash to 64-bit Raspberry Pi OS first.** |
| Accelerator | none (no CUDA) | int8 CPU inference via CTranslate2 |

> Note: most real Raspberry Pis are quad-core (Pi 4/5). If the device is actually a
> Pi 4 (4×A72) or Pi 5 (4×A76 @2.4 GHz, has int8 SDOT/UDOT), it will be **faster**
> than this 2-core planning assumption — the plan is conservative.

---

## 2. THE key decision: how do we get a CT2 model from *our* fine-tuned weights?

Low latency ⇒ we want **CTranslate2 int8**. But CT2 does **not** cleanly accept a
fine-tuned HF IndicTrans2 model, because IndicTrans2 ships a *custom* architecture
(`IndicTransForConditionalGeneration`, `trust_remote_code=True`) that the generic
`ct2-transformers-converter` doesn't list among supported archs (M2M100/NLLB/Marian…).
So there are three routes, with an explicit recommendation:

| Route | Fine-tune with | → CT2? | Latency | Effort / risk |
|---|---|---|---|---|
| **A (recommended)** | **fairseq** (`finetune.sh`, arch `transformer_base18L`) | ✅ official `fairseq-ct2-converter` | 🟢 lowest (CT2 int8) | Heavier training setup, but the CT2 path is *supported* and reliable |
| **B** | **HF LoRA** (`train_lora.sh`) → `merge_and_unload()` | ⚠️ HF→CT2 likely **fails** on custom arch | 🟢 if conversion works | Easiest training; **CT2 conversion is the gamble** — treat as experimental, verify against current CT2 release notes |
| **C (fallback)** | **HF LoRA** → merge, **deploy merged HF model as-is** (PyTorch) | ❌ no CT2 | 🔴 highest, torch in 2 GB is tight | Simplest to get *working*, worst latency — only if A and B both stall |

**Recommendation:** pursue **Route A (fairseq → CT2)** as the primary path because
it is the only one where "fine-tuned" *and* "ultra-low-latency CT2 int8 on the Pi"
are both guaranteed. Keep **Route B** as a quick experiment (it's cheap to try the
HF LoRA fine-tune first since it's the friendlier workflow), and **Route C** as the
guaranteed-works safety net if the CT2 conversion of custom weights proves impossible.

> Reality check to do in Phase 0: attempt `fairseq-ct2-converter` on the *stock*
> distilled fairseq checkpoint **before** investing in fine-tuning. If that
> round-trips to a working CT2 model, Route A is confirmed viable end-to-end.

---

## 3. Phase 1 — Fine-tuning (on the dev machine with a GPU)

### 3a. Hardware to fine-tune
- A single **12–16 GB GPU** is plenty (free Colab/Kaggle **T4 16 GB**, or RTX 3060 12 GB).
  200M params, and LoRA trains only ~0.8 M params (~0.4%).
- **CPU-only fine-tuning is impractical** (the `inverse_sqrt` schedule + long
  `max_steps` assume GPU throughput). Use CPU only for a tiny smoke test.

### 3b. Data format (parallel corpus)
Line-aligned plain text, one sentence per line, in the layout the scripts expect:
```
en-indic-exp/
  train/eng_Latn-hin_Deva/train.eng_Latn   # English lines
  train/eng_Latn-hin_Deva/train.hin_Deva   # aligned Hindi lines
  dev/eng_Latn-hin_Deva/dev.eng_Latn
  dev/eng_Latn-hin_Deva/dev.hin_Deva
```
- Language codes are FLORES-style: `eng_Latn` → `hin_Deva`, `tam_Taml`, `ben_Beng`, `tel_Telu`, …
- **Feed RAW text.** `IndicProcessor` (from IndicTransToolkit) does Unicode
  normalization, script unification, language tagging, and entity masking
  *internally*, before tokenization. Do **not** pre-transliterate yourself — training
  and inference must use the *same* IndicProcessor pass.

### 3c. Route B (HF LoRA) — the easy first attempt
From `IndicTrans2/huggingface_interface/`:
```bash
bash install.sh    # clones + installs IndicTransToolkit, peft, transformers, etc.
bash train_lora.sh <data_dir> ai4bharat/indictrans2-en-indic-dist-200M <out_dir> \
     "eng_Latn" "hin_Deva,tam_Taml,ben_Beng"
```
Verified defaults (from `train_lora.sh`): `batch_size 32`, `grad_accum 4`,
`lr 2e-4`, `warmup 4000`, `inverse_sqrt`, `adamw_torch`, `weight_decay 0.01`,
`save_steps 1000`, early-stop patience 10 on `eval_BLEU`.
LoRA: `r=16, alpha=32, dropout=0.1, target_modules=[q_proj,k_proj], task=SEQ_2_SEQ_LM`.

> **OOM warning:** `batch_size=32` will likely OOM a 16 GB T4 (activations dominate).
> Drop to `--batch_size 8` (raise `--grad_accum_steps` to keep effective batch), or use
> QLoRA 4-bit (unofficial). Hyperparameters are explicitly "may need tuning."

### 3d. Merge to one deployable model (both routes need this)
`train_lora.py` saves **adapters only** — it never merges, and **CT2 can't read
adapters**. Merge first:
```python
from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel
base = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR, trust_remote_code=True)
merged = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
merged.save_pretrained(FINAL_DIR)   # + tokenizer.save_pretrained(FINAL_DIR)
```
⚠️ Remember the earlier bug: `save_pretrained` writes vocab-file keys that break the
tokenizer on reload. Reuse our `tokenizer_utils.load_indictrans_tokenizer` fallback,
or snapshot the tokenizer files instead of `save_pretrained` for the tokenizer.

### 3e. Route A (fairseq) — the reliable-CT2 path
From the main `IndicTrans2` repo:
```bash
bash prepare_data_joint_finetuning.sh <exp_dir>          # binarize
bash finetune.sh <exp_dir> transformer_base18L <pretrained_ckpt>   # 18L = distilled arch
```
Produces a fairseq checkpoint that `fairseq-ct2-converter` can convert (Phase 2).

---

## 4. Phase 2 — Convert to CTranslate2 int8

### Route A (recommended)
```bash
pip install ctranslate2
ct2-fairseq-converter \
    --model_path <ft>/checkpoint_best.pt \
    --data_dir   <ft>/final_bin \
    --output_dir model_cache_ct2 \
    --quantization int8
```

### Route B (experiment — may fail on custom arch)
Reuse our existing `convert_ct2.py` (it already tries `ct2-transformers-converter
--trust_remote_code` and prints the fairseq fallback if it fails).

**Quantization choice:** use `int8`. On ARM this resolves to `int8_float32` via the
Ruy backend. `int8_float16` and `int16` are **not** natively supported on ARM (they
fall back to int8_float32) — requesting them buys nothing.

Output: a `model_cache_ct2/` directory (~200 MB) — this is what goes in the image.

---

## 5. Phase 3 — Build the ARM64 Docker image (on the dev machine)

### Base & dependencies (all prebuilt aarch64 wheels — **no compilation**)
- Base: **`python:3.11-slim`** (Debian bookworm, glibc 2.36). **Not** Alpine/musl —
  manylinux wheels are glibc-only and would trigger a source build on musl.
- `ctranslate2` **4.8.x** ships aarch64 manylinux wheels (glibc ≥2.27), CPU-only,
  with OpenBLAS+Ruy statically bundled → no `apt` BLAS needed.
- `sentencepiece`, `numpy` also ship aarch64 wheels.
- **Pin exact versions** and never pass `--no-binary`, so pip can't fall back to an
  sdist source build (that's the one thing that makes QEMU builds slow).
- Requires **pip ≥ 20.3** in the image (PEP 600 tag support) — trivially satisfied by slim.

### Dockerfile sketch
```dockerfile
FROM python:3.11-slim
ENV OMP_NUM_THREADS=2 CT2_INTER_THREADS=1 CT2_INTRA_THREADS=2
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt   # ct2, sentencepiece, IndicTransToolkit, fastapi, uvicorn
COPY app.py .
# model in its own LATE layer so code changes reuse the cached ~200MB layer
COPY model_cache_ct2/ /app/model_cache_ct2/
COPY model_cache/indictrans2-en-indic-dist-200M/ /app/tokenizer/
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

### Cross-build for arm64 from x86 (buildx + QEMU)
```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64        # once per boot
docker buildx create --name multi --driver docker-container --bootstrap
docker buildx use multi
docker buildx build --platform linux/arm64 -t <registry>/it2-ct2:latest --push .
```
- A cross-platform image **cannot** be `--load`ed into the x86 daemon — it **must be
  `--push`ed** to a registry, then pulled on the Pi.
- Because everything is wheels, QEMU only *unpacks* wheels (no C++ compile) → build is
  minutes, not hours. (The horror stories are about source builds.)
- Alternative with zero emulation: **build natively on the Pi** (viable precisely
  because nothing compiles).

### Model: bake in vs. mount
**Bake it in** (recommended for a single-purpose Pi appliance): one immutable
artifact, `docker run` needs nothing else, easy rollback by tag. Cost: image +200 MB,
model updates need a rebuild+repush. Put the `COPY model_...` in a late layer.
Use a **mounted volume** (`-v /srv/models:/models:ro`) only if you expect frequent
model swaps.

**Image size (ESTIMATE):** ~250–300 MB without model / **~450–500 MB with model baked**
(~330–360 MB compressed in the registry).

---

## 6. Phase 4 — Run on the Pi

```bash
docker run -d --name it2 --restart unless-stopped \
  --cpus 2 --memory 1500m --memory-swap 1500m \
  -p 8080:8080 <registry>/it2-ct2:latest
```
- **Long-lived HTTP service, not one-shot CLI** — load the ~200 MB model **once** at
  startup and keep it warm. A CLI would pay the load cost every request.
- **Single uvicorn worker.** On 2 GB, each extra worker duplicates the model in RAM →
  OOM. Get parallelism from CT2 threads, not processes.
- Cap memory ~1.5 GB to leave OS headroom; `--restart unless-stopped` survives reboots.

---

## 7. Phase 5 — Latency optimization (the "very less latency" part)

### Threading (biggest structural lever on 2 cores)
- `ctranslate2.Translator(model, device="cpu", compute_type="int8",
  inter_threads=1, intra_threads=2)` and `OMP_NUM_THREADS=2`.
- Rule from CT2 docs: `inter_threads * intra_threads ≤ physical cores`. For **lowest
  single-sentence latency**, put all cores on one translation → `1 × 2`.
- (Only if you optimize for *throughput* over many concurrent requests would you flip
  to `inter_threads=2, intra_threads=1` — that raises per-sentence latency, so don't.)

### Decoding settings (per-request)
- **`beam_size=1` (greedy)** — the single biggest latency win vs the default beam=5.
  Small quality cost; measure BLEU on your dev set to confirm it's acceptable.
- **`return_scores=False`** — lets CT2 skip the final softmax each step under greedy.
- **Cap `max_decoding_length`** (~1.5–2× expected output length) so runaway/looping
  generations can't blow latency.
- **`max_input_length`** to bound encoder cost; split paragraphs into sentences.
- **`batch=1`** for single-sentence latency; only batch when many sentences arrive,
  then `max_batch_size` with `batch_type="tokens"` and the iterable API (sorts by length).
- Keep the model **warm/resident**; disable unused features (no sampling, no
  `return_alternatives`, minimal repetition penalties).

### Realistic expectations (ESTIMATE — verify on device)
| Sentence length | Latency (greedy, int8, 2 cores @1.5 GHz) |
|---|---|
| short (~10 out tokens) | ~0.2–0.4 s |
| typical (~20–30 tokens) | ~0.4–1.5 s |
| long (50+ tokens) | ~2–3 s |
- Basis: ~15–60 ms/output token on Pi-class CPU; single-stream decode is
  **memory-bandwidth bound** (~90–100 MB int8 weights streamed per token). Beam search
  would multiply this; Pi thermal throttling and SD-card cold-start also matter.
- **Memory:** ~250–500 MB resident at batch=1 — comfortable in 2 GB.

### If you need it faster still (stretch options)
- Shorter outputs / sentence-splitting (latency ∝ tokens generated).
- Distill/prune further or reduce vocab (advanced; changes the model).
- A Pi 5 (A76 @2.4 GHz, native int8 dot-product) is materially faster than the 2-core assumption.

---

## 8. Consolidated risks & gotchas

1. **HF→CT2 conversion of custom IndicTrans2 arch likely fails** → primary path is
   fairseq→CT2 (Route A). *Validate in Phase 0 before fine-tuning.*
2. **32-bit OS = no wheels.** Pi must run **64-bit** Raspberry Pi OS (`uname -m` → `aarch64`).
3. **Alpine/musl base = source build.** Use `python:3.11-slim` (glibc).
4. **`train_lora.py` saves adapters only** — you must `merge_and_unload()` (CT2 can't read adapters).
5. **`batch_size=32` OOMs a T4** — drop to 8–16 or QLoRA 4-bit.
6. **`save_pretrained` tokenizer bug** (already hit) — reuse `tokenizer_utils` fallback / snapshot tokenizer files.
7. **Cross-platform image must be `--push`ed**, not `--load`ed, from x86.
8. **Single worker on 2 GB** — never multi-worker; scale via CT2 threads.
9. **Pin deps, no `--no-binary`** — one unpinned dep lacking an arm64 wheel silently triggers a slow QEMU source build.
10. **Greedy vs beam is a quality/latency tradeoff** — measure BLEU before committing to greedy in production.

---

## 9. Deliverables to build (scripts/files)

| File | Purpose | Status |
|---|---|---|
| `finetune_lora.sh` / notebook | wrap AI4Bharat `train_lora.sh` for our data (Route B) | to build |
| `merge_lora.py` | `merge_and_unload()` → single HF model dir | to build |
| `convert_ct2.py` | HF→CT2 attempt + fairseq fallback message | ✅ exists |
| (Route A) fairseq fine-tune + `ct2-fairseq-converter` runbook | reliable CT2 artifact | to build |
| `app.py` | FastAPI service: warm CT2 model, `/translate`, `/health` | to build |
| `requirements-docker.txt` | pinned arm64 deps (ct2, sentencepiece, IndicTransToolkit, fastapi, uvicorn) | to build |
| `Dockerfile` | arm64 image, model baked in late layer | to build |
| `build_arm64.sh` | buildx + QEMU cross-build + push | to build |
| `translate_ct2.py` | local CLI sanity check (compare vs PyTorch `translate.py`) | ✅ exists |

---

## 10. Suggested sequencing (milestones)

- **M0 — Feasibility (½ day):** confirm fairseq→CT2 round-trips on the *stock* distilled
  checkpoint; confirm 64-bit OS on the Pi; A/B CT2 vs torch latency on the Pi with the stock model.
- **M1 — Fine-tune (1–2 days):** prep data, run HF LoRA (Route B) as the easy first cut, `merge_and_unload()`, sanity-check BLEU.
- **M2 — CT2 artifact (½ day):** convert merged model (Route B) *or* fairseq checkpoint (Route A) to int8; validate output vs PyTorch.
- **M3 — Containerize (1 day):** `app.py` warm service, Dockerfile, buildx arm64, push to registry.
- **M4 — Deploy & tune (1 day):** run on Pi, set threads/greedy/limits, **measure real latency**, iterate.

---

*Prepared from source-verified research (AI4Bharat IndicTrans2 repo, CTranslate2 docs,
PyPI, Docker multi-platform docs). Figures marked ESTIMATE are planning-grade — the
one number that matters most, on-device latency, must be measured on the actual Pi.*
