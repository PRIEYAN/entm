# Plan — Fine-tune, compress, and deploy IndicTrans2 (dist-200M) to a low-power ARM device

> Goal: take `ai4bharat/indictrans2-en-indic-dist-200M`, **fine-tune it (QLoRA)**,
> squeeze it as small/fast as possible (**int8**, optional **structured pruning**),
> package it into a **Podman** container image, ship that to a **Raspberry-Pi-class
> device (dual-core, 2 GB RAM, 64-bit)**, and serve translations locally at **the
> lowest latency the hardware allows**. ONNX Runtime is documented as an alternative
> runtime to CTranslate2.

Every technical specific below was researched against primary sources (AI4Bharat
repo, CTranslate2 / ONNX Runtime / optimum docs, PyPI, bitsandbytes/PEFT docs) and
cross-checked. Numbers marked **ESTIMATE** are order-of-magnitude planning figures —
measure on the real device. See §12 for a verification-confidence note.

---

## 0. Read this first — three corrections to the brief

These reshape the plan, so they're up front (not to nitpick — to save you effort):

1. **There is no "64-bit" model.** Transformer weights ship as **fp32 (32-bit)**,
   or **fp16 (16-bit)** when loaded for GPU work. The right framing is *fp32/fp16 →
   int8 for the Pi*, not "64→8".
2. **QLoRA is 4-bit, and it's a *training* trick — not your deployment artifact.**
   QLoRA = freeze the base in **4-bit NF4** (bitsandbytes) + train 16-bit LoRA
   adapters, purely to fit the fine-tune in low VRAM. After you `merge_and_unload()`
   you get an **fp16** model back (~400 MB) — *bigger and float again*. The small
   **int8** artifact for the Pi is a **separate post-training conversion step**.
   ("8-bit + QLoRA" is a contradiction; QLoRA is 4-bit. If you truly wanted 8-bit
   *at train time* that's `load_in_8bit` LoRA, a different technique — not needed here.)
3. **"Dynamically prune the weights that don't contribute" won't make it faster.**
   That describes *unstructured/magnitude* pruning (zeroing individual weights). On a
   dense CPU int8 backend it gives a **smaller file and identical latency** — neither
   CTranslate2 nor ONNX Runtime's CPU path exploits unstructured sparsity (that needs
   a specialized sparse engine like DeepSparse). **Only *structured* pruning** (dropping
   whole attention heads / FFN neurons / layers) actually shrinks the dense math and
   speeds up CPU inference. §5 uses structured pruning; treat it as optional/advanced.

**Correct end-to-end ordering (memorize this):**
```
QLoRA 4-bit fine-tune ─▶ merge to fp16 ─▶ (optional) STRUCTURED prune + heal (float)
                                                        │
                                                        ▼
                                          int8 conversion LAST (CT2 or ONNX) ─▶ ship
```
int8 goes **last** because pruning/healing need float weights; int8 is a deployment-only transform.

---

## 1. TL;DR pipeline

```
        [ DEV MACHINE (x86-64 + GPU) ]                              [ RASPBERRY PI (aarch64, 2GB) ]
 data ─▶ QLoRA(4bit) ─▶ merge→fp16 ─▶ [opt] struct-prune+heal ─▶ int8 ─▶ build arm64 ──push──▶ pull ─▶ podman run
 (en→xx)  (~0.8M         (one fp16      (fairseq retrain,          (CT2   Podman image      (registry) rootless +
          trainable)     model)         decoder-depth cut)         primary) (+ manifest)               systemd quadlet
```
Runtime on 2 cores @ ~1.5 GHz: **~0.4–1.5 s/sentence** (int8, greedy) — ESTIMATE.
RAM: int8 model + runtime ≈ **250–500 MB** resident; Podman adds **~0** persistent daemon RAM.

---

## 2. Target hardware & hard constraints

| Constraint | Value | Consequence |
|---|---|---|
| CPU | ~2 cores, ~1.5 GHz, ARM | CPU-only; thread pinning matters a lot |
| RAM | 2 GB | one model, one worker; no torch-at-runtime |
| **OS** | **MUST be 64-bit (aarch64)** | `uname -m` → `aarch64`. 32-bit has **no** CT2 / ONNX Runtime wheels → source build. **Reflash to 64-bit first.** |
| Accel | none | int8 CPU inference |

> Most real Pis are quad-core (Pi 4/5). A Pi 5 (A76 @2.4 GHz, native int8 SDOT/UDOT)
> is materially faster than this 2-core planning assumption — the plan is conservative.

---

## 3. Runtime decision: CTranslate2 (primary) vs ONNX Runtime (alternative)

You asked for ONNX Runtime, so both are specified. But the honest ranking for this
model on ARM:

| | CTranslate2 (**recommended**) | ONNX Runtime (alternative) |
|---|---|---|
| Seq2seq latency on ARM CPU | 🟢 lowest — C++ greedy/beam + KV cache, **no per-token Python** | 🟡 higher — `ORTModelForSeq2SeqLM` runs `generate()` in Python, one `session.run` per token |
| Memory | 🟢 int8 model ~100–200 MB | 🟡 higher (float activations, up to 3 graphs, full transformers stack) |
| IndicTrans2 support | 🟢 **essentially solved** — AI4Bharat ships official CT2 ports; `fairseq-ct2-converter` works | 🔴 **custom-arch export required** (see §7b) |
| int8 | `--quantization int8` (one flag, no calibration) | `quantize_dynamic` (U8S8+reduce_range on ARM, no calibration) |
| aarch64 wheels | ✅ ctranslate2 4.8.x | ✅ onnxruntime 1.27.x |

> **Key correction to an earlier assumption:** CT2 is **not** actually blocked for
> IndicTrans2. Only the *HF-transformers* converter (`ct2-transformers-converter`)
> fails on the custom arch. The **fairseq → CT2** route (`fairseq-ct2-converter`) is
> official and works, and AI4Bharat even publishes pre-built CT2 dirs. So CT2 is both
> the fastest *and* the lowest-effort path.

**Recommendation:** make **CT2 the primary runtime**. Pursue **ONNX Runtime only** if
you have a hard reason (an existing ORT-only serving stack, one runtime across many
models, a needed ORT execution provider). §7 covers both.

---

## 4. Phase 1 — QLoRA fine-tuning (dev machine, GPU)

### Hardware
- One **12–16 GB GPU** (free Colab/Kaggle **T4 16 GB**, RTX 3060 12 GB). QLoRA 4-bit
  makes this comfortable; you train only ~0.8 M LoRA params (~0.4% of the model).
- CPU-only fine-tune is impractical (schedule assumes GPU throughput) — smoke-test only.

### Data format (parallel corpus, line-aligned raw text)
```
en-indic-exp/
  train/eng_Latn-hin_Deva/train.eng_Latn   train.hin_Deva
  dev/eng_Latn-hin_Deva/dev.eng_Latn       dev.hin_Deva
```
- FLORES codes: `eng_Latn` → `hin_Deva`, `tam_Taml`, `ben_Beng`, `tel_Telu`, …
- **Feed RAW text.** `IndicProcessor` (from IndicTransToolkit) does normalization,
  script unification, language tagging, entity masking *internally*. Do not
  pre-transliterate — training and inference must share the same IndicProcessor pass.

### QLoRA setup
Start from AI4Bharat's `IndicTrans2/huggingface_interface/` (`train_lora.py` is the
supported HF fine-tune). Add 4-bit loading to make it QLoRA:
```python
from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16", bnb_4bit_use_double_quant=True,
)
base = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-en-indic-dist-200M",
    trust_remote_code=True, quantization_config=bnb, device_map="auto",
)
```
Verified LoRA config from `train_lora.py`: `r=16, lora_alpha=32, lora_dropout=0.1,
target_modules=["q_proj","k_proj"], task_type="SEQ_2_SEQ_LM"`. Baseline HP (from
`train_lora.sh`): `lr 2e-4`, `warmup 4000`, `inverse_sqrt`, `adamw_torch`,
`weight_decay 0.01`, early-stop patience 10 on `eval_BLEU`.
> **OOM note:** default `batch_size=32` will likely OOM a 16 GB T4 — drop to 8
> (raise `grad_accum` to keep effective batch). Hyperparameters "may need tuning."

---

## 5. Phase 2 — Merge, and (optional) structured pruning

### 5a. Merge adapters → single fp16 model (required)
`train_lora.py` saves **adapters only**; CT2/ONNX can't read adapters. Merge — and
**dequantize to fp16 first**, then merge (merging into a still-4-bit base degrades
quality, since adapters were trained against dequantized weights):
```python
from peft import PeftModel
base_fp16 = AutoModelForSeq2SeqLM.from_pretrained(BASE, trust_remote_code=True,
                                                   torch_dtype="float16")
merged = PeftModel.from_pretrained(base_fp16, ADAPTER_DIR).merge_and_unload()
merged.save_pretrained(FINAL_DIR)     # ~400 MB fp16
```
⚠️ Reuse our `tokenizer_utils.load_indictrans_tokenizer` (the `save_pretrained`
tokenizer reload bug we already hit).

### 5b. Structured pruning + heal (OPTIONAL — do this last, only if still too big/slow)
**This is what you asked for, done correctly.** Recall from §0: unstructured/"dynamic"
pruning = no CPU speedup. Use **structured** pruning.

- **What actually pays off (verified insight):** decoder **depth** dominates
  autoregressive latency; encoder **width** preserves quality. So the highest-yield
  structured move is **shorten decoder layers** + modestly **narrow the encoder**,
  not uniform pruning. (Ref: NASH, seq2seq structured pruning.)
- **Tooling** (for a 200M encoder-decoder): `torch-pruning` (DepGraph — actually
  shrinks tensors), HF `prune_heads` (needs `_prune_heads` hooks the custom arch may
  lack — you'd patch them), NASH recipe. **Avoid** `torch.nn.utils.prune` (masks only,
  no dense speedup), LLM-Pruner (decoder-only), Intel Neural Compressor's default
  (produces *sparsity* for sparse runtimes, not dense shrink).
- **Heal is mandatory:** structured pruning drops BLEU immediately → you must
  fine-tune ("heal") on parallel data and re-check BLEU/chrF, then re-convert.
- **Custom-arch constraint (important):** the CT2 converter assumes **uniform
  topology** (same heads/FFN dim across layers). Heterogeneous per-layer pruning
  **won't round-trip**. Constrain pruning to **converter-legal shapes**: uniform
  per-layer head/FFN reductions + whole-layer drops.

**Honest verdict:** for an *already-distilled* 200M, structured pruning has
**diminishing returns** (distillation already spent the redundancy) and is high-effort
+ fragile through the custom arch. Expect maybe **~10–30%** further size/latency after
healing — not transformative — at a measurable BLEU cost. **Do int8 + greedy first
(§7, §9); only attempt pruning if you still need more.**

---

## 6. Phase 3 — int8 conversion for the Pi (do LAST)

### 6a. CTranslate2 (primary)
- If you fine-tuned via **fairseq**: `ct2-fairseq-converter --model_path
  checkpoint_best.pt --data_dir final_bin --output_dir model_cache_ct2
  --quantization int8`.
- If you fine-tuned via **HF QLoRA → merged fp16**: the HF→CT2 converter may reject
  the custom arch (our `convert_ct2.py` already tries it and prints the fairseq
  fallback). The reliable CT2 artifact comes from the **fairseq** side.
- **Quantization:** use `int8`. On ARM it resolves to `int8_float32` (Ruy backend);
  `int8_float16`/`int16` are **not** native on ARM and just fall back — no benefit.
- Output `model_cache_ct2/` ≈ 100–200 MB.

### 6b. ONNX Runtime (alternative — only if mandated)
The `optimum-cli export onnx` CLI **fails** on IndicTrans2 (custom `model_type` not in
optimum's registry — confirmed by an AI4Bharat HF discussion). The working route is
the **Python API with a custom OnnxConfig**:
```python
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import M2M100OnnxConfig
from optimum.utils import NormalizedTextConfig
# IndicTrans2 ≈ M2M100 (sinusoidal pos-emb, enc/dec transformer) but fairseq attr names.
# Subclass M2M100OnnxConfig; remap normalized attrs:
#   hidden_size=encoder_embed_dim, num_attention_heads=encoder_attention_heads, ...
main_export(MODEL_ID, output="onnx/", trust_remote_code=True, no_post_process=True,
            task="text2text-generation-with-past",
            custom_onnx_configs={"encoder_model": enc, "decoder_model": dec,
                                 "decoder_with_past_model": dec_past})
```
Then int8 with **`quantize_dynamic(..., weight_type=QInt8)`** (dynamic = **no
calibration data**; use **U8S8 + `reduce_range`** on ARM). Caveats: `ORTModelForSeq2SeqLM`
may `KeyError` on the unregistered `model_type` (register the normalized config or
hand-write the decode loop), dynamic-quant seq2seq has known bugs (optimum #438), and
quantizing the decoder can hurt MT quality — **re-validate BLEU/chrF**.
> `IndicProcessor` + tokenizer stay in Python in **both** runtimes — they're
> preprocessing, never in the ONNX/CT2 graph.

---

## 7. Phase 4 — Build the ARM64 image with **Podman** (dev machine)

Podman replaces Docker: **daemonless, rootless, same OCI images & Dockerfile**, and
**~0 persistent RAM** on the Pi (no `dockerd`). Same `Containerfile`/`Dockerfile`.

### Containerfile (identical content to a Dockerfile)
```dockerfile
FROM python:3.11-slim
ENV OMP_NUM_THREADS=2
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt   # ctranslate2, sentencepiece, IndicTransToolkit, fastapi, uvicorn
COPY app.py .
COPY model_cache_ct2/ /app/model_cache_ct2/          # int8 model, late layer (cache reuse)
COPY tokenizer/ /app/tokenizer/                      # tokenizer files
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080","--workers","1"]
```
- Base `python:3.11-slim` (glibc). **Not Alpine/musl** — manylinux wheels are
  glibc-only; musl forces a slow source build.
- **All deps are prebuilt aarch64 wheels** (ctranslate2 4.8.x, sentencepiece,
  onnxruntime 1.27.x, numpy) → **no compilation**, even under emulation. **Pin
  versions**, never `--no-binary` (an unpinned dep with no arm64 wheel silently
  triggers a QEMU source build). Needs **pip ≥ 20.3** (PEP 600 tags) — slim satisfies it.

### Cross-build arm64 on x86 with Podman
```bash
# one-time: register QEMU emulators for cross-arch
podman run --rm --privileged docker.io/tonistiigi/binfmt --install arm64
# build arm64 into a manifest, then push to a registry
podman build --platform=linux/arm64 --manifest it2-ct2:latest .
podman manifest push --all it2-ct2:latest docker://<registry>/it2-ct2:latest
```
- A cross-arch image can't run on the x86 box; **push to a registry, pull on the Pi.**
- Zero-emulation alternative: **build natively on the Pi** (viable — nothing compiles).

### Model: bake in vs mount
**Bake it in** for a single-purpose Pi appliance (immutable, one-command deploy;
`COPY` the model in a late layer for cache reuse). Use a mounted volume
(`-v /srv/models:/models:ro`) only if you swap models often.
**Image size (ESTIMATE):** ~250–300 MB without model / **~450–500 MB with int8 model baked**.

---

## 8. Phase 5 — Run on the Pi (rootless Podman + systemd)

```bash
podman pull <registry>/it2-ct2:latest
podman run -d --name it2 --restart unless-stopped \
  --cpus 2 --memory 1500m -p 8080:8080 <registry>/it2-ct2:latest
```
- **Long-lived HTTP service, not one-shot CLI** — load the model **once** at startup,
  keep it warm (a CLI pays the ~200 MB load every request).
- **Single uvicorn worker.** On 2 GB each extra worker duplicates the model → OOM.
  Get parallelism from CT2 threads, not processes.

**Auto-start across reboots — the Podman-native way (Quadlet):** drop
`~/.config/containers/systemd/it2.container`:
```ini
[Container]
Image=<registry>/it2-ct2:latest
PublishPort=8080:8080
Environment=OMP_NUM_THREADS=2
[Service]
Restart=always
[Install]
WantedBy=default.target
```
then `systemctl --user daemon-reload && systemctl --user start it2`. (Enable
`loginctl enable-linger $USER` so the rootless service runs without an active login.)
This is the "much lower RAM than Docker" path you wanted: no daemon, systemd supervises.

---

## 9. Phase 6 — Latency optimization (the "very less latency" part)

### Threading (biggest structural lever on 2 cores)
`ctranslate2.Translator(model, device="cpu", compute_type="int8",
inter_threads=1, intra_threads=2)` + `OMP_NUM_THREADS=2`.
Rule: `inter_threads * intra_threads ≤ physical cores`. For lowest single-sentence
latency, put both cores on one translation → **1 × 2**. (Flip to `2 × 1` only if you
optimize for throughput over many concurrent requests — that raises per-sentence latency.)

### Decoding settings (per request)
- **`beam_size=1` (greedy)** — the single biggest latency win vs default beam=5.
  Measure BLEU to confirm the quality drop is acceptable.
- **`return_scores=False`** under greedy (skips a per-step softmax).
- **Cap `max_decoding_length`** (~1.5–2× expected output) so runaways can't blow latency.
- **`max_input_length`**; split paragraphs into sentences (latency ∝ tokens generated).
- **`batch=1`** for single-sentence latency; batch only when many arrive
  (`max_batch_size`, `batch_type="tokens"`, iterable API sorts by length).
- Keep the model **warm/resident**; disable sampling/alternatives/scoring you don't use.

### Expectations (ESTIMATE — measure on device)
| Sentence | Latency (int8, greedy, 2 cores @1.5 GHz) |
|---|---|
| short (~10 tok) | ~0.2–0.4 s |
| typical (~20–30 tok) | ~0.4–1.5 s |
| long (50+ tok) | ~2–3 s |
- Single-stream decode is **memory-bandwidth bound** (~90–100 MB int8 weights streamed
  per token). Beam search multiplies this. Pi thermal throttling + SD-card cold start matter.
- **RAM:** ~250–500 MB resident at batch=1 — comfortable in 2 GB. Podman daemonless ⇒ no extra idle RAM.

### Ordered ROI of the optimizations you asked for
1. 🟢 **int8 quantization** (CT2 `--quantization int8`) — ~4× smaller, ~3–4× faster, one flag.
2. 🟢 **greedy + capped decode length** — biggest autoregressive-latency lever.
3. 🟢 **already using dist-200M** (not the 1B) — the largest win, already taken.
4. 🟡 **QLoRA** — enables *fine-tuning* cheaply; **neutral** on inference latency.
5. 🔴 **structured pruning** — last resort, ~10–30% more, high effort + BLEU cost + custom-arch fragility.
6. ⚪ **unstructured/"dynamic" pruning** — **skip**: no CPU speedup on this stack.

---

## 10. Consolidated risks & gotchas

1. **32-bit OS = no wheels** (CT2 *and* ONNX Runtime). Pi must be **64-bit** (`uname -m`→`aarch64`).
2. **HF→CT2 conversion fails on custom arch** → use the **fairseq→CT2** route (official).
3. **optimum-cli ONNX export fails on custom arch** → Python `main_export` + custom OnnxConfig (§6b).
4. **QLoRA ≠ deployment size** — merge to fp16, then int8 separately; never ship the 4-bit object.
5. **Merge into fp16, not 4-bit** — dequantize base first or quality drops.
6. **Unstructured pruning gives no CPU speedup** — structured only, and heal after.
7. **Structured pruning must stay converter-legal** (uniform shapes) or it won't round-trip to CT2.
8. **Alpine/musl base = source build** — use `python:3.11-slim`.
9. **Cross-arch image must be pushed**, not run on x86; pin deps, never `--no-binary`.
10. **Single worker on 2 GB**; parallelism via CT2 threads.
11. **`save_pretrained` tokenizer bug** — reuse `tokenizer_utils` fallback.
12. **Re-validate BLEU/chrF** after *every* lossy step (greedy, int8, pruning).

---

## 11. Deliverables to build

| File | Purpose | Status |
|---|---|---|
| `finetune_qlora.py` | QLoRA (4-bit) wrapper over AI4Bharat `train_lora` for our data | to build |
| `merge_lora.py` | dequant→fp16, `merge_and_unload()` → single model dir | to build |
| `prune_structured.py` *(optional)* | torch-pruning decoder-depth/encoder-width, converter-legal shapes | to build |
| `convert_ct2.py` | HF→CT2 attempt + fairseq fallback | ✅ exists |
| `export_onnx.py` *(alternative)* | custom-OnnxConfig `main_export` + `quantize_dynamic` | to build |
| `app.py` | FastAPI: warm CT2 (or ORT) model, `/translate`, `/health` | to build |
| `requirements-docker.txt` | pinned arm64 deps | to build |
| `Containerfile` | arm64 image (Podman/Docker-compatible) | to build |
| `build_arm64.sh` | `podman build --platform --manifest` + push | to build |
| `it2.container` | systemd **Quadlet** unit for auto-start on the Pi | to build |
| `translate_ct2.py` | local CLI sanity check vs PyTorch `translate.py` | ✅ exists |

---

## 12. Sequencing (milestones)

- **M0 — Feasibility (½ day):** confirm 64-bit Pi OS; confirm **fairseq→CT2** round-trips
  on the *stock* dist-200M; A/B **CT2-int8 vs ORT-int8** latency on the Pi with the stock model.
- **M1 — QLoRA fine-tune (1–2 days):** data prep, 4-bit LoRA train, merge→fp16, BLEU check.
- **M2 — int8 artifact (½ day):** CT2 `--quantization int8` (fairseq route); validate output vs PyTorch.
- **M3 — Containerize with Podman (1 day):** `app.py`, Containerfile, `podman build --manifest`, push.
- **M4 — Deploy & tune (1 day):** rootless Podman + Quadlet on the Pi, set threads/greedy/limits, **measure real latency**.
- **M5 — (optional) structured pruning:** only if M4 latency/size still insufficient; prune decoder depth + heal + re-convert + re-benchmark.

---

### Verification confidence note
The ONNX / pruning / quantization findings here come from a focused research pass with
primary-source citations (optimum & ONNX Runtime docs, CTranslate2 docs, bitsandbytes/
PEFT, pruning literature, AI4Bharat repo). The two **adversarial-verification** agents
that would have double-checked the ONNX-export and pruning claims **did not complete
(session limit)**, so treat those two as single-pass research-grade. The load-bearing
practical claims (CT2 fairseq route works; optimum CLI fails but `main_export` works;
unstructured pruning gives no dense CPU speedup; QLoRA is 4-bit/training-only) are each
backed by cited sources. The one number that matters most — on-device latency — must be
measured on the actual Pi regardless.
