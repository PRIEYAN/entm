# English → Hindi Translation Model — Build Summary

A compact MarianMT model fine-tuned for English→Hindi and quantized to int8
CTranslate2 for CPU/edge inference.

---

## 1. Base model

| Property | Value |
|---|---|
| Base model | [`AI4INDIANS/better-opus-mt-en-hi`](https://huggingface.co/AI4INDIANS/better-opus-mt-en-hi) (OPUS-MT / Marian) |
| Pinned revision | `b513145b43f1` |
| Architecture | `MarianMTModel` (encoder–decoder Transformer) |

### Parameters (measured from the actual weights)

| Metric | Value |
|---|---|
| **Total parameters** | **77.0 M** (77,026,926) |
| Hidden size (`d_model`) | 512 |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Attention heads | 8 (enc) / 8 (dec) |
| FFN dim | 2048 |
| Vocab size | 64,110 |
| Max positions | 512 |

> ~77 M params, but the fp32 safetensors file is **308 MB** (embeddings + weights
> stored at full precision). This is what quantization shrinks.

---

## 2. Fine-tuning

**Command used:**
```bash
python -m it2edge.train.finetune_marian --data_dir en-indic-exp
```

**Dataset:** `en-indic-exp` — ~129k train pairs / ~1.3k validation pairs (eng_Latn → hin_Deva).

### Training configuration (defaults, tuned for 4 GB VRAM)

| Hyperparameter | Value |
|---|---|
| Epochs | 3 (early-stop on validation chrF++) |
| Batch size | 1 |
| Gradient accumulation | 16 (→ effective batch 16) |
| Precision | fp16 |
| Gradient checkpointing | on |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Max source length | 96 tokens |
| Max target length | 128 tokens |

### Hardware & run

| | |
|---|---|
| GPU | RTX 3050 Laptop, **4 GB VRAM** |
| Total steps | 24,162 |
| Wall-clock time | **~4 h 28 min** (16,073 s) |
| Throughput | ~24 samples/s, ~1.5 steps/s |

### Result (validation, epoch 3)

| Metric | Value |
|---|---|
| chrF | **35.61** |
| BLEU | **12.22** |
| eval_loss | 1.428 |

Output: `model_cache_compact_ft/` (fine-tuned HF checkpoint + tokenizer).

---

## 3. Quantization (int8 CTranslate2)

**Command used:**
```bash
rm -rf model_cache_compact_ct2
python -m it2edge.convert.convert_ct2
```

Converts the fine-tuned fp32 Marian model → **CTranslate2 int8** for fast CPU inference.

### Weight reduction (measured)

| Stage | File | Size | vs fp32 |
|---|---|---|---|
| fp32 (base / fine-tuned) | `model.safetensors` | **308 MB** | — |
| **int8 CTranslate2** | `model.bin` | **79 MB** | **≈ 3.9× smaller (−74%)** |

> Weights go from 32-bit floats → 8-bit integers, cutting the model to roughly a
> quarter of its size with negligible quality loss for MT.

Output: `model_cache_compact_ct2/` (deployable int8 package + tokenizer).

**Verified working:**
```
EN: Hello, how are you?
-> हेलो, आप कैसे हैं?
```

---

## 4. Latency

### On this laptop — **measured** (Intel Core i5-12450H, CPU, int8, 2 threads)

Running the CLI (`python -m it2edge.serve.translate_ct2 "..."`) takes **~2.19 s
end-to-end**. But almost all of that is one-time startup, **not** translation:

| Phase | Time | What it is |
|---|---|---|
| Import libraries | ~1611 ms | Python + ctranslate2 + transformers load |
| Load model + tokenizer | ~346 ms | reading the 79 MB int8 model into RAM |
| **Actual translation** | **~37 ms** | the real inference (per sentence) |
| **Total cold CLI run** | **~2000 ms (≈2.19 s)** | measured, one sentence from cold |

**The two numbers that matter:**
- **Cold start (one-shot CLI):** ~**2.19 s** — dominated by Python/library import.
- **Per-sentence translation (model already loaded):** ~**37 ms** (median ~65 ms
  across mixed lengths; long ~18-word sentence ~153 ms).

> ⚠️ Don't confuse the two. The ~2.19 s is what you see running the CLI fresh each
> time. In a **long-running server** (load the model once, then translate many
> sentences), each translation is only ~**37–65 ms** — the 2 s startup happens
> once, not per sentence.

Measured with CTranslate2's default `intra_threads=2` (edge-realistic), **not** all 12 cores.

### On a Raspberry Pi (1 GB RAM) — **estimated**

> ⚠️ These are projections, not measured. Actual numbers depend on the exact Pi
> board, cooling, and thread count — treat as a ballpark.

Reasoning: scale the laptop's **~37 ms per-sentence inference** by how much slower
the Pi's ARM cores are (~8–15× per-core for this int8 workload). The **~2 s cold
start** (Python + library import) will be *much* worse on a Pi — likely 10–30 s —
which is exactly why you want a long-running server, not one-shot CLI calls, on the Pi.

| Board | Est. **per-sentence** inference (model loaded) | Est. cold start (import + load) | Notes |
|---|---|---|---|
| **Pi Model B+ (BCM2835, single-core ~700 MHz–1 GHz, 512 MB–1 GB)** | **~0.3–0.6 s+** | **~15–40 s** | Single old ARM11 core. **RAM-tight** — 79 MB model + CT2 + Python can approach the limit; expect swapping. |
| Pi 3 B+ (quad Cortex-A53, 1 GB) | ~150–350 ms | ~8–15 s | Usable for occasional translation |
| Pi 4 (Cortex-A72, ≥2 GB) | ~60–150 ms | ~4–8 s | Comfortable |

**Bottom line for a Pi B+/1 GB:** load the model **once** in a running process, then
each sentence is a few hundred ms. If you run the CLI fresh each time, the cold start
(seconds to tens of seconds) dominates. Also watch memory — the original B+ is the
tightest target. A Pi 4 (or Pi 3 minimum) is the realistic edge target for snappy use.

---

## 5. Full pipeline (reproduce end-to-end)

```bash
source venv/bin/activate

# 1. download base model (once)
python -m it2edge.download_model

# 2. fine-tune (~4.5 h on RTX 3050 4GB)
python -m it2edge.train.finetune_marian --data_dir en-indic-exp

# 3. quantize to int8 CTranslate2
rm -rf model_cache_compact_ct2
python -m it2edge.convert.convert_ct2

# 4. translate
python -m it2edge.serve.translate_ct2 "Hello, how are you?"
```

### Environment fixes applied during this build
- Upgraded `transformers` → **4.57.6** (ctranslate2 4.8.1 needs ≥4.56 for the `dtype` kwarg).
- Patched `it2edge/convert/convert_ct2.py` to stop copying the HF `config.json`
  over the CT2-generated one (the HF config's `max_length: null` crashed
  `ctranslate2.Translator`).


--Start--
uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1

curl -X POST http://127.0.0.1:8080/translate \
  -H "Content-Type: application/json" \
  -d '{"texts":["Hello","How are you?","Good morning"]}'
