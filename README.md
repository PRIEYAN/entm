# it2edge — Offline English → Hindi translation for edge devices

Local neural machine translation for **English → Hindi**, optimized for a
**1 GB Raspberry Pi**. The recommended on-device model is a compact **~77M
MarianMT** pipeline (download → fine-tune → CTranslate2 int8 → serve). The
larger IndicTrans2 200M model is retained as an **off-device quality reference**
only — it is too slow for acceptable latency on a 1 GB Pi.

| | Compact Marian (recommended) | IndicTrans2 (reference) |
|---|---|---|
| Model | ~77M en→hi MarianMT | ~200M en→Indic distilled |
| On-device use | Yes (Pi runtime) | No (too slow on 1 GB) |
| Languages | English → Hindi | English → 22 Indic languages |
| Toolkit | transformers + ctranslate2 | + IndicTransToolkit |

Package: `it2edge` · Python **≥ 3.11** · Offline after model download.

---

## Prerequisites (developer laptop)

| Requirement | Notes |
|---|---|
| OS | Windows 10/11, macOS, or Linux (x86-64) |
| Python | **3.11.x** (see `.python-version`) — create a fresh venv |
| Disk | ~5–15 GB free (models + checkpoints + CT2 artifacts) |
| Network | Required once for Hugging Face downloads |
| GPU (optional but recommended) | NVIDIA GPU + CUDA driver for fine-tuning. A **4 GB** card (e.g. GTX 1650) works with the compact Marian defaults. |

> **Critical:** `pip install torch` often installs a **CPU-only** wheel. Fine-tuning
> then silently runs on CPU and can take days. Install a **CUDA build** of PyTorch
> if you have an NVIDIA GPU (see [Step 2](#2-install-pytorch-cuda-if-you-have-an-nvidia-gpu)).

---

## Quick start — laptop setup

Run every command from the **repository root** (`d:\entm` / wherever you cloned).

### 1. Clone and create a virtualenv

```bash
git clone <this-repo-url> entm
cd entm

python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows cmd
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate

python -m pip install --upgrade pip
```

### 2. Install PyTorch (CUDA if you have an NVIDIA GPU)

**With NVIDIA GPU** (example: CUDA 12.1 — match your driver; see
[pytorch.org/get-started](https://pytorch.org/get-started/locally/)):

```bash
python -m pip uninstall -y torch
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expect something like `2.5.1+cu121 True NVIDIA GeForce GTX 1650`.

**CPU only** (smoke tests / conversion only — full fine-tune will be very slow):

```bash
python -m pip install torch
```

### 3. Install project dependencies

```bash
python -m pip install -r requirements/dev.txt
python -m pip install -e .
```

Optional (only if you will run the IndicTrans2 reference path):

```bash
python -m pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
```

`IndicTransToolkit` is **not** on PyPI and is **not** required for the compact
Marian path.

### 4. Confirm the install

```bash
python -c "import transformers, ctranslate2, sacrebleu; print('ok')"
python -m it2edge.download_compact_model --help
```

---

## Recommended path — compact Marian (Pi-ready)

Full detail: [`docs/compact-model.md`](docs/compact-model.md).

### A. Download the base model (once, needs internet)

```bash
python -m it2edge.download_compact_model
# optional control model for A/B quality comparison:
python -m it2edge.download_compact_model --control
```

Writes `model_cache_compact/` plus `provenance.json` (pinned Hugging Face revision).

### B. Prepare / use the parallel corpus

The repo expects line-aligned English–Hindi files:

```
en-indic-exp/
  train/eng_Latn-hin_Deva/train.eng_Latn
  train/eng_Latn-hin_Deva/train.hin_Deva
  dev/eng_Latn-hin_Deva/dev.eng_Latn
  dev/eng_Latn-hin_Deva/dev.hin_Deva
```

If you start from `Dataset_English_Hindi.csv`:

```bash
python -m it2edge.train.prepare_data \
  --csv Dataset_English_Hindi.csv \
  --tgt_lang hin_Deva \
  --out_dir en-indic-exp
```

### C. Fine-tune (GPU recommended)

Smoke test first (~2k rows):

```bash
python -m it2edge.train.finetune_compact_marian \
  --data_dir en-indic-exp \
  --max_train_samples 2000
```

Full corpus (~129k train / ~1.3k validation):

```bash
python -m it2edge.train.finetune_compact_marian --data_dir en-indic-exp
```

Defaults are tuned for **4 GB VRAM**: fp16, batch size 1, gradient accumulation 16,
gradient checkpointing, sequence caps 96/128, early stopping on validation chrF++.

Output: `model_cache_compact_ft/`

### D. Convert to CTranslate2 int8

```bash
python -m it2edge.convert.convert_compact_ct2
```

Output: `model_cache_compact_ct2/` (weights + tokenizer files in one directory).

### E. Local smoke translate

```bash
python -m it2edge.serve.translate_ct2 --marian "Hello, how are you?"
```

### F. Quality benchmark (dev laptop)

```bash
python -m it2edge.evaluate.benchmark_models quality \
  --model_type marian \
  --ct2_dir model_cache_compact_ct2 \
  --data_dir en-indic-exp \
  --out report_marian_ft.json
```

### G. Serve locally (HTTP)

```bash
# Windows PowerShell
$env:MODEL_TYPE = "marian"
uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1

# macOS / Linux
MODEL_TYPE=marian uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
```

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/translate \
  -H "content-type: application/json" \
  -d "{\"text\":\"Hello, how are you?\",\"tgt_lang\":\"hin_Deva\"}"
```

---

## Raspberry Pi (64-bit) — deploy the compact model

**Requires 64-bit Raspberry Pi OS (aarch64).** `ctranslate2` has no armv7 wheels.

1. On the laptop, complete steps A–D above.
2. Copy `model_cache_compact_ct2/` to the Pi.
3. On the Pi:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements/pi.txt
pip install -e .

python -m it2edge.serve.translate_ct2 --marian "Hello, how are you?"

MODEL_TYPE=marian uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
```

Latency check on the Pi:

```bash
python -m it2edge.evaluate.benchmark_models latency \
  --model_type marian \
  --ct2_dir model_cache_compact_ct2 \
  --data_dir en-indic-exp \
  --intra 4 \
  --out pi_latency_marian.json
```

**Release targets (see `docs/compact-model.md`):** median translate ≤ 5 s, p95 ≤ 8 s,
peak RSS ≤ 700 MB, quality gates vs the off-device IndicTrans2 reference.

---

## IndicTrans2 path (quality reference / multi-language)

Use this on a **dev laptop** for quality comparison or multi-Indic experiments.
Do **not** treat it as the 1 GB Pi runtime.

```bash
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
python -m it2edge.download_model
python -m it2edge.serve.translate "Hello, how are you?"          # torch
python -m it2edge.convert.convert_ct2                            # -> model_cache_ct2/
python -m it2edge.serve.translate_ct2 "Hello, how are you?"      # CT2
```

QLoRA fine-tune of IndicTrans2 needs roughly a **12–16 GB** GPU — see
[`docs/plan.md`](docs/plan.md) and [`docs/start.md`](docs/start.md).

> **transformers pin for IndicTrans2:** that model’s remote tokenizer breaks on
> some newer transformers releases. Prefer `transformers>=4.51,<4.53` when using
> IndicTrans2. The compact Marian path is more tolerant of newer transformers;
> keep one venv dedicated to each path if you hit version conflicts.

### Language codes (IndicTrans2)

Source is always `eng_Latn`. Common targets: `hin_Deva`, `tam_Taml`, `ben_Beng`,
`tel_Telu`, `mar_Deva`, `guj_Gujr`, `kan_Knda`, `mal_Mlym`, `pan_Guru`, `urd_Arab`.

---

## Project layout

```
it2edge/
  paths.py                     # paths + model IDs (single source of truth)
  download_model.py            # IndicTrans2 HF snapshot
  download_compact_model.py    # Marian base (revision-pinned)
  train/                       # QLoRA, Marian fine-tune, prepare_data, prune
  convert/                     # CT2 / ONNX converters
  serve/                       # FastAPI app, CLI translate, Marian CT2 helper
  evaluate/                    # quality + latency benchmarks
deploy/                        # Containerfile, arm64 build, Quadlet
requirements/                  # dev.txt · pi.txt · docker.txt · laptop.txt
docs/                          # plan · start · compact-model · STT-TTS · reduceParams
en-indic-exp/                  # parallel English–Hindi corpus
```

### Console commands (after `pip install -e .`)

| Command | Purpose |
|---|---|
| `it2-download-compact` | Download Marian base |
| `it2-finetune-compact` | Fine-tune Marian on project corpus |
| `it2-convert-compact-ct2` | Marian → CT2 int8 |
| `it2-benchmark` | Quality / latency reports |
| `it2-download` | Download IndicTrans2 |
| `it2-translate` / `it2-translate-ct2` | CLI translation |
| `it2-finetune` / `it2-merge` / `it2-convert-ct2` | IndicTrans2 train/convert |

Equivalent: `python -m it2edge.<module>`.

### Artifacts (git-ignored — regenerate locally)

| Directory | Contents |
|---|---|
| `model_cache_compact/` | Downloaded Marian base |
| `model_cache_compact_ft/` | Fine-tuned Marian checkpoint |
| `model_cache_compact_ct2/` | Deployable CT2 int8 (Pi) |
| `model_cache/` / `model_cache_ct2/` | IndicTrans2 snapshot / CT2 |
| `lora_adapters/` / `model_cache_merged/` | IndicTrans2 QLoRA artifacts |

---

## Environment variables (serving)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_TYPE` | `indictrans` | `marian` for compact Pi path |
| `CT2_MODEL_DIR` | path by model type | CT2 model directory |
| `TOKENIZER_DIR` | snapshot / CT2 dir | Tokenizer location |
| `CT2_INTER_THREADS` | `1` | Parallel translations |
| `CT2_INTRA_THREADS` | `2` | Threads per translation |
| `DEFAULT_TGT_LANG` | `hin_Deva` | Default target language |
| `DEFAULT_BEAM_SIZE` | `1` | Greedy decode (fastest) |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no CUDA GPU detected` but `nvidia-smi` works | You installed CPU torch. Reinstall the CUDA wheel (Step 2). |
| `ModuleNotFoundError: sacrebleu` | `pip install sacrebleu` (also in `requirements/dev.txt`). |
| `Seq2SeqTrainer ... unexpected keyword argument 'tokenizer'` | Newer transformers renamed this to `processing_class` — already handled in `finetune_compact_marian.py`. Update to latest repo code. |
| CUDA out of memory on 4 GB | Lower `--max_source_length` / `--max_target_length`, keep batch size 1, or use `--max_train_samples` for a smoke run. |
| `got multiple values for keyword argument 'src_vocab_file'` (IndicTrans2) | Clear `model_cache` and HF module cache; re-run `it2edge.download_model`. Prefer transformers `<4.53` for that path. |
| Pi install fails for ctranslate2 | Device is likely 32-bit. Reflash to **64-bit** Raspberry Pi OS. |

---

## Documentation map

| Doc | Contents |
|---|---|
| [`docs/compact-model.md`](docs/compact-model.md) | Marian pipeline, release gates, rollback |
| [`docs/start.md`](docs/start.md) | Ordered IndicTrans2 → deploy walkthrough |
| [`docs/plan.md`](docs/plan.md) | Design rationale (QLoRA, CT2, pruning) |
| [`docs/STT-TTS.md`](docs/STT-TTS.md) | Optional voice path (Whisper + Piper) |
| [`docs/reduceParams.md`](docs/reduceParams.md) | Why shrinking IndicTrans2 in-place is wrong |
| [`des.txt`](des.txt) | Hardware cost and latency notes |

---

## License notes

- Compact Marian base / control: **Apache-2.0** (Helsinki-NLP OPUS lineage;
  Samanantar fine-tune per model card).
- IndicTrans2 / IndicTransToolkit: follow upstream model and toolkit licenses.
- Project code: see repository license (if present) or ask the maintainers.
