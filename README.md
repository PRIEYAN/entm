# IndicTrans2 — local English→Indic translation

Runs [`ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
fully locally, plus a full pipeline to **fine-tune (QLoRA), compress to int8, and
deploy to a Raspberry-Pi-class ARM device**. Works on CPU; uses GPU automatically
if CUDA is available.

## Project layout

Everything lives in the `it2edge` package, grouped by pipeline stage. Run scripts
as modules **from the project root** (`python -m it2edge.<stage>.<script>`), or
`pip install -e .` to get the `it2-*` console commands.

```
it2edge/                package root
  paths.py              project-relative paths + model id (single source of truth)
  tokenizer_utils.py    robust IndicTrans2 tokenizer loader
  download_model.py     one-time model snapshot into ./model_cache
  train/                finetune_qlora · merge_lora · prune_structured
  convert/              convert_ct2 (primary) · export_onnx (alternative)
  serve/                app (FastAPI) · translate (torch) · translate_ct2 (CT2)
deploy/                 Containerfile · build_arm64.sh · it2.container
requirements/           dev.txt · pi.txt · docker.txt
pyproject.toml          package metadata + console-script entry points
plan.md                 the full plan and rationale
```

## Setup (run once, needs internet)

```bash
python -m venv venv
# Windows PowerShell:  venv\Scripts\Activate.ps1
# Windows cmd:         venv\Scripts\activate.bat
# macOS / Linux:       source venv/bin/activate

pip install -r requirements/dev.txt
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

python -m it2edge.download_model   # caches weights into ./model_cache
```

`IndicTransToolkit` is installed separately because it is not on PyPI — it
provides the `IndicProcessor` that tags input with language tokens and handles
script conversion. Without it the model does **not** translate correctly.

## Run (offline after the download step)

```bash
python -m it2edge.serve.translate "Hello, how are you?"            # -> Hindi by default
python -m it2edge.serve.translate --tgt tam_Taml "This is a test." # -> Tamil
python -m it2edge.serve.translate                                  # built-in demo
```

### Already hit `got multiple values for keyword argument 'src_vocab_file'`?

An earlier version of the downloader used `save_pretrained`, which writes
`src_vocab_file` into `tokenizer_config.json`; on reload the remote tokenizer
then receives it both positionally and by keyword and raises that TypeError.
`it2edge.download_model` snapshots the raw repo instead, which avoids it.
To recover, clear the poisoned caches and re-download:

```bash
rm -rf model_cache
rm -rf ~/.cache/huggingface/modules/transformers_modules/indictrans2-en-indic-dist-200M
python -m it2edge.download_model
python -m it2edge.serve.translate "hello"
```

## Target language codes

Source is always `eng_Latn`. Common targets:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `hin_Deva` | Hindi | `kan_Knda` | Kannada |
| `tam_Taml` | Tamil | `mal_Mlym` | Malayalam |
| `tel_Telu` | Telugu | `pan_Guru` | Punjabi |
| `ben_Beng` | Bengali | `guj_Gujr` | Gujarati |
| `mar_Deva` | Marathi | `urd_Arab` | Urdu |

Full list of the 22 supported languages is on the model card.

## Running on a Raspberry Pi (CTranslate2 int8)

The torch path needs PyTorch and loads the model in fp32 (~800 MB), which is too
heavy for a Pi 3 (1 GB RAM). For low-memory / ARM devices there is a
**CTranslate2 int8** path (~200 MB, no PyTorch at runtime).

**Requires a 64-bit Raspberry Pi OS** — `ctranslate2` ships aarch64 wheels but
**no 32-bit/armv7 wheels**, and PyTorch has dropped armv7 too. On 32-bit Pi OS
neither this path nor the default one will `pip install` without building from
source. Reflash to 64-bit first.

**1. Convert on your dev machine** (x86-64 — this is where the heavy tooling lives):

```bash
pip install ctranslate2 transformers sentencepiece protobuf
python -m it2edge.download_model      # if you haven't already
python -m it2edge.convert.convert_ct2 # writes ./model_cache_ct2/ (int8)
```

**2. Copy to the Pi** — both the CT2 weights and the tokenizer:

```
model_cache_ct2/                              # int8 CT2 model
model_cache/indictrans2-en-indic-dist-200M/   # tokenizer files
```

**3. Run on the Pi** (64-bit OS):

```bash
pip install -r requirements/pi.txt
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

python -m it2edge.serve.translate_ct2 "Hello, how are you?"        # greedy, fastest
python -m it2edge.serve.translate_ct2 --beams 5 "Higher quality, slower."
```

Greedy decoding (`--beams 1`, the default) is several times faster than beam
search on a Pi 3. If conversion fails on IndicTrans2's custom architecture, it
prints the official AI4Bharat fairseq→CT2 fallback route.

## Fine-tune → compress → containerize → deploy

The full pipeline in [`plan.md`](plan.md): QLoRA fine-tune on a GPU dev box,
merge to fp16, quantize to int8 **last**, ship an arm64 Podman image to a Pi.

**Correct ordering** (int8 goes last — pruning/healing need float weights):

```
QLoRA 4-bit fine-tune ─▶ merge to fp16 ─▶ (optional) structured prune + heal ─▶ int8 ─▶ arm64 image ─▶ Pi
```

**On the dev machine (x86-64 + GPU):**

```bash
# 0. Deps for the whole dev pipeline (torch + QLoRA + convert).
pip install -r requirements/dev.txt
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

# 1. QLoRA fine-tune (4-bit base + LoRA adapters). Needs a 12–16 GB GPU.
python -m it2edge.train.finetune_qlora --data_dir en-indic-exp --output_dir lora_adapters

# 2. Merge adapters into one fp16 model (dequantize base first, then merge).
python -m it2edge.train.merge_lora --adapter_dir lora_adapters

# 3. (OPTIONAL) structured pruning, then HEAL (re-fine-tune) — only if still too slow.
python -m it2edge.train.prune_structured --drop_decoder_layers 2

# 4. int8 conversion — LAST. Picks up model_cache_merged/ automatically.
python -m it2edge.convert.convert_ct2          # -> model_cache_ct2/

# 5. Build the arm64 image with Podman and push to a registry.
REGISTRY=docker.io/youruser deploy/build_arm64.sh
```

**On the Raspberry Pi (64-bit / aarch64):**

```bash
podman pull <registry>/it2-ct2:latest
# Auto-start via the systemd Quadlet unit (rootless, no daemon):
cp deploy/it2.container ~/.config/containers/systemd/it2.container  # edit <registry> first
loginctl enable-linger $USER
systemctl --user daemon-reload && systemctl --user start it2

# Translate over HTTP (model stays warm; single worker; CT2 threads = 2):
curl -s localhost:8080/translate \
  -H 'content-type: application/json' \
  -d '{"text":"Hello, how are you?","tgt_lang":"hin_Deva"}'
```

`it2edge.serve.app` serves `/translate` and `/health` with the model loaded once
at startup (greedy by default, both Pi cores on one translation). ONNX Runtime is
a documented alternative runtime (`it2edge.convert.export_onnx`) — see `plan.md`
§3/§6b for why CTranslate2 is the recommended path.

## Files

**`it2edge/` (the package)**
- `paths.py` — project-relative paths + model id; every script imports from here
- `tokenizer_utils.py` — robust tokenizer loader (works around the `src_vocab_file` reload bug)
- `download_model.py` — one-time model snapshot into `model_cache/`
- `train/finetune_qlora.py` — QLoRA (4-bit) fine-tune on a parallel corpus (dev GPU)
- `train/merge_lora.py` — dequantize base → fp16 and merge the LoRA adapters into one model
- `train/prune_structured.py` — *optional* structured pruning (decoder depth / encoder width)
- `convert/convert_ct2.py` — convert the HF model (stock or merged) to a CTranslate2 int8 package
- `convert/export_onnx.py` — *alternative* runtime: ONNX export (custom OnnxConfig) + int8
- `serve/app.py` — FastAPI service: warm CT2 model, `POST /translate`, `GET /health`
- `serve/translate.py` — offline PyTorch inference (CLI + importable `translate()`)
- `serve/translate_ct2.py` — CTranslate2 int8 inference, no PyTorch (run on the Pi)

**`deploy/`**
- `Containerfile` — arm64 image (Podman/Docker-compatible)
- `build_arm64.sh` — cross-build the arm64 image with Podman and push to a registry
- `it2.container` — systemd Quadlet unit to auto-start the service on the Pi

**`requirements/`**
- `dev.txt` — dev-box deps: torch inference + the full fine-tune/convert pipeline
- `pi.txt` — slim CTranslate2 deps for the Pi (no torch)
- `docker.txt` — pinned aarch64 deps baked into the serving image

**root**
- `pyproject.toml` — package metadata + `it2-*` console-script entry points
- `plan.md` — the full fine-tune → compress → deploy plan and rationale
