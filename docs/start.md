# Start here — end-to-end walkthrough

Take `ai4bharat/indictrans2-en-indic-dist-200M`, fine-tune it, shrink it to int8,
put it in a Podman image, and serve translations on a Raspberry Pi.

This is the hands-on version of [`plan.md`](plan.md) (which explains *why* each
step is done the way it is). Run the commands in order.

> **Two machines.** Everything heavy happens on a **dev machine** (x86-64, ideally
> with a 12–16 GB GPU). Only the final image runs on the **Raspberry Pi**
> (64-bit / aarch64). Each step below is tagged 🖥️ **DEV** or 🍓 **PI**.

> **Run scripts as modules from the project root** — `python -m it2edge.<...>`,
> not `python some_file.py`. (Or `pip install -e .` once to get the `it2-*`
> commands.) Numbers marked **ESTIMATE** are planning figures — measure on the Pi.

---

## Choose your path

| You want… | Do these steps |
|---|---|
| **Just deploy the stock model** (no fine-tuning) | A1 → A2 → **A8** → A9 → A10 → B1 → B2 → B3 |
| **Fine-tune on your own data, then deploy** | A1 → A2 → A4 → A5 → A6 → A8 → A9 → A10 → B1 → B2 → B3 |
| **Squeeze it further** (advanced, usually skip) | …insert A7 between A6 and A8 |

`convert_ct2` (A8) automatically uses your fine-tuned model if `model_cache_merged/`
exists, otherwise the stock snapshot — so the same command works for both paths.

---

# Part A — Dev machine 🖥️

## A1. Set up the environment 🖥️

```bash
python -m venv venv
# Windows PowerShell:  venv\Scripts\Activate.ps1
# macOS / Linux:       source venv/bin/activate

pip install -r requirements/dev.txt
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
```

`IndicTransToolkit` is installed separately (not on PyPI). It provides the
`IndicProcessor` that tags input with language tokens — without it the model does
**not** translate correctly.

**Check:** `python -c "import torch; print(torch.cuda.is_available())"` → `True`
means the GPU is visible (needed for A5). CPU-only is fine for everything except
fine-tuning.

## A2. Download the base model 🖥️

```bash
python -m it2edge.download_model      # caches into ./model_cache (needs internet)
```

**Check:** `model_cache/indictrans2-en-indic-dist-200M/` now holds the weights,
config, remote code, and tokenizer/vocab files. Everything after this is offline.

## A3. Sanity-check the base model (optional) 🖥️

Confirm the stock model translates before you invest in fine-tuning:

```bash
python -m it2edge.serve.translate "Hello, how are you?"            # -> Hindi
python -m it2edge.serve.translate --tgt tam_Taml "This is a test." # -> Tamil
python -m it2edge.serve.translate                                  # multi-lang demo
```

Common target codes (source is always `eng_Latn`): `hin_Deva` Hindi ·
`tam_Taml` Tamil · `tel_Telu` Telugu · `ben_Beng` Bengali · `mar_Deva` Marathi ·
`guj_Gujr` Gujarati · `kan_Knda` Kannada · `mal_Mlym` Malayalam · `pan_Guru`
Punjabi · `urd_Arab` Urdu.

## A4. Prepare your parallel data 🖥️

Line-aligned **raw** text: line *N* of the English file pairs with line *N* of the
target file. Feed raw text — `IndicProcessor` does normalization, script
unification and language tagging internally. **Do not pre-transliterate.**

```
en-indic-exp/
  train/eng_Latn-hin_Deva/train.eng_Latn   train.hin_Deva
  train/eng_Latn-tam_Taml/train.eng_Latn   train.tam_Taml
  dev/eng_Latn-hin_Deva/dev.eng_Latn       dev.hin_Deva
```

- The folder name encodes the pair: `eng_Latn-<tgt>`. Add as many pair folders as
  you have languages; all of them are picked up.
- `train/` is required; `dev/` is optional but recommended (enables eval + early
  best-checkpoint selection).

## A5. Fine-tune with QLoRA 🖥️ (needs GPU)

Freezes the base in 4-bit NF4 and trains small LoRA adapters (~0.8 M params).

```bash
python -m it2edge.train.finetune_qlora --data_dir en-indic-exp --output_dir lora_adapters
```

Useful flags: `--batch_size 8 --grad_accum 4` (defaults; raise batch only if VRAM
allows — the AI4Bharat default of 32 OOMs a 16 GB T4), `--epochs 3`, `--lr 2e-4`.

**Check:** `lora_adapters/` contains adapter files. These are **not** deployable on
their own — that's the next step.

> No GPU? QLoRA is a GPU technique; a CPU run is a smoke test only and is
> impractically slow. Use a free Colab/Kaggle T4, or skip fine-tuning entirely
> (the "stock model" path above).

## A6. Merge adapters → one fp16 model 🖥️

Neither CTranslate2 nor ONNX can read LoRA adapters. Merge them in — dequantizing
the base to fp16 first (merging into a still-4-bit base hurts quality).

```bash
python -m it2edge.train.merge_lora --adapter_dir lora_adapters
```

**Check:** `model_cache_merged/` holds one ~400 MB fp16 model. It's *bigger* and
float again — that's expected. The small int8 artifact comes last (A8).

## A7. (Optional) structured pruning + heal 🖥️ — advanced, usually skip

Only if int8 + greedy (A8 + the decoding settings) still leaves it too slow/big.
Drops whole decoder layers / narrows FFN width (real CPU speedup, unlike
"dynamic"/unstructured pruning which gives none on this stack).

```bash
pip install torch-pruning
python -m it2edge.train.prune_structured --drop_decoder_layers 2 --ffn_ratio 0.25
```

> ⚠️ Pruning **drops BLEU immediately**. You **must** then *heal* — fine-tune the
> pruned model on parallel data again (repeat A5-style training against
> `model_cache_pruned/`) and re-check quality — before converting. For an
> already-distilled 200M this has diminishing returns (~10–30% at a quality cost).
> If you run this, point A8 at the healed pruned model with `--model`.

## A8. Convert to int8 for the Pi (CTranslate2) 🖥️ — do this LAST

```bash
python -m it2edge.convert.convert_ct2        # writes ./model_cache_ct2/
```

- Auto-selects `model_cache_merged/` (fine-tuned) if present, else the stock
  snapshot. Override with `--model <dir>` (e.g. a healed pruned model).
- On ARM, `int8` resolves to `int8_float32` (the right choice); other types just
  fall back with no benefit.

**Check:** `model_cache_ct2/` is ~100–200 MB.

> **If conversion fails** on IndicTrans2's custom architecture, the script prints
> the official AI4Bharat **fairseq → CT2** fallback route. That path is supported
> and is what AI4Bharat themselves ship. ONNX Runtime is a documented alternative
> (`python -m it2edge.convert.export_onnx`) — see `plan.md` §3/§6b, but CT2 is the
> recommended path for this model on ARM.

## A9. Test the int8 model locally 🖥️

```bash
pip install ctranslate2
python -m it2edge.serve.translate_ct2 "Hello, how are you?"        # greedy, fastest
python -m it2edge.serve.translate_ct2 --beams 5 "Higher quality, slower."
```

> **Important sanity check.** Compare a known sentence against the PyTorch path
> (A3). The raw-CT2 token round-trip is the one part not yet verified end-to-end;
> if output is empty, wrong-language, or garbled, see Troubleshooting.

## A10. Build the arm64 Podman image and push 🖥️

A cross-arch image can't run on x86, so you build it, push to a registry, and pull
on the Pi.

```bash
# needs: podman installed; model_cache_ct2/ (A8) + model_cache/ (A2) present
REGISTRY=docker.io/youruser deploy/build_arm64.sh
```

The script registers QEMU arm64 emulation, stages the tokenizer files into
`tokenizer/`, builds `linux/arm64` into a manifest, and pushes to
`$REGISTRY/it2-ct2:latest`. Set `TAG=v1` to version it.

**Image size (ESTIMATE):** ~450–500 MB with the int8 model baked in.

> No registry? Run `deploy/build_arm64.sh` **natively on the Pi** instead — nothing
> compiles (all deps are prebuilt aarch64 wheels), so emulation isn't needed.

---

# Part B — Raspberry Pi 🍓

## B1. Flash a 64-bit OS and install Podman 🍓

```bash
uname -m        # MUST print: aarch64
```

If it prints `armv7l`, you're on 32-bit — **reflash Raspberry Pi OS (64-bit)
first.** There are no 32-bit CTranslate2/ONNX wheels; nothing will install.

```bash
sudo apt update && sudo apt install -y podman
```

## B2. Pull and run with auto-start (systemd Quadlet) 🍓

Rootless, daemonless — systemd supervises the container, no `dockerd` eating RAM.

```bash
podman pull <registry>/it2-ct2:latest

mkdir -p ~/.config/containers/systemd
cp deploy/it2.container ~/.config/containers/systemd/it2.container
# edit that file: replace <registry> with your registry/user

loginctl enable-linger $USER          # run without an active login session
systemctl --user daemon-reload
systemctl --user start it2
systemctl --user status it2           # should be active (running)
```

Logs: `journalctl --user -u it2 -f`.

**Quick one-off alternative** (no auto-start):

```bash
podman run -d --name it2 --restart unless-stopped \
  --cpus 2 --memory 1500m -p 8080:8080 <registry>/it2-ct2:latest
```

The service loads the model **once** at startup and keeps it warm. Give it ~30–60 s
on first boot; `/health` returns 503 until the model is loaded, then `{"status":"ok"}`.

## B3. Use it 🍓

```bash
# health
curl -s localhost:8080/health

# single sentence (default target Hindi, greedy)
curl -s localhost:8080/translate \
  -H 'content-type: application/json' \
  -d '{"text":"Hello, how are you?","tgt_lang":"hin_Deva"}'

# a batch, into Tamil
curl -s localhost:8080/translate \
  -H 'content-type: application/json' \
  -d '{"texts":["Good morning.","Where is the station?"],"tgt_lang":"tam_Taml"}'

# higher quality (slower): beam search
curl -s localhost:8080/translate \
  -H 'content-type: application/json' \
  -d '{"text":"A longer, more careful translation.","tgt_lang":"ben_Beng","beam_size":5}'
```

**Request fields:** `text` *or* `texts` (required), `tgt_lang` (default `hin_Deva`),
`beam_size` (default 1 = greedy, fastest), `max_decoding_length` (default 256).

**Expected latency** (int8, greedy, 2 cores @1.5 GHz — **ESTIMATE**, measure yours):
short ~0.2–0.4 s · typical ~0.4–1.5 s · long (50+ tokens) ~2–3 s. A Pi 4/5 is
materially faster. **Resident RAM:** ~250–500 MB, comfortable in 2 GB.

---

## Latency tuning (on the Pi) 🍓

The image already sets the big levers, tunable via env in `deploy/it2.container`:

- **Threads:** `CT2_INTER_THREADS=1`, `CT2_INTRA_THREADS=2`, `OMP_NUM_THREADS=2` —
  both cores on one translation for lowest single-sentence latency. Rule:
  `inter × intra ≤ physical cores`.
- **Greedy** (`beam_size=1`) is the default and the biggest decode-latency win.
- **One worker only.** On 2 GB, extra workers duplicate the model → OOM. Get
  parallelism from CT2 threads, not processes.

Full ordered ROI of every optimization is in `plan.md` §9.

---

## Troubleshooting

**`got multiple values for keyword argument 'src_vocab_file'`** — a poisoned
tokenizer cache. Clear and re-download:
```bash
rm -rf model_cache
rm -rf ~/.cache/huggingface/modules/transformers_modules/indictrans2-en-indic-dist-200M
python -m it2edge.download_model
```

**CT2 output empty / wrong language / garbled** (A9) — likely target-token
handling in the raw-CT2 round-trip. Check the notes in
[`it2edge/serve/translate_ct2.py`](it2edge/serve/translate_ct2.py): try passing
`target_prefix`, or appending EOS to each source token list. Always A/B against the
PyTorch path (A3) first.

**`ct2-transformers-converter` fails** (A8) — expected on the custom arch; use the
printed fairseq → CT2 fallback (`plan.md` §6a).

**Fine-tune OOM** (A5) — lower `--batch_size` (e.g. 4) and raise `--grad_accum` to
keep the effective batch size.

**`pip install` fails on the Pi** — you're almost certainly on 32-bit. Confirm
`uname -m` is `aarch64` (B1).

**`/health` stuck on 503** — the model is still loading, or `CT2_MODEL_DIR` /
`TOKENIZER_DIR` point at nothing. Check `journalctl --user -u it2 -f`.

**Re-validate quality after every lossy step** (greedy, int8, pruning) — compute
BLEU/chrF on your dev set, not just eyeball a sentence.
