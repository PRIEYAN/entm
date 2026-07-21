# IndicTrans2 en→Indic — Full Model & Training Details

Everything done to `ai4bharat/indictrans2-en-indic-dist-200M` in this project:
fine-tuning, merging, quantization, deployment — with the **actual numbers read
off the artifacts on disk**, not estimates.

---

## 0. One-paragraph summary

I took AI4Bharat's distilled 200M English→Indic translation model and **QLoRA
fine-tuned** it on **my own English↔Hindi dataset (128,861 training pairs)**. The
base was frozen in **4-bit NF4** and I trained only small **LoRA adapters
(1,769,472 params ≈ 0.64% of the model)** for **1 epoch / 4027 steps**, taking eval
loss from the start down to **2.369**. I then **merged** the adapters back into a
full **fp16** model and **quantized it to int8 with CTranslate2**, shrinking it from
~1.1 GB → **204 MB** so it runs fully offline on a Raspberry Pi. The result is an
English→Hindi speech translator that runs on a ~₹5,000 Pi 3 B+ with no internet.

---

## 1. The base model (before I touched it)

**`ai4bharat/indictrans2-en-indic-dist-200M`** — a *distilled* IndicTrans2 model.
It is **already** a compressed model (distilled down from AI4Bharat's larger 1B
IndicTrans2), which matters later when talking about further shrinking.

### Architecture (from [config.json](model_cache/indictrans2-en-indic-dist-200M/config.json))

| Property | Value |
|---|---|
| Architecture | `IndicTransForConditionalGeneration` (custom HF arch, `model_type: IndicTrans`) |
| Type | Encoder–decoder (seq2seq) transformer, BART/M2M100-style |
| Encoder layers | **18** |
| Decoder layers | **18** |
| Embedding dim (enc & dec) | **512** |
| FFN dim (enc & dec) | **2048** |
| Attention heads (enc & dec) | **8** |
| Source vocab (English) | **32,322** |
| Target vocab (Indic) | **122,672** (separate, larger — 22 languages) |
| Max source / target positions | 256 / 256 |
| Activation | GELU, pre-norm |
| Weight tying | decoder input embed **tied** to output projection |

### ⚠️ Actual parameter count — the real "200M"

The name says "200M" but the **true total is larger**, because the two vocabularies
make the embedding tables huge. Counted directly from the weights on disk:

| | Params | Share |
|---|---|---|
| **Total parameters** | **274,584,576** (~274.6M) | 100% |
| Embedding + output projection | 142,164,992 (~142M) | **51.8%** |
| Transformer body (attn + FFN, non-embedding) | 132,419,584 (~132M) | 48.2% |

**Key takeaway:** more than **half the model is just the embedding/vocab tables**
(especially the 122,672-row Indic target vocab). The "200M" figure refers to the
*non-embedding* compute; the file-on-disk parameter count is ~275M. This is also why
you *can't* shrink it to something like 5M without changing the tokenizer — see §8.

- Base weights on disk: **1.1 GB** (`model.safetensors` / `pytorch_model.bin`, fp32)

---

## 2. My dataset (the fine-tune data)

The fine-tune data is **my own dataset**, not AI4Bharat's.

- **Source file:** [`Dataset_English_Hindi.csv`](Dataset_English_Hindi.csv) — a
  2-column `English,Hindi` CSV, **130,476 rows** (~39 MB).
  - Example rows: `Help!,बचाओ!` · `Jump.,उछलो.`
- **Prepared** by [prepare_data.py](it2edge/train/prepare_data.py): deduplicated of
  blank rows, deterministically shuffled (seed 42), split 99% / 1% into train/dev,
  and written as raw **line-aligned parallel text** (the format the trainer expects):

  ```
  en-indic-exp/
    train/eng_Latn-hin_Deva/train.eng_Latn   train.hin_Deva
    dev/eng_Latn-hin_Deva/dev.eng_Latn       dev.hin_Deva
  ```

- **Final corpus sizes** (counted on disk):

  | Split | Pairs |
  |---|---|
  | **Train** | **128,861** |
  | **Dev (held-out)** | **1,301** |

- Text was fed **raw** — `IndicProcessor` (from IndicTransToolkit) applies the exact
  same language-tagging, script-unification and entity-masking at train time that the
  model uses at inference, so training and serving stay consistent.
- There is also a tiny **`en-indic-exp-mini/`** (300 train / 50 dev) used only as a
  smoke-test corpus.

---

## 3. Fine-tuning: QLoRA (4-bit) — what actually ran

Script: [finetune_qlora.py](it2edge/train/finetune_qlora.py). **QLoRA = quantized
base + LoRA adapters.** The base is frozen in 4-bit *only so it fits in low VRAM*;
the actual trained weights are the tiny LoRA adapters.

### Quantization used *during training* (bitsandbytes NF4)

```
load_in_4bit           = True
bnb_4bit_quant_type    = "nf4"          # 4-bit NormalFloat
bnb_4bit_compute_dtype = bfloat16
bnb_4bit_use_double_quant = True        # double quantization
```

This is a **training trick, not the shipped model.** The base is 4-bit only while
training; the deployed model is re-dequantized then re-quantized to int8 (§5–6).

### LoRA config (from [adapter_config.json](lora_adapters/adapter_config.json))

| Setting | Value |
|---|---|
| Rank `r` | **16** |
| `lora_alpha` | **32** |
| `lora_dropout` | **0.1** |
| Target modules | **`q_proj`, `k_proj`** (query & key projections only) |
| Task type | `SEQ_2_SEQ_LM` |
| PEFT version | 0.19.1 |

### 🔑 Trainable parameters after LoRA (the "params after fine-tune")

Counted directly from the adapter file:

| | Value |
|---|---|
| **Trainable params (LoRA adapters)** | **1,769,472** (~1.77M) |
| Frozen base params | 274,584,576 |
| **Trainable share of the model** | **≈ 0.644 %** |
| Adapter file size | **6.8 MB** ([adapter_model.safetensors](lora_adapters/adapter_model.safetensors)) |

So the fine-tune only ever updated **~1.77M weights** — everything else stayed
frozen. This is the whole point of QLoRA: full quality-relevant adaptation at a
fraction of the memory/compute of full fine-tuning.

### Hyperparameters that actually ran

Read from the saved `training_args.bin` in the checkpoint (these reflect the **real
run**, which differs from the script's argparse defaults):

| Hyperparameter | Value used |
|---|---|
| Epochs | **1.0** |
| Per-device batch size | **4** |
| Gradient accumulation | **8** → **effective batch 32** |
| Learning rate | **2e-4** |
| Warmup steps | 4000 |
| LR schedule | **inverse_sqrt** |
| Optimizer | **adamw_torch** |
| Weight decay | 0.01 |
| Precision | **bf16** |
| Max sequence length | 256 |

### Training results (from [trainer_state.json](lora_adapters/checkpoint-4027/trainer_state.json))

| Metric | Value |
|---|---|
| Total steps | **4027** (= 1 epoch over 128,861 pairs at effective batch 32) |
| Best checkpoint | `checkpoint-4027` |
| **Best eval loss** | **2.3688** |
| Train loss trajectory | **~3.23 → ~2.47** (start → end of epoch) |
| Eval throughput | ~57.9 samples/s |
| Total FLOPs | ~4.09e15 |

The loss curve is clean and monotonically decreasing — the adapters clearly learned
from the dataset. Checkpoint `checkpoint-4027` (with optimizer/scheduler/RNG state)
is the best model and what got merged.

> Note: QLoRA is a **GPU** technique — this run needs a 12–16 GB GPU (T4 / RTX 3060
> class). It only saves the adapters, never the base.

---

## 4. Pipeline ordering (why int8 comes *last*)

```
QLoRA 4-bit fine-tune ─▶ merge to fp16 ─▶ (optional prune + heal) ─▶ int8 ─▶ arm64 image ─▶ Pi
```

The rule enforced throughout the repo: **quantize to int8 LAST.** Pruning and
adapter-merging need real float weights; if you quantize early you lose the ability
to heal/merge cleanly. "8-bit + QLoRA" would be a contradiction — QLoRA is 4-bit,
and that 4-bit object is thrown away after merging.

---

## 5. Merge — back to a full fp16 model

Script: [merge_lora.py](it2edge/train/merge_lora.py) → output
[`model_cache_merged/`](model_cache_merged/).

- Reloads the **base in fp16 (dequantized, NOT 4-bit)** and merges the LoRA adapters
  into it with `merge_and_unload()`. Merging into a still-4-bit base would degrade
  quality, so it dequantizes first.
- Re-asserts the tied decoder-embed ↔ output-projection weight tie, and saves as
  `.bin` (not safetensors) to preserve the aliased tied tensor.
- **Result:** one standalone fp16 model, no adapters, no LoRA.

| Artifact | Size |
|---|---|
| Merged fp16 model (`model_cache_merged/pytorch_model.bin`) | **405 MB** |

Parameter count is **unchanged (~274.6M)** — LoRA merging folds the adapter deltas
into the existing weights; it doesn't add parameters. Only the *values* changed
(fine-tuned), the *shape/count* is identical to the base.

---

## 6. Quantization — int8 with CTranslate2 (the shipped model)

Script: [convert_ct2.py](it2edge/convert/convert_ct2.py) → output
[`model_cache_ct2/`](model_cache_ct2/).

- Converts the merged fp16 HF model into a **CTranslate2 int8** package
  (`quantization="int8"`). At runtime this needs **no PyTorch** — ideal for a Pi.
- A **custom loader** was required: IndicTrans2 has **two separate vocabularies**
  (32,322 source / 122,672 target), which CT2's stock M2M100/BART converter can't
  handle. The project registers an `IndicTransLoader` (subclassing `BartLoader`) that
  wires the dual embeddings, the sinusoidal position table (`.weights`/`.offset`),
  and both vocabularies correctly. Verified vocab counts in the CT2 model: source
  **32,322**, target **122,672** — matching the base exactly.

| Stage | Artifact | Size |
|---|---|---|
| Base (fp32) | model.safetensors | 1.1 GB |
| Merged (fp16) | pytorch_model.bin | 405 MB |
| **Quantized (int8)** | **model_cache_ct2/model.bin** | **204 MB** |

**≈ 5.4× smaller than the fp32 base**, ~2× smaller than the fp16 merge. int8 is
lossy, so a few sentences should be A/B-checked before shipping. Parameter *count*
is again unchanged (~274.6M) — quantization changes the **bytes per weight**
(fp32→int8, 4×), not the number of weights.

---

## 7. Optional structured pruning (available, not required)

Script: [prune_structured.py](it2edge/train/prune_structured.py). Provided but
**optional** — only if int8 + greedy is still too slow.

- Does **structured** pruning (drops whole decoder layers / FFN neurons via
  torch-pruning DepGraph), which actually shrinks the dense CPU math — unlike
  unstructured pruning, which only zeros weights and gives no CPU speedup.
- Highest-yield move: **drop decoder layers** (decoder depth dominates
  autoregressive latency), narrow the encoder only modestly.
- **Healing is mandatory:** pruning drops BLEU immediately, so you must re-fine-tune
  on the parallel corpus afterward, re-check BLEU/chrF, then re-convert to int8.
- **Honest verdict** (also in [docs/reduceParams.md](docs/reduceParams.md)): on an
  *already-distilled* 200M model, expect only **~10–30%** further reduction before
  quality collapses. There is no evidence the pruned path was actually used to
  produce the shipped model — the deployed artifact is the merged+int8 model.

---

## 8. Can it go to 5M params? (from [docs/reduceParams.md](docs/reduceParams.md))

Short answer: **no, not by "reduction."**

- You cannot delete 97.5% of a trained network and keep it working. Structured
  pruning caps at ~10–30% here.
- **>51% of the params are embeddings** you can't touch without breaking the
  tokenizer / 22-language vocab.
- A real 5M model would be a **new, from-scratch distilled model** (needs a GPU, a
  large corpus, days of work) and would translate noticeably worse. Realistic
  smaller target is **~50M**, not 5M.
- A 40× smaller model does **not** give 40× less latency — decoding is
  autoregressive (one token at a time); a smaller model cuts per-step cost, not the
  number of steps (~3–6× at best).

---

## 9. Serving & deployment

- **On the Pi (int8, no torch):** [translate_ct2.py](it2edge/serve/translate_ct2.py)
  and the FastAPI service [app.py](it2edge/serve/app.py) — model loaded once, warm,
  `POST /translate`, greedy decoding by default (beam=1 is several× faster on a Pi 3),
  both cores pinned (CT2 `intra_threads=2`, `OMP_NUM_THREADS`).
- **On the dev box (torch, fp32):** [translate.py](it2edge/serve/translate.py).
- **Container:** arm64 Podman image ([deploy/](deploy/)) with a systemd Quadlet unit
  (`it2.container`) to auto-start on the Pi.
- **Full voice pipeline:** STT → this translator → TTS, driven from
  [laptop_mic.py](laptop_mic.py) / [speak.py](it2edge/serve/speak.py).

### Measured performance (from [des.txt](des.txt))

| Board | Latency (full STT+translate+TTS) |
|---|---|
| **Pi 3 B+ (1 GB)** — current prototype | **35–40 s** |
| Pi 4 (4 GB) | ~10–15 s (~3×) |
| Pi 5 (4–8 GB) | ~3–6 s (~6–8×) |

The work is CPU-bound; the model already fits in RAM and is fully optimized (int8 +
greedy). Speed now scales with the board, not the model.

---

## 10. Artifact map (what each directory is)

| Path | What it is | Size |
|---|---|---|
| [model_cache/indictrans2-en-indic-dist-200M/](model_cache/) | Original base (fp32) | 1.1 GB |
| [lora_adapters/](lora_adapters/) | Trained LoRA adapters + best checkpoint | 6.8 MB adapter |
| [model_cache_merged/](model_cache_merged/) | Base + adapters merged (fp16) | 405 MB |
| [model_cache_ct2/](model_cache_ct2/) | **Shipped** int8 CTranslate2 model | 204 MB |
| [en-indic-exp/](en-indic-exp/) | My prepared parallel corpus (128,861 / 1,301) | — |
| [Dataset_English_Hindi.csv](Dataset_English_Hindi.csv) | My raw English↔Hindi dataset (130,476 rows) | 39 MB |

---

## 11. The numbers you asked for, in one place

| Question | Answer |
|---|---|
| Base model | `ai4bharat/indictrans2-en-indic-dist-200M` (distilled seq2seq) |
| **Actual base parameter count** | **274,584,576 (~274.6M)** — "200M" = non-embedding compute |
| Embeddings share | 142M / 51.8% of the model |
| Architecture | 18 enc + 18 dec layers, dim 512, FFN 2048, 8 heads |
| Vocab | 32,322 source (En) / 122,672 target (Indic) |
| Fine-tune method | **QLoRA** (4-bit NF4 frozen base + LoRA adapters) |
| **Trainable params after fine-tune** | **1,769,472 (~1.77M) = 0.644% of the model** |
| LoRA config | r=16, α=32, dropout 0.1, targets q_proj+k_proj |
| Dataset | My own English↔Hindi CSV, **128,861 train / 1,301 dev** |
| Training | 1 epoch, 4027 steps, eff. batch 32, lr 2e-4, bf16 |
| **Best eval loss** | **2.3688** |
| Total params after merge | **~274.6M (unchanged — LoRA merged in, not added)** |
| Quantization | fp32 → merged fp16 → **int8 (CTranslate2)** |
| **Final shipped model size** | **204 MB** (from 1.1 GB base ≈ 5.4× smaller) |
| Deployment | Raspberry Pi 3 B+, offline, int8 + greedy, ~35–40 s end-to-end |
