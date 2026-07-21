# Compact English→Hindi model (Pi runtime)

Why this exists: `ai4bharat/indictrans2-en-indic-dist-200M` is too slow on a
**1 GB Raspberry Pi** (~15 s per translation, ~35–40 s voice end-to-end — see
[`des.txt`](../des.txt)). This path replaces the on-device translator with a
compact **~77M English→Hindi MarianMT** model, fine-tuned on the project corpus,
quantized to CTranslate2 int8.

IndicTrans2 is kept **only as an off-device quality reference**. It is not served
or benchmarked on the Pi.

## Model

| | Value |
|---|---|
| Base | [`AI4INDIANS/better-opus-mt-en-hi`](https://huggingface.co/AI4INDIANS/better-opus-mt-en-hi) (Samanantar-fine-tuned OPUS) |
| Control | [`Helsinki-NLP/opus-mt-en-hi`](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi) (official OPUS) |
| Arch | MarianMT, ~77M params, `d_model=512`, 6+6 layers |
| Pinned revisions | see `COMPACT_REVISION` / `CONTROL_REVISION` in [`it2edge/download_compact_model.py`](../it2edge/download_compact_model.py) |
| License | Apache-2.0 |

Published scores are on *different* test sets and are **not** comparable to our
system. The only numbers that decide deployment come from the benchmark harness
run on our own dev set and Pi.

## Pipeline (all dev-machine steps except the Pi benchmark)

```
download_compact_model → finetune_compact_marian → convert_compact_ct2
   → benchmark (quality, DEV) → copy to Pi → benchmark (latency, PI) → deploy
```

### 1. Download the base (revision-pinned)

```bash
python -m it2edge.download_compact_model          # fine-tuned base
python -m it2edge.download_compact_model --control # optional OPUS control
```

Writes `model_cache_compact/<name>/` plus a `provenance.json`.

### 2. Fine-tune on the project corpus (4 GB GPU)

```bash
python -m it2edge.train.finetune_compact_marian --data_dir en-indic-exp
# smoke test first:
python -m it2edge.train.finetune_compact_marian --data_dir en-indic-exp --max_train_samples 2000
```

Memory-safe defaults: fp16, batch 1, grad-accum 16, gradient checkpointing,
`max_source_length=96`, `max_target_length=128`, LR 2e-5, ≤3 epochs, eval each
epoch, early stopping (patience 1). Best checkpoint by validation **chrF++** is
saved to `model_cache_compact_ft/`. Uses the 128,861-pair train split; the
1,301-pair dev split is validation only.

### 3. Convert to CT2 int8

```bash
python -m it2edge.convert.convert_compact_ct2      # -> model_cache_compact_ct2/
```

Stock CTranslate2 Marian converter (no custom loader). The Marian tokenizer
files are copied into the CT2 dir, so serving needs just that one directory.

### 4. Quality benchmark (DEV machine, same dev set for all models)

```bash
python -m it2edge.evaluate.benchmark_models quality \
    --model_type marian --ct2_dir model_cache_compact_ct2 \
    --data_dir en-indic-exp --out report_marian_ft.json

python -m it2edge.evaluate.benchmark_models quality \
    --model_type indictrans --ct2_dir model_cache_ct2 \
    --tokenizer_dir model_cache/indictrans2-en-indic-dist-200M \
    --data_dir en-indic-exp --out report_indictrans.json
```

Reports BLEU, chrF++, output-safety counts (empty / untranslated / repetition /
malformed Devanagari / bad length ratio) and a deterministic 100-sentence sample
for human review.

### 5. Latency benchmark (on the 1 GB Pi, Marian only)

Copy `model_cache_compact_ct2/` to the Pi, then:

```bash
pip install -r requirements/pi.txt
python -m it2edge.evaluate.benchmark_models latency \
    --model_type marian --ct2_dir model_cache_compact_ct2 \
    --data_dir en-indic-exp --intra 4 --out pi_latency_marian.json
```

Discards 5 warm-ups, times 50 short/medium/long sentences, reports median, p95,
peak RSS, load time, and artifact size.

### 6. Serve / deploy

```bash
# Local or Pi:
MODEL_TYPE=marian uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
# CLI check:
python -m it2edge.serve.translate_ct2 --marian "Hello, how are you?"
```

`MODEL_TYPE=marian` makes the FastAPI service load the compact model from
`model_cache_compact_ct2/` and skip IndicProcessor entirely. `MODEL_TYPE`
defaults to `indictrans` so the legacy path is unchanged.

## Release gates (all must pass to deploy)

| Gate | Threshold |
|---|---|
| Warm latency (Pi) | median ≤ 5 s, p95 ≤ 8 s |
| Peak RSS (Pi) | ≤ 700 MB |
| Quality | chrF++ ≤ 10% relative below IndicTrans2 reference |
| Reliability | no empty outputs; unsafe fraction < 1% |
| Human review | ≥ 90/100 sampled translations acceptable |
| Fine-tuning value | fine-tuned ≥ Samanantar base chrF++ |

If a gate fails, do not deploy; the report shows whether the blocker is quality
or Pi compute.

## Rollback

Model selection is via `MODEL_TYPE` + `CT2_MODEL_DIR`. To roll back the
on-device model, point at the previous Marian CT2 artifact. IndicTrans2 is a
dev-machine quality reference only and is not a Pi runtime fallback.

## Out of scope

Whisper/TTS, non-Hindi languages, pruning/distillation, and serving IndicTrans2
on the Pi.
