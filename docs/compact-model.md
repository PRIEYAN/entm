# MarianMT English→Hindi (production path)

Primary model: [`AI4INDIANS/better-opus-mt-en-hi`](https://huggingface.co/AI4INDIANS/better-opus-mt-en-hi)
(~77M MarianMT, Samanantar-tuned OPUS). This is the **only** translation model
in this repository.

Pinned revision: see `MODEL_REVISION` in [`it2edge/paths.py`](../it2edge/paths.py).

## A/B compare bases (recommended before a long fine-tune)

```bash
python -m it2edge.download_model --all
python -m it2edge.evaluate.ab_compare_models --data_dir en-indic-exp
```

Default pair: `better-opus-mt-en-hi` vs `opus-mt-en-hi` on the same 1,301-pair
dev set. Optional `--extra <hub-or-path>` (e.g. HPLT). Report: `ab_report.json`.

## Pipeline

```
download_model → finetune_marian → convert_ct2 → benchmark → deploy to Pi
```

### Download

```bash
python -m it2edge.download_model
python -m it2edge.download_model --control   # optional Helsinki OPUS baseline
```

### Fine-tune

```bash
python -m it2edge.train.finetune_marian --data_dir en-indic-exp
# smoke: --max_train_samples 2000
```

Defaults: fp16, batch 1, grad-accum 16, gradient checkpointing, seq 96/128,
LR 2e-5, ≤3 epochs, early stop on validation chrF++.
→ `model_cache_compact_ft/`

### Convert

```bash
python -m it2edge.convert.convert_ct2
```

→ `model_cache_compact_ct2/` (int8 + tokenizer)

### Benchmark

```bash
# DEV quality
python -m it2edge.evaluate.benchmark_models quality \
  --ct2_dir model_cache_compact_ct2 --data_dir en-indic-exp --out report.json

# PI latency
python -m it2edge.evaluate.benchmark_models latency \
  --ct2_dir model_cache_compact_ct2 --data_dir en-indic-exp --intra 4
```

### Serve

```bash
uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
python -m it2edge.serve.translate_ct2 "Hello, how are you?"
```

## Release gates (Pi)

| Gate | Threshold |
|---|---|
| Warm latency | median ≤ 5 s, p95 ≤ 8 s |
| Peak RSS | ≤ 700 MB |
| Reliability | no empty outputs; unsafe fraction < 1% |
| Human review | ≥ 90/100 acceptable samples |

## Rollback

Keep the previous `model_cache_compact_ct2/` build (or the pre-fine-tune base
converted with `--model model_cache_compact/better-opus-mt-en-hi`) and point
`CT2_MODEL_DIR` at it.
