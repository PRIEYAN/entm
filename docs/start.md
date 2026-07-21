# Start here — MarianMT en→hi walkthrough

End-to-end path for [`AI4INDIANS/better-opus-mt-en-hi`](https://huggingface.co/AI4INDIANS/better-opus-mt-en-hi).
Full rationale and release gates: [`compact-model.md`](compact-model.md).
Laptop setup details: [`../README.md`](../README.md).

## Dev laptop

```bash
python -m venv venv && source venv/bin/activate   # or Windows Activate.ps1
# Install CUDA torch if you have an NVIDIA GPU (see README)
pip install -r requirements/dev.txt && pip install -e .

python -m it2edge.download_model
python -m it2edge.train.finetune_marian --data_dir en-indic-exp --max_train_samples 2000
python -m it2edge.train.finetune_marian --data_dir en-indic-exp
python -m it2edge.convert.convert_ct2
python -m it2edge.serve.translate_ct2 "Hello, how are you?"
```

## Raspberry Pi (64-bit)

```bash
# copy model_cache_compact_ct2/ from the laptop first
pip install -r requirements/pi.txt && pip install -e .
uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
```

> Historical IndicTrans2 notes in `plan.md` / `reduceParams.md` are archived
> design discussion only — that model is no longer part of this codebase.
