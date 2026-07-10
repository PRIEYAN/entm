# IndicTrans2 — local English→Indic translation

Runs [`ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
fully locally. Works on CPU; uses GPU automatically if CUDA is available.

## Setup (run once, needs internet)

```bash
python -m venv venv
# Windows PowerShell:  venv\Scripts\Activate.ps1
# Windows cmd:         venv\Scripts\activate.bat
# macOS / Linux:       source venv/bin/activate

pip install -r requirements.txt
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

python download_model.py   # caches weights into ./model_cache
```

`IndicTransToolkit` is installed separately because it is not on PyPI — it
provides the `IndicProcessor` that tags input with language tokens and handles
script conversion. Without it the model does **not** translate correctly.

## Run (offline after the download step)

```bash
python translate.py "Hello, how are you?"            # -> Hindi by default
python translate.py --tgt tam_Taml "This is a test." # -> Tamil
python translate.py                                  # built-in multi-language demo
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

## Files

- `requirements.txt` — pip dependencies (`transformers<5`, torch, etc.)
- `download_model.py` — one-time weight download into `model_cache/`
- `translate.py` — offline inference (CLI + importable `translate()`)
