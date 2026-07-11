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

### Already hit `got multiple values for keyword argument 'src_vocab_file'`?

An earlier version of `download_model.py` used `save_pretrained`, which writes
`src_vocab_file` into `tokenizer_config.json`; on reload the remote tokenizer
then receives it both positionally and by keyword and raises that TypeError.
The current `download_model.py` snapshots the raw repo instead, which avoids it.
To recover, clear the poisoned caches and re-download:

```bash
rm -rf model_cache
rm -rf ~/.cache/huggingface/modules/transformers_modules/indictrans2-en-indic-dist-200M
python download_model.py
python translate.py "hello"
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

The default `translate.py` needs PyTorch and loads the model in fp32 (~800 MB),
which is too heavy for a Pi 3 (1 GB RAM). For low-memory / ARM devices there is
a **CTranslate2 int8** path (~200 MB, no PyTorch at runtime).

**Requires a 64-bit Raspberry Pi OS** — `ctranslate2` ships aarch64 wheels but
**no 32-bit/armv7 wheels**, and PyTorch has dropped armv7 too. On 32-bit Pi OS
neither this path nor the default one will `pip install` without building from
source. Reflash to 64-bit first.

**1. Convert on your dev machine** (x86-64 — this is where the heavy tooling lives):

```bash
pip install ctranslate2 transformers sentencepiece protobuf
python download_model.py    # if you haven't already
python convert_ct2.py       # writes ./model_cache_ct2/ (int8)
```

**2. Copy to the Pi** — both the CT2 weights and the tokenizer:

```
model_cache_ct2/                              # int8 CT2 model
model_cache/indictrans2-en-indic-dist-200M/   # tokenizer files
```

**3. Run on the Pi** (64-bit OS):

```bash
pip install -r requirements-pi.txt
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

python translate_ct2.py "Hello, how are you?"             # greedy, fastest
python translate_ct2.py --beams 5 "Higher quality, slower."
```

Greedy decoding (`--beams 1`, the default) is several times faster than beam
search on a Pi 3. If `convert_ct2.py` fails on IndicTrans2's custom
architecture, it prints the official AI4Bharat fairseq→CT2 fallback route.

## Files

- `requirements.txt` — pip dependencies for the default torch path (`transformers>=4.51,<4.53`, torch, etc.)
- `requirements-pi.txt` — slim CTranslate2 deps for the Pi (no torch)
- `download_model.py` — one-time model snapshot into `model_cache/`
- `translate.py` — offline PyTorch inference (CLI + importable `translate()`)
- `tokenizer_utils.py` — shared robust tokenizer loader (works around the `src_vocab_file` reload bug)
- `convert_ct2.py` — convert the HF model to a CTranslate2 int8 package (run on dev machine)
- `translate_ct2.py` — CTranslate2 int8 inference, no PyTorch (run on the Pi)
