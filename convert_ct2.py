"""Convert IndicTrans2 (en-indic distilled 200M) to a CTranslate2 int8 package.

RUN THIS ON YOUR DEV MACHINE (x86-64, e.g. this Windows box), NOT on the Pi.
It produces a small int8 CTranslate2 model directory that you then copy to a
64-bit Raspberry Pi and run with translate_ct2.py -- no PyTorch needed there.

Why: a 200M model in fp32 is ~800 MB and needs PyTorch, which is too heavy for
a 1 GB Pi 3. The int8 CTranslate2 package is ~200 MB and runs on the light
ctranslate2 CPU runtime, which has aarch64 wheels (64-bit Raspberry Pi OS).

    pip install ctranslate2 transformers sentencepiece protobuf
    python convert_ct2.py            # writes ./model_cache_ct2/

Then copy BOTH of these to the Pi:
    - model_cache_ct2/                         (the CT2 int8 weights)
    - model_cache/indictrans2-en-indic-dist-200M/   (the tokenizer files)

IMPORTANT compatibility note
----------------------------
IndicTrans2 ships a *custom* HF architecture (trust_remote_code=True). The
ct2-transformers-converter dispatches on the model type; if it does not
recognise IndicTrans2's, conversion will fail. If that happens, this script
prints the official fairseq -> CT2 fallback route instead of dying silently.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
# The self-contained HF copy saved by download_model.py (preferred), else hub id.
LOCAL_HF_DIR = os.path.join(HERE, "model_cache", "indictrans2-en-indic-dist-200M")
OUTPUT_DIR = os.path.join(HERE, "model_cache_ct2")

FALLBACK_MSG = f"""
[!] The Hugging Face -> CTranslate2 conversion failed. This usually means the
    generic converter does not recognise IndicTrans2's custom architecture.

    Official fallback (AI4Bharat ship CT2 ports directly):
      1. Get the fairseq checkpoint + CT-ported dirs from the IndicTrans2 repo:
             https://github.com/AI4Bharat/IndicTrans2   (see the models table)
         The distilled en-indic download already contains 2 CT-ported dirs.
      2. Or convert a fairseq checkpoint yourself with the fairseq converter:
             https://opennmt.net/CTranslate2/guides/fairseq.html
      3. Copy the resulting CT2 dir to the Pi and point translate_ct2.py at it
         via --model_dir.
"""


def resolve_source() -> str:
    if os.path.isdir(LOCAL_HF_DIR):
        return LOCAL_HF_DIR
    print(f"[warn] {LOCAL_HF_DIR} not found; converting straight from the hub.")
    print("       (Run download_model.py first to avoid a re-download.)")
    return MODEL_ID


def main() -> None:
    if os.path.isdir(OUTPUT_DIR):
        raise SystemExit(
            f"{OUTPUT_DIR} already exists. Remove it first to re-convert."
        )

    source = resolve_source()
    cmd = [
        "ct2-transformers-converter",
        "--model", source,
        "--output_dir", OUTPUT_DIR,
        "--quantization", "int8",
        # IndicTrans2 needs its remote code to load the custom classes.
        "--trust_remote_code",
    ]
    print("[info] running:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit(
            "ct2-transformers-converter not found. Install it with:\n"
            "    pip install ctranslate2 transformers"
        )
    except subprocess.CalledProcessError:
        print(FALLBACK_MSG, file=sys.stderr)
        raise SystemExit(1)

    print(f"\n[ok] int8 CTranslate2 model written to: {OUTPUT_DIR}")
    print("     Copy it (and the tokenizer dir) to the Pi, then run translate_ct2.py.")


if __name__ == "__main__":
    main()
