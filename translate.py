"""Run IndicTrans2 (en-indic distilled 200M) locally, fully offline.

English -> Indic-language translation. Loads the model saved by download_model.py
(falls back to the HF hub id if the local copy is absent), then translates.

Usage:
    # translate a one-off sentence (default target: Hindi)
    python translate.py "Hello, how are you?"

    # pick a target language by its IndicTrans2 code
    python translate.py --tgt tam_Taml "This is a test sentence."

    # no args -> runs a small built-in demo across a few languages
    python translate.py

Common target codes: hin_Deva (Hindi), tam_Taml (Tamil), tel_Telu (Telugu),
ben_Beng (Bengali), mar_Deva (Marathi), guj_Gujr (Gujarati), kan_Knda (Kannada),
mal_Mlym (Malayalam), pan_Guru (Punjabi), urd_Arab (Urdu).
Source is always eng_Latn for this en-indic model.
"""

import argparse
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError as exc:  # pragma: no cover - guidance for a missing dep
    raise SystemExit(
        "IndicTransToolkit is not installed. Install it with:\n"
        "    pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
    ) from exc

MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model_cache",
    "indictrans2-en-indic-dist-200M",
)
SRC_LANG = "eng_Latn"


def resolve_model_path() -> str:
    """Prefer the offline copy; fall back to the hub id (needs internet)."""
    if os.path.isdir(LOCAL_DIR):
        return LOCAL_DIR
    print(f"[warn] {LOCAL_DIR} not found; loading from the hub ({MODEL_ID}).")
    print("       Run download_model.py first to enable offline use.")
    return MODEL_ID


def load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = resolve_model_path()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path,
        trust_remote_code=True,
        # float16 only helps on GPU; keep float32 on CPU for correctness/speed.
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    processor = IndicProcessor(inference=True)
    return tokenizer, model, processor, device


def translate(sentences, tgt_lang, tokenizer, model, processor, device):
    """Translate a list of English strings into tgt_lang. Returns a list."""
    batch = processor.preprocess_batch(sentences, src_lang=SRC_LANG, tgt_lang=tgt_lang)

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess converts back to native script and cleans up spacing.
    return processor.postprocess_batch(decoded, lang=tgt_lang)


def demo(tokenizer, model, processor, device):
    sentences = [
        "Hello, how are you today?",
        "The weather is beautiful this morning.",
        "I would like a cup of coffee, please.",
    ]
    for tgt in ["hin_Deva", "tam_Taml", "ben_Beng"]:
        print(f"\n=== eng_Latn -> {tgt} ===")
        out = translate(sentences, tgt, tokenizer, model, processor, device)
        for src, dst in zip(sentences, out):
            print(f"  EN: {src}")
            print(f"  ->  {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="IndicTrans2 en-indic local translator")
    parser.add_argument("text", nargs="*", help="English text to translate")
    parser.add_argument(
        "--tgt", default="hin_Deva", help="target language code (default: hin_Deva)"
    )
    args = parser.parse_args()

    tokenizer, model, processor, device = load()
    print(f"[info] loaded on device: {device}")

    if not args.text:
        demo(tokenizer, model, processor, device)
        return

    sentence = " ".join(args.text)
    out = translate([sentence], args.tgt, tokenizer, model, processor, device)
    print(f"\nEN ({SRC_LANG}): {sentence}")
    print(f"-> ({args.tgt}): {out[0]}")


if __name__ == "__main__":
    main()
