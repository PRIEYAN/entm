"""Run IndicTrans2 via CTranslate2 int8 -- Pi-friendly, no PyTorch.

This is the inference script meant to run ON THE RASPBERRY PI (64-bit OS).
It uses the light `ctranslate2` CPU runtime for the model, and keeps the HF
tokenizer + IndicProcessor (from IndicTransToolkit) for pre/post-processing,
exactly like translate.py -- but with no torch dependency.

Prereqs on the Pi (see requirements/pi.txt):
    pip install ctranslate2 transformers sentencepiece sacremoses protobuf
    pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

Layout it expects (copied from the dev machine):
    ./model_cache_ct2/                              (int8 CT2 weights)
    ./model_cache/indictrans2-en-indic-dist-200M/   (tokenizer files)

Usage (run from the project root):
    python -m it2edge.serve.translate_ct2 "Hello, how are you?"             # -> Hindi
    python -m it2edge.serve.translate_ct2 --tgt tam_Taml "This is a test."  # -> Tamil
    python -m it2edge.serve.translate_ct2 --beams 5 "Slower, higher quality."
    python -m it2edge.serve.translate_ct2                                   # built-in demo

Greedy decoding (beam_size=1) is the default: on a Pi 3 it is several times
faster than beam search for a small, usually acceptable quality drop.

Token handling (verified against the PyTorch path): IndicTrans2 routes the
target language via the tags IndicProcessor.preprocess_batch prepends to the
SOURCE (unlike vanilla M2M100), so NO target_prefix is passed. The predicted
target subword pieces are detokenized directly ('▁'->space) rather than via
tokenizer.decode, because the tokenizer's convert_tokens_to_ids/decode use the
SOURCE vocab and would blank out the Hindi output.
"""

import argparse
import os

import ctranslate2

from it2edge.paths import COMPACT_CT2_DIR, CT2_DIR, HF_SNAPSHOT
from it2edge.tokenizer_utils import detokenize_target, load_indictrans_tokenizer

DEFAULT_CT2_DIR = str(CT2_DIR)
DEFAULT_TOKENIZER_DIR = str(HF_SNAPSHOT)
SRC_LANG = "eng_Latn"


def _require_indicprocessor():
    try:
        from IndicTransToolkit.processor import IndicProcessor
    except ImportError as exc:  # pragma: no cover - guidance for a missing dep
        raise SystemExit(
            "IndicTransToolkit is not installed. Install it with:\n"
            "    pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
        ) from exc
    return IndicProcessor


def load(model_dir: str, tokenizer_dir: str):
    if not os.path.isdir(model_dir):
        raise SystemExit(
            f"CT2 model not found at {model_dir}.\n"
            "Convert it on your dev machine with `python -m it2edge.convert.convert_ct2`\n"
            "and copy it here, or pass --model_dir pointing at your CT2 directory."
        )
    if not os.path.isdir(tokenizer_dir):
        raise SystemExit(
            f"Tokenizer not found at {tokenizer_dir}.\n"
            "Copy model_cache/indictrans2-en-indic-dist-200M/ from the dev machine,\n"
            "or pass --tokenizer_dir."
        )

    tokenizer = load_indictrans_tokenizer(tokenizer_dir)
    # CPU int8 runtime. intra_threads defaults to the core count; the Pi 3 has 4.
    translator = ctranslate2.Translator(model_dir, device="cpu", compute_type="int8")
    processor = _require_indicprocessor()(inference=True)
    return tokenizer, translator, processor


def translate(sentences, tgt_lang, tokenizer, translator, processor, beam_size=1):
    """Translate a list of English strings into tgt_lang. Returns a list."""
    batch = processor.preprocess_batch(sentences, src_lang=SRC_LANG, tgt_lang=tgt_lang)

    # CTranslate2 works on token strings, not tensors: encode to ids, then map
    # ids back to the tokenizer's subword tokens for each sentence.
    encoded = tokenizer(batch, truncation=True, padding=False).input_ids
    source_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in encoded]

    results = translator.translate_batch(
        source_tokens,
        beam_size=beam_size,
        max_decoding_length=256,
    )

    # Detokenize target pieces directly (see detokenize_target): the id
    # round-trip uses the SOURCE vocab and would blank out the Hindi output.
    decoded = [detokenize_target(res.hypotheses[0]) for res in results]

    # Postprocess converts back to native script and cleans up spacing.
    return processor.postprocess_batch(decoded, lang=tgt_lang)


def demo(tokenizer, translator, processor, beam_size):
    sentences = [
        "Hello, how are you today?",
        "The weather is beautiful this morning.",
        "I would like a cup of coffee, please.",
    ]
    for tgt in ["hin_Deva", "tam_Taml", "ben_Beng"]:
        print(f"\n=== eng_Latn -> {tgt} ===")
        out = translate(sentences, tgt, tokenizer, translator, processor, beam_size)
        for src, dst in zip(sentences, out):
            print(f"  EN: {src}")
            print(f"  ->  {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IndicTrans2 en-indic local translator (CTranslate2 int8)"
    )
    parser.add_argument("text", nargs="*", help="English text to translate")
    parser.add_argument(
        "--marian",
        action="store_true",
        help="use the compact en->hi MarianMT CT2 model (Pi runtime) instead of IndicTrans2",
    )
    parser.add_argument(
        "--tgt", default="hin_Deva", help="target language code (default: hin_Deva)"
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=1,
        help="beam size; 1 = greedy (fastest, default). Try 5 for best quality.",
    )
    parser.add_argument(
        "--model_dir", default=DEFAULT_CT2_DIR, help="CT2 model directory"
    )
    parser.add_argument(
        "--tokenizer_dir", default=DEFAULT_TOKENIZER_DIR, help="HF tokenizer directory"
    )
    args = parser.parse_args()

    if args.marian:
        from it2edge.serve.marian_ct2 import load_marian, translate_marian

        model_dir = args.model_dir if args.model_dir != DEFAULT_CT2_DIR else str(
            COMPACT_CT2_DIR
        )
        tokenizer, translator = load_marian(model_dir)
        print(f"[info] loaded compact Marian CT2 int8 model (beam_size={args.beams})")
        sentences = [" ".join(args.text)] if args.text else [
            "Hello, how are you today?",
            "The weather is beautiful this morning.",
        ]
        out = translate_marian(sentences, tokenizer, translator, beam_size=args.beams)
        for src, dst in zip(sentences, out):
            print(f"\nEN: {src}\n-> {dst}")
        return

    tokenizer, translator, processor = load(args.model_dir, args.tokenizer_dir)
    print(f"[info] loaded CT2 int8 model (beam_size={args.beams})")

    if not args.text:
        demo(tokenizer, translator, processor, args.beams)
        return

    sentence = " ".join(args.text)
    out = translate([sentence], args.tgt, tokenizer, translator, processor, args.beams)
    print(f"\nEN ({SRC_LANG}): {sentence}")
    print(f"-> ({args.tgt}): {out[0]}")


if __name__ == "__main__":
    main()
