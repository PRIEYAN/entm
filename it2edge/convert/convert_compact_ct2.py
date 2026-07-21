"""Convert the compact en->hi MarianMT model to a CTranslate2 int8 package.

RUN ON THE DEV MACHINE. Unlike IndicTrans2 (which needs the custom dual-vocab
loader in convert_ct2.py), MarianMT is natively supported by CTranslate2's stock
converter -- no custom loader, no trust_remote_code.

    python -m it2edge.convert.convert_compact_ct2                       # fine-tuned model
    python -m it2edge.convert.convert_compact_ct2 --model model_cache_compact/opus-mt-en-hi

Output: model_cache_compact_ct2/ (int8). Copy it plus the tokenizer files
(source.spm, target.spm, vocab.json, *tokenizer*.json) to the Pi. The Marian
tokenizer travels WITH the CT2 dir here (it is copied in), so serving needs only
one directory.
"""

import argparse
import os
import shutil
import sys

from it2edge.paths import COMPACT_CT2_DIR, COMPACT_FINETUNED

# Tokenizer/vocab files CT2 does not emit but the Marian serve path needs.
_TOKENIZER_FILES = (
    "source.spm",
    "target.spm",
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "config.json",
)


def _copy_tokenizer(src_model_dir: str, out_dir: str) -> None:
    if not os.path.isdir(src_model_dir):
        # A bare hub id was converted; nothing local to copy. The serve path can
        # still load the tokenizer from the hub id if given one.
        print("[warn] source is not a local dir; tokenizer files not copied.")
        return
    for name in _TOKENIZER_FILES:
        src = os.path.join(src_model_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))
    print(f"[info] copied Marian tokenizer files into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert compact en->hi MarianMT to CTranslate2 int8"
    )
    parser.add_argument(
        "--model",
        default=str(COMPACT_FINETUNED),
        help="HF Marian model dir or hub id (default: the fine-tuned model)",
    )
    parser.add_argument("--output_dir", default=str(COMPACT_CT2_DIR))
    parser.add_argument("--quantization", default="int8")
    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        raise SystemExit(
            f"{args.output_dir} already exists. Remove it first to re-convert."
        )
    if os.sep in args.model and not os.path.isdir(args.model):
        raise SystemExit(
            f"Model not found at {args.model}. Fine-tune it first with "
            "`python -m it2edge.train.finetune_compact_marian`, or pass --model."
        )

    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        raise SystemExit("ctranslate2 is not installed:  pip install ctranslate2")

    print(
        f"[info] converting {args.model} -> {args.output_dir} "
        f"(quantization={args.quantization}, stock Marian loader)"
    )
    try:
        converter = TransformersConverter(args.model)
        converter.convert(args.output_dir, quantization=args.quantization, force=True)
    except Exception as exc:
        print(f"[error] conversion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    _copy_tokenizer(args.model, args.output_dir)

    print(f"\n[ok] int8 CTranslate2 model written to: {args.output_dir}")
    print("     Verify locally:")
    print(f'       python -m it2edge.serve.translate_ct2 --marian "Hello, how are you?"')
    print("     Then benchmark on the Pi (docs/compact-model.md).")


if __name__ == "__main__":
    main()
