"""Convert the MarianMT en→hi model to a CTranslate2 int8 package.

RUN ON THE DEV MACHINE. MarianMT is natively supported by CTranslate2's stock
converter (no custom loader).

    python -m it2edge.convert.convert_ct2                       # fine-tuned model
    python -m it2edge.convert.convert_ct2 --model model_cache_compact/better-opus-mt-en-hi

Output: model_cache_compact_ct2/ (int8 + tokenizer files copied in).
"""

import argparse
import os
import shutil
import sys

from it2edge.paths import CT2_DIR, FINETUNED_DIR

_TOKENIZER_FILES = (
    "source.spm",
    "target.spm",
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    # NOTE: do NOT copy the HF model config.json here. The CT2 converter writes
    # its own config.json into the output dir, and copying the HF Marian config
    # over it (which contains `max_length: null` etc.) breaks
    # ctranslate2.Translator with a JSON null type error.
)


def _copy_tokenizer(src_model_dir: str, out_dir: str) -> None:
    if not os.path.isdir(src_model_dir):
        print("[warn] source is not a local dir; tokenizer files not copied.")
        return
    for name in _TOKENIZER_FILES:
        src = os.path.join(src_model_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))
    print(f"[info] copied Marian tokenizer files into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MarianMT en→hi to CTranslate2 int8"
    )
    parser.add_argument(
        "--model",
        default=str(FINETUNED_DIR),
        help="HF Marian model dir or hub id (default: fine-tuned model)",
    )
    parser.add_argument("--output_dir", default=str(CT2_DIR))
    parser.add_argument("--quantization", default="int8")
    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        raise SystemExit(
            f"{args.output_dir} already exists. Remove it first to re-convert."
        )
    if os.sep in args.model and not os.path.isdir(args.model):
        raise SystemExit(
            f"Model not found at {args.model}. Fine-tune it first with "
            "`python -m it2edge.train.finetune_marian`, or pass --model."
        )

    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        raise SystemExit("ctranslate2 is not installed:  pip install ctranslate2")

    print(
        f"[info] converting {args.model} -> {args.output_dir} "
        f"(quantization={args.quantization})"
    )
    try:
        converter = TransformersConverter(args.model)
        converter.convert(args.output_dir, quantization=args.quantization, force=True)
    except Exception as exc:
        print(f"[error] conversion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    _copy_tokenizer(args.model, args.output_dir)

    print(f"\n[ok] int8 CTranslate2 model written to: {args.output_dir}")
    print("     Verify (CPU fallback backend on the Jetson):")
    print('       TRANSLATE_BACKEND=ct2 python -m nvidia.jetson_mic --text "Hello, how are you?"')


if __name__ == "__main__":
    main()
