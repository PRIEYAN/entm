"""Merge QLoRA adapters into a single fp16 IndicTrans2 model.

RUN ON THE DEV MACHINE after finetune_qlora.py. Plan reference: §5a.

Why this step exists:
  * finetune_qlora.py saves ADAPTERS ONLY. Neither CTranslate2 nor ONNX can read
    LoRA adapters -- they need one merged model.
  * We deliberately reload the base in fp16 (NOT 4-bit) and merge into that.
    The adapters were trained against the *dequantized* NF4 weights, so merging
    them into a still-4-bit base degrades quality. Dequantize first, then merge.
  * The result is an fp16 model (~400 MB) -- bigger and float again. That is
    expected: the small int8 Pi artifact is a SEPARATE, LATER step (convert_ct2.py).
    Never ship the 4-bit training object. See plan §0 and §11.

Usage:
    pip install "transformers>=4.51,<4.53" torch peft sentencepiece protobuf

    python -m it2edge.train.merge_lora --adapter_dir lora_adapters
    # then, LAST, quantize for the Pi (auto-detects the merged model):
    #   python -m it2edge.convert.convert_ct2
"""

import argparse
import os

from it2edge.paths import HF_SNAPSHOT, MERGED_DIR, MODEL_ID


def resolve_base_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    if HF_SNAPSHOT.is_dir():
        return str(HF_SNAPSHOT)
    print(f"[warn] {HF_SNAPSHOT} not found; loading base from the hub ({MODEL_ID}).")
    return MODEL_ID


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dequantize base to fp16 and merge QLoRA adapters into it"
    )
    parser.add_argument(
        "--adapter_dir", required=True, help="LoRA adapter dir from finetune_qlora"
    )
    parser.add_argument(
        "--output_dir",
        default=str(MERGED_DIR),
        help="where to save the merged fp16 model (default: model_cache_merged/)",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="base model path/id (default: local snapshot, else hub)",
    )
    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        raise SystemExit(
            f"{args.output_dir} already exists. Remove it first to re-merge."
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM

    from it2edge.tokenizer_utils import load_indictrans_tokenizer

    base_path = resolve_base_path(args.base)
    print(f"[info] loading base in fp16 (dequantized) from: {base_path}")

    # fp16, NOT 4-bit -- merging into a 4-bit base would degrade quality (§5a).
    base_fp16 = AutoModelForSeq2SeqLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    print(f"[info] applying adapters from: {args.adapter_dir}")
    merged = PeftModel.from_pretrained(base_fp16, args.adapter_dir).merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir)

    # Reuse the robust loader, then save the tokenizer alongside the weights so the
    # merged dir is self-contained. (Reload via tokenizer_utils to dodge the
    # save_pretrained src_vocab_file reload bug -- plan §5a caveat.)
    tokenizer = load_indictrans_tokenizer(base_path)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n[ok] merged fp16 model written to: {args.output_dir}  (~400 MB)")
    print("     This is float and full-size on purpose. Next, LAST step:")
    print("     python -m it2edge.convert.convert_ct2   # -> int8 model_cache_ct2/")


if __name__ == "__main__":
    main()
