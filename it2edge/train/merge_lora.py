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

    # Make the merged dir self-contained and loadable, WITHOUT re-serializing
    # the remote tokenizer. tokenizer.save_pretrained() poisons
    # tokenizer_config.json: it injects src_vocab_file (the classic reload
    # TypeError) AND writes added_tokens_decoder as plain dicts, later crashing
    # with "'dict' object has no attribute 'content'". So we COPY the raw repo
    # files verbatim instead. This copies:
    #   * every remote-code .py (modeling_indictrans.py, configuration_*,
    #     tokenization_*) -- trust_remote_code needs these or the model load
    #     fails with "does not appear to have a file named modeling_indictrans.py"
    #   * the tokenizer vocab/config files (dict.*, *.SRC/.TGT, tokenizer_config,
    #     special_tokens_map)
    # It never overwrites config.json (save_pretrained already wrote the model's).
    import shutil

    if os.path.isdir(base_path):
        skip = {"config.json"}  # keep the merged model's own config
        copied = []
        for name in sorted(os.listdir(base_path)):
            src = os.path.join(base_path, name)
            if not os.path.isfile(src) or name in skip:
                continue
            # weight files belong to the merged model, not the base -- skip them.
            if name.endswith((".safetensors", ".bin", ".pt", ".h5", ".msgpack")):
                continue
            is_py = name.endswith(".py")
            is_tok = name in {
                "tokenizer_config.json", "special_tokens_map.json",
                "dict.SRC.json", "dict.TGT.json", "model.SRC", "model.TGT",
                "sentencepiece.bpe.model", "tokenizer.json",
            }
            if not (is_py or is_tok):
                continue
            dst = os.path.join(args.output_dir, name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(name)
        print(f"[info] copied remote-code + tokenizer files from snapshot: {copied}")
        # Sanity-check that the merged dir's tokenizer now loads.
        load_indictrans_tokenizer(args.output_dir)
        print("[info] tokenizer in merged dir loads OK")
    else:
        print("[warn] base is a hub id, not a local dir; the merged dir has no "
              "remote-code/tokenizer files and will not load standalone. Merge "
              "against a local snapshot (run it2edge.download_model first).")

    print(f"\n[ok] merged fp16 model written to: {args.output_dir}  (~400 MB)")
    print("     This is float and full-size on purpose. Next, LAST step:")
    print("     python -m it2edge.convert.convert_ct2   # -> int8 model_cache_ct2/")


if __name__ == "__main__":
    main()
