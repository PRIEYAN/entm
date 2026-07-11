"""QLoRA (4-bit) fine-tune of IndicTrans2 en-indic distilled 200M.

RUN THIS ON A DEV MACHINE WITH A GPU (12-16 GB: free Colab/Kaggle T4, RTX 3060).
Plan reference: §4 (Phase 1 -- QLoRA fine-tuning).

What QLoRA does here (and what it does NOT do):
  * Freezes the base model in 4-bit NF4 (bitsandbytes) purely so the fine-tune
    fits in low VRAM, and trains small 16-bit LoRA adapters on top
    (~0.8M trainable params, ~0.4% of the model).
  * It is a TRAINING trick, not a deployment artifact. After training you have
    adapters only. You then run merge_lora.py (dequant base -> fp16, merge) to
    get one fp16 model, and convert_ct2.py LAST to get the int8 Pi artifact.
    "8-bit + QLoRA" is a contradiction -- QLoRA is 4-bit. See plan §0.

Data layout (line-aligned RAW parallel text -- feed raw, IndicProcessor handles
normalization/tagging/masking internally; do NOT pre-transliterate). See §4:

    en-indic-exp/
      train/eng_Latn-hin_Deva/train.eng_Latn   train.hin_Deva
      dev/eng_Latn-hin_Deva/dev.eng_Latn       dev.hin_Deva

Each *.eng_Latn line pairs with the same line number in the matching target file.
Multiple language-pair directories under train/ and dev/ are all picked up.

Usage:
    pip install "transformers>=4.51,<4.53" torch peft bitsandbytes datasets \
        accelerate sentencepiece sacremoses protobuf evaluate sacrebleu
    pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

    python -m it2edge.train.finetune_qlora --data_dir en-indic-exp --output_dir lora_adapters
    python -m it2edge.train.finetune_qlora --data_dir en-indic-exp --batch_size 8 --grad_accum 4

Note the OOM guard: the AI4Bharat default batch_size=32 will likely OOM a 16 GB
T4, so the default here is 8 with grad_accum 4 (effective batch 32). See §4.

This is a thin, self-contained wrapper over the technique in AI4Bharat's
huggingface_interface/train_lora.py; the verified LoRA config and baseline
hyperparameters below come from that repo (plan §4).
"""

import argparse
import glob
import os

from it2edge.paths import HF_SNAPSHOT, MODEL_ID

SRC_LANG = "eng_Latn"


def resolve_model_path() -> str:
    """Prefer the offline snapshot; fall back to the hub id (needs internet)."""
    if HF_SNAPSHOT.is_dir():
        return str(HF_SNAPSHOT)
    print(f"[warn] {HF_SNAPSHOT} not found; loading from the hub ({MODEL_ID}).")
    print("       Run `python -m it2edge.download_model` first for offline use.")
    return MODEL_ID


def discover_pairs(split_dir: str):
    """Find every <split>.eng_Latn / <split>.<tgt> file pair under split_dir.

    Layout: split_dir/eng_Latn-<tgt>/<split>.eng_Latn + <split>.<tgt>
    Returns a list of (src_file, tgt_file, tgt_lang) tuples.
    """
    pairs = []
    for pair_dir in sorted(glob.glob(os.path.join(split_dir, f"{SRC_LANG}-*"))):
        tgt_lang = os.path.basename(pair_dir).split("-", 1)[1]
        src_files = glob.glob(os.path.join(pair_dir, f"*.{SRC_LANG}"))
        for src_file in sorted(src_files):
            tgt_file = src_file[: -len(SRC_LANG)] + tgt_lang
            if os.path.isfile(tgt_file):
                pairs.append((src_file, tgt_file, tgt_lang))
            else:
                print(f"[warn] no target for {src_file} (expected {tgt_file}); skipping.")
    return pairs


def read_lines(path: str):
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


def build_examples(split_dir: str, processor):
    """Load all pairs in split_dir and return preprocessed (src, tgt) records.

    IndicProcessor.preprocess_batch applies IndicTrans2's language tags, script
    unification and entity masking -- the exact same pass used at inference, so
    training and serving stay consistent (plan §4).
    """
    srcs, tgts = [], []
    for src_file, tgt_file, tgt_lang in discover_pairs(split_dir):
        src_lines = read_lines(src_file)
        tgt_lines = read_lines(tgt_file)
        if len(src_lines) != len(tgt_lines):
            raise SystemExit(
                f"Line count mismatch: {src_file} ({len(src_lines)}) vs "
                f"{tgt_file} ({len(tgt_lines)}). Corpus must be line-aligned."
            )
        # Preprocess source with tags; target is normalized without the tag prefix.
        proc_src = processor.preprocess_batch(
            src_lines, src_lang=SRC_LANG, tgt_lang=tgt_lang
        )
        proc_tgt = processor.preprocess_batch(
            tgt_lines, src_lang=tgt_lang, tgt_lang=SRC_LANG, is_target=True
        )
        srcs.extend(proc_src)
        tgts.extend(proc_tgt)
    if not srcs:
        raise SystemExit(
            f"No training pairs found under {split_dir}. Expected "
            f"{split_dir}/{SRC_LANG}-<tgt>/<split>.{SRC_LANG} + .<tgt> files."
        )
    print(f"[info] {split_dir}: {len(srcs)} examples")
    return {"src": srcs, "tgt": tgts}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA (4-bit) fine-tune of IndicTrans2 en-indic dist-200M"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="corpus root containing train/ and dev/ (see module docstring)",
    )
    parser.add_argument(
        "--output_dir", default="lora_adapters", help="where to save LoRA adapters"
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    # Default 8 (not AI4Bharat's 32) to avoid OOM on a 16 GB T4 -- plan §4.
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="disable 4-bit loading (plain LoRA -- needs much more VRAM)",
    )
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForSeq2SeqLM,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from it2edge.tokenizer_utils import load_indictrans_tokenizer

    try:
        from IndicTransToolkit.processor import IndicProcessor
    except ImportError as exc:  # pragma: no cover - guidance for a missing dep
        raise SystemExit(
            "IndicTransToolkit is not installed. Install it with:\n"
            "    pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
        ) from exc

    if not torch.cuda.is_available():
        print(
            "[warn] no CUDA GPU detected. QLoRA is a GPU technique; a CPU run is "
            "only a smoke test and will be impractically slow (plan §4)."
        )

    path = resolve_model_path()
    tokenizer = load_indictrans_tokenizer(path)
    processor = IndicProcessor(inference=False)

    # ---- 4-bit NF4 base load: this is what makes LoRA "QLoRA" (plan §4). ----
    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        path,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Verified LoRA config from AI4Bharat train_lora.py (plan §4).
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- data ----
    train_raw = build_examples(os.path.join(args.data_dir, "train"), processor)
    dev_dir = os.path.join(args.data_dir, "dev")
    eval_raw = build_examples(dev_dir, processor) if os.path.isdir(dev_dir) else None

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["src"], truncation=True, max_length=args.max_length
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["tgt"], truncation=True, max_length=args.max_length
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_dict(train_raw).map(
        tokenize, batched=True, remove_columns=["src", "tgt"]
    )
    eval_ds = (
        Dataset.from_dict(eval_raw).map(
            tokenize, batched=True, remove_columns=["src", "tgt"]
        )
        if eval_raw
        else None
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Baseline HP from AI4Bharat train_lora.sh (plan §4): inverse_sqrt schedule,
    # adamw_torch, weight_decay 0.01. "May need tuning."
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        lr_scheduler_type="inverse_sqrt",
        optim="adamw_torch",
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        save_total_limit=2,
        load_best_model_at_end=bool(eval_ds),
        predict_with_generate=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("[info] starting QLoRA fine-tune ...")
    trainer.train()

    # Save ADAPTERS ONLY (this is not the deployment artifact -- plan §5a).
    model.save_pretrained(args.output_dir)
    print(f"\n[ok] LoRA adapters saved to: {args.output_dir}")
    print("     Next: python -m it2edge.train.merge_lora --adapter_dir "
          f"{args.output_dir}  (dequant base -> fp16, merge)")


if __name__ == "__main__":
    main()
