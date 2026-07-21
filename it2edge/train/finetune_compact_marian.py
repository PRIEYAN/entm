"""Fine-tune the compact en->hi MarianMT model on the project corpus.

RUN ON THE DEV MACHINE. Designed to fit a 4 GB GPU: full fine-tune (not LoRA)
of the ~77M model, with fp16, per-device batch 1, gradient accumulation,
gradient checkpointing, and short sequence caps. Only the 77M model is loaded,
so this is far lighter than IndicTrans2 QLoRA or online distillation.

    python -m it2edge.download_compact_model                     # get the base first
    python -m it2edge.train.finetune_compact_marian --data_dir en-indic-exp

Data layout (RAW line-aligned parallel text -- Marian consumes raw text, there
is NO IndicProcessor tagging here, unlike the IndicTrans2 path):

    en-indic-exp/
      train/eng_Latn-hin_Deva/train.eng_Latn   train.hin_Deva
      dev/eng_Latn-hin_Deva/dev.eng_Latn       dev.hin_Deva

The best checkpoint (by validation chrF++) is saved to model_cache_compact_ft/.
Convert it next with `python -m it2edge.convert.convert_compact_ct2`.
"""

import argparse
import os

from it2edge.paths import COMPACT_CACHE, COMPACT_FINETUNED
from it2edge.train.finetune_qlora import discover_pairs, read_lines

SRC_LANG = "eng_Latn"
BASE_SUBDIR = "better-opus-mt-en-hi"


def load_raw_pairs(split_dir: str):
    """Return {'src': [...], 'tgt': [...]} of RAW line-aligned text pairs."""
    srcs, tgts = [], []
    for src_file, tgt_file, _tgt_lang in discover_pairs(split_dir):
        src_lines = read_lines(src_file)
        tgt_lines = read_lines(tgt_file)
        if len(src_lines) != len(tgt_lines):
            raise SystemExit(
                f"Line count mismatch: {src_file} ({len(src_lines)}) vs "
                f"{tgt_file} ({len(tgt_lines)}). Corpus must be line-aligned."
            )
        srcs.extend(src_lines)
        tgts.extend(tgt_lines)
    if not srcs:
        raise SystemExit(
            f"No pairs found under {split_dir}. Expected "
            f"{split_dir}/{SRC_LANG}-<tgt>/<split>.{SRC_LANG} + .<tgt> files."
        )
    print(f"[info] {split_dir}: {len(srcs)} raw pairs")
    return {"src": srcs, "tgt": tgts}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune compact en->hi MarianMT on the project corpus (4GB-GPU safe)"
    )
    parser.add_argument(
        "--data_dir", required=True, help="corpus root containing train/ and dev/"
    )
    parser.add_argument(
        "--base_model",
        default=str(COMPACT_CACHE / BASE_SUBDIR),
        help="path/id of the base Marian model (default: downloaded snapshot)",
    )
    parser.add_argument("--output_dir", default=str(COMPACT_FINETUNED))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_source_length", type=int, default=96)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="cap training rows for a quick smoke run (e.g. 2000)",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    base = args.base_model
    if not os.path.isdir(base) and os.sep not in base:
        pass  # allow a bare hub id
    elif not os.path.isdir(base):
        raise SystemExit(
            f"Base model not found at {base}. Run "
            "`python -m it2edge.download_compact_model` first, or pass --base_model."
        )

    cuda = torch.cuda.is_available()
    if not cuda:
        print(
            "[warn] no CUDA GPU detected. This will run on CPU and be slow; use it "
            "only as a smoke test with --max_train_samples."
        )

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSeq2SeqLM.from_pretrained(base)
    # Gradient checkpointing trades compute for memory -- key to fitting 4 GB.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_raw = load_raw_pairs(os.path.join(args.data_dir, "train"))
    dev_dir = os.path.join(args.data_dir, "dev")
    eval_raw = load_raw_pairs(dev_dir) if os.path.isdir(dev_dir) else None
    if eval_raw is None:
        raise SystemExit(
            f"No dev split at {dev_dir}. A validation set is required for "
            "best-checkpoint selection and early stopping."
        )

    if args.max_train_samples:
        n = args.max_train_samples
        train_raw = {k: v[:n] for k, v in train_raw.items()}
        print(f"[info] capped training set to {len(train_raw['src'])} rows (smoke run)")

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["src"], truncation=True, max_length=args.max_source_length
        )
        labels = tokenizer(
            text_target=batch["tgt"],
            truncation=True,
            max_length=args.max_target_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_dict(train_raw).map(
        tokenize, batched=True, remove_columns=["src", "tgt"]
    )
    eval_ds = Dataset.from_dict(eval_raw).map(
        tokenize, batched=True, remove_columns=["src", "tgt"]
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    import sacrebleu

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        chrf = sacrebleu.corpus_chrf(
            decoded_preds, [decoded_labels], word_order=2  # chrF++
        ).score
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        return {"chrf": chrf, "bleu": bleu}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        weight_decay=0.01,
        fp16=cuda,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=1,
        report_to="none",
    )

    # transformers >=4.46 renamed the Trainer's `tokenizer=` arg to
    # `processing_class=` (the old name is removed in the v5 line).
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print("[info] starting compact Marian fine-tune ...")
    trainer.train()

    # Save the best model (loaded at end) + tokenizer as a full HF model dir.
    model.config.use_cache = True
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n[ok] fine-tuned model saved to: {args.output_dir}")
    print("     Next: python -m it2edge.convert.convert_compact_ct2")


if __name__ == "__main__":
    main()
