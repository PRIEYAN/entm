"""OPTIONAL structured pruning of the merged fp16 IndicTrans2 model.

RUN ON THE DEV MACHINE, and only if int8 + greedy (convert_ct2.py + §9 decoding)
still leaves the model too big/slow on the Pi. Plan reference: §5b.

Read this first -- what "pruning" means here:
  * The brief said "dynamically prune weights that don't contribute". That is
    UNSTRUCTURED (magnitude) pruning: it zeros individual weights. On a dense CPU
    int8 backend (CTranslate2 / ONNX Runtime CPU) it yields a smaller *file* but
    IDENTICAL latency -- neither runtime exploits unstructured sparsity. So it is
    NOT what you want for speed. See plan §0 and §9 (ROI item 6: skip it).
  * This script does STRUCTURED pruning: it drops whole FFN neurons / attention
    heads / decoder layers, which actually shrinks the dense math and speeds up
    CPU inference. torch-pruning's DepGraph rewrites the tensors for real (unlike
    torch.nn.utils.prune, which only masks).

Highest-yield moves (verified insight, plan §5b):
  * Decoder DEPTH dominates autoregressive latency  -> drop decoder layers.
  * Encoder WIDTH preserves quality               -> narrow the encoder modestly.
  So prefer: shorten decoder layers + modestly narrow encoder, not uniform pruning.

Two hard constraints (plan §5b, §10.7):
  1. HEAL IS MANDATORY. Structured pruning drops BLEU immediately. After pruning
     you MUST fine-tune ("heal") on parallel data (re-run finetune_qlora.py-style
     training on the pruned model) and re-check BLEU/chrF, then re-convert.
  2. CONVERTER-LEGAL SHAPES ONLY. The CT2 converter assumes uniform topology
     (same heads/FFN dim across layers). Heterogeneous per-layer pruning WON'T
     round-trip. Keep reductions uniform across layers + only whole-layer drops.

Honest verdict (plan §5b): for an already-distilled 200M model, structured
pruning has diminishing returns (distillation already spent the redundancy) and
is high-effort + fragile through the custom arch. Expect ~10-30% more size/latency
after healing, at a measurable BLEU cost. Do int8 + greedy FIRST.

Usage:
    pip install "transformers>=4.51,<4.53" torch torch-pruning sentencepiece protobuf

    python -m it2edge.train.prune_structured --model_dir model_cache_merged \
        --output_dir model_cache_pruned \
        --ffn_ratio 0.25 --drop_decoder_layers 2
    # THEN heal (fine-tune on parallel data), THEN convert_ct2, THEN benchmark.
"""

import argparse
import os

from it2edge.paths import MERGED_DIR, PRUNED_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structured pruning (decoder depth + encoder/FFN width) for IndicTrans2"
    )
    parser.add_argument(
        "--model_dir",
        default=str(MERGED_DIR),
        help="merged fp16 model dir (from merge_lora; default: model_cache_merged/)",
    )
    parser.add_argument(
        "--output_dir", default=str(PRUNED_DIR), help="where to save the pruned model"
    )
    parser.add_argument(
        "--ffn_ratio",
        type=float,
        default=0.25,
        help="fraction of FFN neurons to remove, uniform across layers (0=off)",
    )
    parser.add_argument(
        "--drop_decoder_layers",
        type=int,
        default=0,
        help="number of whole decoder layers to drop from the top (0=off)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.ffn_ratio < 1.0:
        raise SystemExit("--ffn_ratio must be in [0, 1).")
    if os.path.isdir(args.output_dir):
        raise SystemExit(f"{args.output_dir} already exists. Remove it to re-prune.")

    import torch
    from transformers import AutoModelForSeq2SeqLM

    from it2edge.tokenizer_utils import load_indictrans_tokenizer

    try:
        import torch_pruning as tp
    except ImportError as exc:  # pragma: no cover - guidance for a missing dep
        raise SystemExit(
            "torch-pruning is not installed. Install it with:\n"
            "    pip install torch-pruning\n"
            "(Avoid torch.nn.utils.prune -- it only masks, giving no dense speedup.)"
        ) from exc

    print(f"[info] loading merged fp16 model from: {args.model_dir}")
    # fp32 on CPU for a stable pruning/dependency-graph pass; re-cast on save.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    tokenizer = load_indictrans_tokenizer(args.model_dir)

    # ---- 1. Drop whole decoder layers (biggest autoregressive-latency win). ----
    # Whole-layer removal keeps every remaining layer's topology uniform, so it
    # stays converter-legal (plan §5b). We locate the decoder layer ModuleList by
    # duck-typing because the custom arch's attribute names may vary by revision.
    if args.drop_decoder_layers > 0:
        layers = _find_decoder_layers(model)
        if layers is None:
            raise SystemExit(
                "Could not locate the decoder layer list on this arch. Inspect "
                "model.named_modules() and adapt _find_decoder_layers()."
            )
        n = len(layers)
        keep = n - args.drop_decoder_layers
        if keep < 1:
            raise SystemExit(f"Cannot drop {args.drop_decoder_layers} of {n} layers.")
        # Drop from the TOP -- late layers are the usual choice for depth pruning.
        del layers[keep:]
        _sync_layer_count(model, keep)
        print(f"[info] decoder layers: {n} -> {keep}")

    # ---- 2. Narrow FFN width uniformly via DepGraph (real tensor shrink). ----
    if args.ffn_ratio > 0.0:
        example = _example_inputs(tokenizer)
        # Prune Linear layers whose name marks them as FFN (fc1/fc2). Uniform ratio
        # across all layers keeps shapes converter-legal.
        ignored = [m for n_, m in model.named_modules() if _is_head_or_embed(n_)]
        pruner = tp.pruning.MagnitudePruner(
            model,
            example_inputs=example,
            importance=tp.importance.MagnitudeImportance(p=2),
            pruning_ratio=args.ffn_ratio,
            ignored_layers=ignored,
            global_pruning=False,  # uniform per-layer -> converter-legal shapes
        )
        pruner.step()
        print(f"[info] FFN width reduced by ratio {args.ffn_ratio} (uniform).")

    os.makedirs(args.output_dir, exist_ok=True)
    model.half().save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n[ok] pruned model written to: {args.output_dir}")
    print("[!]  BLEU has DROPPED. You MUST now heal (fine-tune on parallel data)")
    print("     and re-check BLEU/chrF before converting. Then, LAST:")
    print("       python -m it2edge.convert.convert_ct2   # int8 -- re-benchmark on the Pi")


def _find_decoder_layers(model):
    """Best-effort locate the decoder's layer ModuleList across arch revisions."""
    import torch.nn as nn

    candidates = ["model.decoder.layers", "decoder.layers", "model.model.decoder.layers"]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    return None


def _sync_layer_count(model, keep: int) -> None:
    """Update config counters so the model + converter agree on decoder depth."""
    for attr in ("decoder_layers", "num_decoder_layers"):
        if hasattr(model.config, attr):
            setattr(model.config, attr, keep)


def _is_head_or_embed(name: str) -> bool:
    """Keep embeddings and the output projection out of width pruning."""
    lowered = name.lower()
    return any(k in lowered for k in ("embed", "lm_head", "output_projection"))


def _example_inputs(tokenizer):
    enc = tokenizer(["This is an example sentence."], return_tensors="pt")
    # Seq2seq DepGraph needs decoder_input_ids to trace the decoder subgraph.
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "decoder_input_ids": enc["input_ids"][:, :1],
    }


if __name__ == "__main__":
    main()
