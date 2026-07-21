"""A/B quality comparison of English->Hindi Marian models on the project dev set.

Scores candidates on the SAME held-out pairs so you pick the base with numbers,
not model-card claims. Runs on the DEV laptop with Hugging Face Transformers
(torch) -- no CT2 conversion required for this step.

Default candidates:
  * primary  -- AI4INDIANS/better-opus-mt-en-hi (local snapshot if present)
  * control  -- Helsinki-NLP/opus-mt-en-hi (local snapshot if present)

Optional third model via --extra (local path or hub id), e.g. an HPLT checkpoint.

    python -m it2edge.download_model --all
    python -m it2edge.evaluate.ab_compare_models --data_dir en-indic-exp
    python -m it2edge.evaluate.ab_compare_models --data_dir en-indic-exp \\
        --extra HPLT/translate-en-hi-v1.0-hplt --out ab_report.json

Smoke (first N pairs):
    python -m it2edge.evaluate.ab_compare_models --data_dir en-indic-exp --max_samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from it2edge.evaluate.benchmark_models import _dev_pairs, _safety_counts
from it2edge.paths import (
    CONTROL_MODEL_ID,
    CONTROL_SNAPSHOT,
    HF_SNAPSHOT,
    MODEL_ID,
    PROJECT_ROOT,
)


def _resolve_model(label: str, preferred_local: Path | None, fallback_id: str) -> str:
    if preferred_local is not None and preferred_local.is_dir():
        # Prefer a real weight file so we don't treat an empty dir as ready.
        if any(preferred_local.glob("*.safetensors")) or any(
            preferred_local.glob("pytorch_model.bin")
        ):
            return str(preferred_local)
    return fallback_id


def _ensure_safetensors(model_path: str) -> None:
    """If a local Marian dir only has pytorch_model.bin, write model.safetensors.

    transformers + torch<2.6 refuse to load .bin weights (CVE-2025-32434).
    Official OPUS snapshots often ship only .bin; convert once in-place.
    """
    root = Path(model_path)
    if not root.is_dir():
        return
    if any(root.glob("*.safetensors")):
        return
    bin_path = root / "pytorch_model.bin"
    if not bin_path.is_file():
        return

    import torch
    from safetensors.torch import save_model
    from transformers import AutoConfig, AutoModelForSeq2SeqLM

    print(f"[info] converting {bin_path.name} -> model.safetensors (one-time) ...")
    cfg = AutoConfig.from_pretrained(str(root))
    model = AutoModelForSeq2SeqLM.from_config(cfg)
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    save_model(model, str(root / "model.safetensors"))
    print(f"[ok] wrote {root / 'model.safetensors'}")


def _load_hf_translator(model_path: str, device: str, max_length: int, beam_size: int):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _ensure_safetensors(model_path)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    load_s = time.perf_counter() - t0

    @torch.inference_mode()
    def translate_fn(batch: list[str]) -> list[str]:
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model.generate(
            **enc,
            num_beams=beam_size,
            max_new_tokens=max_length,
        )
        return tokenizer.batch_decode(out, skip_special_tokens=True)

    return translate_fn, load_s


def _score_one(
    label: str,
    model_path: str,
    srcs: list[str],
    refs: list[str],
    device: str,
    batch_size: int,
    beam_size: int,
    max_length: int,
) -> dict:
    import sacrebleu

    print(f"\n=== {label} ===")
    print(f"[info] loading {model_path} on {device} ...")
    translate_fn, load_s = _load_hf_translator(
        model_path, device, max_length=max_length, beam_size=beam_size
    )
    print(f"[info] loaded in {load_s:.1f}s; translating {len(srcs)} sentences ...")

    outs: list[str] = []
    t0 = time.perf_counter()
    for i in range(0, len(srcs), batch_size):
        outs.extend(translate_fn(srcs[i : i + batch_size]))
        if (i // batch_size) % 10 == 0:
            print(f"  {i}/{len(srcs)}")
    decode_s = time.perf_counter() - t0

    bleu = sacrebleu.corpus_bleu(outs, [refs]).score
    chrf = sacrebleu.corpus_chrf(outs, [refs], word_order=2).score
    safety = _safety_counts(srcs, outs)

    # Evenly spaced sample for human spot-check.
    step = max(1, len(srcs) // 20)
    sample = [
        {"src": srcs[i], "ref": refs[i], "hyp": outs[i]}
        for i in range(0, len(srcs), step)
    ][:20]

    result = {
        "label": label,
        "model_path": model_path,
        "num_sentences": len(srcs),
        "beam_size": beam_size,
        "bleu": round(bleu, 2),
        "chrf_pp": round(chrf, 2),
        "safety": safety,
        "load_seconds": round(load_s, 2),
        "decode_seconds": round(decode_s, 2),
        "review_sample": sample,
    }
    print(
        f"[ok] {label}: BLEU={bleu:.2f}  chrF++={chrf:.2f}  "
        f"unsafe={safety['unsafe_fraction'] * 100:.2f}%  "
        f"decode={decode_s:.1f}s"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B compare en-hi Marian models on the project dev set"
    )
    parser.add_argument(
        "--data_dir",
        default=str(PROJECT_ROOT / "en-indic-exp"),
        help="corpus root with dev/ (default: en-indic-exp)",
    )
    parser.add_argument(
        "--primary",
        default=None,
        help="primary model path/id (default: local better-opus snapshot or hub id)",
    )
    parser.add_argument(
        "--control",
        default=None,
        help="control model path/id (default: local opus-mt-en-hi or hub id)",
    )
    parser.add_argument(
        "--extra",
        default=None,
        help="optional third model path/id (e.g. HPLT/translate-en-hi-v1.0-hplt)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="score only the first N pairs (smoke)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda|cpu (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "ab_report.json"),
        help="write JSON comparison report here",
    )
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    srcs, refs = _dev_pairs(args.data_dir)
    if args.max_samples:
        srcs, refs = srcs[: args.max_samples], refs[: args.max_samples]
        print(f"[info] capped to {len(srcs)} pairs (--max_samples)")

    candidates: list[tuple[str, str]] = [
        (
            "primary_better_opus",
            args.primary
            or _resolve_model("primary", HF_SNAPSHOT, MODEL_ID),
        ),
        (
            "control_opus_mt_en_hi",
            args.control
            or _resolve_model("control", CONTROL_SNAPSHOT, CONTROL_MODEL_ID),
        ),
    ]
    if args.extra:
        candidates.append(("extra", args.extra))

    results = []
    for label, path in candidates:
        try:
            results.append(
                _score_one(
                    label,
                    path,
                    srcs,
                    refs,
                    device=device,
                    batch_size=args.batch_size,
                    beam_size=args.beam_size,
                    max_length=args.max_length,
                )
            )
        except Exception as exc:
            print(f"[error] {label} ({path}): {type(exc).__name__}: {exc}")
            results.append(
                {
                    "label": label,
                    "model_path": path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    # Rank successful runs by chrF++ then BLEU.
    scored = [r for r in results if "chrf_pp" in r]
    scored.sort(key=lambda r: (r["chrf_pp"], r["bleu"]), reverse=True)
    winner = scored[0]["label"] if scored else None

    report = {
        "data_dir": os.path.abspath(args.data_dir),
        "num_sentences": len(srcs),
        "device": device,
        "beam_size": args.beam_size,
        "winner_by_chrf_pp": winner,
        "ranking": [
            {
                "label": r["label"],
                "chrf_pp": r.get("chrf_pp"),
                "bleu": r.get("bleu"),
                "unsafe_fraction": (r.get("safety") or {}).get("unsafe_fraction"),
            }
            for r in scored
        ],
        "results": results,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] wrote {out_path}")
    if winner:
        print(f"[ok] winner by chrF++: {winner}")
        print("     Use that as --base_model for finetune_marian if it is not already primary.")


if __name__ == "__main__":
    main()
