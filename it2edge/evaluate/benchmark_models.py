"""Benchmark harness for MarianMT en→hi (quality on DEV, latency on Pi).

    python -m it2edge.evaluate.benchmark_models quality \
        --ct2_dir model_cache_compact_ct2 --data_dir en-indic-exp --out report.json

    python -m it2edge.evaluate.benchmark_models latency \
        --ct2_dir model_cache_compact_ct2 --data_dir en-indic-exp --intra 4
"""

import argparse
import json
import os
import re
import statistics
import time

SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")


def _dev_pairs(data_dir: str):
    base = os.path.join(data_dir, "dev", f"{SRC_LANG}-{TGT_LANG}")
    src_path = os.path.join(base, f"dev.{SRC_LANG}")
    ref_path = os.path.join(base, f"dev.{TGT_LANG}")
    if not (os.path.isfile(src_path) and os.path.isfile(ref_path)):
        raise SystemExit(f"Dev pair files not found under {base}.")
    with open(src_path, encoding="utf-8") as fh:
        srcs = [ln.rstrip("\n") for ln in fh]
    with open(ref_path, encoding="utf-8") as fh:
        refs = [ln.rstrip("\n") for ln in fh]
    if len(srcs) != len(refs):
        raise SystemExit("Dev src/ref line counts differ; corpus must be aligned.")
    return srcs, refs


def _load_model(ct2_dir: str, tokenizer_dir: str | None, inter: int, intra: int):
    from it2edge.serve.marian_ct2 import load_marian, translate_marian

    t0 = time.perf_counter()
    tokenizer, translator = load_marian(ct2_dir, tokenizer_dir, inter, intra)
    load_s = time.perf_counter() - t0

    def translate_fn(batch, beam_size=1):
        return translate_marian(batch, tokenizer, translator, beam_size=beam_size)

    return translate_fn, load_s


def _has_repetition(text: str, n: int = 3) -> bool:
    toks = text.split()
    if len(toks) < n * 2:
        return False
    seen = set()
    for i in range(len(toks) - n + 1):
        gram = tuple(toks[i : i + n])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def _safety_counts(srcs, outs):
    empty = untranslated = repetition = malformed = bad_ratio = 0
    for src, out in zip(srcs, outs):
        stripped = out.strip()
        if not stripped:
            empty += 1
            continue
        deva = len(_DEVANAGARI.findall(out))
        latin = len(_LATIN.findall(out))
        if deva == 0 or latin > deva:
            untranslated += 1
        if deva < max(1, 0.3 * len(stripped.replace(" ", ""))):
            malformed += 1
        if _has_repetition(out):
            repetition += 1
        ratio = len(stripped) / max(1, len(src.strip()))
        if ratio < 0.2 or ratio > 3.0:
            bad_ratio += 1
    total = len(srcs)
    return {
        "total": total,
        "empty": empty,
        "untranslated_english": untranslated,
        "repetition": repetition,
        "malformed_devanagari": malformed,
        "bad_length_ratio": bad_ratio,
        "unsafe_fraction": round(
            (empty + untranslated + repetition + malformed) / max(1, total), 4
        ),
    }


def run_quality(args):
    import sacrebleu

    srcs, refs = _dev_pairs(args.data_dir)
    translate_fn, load_s = _load_model(
        args.ct2_dir, args.tokenizer_dir, args.inter, args.intra
    )
    print(
        f"[info] Marian loaded in {load_s:.1f}s; translating "
        f"{len(srcs)} dev sentences (beam={args.beam_size}) ..."
    )

    outs = []
    bs = args.batch_size
    for i in range(0, len(srcs), bs):
        outs.extend(translate_fn(srcs[i : i + bs], beam_size=args.beam_size))
        if (i // bs) % 20 == 0:
            print(f"  {i}/{len(srcs)}")

    chrf = sacrebleu.corpus_chrf(outs, [refs], word_order=2).score
    bleu = sacrebleu.corpus_bleu(outs, [refs]).score
    safety = _safety_counts(srcs, outs)

    step = max(1, len(srcs) // 100)
    sample = [
        {"src": srcs[i], "ref": refs[i], "hyp": outs[i]}
        for i in range(0, len(srcs), step)
    ][:100]

    report = {
        "mode": "quality",
        "model": "marian",
        "ct2_dir": args.ct2_dir,
        "num_sentences": len(srcs),
        "beam_size": args.beam_size,
        "bleu": round(bleu, 2),
        "chrf_pp": round(chrf, 2),
        "safety": safety,
        "load_seconds": round(load_s, 2),
        "review_sample": sample,
    }
    _write(args.out, report)
    print(
        f"\n[ok] BLEU={bleu:.2f}  chrF++={chrf:.2f}  "
        f"unsafe={safety['unsafe_fraction'] * 100:.2f}%"
    )


def run_latency(args):
    try:
        import psutil
    except ImportError:
        psutil = None

    srcs, _ = _dev_pairs(args.data_dir)
    ordered = sorted(srcs, key=len)
    step = max(1, len(ordered) // args.num)
    sample = ordered[::step][: args.num]

    translate_fn, load_s = _load_model(
        args.ct2_dir, args.tokenizer_dir, args.inter, args.intra
    )

    for s in sample[: args.warmup]:
        translate_fn([s], beam_size=args.beam_size)

    lat = []
    for s in sample:
        t0 = time.perf_counter()
        translate_fn([s], beam_size=args.beam_size)
        lat.append((time.perf_counter() - t0) * 1000.0)

    lat.sort()
    p95 = lat[min(len(lat) - 1, int(round(0.95 * len(lat))) - 1)]
    peak_rss_mb = None
    if psutil is not None:
        peak_rss_mb = round(psutil.Process().memory_info().rss / 1e6, 1)

    ct2_bin = os.path.join(args.ct2_dir, "model.bin")
    artifact_mb = (
        round(os.path.getsize(ct2_bin) / 1e6, 1) if os.path.isfile(ct2_bin) else None
    )

    report = {
        "mode": "latency",
        "model": "marian",
        "ct2_dir": args.ct2_dir,
        "num_sentences": len(sample),
        "warmup_discarded": args.warmup,
        "beam_size": args.beam_size,
        "median_ms": round(statistics.median(lat), 1),
        "p95_ms": round(p95, 1),
        "min_ms": round(lat[0], 1),
        "max_ms": round(lat[-1], 1),
        "load_seconds": round(load_s, 2),
        "peak_rss_mb": peak_rss_mb,
        "artifact_model_bin_mb": artifact_mb,
    }
    _write(args.out, report)
    print(
        f"\n[ok] median={report['median_ms']}ms  p95={report['p95_ms']}ms  "
        f"rss={peak_rss_mb}MB  artifact={artifact_mb}MB"
    )


def _write(path, report):
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"[info] wrote {path}")
    else:
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "review_sample"},
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MarianMT en→hi (quality/latency)"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    def _common(p):
        p.add_argument("--ct2_dir", required=True, help="CT2 int8 model directory")
        p.add_argument(
            "--tokenizer_dir",
            default=None,
            help="tokenizer dir (default: same as --ct2_dir)",
        )
        p.add_argument("--data_dir", required=True, help="corpus root with dev/")
        p.add_argument("--beam_size", type=int, default=1)
        p.add_argument("--inter", type=int, default=1)
        p.add_argument("--intra", type=int, default=2)
        p.add_argument("--out", default=None, help="write JSON report here")

    pq = sub.add_parser("quality", help="BLEU/chrF++ + safety on the dev set")
    _common(pq)
    pq.add_argument("--batch_size", type=int, default=16)
    pq.set_defaults(func=run_quality)

    pl = sub.add_parser("latency", help="warm latency + RSS on the Pi")
    _common(pl)
    pl.add_argument("--num", type=int, default=50)
    pl.add_argument("--warmup", type=int, default=5)
    pl.set_defaults(func=run_latency)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
