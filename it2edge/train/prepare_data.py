"""Turn a 2-column English,<Target> CSV into the line-aligned corpus layout.

The fine-tune step (finetune_qlora.py) reads RAW line-aligned parallel files:

    en-indic-exp/
      train/eng_Latn-hin_Deva/train.eng_Latn   train.hin_Deva
      dev/eng_Latn-hin_Deva/dev.eng_Latn       dev.hin_Deva

This script converts a CSV like Dataset_English_Hindi.csv --

    English,Hindi
    Help!,बचाओ!
    Jump.,उछलो.
    ...

-- into exactly that layout: it reads the CSV (properly handling quoted,
comma- and newline-containing fields), shuffles deterministically, holds out a
small dev split, and writes the four files. Feed RAW text -- IndicProcessor does
normalization/tagging at train time, so do NOT transliterate or clean here.

Usage:
    python -m it2edge.train.prepare_data                      # defaults below
    python -m it2edge.train.prepare_data --csv Dataset_English_Hindi.csv \
        --tgt_lang hin_Deva --dev_frac 0.01 --out_dir en-indic-exp

Then fine-tune:
    python -m it2edge.train.finetune_qlora --data_dir en-indic-exp
"""

import argparse
import csv
import random

from it2edge.paths import PROJECT_ROOT

SRC_LANG = "eng_Latn"


def read_csv_pairs(csv_path, src_col, tgt_col):
    """Read (src, tgt) pairs from a CSV, skipping blank/incomplete rows.

    Uses csv.DictReader so quoted fields containing commas or embedded newlines
    (present in this corpus) are parsed correctly rather than split naively.
    Newlines inside a field are flattened to spaces so each pair stays on one
    physical line in the output files (the corpus format is line-aligned).
    """
    pairs = []
    skipped = 0
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if src_col not in reader.fieldnames or tgt_col not in reader.fieldnames:
            raise SystemExit(
                f"CSV columns {reader.fieldnames} do not include "
                f"'{src_col}' and '{tgt_col}'. Pass --src_col/--tgt_col."
            )
        for row in reader:
            src = (row.get(src_col) or "").strip()
            tgt = (row.get(tgt_col) or "").strip()
            if not src or not tgt:
                skipped += 1
                continue
            # Flatten any embedded newlines so one pair == one line each side.
            src = " ".join(src.split())
            tgt = " ".join(tgt.split())
            pairs.append((src, tgt))
    return pairs, skipped


def write_split(split_dir, split_name, tgt_lang, pairs):
    """Write <split>.eng_Latn and <split>.<tgt_lang> under eng_Latn-<tgt>/."""
    pair_dir = split_dir / f"{SRC_LANG}-{tgt_lang}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    src_path = pair_dir / f"{split_name}.{SRC_LANG}"
    tgt_path = pair_dir / f"{split_name}.{tgt_lang}"
    with open(src_path, "w", encoding="utf-8") as sf, \
            open(tgt_path, "w", encoding="utf-8") as tf:
        for src, tgt in pairs:
            sf.write(src + "\n")
            tf.write(tgt + "\n")
    print(f"[ok] {split_name}: {len(pairs)} pairs -> {pair_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a 2-column CSV into the line-aligned corpus layout"
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "Dataset_English_Hindi.csv"),
        help="path to the source CSV (default: Dataset_English_Hindi.csv at root)",
    )
    parser.add_argument(
        "--out_dir",
        default=str(PROJECT_ROOT / "en-indic-exp"),
        help="corpus root to create (default: en-indic-exp/ at project root)",
    )
    parser.add_argument("--tgt_lang", default="hin_Deva", help="target lang code")
    parser.add_argument("--src_col", default="English", help="source CSV column")
    parser.add_argument("--tgt_col", default="Hindi", help="target CSV column")
    parser.add_argument(
        "--dev_frac", type=float, default=0.01, help="fraction held out for dev"
    )
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed")
    args = parser.parse_args()

    from pathlib import Path
    out_dir = Path(args.out_dir)

    pairs, skipped = read_csv_pairs(args.csv, args.src_col, args.tgt_col)
    if not pairs:
        raise SystemExit(f"No usable pairs found in {args.csv}.")
    print(f"[info] read {len(pairs)} pairs ({skipped} blank/incomplete rows skipped)")

    random.seed(args.seed)
    random.shuffle(pairs)

    n_dev = int(len(pairs) * args.dev_frac)
    if 0 < args.dev_frac and n_dev == 0:
        n_dev = 1  # tiny corpus: still hold out at least one
    dev_pairs = pairs[:n_dev]
    train_pairs = pairs[n_dev:]

    write_split(out_dir / "train", "train", args.tgt_lang, train_pairs)
    if dev_pairs:
        write_split(out_dir / "dev", "dev", args.tgt_lang, dev_pairs)
    else:
        print("[info] dev_frac=0 -> no dev split written")

    print(f"\n[done] corpus ready at {out_dir}")
    print(f"       train={len(train_pairs)}  dev={len(dev_pairs)}  "
          f"(seed={args.seed})")
    print("       Next: python -m it2edge.train.finetune_qlora "
          f"--data_dir {out_dir.name}")


if __name__ == "__main__":
    main()
