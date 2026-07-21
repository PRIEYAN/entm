"""Shared helpers for line-aligned English–Hindi parallel corpora."""

import glob
import os

SRC_LANG = "eng_Latn"


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
