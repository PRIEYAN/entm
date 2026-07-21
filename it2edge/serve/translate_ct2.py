"""Run MarianMT en→hi via CTranslate2 int8 — Pi-friendly, no PyTorch.

    python -m it2edge.serve.translate_ct2 "Hello, how are you?"
    python -m it2edge.serve.translate_ct2 --beams 5 "Higher quality, slower."
    python -m it2edge.serve.translate_ct2
"""

import argparse

from it2edge.paths import CT2_DIR
from it2edge.serve.marian_ct2 import load_marian, translate_marian

DEFAULT_CT2_DIR = str(CT2_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MarianMT en→hi translator (CTranslate2 int8)"
    )
    parser.add_argument("text", nargs="*", help="English text to translate")
    parser.add_argument(
        "--beams",
        type=int,
        default=1,
        help="beam size; 1 = greedy (fastest, default)",
    )
    parser.add_argument(
        "--model_dir", default=DEFAULT_CT2_DIR, help="CT2 model directory"
    )
    parser.add_argument(
        "--tokenizer_dir",
        default=None,
        help="tokenizer directory (default: same as --model_dir)",
    )
    args = parser.parse_args()

    tokenizer, translator = load_marian(args.model_dir, args.tokenizer_dir)
    print(f"[info] loaded Marian CT2 int8 (beam_size={args.beams})")

    sentences = (
        [" ".join(args.text)]
        if args.text
        else [
            "Hello, how are you today?",
            "The weather is beautiful this morning.",
            "I would like a cup of coffee, please.",
        ]
    )
    out = translate_marian(sentences, tokenizer, translator, beam_size=args.beams)
    for src, dst in zip(sentences, out):
        print(f"\nEN: {src}\n-> {dst}")


if __name__ == "__main__":
    main()
