"""MarianMT English→Hindi inference over CTranslate2 int8 (Pi runtime).

Standard CTranslate2 Marian recipe:
    ids    = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    hyp    = translator.translate_batch([tokens])[0].hypotheses[0]
    text   = tokenizer.decode(tokenizer.convert_tokens_to_ids(hyp),
                              skip_special_tokens=True)
"""

# The Jetson Nano runs Python 3.6 — no PEP 604 unions (`str | None`) and no
# `from __future__ import annotations` (3.7+). Use typing.Optional instead.

import os
from typing import Optional

import ctranslate2
from transformers import AutoTokenizer


def load_marian(
    model_dir: str,
    tokenizer_dir: Optional[str] = None,
    inter_threads: int = 1,
    intra_threads: int = 2,
):
    """Load a CT2 Marian model + tokenizer. Returns (tokenizer, translator)."""
    if not os.path.isdir(model_dir):
        raise SystemExit(
            f"CT2 Marian model not found at {model_dir}. Convert it with "
            "`python -m it2edge.convert.convert_ct2` and copy it here."
        )
    tok_dir = tokenizer_dir or model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    translator = ctranslate2.Translator(
        model_dir,
        device="cpu",
        compute_type="int8",
        inter_threads=inter_threads,
        intra_threads=intra_threads,
    )
    return tokenizer, translator


def translate_marian(
    sentences,
    tokenizer,
    translator,
    beam_size: int = 1,
    max_decoding_length: int = 256,
):
    """Translate a list of English strings to Hindi. Returns a list of strings."""
    source_tokens = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(s)) for s in sentences
    ]
    results = translator.translate_batch(
        source_tokens,
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
        return_scores=False,
    )
    out = []
    for res in results:
        hyp_ids = tokenizer.convert_tokens_to_ids(res.hypotheses[0])
        out.append(tokenizer.decode(hyp_ids, skip_special_tokens=True))
    return out
