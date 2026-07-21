"""Compact en->hi MarianMT inference over CTranslate2 int8 (Pi runtime).

This is the on-device translation path for the 1 GB Raspberry Pi. Marian is a
plain shared-vocab seq2seq model, so -- unlike the IndicTrans2 path -- there is
NO IndicProcessor and NO dual-vocab detokenize workaround. It needs only
ctranslate2 + transformers (MarianTokenizer); IndicTransToolkit is not required.

Standard CTranslate2 Marian recipe (OpenNMT transformers guide):
    ids    = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    hyp    = translator.translate_batch([tokens])[0].hypotheses[0]
    text   = tokenizer.decode(tokenizer.convert_tokens_to_ids(hyp),
                              skip_special_tokens=True)
"""

import os

import ctranslate2
from transformers import AutoTokenizer


def load_marian(model_dir: str, tokenizer_dir: str | None = None,
                inter_threads: int = 1, intra_threads: int = 2):
    """Load a CT2 Marian model + its tokenizer. Returns (tokenizer, translator).

    tokenizer_dir defaults to model_dir because convert_compact_ct2 copies the
    Marian tokenizer files into the CT2 directory.
    """
    if not os.path.isdir(model_dir):
        raise SystemExit(
            f"CT2 Marian model not found at {model_dir}. Convert it with "
            "`python -m it2edge.convert.convert_compact_ct2` and copy it here."
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


def translate_marian(sentences, tokenizer, translator, beam_size: int = 1,
                     max_decoding_length: int = 256):
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
