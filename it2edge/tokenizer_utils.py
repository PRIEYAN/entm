"""Shared robust loader for IndicTrans2's remote tokenizer.

Both translate.py (PyTorch) and translate_ct2.py (CTranslate2) use this so the
src_vocab_file reload collision is handled in exactly one place.

Root cause of that collision: the model's remote IndicTransTokenizer passes
src_vocab_file / tgt_vocab_file to super().__init__() as keywords. transformers
stores them in init_kwargs and, on reload, also replays them as the first
*positional* init_inputs -- so the subclass parameter gets the same value both
positionally and by keyword:
    TypeError: __init__() got multiple values for keyword argument 'src_vocab_file'
(Sometimes masked by a misleading protobuf ImportError.)

Loading from a clean hub snapshot (see download_model.py) avoids it because the
repo's original tokenizer_config.json has no vocab-file keys to replay. This
loader is the belt-and-suspenders fallback for copies that were saved with
save_pretrained (which does inject those keys).
"""

import os


def load_indictrans_tokenizer(path):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except (TypeError, ImportError) as exc:
        print(f"[warn] default tokenizer load failed ({exc}); retrying explicitly.")

    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    config = get_tokenizer_config(path, trust_remote_code=True)
    ref = config.get("auto_map", {}).get("AutoTokenizer")
    if isinstance(ref, (list, tuple)):
        ref = ref[0]
    if not ref:
        raise RuntimeError(
            "Could not locate the remote tokenizer class in tokenizer_config.json"
        )

    tok_cls = get_class_from_dynamic_module(ref, path, trust_remote_code=True)

    # The two vocab dicts shipped in the repo (verified filenames).
    def find(*names):
        for n in names:
            p = os.path.join(path, n)
            if os.path.isfile(p):
                return p
        return None

    src_vocab = find("dict.SRC.json", "dict.SRC", "model.SRC")
    tgt_vocab = find("dict.TGT.json", "dict.TGT", "model.TGT")

    # Instantiate with keyword-only args so nothing is passed twice.
    kwargs = {k: v for k, v in config.items() if k not in ("auto_map", "tokenizer_class")}
    kwargs.pop("src_vocab_file", None)
    kwargs.pop("tgt_vocab_file", None)
    if src_vocab:
        kwargs["src_vocab_fp"] = src_vocab
    if tgt_vocab:
        kwargs["tgt_vocab_fp"] = tgt_vocab

    try:
        return tok_cls(**kwargs)
    except TypeError:
        # Some revisions name the params src_vocab_file / tgt_vocab_file.
        kwargs.pop("src_vocab_fp", None)
        kwargs.pop("tgt_vocab_fp", None)
        if src_vocab:
            kwargs["src_vocab_file"] = src_vocab
        if tgt_vocab:
            kwargs["tgt_vocab_file"] = tgt_vocab
        return tok_cls(**kwargs)
