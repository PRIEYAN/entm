"""Convert the FINE-TUNED dual-vocab IndicTrans2 to a CTranslate2 int8 package.

THE CORRECT CT2 ROUTE for this model. Unlike convert_ct2.py (which restages the
model as m2m_100 and dies on the shared-vocab assumption), this registers a
custom CT2 loader keyed on IndicTrans2's own config class and loads the REAL
trust_remote_code model -- whose encoder/decoder embeddings are already the two
different sizes (32322 source / 122672 target). We copy each embedding from its
own module and register the source and target vocabularies SEPARATELY, which
CTranslate2's TransformerSpec supports natively.

Why this works where convert_ct2.py fails
------------------------------------------
* CT2's whole converter operates on a *loaded torch nn.Module*, chosen by the
  config's Python class name (`config.__class__.__name__`). BartLoader already
  copies encoder and decoder `embed_tokens` from SEPARATE modules -- it never
  assumes they are the same size. Only `set_vocabulary` (which feeds ONE token
  list to both register_* calls) and `get_vocabulary` (single tokenizer.vocab)
  hard-code the shared-vocab assumption.
* The M2M100 restage failed *before CT2 even ran*: rewriting config.json to
  model_type=m2m_100 makes transformers build a single [vocab_size,512]
  M2M100ScaledWordEmbedding, which cannot ingest the [32322,512] encoder
  checkpoint -> "size mismatch". The real IndicTrans class has no such problem.
* validate() checks token COUNTS against get_source_vocabulary_size() (=encoder
  embed rows, 32322) and get_target_vocabulary_size() (=decoder embed rows,
  122672). So we must register 32322 source tokens and 122672 target tokens.

Run on the dev box:
    pip install "transformers>=4.51,<4.53" torch ctranslate2 sentencepiece protobuf
    python -m it2edge.convert.convert_ct2_dualvocab   # writes ./model_cache_ct2/

VERIFY the field names against your checkpoint's modeling_indictrans.py before
trusting the output (see the asserts in get_model_spec). If an attribute name
differs, the assert points at the exact spot to fix.
"""

import argparse
import math
import os
import sys

from it2edge.paths import CT2_DIR, HF_SNAPSHOT, MERGED_DIR, MODEL_ID


def _register_loader():
    """Register a dual-vocab loader under IndicTrans2's config class name.

    Returns the config class name string it registered under, so the converter
    dispatch (which keys on config.__class__.__name__) finds it.
    """
    from ctranslate2.converters import transformers as ct2t
    from ctranslate2.specs import transformer_spec

    _ACT = ct2t._SUPPORTED_ACTIVATIONS  # {"gelu": Activation.GELU, ...}

    class IndicTrans2Loader(ct2t.BartLoader):
        # architecture_name -> the model class CT2 will from_pretrained().
        @property
        def architecture_name(self):
            return "IndicTransForConditionalGeneration"

        def load_model(self, model_class, model_name_or_path, **kwargs):
            # transformers<4.53 knows torch_dtype=, not dtype= (see convert_ct2).
            if "dtype" in kwargs:
                kwargs.setdefault("torch_dtype", kwargs.pop("dtype"))
            return model_class.from_pretrained(
                model_name_or_path, trust_remote_code=True, **kwargs
            )

        def get_model_spec(self, model):
            cfg = model.config
            self._cfg = cfg  # used by set_common_layers for embed_scale
            pre_norm = getattr(cfg, "encoder_normalize_before", True)
            act = _ACT[getattr(cfg, "activation_function", "gelu")]
            spec = transformer_spec.TransformerSpec.from_config(
                (cfg.encoder_layers, cfg.decoder_layers),
                cfg.encoder_attention_heads,
                pre_norm=pre_norm,
                activation=act,
                layernorm_embedding=getattr(cfg, "layernorm_embedding", True),
            )
            # model.model.{encoder,decoder}; model.lm_head -- BartLoader layout.
            enc = model.model.encoder
            dec = model.model.decoder
            assert hasattr(enc, "embed_tokens") and hasattr(dec, "embed_tokens"), (
                "encoder/decoder embed_tokens not found -- check modeling_indictrans.py "
                "attribute names against BartLoader.set_common_layers()."
            )
            self.set_encoder(spec.encoder, enc)   # copies the 32322-row embedding
            self.set_decoder(spec.decoder, dec)   # copies the 122672-row embedding

            # Tied output projection (share_decoder_input_output_embed=True).
            # Prefer an explicit lm_head if the arch exposes one; otherwise tie
            # to the decoder input embedding (both [122672,512]). An nn.Embedding
            # has a .weight of shape [vocab,dim], exactly the projection weight,
            # so set_linear (which only reads .weight/.bias) accepts it directly.
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None and getattr(lm_head, "weight", None) is not None:
                self.set_linear(spec.decoder.projection, lm_head)
            else:
                self.set_linear(spec.decoder.projection, dec.embed_tokens)
            return spec

        def set_position_encodings(self, spec, module):
            # Sinusoidal table lives on the module as a buffer. IndicTrans2's
            # class mirrors M2M100SinusoidalPositionalEmbedding: `.weights` with
            # an `.offset` (usually 2 for pad/bos). Fall back gracefully.
            if hasattr(module, "weights"):
                offset = getattr(module, "offset", 0)
                spec.encodings = module.weights[offset:]
            elif hasattr(module, "weight"):
                spec.encodings = module.weight
            else:
                raise AttributeError(
                    "positional-embedding module exposes neither .weights nor "
                    ".weight -- inspect modeling_indictrans.py and adjust."
                )

        def set_common_layers(self, spec, module):
            # Override BartLoader's version because IndicTrans2's encoder/decoder
            # submodules may not carry .config or .embed_scale. Read the scale
            # from the top-level config we captured, or off the module if present.
            if hasattr(module, "embed_scale"):
                embed_scale = module.embed_scale
            else:
                d_model = getattr(self._cfg, "d_model", None) or getattr(
                    self._cfg, "encoder_embed_dim", 512
                )
                embed_scale = (
                    math.sqrt(d_model)
                    if getattr(self._cfg, "scale_embedding", True)
                    else 1.0
                )
            spec.scale_embeddings = embed_scale
            self.set_position_encodings(spec.position_encodings, module.embed_positions)
            self.set_embeddings(
                spec.embeddings[0] if isinstance(spec.embeddings, list) else spec.embeddings,
                module.embed_tokens,
            )
            if getattr(module, "layer_norm", None) is not None:
                self.set_layer_norm(spec.layer_norm, module.layer_norm)
            if getattr(module, "layernorm_embedding", None) is not None:
                self.set_layer_norm(spec.layernorm_embedding, module.layernorm_embedding)

        def get_vocabulary(self, model, tokenizer):
            # THE CRUX: two different vocabularies. The IndicTrans2 tokenizer
            # keeps separate src/tgt SentencePiece dicts (dict.SRC.json /
            # dict.TGT.json). We do NOT use this single-list method; set_vocabulary
            # below builds both lists explicitly. Return [] so nothing relies on it.
            return []

        def set_vocabulary(self, spec, tokens):
            src, tgt = self._src_tokens, self._tgt_tokens
            spec.register_source_vocabulary(src)   # must be 32322 long
            spec.register_target_vocabulary(tgt)   # must be 122672 long

    return IndicTrans2Loader


def _build_vocab_lists(tokenizer, enc_rows, dec_rows, model_dir):
    """Return (src_tokens, tgt_tokens) in EMBEDDING-ROW order.

    CORRECTNESS, not just count: translate_ct2.py runs entirely in token-STRING
    space -- it feeds CT2 the source subword strings (tokenizer.convert_ids_to_
    tokens) and turns CT2's emitted target strings back into ids
    (tokenizer.convert_tokens_to_ids). CT2 maps a token string to an embedding
    row by that string's POSITION in source_vocabulary / target_vocabulary. So
    the string at row i here MUST equal the string the HF tokenizer assigns to
    id i, or every token lands on the wrong row and output is garbage.

    The authoritative id->token map is the fairseq dicts the tokenizer itself
    loads: dict.SRC.json / dict.TGT.json ({token: id}). We order by id. That is
    exactly the ordering the model was trained with and the HF tokenizer uses,
    so CT2 <-> HF token strings line up. SentencePiece id_to_piece order is a
    LAST resort (it omits the fairseq specials/offset and can misalign).
    """
    import json

    def from_dict_json(*names):
        for n in names:
            p = os.path.join(model_dir, n)
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as fh:
                    d = json.load(fh)
                if isinstance(d, dict):
                    # {token: id} -> id-ordered list
                    return [t for t, _ in sorted(d.items(), key=lambda kv: kv[1])]
        return None

    src = from_dict_json("dict.SRC.json", "dict.SRC")
    tgt = from_dict_json("dict.TGT.json", "dict.TGT")

    # Fallback ONLY if the dict files are absent: SentencePiece piece order.
    if src is None or tgt is None:
        print("[warn] dict.SRC.json/dict.TGT.json not found; falling back to "
              "SentencePiece piece order -- VERIFY output, rows may misalign.")

        def _sp_tokens(*attrs):
            for a in attrs:
                sp = getattr(tokenizer, a, None)
                if sp is not None:
                    return [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
            return None

        if src is None:
            src = _sp_tokens("src_spm", "src_sp_model", "sp_model_src", "source_spm")
        if tgt is None:
            tgt = _sp_tokens("tgt_spm", "tgt_sp_model", "sp_model_tgt", "target_spm")

    def _fit(lst, n, name):
        if lst is None:
            lst = ["madeupword%d" % i for i in range(n)]
        if len(lst) < n:
            lst = lst + ["madeupword%d" % i for i in range(n - len(lst))]
        elif len(lst) > n:
            print(f"[warn] {name} vocab {len(lst)} > {n} embed rows; truncating.")
            lst = lst[:n]
        return lst

    return _fit(src, enc_rows, "source"), _fit(tgt, dec_rows, "target")


def resolve_source(explicit):
    if explicit:
        return explicit
    if os.path.isdir(str(MERGED_DIR)):
        print(f"[info] using merged fine-tuned model at {MERGED_DIR}")
        return str(MERGED_DIR)
    if os.path.isdir(str(HF_SNAPSHOT)):
        return str(HF_SNAPSHOT)
    return MODEL_ID


def main():
    p = argparse.ArgumentParser(description="Dual-vocab IndicTrans2 -> CT2 int8")
    p.add_argument("--model", default=None)
    p.add_argument("--output_dir", default=str(CT2_DIR))
    p.add_argument("--quantization", default="int8")
    args = p.parse_args()

    if os.path.isdir(args.output_dir):
        raise SystemExit(f"{args.output_dir} exists; remove it to re-convert.")

    source = resolve_source(args.model)
    Loader = _register_loader()

    from ctranslate2.converters.transformers import TransformersConverter
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    # The converter dispatches on config.__class__.__name__. Register our loader
    # under that exact name so it is selected instead of failing as "unsupported".
    from ctranslate2.converters import transformers as ct2t
    cfg = AutoConfig.from_pretrained(source, trust_remote_code=True)
    ct2t._MODEL_LOADERS[cfg.__class__.__name__] = Loader()

    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    enc_rows = int(getattr(cfg, "encoder_vocab_size", 32322))
    dec_rows = int(getattr(cfg, "decoder_vocab_size", getattr(cfg, "vocab_size", 122672)))
    src_tokens, tgt_tokens = _build_vocab_lists(tok, enc_rows, dec_rows, source)

    loader = ct2t._MODEL_LOADERS[cfg.__class__.__name__]
    loader._src_tokens = src_tokens
    loader._tgt_tokens = tgt_tokens
    print(f"[info] source vocab={len(src_tokens)} target vocab={len(tgt_tokens)}")

    # CRITICAL: the stock TransformersConverter._load() does
    #     model_class = getattr(transformers, loader.architecture_name)
    # which raises AttributeError for "IndicTransForConditionalGeneration"
    # (that class lives in remote code, NOT the transformers namespace). So we
    # subclass _load() to build the model with AutoModelForSeq2SeqLM +
    # trust_remote_code, then hand it to our already-registered loader. This
    # never touches the M2M100 path, so the dual embeddings load as trained.
    class IndicTrans2Converter(TransformersConverter):
        def _load(self):
            import torch

            with torch.no_grad():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    source, trust_remote_code=True, torch_dtype=cfg.torch_dtype
                    if getattr(cfg, "torch_dtype", None) else None,
                )
                model.eval()
                spec = loader(model, tok)  # __call__ -> get_model_spec + set_vocabulary
                return spec

    try:
        conv = IndicTrans2Converter(source, trust_remote_code=True)
        conv.convert(args.output_dir, quantization=args.quantization, force=True)
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(f"\n[ok] int8 CT2 (dual-vocab) written to {args.output_dir}")
    print("     A/B a few sentences against it2edge.serve.translate before shipping.")


if __name__ == "__main__":
    main()
