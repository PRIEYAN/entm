"""Convert IndicTrans2 (en-indic distilled 200M) to a CTranslate2 int8 package.

RUN THIS ON YOUR DEV MACHINE (x86-64), NOT on the Pi. It produces a small int8
CTranslate2 model directory you copy to a 64-bit Raspberry Pi and run with
it2edge.serve.translate_ct2 -- no PyTorch needed there.

    pip install "transformers>=4.51,<4.53" ctranslate2 torch sentencepiece protobuf
    python -m it2edge.convert.convert_ct2    # writes ./model_cache_ct2/

Then copy BOTH of these to the Pi:
    - model_cache_ct2/                              (the CT2 int8 weights)
    - model_cache/indictrans2-en-indic-dist-200M/   (the tokenizer files)

Why a custom loader (the real story)
------------------------------------
IndicTrans2 uses a custom HF arch (model_type "IndicTrans") that CT2's stock
converter does not know:  "No conversion is registered for ... IndicTransConfig".

It is NOT plain M2M100: it has SEPARATE source/target vocabularies --
  encoder.embed_tokens = [32322, 512]  (English source)
  decoder.embed_tokens = [122672, 512] (Indic target, tied to the output proj)
Re-presenting it as m2m_100 fails because M2M100 builds ONE shared embedding and
rejects the 32322-row encoder tensor (size mismatch).

But the layer layout is byte-for-byte BART/M2M100, and CTranslate2's BartLoader
already reads the encoder and decoder embeddings from their OWN modules (sized
per-tensor) and supports distinct source/target vocabularies. So the fix is to
LOAD THE REAL model via trust_remote_code (its own classes build the two
embeddings correctly and preserve the fine-tune) and hand it to a BartLoader
subclass registered for IndicTransConfig -- reusing CT2's verified layer wiring,
overriding only the dual-vocabulary registration.

Verified model facts (from the merged model on disk):
  config class: IndicTransConfig; encoder_vocab_size=32322, decoder_vocab_size=122672
  model.model.{encoder,decoder}; model.lm_head; no final_logits_bias
  encoder children: embed_tokens, embed_positions, layers, layer_norm, layernorm_embedding
  embed_positions: IndicTransSinusoidalPositionalEmbedding with .weights + .offset
  tokenizer: get_src_vocab()/get_tgt_vocab(), src_encoder/tgt_encoder dicts
"""

import argparse
import os
import sys

from it2edge.paths import CT2_DIR, HF_SNAPSHOT, MERGED_DIR, MODEL_ID

LOCAL_HF_DIR = str(HF_SNAPSHOT)
MERGED_MODEL_DIR = str(MERGED_DIR)
OUTPUT_DIR = str(CT2_DIR)

FALLBACK_MSG = """
[!] CTranslate2 conversion failed. If this is a NEW error (not the old
    IndicTransConfig one), paste it -- the custom loader below handles the
    known dual-vocabulary case. Alternative routes, if ever needed:
      * ONNX: python -m it2edge.convert.export_onnx  (needs a separate serve path)
      * fairseq -> CT2 (official AI4Bharat path; needs a fairseq checkpoint)
"""


def resolve_source(explicit):
    if explicit:
        return explicit
    if os.path.isdir(MERGED_MODEL_DIR):
        print(f"[info] using merged fine-tuned model at {MERGED_MODEL_DIR}")
        return MERGED_MODEL_DIR
    if os.path.isdir(LOCAL_HF_DIR):
        return LOCAL_HF_DIR
    print(f"[warn] {LOCAL_HF_DIR} not found; converting straight from the hub.")
    return MODEL_ID


def _register_indictrans_loader():
    """Register a CT2 loader for IndicTransConfig, subclassing BartLoader.

    Returns the RobustTransformersConverter class (with the dtype->torch_dtype
    shim). Importing/calling this runs @register_loader as a side effect, so the
    converter will dispatch IndicTransConfig to our loader.
    """
    from ctranslate2.converters import transformers as ct2t
    from ctranslate2.converters.transformers import (
        BartLoader,
        TransformersConverter,
        register_loader,
    )

    @register_loader("IndicTransConfig")
    class IndicTransLoader(BartLoader):
        @property
        def architecture_name(self):
            # CT2's _load does getattr(transformers, architecture_name) BEFORE
            # calling load_model. IndicTransForConditionalGeneration is NOT in
            # the transformers namespace (it lives in remote code), so name a
            # class that IS present as a decoy; load_model below ignores it and
            # loads the real model via AutoModelForSeq2SeqLM + trust_remote_code.
            return "M2M100ForConditionalGeneration"

        def get_model_spec(self, model):
            # Same spec shape as M2M100/BART: pre-norm, GELU, layernorm_embedding.
            spec = ct2t.transformer_spec.TransformerSpec.from_config(
                (model.config.encoder_layers, model.config.decoder_layers),
                model.config.encoder_attention_heads,
                pre_norm=True,
                activation=ct2t.common_spec.Activation.GELU,
                layernorm_embedding=True,
            )
            # BartLoader's verified wiring reads enc/dec embeddings from their own
            # (differently sized) modules -- exactly what dual-vocab needs.
            self.set_encoder(spec.encoder, model.model.encoder)
            self.set_decoder(spec.decoder, model.model.decoder)
            self.set_linear(spec.decoder.projection, model.lm_head)
            return spec

        def set_position_encodings(self, spec, module):
            # IndicTransSinusoidalPositionalEmbedding stores its table as
            # `.weights` (plural) with an `.offset`, exactly like M2M100 -- not
            # `.weight` as BartLoader's base method assumes. Read it the M2M100 way.
            spec.encodings = module.weights[module.offset:]

        def get_vocabulary(self, model, tokenizer):
            # Return the two vocabularies as (source, target) ordered token lists.
            # Order MUST match embedding rows == HF token ids, so we invert the
            # tokenizer's src/tgt encoder dicts (token -> id) by id.
            def ordered(token_to_id, n):
                toks = [None] * n
                for tok, idx in token_to_id.items():
                    if 0 <= idx < n:
                        toks[idx] = str(tok)  # CT2 vocab entries must be strings
                # Fill any gaps so the count matches the embedding rows exactly.
                for i in range(n):
                    if toks[i] is None:
                        toks[i] = f"<madeupword{i}>"
                return toks

            # Use the raw {token: id} encoder dicts directly. Do NOT call
            # get_src_vocab()/get_tgt_vocab(): those do
            # dict(self.tgt_encoder, **self.added_tokens_decoder), and
            # added_tokens_decoder is keyed by INT ids, so the ** splat raises
            # "TypeError: keywords must be strings". CT2 only needs the base
            # encoder vocab in id order to size the source/target embeddings; any
            # added special tokens are already rows in tgt_encoder/src_encoder.
            src = ordered(tokenizer.src_encoder, model.config.encoder_vocab_size)
            tgt = ordered(tokenizer.tgt_encoder, model.config.decoder_vocab_size)
            return src, tgt

        def set_vocabulary(self, spec, tokens):
            src, tgt = tokens
            spec.register_source_vocabulary(src)
            spec.register_target_vocabulary(tgt)

        def set_config(self, config, model, tokenizer):
            config.bos_token = tokenizer.bos_token
            config.eos_token = tokenizer.eos_token
            config.unk_token = tokenizer.unk_token
            config.decoder_start_token = tokenizer.convert_ids_to_tokens(
                model.config.decoder_start_token_id
            )

    class RobustTransformersConverter(TransformersConverter):
        """Load the REAL IndicTrans model (not the M2M100 decoy) and bridge the
        dtype/torch_dtype kwarg mismatch (new ct2 vs transformers<4.53)."""

        def load_model(self, model_class, model_name_or_path, **kwargs):
            from transformers import AutoModelForSeq2SeqLM

            if "dtype" in kwargs:
                kwargs.setdefault("torch_dtype", kwargs.pop("dtype"))
            # Ignore model_class (the M2M100 decoy from architecture_name) and
            # build the actual IndicTrans model via its remote code.
            kwargs.setdefault("trust_remote_code", True)
            return AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **kwargs)

    return RobustTransformersConverter


def main():
    parser = argparse.ArgumentParser(
        description="Convert IndicTrans2 (HF) to a CTranslate2 int8 package"
    )
    parser.add_argument("--model", default=None, help="model path/id (default: merged)")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--quantization", default="int8")
    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        raise SystemExit(
            f"{args.output_dir} already exists. Remove it first to re-convert."
        )

    source = resolve_source(args.model)

    try:
        converter_cls = _register_indictrans_loader()
    except ImportError:
        raise SystemExit("ctranslate2 is not installed:  pip install ctranslate2")

    print(f"[info] converting {source} -> {args.output_dir} "
          f"(quantization={args.quantization}, custom IndicTrans dual-vocab loader)")
    try:
        # trust_remote_code=True so from_pretrained builds the REAL IndicTrans
        # model (two separate embeddings); our registered loader handles it.
        converter = converter_cls(source, trust_remote_code=True)
        converter.convert(args.output_dir, quantization=args.quantization, force=True)
    except Exception as exc:
        print(f"[error] conversion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(FALLBACK_MSG, file=sys.stderr)
        raise SystemExit(1)

    print(f"\n[ok] int8 CTranslate2 model written to: {args.output_dir}")
    print("     Next: verify it loads, then A/B against the torch path:")
    print("       python -m it2edge.serve.translate_ct2 \"Jump\"")
    print("       python -m it2edge.serve.translate     --model merged \"Jump\"")
    print("[!]  int8 is lossy -- compare a few sentences before shipping to the Pi.")


if __name__ == "__main__":
    main()
