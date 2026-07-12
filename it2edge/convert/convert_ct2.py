"""Convert IndicTrans2 (en-indic distilled 200M) to a CTranslate2 int8 package.

RUN THIS ON YOUR DEV MACHINE (x86-64, e.g. this Windows box), NOT on the Pi.
It produces a small int8 CTranslate2 model directory that you then copy to a
64-bit Raspberry Pi and run with it2edge.serve.translate_ct2 -- no PyTorch there.

Why: a 200M model in fp32 is ~800 MB and needs PyTorch, which is too heavy for
a 1 GB Pi 3. The int8 CTranslate2 package is ~200 MB and runs on the light
ctranslate2 CPU runtime, which has aarch64 wheels (64-bit Raspberry Pi OS).

    pip install ctranslate2 transformers sentencepiece protobuf
    python -m it2edge.convert.convert_ct2    # writes ./model_cache_ct2/

Then copy BOTH of these to the Pi:
    - model_cache_ct2/                         (the CT2 int8 weights)
    - model_cache/indictrans2-en-indic-dist-200M/   (the tokenizer files)

IMPORTANT compatibility note
----------------------------
IndicTrans2 ships a *custom* HF architecture (model_type "IndicTrans"), which
the stock ct2-transformers-converter does NOT recognise:

    ValueError: No conversion is registered for the model configuration
    IndicTransConfig

But IndicTrans2 is architecturally M2M100: its weight tensor names are an exact
match for M2M100/MBart (self_attn.{q,k,v,out}_proj, encoder_attn.*, fc1/fc2,
self_attn_layer_norm, final_layer_norm, layernorm_embedding, sinusoidal
positional embeddings created at load with no stored weight). So the default
here (--force_m2m100, on) STAGES a copy of the model with config.json rewritten
to present as m2m_100, then converts with CT2's native M2M100 loader (no
--trust_remote_code). No weight tensors are renamed -- only config keys.

Pass --no_force_m2m100 to try the plain converter instead (will fail on this
model, but useful if a future CT2 registers IndicTransConfig natively).
"""

import argparse
import json
import os
import shutil
import sys

from it2edge.paths import CT2_DIR, HF_SNAPSHOT, MERGED_DIR, MODEL_ID

LOCAL_HF_DIR = str(HF_SNAPSHOT)
MERGED_MODEL_DIR = str(MERGED_DIR)
OUTPUT_DIR = str(CT2_DIR)

FALLBACK_MSG = f"""
[!] The Hugging Face -> CTranslate2 conversion failed. This usually means the
    generic converter does not recognise IndicTrans2's custom architecture.

    Official fallback (AI4Bharat ship CT2 ports directly):
      1. Get the fairseq checkpoint + CT-ported dirs from the IndicTrans2 repo:
             https://github.com/AI4Bharat/IndicTrans2   (see the models table)
         The distilled en-indic download already contains 2 CT-ported dirs.
      2. Or convert a fairseq checkpoint yourself with the fairseq converter:
             https://opennmt.net/CTranslate2/guides/fairseq.html
      3. Copy the resulting CT2 dir to the Pi and point it2edge.serve.translate_ct2
         at it via --model_dir.
"""


def resolve_source(explicit: str | None) -> str:
    if explicit:
        return explicit
    # Prefer a fine-tuned + merged model over the stock snapshot (plan §6a).
    if os.path.isdir(MERGED_MODEL_DIR):
        print(f"[info] using merged fine-tuned model at {MERGED_MODEL_DIR}")
        return MERGED_MODEL_DIR
    if os.path.isdir(LOCAL_HF_DIR):
        return LOCAL_HF_DIR
    print(f"[warn] {LOCAL_HF_DIR} not found; converting straight from the hub.")
    print("       (Run `python -m it2edge.download_model` first to avoid a re-download.)")
    return MODEL_ID


def stage_as_m2m100(source: str, stage_dir: str) -> str:
    """Copy `source` to `stage_dir` and rewrite config.json to present as m2m_100.

    IndicTrans2's weights already follow the M2M100 layout; only the config
    identifies it as a custom arch. We copy everything, then patch config.json:
      * model_type / architectures -> m2m_100
      * add the M2M100 attribute names the CT2 loader reads (d_model,
        max_position_embeddings) aliased from the fairseq-style ones
      * drop auto_map so the converter does NOT try to import the remote code
    Returns stage_dir. Does not touch weight files.
    """
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    if os.path.isdir(source):
        shutil.copytree(source, stage_dir)
    else:
        raise SystemExit(
            f"--force_m2m100 needs a local model dir, got '{source}'. Download it "
            "first with `python -m it2edge.download_model`, or pass --model <dir>."
        )

    cfg_path = os.path.join(stage_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    d_model = cfg.get("encoder_embed_dim") or cfg.get("d_model")
    max_pos = max(
        cfg.get("max_source_positions", 0),
        cfg.get("max_target_positions", 0),
    ) or cfg.get("max_position_embeddings", 1024)

    # --- present as vanilla M2M100 so CT2's native loader picks it up ---
    cfg["model_type"] = "m2m_100"
    cfg["architectures"] = ["M2M100ForConditionalGeneration"]
    cfg["d_model"] = d_model
    cfg["max_position_embeddings"] = max_pos
    # These are the names the M2M100 loader / config expects; alias them across
    # in case they are only present under the fairseq-style spelling.
    cfg.setdefault("encoder_ffn_dim", cfg.get("encoder_ffn_dim"))
    cfg.setdefault("decoder_ffn_dim", cfg.get("decoder_ffn_dim"))
    cfg["scale_embedding"] = cfg.get("scale_embedding", True)
    cfg["activation_function"] = cfg.get("activation_function", "gelu")
    # Do NOT let the converter import the custom remote classes.
    cfg.pop("auto_map", None)
    cfg.pop("tokenizer_class", None)

    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)

    print(f"[info] staged M2M100-shaped copy at {stage_dir} "
          f"(d_model={d_model}, max_position_embeddings={max_pos})")
    return stage_dir


def _build_converter(source, trust_remote_code):
    """CT2 TransformersConverter that bridges new-ct2 vs transformers<4.53.

    Newer ctranslate2 (>=~4.7) calls model_class.from_pretrained(dtype=...), but
    transformers<4.53 only knows torch_dtype=; the stray `dtype` then leaks into
    M2M100ForConditionalGeneration.__init__ and raises
        TypeError: __init__() got an unexpected keyword argument 'dtype'
    load_model is ct2's signature-stable override point. Remapping
    dtype->torch_dtype preserves the load dtype (so int8 quant stays faithful)
    and is a harmless no-op on older ct2 that already passes torch_dtype.
    """
    from ctranslate2.converters.transformers import TransformersConverter

    class RobustTransformersConverter(TransformersConverter):
        def load_model(self, model_class, model_name_or_path, **kwargs):
            if "dtype" in kwargs:
                kwargs.setdefault("torch_dtype", kwargs.pop("dtype"))
            return model_class.from_pretrained(model_name_or_path, **kwargs)

    return RobustTransformersConverter(source, trust_remote_code=trust_remote_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert IndicTrans2 (HF) to a CTranslate2 int8 package"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model path/id to convert (default: merged, else stock snapshot, else hub)",
    )
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="CT2 output directory")
    # On ARM int8 resolves to int8_float32 (Ruy); int8_float16/int16 just fall
    # back with no benefit -- so int8 is the right default (plan §6a).
    parser.add_argument("--quantization", default="int8")
    parser.add_argument(
        "--no_force_m2m100",
        dest="force_m2m100",
        action="store_false",
        help="do NOT re-present the model as m2m_100 (will fail on IndicTrans2)",
    )
    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        raise SystemExit(
            f"{args.output_dir} already exists. Remove it first to re-convert."
        )

    source = resolve_source(args.model)

    # Default path: stage an M2M100-shaped copy so CT2's native loader accepts
    # it (IndicTrans2's weights are already M2M100-layout; only config differs).
    if args.force_m2m100:
        stage_dir = args.output_dir + "_m2m100_src"
        source = stage_as_m2m100(source, stage_dir)

    # In-process conversion (equivalent to the ct2-transformers-converter CLI,
    # which is just TransformersConverter(...).convert(..., quantization=...)).
    # We use a subclass so the dtype->torch_dtype remap works with
    # transformers<4.53 regardless of the installed ctranslate2 version. For the
    # staged M2M100 copy we must NOT trust remote code -- we want CT2's native
    # M2M100 loader; only the (failing) custom-arch path would need it True.
    trust_remote_code = not args.force_m2m100
    print(f"[info] converting {source} -> {args.output_dir} "
          f"(quantization={args.quantization}, trust_remote_code={trust_remote_code})")

    try:
        converter = _build_converter(source, trust_remote_code)
        converter.convert(args.output_dir, quantization=args.quantization, force=True)
    except ImportError:
        raise SystemExit(
            "ctranslate2 is not installed. Install it with:\n"
            "    pip install ctranslate2"
        )
    except Exception as exc:  # conversion failed for any reason
        print(f"[error] conversion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(FALLBACK_MSG, file=sys.stderr)
        raise SystemExit(1)

    # Remove the staged M2M100 copy; the CT2 output is self-contained.
    if args.force_m2m100:
        shutil.rmtree(source, ignore_errors=True)

    print(f"\n[ok] int8 CTranslate2 model written to: {args.output_dir}")
    print("     Copy it (and the tokenizer dir) to the Pi, then run "
          "`python -m it2edge.serve.translate_ct2`.")
    print("[!]  int8 + the M2M100 re-presentation are both lossy re: the source "
          "model -- A/B a few sentences against the torch path before shipping.")


if __name__ == "__main__":
    main()
