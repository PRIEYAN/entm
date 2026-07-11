"""ALTERNATIVE runtime: export IndicTrans2 to ONNX and int8-quantize it.

RUN ON THE DEV MACHINE. Plan reference: §3, §6b, §11.

Use this ONLY if you have a hard reason to run ONNX Runtime instead of the
recommended CTranslate2 path (an existing ORT-only serving stack, one runtime
across many models, a needed ORT execution provider). For this model on ARM,
CTranslate2 is both faster and lower-effort -- see plan §3. Prefer it2edge.convert.convert_ct2.

Why the plain CLI does not work:
  * `optimum-cli export onnx` FAILS on IndicTrans2 -- its custom `model_type` is
    not in optimum's registry (confirmed by an AI4Bharat HF discussion). The
    working route is the Python API `main_export` with a CUSTOM OnnxConfig
    subclassing M2M100OnnxConfig and remapping the fairseq-style attribute names.

Known caveats (plan §6b, §10):
  * `ORTModelForSeq2SeqLM` may KeyError on the unregistered model_type at load
    time -- register the normalized config or hand-write the decode loop.
  * Dynamic-quant seq2seq has known bugs (optimum #438).
  * Quantizing the DECODER can hurt MT quality -- RE-VALIDATE BLEU/chrF after.
  * On ARM, dynamic quant uses U8S8 + reduce_range (no calibration data needed).
  * IndicProcessor + tokenizer always stay in Python -- never in the ONNX graph.

Usage:
    pip install "transformers>=4.51,<4.53" torch "optimum[exporters]" \
        onnx onnxruntime sentencepiece protobuf

    python -m it2edge.convert.export_onnx                 # fp32 export + int8
    python -m it2edge.convert.export_onnx --no_quantize   # export only
"""

import argparse
import os

from it2edge.paths import HF_SNAPSHOT, MODEL_ID, ONNX_DIR


def resolve_model_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    if HF_SNAPSHOT.is_dir():
        return str(HF_SNAPSHOT)
    print(f"[warn] {HF_SNAPSHOT} not found; exporting from the hub ({MODEL_ID}).")
    return MODEL_ID


def build_custom_configs(config):
    """Subclass M2M100OnnxConfig, remapping fairseq attr names to normalized ones.

    IndicTrans2 is architecturally close to M2M100 (sinusoidal positional
    embeddings, encoder/decoder transformer) but uses fairseq-style attribute
    names. We map the normalized names optimum expects onto the real ones.
    Plan §6b.
    """
    from optimum.exporters.onnx.model_configs import M2M100OnnxConfig
    from optimum.utils import NormalizedTextConfig

    # Remap: normalized attr -> actual attr on IndicTrans2's config.
    normalized = NormalizedTextConfig.with_args(
        hidden_size="encoder_embed_dim",
        num_attention_heads="encoder_attention_heads",
        num_layers="encoder_layers",
        vocab_size="src_vocab_size",
    )

    class IndicTrans2OnnxConfig(M2M100OnnxConfig):
        NORMALIZED_CONFIG_CLASS = normalized

    enc = IndicTrans2OnnxConfig(config, task="text2text-generation-with-past")
    dec = IndicTrans2OnnxConfig(
        config, task="text2text-generation-with-past", use_past=False
    )
    dec_past = IndicTrans2OnnxConfig(
        config, task="text2text-generation-with-past", use_past=True
    )
    return {
        "encoder_model": enc,
        "decoder_model": dec,
        "decoder_with_past_model": dec_past,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export IndicTrans2 to ONNX (custom OnnxConfig) + int8 quantize"
    )
    parser.add_argument("--output_dir", default=str(ONNX_DIR))
    parser.add_argument("--model", default=None, help="model path/id (default: local)")
    parser.add_argument(
        "--no_quantize", action="store_true", help="export fp32 only, skip int8"
    )
    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        raise SystemExit(f"{args.output_dir} already exists. Remove it to re-export.")

    from transformers import AutoConfig

    try:
        from optimum.exporters.onnx import main_export
    except ImportError as exc:  # pragma: no cover - guidance for a missing dep
        raise SystemExit(
            "optimum[exporters] is not installed. Install it with:\n"
            '    pip install "optimum[exporters]" onnx onnxruntime'
        ) from exc

    model_path = resolve_model_path(args.model)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    custom_configs = build_custom_configs(config)

    print(f"[info] exporting to ONNX (custom OnnxConfig) -> {args.output_dir}")
    main_export(
        model_path,
        output=args.output_dir,
        trust_remote_code=True,
        no_post_process=True,
        task="text2text-generation-with-past",
        custom_onnx_configs=custom_configs,
    )
    print(f"[ok] fp32 ONNX written to: {args.output_dir}")

    if args.no_quantize:
        return

    # ---- int8 dynamic quantization (no calibration data). Plan §6b. ----
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_files = [f for f in os.listdir(args.output_dir) if f.endswith(".onnx")]
    if not onnx_files:
        raise SystemExit(f"No .onnx files found in {args.output_dir} to quantize.")

    for fname in onnx_files:
        src = os.path.join(args.output_dir, fname)
        dst = os.path.join(args.output_dir, fname.replace(".onnx", ".int8.onnx"))
        print(f"[info] quantizing {fname} -> {os.path.basename(dst)}")
        # U8S8 + reduce_range are the ARM-friendly settings (plan §6b).
        quantize_dynamic(
            src,
            dst,
            weight_type=QuantType.QInt8,
            reduce_range=True,
        )

    print(f"\n[ok] int8 ONNX models written to: {args.output_dir}")
    print("[!]  Quantizing the decoder can hurt MT quality -- RE-VALIDATE BLEU/chrF.")
    print("     Serve with ORTModelForSeq2SeqLM (register the normalized config if")
    print("     it KeyErrors on model_type) or a hand-written decode loop.")


if __name__ == "__main__":
    main()
