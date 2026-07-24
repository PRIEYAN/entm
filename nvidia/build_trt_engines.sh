#!/usr/bin/env bash
# Build TensorRT engines for the MarianMT en->hi model, ON THE JETSON NANO.
#
#   bash nvidia/build_trt_engines.sh
#
# Engines are HARDWARE-SPECIFIC and built ONCE, then cached to model_trt/.
# Never copy a .engine from another machine. Re-run this only if you change the
# ONNX model or the TensorRT version.
#
# Inputs:  $ONNX_MODEL_DIR (default model_onnx/) — the fp32 ONNX export with
#          encoder_model.onnx + decoder_model.onnx (the NO-past decoder; we don't
#          use decoder_with_past here — see nvidia/marian_trt.py for why).
# Outputs: $TRT_ENGINE_DIR (default model_trt/) — encoder.engine + decoder.engine
#
# Shapes: MarianMT hidden size is 512. Sequence lengths are dynamic; the profiles
# below cover 1..MAXLEN tokens. Bump MAXLEN if you translate very long sentences.

set -euo pipefail

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ONNX_DIR="${ONNX_MODEL_DIR:-model_onnx}"
OUT_DIR="${TRT_ENGINE_DIR:-model_trt}"
HIDDEN="${MARIAN_HIDDEN:-512}"     # d_model of the Marian encoder output
MAXLEN="${TRT_MAXLEN:-128}"        # max tokens (source and target)
OPTLEN="${TRT_OPTLEN:-32}"         # the length TRT optimizes hardest for

mkdir -p "$OUT_DIR"

if [ ! -x "$TRTEXEC" ]; then
  echo "ERROR: trtexec not found at $TRTEXEC (set TRTEXEC=...)" >&2
  exit 1
fi
for f in encoder_model.onnx decoder_model.onnx; do
  [ -f "$ONNX_DIR/$f" ] || { echo "ERROR: missing $ONNX_DIR/$f" >&2; exit 1; }
done

echo "==> Building ENCODER engine (this can take several minutes)"
# encoder inputs: input_ids [B,S], attention_mask [B,S]
"$TRTEXEC" \
  --onnx="$ONNX_DIR/encoder_model.onnx" \
  --saveEngine="$OUT_DIR/encoder.engine" \
  --minShapes=input_ids:1x1,attention_mask:1x1 \
  --optShapes=input_ids:1x${OPTLEN},attention_mask:1x${OPTLEN} \
  --maxShapes=input_ids:1x${MAXLEN},attention_mask:1x${MAXLEN} \
  --fp16 --workspace=1024

echo "==> Building DECODER engine (no-KV-cache; re-feeds the sequence each step)"
# decoder inputs: input_ids [B,T], encoder_attention_mask [B,S],
#                 encoder_hidden_states [B,S,HIDDEN]
"$TRTEXEC" \
  --onnx="$ONNX_DIR/decoder_model.onnx" \
  --saveEngine="$OUT_DIR/decoder.engine" \
  --minShapes=input_ids:1x1,encoder_attention_mask:1x1,encoder_hidden_states:1x1x${HIDDEN} \
  --optShapes=input_ids:1x${OPTLEN},encoder_attention_mask:1x${OPTLEN},encoder_hidden_states:1x${OPTLEN}x${HIDDEN} \
  --maxShapes=input_ids:1x${MAXLEN},encoder_attention_mask:1x${MAXLEN},encoder_hidden_states:1x${MAXLEN}x${HIDDEN} \
  --fp16 --workspace=1024

echo ""
echo "==> DONE. Engines in $OUT_DIR/:"
ls -lh "$OUT_DIR"/*.engine
echo ""
echo "Test:  ONNX_MODEL_DIR=$ONNX_DIR TRT_ENGINE_DIR=$OUT_DIR \\"
echo "       python -m nvidia.marian_trt \"Hello, how are you?\""
