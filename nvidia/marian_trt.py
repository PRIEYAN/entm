"""MarianMT en->hi inference over TensorRT — the GPU path for the Jetson Nano.

Why this exists: on JetPack 4.6 / CUDA 10.2 the ONLY way to reach the GPU is
TensorRT (CTranslate2 and onnxruntime-gpu both need CUDA 11+). This module runs
the SAME fine-tuned MarianMT model on the GPU by loading TensorRT engines built
from the ONNX export, and driving a greedy decode loop with pycuda.

Design choice — NO KV-cache (deliberate):
    The ONNX export ships a `decoder_with_past` graph (KV-cache) for speed, but
    driving a growing-KV-cache decoder through TensorRT 8.2 by hand is fragile
    (a dozen KV tensors reshaped every step). Instead we use the plain
    `decoder_model.onnx` (no past) and RE-FEED the whole generated sequence each
    step. For short sentences on a tiny 77M model this is simpler, robust, and
    still fast — and it needs only two engines (encoder + decoder), each with one
    dynamic-shape optimization profile.

Engines are built ON THE NANO with trtexec (see build_trt_engines.sh / final.txt).
They are hardware-specific — never copy an .engine from another machine.

Interface matches nvidia/marian_onnx.OnnxMarian so nvidia/engine.py can pick it
up unchanged: `.tokenizer`, `.device`, `.translate([str]) -> [str]`.

Runtime deps: tensorrt (system cp36), pycuda, numpy, transformers tokenizer.
"""

from __future__ import annotations

import os

import numpy as np

# pycuda.autoinit creates a CUDA context on import. Import lazily inside __init__
# so merely importing this module doesn't grab the GPU.


def _default_engine_dir() -> str:
    return os.environ.get("TRT_ENGINE_DIR", "model_trt")


class _TrtRunner:
    """One TensorRT engine + its pycuda I/O buffers, with dynamic-shape support.

    Handles the TensorRT 8.x explicit-batch, dynamic-shape execution flow:
      * set the concrete input shapes for this call (set_binding_shape),
      * (re)allocate device buffers big enough for those shapes,
      * H2D copy inputs -> execute_v2 -> D2H copy outputs.
    Kept dead simple (synchronous, no streams) — correctness first on the Nano.
    """

    def __init__(self, engine_path: str, trt, cuda):
        self._trt = trt
        self._cuda = cuda
        if not os.path.isfile(engine_path):
            raise SystemExit(
                f"TensorRT engine not found: {engine_path}\n"
                "Build it on the Nano first:  bash nvidia/build_trt_engines.sh"
            )
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # Split binding names into inputs vs outputs (order = binding index).
        self._names = [self.engine.get_binding_name(i)
                       for i in range(self.engine.num_bindings)]
        self._is_input = {n: self.engine.binding_is_input(n) for n in self._names}
        self._dtype = {n: trt.nptype(self.engine.get_binding_dtype(n))
                       for n in self._names}
        self._dev = {}   # name -> device allocation
        self._dev_nbytes = {}  # name -> current allocation size

    def _ensure_dev(self, name: str, nbytes: int):
        if self._dev_nbytes.get(name, 0) < nbytes:
            if name in self._dev:
                self._dev[name].free()
            self._dev[name] = self._cuda.mem_alloc(nbytes)
            self._dev_nbytes[name] = nbytes
        return self._dev[name]

    def run(self, feeds: dict) -> dict:
        """feeds: {input_name: np.ndarray}. Returns {output_name: np.ndarray}."""
        cuda = self._cuda
        bindings = [0] * self.engine.num_bindings

        # --- inputs: set shape, upload ---
        for name, arr in feeds.items():
            idx = self.engine.get_binding_index(name)
            arr = np.ascontiguousarray(arr.astype(self._dtype[name]))
            self.context.set_binding_shape(idx, arr.shape)
            dptr = self._ensure_dev(name, arr.nbytes)
            cuda.memcpy_htod(dptr, arr)
            bindings[idx] = int(dptr)

        assert self.context.all_binding_shapes_specified, "unset input shape"

        # --- outputs: allocate to the shapes TRT now reports ---
        out_meta = {}
        for name in self._names:
            if self._is_input[name]:
                continue
            idx = self.engine.get_binding_index(name)
            shape = tuple(self.context.get_binding_shape(idx))
            nbytes = int(np.prod(shape)) * np.dtype(self._dtype[name]).itemsize
            dptr = self._ensure_dev(name, nbytes)
            bindings[idx] = int(dptr)
            out_meta[name] = (shape, dptr)

        # --- execute ---
        ok = self.context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 failed")

        # --- download outputs ---
        outputs = {}
        for name, (shape, dptr) in out_meta.items():
            host = np.empty(shape, dtype=self._dtype[name])
            cuda.memcpy_dtoh(host, dptr)
            outputs[name] = host
        return outputs


class TrtMarian:
    """Greedy en->hi translator over TensorRT engines (no KV-cache)."""

    def __init__(self, model_dir: str, engine_dir: str | None = None):
        import pycuda.autoinit  # noqa: F401 — creates the CUDA context
        import pycuda.driver as cuda
        import tensorrt as trt
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.device = "cuda"   # if we got here, TensorRT/CUDA are live

        engine_dir = engine_dir or _default_engine_dir()
        self._encoder = _TrtRunner(os.path.join(engine_dir, "encoder.engine"), trt, cuda)
        self._decoder = _TrtRunner(os.path.join(engine_dir, "decoder.engine"), trt, cuda)

        self.pad_id = self.tokenizer.pad_token_id or 0
        self.eos_id = self.tokenizer.eos_token_id
        self.decoder_start_id = self.pad_id   # Marian starts decoding with pad as BOS

    # -------------------------------------------------------------------------

    def _encode(self, text: str):
        enc = self.tokenizer(text, return_tensors="np")
        input_ids = enc["input_ids"].astype(np.int64)
        attn = enc["attention_mask"].astype(np.int64)
        out = self._encoder.run({"input_ids": input_ids, "attention_mask": attn})
        # encoder has a single output (last_hidden_state); take it by position.
        hidden = next(iter(out.values()))
        return hidden, attn

    def translate_one(self, text: str, max_length: int = 128) -> str:
        hidden, enc_attn = self._encode(text)

        generated = [self.decoder_start_id]   # feed the whole sequence each step
        result: list[int] = []
        for _ in range(max_length):
            dec_ids = np.array([generated], dtype=np.int64)
            out = self._decoder.run({
                "input_ids": dec_ids,
                "encoder_attention_mask": enc_attn,
                "encoder_hidden_states": hidden,
            })
            logits = out["logits"] if "logits" in out else next(iter(out.values()))
            next_id = int(np.argmax(logits[0, -1]))
            if next_id == self.eos_id:
                break
            result.append(next_id)
            generated.append(next_id)

        return self.tokenizer.decode(result, skip_special_tokens=True)

    def translate(self, sentences, max_length: int = 128):
        return [self.translate_one(s, max_length=max_length) for s in sentences]


if __name__ == "__main__":
    import sys

    model_dir = os.environ.get("ONNX_MODEL_DIR", "model_onnx")
    m = TrtMarian(model_dir)
    print(f"[trt] MarianMT translator on {m.device} (TensorRT)")
    texts = sys.argv[1:] or ["Hello, how are you?", "Good morning"]
    for s, h in zip(texts, m.translate(texts)):
        print(f"EN: {s}\n-> {h}\n")
