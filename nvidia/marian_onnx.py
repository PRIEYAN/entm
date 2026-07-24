"""MarianMT en->hi inference over ONNX Runtime — GPU-capable on the Jetson Nano.

Why this exists: CTranslate2 4.x has no CUDA build for the Nano's Maxwell GPU
(sm_53 / CUDA 10.2), so it can only run the translator on CPU. ONNX Runtime DOES
have a GPU build for JetPack 4.6, so this module runs the SAME fine-tuned model on
the GPU via CUDAExecutionProvider (falling back to CPU if the GPU provider isn't
available).

Runtime deps: onnxruntime (or onnxruntime-gpu on the Jetson) + transformers
tokenizer + numpy. No torch, no optimum — we drive the encoder/decoder ONNX
sessions directly with a greedy decode loop.

Model layout (produced by `optimum-cli export onnx ... text2text-generation-with-past`):
    encoder_model.onnx            input_ids, attention_mask -> last_hidden_state
    decoder_model.onnx            first step: no past; emits logits + all present KV
    decoder_with_past_model.onnx  cached steps: takes past KV, emits logits + new
                                  decoder-present KV (encoder KV is constant, reused)

We use the two non-merged decoder files rather than decoder_model_merged.onnx —
the merged graph's use_cache branch is awkward to drive by hand, while this pair
is a clean no-past / with-past split.
"""


import os
from typing import Optional

import numpy as np


def _providers(device: str):
    """Ordered ONNX Runtime providers for the requested device.

    'cuda' -> try GPU first, then CPU. Anything else -> CPU only.
    """
    if device.lower() == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxMarian:
    """Greedy en->hi translator over ONNX Runtime encoder/decoder sessions."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_THREADS", os.cpu_count() or 4))
        want = _providers(device)

        self.encoder = ort.InferenceSession(
            os.path.join(model_dir, "encoder_model.onnx"), sess_options=so, providers=want
        )
        # Step 1 (no cache) and subsequent cached steps are two separate graphs.
        self.decoder = ort.InferenceSession(
            os.path.join(model_dir, "decoder_model.onnx"), sess_options=so, providers=want
        )
        self.decoder_past = ort.InferenceSession(
            os.path.join(model_dir, "decoder_with_past_model.onnx"), sess_options=so, providers=want
        )

        # Report what actually bound (CUDA may silently fall back to CPU).
        self.device = "cuda" if "CUDAExecutionProvider" in self.encoder.get_providers() else "cpu"

        # Layer count from the with-past decoder's inputs.
        self._n_layers = sum(
            1 for i in self.decoder_past.get_inputs() if i.name.endswith(".decoder.key")
        )
        # Which past-KV inputs the with-past graph actually expects. Some exports
        # take only decoder-side past (encoder KV is baked in); others take both.
        self._past_inputs = {i.name for i in self.decoder_past.get_inputs()}

        self.pad_id = self.tokenizer.pad_token_id or 0
        self.eos_id = self.tokenizer.eos_token_id
        # Marian starts decoding with the pad token as BOS.
        self.decoder_start_id = self.pad_id

    # -------------------------------------------------------------------------

    def _encode(self, text: str):
        enc = self.tokenizer(text, return_tensors="np")
        input_ids = enc["input_ids"].astype(np.int64)
        attn = enc["attention_mask"].astype(np.int64)
        (hidden,) = self.encoder.run(None, {"input_ids": input_ids, "attention_mask": attn})
        return hidden, attn

    def _present_to_past(self, out: dict, prev_past: Optional[dict]) -> dict:
        """Build the next step's past-KV from a decoder run's `present.*` outputs.

        Decoder-side KV grows each step (comes from `present`); encoder-side KV is
        constant across steps, so reuse the previous past when the with-past graph
        no longer emits it.
        """
        past = {}
        for i in range(self._n_layers):
            past[f"past_key_values.{i}.decoder.key"] = out[f"present.{i}.decoder.key"]
            past[f"past_key_values.{i}.decoder.value"] = out[f"present.{i}.decoder.value"]
            for side in ("encoder.key", "encoder.value"):
                name = f"past_key_values.{i}.{side}"
                if name not in self._past_inputs:
                    continue
                pres = f"present.{i}.{side}"
                if pres in out:
                    past[name] = out[pres]
                elif prev_past is not None and name in prev_past:
                    past[name] = prev_past[name]  # encoder KV is constant
        return past

    def translate_one(self, text: str, max_length: int = 128) -> str:
        hidden, enc_attn = self._encode(text)

        # --- step 1: no-past decoder, seeds the cache ---
        first_out = self.decoder.run(None, {
            "encoder_attention_mask": enc_attn,
            "input_ids": np.array([[self.decoder_start_id]], dtype=np.int64),
            "encoder_hidden_states": hidden,
        })
        names = [o.name for o in self.decoder.get_outputs()]
        out = dict(zip(names, first_out))

        next_id = int(np.argmax(out["logits"][0, -1]))
        generated: list[int] = []
        if next_id == self.eos_id:
            return ""
        generated.append(next_id)
        past = self._present_to_past(out, None)

        # --- steps 2..N: with-past decoder ---
        past_out_names = [o.name for o in self.decoder_past.get_outputs()]
        for _ in range(max_length - 1):
            feeds = {
                "encoder_attention_mask": enc_attn,
                "input_ids": np.array([[next_id]], dtype=np.int64),
            }
            feeds.update(past)
            step = self.decoder_past.run(None, feeds)
            out = dict(zip(past_out_names, step))

            next_id = int(np.argmax(out["logits"][0, -1]))
            if next_id == self.eos_id:
                break
            generated.append(next_id)
            past = self._present_to_past(out, past)

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def translate(self, sentences, max_length: int = 128):
        return [self.translate_one(s, max_length=max_length) for s in sentences]


if __name__ == "__main__":
    import sys

    model_dir = os.environ.get("ONNX_MODEL_DIR", "model_onnx")
    m = OnnxMarian(model_dir, device=os.environ.get("CT2_DEVICE", "cuda"))
    print(f"[onnx] translator on {m.device} ({m._n_layers} layers)")
    texts = sys.argv[1:] or ["Hello, how are you?", "Good morning"]
    for s, h in zip(texts, m.translate(texts)):
        print(f"EN: {s}\n-> {h}\n")
