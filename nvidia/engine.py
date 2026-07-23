"""Jetson Nano voice pipeline engine — warm models, one process, minimal moving parts.

Device placement (per NvidiaRun.md):
    Whisper STT      -> GPU (CUDA)      faster-whisper / CTranslate2
    Translation      -> GPU (CUDA)      your fine-tuned MarianMT int8 CT2
    Piper TTS        -> CPU subprocess  (reuses it2edge.serve.speak, unchanged)

Everything the loop needs is loaded ONCE here and kept warm. Per-utterance work is
then just three calls. Each stage is timed so the caller can report latency.

Env (all optional, sane Jetson defaults):
    WHISPER_MODEL     tiny|base|small        (default: base — fits 4 GB with the translator)
    WHISPER_DEVICE    cuda|cpu               (default: cuda; auto-falls back to cpu)
    WHISPER_COMPUTE   int8_float16|float16|int8   (default: int8_float16 on cuda)
    CT2_DEVICE        cuda|cpu               (default: cuda; auto-falls back to cpu)
    CT2_MODEL_DIR     path to the CT2 package (default: model_cache_compact_ct2)
    (Piper paths / audio: PIPER_BIN, PIPER_VOICE, AUDIO_OUT, ALSA_DEVICE — see speak.py)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

# The Jetson's CUDA libs (cuBLAS/cuDNN) must be on the loader path before either
# CTranslate2 model touches the GPU. JetPack installs them under /usr/local/cuda.
def _ensure_cuda_on_path() -> None:
    extra = [
        "/usr/local/cuda/lib64",
        "/usr/lib/aarch64-linux-gnu",
    ]
    present = [d for d in extra if os.path.isdir(d)]
    if not present:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = present + ([cur] if cur else [])
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


@dataclass
class Stage:
    """One timed stage result."""

    text: str
    ms: float


@dataclass
class Turn:
    """One full mic->speak turn, with per-stage latency in milliseconds."""

    english: str
    hindi: str
    stt_ms: float
    translate_ms: float
    tts_ms: float

    @property
    def compute_ms(self) -> float:
        """STT + translate + TTS — excludes recording time (not compute)."""
        return self.stt_ms + self.translate_ms + self.tts_ms


class Engine:
    """Warm STT + translation models and a CPU Piper speaker, loaded once."""

    def __init__(self) -> None:
        self.whisper = None
        self.whisper_device = "?"
        self.tokenizer = None
        self.translator = None
        self.ct2_device = "?"
        self._speak = None  # it2edge.serve.speak.speak, bound lazily

    # ----- one-time warm load -------------------------------------------------

    def load(self) -> None:
        """Load both GPU models and bind the CPU speaker. Call once at startup."""
        _ensure_cuda_on_path()
        self._load_whisper()
        self._load_translator()
        self._bind_speaker()

    def _load_whisper(self) -> None:
        from faster_whisper import WhisperModel

        name = os.environ.get("WHISPER_MODEL", "base")
        want = os.environ.get("WHISPER_DEVICE", "cuda").lower()

        if want == "cuda":
            compute = os.environ.get("WHISPER_COMPUTE", "int8_float16")
            try:
                import numpy as np

                m = WhisperModel(name, device="cuda", compute_type=compute)
                # Force a real CUDA transcribe now so a broken CUDA surfaces HERE,
                # at startup, not mid-utterance.
                m.transcribe(np.zeros(16000, dtype=np.float32), language="en")
                self.whisper, self.whisper_device = m, "cuda"
                print(f"[engine] Whisper '{name}' on CUDA ({compute})")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[engine] Whisper CUDA unavailable ({type(exc).__name__}: {exc});"
                      " falling back to CPU")

        self.whisper = WhisperModel(name, device="cpu", compute_type="int8")
        self.whisper_device = "cpu"
        print(f"[engine] Whisper '{name}' on CPU (int8)")

    def _load_translator(self) -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        from it2edge.paths import CT2_DIR

        model_dir = os.environ.get("CT2_MODEL_DIR", str(CT2_DIR))
        if not os.path.isdir(model_dir):
            raise SystemExit(
                f"CT2 model not found at {model_dir}. Copy model_cache_compact_ct2/ "
                "onto the Jetson or set CT2_MODEL_DIR."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        want = os.environ.get("CT2_DEVICE", "cuda").lower()

        if want == "cuda":
            try:
                self.translator = ctranslate2.Translator(
                    model_dir, device="cuda", compute_type="int8"
                )
                self.ct2_device = "cuda"
                print("[engine] Translator (MarianMT int8 CT2) on CUDA")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[engine] CT2 CUDA unavailable ({type(exc).__name__}: {exc});"
                      " falling back to CPU")

        self.translator = ctranslate2.Translator(
            model_dir, device="cpu", compute_type="int8",
            inter_threads=1, intra_threads=os.cpu_count() or 4,
        )
        self.ct2_device = "cpu"
        print("[engine] Translator (MarianMT int8 CT2) on CPU")

    def _bind_speaker(self) -> None:
        # Piper stays exactly as the Pi build has it: a CPU subprocess per utterance.
        from it2edge.serve.speak import speak, tts_available

        ok, reason = tts_available()
        if not ok:
            print(f"[engine] Piper TTS not ready: {reason} (translation still works)")
        else:
            print("[engine] Piper TTS (CPU) ready")
        self._speak = speak

    # ----- per-utterance stages (each timed) ----------------------------------

    def stt(self, wav_path: str) -> Stage:
        t0 = time.perf_counter()
        segments, _ = self.whisper.transcribe(wav_path, language="en")
        text = " ".join(s.text for s in segments).strip()
        return Stage(text, (time.perf_counter() - t0) * 1000.0)

    def translate(self, english: str) -> Stage:
        from it2edge.serve.marian_ct2 import translate_marian

        t0 = time.perf_counter()
        hindi = translate_marian(
            [english], self.tokenizer, self.translator, beam_size=1
        )[0]
        return Stage(hindi, (time.perf_counter() - t0) * 1000.0)

    def speak(self, hindi: str) -> float:
        if self._speak is None:
            return 0.0
        t0 = time.perf_counter()
        try:
            self._speak(hindi)
        except Exception as exc:  # noqa: BLE001 — never crash the loop on audio
            print(f"[engine] TTS failed: {exc}")
        return (time.perf_counter() - t0) * 1000.0

    def run_turn(self, wav_path: str, do_speak: bool = True) -> Turn:
        """STT -> translate -> (speak). Returns text + per-stage latency."""
        s = self.stt(wav_path)
        t = self.translate(s.text) if s.text else Stage("", 0.0)
        tts_ms = self.speak(t.text) if (do_speak and t.text) else 0.0
        return Turn(
            english=s.text, hindi=t.text,
            stt_ms=s.ms, translate_ms=t.ms, tts_ms=tts_ms,
        )
