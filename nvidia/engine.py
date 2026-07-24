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
        self._onnx = None  # OnnxMarian when TRANSLATE_BACKEND=onnx, else None
        self._speak = None  # it2edge.serve.speak.speak, bound lazily

    # ----- one-time warm load -------------------------------------------------

    def load(self, with_stt: bool = True) -> None:
        """Warm the models + bind the CPU speaker. Call once at startup.

        with_stt=False skips Whisper entirely — text-only and text->audio modes
        never touch the mic, so they shouldn't pay Whisper's import/load cost or
        even require faster_whisper to be installed.
        """
        _ensure_cuda_on_path()
        if with_stt:
            self._load_whisper()
        else:
            self.whisper_device = "off"
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
        # Backend choice:
        #   'trt'  -> TensorRT engines on the Jetson GPU (JetPack 4.6 / CUDA 10.2).
        #             THE real GPU path on the Nano — CT2 & onnxruntime-gpu need CUDA 11+.
        #   'onnx' -> ONNX Runtime (GPU only where a CUDA onnxruntime wheel exists;
        #             CPU otherwise). Useful on dev boxes / newer Jetsons.
        #   'ct2'  -> CPU-only CTranslate2 int8. The safe fallback everywhere.
        backend = os.environ.get("TRANSLATE_BACKEND", "onnx").lower()
        if backend == "trt":
            self._load_translator_trt()
        elif backend == "onnx":
            self._load_translator_onnx()
        else:
            self._load_translator_ct2()

    def _load_translator_trt(self) -> None:
        from nvidia.marian_trt import TrtMarian

        model_dir = os.environ.get("ONNX_MODEL_DIR", "model_onnx")  # tokenizer lives here
        if not os.path.isdir(model_dir):
            raise SystemExit(
                f"Model/tokenizer dir not found at {model_dir}. Set ONNX_MODEL_DIR."
            )
        # TrtMarian raises SystemExit with build instructions if engines are missing.
        self._onnx = TrtMarian(model_dir)  # reuses the ONNX sentinel path in translate()
        self.tokenizer = self._onnx.tokenizer
        self.translator = self._onnx
        self.ct2_device = self._onnx.device
        print("[engine] Translator (MarianMT TensorRT) on CUDA")

    def _load_translator_onnx(self) -> None:
        from nvidia.marian_onnx import OnnxMarian

        model_dir = os.environ.get("ONNX_MODEL_DIR", "model_onnx")
        if not os.path.isdir(model_dir):
            raise SystemExit(
                f"ONNX model not found at {model_dir}. Export it with "
                "`optimum-cli export onnx --model model_cache_compact_ft "
                "--task text2text-generation-with-past model_onnx` and copy it over, "
                "or set TRANSLATE_BACKEND=ct2 to use the CPU CTranslate2 model."
            )
        want = os.environ.get("CT2_DEVICE", "cuda").lower()
        self._onnx = OnnxMarian(model_dir, device=want)
        self.tokenizer = self._onnx.tokenizer
        self.translator = self._onnx  # sentinel; translate() picks the ONNX path
        self.ct2_device = self._onnx.device
        print(f"[engine] Translator (MarianMT ONNX) on "
              f"{'CUDA' if self._onnx.device == 'cuda' else 'CPU'}")

    def _load_translator_ct2(self) -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        from it2edge.paths import CT2_DIR

        self._onnx = None
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

    def stt_segments(self, audio):
        """Yield (english_sentence, elapsed_ms) per Whisper segment, as they decode.

        This is the sentence-level lever from jetsonNvidia.md §11.3: instead of
        waiting for the whole utterance to transcribe before anything downstream
        runs, emit each segment the moment faster-whisper produces it. The server
        pushes each one straight into MT->TTS, so the first Hindi line plays while
        later lines are still being transcribed.

        `audio` may be a WAV path or a float32 numpy array at 16 kHz (what the
        socket server assembles from the streamed PCM). Times are cumulative from
        the start of transcription so the caller can see per-segment latency.
        """
        t0 = time.perf_counter()
        segments, _ = self.whisper.transcribe(audio, language="en")
        for seg in segments:              # faster-whisper yields lazily as it decodes
            text = seg.text.strip()
            if text:
                yield text, (time.perf_counter() - t0) * 1000.0

    def translate(self, english: str) -> Stage:
        t0 = time.perf_counter()
        if self._onnx is not None:
            hindi = self._onnx.translate([english])[0]
        else:
            from it2edge.serve.marian_ct2 import translate_marian

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
