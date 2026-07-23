"""Full voice pipeline on ONE Jetson Nano: mic -> STT(GPU) -> translate(GPU) -> Piper(CPU).

Implements nvidia/NvidiaRun.md. Models are loaded once and kept warm; each mic press
is just three calls. Run from the project root so `it2edge` imports resolve:

    python -m nvidia.jetson_mic                    # push-to-talk loop (records mic)
    python -m nvidia.jetson_mic --seconds 7        # longer recording window
    python -m nvidia.jetson_mic --text "Hello"     # skip the mic, type the English
    python -m nvidia.jetson_mic --bench            # latency benchmark, no mic/audio
    python -m nvidia.jetson_mic --bench --speak    # benchmark including TTS playback

Recommended launch on the Jetson (max clocks + both models on GPU):
    sudo nvpmodel -m 0 && sudo jetson_clocks
    WHISPER_MODEL=base WHISPER_DEVICE=cuda CT2_DEVICE=cuda python -m nvidia.jetson_mic
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from nvidia.engine import Engine, Turn

SR = 16000  # Whisper wants 16 kHz mono

# A fixed, mixed-length English set for --bench (short / medium / long).
BENCH_SENTENCES = [
    "Hello.",
    "Good morning.",
    "How are you today?",
    "The weather is beautiful this morning.",
    "I would like a cup of coffee, please.",
    "The committee will review the proposal next week.",
    "Please turn left at the next intersection and continue for two kilometers.",
    "Thank you very much for your help; I really appreciate it.",
]


def _record(seconds: int) -> str:
    import sounddevice as sd
    import soundfile as sf

    print(f"[recording {seconds}s — speak English now]")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1)
    sd.wait()
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(wav, audio, SR)
    return wav


def _print_turn(turn: Turn) -> None:
    print("-" * 56)
    print(f"  🎤 EN : {turn.english or '[no speech]'}")
    print(f"  🔊 HI : {turn.hindi or '—'}")
    print(f"  ⏱  STT {turn.stt_ms:7.1f} ms | MT {turn.translate_ms:7.1f} ms "
          f"| TTS {turn.tts_ms:7.1f} ms | compute {turn.compute_ms:7.1f} ms")
    print("-" * 56)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round((pct / 100.0) * (len(s) - 1))))
    return s[k]


def bench(engine: Engine, do_speak: bool) -> None:
    """Latency benchmark: translation always; STT only if we can synthesize audio.

    STT needs an audio file; here we synthesize each sentence to text->speech isn't
    the input. So --bench measures the TWO always-on stages we control precisely:
    translation (the GPU model in question) and, if --speak, TTS. It also does a
    warmup so cold-start import cost is excluded from the numbers.
    """
    print(f"\n[bench] warmup + {len(BENCH_SENTENCES)} sentences "
          f"(translate on {engine.ct2_device}, speak={'yes' if do_speak else 'no'})\n")

    # Warmup — pay any first-call JIT/allocation cost outside the measurement.
    engine.translate("Warmup sentence to prime the translator.")

    mt_ms: list[float] = []
    tts_ms: list[float] = []
    for i, en in enumerate(BENCH_SENTENCES, 1):
        t = engine.translate(en)
        mt_ms.append(t.ms)
        line = f"  [{i}/{len(BENCH_SENTENCES)}] MT {t.ms:7.1f} ms  {en[:40]!r} -> {t.text}"
        if do_speak:
            ms = engine.speak(t.text)
            tts_ms.append(ms)
            line += f"   (TTS {ms:.0f} ms)"
        print(line)

    def stats(name: str, xs: list[float]) -> None:
        if not xs:
            return
        print(f"  {name:12s} median {_percentile(xs,50):7.1f} ms | "
              f"p95 {_percentile(xs,95):7.1f} ms | "
              f"min {min(xs):7.1f} | max {max(xs):7.1f} ms")

    print("\n[bench] results")
    print(f"  translator device : {engine.ct2_device}")
    print(f"  whisper device    : {engine.whisper_device} "
          f"(model '{os.environ.get('WHISPER_MODEL','base')}')")
    stats("translate", mt_ms)
    if do_speak:
        stats("tts", tts_ms)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Jetson voice pipeline (STT+MT on GPU, Piper on CPU)")
    ap.add_argument("--seconds", type=int, default=5, help="mic recording window")
    ap.add_argument("--text", default=None, help="skip the mic; translate+speak this English")
    ap.add_argument("--bench", action="store_true", help="latency benchmark (no mic)")
    ap.add_argument("--speak", action="store_true", help="with --bench, also play TTS")
    ap.add_argument("--no-audio", action="store_true", help="translate only, don't speak")
    ap.add_argument("--once", action="store_true", help="one utterance then exit")
    args = ap.parse_args()

    # Only the mic loop needs Whisper. --text / --bench skip it, so they don't
    # require faster_whisper and start much faster.
    needs_stt = args.text is None and not args.bench

    engine = Engine()
    engine.load(with_stt=needs_stt)

    if args.bench:
        bench(engine, do_speak=args.speak)
        return

    if args.text is not None:
        turn = engine.translate(args.text)
        full = Turn(args.text, turn.text, 0.0, turn.ms,
                    engine.speak(turn.text) if not args.no_audio else 0.0)
        _print_turn(full)
        return

    print(f"\nPush-to-talk  EN -> HI on this Jetson  "
          f"(STT:{engine.whisper_device}  MT:{engine.ct2_device}  TTS:cpu). Ctrl-C to quit.")
    try:
        while True:
            input("\npress Enter, then speak... ")
            wav = _record(args.seconds)
            turn = engine.run_turn(wav, do_speak=not args.no_audio)
            os.unlink(wav)
            if not turn.english:
                print("  [no speech detected — try again]")
                continue
            _print_turn(turn)
            if args.once:
                break
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
