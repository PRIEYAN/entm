"""Direct text-input tester — type English, get Hindi (translate + optional speak).

No mic, no socket, no VAD, no Whisper. Loads the translator (and Piper) ONCE and
then loops on your typed input, so you can test translation + TTS quickly without
relaunching (which would reload the model every time).

    python -m nvidia.text_test                       # interactive REPL
    python -m nvidia.text_test "Good morning"        # one-shot, then exit
    python -m nvidia.text_test --no-audio            # translate only, don't speak
    TRANSLATE_BACKEND=ct2 python -m nvidia.text_test # force the CPU int8 backend

This is the fastest way to prove the translation half + Piper audio path work
(including SoundWire output to a phone — see soundwire.txt) before wiring up the
mic/STT half. Whisper is never loaded here.
"""

from __future__ import annotations

import argparse

from nvidia.engine import Engine


def _run_one(engine: Engine, text: str, do_speak: bool) -> None:
    t = engine.translate(text)
    print(f"  EN: {text}")
    print(f"  HI: {t.text}   (MT {t.ms:.1f} ms)")
    if do_speak and t.text:
        ms = engine.speak(t.text)
        print(f"  🔊 spoke ({ms:.0f} ms)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Direct text -> Hindi tester (no mic/STT)")
    ap.add_argument("text", nargs="*", help="English to translate; omit for interactive mode")
    ap.add_argument("--no-audio", action="store_true", help="translate only, don't speak")
    args = ap.parse_args()

    do_speak = not args.no_audio

    # with_stt=False -> never import/load Whisper. Loads translator + binds Piper only.
    engine = Engine()
    engine.load(with_stt=False)
    print(f"[text-test] translator on {engine.ct2_device} | "
          f"audio {'on' if do_speak else 'off'}")

    if args.text:                      # one-shot mode
        _run_one(engine, " ".join(args.text), do_speak)
        return

    print("Type English and press Enter (blank line or Ctrl-C/Ctrl-D to quit).")
    try:
        while True:
            line = input("\nEN> ").strip()
            if not line:
                break
            _run_one(engine, line, do_speak)
    except (KeyboardInterrupt, EOFError):
        pass
    print("\nbye")


if __name__ == "__main__":
    main()
