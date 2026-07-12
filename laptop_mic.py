"""STEP 1 of the voice pipeline: laptop mic -> local Whisper STT -> Pi translate.

Runs on the GPU LAPTOP. Records your English speech, transcribes it LOCALLY with
Whisper (on the laptop GPU -- no cloud), then sends the text over SSH to the Pi,
which runs the existing translate CLI and prints the Hindi translation.

This step does NOT do TTS yet -- it just proves: voice -> English text ->
(SSH) -> Hindi text printed back. Speaker/TTS is a later step.

Fully offline after a one-time model download (Whisper caches to disk).

Setup (laptop, once):
    pip install faster-whisper sounddevice soundfile
    # make SSH password-less so each utterance doesn't prompt:
    ssh-keygen -t ed25519            # if you don't have a key
    ssh-copy-id <PI_USER>@<PI_IP>

Configure PI_HOST / PI_PROJECT below (or via env), then:
    python laptop_mic.py                 # push-to-talk loop
    python laptop_mic.py --seconds 7     # longer recording window
    python laptop_mic.py --text "Hello"  # skip the mic, just send text (test SSH)
"""

import argparse
import os
import subprocess
import sys
import tempfile

# --- configure these for your setup (env overrides win) ---
PI_HOST = os.environ.get("PI_HOST", "root@192.168.1.50")   # <PI_USER>@<PI_IP>
PI_PROJECT = os.environ.get("PI_PROJECT", "~/entm")         # repo path on the Pi
PI_VENV_PY = os.environ.get("PI_VENV_PY", "venv/bin/python")  # python on the Pi
TGT_LANG = os.environ.get("TGT_LANG", "hin_Deva")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")     # tiny/base/small/medium
SR = 16000


def transcribe(seconds: int) -> str:
    """Record `seconds` of mic audio and transcribe it locally with Whisper."""
    import sounddevice as sd
    import soundfile as sf
    from faster_whisper import WhisperModel

    # device="cuda" -> laptop GPU. Falls back to CPU if CUDA isn't available.
    try:
        model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    except Exception:
        print("[warn] CUDA Whisper unavailable; using CPU (slower).")
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    print(f"[recording {seconds}s -- speak English now]")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1)
    sd.wait()
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(wav, audio, SR)

    segments, _ = model.transcribe(wav, language="en")
    text = " ".join(s.text for s in segments).strip()
    os.unlink(wav)
    return text


def send_to_pi(text: str) -> int:
    """SSH the English text to the Pi's translate CLI; the Pi prints the Hindi."""
    safe = text.replace('"', '\\"')
    remote = (
        f'cd {PI_PROJECT} && {PI_VENV_PY} -m it2edge.serve.translate_ct2 '
        f'--tgt {TGT_LANG} "{safe}"'
    )
    print(f"[sending to {PI_HOST}] {text}")
    # Stream the Pi's stdout straight to our terminal so we see the Hindi.
    return subprocess.run(["ssh", PI_HOST, remote]).returncode


def main():
    ap = argparse.ArgumentParser(description="Laptop STT -> Pi translate (step 1)")
    ap.add_argument("--seconds", type=int, default=5, help="recording length")
    ap.add_argument("--text", default=None,
                    help="skip the mic and send this text (SSH/translate test)")
    ap.add_argument("--once", action="store_true", help="one utterance, then exit")
    args = ap.parse_args()

    # Mic-less mode: just test the SSH -> translate path with typed text.
    if args.text is not None:
        sys.exit(send_to_pi(args.text))

    print(f"Push-to-talk EN -> {TGT_LANG} on {PI_HOST}. Ctrl-C to quit.")
    try:
        while True:
            input("press Enter to speak ")
            text = transcribe(args.seconds)
            if not text:
                print("[no speech detected]")
                continue
            print("EN:", text)
            send_to_pi(text)
            if args.once:
                break
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
