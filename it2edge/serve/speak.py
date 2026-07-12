"""STEP 2: translate English -> Hindi and SPEAK it on the Pi's speaker (offline).

Runs ON THE RASPBERRY PI. Reuses the working CT2 int8 translator, then pipes the
Hindi text through Piper TTS and plays it out the Pi's default ALSA device.
No network calls -- fully offline once Piper + the voice model are installed.

Invoked locally or over SSH from the laptop:
    python -m it2edge.serve.speak "Hello, how are you?"
    python -m it2edge.serve.speak --tgt hin_Deva "Good morning."
    python -m it2edge.serve.speak --no-audio "test"     # translate + print only

Setup on the Pi (see STT-TTS.md Part 3):
    # 1) Piper binary (aarch64):
    #    wget .../piper_linux_aarch64.tar.gz && tar -xf ...   -> ~/piper/piper
    # 2) a Hindi voice (.onnx + .onnx.json) into ~/piper_voices/
    # 3) a speaker on the Pi (3.5mm/USB), tested with `speaker-test`
Configure the paths below via env if yours differ:
    PIPER_BIN   (default ~/piper/piper)
    PIPER_VOICE (default ~/piper_voices/hi_IN-pratham-medium.onnx)
    PIPER_SR    (default 22050; match the voice's .onnx.json audio.sample_rate)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

from it2edge.paths import CT2_DIR, HF_SNAPSHOT
from it2edge.serve.translate_ct2 import load, translate

PIPER_BIN = os.environ.get("PIPER_BIN", os.path.expanduser("~/piper/piper"))
PIPER_VOICE = os.environ.get(
    "PIPER_VOICE", os.path.expanduser("~/piper_voices/hi_IN-pratham-medium.onnx")
)


def _voice_sample_rate(default="22050"):
    """Read the voice's sample rate from its .onnx.json so aplay matches it."""
    cfg = PIPER_VOICE + ".json"
    try:
        with open(cfg, encoding="utf-8") as fh:
            return str(json.load(fh).get("audio", {}).get("sample_rate", default))
    except Exception:
        return os.environ.get("PIPER_SR", default)


def speak(text: str):
    """Pipe `text` through Piper and play the audio out the default ALSA device."""
    if not os.path.isfile(PIPER_BIN):
        raise SystemExit(
            f"Piper binary not found at {PIPER_BIN}. Install it (see STT-TTS.md) "
            "or set PIPER_BIN."
        )
    if not os.path.isfile(PIPER_VOICE):
        raise SystemExit(
            f"Piper voice not found at {PIPER_VOICE}. Download a Hindi voice "
            "(.onnx + .onnx.json) or set PIPER_VOICE."
        )
    if shutil.which("aplay") is None:
        raise SystemExit(
            "aplay not found. Install ALSA utils on the Pi:  sudo apt install -y alsa-utils"
        )

    sr = _voice_sample_rate()
    # ALSA device. "Host is down" / silence usually means the DEFAULT device is
    # wrong; point ALSA_DEVICE at the real card (e.g. hw:0,0 for the Pi 3 B+
    # headphone jack, which aplay -l shows as card 0). Empty = aplay's default.
    aplay_cmd = ["aplay", "-q", "-r", sr, "-f", "S16_LE", "-t", "raw"]
    device = os.environ.get("ALSA_DEVICE", "hw:0,0")
    if device:
        aplay_cmd += ["-D", device]
    aplay_cmd.append("-")

    piper = subprocess.Popen(
        [PIPER_BIN, "--model", PIPER_VOICE, "--output-raw"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    aplay = subprocess.Popen(aplay_cmd, stdin=piper.stdout)
    piper.stdout.close()  # let aplay own the pipe end
    piper.stdin.write(text.encode("utf-8"))
    piper.stdin.close()
    aplay.wait()
    piper.wait()


def main():
    ap = argparse.ArgumentParser(description="Translate EN->Indic and speak on the Pi")
    ap.add_argument("text", nargs="*", help="English text to translate + speak")
    ap.add_argument("--tgt", default="hin_Deva", help="target language code")
    ap.add_argument("--beams", type=int, default=1, help="beam size (1=greedy)")
    ap.add_argument("--no-audio", action="store_true",
                    help="translate + print only, do not synthesize/play")
    ap.add_argument("--model_dir", default=str(CT2_DIR))
    ap.add_argument("--tokenizer_dir", default=str(HF_SNAPSHOT))
    args = ap.parse_args()

    if not args.text:
        raise SystemExit('usage: python -m it2edge.serve.speak "English text"')
    english = " ".join(args.text)

    tokenizer, translator, processor = load(args.model_dir, args.tokenizer_dir)
    hindi = translate([english], args.tgt, tokenizer, translator, processor, args.beams)[0]
    print(f"EN: {english}")
    print(f"-> ({args.tgt}): {hindi}")

    if not args.no_audio:
        speak(hindi)


if __name__ == "__main__":
    main()
