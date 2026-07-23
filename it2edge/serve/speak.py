"""Translate English → Hindi and speak it on the Pi's speaker (offline).

    python -m it2edge.serve.speak "Hello, how are you?"
    python -m it2edge.serve.speak --no-audio "test"     # translate + print only

Also used by it2edge/serve/app.py to speak translations served over HTTP.

Configure Piper paths via env if yours differ:
    PIPER_BIN   (default ~/piper/piper)
    PIPER_VOICE (default ~/piper_voices/hi_IN-pratham-medium.onnx)
    PIPER_SR    (default 22050)
    AUDIO_OUT   alsa (default) | pulse
    ALSA_DEVICE default hw:0,0     (alsa backend)
    PULSE_SINK  optional sink name (pulse backend)

Design notes for the 1 GB Pi:
  * Piper runs as a SHORT-LIVED SUBPROCESS, one per utterance, so its RAM is
    released between sentences instead of being held by the long-running server.
  * Audio is streamed (`--output-raw | aplay`) — playback starts before
    synthesis finishes, and nothing is written to the SD card.
"""

# `tuple[bool, str]` and other PEP 585 hints below are only subscriptable at
# runtime on Python 3.9+. The Jetson Nano runs Python 3.8, so defer hint
# evaluation to keep this module importable there.
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading

from it2edge.paths import CT2_DIR
from it2edge.serve.marian_ct2 import load_marian, translate_marian

PIPER_BIN = os.environ.get("PIPER_BIN", os.path.expanduser("~/piper/piper"))
PIPER_VOICE = os.environ.get(
    "PIPER_VOICE", os.path.expanduser("~/piper_voices/hi_IN-pratham-medium.onnx")
)

# One utterance at a time: two Piper processes sharing one sound card produce
# garbled audio, and on a 1 GB board they can also fight for memory.
_speak_lock = threading.Lock()


def tts_available() -> tuple[bool, str]:
    """Report whether Piper TTS can run, without raising.

    Returns (ok, reason). Used by /health so the service can start and serve
    text translations even on a machine with no Piper or no sound card.
    """
    if not os.path.isfile(PIPER_BIN):
        return False, f"piper binary not found at {PIPER_BIN}"
    if not os.path.isfile(PIPER_VOICE):
        return False, f"piper voice not found at {PIPER_VOICE}"
    backend = os.environ.get("AUDIO_OUT", "alsa").lower()
    player = "paplay" if backend == "pulse" else "aplay"
    if shutil.which(player) is None:
        return False, f"{player} not found (backend={backend})"
    return True, "ok"


def _voice_sample_rate(default="22050"):
    cfg = PIPER_VOICE + ".json"
    try:
        with open(cfg, encoding="utf-8") as fh:
            return str(json.load(fh).get("audio", {}).get("sample_rate", default))
    except Exception:
        return os.environ.get("PIPER_SR", default)


def speak(text: str):
    if not os.path.isfile(PIPER_BIN):
        raise SystemExit(
            f"Piper binary not found at {PIPER_BIN}. Install it (see docs/STT-TTS.md) "
            "or set PIPER_BIN."
        )
    if not os.path.isfile(PIPER_VOICE):
        raise SystemExit(
            f"Piper voice not found at {PIPER_VOICE}. Download a Hindi voice "
            "(.onnx + .onnx.json) or set PIPER_VOICE."
        )

    sr = _voice_sample_rate()
    backend = os.environ.get("AUDIO_OUT", "alsa").lower()
    if backend == "pulse":
        play_cmd = [
            "paplay", "--raw", f"--rate={sr}", "--format=s16le", "--channels=1"
        ]
        sink = os.environ.get("PULSE_SINK", "")
        if sink:
            play_cmd += [f"--device={sink}"]
        if shutil.which("paplay") is None:
            raise SystemExit(
                "paplay not found. Install:  sudo apt install -y pulseaudio-utils"
            )
    else:
        play_cmd = ["aplay", "-q", "-r", sr, "-f", "S16_LE", "-t", "raw"]
        device = os.environ.get("ALSA_DEVICE", "hw:0,0")
        if device:
            play_cmd += ["-D", device]
        play_cmd.append("-")
        if shutil.which("aplay") is None:
            raise SystemExit(
                "aplay not found. Install:  sudo apt install -y alsa-utils"
            )

    # Serialize: concurrent utterances would garble the single sound card.
    with _speak_lock:
        piper = subprocess.Popen(
            [PIPER_BIN, "--model", PIPER_VOICE, "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        try:
            player = subprocess.Popen(play_cmd, stdin=piper.stdout)
        except OSError as exc:
            piper.kill()
            piper.wait()
            raise SystemExit(f"could not start audio player: {exc}")

        # Close our copy so piper sees EOF/SIGPIPE if the player dies.
        piper.stdout.close()
        try:
            piper.stdin.write(text.encode("utf-8"))
            piper.stdin.close()
        except BrokenPipeError:
            pass  # piper exited early; the wait() below surfaces its status

        # Wait for BOTH, always — otherwise a failed player leaves piper as a
        # zombie holding memory on a board that has little to spare.
        player_rc = player.wait()
        piper_rc = piper.wait()

    if piper_rc != 0:
        raise RuntimeError(f"piper exited with status {piper_rc}")
    if player_rc != 0:
        raise RuntimeError(f"audio player exited with status {player_rc}")


def main():
    ap = argparse.ArgumentParser(description="Translate EN→HI and speak on the Pi")
    ap.add_argument("text", nargs="*", help="English text to translate + speak")
    ap.add_argument("--beams", type=int, default=1, help="beam size (1=greedy)")
    ap.add_argument(
        "--no-audio",
        action="store_true",
        help="translate + print only, do not synthesize/play",
    )
    ap.add_argument("--model_dir", default=str(CT2_DIR))
    ap.add_argument("--tokenizer_dir", default=None)
    args = ap.parse_args()

    if not args.text:
        raise SystemExit('usage: python -m it2edge.serve.speak "English text"')
    english = " ".join(args.text)

    tokenizer, translator = load_marian(args.model_dir, args.tokenizer_dir)
    hindi = translate_marian(
        [english], tokenizer, translator, beam_size=args.beams
    )[0]
    print(f"EN: {english}")
    print(f"-> {hindi}")

    if not args.no_audio:
        speak(hindi)


if __name__ == "__main__":
    main()
