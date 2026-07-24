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

# The Jetson Nano runs Python 3.6, which can't subscript builtin generics
# (`tuple[...]`) at runtime and has no `from __future__ import annotations`
# (that's 3.7+). So use typing.Tuple for the one annotated return below.

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import wave
try:
    import queue as _queue          # py3
except ImportError:                 # pragma: no cover
    import Queue as _queue          # py2 fallback (unused on this board)
from typing import Tuple

from it2edge.paths import CT2_DIR
from it2edge.serve.marian_ct2 import load_marian, translate_marian

PIPER_BIN = os.environ.get("PIPER_BIN", os.path.expanduser("~/piper/piper"))
PIPER_VOICE = os.environ.get(
    "PIPER_VOICE", os.path.expanduser("~/piper_voices/hi_IN-pratham-medium.onnx")
)

# One utterance at a time: two Piper processes sharing one sound card produce
# garbled audio, and on a 1 GB board they can also fight for memory.
_speak_lock = threading.Lock()


# --- espeak-ng + adb backend (AUDIO_OUT=adb) --------------------------------
# The prebuilt Piper binary needs GLIBC_2.29 / GLIBCXX_3.4.26, which the Jetson
# Nano (Ubuntu 18.04, glibc 2.27) does not have, so the binary can't run here.
# This backend instead synthesizes Hindi with espeak-ng (built against the
# board's own glibc) and plays it on an ADB-connected phone via VLC. It writes
# a small WAV, pushes it, and asks VLC to open it — one utterance at a time.
ESPEAK_VOICE = os.environ.get("ESPEAK_VOICE", "hi")
ADB_BIN = os.environ.get("ADB_BIN", "adb")
ADB_REMOTE = os.environ.get("ADB_REMOTE", "/sdcard/Movies/entm_tts.wav")
VLC_PKG = os.environ.get("VLC_PKG", "org.videolan.vlc")


# A single background consumer drains this queue so utterances play strictly
# one-after-another. The phone (VLC via an intent) gives no "playback finished"
# callback, so the consumer waits out each clip's real duration — read from the
# WAV header — before sending the next. Producers (translation) never block on
# audio: they synthesize a WAV and drop it on the queue.
ADB_CACHE_DIR = os.environ.get("ADB_CACHE_DIR", "/tmp/entm_tts")
ADB_REMOTE_DIR = os.environ.get("ADB_REMOTE_DIR", "/sdcard/Movies")
# Fixed pad after each clip so VLC has time to launch/route before the next
# push, and clips never clip into each other.
ADB_GAP_S = float(os.environ.get("ADB_GAP_S", "0.35"))

_adb_queue = None          # type: ignore  # lazily created queue.Queue
_adb_worker = None         # type: ignore  # consumer thread
_adb_seq = 0               # unique id per clip so a fast producer can't
                           # overwrite a WAV that's still playing on the phone


def _wav_duration_s(path):
    """Seconds of audio in a PCM WAV, from its header (no external tool)."""
    try:
        with wave.open(path, "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
            return frames / float(rate) if rate else 0.0
    except Exception:
        return 0.0


def _adb_consumer():
    """Pop WAVs and play them on the phone, one fully before the next."""
    while True:
        local_wav, remote = _adb_queue.get()
        try:
            subprocess.run([ADB_BIN, "push", local_wav, remote], check=True)
            subprocess.run(
                [ADB_BIN, "shell", "am", "start",
                 "-a", "android.intent.action.VIEW",
                 "-d", "file://" + remote, "-t", "audio/wav", VLC_PKG],
                check=True,
            )
            # Block until this clip has finished playing before the next push.
            time.sleep(_wav_duration_s(local_wav) + ADB_GAP_S)
        except Exception as exc:  # noqa: BLE001 — one bad clip must not kill the queue
            print("[speak] adb playback failed: {}".format(exc))
        finally:
            _adb_queue.task_done()


def _ensure_adb_worker():
    global _adb_queue, _adb_worker
    if _adb_worker is None:
        os.makedirs(ADB_CACHE_DIR, exist_ok=True)
        _adb_queue = _queue.Queue()
        _adb_worker = threading.Thread(target=_adb_consumer, daemon=True)
        _adb_worker.start()


def _speak_adb(text: str):
    """espeak-ng -> cached WAV -> enqueue for sequential playback on the phone.

    Returns as soon as the WAV is queued; the background consumer pushes and
    plays it, waiting out each clip before starting the next.
    """
    if shutil.which("espeak-ng") is None:
        raise SystemExit("espeak-ng not found. Install:  sudo apt install -y espeak-ng")
    if shutil.which(ADB_BIN) is None:
        raise SystemExit(f"{ADB_BIN} not found (set ADB_BIN or install android-tools-adb)")

    global _adb_seq
    _ensure_adb_worker()
    with _speak_lock:                      # serialize espeak + seq bump only
        _adb_seq += 1
        seq = _adb_seq
    local_wav = os.path.join(ADB_CACHE_DIR, "utt_{:06d}.wav".format(seq))
    remote = ADB_REMOTE_DIR.rstrip("/") + "/entm_tts_{:06d}.wav".format(seq)
    subprocess.run(
        ["espeak-ng", "-v", ESPEAK_VOICE, "-w", local_wav, text], check=True
    )
    _adb_queue.put((local_wav, remote))    # consumer plays it in order


def tts_available() -> Tuple[bool, str]:
    """Report whether TTS can run, without raising.

    Returns (ok, reason). Used by /health so the service can start and serve
    text translations even on a machine with no Piper or no sound card.
    """
    backend = os.environ.get("AUDIO_OUT", "alsa").lower()
    if backend == "adb":
        if shutil.which("espeak-ng") is None:
            return False, "espeak-ng not found (AUDIO_OUT=adb)"
        if shutil.which(ADB_BIN) is None:
            return False, f"{ADB_BIN} not found (AUDIO_OUT=adb)"
        return True, "ok"
    if not os.path.isfile(PIPER_BIN):
        return False, f"piper binary not found at {PIPER_BIN}"
    if not os.path.isfile(PIPER_VOICE):
        return False, f"piper voice not found at {PIPER_VOICE}"
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
    # espeak-ng -> adb -> phone speaker (works on the glibc-2.27 Jetson, where
    # the prebuilt Piper binary cannot link). Selected with AUDIO_OUT=adb.
    if os.environ.get("AUDIO_OUT", "alsa").lower() == "adb":
        _speak_adb(text)
        return

    if not os.path.isfile(PIPER_BIN):
        raise SystemExit(
            f"Piper binary not found at {PIPER_BIN}. Install it (see nvidia/NvidiaRun.md) "
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
