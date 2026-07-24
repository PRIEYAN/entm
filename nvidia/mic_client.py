"""Laptop mic client: record -> faster-whisper -> send English text to the Jetson.

    python -m nvidia.mic_client --host 192.168.1.42          # Jetson's IP
    python -m nvidia.mic_client --host jetson.local --port 8765
    WHISPER_MODEL=small python -m nvidia.mic_client --host 192.168.1.42

Runs on the LAPTOP (not the Jetson). It:
  1. captures the microphone at 16 kHz mono,
  2. watches the audio level; ~1 s of silence marks the end of a sentence,
  3. transcribes that sentence with faster-whisper (English),
  4. sends the text, one '\n'-terminated line, over TCP to nvidia.text_server
     on the Jetson, which translates + speaks it on the phone.

The "1 s of silence = one sentence" rule is the segmenter: speech accumulates
into a buffer; once the mic stays quiet for SILENCE_HANG seconds, the buffer is
closed, transcribed, and sent. Talk again and the next buffer starts.

Dependencies (laptop):  pip install faster-whisper sounddevice numpy
sounddevice needs PortAudio (Linux: sudo apt install libportaudio2).
"""

import argparse
import os
import queue
import socket
import sys
import threading

import numpy as np

SAMPLE_RATE = 16000               # faster-whisper wants 16 kHz mono
BLOCK_S = 0.1                     # mic callback granularity (100 ms blocks)
SILENCE_HANG = float(os.environ.get("SILENCE_HANG", "1.0"))   # your 1 s rule
# RMS below this (on -1..1 float audio) counts as silence. Tune per mic/room:
# raise it in a noisy room, lower it if quiet speech gets cut off.
SILENCE_RMS = float(os.environ.get("SILENCE_RMS", "0.015"))
MIN_SPEECH_S = float(os.environ.get("MIN_SPEECH_S", "0.3"))   # ignore blips


def _rms(block):
    return float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0


class Segmenter:
    """Turn a stream of audio blocks into finished-sentence PCM buffers.

    Emits (via the out_queue) one float32 numpy array per sentence: everything
    from the first non-silent block up to SILENCE_HANG seconds of trailing quiet.
    """

    def __init__(self, out_queue):
        self.out = out_queue
        self.buf = []               # list of blocks in the current sentence
        self.silence_s = 0.0        # trailing quiet accumulated
        self.in_speech = False

    def feed(self, block):
        loud = _rms(block) >= SILENCE_RMS
        if loud:
            self.in_speech = True
            self.silence_s = 0.0
            self.buf.append(block)
        elif self.in_speech:
            # Quiet, but we were mid-sentence: keep the gap (natural pauses) and
            # count how long the silence has run.
            self.buf.append(block)
            self.silence_s += BLOCK_S
            if self.silence_s >= SILENCE_HANG:
                self._flush()

    def _flush(self):
        audio = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
        self.buf, self.silence_s, self.in_speech = [], 0.0, False
        if audio.size / SAMPLE_RATE >= MIN_SPEECH_S:
            self.out.put(audio)

    def close(self):
        if self.in_speech:
            self._flush()


def _transcribe_and_send(model, audio_q, sock_file, lang):
    """Consumer: pop finished sentences, run Whisper, send text to the Jetson."""
    while True:
        audio = audio_q.get()
        if audio is None:                       # shutdown sentinel
            break
        segments, _ = model.transcribe(audio, language=lang)
        text = " ".join(s.text for s in segments).strip()
        if not text:
            continue
        print(">> {}".format(text))
        try:
            sock_file.write((text + "\n").encode("utf-8"))
            sock_file.flush()
        except Exception as exc:                # noqa: BLE001
            print("[mic-client] send failed: {}".format(exc))
            break


def main():
    ap = argparse.ArgumentParser(description="Laptop mic -> Whisper -> Jetson text server")
    ap.add_argument("--host", required=True, help="Jetson IP or hostname")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("TEXT_PORT", "8765")))
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    # Import here so --help works without the heavy deps installed.
    import sounddevice as sd
    from faster_whisper import WhisperModel

    name = os.environ.get("WHISPER_MODEL", "base")
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    compute = os.environ.get("WHISPER_COMPUTE", "int8")
    print("[mic-client] loading faster-whisper '{}' on {} ({})".format(
        name, device, compute))
    model = WhisperModel(name, device=device, compute_type=compute)

    print("[mic-client] connecting to {}:{} ...".format(args.host, args.port))
    sock = socket.create_connection((args.host, args.port))
    sock_file = sock.makefile("rwb")
    print("[mic-client] connected. Speak — 1 s of silence ends a sentence. Ctrl-C to stop.")

    audio_q = queue.Queue()
    seg = Segmenter(audio_q)

    worker = threading.Thread(
        target=_transcribe_and_send,
        args=(model, audio_q, sock_file, args.lang), daemon=True)
    worker.start()

    block_frames = int(SAMPLE_RATE * BLOCK_S)

    def on_audio(indata, frames, time_info, status):   # sounddevice callback
        if status:
            print("[mic-client] audio status: {}".format(status), file=sys.stderr)
        seg.feed(indata[:, 0].copy())          # mono channel 0, -1..1 float32

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=block_frames,
                            callback=on_audio):
            threading.Event().wait()           # run until Ctrl-C
    except KeyboardInterrupt:
        print("\n[mic-client] stopping")
    finally:
        seg.close()                            # flush any half-spoken sentence
        audio_q.put(None)
        worker.join(timeout=5)
        try:
            sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
