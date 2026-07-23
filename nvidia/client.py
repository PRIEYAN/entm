"""Always-listening mic client for the Jetson voice pipeline.

    python -m nvidia.client                        # listen forever, stream on speech
    python -m nvidia.client --socket /tmp/it2.sock
    python -m nvidia.client --vad-test             # no socket: just show speech/silence

Implements the CLIENT half of jetsonNvidia.md §2. It holds NO models. It:

  1. opens the mic at 16 kHz mono and reads it in VAD-sized frames,
  2. runs the VAD gate (nvidia/vad.py): SILENCE sends nothing (idle), SPEECH
     streams raw PCM frames over the socket, and a trailing pause sends END,
  3. prints the Hindi results the server PUSHES back — partial (per sentence,
     as soon as each is ready) and final.

Because frames only cross the socket while you're talking, the socket is silent
during silence and the whole box idles — no polling anywhere.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading

from nvidia import protocol as proto
from nvidia import vad as vadmod
from nvidia.protocol import EventKind


def _connect(path: str) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(path)
    except OSError as exc:
        raise SystemExit(
            f"could not connect to server at {path}: {exc}\n"
            "start it first:  python -m nvidia.server"
        )
    return s


def _event_reader(sock: socket.socket, stop: threading.Event) -> None:
    """Print events the server pushes back, until the socket closes."""
    while not stop.is_set():
        try:
            mtype, payload = proto.recv_msg(sock)
        except (EOFError, OSError):
            break
        except proto.ProtocolError as exc:
            print(f"  [protocol error: {exc}]")
            continue
        if mtype != proto.EVENT:
            continue
        ev = proto.parse_event(payload)
        kind = ev.get("kind")
        if kind == EventKind.READY:
            print("[server ready — start talking]")
        elif kind == EventKind.LISTENING:
            print("  🎤 …", flush=True)
        elif kind == EventKind.PARTIAL:
            print(f"  EN: {ev.get('english','')}")
            print(f"  HI: {ev.get('hindi','')}   "
                  f"(STT {ev.get('stt_ms','?')} ms | MT {ev.get('mt_ms','?')} ms)")
        elif kind == EventKind.FINAL:
            note = ev.get("note")
            if note:
                print(f"  [{note}]")
        elif kind == EventKind.DROPPED:
            print(f"  ⚠ dropped: {ev.get('reason','overloaded')}")
        elif kind == EventKind.ERROR:
            print(f"  ⚠ error: {ev.get('detail','?')}")
    stop.set()


def run(socket_path: str) -> None:
    import sounddevice as sd

    vad = vadmod.Vad()
    gate = vadmod.SpeechGate()
    frame_len = vadmod.frame_samples()

    sock = _connect(socket_path)
    stop = threading.Event()
    reader = threading.Thread(target=_event_reader, args=(sock, stop), daemon=True)
    reader.start()

    print(f"Always-listening EN→HI (VAD:{vad.name}  frame:{vadmod.FRAME_MS}ms). Ctrl-C to quit.")
    stream = sd.RawInputStream(
        samplerate=proto.SAMPLE_RATE, channels=proto.CHANNELS,
        dtype="int16", blocksize=frame_len,
    )
    try:
        with stream:
            while not stop.is_set():
                data, _ = stream.read(frame_len)
                frame = bytes(data)
                state = gate.update(vad.is_speech(frame))
                if state == "start":
                    proto.send_audio(sock, frame)
                elif state == "speech":
                    proto.send_audio(sock, frame)
                elif state == "end":
                    proto.send_end(sock)
                # "silence" -> send nothing (idle)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        stop.set()
        try:
            proto.send_bye(sock)
        except OSError:
            pass
        sock.close()


def vad_test() -> None:
    """Local mic + VAD only — no socket. Prove the gate before wiring the server."""
    import sounddevice as sd

    vad = vadmod.Vad()
    gate = vadmod.SpeechGate()
    frame_len = vadmod.frame_samples()
    print(f"VAD test (backend:{vad.name}). Speak — Ctrl-C to quit.")
    stream = sd.RawInputStream(
        samplerate=proto.SAMPLE_RATE, channels=proto.CHANNELS,
        dtype="int16", blocksize=frame_len,
    )
    try:
        with stream:
            while True:
                data, _ = stream.read(frame_len)
                state = gate.update(vad.is_speech(bytes(data)))
                if state == "start":
                    print("🎤 speech start", flush=True)
                elif state == "end":
                    print("… silence (utterance end)", flush=True)
    except KeyboardInterrupt:
        print("\nbye")


def main() -> None:
    ap = argparse.ArgumentParser(description="Jetson voice pipeline mic client")
    ap.add_argument("--socket", default=os.environ.get("IT2_SOCKET", proto.DEFAULT_SOCKET_PATH),
                    help="server UNIX socket path")
    ap.add_argument("--vad-test", action="store_true",
                    help="no socket; print speech/silence edges to tune the VAD")
    args = ap.parse_args()

    if args.vad_test:
        vad_test()
    else:
        run(args.socket)


if __name__ == "__main__":
    main()
