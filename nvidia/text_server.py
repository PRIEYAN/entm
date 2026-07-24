"""TCP text server for the Jetson: receive English lines, translate + speak.

    python -m nvidia.text_server                  # listen on 0.0.0.0:8765
    python -m nvidia.text_server --port 9000
    python -m nvidia.text_server --no-audio       # translate + print only

This is the network-facing half of the "Whisper on the laptop" split: the
laptop runs speech-to-text and streams finished ENGLISH sentences here, one
per line, over TCP. The Jetson never touches the mic in this mode — it only
translates (CT2/ONNX/TRT, per TRANSLATE_BACKEND) and speaks (see
it2edge.serve.speak; AUDIO_OUT=adb sends audio to the phone).

Why a separate server from nvidia/server.py:
  * nvidia/server.py is a UNIX-socket, AUDIO-in server (STT runs on the Jetson).
  * This one is a TCP, TEXT-in server (STT runs on the laptop). Different
    topology, different wire format — kept separate so neither is muddied.

Wire format (deliberately trivial): UTF-8, one English sentence per '\n'-
terminated line, client -> server. Server -> client: one '\n'-terminated UTF-8
JSON line per sentence, {"english":..., "hindi":..., "mt_ms":...}, so the
laptop can log/inspect what was said. Text has no length ambiguity against a
newline, so no binary framing is needed here.
"""

import argparse
import json
import os
import socket
import threading

from nvidia.engine import Engine

DEFAULT_HOST = "0.0.0.0"          # all interfaces, so the laptop can reach it
DEFAULT_PORT = 8765


class TextServer:
    def __init__(self, host, port, do_speak=True):
        self.host = host
        self.port = port
        self.do_speak = do_speak
        self.engine = Engine()

    def start(self):
        # with_stt=False: this box never transcribes, so skip Whisper entirely.
        self.engine.load(with_stt=False)
        self._serve()

    def _serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(4)
        print("[text-server] listening on {}:{}  (MT:{}  TTS:{})".format(
            self.host, self.port, self.engine.ct2_device,
            "on" if self.do_speak else "off"))
        try:
            while True:
                conn, addr = srv.accept()
                print("[text-server] client connected: {}".format(addr))
                threading.Thread(
                    target=self._handle, args=(conn, addr), daemon=True
                ).start()
        except KeyboardInterrupt:
            print("\n[text-server] shutting down")
        finally:
            srv.close()

    def _handle(self, conn, addr):
        """One English sentence per line -> translate -> speak -> JSON reply."""
        f = conn.makefile("rwb")
        try:
            for raw in f:                       # blocks per line; real streaming
                english = raw.decode("utf-8", "replace").strip()
                if not english:
                    continue
                try:
                    t = self.engine.translate(english)
                    hindi = t.text
                    print("  EN: {}\n  HI: {}   (MT {:.1f} ms)".format(
                        english, hindi, t.ms))
                    if self.do_speak and hindi:
                        # speak() enqueues; playback is sequential on the phone.
                        self.engine.speak(hindi)
                    reply = {"english": english, "hindi": hindi,
                             "mt_ms": round(t.ms, 1)}
                except Exception as exc:        # noqa: BLE001 — never drop the connection
                    reply = {"english": english, "error": str(exc)}
                    print("[text-server] error: {}".format(exc))
                f.write((json.dumps(reply, ensure_ascii=False) + "\n").encode("utf-8"))
                f.flush()
        except Exception as exc:                # noqa: BLE001
            print("[text-server] client {} dropped: {}".format(addr, exc))
        finally:
            try:
                conn.close()
            except OSError:
                pass
            print("[text-server] client disconnected: {}".format(addr))


def main():
    ap = argparse.ArgumentParser(description="Jetson TCP text->translate->speak server")
    ap.add_argument("--host", default=os.environ.get("TEXT_HOST", DEFAULT_HOST))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("TEXT_PORT", DEFAULT_PORT)))
    ap.add_argument("--no-audio", action="store_true",
                    help="translate + print only, do not speak")
    args = ap.parse_args()
    TextServer(args.host, args.port, do_speak=not args.no_audio).start()


if __name__ == "__main__":
    main()
