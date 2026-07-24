"""Warm-model socket server for the Jetson voice pipeline.

    python -m nvidia.server                       # listen on the default UNIX socket
    python -m nvidia.server --socket /tmp/it2.sock
    python -m nvidia.server --no-audio            # translate only, don't run Piper

Implements jetsonNvidia.md §2. One long-lived process:

  * loads Whisper (GPU) + MarianMT int8 ONNX (GPU) + binds Piper (CPU) ONCE,
  * listens on a UNIX domain socket (no polling — the client streams; the server
    pushes results back as events),
  * runs a SENTENCE-LEVEL pipeline of three worker threads connected by bounded
    queues, so utterance N+1 transcribes on the GPU while Piper speaks N on the
    CPU, and the first Hindi line of a 3-line utterance plays before the last
    line is even transcribed (§11.3).

Idle-when-silent: during silence the client sends nothing, the queues are empty,
and all three workers block in queue.get() at 0% CPU. The GPU models stay warm.

Backpressure: each queue is bounded. If speech arrives faster than the box can
keep up, the OLDEST pending item is dropped and a DROPPED event is sent to the
client — never a silent drop, never unbounded RAM growth.
"""


import argparse
import os
import queue
import socket
import threading

from nvidia import protocol as proto
from nvidia.engine import Engine
from nvidia.protocol import EventKind

# Queue depths — small on purpose. Bigger queues just add latency under load;
# the right response to overload is to drop, visibly, not to buffer.
UTTERANCE_Q_MAX = int(os.environ.get("IT2_UTTERANCE_Q", "3"))
MT_Q_MAX = int(os.environ.get("IT2_MT_Q", "8"))
TTS_Q_MAX = int(os.environ.get("IT2_TTS_Q", "8"))


class _Client:
    """One connected client: its socket + a write lock (events come from many threads)."""

    __slots__ = ("sock", "_wlock", "id")

    def __init__(self, sock: socket.socket, cid: int):
        self.sock = sock
        self._wlock = threading.Lock()
        self.id = cid

    def event(self, kind: str, **fields) -> None:
        """Push one event to this client. Never raises — a dead client is fine."""
        try:
            with self._wlock:
                proto.send_event(self.sock, kind, **fields)
        except OSError:
            pass  # client went away mid-event; the recv loop will clean up


class _Utterance:
    """One captured utterance handed from the recv loop to the STT worker."""

    __slots__ = ("pcm", "client", "seq")

    def __init__(self, pcm: bytes, client: _Client, seq: int):
        self.pcm = pcm
        self.client = client
        self.seq = seq


def _put_drop_oldest(q: "queue.Queue", item, on_drop) -> None:
    """Enqueue, dropping the OLDEST item if the queue is full (bounded backpressure)."""
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        try:
            dropped = q.get_nowait()
            on_drop(dropped)
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            on_drop(item)  # racing producers; drop the newcomer rather than block


class Server:
    def __init__(self, socket_path: str, do_speak: bool = True):
        self.socket_path = socket_path
        self.do_speak = do_speak
        self.engine = Engine()

        self._utterance_q: "queue.Queue[_Utterance]" = queue.Queue(UTTERANCE_Q_MAX)
        # MT/TTS carry (client, seq, sentence_index, text). Bounded, drop-oldest.
        self._mt_q: "queue.Queue[tuple]" = queue.Queue(MT_Q_MAX)
        self._tts_q: "queue.Queue[tuple]" = queue.Queue(TTS_Q_MAX)
        self._stop = threading.Event()
        self._cid = 0

    # ----- lifecycle ------------------------------------------------------

    def start(self) -> None:
        self.engine.load(with_stt=True)
        threading.Thread(target=self._stt_worker, name="stt", daemon=True).start()
        threading.Thread(target=self._mt_worker, name="mt", daemon=True).start()
        if self.do_speak:
            threading.Thread(target=self._tts_worker, name="tts", daemon=True).start()
        self._serve()

    # ----- socket accept loop --------------------------------------------

    def _serve(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        os.makedirs(os.path.dirname(self.socket_path) or ".", exist_ok=True)

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.socket_path)
        os.chmod(self.socket_path, 0o660)
        srv.listen(4)
        print(f"[server] listening on {self.socket_path} "
              f"(STT:{self.engine.whisper_device}  MT:{self.engine.ct2_device}  "
              f"TTS:{'cpu' if self.do_speak else 'off'})")
        try:
            while not self._stop.is_set():
                conn, _ = srv.accept()
                self._cid += 1
                client = _Client(conn, self._cid)
                threading.Thread(
                    target=self._recv_loop, args=(client,),
                    name=f"conn-{self._cid}", daemon=True,
                ).start()
        except KeyboardInterrupt:
            print("\n[server] shutting down")
        finally:
            srv.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    def _recv_loop(self, client: _Client) -> None:
        """Assemble streamed AUDIO frames into utterances; enqueue on END."""
        buf = bytearray()
        seq = 0
        client.event(EventKind.READY)
        try:
            while not self._stop.is_set():
                try:
                    mtype, payload = proto.recv_msg(client.sock)
                except EOFError:
                    break
                except proto.ProtocolError as exc:
                    client.event(EventKind.ERROR, detail=f"protocol: {exc}")
                    break

                if mtype == proto.AUDIO:
                    if not buf:
                        client.event(EventKind.LISTENING)
                    buf.extend(payload)
                elif mtype == proto.END:
                    if buf:
                        seq += 1
                        self._enqueue_utterance(bytes(buf), client, seq)
                        buf.clear()
                elif mtype == proto.BYE:
                    break
        finally:
            try:
                client.sock.close()
            except OSError:
                pass

    def _enqueue_utterance(self, pcm: bytes, client: _Client, seq: int) -> None:
        def on_drop(u: _Utterance):
            u.client.event(EventKind.DROPPED, seq=u.seq,
                           reason="server overloaded (utterance queue full)")
            print(f"[server] DROPPED utterance seq={u.seq} (client {u.client.id})")
        _put_drop_oldest(self._utterance_q, _Utterance(pcm, client, seq), on_drop)

    # ----- stage workers --------------------------------------------------

    def _stt_worker(self) -> None:
        import numpy as np

        while not self._stop.is_set():
            u = self._utterance_q.get()
            try:
                audio = np.frombuffer(u.pcm, dtype=np.int16).astype(np.float32) / 32768.0
                idx = 0
                for english, ms in self.engine.stt_segments(audio):
                    idx += 1
                    self._enqueue_mt(u.client, u.seq, idx, english, ms)
                if idx == 0:
                    u.client.event(EventKind.FINAL, seq=u.seq, sentences=0,
                                   note="no speech detected")
            except Exception as exc:  # noqa: BLE001 — one bad utterance mustn't kill the worker
                u.client.event(EventKind.ERROR, seq=u.seq, detail=f"stt: {exc}")

    def _enqueue_mt(self, client, seq, idx, english, stt_ms) -> None:
        def on_drop(item):
            item[0].event(EventKind.DROPPED, seq=item[1], sentence=item[2],
                          reason="server overloaded (MT queue full)")
        _put_drop_oldest(self._mt_q, (client, seq, idx, english, stt_ms), on_drop)

    def _mt_worker(self) -> None:
        while not self._stop.is_set():
            client, seq, idx, english, stt_ms = self._mt_q.get()
            try:
                t = self.engine.translate(english)
                hindi = t.text
                # PARTIAL: one finished sentence, pushed back the instant it's ready.
                client.event(EventKind.PARTIAL, seq=seq, sentence=idx,
                             english=english, hindi=hindi,
                             stt_ms=round(stt_ms, 1), mt_ms=round(t.ms, 1))
                if self.do_speak and hindi:
                    self._enqueue_tts(client, seq, idx, hindi)
            except Exception as exc:  # noqa: BLE001
                client.event(EventKind.ERROR, seq=seq, sentence=idx, detail=f"mt: {exc}")

    def _enqueue_tts(self, client, seq, idx, hindi) -> None:
        def on_drop(item):
            item[0].event(EventKind.DROPPED, seq=item[1], sentence=item[2],
                          reason="server overloaded (TTS queue full)")
        _put_drop_oldest(self._tts_q, (client, seq, idx, hindi), on_drop)

    def _tts_worker(self) -> None:
        while not self._stop.is_set():
            client, seq, idx, hindi = self._tts_q.get()
            try:
                self.engine.speak(hindi)   # Piper subprocess on the CPU; streams to aplay
            except Exception as exc:  # noqa: BLE001 — never crash the loop on audio
                client.event(EventKind.ERROR, seq=seq, sentence=idx, detail=f"tts: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Jetson voice pipeline socket server")
    ap.add_argument("--socket", default=os.environ.get("IT2_SOCKET", proto.DEFAULT_SOCKET_PATH),
                    help="UNIX socket path to listen on")
    ap.add_argument("--no-audio", action="store_true",
                    help="translate only; don't spawn Piper/TTS")
    args = ap.parse_args()

    Server(args.socket, do_speak=not args.no_audio).start()


if __name__ == "__main__":
    main()
