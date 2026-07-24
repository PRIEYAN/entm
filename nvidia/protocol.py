"""Wire protocol for the Jetson voice pipeline's client <-> server socket.

One UNIX domain socket, two directions:

    CLIENT -> SERVER   audio frames + control markers   (length-prefixed binary)
    SERVER -> CLIENT   status/result events             (length-prefixed JSON)

Both directions use the SAME framing: a 5-byte header (1 type byte + 4-byte
big-endian length) followed by `length` bytes of payload. Length-prefixing is
what makes this a real stream and not polling — a reader blocks in recv() until
a whole message arrives, then hands it up; there's no "is it ready yet?" loop.

Message types (the header's first byte)
----------------------------------------
CLIENT -> SERVER
    AUDIO   raw PCM: int16 mono @ 16 kHz, one VAD-passed chunk. Payload = bytes.
    END     "I stopped talking" — close the current utterance and process it.
            Payload = empty. (The server may also close an utterance on its own
            max-length guard; END is the client's normal signal.)
    BYE     client is disconnecting; server drops the connection cleanly.

SERVER -> CLIENT   (payload = UTF-8 JSON object, one per message)
    EVENT   {"kind": ...} — see EventKind below. This is how the server PUSHES
            partial and final results back without the client ever asking.

Design choices
--------------
* Binary up, JSON down: audio is bulk bytes (framing only), while results are
  small structured records (JSON is convenient and human-debuggable).
* 4-byte length caps a single message at 4 GiB — far more than any audio chunk
  or event needs; MAX_PAYLOAD below rejects anything absurd early.
* No versioning handshake: client and server ship together in nvidia/. If the
  wire format changes, both change.
"""


import json
import socket
import struct

SAMPLE_RATE = 16000          # Whisper wants 16 kHz mono
SAMPLE_WIDTH = 2             # int16 == 2 bytes/sample
CHANNELS = 1

DEFAULT_SOCKET_PATH = "/run/it2/it2.sock"

# --- message type bytes ---------------------------------------------------
AUDIO = 0x01
END = 0x02
BYE = 0x03
EVENT = 0x10

_HEADER = struct.Struct(">BI")   # 1 type byte + 4-byte big-endian length
HEADER_LEN = _HEADER.size
MAX_PAYLOAD = 64 * 1024 * 1024   # 64 MiB hard ceiling; a real chunk is a few KB


class ProtocolError(Exception):
    """Malformed frame on the wire (bad length, truncated stream, etc.)."""


# --- server->client event vocabulary --------------------------------------
class EventKind:
    """`kind` values for EVENT messages (server -> client)."""

    READY = "ready"          # server has warm models and is accepting audio
    LISTENING = "listening"  # utterance opened (first audio frame received)
    PARTIAL = "partial"      # one sentence transcribed+translated (streamed early)
    FINAL = "final"          # the whole utterance is done (summary of all sentences)
    DROPPED = "dropped"      # utterance skipped (queue full / overloaded) — visible, never silent
    ERROR = "error"          # a stage failed; pipeline stays alive


# --- framing: send / recv one message -------------------------------------

def send_msg(sock: socket.socket, msg_type: int, payload: bytes = b"") -> None:
    """Send one framed message. Thread-unsafe per socket — serialize writes."""
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload too large: {len(payload)} > {MAX_PAYLOAD}")
    sock.sendall(_HEADER.pack(msg_type, len(payload)) + payload)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes, or raise if the peer closed mid-message."""
    chunks = []
    remaining = n
    while remaining:
        b = sock.recv(remaining)
        if not b:
            if remaining == n:
                raise EOFError("peer closed")          # clean close between messages
            raise ProtocolError("peer closed mid-message")
        chunks.append(b)
        remaining -= len(b)
    return b"".join(chunks)


def recv_msg(sock: socket.socket):
    """Receive one framed message. Returns (msg_type, payload).

    Raises EOFError on a clean close (call it a disconnect), ProtocolError on a
    corrupt/truncated frame.
    """
    header = _recv_exactly(sock, HEADER_LEN)
    msg_type, length = _HEADER.unpack(header)
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"declared length too large: {length}")
    payload = _recv_exactly(sock, length) if length else b""
    return msg_type, payload


# --- typed helpers on top of the framing -----------------------------------

def send_audio(sock: socket.socket, pcm: bytes) -> None:
    send_msg(sock, AUDIO, pcm)


def send_end(sock: socket.socket) -> None:
    send_msg(sock, END)


def send_bye(sock: socket.socket) -> None:
    send_msg(sock, BYE)


def send_event(sock: socket.socket, kind: str, **fields) -> None:
    """Server -> client: one JSON event. Extra fields are merged into the object."""
    obj = {"kind": kind, **fields}
    send_msg(sock, EVENT, json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def parse_event(payload: bytes) -> dict:
    """Decode an EVENT payload into a dict. Raises ProtocolError on bad JSON."""
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"bad event JSON: {exc}") from exc
