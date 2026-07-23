"""Voice-activity detection — the always-listening gate for the mic client.

This is what makes the pipeline idle-when-silent: the client feeds fixed-size
audio frames in; the VAD says speech / not-speech per frame; a small state
machine turns that into "utterance started" / "utterance ended" so the client
only streams over the socket while someone is actually talking.

Two backends, same interface (`Vad.is_speech(frame_bytes)`):

    webrtcvad  (default)  tiny pure-C energy VAD, cp38/aarch64 wheel, no torch.
                          Set VAD_BACKEND=webrtc (default).
    silero     (opt-in)   neural VAD, more robust in noise, needs torch (the
                          Jetson torch wheel). Set VAD_BACKEND=silero.

Frame size: webrtcvad only accepts 10, 20, or 30 ms frames of 16-bit mono PCM.
At 16 kHz a 30 ms frame is 480 samples = 960 bytes. The client MUST feed frames
of exactly FRAME_MS; `frame_bytes()` gives the byte count to slice.

The utterance state machine (SpeechGate) adds:
  * a short START debounce (a couple of speech frames before we "open") so a
    click doesn't start an utterance, and
  * a trailing-silence HANGOVER (default 500 ms) so a natural pause mid-sentence
    doesn't prematurely end the utterance — the utterance ends only after the
    speaker has been quiet for the whole hangover.
"""

from __future__ import annotations

import os

from nvidia.protocol import SAMPLE_RATE, SAMPLE_WIDTH

FRAME_MS = int(os.environ.get("VAD_FRAME_MS", "30"))   # 10/20/30 for webrtcvad
_ALLOWED_FRAME_MS = (10, 20, 30)


def frame_samples() -> int:
    """Samples per VAD frame at 16 kHz."""
    return SAMPLE_RATE * FRAME_MS // 1000


def frame_bytes() -> int:
    """Bytes per VAD frame (int16 mono)."""
    return frame_samples() * SAMPLE_WIDTH


# --- backends -------------------------------------------------------------

class _WebrtcVad:
    """webrtcvad backend. aggressiveness 0..3 (higher = more aggressive filtering)."""

    def __init__(self, aggressiveness: int = 2):
        import webrtcvad

        if FRAME_MS not in _ALLOWED_FRAME_MS:
            raise ValueError(
                f"webrtcvad needs VAD_FRAME_MS in {_ALLOWED_FRAME_MS}, got {FRAME_MS}"
            )
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes) -> bool:
        # webrtcvad is strict about frame length; a short tail frame isn't speech.
        if len(frame) != frame_bytes():
            return False
        return self._vad.is_speech(frame, SAMPLE_RATE)


class _SileroVad:
    """Silero VAD backend (neural, needs torch). Threshold 0..1."""

    def __init__(self, threshold: float = 0.5):
        import numpy as np
        import torch

        self._np = np
        self._torch = torch
        self._threshold = threshold
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False
        )
        model.eval()
        self._model = model

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != frame_bytes():
            return False
        audio = self._np.frombuffer(frame, dtype=self._np.int16).astype("float32") / 32768.0
        with self._torch.no_grad():
            prob = self._model(self._torch.from_numpy(audio), SAMPLE_RATE).item()
        return prob >= self._threshold


class Vad:
    """Per-frame speech/not-speech. Picks the backend from VAD_BACKEND."""

    def __init__(self) -> None:
        backend = os.environ.get("VAD_BACKEND", "webrtc").lower()
        if backend == "silero":
            self._impl = _SileroVad(
                threshold=float(os.environ.get("VAD_SILERO_THRESHOLD", "0.5"))
            )
            self.name = "silero"
        else:
            self._impl = _WebrtcVad(
                aggressiveness=int(os.environ.get("VAD_AGGRESSIVENESS", "2"))
            )
            self.name = "webrtc"

    def is_speech(self, frame: bytes) -> bool:
        return self._impl.is_speech(frame)


# --- utterance state machine ----------------------------------------------

class SpeechGate:
    """Turn a stream of per-frame speech flags into utterance start/end edges.

    Feed one frame at a time via `update(is_speech)`. Returns one of:
        "start"    an utterance just began (send this frame + following ones)
        "speech"   still inside an utterance (keep sending)
        "end"      the utterance just ended (send END, stop until the next start)
        "silence"  outside an utterance (send nothing — this is the idle state)
    """

    def __init__(
        self,
        start_frames: int = 2,
        hangover_ms: int | None = None,
        max_utterance_ms: int | None = None,
    ):
        self._in_speech = False
        self._start_frames = max(1, start_frames)
        self._speech_run = 0            # consecutive speech frames while silent
        self._silence_run = 0           # consecutive silent frames while in speech
        self._utterance_frames = 0

        hang = hangover_ms if hangover_ms is not None else int(
            os.environ.get("VAD_HANGOVER_MS", "500")
        )
        self._hangover_frames = max(1, hang // FRAME_MS)

        maxu = max_utterance_ms if max_utterance_ms is not None else int(
            os.environ.get("VAD_MAX_UTTERANCE_MS", "20000")  # 20 s guard (~3+ lines)
        )
        self._max_frames = max(1, maxu // FRAME_MS)

    def update(self, is_speech: bool) -> str:
        if not self._in_speech:
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self._start_frames:
                    self._in_speech = True
                    self._speech_run = 0
                    self._silence_run = 0
                    self._utterance_frames = 1
                    return "start"
            else:
                self._speech_run = 0
            return "silence"

        # inside an utterance
        self._utterance_frames += 1
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1

        # end on sustained trailing silence, or on the max-length safety guard
        if self._silence_run >= self._hangover_frames or \
                self._utterance_frames >= self._max_frames:
            self._in_speech = False
            self._silence_run = 0
            self._utterance_frames = 0
            return "end"
        return "speech"
