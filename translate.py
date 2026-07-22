"""Voice → Hindi speech. Runs on the GPU LAPTOP; the Pi translates and speaks.

    🎤 laptop mic → Whisper STT (this machine, GPU) → English text
                         │  HTTP POST /speak
                         ▼
              Raspberry Pi → MarianMT int8 → Hindi text → Piper TTS → 🔊

Unlike laptop_mic.py (which drives the Pi over SSH), this talks to the Pi's
HTTP service — no SSH round-trip, no password prompts, and the Pi's model stays
warm between utterances.

Setup on the laptop (once):
    pip install faster-whisper sounddevice soundfile requests
    # GPU Whisper also needs:
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

Make sure the Pi's service is running:
    ssh root@<PI_IP>
    cd ~/entm && source venv/bin/activate
    uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1

Usage:
    python translate.py                      # push-to-talk loop
    python translate.py --seconds 7          # longer recording window
    python translate.py --text "Hello"       # skip the mic (test the endpoint)
    python translate.py --no-speak           # translate only, Pi stays silent
    python translate.py --list-devices       # show microphones, then exit
    python translate.py --device 5           # pick a specific mic

Env overrides:
    PI_URL          default http://10.42.0.187:8080
    WHISPER_MODEL   tiny|base|small|medium   (default small)
    WHISPER_DEVICE  cuda|cpu                 (default cuda, falls back to cpu)
    MIC_DEVICE      sounddevice input index
"""

import argparse
import os
import sys
import tempfile
import time

import requests

# --- configuration (env wins) ---------------------------------------------
PI_URL = os.environ.get("PI_URL", "http://10.42.0.187:8080").rstrip("/")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
SR = 16000  # Whisper expects 16 kHz mono

_WHISPER = None


# --- Whisper (STT) ---------------------------------------------------------
def _add_pip_cuda_libs() -> None:
    """Put the pip `nvidia-*-cu12` lib dirs on LD_LIBRARY_PATH.

    ctranslate2 (faster-whisper's backend) dlopens libcublas/libcudnn at
    runtime; when CUDA came from pip wheels rather than a system install, those
    live inside the `nvidia` namespace package and aren't on the loader path.
    No-op when the packages aren't present.
    """
    try:
        import nvidia
    except ImportError:
        print("[warn] nvidia CUDA pip libs not found. For GPU Whisper:\n"
              "       pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
        return

    libdirs = []
    for base in list(getattr(nvidia, "__path__", [])):
        for sub in ("cublas", "cudnn"):
            d = os.path.join(base, sub, "lib")
            if os.path.isdir(d):
                libdirs.append(d)
    if not libdirs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(libdirs + ([cur] if cur else []))


def get_whisper():
    """Load Whisper once, preferring the GPU.

    A broken CUDA install otherwise only fails midway through the first real
    transcription, so we force the GPU path immediately with a second of
    silence and fall back to CPU if that raises.
    """
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER

    want = os.environ.get("WHISPER_DEVICE", "cuda").lower()
    if want == "cuda":
        _add_pip_cuda_libs()

    from faster_whisper import WhisperModel

    if want == "cuda":
        try:
            import numpy as np

            t0 = time.perf_counter()
            model = WhisperModel(WHISPER_MODEL, device="cuda",
                                 compute_type="float16")
            model.transcribe(np.zeros(SR, dtype=np.float32), language="en")
            print(f"[info] Whisper '{WHISPER_MODEL}' on CUDA "
                  f"({time.perf_counter() - t0:.1f}s to load)")
            _WHISPER = model
            return _WHISPER
        except Exception as exc:  # noqa: BLE001 — any CUDA fault means fall back
            print(f"[warn] CUDA unavailable ({type(exc).__name__}: {exc})")
            print("       falling back to CPU; set WHISPER_DEVICE=cpu to skip this")

    t0 = time.perf_counter()
    _WHISPER = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print(f"[info] Whisper '{WHISPER_MODEL}' on CPU "
          f"({time.perf_counter() - t0:.1f}s to load)")
    return _WHISPER


def _pick_input(device=None):
    """Find an (device, samplerate) pair the sound card will actually open.

    Most laptop codecs (e.g. Realtek ALC245) refuse a 16 kHz capture stream even
    though Whisper wants 16 kHz, and the ALSA 'default'/pipewire aliases often
    fail outright under PortAudio. So probe real rates and resample later.
    """
    import sounddevice as sd

    if device is not None:
        candidates = [device]
    else:
        candidates = []
        try:
            default_in = sd.default.device[0]
            if default_in is not None and default_in >= 0:
                candidates.append(default_in)
        except (TypeError, IndexError):
            pass
        # Real hardware inputs, preferred over 'default'/'pipewire' aliases,
        # which frequently fail to open here.
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and i not in candidates:
                candidates.append(i)

    # 16k first (no resample needed), then the rates codecs actually support.
    for idx in candidates:
        for rate in (SR, 48000, 44100):
            try:
                sd.check_input_settings(device=idx, samplerate=rate, channels=1)
                return idx, rate
            except Exception:  # noqa: BLE001 — probing; any failure means "try next"
                continue

    raise SystemExit(
        "No usable microphone found.\n"
        "  List devices:   python translate.py --list-devices\n"
        "  Pick one:       python translate.py --device N\n"
        "  Check the OS sees a mic:  arecord -l"
    )


def _resample_to_16k(audio, rate: int):
    """Linear-resample mono audio to 16 kHz (what Whisper expects)."""
    import numpy as np

    if rate == SR:
        return audio
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n_out = int(round(len(audio) * SR / rate))
    # np.interp is enough for speech: mics are already band-limited well below
    # the 8 kHz Nyquist we're downsampling to, so aliasing isn't a concern.
    return np.interp(
        np.linspace(0.0, len(audio) - 1, n_out, dtype=np.float64),
        np.arange(len(audio), dtype=np.float64),
        audio,
    ).astype(np.float32)


def record(seconds: int, device=None) -> str:
    """Record from the mic and return a temp 16 kHz mono .wav path."""
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    idx, rate = _pick_input(device)
    note = "" if rate == SR else f" (device rate {rate} Hz → 16 kHz)"
    print(f"  🎤 recording {seconds}s — speak English now...{note}")

    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, device=idx)
    sd.wait()

    audio = _resample_to_16k(audio, rate)

    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak < 0.001:
        print("  [warn] the microphone captured near-silence — check it isn't "
              "muted, and that the right device is selected (--list-devices)")

    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(wav, audio, SR)
    return wav


def transcribe(seconds: int, device=None) -> str:
    """Record and transcribe one utterance. Returns English text ('' if silent)."""
    wav = record(seconds, device)
    try:
        t0 = time.perf_counter()
        segments, _ = get_whisper().transcribe(wav, language="en")
        text = " ".join(s.text for s in segments).strip()
        print(f"  [transcribed in {time.perf_counter() - t0:.2f}s]")
        return text
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


# --- Pi HTTP client --------------------------------------------------------
def send_to_pi(english: str, speak: bool = True, timeout: int = 120) -> str:
    """POST English to the Pi; it translates (and speaks). Returns Hindi text.

    Uses /translate with speak=… rather than /speak so --no-speak works against
    a Pi that has no audio configured — /speak deliberately 503s in that case.
    """
    url = f"{PI_URL}/translate"
    try:
        t0 = time.perf_counter()
        resp = requests.post(
            url,
            json={"text": english, "speak": speak},
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
    except requests.exceptions.ConnectionError:
        print(f"[error] cannot reach the Pi at {PI_URL}", file=sys.stderr)
        print("        Is the service running? On the Pi:", file=sys.stderr)
        print("          cd ~/entm && source venv/bin/activate", file=sys.stderr)
        print("          uvicorn it2edge.serve.app:app --host 0.0.0.0 "
              "--port 8080 --workers 1", file=sys.stderr)
        return ""
    except requests.exceptions.Timeout:
        print(f"[error] the Pi did not respond within {timeout}s", file=sys.stderr)
        return ""

    if resp.status_code == 503:
        # Model still warming, or audio requested on a Pi without working TTS.
        print(f"[error] Pi returned 503: {resp.text}", file=sys.stderr)
        return ""
    if resp.status_code != 200:
        print(f"[error] Pi returned {resp.status_code}: {resp.text}", file=sys.stderr)
        return ""

    data = resp.json()
    hindi = (data.get("translations") or [""])[0]
    spoken = data.get("spoken", False)

    print("-" * 56)
    print(f"  🎤 EN : {english}")
    print(f"  🔊 HI : {hindi}")
    print(f"  [{elapsed * 1000:.0f}ms round trip"
          f"{'; playing on the Pi' if spoken else '; not spoken'}]")
    print("-" * 56)

    if speak and not spoken:
        print("[warn] the Pi did not speak this. Check TTS readiness:",
              file=sys.stderr)
        print(f"       curl -s {PI_URL}/health", file=sys.stderr)
    return hindi


def check_pi() -> bool:
    """Report the Pi's health before the loop starts, so failures are obvious."""
    try:
        resp = requests.get(f"{PI_URL}/health", timeout=10)
    except requests.exceptions.RequestException as exc:
        print(f"[error] cannot reach the Pi at {PI_URL}: {exc}", file=sys.stderr)
        return False

    if resp.status_code == 503:
        print("[warn] the Pi is still loading its model — wait a moment.")
        return False
    if resp.status_code != 200:
        print(f"[error] /health returned {resp.status_code}", file=sys.stderr)
        return False

    data = resp.json()
    tts = data.get("tts", {})
    if tts.get("available"):
        print(f"[info] Pi at {PI_URL} is ready (TTS available)")
    else:
        print(f"[warn] Pi is up but TTS is unavailable: {tts.get('detail')}")
        print("       Translation will work; the Pi just won't speak.")
    return True


def list_devices() -> None:
    import sounddevice as sd

    print("Input devices (use --device N):\n")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  {i:3d}  {dev['name']}  "
                  f"({dev['max_input_channels']} in)")
    print(f"\nCurrent default input: {sd.query_devices(kind='input')['name']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Laptop mic → Whisper STT → Pi translate + speak (HTTP)"
    )
    ap.add_argument("--seconds", type=int, default=5,
                    help="recording length per utterance (default 5)")
    ap.add_argument("--text", default=None,
                    help="skip the mic and send this text (endpoint test)")
    ap.add_argument("--once", action="store_true",
                    help="handle one utterance, then exit")
    ap.add_argument("--no-speak", action="store_true",
                    help="translate only; do not play audio on the Pi")
    ap.add_argument("--device", type=int,
                    default=(int(os.environ["MIC_DEVICE"])
                             if os.environ.get("MIC_DEVICE") else None),
                    help="input device index (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true",
                    help="list microphones and exit")
    ap.add_argument("--url", default=None,
                    help="Pi base URL (default %s)" % PI_URL)
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.url:
        globals()["PI_URL"] = args.url.rstrip("/")

    speak = not args.no_speak

    # Text mode: exercise the Pi path without touching the mic or Whisper.
    if args.text is not None:
        sys.exit(0 if send_to_pi(args.text, speak=speak) else 1)

    if not check_pi():
        sys.exit(1)

    get_whisper()  # load before the first prompt, so Enter feels instant

    print(f"\nPush-to-talk: EN → HI, spoken on the Pi. Ctrl-C to quit.")
    try:
        while True:
            input("\npress Enter, then speak... ")
            english = transcribe(args.seconds, args.device)
            if not english:
                print("  [no speech detected — try again]")
                continue
            send_to_pi(english, speak=speak)
            if args.once:
                break
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
