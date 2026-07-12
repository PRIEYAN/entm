"""STEP 1 of the voice pipeline: laptop mic -> local Whisper STT -> Pi translate.

Runs on the GPU LAPTOP. Records your English speech, transcribes it LOCALLY with
Whisper (on the laptop GPU -- no cloud), then sends the text over SSH to the Pi,
which runs the existing translate CLI and prints the Hindi translation.

This step does NOT do TTS yet -- it just proves: voice -> English text ->
(SSH) -> Hindi text printed back. Speaker/TTS is a later step.

Fully offline after a one-time model download (Whisper caches to disk).

Setup (laptop, once):
    pip install faster-whisper sounddevice soundfile
    # make SSH password-less so each utterance doesn't prompt:
    ssh-keygen -t ed25519            # if you don't have a key
    ssh-copy-id <PI_USER>@<PI_IP>

Configure PI_HOST / PI_PROJECT below (or via env), then:
    python laptop_mic.py                 # push-to-talk loop
    python laptop_mic.py --seconds 7     # longer recording window
    python laptop_mic.py --text "Hello"  # skip the mic, just send text (test SSH)
"""

import argparse
import os
import subprocess
import sys
import tempfile

# --- configure these for your setup (env overrides win) ---
PI_HOST = os.environ.get("PI_HOST", "root@192.168.1.50")   # <PI_USER>@<PI_IP>
PI_PROJECT = os.environ.get("PI_PROJECT", "~/entm")         # repo path on the Pi
PI_VENV_PY = os.environ.get("PI_VENV_PY", "venv/bin/python")  # python on the Pi
TGT_LANG = os.environ.get("TGT_LANG", "hin_Deva")
# "speak" -> Pi translates AND speaks aloud (full pipeline).
# "translate" -> Pi only prints the Hindi text (step-1 behavior, no audio).
PI_MODE = os.environ.get("PI_MODE", "speak")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")     # tiny/base/small/medium
SR = 16000


def transcribe(seconds: int) -> str:
    """Record `seconds` of mic audio and transcribe it locally with Whisper."""
    import sounddevice as sd
    import soundfile as sf

    model = _get_whisper()

    print(f"[recording {seconds}s -- speak English now]")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1)
    sd.wait()
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(wav, audio, SR)

    segments, _ = model.transcribe(wav, language="en")
    text = " ".join(s.text for s in segments).strip()
    os.unlink(wav)
    return text


_WHISPER = None


def _add_pip_cuda_libs():
    """Add the pip nvidia-*-cu12 lib dirs to LD_LIBRARY_PATH so ctranslate2 finds
    libcublas.so.12 / libcudnn. No-op if the packages aren't installed.

    `nvidia` is a namespace package (no __init__.py), so nvidia.__file__ is None.
    Use its __path__ entries instead to locate cublas/lib and cudnn/lib.
    """
    try:
        import nvidia
    except ImportError:
        print("[warn] nvidia CUDA pip libs not found. For GPU Whisper install:\n"
              "       pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
        return

    libdirs = []
    for base in list(getattr(nvidia, "__path__", [])):
        for sub in ("cublas", "cudnn"):
            d = os.path.join(base, sub, "lib")
            if os.path.isdir(d):
                libdirs.append(d)
    if not libdirs:
        print("[warn] nvidia namespace found but no cublas/cudnn lib dirs; "
              "install: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(libdirs + ([cur] if cur else []))
    print(f"[info] added CUDA libs to LD_LIBRARY_PATH: {libdirs}")


def _get_whisper():
    """Load Whisper once. Honors WHISPER_DEVICE (cpu/cuda). Default cpu -- it
    always works; cuda needs CUDA libs (libcublas etc.) installed on the laptop.
    Verifies the device with a tiny encode so a broken CUDA falls back to CPU
    (the cuda error otherwise only surfaces mid-transcribe)."""
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER

    want = os.environ.get("WHISPER_DEVICE", "cpu").lower()
    if want == "cuda":
        _add_pip_cuda_libs()  # so ctranslate2 finds libcublas/libcudnn from pip

    from faster_whisper import WhisperModel

    if want == "cuda":
        try:
            import numpy as np
            m = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
            # Force actual CUDA use now; if libcublas is missing this raises here.
            m.encode(np.zeros((80, 3000), dtype=np.float32), to_cpu=False)
            print("[info] Whisper on CUDA (GPU)")
            _WHISPER = m
            return _WHISPER
        except Exception as exc:
            print(f"[warn] CUDA Whisper unavailable ({type(exc).__name__}); "
                  "using CPU. (Set WHISPER_DEVICE=cpu to silence.)")

    _WHISPER = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("[info] Whisper on CPU")
    return _WHISPER


def send_to_pi(english: str) -> int:
    """SSH the English text to the Pi. The Pi translates + speaks it aloud, and
    returns the Hindi text, which we print on the laptop next to the English."""
    safe = english.replace('"', '\\"')
    module = "it2edge.serve.speak" if PI_MODE == "speak" else "it2edge.serve.translate_ct2"
    remote = f'cd {PI_PROJECT} && {PI_VENV_PY} -m {module} --tgt {TGT_LANG} "{safe}"'

    # Capture the Pi's output so we can show a clean transcript on the laptop
    # (the Pi prints a "-> (hin_Deva): ..." line; other lines are noise/logs).
    proc = subprocess.run(["ssh", PI_HOST, remote], capture_output=True, text=True)

    hindi = ""
    for line in proc.stdout.splitlines():
        if line.startswith("-> (") and "):" in line:
            hindi = line.split("):", 1)[1].strip()

    print("-" * 48)
    print(f"  🎤 English : {english}")
    print(f"  🔊 {TGT_LANG} : {hindi if hindi else '[see Pi output below]'}")
    print("-" * 48)
    if not hindi and proc.stdout.strip():
        # Fallback: show raw Pi output if we couldn't parse the Hindi line.
        print(proc.stdout.strip())
    if proc.returncode != 0 and proc.stderr.strip():
        # Real errors (not the harmless "None of PyTorch..." note) go to stderr.
        errs = [l for l in proc.stderr.splitlines()
                if "None of PyTorch" not in l and l.strip()]
        if errs:
            print("[pi stderr]\n" + "\n".join(errs), file=sys.stderr)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Laptop mic -> STT -> Pi translate+speak")
    ap.add_argument("--seconds", type=int, default=5, help="recording length")
    ap.add_argument("--text", default=None,
                    help="skip the mic and send this text (SSH/translate test)")
    ap.add_argument("--once", action="store_true", help="one utterance, then exit")
    args = ap.parse_args()

    # Mic-less mode: just test the SSH -> translate path with typed text.
    if args.text is not None:
        sys.exit(send_to_pi(args.text))

    print(f"Push-to-talk  EN -> {TGT_LANG}  (spoken on {PI_HOST}). Ctrl-C to quit.")
    try:
        while True:
            input("\npress Enter, then speak... ")
            print("  [transcribing locally...]")
            text = transcribe(args.seconds)
            if not text:
                print("  [no speech detected -- try again]")
                continue
            send_to_pi(text)
            if args.once:
                break
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
