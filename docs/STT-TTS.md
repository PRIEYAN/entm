# Voice pipeline — laptop mic → STT → Pi translate → Pi TTS → speaker

End-to-end spoken translation. You speak English into the **GPU laptop**; it
transcribes to text (STT), sends the text to the **Raspberry Pi 3 B+** over the
network; the Pi translates English→Hindi (the MarianMT int8 model already
running there) and speaks the Hindi aloud through a speaker wired to the Pi (TTS).

```
🎤 laptop mic
   │  (record while a key is held)
   ▼
[GPU laptop]  Whisper STT  ──►  English text
   │
   │  HTTP POST (or SSH) over the ethernet link
   ▼
[Raspberry Pi 3 B+]  MarianMT int8  ──►  Hindi text
   │
   ▼
[Raspberry Pi 3 B+]  Piper TTS  ──►  Hindi .wav
   │
   ▼
🔊 speaker on the Pi (3.5 mm jack / USB / Bluetooth)
```

**Why this split:** STT (Whisper) is the heavy part — it belongs on the GPU
laptop. Translation + TTS are light enough for the 1 GB Pi. Piper in particular
is purpose-built to run on Raspberry Pis.

---

## Part 1 — GPU laptop: microphone + STT (voice → English text)

The laptop captures audio from its mic and runs Whisper on the GPU (fast).

```bash
pip install faster-whisper sounddevice soundfile requests
```

`faster-whisper` is a CTranslate2 build of Whisper — the same runtime family as
the Pi model, fast on GPU. Use `small` or `base` for a good speed/accuracy trade;
`small` is plenty for clear English.

Minimal push-to-talk client (`laptop_mic.py`) — record while Enter is held-ish,
transcribe, POST the text to the Pi:

```python
import sounddevice as sd, soundfile as sf, requests, tempfile, sys
from faster_whisper import WhisperModel

PI_URL = "http://<PI_IP>:8080/translate"   # the FastAPI app on the Pi
SR = 16000
model = WhisperModel("small", device="cuda", compute_type="float16")

def record(seconds=5):
    print(f"[recording {seconds}s — speak now]")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1)
    sd.wait()
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(f, audio, SR)
    return f

while True:
    input("press Enter to speak (Ctrl-C to quit) ")
    wav = record(5)
    segs, _ = model.transcribe(wav, language="en")
    text = " ".join(s.text for s in segs).strip()
    print("EN:", text)
    r = requests.post(PI_URL, json={"text": text, "tgt_lang": "hin_Deva"}, timeout=120)
    print("HI:", r.json()["translations"][0])   # the Pi also speaks it aloud
```

> If you'd rather not run an HTTP server on the Pi, send the text over SSH
> instead: `ssh pi@<PI_IP> 'python -m it2edge.serve.speak "<TEXT>"'` — but HTTP is
> cleaner and reuses the service you already have.

---

## Part 2 — Raspberry Pi: translation is already done ✅

The Pi already runs the IndicTrans2 int8 model. Two ways it receives text:

- **HTTP (recommended):** run the FastAPI app so the laptop can POST to it:
  ```bash
  cd ~/entm && source venv/bin/activate
  uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
  ```
  Then `/translate` returns the Hindi text. We extend it below to ALSO speak.
- **CLI:** `python -m it2edge.serve.translate_ct2 "some text"` (what you tested).

---

## Part 3 — Raspberry Pi: TTS (Hindi text → speech)

### Use Piper — the right TTS for a Pi 3 B+

Piper is a fast, offline neural TTS designed for Raspberry Pi. It has Hindi
voices, is a single small binary + a small model file, and needs **no GPU and no
PyTorch** — same philosophy as the translation runtime.

**Install Piper on the Pi:**

```bash
# 1. Piper binary (aarch64). Grab the latest arm64 release asset:
cd ~
wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz
tar -xf piper_linux_aarch64.tar.gz     # -> ./piper/piper

# 2. A Hindi voice model (two files: .onnx + .onnx.json). Example voice:
mkdir -p ~/piper_voices && cd ~/piper_voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json
```

> Voice availability changes; browse https://huggingface.co/rhasspy/piper-voices
> under `hi/hi_IN/` and pick a `low` or `medium` quality voice. **On a Pi 3 B+,
> prefer `low` or `medium` — not `high`** (high is slower and larger).

**Test it (text → speech out the speaker):**

```bash
echo "नमस्ते, आप कैसे हैं?" | ~/piper/piper \
  --model ~/piper_voices/hi_IN-pratham-medium.onnx \
  --output-raw | aplay -r 22050 -f S16_LE -t raw -
```

`--output-raw | aplay` streams straight to the speaker with no temp file (lower
latency, less SD wear). Or write a WAV: `... --output_file out.wav && aplay out.wav`.

### Timing & RAM on a Pi 3 B+ (1 GB) — ESTIMATES, measure yours

| Voice quality | Model size | RAM while speaking | Time for a ~10-word sentence |
|---|---|---|---|
| `low`    | ~15–20 MB | ~80–150 MB | **~0.5–1.5 s** |
| `medium` | ~40–60 MB | ~150–250 MB | **~1.5–3 s** |
| `high`   | ~60–110 MB | ~250–400 MB | ~4–8 s (avoid on Pi 3) |

Notes:
- Piper streams audio as it synthesizes, so **playback often starts before the
  whole sentence is done** — perceived latency is lower than the totals above.
- These are **on top of** the translation cost you already measured (~20 s
  greedy). TTS is not your bottleneck — translation is.
- **Total RAM budget check (1 GB Pi):** DietPi ~80–150 MB + translation runtime
  ~250–350 MB + Piper (medium) ~150–250 MB ≈ **500–750 MB**. Fits, but keep the
  translation and TTS processes lean; don't also run a desktop.
- If RAM is tight, run TTS as a **separate short-lived process** (spawn Piper per
  utterance, as the CLI does) so its memory is released between sentences, rather
  than importing a heavy TTS library into the long-running server.

---

## Part 4 — Connecting a speaker to the Raspberry Pi 3 B+

The Pi 3 B+ has a **3.5 mm audio+composite jack** on board. Three options:

### Option A — 3.5 mm jack (simplest)
Plug powered speakers / earphones into the Pi's 3.5 mm jack.

```bash
# Force audio out the 3.5 mm jack (not HDMI):
sudo raspi-config          # System Options → Audio → Headphones     (Raspberry Pi OS)
# On DietPi: use `dietpi-config` → Audio Options, or set the ALSA default card.
amixer set PCM 90%         # volume
speaker-test -t wav -c 2   # hear a test tone; Ctrl-C to stop
```

> The Pi 3 B+ headphone-jack audio is driven by a simple PWM DAC — fine for
> speech, modest quality. For better audio use USB or Bluetooth below.

### Option B — USB speaker / USB sound card (best quality, recommended)
Plug in a USB speaker or a cheap USB audio dongle + speakers. It shows up as a
second ALSA card, usually higher quality than the built-in jack.

```bash
aplay -l                   # list cards; note the USB card number, e.g. card 1
# make it the default:
cat > ~/.asoundrc <<'EOF'
defaults.pcm.card 1
defaults.ctl.card 1
EOF
aplay -l && speaker-test -t wav -c 2
```

### Option C — Bluetooth speaker (no wires, more setup/RAM)
Workable but adds `bluealsa`/`pulseaudio` overhead — heavier on a 1 GB Pi and
occasionally flaky. Only if you specifically need wireless.

```bash
sudo apt install -y bluetooth bluez bluealsa
bluetoothctl                # scan on → pair <MAC> → trust <MAC> → connect <MAC>
```

**Recommendation for a Pi 3 B+:** USB speaker (Option B) for quality, or the
3.5 mm jack (Option A) for zero fuss. Skip Bluetooth unless wireless is required.

---

## Part 5 — Wire TTS into the service (so the Pi speaks automatically)

Add a tiny helper the server calls after translating, so one HTTP request from
the laptop produces spoken Hindi on the Pi. Sketch (`it2edge/serve/speak.py`):

```python
import subprocess

PIPER = "/home/<user>/piper/piper"
VOICE = "/home/<user>/piper_voices/hi_IN-pratham-medium.onnx"

def speak(text: str):
    """Synthesize `text` with Piper and play it out the default ALSA device."""
    piper = subprocess.Popen(
        [PIPER, "--model", VOICE, "--output-raw"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    aplay = subprocess.Popen(
        ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
        stdin=piper.stdout,
    )
    piper.stdin.write(text.encode("utf-8"))
    piper.stdin.close()
    aplay.wait()
```

Then, in the `/translate` handler (`it2edge/serve/app.py`), after producing the
Hindi string, call `speak(hindi_text)` (ideally in a background thread so the
HTTP response returns immediately while audio plays). Match `aplay`'s
`-r 22050` to the voice's sample rate in its `.onnx.json` (`audio.sample_rate`).

---

## End-to-end latency budget (Pi 3 B+, ESTIMATE)

| Stage | Where | Time |
|---|---|---|
| Record + Whisper STT | GPU laptop | ~0.5–2 s (GPU is fast) |
| Network (text, tiny) | ethernet | negligible |
| **Translate (int8, greedy)** | **Pi** | **~20 s (your measured bottleneck)** |
| Piper TTS (medium) | Pi | ~1.5–3 s (streams, starts sooner) |
| **Total** | | **~22–25 s**, dominated by translation |

**To speed up the whole thing, optimize translation, not TTS:** try `--beams 1`
(already default), shorter `max_decoding_length`, or the structured-pruning step
(plan A7). TTS and STT are already cheap in this design.

---

## Quick start checklist

- [ ] Laptop: `pip install faster-whisper sounddevice soundfile requests`
- [ ] Pi: Piper binary + a Hindi `low`/`medium` voice downloaded
- [ ] Pi: speaker connected (USB or 3.5 mm) and `speaker-test` heard
- [ ] Pi: `uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080` running
- [ ] Pi: `speak()` wired into `/translate` (optional; or speak via CLI)
- [ ] Laptop: run `laptop_mic.py`, hold-to-talk, hear Hindi from the Pi

---

# ⭐ FINAL DEPLOYMENT WORKFLOW (SSH, fully offline)

This is THE workflow to use. It runs **100% offline** — no internet, no cloud,
no external API at any step. The laptop and Pi talk over the **direct ethernet +
SSH** link you already have.

```
┌──────────────── GPU LAPTOP ────────────────┐        ┌──────────── RASPBERRY PI 3 B+ ────────────┐
│  🎤 mic → your voice                         │        │  receives English text (over SSH)          │
│      ↓  (record while you speak)             │        │      ↓                                      │
│  Whisper STT  (LOCAL, on the laptop GPU)     │        │  IndicTrans2 int8  (YOUR model, on the Pi) │
│      ↓                                       │        │      ↓                                      │
│  English text  ───────── SSH ───────────────┼───────▶│  Hindi text                                │
│                                              │        │      ↓                                      │
│                                              │        │  Piper TTS  (LOCAL, on the Pi)             │
│                                              │        │      ↓                                      │
│                                              │        │  🔊 speaker on the Pi                       │
└──────────────────────────────────────────────┘        └────────────────────────────────────────────┘
```

**Offline guarantee — every stage is local, nothing leaves your two machines:**

| Stage | Runs on | Internet needed? |
|---|---|---|
| Voice capture + Whisper STT | GPU laptop | ❌ (one-time model download, then offline forever) |
| Send English text → Pi | SSH over the ethernet cable | ❌ (direct link, no internet) |
| Translate EN → HI | Pi (your int8 model) | ❌ already offline |
| Piper TTS (HI → audio) | Pi | ❌ (voice model is a local file) |
| Play audio | Pi's speaker | ❌ |

> **One-time setup needs internet** (downloading the Whisper model on the laptop
> and the Piper voice on the Pi). After that, unplug from the internet — the whole
> pipeline runs on the ethernet-only link between laptop and Pi.

---

## Step 1 — Pi side: one command that translates + speaks (`it2edge/serve/speak.py`)

This is what the laptop invokes over SSH. It takes English text, translates with
your model, synthesizes Hindi with Piper, and plays it on the Pi's speaker — all
offline, all on the Pi.

```python
"""Translate English -> Hindi and speak it on the Pi's speaker (offline).

Invoked locally or over SSH from the laptop:
    python -m it2edge.serve.speak "Hello, how are you?"
    ssh pi@<PI_IP> 'cd ~/entm && venv/bin/python -m it2edge.serve.speak "Hello"'

Uses the already-working CT2 int8 translator + Piper TTS. No network calls.
"""
import os
import subprocess
import sys

from it2edge.serve.translate_ct2 import load, translate

# Adjust these to where you installed Piper + the voice on the Pi.
PIPER_BIN = os.environ.get("PIPER_BIN", os.path.expanduser("~/piper/piper"))
PIPER_VOICE = os.environ.get(
    "PIPER_VOICE", os.path.expanduser("~/piper_voices/hi_IN-pratham-medium.onnx")
)
# Match this to "audio":{"sample_rate":...} in the voice's .onnx.json.
SAMPLE_RATE = os.environ.get("PIPER_SR", "22050")


def speak(text: str):
    """Pipe `text` through Piper and play the audio out the default ALSA device."""
    piper = subprocess.Popen(
        [PIPER_BIN, "--model", PIPER_VOICE, "--output-raw"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    aplay = subprocess.Popen(
        ["aplay", "-r", SAMPLE_RATE, "-f", "S16_LE", "-t", "raw", "-"],
        stdin=piper.stdout,
    )
    piper.stdin.write(text.encode("utf-8"))
    piper.stdin.close()
    aplay.wait()


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m it2edge.serve.speak "English text"')
    english = " ".join(sys.argv[1:])

    # Load the CT2 model + tokenizer + processor (same as the CLI).
    tokenizer, translator, processor = load(
        os.environ.get("CT2_MODEL_DIR", None) or _default_model_dir(),
        os.environ.get("TOKENIZER_DIR", None) or _default_tok_dir(),
    )
    hindi = translate([english], "hin_Deva", tokenizer, translator, processor)[0]
    print(f"EN: {english}")
    print(f"HI: {hindi}")
    speak(hindi)


def _default_model_dir():
    from it2edge.paths import CT2_DIR
    return str(CT2_DIR)


def _default_tok_dir():
    from it2edge.paths import HF_SNAPSHOT
    return str(HF_SNAPSHOT)


if __name__ == "__main__":
    main()
```

Test it directly on the Pi first (no laptop involved):

```bash
cd ~/entm && source venv/bin/activate
python -m it2edge.serve.speak "Hello, how are you?"
# -> prints Hindi AND plays it out the Pi's speaker
```

> Note: this loads the model per call (~a few seconds cold). For a single
> mic-press-per-sentence demo that's fine. If you want the model kept warm, use
> the FastAPI service (Part 2/5 above) with `speak()` wired into `/translate`
> instead — but for the SSH workflow, per-call loading keeps it dead simple.

## Step 2 — Laptop side: mic → STT → SSH the text to the Pi (`laptop_mic.py`)

Runs on the GPU laptop. Records your voice, transcribes locally with Whisper,
and SSHes the English text to the Pi (which then translates + speaks it).

```python
"""Laptop mic -> local Whisper STT -> SSH the text to the Pi (fully offline).

The Pi translates + speaks via it2edge.serve.speak. No cloud, no API.
"""
import subprocess
import tempfile

import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

PI_HOST = "pi@<PI_IP>"                 # your Pi's SSH target (ethernet IP)
PI_PROJECT = "~/entm"                  # repo path on the Pi
SR = 16000

# LOCAL, on the laptop GPU. Model caches after first download, then works offline.
model = WhisperModel("small", device="cuda", compute_type="float16")


def record(seconds=5):
    print(f"[recording {seconds}s — speak now]")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1)
    sd.wait()
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(path, audio, SR)
    return path


def send_to_pi(text: str):
    # Escape double quotes so the remote shell gets the text intact.
    safe = text.replace('"', '\\"')
    remote = f'cd {PI_PROJECT} && venv/bin/python -m it2edge.serve.speak "{safe}"'
    subprocess.run(["ssh", PI_HOST, remote], check=True)


def main():
    print("Push-to-talk English -> Hindi (spoken on the Pi). Ctrl-C to quit.")
    while True:
        input("press Enter to speak ")
        wav = record(5)
        segments, _ = model.transcribe(wav, language="en")
        text = " ".join(s.text for s in segments).strip()
        if not text:
            print("[no speech detected]")
            continue
        print("EN:", text)
        send_to_pi(text)     # Pi translates + speaks it aloud


if __name__ == "__main__":
    main()
```

Run it on the laptop:

```bash
pip install faster-whisper sounddevice soundfile     # one time
python laptop_mic.py
# press Enter, speak English, hear Hindi from the Pi's speaker
```

## Step 3 — Make SSH password-less (so it's seamless)

So each utterance doesn't prompt for a password:

```bash
# on the laptop, once:
ssh-keygen -t ed25519          # if you don't already have a key
ssh-copy-id pi@<PI_IP>         # installs your key on the Pi
ssh pi@<PI_IP> 'echo ok'       # should print ok with no password
```

## Offline operation — final checklist

- [ ] **One-time, online:** download the Whisper model on the laptop
      (`WhisperModel("small", ...)` once) and the Piper voice on the Pi.
- [ ] **One-time, online:** `git clone` + `pip install` on both machines.
- [ ] After that: **disconnect the internet.** Only the laptop↔Pi ethernet link
      is needed. Nothing in this pipeline calls out to the network.
- [ ] Verify offline: unplug internet, run `laptop_mic.py`, confirm you still hear
      Hindi from the Pi.

**Result:** speak English into the laptop → hear the Hindi translation from the
Raspberry Pi's speaker, entirely offline, with your fine-tuned model doing the
translation on the Pi itself.
