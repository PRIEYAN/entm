# Runtime on the Raspberry Pi — what's done, and how to run the model there

**Goal (eventual):** real-time voice→voice English→Hindi translation.
**Goal (this document):** get **text→text** translation running on the Raspberry Pi.
Voice comes later — see [§8](#8-where-this-fits-in-the-voice-to-voice-goal).

---

## 1. What has been done already

All of this ran on the **GPU laptop** (RTX 3050, 4 GB VRAM) and is finished:

| Stage | Result | Where |
|---|---|---|
| Base model cloned | `AI4INDIANS/better-opus-mt-en-hi` (OPUS-MT / Marian), rev `b513145b43f1` | `model_cache_compact/` |
| **Parameters** | **77.0 M** (77,026,926) — d_model 512, 6+6 layers, 8 heads, vocab 64,110 | — |
| Fine-tuned | 3 epochs on ~129k en→hi pairs, ~4 h 28 min | `model_cache_compact_ft/` |
| Quality | **chrF 35.61 / BLEU 12.22** (validation) | [../final.md](../final.md) |
| **Converted to runtime** | **CTranslate2 int8** — fp32 308 MB → **79 MB** (≈3.9× smaller) | `model_cache_compact_ct2/` |

**The runtime conversion is already done.** This is worth being clear about, because it's
the step you asked about: you do **not** need to run CTranslate2 conversion again, and you
do **not** need ONNX. The deployable artifact exists:

```
model_cache_compact_ct2/       80 MB total
├── model.bin                  79 MB   int8 weights
├── config.json                        CT2 runtime config
├── shared_vocabulary.json
├── source.spm / target.spm            SentencePiece models
├── vocab.json
├── tokenizer_config.json
├── special_tokens_map.json
└── generation_config.json
```

`sha256(model.bin) = 995c3f293720a86cb07281a96710367c404dddb9512bb7b141a7d23cb578106a`

It is **self-contained** — model, tokenizer, and vocab travel together. Nothing is
downloaded from HuggingFace on the Pi. That matters: the Pi may have no internet, and
`ab_report.json` shows what happens when HF is unreachable (`OSError: couldn't connect`).

### Why CTranslate2 and not ONNX

You asked about both. CTranslate2 is the right call here and is already done:

| | CTranslate2 (**in use**) | ONNX Runtime |
|---|---|---|
| Seq2seq latency on ARM | 🟢 lowest — C++ decode loop + KV cache, no per-token Python | 🟡 `generate()` runs in Python, one `session.run` per token |
| Memory | 🟢 int8, 79 MB weights | 🟡 higher — float activations, up to 3 graphs |
| Marian support | 🟢 native, one command | 🟡 export works but is fiddlier |
| int8 | one flag, no calibration | `quantize_dynamic`, quality needs re-validation |

Don't switch to ONNX unless something forces it. The conversion command, for reference
only (**already run — do not re-run**):

```bash
rm -rf model_cache_compact_ct2
python -m it2edge.convert.convert_ct2
```

---

## 2. Your target board — read this first

You have a **Raspberry Pi B+, 1 GB RAM, dual-core**. Two things follow.

> ⚠️ **[STT-TTS.md](STT-TTS.md) assumes a Pi 3 B+.** Its thread and memory advice is
> sized for 4 cores. On a dual-core 1 GB board, use the numbers in this document instead.

**Check the architecture before anything else** — this is the one blocker that wastes
hours if missed:

```bash
uname -m        # MUST print aarch64
nproc           # core count
free -m         # total RAM
python3 --version
```

If `uname -m` prints `armv7l` or `armv6l`, **stop**. There is no prebuilt `ctranslate2`
wheel for 32-bit ARM; pip will attempt a source build that takes hours and usually fails.
Reflash 64-bit Raspberry Pi OS (or DietPi arm64) first. Everything below assumes `aarch64`.

**RAM is the real constraint at 1 GB.** The int8 weights are only 79 MB, but the Python
interpreter, ctranslate2, and transformers add more on top. Budget carefully:

- Run **one** process. Never more than one uvicorn worker — each worker loads its own
  copy of the model.
- Keep the model **loaded once and warm**. Cold start on this class of board is seconds
  to tens of seconds; paying it per sentence makes the system unusable.
- If memory gets tight, add zram/swap (§7).

> On x86 the laptop showed ~580 MB resident, but **~390 MB of that is Intel MKL/oneDNN
> arenas bundled in the x86 wheel**. The aarch64 wheel has no MKL — it uses the lighter
> Ruy backend — so the Pi figure should be far lower. Measure it (§6); don't assume either
> number.

---

## 3. Get the model onto the Pi

### Option A — one command from the laptop (preferred)

```bash
# on the LAPTOP, from the project root
bash deploy/push_to_pi.sh root@<pi-ip> --setup
```

This refuses to proceed unless the Pi is `aarch64`, copies model + code, **verifies
`model.bin` by sha256 on both ends**, creates the venv, installs deps, and runs a smoke
translation. Needs passwordless SSH first: `ssh-copy-id root@<pi-ip>`.

> Run it on the **laptop**, not the Pi. Its job is to push files *to* the Pi; on the Pi it
> will just report "No such file or directory".

### Option B — no SSH password? Pull over HTTP

On the **laptop**:

```bash
cd /home/prieyan/weeb/entm
python3 -m http.server 8000
```

On the **Pi**:

```bash
mkdir -p ~/entm && cd ~/entm
for d in model_cache_compact_ct2 it2edge deploy requirements; do
  wget -r -np -nH -R "index.html*" "http://<laptop-ip>:8000/$d/"
done
wget http://<laptop-ip>:8000/pyproject.toml
```

### Verify the transfer — do not skip this

```bash
sha256sum ~/entm/model_cache_compact_ct2/model.bin
```

Must equal `995c3f29...78106a` (§1). A truncated `model.bin` fails later in confusing
ways; checking here saves real debugging time.

---

## 4. Python environment on the Pi

```bash
cd ~/entm
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements/pi.txt
```

**Watch the log. If you see `Building wheel for ctranslate2` or
`Building wheel for sentencepiece`, something is wrong** — abort rather than waiting. Both
should install as prebuilt `manylinux…aarch64` wheels. A source build means either a
32-bit OS (§2) or a version with no wheel for your Python.

### Two failures already hit on a real board (DietPi, Python 3.13)

1. **`sentencepiece==0.2.0` has no aarch64 wheel for Python 3.13.** pip fell back to a
   source build needing `cmake` and `pkg-config` — absent on stock DietPi — and failed
   with exit 127. Fixed: `requirements/pi.txt` now uses `sentencepiece==0.2.1`, which has
   the wheel.
2. **That failure took everything with it.** pip installs atomically, so `fastapi`,
   `uvicorn` and `psutil` — listed after sentencepiece — never installed, producing a
   baffling `uvicorn: command not found` on a board where the model already translated
   fine. If you hit this, install the stragglers directly:

```bash
pip install "sentencepiece==0.2.1" fastapi uvicorn[standard] psutil
```

`requirements/pi.txt` now exact-pins only the three packages that affect model correctness
(`ctranslate2`, `transformers`, `sentencepiece`) and floors the rest, so a single missing
wheel can't abort the whole install again.

### Harmless messages

```
None of PyTorch, TensorFlow >= 2.0, or Flax have been found.
```

**Expected and correct.** The whole point of the int8 CT2 build is running with **no
torch** — it wouldn't fit comfortably on a 1 GB board anyway. transformers is present only
for the tokenizer.

---

## 5. Run it

### 5.1 One-shot CLI — the smoke test

```bash
cd ~/entm && source venv/bin/activate
python -m it2edge.serve.translate_ct2 "Hello, how are you?"
```

Expected — this must match the laptop **exactly**:

```
EN: Hello, how are you?
-> हेलो, आप कैसे हैं?
```

Identical model + identical tokenizer + greedy decode is deterministic across
architectures. **If the Hindi differs, the artifact moved wrong** — recheck the §3
checksum before debugging anything else.

This path is for verification only. It reloads the model every run, so it's slow — which
is exactly why the service below exists.

### 5.2 The HTTP service — how you actually use it

```bash
cd ~/entm && source venv/bin/activate
CT2_INTER_THREADS=1 CT2_INTRA_THREADS=2 OMP_NUM_THREADS=2 \
  uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
```

If `uvicorn` isn't found (see §4), the module form always works inside a venv:

```bash
python -m uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
```

Thread settings for a **dual-core** board: `inter × intra ≤ cores`, so `1 × 2` puts both
cores on one sentence — lowest single-sentence latency, which is what a voice pipeline
needs. (`2 × 1` favours concurrent throughput instead; not what you want here.)

Test from the **laptop**, which is the real end-to-end check:

```bash
curl -s http://<pi-ip>:8080/health
curl -s -X POST http://<pi-ip>:8080/translate \
  -H 'content-type: application/json' -d '{"text":"Hello, how are you?"}'
curl -s -X POST http://<pi-ip>:8080/translate \
  -H 'content-type: application/json' -d '{"texts":["Hello","Good morning"]}'
```

Verified responses:

| Input | Output |
|---|---|
| `Hello, how are you?` | हेलो, आप कैसे हैं? |
| `Good morning` | सुप्रभात |
| `The weather is beautiful this morning.` | आज सुबह मौसम सुंदर है। |

> `/health` returns **503**, not an error, until the model finishes loading. On this board
> that can take tens of seconds. Note `-X POST` — a plain `curl` sends GET and gets **405**.

### 5.3 Auto-start on boot

For a 1 GB board use the **bare-metal** unit; skip containers (Podman's overhead is a
meaningful slice of 1 GB).

```bash
mkdir -p ~/.config/systemd/user
cp deploy/it2-baremetal.service ~/.config/systemd/user/it2.service
loginctl enable-linger $USER
systemctl --user daemon-reload
systemctl --user enable --now it2
journalctl --user -u it2 -f
```

Then **reboot and confirm it comes back on its own**. An auto-start never tested through a
real reboot isn't an auto-start.

---

## 6. Measure it — the numbers nobody has yet

[../final.md](../final.md) §4 only *estimates* Pi latency (~0.3–0.6 s/sentence for a
B+-class board). Replace the estimate with a measurement:

```bash
python -m it2edge.evaluate.benchmark_models latency \
  --ct2_dir model_cache_compact_ct2 --data_dir en-indic-exp \
  --inter 1 --intra 2 --num 40 --warmup 5 \
  --out pi_latency_aarch64.json
```

It reports median/p95 latency, load time, a baseline/loaded/peak RSS split, and board
temperature, and **warns automatically** if `vcgencmd get_throttled` isn't `0x0` (a
throttled run makes the numbers pessimistic).

Laptop reference (i5-12450H, 2 threads, greedy): **median 98.8 ms, p95 338.9 ms**, model
load 0.34 s. Expect the Pi to be several times slower per sentence — and the **cold start**
much worse, which is the entire argument for the warm service in §5.2.

---

## 7. If it's slow or runs out of memory

In order of payoff:

1. **Keep the model warm.** Non-negotiable. Use the service (§5.2), never one-shot CLI.
2. **Greedy decode** — already the default (`beam_size=1`). Don't raise it on this board.
3. **Cap `max_decoding_length`.** The API default is 256, but training used a max target
   length of 128 — sending 128 bounds worst-case latency at no real quality cost, since the
   model never learned to emit longer.
4. **Send sentences, not paragraphs.** Latency scales with tokens *generated*. Split first;
   batch several sentences in one `texts` call rather than N calls.
5. **zram / swap** if RSS approaches the 1 GB limit:
   ```bash
   sudo apt install -y zram-tools     # or dietpi-config → Performance
   ```
   It trades some latency for not being OOM-killed.
6. **Cooling.** Sustained load throttles these boards; check `vcgencmd measure_temp`.

Not worth attempting: quantizing below int8, or unstructured pruning — neither gives a
dense CPU speedup on this stack (see [plan.md](plan.md) §0).

---

## 7b. Text→speech (TTS) — implemented

Piper TTS is now wired into the service. Text→speech is **step 2 of 3** toward
voice→voice; STT is the only remaining piece.

### What changed

- [../it2edge/serve/speak.py](../it2edge/serve/speak.py) — already had the Piper CLI. Added
  `tts_available()` (reports readiness without raising), a lock so two utterances can't
  garble the single sound card, and proper subprocess cleanup so a failed player can't
  leave a Piper zombie holding RAM on a 1 GB board.
- [../it2edge/serve/app.py](../it2edge/serve/app.py) — new `POST /speak`, a `speak` flag on
  `/translate`, and TTS status in `/health`.

### Design rules that matter on this board

1. **Synthesis runs in a background thread.** The HTTP response returns as soon as the
   Hindi text exists — measured **46 ms** — while audio plays afterwards. The caller is
   never blocked by playback.
2. **TTS failure never breaks translation.** No Piper, no speaker, no voice file → text
   still returns, `spoken: false`, reason logged. This is why the dev laptop still works.
3. **Piper is a short-lived subprocess**, one per utterance, so its RAM is released between
   sentences rather than held by the long-running server.
4. **One utterance at a time**, enforced by a lock — concurrent Piper processes on one
   sound card produce garbled audio.

### Endpoints

| Endpoint | Behaviour |
|---|---|
| `POST /translate` | Text only. Add `"speak": true` to also play audio. |
| `POST /speak` | Translate **and** speak. Returns **503** if TTS is unavailable — audio is the point here, so failing loudly is correct. |
| `GET /health` | Now reports `tts.available` and `speak_by_default`. |

Set `SPEAK_BY_DEFAULT=1` to make every `/translate` speak without the flag.

### Prerequisites on the Pi

Piper and a Hindi voice (you already have `piper/` and `piper_voices/`):

```bash
ls ~/piper/piper                                          # binary
ls ~/piper_voices/hi_IN-pratham-medium.onnx*              # voice + .json
sudo apt install -y alsa-utils                            # aplay
```

Override paths via `PIPER_BIN`, `PIPER_VOICE`, `AUDIO_OUT` (`alsa`|`pulse`),
`ALSA_DEVICE` (default `hw:0,0`).

### How to test — in order

**1. Speaker works at all** (before involving any code):

```bash
speaker-test -t wav -c 2       # Ctrl-C to stop
```

Nothing audible → fix audio routing first ([STT-TTS.md](STT-TTS.md) Part 4); no amount of
Python will help. On DietPi use `dietpi-config` → Audio Options.

**2. Piper alone**, bypassing the model:

```bash
echo "नमस्ते, आप कैसे हैं?" | ~/piper/piper \
  --model ~/piper_voices/hi_IN-pratham-medium.onnx \
  --output-raw | aplay -r 22050 -f S16_LE -t raw -
```

Match `-r` to `audio.sample_rate` in the voice's `.onnx.json`; a mismatch plays at the
wrong pitch/speed.

**3. Translate + speak from the CLI:**

```bash
cd ~/entm && source venv/bin/activate
python -m it2edge.serve.speak "Hello, how are you?"
python -m it2edge.serve.speak --no-audio "Hello"      # text only
```

**4. Over HTTP.** Restart the service to pick up the new endpoints:

```bash
CT2_INTER_THREADS=1 CT2_INTRA_THREADS=2 OMP_NUM_THREADS=2 \
  uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
```

Then **from the laptop**:

```bash
# TTS readiness — expect "available": true on the Pi
curl -s http://<pi-ip>:8080/health

# translate AND speak; returns immediately, audio plays on the Pi
curl -s -X POST http://<pi-ip>:8080/speak \
  -H 'content-type: application/json' -d '{"text":"Hello, how are you?"}'

# same via the flag
curl -s -X POST http://<pi-ip>:8080/translate \
  -H 'content-type: application/json' -d '{"text":"Good morning","speak":true}'

# several sentences — spoken in order, not overlapping
curl -s -X POST http://<pi-ip>:8080/speak \
  -H 'content-type: application/json' \
  -d '{"texts":["Hello","Good morning","The weather is beautiful."]}'
```

Expect `{"tgt_lang":"hi","translations":[...],"spoken":true}` **and** Hindi audio from the
speaker. The response arriving before the audio finishes is correct — that's the
background thread.

### Verified on the laptop (with a stubbed Piper)

| Check | Result |
|---|---|
| UTF-8 Hindi reaches Piper intact | ✅ `हेलो, आप कैसे हैं?` |
| Sample rate read from voice `.onnx.json` | ✅ 22050 |
| `/speak` response time (audio in background) | ✅ **46 ms** |
| Batch spoken sequentially, not overlapping | ✅ exactly 2× bytes for 2 sentences |
| No Piper → translation still succeeds | ✅ `spoken: false`, 200 |
| `/speak` with no TTS | ✅ 503 with reason |
| Failed audio player | ✅ raises cleanly, no zombie process |
| Existing text endpoints | ✅ no regression (batch, 422, 405 all unchanged) |

> These used a stub standing in for Piper, which proves the **plumbing** — process wiring,
> encoding, threading, cleanup. It cannot prove **audio quality or real Piper timing**;
> only the Pi with a speaker attached can. Run steps 1–4 above there.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `"available": false` in `/health` | Piper/voice/`aplay` missing | The `detail` field names which |
| `/speak` → 503 | same | as above |
| `spoken: true` but silence | audio routing, not the code | `speaker-test`; check `ALSA_DEVICE`, volume |
| Wrong pitch/speed | sample-rate mismatch | match `-r` to the voice `.onnx.json` |
| `aplay: command not found` | missing ALSA tools | `sudo apt install -y alsa-utils` |
| Garbled/overlapping audio | two players on one card | the lock prevents this internally; check nothing else uses the card |
| Audio cuts off / OOM | 1 GB exhausted | use a `low` voice; see §7 |

---

## 8. Where this fits in the voice-to-voice goal

Two of the three stages now work on the Pi:

```
🎤 laptop mic → Whisper STT (GPU laptop) → English text     ← ❌ REMAINING
                     │ HTTP
                     ▼
        MarianMT int8 on the Pi  → Hindi text               ← ✅ DONE (§1–§6)
                     │
                     ▼
              Piper TTS on the Pi → 🔊 speaker              ← ✅ DONE (§7b)
```

Only **STT** is left. It runs on the **GPU laptop**, not the Pi — Whisper is the heavy
part, and the Pi is already near its limits. [STT-TTS.md](STT-TTS.md) Part 1 has the
`faster-whisper` push-to-talk client; [../laptop_mic.py](../laptop_mic.py) is a starting
point. It POSTs to `/speak`, so the Pi will translate and speak in one call.

The split is deliberate: **STT is the heavy part and stays on the GPU laptop**; translation
and TTS are light enough for the Pi. Piper is purpose-built for Raspberry Pis, and
[../it2edge/serve/speak.py](../it2edge/serve/speak.py) already wires translation → Piper:

```bash
python -m it2edge.serve.speak "Hello, how are you?"
python -m it2edge.serve.speak --no-audio "test"     # translate + print only
```

You already have `piper/` and `piper_voices/` on the Pi, so once §5 works, TTS is the
natural next step. **Finish and benchmark text→text first** — if per-sentence latency is
too high on a dual-core B+, that's much cheaper to discover now than after the audio
layers are attached. On a board this small, moving translation to the laptop and leaving
only Piper on the Pi is a legitimate fallback.

---

## 9. Quick reference

```bash
# --- laptop ---
bash deploy/push_to_pi.sh root@<pi-ip> --setup     # copy + verify + install

# --- Pi ---
uname -m                                            # must be aarch64
cd ~/entm && source venv/bin/activate
python -m it2edge.serve.translate_ct2 "Hello, how are you?"     # smoke test
CT2_INTER_THREADS=1 CT2_INTRA_THREADS=2 OMP_NUM_THREADS=2 \
  uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1

# --- laptop, testing the Pi ---
curl -s -X POST http://<pi-ip>:8080/translate \
  -H 'content-type: application/json' -d '{"text":"Hello, how are you?"}'
```

| Symptom | Cause | Fix |
|---|---|---|
| `Building wheel for ctranslate2` | 32-bit OS | Reflash 64-bit; `uname -m` must be `aarch64` |
| `cmake: not found` building sentencepiece | pinned version has no wheel for your Python | `pip install "sentencepiece==0.2.1"` |
| `uvicorn: command not found` | earlier pip failure aborted the whole install | `pip install fastapi uvicorn[standard] psutil`, or use `python -m uvicorn` |
| `/health` → 503 | model still loading | Wait; normal on a slow board |
| `POST /translate` → 405 | missing `-X POST` | Add it |
| Hindi differs from laptop | corrupt transfer | Recheck `sha256sum model.bin` |
| `No such file or directory: deploy/push_to_pi.sh` | running it on the Pi | Run it on the **laptop** |
| `OSError: couldn't connect to huggingface.co` | code trying to fetch a hub model | Use the local `model_cache_compact_ct2/` path |
| Killed / OOM | 1 GB exhausted | One worker only; add zram (§7) |
