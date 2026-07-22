# Plan — Ship the CTranslate2 int8 runtime to a Raspberry Pi

> **STATUS (2026-07-22): RUNNING ON THE PI.** ✅
> The model was transferred to the DietPi board (aarch64, Python 3.13) and
> translates correctly there:
> `EN: Hello, how are you? -> हेलो, आप कैसे हैं?` — byte-identical to the laptop.
> All bugs in §0 are fixed, including one (#10) found only by running on the real board.
> **Remaining:** start the HTTP service, benchmark it, and enable auto-start —
> see [§11.3](#113-benchmark-and-make-it-permanent).

**Goal:** take the already-built int8 CTranslate2 package
([model_cache_compact_ct2/](model_cache_compact_ct2/), 79 MB) off this laptop and run it
as a warm, long-lived translation service on a Raspberry Pi.

**Status going in:** the model side is *done*. Fine-tuning (chrF 35.61 / BLEU 12.22) and
int8 conversion both completed — see [final.md](final.md). Nothing in this plan re-trains
or re-converts anything. This is purely a *deployment* plan.

---

## 0. What already exists vs. what this plan builds

Read this before doing anything — most of the scaffolding is already in the repo, and
half of the work below is verification rather than writing new code.

| Piece | File | State |
|---|---|---|
| int8 CT2 model + tokenizer | [model_cache_compact_ct2/](model_cache_compact_ct2/) | ✅ built (79 MB `model.bin`) |
| Inference wrapper | [it2edge/serve/marian_ct2.py](it2edge/serve/marian_ct2.py) | ✅ works, no torch |
| One-shot CLI | [it2edge/serve/translate_ct2.py](it2edge/serve/translate_ct2.py) | ✅ works |
| HTTP service | [it2edge/serve/app.py](it2edge/serve/app.py) | ✅ warm model, single worker |
| arm64 image | [deploy/Containerfile](deploy/Containerfile) | ✅ written, not built (no container daemon locally) |
| Native Pi build script | [deploy/build_on_pi.sh](deploy/build_on_pi.sh) | ✅ **fixed** — engine auto-detect, core sizing, arch guard |
| Cross-build script | [deploy/build_arm64.sh](deploy/build_arm64.sh) | ✅ **fixed** — fails clearly without podman |
| systemd Quadlet unit | [deploy/it2.container](deploy/it2.container) | ✅ **fixed** — real `Image=` value |
| Bare-metal systemd unit | [deploy/it2-baremetal.service](deploy/it2-baremetal.service) | ✅ **new** — for 1 GB boards |
| One-command Pi push | [deploy/push_to_pi.sh](deploy/push_to_pi.sh) | ✅ **new** — copy + checksum + setup |
| Pi requirements | [requirements/pi.txt](requirements/pi.txt) | ✅ **fixed** — fully pinned |
| Latency benchmark | [it2edge/evaluate/benchmark_models.py](it2edge/evaluate/benchmark_models.py) | ✅ **fixed** — existed already; min/max + RSS bugs repaired |

### Bugs found and fixed

1. **`requirements/pi.txt` was unpinned** while
   [requirements/docker.txt](requirements/docker.txt) was pinned — bare metal and
   container would install different versions. Now both pinned identically.
2. **`transformers` version skew.** docker.txt pinned `4.52.4`, but [final.md](final.md)
   records the build needing `4.57.6`. Resolved to **4.57.6 in both files** and verified
   the tokenizer loads (`MarianTokenizer`, vocab 64110).
3. **`str | None` on Python 3.9.** [marian_ct2.py](it2edge/serve/marian_ct2.py) and
   `benchmark_models.py` used PEP-604 syntax that raises `TypeError` at import on 3.9,
   which some Pi OS images still ship. Added `from __future__ import annotations` to both.
4. **`<registry>` placeholder** in [deploy/it2.container](deploy/it2.container) would
   never start. Now `Image=localhost/it2-ct2:latest`, matching what `build_on_pi.sh` produces.
5. **Missing `-X POST`** in `build_on_pi.sh`'s printed curl example — as written it sent a
   GET and returned **405** (confirmed by test). Fixed.
6. **Hardcoded `--cpus 2`** in `build_on_pi.sh` regardless of board, which also silently
   contradicted a 4-thread CT2 setting. Now detects `nproc` and sets threads to match.
7. **`min_ms`/`max_ms` read from the wrong list** in the latency benchmark: `lat.sort()`
   sorted in place, so those fields were correct only by accident — and after my RSS fix
   they'd have been outright wrong. Now uses an explicit sorted copy.
8. **`peak_rss_mb` was a single end-of-run sample**, not a peak — the number that decides
   OOM on a small board. Now sampled every iteration, plus a
   baseline/loaded/model RSS split.
9. **No arch guard.** `build_on_pi.sh` and `push_to_pi.sh` now refuse anything that isn't
   `aarch64`, instead of letting pip start an hours-long source build.
10. **`sentencepiece==0.2.0` has no aarch64 wheel for Python 3.13** — found on the real
    Pi (DietPi, Python 3.13). pip fell back to a source build that needs `cmake` and
    `pkg-config`, neither present on a stock DietPi, and failed with exit 127. The pin was
    also simply wrong: the laptop that built and verified this model runs **0.2.1**. Both
    requirement files now use `sentencepiece==0.2.1`, which has the wheel.
11. **Over-pinning made that failure total, not partial.** pip installs atomically: the
    sentencepiece build error aborted the entire run, so `fastapi`, `uvicorn` and `psutil`
    — listed *after* it — were never installed, producing a confusing
    `uvicorn: command not found` on a board where the model itself already worked.
    Several of those pins were also never verified: they were inherited from the old
    `docker.txt`, and the laptop actually runs fastapi 0.139.2 / numpy 2.4.6, not the
    pinned 0.115.12 / 2.2.6. **[requirements/pi.txt](requirements/pi.txt) now exact-pins
    only the three model-critical packages** (ctranslate2, transformers, sentencepiece)
    and uses floors for the HTTP layer. [requirements/docker.txt](requirements/docker.txt)
    stays exact-pinned on purpose — it fixes the interpreter at `python:3.11-slim`, where
    all those wheels exist.

---

## 1. Decide the target board first

Everything downstream (RAM limits, thread counts, whether containers are even worth it)
depends on which Pi this is. Run on the Pi:

```bash
uname -m                                  # MUST print aarch64
cat /proc/cpuinfo | grep -m1 'Model'      # board revision
nproc                                     # core count
free -m                                   # total RAM
cat /etc/os-release                       # Debian version
```

**Hard requirement: `uname -m` must print `aarch64`.** On 32-bit (`armv7l`) there are no
prebuilt `ctranslate2` wheels — pip falls back to a source build that takes hours on a Pi
and usually fails. If you get `armv7l`, **stop and reflash 64-bit Raspberry Pi OS**; every
later step assumes 64-bit.

Then pick your lane:

| Board | Lane | Rationale |
|---|---|---|
| **Pi 4 / Pi 5, ≥2 GB** | **Container (§5)** — the intended path | Enough RAM for Podman + model; matches the existing deploy scripts |
| **Pi 3B+, 1 GB** | **Bare metal (§4)** | Works, but container overhead eats a meaningful slice of 1 GB |
| **Pi B+ / Zero, 512 MB–1 GB, ARM11** | **Bare metal, and expect pain** | [final.md](final.md) §4 already flags this as RAM-tight and 15–40 s cold start. Single old core. Treat as a "does it run at all" experiment, not a service |

> The estimates in [final.md](final.md) §4 are explicitly projections. **§6 of this plan
> replaces them with measured numbers.** Do not report the estimates as results.

---

## 2. Phase 1 — Get the artifacts onto the Pi

The CT2 package is *self-contained*: model weights, both SentencePiece models, and the
tokenizer config all live in one directory (`convert_ct2.py` copies the tokenizer files
in). Nothing needs to be downloaded from HuggingFace on the Pi.

```bash
# from the laptop, project root
scp -r model_cache_compact_ct2 pi@<pi-ip>:~/entm/
scp -r it2edge deploy requirements pyproject.toml pi@<pi-ip>:~/entm/
```

**Verify the transfer** (a truncated `model.bin` fails in a confusing way later):

```bash
# laptop
sha256sum model_cache_compact_ct2/model.bin
# Pi
sha256sum ~/entm/model_cache_compact_ct2/model.bin   # must match
du -sh ~/entm/model_cache_compact_ct2                # ~83 MB
```

Expected contents (9 files): `model.bin`, `config.json`, `generation_config.json`,
`shared_vocabulary.json`, `vocab.json`, `source.spm`, `target.spm`,
`tokenizer_config.json`, `special_tokens_map.json`.

> Note `model_cache_compact_ct2/` is in [.gitignore](.gitignore) — it is deliberately not
> in git. `scp` (or a USB stick / registry image) is the transport, not `git pull`.

---

## 3. Phase 2 — Environment on the Pi

### 3.1 System packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
python3 --version    # 3.11+ preferred; pyproject requires >=3.11
```

If Pi OS ships Python 3.9 (older Bullseye images), either upgrade the OS image or accept
a bare-metal install outside the `pyproject.toml` constraint — the *serving* code itself
uses `str | None` syntax in
[marian_ct2.py](it2edge/serve/marian_ct2.py#L20), which needs **3.10+**. On 3.9 that file
raises `TypeError` at import. Check this early; it is a silent trap.

### 3.2 Venv + deps (bare-metal lane)

```bash
cd ~/entm
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip          # needs >=20.3 for manylinux/aarch64 tags
pip install -r requirements/pi.txt
```

Watch the install log. **If anything says "Building wheel for ctranslate2" you are on the
wrong architecture or an unsupported Python** — abort, don't wait it out.

### 3.3 Pin `requirements/pi.txt` (task)

Bring it in line with the tested container pins so bare metal and container agree:

```
ctranslate2==4.8.1
transformers==4.52.4
sentencepiece==0.2.0
sacremoses==0.1.1
protobuf==5.29.5
numpy==2.2.6
fastapi==0.115.12
uvicorn[standard]==0.34.2
psutil==7.0.0
```

Then **verify the `transformers` pin actually loads this tokenizer** (this is the untested
assumption from §0.2):

```bash
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('model_cache_compact_ct2')
print(type(t).__name__, t.vocab_size)
"
```

If 4.52.4 fails on the tokenizer, bump both `pi.txt` and
[requirements/docker.txt](requirements/docker.txt) to `transformers==4.57.6` and rebuild
the image. Do not leave the two files disagreeing.

---

## 4. Phase 3 — Bare-metal run (do this FIRST, on every board)

Even if the container is the destination, run bare metal once. It isolates
model/tokenizer problems from container problems, and on 1 GB boards it may be the final
answer.

### 4.1 Smoke test

```bash
cd ~/entm && source venv/bin/activate
python -m it2edge.serve.translate_ct2 "Hello, how are you?"
```

Expected (matches the laptop result in [final.md](final.md) §3):

```
EN: Hello, how are you?
-> हेलो, आप कैसे हैं?
```

**If the Hindi output differs from the laptop's**, the artifact or tokenizer moved wrong
— re-check the §2 checksum before touching anything else. Identical model + identical
tokenizer + greedy decode is deterministic across architectures.

Time it, and expect it to be slow — this is the cold-start path, not the real latency:

```bash
time python -m it2edge.serve.translate_ct2 "Hello, how are you?"
```

### 4.2 Run the service

```bash
export OMP_NUM_THREADS=2 CT2_INTER_THREADS=1 CT2_INTRA_THREADS=2
uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
```

From another shell (or the laptop, using the Pi's IP):

```bash
curl -s localhost:8080/health
curl -s -X POST localhost:8080/translate \
  -H 'content-type: application/json' \
  -d '{"texts":["Hello","How are you?","Good morning"]}'
```

`/health` returns **503** until the model finishes loading — that is by design
([app.py](it2edge/serve/app.py#L104-L108)), not a failure. On a slow Pi it may stay 503
for tens of seconds.

### 4.3 Thread tuning

`inter_threads × intra_threads ≤ physical cores`. For lowest **single-sentence** latency
put all cores on one translation:

| Board | Setting |
|---|---|
| Pi 4/5 (4 cores) | `CT2_INTER_THREADS=1 CT2_INTRA_THREADS=4`, `OMP_NUM_THREADS=4` |
| Pi 3B+ (4 cores, 1 GB) | `1 × 4`, but watch RAM under load |
| Single-core (B+) | `1 × 1`, `OMP_NUM_THREADS=1` |

Flip to `inter=2, intra=2` **only** if you care about many concurrent requests more than
per-sentence latency. Measure both in §6 rather than guessing.

---

## 5. Phase 4 — Containerized run (Pi 4/5, ≥2 GB)

### 5.1 Native build on the Pi (recommended)

Nothing compiles — every dep in [requirements/docker.txt](requirements/docker.txt) has a
prebuilt aarch64 wheel — so building on the Pi is straightforward and avoids the
registry round-trip entirely.

```bash
sudo apt install -y podman
cd ~/entm
bash deploy/build_on_pi.sh          # builds AND runs
bash deploy/build_on_pi.sh --build  # build only
```

The script already validates that `model_cache_compact_ct2/` is present and starts the
container with `--cpus 2 --memory 1500m -p 8080:8080`.

**Raise the memory cap on a 4 GB+ Pi** and match `--cpus` to `nproc`; 1500m is sized for a
2 GB board. The `--cpus 2` cap also silently contradicts a 4-thread CT2 setting — keep
them consistent or the extra threads just contend.

### 5.2 Cross-build on the laptop (alternative)

Only if the Pi is too slow to build or you want one image for several Pis:

```bash
REGISTRY=docker.io/<youruser> deploy/build_arm64.sh
```

It registers QEMU arm64 via binfmt, builds `--platform=linux/arm64` into a manifest, and
pushes. A cross-arch image **cannot run on the laptop** — push, then pull on the Pi.
Requires a registry account and a `podman login`; this is why §5.1 is the default.

### 5.3 Expected image size

~250–300 MB of base + deps, plus the 79 MB model baked in ⇒ **~350–450 MB**. Check free
space on the Pi's SD card before building (`df -h`); Podman's build cache needs headroom
beyond the final image.

### 5.4 Auto-start on boot (Quadlet)

[deploy/it2.container](deploy/it2.container) contains a literal `<registry>` placeholder —
**it will not work as-is.** Fix it for whichever build path you used:

- Native build (§5.1): change `Image=<registry>/it2-ct2:latest` → `Image=localhost/it2-ct2:latest`
- Registry pull (§5.2): substitute your actual registry

Then install it rootless:

```bash
mkdir -p ~/.config/containers/systemd
cp deploy/it2.container ~/.config/containers/systemd/it2.container
# edit the Image= line first!
loginctl enable-linger $USER        # so it runs without an active login
systemctl --user daemon-reload
systemctl --user start it2
systemctl --user status it2
```

Also update the unit's `PodmanArgs=--cpus 2 --memory 1500m` and the
`CT2_INTER_THREADS`/`CT2_INTRA_THREADS` env lines to whatever §4.3 and §6 showed to be
best — the file currently ships the conservative 2-core defaults.

**Reboot the Pi and confirm the service comes back on its own.** An auto-start unit that
was never tested through an actual reboot is not an auto-start unit.

---

## 6. Phase 5 — Measure real latency (replaces the estimates)

This is the deliverable that [final.md](final.md) §4 explicitly leaves open. There is
**no Pi benchmark script in the repo** — write one.

### 6.1 Use the existing benchmark (no new script needed)

The repo **already had** `it2edge/evaluate/benchmark_models.py` with a `latency`
subcommand — an earlier draft of this plan wrongly said it didn't exist. It has been
fixed (bugs 7–9 in §0) rather than replaced, and now also records board, arch, temperature
and Pi throttling flags.

```bash
python -m it2edge.evaluate.benchmark_models latency \
  --ct2_dir model_cache_compact_ct2 --data_dir en-indic-exp \
  --inter 1 --intra $(nproc) --num 40 --warmup 5 \
  --out pi_latency_$(uname -m).json
```

`pi_latency_*.json` is already covered by [.gitignore](.gitignore), so reports won't be
committed by accident. It prints a **warning automatically** if `vcgencmd get_throttled`
is anything but `0x0`.

### 6.2 Also record thermals

Latency on a Pi is a function of temperature. Alongside the benchmark:

```bash
vcgencmd measure_temp
vcgencmd get_throttled     # 0x0 == never throttled
watch -n1 vcgencmd measure_temp   # during a sustained run
```

`get_throttled` returning anything but `0x0` invalidates the benchmark — the board was
throttling and the numbers are pessimistic. Add cooling and re-run.

### 6.3 Fill in the results table

Laptop column is **measured 2026-07-22** (i5-12450H, 40 dev sentences, greedy):

| Metric | Laptop, 2 threads | Laptop, 4 threads | Pi (**to measure**) |
|---|---|---|---|
| Median per-sentence | **98.8 ms** | **85.5 ms** | |
| p95 | 338.9 ms | 281.5 ms | |
| min / max | 12.2 / 458.9 ms | — | |
| Model load | 0.34 s | 0.34 s | |
| Cold CLI total | ~2.41 s | — | |
| Baseline RSS (pre-load) | 19.5 MB | — | |
| **Model-only RSS** | 556 MB (x86/MKL — see note) | — | |
| Peak RSS | 582 MB | 582 MB | |
| Temp / throttled | 43 °C / n-a | — | |

> **The 556 MB is an x86 artifact, not the model.** `import ctranslate2` alone costs
> ~390 MB on x86_64 because the wheel bundles a 60.7 MB extension with Intel MKL/oneDNN
> kernels that reserve thread arenas at import. **The aarch64 wheel has no MKL.** The int8
> weights are 79 MB. Do not size the Pi against 582 MB — measure it. This is why the
> benchmark now reports baseline/loaded/model RSS separately.

Then **update [final.md](final.md) §4** — replace the "estimated" Pi table with measured
numbers and drop the ⚠️ projection warning for the row you actually tested. Leave the
untested boards marked as estimates.

---

## 7. Phase 6 — Tuning, only if §6 says it's needed

In rough order of payoff:

1. **Greedy decode** — already the default (`DEFAULT_BEAM_SIZE=1` in
   [app.py](it2edge/serve/app.py#L45)). Note the CT2 package's
   `generation_config.json` still says `num_beams: 4`, but that file is not what drives
   decoding here — `translate_marian` passes `beam_size` explicitly. No action, just don't
   be confused by it.
2. **Cap `max_decoding_length`.** The API default is 256
   ([app.py](it2edge/serve/app.py#L86)) but training used `max_target_length=128`
   ([final.md](final.md) §2). Dropping the default to ~128 bounds worst-case latency at no
   real quality cost, since the model never learned to emit longer.
3. **Sentence-split long inputs** before sending. Latency scales with tokens *generated*;
   one paragraph is much worse than N sentences, and N sentences batch well.
4. **Batch when requests arrive together.** `/translate` already accepts `texts` and
   `translate_batch` handles the batch natively — one round trip beats N.
5. **Keep the model warm.** Non-negotiable on a Pi. The cold start is seconds to tens of
   seconds; one-shot CLI invocations pay it every time. This is the single biggest
   practical difference between a usable and unusable deployment.
6. **Swap / zram on 1 GB boards.** If §6 shows RSS near the limit, a small zram swap keeps
   the OOM killer away at the cost of some latency.

Explicitly **not** worth doing: further quantization below int8 (CT2 has no lower dense
mode that helps here), or unstructured pruning — [docs/plan.md](docs/plan.md) §0 already
covers why it gives zero CPU speedup on this stack.

---

## 8. Risks & gotchas

| # | Risk | Mitigation |
|---|---|---|
| 1 | **32-bit OS** (`armv7l`) | No aarch64 wheels ⇒ hours-long source build. Reflash 64-bit. Check in §1 first. |
| 2 | **Python < 3.10** on the Pi | `str \| None` in [marian_ct2.py](it2edge/serve/marian_ct2.py#L20) fails at import. Check in §3.1. |
| 3 | **Unpinned `pi.txt`** | Bare metal drifts from the tested container versions. Pin it (§3.3). |
| 4 | **`transformers` 4.52.4 vs 4.57.6 skew** | Verify tokenizer load (§3.3); bump both requirement files together if it fails. |
| 5 | **`<registry>` placeholder in Quadlet** | Unit fails to start silently-ish. Fix before install (§5.4). |
| 6 | **Truncated `scp` of `model.bin`** | Checksum both ends (§2). |
| 7 | **`/health` 503 read as failure** | It's the documented pre-warm state. Wait, or watch `podman logs -f it2`. |
| 8 | **Multiple uvicorn workers** | Each duplicates the model ⇒ OOM. Keep `--workers 1`; parallelism comes from CT2 threads. |
| 9 | **`--cpus` cap contradicting CT2 threads** | Keep the Podman cap ≥ `inter × intra` (§5.1). |
| 10 | **SD card space** | ~450 MB image + build cache. `df -h` before building. |
| 11 | **Thermal throttling skewing benchmarks** | `vcgencmd get_throttled` must be `0x0` (§6.2). |
| 12 | **Untested reboot persistence** | Actually reboot and re-check (§5.4). |

---

## 9. Sequencing

| Milestone | Work | Gate |
|---|---|---|
| **M0 — Recon** (30 min) | §1 board check, `uname -m`, RAM, Python version | `aarch64` confirmed, lane chosen |
| **M1 — Transfer** (30 min) | §2 scp + checksum | Checksums match on both ends |
| **M2 — Bare metal** (1–2 h) | §3 env + §3.3 pins, §4 smoke test + service | Hindi output matches the laptop exactly |
| **M3 — Benchmark** (1–2 h) | §6 write `bench_pi.py`, sweep threads, record thermals | `pi_latency_*.json` written, [final.md](final.md) §4 updated with real numbers |
| **M4 — Container** (2–3 h) | §5 build on Pi, run, Quadlet + reboot test | Service survives a reboot on its own |
| **M5 — Tune** (optional) | §7, only against M3 numbers | Latency target met, or documented as not reachable on this board |

**M2 and M3 are the load-bearing milestones.** M4 is packaging convenience; if the board
is 1 GB or smaller, stopping after M3 with a systemd unit wrapping the bare-metal uvicorn
is a perfectly good outcome.

---

## 10. Definition of done

- [ ] `uname -m` on the Pi is `aarch64`
- [ ] `model_cache_compact_ct2/model.bin` checksum matches the laptop
- [ ] `python -m it2edge.serve.translate_ct2 "Hello, how are you?"` on the Pi prints the same Hindi as the laptop
- [ ] `POST /translate` returns correct Hindi over the network from another machine
- [ ] `requirements/pi.txt` pinned and verified to load the tokenizer
- [ ] `pi_latency_<board>.json` exists with median + p95 per-sentence latency and RSS
- [ ] `vcgencmd get_throttled` was `0x0` during the benchmark
- [ ] [final.md](final.md) §4 Pi table shows **measured** numbers for the tested board
- [ ] Service auto-starts after a physical reboot (Quadlet, or a bare-metal systemd unit)
- [x] `deploy/it2.container` has a real `Image=` value, not `<registry>`

---

## 11. How to test

### 11.1 On this laptop — the service is already running

Started with:

```bash
source venv/bin/activate
CT2_INTER_THREADS=1 CT2_INTRA_THREADS=2 OMP_NUM_THREADS=2 \
  uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
```

Test it:

```bash
# health — expect {"status":"ok"}
curl -s http://127.0.0.1:8080/health

# single sentence
curl -s -X POST http://127.0.0.1:8080/translate \
  -H 'content-type: application/json' \
  -d '{"text":"Hello, how are you?"}'

# batch — one round trip, much faster than N calls
curl -s -X POST http://127.0.0.1:8080/translate \
  -H 'content-type: application/json' \
  -d '{"texts":["Hello","How are you?","Good morning"]}'

# higher quality, slower
curl -s -X POST http://127.0.0.1:8080/translate \
  -H 'content-type: application/json' \
  -d '{"text":"The committee will review the proposal next week.","beam_size":5}'
```

**Verified outputs** (all confirmed working):

| Input | Output |
|---|---|
| `Hello, how are you?` | हेलो, आप कैसे हैं? |
| `Good morning` | सुप्रभात |
| `The weather is beautiful this morning.` | आज सुबह मौसम सुंदर है। |
| `The committee will review the proposal next week.` (beam 5) | समिति प्रस्ताव की पुनरीक्षा अगले सप्ताह करेगी। |

Error paths, also verified: `{}` → **422** `provide 'text' or 'texts'`; `{"texts":[]}` →
**200** with `[]`; `GET /translate` → **405** (this is the bug that was hiding in the old
curl example).

CLI, without the server:

```bash
python -m it2edge.serve.translate_ct2 "Hello, how are you?"
```

Stop the server with `pkill -f 'uvicorn it2edge.serve.app'`.

### 11.2 Deploy to the Pi — the one command that's left

The Pi at `10.42.0.187` is up but needs a password (no SSH key installed). Make it
passwordless first, then push:

```bash
ssh-copy-id pi@10.42.0.187                      # one time, asks for the Pi password
bash deploy/push_to_pi.sh pi@10.42.0.187 --setup
```

`push_to_pi.sh` refuses to continue unless the Pi reports `aarch64`, copies the model +
code, **verifies `model.bin` by sha256 on both ends**, creates the venv, installs the
pinned deps, and runs a smoke translation. Drop `--setup` to copy only.

Then start the service on the Pi:

```bash
ssh pi@10.42.0.187
cd entm && source venv/bin/activate
CT2_INTRA_THREADS=$(nproc) OMP_NUM_THREADS=$(nproc) \
  uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
```

And test it **from the laptop**, which is the real end-to-end check:

```bash
curl -s http://10.42.0.187:8080/health
curl -s -X POST http://10.42.0.187:8080/translate \
  -H 'content-type: application/json' \
  -d '{"text":"Hello, how are you?"}'
```

The Hindi must be byte-identical to the laptop output above. If it differs, the artifact
moved wrong — re-check the checksum before debugging anything else.

> `/health` returns **503**, not an error, until the model finishes loading. On a Pi that
> can take tens of seconds. Wait, or watch the log.

### 11.3 Benchmark and make it permanent

```bash
# on the Pi
python -m it2edge.evaluate.benchmark_models latency \
  --ct2_dir model_cache_compact_ct2 --data_dir en-indic-exp \
  --inter 1 --intra $(nproc) --num 40 --warmup 5 \
  --out pi_latency_$(uname -m).json

# auto-start on boot — bare metal (best for 1 GB boards)
mkdir -p ~/.config/systemd/user
cp deploy/it2-baremetal.service ~/.config/systemd/user/it2.service
loginctl enable-linger $USER
systemctl --user daemon-reload && systemctl --user enable --now it2
journalctl --user -u it2 -f

# or containerized (Pi 4/5, >=2 GB)
bash deploy/build_on_pi.sh
```

Then **reboot the Pi** and confirm `curl http://10.42.0.187:8080/health` comes back on its
own. An auto-start that was never tested through a real reboot is not an auto-start.
