# Reducing 200M → 5M: will it cut latency, and how?

> Question: the model (`ai4bharat/indictrans2-en-indic-dist-200M`) takes ~35–45 s
> per output on a Raspberry Pi 3 B+. It's already fine-tuned, merged, int8-quantized
> (see `model_cache_ct2/model.bin`, 213 MB). Can I drop it from **200M → 5M**
> parameters, and will that reduce latency?

---

## 0. Short answer

**Yes, a smaller model would be faster — but "shrink 200M to 5M" is not a real
knob you can turn, and chasing it is the wrong move for you.** Three hard truths:

1. **You cannot "reduce" a 200M model to 5M.** There is no operation that takes an
   existing trained network and deletes 97.5% of its parameters while keeping it
   working. Structured pruning can remove maybe **10–30%** of a model before quality
   collapses — and on an *already-distilled* model like this one, even that is
   optimistic (distillation already removed the slack). A 5M model is a **different,
   from-scratch model** that you would have to **train yourself** (knowledge
   distillation), which needs a GPU, a big parallel corpus, days of work, and will
   translate noticeably worse.
2. **A 40× smaller model does NOT give 40× less latency.** Translation latency here is
   dominated by **autoregressive decoding** — the model generates the Hindi output
   **one token at a time**, and each token is a full pass through the decoder. Latency
   scales roughly with *(output length) × (decoder depth × width)*. Shrinking the model
   cuts the *per-step* cost but not the *number of steps*. Realistically a 5M model
   might get you **~3–6×** faster, not 40×.
3. **Most of your 35–45 s is probably not even the translation model.** On a Pi 3 B+
   the pipeline is STT (speech→text) + translation + TTS (text→speech). Before
   rebuilding the translator, you need to **measure which stage is slow** (§2). It is
   very common for STT (e.g. Whisper) to be the real bottleneck, in which case a
   smaller translator changes almost nothing.

**Recommendation:** do **not** try to build a 5M model. Instead (a) measure where the
time goes, (b) apply the cheap decoding/threading wins in §3 that can cut translation
latency with zero quality loss, and (c) if you still need a smaller *translation* model,
distill to **~50M**, not 5M. Details below.

---

## 1. Why "200M → 5M" isn't a parameter you can set

Parameter count is a property of the **architecture** (how many layers, how wide each
layer, vocabulary size), fixed the moment the model was designed and trained. To change
it you change the architecture and re-train. The two families of size reduction:

| Technique | What it does | Max realistic reduction | Speeds up CPU? | Quality cost |
|---|---|---|---|---|
| **Quantization** (int8) | Stores weights in 8-bit instead of fp32/fp16 | ~4× smaller file | ✅ some | tiny — **you already did this** |
| **Unstructured pruning** | Zeroes individual weights | file smaller only | ❌ **no** (dense CPU ignores sparsity) | grows fast |
| **Structured pruning** | Drops whole heads / FFN neurons / layers | ~10–30% before collapse | ✅ yes | measurable, needs "healing" retrain |
| **Knowledge distillation** | Train a *new smaller* model to imitate this one | any size (5M possible) | ✅ yes | **large** for 40× — this is a new model, not a reduction |

Note the vocabulary tax: IndicTrans2's embedding + output tables are **huge** (the
target vocab file is ~3.4 MB of tokens). A large chunk of the 200M parameters is
**embeddings**, which you cannot prune without breaking the tokenizer. That alone
makes 5M nearly impossible while keeping IndicTrans2's multilingual vocabulary.

**Conclusion:** 5M via "reduction" is not achievable. 5M via "train a new tiny model"
is achievable but is a research-grade effort with a real quality hit.

---

## 2. FIRST — measure where the 35–45 s actually goes (do this before anything)

Don't optimize blind. Time each stage on the Pi:

```bash
# On the Pi. Time ONLY the translation model on a fixed sentence, no STT/TTS:
python - <<'PY'
import time
import ctranslate2, sentencepiece  # or however translate_ct2 loads it
# load your CT2 model from model_cache_ct2/
t0 = time.time()
# ... run one translation of a typical sentence ...
print("translate only:", round(time.time()-t0, 2), "s")
PY
```

Then time the full pipeline and subtract. You are looking for the split between:

- **STT** (mic → English text) — often the biggest cost on a Pi 3 B+
- **Translation** (English → Hindi) — the model in this doc
- **TTS** (Hindi text → speech) — usually small
- **Model load time** — if you load the model per request, that alone can be many
  seconds; load **once** and keep it resident.

**If translation is, say, 8 s of the 40 s, then shrinking it to 5M can save you at most
~6 s of the 40** — not worth building a new model. Decide based on this number.

> The project's own `des.txt` already states the work is "CPU-bound" and that a
> Pi 4/Pi 5 gives 3–8× speedup with the **same model**. That is the honest, low-risk
> path and matches truth #2 above.

---

## 3. Cheap wins that reduce translation latency with NO new model (do these first)

These require **no retraining** and **no quality loss** (or negligible):

1. **Greedy decoding, beam size = 1.** If you're using beam search (beam 4/5), each
   token does N× the work. Greedy (`beam_size=1`) is often 3–5× faster with minor
   quality change for short sentences.
   - CTranslate2: pass `beam_size=1` to `translate_batch(...)`.
2. **Cap `max_decoding_length`.** Don't let the decoder run to 256 tokens for a short
   phrase. Set it to something like 1.5× the source length.
3. **Pin threads to the Pi's cores.** The Pi 3 B+ has 4 cores. Set
   `intra_threads=4` (CTranslate2) and export `OMP_NUM_THREADS=4`. Wrong thread
   counts can *halve* throughput.
4. **Load the model once, keep it warm.** Verify you're not re-loading `model.bin`
   (213 MB) on every request. Use a long-lived server process.
5. **Confirm int8 is actually active at runtime.** The file is int8, but make sure the
   CT2 model is loaded with `compute_type="int8"` (or `int8` for ARM). On some builds
   it silently falls back to float.
6. **Shorten inputs.** Send one sentence at a time; long paragraphs = many decode steps.

Apply §3, re-measure §2. Frequently this alone takes translation from "slow" to "fine"
and the 5M question disappears.

---

## 4. If you STILL need a smaller translation model

In order of increasing effort / risk:

### 4a. Structured pruning + healing (moderate effort, ~10–30% faster)
The full recipe is already written up in `docs/plan.md` §5b. Summary:
- Prune **decoder depth** (drop whole decoder layers) — decoder depth dominates
  autoregressive latency. Narrow the encoder only modestly.
- **Heal**: fine-tune the pruned model on your parallel corpus (you have
  `en-indic-exp-mini/`), re-check BLEU/chrF, then re-convert to int8 CT2.
- **Constraint**: the CT2 converter needs **uniform topology** — prune the same
  heads/FFN per layer, or drop whole layers. Heterogeneous pruning won't round-trip.
- **Honest verdict** (from plan.md): on an already-distilled 200M, expect only
  ~10–30% and a measurable quality cost. Not transformative.

### 4b. Distill to a NEW ~50M model (high effort, ~3–4× faster, real quality cost)
This is the closest realistic version of "make it much smaller":
1. Define a smaller architecture (fewer/narrower layers, **same tokenizer/vocab** so
   embeddings still work).
2. Use the current 200M as the **teacher**; train the small **student** to match the
   teacher's outputs (sequence-level KD) on a large en→hi parallel corpus.
3. Needs a **GPU** and a **large dataset** (your `Dataset_English_Hindi.csv` is a
   start; real KD wants millions of pairs). Days of training.
4. Quantize the student to int8, convert to CT2, ship.
- **Target ~50M, not 5M.** 50M keeps usable quality; 5M for 10-language-vocab NMT will
  produce badly broken Hindi.

### 4c. Swap to an inherently smaller model (least effort if quality acceptable)
Rather than build one, check whether AI4Bharat or others publish a smaller/tinier
IndicTrans variant, or an en→hi-only single-pair model (single-pair models can be far
smaller than the 22-language model because they drop most of the vocabulary).

---

## 5. Concrete step-by-step for what I actually recommend

```
Step 1  Measure: time STT vs translate vs TTS vs model-load on the Pi (§2).
Step 2  If translation is NOT the dominant cost → stop; optimize the real bottleneck
        (usually STT) or upgrade the board (Pi 4/5 = 3–8× per des.txt).
Step 3  If translation IS a big chunk → apply §3 (beam=1, thread pinning, warm load,
        confirm int8, cap length). Re-measure.
Step 4  Still too slow AND translation-bound → structured-prune decoder depth + heal
        (plan.md §5b), re-convert int8. Expect ~10–30%.
Step 5  Need much smaller → distill a NEW ~50M student with the 200M as teacher (§4b).
        Do NOT target 5M.
```

---

## 6. Bottom line

- **Can you reduce 200M → 5M?** No — not by reducing. You'd have to train a new 5M
  model from scratch (distillation), and at 5M for multilingual NMT the output quality
  would be too poor to use.
- **Would a smaller model reduce latency?** Yes, but only for the *translation* stage,
  and roughly proportional to model shrink in the *per-token* cost — realistically
  **~3–6×** for a well-built small model, not 40×, and never for free.
- **What you should do instead:** measure the pipeline (§2), take the free decoding/
  threading wins (§3), and only then consider pruning (~10–30%) or distilling to ~50M.
  A Pi 4/Pi 5 upgrade is the lowest-risk 3–8× and needs no model changes at all.
