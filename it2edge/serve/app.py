"""FastAPI translation service -- warm CTranslate2 int8 model, low latency.

This is what runs INSIDE the Podman container on the Raspberry Pi. Plan §7, §8, §9.

Design decisions (all from the plan):
  * LONG-LIVED HTTP SERVICE, not a one-shot CLI. The model is loaded ONCE at
    startup and kept warm; a CLI would pay the ~200 MB load every request (§8).
  * SINGLE uvicorn worker. On 2 GB RAM each extra worker duplicates the model
    and OOMs. Parallelism comes from CT2 threads, not processes (§8).
  * Threading tuned for lowest single-sentence latency: inter_threads=1,
    intra_threads=2 puts both Pi cores on one translation. Rule:
    inter_threads * intra_threads <= physical cores (§9). Overridable via env.
  * Greedy (beam_size=1) default -- the biggest autoregressive-latency lever (§9).

Endpoints:
    GET  /health                      -> {"status": "ok"} once the model is warm
    POST /translate  {text|texts, tgt_lang, beam_size?, max_decoding_length?}

Config via env (with plan-recommended defaults):
    CT2_MODEL_DIR   default ./model_cache_ct2
    TOKENIZER_DIR   default ./tokenizer  (falls back to the model_cache snapshot)
    CT2_INTER_THREADS  default 1
    CT2_INTRA_THREADS  default 2   (both cores on one translation)
    OMP_NUM_THREADS    default 2   (set in the Containerfile too)
    DEFAULT_TGT_LANG   default hin_Deva
    DEFAULT_BEAM_SIZE  default 1   (greedy)

Local dev:
    pip install fastapi "uvicorn[standard]" ctranslate2 "transformers>=4.51,<4.53" \
        sentencepiece sacremoses protobuf
    pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
    uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

import ctranslate2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from it2edge.paths import COMPACT_CT2_DIR, CT2_DIR, HF_SNAPSHOT, TOKENIZER_STAGE
from it2edge.tokenizer_utils import detokenize_target, load_indictrans_tokenizer

SRC_LANG = "eng_Latn"

# "indictrans" (default, legacy 200M) or "marian" (compact en->hi Pi runtime).
MODEL_TYPE = os.environ.get("MODEL_TYPE", "indictrans").lower()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


_default_ct2 = COMPACT_CT2_DIR if MODEL_TYPE == "marian" else CT2_DIR
CT2_MODEL_DIR = os.environ.get("CT2_MODEL_DIR", str(_default_ct2))
# The tokenizer dir is baked into the image at /app/tokenizer (Containerfile),
# but fall back to the dev-machine snapshot path so local runs work unchanged.
_tok_default = TOKENIZER_STAGE if TOKENIZER_STAGE.is_dir() else HF_SNAPSHOT
TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR", str(_tok_default))

INTER_THREADS = _env_int("CT2_INTER_THREADS", 1)
INTRA_THREADS = _env_int("CT2_INTRA_THREADS", 2)
DEFAULT_TGT_LANG = os.environ.get("DEFAULT_TGT_LANG", "hin_Deva")
DEFAULT_BEAM_SIZE = _env_int("DEFAULT_BEAM_SIZE", 1)

# Shared warm state, populated at startup.
_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model ONCE at startup and keep it warm (plan §8).
    if not os.path.isdir(CT2_MODEL_DIR):
        raise SystemExit(
            f"CT2 model not found at {CT2_MODEL_DIR}. Build it with "
            "`python -m it2edge.convert.convert_compact_ct2` (marian) or "
            "`python -m it2edge.convert.convert_ct2` (indictrans), or set CT2_MODEL_DIR."
        )

    print(
        f"[startup] loading {MODEL_TYPE} CT2 int8 model from {CT2_MODEL_DIR} "
        f"(inter={INTER_THREADS}, intra={INTRA_THREADS})"
    )

    if MODEL_TYPE == "marian":
        # Compact en->hi Marian: tokenizer lives in the CT2 dir; no IndicProcessor.
        from it2edge.serve.marian_ct2 import load_marian

        tok_dir = TOKENIZER_DIR if os.path.isdir(TOKENIZER_DIR) else CT2_MODEL_DIR
        _state["tokenizer"], _state["translator"] = load_marian(
            CT2_MODEL_DIR, tok_dir, INTER_THREADS, INTRA_THREADS
        )
    else:
        if not os.path.isdir(TOKENIZER_DIR):
            raise SystemExit(
                f"Tokenizer not found at {TOKENIZER_DIR}. Copy the tokenizer files "
                "into the image, or set TOKENIZER_DIR."
            )
        from IndicTransToolkit.processor import IndicProcessor

        _state["tokenizer"] = load_indictrans_tokenizer(TOKENIZER_DIR)
        _state["translator"] = ctranslate2.Translator(
            CT2_MODEL_DIR,
            device="cpu",
            compute_type="int8",
            inter_threads=INTER_THREADS,
            intra_threads=INTRA_THREADS,
        )
        _state["processor"] = IndicProcessor(inference=True)

    print("[startup] model warm; ready to serve.")
    yield
    _state.clear()


_title = (
    "Compact en-hi Marian (CTranslate2 int8)"
    if MODEL_TYPE == "marian"
    else "IndicTrans2 en-indic (CTranslate2 int8)"
)
app = FastAPI(title=_title, lifespan=lifespan)


class TranslateRequest(BaseModel):
    text: Optional[str] = Field(None, description="single English sentence")
    texts: Optional[List[str]] = Field(None, description="batch of English sentences")
    tgt_lang: str = Field(DEFAULT_TGT_LANG, description="target language code")
    beam_size: int = Field(DEFAULT_BEAM_SIZE, ge=1, description="1 = greedy (fastest)")
    # Cap so a runaway decode can't blow latency (plan §9).
    max_decoding_length: int = Field(256, ge=1, le=1024)


class TranslateResponse(BaseModel):
    tgt_lang: str
    translations: List[str]


def _translate(sentences: List[str], tgt_lang: str, beam_size: int,
               max_decoding_length: int) -> List[str]:
    if MODEL_TYPE == "marian":
        from it2edge.serve.marian_ct2 import translate_marian

        return translate_marian(
            sentences, _state["tokenizer"], _state["translator"],
            beam_size=beam_size, max_decoding_length=max_decoding_length,
        )

    tokenizer = _state["tokenizer"]
    translator = _state["translator"]
    processor = _state["processor"]

    batch = processor.preprocess_batch(sentences, src_lang=SRC_LANG, tgt_lang=tgt_lang)
    encoded = tokenizer(batch, truncation=True, padding=False).input_ids
    source_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in encoded]

    results = translator.translate_batch(
        source_tokens,
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
        # Skip a per-step softmax under greedy -- small latency win (plan §9).
        return_scores=False,
    )

    # Detokenize target pieces directly; convert_tokens_to_ids/decode use the
    # SOURCE vocab and blank out the target output. See detokenize_target.
    decoded = [detokenize_target(res.hypotheses[0]) for res in results]
    return processor.postprocess_batch(decoded, lang=tgt_lang)


@app.get("/health")
def health():
    # Only report healthy once the model is warm (matches the Containerfile probe).
    if "translator" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return {"status": "ok"}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if "translator" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    if req.texts is not None:
        sentences = req.texts
    elif req.text is not None:
        sentences = [req.text]
    else:
        raise HTTPException(status_code=422, detail="provide 'text' or 'texts'")

    if not sentences:
        return TranslateResponse(tgt_lang=req.tgt_lang, translations=[])

    try:
        out = _translate(
            sentences, req.tgt_lang, req.beam_size, req.max_decoding_length
        )
    except Exception as exc:  # surface model errors as 500s, not a bare crash
        raise HTTPException(status_code=500, detail=f"translation failed: {exc}")

    return TranslateResponse(tgt_lang=req.tgt_lang, translations=out)
