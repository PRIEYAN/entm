"""FastAPI translation service — warm MarianMT CTranslate2 int8, low latency.

Runs inside the Podman container (or bare metal) on the Raspberry Pi.

Design:
  * LONG-LIVED HTTP SERVICE — model loaded once at startup and kept warm.
  * SINGLE uvicorn worker — avoid duplicating the model on 1–2 GB RAM.
  * Greedy decode (beam_size=1) by default for lowest latency.

Endpoints:
    GET  /health
    POST /translate  {text|texts, beam_size?, max_decoding_length?}

Env:
    CT2_MODEL_DIR      default ./model_cache_compact_ct2
    TOKENIZER_DIR      default = CT2_MODEL_DIR (tokenizer lives with the CT2 package)
    CT2_INTER_THREADS  default 1
    CT2_INTRA_THREADS  default 2
    DEFAULT_BEAM_SIZE  default 1
    OMP_NUM_THREADS    default 2

    uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from it2edge.paths import CT2_DIR
from it2edge.serve.marian_ct2 import load_marian, translate_marian


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


CT2_MODEL_DIR = os.environ.get("CT2_MODEL_DIR", str(CT2_DIR))
TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR", CT2_MODEL_DIR)
INTER_THREADS = _env_int("CT2_INTER_THREADS", 1)
INTRA_THREADS = _env_int("CT2_INTRA_THREADS", 2)
DEFAULT_BEAM_SIZE = _env_int("DEFAULT_BEAM_SIZE", 1)
DEFAULT_TGT_LANG = os.environ.get("DEFAULT_TGT_LANG", "hi")

_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not os.path.isdir(CT2_MODEL_DIR):
        raise SystemExit(
            f"CT2 model not found at {CT2_MODEL_DIR}. Build it with "
            "`python -m it2edge.convert.convert_ct2`, or set CT2_MODEL_DIR."
        )

    print(
        f"[startup] loading Marian CT2 int8 from {CT2_MODEL_DIR} "
        f"(inter={INTER_THREADS}, intra={INTRA_THREADS})"
    )
    tok_dir = TOKENIZER_DIR if os.path.isdir(TOKENIZER_DIR) else CT2_MODEL_DIR
    _state["tokenizer"], _state["translator"] = load_marian(
        CT2_MODEL_DIR, tok_dir, INTER_THREADS, INTRA_THREADS
    )
    print("[startup] model warm; ready to serve.")
    yield
    _state.clear()


app = FastAPI(title="MarianMT en-hi (CTranslate2 int8)", lifespan=lifespan)


class TranslateRequest(BaseModel):
    text: Optional[str] = Field(None, description="single English sentence")
    texts: Optional[List[str]] = Field(None, description="batch of English sentences")
    tgt_lang: str = Field(DEFAULT_TGT_LANG, description="target language (en→hi only)")
    beam_size: int = Field(DEFAULT_BEAM_SIZE, ge=1, description="1 = greedy (fastest)")
    max_decoding_length: int = Field(256, ge=1, le=1024)


class TranslateResponse(BaseModel):
    tgt_lang: str
    translations: List[str]


def _translate(sentences: List[str], beam_size: int,
               max_decoding_length: int) -> List[str]:
    return translate_marian(
        sentences,
        _state["tokenizer"],
        _state["translator"],
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
    )


@app.get("/health")
def health():
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
        out = _translate(sentences, req.beam_size, req.max_decoding_length)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"translation failed: {exc}")

    return TranslateResponse(tgt_lang=req.tgt_lang, translations=out)
