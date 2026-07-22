"""FastAPI translation service — warm MarianMT CTranslate2 int8, low latency.

Runs inside the Podman container (or bare metal) on the Raspberry Pi.

Design:
  * LONG-LIVED HTTP SERVICE — model loaded once at startup and kept warm.
  * SINGLE uvicorn worker — avoid duplicating the model on 1–2 GB RAM.
  * Greedy decode (beam_size=1) by default for lowest latency.

Endpoints:
    GET  /health
    POST /translate  {text|texts, beam_size?, max_decoding_length?, speak?}
    POST /speak      {text|texts, ...}   translate + speak on the Pi's speaker

Env:
    CT2_MODEL_DIR      default ./model_cache_compact_ct2
    TOKENIZER_DIR      default = CT2_MODEL_DIR (tokenizer lives with the CT2 package)
    CT2_INTER_THREADS  default 1
    CT2_INTRA_THREADS  default 2
    DEFAULT_BEAM_SIZE  default 1
    OMP_NUM_THREADS    default 2
    SPEAK_BY_DEFAULT   default 0 — set 1 to speak every /translate response
    (Piper paths: PIPER_BIN, PIPER_VOICE, AUDIO_OUT, ALSA_DEVICE — see speak.py)

    uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1

TTS design: synthesis runs in a BACKGROUND THREAD so the HTTP response returns
as soon as the Hindi text is ready. Audio then plays on the Pi while the caller
is already free. Speaking is therefore fire-and-forget: a TTS failure is logged,
never turned into a failed translation.
"""

import os
import threading
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from it2edge.paths import CT2_DIR
from it2edge.serve.marian_ct2 import load_marian, translate_marian
from it2edge.serve.speak import speak, tts_available


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
SPEAK_BY_DEFAULT = _env_int("SPEAK_BY_DEFAULT", 0) == 1

_state: dict = {}


def _speak_async(sentences: List[str]) -> None:
    """Synthesize in the background so the HTTP response isn't blocked.

    Fire-and-forget by design: TTS problems (no speaker, no Piper) must never
    turn a successful translation into a failed request.
    """

    def _run():
        for sentence in sentences:
            if not sentence.strip():
                continue
            try:
                speak(sentence)
            except Exception as exc:  # noqa: BLE001 — never propagate to HTTP
                print(f"[tts] failed to speak {sentence!r}: {exc}")

    threading.Thread(target=_run, daemon=True).start()


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
    speak: Optional[bool] = Field(
        None,
        description="speak the Hindi on the Pi's speaker; "
        "None = use SPEAK_BY_DEFAULT env",
    )


class TranslateResponse(BaseModel):
    tgt_lang: str
    translations: List[str]
    spoken: bool = Field(False, description="audio was queued on the Pi's speaker")


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
    # TTS readiness is reported, never required: text translation must keep
    # working on a box with no Piper and no sound card (e.g. the dev laptop).
    tts_ok, tts_reason = tts_available()
    return {
        "status": "ok",
        "tts": {"available": tts_ok, "detail": tts_reason},
        "speak_by_default": SPEAK_BY_DEFAULT,
    }


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

    want_speak = SPEAK_BY_DEFAULT if req.speak is None else req.speak
    spoken = False
    if want_speak:
        tts_ok, tts_reason = tts_available()
        if tts_ok:
            _speak_async(out)
            spoken = True
        else:
            # Explicitly asked for audio but TTS isn't usable: say so in the
            # log and return spoken=false rather than failing the translation.
            print(f"[tts] speak requested but unavailable: {tts_reason}")

    return TranslateResponse(tgt_lang=req.tgt_lang, translations=out, spoken=spoken)


@app.post("/speak", response_model=TranslateResponse)
def speak_endpoint(req: TranslateRequest):
    """Translate and speak on the Pi's speaker.

    Same body as /translate, but audio is the point — so unlike /translate this
    returns 503 when TTS is unavailable instead of silently returning text.
    """
    tts_ok, tts_reason = tts_available()
    if not tts_ok:
        raise HTTPException(status_code=503, detail=f"TTS unavailable: {tts_reason}")

    req.speak = True
    return translate(req)
