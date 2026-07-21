"""it2edge — fine-tune, compress, and deploy MarianMT en→hi to a low-power ARM device.

Pipeline:
    it2edge.train     — Marian fine-tune on project corpus
    it2edge.convert   — CTranslate2 int8 conversion
    it2edge.serve     — FastAPI + CLI translators
    it2edge.evaluate  — quality / latency benchmarks
"""

__all__ = ["paths", "corpus_utils"]
