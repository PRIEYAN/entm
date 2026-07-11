"""it2edge -- fine-tune, compress, and deploy IndicTrans2 to a low-power ARM device.

Pipeline stages (see plan.md and the README):
    it2edge.train    -- QLoRA fine-tune, merge, optional structured prune
    it2edge.convert  -- int8 conversion (CTranslate2 primary, ONNX alternative)
    it2edge.serve    -- FastAPI service + CLI translators

Shared helpers live at the package root: paths (project-relative locations) and
tokenizer_utils (robust IndicTrans2 tokenizer loader).
"""

__all__ = ["paths", "tokenizer_utils"]
