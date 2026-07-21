"""Centralized project paths and constants.

Every script derives file locations from here. Model caches and CT2 outputs live
at the PROJECT ROOT (parent of this package), not inside the package.
"""

from pathlib import Path

# paths.py is it2edge/paths.py -> parents[0] = it2edge, parents[1] = project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Primary model: compact English→Hindi MarianMT (~77M).
MODEL_ID = "AI4INDIANS/better-opus-mt-en-hi"
CONTROL_MODEL_ID = "Helsinki-NLP/opus-mt-en-hi"  # optional A/B control
MODEL_REVISION = "b513145b43f11b01c0b25120f5dd5509e8698fd8"
CONTROL_REVISION = "75d7f7c9232b2891c7d65fe4ef635616c72be867"

# Downloaded HF snapshot (base) + fine-tuned checkpoint + CT2 int8 deployable.
MODEL_CACHE = PROJECT_ROOT / "model_cache_compact"
HF_SNAPSHOT = MODEL_CACHE / "better-opus-mt-en-hi"
CONTROL_SNAPSHOT = MODEL_CACHE / "opus-mt-en-hi"
FINETUNED_DIR = PROJECT_ROOT / "model_cache_compact_ft"
CT2_DIR = PROJECT_ROOT / "model_cache_compact_ct2"

# Corpus root (line-aligned en-hi).
DATA_DIR = PROJECT_ROOT / "en-indic-exp"
