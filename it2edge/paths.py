"""Centralized project paths and constants.

Every script derives its file locations from here instead of recomputing a local
`HERE = os.path.dirname(__file__)`. The data directories (model_cache/, the CT2
and merged/pruned/ONNX outputs) live at the PROJECT ROOT -- the parent of this
package -- not inside it, so they stay put no matter which submodule imports them.
"""

from pathlib import Path

# paths.py is it2edge/paths.py -> parents[0] = it2edge, parents[1] = project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"

# Downloaded HF snapshot (from it2edge.download_model) + its tokenizer files.
MODEL_CACHE = PROJECT_ROOT / "model_cache"
HF_SNAPSHOT = MODEL_CACHE / "indictrans2-en-indic-dist-200M"

# Pipeline artifacts (all regenerated; git-ignored).
MERGED_DIR = PROJECT_ROOT / "model_cache_merged"   # merge_lora.py  -> fp16
PRUNED_DIR = PROJECT_ROOT / "model_cache_pruned"   # prune_structured.py
CT2_DIR = PROJECT_ROOT / "model_cache_ct2"         # convert_ct2.py -> int8
ONNX_DIR = PROJECT_ROOT / "onnx_model"             # export_onnx.py

# Tokenizer files staged into the container image by deploy/build_arm64.sh.
TOKENIZER_STAGE = PROJECT_ROOT / "tokenizer"
