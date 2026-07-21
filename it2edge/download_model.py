"""Download the English→Hindi MarianMT model to a local folder.

Run once (with internet). Snapshots the pinned Hugging Face revision into
./model_cache_compact and writes provenance.json for reproducibility.

    python -m it2edge.download_model
    python -m it2edge.download_model --control   # official OPUS baseline
    python -m it2edge.download_model --all
"""

import argparse
import json
from datetime import datetime, timezone

from huggingface_hub import snapshot_download

from it2edge.paths import (
    CONTROL_MODEL_ID,
    CONTROL_REVISION,
    CONTROL_SNAPSHOT,
    HF_SNAPSHOT,
    MODEL_CACHE,
    MODEL_ID,
    MODEL_REVISION,
    PROJECT_ROOT,
)

_TARGETS = {
    MODEL_ID: (
        MODEL_REVISION,
        HF_SNAPSHOT,
        "Samanantar-fine-tuned MarianMT en→hi (~77M). Apache-2.0.",
    ),
    CONTROL_MODEL_ID: (
        CONTROL_REVISION,
        CONTROL_SNAPSHOT,
        "Official OPUS MarianMT en→hi control (~77M). Apache-2.0.",
    ),
}


def _download_one(repo_id: str) -> None:
    revision, local_dir, note = _TARGETS[repo_id]
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id}@{revision[:12]} -> {local_dir} ...")
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(local_dir))

    provenance = {
        "repo_id": repo_id,
        "revision": revision,
        "note": note,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "source": "huggingface_hub.snapshot_download",
    }
    (local_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    rel = local_dir.relative_to(PROJECT_ROOT)
    print(f"Done. Snapshot + provenance.json saved to {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the MarianMT en→hi model(s), revision-pinned"
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="download the official OPUS control instead of the primary model",
    )
    parser.add_argument(
        "--all", action="store_true", help="download both primary and control"
    )
    args = parser.parse_args()

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    if args.all:
        _download_one(MODEL_ID)
        _download_one(CONTROL_MODEL_ID)
    elif args.control:
        _download_one(CONTROL_MODEL_ID)
    else:
        _download_one(MODEL_ID)

    print("\nNext: fine-tune with")
    print("      python -m it2edge.train.finetune_marian --data_dir en-indic-exp")


if __name__ == "__main__":
    main()
