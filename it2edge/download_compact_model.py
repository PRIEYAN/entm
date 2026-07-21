"""Download the compact English->Hindi Marian model(s) to a local folder.

Run this ONCE (with internet) on the DEV MACHINE. It snapshots the pinned HF
revision into ./model_cache_compact and records provenance (repo id, revision,
license, param count) so the artifact that gets fine-tuned and shipped is
reproducible. See docs/compact-model.md.

    python -m it2edge.download_compact_model                 # the fine-tuned base
    python -m it2edge.download_compact_model --control       # official OPUS control
    python -m it2edge.download_compact_model --all           # both

Why pin a revision: model cards and weights on the hub can change under the same
name. Pinning the exact commit SHA means the benchmark you sign off on is the
model you deploy. Update the SHAs here deliberately, not by accident.
"""

import argparse
import json
from datetime import datetime, timezone

from huggingface_hub import snapshot_download

from it2edge.paths import (
    COMPACT_CACHE,
    COMPACT_CONTROL_ID,
    COMPACT_MODEL_ID,
    PROJECT_ROOT,
)

# Pinned commit SHAs (verified on the hub). Change only on purpose.
COMPACT_REVISION = "b513145b43f11b01c0b25120f5dd5509e8698fd8"   # ~77M, F32
CONTROL_REVISION = "75d7f7c9232b2891c7d65fe4ef635616c72be867"

# repo_id -> (revision, local subdir name, human note)
_TARGETS = {
    COMPACT_MODEL_ID: (
        COMPACT_REVISION,
        "better-opus-mt-en-hi",
        "Samanantar-fine-tuned MarianMT en->hi (~77M). Apache-2.0.",
    ),
    COMPACT_CONTROL_ID: (
        CONTROL_REVISION,
        "opus-mt-en-hi",
        "Official OPUS MarianMT en->hi control (~77M). Apache-2.0.",
    ),
}


def _download_one(repo_id: str) -> None:
    revision, subdir, note = _TARGETS[repo_id]
    local_dir = COMPACT_CACHE / subdir
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
        description="Download the compact en->hi Marian model(s), revision-pinned"
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="download the official OPUS control instead of the fine-tuned base",
    )
    parser.add_argument(
        "--all", action="store_true", help="download both the base and the control"
    )
    args = parser.parse_args()

    if args.all:
        _download_one(COMPACT_MODEL_ID)
        _download_one(COMPACT_CONTROL_ID)
    elif args.control:
        _download_one(COMPACT_CONTROL_ID)
    else:
        _download_one(COMPACT_MODEL_ID)

    print("\nNext: fine-tune on the project corpus with")
    print("      python -m it2edge.train.finetune_compact_marian --data_dir en-indic-exp")


if __name__ == "__main__":
    main()
