"""Download the IndicTrans2 en-indic distilled 200M model to a local folder.

Run this ONCE (with internet). It snapshots the model repo into ./model_cache
so translate.py can run fully offline afterwards.

    python download_model.py

IMPORTANT: we use snapshot_download (a raw file copy of the HF repo), NOT
tokenizer.save_pretrained(). save_pretrained re-serialises the tokenizer's
constructor args (src_vocab_file / tgt_vocab_file) into tokenizer_config.json;
on reload the model's remote tokenizer then receives those paths BOTH
positionally and as keywords, raising:
    TypeError: __init__() got multiple values for keyword argument 'src_vocab_file'
A raw snapshot keeps the repo's original config untouched and avoids that.

If IndicTransToolkit is missing, install it first (not on PyPI):

    pip install -r requirements.txt
    pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
"""

import os

from huggingface_hub import snapshot_download

MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
LOCAL_DIR = os.path.join(CACHE_DIR, "indictrans2-en-indic-dist-200M")


def main() -> None:
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"Downloading {MODEL_ID} into {LOCAL_DIR} ...")

    # Raw snapshot of the repo: weights, config, remote code, vocab/spm files.
    # A snapshot writes real files into local_dir (safe to copy to another
    # machine) without going through save_pretrained, so the repo's original
    # tokenizer_config.json is preserved as-is.
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
    )

    print(f"Done. Model snapshot saved to {LOCAL_DIR}")
    print("You can now run translate.py fully offline.")


if __name__ == "__main__":
    main()
