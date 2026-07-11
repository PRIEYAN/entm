"""Download the IndicTrans2 en-indic distilled 200M model to a local folder.

Run this ONCE (with internet). It snapshots the model repo into ./model_cache
so the translators can run fully offline afterwards.

    python -m it2edge.download_model

IMPORTANT: we use snapshot_download (a raw file copy of the HF repo), NOT
tokenizer.save_pretrained(). save_pretrained re-serialises the tokenizer's
constructor args (src_vocab_file / tgt_vocab_file) into tokenizer_config.json;
on reload the model's remote tokenizer then receives those paths BOTH
positionally and as keywords, raising:
    TypeError: __init__() got multiple values for keyword argument 'src_vocab_file'
A raw snapshot keeps the repo's original config untouched and avoids that.

If IndicTransToolkit is missing, install it first (not on PyPI):

    pip install -r requirements/dev.txt
    pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
"""

from huggingface_hub import snapshot_download

from it2edge.paths import HF_SNAPSHOT, MODEL_ID

LOCAL_DIR = str(HF_SNAPSHOT)


def main() -> None:
    HF_SNAPSHOT.mkdir(parents=True, exist_ok=True)
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
    print("You can now run `python -m it2edge.serve.translate` fully offline.")


if __name__ == "__main__":
    main()
