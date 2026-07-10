"""Download the IndicTrans2 en-indic distilled 200M model to a local folder.

Run this ONCE (with internet). It caches the model + tokenizer weights into
./model_cache so that translate.py can run fully offline afterwards.

    python download_model.py

If IndicTransToolkit is missing, install it first (not on PyPI):

    pip install -r requirements.txt
    pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
"""

import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Downloading {MODEL_ID} into {CACHE_DIR} ...")

    # trust_remote_code=True: the model repo ships a custom tokenizer + model class.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR
    )

    # Save a clean, self-contained copy so inference never touches the network.
    local_dir = os.path.join(CACHE_DIR, "indictrans2-en-indic-dist-200M")
    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)

    print(f"Done. Model saved to {local_dir}")
    print("You can now run translate.py fully offline.")


if __name__ == "__main__":
    main()
