#!/usr/bin/env bash
# Build AND run the serving image NATIVELY on the Raspberry Pi (aarch64).
# No registry, no QEMU, no cross-build -- the Pi builds its own arch directly.
# All Python deps are prebuilt aarch64 manylinux wheels, so nothing compiles.
#
# Prereqs on the Pi (64-bit Raspberry Pi OS):
#   - podman installed            (sudo apt install -y podman)
#   - this repo checked out, WITH model_cache_ct2/ present. That dir is
#     git-ignored, so copy it from the dev/GPU box, e.g. from your laptop:
#         scp -r model_cache_ct2 pi@<pi-ip>:~/entm/
#     (also copy model_cache/indictrans2-en-indic-dist-200M/ if you don't
#      already have tokenizer/ -- this script stages it for you if present.)
#
# Usage (on the Pi, from the project root):
#   bash deploy/build_on_pi.sh            # build + run, serves on :8080
#   bash deploy/build_on_pi.sh --build    # build only

set -euo pipefail

IMAGE="${IMAGE:-it2-ct2}"
TAG="${TAG:-latest}"
CONTAINERFILE="deploy/Containerfile"
CONTAINER_NAME="${CONTAINER_NAME:-it2}"

cd "$(dirname "$0")/.."

# --- sanity: the int8 model must exist to bake into the image ---
if [ ! -d model_cache_ct2 ]; then
  echo "error: model_cache_ct2/ not found in $(pwd)." >&2
  echo "       Copy it from the dev box: scp -r model_cache_ct2 pi@<pi>:$(pwd)/" >&2
  exit 1
fi

# --- stage tokenizer/ from the snapshot if not already present ---
if [ ! -d tokenizer ]; then
  SNAP="model_cache/indictrans2-en-indic-dist-200M"
  if [ ! -d "$SNAP" ]; then
    echo "error: neither tokenizer/ nor $SNAP found." >&2
    echo "       Copy the snapshot's tokenizer files or the whole $SNAP dir here." >&2
    exit 1
  fi
  echo "[info] staging tokenizer/ from $SNAP"
  mkdir -p tokenizer
  cp "$SNAP"/*.json "$SNAP"/*.model tokenizer/ 2>/dev/null || true
  cp "$SNAP"/*.py tokenizer/ 2>/dev/null || true
  cp "$SNAP"/dict.* "$SNAP"/model.* tokenizer/ 2>/dev/null || true
fi

# --- build for THIS machine's arch (aarch64) -- no --platform, no manifest ---
echo "[info] building $IMAGE:$TAG natively (aarch64) ..."
podman build -t "$IMAGE:$TAG" -f "$CONTAINERFILE" .

echo "[ok] built $IMAGE:$TAG"

if [ "${1:-}" = "--build" ]; then
  echo "     Run it with:  podman run -d --name $CONTAINER_NAME --restart unless-stopped \\"
  echo "                     --cpus 2 --memory 1500m -p 8080:8080 $IMAGE:$TAG"
  exit 0
fi

# --- run it (replace any existing container of the same name) ---
podman rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "[info] starting container '$CONTAINER_NAME' on :8080 ..."
podman run -d --name "$CONTAINER_NAME" --restart unless-stopped \
  --cpus 2 --memory 1500m -p 8080:8080 "$IMAGE:$TAG"

cat <<EOF

[ok] '$CONTAINER_NAME' is starting. The model loads once (~30-60s on first boot);
     /health returns 503 until it is warm, then {"status":"ok"}.

  watch logs:   podman logs -f $CONTAINER_NAME
  health:       curl -s localhost:8080/health
  translate:    curl -s localhost:8080/translate \\
                  -H 'content-type: application/json' \\
                  -d '{"text":"Hello, how are you?","tgt_lang":"hin_Deva"}'
EOF
