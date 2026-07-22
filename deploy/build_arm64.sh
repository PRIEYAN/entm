#!/usr/bin/env bash
# Cross-build the arm64 Marian CT2 serving image on an x86 dev box with Podman.
#
# Usage:
#   REGISTRY=docker.io/youruser deploy/build_arm64.sh
#
# Prereqs: model already converted
#   python -m it2edge.download_model
#   python -m it2edge.train.finetune_marian --data_dir en-indic-exp
#   python -m it2edge.convert.convert_ct2

set -euo pipefail

IMAGE="${IMAGE:-it2-ct2}"
TAG="${TAG:-latest}"
REGISTRY="${REGISTRY:-}"
PLATFORM="linux/arm64"
CONTAINERFILE="deploy/Containerfile"

cd "$(dirname "$0")/.."

if [ ! -d model_cache_compact_ct2 ]; then
  echo "error: model_cache_compact_ct2/ not found. Run it2edge.convert.convert_ct2 first." >&2
  exit 1
fi

if ! command -v podman >/dev/null 2>&1; then
  echo "error: podman not found. This cross-build path uses podman manifests." >&2
  echo "       Either install podman, or build natively on the Pi instead:" >&2
  echo "         bash deploy/push_to_pi.sh pi@<ip> && ssh pi@<ip> 'cd entm && bash deploy/build_on_pi.sh'" >&2
  exit 1
fi

echo "[info] ensuring QEMU arm64 emulation is registered ..."
podman run --rm --privileged docker.io/tonistiigi/binfmt --install arm64 >/dev/null

echo "[info] building $PLATFORM image $IMAGE:$TAG ..."
podman build --platform="$PLATFORM" --manifest "$IMAGE:$TAG" -f "$CONTAINERFILE" .

if [ -z "$REGISTRY" ]; then
  cat <<EOF

[ok] Built manifest $IMAGE:$TAG for $PLATFORM.
     Set REGISTRY and re-run to push, e.g.:
         REGISTRY=docker.io/youruser deploy/build_arm64.sh
EOF
  exit 0
fi

DEST="docker://$REGISTRY/$IMAGE:$TAG"
echo "[info] pushing manifest -> $DEST"
podman manifest push --all "$IMAGE:$TAG" "$DEST"

echo "[ok] Pushed $REGISTRY/$IMAGE:$TAG ($PLATFORM)."
