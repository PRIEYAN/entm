#!/usr/bin/env bash
# Cross-build the arm64 serving image on an x86 dev box with Podman, then push.
# Plan §7. A cross-arch image can't run on x86 -- you push to a registry and
# pull on the Pi. (Zero-emulation alternative: run this natively on the Pi,
# nothing compiles.)
#
# Usage (can be run from anywhere; it cd's to the project root itself):
#   REGISTRY=docker.io/youruser deploy/build_arm64.sh
#   REGISTRY=ghcr.io/youruser TAG=v1 deploy/build_arm64.sh
#
# Prereqs on the dev box: podman, and the model already converted:
#   python -m it2edge.download_model && python -m it2edge.convert.convert_ct2

set -euo pipefail

IMAGE="${IMAGE:-it2-ct2}"
TAG="${TAG:-latest}"
REGISTRY="${REGISTRY:-}"
PLATFORM="linux/arm64"
CONTAINERFILE="deploy/Containerfile"

# Build context is the PROJECT ROOT (parent of this script's deploy/ dir) so the
# Containerfile can COPY it2edge/, model_cache_ct2/, tokenizer/ and requirements/.
cd "$(dirname "$0")/.."

# --- sanity: the int8 model must exist to bake into the image (plan §7) ---
if [ ! -d model_cache_ct2 ]; then
  echo "error: model_cache_ct2/ not found. Run it2edge.convert.convert_ct2 first." >&2
  exit 1
fi

# The Containerfile COPYs ./tokenizer/. Stage it from the downloaded snapshot so
# the image is self-contained (reuses the exact tokenizer files, plan §6b/§7).
if [ ! -d tokenizer ]; then
  SNAP="model_cache/indictrans2-en-indic-dist-200M"
  if [ ! -d "$SNAP" ]; then
    echo "error: neither tokenizer/ nor $SNAP found. Run it2edge.download_model." >&2
    exit 1
  fi
  echo "[info] staging tokenizer/ from $SNAP"
  mkdir -p tokenizer
  # Tokenizer + remote code files only -- not the multi-hundred-MB weights.
  cp "$SNAP"/*.json "$SNAP"/*.model tokenizer/ 2>/dev/null || true
  cp "$SNAP"/*.py tokenizer/ 2>/dev/null || true
  cp "$SNAP"/dict.* "$SNAP"/model.* tokenizer/ 2>/dev/null || true
fi

# --- one-time: register QEMU emulators for cross-arch builds (plan §7) ---
echo "[info] ensuring QEMU arm64 emulation is registered ..."
podman run --rm --privileged docker.io/tonistiigi/binfmt --install arm64 >/dev/null

# --- build arm64 into a manifest ---
echo "[info] building $PLATFORM image $IMAGE:$TAG ..."
podman build --platform="$PLATFORM" --manifest "$IMAGE:$TAG" -f "$CONTAINERFILE" .

if [ -z "$REGISTRY" ]; then
  cat <<EOF

[ok] Built manifest $IMAGE:$TAG for $PLATFORM.
     This is a cross-arch image -- it will NOT run on this x86 box.
     Set REGISTRY and re-run to push, e.g.:
         REGISTRY=docker.io/youruser deploy/build_arm64.sh
     Then on the Pi:  podman pull \$REGISTRY/$IMAGE:$TAG
EOF
  exit 0
fi

DEST="docker://$REGISTRY/$IMAGE:$TAG"
echo "[info] pushing manifest -> $DEST"
podman manifest push --all "$IMAGE:$TAG" "$DEST"

cat <<EOF

[ok] Pushed $REGISTRY/$IMAGE:$TAG ($PLATFORM).
     On the Pi:
         podman pull $REGISTRY/$IMAGE:$TAG
         # then use the it2.container Quadlet unit (plan §8)
EOF
