#!/usr/bin/env bash
# Build AND run the Marian CT2 serving image NATIVELY on the Raspberry Pi.
#
# Prereqs: copy model_cache_compact_ct2/ from the laptop, e.g.
#   bash deploy/push_to_pi.sh pi@<pi-ip>
# or manually:
#   scp -r model_cache_compact_ct2 pi@<pi-ip>:~/entm/
#
# Usage (on the Pi, from the project root):
#   bash deploy/build_on_pi.sh              # build, then run
#   bash deploy/build_on_pi.sh --build      # build only, don't start
#
# Env overrides: IMAGE, TAG, CONTAINER_NAME, ENGINE, CPUS, MEMORY

set -euo pipefail

IMAGE="${IMAGE:-it2-ct2}"
TAG="${TAG:-latest}"
CONTAINERFILE="deploy/Containerfile"
CONTAINER_NAME="${CONTAINER_NAME:-it2}"

cd "$(dirname "$0")/.."

# --- container engine: podman preferred, docker accepted -------------------
if [ -n "${ENGINE:-}" ]; then
  :
elif command -v podman >/dev/null 2>&1; then
  ENGINE=podman
elif command -v docker >/dev/null 2>&1; then
  ENGINE=docker
else
  echo "error: neither podman nor docker found." >&2
  echo "       sudo apt install -y podman" >&2
  exit 1
fi
echo "[info] container engine: $ENGINE"

# --- sanity: 64-bit only ---------------------------------------------------
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "x86_64" ]; then
  echo "error: architecture '$ARCH' is not supported." >&2
  echo "       ctranslate2 ships no wheels for 32-bit ARM (armv7l/armhf)." >&2
  echo "       Reflash 64-bit Raspberry Pi OS so 'uname -m' reports aarch64." >&2
  exit 1
fi

if [ ! -d model_cache_compact_ct2 ]; then
  echo "error: model_cache_compact_ct2/ not found in $(pwd)." >&2
  echo "       Copy it from the laptop: scp -r model_cache_compact_ct2 pi@<pi>:$(pwd)/" >&2
  exit 1
fi

# --- size the container to the actual board, not a hardcoded 2 cores -------
CORES="$(nproc)"
CPUS="${CPUS:-$CORES}"
MEMORY="${MEMORY:-1500m}"
echo "[info] detected $CORES core(s); running with --cpus $CPUS --memory $MEMORY"

echo "[info] building $IMAGE:$TAG natively ($ARCH) ..."
"$ENGINE" build -t "$IMAGE:$TAG" -f "$CONTAINERFILE" .

echo "[ok] built $IMAGE:$TAG"

if [ "${1:-}" = "--build" ]; then
  cat <<EOF
     Built only (--build). To start it:
       $ENGINE run -d --name $CONTAINER_NAME --restart unless-stopped \\
         --cpus $CPUS --memory $MEMORY \\
         -e CT2_INTRA_THREADS=$CPUS -e OMP_NUM_THREADS=$CPUS \\
         -p 8080:8080 $IMAGE:$TAG
EOF
  exit 0
fi

"$ENGINE" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "[info] starting container '$CONTAINER_NAME' on :8080 ..."
"$ENGINE" run -d --name "$CONTAINER_NAME" --restart unless-stopped \
  --cpus "$CPUS" --memory "$MEMORY" \
  -e CT2_INTER_THREADS=1 -e CT2_INTRA_THREADS="$CPUS" -e OMP_NUM_THREADS="$CPUS" \
  -p 8080:8080 "$IMAGE:$TAG"

cat <<EOF

[ok] '$CONTAINER_NAME' is starting. /health returns 503 until the model is warm.

  watch logs:   $ENGINE logs -f $CONTAINER_NAME
  health:       curl -s localhost:8080/health
  translate:    curl -s -X POST localhost:8080/translate \\
                  -H 'content-type: application/json' \\
                  -d '{"text":"Hello, how are you?"}'
EOF
