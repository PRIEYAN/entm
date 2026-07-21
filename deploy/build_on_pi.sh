#!/usr/bin/env bash
# Build AND run the Marian CT2 serving image NATIVELY on the Raspberry Pi.
#
# Prereqs: copy model_cache_compact_ct2/ from the laptop, e.g.
#   scp -r model_cache_compact_ct2 pi@<pi-ip>:~/entm/
#
# Usage (on the Pi, from the project root):
#   bash deploy/build_on_pi.sh
#   bash deploy/build_on_pi.sh --build

set -euo pipefail

IMAGE="${IMAGE:-it2-ct2}"
TAG="${TAG:-latest}"
CONTAINERFILE="deploy/Containerfile"
CONTAINER_NAME="${CONTAINER_NAME:-it2}"

cd "$(dirname "$0")/.."

if [ ! -d model_cache_compact_ct2 ]; then
  echo "error: model_cache_compact_ct2/ not found in $(pwd)." >&2
  echo "       Copy it from the laptop: scp -r model_cache_compact_ct2 pi@<pi>:$(pwd)/" >&2
  exit 1
fi

echo "[info] building $IMAGE:$TAG natively (aarch64) ..."
podman build -t "$IMAGE:$TAG" -f "$CONTAINERFILE" .

echo "[ok] built $IMAGE:$TAG"

if [ "${1:-}" = "--build" ]; then
  echo "     Run:  podman run -d --name $CONTAINER_NAME --restart unless-stopped \\"
  echo "             --cpus 2 --memory 1500m -p 8080:8080 $IMAGE:$TAG"
  exit 0
fi

podman rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "[info] starting container '$CONTAINER_NAME' on :8080 ..."
podman run -d --name "$CONTAINER_NAME" --restart unless-stopped \
  --cpus 2 --memory 1500m -p 8080:8080 "$IMAGE:$TAG"

cat <<EOF

[ok] '$CONTAINER_NAME' is starting. /health returns 503 until the model is warm.

  watch logs:   podman logs -f $CONTAINER_NAME
  health:       curl -s localhost:8080/health
  translate:    curl -s localhost:8080/translate \\
                  -H 'content-type: application/json' \\
                  -d '{"text":"Hello, how are you?"}'
EOF
