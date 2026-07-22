#!/usr/bin/env bash
# Copy the deployable artifacts from this laptop to the Raspberry Pi, verify the
# transfer with a checksum, and (optionally) set up the venv + service remotely.
#
# Usage:
#   bash deploy/push_to_pi.sh pi@192.168.1.42
#   bash deploy/push_to_pi.sh pi@192.168.1.42 --setup     # also install deps
#   REMOTE_DIR=~/entm bash deploy/push_to_pi.sh pi@<ip>
#
# You will be prompted for the Pi's password unless you have an SSH key
# installed (ssh-copy-id pi@<ip> makes this passwordless).

set -euo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "usage: bash deploy/push_to_pi.sh user@host [--setup]" >&2
  exit 1
fi
SETUP="${2:-}"
REMOTE_DIR="${REMOTE_DIR:-entm}"

cd "$(dirname "$0")/.."

if [ ! -d model_cache_compact_ct2 ]; then
  echo "error: model_cache_compact_ct2/ not found. Convert it first:" >&2
  echo "       python -m it2edge.convert.convert_ct2" >&2
  exit 1
fi

echo "[info] target: $TARGET:$REMOTE_DIR"

# --- preflight: is the Pi 64-bit? -----------------------------------------
echo "[info] checking remote architecture ..."
REMOTE_ARCH="$(ssh "$TARGET" 'uname -m' | tr -d '\r')"
echo "[info] remote arch: $REMOTE_ARCH"
if [ "$REMOTE_ARCH" != "aarch64" ]; then
  echo "error: remote reports '$REMOTE_ARCH', not aarch64." >&2
  echo "       ctranslate2 has no 32-bit ARM wheels; pip would try a source" >&2
  echo "       build that takes hours and usually fails." >&2
  echo "       Reflash 64-bit Raspberry Pi OS before deploying." >&2
  exit 1
fi

REMOTE_PY="$(ssh "$TARGET" 'python3 --version' | tr -d '\r')"
echo "[info] remote python: $REMOTE_PY"

# --- copy ------------------------------------------------------------------
echo "[info] creating $REMOTE_DIR on the Pi ..."
ssh "$TARGET" "mkdir -p '$REMOTE_DIR'"

echo "[info] copying model + code (this is ~83 MB, be patient) ..."
if command -v rsync >/dev/null 2>&1; then
  rsync -az --info=progress2 \
    model_cache_compact_ct2 it2edge deploy requirements pyproject.toml \
    "$TARGET:$REMOTE_DIR/"
else
  scp -r model_cache_compact_ct2 it2edge deploy requirements pyproject.toml \
    "$TARGET:$REMOTE_DIR/"
fi

# --- verify ----------------------------------------------------------------
echo "[info] verifying model.bin checksum ..."
LOCAL_SUM="$(sha256sum model_cache_compact_ct2/model.bin | cut -d' ' -f1)"
REMOTE_SUM="$(ssh "$TARGET" "sha256sum '$REMOTE_DIR/model_cache_compact_ct2/model.bin'" | cut -d' ' -f1)"
if [ "$LOCAL_SUM" != "$REMOTE_SUM" ]; then
  echo "error: checksum MISMATCH — the transfer was corrupted." >&2
  echo "       local:  $LOCAL_SUM" >&2
  echo "       remote: $REMOTE_SUM" >&2
  exit 1
fi
echo "[ok] checksum matches: ${LOCAL_SUM:0:16}..."

if [ "$SETUP" != "--setup" ]; then
  cat <<EOF

[ok] Artifacts are on the Pi. Next, on the Pi:

  ssh $TARGET
  cd $REMOTE_DIR
  python3 -m venv venv && source venv/bin/activate
  pip install --upgrade pip && pip install -r requirements/pi.txt
  python -m it2edge.serve.translate_ct2 "Hello, how are you?"

Or re-run this script with --setup to do that automatically.
EOF
  exit 0
fi

# --- remote setup ----------------------------------------------------------
echo "[info] setting up venv + deps on the Pi (several minutes) ..."
ssh "$TARGET" bash -s <<EOF
set -euo pipefail
cd '$REMOTE_DIR'
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements/pi.txt
echo "[remote] smoke test:"
python -m it2edge.serve.translate_ct2 "Hello, how are you?"
EOF

cat <<EOF

[ok] Pi is set up and the model translates.

Start the service on the Pi:
  ssh $TARGET
  cd $REMOTE_DIR && source venv/bin/activate
  CT2_INTRA_THREADS=\$(nproc) OMP_NUM_THREADS=\$(nproc) \\
    uvicorn it2edge.serve.app:app --host 0.0.0.0 --port 8080 --workers 1

Then from this laptop:
  curl -s -X POST http://<pi-ip>:8080/translate \\
    -H 'content-type: application/json' -d '{"text":"Hello, how are you?"}'
EOF
