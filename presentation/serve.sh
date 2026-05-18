#!/usr/bin/env bash
# Launch the presentation locally and open it in the default browser.
set -e
cd "$(dirname "$0")/.."        # serve from repo root so ../data/... paths resolve
PORT=${PORT:-8765}
URL="http://localhost:${PORT}/presentation/"
echo "→ $URL"
echo "  (Ctrl-C to stop)"
( sleep 0.6 && open "$URL" 2>/dev/null || true ) &
exec python3 -m http.server "$PORT" --bind 127.0.0.1
