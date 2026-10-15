#!/usr/bin/env bash
# Start the Meridian dashboard and the robot digital twin; stop both on exit.
# Usage: ./run_robot_demo.sh [extra robot_sim flags, e.g. --verbose]
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
PORT="${PORT:-5000}"
URL="http://127.0.0.1:${PORT}/api/robot/state"

missing=0
[ -d results/bert_model ] || { echo "ERROR: results/bert_model/ is missing (BERT weights). Train/copy it first."; missing=1; }
ls results/*.joblib >/dev/null 2>&1 || ls results/**/*.joblib >/dev/null 2>&1 || \
  { echo "ERROR: no baseline *.joblib artefacts under results/. Run the baseline training step first."; missing=1; }
if [ "$missing" = 1 ]; then
  echo "The dashboard cannot analyse text without these. You can still run: $PY -m robot_sim --manual"
  exit 1
fi

PYTHONPATH=. "$PY" src/dashboard/app.py > runs_dashboard.log 2>&1 &
DASH_PID=$!
cleanup() {
  kill "$DASH_PID" 2>/dev/null
  pkill -P "$DASH_PID" 2>/dev/null     # Flask debug reloader child
  wait "$DASH_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "Waiting for the dashboard at $URL ..."
for i in $(seq 1 120); do
  if "$PY" - "$URL" <<'PYEOF' 2>/dev/null; then break; fi
import sys, urllib.request
urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1], timeout=0.5)
PYEOF
  if ! kill -0 "$DASH_PID" 2>/dev/null; then
    echo "ERROR: the dashboard exited. Last lines of runs_dashboard.log:"; tail -20 runs_dashboard.log; exit 1
  fi
  sleep 0.5
done
echo "Dashboard: http://127.0.0.1:${PORT}   (log: runs_dashboard.log)"
"$PY" -m robot_sim --url "$URL" "$@"
