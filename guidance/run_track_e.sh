#!/usr/bin/env bash
# One-command, resumable Track E driver. Usage: guidance/run_track_e.sh {status | run --stage e1 | run --stage e2 | run --stage pcgrad}
# Safe to Ctrl+C or suspend at any time; re-run the same command to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/miniconda3/envs/drugdisc/bin:$PATH"
export PYTHONPATH=.
exec python guidance/track_e/run_track_e.py "$@"
