#!/bin/bash
# run_compute_quote_metrics.sh
#
# Run compute_quote_metrics.py to produce Layer 1 quote panel data.
# Supports resume via checkpoint; re-running this script will skip
# already-completed earnings calls.
#
# Usage:
#   tmux new -s layer1_quote
#   bash run_compute_quote_metrics.sh
#   # After disconnect: tmux attach -t layer1_quote
#
# Adjust the path variables below before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RAW_ROOT="/tmp2/earnings_call/quote_raw_data"
CALENDAR="${SCRIPT_DIR}/../../data/ec_calendar.csv"
OUTPUT_DIR="${SCRIPT_DIR}/../../data/layer1/quote"
WINDOW_DAYS=1

python3 "${SCRIPT_DIR}/compute_quote_metrics.py" \
    --raw_root    "$RAW_ROOT" \
    --calendar    "$CALENDAR" \
    --output_dir  "$OUTPUT_DIR" \
    --window_days "$WINDOW_DAYS"
