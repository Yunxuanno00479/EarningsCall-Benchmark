#!/bin/bash
# run_compute_quote_metrics.sh
#
# Run compute_quote_metrics.py to produce the Layer 1 quote panel.
# Supports resume via checkpoint: re-running this script will skip
# already-completed earnings calls.
#
# It is recommended to run this inside a tmux session to guard against
# SSH disconnection on long runs:
#
#   tmux new -s layer1_quote
#   bash run_compute_quote_metrics.sh
#   # To reattach after disconnect: tmux attach -t layer1_quote
#
# Usage:
#   Set the path variables in the "Configuration" section below, then run:
#       bash run_compute_quote_metrics.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Configuration: edit these paths before running.
# ---------------------------------------------------------------------------

# Root directory of the filtered Parquet data (output of run_filter_taq.sh).
RAW_ROOT="/path/to/output/raw_data"

# Path to the earnings call calendar CSV.
# Required columns: tic, year, quarter, timestamp_start_utc
CALENDAR="${SCRIPT_DIR}/../../data/ec_calendar.csv"

# Directory for Layer 1 output Parquet files.
OUTPUT_DIR="${SCRIPT_DIR}/../../data/layer1/quote"

# Number of calendar days before and after the EC date to include.
WINDOW_DAYS=1

# Python interpreter.
PYTHON="python3"

# ---------------------------------------------------------------------------
# End of configuration.
# ---------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"

"$PYTHON" "${SCRIPT_DIR}/compute_quote_metrics.py" \
    --raw_root    "$RAW_ROOT" \
    --calendar    "$CALENDAR" \
    --output_dir  "$OUTPUT_DIR" \
    --window_days "$WINDOW_DAYS"