#!/bin/bash
# run_aggregate_quote_windows.sh
#
# Run aggregate_quote_windows.py to produce Layer 2 quote window features.
# Supports resume via checkpoint; re-running will skip completed earnings calls.
#
# Recommended: run inside a tmux session to guard against SSH disconnection.
#
#   tmux new -s layer2
#   bash run_aggregate_quote_windows.sh
#   # To reattach: tmux attach -t layer2
#
# Edit the path variables below before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Configuration: edit these paths before running.
# ---------------------------------------------------------------------------

# Layer 1 quote panel directory (output of compute_quote_metrics.py).
# Each file: {TIC}_{YEAR}_{QUARTER}.parquet (or .csv for pilot data).
QUOTE_DIR="/path/to/layer1/quote"

# Released sentiment panel root directory (output of merge_sentiment_panels.py).
# Must contain pre/ and qa_score/ subdirectories.
SENTIMENT_DIR="/path/to/released_panel"

# Earnings call calendar CSV.
# Required columns: tic, year, quarter, timestamp_start_et
CALENDAR="${SCRIPT_DIR}/../../data/ec_calendar.csv"

# Output directory for Layer 2 files.
OUTPUT_DIR="${SCRIPT_DIR}/../../data/layer2"

# Anchor types to process: pre, qa, or all.
ANCHOR_TYPE="all"

# Python interpreter.
PYTHON="python3"

# ---------------------------------------------------------------------------
# End of configuration.
# ---------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"

"$PYTHON" "${SCRIPT_DIR}/aggregate_quote_windows.py" \
    --quote_dir     "$QUOTE_DIR" \
    --sentiment_dir "$SENTIMENT_DIR" \
    --calendar      "$CALENDAR" \
    --output_dir    "$OUTPUT_DIR" \
    --anchor_type   "$ANCHOR_TYPE"