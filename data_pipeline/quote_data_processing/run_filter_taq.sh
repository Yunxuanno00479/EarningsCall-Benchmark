#!/bin/bash
# run_filter_taq.sh
#
# Batch runner for filter_taq_by_sp500.py.
# Iterates over all .gz files under INPUT_BASE and filters each one to keep
# only S&P 500 symbols, writing partitioned Parquet output for each day.
#
# Failed files are logged to ERROR_LOG for reprocessing.
#
# Usage:
#   Set the path variables in the "Configuration" section below, then run:
#       bash run_filter_taq.sh
#
# Prerequisites:
#   - Java 17 available at JAVA_HOME.
#   - A Python environment with PySpark and pandas installed.
#   - Activate the environment before running this script, or set PYTHON below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Configuration: edit these paths before running.
# ---------------------------------------------------------------------------

# Root directory containing raw TAQ NBBO .gz files.
# Expected structure: INPUT_BASE/EQY_US_ALL_NBBO_{YYYY}/.../EQY_US_ALL_NBBO_{YYYYMMDD}.gz
INPUT_BASE="/path/to/raw_taq"

# Root directory for filtered Parquet output (partitioned by symbol).
OUTPUT_BASE="/path/to/output/raw_data"

# Path to S&P 500 company list CSV. Must contain a column named 'tic'.
SP500_CSV="${SCRIPT_DIR}/../../data/sp500_companies.csv"

# Log file for failed .gz files.
ERROR_LOG="${SCRIPT_DIR}/logs/filter_errors.txt"

# Java 17 home directory (required by PySpark).
export JAVA_HOME="/path/to/java17"
export PATH="$JAVA_HOME/bin:$PATH"

# Python interpreter (use the one with PySpark installed).
PYTHON="python3"

# ---------------------------------------------------------------------------
# End of configuration.
# ---------------------------------------------------------------------------

mkdir -p "$(dirname "$ERROR_LOG")"

find "$INPUT_BASE" -type f -name "*.gz" | sort | while read -r gz_file; do
    echo "Processing: $gz_file"
    if "$PYTHON" "${SCRIPT_DIR}/filter_taq_by_sp500.py" \
            "$gz_file" "$OUTPUT_BASE" "$SP500_CSV"; then
        echo "Done: $gz_file"
    else
        echo "FAILED: $gz_file" | tee -a "$ERROR_LOG"
    fi
done

echo "All files processed. Check $ERROR_LOG for any failures."