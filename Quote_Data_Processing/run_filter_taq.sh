#!/bin/bash
# run_filter_taq.sh
#
# Batch runner for filter_taq_by_sp500.py.
# Iterates over all .gz files under INPUT_BASE and filters each one.
# Failed files are logged to ERROR_LOG for reprocessing.
#
# Usage:
#   bash run_filter_taq.sh
#
# Prerequisites:
#   - Set JAVA_HOME to a Java 17 installation.
#   - Activate the Python virtual environment that contains PySpark.
#   - Adjust INPUT_BASE, OUTPUT_BASE, SP500_CSV, and ERROR_LOG as needed.

set -euo pipefail

export JAVA_HOME=/tmp2/taq/java17
export PATH=$JAVA_HOME/bin:$PATH
source /tmp2/taq/myenv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_BASE="/tmp2/earnings_call/EQY_US_ALL_NBBO_2021"
OUTPUT_BASE="/tmp2/taq/sp500/raw_data"
SP500_CSV="${SCRIPT_DIR}/../../data/sp500_companies.csv"
ERROR_LOG="${SCRIPT_DIR}/logs/filter_errors.txt"

mkdir -p "$(dirname "$ERROR_LOG")"

find "$INPUT_BASE" -type f -name "*.gz" | sort | while read -r gz_file; do
    echo "Processing: $gz_file"
    if python3 "${SCRIPT_DIR}/filter_taq_by_sp500.py" \
            "$gz_file" "$OUTPUT_BASE" "$SP500_CSV"; then
        echo "Done: $gz_file"
    else
        echo "Failed: $gz_file"
        echo "$gz_file" >> "$ERROR_LOG"
    fi
done
