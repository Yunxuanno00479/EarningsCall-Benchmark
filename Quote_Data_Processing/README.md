# Layer 0 to Layer 1: Quote Panel Processing

This directory contains the code to transform raw TAQ NBBO quote data (Layer 0)
into a tick-level market microstructure panel (Layer 1).

## Directory Structure

```
layer0_to_layer1/
    filter_taq_by_sp500.py          Step 1: filter raw TAQ .gz files by S&P 500 symbol list
    run_filter_taq.sh               Shell runner for Step 1 (batch over all .gz files)
    compute_quote_metrics.py        Step 2 + 3: compute metrics and produce Layer 1 panel
    run_compute_quote_metrics.sh    Shell runner for Step 2 + 3
    README.md                       This file
```

## Prerequisites

### Step 1 (filter_taq_by_sp500.py)
- Java 17
- PySpark
- pandas

### Step 2 / 3 (compute_quote_metrics.py)
- Python >= 3.10
- pandas >= 2.0
- pyarrow >= 12.0
- numpy

## Data Requirements

| File | Description |
|------|-------------|
| Raw TAQ NBBO .gz files | Daily pipe-delimited quote files from NYSE TAQ |
| `sp500_companies.csv` | S&P 500 company list with a `tic` column |
| `ec_calendar.csv` | Earnings call calendar with columns: `tic`, `year`, `quarter`, `timestamp_start_utc` |

## Step-by-Step Instructions

### Step 1: Filter raw TAQ files by S&P 500 symbols

Edit `run_filter_taq.sh` to set:
- `INPUT_BASE`: directory containing the raw TAQ .gz files
- `OUTPUT_BASE`: output directory for filtered Parquet files
- `SP500_CSV`: path to `sp500_companies.csv`

Then run:
```bash
bash run_filter_taq.sh
```

Output layout:
```
{OUTPUT_BASE}/
    EQY_US_ALL_NBBO_{YYYY}/
        EQY_US_ALL_NBBO_{YYYYMM}/
            EQY_US_ALL_NBBO_{YYYYMMDD}/
                Symbol_Copy={TICKER}/*.parquet
```

### Step 2 + 3: Compute quote metrics and produce Layer 1 panel

Edit `run_compute_quote_metrics.sh` to set:
- `RAW_ROOT`: the `OUTPUT_BASE` from Step 1
- `CALENDAR`: path to `ec_calendar.csv`
- `OUTPUT_DIR`: directory for Layer 1 output files

Then run (inside a tmux session to guard against disconnection):
```bash
tmux new -s layer1_quote
bash run_compute_quote_metrics.sh
```

The script writes one Parquet file per earnings call:
```
{OUTPUT_DIR}/{TICKER}_{YEAR}_{QUARTER}.parquet
```

Progress is saved to `{OUTPUT_DIR}/logs/checkpoint.txt`.
Re-running the script will skip already-completed earnings calls.

## Layer 1 Output Schema

Each row corresponds to one quote event (tick).

| Column | Type | Description |
|--------|------|-------------|
| `ec_id` | str | `{tic}_{year}_{quarter}` |
| `tic` | str | Ticker symbol |
| `year` | int | Earnings call year |
| `quarter` | str | Q1 / Q2 / Q3 / Q4 |
| `ec_timestamp_utc` | str | EC start time (UTC) |
| `ec_timestamp_et` | str | EC start time (ET) |
| `timestamp_utc` | timestamp | Quote event time (UTC) |
| `timestamp_et` | timestamp | Quote event time (ET) |
| `date_label` | str | `pre` / `ec` / `post` (day relative to EC date) |
| `seconds_to_ec` | float | Signed seconds from tick to EC start (negative = before) |
| `session` | int8 | -1=non-trading, 0=pre-market, 1=regular, 2=after-hours |
| `minute_bucket` | timestamp | Floored-to-minute timestamp for minute-level joins |
| `is_open_auction_overlap` | int8 | 1 if 30-min pre-window crosses 09:30 ET open |
| `bid_ask_spread` | float | (ask - bid) / mid |
| `half_spread` | float | (ask - bid) / 2, in dollars |
| `log_spread` | float | log(ask / bid) |
| `bid_depth` | float | Bid size in shares (round lots x 100) |
| `ask_depth` | float | Ask size in shares |
| `total_depth` | float | bid_depth + ask_depth |
| `obi` | float | (bid_sz - ask_sz) / (bid_sz + ask_sz) |
| `depth_ratio` | float | log(bid_size / ask_size) |
| `mid` | float | (bid + ask) / 2 |
| `rv_contrib` | float | (log(mid_i / mid_{i-1}))^2, building block for RV |
| `log_mid_return` | float | log(mid_i / mid_{i-1}) |
| `price_range` | float | (max(mid) - min(mid)) / mean(mid), window-level |
| `sip_latency` | float | SIP dissemination latency in seconds |
| `qrf` | int | Quote Revision Frequency (ticks per minute) |
| `iqd` | float | Inter-Quote Duration mean (seconds per minute) |
| `quote_volatility` | float | std(delta mid) per minute |
| `qar` | float | Quote Arrival Rate (ticks per second, window-level) |
| `liquidity_composite` | float | total_depth / bid_ask_spread |
| `spread_depth_corr` | float | corr(spread, total_depth) within date_label window |
| `amihud_approx` | float | abs(log_mid_return) / total_depth |
| `is_valid_nbbo` | int8 | 1 if National_BBO_Ind == "4" |
| `is_luld` | int8 | 1 if LULD_NBBO_Indicator is non-blank |

## Integration with Other Panels

Layer 1 Quote Panel can be joined to the Sentiment Panel and Trade Panel
at Layer 2 using the composite key `(ec_id, anchor_id, window_type)`.
See `processing/layer1_to_layer2/` for the aggregation step.
