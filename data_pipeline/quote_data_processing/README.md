# Quote Data Processing: Layer 0 to Layer 1

This directory transforms raw TAQ NBBO quote data (Layer 0) into a
tick-level market microstructure panel (Layer 1).

## Directory Structure

```
quote_data_processing/
    filter_taq_by_sp500.py          Step 1: filter raw TAQ .gz files by S&P 500 symbol list
    run_filter_taq.sh               Runner for Step 1 (batch over all .gz files)
    compute_quote_metrics.py        Step 2: compute metrics and produce Layer 1 panel
    run_compute_quote_metrics.sh    Runner for Step 2
    README.md                       This file
```

## Prerequisites

### Step 1 (filter_taq_by_sp500.py)

- Java 17
- Python >= 3.10
- PySpark
- pandas

### Step 2 (compute_quote_metrics.py)

- Python >= 3.10
- pandas >= 2.0
- pyarrow >= 12.0
- numpy
- pytz

Install dependencies:

```bash
pip install -r ../../requirements.txt
```

## Data Requirements

| File | Description |
|------|-------------|
| Raw TAQ NBBO .gz files | Daily pipe-delimited NBBO quote files from NYSE TAQ |
| `sp500_companies.csv` | S&P 500 company list; must contain a `tic` column |
| `ec_calendar.csv` | Earnings call calendar; required columns: `tic`, `year`, `quarter`, `timestamp_start_utc` |

## Step-by-Step Instructions

### Step 1: Filter raw TAQ files by S&P 500 symbols

Edit `run_filter_taq.sh` to set:
- `INPUT_BASE`: directory containing the raw TAQ .gz files
- `OUTPUT_BASE`: output directory for filtered Parquet files
- `SP500_CSV`: path to `sp500_companies.csv`
- `JAVA_HOME`: path to a Java 17 installation

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
                Symbol_Copy={TICKER}/
                    *.parquet
```

Failed files are logged to `logs/filter_errors.txt`.

### Step 2: Compute quote metrics and produce Layer 1 panel

Edit `run_compute_quote_metrics.sh` to set:
- `RAW_ROOT`: the `OUTPUT_BASE` from Step 1
- `CALENDAR`: path to `ec_calendar.csv`
- `OUTPUT_DIR`: directory for Layer 1 output files

Run inside a tmux session to guard against SSH disconnection:

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

### Identity

| Column | Type | Description |
|--------|------|-------------|
| `ec_id` | str | `{tic}_{year}_{quarter}` |
| `tic` | str | Ticker symbol |
| `year` | int | Earnings call year |
| `quarter` | str | Q1 / Q2 / Q3 / Q4 |
| `ec_timestamp_utc` | str | EC start time (UTC) |
| `ec_timestamp_et` | str | EC start time (ET) |

### Time

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_utc` | timestamp | Quote event time (UTC) |
| `timestamp_et` | timestamp | Quote event time (ET) |
| `date_label` | str | `pre` / `ec` / `post` (day relative to EC date) |
| `seconds_to_ec` | float | Signed seconds from tick to EC start; negative = before EC |
| `session` | int8 | -1 = non-trading, 0 = pre-market, 1 = regular, 2 = after-hours |
| `minute_bucket` | timestamp | Floored-to-minute timestamp for minute-level joins |
| `is_open_auction_overlap` | int8 | 1 if the 30-min pre-EC window crosses the 09:30 ET open |

### Group 1: Spread

| Column | Type | Description |
|--------|------|-------------|
| `bid_ask_spread` | float | (ask - bid) / mid |
| `half_spread` | float | (ask - bid) / 2, in dollars |
| `log_spread` | float | log(ask / bid) |

### Group 2: Depth

| Column | Type | Description |
|--------|------|-------------|
| `bid_depth` | float | Bid size in shares (round lots x 100) |
| `ask_depth` | float | Ask size in shares |
| `total_depth` | float | bid_depth + ask_depth |
| `obi` | float | Order book imbalance: (bid_sz - ask_sz) / (bid_sz + ask_sz) |
| `depth_ratio` | float | log(bid_size / ask_size) |

### Group 3: Mid-quote

| Column | Type | Description |
|--------|------|-------------|
| `mid` | float | (bid + ask) / 2 |
| `rv_contrib` | float | (log mid return)^2; building block for realized variance |
| `log_mid_return` | float | log(mid_i / mid_{i-1}) |
| `price_range` | float | (max(mid) - min(mid)) / mean(mid), computed per date_label window |

### Group 4: Quote Dynamics

| Column | Granularity | Description |
|--------|-------------|-------------|
| `sip_latency` | tick | SIP dissemination latency in seconds |
| `qrf` | minute | Quote Revision Frequency: tick count per minute |
| `iqd` | minute | Inter-Quote Duration: mean seconds between consecutive ticks |
| `quote_volatility` | minute | Standard deviation of mid-quote changes within the minute |
| `qar` | window | Quote Arrival Rate: ticks per second over the full date_label window |

### Group 5: Composite Liquidity

| Column | Type | Description |
|--------|------|-------------|
| `liquidity_composite` | float | total_depth / bid_ask_spread |
| `spread_depth_corr` | float | Pearson correlation between bid_ask_spread and total_depth within the date_label window (Lee, Mucklow & Ready, 1993) |
| `amihud_approx` | float | abs(log_mid_return) / total_depth |

### Group 6: Condition Codes

| Column | Type | Description |
|--------|------|-------------|
| `is_valid_nbbo` | int8 | 1 if National_BBO_Ind == "4" |
| `is_luld` | int8 | 1 if LULD_NBBO_Indicator is non-blank |

## Integration with Other Panels

The Layer 1 quote panel can be joined to the sentiment panel and trade panel
at Layer 2 using the composite key `(ec_id, anchor_id, window_type)`.
See `data_pipeline/layer1_to_layer2/` for the aggregation step.