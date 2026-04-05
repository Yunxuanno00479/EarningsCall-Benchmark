# Layer 2: Quote Window Aggregation and Benchmark Dataset Construction

This directory produces the final benchmark dataset from tick-level quote
metrics (Layer 1) and the released sentiment panel.

Two scripts are run in sequence:

    aggregate_quote_windows.py      Step 1: aggregate quote metrics into anchor windows
    build_benchmark_dataset.py      Step 2: annotate sessions, join sentiment, filter anchors

## Directory Structure

```
layer2/
    aggregate_quote_windows.py          Step 1
    run_aggregate_quote_windows.sh      Runner for Step 1
    build_benchmark_dataset.py          Step 2
    run_build_benchmark_dataset.sh      Runner for Step 2
    README.md                           This file
```

## Prerequisites

```bash
pip install pandas>=2.0 numpy pytz pyarrow>=12.0
```

## Step 1: Aggregate quote metrics into anchor windows

For each earnings call anchor (presentation sentence or QA pair), compute
aggregate statistics from the tick-level quote panel over fixed pre- and
post-windows.

Anchor types:
- `pre`: each sentence in the Presentation section.
  Anchor timestamp = `timestamp_p` (seconds from EC start, t=0).
- `qa`: each QA pair.
  Anchor timestamp = `Q_Timestamp` (question start, seconds from EC start).

Window design:
```
|<---- pre-window (fixed 30s) ----|---- post-window (variable) ---->|
t - 30s                    timestamp_anchor              t + {30,60,120,300}s
```

Each anchor produces four output rows (one per post-window width).

```bash
bash run_aggregate_quote_windows.sh
```

Output: one CSV per earnings call in `{OUTPUT_DIR}/`:
```
{TIC}_{YEAR}_{QUARTER}_layer2.csv
```

Run inside tmux to guard against SSH disconnection. Progress is saved to
`{OUTPUT_DIR}/logs/checkpoint.txt`; re-running skips completed calls.

## Step 2: Build benchmark dataset

Reads all `*_layer2.csv` files, then:

1. Annotates each EC with its trading session based on EC start time (ET):

   | Session | Hours (ET) | Typical quote density |
   |---------|------------|-----------------------|
   | `pre_market` | 04:00 - 09:29 | Low |
   | `regular` | 09:30 - 15:59 | High |
   | `after_hours` | 16:00 - 19:59 | Low |
   | `non_trading` | other | Very low |

   In this dataset: ~44% pre_market, ~30% after_hours, ~26% regular.
   After_hours and pre_market ECs have systematically fewer ticks in each
   window. The `ec_session` column allows downstream users to stratify or
   filter by session.

2. Joins sentiment features from the released panel:
   - `pre` anchors: FinBERT-tone scores from `released_panel/pre/`
   - `qa` anchors: SubjECTive-QA scores from `released_panel/qa_score/`

3. Filters anchors where `n_ticks_post < --min_ticks_post`.
   Excluded anchors are written to `excluded_anchors_{W}s.csv` for
   transparency. The default threshold is 3 ticks.

```bash
bash run_build_benchmark_dataset.sh
```

Output files in `{OUTPUT_DIR}/`:

```
benchmark_pre_30s.csv       Presentation anchors, post-window = 30s
benchmark_pre_60s.csv       Presentation anchors, post-window = 60s
benchmark_pre_120s.csv      Presentation anchors, post-window = 120s
benchmark_pre_300s.csv      Presentation anchors, post-window = 300s
benchmark_qa_30s.csv        QA-pair anchors, post-window = 30s
benchmark_qa_60s.csv        QA-pair anchors, post-window = 60s
benchmark_qa_120s.csv       QA-pair anchors, post-window = 120s
benchmark_qa_300s.csv       QA-pair anchors, post-window = 300s
excluded_anchors_30s.csv    Anchors excluded at post=30s (n_ticks_post < threshold)
...
benchmark_coverage.csv      Per-EC counts of valid and excluded anchors
logs/
    build_benchmark_dataset.log
```

## Output Schema

### Benchmark files (benchmark_{pre|qa}_{W}s.csv)

#### Identity columns

| Column | Description |
|--------|-------------|
| `tic` | Ticker symbol |
| `year` | Earnings call year |
| `quarter` | Quarter (Q1 / Q2 / Q3 / Q4) |
| `anchor_type` | `pre` or `qa` |
| `anchor_id` | `section_id` for pre; `qa_index` for qa |
| `post_window_sec` | Post-window width in seconds |
| `timestamp_anchor` | Seconds from EC start to anchor |
| `ec_session` | `pre_market` / `regular` / `after_hours` / `non_trading` |

#### Pre-window quote features (suffix: _pre)

Computed over ticks in `[timestamp_anchor - 30, timestamp_anchor)`.

| Column | Description |
|--------|-------------|
| `bid_ask_spread_mean_pre` | Mean relative bid-ask spread |
| `bid_ask_spread_std_pre` | Std of bid-ask spread |
| `obi_mean_pre` | Mean order book imbalance |
| `total_depth_mean_pre` | Mean total quoted depth (shares) |
| `qrf_mean_pre` | Mean quote revision frequency (ticks/min) |
| `quote_volatility_mean_pre` | Mean quote volatility (std of mid changes/min) |
| `n_ticks_pre` | Number of quote ticks in the pre-window |

#### Post-window quote features (suffix: _post)

Same columns as pre-window with `_post` suffix. Computed over
`[timestamp_anchor, timestamp_anchor + post_window_sec)`.

#### Sentiment features (pre anchors)

| Column | Description |
|--------|-------------|
| `section_id` | Sentence index within the earnings call |
| `timestamp_p` | Sentence start time (seconds from EC start) |
| `section` | Always `Pre` |
| `finberttone_expected_value` | P(Positive) - P(Negative) for this sentence |
| `finberttone_cumulative_tone` | Running sum of expected_value up to this sentence |
| `finberttone_change_point` | 1 if this sentence starts a new sentiment regime (PELT+AIC); 0 otherwise |

#### Sentiment features (qa anchors)

| Column | Description |
|--------|-------------|
| `qa_index` | QA pair index within the earnings call |
| `Q_Timestamp` | Question start time (seconds from EC start) |
| `A_Timestamp` | Answer start time (seconds from EC start) |
| `{dim}_negative_score` | P(class 0) for dimension `dim` |
| `{dim}_neutral_score` | P(class 1) for dimension `dim` |
| `{dim}_positive_score` | P(class 2) for dimension `dim` |

Where `dim` is one of: `assertive`, `cautious`, `optimistic`, `specific`,
`clear`, `relevant`.

### Coverage file (benchmark_coverage.csv)

| Column | Description |
|--------|-------------|
| `tic`, `year`, `quarter` | EC identifier |
| `ec_session` | Session label |
| `n_valid_30s` | Valid anchors at post=30s |
| `n_excluded_30s` | Excluded anchors at post=30s |

## Reproducibility Notes

The `ec_session` classification depends only on the EC start time from the
calendar CSV. Given the same calendar and the same Layer 2 output, the
benchmark dataset is deterministically reproducible.

The `--min_ticks_post` threshold is the only free parameter. The default
value of 3 is reported in the benchmark paper. To replicate exactly, use:

```bash
python build_benchmark_dataset.py --min_ticks_post 3 ...
```

## References

- Lee, C. M. C., Mucklow, B., and Ready, M. J. (1993). Spreads, depths,
  and the impact of earnings information: An intraday analysis. Review of
  Financial Studies, 6(2), 345-374.

- Beaver, W. H. (1968). The information content of annual earnings
  announcements. Journal of Accounting Research, 6, 67-92.

- Campbell, C. J. and Wasley, C. E. (1996). Measuring abnormal daily
  trading volume for samples of NYSE/ASE and NASDAQ securities using
  parametric and nonparametric test statistics. Review of Quantitative
  Finance and Accounting, 6(3), 309-326.

## Step 3: Train/test split

Applies session filtering and time-based train/test split to the benchmark
files produced in Step 2.

```bash
python split_benchmark_dataset.py \
    --benchmark_dir  /path/to/benchmark \
    --output_dir     /path/to/benchmark_split \
    --train_years    2021 2022 \
    --test_years     2023 \
    --main_sessions  regular after_hours
```

### Split design

A time-based split is used to prevent look-ahead bias:

```
Train : 2021 Q1 - 2022 Q4
Test  : 2023 Q1 - 2023 Q3
```

### Session filtering

Pre_market ECs (04:00-09:29 ET) are excluded from the main benchmark because
NBBO quote data is systematically sparse before market open, making the
post-window microstructure response unreliable as a prediction target.

| Subset | Sessions included | Destination |
|--------|------------------|-------------|
| Main benchmark | `regular`, `after_hours` | `train/`, `test/` |
| Robustness check | `pre_market` | `robustness/` |
| Excluded | `non_trading` | not written |

### Output layout

```
benchmark_split/
    train/
        benchmark_pre_{W}s.csv      Presentation anchors, training set
        benchmark_qa_{W}s.csv       QA-pair anchors, training set
    test/
        benchmark_pre_{W}s.csv      Presentation anchors, test set
        benchmark_qa_{W}s.csv       QA-pair anchors, test set
    robustness/
        pre_market_pre_{W}s.csv     Pre_market ECs (supplementary analysis)
        pre_market_qa_{W}s.csv
    split_summary.csv               Anchor and EC counts per split
    logs/
        split_benchmark_dataset.log
```

### Expected scale (post_window=30s, min_ticks_post=3)

| Split | Pre anchors | QA anchors | ECs |
|-------|-------------|-----------|-----|
| Train (2021-2022) | ~90,000 | ~11,800 | ~720 |
| Test (2023) | ~51,000 | ~3,600 | ~389 |
| Robustness (pre_market) | ~37,000 | ~7,000 | ~747 |