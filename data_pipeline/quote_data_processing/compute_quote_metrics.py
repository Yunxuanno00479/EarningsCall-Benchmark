"""
compute_quote_metrics.py

Compute tick-level market microstructure metrics from filtered TAQ NBBO Parquet
files and produce a per-earnings-call panel dataset (Layer 1).

Each output file covers the earnings call date plus the preceding and following
calendar day, providing sufficient context for pre/post window construction.

Input layout (output of filter_taq_by_sp500.py after repartitioning by year):
    {raw_root}/
        EQY_US_ALL_NBBO_{YYYY}/
            EQY_US_ALL_NBBO_{YYYYMM}/
                EQY_US_ALL_NBBO_{YYYYMMDD}/
                    Symbol_Copy={TICKER}/*.parquet

Output layout:
    {output_dir}/{TICKER}_{YYYY}_{QUARTER}.parquet

Each output row corresponds to one quote event (tick). Minute-level metrics
(QRF, IQD, Quote_Volatility) are aggregated per minute and left-joined back
onto the tick-level rows within each minute bucket.

Column groups:
    Identity       : ec_id, tic, year, quarter, ec_timestamp_utc, ec_timestamp_et
    Time           : timestamp_utc, timestamp_et, date_label, seconds_to_ec,
                     session, minute_bucket
    Session flag   : is_open_auction_overlap
    Spread (G1)    : bid_ask_spread, half_spread, log_spread
    Depth (G2)     : bid_depth, ask_depth, total_depth, obi, depth_ratio
    Mid (G3)       : mid, rv_contrib, log_mid_return, price_range
    Quote dynamics : qrf, qar, iqd, sip_latency, quote_volatility  (G4)
    Composite (G5) : liquidity_composite, spread_depth_corr, amihud_approx
    Condition (G6) : is_valid_nbbo, is_luld

Usage:
    python compute_quote_metrics.py \\
        --raw_root   /path/to/raw_data \\
        --calendar   /path/to/ec_calendar.csv \\
        --output_dir /path/to/layer1/quote \\
        [--window_days 1]

Requirements:
    pandas >= 2.0, pyarrow >= 12.0, numpy
"""

import argparse
import glob
import logging
import os
import sys
from datetime import time as dtime, timedelta

import numpy as np
import pandas as pd
import pytz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ET_TZ = pytz.timezone("America/New_York")

# Session codes for each tick.
SESSION_BEFORE_PRE  = -1   # before 04:00 ET (non-trading)
SESSION_PRE_MARKET  =  0   # 04:00 - 09:30 ET
SESSION_REGULAR     =  1   # 09:30 - 16:00 ET
SESSION_AFTER_HOURS =  2   # 16:00 - 20:00 ET

# Minimum number of ticks required in a window to compute correlation.
MIN_TICKS_FOR_CORR = 3


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> logging.Logger:
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> set:
    """Return the set of already-completed keys from a checkpoint file."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def write_checkpoint(path: str, key: str) -> None:
    with open(path, "a") as f:
        f.write(key + "\n")


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw_parquet(raw_root: str, tic: str, date_str: str) -> "pd.DataFrame | None":
    """
    Load all Parquet files for a given ticker on a given date.

    Args:
        raw_root  : Root directory of the raw partitioned Parquet data.
        tic       : Ticker symbol (e.g., "AAPL").
        date_str  : Date string in YYYYMMDD format.

    Returns:
        DataFrame with raw quote data, or None if no files are found.
    """
    year = date_str[:4]
    ym   = date_str[:6]
    base = os.path.join(
        raw_root,
        f"EQY_US_ALL_NBBO_{year}",
        f"EQY_US_ALL_NBBO_{ym}",
        f"EQY_US_ALL_NBBO_{date_str}",
        f"Symbol_Copy={tic}",
    )
    files = glob.glob(os.path.join(base, "*.parquet")) + \
            glob.glob(os.path.join(base, "*.snappy.parquet"))
    files = list(set(files))
    if not files:
        return None

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        # The Date field is not stored inside the Parquet file;
        # it is derived from the directory name.
        df["Date"] = date_str
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the TAQ Time field and the Date field into timezone-aware timestamps.

    TAQ Time format: HHMMSSxxxxxxxxx (15 characters).
    The last 9 characters are nanoseconds; only the first 6 are used (microseconds).
    Date format: YYYYMMDD (string, injected from directory name).
    """
    date_s = df["Date"].astype(str).str.zfill(8)
    time_s = df["Time"].astype(str).str.zfill(15)

    dt_str = (
        date_s.str[:4] + "-" + date_s.str[4:6] + "-" + date_s.str[6:8] + " "
        + time_s.str[:2] + ":" + time_s.str[2:4] + ":" + time_s.str[4:6]
        + "." + time_s.str[6:12]
    )
    df["timestamp_et"] = pd.to_datetime(dt_str).dt.tz_localize(
        ET_TZ, ambiguous="infer", nonexistent="shift_forward"
    )
    df["timestamp_utc"] = df["timestamp_et"].dt.tz_convert("UTC")
    return df


def parse_sip_latency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SIP latency in seconds as the difference between the TAQ
    dissemination timestamp (Time) and the exchange Participant_Timestamp.

    Rows with missing or zero Participant_Timestamp receive NaN.
    """
    if "Participant_Timestamp" not in df.columns:
        df["sip_latency"] = np.nan
        return df

    date_s = df["Date"].astype(str).str.zfill(8)
    pt_s   = df["Participant_Timestamp"].astype(str).str.zfill(15)
    valid  = pt_s.str.strip() != "000000000000000"

    pt_dt_str = (
        date_s.str[:4] + "-" + date_s.str[4:6] + "-" + date_s.str[6:8] + " "
        + pt_s.str[:2] + ":" + pt_s.str[2:4] + ":" + pt_s.str[4:6]
        + "." + pt_s.str[6:12]
    )
    pt_parsed = pd.to_datetime(pt_dt_str, errors="coerce").dt.tz_localize(
        ET_TZ, ambiguous="infer", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")

    latency = (df["timestamp_utc"] - pt_parsed).dt.total_seconds()
    latency[~valid] = np.nan
    df["sip_latency"] = latency
    return df


# ---------------------------------------------------------------------------
# Session classification
# ---------------------------------------------------------------------------

def assign_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a session code to each tick based on Eastern Time.

    Codes:
        -1 : outside trading hours (before 04:00 or after 20:00 ET)
         0 : pre-market  (04:00 - 09:30 ET)
         1 : regular     (09:30 - 16:00 ET)
         2 : after-hours (16:00 - 20:00 ET)
    """
    t               = df["timestamp_et"].dt.time
    pre_open_time   = dtime(4,  0)
    open_time       = dtime(9, 30)
    close_time      = dtime(16, 0)
    post_close_time = dtime(20, 0)

    session = np.full(len(df), SESSION_BEFORE_PRE, dtype=np.int8)
    session = np.where((t >= pre_open_time)  & (t < open_time),       SESSION_PRE_MARKET,  session)
    session = np.where((t >= open_time)      & (t < close_time),      SESSION_REGULAR,     session)
    session = np.where((t >= close_time)     & (t < post_close_time), SESSION_AFTER_HOURS, session)
    df["session"] = session.astype(np.int8)
    return df


# ---------------------------------------------------------------------------
# Tick-level metric computation
# ---------------------------------------------------------------------------

def compute_tick_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all tick-level market microstructure metrics.
    The DataFrame must already contain timestamp_utc after parse_timestamps().
    """
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    bid = df["Bid_Price"].astype(float)
    ask = df["Offer_Price"].astype(float)
    bid_sz = df["Bid_Size"].astype(float)
    ask_sz = df["Offer_Size"].astype(float)
    mid    = (bid + ask) / 2.0

    # --- Group 1: Spread ---
    df["mid"]            = mid
    df["bid_ask_spread"] = np.where(mid > 0, (ask - bid) / mid, np.nan)
    df["half_spread"]    = (ask - bid) / 2.0
    df["log_spread"]     = np.where(
        (ask > 0) & (bid > 0), np.log(ask / bid), np.nan
    )

    # --- Group 2: Depth ---
    df["bid_depth"]   = bid_sz * 100.0
    df["ask_depth"]   = ask_sz * 100.0
    df["total_depth"] = (bid_sz + ask_sz) * 100.0
    total_sz = bid_sz + ask_sz
    df["obi"] = np.where(
        total_sz > 0, (bid_sz - ask_sz) / total_sz, np.nan
    )
    df["depth_ratio"] = np.where(
        (bid_sz > 0) & (ask_sz > 0), np.log(bid_sz / ask_sz), np.nan
    )

    # --- Group 3: Mid-quote derivatives ---
    log_mid             = np.log(mid.clip(lower=1e-10))
    log_mid_ret         = log_mid - log_mid.shift(1)
    df["log_mid_return"] = log_mid_ret
    df["rv_contrib"]     = log_mid_ret ** 2   # first row is NaN

    # --- Group 5: Composite liquidity (tick-level components) ---
    df["liquidity_composite"] = np.where(
        df["bid_ask_spread"] > 0,
        df["total_depth"] / df["bid_ask_spread"],
        np.nan,
    )
    df["amihud_approx"] = np.where(
        df["total_depth"] > 0,
        log_mid_ret.abs() / df["total_depth"],
        np.nan,
    )

    # --- Group 6: Condition codes ---
    # National_BBO_Ind == "4" identifies a valid NBBO quote.
    if "National_BBO_Ind" in df.columns:
        df["is_valid_nbbo"] = (
            df["National_BBO_Ind"].astype(str).str.strip() == "4"
        ).astype(np.int8)
    else:
        df["is_valid_nbbo"] = np.int8(-1)

    # Non-blank LULD_NBBO_Indicator means the quote is under LULD constraint.
    if "LULD_NBBO_Indicator" in df.columns:
        df["is_luld"] = (
            df["LULD_NBBO_Indicator"].astype(str).str.strip() != ""
        ).astype(np.int8)
    else:
        df["is_luld"] = np.int8(-1)

    return df


# ---------------------------------------------------------------------------
# Minute-level metric computation
# ---------------------------------------------------------------------------

def compute_minute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-minute aggregates and left-join them onto the tick-level rows.

    Metrics:
        qrf             : Quote Revision Frequency (count of ticks per minute).
        iqd             : Inter-Quote Duration, mean seconds between consecutive
                          ticks within the minute.
        quote_volatility: Standard deviation of mid-quote changes within the minute.
    """
    df["minute_bucket"] = df["timestamp_utc"].dt.floor("min")
    grp = df.groupby("minute_bucket")

    qrf = grp.size().rename("qrf")

    def mean_iqd(g):
        diffs = g["timestamp_utc"].sort_values().diff().dt.total_seconds().dropna()
        return diffs.mean() if len(diffs) > 0 else np.nan

    iqd  = grp.apply(mean_iqd).rename("iqd")

    def quote_vol(g):
        return g.sort_values("timestamp_utc")["mid"].diff().std()

    qvol = grp.apply(quote_vol).rename("quote_volatility")

    minute_df = pd.concat([qrf, iqd, qvol], axis=1).reset_index()
    df = df.merge(minute_df, on="minute_bucket", how="left")
    return df


# ---------------------------------------------------------------------------
# Window-level metric computation
# ---------------------------------------------------------------------------

def compute_window_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-date-label window aggregates and broadcast them onto every
    tick within the corresponding window.

    Window metrics:
        price_range       : (max(mid) - min(mid)) / mean(mid)
        qar               : Quote Arrival Rate = tick count / window duration (s)
        spread_depth_corr : Pearson correlation between bid_ask_spread and
                            total_depth within the window (Lee, Mucklow & Ready 1993).
    """
    records = []
    for label, grp in df.groupby("date_label"):
        grp_s = grp.sort_values("timestamp_utc")
        m = grp_s["mid"]

        price_range = (
            (m.max() - m.min()) / m.mean()
            if m.mean() > 0 else np.nan
        )

        duration = (
            grp_s["timestamp_utc"].max() - grp_s["timestamp_utc"].min()
        ).total_seconds()
        qar = len(grp_s) / duration if duration > 0 else np.nan

        corr = (
            grp_s["bid_ask_spread"].corr(grp_s["total_depth"])
            if len(grp_s) >= MIN_TICKS_FOR_CORR else np.nan
        )

        records.append({
            "date_label":        label,
            "price_range":       price_range,
            "qar":               qar,
            "spread_depth_corr": corr,
        })

    window_df = pd.DataFrame(records)
    return df.merge(window_df, on="date_label", how="left")


# ---------------------------------------------------------------------------
# EC-level derived fields
# ---------------------------------------------------------------------------

def assign_ec_fields(
    df: pd.DataFrame,
    ec_ts_utc: pd.Timestamp,
    tic: str,
    year: int,
    quarter: str,
) -> pd.DataFrame:
    """
    Add identity and event-relative fields that are constant for all ticks
    belonging to the same earnings call.

    Fields added:
        ec_id                  : "{tic}_{year}_{quarter}"
        tic, year, quarter
        ec_timestamp_utc       : earnings call start time (UTC)
        ec_timestamp_et        : earnings call start time (ET)
        seconds_to_ec          : signed seconds from tick timestamp to EC start
                                 (negative = before EC, positive = after EC)
        is_open_auction_overlap: 1 if the 30-minute pre-EC window crosses 09:30 ET
    """
    ec_ts_et             = ec_ts_utc.astimezone(ET_TZ)
    pre_window_start_et  = (ec_ts_utc - timedelta(minutes=30)).astimezone(ET_TZ)
    market_open_et       = ET_TZ.localize(
        ec_ts_et.replace(hour=9, minute=30, second=0, microsecond=0, tzinfo=None)
    )
    is_overlap = int(pre_window_start_et < market_open_et)

    df["seconds_to_ec"]           = (df["timestamp_utc"] - ec_ts_utc).dt.total_seconds()
    df["ec_id"]                   = f"{tic}_{year}_{quarter}"
    df["tic"]                     = tic
    df["year"]                    = year
    df["quarter"]                 = quarter
    df["ec_timestamp_utc"]        = str(ec_ts_utc)
    df["ec_timestamp_et"]         = str(ec_ts_et)
    df["is_open_auction_overlap"] = np.int8(is_overlap)
    return df


# ---------------------------------------------------------------------------
# Main per-EC processing
# ---------------------------------------------------------------------------

def process_earnings_call(
    raw_root: str,
    tic: str,
    year: int,
    quarter: str,
    ec_ts_utc: pd.Timestamp,
    window_days: int,
    log: logging.Logger,
) -> "pd.DataFrame | None":
    """
    Load raw quote data for the EC date window, compute all metrics,
    and return a single tick-level DataFrame ready for output.
    """
    frames = []
    for delta, label in [(-1, "pre"), (0, "ec"), (1, "post")]:
        date_str = (ec_ts_utc + timedelta(days=delta)).strftime("%Y%m%d")
        raw = load_raw_parquet(raw_root, tic, date_str)
        if raw is None or raw.empty:
            log.warning("No data: %s %s (%s)", tic, date_str, label)
            continue
        raw["date_label"] = label
        frames.append(raw)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = parse_timestamps(df)
    df = assign_session(df)
    df = compute_tick_metrics(df)
    df = parse_sip_latency(df)
    df = compute_minute_metrics(df)
    df = compute_window_metrics(df)
    df = assign_ec_fields(df, ec_ts_utc, tic, year, quarter)
    return df


# ---------------------------------------------------------------------------
# Output column ordering
# ---------------------------------------------------------------------------

LAYER1_COLUMNS = [
    # Identity
    "ec_id", "tic", "year", "quarter",
    "ec_timestamp_utc", "ec_timestamp_et",
    # Time
    "timestamp_utc", "timestamp_et",
    "date_label", "seconds_to_ec",
    "session", "minute_bucket",
    # EC-level flags
    "is_open_auction_overlap",
    # Group 1: Spread
    "bid_ask_spread", "half_spread", "log_spread",
    # Group 2: Depth
    "bid_depth", "ask_depth", "total_depth", "obi", "depth_ratio",
    # Group 3: Mid-quote
    "mid", "rv_contrib", "log_mid_return", "price_range",
    # Group 4: Quote dynamics (tick-level)
    "sip_latency",
    # Group 4: Quote dynamics (minute-level, left-joined)
    "qrf", "iqd", "quote_volatility",
    # Group 4: Quote dynamics (window-level)
    "qar",
    # Group 5: Composite liquidity
    "liquidity_composite", "spread_depth_corr", "amihud_approx",
    # Group 6: Condition codes
    "is_valid_nbbo", "is_luld",
]


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in LAYER1_COLUMNS if c in df.columns]
    return df[present]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute tick-level quote metrics for earnings call windows (Layer 1)."
    )
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Root directory of the filtered TAQ Parquet data.",
    )
    parser.add_argument(
        "--calendar",
        required=True,
        help="Path to the earnings call calendar CSV (must contain: "
             "tic, year, quarter, timestamp_start_utc).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for Layer 1 output Parquet files.",
    )
    parser.add_argument(
        "--window_days",
        type=int,
        default=1,
        help="Number of calendar days before and after the EC date to include "
             "(default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_path   = os.path.join(args.output_dir, "logs", "compute_quote_metrics.log")
    ckpt_path  = os.path.join(args.output_dir, "logs", "checkpoint.txt")
    log        = setup_logging(log_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load earnings call calendar.
    calendar = pd.read_csv(args.calendar, parse_dates=["timestamp_start_utc"])
    if calendar["timestamp_start_utc"].dt.tz is None:
        calendar["timestamp_start_utc"] = calendar[
            "timestamp_start_utc"
        ].dt.tz_localize("UTC")
    else:
        calendar["timestamp_start_utc"] = calendar[
            "timestamp_start_utc"
        ].dt.tz_convert("UTC")

    done      = load_checkpoint(ckpt_path)
    total     = len(calendar)
    processed = 0

    for idx, row in calendar.iterrows():
        tic     = row["tic"]
        year    = int(row["year"])
        quarter = str(row["quarter"])
        ec_ts   = row["timestamp_start_utc"]
        key     = f"{tic}|{year}|{quarter}"

        if key in done:
            continue

        log.info("[%d/%d] %s %d %s  (%s)", idx + 1, total, tic, year, quarter,
                 ec_ts.date())

        try:
            result = process_earnings_call(
                raw_root    = args.raw_root,
                tic         = tic,
                year        = year,
                quarter     = quarter,
                ec_ts_utc   = ec_ts,
                window_days = args.window_days,
                log         = log,
            )

            if result is None or result.empty:
                log.warning("No output for %s %d %s; skipping.", tic, year, quarter)
                write_checkpoint(ckpt_path, key)
                continue

            result = select_output_columns(result)

            out_path = os.path.join(args.output_dir, f"{tic}_{year}_{quarter}.parquet")
            result.to_parquet(out_path, index=False, engine="pyarrow")

            write_checkpoint(ckpt_path, key)
            processed += 1
            log.info("Saved %s  (%d rows)", out_path, len(result))

        except Exception:
            log.exception("Error processing %s %d %s", tic, year, quarter)

    log.info("Finished. Processed %d / %d earnings calls.", processed, total)


if __name__ == "__main__":
    main()