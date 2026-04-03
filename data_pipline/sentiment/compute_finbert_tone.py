"""
compute_finbert_tone.py

Compute FinBERT-tone sentiment scores for earnings call transcript panels.

Supported panels:
    pre         -- Presentation section, one row per sentence
    qa_sentence -- QA section, one row per sentence

Model: yiyanghkust/finbert-tone (Huang et al., 2022)
  Output labels: Positive, Neutral, Negative
  Expected value: P(Positive) - P(Negative)

Change point detection (optional):
  Algorithm: PELT with BIC penalty, pen = log(n)
  Reference: Killick, Fearnhead & Eckley (2012), JASA 107(500), 1590-1598
  Library:   ruptures (Truong et al., 2020, Signal Processing 167, 107299)
  Flag 1 marks the first sentence of each new sentiment regime.

Input format (pre panel):
    CSV with columns: tic, year, quarter, section_id, presentation_text, ...

Input format (qa_sentence panel):
    CSV with columns: tic, year, quarter, qa_index, sentence_index, sentence, ...

Output columns (pre):
    tic, year, quarter, section_id,
    finberttone_expected_value, finberttone_cumulative_tone, finberttone_change_point

Output columns (qa_sentence):
    tic, year, quarter, qa_index, sentence_index,
    finberttone_expected_value, finberttone_cumulative_tone, finberttone_change_point

    finberttone_change_point = -1 when --compute_changepoint is not set.

Usage:
    # Create and activate environment
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    # Presentation panel (GPU 0)
    python compute_finbert_tone.py \\
        --panel      pre \\
        --input_dir  /path/to/transcript_panel/pre \\
        --output_dir /path/to/sentiment_panel/pre \\
        --device     0 \\
        --compute_changepoint

    # QA sentence panel (GPU 1)
    python compute_finbert_tone.py \\
        --panel      qa_sentence \\
        --input_dir  /path/to/transcript_panel/qa_sentence \\
        --output_dir /path/to/sentiment_panel/qa_sentence \\
        --device     1 \\
        --compute_changepoint
"""

import argparse
import csv
import glob
import logging
import math
import os
import sys

import numpy as np
import torch
from transformers import pipeline


FINBERT_MODEL = "yiyanghkust/finbert-tone"

PANEL_CONFIGS = {
    "pre": {
        "file_pattern": "*_pre.csv",
        "output_suffix": "_pre_score.csv",
        "input_suffix": "_pre.csv",
        "text_col": "presentation_text",
        "id_cols": ["tic", "year", "quarter", "section_id"],
        "output_cols": [
            "tic", "year", "quarter", "section_id",
            "finberttone_expected_value",
            "finberttone_cumulative_tone",
            "finberttone_change_point",
        ],
    },
    "qa_sentence": {
        "file_pattern": "*_qa_sentence.csv",
        "output_suffix": "_qa_sentence_score.csv",
        "input_suffix": "_qa_sentence.csv",
        "text_col": "sentence",
        "id_cols": ["tic", "year", "quarter", "qa_index", "sentence_index"],
        "output_cols": [
            "tic", "year", "quarter", "qa_index", "sentence_index",
            "finberttone_expected_value",
            "finberttone_cumulative_tone",
            "finberttone_change_point",
        ],
    },
}


def setup_logging(output_dir):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "compute_finbert_tone.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_checkpoint(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                stem = line.strip()
                if stem:
                    done.add(stem)
    return done


def mark_done(path, stem):
    with open(path, "a") as f:
        f.write(stem + "\n")


def load_ruptures():
    try:
        import ruptures as rpt
        return rpt
    except ImportError:
        print(
            "ERROR: ruptures is not installed.\n"
            "Install with: pip install ruptures"
        )
        sys.exit(1)


def detect_changepoints_bic(ev_list, cp_model, cp_min_size, rpt):
    """
    Apply PELT with BIC penalty to a sequence of expected values.

    The BIC penalty pen = log(n) is the standard criterion from
    Killick et al. (2012) for penalized likelihood change point detection.

    Parameters
    ----------
    ev_list : list of float
        Sequence of expected values (P(Positive) - P(Negative)) for one EC.
    cp_model : str
        PELT cost function. "rbf" is recommended for bounded sentiment scores.
    cp_min_size : int
        Minimum number of sentences between consecutive change points.
    rpt : module
        The ruptures module.

    Returns
    -------
    list of int
        Binary flags of length len(ev_list).
        Flag 1 marks the first sentence of each new sentiment regime.
    """
    n = len(ev_list)
    if n < cp_min_size * 2:
        return [0] * n

    signal = np.array(ev_list, dtype=float).reshape(-1, 1)
    pen = math.log(n)
    algo = rpt.Pelt(model=cp_model, min_size=cp_min_size, jump=1).fit(signal)
    breakpoints = algo.predict(pen=pen)

    flags = [0] * n
    for bp in breakpoints[:-1]:
        if 0 < bp < n:
            flags[bp] = 1
    return flags


def run_finbert(texts, clf, batch_size):
    """
    Run FinBERT-tone inference on a list of texts.

    Returns a list of expected values: P(Positive) - P(Negative).
    """
    ev_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        predictions = clf(batch, truncation=True, max_length=512)
        for pred in predictions:
            score_map = {item["label"]: item["score"] for item in pred}
            pos = score_map.get("Positive", 0.0)
            neg = score_map.get("Negative", 0.0)
            ev_list.append(pos - neg)
    return ev_list


def process_file(input_path, output_path, clf, config,
                 batch_size, compute_changepoint, cp_model, cp_min_size,
                 rpt, log):
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        log.warning("%s: empty file, skipping.", input_path)
        return

    texts = [row[config["text_col"]] for row in rows]
    ev_list = run_finbert(texts, clf, batch_size)

    # Cumulative tone: running sum across all sentences in this EC.
    cumulative, cum_list = 0.0, []
    for ev in ev_list:
        cumulative += ev
        cum_list.append(cumulative)

    if compute_changepoint:
        cp_flags = detect_changepoints_bic(ev_list, cp_model, cp_min_size, rpt)
    else:
        cp_flags = [-1] * len(ev_list)

    out_rows = []
    for row, ev, cum_val, cp in zip(rows, ev_list, cum_list, cp_flags):
        out_row = {col: row[col] for col in config["id_cols"]}
        out_row["finberttone_expected_value"] = round(ev, 9)
        out_row["finberttone_cumulative_tone"] = round(cum_val, 9)
        out_row["finberttone_change_point"] = cp
        out_rows.append(out_row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config["output_cols"])
        writer.writeheader()
        writer.writerows(out_rows)

    n_cp = sum(c for c in cp_flags if c == 1)
    log.info(
        "%s: %d rows, %d change points (rate=%.3f), cp_computed=%s",
        os.path.basename(input_path),
        len(out_rows),
        n_cp,
        n_cp / len(out_rows) if out_rows else 0.0,
        compute_changepoint,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute FinBERT-tone scores for transcript panels."
    )
    p.add_argument(
        "--panel", required=True, choices=["pre", "qa_sentence"],
        help="Which panel to score: 'pre' or 'qa_sentence'.",
    )
    p.add_argument("--input_dir",  required=True,
                   help="Directory containing input CSV files.")
    p.add_argument("--output_dir", required=True,
                   help="Directory for output score CSV files.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Inference batch size (default: 64).")
    p.add_argument("--device", type=int, default=0,
                   help="GPU index. Use -1 for CPU (default: 0).")
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint file for resuming interrupted runs.")
    p.add_argument("--compute_changepoint", action="store_true",
                   help=(
                       "Run PELT change point detection after scoring. "
                       "Requires the ruptures package."
                   ))
    p.add_argument("--cp_model", default="rbf", choices=["rbf", "l1", "l2"],
                   help="PELT cost model (default: rbf).")
    p.add_argument("--cp_min_size", type=int, default=3,
                   help="Minimum segment length between change points (default: 3).")
    return p.parse_args()


def main():
    args   = parse_args()
    config = PANEL_CONFIGS[args.panel]
    log    = setup_logging(args.output_dir)

    checkpoint_path = args.checkpoint or os.path.join(
        args.output_dir, f"checkpoint_{args.panel}.txt"
    )
    done = load_checkpoint(checkpoint_path)

    rpt = None
    if args.compute_changepoint:
        rpt = load_ruptures()
        log.info(
            "PELT enabled: model=%s, min_size=%d, penalty=log(n) per EC.",
            args.cp_model, args.cp_min_size,
        )

    device = args.device if torch.cuda.is_available() else -1
    log.info("Loading %s on device=%d ...", FINBERT_MODEL, device)
    clf = pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        top_k=None,
        device=device,
    )
    log.info("Model loaded.")

    files = sorted(
        glob.glob(os.path.join(args.input_dir, config["file_pattern"]))
    )
    log.info("Found %d input files for panel '%s'.", len(files), args.panel)

    for input_path in files:
        stem = os.path.basename(input_path)
        if stem in done:
            log.info("Skip (already done): %s", stem)
            continue

        output_stem = stem.replace(
            config["input_suffix"], config["output_suffix"]
        )
        output_path = os.path.join(args.output_dir, output_stem)

        try:
            process_file(
                input_path, output_path, clf, config,
                args.batch_size,
                args.compute_changepoint,
                args.cp_model, args.cp_min_size,
                rpt, log,
            )
            mark_done(checkpoint_path, stem)
        except Exception:
            log.exception("Error processing %s.", stem)

    log.info("Done.")


if __name__ == "__main__":
    main()