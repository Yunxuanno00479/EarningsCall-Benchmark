"""
filter_taq_by_sp500.py

Layer 0 processing: Filter raw TAQ NBBO quote files (.gz) by S&P 500 symbol list
and write partitioned Parquet output.

Input:
    A single daily TAQ NBBO gzip file (pipe-delimited CSV) and an output directory.
    File path must follow the TAQ naming convention:
        .../EQY_US_ALL_NBBO_{YYYY}/EQY_US_ALL_NBBO_{YYYYMM}/EQY_US_ALL_NBBO_{YYYYMMDD}.gz

Output:
    Parquet files partitioned by Symbol_Copy under:
        {output_dir}/EQY_US_ALL_NBBO_{YYYY}/EQY_US_ALL_NBBO_{YYYYMM}/EQY_US_ALL_NBBO_{YYYYMMDD}/
            Symbol_Copy={TICKER}/*.parquet

Usage:
    python filter_taq_by_sp500.py <input_gz_path> <output_base_dir> <sp500_csv_path>

Arguments:
    input_gz_path   : Path to the daily TAQ .gz file.
    output_base_dir : Root directory for partitioned Parquet output.
    sp500_csv_path  : Path to the S&P 500 company list CSV (must contain a 'tic' column).

Requirements:
    PySpark, pandas
"""

import os
import sys

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


# TAQ NBBO schema as documented in the NYSE TAQ data specification.
TAQ_NBBO_SCHEMA = StructType([
    StructField("Time",                          StringType(),  True),
    StructField("Exchange",                      StringType(),  True),
    StructField("Symbol",                        StringType(),  True),
    StructField("Bid_Price",                     DoubleType(),  True),
    StructField("Bid_Size",                      IntegerType(), True),
    StructField("Offer_Price",                   DoubleType(),  True),
    StructField("Offer_Size",                    IntegerType(), True),
    StructField("Quote_Condition",               StringType(),  True),
    StructField("Sequence_Number",               LongType(),    True),
    StructField("National_BBO_Ind",              StringType(),  True),
    StructField("FINRA_BBO_Indicator",           StringType(),  True),
    StructField("FINRA_ADF_MPID_Indicator",      IntegerType(), True),
    StructField("Quote_Cancel_Correction",       StringType(),  True),
    StructField("Source_Of_Quote",               StringType(),  True),
    StructField("Best Bid Quote Condition",      StringType(),  True),
    StructField("Best_Bid_Exchange",             StringType(),  True),
    StructField("Best_Bid_Price",                DoubleType(),  True),
    StructField("Best_Bid_Size",                 IntegerType(), True),
    StructField("Best_Bid_FINRA_Market_Maker_ID", StringType(), True),
    StructField("Best_Offer_Quote_Condition",    StringType(),  True),
    StructField("Best_Offer_Exchange",           StringType(),  True),
    StructField("Best_Offer_Price",              DoubleType(),  True),
    StructField("Best_Offer_Size",               IntegerType(), True),
    StructField("Best_Offer_FINRA_Market_Maker_ID", StringType(), True),
    StructField("LULD_Indicator",                StringType(),  True),
    StructField("LULD_NBBO_Indicator",           StringType(),  True),
    StructField("SIP_Generated_Message_Identifier", StringType(), True),
    StructField("Participant_Timestamp",         StringType(),  True),
    StructField("FINRA_ADF_Timestamp",           StringType(),  True),
    StructField("Security_Status_Indicator",     StringType(),  True),
])


def load_symbol_list(sp500_csv_path: str) -> list[str]:
    """Load the S&P 500 ticker list from a CSV file with a 'tic' column."""
    df = pd.read_csv(sp500_csv_path, usecols=["tic"])
    symbols = df["tic"].dropna().str.strip().tolist()
    if not symbols:
        raise ValueError(f"No symbols found in {sp500_csv_path}")
    return symbols


def build_output_dir(input_gz_path: str, output_base_dir: str) -> str:
    """
    Derive the output directory from the input file path.

    Input path structure:
        .../EQY_US_ALL_NBBO_{YYYY}/EQY_US_ALL_NBBO_{YYYYMM}/EQY_US_ALL_NBBO_{YYYYMMDD}.gz

    Output path structure:
        {output_base_dir}/EQY_US_ALL_NBBO_{YYYY}/EQY_US_ALL_NBBO_{YYYYMM}/EQY_US_ALL_NBBO_{YYYYMMDD}/
    """
    file_name    = os.path.basename(input_gz_path).replace(".gz", "")
    month_folder = os.path.basename(os.path.dirname(input_gz_path))
    year_folder  = os.path.basename(os.path.dirname(os.path.dirname(input_gz_path)))
    return os.path.join(output_base_dir, year_folder, month_folder, file_name)


def filter_and_write(
    input_gz_path: str,
    output_base_dir: str,
    sp500_csv_path: str,
) -> None:
    """
    Read one daily TAQ NBBO .gz file, filter to S&P 500 symbols,
    and write partitioned Parquet output.
    """
    symbols = load_symbol_list(sp500_csv_path)
    output_dir = build_output_dir(input_gz_path, output_base_dir)
    os.makedirs(output_dir, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("EarningsCallBench_FilterTAQ")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )

    try:
        df = spark.read.csv(
            input_gz_path,
            sep="|",
            header=True,
            inferSchema=False,
            schema=TAQ_NBBO_SCHEMA,
        )

        # Keep only S&P 500 symbols. Add Symbol_Copy as the partition key
        # so downstream readers can locate a symbol's data without scanning
        # unrelated partitions.
        df_filtered = (
            df.filter(col("Symbol").isin(symbols))
              .withColumn("Symbol_Copy", col("Symbol"))
        )

        df_filtered.write.mode("append").partitionBy("Symbol_Copy").parquet(output_dir)

    finally:
        spark.stop()


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: python filter_taq_by_sp500.py "
            "<input_gz_path> <output_base_dir> <sp500_csv_path>"
        )
        sys.exit(1)

    input_gz_path   = sys.argv[1]
    output_base_dir = sys.argv[2]
    sp500_csv_path  = sys.argv[3]

    filter_and_write(input_gz_path, output_base_dir, sp500_csv_path)


if __name__ == "__main__":
    main()