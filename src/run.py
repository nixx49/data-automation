#!/usr/bin/env python
"""
This script processes the AG News dataset using PySpark to generate two parquet files:
1. A fixed word count table for the words: ["president", "the", "Asia"].
2. A table counting all unique words from the "description" column.

Usage:
    python src/run.py process_data --cfg config/cfg.yaml --dataset news --dirout ztmp/data/
    python src/run.py process_data_all --cfg config/cfg.yaml --dataset news --dirout ztmp/data/
    python src/run.py run_tests  --cfg config/cfg.yaml --dataset news --dirout ztmp/data/
"""

import argparse
import logging
import re
from collections import namedtuple
from datetime import datetime
from typing import List, Tuple, Dict, Any

import yaml
from datasets import load_dataset
from pyspark.sql import SparkSession, DataFrame, Row, functions as F
from pyspark.sql.types import IntegerType

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_config(cfg_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    :param cfg_path: Path to the config file.
    :return: Configuration as a dictionary.
    """
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully from %s", cfg_path)
    return config


def load_ag_news_data(spark: SparkSession) -> DataFrame:
    """
    Load the AG News dataset (test split) using HuggingFace datasets and convert it to a Spark DataFrame.

    :param spark: SparkSession.
    :return: Spark DataFrame containing the AG News test data.
    """
    logger.info("Loading AG News dataset using HuggingFace datasets...")
    dataset = load_dataset("sh0416/ag_news", split="test")
    pandas_df = dataset.to_pandas()
    spark_df = spark.createDataFrame(pandas_df)
    logger.info("AG News dataset loaded with %d rows.", spark_df.count())
    return spark_df


def get_count_udf(word: str):
    """
    Create a UDF to count occurrences of a given word (case-sensitive) in a text.

    :param word: Word to count.
    :return: A Spark UDF that counts the number of occurrences.
    """
    pattern = r'\b{}\b'.format(re.escape(word))

    def count(text: str) -> int:
        if text is None:
            return 0
        return len(re.findall(pattern, text))

    return F.udf(count, IntegerType())


def compute_fixed_word_counts(spark_df: DataFrame, words: List[str]) -> List[Tuple[str, int]]:
    """
    Compute the total frequency of each fixed word in the 'description' column.

    :param spark_df: Spark DataFrame with column 'description'.
    :param words: List of words to count.
    :return: List of tuples (word, word_count).
    """
    results: List[Tuple[str, int]] = []
    for word in words:
        count_udf = get_count_udf(word)
        count_val = spark_df.select(F.sum(count_udf(F.col("description"))).alias("word_count")).collect()[0]["word_count"]
        results.append((word, count_val))
        logger.info("Word '%s' count: %d", word, count_val)
    return results


def process_fixed_words(spark: SparkSession, spark_df: DataFrame, output_dir: str) -> None:
    """
    Process fixed words (["president", "the", "Asia"]) and save the result as a parquet file.

    :param spark: SparkSession.
    :param spark_df: Input Spark DataFrame with column 'description'.
    :param output_dir: Output directory where the parquet file will be saved.
    """
    logger.info("Processing fixed words...")
    words = ["president", "the", "Asia"]
    results = compute_fixed_word_counts(spark_df, words)
    # Create a Spark DataFrame from the results list.
    result_df = spark.createDataFrame(results, schema=["word", "word_count"])
    current_date = datetime.now().strftime("%Y%m%d")
    file_name = f"{output_dir}/word_count_{current_date}.parquet"
    result_df.write.mode("overwrite").parquet(file_name)
    logger.info("Parquet file '%s' saved successfully!", file_name)


def compute_all_word_counts(spark_df: DataFrame) -> DataFrame:
    """
    Compute the total frequency of all unique words in the 'description' column.

    :param spark_df: Spark DataFrame with column 'description'.
    :return: Spark DataFrame with columns 'word' and 'word_count'.
    """
    logger.info("Computing counts for all unique words...")
    # Split the description by whitespace and explode the list to get one word per row.
    words_df = spark_df.select(F.explode(F.split(F.col("description"), r"\s+")).alias("word"))
    word_counts_df = words_df.groupBy("word").agg(F.count("*").alias("word_count"))
    logger.info("Computed counts for %d unique words.", word_counts_df.count())
    return word_counts_df


def process_all_words(spark: SparkSession, spark_df: DataFrame, output_dir: str) -> None:
    """
    Process all words in the 'description' column and save the result as a parquet file.

    :param spark: SparkSession.
    :param spark_df: Input Spark DataFrame with column 'description'.
    :param output_dir: Output directory where the parquet file will be saved.
    """
    logger.info("Processing all words...")
    word_counts_df = compute_all_word_counts(spark_df)
    current_date = datetime.now().strftime("%Y%m%d")
    file_name = f"{output_dir}/word_count_all_{current_date}.parquet"
    word_counts_df.write.mode("overwrite").parquet(file_name)
    logger.info("Parquet file '%s' saved successfully!", file_name)


def run_tests(spark: SparkSession) -> None:
    """
    Run basic tests on the word count functions.
    """
    logger.info("Running tests...")

    # Test fixed words count
    test_data = [Row(description="president the Asia president")]
    test_df = spark.createDataFrame(test_data)
    fixed_results = compute_fixed_word_counts(test_df, ["president", "the", "Asia"])
    expected_fixed = {"president": 2, "the": 1, "Asia": 1}
    assert dict(fixed_results) == expected_fixed, f"Fixed words count test failed: {fixed_results}"
    logger.info("Fixed words count test passed.")

    # Test all words count
    test_data2 = [Row(description="Today this is raining")]
    test_df2 = spark.createDataFrame(test_data2)
    all_counts_df = compute_all_word_counts(test_df2)
    result = {row["word"]: row["word_count"] for row in all_counts_df.collect()}
    expected_all = {"Today": 1, "this": 1, "is": 1, "raining": 1}
    assert result == expected_all, f"All words count test failed: {result}"
    logger.info("All words count test passed.")


def main() -> None:
    """
    Main function to parse arguments and run the appropriate processing command.
    """
    parser = argparse.ArgumentParser(description="Process AG News dataset using PySpark.")
    parser.add_argument("command", choices=["process_data", "process_data_all", "run_tests"],
                        help="Command to run: process_data for fixed words, process_data_all for all words, or run_tests for testing.")
    parser.add_argument("--cfg", type=str, default="config/cfg.yaml", help="Path to configuration file.")
    parser.add_argument("--dataset", type=str, default="news", help="Dataset name (not used in processing).")
    parser.add_argument("--dirout", type=str, default="ztmp/data", help="Output directory.")
    args = parser.parse_args()

    # Initialize Spark session
    spark = SparkSession.builder.appName("AGNewsWordCount").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load config (if needed)
    if args.cfg:
        _ = load_config(args.cfg)

    # Run tests if requested
    if args.command == "run_tests":
        run_tests(spark)
        spark.stop()
        return

    # Load AG News dataset
    spark_df = load_ag_news_data(spark)

    if args.command == "process_data":
        process_fixed_words(spark, spark_df, args.dirout)
    elif args.command == "process_data_all":
        process_all_words(spark, spark_df, args.dirout)

    spark.stop()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
    main()
