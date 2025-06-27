import argparse
import logging

import pandas as pd

from iai.config import GOOGLE_API_KEY, LLM_RATE_LIMIT_SECONDS
from iai.llm_utils import generate_summaries
from iai.utils import DATA_DIR, get_filepath_in_run_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args():
    """Parses command-line arguments for summarization."""
    parser = argparse.ArgumentParser(description="Run LLM summarization on repository data.")
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="The unique run ID for this execution.",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="gov_repositories_catalog.csv",
        help="Filename of the input CSV within the run_id directory (output of run_scan.py).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="summarized_gov_repositories.csv",
        help="Filename for the output CSV within the run_id directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of repositories to summarize (processed in descending order of stars). " "Default: no limit.",
    )
    parser.add_argument(
        "--rate_limit",
        type=float,
        default=LLM_RATE_LIMIT_SECONDS,
        help=("Seconds to wait between LLM calls to avoid quota limits. " f"Default from config: {LLM_RATE_LIMIT_SECONDS}s."),
    )
    return parser.parse_args()


def _load_and_prepare_data(args):
    """Loads data from CSV, sorts, limits, and prepares DataFrame columns for summarization."""
    logger.info(f"Starting LLM summarization for Run ID: {args.run_id}")
    run_data_path = DATA_DIR / args.run_id
    logger.info(f"Data for this run will be read from and saved to: {run_data_path}")

    if not GOOGLE_API_KEY:
        logger.error("Google API key (GOOGLE_API_KEY) is not set in environment or config.")
        logger.error("LLM summarization cannot proceed without Google API credentials.")
        return None, None

    input_file_path = get_filepath_in_run_data(args.run_id, args.input_csv)
    output_file_path = get_filepath_in_run_data(args.run_id, args.output_csv)

    try:
        logger.info(f"Attempting to load data from {input_file_path}...")
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading CSV {input_file_path}: {e}")
        return None, None

    if df.empty:
        logger.info("Input DataFrame is empty. No repositories to summarize.")
    else:
        if "stars" in df.columns:
            logger.info("Sorting repositories by 'stars' in descending order.")
            df = df.sort_values(by="stars", ascending=False)

        if args.limit is not None and args.limit > 0:
            original_count = len(df)
            if original_count > args.limit:
                logger.info(f"Limiting summarization to the top {args.limit} repositories (out of {original_count}).")
                df = df.head(args.limit)
        df = df.reset_index(drop=True)

    # Ensure 'description' and 'readme' columns exist for summarization
    if "description" not in df.columns:
        df["description"] = ""
    if "readme_snippet" in df.columns:
        df.rename(columns={"readme_snippet": "readme"}, inplace=True)
    elif "readme" not in df.columns:
        df["readme"] = ""

    return df, output_file_path


def main():
    args = _parse_args()
    df, output_file_path = _load_and_prepare_data(args)

    if df is None:  # Error during loading or API key missing
        return

    logger.info(f"Generating summaries for {len(df)} repositories...")
    summarized_df = generate_summaries(df, output_csv_path=output_file_path, rate_limit_seconds=args.rate_limit)

    try:
        summarized_df.to_csv(output_file_path, index=False, encoding="utf-8")
        logger.info(f"Successfully saved summarized data to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving CSV to {output_file_path}: {e}")


if __name__ == "__main__":
    main()
