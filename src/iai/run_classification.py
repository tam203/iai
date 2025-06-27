import argparse
import logging

import pandas as pd

from iai.config import GOOGLE_API_KEY, LLM_RATE_LIMIT_SECONDS, TOPIC_LIST
from iai.classifiers.one_at_a_time_classifier import OneAtATimeClassifier
from iai.classifiers.goal_based_classifier import GoalBasedClassifier
from iai.classifiers.batch_topic_classifier import BatchTopicClassifier
from iai.utils import DATA_DIR, get_filepath_in_run_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM classification on repository data.")
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="The unique run ID for this execution.",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="summarized_gov_repositories.csv",
        help="Filename of the input CSV within the run_id directory.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Filename for the output CSV within the run_id directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of repositories to classify (processed in descending order of stars). " "Default: no limit.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["one_at_a_time", "batch", "goal_based"],
        default="one_at_a_time",
        help=(
            "The classification strategy to use. 'one_at_a_time' classifies each repo individually. "
            "'batch' clusters all repos and generates topics dynamically."
        ),
    )
    parser.add_argument(
        "--rate_limit",
        type=float,
        default=LLM_RATE_LIMIT_SECONDS,
        help=("Seconds to wait between LLM calls to avoid quota limits. " f"Default from config: {LLM_RATE_LIMIT_SECONDS}s."),
    )
    return parser.parse_args()


def _load_and_prepare_data(args):
    """Loads data from CSV, sorts, limits, and prepares DataFrame columns."""
    logger.info(f"Starting LLM classification for Run ID: {args.run_id}")
    run_data_path = DATA_DIR / args.run_id
    logger.info(f"Data for this run will be read from and saved to: {run_data_path}")

    if not GOOGLE_API_KEY:  # Moved API key check here as it's part of setup/preparation
        logger.error("Google API key (GOOGLE_API_KEY) is not set in environment or config.")
        logger.error("LLM classification cannot proceed without Google API credentials.")
        return None, None, None

    # Determine output_csv_filename based on whether it was provided or needs to be generated
    output_csv_filename = args.output_csv
    if output_csv_filename is None:
        output_csv_filename = f"classified_gov_repositories_{args.classifier}.csv"
        logger.info(f"No output CSV filename provided. Using default: {output_csv_filename}")

    input_file_path = get_filepath_in_run_data(args.run_id, args.input_csv)
    output_file_path = get_filepath_in_run_data(args.run_id, output_csv_filename)

    try:
        logger.info(f"Attempting to load data from {input_file_path}...")
        df = pd.read_csv(input_file_path)  # This is now the file that will be updated with summaries
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file_path}")
        return None, input_file_path, output_file_path  # Return None for df to signal failure
    except Exception as e:
        logger.error(f"Error loading CSV {input_file_path}: {e}")
        return None, input_file_path, output_file_path

    if df.empty:
        logger.info("Input DataFrame is empty. No repositories to classify.")
        # If the DataFrame is empty, we can't sort or limit.
        # Return the empty df and output path so main can save an empty file.
        # LLMClassAnalysis should handle an empty df, producing an empty classified_df.
    else:
        # Sort by stars if the column exists
        if "stars" in df.columns:
            logger.info("Sorting repositories by 'stars' in descending order.")
            df = df.sort_values(by="stars", ascending=False)
        elif args.limit is not None and args.limit > 0:
            # Only warn about missing 'stars' if a limit is actually going to be applied
            logger.warning(
                "'stars' column not found. Cannot sort by stars. " "If a limit is applied, it will be based on the current order of rows in the CSV."
            )

        # Apply limit
        if args.limit is not None and args.limit > 0:
            original_count = len(df)
            if original_count > args.limit:
                logger.info(f"Limiting classification to the top {args.limit} repositories (out of {original_count}).")
                df = df.head(args.limit)
            else:
                logger.info(f"Limit of {args.limit} requested, but only {original_count} " "repositories available. Processing all available.")
        elif args.limit is not None and args.limit <= 0:
            logger.warning(f"Invalid limit '{args.limit}' (must be a positive integer). Processing all repositories.")
        # Reset index after sorting/limiting to ensure clean indexing for subsequent operations
        df = df.reset_index(drop=True)

    # Ensure 'description' and 'readme_snippet' (to be renamed to 'readme') columns exist
    if "description" not in df.columns:
        logger.warning("'description' column not found in input CSV. It will be treated as empty for classification.")
        df["description"] = ""

    if "readme_snippet" in df.columns:
        df.rename(columns={"readme_snippet": "readme"}, inplace=True)
        logger.info("Renamed 'readme_snippet' column to 'readme'.")
    elif "readme" not in df.columns:
        logger.warning("'readme_snippet' or 'readme' column not found. 'readme' content will be treated as empty for classification.")
        df["readme"] = ""

    if "summary" not in df.columns:
        logger.error("'summary' column not found in input CSV. Summarization must be run before classification.")
        return None, input_file_path, output_file_path

    return df, input_file_path, output_file_path


def main():
    """Main function to run the LLM classification process."""
    args = _parse_args()
    df, input_file_path, output_file_path = _load_and_prepare_data(args)

    # If data loading failed or resulted in an empty DataFrame, exit early
    if df is None or input_file_path is None or df.empty:
        if df is not None and df.empty:  # Only save if df was loaded but empty
            df.to_csv(output_file_path, index=False, encoding="utf-8")
            logger.info(f"Empty DataFrame saved to {output_file_path}. Exiting.")
        return  # Exit if df is None or empty

    # --- Classifier Selection ---
    if args.classifier == "one_at_a_time":
        analyzer = OneAtATimeClassifier(run_id=args.run_id, rate_limit_seconds=args.rate_limit)
    elif args.classifier == "batch":
        analyzer = BatchTopicClassifier(run_id=args.run_id, rate_limit_seconds=args.rate_limit)
    elif args.classifier == "goal_based":
        analyzer = GoalBasedClassifier(run_id=args.run_id, rate_limit_seconds=args.rate_limit)
    else:
        # This case is handled by argparse choices, but as a safeguard:
        logger.error(f"Unknown classifier: {args.classifier}")
        return

    logger.info(f"Starting classification using {analyzer.__class__.__name__} for {len(df)} items.")

    # Use a copy of TOPIC_LIST for one_at_a_time; the batch classifier will ignore it and generate its own.
    initial_topics = list(TOPIC_LIST)  # This is ignored by batch classifier, but needed for one_at_a_time
    classified_df, updated_topic_list = analyzer.classify_repos(df, initial_topics, summaries_output_path=str(output_file_path))
    logger.info(f"Classification complete. {len(classified_df)} items classified. Final topic list has {len(updated_topic_list)} topics.")
    logger.info(f"Final topic list: {updated_topic_list}")

    try:
        classified_df.to_csv(output_file_path, index=False, encoding="utf-8")
        logger.info(f"Successfully saved classified data to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving CSV to {output_file_path}: {e}")


if __name__ == "__main__":
    main()
