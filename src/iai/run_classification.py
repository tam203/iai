import argparse
import logging

import pandas as pd

from iai.config import AZURE_OPENAI_API_KEY, TOPIC_LIST
from iai.llm_annalysis import LLMClassAnalysis
from iai.utils import DATA_DIR, get_filepath_in_run_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the LLM classification process."""
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
        default="gov_repositories_catalog.csv",
        help="Filename of the input CSV within the run_id directory.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="classified_gov_repositories.csv",
        help="Filename for the output CSV within the run_id directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of repositories to classify (processed in descending order of stars). " "Default: no limit.",
    )
    args = parser.parse_args()

    logger.info(f"Starting LLM classification for Run ID: {args.run_id}")
    run_data_path = DATA_DIR / args.run_id
    logger.info(f"Data for this run will be read from and saved to: {run_data_path}")

    # Check for Azure OpenAI API Key
    if not AZURE_OPENAI_API_KEY:
        logger.error("Azure OpenAI API key (AZURE_OPENAI_API_KEY) is not set in environment or config.")
        logger.error("LLM classification cannot proceed without API credentials.")
        return

    input_file_path = get_filepath_in_run_data(args.run_id, args.input_csv)
    output_file_path = get_filepath_in_run_data(args.run_id, args.output_csv)

    try:
        logger.info(f"Loading data from {input_file_path}...")
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file_path}")
        return
    except Exception as e:
        logger.error(f"Error loading CSV {input_file_path}: {e}")
        return

    if df.empty:
        logger.info("Input DataFrame is empty. No repositories to classify.")
        # If the DataFrame is empty, we can't sort or limit.
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

    if df.empty and args.limit is not None and args.limit > 0:  # Check if df became empty after limiting
        logger.info("DataFrame is empty after applying the limit. No repositories to classify.")
        # classified_df will be empty, and an empty CSV will be saved.

    # Initialize LLMClassAnalysis
    # API token, deployment, and endpoint will be picked from iai.config by default
    analyzer = LLMClassAnalysis(run_id=args.run_id)

    topic_list = TOPIC_LIST
    logger.info(f"Starting classification with initial topics: {topic_list}")

    classified_df, updated_topic_list = analyzer.classify_repos(df, topic_list)

    logger.info(f"Classification complete. Updated topic list: {updated_topic_list}")

    try:
        classified_df.to_csv(output_file_path, index=False, encoding="utf-8")
        logger.info(f"Successfully saved classified data to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving CSV to {output_file_path}: {e}")


if __name__ == "__main__":
    main()
