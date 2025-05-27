import argparse
import logging
from pathlib import Path

import pandas as pd

from iai.config import AZURE_OPENAI_API_KEY, DEPLOYMENT_NAME, ENDPOINT_URL
from iai.llm_annalysis import LLMClassAnalysis
from iai.utils import DATA_DIR, get_filepath_in_run_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the LLM classification process."""
    parser = argparse.ArgumentParser(
        description="Run LLM classification on repository data."
    )
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
    args = parser.parse_args()

    logger.info(f"Starting LLM classification for Run ID: {args.run_id}")
    run_data_path = DATA_DIR / args.run_id
    logger.info(f"Data for this run will be read from and saved to: {run_data_path}")

    # Check for Azure OpenAI API Key
    if not AZURE_OPENAI_API_KEY:
        logger.error(
            "Azure OpenAI API key (AZURE_OPENAI_API_KEY) is not set in environment or config."
        )
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
        # Save an empty CSV with expected columns if needed, or just skip.
        # For now, we'll let it proceed, and LLMClassAnalysis should handle empty df.
        # If classify_repos creates columns, an empty df with those columns will be saved.

    # Ensure 'description' and 'readme_snippet' (to be renamed to 'readme') columns exist
    if "description" not in df.columns:
        logger.warning(
            "'description' column not found in input CSV. It will be treated as empty for classification."
        )
        df["description"] = ""

    if "readme_snippet" in df.columns:
        df.rename(columns={"readme_snippet": "readme"}, inplace=True)
        logger.info("Renamed 'readme_snippet' column to 'readme'.")
    elif "readme" not in df.columns:
        logger.warning(
            "'readme_snippet' or 'readme' column not found. 'readme' content will be treated as empty for classification."
        )
        df["readme"] = ""

    # Initialize LLMClassAnalysis
    # API token, deployment, and endpoint will be picked from iai.config by default
    analyzer = LLMClassAnalysis(run_id=args.run_id)

    topic_list = ["ai", "web development", "data science", "cybersecurity"]
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