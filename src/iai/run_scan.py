import logging
import argparse

from iai import GitHubScanner, generate_run_id
from iai.config import GITHUB_API_TOKEN
from iai.utils import DATA_DIR


def main():
    """Main function to run the GitHub scan and catalogue process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run GitHub project scan.")
    parser.add_argument(
        "--run_id",
        type=str,
        help="Optional existing Run ID to resume or update a scan. If not provided, a new scan run is initiated.",
        default=None,
    )
    args = parser.parse_args()

    logger.info("Starting GitHub project scan...")

    current_run_id = args.run_id

    if current_run_id:
        run_path = DATA_DIR / current_run_id
        if not run_path.is_dir():
            logger.error(
                f"Provided Run ID '{current_run_id}' directory does not exist at {run_path}. "
                "Cannot resume. Please provide a valid Run ID or omit --run_id for a new scan."
            )
            return
        logger.info(f"Resuming or updating scan for existing Run ID: {current_run_id}")
    else:
        current_run_id = generate_run_id()
        logger.info(f"Generated new Run ID: {current_run_id}")

    # Ensure the run directory exists (it will be created if new)
    run_data_dir = DATA_DIR / current_run_id
    run_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data for this run will be stored in: {run_data_dir}")

    # 2. Initialize the scanner
    # Ensure GITHUB_API_TOKEN is set as an environment variable or in src/iai/config.py
    scanner = GitHubScanner(run_id=current_run_id, api_token=GITHUB_API_TOKEN)

    # 3. Perform the scan for government repositories
    logger.info("Starting scan for government repositories...")
    gov_catalog_file_path = scanner.scan_and_save_gov_repos_catalog()  # output_filename removed
    if gov_catalog_file_path:
        logger.info(f"Government repositories scan complete. Catalog saved to: {gov_catalog_file_path}")
    else:
        logger.info(
            "Government repositories scan did not produce a catalog (e.g., API token missing or no data found)."
        )  # This message might change slightly due to new file handling


if __name__ == "__main__":
    main()
