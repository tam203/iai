import logging

from iai import GitHubScanner, generate_run_id
from iai.config import DEFAULT_SEARCH_QUERIES, GITHUB_API_TOKEN  # Assuming config.py
from iai.utils import DATA_DIR  # To inform user where data is stored


def main():
    """Main function to run the GitHub scan and catalogue process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting GitHub project scan...")

    # 1. Generate a unique run ID for this execution
    run_id = generate_run_id()
    logger.info(f"Generated Run ID: {run_id}")
    logger.info(f"Data for this run will be stored in: {DATA_DIR / run_id}")

    # 2. Initialize the scanner
    # Ensure GITHUB_API_TOKEN is set as an environment variable or in src/iai/config.py
    print(GITHUB_API_TOKEN)
    scanner = GitHubScanner(run_id=run_id, api_token=GITHUB_API_TOKEN)

    # 3. Perform the scan for government repositories
    logger.info("Starting scan for government repositories...")
    gov_catalog_file_path = scanner.scan_and_save_gov_repos_catalog()
    if gov_catalog_file_path:
        logger.info(
            f"Government repositories scan complete. Catalog saved to: {gov_catalog_file_path}"
        )
    else:
        logger.info(
            "Government repositories scan did not produce a catalog (e.g., API token missing or no data found)."
        )


if __name__ == "__main__":
    main()
