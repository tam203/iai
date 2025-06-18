import json
import logging
from typing import List, Dict, Any, Optional
import datetime

import pandas as pd
import requests
import yaml

from .utils import get_filepath_in_run_data
from .config import GITHUB_API_TOKEN, GOVERNMENT_ACCOUNTS_URL, TARGET_REPOS

logger = logging.getLogger(__name__)


class GitHubScanner:
    """
    Scans GitHub for projects based on search queries and catalogues them.
    """

    EXPECTED_CSV_COLUMNS = [
        "account",
        "name",
        "description",
        "stars",
        "forks",
        "language",
        "url",
        "readme_snippet",
        "last_scanned_utc",
    ]

    def __init__(self, run_id: str, api_token: str = None):
        self.run_id = run_id
        self.api_token = api_token or GITHUB_API_TOKEN
        if not self.api_token or "your_default_token" in self.api_token:
            logger.warning("GitHub API token is not set or is using a default placeholder. API rate limits may be encountered.")
        self.catalog_filepath = get_filepath_in_run_data(self.run_id, "gov_repositories_catalog.csv")
        self.scanned_repo_identifiers = self._load_existing_scanned_repos()

    def _load_existing_scanned_repos(self) -> set:
        """Loads unique identifiers (account/repo_name) of already scanned repos from the CSV."""
        identifiers = set()
        if self.catalog_filepath.exists() and self.catalog_filepath.stat().st_size > 0:
            try:
                df_existing = pd.read_csv(self.catalog_filepath)
                if "account" in df_existing.columns and "name" in df_existing.columns:
                    for _, row in df_existing.iterrows():
                        account = str(row["account"]) if pd.notna(row["account"]) else ""
                        name = str(row["name"]) if pd.notna(row["name"]) else ""
                        if account and name:
                            identifiers.add(f"{account}/{name}")
                else:
                    logger.warning(
                        f"CSV {self.catalog_filepath} is missing 'account' or 'name' columns. "
                        "Cannot load existing repo identifiers. Will re-scan all."
                    )
            except pd.errors.EmptyDataError:
                logger.info(f"Catalog file {self.catalog_filepath} is empty. No existing repos to load.")
            except Exception as e:
                logger.error(f"Error loading existing scanned repos from {self.catalog_filepath}: {e}")
        return identifiers

    def _append_repo_to_csv(self, repo_data: Dict[str, Any]):
        """Appends a single repository's data to the CSV file."""
        for col in self.EXPECTED_CSV_COLUMNS:  # Ensure all columns are present
            repo_data.setdefault(col, None)

        df_to_append = pd.DataFrame([repo_data], columns=self.EXPECTED_CSV_COLUMNS)
        file_exists_and_not_empty = self.catalog_filepath.exists() and self.catalog_filepath.stat().st_size > 0
        df_to_append.to_csv(
            self.catalog_filepath,
            mode="a",
            header=not file_exists_and_not_empty,
            index=False,
            encoding="utf-8",
        )

    # --- Methods for Government Repositories Scan ---

    def fetch_gov_github_accounts_yaml(self, url: str) -> Optional[Dict[str, List[str]]]:
        """Fetches and parses the YAML file listing government GitHub accounts."""
        logger.info(f"Fetching government GitHub accounts from: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            data = yaml.safe_load(response.text)
            if not isinstance(data, dict):
                logger.error(f"Government accounts YAML from {url} did not parse into a dictionary.")
                return None
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching government accounts YAML from {url}: {e}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing government accounts YAML from {url}: {e}")
        return None

    def _extract_usernames_from_gov_data(self, gov_data: Dict[str, List[str]], filter_list: Optional[List[str]] = None) -> List[str]:
        """
        Extracts unique GitHub usernames/organization names from the government data structure.
        The expected structure for gov_data is Dict[str, List[str]] (e.g., {'Country': ['org1', 'org2']}).
        Optionally filters the usernames against a provided list.
        """
        usernames = set()
        if not isinstance(gov_data, dict):
            logger.warning("Government data is not in the expected dictionary format (country: [orgs]).")
            return []

        apply_filter = bool(filter_list)  # True if filter_list is not None and not empty

        for country_name, org_list in gov_data.items():
            if not isinstance(org_list, list):
                logger.warning(f"Expected a list of orgs for country '{country_name}', but got {type(org_list)}. Skipping.")
                continue
            for username in org_list:
                if isinstance(username, str) and (not apply_filter or (filter_list and username in filter_list)):
                    usernames.add(username)
        logger.info(f"Extracted {len(usernames)} unique usernames. Filter applied: {apply_filter} (using {len(filter_list or [])} filter items).")
        return list(usernames)

    def _fetch_repository_details_for_account(self, username: str):  # No longer returns a list
        """Fetches repository details for a given GitHub username/organization."""
        if not self.api_token or "your_default_token" in self.api_token:
            logger.error(f"A valid GitHub API token is required to fetch repository details for {username}. Skipping.")
            return

        headers = {"Authorization": f"token {self.api_token}"}
        repos_url = f"https://api.github.com/users/{username}/repos?per_page=100"  # Adjust per_page as needed
        logger.info(f"Fetching repositories for account: {username} from {repos_url}")

        try:
            repos_response = requests.get(repos_url, headers=headers, timeout=30)
            repos_response.raise_for_status()
            repos_data = repos_response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching repositories for {username}: {e}")
            return
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for {username}'s repositories: {e}")
            return

        logger.info(f"Found {len(repos_data)} repositories for {username}. Processing details...")

        for repo_api_data in repos_data:
            if not isinstance(repo_api_data, dict):
                logger.warning(f"Skipping non-dict item in repo data for {username}")
                continue

            repo_name = repo_api_data.get("name")
            if not repo_name:
                logger.warning(f"Skipping repository with no name for account {username}.")
                continue

            repo_identifier = f"{username}/{repo_name}"

            if repo_identifier in self.scanned_repo_identifiers:
                logger.info(f"Skipping already scanned repository: {repo_identifier}")
                continue

            logger.info(f"Fetching details for new repository: {repo_identifier}")
            repo_details = {
                "account": username,
                "name": repo_name,
                "description": repo_api_data.get("description") or "No description",
                "stars": repo_api_data.get("stargazers_count", 0),
                "forks": repo_api_data.get("forks_count", 0),
                "language": repo_api_data.get("language") or "None specified",
                "url": repo_api_data.get("html_url", ""),
                "readme_snippet": "README not available",  # Default
            }

            # Fetch the README file
            readme_url = f"https://api.github.com/repos/{username}/{repo_name}/readme"
            try:
                readme_api_response = requests.get(readme_url, headers=headers, timeout=10)
                if readme_api_response.status_code == 200:
                    readme_api_data = readme_api_response.json()
                    readme_download_url = readme_api_data.get("download_url")
                    if readme_download_url:
                        readme_content_response = requests.get(readme_download_url, timeout=10)
                        if readme_content_response.status_code == 200:
                            repo_details["readme_snippet"] = readme_content_response.text[:150]  # Truncate
                        else:
                            logger.debug(f"Failed to download README for {repo_identifier}, status: {readme_content_response.status_code}")
                    else:
                        logger.debug(f"No download_url for README of {repo_identifier}")
                elif readme_api_response.status_code != 404:  # Log if not a simple "Not Found"
                    logger.warning(f"Failed to fetch README metadata for {repo_identifier}, status: {readme_api_response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"RequestException for README {repo_identifier}: {e}")
            except json.JSONDecodeError:
                logger.warning(f"JSONDecodeError for README metadata {repo_identifier}")

            repo_details["last_scanned_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
            self._append_repo_to_csv(repo_details)
            self.scanned_repo_identifiers.add(repo_identifier)  # Add after successful processing and append
            # time.sleep(0.1) # Optional: be polite to the API if fetching many READMEs

    # _save_gov_repos_to_csv is no longer needed as we append incrementally.

    def scan_and_save_gov_repos_catalog(
        self,
        gov_accounts_url: Optional[str] = None,
        # output_filename: str = "gov_repositories_catalog.csv", # No longer needed
    ) -> Optional[str]:
        """
        Orchestrates fetching government account data, their repositories, and saving to a CSV catalog.
        """
        url_to_fetch = gov_accounts_url or GOVERNMENT_ACCOUNTS_URL
        logger.info(f"Starting scan for government repositories from {url_to_fetch}. Run ID: {self.run_id}")

        # Ensure catalog file exists with headers, especially for new runs or if it was deleted.
        if not self.catalog_filepath.exists() or self.catalog_filepath.stat().st_size == 0:
            logger.info(f"Catalog file {self.catalog_filepath} is new or empty. Creating with headers.")
            try:
                pd.DataFrame(columns=self.EXPECTED_CSV_COLUMNS).to_csv(self.catalog_filepath, index=False, encoding="utf-8")
            except Exception as e:
                logger.error(f"Failed to create initial catalog file {self.catalog_filepath}: {e}")
                return None

        if not self.api_token or "your_default_token" in self.api_token:
            logger.error(
                "A valid GitHub API token is required to fetch government repository details."
                "Please set GITHUB_API_TOKEN in your environment or config."
            )
            return None

        gov_yaml_data = self.fetch_gov_github_accounts_yaml(url_to_fetch)
        if not gov_yaml_data:
            return None

        # Use TARGET_REPOS from config for filtering if it's populated
        filter_to_apply = TARGET_REPOS if TARGET_REPOS else None
        gov_usernames = self._extract_usernames_from_gov_data(gov_yaml_data, filter_list=filter_to_apply)

        if not gov_usernames:
            log_message = "No government GitHub usernames found to scan."
            if filter_to_apply:
                log_message += f" (Filter applied: {filter_to_apply})"
            logger.info(log_message)
            return str(self.catalog_filepath)  # Return path to the (potentially empty) catalog

        for i, username in enumerate(gov_usernames):
            logger.info(f"Processing account {i+1}/{len(gov_usernames)}: {username}")
            # _fetch_repository_details_for_account now writes directly to CSV and handles skipping
            self._fetch_repository_details_for_account(username)
            # time.sleep(1) # Be polite to the GitHub API, especially if iterating many accounts

        logger.info(f"Government repositories scan complete. Catalog updated at: {self.catalog_filepath}")
        return str(self.catalog_filepath)
