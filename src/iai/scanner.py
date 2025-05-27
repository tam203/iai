import json
import logging
from typing import List, Dict, Any, Optional
import time

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

    def __init__(self, run_id: str, api_token: str = None):
        self.run_id = run_id
        self.api_token = api_token or GITHUB_API_TOKEN
        if not self.api_token or "your_default_token" in self.api_token:
            logger.warning(
                "GitHub API token is not set or is using a default placeholder. API rate limits may be encountered."
            )


    # --- Methods for Government Repositories Scan ---

    def _fetch_gov_github_accounts_yaml(
        self, url: str
    ) -> Optional[Dict[str, List[str]]]:
        """Fetches and parses the YAML file listing government GitHub accounts."""
        logger.info(f"Fetching government GitHub accounts from: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            data = yaml.safe_load(response.text)
            if not isinstance(data, dict):
                logger.error(
                    f"Government accounts YAML from {url} did not parse into a dictionary."
                )
                return None
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching government accounts YAML from {url}: {e}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing government accounts YAML from {url}: {e}")
        return None

    def _extract_usernames_from_gov_data(
        self, gov_data: Dict[str, List[str]], filter_list: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extracts unique GitHub usernames/organization names from the government data structure.
        The expected structure for gov_data is Dict[str, List[str]] (e.g., {'Country': ['org1', 'org2']}).
        Optionally filters the usernames against a provided list.
        """
        usernames = set()
        if not isinstance(gov_data, dict):
            logger.warning(
                "Government data is not in the expected dictionary format (country: [orgs])."
            )
            return []

        apply_filter = bool(
            filter_list
        )  # True if filter_list is not None and not empty

        for country_name, org_list in gov_data.items():
            if not isinstance(org_list, list):
                logger.warning(
                    f"Expected a list of orgs for country '{country_name}', but got {type(org_list)}. Skipping."
                )
                continue
            for username in org_list:
                if isinstance(username, str) and (
                    not apply_filter or (filter_list and username in filter_list)
                ):
                    usernames.add(username)
        logger.info(
            f"Extracted {len(usernames)} unique usernames. Filter applied: {apply_filter} (using {len(filter_list or [])} filter items)."
        )
        return list(usernames)

    def _fetch_repository_details_for_account(
        self, username: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetches repository details for a given GitHub username/organization."""
        if not self.api_token or "your_default_token" in self.api_token:
            logger.error(
                f"A valid GitHub API token is required to fetch repository details for {username}. Skipping."
            )
            return None

        headers = {"Authorization": f"token {self.api_token}"}
        repos_url = f"https://api.github.com/users/{username}/repos?per_page=100"  # Adjust per_page as needed
        logger.info(f"Fetching repositories for account: {username} from {repos_url}")

        try:
            repos_response = requests.get(repos_url, headers=headers, timeout=30)
            repos_response.raise_for_status()
            repos_data = repos_response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching repositories for {username}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for {username}'s repositories: {e}")
            return None

        full_repo_details = []
        logger.info(
            f"Found {len(repos_data)} repositories for {username}. Fetching details..."
        )

        for repo in repos_data:
            if not isinstance(repo, dict):
                continue  # Basic type check
            repo_details = {
                "name": repo.get("name", "N/A"),
                "description": repo.get("description") or "No description",
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "language": repo.get("language") or "None specified",
                "url": repo.get("html_url", ""),
            }

            # Fetch the README file
            readme_url = (
                f"https://api.github.com/repos/{username}/{repo_details['name']}/readme"
            )
            repo_details["readme_snippet"] = "README not available"  # Default
            try:
                readme_api_response = requests.get(
                    readme_url, headers=headers, timeout=10
                )
                if readme_api_response.status_code == 200:
                    readme_api_data = readme_api_response.json()
                    download_url = readme_api_data.get("download_url")
                    if download_url:
                        readme_content_response = requests.get(download_url, timeout=10)
                        if readme_content_response.status_code == 200:
                            repo_details["readme_snippet"] = (
                                readme_content_response.text[:150]
                            )  # Truncate
                        else:
                            logger.debug(
                                f"Failed to download README for {username}/{repo_details['name']}, status: {readme_content_response.status_code}"
                            )
                    else:
                        logger.debug(
                            f"No download_url for README of {username}/{repo_details['name']}"
                        )
                elif (
                    readme_api_response.status_code != 404
                ):  # Log if not a simple "Not Found"
                    logger.warning(
                        f"Failed to fetch README metadata for {username}/{repo_details['name']}, status: {readme_api_response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"RequestException for README {username}/{repo_details['name']}: {e}"
                )
            except json.JSONDecodeError:
                logger.warning(
                    f"JSONDecodeError for README metadata {username}/{repo_details['name']}"
                )

            full_repo_details.append(repo_details)
            # time.sleep(0.1) # Optional: be polite to the API if fetching many READMEs

        logger.info(
            f"Fetched details for {len(full_repo_details)} repositories for account: {username}"
        )
        return full_repo_details

    def _save_gov_repos_to_csv(
        self, gov_repos_data: List[Dict[str, Any]], output_filename: str
    ) -> str:
        """Saves the fetched government repository data to a CSV file."""
        output_path = get_filepath_in_run_data(self.run_id, output_filename)

        if not gov_repos_data:
            logger.info(
                f"No government repository data to save. Creating an empty CSV: {output_path}"
            )
            # Define columns for an empty CSV to ensure consistency
            columns = [
                "account",
                "name",
                "description",
                "stars",
                "forks",
                "language",
                "url",
                "readme_snippet",
            ]
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.DataFrame(gov_repos_data)
            # Optionally, ensure specific column order if needed, though DataFrame constructor from list of dicts is usually fine.
            # columns_order = ['account', 'name', 'description', 'stars', 'forks', 'language', 'url', 'readme_snippet']
            # df = df.reindex(columns=columns_order) # Ensures all columns exist, in order, filling missing with NaN

        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(
            f"Saved government repositories catalog ({len(df)} rows) to {output_path}"
        )
        return str(output_path)

    def scan_and_save_gov_repos_catalog(
        self,
        gov_accounts_url: Optional[str] = None,
        output_filename: str = "gov_repositories_catalog.csv",
    ) -> Optional[str]:
        """
        Orchestrates fetching government account data, their repositories, and saving to a CSV catalog.
        """
        url_to_fetch = gov_accounts_url or GOVERNMENT_ACCOUNTS_URL
        logger.info(
            f"Starting scan for government repositories from {url_to_fetch}. Run ID: {self.run_id}"
        )

        if not self.api_token or "your_default_token" in self.api_token:
            logger.error(
                "A valid GitHub API token is required to fetch government repository details. Please set GITHUB_API_TOKEN in your environment or config."
            )
            return None

        gov_yaml_data = self._fetch_gov_github_accounts_yaml(url_to_fetch)
        if not gov_yaml_data:
            return None

        # Use TARGET_REPOS from config for filtering if it's populated
        filter_to_apply = TARGET_REPOS if TARGET_REPOS else None
        gov_usernames = self._extract_usernames_from_gov_data(
            gov_yaml_data, filter_list=filter_to_apply
        )

        if not gov_usernames:
            log_message = "No government GitHub usernames found to scan."
            if filter_to_apply:
                log_message += " (After applying filter based on TARGET_REPOS)"
            logger.info(log_message)
            return None

        # TODO: This would be better if done in parallel and also if results were written out as went and aggregated at the end.
        all_repos_details = []
        for i, username in enumerate(gov_usernames):
            logger.info(f"Processing account {i+1}/{len(gov_usernames)}: {username}")
            repo_details_for_account = self._fetch_repository_details_for_account(
                username
            )
            if repo_details_for_account:
                for detail in repo_details_for_account:
                    detail["account"] = username  # Add account context
                all_repos_details.extend(repo_details_for_account)
            # time.sleep(1) # Be polite to the GitHub API, especially if iterating many accounts

        if not all_repos_details:
            logger.info(
                "No repository details were fetched from any government account."
            )
            # Save an empty catalog file with headers
            return self._save_gov_repos_to_csv([], output_filename)

        return self._save_gov_repos_to_csv(all_repos_details, output_filename)
