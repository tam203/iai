# iai Project

This project provides a toolkit for scanning GitHub repositories, generating AI-powered summaries, and classifying them based on their content. It's designed to help users quickly understand and categorize large numbers of repositories.

The core functionality is a three-step pipeline:
1.  **Scan:** Collects repository metadata from GitHub.
2.  **Summarize:** Uses a Large Language Model (LLM) to generate a concise summary of each repository's purpose and functionality based on its README file and description.
3.  **Classify:** Categorizes each repository into predefined or dynamically generated topics using an LLM.

Some of the code is based off https://github.com/lexicond/gov_tech_aggregator and some is AI Generated (Google Gemini).

## Getting Started

### Prerequisites

*   Python 3.9+
*   [uv](https://docs.astral.sh/uv/getting-started/) for package management.
*   A GitHub API Token.
*   A Google API Key for LLM operations.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/iai.git
    cd iai
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

3.  **Set up your API keys:**
    Create a `.env` file in the project root and add your API keys:
    ```
    GITHUB_API_TOKEN="your_github_api_token"
    GOOGLE_API_KEY="your_google_api_key"
    ```

## Usage

The main workflow is a three-step pipeline. You must run the steps in order, using the same `run_id` to link the operations.

### Step 1: Scan GitHub Repositories

This step scans GitHub for repositories based on a query and creates a catalog of their metadata.

```bash
uv run -- python src/iai/run_scan.py [--run_id <YOUR_RUN_ID>]
```

**Arguments:**

*   `--run_id` (optional): A unique ID for the run. If you provide an existing `run_id`, the scan will be resumed or updated. If omitted, a new `run_id` will be generated.

### Step 2: Generate LLM Summaries

This step reads the scanned repositories, fetches their READMEs, and generates a summary for each one using an LLM.

```bash
uv run -- python src/iai/run_summarization.py --run_id <YOUR_RUN_ID> [options]
```

**Arguments:**

*   `--run_id` (required): The `run_id` from the scanning step.
*   `--input_csv` (optional): The input CSV file from the previous step. Defaults to `gov_repositories_catalog.csv`.
*   `--output_csv` (optional): The name of the output file. Defaults to `summarized_gov_repositories.csv`.
*   `--limit` (optional): The maximum number of repositories to summarize.
*   `--sort_by` (optional): The field to sort by before applying the limit (`stars` or `created_at`). Defaults to `created_at`.
*   `--rate_limit` (optional): Seconds to wait between LLM calls. Defaults to the value in `src/iai/config.py`.

### Step 3: Classify Repositories

This step classifies the summarized repositories into topics using an LLM.

```bash
uv run -- python src/iai/run_classification.py --run_id <YOUR_RUN_ID> --classifier <CLASSIFIER_TYPE> [options]
```

**Arguments:**

*   `--run_id` (required): The `run_id` from the previous steps.
*   `--classifier` (required): The classification strategy to use.
    *   `one_at_a_time`: Classifies each repository individually against a predefined list of topics.
    *   `batch`: Clusters all repositories and generates topics dynamically.
    *   `goal_based`: A more advanced classifier that attempts to determine the "goal" of the repository.
*   `--input_csv` (optional): The input CSV file from the summarization step. Defaults to `summarized_gov_repositories.csv`.
*   `--output_csv` (optional): The name of the output file. Defaults to `classified_gov_repositories_{classifier}.csv`.
*   `--limit` (optional): The maximum number of repositories to classify.
*   `--sort_by` (optional): The field to sort by before applying the limit (`stars` or `created_at`). Defaults to `created_at`.
*   `--rate_limit` (optional): Seconds to wait between LLM calls.

## Example Workflow

Here is an example of a full pipeline run:

1.  **Start the scan and get a new `run_id`:**
    ```bash
    uv run -- python src/iai/run_scan.py
    # Note the generated run_id from the output, e.g., 20250701_120000_abcdef12
    ```

2.  **Run the summarization:**
    ```bash
    uv run -- python src/iai/run_summarization.py --run_id 20250701_120000_abcdef12
    ```

3.  **Run the classification:**
    ```bash
    uv run -- python src/iai/run_classification.py --run_id 20250701_120000_abcdef12 --classifier batch
    ```

## Development

### Setting up for Development

1.  **Install development dependencies:**
    ```bash
    uv add --dev flake8
    uv add --dev jupyter
    ```

2.  **Install pre-commit hooks:**
    ```bash
    uv run -- pre-commit install
    ```

### Running Checks

To run all pre-commit checks on all files:
```bash
uv run -- pre-commit run --all-files
```

### Running Notebooks

To launch Jupyter Lab and work with the analysis notebooks:
```bash
uv run -- jupyter jupyter lab --notebook-dir=notebooks
```
