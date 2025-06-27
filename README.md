# iai
A repo for an i dot ai assessment. This has code for scanning repos and clasifing their topics.

Some of the code is based off https://github.com/lexicond/gov_tech_aggregator
Some is AI Generated (Google Gemini)


## UV

This project uses `uv` for package management.

See: [uv Getting started](https://docs.astral.sh/uv/getting-started/).

## Some useful commands

Install local version of package
`uv pip install -e .`

Install pre-commit hooks
`uv run -- pre-commit install`

Run all pre-commit checks on all files
`uv run -- pre-commit run --all-files`

Add a dev dependency e.g. `flake8`
`uv add --dev flake8`

Run notebooks
`uv run -- jupyter jupyter lab --notebook-dir=notebooks`

Run the scanning of GitHub orgs/repos. Use `--run_id` to update existing run / recover from errors

The full pipeline involves three steps:

1.  **Scan GitHub repositories:**
    `uv run -- python src/iai/run_scan.py --run_id <YOUR_RUN_ID>`
2.  **Generate LLM summaries for repositories:**
    `uv run -- python src/iai/run_summarization.py --run_id <YOUR_RUN_ID>`
3.  **Classify repositories using LLM:**
    `uv run -- python src/iai/run_classification.py --run_id <YOUR_RUN_ID> --classifier [one_at_a_time|batch|goal_based]`

Example usage for a full pipeline run:
`uv run -- python src/iai/run_scan.py --run_id 20250618_153140_a475925c`
`uv run -- python src/iai/run_summarization.py --run_id 20250618_153140_a475925c`
`uv run -- python src/iai/run_classification.py --run_id 20250618_153140_a475925c --classifier batch`
