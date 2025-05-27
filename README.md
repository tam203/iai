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
