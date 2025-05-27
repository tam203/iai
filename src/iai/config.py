import os
import dotenv

dotenv.load_dotenv()

# Example: Get GitHub token from environment variable
# In a real scenario, ensure this is handled securely.
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN", "your_default_token_here_if_any")

DEFAULT_SEARCH_QUERIES = ["topic:python language:python stars:>1000", "org:google language:go"]

GOVERNMENT_ACCOUNTS_URL = "https://raw.githubusercontent.com/github/government.github.com/gh-pages/_data/governments.yml"

TARGET_REPOS = [
    'alphagov', 'i-dot-ai', 'canada-ca', 'govtechsg', 'GSA', 'ec-europa', 'opengovsg'
]