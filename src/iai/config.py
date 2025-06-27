import os
import dotenv

dotenv.load_dotenv()

# Example: Get GitHub token from environment variable
# In a real scenario, ensure this is handled securely.
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
ENDPOINT_URL = os.getenv("ENDPOINT_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_RATE_LIMIT_SECONDS = float(os.getenv("LLM_RATE_LIMIT_SECONDS", "0"))
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")  # gemini-2.5-flash-lite-preview-06-17
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/text-embedding-004")


DEFAULT_SEARCH_QUERIES = [
    "topic:python language:python stars:>1000",
    "org:google language:go",
]

GOVERNMENT_ACCOUNTS_URL = "https://raw.githubusercontent.com/github/government.github.com/gh-pages/_data/governments.yml"

TARGET_REPOS = None
# TARGET_REPOS = [
#     "alphagov",
#     "i-dot-ai",
#     "canada-ca",
#     "govtechsg",
#     "GSA",
#     "ec-europa",
#     "opengovsg",
# ]

TOPIC_LIST = [
    "ai",
    "data",
    "data science",
    "cybersecurity",
    "ai agents",
    "ai ethics",
    "ai regulation",
    "automation",
    "innovation",
]

GOALS = [
    {
        "name": "Economic Growth and Stability",
        "description": "Contributes to a stronger UK economy by providing tools"
        " for financial analysis, supporting businesses, or increasing public "
        "sector efficiency to help reduce national debt and control inflation.",
    },
    {
        "name": "Healthcare Modernisation",
        "description": "Supports the NHS by developing digital tools,"
        " data analytics, or new technologies to improve patient care,"
        " streamline operations, and cut waiting lists.",
    },
    {
        "name": "National Security and Border Control",
        "description": "Enhances the UK's national security through projects related to border management,"
        " law enforcement technology, cybersecurity, or defence systems.",
    },
    {
        "name": "Energy Security and Net Zero",
        "description": "Aids the UK's transition to Net Zero and enhances energy security through renewable energy technology,"
        " grid management, or climate data analysis.",
    },
    {
        "name": "Education and Skills",
        "description": "Advances the UK's education system and workforce skills through educational technology,"
        " data analysis on attainment, or platforms for vocational training.",
    },
]
