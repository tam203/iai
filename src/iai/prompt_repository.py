import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Assumes this file is in src/iai/
PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptType(str, Enum):
    """Enumeration of available prompt types for LLM tasks."""

    SUMMARY_DEFAULT = "summary_default"
    SUMMARY_SHORT = "summary_short"

    def __str__(self):
        """Returns the string value, which is used by argparse."""
        return self.value

    def get_path(self) -> Path:
        """Returns the file path for the prompt."""
        return PROMPTS_DIR / f"{self.value}.txt"

    def get_content(self) -> str:
        """Reads and returns the content of the prompt file."""
        prompt_path = self.get_path()
        logger.debug(f"Loading prompt '{self.name}' from {prompt_path}")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found for {self.name} at {prompt_path}")
            raise
