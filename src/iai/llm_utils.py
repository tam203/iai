import logging
import time

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm

from iai.config import GOOGLE_API_KEY, LLM_MODEL_NAME
from iai.prompt_repository import PromptType

logger = logging.getLogger(__name__)


def generate_summaries(
    df: pd.DataFrame,
    output_csv_path: str,
    prompt_type: PromptType,
    rate_limit_seconds: float = 0.0,
) -> pd.DataFrame:
    """
    Generates summaries for each repository in the DataFrame.

    Summaries are written to the output CSV file as they are generated to
    prevent data loss on interruption. If a 'summary' column already exists,
    it only generates summaries for rows where the summary is missing.

    Args:
        df: DataFrame with 'description' and 'readme' columns.
        output_csv_path: Path to the output CSV file. Summaries will be
                         written here progressively.
        prompt_type: The type of prompt to use for summarization.
        rate_limit_seconds: Seconds to wait between LLM calls to avoid quota
                            limits. Defaults to 0.0.

    Returns:
        DataFrame with an added or updated 'summary' column.
    """
    if df.empty:
        return df

    if "summary" not in df.columns:
        df["summary"] = pd.NA

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
    )

    prompt_content = prompt_type.get_content()
    summary_prompt = PromptTemplate(
        input_variables=["description", "readme"],
        template=prompt_content,
    )

    summary_chain = summary_prompt | llm | StrOutputParser()

    rows_to_process = df[df["summary"].isnull()]

    if rows_to_process.empty:
        logger.info("All repositories already have summaries. No new summaries to generate.")
        return df

    logger.info(f"Generating summaries for {len(rows_to_process)} repositories...")

    save_interval = 40  # Save progress every 40 summaries
    rows_since_last_save = 0

    for index, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="Generating Summaries"):
        description = row["description"] if pd.notnull(row["description"]) else ""
        readme = row["readme"] if pd.notnull(row["readme"]) else ""
        try:
            summary = summary_chain.invoke({"description": description, "readme": readme})
            if rate_limit_seconds > 0:
                logger.debug(f"Rate limiting summary generation: sleeping for {rate_limit_seconds}s.")
                time.sleep(rate_limit_seconds)
            df.loc[index, "summary"] = summary
        except Exception as e:
            logger.error(f"Error generating summary for row index {index}: {e}")
            df.loc[index, "summary"] = "Error generating summary."

        rows_since_last_save += 1
        # Save progress periodically to avoid data loss on interruption without
        # incurring the I/O overhead of saving on every single iteration.
        if rows_since_last_save >= save_interval:
            df.to_csv(output_csv_path, index=False, encoding="utf-8")
            rows_since_last_save = 0

    # Perform a final save to ensure all generated summaries are written to disk.
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    logger.info("Summary generation complete.")
    return df
