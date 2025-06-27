import logging
import time
import json
from typing import List, Tuple, Dict, Any

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

from iai.classifiers.base import ClassifierInterface
from iai.config import GOOGLE_API_KEY, GOALS, LLM_MODEL_NAME


logger = logging.getLogger(__name__)


class GoalBasedClassifier(ClassifierInterface):
    """
    A classifier that evaluates repositories against a predefined set of goals
    using an LLM, providing a score and a reason for each goal.
    """

    def __init__(self, run_id: str, rate_limit_seconds: float):
        """
        Initializes the classifier and LLM.

        Args:
            run_id: The identifier for the run.
            rate_limit_seconds: Seconds to wait between LLM calls to avoid quota
                                limits.
        """
        super().__init__(run_id)
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
        )
        self.rate_limit_seconds = rate_limit_seconds
        self.goals = GOALS

    def _evaluate_repo_for_goals(self, summary: str) -> Dict[str, Any]:
        """
        Asks the LLM to evaluate a repository summary against all defined goals,
        providing a score (0-5) and a brief reason (1-3 sentences) for each.
        """
        goals_formatted = "\n".join([f"- {g['name']}: {g['description']}" for g in self.goals])

        prompt_template = PromptTemplate(
            input_variables=["summary", "goals_formatted"],
            template="""
            You are an expert in public sector technology and innovation.
            Given the following summary of a GitHub repository, evaluate its relevance to each of the provided goals.
            For each goal, provide a score from 0 (no relevance) to 5 (highly relevant) and a brief reason (1-3 sentences) for that score.

            Repository Summary:
            {summary}

            Goals to evaluate:
            {goals_formatted}

            Output your response as a JSON object. The keys of the JSON object should be the goal names.
            Each value should be an object with two fields: "score" (integer, 0-5) and "reason" (string, 1-3 sentences).
            Example format:
            {{
                "Public Sector Innovation": {{"score": 4, "reason": "This project introduces a novel approach to data sharing, fostering innovation."}},
                "Citizen Engagement": {{"score": 2, "reason": "While not directly citizen-facing, it could indirectly improve service delivery."}}
            }}
            Ensure the output is valid JSON and contains evaluations for ALL listed goals.
            JSON Output:
            """,
        )

        chain = prompt_template | self.llm | StrOutputParser()

        try:
            raw_output = chain.invoke({"summary": summary, "goals_formatted": goals_formatted})
            # Attempt to parse JSON. Sometimes LLMs add conversational text before/after JSON.
            # Try to find the first and last curly brace to extract pure JSON.
            json_start = raw_output.find("{")
            json_end = raw_output.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = raw_output[json_start : json_end + 1]
                return json.loads(json_string)
            else:
                logger.error(f"Could not find valid JSON in LLM output: {raw_output}")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}\nRaw output: {raw_output}")
            return {}
        except Exception as e:
            logger.error(f"Error during LLM call for goal evaluation: {e}")
            return {}

    def classify_repos(self, df: pd.DataFrame, topic_list: List[str], summaries_output_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Evaluates repositories against predefined goals and adds score/reason columns.

        Args:
            df: A pandas DataFrame with 'description' and 'readme' columns.
            topic_list: An initial list of topics (ignored by this classifier).
            summaries_output_path: The path to the CSV file where summaries
                                   are read from and written to.

        Returns:
            A tuple containing:
            - pd.DataFrame: The input DataFrame augmented with goal score and reason columns.
            - List[str]: An empty list, as this classifier doesn't generate traditional topics.
        """
        if df.empty:
            return df, []
        logger.info("Starting goal-based classification (assuming summaries are already generated)...")
        logger.info("Step 2: Evaluating repositories against defined goals...")
        tqdm.pandas(desc="Evaluating Repositories for Goals")

        # Prepare new columns
        for goal in self.goals:
            df[f"{goal['name'].replace(' ', '_').lower()}_score"] = pd.NA
            df[f"{goal['name'].replace(' ', '_').lower()}_reason"] = pd.NA

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Repositories"):
            summary = row["summary"]
            if pd.isna(summary) or summary == "Error generating summary.":
                logger.warning(f"Skipping goal evaluation for row {index} due to missing or error summary.")
                continue

            evaluation_results = self._evaluate_repo_for_goals(summary)

            for goal in self.goals:
                goal_name = goal["name"]
                col_prefix = goal_name.replace(" ", "_").lower()
                if goal_name in evaluation_results:
                    df.loc[index, f"{col_prefix}_score"] = evaluation_results[goal_name].get("score", pd.NA)
                    df.loc[index, f"{col_prefix}_reason"] = evaluation_results[goal_name].get("reason", pd.NA)
                else:
                    logger.warning(f"LLM output missing evaluation for goal '{goal_name}' for row {index}.")
                    df.loc[index, f"{col_prefix}_score"] = 0  # Default to 0 if missing
                    df.loc[index, f"{col_prefix}_reason"] = "Evaluation missing from LLM output."

            if self.rate_limit_seconds > 0:
                logger.debug(f"Rate limiting goal evaluation: sleeping for {self.rate_limit_seconds}s.")
                time.sleep(self.rate_limit_seconds)

        logger.info("Goal-based classification complete.")
        # Return an empty list for topics, as this classifier doesn't generate them in the traditional sense
        return df, []
