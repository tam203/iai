import logging
import time

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm
from langchain_core.output_parsers import StrOutputParser

from iai.config import GOOGLE_API_KEY, LLM_MODEL_NAME
from iai.classifiers.base import ClassifierInterface

logger = logging.getLogger(__name__)


class OneAtATimeClassifier(ClassifierInterface):
    def __init__(self, run_id: str, rate_limit_seconds: float):
        """
        Initializes the classifier and LLM.

        Args:
            run_id: The identifier for the run.
            rate_limit_seconds: Seconds to wait between LLM calls to avoid quota
                                limits.
        """
        super().__init__(run_id)
        self.google_api_key = GOOGLE_API_KEY
        self.rate_limit_seconds = rate_limit_seconds

    def classify_repos(self, df, topic_list, summaries_output_path: str):
        logger.info("Starting one-at-a-time classification (assuming summaries are already generated)...")
        if "summary" not in df.columns:
            raise ValueError("DataFrame must contain a 'summary' column for one-at-a-time classification.")

        # Initialize Google Gemini client
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=self.google_api_key,
        )

        topic_prompt = PromptTemplate(
            input_variables=["summary", "topic_list"],
            template="""
            Given the following summary of a GitHub repository and a list of potential topics,
            select the most appropriate topic for the repository. If none of the topics in the list are suitable,
            provide a new, concise topic label.

            Repository summary:
            {summary}

            Potential topics:
            {topic_list}

            Return only the selected or new topic label, and nothing else.
            Topic:
            """,
        )

        topic_chain = topic_prompt | llm | StrOutputParser()

        def classify_row(row):
            """
            Classifies a repository based on its summary.
            """
            summary = row["summary"]
            topic = topic_chain.invoke({"summary": summary, "topic_list": ", ".join(topic_list)})

            if self.rate_limit_seconds > 0:
                logger.debug(f"Rate limiting topic classification: sleeping for {self.rate_limit_seconds}s.")
                time.sleep(self.rate_limit_seconds)

            # Update master topic list if a new topic is found
            if topic not in topic_list:
                topic_list.append(topic)

            return topic

        tqdm.pandas(desc="Classifying Repositories")
        df["topic"] = df.progress_apply(classify_row, axis=1)

        return df, topic_list
