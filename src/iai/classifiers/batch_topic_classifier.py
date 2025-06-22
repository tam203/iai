import logging
from typing import List, Tuple
import time

import hdbscan
import numpy as np
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from iai.classifiers.base import ClassifierInterface
from iai.config import GOOGLE_API_KEY
from iai.llm_utils import generate_summaries


logger = logging.getLogger(__name__)


class BatchTopicClassifier(ClassifierInterface):
    """
    A classifier that analyzes a batch of repositories to dynamically generate
    topics using clustering, rather than classifying them one by one against a
    predefined list.
    """

    def __init__(self, run_id: str, rate_limit_seconds: float):
        """
        Initializes the classifier, embedding model, and LLM.

        Args:
            run_id: The identifier for the run.
            rate_limit_seconds: Seconds to wait between LLM calls to avoid quota
                                limits.
        """
        super().__init__(run_id)
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
        self.rate_limit_seconds = rate_limit_seconds

    def _generate_topic_label(self, summaries: List[str]) -> str:
        """Asks LLM to generate a topic for a cluster of summaries."""
        prompt = PromptTemplate(
            input_variables=["summaries"],
            template="""The following are summaries of similar GitHub projects:
            ---
            - {summaries}
            ---
            What is a short, descriptive topic label (3 words maximum) for this group of projects?
            Return only the label, and nothing else. For example: "Web Development Tools" or "Data Visualization".
            Topic Label:""",
        )
        chain = prompt | self.llm | StrOutputParser()
        # Join with newlines for better formatting in the prompt
        summaries_str = "\n- ".join(summaries)
        return chain.invoke({"summaries": summaries_str}).strip()

    def classify_repos(self, df: pd.DataFrame, topic_list: List[str], summaries_output_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Classifies repositories by clustering them based on their content and
        then generating topic labels for those clusters.

        The input `topic_list` is ignored as topics are generated dynamically.

        Args:
            df: A pandas DataFrame with 'description' and 'readme' columns.
            topic_list: An initial list of topics (ignored by this classifier).
            summaries_output_path: The path to the CSV file where summaries
                                   are read from and written to.

        Returns:
            A tuple containing:
            - pd.DataFrame: The input DataFrame augmented with a 'topic' column.
            - List[str]: The new list of topics generated from the data.
        """
        if df.empty:
            df["topic"] = None
            return df, []

        logger.info("Starting batch topic classification...")

        # --- Step 1: Generate Summaries ---
        logger.info("Step 1: Generating summaries for repositories...")
        df = generate_summaries(df, output_csv_path=summaries_output_path, rate_limit_seconds=self.rate_limit_seconds)
        logger.info("Summaries generated.")

        # --- Step 2: Generate Embeddings ---
        logger.info("Step 2: Generating embeddings from summaries...")
        summaries = df["summary"].tolist()
        embeddings = self.embedding_model.embed_documents(summaries)
        logger.info(f"Embeddings generated for {len(embeddings)} summaries.")

        # --- Step 3: Cluster the Embeddings ---
        logger.info("Step 3: Clustering embeddings using HDBSCAN...")
        # min_cluster_size can be tuned. A smaller value allows for more niche topics.
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric="euclidean", gen_min_span_tree=True)
        cluster_labels = clusterer.fit_predict(np.array(embeddings))
        df["cluster_id"] = cluster_labels
        logger.info(f"Clustering complete. Found {len(np.unique(cluster_labels)) - 1} topics (excluding outliers).")

        # --- Step 4: Generate Topic Labels for Each Cluster ---
        logger.info("Step 4: Generating topic labels for each cluster using LLM...")
        unique_clusters = sorted(df["cluster_id"].unique())
        topic_map = {}
        generated_topics = []

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                topic_map[cluster_id] = "Uncategorized"
                continue

            # Get representative summaries for the cluster
            cluster_df = df[df["cluster_id"] == cluster_id]
            sample_size = min(5, len(cluster_df))
            cluster_summaries = cluster_df["summary"].sample(n=sample_size, random_state=42).tolist()

            topic_label = self._generate_topic_label(cluster_summaries)
            logger.info(f"Generated label for cluster {cluster_id}: '{topic_label}'")
            topic_map[cluster_id] = topic_label
            generated_topics.append(topic_label)

            if self.rate_limit_seconds > 0:
                logger.debug(f"Rate limiting: sleeping for {self.rate_limit_seconds}s.")
                time.sleep(self.rate_limit_seconds)

        # --- Step 5: Assign Topic Labels ---
        logger.info("Step 5: Assigning topic labels to repositories...")
        df["topic"] = df["cluster_id"].map(topic_map)
        df = df.drop(columns=["cluster_id"])

        final_topic_list = sorted(list(set(generated_topics + ["Uncategorized"])))

        return df, final_topic_list
