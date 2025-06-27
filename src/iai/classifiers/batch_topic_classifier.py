import logging
from typing import List, Tuple
import ast
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
from iai.config import EMBEDDING_MODEL_NAME, GOOGLE_API_KEY, LLM_MODEL_NAME


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
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
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

        logger.info("Starting batch topic classification (assuming summaries are already generated)...")

        # --- Step 2: Generate/Load Embeddings ---
        embedding_col = "embedding"
        if embedding_col not in df.columns:
            df[embedding_col] = pd.NA

        # Identify rows that need embeddings (where summary exists but embedding is missing)
        rows_to_embed_mask = df["summary"].notna() & df[embedding_col].isnull()
        rows_to_embed = df[rows_to_embed_mask]

        if not rows_to_embed.empty:
            logger.info(f"Step 2: Generating embeddings for {len(rows_to_embed)} new summaries...")
            summaries_to_embed = rows_to_embed["summary"].tolist()
            new_embeddings = self.embedding_model.embed_documents(summaries_to_embed)

            # Store as string representation of the list
            df.loc[rows_to_embed_mask, embedding_col] = [str(e) for e in new_embeddings]

            # Save the DataFrame with the new embeddings, overwriting the file
            logger.info(f"Saving DataFrame with new embeddings to {summaries_output_path}")
            try:
                df.to_csv(summaries_output_path, index=False, encoding="utf-8")
            except Exception as e:
                logger.error(f"Error saving CSV with embeddings to {summaries_output_path}: {e}")
        else:
            logger.info("Step 2: All embeddings already present in the file.")

        # --- Step 3: Parse Embeddings and Prepare for Clustering ---
        logger.info("Step 3: Parsing stored embeddings for clustering...")

        def parse_embedding(embedding_str: str) -> list | None:
            if pd.isnull(embedding_str):
                return None
            try:
                return ast.literal_eval(embedding_str)
            except (ValueError, SyntaxError):
                logger.warning(f"Could not parse embedding string, returning None. String: {embedding_str[:100]}...")
                return None

        df["parsed_embedding"] = df[embedding_col].apply(parse_embedding)
        df_to_cluster = df[df["parsed_embedding"].notna()].copy()
        if df_to_cluster.empty:
            logger.warning("No valid embeddings found to perform clustering. All repos will be 'Uncategorized'.")
            df["topic"] = "Uncategorized"
            df = df.drop(columns=["parsed_embedding"])
            return df, ["Uncategorized"]

        embeddings = np.array(df_to_cluster["parsed_embedding"].tolist())
        logger.info(f"Embeddings parsed for {len(embeddings)} summaries.")

        # --- Step 4: Cluster the Embeddings ---
        logger.info("Step 4: Clustering embeddings using HDBSCAN...")
        # min_cluster_size can be tuned. A smaller value allows for more niche topics.
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric="euclidean", gen_min_span_tree=True)
        cluster_labels = clusterer.fit_predict(embeddings)
        df_to_cluster["cluster_id"] = cluster_labels
        logger.info(f"Clustering complete. Found {len(np.unique(cluster_labels)) - 1} topics (excluding outliers).")

        # --- Step 5: Generate Topic Labels for Each Cluster ---
        logger.info("Step 5: Generating topic labels for each cluster using LLM...")
        unique_clusters = sorted(df_to_cluster["cluster_id"].unique())
        topic_map = {}
        generated_topics = []

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                topic_map[cluster_id] = "Uncategorized"
                continue

            # Get representative summaries for the cluster
            cluster_df = df_to_cluster[df_to_cluster["cluster_id"] == cluster_id]
            sample_size = min(5, len(cluster_df))
            cluster_summaries = cluster_df["summary"].sample(n=sample_size, random_state=42).tolist()

            topic_label = self._generate_topic_label(cluster_summaries)
            logger.info(f"Generated label for cluster {cluster_id}: '{topic_label}'")
            topic_map[cluster_id] = topic_label
            generated_topics.append(topic_label)

            if self.rate_limit_seconds > 0:
                logger.debug(f"Rate limiting: sleeping for {self.rate_limit_seconds}s.")
                time.sleep(self.rate_limit_seconds)

        # --- Step 6: Assign Topic Labels ---
        logger.info("Step 6: Assigning topic labels to repositories...")
        df_to_cluster["topic"] = df_to_cluster["cluster_id"].map(topic_map)

        # Merge the topic back into the original dataframe
        df = df.merge(df_to_cluster[["topic"]], left_index=True, right_index=True, how="left")
        df["topic"] = df["topic"].fillna("Uncategorized")

        df = df.drop(columns=["parsed_embedding"])

        final_topic_list = sorted(list(set(generated_topics + ["Uncategorized"])))

        return df, final_topic_list
