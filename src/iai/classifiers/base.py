# /home/theo/projects/iai/src/iai/classifiers/base.py
from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd


class ClassifierInterface(ABC):
    """
    Interface for repository classification.
    """

    def __init__(self, run_id: str):
        """
        Initializes the classifier with a run ID.

        Args:
            run_id: The unique identifier for the current execution run.
        """
        self.run_id = run_id

    @abstractmethod
    def classify_repos(self, df: pd.DataFrame, topic_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Classifies repositories based on their data and a given list of topics.

        Args:
            df: A pandas DataFrame containing repository information.
                It's expected to have columns like 'description' and 'readme'
                that the classifier can use.
            topic_list: A list of predefined topics. The classifier might
                        add new topics to this list if it discovers them.

        Returns:
            A tuple containing:
            - pd.DataFrame: The input DataFrame, augmented with a 'topic'
                            column and potentially other columns like 'summary'.
            - List[str]: The (potentially updated) list of topics.
        """
        pass
