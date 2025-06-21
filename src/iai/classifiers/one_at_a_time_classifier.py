from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

from iai.config import GOOGLE_API_KEY  # Assuming GOOGLE_API_KEY is in iai.config
from iai.classifiers.base import ClassifierInterface


class OneAtATimeClassifier(ClassifierInterface):

    def __init__(self, run_id: str):
        self.run_id = run_id
        # Assuming GOOGLE_API_KEY is set in iai.config or environment
        self.google_api_key = GOOGLE_API_KEY

    def classify_repos(self, df, topic_list):
        # Initialize Azure OpenAI client
        # Initialize Google Gemini client
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # TODO: move to config
            google_api_key=self.google_api_key,
        )

        summary_prompt = PromptTemplate(
            input_variables=["description", "readme"],
            template="""
            Please provide a summary of the following GitHub repository based on its description and README.md content.
            If the README.md is not in English, please first translate it to English and then generate a summary.
            The summary should be concise and in fewer than 5 sentences.

            Repository description:
            {description}

            README.md content:
            {readme}

            Summary:
            """,
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

        def classify_and_summarize_row(row):
            """
            Generates a summary and classifies a repository based on a DataFrame row.
            This function is applied once per row to avoid redundant API calls.
            """
            # Chain 1: Generate Summary
            description = row["description"] if pd.notnull(row["description"]) else ""
            readme = row["readme"] if pd.notnull(row["readme"]) else ""
            summary_chain = summary_prompt | llm | StrOutputParser()
            summary = summary_chain.invoke({"description": description, "readme": readme})

            # Chain 2: Classify Topic using the generated summary
            topic_chain = topic_prompt | llm | StrOutputParser()
            topic = topic_chain.invoke({"summary": summary, "topic_list": ", ".join(topic_list)})

            # Update master topic list if a new topic is found
            if topic not in topic_list:
                topic_list.append(topic)

            # Return a Series to efficiently create both columns in the DataFrame
            return pd.Series([topic, summary], index=["topic", "summary"])

        tqdm.pandas(desc="Classifying and Summarizing Repositories")
        # Apply the combined function once per row, creating two new columns
        df[["topic", "summary"]] = df.progress_apply(classify_and_summarize_row, axis=1)

        return df, topic_list
