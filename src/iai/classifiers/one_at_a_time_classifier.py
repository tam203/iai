from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.exceptions import OutputParserException
import pandas as pd

from iai.config import GOOGLE_API_KEY  # Assuming GOOGLE_API_KEY is in iai.config
from iai.classifiers.base import ClassifierInterface


class OneAtATimeClassifier(ClassifierInterface):

    def __init__(self, run_id: str):
        self.run_id = run_id
        # Assuming GOOGLE_API_KEY is set in iai.config or environment
        self.google_api_key = GOOGLE_API_KEY

    def classify_repos(self, df, topic_list, model_name="gpt-4o"):
        # Initialize Azure OpenAI client
        # Initialize Google Gemini client
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # Use the specified Gemini model
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

        response_schemas = [ResponseSchema(name="topic", description="The selected topic for the repository.")]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        topic_prompt = PromptTemplate(
            input_variables=["summary", "topic_list"],
            template="""
            Given the following summary of a GitHub repository and a list of potential topics,
            select the most appropriate topic for the repository. If none of the topics in the list are suitable,
            generate a new topic label.

            Repository summary:
            {summary}

            Potential topics:
            {topic_list}

            Selected topic:
            {{topic}}

            PLEASE ONLY RETURN THE TOPIC LABEL AS THE RESPONSE. DO NOT INCLUDE ANY ADDITIONAL TEXT.
            """,
            output_parser=output_parser,
        )

        def generate_summary(row):
            description = row["description"] if pd.notnull(row["description"]) else ""
            readme = row["readme"] if pd.notnull(row["readme"]) else ""
            prompt = summary_prompt.format(description=description, readme=readme)
            summary = llm.invoke(prompt).content
            return summary

        def classify_repo(readme):
            row = df.loc[df["readme"] == readme].iloc[0]
            summary = generate_summary(row)
            prompt = topic_prompt.format(summary=summary, topic_list=", ".join(topic_list))
            response = llm.invoke(prompt).content

            try:
                topic = output_parser.parse(response)["topic"]
            except OutputParserException:
                # Handle the case when the response is not a valid JSON
                topic = response.strip()  # Extract the topic as a plain string

            if topic not in topic_list:
                topic_list.append(topic)

            return topic

        tqdm.pandas(desc="Classifying repositories")
        df["topic"] = df["readme"].progress_apply(classify_repo)

        df["summary"] = df.apply(generate_summary, axis=1)

        return df, topic_list
