"""
llm.py

This module defines the LLMQuery class that handles interactions with the language model (LLM),
including sending queries and creating a question-answering chain.
"""

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import Config


class LLMQuery:
    def __init__(self):
        """
        Initializes the LLMQuery class and sets up the LLM using configuration values.
        """
        self.llm = init_chat_model(
            model=Config.LLM_MODEL,
            model_provider=Config.MODEL_PROVIDER,
            api_key=Config.API_KEY,
        )

    def create_qa_chain(self) -> tuple:
        """
        Creates a QA chain combining the prompt template, LLM, and output parser.

        """
        template = (
            "Answer the question based only on the following context:\n"
            "{context}\n\n"
            "Question: {evaluation_query}\n"
        )
        prompt = ChatPromptTemplate.from_template(template)

        return prompt | self.llm | StrOutputParser(), self.llm

    def send_query(
        self, evaluation_query: str, relevant_docs: list[Document], qa_chain: tuple
    ) -> str:
        """
        Sends a query to the LLM using the context and query string.

        """
        response = qa_chain.invoke(
            {
                "context": format_docs(relevant_docs),
                "evaluation_query": evaluation_query,
            }
        )
        return response


def format_docs(relevant_docs: list[Document]) -> str:
    """
    Formats a list of Document objects into a single string.
    :param relevant_docs: List of Document objects.
    :return: Combined string of all document contents.
    """
    return "\n".join(doc.page_content for doc in relevant_docs)