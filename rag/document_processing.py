""" 
document_processing.py

Handles document processing for a RAG pipeline.  
Extracts text from PDFs, converts them into Document objects,  
and initializes a Chroma vector store with OpenAI embeddings.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from uuid import uuid4
import PyPDF2

from config import Config


class DocumentProcessor:
    def __init__(self):
        """
        Initialize the DocumentProcessor.
        """
        self.embeddings = OpenAIEmbeddings(
            api_key=Config.API_KEY, model=Config.EMBEDDING_MODEL
        )
        self.vector_store = self.init_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 1})

    def init_vector_store(self):
        """
        Create embeddings and load them to vector db.
        :return: A vector store that contains the indexed documents.
        """
        vector_store = Chroma(
            collection_name="paper_collection",
            embedding_function=self.embeddings,
            persist_directory="./chroma_langchain_db",
        )

        return vector_store

    def read_pdf(self, file) -> str:
        """
        Read pdf document, and extract its text.
        :param file: uploaded PDF file.
        :return: Extracted text from the PDF.
        """
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n".join(page.extract_text() or " " for page in pdf_reader.pages)
        return text

    def create_document(self, content_list: list[str]):
        """
        Create Document objects for a list of strings
        :param content_list: List of strings representing document content.
        :return: List of Document objects.
        """
        return [Document(page_content=content) for content in content_list]

    def load_documents(self, documents: list[Document]) -> None:
        """
        Load list of Document objects to vector_store.
        :param vector_store: Vector store retriever.
        :param documents: List of Document objects.
        """
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents, uuids=uuids)