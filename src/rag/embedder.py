from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List
import os

class Embedder:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Inicializa el embedder usando OpenAI API.
        :param model_name: Nombre del modelo de embeddings de OpenAI.
        """
        self.embedder = OpenAIEmbeddings(model=model_name)

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Genera embeddings para una lista de documentos.
        :param documents: Lista de objetos Document de LangChain.
        :return: Lista de embeddings, cada embedding es una lista de floats.
        """
        texts = [doc.page_content for doc in documents]
        return self.embedder.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """
        Genera el embedding de una query individual.
        :param query: Texto de la query a vectorizar.
        :return: Embedding de la query como lista de floats.
        """
        return self.embedder.embed_query(query)
