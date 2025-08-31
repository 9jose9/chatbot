# src/rag/retriever.py

from typing import List, Tuple
from langchain.docstore.document import Document

from .embedder import Embedder
from .vectorstore import FAISSVectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vectorstore: FAISSVectorStore):
        """
        Inicializa el retriever.

        :param embedder: Clase Embedder para generar embeddings.
        :param vectorstore: Clase FAISSVectorStore para recuperar información.
        """
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Dada una query, devuelve los documentos más relevantes del vectorstore.

        :param query: Pregunta del usuario.
        :param top_k: Número de resultados a devolver.
        :return: Lista de (Documento, score)
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.vectorstore.search(query_embedding, top_k=top_k)
        return results

    def format_context(self, results: List[Tuple[Document, float]]) -> str:
        """
        Convierte los documentos recuperados en un contexto de texto para el LLM.

        :param results: Documentos y scores devueltos por retrieve().
        :return: String concatenado con los contenidos.
        """
        context = ""
        for doc, score in results:
            context += f"[Fuente: {doc.metadata.get('source', 'desconocido')} | Score: {score:.4f}]\n"
            context += doc.page_content.strip() + "\n\n"
        return context.strip()
