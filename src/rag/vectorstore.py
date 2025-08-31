import faiss
import numpy as np
from langchain.docstore.document import Document
from typing import List
import os

class FAISSVectorStore:
    def __init__(self, index_path: str = "../data/faiss_index"):
        """
        :param index_path: carpeta para guardar/cargar el índice FAISS
        """
        self.index_path = index_path
        self.index = None
        self.documents: List[Document] = []

    def add_embeddings(self, embeddings: List[List[float]], documents: List[Document]):
        """
        Añade embeddings y documentos al índice FAISS.
        Detecta automáticamente la dimensión del índice.
        :param embeddings: Lista de embeddings a añadir.
        :param documents: Lista de documentos correspondientes a los embeddings.
        """
        embeddings = np.array(embeddings).astype("float32")

        if self.index is None:
            # Crear índice con la dimensión correcta
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            print(f"📏 Creando índice FAISS con dimensión: {embeddings.shape[1]}")

        # Comprobar dimensiones
        assert embeddings.shape[1] == self.index.d, f"Dim {embeddings.shape[1]} != index {self.index.d}"

        # Añadir embeddings al índice
        self.index.add(embeddings)
        self.documents.extend(documents)
        print(f"✅ Añadidos {len(documents)} documentos al índice")

    def save_index(self):
        """
        Guarda el índice y los documentos en disco
        """
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
        faiss.write_index(self.index, os.path.join(self.index_path, "faiss.index"))

        # Guardar documentos
        import pickle
        with open(os.path.join(self.index_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

        print(f"💾 Índice FAISS y documentos guardados en {self.index_path}")

    def load_index(self):
        """
        Carga el índice y documentos desde disco
        """
        import pickle
        self.index = faiss.read_index(os.path.join(self.index_path, "faiss.index"))
        with open(os.path.join(self.index_path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        print(f"📂 Índice FAISS y documentos cargados desde {self.index_path}")

    def search(self, query_embedding: List[float], top_k: int = 5):
        """
        Realiza búsqueda por similitud usando un embedding de consulta.
        Retorna tuplas (Document, score)
        :param query_embedding: Embedding de la consulta.
        :param top_k: Número máximo de resultados a retornar.
        :return: Lista de tuplas (Document, score) con los resultados más cercanos.
        """
        query_vec = np.array(query_embedding).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], dist))
        return results
