from typing import List
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader

class DocumentLoader:
    def __init__(self, chunk_size=300, chunk_overlap=50, split_by_page=False):
        """
        :param chunk_size: Tamaño máximo de cada chunk en caracteres.
        :param chunk_overlap: Superposición entre chunks consecutivos.
        :param split_by_page: Si True, cada página se considera un chunk independiente.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_by_page = split_by_page
    
    def load_pdf(self, file_path):
        """
        Carga un PDF y lo divide en chunks según la configuración.

        :param file_path: Ruta al archivo PDF.
        :return: Lista de Document, cada uno representando un chunk o una página.
        """
        loader = PyPDFLoader(file_path)
        if self.split_by_page:
            return loader.load_and_split() 
        else:
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            return splitter.split_documents(docs)

    def load_directory(self, dir_path: str, extensions: List[str] = [".pdf"]) -> List[Document]:
        """
        Carga múltiples documentos de un directorio.

        :param dir_path: Carpeta donde están los archivos.
        :param extensions: Extensiones soportadas.
        :return: Lista de Document con todos los chunks de los archivos cargados.
        """
        documents = []
        for file in Path(dir_path).rglob("*"):
            if file.suffix.lower() in extensions:
                if file.suffix.lower() == ".pdf":
                    documents.extend(self.load_pdf(str(file)))
        return documents
