from typing import List
from langchain.docstore.document import Document

class MemoryManager:
    def __init__(self, max_tokens: int, llm):
        """
        :param max_tokens: Número máximo de tokens antes de generar un resumen
        :param llm: Instancia de LLMHandler para generar resúmenes
        """
        self.memory: List[Document] = []
        self.max_tokens = max_tokens
        self.llm = llm
        self.summary_shown = True

    def add_to_memory(self, doc: Document):
        """
        Añade un documento a la memoria y genera resumen si se supera max_tokens
        :param doc: Documento a agregar (puede ser del usuario o del chatbot).
        """
        self.memory.append(doc)
        total_tokens = self.count_tokens()
        if total_tokens > self.max_tokens:
            self.summarize_memory()
            self.memory = [d for d in self.memory if d.metadata.get("source") == "memory_summary"]
            self.summary_shown = False

    def summarize_memory(self):
        """
        Genera un resumen de la memoria y lo añade como un nuevo documento
        Utiliza el LLM para condensar el contenido y lo marca con metadata 'memory_summary'.
        """
        if not self.memory:
            return
        memory_text = self.get_memory_context()
        prompt = f"Resume brevemente la siguiente memoria:\n{memory_text}\nResumen:"
        summary = self.llm.answer("Resumen de memoria", prompt)
        self.memory.append(Document(
            page_content=summary,
            metadata={"source": "memory_summary"}
        ))

    def get_memory_context(self) -> str:
        """
        Devuelve todo el texto de la memoria concatenado
        :return: Texto concatenado de todos los documentos en memoria.
        """
        return "\n".join([doc.page_content for doc in self.memory])

    def count_tokens(self) -> int:
        """
        Cuenta tokens aproximados (1 palabra ~ 1 token)
        :return: Número total de tokens en la memoria.
        """
        return sum(len(doc.page_content.split()) for doc in self.memory)
