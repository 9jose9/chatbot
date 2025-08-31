import gradio as gr
from rag.loader import DocumentLoader
from rag.embedder import Embedder
from rag.vectorstore import FAISSVectorStore
from rag.retriever import Retriever
from rag.llm_handler import LLMHandler
from memory.memory_manager import MemoryManager
from langchain.docstore.document import Document
from typing import List, Tuple


loader = DocumentLoader(chunk_size=None, chunk_overlap=0, split_by_page=True)
documents = loader.load_directory("../data/raw", extensions=[".pdf"])

embedder = Embedder(model_name="text-embedding-3-small")
embeddings = embedder.embed_documents(documents)
dim = len(embeddings[0])

vectorstore = FAISSVectorStore()
vectorstore.add_embeddings(embeddings, documents)

retriever = Retriever(embedder, vectorstore)

llm = LLMHandler(model_name="gpt-4o-mini", temperature=0.0, max_tokens=500)
memory = MemoryManager(max_tokens=200, llm=llm)


def chat(query: str, history: List[Tuple[str, str]]):
    """
    Función de chat que recibe una consulta del usuario y mantiene la memoria dinámica.

    :param query: Pregunta o mensaje del usuario.
    :param history: Historial del chat en formato [(usuario, chatbot), ...]
    :return: Tupla (historial actualizado, historial para Chatbot, limpiar input)
    """
    memory.add_to_memory(Document(page_content=f"Usuario: {query}"))

    results = retriever.retrieve(query, top_k=3)
    retrieved_context = retriever.format_context(results)
    full_context = retrieved_context + "\n" + memory.get_memory_context()

    answer = llm.answer(query, full_context)
    memory.add_to_memory(Document(page_content=f"Chatbot: {answer}"))

    history = history + [(query, answer)]

    summary_text = ""
    for doc in memory.memory:
        if doc.metadata.get("source") == "memory_summary" and not memory.summary_shown:
            summary_text = doc.page_content
            memory.summary_shown = True
            break

    if summary_text:
        history.append(("📝 Memoria resumida", summary_text))
    return history, history, ""


with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chatbot RAG + Memoria dinámica (OpenAI)")

    chatbot = gr.Chatbot(label="Chat")
    user_input = gr.Textbox(label="Escribe tu mensaje", placeholder="Escribe aquí...")

    user_input.submit(
        chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, chatbot, user_input] 
    )

if __name__ == "__main__":
    demo.launch()
