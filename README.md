# Chatbot RAG con Memoria Dinámica (OpenAI + Gradio)

Este proyecto implementa un **chatbot** basado en **RAG (Retrieval-Augmented Generation)** que combina recuperación de información de documentos con memoria dinámica. El chatbot puede responder preguntas usando los documentos cargados y mantener un **historial resumido** de la conversación para mejorar la coherencia y la eficiencia. Además tiene la capacidad de responder a consultas fuera de su base de conocimiento. 

---

## 🛠 Tecnologías utilizadas

* **Python 3.10+**
* **LangChain**: gestión de memoria y RAG
* **FAISS**: motor de búsqueda vectorial
* **OpenAI**: embeddings (`text-embedding-3-small`) y LLM (`gpt-4o-mini`)
* **Gradio**: interfaz web interactiva
* **PyPDF2**: carga de documentos PDF

---

## 📂 Estructura del proyecto

```
chatbot-rag/
│
├─ data/
│   └─ raw/                # Documentos para RAG (.pdf)
│
├─ src/
│   ├─ gradio_app.py        # Interfaz principal
│   ├─ rag/
│   │   ├─ loader.py        # Carga de documentos
│   │   ├─ embedder.py      # Generación de embeddings
│   │   ├─ vectorstore.py   # FAISS VectorStore
│   │   ├─ retriever.py     # Recuperador de contexto
│   │   └─ llm_handler.py   # Interfaz LLM OpenAI
│   └─ memory/
│       └─ memory_manager.py # Gestión de memoria dinámica y resúmenes
│
├─ requirements.txt         # Dependencias del proyecto
└─ README.md
```

---

## ⚡ Características

1. **RAG (Retrieval-Augmented Generation)**:
   Recupera información relevante de documentos cargados y la combina con la memoria del usuario para generar respuestas más precisas. Tal y como está implementado el RAG, es posible preguntar sobre imágenes y figuras de los documentos, así como sobre ciertos datos estructurados que aparezcan en ellos. 

2. **Memoria dinámica con resumen automático**:
   Cuando la memoria supera un límite de tokens, se genera automáticamente un **resumen**, que luego se integra en la memoria para mantener la coherencia de la conversación.

3. **Soporte documentos PDF**:
   Compatible con `.pdf`.

4. **Interfaz web con Gradio**:
   Chat interactivo fácil de usar que limpia automáticamente la entrada del usuario.

5. **Embeddings y LLM con OpenAI**:

   * Embeddings: `text-embedding-3-small`
   * LLM: `gpt-4o-mini` (personalizable con la API key de OpenAI)

---

## 🚀 Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/9jose9/prueba-turing.git
```

2. Crear un entorno virtual:

```bash
python -m venv venv
```

3. Activar el entorno:

* Windows:

```bash
venv\Scripts\activate
```

* Linux / macOS:

```bash
source venv/bin/activate
```

4. Instalar dependencias:

```bash
pip install -r requirements.txt
```

5. Configurar API key de OpenAI:

```bash
export OPENAI_API_KEY="tu_api_key"
```

> En Windows PowerShell:

```powershell
setx OPENAI_API_KEY "tu_api_key"
```

---

## 📝 Uso

1. Coloca tus documentos PDF en `data/raw/`.
2. Ejecuta la aplicación Gradio:

```bash
cd src
python gradio_app.py
```

3. Abre el navegador en la URL que Gradio indique (normalmente `http://127.0.0.1:7860/`).
4. Escribe tu pregunta en el chat y obtén respuestas usando RAG + memoria dinámica.

---

## ⚙ Configuración

* `chunk_size` y `chunk_overlap` en `loader.py` para ajustar cómo se fragmentan los documentos.
* `max_tokens` en `memory_manager.py` para definir cuándo se genera un resumen de memoria.
* `top_k` en `retriever.py` para definir cuántos documentos se recuperan para la generación de la respuesta.

---

## 📌 Consideraciones

* Cada resumen de memoria se genera usando la información acumulada hasta el momento, lo que permite mantener la coherencia incluso en conversaciones largas.

