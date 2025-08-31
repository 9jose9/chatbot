from langchain_openai import ChatOpenAI

class LLMHandler:
    def __init__(self, model_name="gpt-4o-mini", temperature=0.0, max_tokens=500):
        """
        :param model_name: modelo de OpenAI (ej: gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo)
        :param temperature: control de aleatoriedad
        :param max_tokens: límite de salida
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def answer(self, query: str, context: str) -> str:
        """
        Genera una respuesta profesional basada en el contexto proporcionado y la pregunta del usuario.

        :param query: Pregunta del usuario a la que se desea responder.
        :param context: Contexto relevante extraído de los documentos o memoria previa.
        :return: Respuesta generada por el modelo como texto limpio.
        """
        prompt = (
            "Eres un asistente para un chatbot. "
            "Tu tarea es responder preguntas del usuario utilizando el contexto proporcionado. "
            "Además, puedes identificar y describir figuras, imágenes y tablas presentes en los documentos.\n\n"
            "Instrucciones:\n"
            "- Mantén un tono formal y conciso.\n"
            "- Utiliza ejemplos o explicaciones cuando sean útiles.\n"
            "- Responde únicamente a preguntas legales y apropiadas.\n\n"
            f"Contexto relevante:\n{context}\n\n"
            f"Pregunta del usuario:\n{query}\n\n"
            "Respuesta detallada y profesional:"
        )
    
        response = self.llm.invoke(prompt)
        return response.content.strip()

