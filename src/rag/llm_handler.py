from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import initialize_agent, AgentType

class LLMHandler:
    def __init__(self, model_name="gpt-4o-mini", temperature=0.0, max_tokens=500):
        """
        :param model_name: Nombre del modelo de OpenAI (ej: "gpt-4o-mini", "gpt-4-turbo").
        :param temperature: Control de aleatoriedad en las respuestas (0.0 = determinista).
        :param max_tokens: Límite máximo de tokens generados en la salida.
        """

        self.python_flags = [ "python", "código", "code", "ejecuta", "programa",
                            "calcula", "resolver", "script", "algoritmo",
                            "función", "matriz", "integral", "derivada",
                            "media", "varianza", "desviación estándar", "probabilidad",
                            "ecuación", "número primo", "operación matemática"
                        ]

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        self.python_tool = PythonAstREPLTool()

        self.agent = initialize_agent(
            tools=[self.python_tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )


    def requires_python(self, query: str) -> bool:
        """
        Evalúa si la consulta requiere ejecutar código Python usando
            un clasificador basado en el LLM.
        """
        classification_prompt = (
            "Dada la siguiente consulta, responde SOLO con 'yes' si requiere "
            "ejecutar código Python (por cálculos, estadísticas, algoritmos, etc.), "
            "o 'no' en caso contrario.\n\n"
            f"Consulta: {query}"
        )
        result = self.llm.invoke(classification_prompt).content.strip().lower()
        return "yes" in result


    def answer(self, query: str, context: str) -> str:
        """
        Genera una respuesta profesional basada en el contexto proporcionado y la pregunta del usuario.
        Usa el agente con capacidad de ejecutar Python si es necesario, de lo contrario responde solo con el LLM.

        :param query: Pregunta del usuario a la que se desea responder.
        :param context: Contexto relevante extraído de los documentos o memoria previa.
        :return: Respuesta generada por el modelo como texto limpio.
        """

        if self.requires_python(query) or any(flag in query.lower() for flag in self.python_flags):
            response = self.agent.run(
                f"Contexto:\n{context}\n\nPregunta:\n{query}\n\n"
                "Usa Python si es necesario para calcular la respuesta."
            )
        
        else: 

            prompt = (
                "Eres un asistente para un chatbot. "
                "Tu tarea es responder preguntas del usuario utilizando el contexto proporcionado. "
                "Además, puedes identificar y describir figuras, imágenes y tablas presentes en los documentos.\n\n"
                "Instrucciones:\n"
                "- Mantén un tono formal y conciso.\n"
                "- Utiliza ejemplos o explicaciones cuando sean útiles.\n"
                "- Responde únicamente a preguntas legales y apropiadas.\n\n"
                "Si necesitas usar Python para calcular algo, incluye el código en un bloque "
                "```python ... ``` y lo ejecutaré.\n\n"
                f"Contexto relevante:\n{context}\n\n"
                f"Pregunta del usuario:\n{query}\n\n"
                "Respuesta detallada y profesional:"
            )

            response = self.llm.invoke(prompt)
            response = response.content.strip()
    

        return response

