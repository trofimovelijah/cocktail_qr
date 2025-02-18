from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain.chains import RetrievalQA
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class RecipeFinder:
    def __init__(self):
        # Инициализация эмбеддингов с новыми параметрами
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu", "token": hf_token}#,
            #encode_kwargs={'normalize_embeddings': True}
        )
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        index_path = os.path.join(base_dir, "data", "faiss_db_epub_mistral")
        
        self.db = FAISS.load_local(
            folder_path=index_path,
            index_name="index",
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Инициализация LLM с новыми параметрами
        self.llm = Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model="mistral:instruct",
            temperature=0.2,
            num_predict=512,
            stop=["\n\n", "###"],
            top_k=40
        )
        
        # Настройка ретривера с MMR
        self.retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        
        # Инициализация цепочки RetrievalQA
        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )
        
        # Системный промпт из отладочного кода
        self.system_prompt = """Ты — помощник для поиска рецептов коктейлей. У тебя есть доступ к базе данных с рецептами, вдохновлёнными литературными произведениями.
        Когда пользователь вводит ингредиенты, используй инструмент "Cocktail Recipe Finder", чтобы найти подходящие рецепты.
        Если рецепт не найден, предложи альтернативные варианты или уточни запрос."""

        # Инициализация агента
        self.agent = initialize_agent(
            tools=self._create_tools(),
            llm=self.llm,
            agent="conversational-react-description",
            verbose=True,
            agent_kwargs={'system_message': self.system_prompt}
        )
        
        # Стили для преобразования
        self.styles = {
            "1": "стиль космического ужаса и хтонического мрака Говарда Ф. Лавкрафта",
            "2": "гопническо-быдляцкий жаргон",
            "3": "экспериментальный стиль нарезок Уильяма Берроуза",
            "4": "стиль без изменений"
        }

    def _create_tools(self):
        return [
            Tool(
                name="Cocktail Recipe Finder",
                func=self.retrieval_qa.run,
                description="Используй для поиска рецептов по ингредиентам. Пример: ['сок', 'виски']."
            )
        ]

    async def find_recipe(self, ingredients: list, style: str) -> str:
        # Формирование запроса
        ingredients_str = ", ".join(ingredients)
        user_prompt = f"""Напиши, какие коктейли можно изготовить из: {ingredients_str}.
        Опиши способ приготовления и пропорции. Отвечай на русском, четко и лаконично."""
        
        # Поиск базового рецепта
        response = self.retrieval_qa.invoke({"query": user_prompt})
        original_response = response["result"]
        
        # Применение стиля
        if style != "4":
            styled_response = self._apply_style(original_response, style)
            return self._format_response(styled_response)
        
        return self._format_response(original_response)

    def _apply_style(self, text: str, style: str) -> str:
        style_description = self.styles.get(style, self.styles["4"])
        prompt_template = PromptTemplate(
            input_variables=['output_text', 'style'],
            template='''Перепиши текст в заданном стиле:
            Текст: {output_text}
            Стиль: {style}
            Результат:'''
        )
        
        styled_prompt = prompt_template.format(
            output_text=text,
            style=style_description
        )
        
        try:
            return self.llm.invoke(styled_prompt)
        except Exception as e:
            print(f"Ошибка стилизации: {e}")
            return text

    def _format_response(self, response: str) -> str:
        if "не знаю" in response.lower() or "не найдено" in response.lower():
            return "К сожалению, не нашёл коктейлей с этими ингредиентами. Попробуйте другие."
        
        # Ограничение длины ответа
        return response[:1000] if len(response) > 1000 else response
