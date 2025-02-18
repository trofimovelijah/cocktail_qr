from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain.chains import RetrievalQA
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class RecipeFinder:
    def __init__(self):
        # Инициализация эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu", "token": hf_token}
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
            temperature=0.8,
            num_predict=512,
            stop=["\n\n", "###"],
            #top_k=40
        )
        
        # Настройка ретривера
        self.retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        
        # Системный промпт
        self.system_prompt = """Ты — помощник для поиска рецептов коктейлей, знаток классической литературы и просто культурный деятель. 
        У тебя есть доступ к базе данных с рецептами, вдохновлёнными литературными произведениями. На основе её сгенери рецепт по имеющимся ингредиентам.
        Когда пользователь вводит ингредиенты, используй инструмент "Cocktail Recipe Finder", чтобы найти подходящие рецепты.
        Если рецепт не найден, предложи альтернативные варианты или уточни запрос.
        ===
           
        Пример:
        Пользователь: Напиши, какие коктейли можно изготовить из ржаного виски и грейпфрутового сока? Опиши способ приготовления и пропорции. Отвечай на русском, четко и лаконично. 
        Агент: Использую инструмент "Cocktail Recipe Finder" для поиска рецептов. Вот что найдено: [рецепт].

        ===

        Пример корректного ответа по введённым ингредиентам (ржаной виски, грейпфрутовый сок):
        Коктейль "Рожь и предубеждение"
        Состав:
        - Ржаной виски (50 мл)
        - Грейпфрутовый сок (90 мл)
        Способ приготовления:
        - Ингредиенты выливаем в стакан рокс поверх кубиков льда, размешиваем в ритме «сердце вскачь». 
        - Хотим подчеркнуть, юные леди, мы нисколечко не предубеждены против замужества, а лишь призываем вас уяснить главное: чтобы почувствовать себя королевой, не нужно и дворца (как, впрочем, и короля).
        """

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
            "1": "стиль космического ужаса, Иных богов, Некрономикона и хтонического мрака Говарда Ф. Лавкрафта",
            "2": "урко-гопническо-быдляцкий жаргон",
            "3": "экспериментальный стиль нарезок Уильяма Берроуза",
            "4": "стиль меню без изменений",
            "5": "сухое математическое изложение учебника по математической статистике и теории вероятностей с наличием формул",
            "6": "метафорическая образность песен Егора Летова и Гражданской обороны + немного инвективной и яростной лексики + грязный звук"
        }

    def _create_tools(self):
        return [
            Tool(
                name="Cocktail Recipe Finder",
                func=self.find_recipe,
                description="Используй для поиска рецептов по ингредиентам. Пример: ['сок', 'виски']."
            )
        ]

    async def find_recipe(self, ingredients: list, style: str) -> str:
        # Формирование запроса
        ingredients_str = ", ".join(ingredients)
        user_prompt = f"""Напиши, какой коктейль можно изготовить из представленных ингредиентов.
            Укажи пропорции каждого ингредиента. 
            Отвечай строго на русском языке, используя чёткий и лаконичный формат.
            Не используй повторений.
            Ингредиенты: {ingredients_str}"""
        
        # Получение контекста из базы данных
        docs = self.retriever.invoke(ingredients_str)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Формирование финального промта
        final_prompt = ChatPromptTemplate.from_template(f"""
        {self.system_prompt}
        
        {context}
        
        Задача: {user_prompt}
        """)

        # Получение текста из final_prompt
        prompt_text = final_prompt.format()  # Извлекаем текст из шаблона

        # Получение ответа от LLM
        response = self.llm.invoke(prompt_text)
        
        # Применение стиля
        if style != "4":
            styled_response = self._apply_style(response, style)
            return self._format_response(styled_response)
        
        return self._format_response(response)

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
