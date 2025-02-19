from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 1. Системный промт
SYSTEM_PROMPT = """
        Ты — AI-агент, мой помощник для поиска рецептов коктейлей, знаток хорошей литературы и просто культурный деятель.
        Твоя задача — отвечать на вопросы пользователей, используя только предоставленные рецепты.
        Если рецепт не найден, предложи рецепт, который возможно приготовить по имеющимся ингредиентам.
        Отвечай чётко и по делу, но сохраняй культурную точность.
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

# 2. Пользовательский промт
USER_PROMPT_TEMPLATE = """
Вопрос: {question}

Контекст:
{context}

На основе контекста ответь на вопрос. Если в контексте ответа нет, предложи свой рецепт, но ориентируйся на имеющиеся ингредиенты и стилистику контекста.
"""

class RecipeFinder:
    def __init__(self):
        # Инициализация эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={
                "device": "cpu", 
                "token": hf_token
            }
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
        )
        
        # Инициализация RetrievalQA
        self.qa_agent = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 3, "fetch_k": 20}),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    input_variables=["question", "context"],
                    template=USER_PROMPT_TEMPLATE
                ),
            },
            return_source_documents=True
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

    async def find_recipe(self, ingredients: list, style: str) -> str:
        # Формирование запроса
        ingredients_str = ", ".join(ingredients)
        user_question = """Напиши, какой коктейль (название) можно изготовить из представленных ингредиентов. 
        Укажи пропорции каждого ингредиента и способ приготовления. Отвечай строго на русском языке, используя чёткий и лаконичный формат. 
        Не используй повторений. 
        Ингредиенты: {ingredients_str}"""
        
        # Получение контекста из базы данных
        result = self.qa_agent({"query": user_question})
        context = "\n\n".join([d.page_content for d in result["source_documents"]])
        
        # Формирование финального промта с системным промтом
        final_prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\nЗадача: {user_question}"

        # Получение ответа от LLM
        response = self.llm.invoke(final_prompt)
        
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

# Пример использования
if __name__ == "__main__":
    # Инициализация агента
    recipe_finder = RecipeFinder()
    
    # Пример запроса
    ingredients = ["ржаной виски", "грейпфрутовый сок"]
    style = "1"
    result = recipe_finder.find_recipe(ingredients, style)
    
    print("Ответ:", result)