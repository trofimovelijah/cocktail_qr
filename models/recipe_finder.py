from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

class RecipeFinder:
    def __init__(self):
        # Инициализация эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Загрузка векторной базы
        self.db = FAISS.load_local(
            "data/faiss_db_epub_mistral", 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Инициализация LLM
        self.llm = Ollama(model="saiga_mistral_7b_gguf:Q8_0")

    async def find_recipe(self, ingredients: list, style: str) -> str:
        # Формирование запроса
        ingredients_str = ", ".join(ingredients)
        query = f"""Напиши, какие коктейли можно изготовить из представленных ингредиентов.
        Для каждого коктейля опиши подробный способ приготовления и укажи пропорции.
        Ингредиенты: {ingredients_str}"""
        
        # Поиск рецепта
        results = self.db.similarity_search(query)
        recipe = results[0].page_content if results else "Рецепты не найдены"
        
        # Применение стиля
        if style != "4":  # Если не стандартный стиль
            prompt = PromptTemplate(
                input_variables=['text', 'style'],
                template='Перепиши текст в указанном стиле:\n{text}\nСтиль: {style}'
            )
            styled_prompt = prompt.format(text=recipe, style=style)
            recipe = self.llm.invoke(styled_prompt)
        
        return recipe 