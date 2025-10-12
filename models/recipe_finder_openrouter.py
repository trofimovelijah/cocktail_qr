# Упрощенный RecipeFinderOpenRouter на базе отлаженного ноутбука

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import requests
import logging
import json
from fenya_vocab import fenya
 
# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RecipeFinderOpenRouter:
    """
    Упрощенная версия RecipeFinder с OpenRouter вместо локального Ollama
    """
    
    def __init__(self):
        logger.info("Инициализация RecipeFinderOpenRouter...")
        
        # Получаем API ключи
        self.hf_token = self._read_token_from_env()
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY не найден в переменных окружения")
        
        # Инициализация эмбеддингов (идентично основному классу)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="cointegrated/LaBSE-en-ru",
            model_kwargs={
                "device": "cpu", 
                "token": self.hf_token
            },
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
        
        # Загружаем векторное хранилище
        self._load_vector_database()
        
        # Настройка ретривера (как в основном классе)
        self.retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 20,
            }
        )

        # Стили (те же что в основном классе)
        self.styles = {
            "1": "стиль космического ужаса, Иных богов, Некрономикона и хтонического мрака Говарда Ф. Лавкрафта",
            "2": f"гопническо-быдляцкий жаргон с использованием {fenya} и подобных слов и фраз",
            "3": "экспериментальный стиль нарезок Уильяма Берроуза",
            "4": "стиль меню без изменений",
            "5": "сухое математическое изложение учебника по математической статистике и теории вероятностей с наличием формул",
            "6": "метафорическая образность песен Егора Летова и Гражданской обороны + немного инвективной и яростной лексики + грязный звук"
        }
        
        logger.info("✓ RecipeFinderOpenRouter успешно инициализирован")

    def _read_token_from_env(self):
        """Чтение токена (идентично основному классу)"""
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith("HUGGINGFACEHUB_API_TOKEN"):
                            return line.split('=')[1].strip()
            except FileNotFoundError:
                pass
            raise ValueError("HUGGINGFACEHUB_API_TOKEN не найден")
        return hf_token

    def _load_vector_database(self):
        """Загрузка векторной базы (идентично основному классу)"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        improved_path = os.path.join(base_dir, "data", "md_vectorstore")
        fallback_path = os.path.join(base_dir, "data", "faiss_db_epub_mistral")
        
        try:
            if os.path.exists(improved_path):
                logger.info("Загружаем md_vectorstore...")
                self.db = FAISS.load_local(
                    folder_path=improved_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✓ Загружена улучшенная векторная база")
            else:
                logger.warning("md_vectorstore не найден, fallback...")
                self.db = FAISS.load_local(
                    folder_path=fallback_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✓ Загружена оригинальная векторная база")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            raise RuntimeError(f"Не удалось загрузить векторную базу: {e}")

    def _call_openrouter_api(self, prompt: str, model: str = "openai/gpt-4o-mini") -> str:
        """Вызов OpenRouter API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/trofimovelijah/cocktail_qr",
                "X-Title": "Cocktail Recipe Bot"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "Ты эксперт по коктейлям. Отвечай на русском языке четко и структурированно."
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 800,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Ошибка OpenRouter API: {e}")
            raise RuntimeError(f"Не удалось получить ответ от OpenRouter: {e}")

    async def find_recipe(self, ingredients: list, style: str) -> str:
        """
        Поиск рецептов с использованием OpenRouter
        """
        try:
            logger.info(f"Поиск рецепта через OpenRouter для: {ingredients}")
            
            # Получаем контекст из векторной базы (как в ноутбуке)
            ingredients_str = ", ".join(ingredients)
            search_query = f"Ингредиенты: {ingredients_str}"
            
            # Поиск релевантных документов
            relevant_docs = self.retriever.get_relevant_documents(search_query)
            context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Формируем промпт для OpenRouter (упрощенный, на основе ноутбука)
            prompt = f"""На основе следующего контекста из книги рецептов коктейлей найди подходящий рецепт:

Контекст:
{context}

Запрос пользователя: Какие коктейли можно изготовить из ингредиентов: {ingredients_str}

Требования к ответу:
- Назови конкретный коктейль из контекста
- Укажи точные пропорции ингредиентов
- Опиши способ приготовления  
- Отвечай на русском языке в литературном стиле
- Если точного соответствия нет, выбери наиболее подходящий рецепт"""

            # Получаем ответ от OpenRouter
            response = self._call_openrouter_api(prompt)
            
            logger.info("✓ Получен ответ от OpenRouter")
            
            # Применяем стиль если нужно
            if style != "4":
                response = await self._apply_style_with_openrouter(response, style)
            
            return self._format_response(response)
            
        except Exception as e:
            logger.error(f"Ошибка при поиске рецепта: {e}")
            return f"Извините, произошла ошибка: {str(e)}"

    async def _apply_style_with_openrouter(self, text: str, style: str) -> str:
        """Применение стиля через OpenRouter"""
        style_description = self.styles.get(style, self.styles["4"])
        
        style_prompt = f"""Перепиши следующий текст рецепта коктейля в указанном стиле, сохраняя всю важную информацию:

Оригинальный текст:
{text}

Целевой стиль: {style_description}

Результат должен быть на русском языке и сохранять структуру рецепта."""

        try:
            return self._call_openrouter_api(style_prompt)
        except Exception as e:
            logger.error(f"Ошибка применения стиля: {e}")
            return text

    def _format_response(self, response: str) -> str:
        """Форматирование ответа (идентично основному классу)"""
        if "не знаю" in response.lower() or "не найдено" in response.lower():
            return "К сожалению, не нашёл коктейлей с этими ингредиентами. Попробуйте другие."
        
        return response[:1500] if len(response) > 1500 else response

    def get_database_stats(self):
        """Статистика базы данных"""
        try:
            stats = {
                'total_documents': self.db.index.ntotal,
                'vector_dimension': self.db.index.d,
                'index_type': type(self.db.index).__name__,
                'api_provider': 'OpenRouter',
                'model': 'gpt-4o-mini'
            }
            return stats
        except Exception as e:
            return {'error': str(e)}

# Пример использования
if __name__ == "__main__":
    import asyncio
    
    async def test_openrouter():
        finder = RecipeFinderOpenRouter()
        
        # Статистика
        stats = finder.get_database_stats()
        print("Статистика:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
        # Тест
        ingredients = ["ржаной виски", "грейпфрутовый сок"]
        result = await finder.find_recipe(ingredients, "4")
        print(f"\nРезультат для {ingredients}:")
        print(result)
    
    asyncio.run(test_openrouter())
