# Обновленный RecipeFinder на основе отлаженного notebook debug_mistral_md.ipynb

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import logging
from fenya_vocab import fenya

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RecipeFinder:
    """
    Основной класс для поиска рецептов коктейлей с использованием 
    улучшенного векторного хранилища md_vectorstore
    """
    
    def __init__(self):
        logger.info("Инициализация RecipeFinder...")
        
        # Получаем токен из .env файла (как в ноутбуке)
        self.hf_token = self._read_token_from_env()
        
        # Инициализация эмбеддингов (идентично ноутбуку)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="cointegrated/LaBSE-en-ru",
            model_kwargs={
                "device": "cpu", 
                "token": self.hf_token
            },
            encode_kwargs={
                'normalize_embeddings': True  # Для косинусной схожести
            }
        )
        
        # Загружаем улучшенное векторное хранилище md_vectorstore
        self._load_vector_database()
        
        # Инициализация LLM (идентично ноутбуку) 
        self.llm = Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model="mistral:instruct",
            temperature=0.8,
            num_predict=512,
            stop=["\n\n", "###", "\n\n\n"],
        )

        # Настройка ретривера с параметрами из ноутбука
        self.retriever = self.db.as_retriever(
            search_type="mmr",              # Maximal Marginal Relevance из ноутбука
            search_kwargs={
                "k": 3,                     # Количество документов как в ноутбуке  
                "fetch_k": 20,              # Размер пула для поиска
            }
        )

        # Инициализация RetrievalQA с упрощенной логикой
        self.qa_agent = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )

        # Стили для преобразования (оставляем как есть)
        self.styles = {
            "1": "стиль космического ужаса, Иных богов, Некрономикона и хтонического мрака Говарда Ф. Лавкрафта",
            "2": f"гопническо-быдляцкий жаргон с использованием {fenya} и подобных слов и фраз",
            "3": "экспериментальный стиль нарезок Уильяма Берроуза",
            "4": "стиль меню без изменений",
            "5": "сухое математическое изложение учебника по математической статистике и теории вероятностей с наличием формул",
            "6": "метафорическая образность песен Егора Летова и Гражданской обороны + немного инвективной и яростной лексики + грязный звук"
        }
        
        logger.info("✓ RecipeFinder успешно инициализирован")

    def _read_token_from_env(self):
        """Чтение токена из .env файла (как в ноутбуке)"""
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            # Fallback: читаем из файла как в ноутбуке
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith("HUGGINGFACEHUB_API_TOKEN"):
                            return line.split('=')[1].strip()
            except FileNotFoundError:
                pass
            raise ValueError("HUGGINGFACEHUB_API_TOKEN не найден в переменных окружения или .env файле")
        return hf_token

    def _load_vector_database(self):
        """Загружает улучшенное векторное хранилище (приоритет md_vectorstore)"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Сначала пробуем загрузить улучшенное хранилище md_vectorstore
        improved_path = os.path.join(base_dir, "data", "md_vectorstore")
        fallback_path = os.path.join(base_dir, "data", "faiss_db_epub_mistral")
        
        try:
            # Пробуем загрузить улучшенную базу
            if os.path.exists(improved_path):
                logger.info("Загружаем улучшенную векторную базу md_vectorstore...")
                self.db = FAISS.load_local(
                    folder_path=improved_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✓ Загружена улучшенная векторная база")
            else:
                # Fallback на оригинальную базу
                logger.warning("md_vectorstore не найден, загружаем оригинальную базу...")
                self.db = FAISS.load_local(
                    folder_path=fallback_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✓ Загружена оригинальная векторная база")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки векторной базы: {e}")
            raise RuntimeError(f"Не удалось загрузить векторную базу: {e}")

    async def find_recipe(self, ingredients: list, style: str) -> str:
        """
        Упрощенный поиск рецептов на основе логики из ноутбука
        """
        try:
            logger.info(f"Поиск рецепта для ингредиентов: {ingredients}")
            
            # Формируем запрос как в ноутбуке
            ingredients_str = ", ".join(ingredients)
            user_prompt = f"""Напиши, какие коктейли можно изготовить из представленных ниже ингредиентов. 
            Для каждого коктейля опиши подробный способ приготовления и укажи пропорции. 
            Отвечай строго на русском языке, используя чёткий и лаконичный формат. 
            Не добавляй лишнюю информацию, которая не относится к рецепту. 
            Ингредиенты: {ingredients_str}"""
            
            # Получаем результат через RetrievalQA (как в ноутбуке)
            result = self.qa_agent.invoke({"query": user_prompt})
            response = result["result"]
            
            logger.info(f"Получен ответ от LLM, длина: {len(response)} символов")
            
            # Применяем стиль если нужно
            if style != "4":
                response = self._apply_style(response, style)
            
            return self._format_response(response)
            
        except Exception as e:
            logger.error(f"Ошибка при поиске рецепта: {e}")
            return f"Извините, произошла ошибка при поиске рецепта: {str(e)}"

    def _apply_style(self, text: str, style: str) -> str:
        """Применение стиля (упрощено по сравнению с ноутбуком)"""
        style_description = self.styles.get(style, self.styles["4"])
        
        # Упрощенный промпт для стилизации (как в ноутбуке)
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
            logger.error(f"Ошибка стилизации: {e}")
            return text

    def _format_response(self, response: str) -> str:
        """Форматирование ответа (как в оригинале)"""
        if "не знаю" in response.lower() or "не найдено" in response.lower():
            return "К сожалению, не нашёл коктейлей с этими ингредиентами. Попробуйте другие."
        
        # Ограничение длины ответа (увеличено с 1000 до 1500)
        return response[:1500] if len(response) > 1500 else response

    def get_database_stats(self):
        """Получение статистики базы данных (новый метод)"""
        try:
            stats = {
                'total_documents': self.db.index.ntotal,
                'vector_dimension': self.db.index.d,
                'index_type': type(self.db.index).__name__,
            }
            return stats
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {'error': str(e)}

# Пример использования
if __name__ == "__main__":
    import asyncio
    
    async def test_recipe_finder():
        # Инициализация 
        recipe_finder = RecipeFinder()
        
        # Вывод статистики
        stats = recipe_finder.get_database_stats()
        print("Статистика векторной базы:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Тестовый запрос (тот же что тестировался в ноутбуке)
        ingredients = ["ржаной виски", "грейпфрутовый сок"]
        style = "4"
        
        print(f"\nТестовый поиск для: {ingredients}")
        result = await recipe_finder.find_recipe(ingredients, style)
        print("Результат:")
        print(result)
    
    # Запуск теста
    asyncio.run(test_recipe_finder())