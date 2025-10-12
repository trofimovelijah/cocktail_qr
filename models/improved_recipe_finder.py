# Улучшенный RecipeFinder с оптимизированным FAISS и Cohere Rerank

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

class ImprovedRecipeFinder:
    """
    Улучшенный поиск рецептов с оптимизированным FAISS индексом
    и интеграцией Cohere Rerank для повышения качества поиска.
    """
    
    def __init__(self):
        """Инициализация всех компонентов системы"""
        logger.info("Инициализация ImprovedRecipeFinder...")
        
        # Загружаем токены из переменных окружения
        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        
        if not self.hf_token:
            raise ValueError("Не найден токен HUGGINGFACEHUB_API_TOKEN в .env файле")
        
        # Инициализация эмбеддингов (те же что использовались при создании индекса)
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
        
        # Загружаем улучшенный FAISS индекс
        self._load_vector_database()
        
        # Инициализация Cohere для реранкинга (если доступен API ключ)
        self._init_cohere_reranker()
        
        # Инициализация LLM (Ollama)
        self._init_llm()
        
        # Создание RetrievalQA цепочки
        self._create_retrieval_qa()
        
        logger.info("✓ ImprovedRecipeFinder успешно инициализирован")
    
    def _load_vector_database(self):
        """Загружает улучшенную векторную базу данных"""
        try:
            # Сначала пробуем загрузить улучшенную базу
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            improved_path = os.path.join(base_dir, "data", "faiss_db_improved_markdown")
            
            if os.path.exists(improved_path):
                logger.info("Загружаем улучшенную векторную базу...")
                self.db = FAISS.load_local(
                    folder_path=improved_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✓ Загружена улучшенная база из {improved_path}")
            else:
                # Fallback на оригинальную базу
                original_path = os.path.join(base_dir, "data", "faiss_db_epub_mistral")
                logger.warning(f"Улучшенная база не найдена, загружаем оригинальную из {original_path}")
                self.db = FAISS.load_local(
                    folder_path=original_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✓ Загружена оригинальная векторная база")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки векторной базы: {e}")
            raise RuntimeError(f"Не удалось загрузить векторную базу: {e}")
    
    def _init_cohere_reranker(self):
        """Инициализирует Cohere реранкер если доступен API ключ"""
        self.cohere_client = None
        
        if self.cohere_api_key:
            try:
                import cohere
                self.cohere_client = cohere.Client(self.cohere_api_key)
                logger.info("✓ Cohere реранкер инициализирован")
            except ImportError:
                logger.warning("Библиотека cohere не установлена. Установите: pip install cohere")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать Cohere: {e}")
        else:
            logger.info("Cohere API ключ не найден, реранкинг будет отключен")
    
    def _init_llm(self):
        """Инициализирует Ollama LLM"""
        try:
            self.llm = Ollama(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                model="mistral:instruct",
                temperature=0.8,
                num_predict=512,
                stop=["\\n\\n", "###", "\\n\\n\\n"],
            )
            logger.info("✓ Ollama LLM инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации Ollama: {e}")
            raise RuntimeError(f"Не удалось инициализировать Ollama: {e}")
    
    def _create_retrieval_qa(self):
        """Создает цепочку RetrievalQA с оптимизированными параметрами"""
        
        # Используем MMR (Maximal Marginal Relevance) для разнообразия результатов
        self.retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,          # Увеличиваем количество кандидатов для реранкинга
                "fetch_k": 20,   # Больше документов для первичного поиска
                "lambda_mult": 0.7  # Баланс между релевантностью и разнообразием
            }
        )
        
        # Создаем системный промпт
        system_prompt = """
        Ты — AI-агент, эксперт по поиску рецептов коктейлей из книги "Мастер шейков и Маргариты".
        Твоя задача — найти точный рецепт коктейля по указанным ингредиентам.
        
        ВАЖНО:
        - Если в контексте есть рецепт с указанными ингредиентами, обязательно его верни
        - Обращай особое внимание на точное соответствие ингредиентов
        - Если точного соответствия нет, предложи наиболее похожий рецепт
        - Всегда указывай название коктейля, ингредиенты и способ приготовления
        - Отвечай на русском языке
        """
        
        user_prompt_template = """
        Контекст из базы рецептов:
        {context}
        
        Вопрос пользователя: {question}
        
        На основе контекста найди подходящий рецепт коктейля. Если точного соответствия нет,
        предложи наиболее похожий вариант из доступных рецептов.
        """
        
        # Создаем RetrievalQA цепочку
        self.qa_agent = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    input_variables=["question", "context"],
                    template=system_prompt + user_prompt_template
                ),
            },
            return_source_documents=True
        )
        
        logger.info("✓ RetrievalQA цепочка создана")
    
    def _rerank_documents_with_cohere(self, query: str, documents: List[Any], top_k: int = 3) -> List[Any]:
        """
        Переранжирует документы с помощью Cohere Rerank для улучшения релевантности
        """
        if not self.cohere_client or len(documents) <= 1:
            return documents[:top_k]
        
        try:
            # Подготавливаем документы для реранкинга
            rerank_docs = []
            for doc in documents:
                # Используем содержимое документа и метаданные
                doc_text = doc.page_content
                if hasattr(doc, 'metadata') and doc.metadata:
                    # Добавляем информацию из метаданных для лучшего реранкинга
                    if 'recipe_name' in doc.metadata:
                        doc_text = f"Рецепт: {doc.metadata['recipe_name']}. {doc_text}"
                    if 'ingredients_str' in doc.metadata:
                        doc_text = f"{doc_text} Ингредиенты: {doc.metadata['ingredients_str']}"
                
                rerank_docs.append({"text": doc_text})
            
            # Выполняем реранкинг
            logger.info(f"Выполняем реранкинг {len(rerank_docs)} документов с Cohere...")
            response = self.cohere_client.rerank(
                query=query,
                documents=rerank_docs,
                top_k=min(top_k, len(documents)),
                model="rerank-multilingual-v2.0"  # Поддерживает русский язык
            )
            
            # Переранжируем оригинальные документы
            reranked_docs = []
            for result in response.results:
                reranked_docs.append(documents[result.index])
            
            logger.info(f"✓ Реранкинг завершен, возвращено {len(reranked_docs)} документов")
            return reranked_docs
            
        except Exception as e:
            logger.warning(f"Ошибка реранкинга с Cohere: {e}")
            return documents[:top_k]
    
    def _enhance_query(self, ingredients: List[str]) -> str:
        """
        Улучшает поисковый запрос для лучшего нахождения рецептов
        """
        # Нормализуем ингредиенты
        normalized_ingredients = []
        
        # Словарь синонимов для популярных ингредиентов
        synonyms = {
            'грейпфрут': ['грейпфрутовый сок', 'грейпфрут'],
            'рожь': ['ржаной виски', 'рожь'],
            'виски': ['виски', 'whiskey', 'whisky'],
            'водка': ['водка', 'vodka'],
            'джин': ['джин', 'gin'],
            'лимон': ['лимонный сок', 'лимон'],
            'апельсин': ['апельсиновый сок', 'апельсин']
        }
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower().strip()
            normalized_ingredients.append(ingredient_lower)
            
            # Добавляем синонимы
            for key, syns in synonyms.items():
                if key in ingredient_lower:
                    normalized_ingredients.extend(syns)
        
        # Создаем расширенный запрос
        base_query = f"Ингредиенты: {', '.join(ingredients)}"
        enhanced_query = f"{base_query}. Коктейль с {' '.join(normalized_ingredients)}"
        
        return enhanced_query
    
    def _extract_recipe_info(self, response: str, source_docs: List[Any]) -> Dict[str, Any]:
        """
        Извлекает структурированную информацию о рецепте из ответа
        """
        recipe_info = {
            'name': 'Неизвестный коктейль',
            'ingredients': [],
            'instructions': '',
            'source': '',
            'confidence': 0.0
        }
        
        # Пытаемся извлечь название из ответа или из метаданных источников
        if source_docs:
            for doc in source_docs:
                if hasattr(doc, 'metadata') and 'recipe_name' in doc.metadata:
                    recipe_info['name'] = doc.metadata['recipe_name']
                    break
        
        # Извлекаем ингредиенты из ответа
        ingredient_pattern = r'(\d+\s*мл\s+[^;\n]+)'
        ingredients_found = re.findall(ingredient_pattern, response)
        recipe_info['ingredients'] = ingredients_found
        
        # Весь ответ как инструкции
        recipe_info['instructions'] = response
        
        return recipe_info
    
    async def find_recipe(self, ingredients: List[str], style: str) -> str:
        """
        Основной метод поиска рецепта с улучшенным алгоритмом
        """
        try:
            logger.info(f"Поиск рецепта для ингредиентов: {ingredients}")
            
            # Улучшаем поисковый запрос
            enhanced_query = self._enhance_query(ingredients)
            logger.info(f"Расширенный запрос: {enhanced_query}")
            
            # Выполняем поиск через RetrievalQA
            result = self.qa_agent({"query": enhanced_query})
            
            # Получаем найденные документы
            source_documents = result.get("source_documents", [])
            logger.info(f"Найдено {len(source_documents)} документов")
            
            # Применяем реранкинг с Cohere если доступен
            if self.cohere_client and source_documents:
                logger.info("Применяем реранкинг с Cohere...")
                reranked_docs = self._rerank_documents_with_cohere(
                    enhanced_query, 
                    source_documents, 
                    top_k=3
                )
                
                # Пересоздаем контекст с переранжированными документами
                reranked_context = "\n\n".join([doc.page_content for doc in reranked_docs])
                
                # Генерируем улучшенный ответ на основе переранжированного контекста
                improved_prompt = f"""
                На основе следующего контекста найди рецепт коктейля:
                
                Контекст:
                {reranked_context}
                
                Запрос: {enhanced_query}
                
                Ответь с указанием названия коктейля, ингредиентов и способа приготовления.
                """
                
                response = self.llm.invoke(improved_prompt)
            else:
                response = result["result"]
            
            # Применяем стиль если нужно
            if style != "4":
                response = self._apply_style(response, style)
            
            # Извлекаем структурированную информацию
            recipe_info = self._extract_recipe_info(response, source_documents)
            logger.info(f"Найден рецепт: {recipe_info['name']}")
            
            return self._format_response(response)
            
        except Exception as e:
            logger.error(f"Ошибка при поиске рецепта: {e}")
            return f"Извините, произошла ошибка при поиске рецепта: {str(e)}"
    
    def _apply_style(self, text: str, style: str) -> str:
        """Применяет выбранный стиль к тексту рецепта"""
        
        # Словарь стилей (как в оригинале)
        styles = {
            "1": "стиль космического ужаса, Иных богов, Некрономикона и хтонического мрака Говарда Ф. Лавкрафта",
            "2": "гопническо-быдляцкий жаргон с использованием соответствующих слов и фраз",
            "3": "экспериментальный стиль нарезок Уильяма Берроуза",
            "4": "стиль меню без изменений",
            "5": "сухое математическое изложение учебника по математической статистике и теории вероятностей с наличием формул",
            "6": "метафорическая образность песен Егора Летова и Гражданской обороны + немного инвективной и яростной лексики"
        }
        
        style_description = styles.get(style, styles["4"])
        
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
        """Форматирует финальный ответ"""
        if "не знаю" in response.lower() or "не найдено" in response.lower():
            return "К сожалению, не нашёл коктейлей с этими ингредиентами. Попробуйте другие."
        
        # Ограничиваем длину ответа
        return response[:1500] if len(response) > 1500 else response
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Возвращает статистику векторной базы данных"""
        try:
            stats = {
                'total_documents': self.db.index.ntotal,
                'vector_dimension': self.db.index.d,
                'index_type': type(self.db.index).__name__,
                'cohere_available': self.cohere_client is not None
            }
            return stats
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {'error': str(e)}


# Пример использования
if __name__ == "__main__":
    # Инициализация улучшенного агента
    recipe_finder = ImprovedRecipeFinder()
    
    # Вывод статистики
    stats = recipe_finder.get_database_stats()
    print("Статистика векторной базы:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Тестовый поиск
    import asyncio
    
    async def test_search():
        # Тест на поиск "Рожь и предубеждение"
        ingredients = ["ржаной виски", "грейпфрутовый сок"]
        style = "4"
        
        print(f"\nТестовый поиск для ингредиентов: {ingredients}")
        result = await recipe_finder.find_recipe(ingredients, style)
        print("Результат:")
        print(result)
    
    # Запуск теста
    asyncio.run(test_search())