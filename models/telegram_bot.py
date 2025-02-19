import os
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ForceReply
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
    filters,
    CallbackContext
)
from recipe_finder import RecipeFinder
from speech_generator import SpeechGenerator
import asyncio

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Определение состояний
START, INGREDIENTS, WAITING_INPUT, STYLE, GENERATING_RECIPE, GENERATING_SPEECH = range(6)

# Загрузка переменных окружения
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

class BotState:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                cls._instance.recipe_finder = RecipeFinder()
                cls._instance.speech_generator = SpeechGenerator()
            except Exception as e:
                logger.error(f"Ошибка инициализации: {str(e)}")
                raise RuntimeError("Не удалось инициализировать компоненты бота. Проверьте наличие необходимых файлов.")
        return cls._instance

    @staticmethod
    def convert_style(style_code):
        style_map = {
            "L": "1",  # Лавкрафт
            "G": "2",  # Гопник
            "U": "3",  # Уильям Берроуз
            "M": "4",   # Минимализм
            "T": "5",   # Тервер
            "E": "6"    # Летов
        }
        return style_map.get(style_code, "4")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [[InlineKeyboardButton("Запустить бот", callback_data="start_bot")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Привет, ДОРОГОЙ БРАТ! Йа бот для создания коктейлей по мотивам литературных произведений. С моей помощью ты сможешь найти, что можно придумать из имеющихся у тебя вкусностей. Нажми кнопку ниже, чтобы начать!",
        reply_markup=reply_markup,
    )
    return START

async def handle_ingredients(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    context.user_data.setdefault("ingredients", [])
    
    # Теперь сообщение содержит кнопку для активации ввода 
    keyboard = [[InlineKeyboardButton("➕ Добавить ингредиент", callback_data="activate_input")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        text="Нажмите кнопку, чтобы добавить ингредиент:",
        reply_markup=reply_markup
    )
    return INGREDIENTS

async def activate_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    context.user_data["current_state"] = "WAITING_INPUT"  # Устанавливаем текущее состояние
    await query.edit_message_text(text="Теперь введите ингредиент:")
    return "WAITING_INPUT"  # Новое состояние для ожидания ввода

async def process_ingredient(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if context.user_data.get("current_state") != "WAITING_INPUT":
        await update.message.reply_text("Пожалуйста, нажмите кнопку 'Добавить ингредиент', чтобы ввести ингредиент.")
        return INGREDIENTS

    user_data = context.user_data
    ingredient = update.message.text
    
    if len(user_data.get("ingredients", [])) >= 3:
        await update.message.reply_text("Максимум 3 ингредиента!")
        return INGREDIENTS
    
    user_data.setdefault("ingredients", []).append(ingredient)
    
    buttons = []
    if len(user_data["ingredients"]) < 3:
        buttons.append(InlineKeyboardButton("➕ Добавить ингредиент", callback_data="add_ingredient"))
    buttons.append(InlineKeyboardButton("⚡ Сгенерировать", callback_data="generate"))
    
    keyboard = [buttons]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"Добавлено ингредиентов: {len(user_data['ingredients'])}/3",
        reply_markup=reply_markup,
    )
    return INGREDIENTS

async def select_style(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    # Определяем кнопки с эмодзи
    keyboard = [
        [
            InlineKeyboardButton("👻 Лавкрафт", callback_data="style_L"),
            InlineKeyboardButton("😎 Типикал гоп", callback_data="style_G"),
        ],
        [
            InlineKeyboardButton("📖 Берроуз", callback_data="style_U"),
            InlineKeyboardButton("📝 Стиль меню", callback_data="style_M"),
        ],
        [
            InlineKeyboardButton("📊 Теорвер и матстат", callback_data="style_T"),
            InlineKeyboardButton("🎶 ГрОб", callback_data="style_E"),
        ],
        [InlineKeyboardButton("Отмена", callback_data="cancel")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(text="Выберите стиль:", reply_markup=reply_markup)
    return STYLE

async def generate_recipe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    user_data = context.user_data
    
    # Отправляем сообщение с анимацией
    progress_message = await query.edit_message_text(text="Генерация рецепта... 🍾🥂")
    
    # Динамическая анимация прогресс-бара
    for i in range(5):  # 5 шагов анимации
        await asyncio.sleep(0.5)  # Задержка для имитации процесса
        await query.edit_message_text(text="Генерация рецепта... 🍾🥂" + " " * i + "💧")
    
    try:
        bot_state = BotState()
        style = bot_state.convert_style(query.data.split("_")[1])
        logger.info(f"Попытка генерации рецепта с ингредиентами: {user_data['ingredients']}, стиль: {style}")
        
        recipe = await bot_state.recipe_finder.find_recipe(user_data["ingredients"], style)
        logger.info("Рецепт успешно сгенерирован")
        
        user_data["recipe"] = recipe
        user_data["style"] = style
        
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"🍹 Ваш рецепт:\n\n{recipe}"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("🔊 Запустить синтез речи", callback_data="synthesize"),
                InlineKeyboardButton("🔄 Новый поиск", callback_data="new_search")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Что бы вы хотели сделать дальше?",
            reply_markup=reply_markup,
        )
        return GENERATING_SPEECH
        
    except Exception as e:
        logger.error(f"Ошибка при генерации рецепта: {str(e)}", exc_info=True)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Произошла ошибка при генерации рецепта. Попробуйте еще раз, либо заведите дефект в изъяноотслеживателе https://github.com/trofimovelijah/cocktail_qr/issues/new"
        )
        return START

async def synthesize_speech(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text(text="Синтез речи...⏳")
    
    bot_state = BotState()
    audio_path = await bot_state.speech_generator.generate_speech(
        context.user_data["recipe"],
        context.user_data["style"]
    )
    
    with open(audio_path, "rb") as audio_file:
        await context.bot.send_audio(
            chat_id=query.message.chat_id,
            audio=audio_file,
            title="Рецепт коктейля"
        )
    
    # Очищаем данные пользователя после синтеза речи
    context.user_data.clear()
    
    keyboard = [[InlineKeyboardButton("🔄 Новый поиск", callback_data="start_bot")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="Что бы вы хотели сделать дальше?",
        reply_markup=reply_markup,
    )
    return START

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    context.user_data.clear()
    await query.edit_message_text(text="Действие отменено")
    return await start_command(update, context)

async def new_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    # Очищаем данные пользователя
    context.user_data.clear()
    
    # Создаем кнопку для начала нового поиска
    keyboard = [[InlineKeyboardButton("➕ Добавить ингредиент", callback_data="add_ingredient")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Отправляем сообщение с кнопкой
    await query.edit_message_text(
        text="Начнем новый поиск! Нажмите кнопку, чтобы добавить ингредиент.",
        reply_markup=reply_markup
    )
    
    return INGREDIENTS

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Произошла ошибка при обработке запроса:", exc_info=context.error)
    
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "Извините, произошла ошибка при обработке вашего запроса. Попробуйте позже."
            )
    except Exception as e:
        logger.error(f"Ошибка при отправке сообщения об ошибке: {e}")

def create_dynamic_keyboard(buttons):
    # Разбиваем кнопки на ряды
    rows = []
    for i in range(0, len(buttons), 2):  # Два элемента в ряд
        rows.append(buttons[i:i + 2])
    return InlineKeyboardMarkup(rows)

def main() -> None:
    application = Application.builder().token(TOKEN).build()
    
    # Добавляем обработчик ошибок
    application.add_error_handler(error_handler)
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            START: [
                CallbackQueryHandler(handle_ingredients, pattern="^start_bot$"),
            ],
            INGREDIENTS: [
                CallbackQueryHandler(activate_input, pattern="^activate_input$"),
                CallbackQueryHandler(select_style, pattern="^generate$"),
                CallbackQueryHandler(handle_ingredients, pattern="^add_ingredient$"),
                CallbackQueryHandler(cancel, pattern="^cancel$"),
                CallbackQueryHandler(new_search, pattern="^new_search$"),
            ],
            "WAITING_INPUT": [  # Новое состояние
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_ingredient),
            ],
            STYLE: [
                CallbackQueryHandler(generate_recipe, pattern="^style_"),
                CallbackQueryHandler(cancel, pattern="^cancel$"),
            ],
            GENERATING_SPEECH: [
                CallbackQueryHandler(synthesize_speech, pattern="^synthesize$"),
                CallbackQueryHandler(new_search, pattern="^new_search$"),
                CallbackQueryHandler(cancel, pattern="^cancel$"),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)
    
    # Запуск бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()