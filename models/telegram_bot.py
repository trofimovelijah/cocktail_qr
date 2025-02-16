import os
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
    filters,
)
from recipe_finder import RecipeFinder
from speech_generator import SpeechGenerator

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Определение состояний
START, INGREDIENTS, STYLE, GENERATING_RECIPE, GENERATING_SPEECH = range(5)

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
            "M": "4"   # Минимализм
        }
        return style_map.get(style_code, "4")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [[InlineKeyboardButton("Запустить бот", callback_data="start_bot")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Привет! Я бот для создания коктейлей. Нажми кнопку ниже, чтобы начать!",
        reply_markup=reply_markup,
    )
    return START

async def handle_ingredients(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    context.user_data.setdefault("ingredients", [])
    await query.edit_message_text(text="Введите ингредиент:")
    return INGREDIENTS

async def process_ingredient(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_data = context.user_data
    ingredient = update.message.text
    
    if len(user_data.get("ingredients", [])) >= 3:
        await update.message.reply_text("Максимум 3 ингредиента!")
        return INGREDIENTS
    
    user_data.setdefault("ingredients", []).append(ingredient)
    
    buttons = []
    if len(user_data["ingredients"]) < 3:
        buttons.append(InlineKeyboardButton("Добавить ингредиент", callback_data="add_ingredient"))
    buttons.append(InlineKeyboardButton("Сгенерировать", callback_data="generate"))
    
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
    
    keyboard = [
        [
            InlineKeyboardButton("Лавкрафт", callback_data="style_L"),
            InlineKeyboardButton("Типикал гоп", callback_data="style_G"),
            InlineKeyboardButton("Берроуз", callback_data="style_U"),
            InlineKeyboardButton("Меню", callback_data="style_M"),
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
    
    await query.edit_message_text(text="Генерация рецепта...⏳")
    
    bot_state = BotState()
    style = bot_state.convert_style(query.data.split("_")[1])
    recipe = await bot_state.recipe_finder.find_recipe(user_data["ingredients"], style)
    user_data["recipe"] = recipe
    user_data["style"] = style  # Сохраняем стиль для генерации речи
    
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=f"🍹 Ваш рецепт:\n\n{recipe}"
    )
    
    keyboard = [[InlineKeyboardButton("Запустить синтез речи", callback_data="synthesize")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="Синтезировать аудиоверсию?",
        reply_markup=reply_markup,
    )
    return GENERATING_SPEECH

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
    
    keyboard = [[InlineKeyboardButton("Новый поиск", callback_data="new_search")]]
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
    context.user_data.clear()
    return await start_command(update, context)

def main() -> None:
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            START: [
                CallbackQueryHandler(handle_ingredients, pattern="^start_bot$"),
            ],
            INGREDIENTS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_ingredient),
                CallbackQueryHandler(select_style, pattern="^generate$"),
                CallbackQueryHandler(handle_ingredients, pattern="^add_ingredient$"),
                CallbackQueryHandler(cancel, pattern="^cancel$"),
            ],
            STYLE: [
                CallbackQueryHandler(generate_recipe, pattern="^style_"),
                CallbackQueryHandler(cancel, pattern="^cancel$"),
            ],
            GENERATING_SPEECH: [
                CallbackQueryHandler(synthesize_speech, pattern="^synthesize$"),
                CallbackQueryHandler(cancel, pattern="^cancel$"),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)
    application.add_handler(CallbackQueryHandler(new_search, pattern="^new_search$"))
    
    application.run_polling()

if __name__ == "__main__":
    main()