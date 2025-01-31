import os
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, ConversationHandler, filters
from recipe_finder import RecipeFinder
from speech_generator import SpeechGenerator

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Состояния разговора
INGREDIENT_INPUT, STYLE_CHOICE, GENERATION = range(3)

# Стили
STYLES = {
    "1": "стиль космического ужаса",
    "2": "гопническо-быдляцкий жаргон",
    "3": "экспериментальный стиль",
    "4": "классический стиль"
}

class CocktailBot:
    def __init__(self):
        self.recipe_finder = RecipeFinder()
        self.speech_generator = SpeechGenerator()
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        context.user_data['ingredients'] = []
        
        await update.message.reply_text(
            "Привет! Я помогу найти рецепт коктейля.\n"
            "Введите первый ингредиент:"
        )
        return INGREDIENT_INPUT

    async def add_ingredient(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик ввода ингредиента"""
        ingredient = update.message.text
        if 'ingredients' not in context.user_data:
            context.user_data['ingredients'] = []
        
        context.user_data['ingredients'].append(ingredient)
        
        keyboard = [
            [
                InlineKeyboardButton("Добавить ещё", callback_data='more'),
                InlineKeyboardButton("Готово", callback_data='done')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"Добавлен ингредиент: {ingredient}\n"
            f"Текущий список: {', '.join(context.user_data['ingredients'])}",
            reply_markup=reply_markup
        )
        return INGREDIENT_INPUT

    async def handle_ingredient_choice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик выбора действия после ввода ингредиента"""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'more':
            await query.edit_message_text("Введите следующий ингредиент:")
            return INGREDIENT_INPUT
        
        elif query.data == 'done':
            keyboard = [[InlineKeyboardButton(desc, callback_data=str(i))] 
                       for i, desc in STYLES.items()]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                "Выберите стиль изложения рецепта:",
                reply_markup=reply_markup
            )
            return STYLE_CHOICE

    async def handle_style_choice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик выбора стиля"""
        query = update.callback_query
        await query.answer()
        
        style = query.data
        context.user_data['style'] = style
        
        keyboard = [
            [
                InlineKeyboardButton("Генерировать", callback_data='generate'),
                InlineKeyboardButton("Отмена", callback_data='cancel')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"Выбран стиль: {STYLES[style]}\n"
            "Начать генерацию?",
            reply_markup=reply_markup
        )
        return GENERATION

    async def generate_recipe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик генерации рецепта"""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'cancel':
            await query.edit_message_text("Генерация отменена. Напишите /start для нового поиска.")
            return ConversationHandler.END
        
        await query.edit_message_text("Генерирую рецепт...")
        
        # Получение рецепта
        ingredients = context.user_data['ingredients']
        style = context.user_data['style']
        recipe_text = await self.recipe_finder.find_recipe(ingredients, style)
        
        # Генерация аудио
        audio_path = await self.speech_generator.generate_speech(recipe_text, style)
        
        # Отправка результатов
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=recipe_text
        )
        
        if audio_path:
            with open(audio_path, 'rb') as audio:
                await context.bot.send_voice(
                    chat_id=update.effective_chat.id,
                    voice=audio
                )
        
        return ConversationHandler.END

    def run(self):
        """Запуск бота"""
        application = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
        
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', self.start)],
            states={
                INGREDIENT_INPUT: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.add_ingredient),
                    CallbackQueryHandler(self.handle_ingredient_choice)
                ],
                STYLE_CHOICE: [
                    CallbackQueryHandler(self.handle_style_choice)
                ],
                GENERATION: [
                    CallbackQueryHandler(self.generate_recipe)
                ]
            },
            fallbacks=[CommandHandler('start', self.start)]
        )
        
        application.add_handler(conv_handler)
        application.run_polling()

if __name__ == '__main__':
    bot = CocktailBot()
    bot.run() 