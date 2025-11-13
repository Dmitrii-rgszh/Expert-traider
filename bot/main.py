import logging
import os
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBAPP_URL = os.getenv("WEBAPP_URL", "https://www.glazok.site")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[KeyboardButton(text="Открыть приложение", web_app=WebAppInfo(url=WEBAPP_URL))]]
    await update.message.reply_text(
        "Добро пожаловать! Откройте Miniapp для анализа новостей.",
        reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True),
    )


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
