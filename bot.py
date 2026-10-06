"""Telegram bot -> FastAPI Hinglish agent.

Text message  -> POST /ask-text
Voice note    -> POST /ask-audio
Replies with the reply text and a voice message.
"""
import io
import os
import random
import httpx
import soundfile as sf
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
TIMEOUT = 300  # TTS can be slow


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Namaste! Text ya voice note bhejiye.")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(f"{API_URL}/reset", headers=HEADERS)
    await update.message.reply_text("Conversation history clear ho gayi.")


async def reply_with_result(update: Update, client: httpx.AsyncClient, result: dict):
    # 0:text reply, 1:voice reply
    if random.randint(0, 1) == 0:
        await update.message.reply_text(result["reply_text"])
        return

    wav = (await client.get(API_URL + result["audio_url"], headers=HEADERS)).content
    try:
        data, sr = sf.read(io.BytesIO(wav))
        ogg = io.BytesIO()
        sf.write(ogg, data, sr, format="OGG", subtype="OPUS")  
        ogg.seek(0)
        await update.message.reply_voice(voice=ogg)
    except Exception as e:
        print(f"Voice conversion failed ({e!r}); sending text instead")
        await update.message.reply_text(result["reply_text"])


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action("record_voice")
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(f"{API_URL}/ask-text", data={"text": update.message.text}, headers=HEADERS)
        r.raise_for_status()
        await reply_with_result(update, client, r.json())


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action("record_voice")
    tg_file = await update.message.voice.get_file()
    audio = bytes(await tg_file.download_as_bytearray())
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(
            f"{API_URL}/ask-audio",
            files={"file": ("voice.ogg", audio, "audio/ogg")},  
            headers=HEADERS,
        )
        r.raise_for_status()
        await reply_with_result(update, client, r.json())


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    print(f"Error: {context.error!r}")
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("Kuch gadbad ho gayi, thodi der baad try kijiye.")


def main():
    if not TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is missing. Put it in .env")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(on_error)
    print(f"Bot is running... (API: {API_URL})")
    app.run_polling()


if __name__ == "__main__":
    main()