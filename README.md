# Hinglish Business Assistant (Telegram Bot)

A self-hosted Telegram bot that answers business questions in conversational Hinglish. It accepts text messages and voice notes, retrieves company records from a ChromaDB knowledge base, generates a grounded answer with a local LLM, and replies with Hinglish text or a Hinglish voice note. All models run locally. No paid third-party APIs are used.

## Architecture

The system runs as three cooperating processes on one machine.

1. **FastAPI server** (`app.py`, `agent.py`): receives text or audio, runs the pipeline, and returns the reply text and a link to the reply audio.
2. **Ollama**: serves the LLM locally.
3. **Telegram bot** (`bot.py`): forwards user messages to the FastAPI server and sends the result back to the user.

### Pipeline

```
Telegram (text or voice note)
        |
   bot.py  ---->  FastAPI (/ask-text or /ask-audio)
                        |
        Voice input: Zero-STT Hinglish  -> transcript
                        |
        Retrieval: ChromaDB (exact ID, employee, client, aggregation, semantic search)
                        |
        LLM: Gemma 4 E4B via Ollama (answer grounded in retrieved records)
                        |
        TTS: Veena (default) with Kokoro fallback  -> WAV audio
                        |
   JSON response (reply text + audio URL)
        |
   bot.py  ---->  Telegram (Hinglish text OR Hinglish voice note, chosen randomly)
```

### Components

| Layer | Implementation |
|---|---|
| Telegram gateway | python-telegram-bot (long polling) |
| API server | FastAPI with optional API-key protection |
| Speech-to-text | Zero-STT Hinglish (`shunyalabs/zero-stt-hinglish`) |
| LLM | Gemma 4 E4B served by Ollama (`gemma4:e4b`) |
| Knowledge base | ChromaDB (`company_records` collection) |
| Text-to-speech | Veena (4-bit) by default, Kokoro-82M as fallback |
| Audio output | 24 kHz WAV, converted to OGG/Opus for Telegram voice notes |

### Project structure

```
app.py               FastAPI server (endpoints, API-key check)
agent.py             Pipeline: STT, retrieval, LLM, TTS, JSON output
retrieval.py         ChromaDB retrieval and query routing
speech_prep.py       Number and text normalisation before TTS
tts_engines.py       Veena and Kokoro engines
bot.py               Telegram bot
generate_test_cases.py  Generates test audio with Veena
requirements.txt     Python dependencies
```

## Prerequisites

- Windows with an NVIDIA GPU (8 GB VRAM or more recommended) and a recent driver
- Python 3.11 and a virtual environment (conda or venv)
- Ollama, with the model pulled: `ollama pull gemma4:e4b`
- PyTorch with CUDA, installed from https://pytorch.org/get-started/locally before the other packages
- A Telegram bot token from @BotFather (`/newbot`)
- `espeak-ng` installed on the system (needed by Kokoro)
- Cloudflared, only if the API must be reachable from another machine

## Setup

### 1. Clone the repository and install dependencies

```
git clone https://github.com/DishaRoy14/hinglish_telegram_bot.git
cd hinglish_telegram_bot
pip install -r requirements.txt
```

### 2. Get the ChromaDB knowledge base

The `chroma_db` folder is not stored in this repository. Obtain it from the project owner and place it in the project root, so the path is `./chroma_db`. The collection name must be `company_records`.

### 3. Create the `.env` file

Create a file named `.env` in the project root:

```
TELEGRAM_BOT_TOKEN=your-token-from-BotFather
API_URL=http://localhost:8000
API_KEY=any-long-random-string
```

- Each developer can create their own bot token. Running two bots with the same token causes a `Conflict` error.
- `API_KEY` can be any random string. The same value must be set in the terminal that runs the API server (see below).
- If a Cloudflare tunnel is used, replace `API_URL` with the tunnel URL after Step 2 of "Running the system".


## Running the system

Use three separate terminals, started in this order. Ollama must already be running.

### To run ollama: 
In terminal run: **ollama run gemma4:e4b**

### Terminal 1: API server

```
set API_KEY=your-same-random-string
uvicorn app:app --host 127.0.0.1 --port 8000
```


### Terminal 2: Cloudflare tunnel (optional)

Needed only when the bot runs on a different machine from the API server. If both run on the same PC, skip this terminal and keep `API_URL=http://localhost:8000`.

```
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8000
```

Copy the `https://....trycloudflare.com` URL printed in the log, set it as `API_URL` in `.env`, and keep this terminal open. The URL changes every time the tunnel is restarted. If the cloudflared command is not recognised, open a new terminal outside the editor, or run the executable by its full path.

### Terminal 3: Telegram bot

```
python bot.py
```

The terminal should print `Bot is running...` with no errors. Restart the bot whenever `.env` is changed.

## Usage

1. Open the bot in Telegram and send `/start`.
2. Send a Hinglish text message or a voice note, for example:
   - `Rahul Sharma ka pending invoice kitna hai?`
   - `Engineering department mein kitne log kaam karte hain?`
3. The bot replies with a Hinglish answer. Each reply is delivered randomly as either text or a voice note.
4. Send `/reset` to clear the conversation history.

Replies take time because speech synthesis runs on every request. Wait for the reply before sending the next message.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ask-text` | Form field `text`; returns reply text and audio URL |
| POST | `/ask-audio` | Audio file upload (`.wav .ogg .mp3 .m4a .flac`); returns transcript, reply text and audio URL |
| GET | `/audio/{filename}` | Downloads a generated reply WAV |
| POST | `/reset` | Clears conversation history |

When `API_KEY` is set, every request must include the header `X-API-Key`. Interactive documentation is available at `/docs`.

Each request also writes a `reply_<timestamp>.wav` and `.json` file to the `outputs/` folder. The JSON contains the transcript, reply text, TTS engine used and per-stage timings.

## Configuration

Optional environment variables (set before starting uvicorn):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `gemma4:e4b` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `STT_MODEL` | `shunyalabs/zero-stt-hinglish` | Speech-to-text model |
| `TTS_ENGINE` | `veena` | `veena` or `kokoro` |
| `VEENA_SPEAKER` | `kavya` | Veena voice |
| `KOKORO_VOICE` | `hf_alpha` | Kokoro voice |
| `CHROMA_DB_PATH` | `./chroma_db` | Location of the knowledge base |
| `RETRIEVAL_TOP_K` | `5` | Records returned by semantic search |
| `OUT_DIR` | `outputs` | Output folder for generated files |

## Known limitations

- **Latency:** the language model responds in about 3 seconds, but Veena speech synthesis is the main bottleneck (about 47 seconds for a short reply on an 8 GB GPU). Setting `TTS_ENGINE=kokoro` is much faster at some cost in voice quality.
- **Shared conversation history:** history is global to the server, so all Telegram users share one conversation until `/reset` is called.
- **Sequential processing:** requests are handled one at a time.
- **Hindi retrieval:** keyword-based routing is written for Roman Hinglish. Devanagari queries may fall back to semantic search and return less precise records.
- **Cloudflare quick tunnels:** requests longer than about 100 seconds are cut off, and the URL changes on every restart.
- **GPU memory:** the LLM, speech-to-text model and Veena share one GPU. Close other GPU applications before running.

## Reference

The design goals, component alternatives and trade-offs are documented in the team's architecture specification, "Hinglish Voice and Text Telegram Bot: System Architecture, Alternatives and Trade-offs" (v1.0).
