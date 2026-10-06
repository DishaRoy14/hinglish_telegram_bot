
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from agent import AUDIO_EXTENSIONS, OUT_DIR, HinglishAgent

API_KEY = os.getenv("API_KEY", "")

state = {}
lock = threading.Lock()


def check_key(x_api_key: str = Header(default="")):
    
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Bad API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["agent"] = HinglishAgent()
    yield
    state.clear()


app = FastAPI(
    title="Hinglish Business Agent",
    lifespan=lifespan,
    dependencies=[Depends(check_key)],
)


def _run(user_input: str) -> dict:
    with lock:
        result = state["agent"].respond(user_input)
    result["audio_url"] = f"/audio/{Path(result['reply_wav']).name}"
    return result


@app.post("/ask-audio", summary="Upload a voice question, get reply text + audio link")
async def ask_audio(file: UploadFile = File(..., description=".wav .ogg .mp3 .m4a .flac")):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use one of {sorted(AUDIO_EXTENSIONS)}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        result = await run_in_threadpool(_run, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    result["input"] = file.filename
    return result


@app.post("/ask-text", summary="Send a text question, get reply text + audio link")
async def ask_text(text: str = Form(..., description="e.g. Rahul Sharma ka pending invoice kitna hai?")):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    return await run_in_threadpool(_run, text)


@app.get("/audio/{filename}", summary="Download a generated reply WAV", response_class=FileResponse)
async def get_audio(filename: str):
    path = OUT_DIR / Path(filename).name
    if path.suffix != ".wav" or not path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.post("/reset", summary="Clear conversation history")
async def reset():
    with lock:
        state["agent"].history = []
    return {"status": "history cleared"}