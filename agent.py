
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
import requests
import librosa
import soundfile as sf

from retrieval import get_context
from tts_engines import KokoroEngine, VeenaEngine

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:e4b")
STT_MODEL = os.getenv("STT_MODEL", "shunyalabs/zero-stt-hinglish")

TTS_ENGINE = os.getenv("TTS_ENGINE", "veena")
VEENA_SPEAKER = os.getenv("VEENA_SPEAKER", "kavya")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "hf_alpha")
OUT_DIR = Path(os.getenv("OUT_DIR", "outputs"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))
HISTORY_TURNS = 6
AUDIO_EXTENSIONS = {".wav", ".ogg", ".mp3", ".m4a", ".flac"}


SYSTEM_PROMPT = """You are a business assistant for a company.

Rules:
1. Always reply in Hinglish: everyday Hindi mixed with common English business words. Write Hindi words in Devanagari and English words in Latin script.
2. Keep the reply short: 2 to 3 sentences, no lists.
3. Use ONLY the COMPANY DATA below to answer.
4. Copy names, amounts, dates and relevant business IDs exactly as written there.
5. Do NOT reveal private/internal identifiers such as Telegram Chat IDs unless the user explicitly asks for them.
6. If the answer is not in the COMPANY DATA, say clearly that you do not have that information. Never guess.
7. Use the earlier conversation to understand follow-up questions like "and its due date?"."""

def is_audio_input(value: str) -> bool:
    p = Path(value.strip().strip('"'))
    return p.suffix.lower() in AUDIO_EXTENSIONS and p.exists()


def save_json(result: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


class HinglishAgent:
    def __init__(self):
       ## from openai import OpenAI
        from transformers import pipeline
        import torch

        t0 = time.perf_counter()
        use_gpu = torch.cuda.is_available()
        print(f"GPU available: {use_gpu}")
        self.stt = pipeline(
            "automatic-speech-recognition",
            model=STT_MODEL,
            chunk_length_s=30,
            device=0 if use_gpu else -1,
            torch_dtype=torch.float16 if use_gpu else torch.float32,
        )
        ##self.llm = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

        self.last_tts_error = None
        self.tts = None
        if TTS_ENGINE == "veena":
            try:
                self.tts = VeenaEngine(VEENA_SPEAKER)
            except Exception as e:
                print(f"Veena failed to load ({e!r}); using Kokoro instead")
                traceback.print_exc()
        else:
            self.tts = KokoroEngine(KOKORO_VOICE)
        self._fallback = self.tts if isinstance(self.tts, KokoroEngine) else KokoroEngine(KOKORO_VOICE)

        self.history = []
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        
        try:
            requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": LLM_MODEL, "messages": [{"role": "user", "content": "hi"}],
        "stream": False, "think": False, "keep_alive": -1,
        "options": {"num_predict": 1, "num_ctx": 2048},
        }, timeout=120)
        except Exception as e:
            print(f"Ollama warm-up failed: {e!r}")
        active = type(self.tts or self._fallback).__name__
        print(f"Models ready in {time.perf_counter() - t0:.1f}s (TTS requested: {TTS_ENGINE}, active: {active})")
    def get_text(self, user_input: str, timings: dict):
        if is_audio_input(user_input):
            t = time.perf_counter()
            path = user_input.strip().strip('"')
            audio, sr = librosa.load(path, sr=16000, mono=True)
            transcript = self.stt({"raw": audio, "sampling_rate": sr})["text"].strip()
            timings["stt"] = time.perf_counter() - t
            return transcript, "audio"
        return user_input.strip(), "text"

    def ask_gemma(self, question: str, timings: dict) -> str:
        t = time.perf_counter()
        context = get_context(question)
        timings["retrieval"] = time.perf_counter() - t

        system = SYSTEM_PROMPT + "\n\nCOMPANY DATA:\n" + (context or "(no matching data found)")
        messages = [{"role": "system", "content": system}] + self.history + [{"role": "user", "content": question}]

        t = time.perf_counter()
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": -1,
            "options": {"temperature": 0.3, "num_predict": 150,"num_ctx": 2048},
            },
            timeout=120,
            )
        r.raise_for_status()
        data = r.json()
        reply = data["message"]["content"].strip()
        timings["llm"] = time.perf_counter() - t
        timings["llm_tokens"] = data.get("eval_count", 0)
        timings["llm_load"] = data.get("load_duration", 0) / 1e9
        timings["llm_prompt_eval"] = data.get("prompt_eval_duration", 0) / 1e9
        timings["llm_gen"] = data.get("eval_duration", 0) / 1e9
        timings["llm_prompt_tokens"] = data.get("prompt_eval_count", 0)
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
        self.history = self.history[-HISTORY_TURNS * 2:]
        return reply

    def speak(self, reply: str, wav_path: Path, timings: dict) -> str:
        
        t = time.perf_counter()
        engine = self.tts
        audio = None
        self.last_tts_error = None

        if engine is None:
            self.last_tts_error = "Veena failed to load at startup"
        else:
            try:
                audio = engine.synthesize(engine.prepare(reply))
            except Exception as e:
                self.last_tts_error = repr(e)
                print(f"TTS failed ({e!r}); falling back to Kokoro")
                traceback.print_exc()
                if isinstance(engine, KokoroEngine):
                    raise

        if audio is None:
            engine = self._fallback
            audio = engine.synthesize(engine.prepare(reply))

        sf.write(str(wav_path), audio, engine.sample_rate)
        timings["tts"] = time.perf_counter() - t
        return type(engine).__name__

    def respond(self, user_input: str) -> dict:
        timings = {}
        start = time.perf_counter()
        stem = "reply_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        wav_path = OUT_DIR / f"{stem}.wav"
        json_path = OUT_DIR / f"{stem}.json"

        question, input_type = self.get_text(user_input, timings)
        reply = self.ask_gemma(question, timings) if question else "आवाज़ clear नहीं आई, एक बार फिर भेजिए?"
        print(f"Gemma reply: {reply!r}")
        if not reply.strip():
            print("Gemma returned an empty reply (its thinking may have used all the tokens). Try a larger LLM_MAX_TOKENS.")
            reply = "मुझे अभी जवाब नहीं मिला, कृपया दोबारा पूछिए।"

        engine_used = self.speak(reply, wav_path, timings)
        timings["total"] = time.perf_counter() - start

        result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input_type": input_type,
            "input": user_input.strip(),
            "transcript": question,
            "reply_text": reply,
            "reply_wav": str(wav_path),
            "reply_json": str(json_path),
            "tts_engine": engine_used,
            "tts_fallback_reason": self.last_tts_error,
            "llm_model": LLM_MODEL,
            "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
        }
        save_json(result, json_path)
        return result


def show(result: dict):
    print(f"\n[{result['input_type']}] You said : {result['transcript']}")
    print(f"Agent reply  : {result['reply_text']}")
    print(f"Reply audio  : {result['reply_wav']}  ({result['tts_engine']})")
    if result.get("tts_fallback_reason"):
        print(f"Fallback why : {result['tts_fallback_reason']}")
    print(f"Reply JSON   : {result['reply_json']}")
    print("Timings (s)  : " + ", ".join(f"{k} {v:.2f}" for k, v in result["timings_seconds"].items()))


if __name__ == "__main__":
    agent = HinglishAgent()
    if len(sys.argv) > 1:
        show(agent.respond(" ".join(sys.argv[1:])))
    else:
        print("Type a question, or the path to an audio file. Type 'quit' to stop.")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"quit", "exit", ""}:
                break
            show(agent.respond(user_input))