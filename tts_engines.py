"""Two TTS engines with the same interface."""

import re
import numpy as np

from speech_prep import (
    apply_lexicon,
    normalize_numbers,
    split_sentences,
)


START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
AUDIO_BASE = 128266

VEENA_SPEAKERS = [
    "kavya",
    "agastya",
    "maitri",
    "vinaya",
]


def deinterleave_snac(tokens):
    tokens = tokens[:len(tokens) // 7 * 7]

    if not tokens:
        raise ValueError("No audio tokens generated")

    lvl = [[], [], []]

    off = [
        AUDIO_BASE + i * 4096
        for i in range(7)
    ]

    for i in range(0, len(tokens), 7):
        lvl[0].append(tokens[i] - off[0])

        lvl[1].append(tokens[i + 1] - off[1])
        lvl[1].append(tokens[i + 4] - off[4])

        lvl[2].append(tokens[i + 2] - off[2])
        lvl[2].append(tokens[i + 3] - off[3])
        lvl[2].append(tokens[i + 5] - off[5])
        lvl[2].append(tokens[i + 6] - off[6])

    if any(
        c < 0 or c > 4095
        for level in lvl
        for c in level
    ):
        raise ValueError("Invalid SNAC token values")

    return lvl


class VeenaEngine:
    sample_rate = 24000

    def __init__(self, speaker="kavya"):
        import torch
        from snac import SNAC
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if speaker not in VEENA_SPEAKERS:
            raise ValueError(
                f"speaker must be one of {VEENA_SPEAKERS}"
            )

        self.torch = torch
        self.speaker = speaker

        use_bf16 = (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16
                if use_bf16
                else torch.float16
            ),
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "maya-research/veena-tts",
            quantization_config=quant,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "maya-research/veena-tts",
            trust_remote_code=True,
        )

        self.snac = (
            SNAC
            .from_pretrained(
                "hubertsiuzdak/snac_24khz"
            )
            .eval()
            .to(self.model.device)
        )

        print("Veena device:", self.model.device)
        print(
            "SNAC device:",
            next(self.snac.parameters()).device
        )

    def prepare(self, text):
        return normalize_numbers(text)

    def _speak_sentence(self, text):
        torch = self.torch

        prompt_tokens = self.tokenizer.encode(
            f"<spk_{self.speaker}> {text}",
            add_special_tokens=False,
        )

        input_tokens = [
            START_OF_HUMAN,
            *prompt_tokens,
            END_OF_HUMAN,
            START_OF_AI,
            START_OF_SPEECH,
        ]

        input_ids = torch.tensor(
            [input_tokens],
            device=self.model.device,
        )

        # ORIGINAL WORKING LIMIT
        max_tokens = min(
            int(len(text) * 1.3) * 7 + 21,
            700,
        )

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[
                    END_OF_SPEECH,
                    END_OF_AI,
                ],
            )

        generated = output[0][
            len(input_tokens):
        ].tolist()

        snac_tokens = [
            t
            for t in generated
            if AUDIO_BASE <= t < AUDIO_BASE + 7 * 4096
        ]

        levels = deinterleave_snac(
            snac_tokens
        )

        snac_device = next(
            self.snac.parameters()
        ).device

        codes = [
            torch.tensor(
                level,
                dtype=torch.int32,
                device=snac_device,
            ).unsqueeze(0)
            for level in levels
        ]

        with torch.no_grad():
            audio = self.snac.decode(codes)

        return (
            audio
            .squeeze()
            .clamp(-1, 1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def synthesize(self, text):
        gap = np.zeros(
            int(0.2 * self.sample_rate),
            dtype=np.float32,
        )

        parts = []

        for sentence in split_sentences(text):
            parts += [
                self._speak_sentence(sentence),
                gap,
            ]

        return np.concatenate(parts)


class KokoroEngine:
    sample_rate = 24000

    def __init__(self, voice="hf_alpha"):
        from kokoro import KPipeline

        self.voice = voice
        self.pipe = KPipeline(lang_code="h")
        self.unknown_words = set()

    def prepare(self, text):
        text = apply_lexicon(
            normalize_numbers(text)
        )

        self.unknown_words.update(
            re.findall(r"[A-Za-z]+", text)
        )

        return text

    def synthesize(self, text):
        chunks = []

        for _, _, audio in self.pipe(
            text,
            voice=self.voice,
            speed=1.0,
        ):
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            else:
                audio = np.asarray(audio)

            chunks.append(audio)

        if not chunks:
            raise ValueError(
                f"Kokoro produced no audio for this text: {text!r}"
            )

        return np.concatenate(
            chunks
        ).astype(np.float32)

    