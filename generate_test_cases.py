from pathlib import Path
import soundfile as sf

from tts_engines import VeenaEngine


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    # ---------------- HINGLISH ----------------

    "INV-1002 ka due amount kitna hai?",
    "Rahul kis department mein kaam karta hai?",
    "Engineering department mein kitne log kaam karte hain?",
    "Abhi kitne invoices overdue hain?",
    "Saare overdue invoices ki total due amount kitni hai?",
    "Apex Tech Solutions ka koi pending invoice hai kya?",

    # ---------------- HINDI ----------------

    "INV-1002 की बकाया राशि कितनी है?",
    "राहुल किस विभाग में काम करते हैं?",
    "इंजीनियरिंग विभाग में कितने लोग काम करते हैं?",
    "अभी कितने चालान अतिदेय हैं?",
    "सभी अतिदेय चालानों की कुल बकाया राशि कितनी है?",
    "Apex Tech Solutions का कोई लंबित चालान है क्या?",
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "test_cases"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# INITIALIZE VEENA
# ============================================================

print("Loading Veena...")
engine = VeenaEngine()

print(f"\nGenerating {len(TEST_CASES)} test cases...\n")


# ============================================================
# GENERATE AUDIO
# ============================================================

for i, sentence in enumerate(TEST_CASES, start=1):

    print("=" * 70)
    print(f"Test case {i}")
    print(f"Text: {sentence}")

    output_path = OUTPUT_DIR / f"test_{i:02d}.wav"

    try:
        # Veena returns numpy.ndarray
        audio = engine.synthesize(sentence)

        # Convert numpy array -> WAV
        sf.write(
            str(output_path),
            audio,
            engine.sample_rate
        )

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"ERROR generating test case {i}: {e}")


print("\n" + "=" * 70)
print("All test cases generated.")
print(f"Output folder: {OUTPUT_DIR}")