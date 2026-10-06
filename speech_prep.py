"""Text preparation before TTS. No models needed, so it is easy to test."""
import re

# hindi numbers in words(0-99)
HINDI_0_99 = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]

### english to devanagari 
LEXICON = {
    "invoice": "इनवॉइस", "payment": "पेमेंट", "order": "ऑर्डर", "refund": "रिफंड",
    "deadline": "डेडलाइन", "client": "क्लाइंट", "project": "प्रोजेक्ट", "task": "टास्क",
    "status": "स्टेटस", "pending": "पेंडिंग", "update": "अपडेट", "amount": "अमाउंट",
    "report": "रिपोर्ट", "team": "टीम", "meeting": "मीटिंग", "blocker": "ब्लॉकर",
    "sales": "सेल्स", "target": "टारगेट", "advance": "एडवांस", "server": "सर्वर",
    "access": "एक्सेस", "outstanding": "आउटस्टैंडिंग", "submit": "सबमिट",
}


def hindi_number(n: int) -> str:
    """225000 -> 'दो लाख पच्चीस हज़ार' (Indian system: करोड़, लाख, हज़ार, सौ)."""
    if n < 100:
        return HINDI_0_99[n]
    parts = []
    for unit, word in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        q, n = divmod(n, unit)
        if q:
            parts.append(f"{hindi_number(q)} {word}")
    if n:
        parts.append(HINDI_0_99[n])
    return " ".join(parts)


def normalize_numbers(text: str) -> str:
    """Rs 45000 / ₹2,25,000 -> spoken Hindi with 'रुपये'; any other number -> spoken Hindi."""
    def money(m):
        return hindi_number(int(m.group(1).replace(",", ""))) + " रुपये"

    def plain(m):
        return hindi_number(int(m.group(0).replace(",", "")))

    text = re.sub(r"(?:₹|Rs\.?|INR)\s?(\d+(?:,\d+)*)", money, text, flags=re.IGNORECASE)
    return re.sub(r"\d+(?:,\d+)*", plain, text)


def apply_lexicon(text: str) -> str:
    return re.sub(r"[A-Za-z]+", lambda m: LEXICON.get(m.group(0).lower(), m.group(0)), text)


def split_sentences(text: str) -> list:
    
    return [s.strip() for s in re.split(r"(?<=[।.?!])\s+", text.strip()) if s.strip()]