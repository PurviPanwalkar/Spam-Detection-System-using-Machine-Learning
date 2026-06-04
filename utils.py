import re


SPAM_KEYWORDS = [
    "free",
    "win",
    "winner",
    "won",
    "prize",
    "cash",
    "urgent",
    "claim",
    "click",
    "offer",
    "reward",
    "selected",
    "guaranteed",
    "call",
    "txt",
    "congratulations",
    "limited",
]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def analyze_message(text):
    lower_text = text.lower()
    words = re.findall(r"\b\w+\b", lower_text)
    matched_keywords = sorted({word for word in SPAM_KEYWORDS if word in words})

    return {
        "characters": len(text),
        "words": len(words),
        "has_link": bool(re.search(r"http\S+|www\.\S+", lower_text)),
        "has_phone": bool(re.search(r"\b\d{10,}\b", text)),
        "has_money": bool(re.search(r"[$₹]\s?\d+|\b\d+\s?(rs|rupees|dollars|usd)\b", lower_text)),
        "suspicious_keywords": matched_keywords,
    }
