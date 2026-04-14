import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-záàâãéêíóôõúç\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
