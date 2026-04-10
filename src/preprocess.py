"""Text preprocessing helpers."""

import re
import string


def clean_text(text: str) -> str:
    """Normalize user text for baseline NLP models."""
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

