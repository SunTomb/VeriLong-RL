import re
import string


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def exact_match(prediction: str | None, gold: str) -> float:
    if prediction is None:
        return 0.0
    return 1.0 if prediction.strip() == gold.strip() else 0.0


def normalized_match(prediction: str | None, gold: str) -> float:
    if prediction is None:
        return 0.0
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = _ARTICLES_RE.sub(" ", without_punctuation)
    return _WHITESPACE_RE.sub(" ", without_articles).strip()
