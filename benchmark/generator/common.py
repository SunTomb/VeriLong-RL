import random

from benchmark.schemas.task import EvidenceDocument


def make_rng(seed: int) -> random.Random:
    return random.Random(seed)


def evidence_id(index: int) -> str:
    return f"E{index:02d}"


def doc_id(index: int) -> str:
    return f"D{index:02d}"


def approximate_tokens(text: str) -> int:
    """Approximate tokens with whitespace-delimited words.

    The pilot generators only need stable sizing without a tokenizer dependency,
    so each whitespace-delimited word is counted as one approximate token.
    """
    return len(text.split())


def pad_neutral_documents(
    documents: list[EvidenceDocument],
    target_context_tokens: int,
    rng: random.Random,
) -> list[EvidenceDocument]:
    padded = list(documents)
    current_tokens = sum(approximate_tokens(document.text) for document in padded)
    neutral_topics = [
        "archive shelving",
        "weather station calibration",
        "library lighting",
        "garden inventory",
        "harbor maintenance",
        "museum cataloging",
    ]

    next_index = len(padded) + 1
    while current_tokens < target_context_tokens:
        topic = rng.choice(neutral_topics)
        batch = rng.randint(1000, 9999)
        text = (
            f"Neutral memorandum {batch} records routine observations about {topic}. "
            "The note lists ordinary scheduling details, supply checks, and review reminders. "
            "It contains no project codes, eligibility decisions, protocol mappings, or active updates."
        )
        document = EvidenceDocument(
            doc_id=doc_id(next_index),
            evidence_id=evidence_id(next_index),
            text=text,
            role="neutral",
        )
        padded.append(document)
        current_tokens += approximate_tokens(text)
        next_index += 1

    return padded
