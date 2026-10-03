import re
from collections.abc import Iterable

from app.data.demo_sources import SOURCES
from app.models import Source

_WORD = re.compile(r"[a-zA-Z]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in",
    "is", "it", "my", "of", "should", "the", "to", "what", "which", "with", "you",
}
_ALIASES = {
    "lifting": {"resistance", "strength", "muscle"},
    "gym": {"resistance", "strength", "muscle"},
    "cramps": {"magnesium"},
    "cramp": {"magnesium"},
    "sun": {"vitamin", "d"},
    "bones": {"vitamin", "d"},
    "food": {"meal", "diet", "plate"},
}


def tokens(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text) if word.lower() not in _STOP_WORDS}


def expanded_tokens(text: str) -> set[str]:
    result = tokens(text)
    for token in tuple(result):
        result.update(_ALIASES.get(token, set()))
    return result


def source_to_model(source: dict) -> Source:
    return Source(
        id=source["id"],
        organization=source["organization"],
        title=source["title"],
        url=source["url"],
        excerpt=source["text"],
    )


class DemoRetriever:
    """Transparent token/alias retriever for an offline demo.

    It is intentionally simple and inspectable. The live adapter will replace this
    with embedded chunks in Pinecone while preserving the `retrieve` interface.
    """

    def retrieve(self, question: str, limit: int = 2) -> list[Source]:
        query = expanded_tokens(question)
        scored: list[tuple[int, dict]] = []
        for source in SOURCES:
            source_terms = source["topics"] | tokens(source["text"])
            # Topic tags mimic metadata filters/semantic prominence in the
            # production index: a direct subject match outweighs a generic word.
            score = len(query & source_terms) + (2 * len(query & source["topics"]))
            if score:
                scored.append((score, source))
        scored.sort(key=lambda row: row[0], reverse=True)
        if not scored:
            return []
        # Avoid padding a response with a weak, coincidental term overlap such as
        # "adults". In production this becomes an embedding-score threshold.
        cutoff = 2 if scored[0][0] >= 2 else scored[0][0]
        return [source_to_model(source) for score, source in scored if score >= cutoff][:limit]

    def all_sources(self) -> Iterable[Source]:
        return (source_to_model(source) for source in SOURCES)
