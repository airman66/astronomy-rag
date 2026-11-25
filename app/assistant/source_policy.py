"""Resolve a source named in the question against actual catalogue records."""

import re

from app.assistant.retrieval import Evidence
from app.catalog.storage import normalized_words

GLOSSARY_FORBIDDEN = re.compile(
    r"(?:^|[.!?,;]\s+)(?:(?:otherwise|в противном случае)[,\s]+)?"
    r"(?:(?:do not|don['’]t|never)\s+(?:(?:answer|explain)\s+from|use|cite)\s+(?:the\s+)?glossary"
    r"|не\s+(?:(?:отвечай(?:те)?|объясняй(?:те)?)\s+по\s+справочнику|используй(?:те)?\s+справочник))"
    r"(?=[.!?](?:\s|$)|\s*$)",
    re.IGNORECASE,
)
ARTICLE_ID = re.compile(r"\barticle:[a-z0-9_:-]+(?:\.[a-z0-9_:-]+)*\b", re.IGNORECASE)
QUOTED_TITLE = re.compile(
    r'\b(?P<kind>articles?|стать\w*|mini[- ]review|мини[- ]обзор\w*)\s+'
    r'(?:OAE\s+)?(?:(?:named|titled|called)\s+)?[«“"](?P<title>[^»”"]+)[»”"]',
    re.IGNORECASE,
)
MINI_REVIEW = re.compile(r"\b(?:mini review|мини обзор\w*)\b")
OPEN_ARTICLE = re.compile(r"\b(?:this article|эт[аоуй]\s+стать\w*)\b", re.IGNORECASE)
ARTICLE_TOPIC = re.compile(r"\b(?:articles?\s+(?:on|about)|стать\w*\s+(?:о|об|про))\s+", re.IGNORECASE)
GENERIC_ARTICLE = re.compile(r"(?:(?:the |an? )?article|(?:по |из )?стать\w*)", re.IGNORECASE)


def _title_key(text: str) -> tuple[str, bool]:
    title = " ".join(word for word in normalized_words(text) if word != "oae")
    is_mini_review = MINI_REVIEW.search(title) is not None
    return " ".join(MINI_REVIEW.sub("", title).split()), is_mini_review


def named_articles(
    question: str, reference: str | None, evidence: dict[str, Evidence], *, page: str | None, require_match: bool,
) -> set[str] | None:
    """None means no named source. An empty set means no verified match."""
    ids = {match.casefold() for match in ARTICLE_ID.findall(question)}
    if ids:
        return ids if len(ids) == 1 else set()

    quoted = list(QUOTED_TITLE.finditer(question))
    if len(quoted) > 1:
        return set()
    if quoted:
        match = quoted[0]
        key = _title_key(match["title"])
        if _title_key(match["kind"])[1]:
            key = (key[0], True)
    elif OPEN_ARTICLE.search(question):
        return {page} if page and ARTICLE_ID.fullmatch(page) else set()
    elif reference:
        if ARTICLE_TOPIC.search(question) or GENERIC_ARTICLE.fullmatch(reference):
            return None
        if " ".join(reference.split()) not in " ".join(question.split()):
            return set() if require_match else None
        key = _title_key(reference)
    else:
        return None

    records = {item.record["id"]: item.record for item in evidence.values() if item.kind == "article"}
    matches = {
        record_id for record_id, record in records.items()
        if any(_title_key(localized["title"]) == key for localized in record["locales"].values())
    }
    if len(matches) == 1:
        return matches
    return set() if matches or quoted or require_match else None
