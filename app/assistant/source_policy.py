"""Resolve explicit article sources and glossary permission from the question."""

import re
from dataclasses import dataclass

from app.catalog import storage

ARTICLE_REQUEST = re.compile(r"\b(?:articles?|стать\w*)\b", re.IGNORECASE)
ARTICLE_ID = re.compile(r"\barticle:[a-z0-9_:-]+(?:\.[a-z0-9_:-]+)*\b", re.IGNORECASE)
ARTICLE_LABEL = r"(?:articles?|стать\w*|mini[- ]review|мини[- ]обзор\w*)"
TITLE_PREFIX = rf"\b(?P<kind>{ARTICLE_LABEL})\s+(?:OAE\s+)?(?:(?P<named>named|titled|called)\s+)?"
QUOTED_TEXT = r"""«[^»]*»|“[^”]*”|"[^"]*"|‘[^’]*’|(?<!\w)'[^'\n]+'(?!\w)"""
QUOTED_TITLE = re.compile(TITLE_PREFIX + rf"(?P<title>{QUOTED_TEXT})", re.IGNORECASE)
QUOTES = re.compile(QUOTED_TEXT)
MINI_REVIEW = re.compile(r"\b(?:mini review|мини обзор\w*)\b")
OPEN_ARTICLE = re.compile(r"\b(?:this article|эт[аоуй]\s+стать\w*)\b", re.IGNORECASE)
ARTICLE_TOPIC = re.compile(r"\b(?:articles?\s+(?:on|about)|стать\w*\s+(?:о|об|про))\s+", re.IGNORECASE)
GENERIC_ARTICLE = re.compile(r"(?:(?:the |an? )?article|(?:по |из )?стать\w*)", re.IGNORECASE)
TITLE_CUE = re.compile(rf"{TITLE_PREFIX}|\b(?:according to|согласно)\s+", re.IGNORECASE)
TITLE_END = re.compile(r"^\s*(?:$|[,.!?;]|(?:and|и|compare|explain|summari[sz]e|сравни\w*|объясни\w*)\b)", re.IGNORECASE)

ARTICLE_ONLY = re.compile(
    r"\b(?:(?:only|exclusively)\s+(?:(?:use|cite|from)\s+)?(?:(?:the|an?|this|that)\s+)?"
    r"(?:(?:named|specified|requested|dedicated|cited|given|linked|specific)\s+)?articles?"
    r"|(?:the\s+)?article\b[^.!?;,]{0,120}\b(?:only|sole)\s+source"
    r"|только\s+(?:(?:по|из)\s+)?стать\w*|исключительно\s+(?:(?:по|из)\s+)?стать\w*)\b",
    re.IGNORECASE,
)
GLOSSARY = r"(?:the\s+)?(?:glossary|справочни\w*|глоссари\w*)"
GLOSSARY_ACTION = (
    rf"(?:(?:use|consult|cite|fall back to|rely on|turn to)\s+{GLOSSARY}"
    rf"|(?:answer|explain|give|summari[sz]e)\b[^.!?;]{{0,160}}?\b(?:from|using|based on)\s+{GLOSSARY}"
    rf"|(?:используй\w*|использовать|возьми\w*)\s+{GLOSSARY}"
    rf"|(?:ответь\w*|отвечай\w*|ответить|объясни\w*|объяснить|расскажи\w*)\b[^.!?;]{{0,160}}?\b(?:по|из)\s+{GLOSSARY})"
)
GLOSSARY_FORBIDDEN = re.compile(
    rf"\b(?:do\s+not|don['’]t|never|не)\s+{GLOSSARY_ACTION}\b"
    r"(?![\s,]+(?:unless|except|кроме|если\s+только)\b)",
    re.IGNORECASE,
)
MISSING_ARTICLE = (
    rf"(?:(?:there\s+(?:is|are)\s+)?no\b[^.!?;,]{{0,70}}\b{ARTICLE_LABEL}\b"
    rf"|(?:the\s+|a\s+)?(?:[^\W_]+\s+){{0,3}}{ARTICLE_LABEL}\b"
    r"[^.!?;,]{0,80}?\b(?:unavailable|missing|not\s+(?:available|found)|can(?:not|['’]t)\s+be\s+(?:found|used)"
    r"|(?:does\s+not|doesn['’]t)\s+(?:cover|address))"
    rf"|(?:you\s+)?(?:can(?:not|['’]t)|don['’]t)\s+find\b[^.!?;,]{{0,60}}\b{ARTICLE_LABEL}\b"
    r"|(?=[^.!?;,]{0,100}\bстать\w*\b)[^.!?;,]{0,90}?\b(?:нет|не\s+(?:найд[её]тся|нашл\w*|найд[её]шь|найден\w*|доступ\w*)|недоступ\w*|отсутств\w*)"
    r"[^.!?;,]{0,70}?)"
)
FALLBACK_CONDITION = re.compile(
    r"\b(?:(?P<relative>otherwise|if\s+not|if\s+it\s+is\s+(?:unavailable|missing|not\s+available)"
    r"|if\s+(?:you\s+)?can(?:not|['’]t)\s+find\s+it|иначе|в\s+противном\s+случае)"
    r"|при\s+отсутствии\s+(?:\w+\s+){0,3}статьи"
    rf"|(?:if|если)\s+{MISSING_ARTICLE}(?:[^.!?;,:—]{{0,100}}(?=[,;:—]))?)(?=\s*[,;:—]|\s)",
    re.IGNORECASE,
)
GLOSSARY_DIRECTIVE = re.compile(
    rf"^[\s,;:—-]*(?:(?:then|то)\s+)?(?:(?:please|you\s+(?:may|can)|feel\s+free\s+to|пожалуйста|можно)\s+)?"
    rf"{GLOSSARY_ACTION}\b",
    re.IGNORECASE,
)
GLOSSARY_EXCEPTION = re.compile(
    rf"\b(?:do\s+not|don['’]t|never|не)\s+{GLOSSARY_ACTION}\b[\s,]+"
    rf"(?:unless|except\s+(?:if|when)|если\s+только|кроме\s+случая,?\s+когда)\s+{MISSING_ARTICLE}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ArticleScope:
    record_ids: tuple[str, ...] = ()
    unresolved: bool = False

    def __bool__(self) -> bool:
        return bool(self.record_ids or self.unresolved)


def allows_glossary(question: str) -> bool:
    plain = QUOTES.sub(" ", question)
    if GLOSSARY_FORBIDDEN.search(plain):
        return False
    return bool(GLOSSARY_EXCEPTION.search(plain) or any(
        GLOSSARY_DIRECTIVE.match(plain[condition.end():])
        and (not condition["relative"] or ARTICLE_REQUEST.search(plain, 0, condition.start()))
        for condition in FALLBACK_CONDITION.finditer(plain)
    ))


def requires_article(question: str, *, requested_article: bool) -> bool:
    plain = QUOTES.sub(" ", question)
    return bool(ARTICLE_ONLY.search(plain) or (
        (requested_article or ARTICLE_REQUEST.search(plain)) and GLOSSARY_FORBIDDEN.search(plain)
    ))


def _title_key(text: str) -> tuple[str, bool]:
    title = " ".join(word for word in storage.normalized_words(text) if word != "oae")
    return " ".join(MINI_REVIEW.sub("", title).split()), MINI_REVIEW.search(title) is not None


def explicit_articles(question: str, page: str | None) -> ArticleScope:
    """Resolve explicit IDs, quoted names, and catalogue titles after a citation cue."""
    ids = [match[0].casefold() for match in ARTICLE_ID.finditer(question)]
    explicit_ids = set(ids)
    unresolved = False
    quoted = list(QUOTED_TITLE.finditer(question))
    cues = [match for match in TITLE_CUE.finditer(question) if not ARTICLE_TOPIC.match(question, match.start())]
    if quoted or cues:
        titles: dict[tuple[str, bool], set[str]] = {}
        for record_id, en, ru in storage.article_titles():
            for title in (en, ru):
                titles.setdefault(_title_key(title), set()).add(record_id)
        for match in quoted:
            title, mini = _title_key(match["title"][1:-1])
            matches = titles.get((title, mini or _title_key(match["kind"])[1]), set())
            matches = matches & explicit_ids or matches
            if len(matches) == 1:
                ids.extend(matches)
            else:
                unresolved = True
        for cue in cues:
            tail = question[cue.end():]
            if not tail or QUOTES.match(tail):
                continue
            words = list(storage.WORD.finditer(tail))
            # Prefer the longest catalogue title at the start of the cited phrase.
            mini_review = _title_key(cue["kind"] or "")[1]
            for word in reversed(words[:40]):
                if not TITLE_END.match(tail[word.end():]):
                    continue
                key = _title_key(tail[:word.end()])
                if mini_review:
                    key = (key[0], True)
                if matches := titles.get(key):
                    matches = matches & explicit_ids or matches
                    if len(matches) == 1:
                        ids.extend(matches)
                    else:
                        unresolved = True
                    break
            else:
                if cue["named"]:
                    unresolved = True
    if OPEN_ARTICLE.search(QUOTES.sub(" ", question)):
        if page and ARTICLE_ID.fullmatch(page):
            ids.append(page)
        else:
            unresolved = True
    return ArticleScope(tuple(dict.fromkeys(ids)), unresolved)


def proposed_article(
    question: str, reference: str, titles: list[tuple[str, str]], *, require_match: bool,
) -> ArticleScope | None:
    """Accept a model-extracted title only if it is a non-topical span in the question."""
    if GENERIC_ARTICLE.fullmatch(reference) or ARTICLE_TOPIC.search(reference):
        return None
    pattern = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, reference.split())) + r"(?!\w)", re.IGNORECASE)
    topic_starts = {match.end() for match in ARTICLE_TOPIC.finditer(question)}
    occurrences = list(pattern.finditer(question))
    if any(match.start() in topic_starts for match in occurrences):
        return None
    if not occurrences:
        return ArticleScope(unresolved=True) if require_match and not topic_starts else None
    key = _title_key(reference)
    matches = {record_id for record_id, title in titles if _title_key(title) == key}
    return ArticleScope(tuple(matches)) if len(matches) == 1 else ArticleScope(unresolved=True)
