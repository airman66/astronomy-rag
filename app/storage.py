"""Rebuildable SQLite catalogue and bilingual FTS5 index."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path

DATA = Path(os.getenv("DATA_DIR", "data"))
DB = DATA / "glossary.db"
SCHEMA_VERSION = "3"
# Semantic neighbours below this cosine similarity are noise for a glossary of short texts.
SEMANTIC_POOL = int(os.getenv("SEMANTIC_POOL", "40"))
RECHECK_SECONDS = 30.0
_lock = threading.Lock()
_checked: dict[str, float] = {}
log = logging.getLogger("glossary.storage")

WORD = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = {
    "ru": set("а без бы в во вот все всё где да для до если есть еще ещё же за и из или их к как ка ко когда кто ли либо мне мой на над не нет ни но о об один от по под при про с со так такое такой там то тот у уже чем что чтобы это эта этот я как почему зачем какой какая какие расскажи объясни покажи".split()),
    "en": set("a an and are as at be by can do does for from how i in is it its me of on or the this that to was what when where which who why with tell explain show".split()),
}


def normalize(text: str) -> str:
    """Case- and ё-insensitive form used for matching; displayed text stays unchanged."""
    return (text or "").casefold().replace("ё", "е")


@lru_cache(maxsize=4)
def _stemmer(locale: str):
    try:
        import snowballstemmer
    except ImportError:  # stemming improves recall but is not required for correctness
        return None
    return snowballstemmer.stemmer("russian" if locale == "ru" else "english")


def stems(text: str, locale: str) -> str:
    words = WORD.findall(normalize(text))
    stemmer = _stemmer(locale)
    if stemmer is None:
        return " ".join(words)
    return " ".join(stemmer.stemWords(words))


def canonical_files() -> list[Path]:
    return sorted((DATA / "processed" / "terms").glob("*.json")) + sorted((DATA / "processed" / "articles").glob("*.json")) + sorted((DATA / "illustrations").glob("manifest.json"))


def fingerprint() -> str:
    digest = hashlib.sha256(SCHEMA_VERSION.encode())
    for path in canonical_files():
        stat = path.stat()
        digest.update(f"{path.relative_to(DATA).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _stored_fingerprint() -> str | None:
    try:
        with closing(sqlite3.connect(DB)) as db:
            row = db.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def connection() -> sqlite3.Connection:
    ensure_database()
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    return db


@contextmanager
def _read():
    """A connection that is always closed (sqlite3's own context manager only commits)."""
    db = connection()
    try:
        yield db
    finally:
        db.close()


def ensure_database(force_check: bool = False) -> None:
    """Rebuild the DB when it is missing or the canonical JSON set changed (checked every 30 s)."""
    key = str(DB)
    if not force_check and DB.exists() and time.monotonic() - _checked.get(key, -RECHECK_SECONDS) < RECHECK_SECONDS:
        return
    with _lock:
        if not force_check and DB.exists() and time.monotonic() - _checked.get(key, -RECHECK_SECONDS) < RECHECK_SECONDS:
            return
        current = fingerprint()
        if not DB.exists() or _stored_fingerprint() != current:
            rebuild_database(current)
        _checked[key] = time.monotonic()


def _illustrations() -> dict[int, dict]:
    path = DATA / "illustrations" / "manifest.json"
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for item in manifest.get("items", []):
        if item.get("status") in ("generated", "approved") and item.get("path"):
            result[int(item["term_number"])] = item
    return result


def rebuild_database(current_fingerprint: str | None = None) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    temporary = DB.with_name(f"{DB.name}.{os.getpid()}.{threading.get_ident()}.building")
    if temporary.exists():
        temporary.unlink()
    db = sqlite3.connect(temporary)
    illustrations = _illustrations()
    try:
        db.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE terms (number INTEGER PRIMARY KEY, title_en TEXT NOT NULL,
                title_ru TEXT NOT NULL, title_en_norm TEXT NOT NULL, title_ru_norm TEXT NOT NULL,
                aliases_norm TEXT NOT NULL, categories TEXT NOT NULL, review_status TEXT NOT NULL,
                payload TEXT NOT NULL);
            CREATE TABLE articles (id TEXT PRIMARY KEY, title_en TEXT NOT NULL,
                title_ru TEXT NOT NULL, title_en_norm TEXT NOT NULL, title_ru_norm TEXT NOT NULL,
                review_status TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE media (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE illustrations (term_number INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE VIRTUAL TABLE search_fts USING fts5(
                record_id UNINDEXED, section_id UNINDEXED, locale UNINDEXED,
                kind UNINDEXED, title UNINDEXED, body UNINDEXED, url UNINDEXED, review_status UNINDEXED,
                title_norm, body_norm, title_stem, body_stem,
                tokenize='unicode61 remove_diacritics 2');
            """
        )

        def index(record_id: str, section_id: str, locale: str, kind: str, title: str, body: str, url: str, status: str) -> None:
            db.execute(
                "INSERT INTO search_fts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (record_id, section_id, locale, kind, title, body, url, status, normalize(title), normalize(body), stems(title, locale), stems(body, locale)),
            )

        seen_media: set[str] = set()
        for path in sorted((DATA / "processed" / "terms").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            number = int(record["source"]["term_number"])
            en, ru = record["locales"]["en"], record["locales"]["ru"]
            status = record["review"]["status"]
            if number in illustrations:
                record["illustration"] = illustration_view(illustrations[number])
                db.execute("INSERT INTO illustrations VALUES (?,?)", (number, json.dumps(illustrations[number], ensure_ascii=False)))
            aliases = " | ".join(en.get("aliases", []) + ru.get("aliases", []))
            db.execute(
                "INSERT INTO terms VALUES (?,?,?,?,?,?,?,?,?)",
                (number, en["title"], ru["title"], normalize(en["title"]), normalize(ru["title"]), normalize(aliases), json.dumps(record["metadata"].get("categories", [])), status, json.dumps(record, ensure_ascii=False)),
            )
            url = f"/api/v1/terms/{number}"
            for locale, loc in (("en", en), ("ru", ru)):
                index(record["id"], "title", locale, "term", loc["title"], ", ".join(loc.get("aliases", [])), url, status)
                for section in record["sections"]:
                    block = loc.get("blocks", {}).get(section["id"], {})
                    if block.get("body"):
                        index(record["id"], section["id"], locale, "term", loc["title"], block["body"], url, status)
            for item in record.get("media", []):
                if item["id"] in seen_media:
                    continue
                seen_media.add(item["id"])
                db.execute("INSERT INTO media VALUES (?,?)", (item["id"], json.dumps(item, ensure_ascii=False)))
                for locale in ("en", "ru"):
                    title = item.get(f"title_{locale}") or ""
                    body = " ".join(filter(None, [item.get(f"short_caption_{locale}"), item.get(f"caption_{locale}"), item.get(f"alt_{locale}")]))
                    if title or body:
                        index(record["id"], item["id"], locale, item["kind"], title, body, f"/api/v1/media/{item['id']}", item["review"]["status"])
        for path in sorted((DATA / "processed" / "articles").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            en, ru = record["locales"]["en"], record["locales"]["ru"]
            status = record["review"]["status"]
            db.execute("INSERT INTO articles VALUES (?,?,?,?,?,?,?)", (record["id"], en["title"], ru["title"], normalize(en["title"]), normalize(ru["title"]), status, json.dumps(record, ensure_ascii=False)))
            url = f"/api/v1/articles/{record['id']}"
            for locale, loc in (("en", en), ("ru", ru)):
                for section in record["sections"]:
                    block = loc["blocks"].get(section["id"], {})
                    index(record["id"], section["id"], locale, "article", loc["title"], block.get("body", ""), url, status)
        db.execute("INSERT INTO meta VALUES ('fingerprint', ?)", (current_fingerprint or fingerprint(),))
        db.execute("INSERT INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
        db.commit()
    finally:
        db.close()
    temporary.replace(DB)
    log.info("glossary database rebuilt")


def illustration_view(item: dict) -> dict:
    return {
        "term_number": int(item["term_number"]),
        "url": f"/api/v1/terms/{int(item['term_number'])}/illustration",
        "width": item.get("width"),
        "height": item.get("height"),
        "alt_ru": item.get("alt_ru", ""),
        "alt_en": item.get("alt_en", ""),
        "caption_ru": item.get("caption_ru", ""),
        "kind": item.get("kind", "conceptual"),
        "status": item.get("status"),
        "generator": item.get("generator", ""),
    }


def get_term(number: int) -> dict | None:
    with _read() as db:
        row = db.execute("SELECT payload FROM terms WHERE number=?", (number,)).fetchone()
    return json.loads(row[0]) if row else None


def get_illustration(number: int) -> dict | None:
    with _read() as db:
        row = db.execute("SELECT payload FROM illustrations WHERE term_number=?", (number,)).fetchone()
    return json.loads(row[0]) if row else None


def get_article(identifier: str) -> dict | None:
    with _read() as db:
        row = db.execute("SELECT payload FROM articles WHERE id=?", (identifier,)).fetchone()
    return json.loads(row[0]) if row else None


def get_media(identifier: str) -> dict | None:
    with _read() as db:
        row = db.execute("SELECT payload FROM media WHERE id=?", (identifier,)).fetchone()
    return json.loads(row[0]) if row else None


def _like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def list_records(kind: str, *, page: int, page_size: int, locale: str, query: str = "", category: str = "", letter: str = "", status: str = "") -> tuple[list[dict], int]:
    table = "terms" if kind == "term" else "articles"
    clauses, args = [], []
    if query:
        needle = f"%{_like(normalize(query))}%"
        searchable = "(title_en_norm LIKE ? ESCAPE '\\' OR title_ru_norm LIKE ? ESCAPE '\\'" + (" OR aliases_norm LIKE ? ESCAPE '\\')" if table == "terms" else ")")
        clauses.append(searchable)
        args.extend([needle] * (3 if table == "terms" else 2))
    if category and table == "terms":
        clauses.append("categories LIKE ?")
        args.append(f'%"{category}"%')
    if letter:
        clauses.append(f"title_{locale}_norm LIKE ? ESCAPE '\\'")
        args.append(f"{_like(normalize(letter)[:1])}%")
    if status:
        clauses.append("review_status=?")
        args.append(status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    order = f"title_{locale}_norm, title_en_norm"
    with _read() as db:
        total = db.execute(f"SELECT COUNT(*) FROM {table}{where}", args).fetchone()[0]
        rows = db.execute(f"SELECT * FROM {table}{where} ORDER BY {order} LIMIT ? OFFSET ?", [*args, page_size, (page - 1) * page_size]).fetchall()
    result = []
    for row in rows:
        payload = json.loads(row["payload"])
        title = row[f"title_{locale}"] or row["title_en"]
        illustration = payload.get("illustration")
        result.append({
            "id": payload["id"],
            "title": title,
            "title_en": row["title_en"],
            "url": f"/api/v1/{table}/{row['number'] if table == 'terms' else row['id']}",
            "review_status": row["review_status"],
            "categories": payload.get("metadata", {}).get("categories", []),
            "categories_ru": payload.get("metadata", {}).get("categories_ru", []),
            "illustration_url": illustration["url"] if illustration else None,
        })
    return result, total


def letters(locale: str) -> list[dict]:
    """First letters of term titles in the requested language, for an alphabet index."""
    with _read() as db:
        rows = db.execute(f"SELECT substr(title_{locale}_norm, 1, 1) AS letter, COUNT(*) AS count FROM terms GROUP BY letter ORDER BY letter").fetchall()
    return [{"letter": row["letter"].upper(), "count": row["count"]} for row in rows if row["letter"] and row["letter"].isalpha()]


def categories(locale: str) -> list[dict]:
    counts: dict[str, dict] = {}
    with _read() as db:
        for (payload,) in db.execute("SELECT payload FROM terms"):
            metadata = json.loads(payload).get("metadata", {})
            names_ru = metadata.get("categories_ru", [])
            for index, name in enumerate(metadata.get("categories", [])):
                entry = counts.setdefault(name, {"id": name, "title": name, "count": 0})
                if locale == "ru" and index < len(names_ru) and names_ru[index]:
                    entry["title"] = names_ru[index]
                entry["count"] += 1
    return sorted(counts.values(), key=lambda item: (-item["count"], normalize(item["title"])))


def _expression(query: str, locale: str) -> str | None:
    words = WORD.findall(normalize(query))[:12]
    meaningful = [word for word in words if word not in STOPWORDS[locale]] or words
    if not meaningful:
        return None
    stemmer = _stemmer(locale)
    parts = []
    for word in meaningful:
        alternatives = [f'{{title_norm body_norm}} : "{word}"*']
        stem = stemmer.stemWord(word) if stemmer else word
        if len(stem) >= 3:
            alternatives.append(f'{{title_stem body_stem}} : "{stem}"' + ("*" if len(stem) >= 4 else ""))
        parts.append("(" + " OR ".join(alternatives) + ")")
    return " OR ".join(parts)


def lexical_search(query: str, locale: str, limit: int) -> list[dict]:
    expression = _expression(query, locale)
    if not expression:
        return []
    with _read() as db:
        rows = db.execute(
            "SELECT record_id,section_id,locale,kind,title,body,url,review_status,bm25(search_fts, 0,0,0,0,0,0,0,0, 8,1,6,0.8) AS score FROM search_fts WHERE search_fts MATCH ? AND locale=? ORDER BY score LIMIT ?",
            (expression, locale, limit),
        ).fetchall()
    return [{"record_id": row["record_id"], "section_id": row["section_id"], "kind": row["kind"], "title": row["title"], "excerpt": row["body"][:400], "url": row["url"], "review_status": row["review_status"], "score": -row["score"]} for row in rows]


_last_semantic_error = [0.0]


def search_with_mode(query: str, locale: str, limit: int = 20) -> tuple[list[dict], str]:
    """Hybrid FTS + vector search; falls back to FTS alone when embeddings are unavailable."""
    lexical = lexical_search(query, locale, limit)
    from app import vector

    try:
        semantic = vector.search(query, locale, min(limit, SEMANTIC_POOL))
    except Exception as exc:  # embedding server/model/index problems must not break search
        if time.monotonic() - _last_semantic_error[0] > 60:
            log.warning("semantic search unavailable, using lexical results: %s: %s", type(exc).__name__, exc)
            _last_semantic_error[0] = time.monotonic()
        return lexical, "lexical"
    if not vector.available():
        return lexical, "lexical"
    fused: dict[tuple[str, str], dict] = {}
    for ranked in (lexical, semantic):
        for rank, item in enumerate(ranked):
            key = item["record_id"], item["section_id"]
            if key not in fused:
                fused[key] = {**item, "score": 0.0}
            fused[key]["score"] += 1.0 / (60 + rank + 1)
    return sorted(fused.values(), key=lambda row: row["score"], reverse=True)[:limit], "hybrid"


def search(query: str, locale: str, limit: int = 20) -> list[dict]:
    return search_with_mode(query, locale, limit)[0]


if __name__ == "__main__":
    rebuild_database()
