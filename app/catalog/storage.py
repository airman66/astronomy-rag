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
import uuid
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path

from app.catalog.corpus import corpus_digest, record_files
from app.infra.config import RELEASE_DIR, data_directory

DATA = data_directory()
DB = DATA / "glossary.db" if RELEASE_DIR else Path(os.getenv("CATALOG_DB_PATH") or DATA / "glossary.db")
READ_ONLY = RELEASE_DIR is not None or os.getenv("CATALOG_READ_ONLY") == "1"
SCHEMA_VERSION = "9"
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


def normalized_words(text: str) -> list[str]:
    return WORD.findall(normalize(text))


@lru_cache(maxsize=4)
def _stemmer(locale: str):
    try:
        import snowballstemmer
    except ImportError:  # stemming improves recall but is not required for correctness
        return None
    return snowballstemmer.stemmer("russian" if locale == "ru" else "english")


def stems(text: str, locale: str) -> str:
    words = normalized_words(text)
    stemmer = _stemmer(locale)
    if stemmer is None:
        return " ".join(words)
    return " ".join(stemmer.stemWords(words))


def canonical_files(root: Path | None = None) -> list[Path]:
    root = DATA if root is None else root
    return record_files(root) + sorted((root / "illustrations").glob("manifest.json"))


def fingerprint(root: Path | None = None) -> str:
    root = DATA if root is None else root
    digest = hashlib.sha256(SCHEMA_VERSION.encode())
    for path in canonical_files(root):
        stat = path.stat()
        digest.update(f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _stored_fingerprint() -> str | None:
    try:
        with closing(sqlite3.connect(f"{DB.resolve().as_uri()}?mode=ro", uri=True)) as db:
            row = db.execute(
                "SELECT value FROM meta WHERE key='fingerprint' "
                "AND EXISTS (SELECT 1 FROM meta WHERE key='schema_version' AND value=?) "
                "AND EXISTS (SELECT 1 FROM terms)",
                (SCHEMA_VERSION,),
            ).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def connection() -> sqlite3.Connection:
    if not READ_ONLY:
        ensure_database()
    db = sqlite3.connect(f"{DB.resolve().as_uri()}?mode=ro", uri=True)
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
    if READ_ONLY:
        if _stored_fingerprint() is None:
            raise ValueError("Read-only catalogue is missing or incompatible")
        return
    key = str(DB)
    if not force_check and DB.exists() and time.monotonic() - _checked.get(key, -RECHECK_SECONDS) < RECHECK_SECONDS:
        return
    with _lock:
        if not force_check and DB.exists() and time.monotonic() - _checked.get(key, -RECHECK_SECONDS) < RECHECK_SECONDS:
            return
        stored = _stored_fingerprint()
        try:
            current = fingerprint()
            if stored != current:
                rebuild_database(current)
        except (OSError, ValueError, LookupError, TypeError, AttributeError, OverflowError, sqlite3.Error):
            if stored is None:
                raise
            log.exception("Catalog rebuild failed, serving the previous database for %s seconds", RECHECK_SECONDS)
        _checked[key] = time.monotonic()


def _illustrations(root: Path) -> dict[int, dict]:
    path = root / "illustrations" / "manifest.json"
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for item in manifest.get("items", []):
        if item.get("status") in ("generated", "approved") and item.get("path"):
            result[int(item["term_number"])] = item
    return result


def rebuild_database(current_fingerprint: str | None = None) -> None:
    build_database(DATA, DB, current_fingerprint)


def build_database(root: Path, destination: Path, current_fingerprint: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.building")
    try:
        initial = current_fingerprint or fingerprint(root)
        source_sha = corpus_digest(root)
        with closing(sqlite3.connect(temporary)) as db:
            _populate_database(db, initial, root, source_sha)
        if fingerprint(root) != initial or corpus_digest(root) != source_sha:
            raise ValueError("Catalogue inputs changed during the build")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    log.info("glossary database rebuilt")


def corpus_version() -> str | None:
    with _read() as db:
        return _corpus_version(db)


def _corpus_version(db: sqlite3.Connection) -> str | None:
    row = db.execute("SELECT value FROM meta WHERE key='corpus_sha256'").fetchone()
    return row[0] if row else None


def _populate_database(db: sqlite3.Connection, current_fingerprint: str, root: Path, source_sha: str) -> None:
    term_paths = sorted((root / "processed" / "terms").glob("*.json"))
    if not term_paths:
        raise ValueError("No canonical term records")

    illustrations = _illustrations(root)
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

    def index(record_id: str, section_id: str, locale: str, kind: str, title: str, body: str, url: str, status: str, heading: str = "") -> None:
        searchable_title = f"{title} {heading}" if heading else title
        db.execute(
            "INSERT INTO search_fts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (record_id, section_id, locale, kind, title, body, url, status, normalize(searchable_title), normalize(body), stems(searchable_title, locale), stems(body, locale)),
        )

    seen_media: set[str] = set()

    def index_media(record: dict) -> None:
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

    for path in term_paths:
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
        index_media(record)
    for path in sorted((root / "processed" / "articles").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        en, ru = record["locales"]["en"], record["locales"]["ru"]
        status = record["review"]["status"]
        db.execute("INSERT INTO articles VALUES (?,?,?,?,?,?,?)", (record["id"], en["title"], ru["title"], normalize(en["title"]), normalize(ru["title"]), status, json.dumps(record, ensure_ascii=False)))
        url = f"/api/v1/articles/{record['id']}"
        for locale, loc in (("en", en), ("ru", ru)):
            for section in record["sections"]:
                block = loc["blocks"].get(section["id"], {})
                if block.get("body"):
                    index(record["id"], section["id"], locale, "article", loc["title"], block["body"], url, status, block.get("heading", ""))
        index_media(record)
    db.execute("INSERT INTO meta VALUES ('fingerprint', ?)", (current_fingerprint,))
    db.execute("INSERT INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
    db.execute("INSERT INTO meta VALUES ('corpus_sha256', ?)", (source_sha,))
    db.commit()


def illustration_view(item: dict) -> dict:
    source = item.get("source") or {}
    number = int(item["term_number"])
    url = f"/api/v1/terms/{number}/illustration"
    if digest := item.get("sha256"):
        url = f"{url}?v={digest}"
    return {
        "term_number": number,
        "url": url,
        "width": item.get("width"),
        "height": item.get("height"),
        "alt_ru": item.get("alt_ru", ""),
        "alt_en": item.get("alt_en", ""),
        "caption_ru": item.get("caption_ru", ""),
        "kind": item.get("kind", "conceptual"),
        "status": item.get("status"),
        "generator": item.get("generator", ""),
        "source_type": item.get("source_type", "generated"),
        "file_page": source.get("file_page"),
        "article_url": source.get("article_url"),
        "author": source.get("author"),
        "license": source.get("license"),
        "license_url": source.get("license_url"),
        "modified": source.get("modified"),
    }


def _get_payload(query: str, identifier: str | int) -> dict | None:
    with _read() as db:
        row = db.execute(query, (identifier,)).fetchone()
    return json.loads(row[0]) if row else None


def get_term(number: int) -> dict | None:
    return _get_payload("SELECT payload FROM terms WHERE number=?", number)


def get_illustration(number: int) -> dict | None:
    return _get_payload("SELECT payload FROM illustrations WHERE term_number=?", number)


def get_article(identifier: str) -> dict | None:
    return _get_payload("SELECT payload FROM articles WHERE id=?", identifier)


def get_media(identifier: str) -> dict | None:
    return _get_payload("SELECT payload FROM media WHERE id=?", identifier)


def _like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def list_records(kind: str, *, page: int, page_size: int, locale: str, query: str = "", category: str = "", letter: str = "", status: str = "") -> tuple[list[dict], int]:
    is_term = kind == "term"
    table = "terms" if is_term else "articles"
    params = {
        "is_term": is_term,
        "ru": locale == "ru",
        "needle": f"%{_like(normalize(query))}%" if query else "",
        "category": category if is_term else "",
        "letter": f"{_like(normalize(letter)[:1])}%" if letter else "",
        "status": status,
        "limit": page_size,
        "offset": min((page - 1) * page_size, 2**63 - 1),
    }
    with _read() as db:
        rows = db.execute(
            r"""WITH records AS (
                SELECT number AS identifier, title_en, title_ru, title_en_norm, title_ru_norm,
                       aliases_norm, categories, review_status, payload FROM terms WHERE :is_term
                UNION ALL
                SELECT id, title_en, title_ru, title_en_norm, title_ru_norm,
                       '', '[]', review_status, payload FROM articles WHERE NOT :is_term
            ), filtered AS (
                SELECT *, CASE WHEN :ru THEN title_ru_norm ELSE title_en_norm END AS sort_title
                FROM records
                WHERE (:needle = '' OR title_en_norm LIKE :needle ESCAPE '\'
                    OR title_ru_norm LIKE :needle ESCAPE '\' OR aliases_norm LIKE :needle ESCAPE '\')
                  AND (:category = '' OR EXISTS (SELECT 1 FROM json_each(records.categories) WHERE value=:category))
                  AND (:letter = '' OR sort_title LIKE :letter ESCAPE '\')
                  AND (:status = '' OR review_status=:status)
            ), page AS (
                SELECT * FROM filtered ORDER BY sort_title, title_en_norm LIMIT :limit OFFSET :offset
            )
            SELECT total, page.* FROM (SELECT COUNT(*) AS total FROM filtered)
            LEFT JOIN page ON 1 ORDER BY sort_title, title_en_norm""",
            params,
        ).fetchall()
    total = rows[0]["total"]
    if rows[0]["identifier"] is None:
        return [], total
    result = []
    for row in rows:
        payload = json.loads(row["payload"])
        title = row["title_ru" if locale == "ru" else "title_en"] or row["title_en"]
        illustration = payload.get("illustration")
        result.append({
            "id": payload["id"],
            "title": title,
            "title_en": row["title_en"],
            "url": f"/api/v1/{table}/{row['identifier']}",
            "review_status": row["review_status"],
            "categories": payload.get("metadata", {}).get("categories", []),
            "categories_ru": payload.get("metadata", {}).get("categories_ru", []),
            "illustration_url": illustration["url"] if illustration else None,
        })
    return result, total


def letters(locale: str) -> list[dict]:
    """First letters of term titles in the requested language, for an alphabet index."""
    with _read() as db:
        rows = db.execute(
            "SELECT substr(CASE WHEN ? THEN title_ru_norm ELSE title_en_norm END, 1, 1) AS letter, "
            "COUNT(*) AS count FROM terms GROUP BY letter ORDER BY letter",
            (locale == "ru",),
        ).fetchall()
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
    words = normalized_words(query)[:12]
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


def lexical_search(query: str, locale: str, limit: int, *, record_id: str | None = None) -> list[dict]:
    expression = _expression(query, locale)
    if not expression:
        return []
    with _read() as db:
        return _lexical_matches(db, expression, locale, limit, record_id)


def lexical_search_snapshot(query: str, locale: str, limit: int) -> tuple[list[dict], str | None]:
    """Read the lexical hits and their corpus version in one SQLite snapshot."""
    expression = _expression(query, locale)
    with _read() as db:
        db.execute("BEGIN")
        version = _corpus_version(db)
        matches = _lexical_matches(db, expression, locale, limit, None) if expression else []
    return matches, version


def _lexical_matches(
    db: sqlite3.Connection, expression: str, locale: str, limit: int, record_id: str | None,
) -> list[dict]:
    rows = db.execute(
        "SELECT record_id,section_id,locale,kind,title,body,url,review_status,"
        "bm25(search_fts, 0,0,0,0,0,0,0,0, 8,1,6,0.8) AS score FROM search_fts "
        "WHERE search_fts MATCH ? AND locale=? AND (? IS NULL OR record_id=?) ORDER BY score LIMIT ?",
        (expression, locale, record_id, record_id, limit),
    ).fetchall()
    return [{"record_id": row["record_id"], "section_id": row["section_id"], "kind": row["kind"], "title": row["title"], "excerpt": row["body"][:400], "url": row["url"], "review_status": row["review_status"], "score": -row["score"]} for row in rows]
