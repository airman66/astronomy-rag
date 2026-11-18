"""Export the cover coverage tables and visual review sheet."""

import csv
import html
import json

from scripts.illustration_files import DATA, MANIFEST


def write_review() -> None:
    manifest = json.loads(MANIFEST.read_text())
    items = sorted(manifest["items"], key=lambda item: int(item["term_number"]))
    directory = DATA / "illustrations"
    with (directory / "coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "term_number",
                "title_ru",
                "status",
                "source_type",
                "file_page",
                "author",
                "license",
                "license_url",
                "review_status",
            )
        )
        for item in items:
            source = item.get("source") or {}
            writer.writerow(
                (
                    item["term_number"],
                    item["title_ru"],
                    item["status"],
                    item.get("source_type", ""),
                    source.get("file_page", ""),
                    source.get("author", ""),
                    source.get("license", ""),
                    source.get("license_url", ""),
                    item.get("review", {}).get("status", ""),
                )
            )
    cards = []
    for item in items:
        source = item.get("source") or {}
        if item.get("path") and item["status"] in ("generated", "approved"):
            picture = f'<img src="../{html.escape(item["path"], quote=True)}" alt="" loading="lazy">'
        else:
            picture = '<div class="empty">Нет обложки</div>'
        file_page = source.get("file_page") or ""
        link = (
            f'<a href="{html.escape(file_page, quote=True)}" rel="noopener">Источник</a>' if file_page else ""
        )
        cards.append(
            f'<article><div class="image">{picture}</div><h2>{item["term_number"]}. {html.escape(item["title_ru"])}</h2>'
            f"<p>{html.escape(item.get('source_type') or item['status'])} · {html.escape(source.get('license') or '')}</p>{link}</article>"
        )
    page = (
        """<!doctype html><html lang="ru"><meta charset="utf-8"><title>Обложки глоссария — 501 ID</title>
<style>body{font:15px/1.4 system-ui,sans-serif;margin:24px;background:#f7f7f4;color:#20221f}h1{font-size:28px}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}article{background:white;padding:10px;border:1px solid #ddd}
.image{height:160px;display:grid;place-items:center;background:#e9e9e5}.image img{max-width:100%;max-height:160px;object-fit:contain}
h2{font-size:15px;margin:8px 0 3px}p{color:#555;margin:0 0 6px}a{color:#174b90}.empty{color:#777}</style>
<h1>Обложки глоссария — 501 ID</h1><main>"""
        + "\n".join(cards)
        + "</main></html>\n"
    )
    (directory / "review.html").write_text(page, encoding="utf-8")
    print(
        json.dumps(
            {
                "items": len(items),
                "ready": sum(i["status"] in ("generated", "approved") for i in items),
            },
            ensure_ascii=False,
        )
    )
