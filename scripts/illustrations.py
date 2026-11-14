"""Team illustrations for every glossary term: manifest, resumable generation queue, import, coverage check.

Files live apart from Astro4Edu media: data/media/illustrations/<term_number>.<ext>.
Manifest: data/illustrations/manifest.json — one item per term ID with title, summary, individual
prompt, alt text, caption, file path, size, SHA-256, generator and review status.

Statuses: pending → generated (file present, automatic checks passed) → approved (human review)
or rejected (regenerate). The API serves only generated/approved items; others show an empty state.

  python -m scripts.illustrations manifest                 # build/refresh from canonical records
  python -m scripts.illustrations prompts --file F.json    # merge individual prompts written by a model
  python -m scripts.illustrations batches --threads 4      # ChatGPT-ready batches for manual runs
  python -m scripts.illustrations run --workers 1          # resumable queue via Codex CLI image generation
  python -m scripts.illustrations import --dir INBOX       # import files named <term_number>.png|jpg|webp
  python -m scripts.illustrations verify                   # coverage and integrity check (exit 1 on problems)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TERMS = DATA / "processed" / "terms"
MANIFEST = DATA / "illustrations" / "manifest.json"
FILES = DATA / "media" / "illustrations"
TASK = DATA / "illustrations" / "CHATGPT_TASK.md"
NL = chr(10)
LOCK = threading.Lock()
MIN_SIDE = 768
SCHEMA_VERSION = 1

STYLE = (
    "Единый стиль портала «Параллакс»: строгая научно-популярная иллюстрация в духе современного астрономического атласа. "
    "Глубокий тёмно-синий фон #18233A, светлые тонкие линии #FBFBF9, приглушённые акценты: зелёный #2C7A4B, синий #2A78D6, "
    "оранжевый #EB6834, бирюзовый #1BAF7A; цвета звёзд — по их температуре. Матовая плоская заливка с мягкими градиентами, "
    "без свечения ради эффекта, без неонов и фиолетовых градиентов. Горизонтальный формат 4:3. Никакого текста, букв, цифр, "
    "подписей, логотипов и водяных знаков на изображении. Не изображать фотореалистичный снимок телескопа, если это не концепт."
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + NL, encoding="utf-8", newline=NL)
    temp.replace(path)


def load_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"schema_version": SCHEMA_VERSION, "style": STYLE, "items": []}


def summary(text: str, limit: int = 320) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[: end + 1] if end > 80 else cut.rsplit(" ", 1)[0] + "…").strip()


def definition(record: dict[str, Any], locale: str) -> str:
    blocks = record["locales"][locale]["blocks"]
    for section in record["sections"]:
        if section["type"] == "definition":
            return blocks.get(section["id"], {}).get("body", "")
    return ""


def template_prompt(item: dict[str, Any]) -> str:
    """Fallback prompt when no model-written prompt exists yet: explains the term from its definition."""
    return (
        f"Illustrate the astronomy concept “{item['title_en']}” for an educational glossary. "
        f"Meaning: {item['summary_en']} "
        "Show the essential idea clearly with a composition chosen for this concept (object portrait, process sequence, "
        "geometry schematic or instrument cutaway), not a generic planet. No text or numbers in the image."
    )


def build_manifest() -> dict[str, Any]:
    manifest = load_manifest()
    existing = {int(item["term_number"]): item for item in manifest["items"]}
    items = []
    for path in sorted(TERMS.glob("*.json"), key=lambda p: int(p.stem)):
        record = json.loads(path.read_text(encoding="utf-8"))
        number = int(record["source"]["term_number"])
        base = {
            "term_id": record["id"],
            "term_number": number,
            "title_en": record["locales"]["en"]["title"],
            "title_ru": record["locales"]["ru"]["title"],
            "categories": record["metadata"].get("categories", []),
            "summary_en": summary(definition(record, "en")),
            "summary_ru": summary(definition(record, "ru")),
        }
        item = existing.get(number, {})
        merged = {
            **base,
            "kind": item.get("kind", "conceptual"),
            "composition": item.get("composition", ""),
            "prompt": item.get("prompt") or "",
            "prompt_source": item.get("prompt_source", "template"),
            "alt_ru": item.get("alt_ru", ""),
            "alt_en": item.get("alt_en", ""),
            "caption_ru": item.get("caption_ru", ""),
            "status": item.get("status", "pending"),
            "path": item.get("path"),
            "width": item.get("width"),
            "height": item.get("height"),
            "bytes": item.get("bytes"),
            "sha256": item.get("sha256"),
            "generator": item.get("generator", ""),
            "generated_at": item.get("generated_at"),
            "attempts": item.get("attempts", 0),
            "last_error": item.get("last_error"),
            "review": item.get("review", {"status": "unreviewed", "reviewer": None, "notes": None}),
        }
        if not merged["prompt"]:
            merged["prompt"] = template_prompt(merged)
            merged["prompt_source"] = "template"
        items.append(merged)
    manifest = {"schema_version": SCHEMA_VERSION, "style": STYLE, "updated_at": now(), "items": items}
    write_json(MANIFEST, manifest)
    print(f"manifest: {len(items)} terms, prompts from model: {sum(i['prompt_source'] != 'template' for i in items)}", flush=True)
    return manifest


def merge_prompts(files: list[Path]) -> None:
    """Merge model-written prompts: [{term_number, kind, composition, prompt, alt_ru, alt_en, caption_ru}]."""
    manifest = load_manifest()
    by_number = {int(item["term_number"]): item for item in manifest["items"]}
    merged = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("items", payload if isinstance(payload, list) else []):
            item = by_number.get(int(row.get("term_number", -1)))
            if not item or not isinstance(row.get("prompt"), str) or len(row["prompt"]) < 40:
                continue
            item["prompt"] = row["prompt"].strip()
            item["prompt_source"] = row.get("source", "gpt-6-astra (Codex CLI)")
            for field in ("kind", "composition", "alt_ru", "alt_en", "caption_ru"):
                if isinstance(row.get(field), str) and row[field].strip():
                    item[field] = row[field].strip()
            if item["kind"] not in ("conceptual", "diagram"):
                item["kind"] = "conceptual"
            merged += 1
    manifest["updated_at"] = now()
    write_json(MANIFEST, manifest)
    print(f"prompts merged: {merged}", flush=True)


def image_size(path: Path) -> tuple[str, int, int]:
    """Format and dimensions from the file header (PNG, JPEG, WebP) without extra libraries."""
    data = path.read_bytes()[:65536]
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24])
        return "png", width, height
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                height, width = struct.unpack(">HH", data[i + 5 : i + 9])
                return "jpeg", width, height
            length = struct.unpack(">H", data[i + 2 : i + 4])[0]
            i += 2 + length
        raise ValueError("JPEG without frame header")
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X":
            width = 1 + int.from_bytes(data[24:27], "little")
            height = 1 + int.from_bytes(data[27:30], "little")
            return "webp", width, height
        if chunk == b"VP8 ":
            width, height = struct.unpack("<HH", data[26:30])
            return "webp", width & 0x3FFF, height & 0x3FFF
        if chunk == b"VP8L":
            bits = int.from_bytes(data[21:25], "little")
            return "webp", (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    raise ValueError("unsupported image format")


def accept_file(item: dict[str, Any], source: Path, generator: str) -> None:
    kind, width, height = image_size(source)
    if min(width, height) < MIN_SIDE:
        raise ValueError(f"image too small: {width}×{height}, need at least {MIN_SIDE} px on the short side")
    ratio = width / height
    if not 0.5 <= ratio <= 2.0:
        raise ValueError(f"unexpected aspect ratio {ratio:.2f}")
    extension = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[kind]
    FILES.mkdir(parents=True, exist_ok=True)
    target = FILES / f"{item['term_number']}{extension}"
    for old in FILES.glob(f"{item['term_number']}.*"):
        if old != target:
            old.unlink()
    shutil.copyfile(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    item.update({
        "status": "generated",
        "path": target.relative_to(DATA).as_posix(),
        "width": width,
        "height": height,
        "bytes": target.stat().st_size,
        "sha256": digest,
        "generator": generator,
        "generated_at": now(),
        "last_error": None,
        "review": {"status": "unreviewed", "reviewer": None, "notes": "Автоматически проверены формат, размер и хэш; содержание не проверено человеком."},
    })
    if not item.get("alt_ru"):
        item["alt_ru"] = f"Иллюстрация к термину «{item['title_ru']}»"
    if not item.get("alt_en"):
        item["alt_en"] = f"Illustration for the term “{item['title_en']}”"


def save_item(item: dict[str, Any]) -> None:
    with LOCK:
        manifest = load_manifest()
        for index, current in enumerate(manifest["items"]):
            if int(current["term_number"]) == int(item["term_number"]):
                manifest["items"][index] = item
        manifest["updated_at"] = now()
        write_json(MANIFEST, manifest)


def import_dir(directory: Path, generator: str) -> None:
    manifest = load_manifest()
    by_number = {int(item["term_number"]): item for item in manifest["items"]}
    ok = failed = 0
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not path.stem.isdigit():
            continue
        item = by_number.get(int(path.stem))
        if not item:
            print(f"{path.name}: no such term ID", flush=True)
            failed += 1
            continue
        try:
            accept_file(item, path, generator)
            save_item(item)
            ok += 1
        except Exception as exc:
            item["last_error"] = str(exc)
            save_item(item)
            failed += 1
            print(f"{path.name}: {exc}", flush=True)
    print(f"import: accepted={ok} failed={failed}", flush=True)


def codex_command() -> list[str]:
    """Codex CLI launcher. On Windows the npm shim is codex.cmd; long multi-line prompts break in cmd.exe,
    so the JS entry point is started with node directly."""
    found = shutil.which("codex")
    if found and found.lower().endswith((".cmd", ".bat")):
        entry = Path(found).parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        node = shutil.which("node")
        if entry.exists() and node:
            return [node, str(entry)]
    return [found or "codex"]


def codex_generate(item: dict[str, Any], inbox: Path, model: str) -> Path:
    """One image through Codex CLI image generation (the operator's ChatGPT account)."""
    inbox.mkdir(parents=True, exist_ok=True)
    target = inbox / f"{item['term_number']}.png"
    if target.exists():
        target.unlink()
    instruction = (
        "Generate exactly one image with your image generation tool. Do not write code and do not create other files. "
        f"After the image is generated, copy it into the current directory as {target.name} using a single shell command. "
        "Machine safety: run shell commands one at a time.\n\n"
        f"Style: {STYLE}\n\nSubject — glossary term “{item['title_en']}” («{item['title_ru']}»).\n"
        f"Definition: {item['summary_en']}\n\nImage prompt: {item['prompt']}"
    )
    started = time.time()
    result = subprocess.run(
        [*codex_command(), "exec", "--skip-git-repo-check", "-m", model, "-c", "model_reasoning_effort=low", "-s", "workspace-write", "-C", str(inbox), "--json", instruction],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
    )
    if target.exists():
        return target
    # Fallback: the tool also stores images under ~/.codex/generated_images/<thread>/.
    thread = None
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        thread = event.get("thread_id") or (event.get("msg") or {}).get("thread_id") or thread
    roots = [Path.home() / ".codex" / "generated_images" / thread] if thread else []
    roots.append(Path.home() / ".codex" / "generated_images")
    for root in roots:
        if root.exists():
            images = [p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and p.stat().st_mtime >= started]
            if images:
                newest = max(images, key=lambda p: p.stat().st_mtime)
                shutil.copyfile(newest, target)
                return target
    raise RuntimeError(f"no image produced (codex exit {result.returncode}): {result.stderr[-300:]}")


def run_queue(workers: int, limit: int | None, model: str, max_attempts: int) -> None:
    manifest = load_manifest()
    queue = [item for item in manifest["items"] if item["status"] in ("pending", "rejected") and item.get("attempts", 0) < max_attempts]
    queue.sort(key=lambda item: item["term_number"])
    if limit:
        queue = queue[:limit]
    inbox = DATA / "illustrations" / "inbox"
    print(f"queue: {len(queue)} items, workers={workers}", flush=True)

    def work(item: dict[str, Any]) -> tuple[int, str]:
        item["attempts"] = item.get("attempts", 0) + 1
        try:
            path = codex_generate(item, inbox / str(item["term_number"]), model)
            accept_file(item, path, f"ChatGPT image generation via Codex CLI ({model})")
            save_item(item)
            return item["term_number"], "generated"
        except Exception as exc:
            item["last_error"] = str(exc)[:500]
            save_item(item)
            return item["term_number"], f"failed: {str(exc)[:160]}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(work, item) for item in queue]):
            number, status = future.result()
            print(f"term {number}: {status}", flush=True)


def export_batches(threads: int, size: int) -> None:
    manifest = load_manifest()
    pending = [item for item in manifest["items"] if item["status"] in ("pending", "rejected")]
    out = DATA / "illustrations" / "batches"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    batches = [pending[i : i + size] for i in range(0, len(pending), size)]
    for index, batch in enumerate(batches, 1):
        thread = (index - 1) % threads + 1
        lines = [
            f"# Пакет {index:02d} из {len(batches)} — поток {thread}",
            "",
            "Вставьте в ChatGPT сначала общее задание из CHATGPT_TASK.md, затем по одному пункту ниже. Каждую картинку сохраните под указанным именем в папку inbox.",
            "",
        ]
        for item in batch:
            lines += [f"## {item['term_number']}.png — {item['title_ru']} ({item['title_en']})", "", item["prompt"], ""]
        (out / f"thread-{thread}-batch-{index:02d}.md").write_text(NL.join(lines), encoding="utf-8", newline=NL)
    print(f"batches: {len(batches)} files for {threads} threads in {out}", flush=True)


def verify() -> int:
    manifest = load_manifest()
    records = {int(p.stem) for p in TERMS.glob("*.json")}
    problems: list[str] = []
    numbers = [int(item["term_number"]) for item in manifest["items"]]
    if sorted(numbers) != sorted(records):
        missing = sorted(records - set(numbers))
        extra = sorted(set(numbers) - records)
        problems.append(f"manifest/record mismatch: missing {missing[:10]}, unknown {extra[:10]}")
    if len(numbers) != len(set(numbers)):
        problems.append("duplicate term numbers in manifest")
    counts: dict[str, int] = {}
    referenced = set()
    for item in manifest["items"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
        if not item.get("prompt") or len(item["prompt"]) < 40:
            problems.append(f"term {item['term_number']}: empty prompt")
        if item["status"] in ("generated", "approved"):
            path = DATA / (item.get("path") or "")
            referenced.add(path.resolve())
            if not path.is_file():
                problems.append(f"term {item['term_number']}: file missing {item.get('path')}")
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
                problems.append(f"term {item['term_number']}: checksum mismatch")
            try:
                _, width, height = image_size(path)
                if (width, height) != (item.get("width"), item.get("height")):
                    problems.append(f"term {item['term_number']}: size differs from manifest")
            except ValueError as exc:
                problems.append(f"term {item['term_number']}: {exc}")
            if not item.get("alt_ru"):
                problems.append(f"term {item['term_number']}: missing alt_ru")
    orphans = [p.name for p in FILES.glob("*") if p.is_file() and p.resolve() not in referenced] if FILES.exists() else []
    if orphans:
        problems.append(f"files without manifest entry: {orphans[:10]}")
    print(json.dumps({"terms": len(numbers), "records": len(records), "statuses": counts, "model_prompts": sum(i["prompt_source"] != "template" for i in manifest["items"]), "problems": len(problems)}, ensure_ascii=False), flush=True)
    for problem in problems[:40]:
        print(" -", problem)
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["manifest", "prompts", "batches", "run", "import", "verify"])
    parser.add_argument("--file", type=Path, nargs="*", default=[])
    parser.add_argument("--dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1, help="parallel Codex generations; keep within the machine's safe limit")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=os.getenv("ILLUSTRATION_MODEL", "gpt-6-astra"))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--generator", default="ChatGPT (team account)")
    args = parser.parse_args()
    if args.command == "manifest":
        build_manifest()
    elif args.command == "prompts":
        merge_prompts(args.file)
    elif args.command == "batches":
        export_batches(args.threads, args.size)
    elif args.command == "run":
        run_queue(args.workers, args.limit, args.model, args.max_attempts)
    elif args.command == "import":
        import_dir(args.dir, args.generator)
    elif args.command == "verify":
        return verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
