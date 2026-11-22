"""Run the packing factorial against the checked production index."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from app.assistant import retrieval
from app.assistant.prompts import ANSWER_PROMPT, PROMPT_VERSION
from app.catalog import search, storage, vector
from app.catalog.corpus import corpus_digest
from app.catalog.index import read_manifest
from app.infra.files import write_json
from scripts.evaluation.retrieval import matches_source, percentile

ROOT = Path(__file__).parent
OUT = Path(os.getenv("RAG_AB_OUTPUT_DIR", str(ROOT)))
OLD = Path("data/eval/assistant")
CONFIGS = {
    "6x8500": (6, 8500), "6x11000": (6, 11000),
    "8x8500": (8, 8500), "8x11000": (8, 11000),
}


def questions(path):
    return json.loads(path.read_bytes())["questions"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ms_since(start):
    return round(1000 * (time.perf_counter() - start), 3)


def matches(row, option):
    quote = " ".join(option["quote"].split()).casefold()
    body = " ".join((row.get("text") or row.get("evidence_text") or "").split()).casefold()
    return matches_source(row, option["source"]) and (quote in body if quote else bool(body))


def first_rank(rows, options):
    return next((rank for rank, row in enumerate(rows, 1)
                 if any(matches(row, option) for option in options)), None)


def stage(rows, groups):
    return [first_rank(rows, options) for options in groups]


def evidence_row(key, item):
    return {
        "id": key, **item.reference(), "kind": item.kind, "title": item.title,
        "heading": item.heading, "text": item.text, "url": item.url,
        "canonical_definition": item.canonical_definition, "review_status": item.review_status,
    }


def capture_search(case, pool):
    captures = {}
    lexical_fn = storage.lexical_search_snapshot
    semantic_fn = vector.search

    def lexical(*args, **kwargs):
        value = lexical_fn(*args, **kwargs)
        captures["lexical"] = value[0]
        return value

    def semantic(*args, **kwargs):
        value = semantic_fn(*args, **kwargs)
        captures["semantic"] = value
        return value

    with (patch.object(storage, "lexical_search_snapshot", side_effect=lexical),
          patch.object(vector, "search", side_effect=semantic),
          patch.object(search, "SEMANTIC_POOL", pool)):
        started = time.perf_counter()
        hits, mode = search.search_with_mode(case["question"], case["locale"], limit=80)
        elapsed = ms_since(started)
    if mode != "hybrid" or captures.get("semantic") is None:
        raise SystemExit(f'{case["id"]}: expected hybrid, observed {mode}')
    return hits, captures, elapsed


def pack(case, hits, config):
    limit, budget = config
    if case.get("requested_mode") == "general":
        return [], 0.0
    started = time.perf_counter()
    with (patch.object(retrieval, "search_with_mode", return_value=(hits, "hybrid")),
          patch.object(retrieval, "EVIDENCE_LIMIT", limit),
          patch.object(retrieval, "CONTEXT_LIMIT", budget)):
        evidence = retrieval.evidence_for(
            case["question"], case["locale"], prefer_article=case.get("requested_mode") == "article",
            context_record_id=case.get("context_record_id"),
        )
    return [evidence_row(key, item) for key, item in evidence.items()], ms_since(started)


def packed_run(case, hits, config, groups, search_ms):
    rows, pack_ms = pack(case, hits, config)
    return {
        "evidence": rows, "ranks": stage(rows, groups), "retrieval_ms": round(search_ms + pack_ms, 3),
        "chars": sum(len(row["text"]) + len(row["title"]) + len(row["heading"]) for row in rows),
    }


def canonical_lexical(rows, locale):
    """A lexical hit denotes a section, so inspect its canonical text before packing."""
    result = []
    for row in rows:
        record_id = row["record_id"]
        record = storage.get_term(int(record_id[5:])) if record_id.startswith("term:") else storage.get_article(record_id)
        block = (record or {}).get("locales", {}).get(locale, {}).get("blocks", {}).get(row["section_id"], {})
        result.append({**row, "text": block.get("body") or row.get("excerpt", "")})
    return result


def lost_siblings(hits, semantic):
    """Additional distinct same-section chunks hidden by the production fusion key."""
    fused = {(row["record_id"], row["section_id"]): row for row in hits}
    return [row for row in semantic
            if (winner := fused.get((row["record_id"], row["section_id"]))) is not None
            and row.get("chunk_id") != winner.get("chunk_id")]


def with_siblings(hits, siblings):
    # Keep every production record/section in the same order. Route extra chunks
    # through the normal evidence packing and context budget.
    positions = {}
    for row in siblings:
        positions.setdefault((row["record_id"], row["section_id"]), []).append(row)
    expanded = []
    for row in hits:
        expanded.append(row)
        expanded.extend(positions.get((row["record_id"], row["section_id"]), []))
    return expanded


def case_run(case, suite):
    groups = [[p] for p in case.get("passages", [])]
    groups.extend(fact["sources"] for fact in case.get("facts", []) if fact["sources"])
    if case.get("requested_mode") == "general":
        hits = other_hits = []
        components = other_components = {"lexical": [], "semantic": []}
        search_ms = other_ms = 0.0
    else:
        # Alternate which pool runs first to balance the warm server/cache effects.
        pool_order = (40, 80) if sum(map(ord, case["id"])) % 2 else (80, 40)
        runs = {pool: capture_search(case, pool) for pool in pool_order}
        hits, components, search_ms = runs[40]
        other_hits, other_components, other_ms = runs[80]

    lexical = canonical_lexical(components["lexical"], case["locale"])
    siblings = lost_siblings(hits, components["semantic"])
    candidates = {
        "lexical_sections": stage(lexical, groups),
        "semantic_40": stage(components["semantic"], groups),
        "semantic_80": stage(other_components["semantic"], groups),
        "fused_chunks_40": stage(hits, groups),
        "fused_chunks_80": stage(other_hits, groups),
        "discarded_siblings": stage(siblings, groups),
        "search_ms_40": search_ms,
        "search_ms_80": other_ms,
        "semantic_hits_40": len(components["semantic"]),
        "semantic_hits_80": len(other_components["semantic"]),
        "distractors_fused": [s for s in case.get("distractors", []) if any(matches_source(row, s) for row in hits)],
    }
    configs = {name: packed_run(case, hits, values, groups, search_ms) for name, values in CONFIGS.items()}
    for name, values in CONFIGS.items():
        configs[f"{name}-pool80"] = packed_run(case, other_hits, values, groups, other_ms)
        configs[f"{name}-siblings"] = packed_run(case, with_siblings(hits, siblings), values, groups, search_ms)
    return {"id": case["id"], "suite": suite, "search_mode": "hybrid" if case.get("requested_mode") != "general" else None,
            "groups": groups, "candidates": candidates, "configs": configs}


def summarize(rows):
    summary = {}
    for suite in ("new", "oae"):
        cases = [row for row in rows if row["suite"] == suite]
        applicable = [row for row in cases if row["groups"]]
        summary[suite] = {
            "cases": len(cases), "with_expected": len(applicable),
            "search_modes": dict(Counter(row["search_mode"] for row in cases)),
            "configs": {
                name: {
                    "all": sum(all(c["configs"][name]["ranks"]) for c in applicable),
                    "found": sum(bool(rank) for c in applicable for rank in c["configs"][name]["ranks"]),
                    "total": sum(len(c["groups"]) for c in applicable),
                    "p95_retrieval_ms": percentile([c["configs"][name]["retrieval_ms"] for c in applicable], .95),
                }
                for name in cases[0]["configs"]
            },
            "candidate_hits": {key: sum(bool(rank) for c in applicable for rank in c["candidates"][key])
                               for key in ("lexical_sections", "semantic_40", "semantic_80",
                                           "fused_chunks_40", "fused_chunks_80", "discarded_siblings")},
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(storage.DATA / "index")
    assert manifest["corpus_sha256"] == corpus_digest(storage.DATA) == storage.corpus_version()
    assert vector.load() is not None
    warm, mode = search.search_with_mode("What is a star?", "en", limit=80)
    assert mode == "hybrid" and warm
    if args.legacy:
        rows = []
        for case in questions(OLD / "full_questions.json"):
            hits, search_ms = [], 0.0
            if case.get("requested_mode") != "general":
                hits, _, search_ms = capture_search(case, 40)
            rows.append({"id": case["id"], "configs": {
                name: packed_run(case, hits, values, [], search_ms) for name, values in CONFIGS.items()
            }})
        write_json(OUT / "legacy-retrieval.json", rows)
        return
    paths = {"new": ROOT / "questions.json", "oae": OLD / "oae_passage_questions.json"}
    rows = []
    for suite, path in paths.items():
        for case in questions(path):
            rows.append(case_run(case, suite))
        print(suite, len(questions(path)), flush=True)
    summary = summarize(rows)
    write_json(OUT / "retrieval.json", {
        "base_commit": subprocess.check_output(["git", "rev-parse", "85ebed8"], text=True).strip(),
        "prompt_version": PROMPT_VERSION, "prompt_sha256": hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest(),
        "index_manifest_sha256": digest(storage.DATA / "index" / "index-manifest.json"),
        "index": manifest, "suite_sha256": {key: digest(path) for key, path in paths.items()},
        "metrics": summary, "results": rows,
    })
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
