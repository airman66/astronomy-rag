"""Run the packing factorial against the checked production index."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import patch

from app.assistant import retrieval
from app.assistant.prompts import ANSWER_PROMPT, PROMPT_VERSION
from app.assistant.retrieval import Evidence, EvidenceKind, EvidenceReference
from app.catalog import search, storage, vector
from app.catalog.corpus import corpus_digest
from app.catalog.index import read_manifest
from app.infra.files import checksum, write_json
from scripts.evaluation.retrieval import matches_source, percentile

ROOT = Path(__file__).parent
OUT = Path(os.getenv("RAG_AB_OUTPUT_DIR", str(ROOT)))
OLD = Path("data/eval/assistant")
CONFIGS = {"6x8500": (6, 8500), "6x11000": (6, 11000), "8x8500": (8, 8500), "8x11000": (8, 11000)}


def packing_config(phase: str, variant: str, default: str) -> str:
    if phase == "packing":
        return variant
    return {"A-base": "6x8500", "A-wide": "8x11000"}.get(variant, default)


class PackedEvidence(EvidenceReference):
    id: str
    kind: EvidenceKind
    title: str
    heading: str
    text: str
    url: str
    canonical_definition: str | None
    review_status: str


class PackedContext(TypedDict):
    evidence: list[PackedEvidence]
    ranks: list[int | None]
    retrieval_ms: float
    chars: int


def questions_with_sha(path: Path) -> tuple[list[dict], str]:
    """Return the questions and the digest of the same file contents."""
    content = path.read_bytes()
    return json.loads(content)["questions"], hashlib.sha256(content).hexdigest()


def ms_since(start: float) -> float:
    return round(1000 * (time.perf_counter() - start), 3)


def matches(row: Mapping[str, Any], option: dict) -> bool:
    quote = " ".join(option["quote"].split()).casefold()
    text = row.get("text") or row.get("evidence_text") or ""
    body = " ".join(text.split()).casefold()
    return matches_source(row, option["source"]) and (quote in body if quote else bool(body))


def stage(rows: Sequence[Mapping[str, Any]], groups: list[list[dict]]) -> list[int | None]:
    ranks = []
    for options in groups:
        rank = next(
            (i for i, row in enumerate(rows, 1) if any(matches(row, option) for option in options)), None
        )
        ranks.append(rank)
    return ranks


def evidence_row(key: str, item: Evidence) -> PackedEvidence:
    return {
        "id": key,
        **item.reference(),
        "kind": item.kind,
        "title": item.title,
        "heading": item.heading,
        "text": item.text,
        "url": item.url,
        "canonical_definition": item.canonical_definition,
        "review_status": item.review_status,
    }


def capture_search(case: dict, pool: int) -> tuple[list[dict], dict[str, list[dict]], float]:
    captures: dict[str, list[dict]] = {}
    lexical_fn = storage.lexical_search_snapshot
    semantic_fn = vector.search

    def lexical(query: str, locale: str, limit: int) -> tuple[list[dict], str | None]:
        rows, corpus_sha = lexical_fn(query, locale, limit)
        captures["lexical"] = rows
        return rows, corpus_sha

    def semantic(
        query: str, locale: str, limit: int, *, corpus_sha: str | None, allow_index_load: bool
    ) -> list[dict] | None:
        rows = semantic_fn(query, locale, limit, corpus_sha=corpus_sha, allow_index_load=allow_index_load)
        if rows is not None:
            captures["semantic"] = rows
        return rows

    with (
        patch.object(storage, "lexical_search_snapshot", side_effect=lexical),
        patch.object(vector, "search", side_effect=semantic),
        patch.object(search, "SEMANTIC_POOL", pool),
    ):
        started = time.perf_counter()
        hits, mode = search.search_with_mode(case["question"], case["locale"], limit=80)
        elapsed = ms_since(started)
    if mode != "hybrid" or "semantic" not in captures:
        raise SystemExit(f"{case['id']}: expected hybrid, observed {mode}")
    return hits, captures, elapsed


def pack(case: dict, hits: list[dict], config: tuple[int, int]) -> tuple[list[PackedEvidence], float]:
    limit, budget = config
    if case.get("requested_mode") == "general":
        return [], 0.0
    started = time.perf_counter()
    with (
        patch.object(retrieval, "search_with_mode", return_value=(hits, "hybrid")),
        patch.object(retrieval, "EVIDENCE_LIMIT", limit),
        patch.object(retrieval, "CONTEXT_LIMIT", budget),
    ):
        evidence = retrieval.evidence_for(
            case["question"],
            case["locale"],
            prefer_article=case.get("requested_mode") == "article",
            context_record_id=case.get("context_record_id"),
        )
    return [evidence_row(key, item) for key, item in evidence.items()], ms_since(started)


def packed_run(
    case: dict, hits: list[dict], config: tuple[int, int], groups: list[list[dict]], search_ms: float
) -> PackedContext:
    rows, pack_ms = pack(case, hits, config)
    return {
        "evidence": rows,
        "ranks": stage(rows, groups),
        "retrieval_ms": round(search_ms + pack_ms, 3),
        "chars": sum(len(row["text"]) + len(row["title"]) + len(row["heading"]) for row in rows),
    }


def canonical_lexical(rows: list[dict], locale: str) -> list[dict]:
    """A lexical hit denotes a section, so inspect its canonical text before packing."""
    result = []
    for row in rows:
        record = storage.get_record(row["record_id"])
        block = (record or {}).get("locales", {}).get(locale, {}).get("blocks", {}).get(row["section_id"], {})
        result.append({**row, "text": block.get("body") or row.get("excerpt", "")})
    return result


def lost_siblings(hits: list[dict], semantic: list[dict]) -> list[dict]:
    """Additional distinct same-section chunks hidden by the production fusion key."""
    fused = {(row["record_id"], row["section_id"]): row for row in hits}
    return [
        row
        for row in semantic
        if (winner := fused.get((row["record_id"], row["section_id"]))) is not None
        and row.get("chunk_id") != winner.get("chunk_id")
    ]


def with_siblings(hits: list[dict], siblings: list[dict]) -> list[dict]:
    # Keep every production record/section in the same order. Route extra chunks
    # through the normal evidence packing and context budget.
    positions: dict[tuple[str, str], list[dict]] = {}
    for row in siblings:
        positions.setdefault((row["record_id"], row["section_id"]), []).append(row)
    expanded = []
    for row in hits:
        expanded.append(row)
        expanded.extend(positions.get((row["record_id"], row["section_id"]), []))
    return expanded


def case_run(case: dict, suite: str) -> dict:
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
        "distractors_fused": [
            s for s in case.get("distractors", []) if any(matches_source(row, s) for row in hits)
        ],
    }
    configs = {name: packed_run(case, hits, values, groups, search_ms) for name, values in CONFIGS.items()}
    expanded = with_siblings(hits, siblings)
    for name, values in CONFIGS.items():
        configs[f"{name}-pool80"] = packed_run(case, other_hits, values, groups, other_ms)
        configs[f"{name}-siblings"] = packed_run(case, expanded, values, groups, search_ms)
    return {
        "id": case["id"],
        "suite": suite,
        "search_mode": "hybrid" if case.get("requested_mode") != "general" else None,
        "groups": groups,
        "candidates": candidates,
        "configs": configs,
    }


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for suite in ("new", "oae"):
        cases = [row for row in rows if row["suite"] == suite]
        applicable = [row for row in cases if row["groups"]]
        summary[suite] = {
            "cases": len(cases),
            "with_expected": len(applicable),
            "search_modes": dict(Counter(row["search_mode"] for row in cases)),
            "configs": {
                name: {
                    "all": sum(all(c["configs"][name]["ranks"]) for c in applicable),
                    "found": sum(bool(rank) for c in applicable for rank in c["configs"][name]["ranks"]),
                    "total": sum(len(c["groups"]) for c in applicable),
                    "p95_retrieval_ms": percentile(
                        [c["configs"][name]["retrieval_ms"] for c in applicable], 0.95
                    ),
                }
                for name in cases[0]["configs"]
            },
            "candidate_hits": {
                key: sum(bool(rank) for c in applicable for rank in c["candidates"][key])
                for key in (
                    "lexical_sections",
                    "semantic_40",
                    "semantic_80",
                    "fused_chunks_40",
                    "fused_chunks_80",
                    "discarded_siblings",
                )
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(storage.DATA / "index")
    assert manifest["corpus_sha256"] == corpus_digest(storage.DATA) == storage.corpus_version()
    index_provenance = {
        "index": manifest,
        "index_manifest_sha256": checksum(storage.DATA / "index" / "index-manifest.json"),
    }
    assert vector.load() is not None
    warm, mode = search.search_with_mode("What is a star?", "en", limit=80)
    assert mode == "hybrid" and warm
    if args.legacy:
        cases, suite_sha = questions_with_sha(OLD / "full_questions.json")
        rows = []
        for case in cases:
            hits, search_ms = [], 0.0
            if case.get("requested_mode") != "general":
                hits, _, search_ms = capture_search(case, 40)
            rows.append(
                {
                    "id": case["id"],
                    "configs": {
                        name: packed_run(case, hits, values, [], search_ms)
                        for name, values in CONFIGS.items()
                    },
                }
            )
        write_json(
            OUT / "legacy-retrieval.json",
            {**index_provenance, "suite_sha256": {"legacy": suite_sha}, "results": rows},
        )
        return
    paths = {"new": ROOT / "questions.json", "oae": OLD / "oae_passage_questions.json"}
    suite_hashes = {}
    rows = []
    for suite, path in paths.items():
        cases, suite_hashes[suite] = questions_with_sha(path)
        for case in cases:
            rows.append(case_run(case, suite))
        print(suite, len(cases), flush=True)
    summary = summarize(rows)
    write_json(
        OUT / "retrieval.json",
        {
            "base_commit": subprocess.check_output(["git", "rev-parse", "85ebed8"], text=True).strip(),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest(),
            **index_provenance,
            "suite_sha256": suite_hashes,
            "metrics": summary,
            "results": rows,
        },
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
