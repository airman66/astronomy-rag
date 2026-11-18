"""Protect editorial edits and apply independently verified corrections."""

import json
import re
from collections import defaultdict
from typing import Any

from app.fileio import write_json as write_json_atomic
from app.translate import protected_tokens
from app.translation_fields import TERM_LINK, get_path, set_path
from scripts.translation_review.common import (
    APPLIED,
    DATA,
    EDITORIAL,
    REVIEW,
    REVIEW_VERSIONS,
    TERMS,
    VERIFY,
    VERIFY_VERSIONS,
    digest,
    latest,
    now,
    read_jsonl,
    write_json,
)

HEADING_PREFIX = "Термин глоссария: "
URL = re.compile(r"https?://\S+")


def sync_editorial(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Propagate manual edits made only in sections[].*_ru to locales.ru.blocks.

    Earlier QA passes always updated both copies to the same value, so a section text that
    differs from its block copy is a later manual edit. The Russian title follows an edited
    "Термин глоссария: …" heading, so related-term links can be synchronised afterwards.
    """
    changes: list[dict[str, Any]] = []
    for record_id, record in records.items():
        if record.get("review", {}).get("status") == "reviewed":
            continue
        ru = record["locales"]["ru"]
        touched = False
        for index, section in enumerate(record["sections"]):
            block = ru["blocks"].get(section["id"])
            if block is None:
                continue
            for field in ("heading", "body"):
                edited, copy = section.get(f"{field}_ru"), block.get(field)
                if section.get(field) and edited and copy and edited != copy:
                    changes.append(
                        {
                            "record_id": record_id,
                            "path": f"/locales/ru/blocks/{section['id']}/{field}",
                            "old": copy,
                            "new": edited,
                            "evidence": f"/sections/{index}/{field}_ru",
                        }
                    )
                    block[field] = edited
                    touched = True
                    if (
                        section["type"] == "definition"
                        and field == "heading"
                        and edited.startswith(HEADING_PREFIX)
                    ):
                        term = edited[len(HEADING_PREFIX) :].strip()
                        title = term[:1].upper() + term[1:]
                        if title.casefold() != ru["title"].casefold():
                            changes.append(
                                {
                                    "record_id": record_id,
                                    "path": "/locales/ru/title",
                                    "old": ru["title"],
                                    "new": title,
                                    "evidence": f"/sections/{index}/heading_ru",
                                }
                            )
                            ru["title"] = title
        if touched:
            path = TERMS / f"{record['source']['term_number']}.json"
            write_json_atomic(path, record)
    if changes:
        history = json.loads(EDITORIAL.read_text(encoding="utf-8")) if EDITORIAL.exists() else {"changes": []}
        history["changes"].extend({**change, "synced_at": now()} for change in changes)
        write_json(EDITORIAL, history)
    print(f"sync-editorial: fields={len(changes)}", flush=True)
    return changes


def previous_edits() -> tuple[set[str], set[str], set[str]]:
    """Earlier edits: manual texts (never auto-replaced) and what earlier QA fixes replaced.

    Earlier automatic QA corrections may be refined further, but a proposal must not restore
    the text they replaced: such a revert is identified by digest(en, proposal) of a valid
    resolution or by the "old" side of a numeric patch.
    """
    manual: set[str] = set()
    if EDITORIAL.exists():
        manual.update(
            change["new"].strip() for change in json.loads(EDITORIAL.read_text(encoding="utf-8"))["changes"]
        )
    replaced_digests: set[str] = set()
    replaced_texts: set[str] = set()
    eval_dir = DATA / "eval"
    resolutions = eval_dir / "translation_resolutions.json"
    if resolutions.exists():
        for identifier, row in json.loads(resolutions.read_text(encoding="utf-8")).get("results", {}).items():
            if row.get("valid") and row.get("corrected_ru"):
                replaced_digests.add(identifier)
    patches = eval_dir / "numeric_patches.json"
    if patches.exists():
        for row in json.loads(patches.read_text(encoding="utf-8")):
            if row.get("valid") and row.get("old") and row.get("old") != row.get("new"):
                replaced_texts.add(row["old"].strip())
    return manual, replaced_digests, replaced_texts


def guard(unit: dict[str, Any], proposed: str) -> str | None:
    """Deterministic rejection reasons for an otherwise accepted proposal."""
    source, current = unit["en"], unit["ru"]
    source_tokens, current_tokens, proposed_tokens = (
        protected_tokens(source),
        protected_tokens(current),
        protected_tokens(proposed),
    )
    if proposed_tokens != source_tokens and current_tokens == source_tokens:
        return "numbers_or_units_changed"
    if len(current) > 40 and not 0.5 <= len(proposed) / max(1, len(current)) <= 2.0:
        return "length_changed_too_much"
    if current.count("\n") != proposed.count("\n") and len(current) > 120:
        return "line_structure_changed"
    if set(URL.findall(current)) - set(URL.findall(proposed)):
        return "url_lost"
    if not proposed.strip():
        return "empty"
    return None


def run_apply(units: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reviews = latest(read_jsonl(REVIEW), REVIEW_VERSIONS)
    verifications = latest(read_jsonl(VERIFY), VERIFY_VERSIONS)
    manual, replaced_digests, replaced_texts = previous_edits()
    history = json.loads(APPLIED.read_text(encoding="utf-8")) if APPLIED.exists() else {"runs": []}
    prior_versions = defaultdict(set)
    for run in history["runs"]:
        for edit in run["applied"]:
            prior_versions[(edit["record_id"], edit["path"])].add(edit["old"].strip())
    applied: list[dict[str, Any]] = []
    disputed: list[dict[str, Any]] = []
    changed: set[str] = set()
    for unit_id, review in reviews.items():
        unit = units.get(unit_id)
        if not unit or review["verdict"] != "fix":
            continue
        decision = verifications.get(unit_id, {}).get("decision")
        proposal = review["corrected_ru"]
        reason = None
        if decision != "accept":
            reason = f"verifier_{decision or 'missing'}"
        elif unit["ru"].strip() in manual:
            reason = "manual_editorial_edit"
        elif digest(unit["en"], proposal) in replaced_digests or proposal.strip() in replaced_texts:
            reason = "reverts_earlier_fix"
        else:
            reason = guard(unit, proposal)
        for record_id, ru_path in unit["occurrences"]:
            record = records[record_id]
            entry = {
                "record_id": record_id,
                "path": ru_path,
                "unit": unit_id,
                "old": unit["ru"],
                "new": proposal,
            }
            if reason:
                disputed.append(
                    {
                        **entry,
                        "reason": reason,
                        "verifier_reason": verifications.get(unit_id, {}).get("reason", ""),
                    }
                )
                continue
            if record.get("review", {}).get("status") == "reviewed":
                disputed.append({**entry, "reason": "record_already_reviewed"})
                continue
            if ru_path.startswith("/media/"):
                media = record["media"][int(ru_path.split("/")[2])]
                if media.get("review", {}).get("status") == "reviewed":
                    disputed.append({**entry, "reason": "media_already_reviewed"})
                    continue
            current = get_path(record, ru_path)
            if current == proposal:
                continue  # applied in an earlier run
            if current != unit["ru"]:
                disputed.append({**entry, "reason": "changed_since_review"})
                continue
            if proposal.strip() in prior_versions.get((record_id, ru_path), ()):
                disputed.append({**entry, "reason": "reverts_earlier_fix"})
                continue
            set_path(record, ru_path, proposal)
            applied.append(
                {
                    **entry,
                    "source": "review",
                    "types": [issue["type"] for issue in review["issues"]],
                    "verification": "accepted_by_verifier",
                }
            )
            changed.add(record_id)
    # Related-term links must name their target with its final Russian title.
    by_number = {record["source"]["term_number"]: record for record in records.values()}
    for record_id, record in records.items():
        if record.get("review", {}).get("status") == "reviewed":
            continue
        for index, section in enumerate(record["sections"]):
            for link_index, link in enumerate(section.get("links", [])):
                match = TERM_LINK.search(link.get("url", ""))
                target = by_number.get(int(match.group(1))) if match else None
                if not target or not link.get("text"):
                    continue
                if link["text"].strip().casefold() != target["locales"]["en"]["title"].strip().casefold():
                    continue
                title_ru = target["locales"]["ru"]["title"]
                current = link.get("text_ru") or ""
                if current != title_ru and current.strip() not in manual and title_ru:
                    path = f"/sections/{index}/links/{link_index}/text_ru"
                    if title_ru.strip() in prior_versions.get((record_id, path), ()):
                        continue
                    applied.append(
                        {
                            "record_id": record_id,
                            "path": path,
                            "unit": None,
                            "old": current,
                            "new": title_ru,
                            "source": "link_sync",
                            "types": ["link"],
                        }
                    )
                    link["text_ru"] = title_ru
                    changed.add(record_id)
    for record_id in sorted(changed, key=lambda rid: int(rid.split(":")[1])):
        record = records[record_id]
        path = TERMS / f"{record['source']['term_number']}.json"
        write_json_atomic(path, record)
    result = {"applied_at": now(), "applied": applied, "disputed": disputed, "touched_records": len(changed)}
    history["runs"].append(result)
    write_json(APPLIED, history)
    print(f"apply: fields={len(applied)} disputed={len(disputed)} records={len(changed)}", flush=True)
    return result
