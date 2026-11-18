"""Build the field-level audit report and Markdown summary."""

import json
from collections import Counter, defaultdict
from typing import Any

from app.translate import protected_tokens
from app.translation_fields import fields
from scripts.translation_review.common import (
    APPLIED,
    REPORT,
    REVIEW,
    REVIEW_MODEL,
    REVIEW_PROMPT_VERSION,
    REVIEW_VERSIONS,
    SUMMARY,
    VERIFY,
    VERIFY_MODEL,
    VERIFY_PROMPT_VERSION,
    VERIFY_VERSIONS,
    latest,
    now,
    read_jsonl,
    write_json,
)


def run_report(records: dict[str, dict[str, Any]], units_payload: dict[str, Any]) -> dict[str, Any]:
    units = units_payload["units"]
    reviews = latest(read_jsonl(REVIEW), REVIEW_VERSIONS)
    verifications = latest(read_jsonl(VERIFY), VERIFY_VERSIONS)
    runs = json.loads(APPLIED.read_text(encoding="utf-8"))["runs"] if APPLIED.exists() else []
    applied_by_field: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:  # later runs refine earlier fixes of the same field; keep the original "old"
        for row in run["applied"]:
            key = (row["record_id"], row["path"])
            applied_by_field[key] = (
                {**row, "old": applied_by_field[key]["old"]} if key in applied_by_field else row
            )
    disputed_by_field = defaultdict(list)
    for row in runs[-1]["disputed"] if runs else []:
        if row["unit"] in units:
            disputed_by_field[(row["record_id"], row["path"])].append(row)
    unit_of_field = {}
    for unit in units.values():
        for record_id, ru_path in unit["occurrences"]:
            unit_of_field[(record_id, ru_path)] = unit
    # Deterministic checks on the reviewed snapshot.
    variants: dict[tuple[str, str], set[str]] = defaultdict(set)
    for unit in units.values():
        for kind in unit["kinds"]:
            if kind in ("category", "link", "heading") and len(unit["en"]) <= 80:
                variants[(kind, unit["en"])].add(unit["ru"])
    report_records: dict[str, Any] = {}
    totals = Counter()
    for record_id, record in records.items():
        checked, issues, fixed, disputed, source_errors, deterministic = [], [], [], [], [], []
        for field in fields(record):
            key = (record_id, field["ru_path"])
            unit = unit_of_field.get(key)
            if unit is None:  # field text changed after collect: report it as not covered
                checked.append([field["ru_path"], None, "not_covered"])
                totals["not_covered"] += 1
                continue
            review = reviews.get(unit["id"])
            verdict = review["verdict"] if review else "pending"
            checked.append([field["ru_path"], unit["id"][:16], verdict])
            totals[f"verdict_{verdict}"] += 1
            if review and review["issues"]:
                issues.append(
                    {
                        "path": field["ru_path"],
                        "types": sorted({issue["type"] for issue in review["issues"]}),
                        "notes": [issue["note"] for issue in review["issues"]],
                        "verdict": verdict,
                    }
                )
            if review and verdict == "source_error":
                source_errors.append(
                    {
                        "path": field["en_path"],
                        "note": review.get("note") or "; ".join(issue["note"] for issue in review["issues"]),
                    }
                )
            if review and verdict == "uncertain":
                disputed.append(
                    {"path": field["ru_path"], "reason": "reviewer_uncertain", "note": review.get("note", "")}
                )
            for row in disputed_by_field.get(key, []):
                disputed.append(
                    {
                        "path": field["ru_path"],
                        "reason": row["reason"],
                        "proposal": row["new"],
                        "note": row.get("verifier_reason", ""),
                    }
                )
            if key in applied_by_field:
                row = applied_by_field[key]
                fixed.append(
                    {
                        "path": field["ru_path"],
                        "old": row["old"],
                        "new": row["new"],
                        "source": row["source"],
                        "types": row["types"],
                        "verification": row.get("verification", "accepted_by_verifier"),
                    }
                )
            if protected_tokens(unit["en"]) != protected_tokens(unit["ru"]):
                deterministic.append(
                    {
                        "check": "numbers_units",
                        "path": field["ru_path"],
                        "en": protected_tokens(unit["en"]),
                        "ru": protected_tokens(unit["ru"]),
                    }
                )
            for kind in unit["kinds"]:
                if len(variants.get((kind, unit["en"]), ())) > 1:
                    deterministic.append(
                        {
                            "check": "inconsistent_translation",
                            "path": field["ru_path"],
                            "variants": sorted(variants[(kind, unit["en"])]),
                        }
                    )
                    break
        for (rid, path), row in applied_by_field.items():
            if (
                rid == record_id
                and row["source"] == "link_sync"
                and not any(item["path"] == path for item in fixed)
            ):
                fixed.append(
                    {
                        "path": path,
                        "old": row["old"],
                        "new": row["new"],
                        "source": "link_sync",
                        "types": ["link"],
                    }
                )
        blocks = record["locales"]["ru"]["blocks"]
        for index, section in enumerate(record["sections"]):
            for field_name in ("heading", "body"):
                if section.get(field_name) and section.get(f"{field_name}_ru") != blocks.get(
                    section["id"], {}
                ).get(field_name):
                    deterministic.append(
                        {"check": "section_block_mismatch", "path": f"/sections/{index}/{field_name}_ru"}
                    )
        totals["fixed_without_verification"] += sum(
            1 for item in fixed if item.get("verification") == "applied_without_verification"
        )
        totals["fields"] += len(checked)
        totals["fixed"] += len(fixed)
        totals["disputed"] += len(disputed)
        totals["source_errors"] += len(source_errors)
        report_records[record_id] = {
            "review_status": record["review"]["status"],
            "fields_checked": checked,
            "issues": issues,
            "fixed": fixed,
            "disputed": disputed,
            "source_errors": source_errors,
            "deterministic": deterministic,
        }
    review_models = Counter(row.get("model", "?") for row in reviews.values())
    verify_models = Counter(row.get("model", "?") for row in verifications.values())
    report = {
        "schema_version": 1,
        "generated_at": now(),
        "method": "Автоматическое ревью: рецензент-модель проверяет каждое поле в контексте термина, каждое предложенное исправление независимо проверяет другая модель, затем детерминированные проверки. Не является подтверждением человеком; статусы review записей не меняются.",
        "models": {"review": REVIEW_MODEL, "verify": VERIFY_MODEL},
        "models_used": {
            "review_units_by_model": dict(review_models),
            "verified_proposals_by_model": dict(verify_models),
        },
        "prompt_versions": {"review": REVIEW_PROMPT_VERSION, "verify": VERIFY_PROMPT_VERSION},
        "field_columns": ["ru_path", "review_unit", "verdict"],
        "totals": dict(totals)
        | {
            "records": len(report_records),
            "review_units": len(units),
            "verified_proposals": len(verifications),
        },
        "records": report_records,
    }
    write_json(REPORT, report)
    write_summary(report)
    print(json.dumps(report["totals"], ensure_ascii=False), flush=True)
    return report


def write_summary(report: dict[str, Any]) -> None:
    totals = report["totals"]
    types = Counter()
    for record in report["records"].values():
        for item in record["fixed"]:
            types.update(item["types"])
    lines = [
        "# Ревью переводов глоссария",
        "",
        report["method"],
        "",
        f"- Записей: {totals['records']}; проверено полей: {totals['fields']}; уникальных пар EN/RU: {totals['review_units']}.",
        f"- Вердикты по полям: ok {totals.get('verdict_ok', 0)}, исправить {totals.get('verdict_fix', 0)}, ошибка оригинала {totals.get('verdict_source_error', 0)}, спорно {totals.get('verdict_uncertain', 0)}, не проверено {totals.get('verdict_pending', 0) + totals.get('not_covered', 0)}.",
        f"- Исправлено полей: {totals['fixed']}, из них без независимой проверки (решение заказчика, второй проход рецензента): {totals.get('fixed_without_verification', 0)}; спорных мест: {totals['disputed']}; замечаний к английскому оригиналу: {totals['source_errors']}.",
        "- Типы исправлений: " + ", ".join(f"{name} {count}" for name, count in types.most_common()) + ".",
        "",
        "Модели по фактическим записям журналов: рецензент — "
        + ", ".join(
            f"`{name}` ({count})" for name, count in report["models_used"]["review_units_by_model"].items()
        )
        + "; проверка исправлений — "
        + ", ".join(
            f"`{name}` ({count})"
            for name, count in report["models_used"]["verified_proposals_by_model"].items()
        )
        + ". Полный отчёт по каждому ID: `report.json`.",
        "",
        "## Ошибки английского оригинала",
        "",
    ]
    for record_id, record in report["records"].items():
        seen: set[str] = set()
        for item in record["source_errors"]:
            if item["note"] in seen:
                continue  # the same English text is stored in several fields of one record
            seen.add(item["note"])
            lines.append(f"- `{record_id}` `{item['path']}`: {item['note']}")
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
