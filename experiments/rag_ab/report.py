"""Aggregate paired outcomes, identify repeat cases, and export a blind comparison."""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from itertools import combinations
from typing import Any

from app.infra.files import write_json
from experiments.rag_ab.live import usage_totals
from experiments.rag_ab.retrieval import CONFIGS, OLD, OUT, ROOT, packing_config, questions_with_sha
from scripts.evaluation.retrieval import percentile

PHASES = ("packing", "llm", "oae", "legacy", "repeat", "repeat2")


def read(name: str) -> Any:
    return json.loads((OUT / name).read_bytes())


def phase_context(phase: str, suite_sha: str, retrieval_sha: str | None = None) -> str:
    """Read the budget used by the phase and verify available input hashes."""
    name = f"{phase}-run.json"
    manifest = read(name)
    context = manifest.get("context")
    choices = {"factorial"} if phase == "packing" else CONFIGS
    if manifest.get("phase") != phase or not isinstance(context, str) or context not in choices:
        raise SystemExit(f"Invalid phase or context in {name}")
    if manifest.get("suite_sha256", suite_sha) != suite_sha:
        raise SystemExit(f"Question suite differs from {name}")
    if retrieval_sha is not None and manifest.get("retrieval_sha256", retrieval_sha) != retrieval_sha:
        raise SystemExit(f"Retrieval snapshot differs from {name}")
    return context


def by_variant(rows: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    return grouped


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    totals = Counter()
    for row in rows:
        totals.update(row["usage"])
    measured = [row["latency_ms"] for row in rows]
    answered = [row for row in rows if row.get("result")]
    result = {
        "n": n,
        "answered": len(answered),
        "passed": sum(bool(row["passed"]) for row in rows),
        "mode_errors": sum(not row["mode_correct"] for row in answered),
        "coverage_errors": sum(not row["coverage_correct"] for row in answered),
        "source_integrity_errors": sum(not row["source_valid"] for row in answered),
        "call_or_schema_errors": sum("error" in row for row in rows),
        "catalog_misses": sum(row["variant"] == "direct" and not row.get("result") for row in rows),
        "invalid_plans": sum("plan_error" in row for row in rows),
        "second_calls": sum(len(row["calls"]) >= 2 for row in rows),
        "p50_ms": percentile(measured, 0.5),
        "p95_ms": percentile(measured, 0.95),
        "mean_calls": round(sum(len(row["calls"]) for row in rows) / n, 2),
        "mean_input_tokens": round(totals.get("prompt_tokens", 0) / n, 1),
        "mean_output_tokens": round(totals.get("completion_tokens", 0) / n, 1),
        "tokens": dict(totals),
    }
    grades = [row["grade"] for row in rows if "grade" in row]
    if grades:
        result.update(
            quality=sum(row["quality"] for row in rows),
            audited_quality=sum(row.get("audited_quality", row["quality"]) for row in rows),
            audited_passed=sum(row.get("audited_passed", row["passed"]) for row in rows),
            request_met=sum(bool(grade.get("request_met")) for grade in grades),
            correct=sum(bool(grade.get("correct")) for grade in grades),
            grounded=sum(bool(grade.get("grounded")) for grade in grades),
            false_refusals=sum(bool(grade.get("false_refusal")) for grade in grades),
            facts_used=sum(sum(grade.get("used_facts", [])) for grade in grades),
        )
    return result


def pair(variants: dict[str, list[dict]], left: str, right: str, metric: str) -> dict:
    a = {row["id"]: row for row in variants.get(left, [])}
    b = {row["id"]: row for row in variants.get(right, [])}
    baseline_key = metric.removeprefix("audited_")
    paired = []
    for case_id, row in a.items():
        if case_id in b:
            left_pass = bool(row.get(metric, row[baseline_key]))
            right_pass = bool(b[case_id].get(metric, b[case_id][baseline_key]))
            paired.append((case_id, left_pass, right_pass))
    return {
        "n": len(paired),
        "left_only": sorted(case_id for case_id, a_pass, b_pass in paired if a_pass and not b_pass),
        "right_only": sorted(case_id for case_id, a_pass, b_pass in paired if b_pass and not a_pass),
        "both_pass": sum(a_pass and b_pass for _, a_pass, b_pass in paired),
        "both_fail": sum(not a_pass and not b_pass for _, a_pass, b_pass in paired),
    }


def select_repeats(cases: list[dict], rows: list[dict]) -> dict[str, list[str]]:
    repeats = {}
    for case in cases:
        if not case.get("ambiguous"):
            continue
        attempts = [row for row in rows if row["id"] == case["id"]]
        graded_modes = {row.get("result", {}).get("mode") for row in attempts}
        disagreed = len({(row["quality"], row["passed"]) for row in attempts}) > 1 or len(graded_modes) > 1
        if disagreed or any("plan_error" in row for row in attempts):
            repeats[case["id"]] = ["A", "B", "C", "C-low"]
    return repeats


def what_happened(snapshot: dict, config: str, judgment: dict) -> list[str]:
    raw = snapshot["candidates"]
    found_ranks = snapshot["configs"][config]["ranks"]
    flags = judgment.get("used_facts", [])
    outcomes = []
    for index, (packed, used) in enumerate(zip(found_ranks, flags)):
        if used:
            outcome = "передан и использован" if packed else "эквивалент использован"
        elif packed:
            outcome = "передан, в ответе не подтверждён"
        elif raw["lexical_sections"][index] or raw["semantic_40"][index]:
            outcome = "найден и отсеян"
        else:
            outcome = "не найден"
        outcomes.append(outcome)
    return outcomes


def apply_review(phases: dict[str, list[dict]]) -> None:
    reviews = json.loads((ROOT / "manual-review.json").read_bytes())
    for phase, reviewed in reviews.items():
        for row in phases[phase]:
            manual = reviewed.get(row["label"])
            if manual is None:
                continue
            assert manual["id"] == row["id"]
            result = row.get("result")
            if not result:
                continue
            payload = [result["answer"], result["mode"], result["coverage"], result["sources"]]
            digest = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            if digest != manual["answer_sha256"]:
                continue
            row["manual_review"] = manual
            grade = {**row["grade"], **manual["grade"]}
            row["audited_quality"] = (
                all(grade[k] for k in ("request_met", "correct", "grounded")) and row["source_valid"]
            )
            row["audited_passed"] = row["audited_quality"] and row["mode_correct"] and row["coverage_correct"]


def write_blind(rows: list[dict], cases: list[dict]) -> None:
    blind = [
        "# Слепое сравнение ответов на новых вопросах",
        "",
        "У каждого вопроса своя пара анонимных меток. Указаны фактический ответ и источники.",
        "",
    ]
    key = {}
    for case in cases:
        ident = case["id"]
        blind.extend([f"## {ident}", "", case["question"], ""])
        key[ident] = {}
        replies = [row for row in rows if row["id"] == ident and row["variant"] in {"A", "B"}]
        for row in sorted(replies, key=lambda row: row["label"]):
            label = row["label"]
            key[ident][label] = row["variant"]
            blind.extend([f"### {label}", ""])
            result = row.get("result")
            if not result:
                blind.extend([f"Ответ недоступен ({row['error']}).", ""])
                continue
            blind.extend([result["answer"], ""])
            if result["sources"]:
                blind.extend(
                    [
                        "Источники",
                        "",
                        *[f"- `{r['record_id']}#{r['section_id']}`" for r in result["sources"]],
                        "",
                    ]
                )
    (OUT / "blind-ab.md").write_text("\n".join(blind), encoding="utf-8")
    write_json(OUT / "blind-key.json", key)


def compact_row(row: dict) -> dict:
    keep = {
        "variant",
        "label",
        "grade",
        "source_valid",
        "mode_correct",
        "coverage_correct",
        "quality",
        "passed",
        "error",
        "plan",
        "plan_error",
        "latency_ms",
        "retrieval_ms",
        "answer_ms",
        "usage",
        "facts",
        "forbidden_hits",
        "manual_review",
        "audited_quality",
        "audited_passed",
    }
    report = {key: value for key, value in row.items() if key in keep}
    report["calls"] = len(row.get("calls", []))
    if result := row.get("result"):
        report.update(
            mode=result["mode"],
            coverage=result["coverage"],
            answer=result["answer"],
            sources=result["sources"],
        )
    return report


def warm_legacy(row: dict) -> dict:
    calls = [{"usage": row["usage"]}]
    return {**row, "calls": calls, "usage": usage_totals([row["usage"]]), "variant": "A-6x8500"}


def build(include_repeats: bool) -> None:
    new, new_sha = questions_with_sha(ROOT / "questions.json")
    llm: list[dict] = read("llm-scored.json")
    if not include_repeats:
        phase_context("llm", new_sha)
        write_json(OUT / "repeat-cases.json", select_repeats(new, llm))
        return
    assert {case["id"] for case in new} == {r["id"] for r in llm if r["variant"] == "A"}
    retrieval_bytes = (OUT / "retrieval.json").read_bytes()
    retrieval_sha = hashlib.sha256(retrieval_bytes).hexdigest()
    sources = json.loads(retrieval_bytes)
    oae, oae_sha = questions_with_sha(OLD / "oae_passage_questions.json")
    legacy_cases, legacy_sha = questions_with_sha(OLD / "full_questions.json")
    suite_hashes = {"oae": oae_sha, "legacy": legacy_sha}
    contexts = {
        phase: phase_context(phase, suite_hashes.get(phase, new_sha), None if phase == "legacy" else retrieval_sha)
        for phase in PHASES
    }
    write_blind(llm, new)
    traces = {row["id"]: row for row in sources["results"]}
    legacy: dict = read("legacy-live.json")
    phases: dict[str, list[dict]] = {name: read(f"{name}-scored.json") for name in PHASES}
    apply_review(phases)
    phases["legacy"].extend(warm_legacy(row) for row in legacy["results"])
    summary = {
        phase: {variant: metrics(rows) for variant, rows in by_variant(data).items()}
        for phase, data in phases.items()
    }
    ambiguous_ids = {case["id"] for case in new if case.get("ambiguous")}
    ambiguous = by_variant(row for row in llm if row["id"] in ambiguous_ids)
    summary["llm_ambiguous"] = {variant: metrics(rows) for variant, rows in ambiguous.items()}
    variants = by_variant(llm)
    summary["comparisons"] = {
        f"{a}/{b}": {
            key: pair(variants, a, b, key)
            for key in ("quality", "passed", "audited_quality", "audited_passed")
        }
        for a, b in combinations(("A", "B", "C", "C-low"), 2)
    }
    summary["repeated"] = {
        variant: metrics(rows) for variant, rows in by_variant(phases["repeat"] + phases["repeat2"]).items()
    }
    summary["retrieval"] = sources["metrics"]
    write_json(OUT / "summary.json", summary)

    case_reports = {}
    for suite, cases in (("new", new), ("oae", oae), ("legacy", legacy_cases)):
        case_reports[suite] = {}
        for case in cases:
            ident = case["id"]
            trace = traces.get(ident)
            report = {**case, "phases": {}}
            if trace:
                report["retrieval"] = {
                    "candidates": trace["candidates"],
                    "configs": {
                        name: {k: v for k, v in c.items() if k != "evidence"}
                        for name, c in trace["configs"].items()
                    },
                }
            for phase, data in phases.items():
                matched = [row for row in data if row["id"] == ident]
                if not matched:
                    continue
                report["phases"][phase] = []
                for row in matched:
                    compact = compact_row(row)
                    if trace and row.get("grade"):
                        config = packing_config(phase, row["variant"], contexts[phase])
                        grade = {**row["grade"], **row.get("manual_review", {}).get("grade", {})}
                        compact["parts"] = what_happened(trace, config, grade)
                    report["phases"][phase].append(compact)
            case_reports[suite][ident] = report
    write_json(OUT / "per-question.json", case_reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-repeats", action="store_true")
    args = parser.parse_args()
    build(not args.select_repeats)
