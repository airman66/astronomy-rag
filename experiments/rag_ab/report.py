"""Aggregate paired outcomes, identify repeat cases, and export a blind comparison."""

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations

from app.infra.files import write_json
from experiments.rag_ab.live import add_calls
from experiments.rag_ab.retrieval import OLD, OUT, ROOT, questions
from scripts.evaluation.retrieval import percentile

CONFIG = "8x8500"
PHASES = ("packing", "llm", "oae", "legacy", "repeat", "repeat2")


def read(name):
    return json.loads((OUT / name).read_bytes())


def by_variant(rows):
    return {variant: [r for r in rows if r["variant"] == variant]
            for variant in dict.fromkeys(row["variant"] for row in rows)}


def metrics(rows):
    n = len(rows)
    grade_key = lambda name: sum(bool(row.get("grade", {}).get(name)) for row in rows)
    totals = Counter()
    for row in rows:
        totals.update(row["usage"])
    measured = [row["latency_ms"] for row in rows]
    answered = [row for row in rows if row.get("result")]
    result = {
        "n": n, "answered": len(answered), "passed": sum(bool(row["passed"]) for row in rows),
        "mode_errors": sum(not row["mode_correct"] for row in answered),
        "coverage_errors": sum(not row["coverage_correct"] for row in answered),
        "source_integrity_errors": sum(not row["source_valid"] for row in answered),
        "call_or_schema_errors": sum("error" in row for row in rows),
        "catalog_misses": sum(row["variant"] == "direct" and not row.get("result") for row in rows),
        "invalid_plans": sum("plan_error" in row for row in rows),
        "second_calls": sum(len(row["calls"]) >= 2 for row in rows),
        "p50_ms": percentile(measured, .5), "p95_ms": percentile(measured, .95),
        "mean_calls": round(sum(len(row["calls"]) for row in rows) / n, 2),
        "mean_input_tokens": round(totals.get("prompt_tokens", 0) / n, 1),
        "mean_output_tokens": round(totals.get("completion_tokens", 0) / n, 1),
        "tokens": dict(totals),
    }
    if any("grade" in row for row in rows):
        result.update(
            quality=sum(row["quality"] for row in rows),
            audited_quality=sum(row.get("audited_quality", row["quality"]) for row in rows),
            audited_passed=sum(row.get("audited_passed", row["passed"]) for row in rows),
            request_met=grade_key("request_met"),
            correct=grade_key("correct"), grounded=grade_key("grounded"), false_refusals=grade_key("false_refusal"),
            facts_used=sum(sum(row.get("grade", {}).get("used_facts", [])) for row in rows),
        )
    return result


def pair(rows, left, right, quality):
    variants = by_variant(rows)
    a = {row["id"]: row for row in variants.get(left, [])}
    b = {row["id"]: row for row in variants.get(right, [])}
    baseline_key = quality.removeprefix("audited_")
    paired = [(key, bool(row.get(quality, row[baseline_key])), bool(b[key].get(quality, b[key][baseline_key])))
              for key, row in a.items() if key in b]
    return {
        "n": len(paired), "left_only": sorted(key for key, yes, no in paired if yes and not no),
        "right_only": sorted(key for key, no, yes in paired if yes and not no),
        "both_pass": sum(x and y for _, x, y in paired),
        "both_fail": sum(not x and not y for _, x, y in paired),
    }


def select_repeats(cases, rows):
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


def what_happened(snapshot, config, judgment):
    raw = snapshot["candidates"]
    found_ranks = snapshot["configs"][config]["ranks"]
    flags = judgment.get("used_facts", [])
    return [
        "эквивалент использован" if used and not packed else
        "передан и использован" if used and packed else
        "передан, в ответе не подтверждён" if packed else
        "найден и отсеян" if raw["lexical_sections"][i] or raw["semantic_40"][i] else
        "не найден"
        for i, (packed, used) in enumerate(zip(found_ranks, flags))
    ]


def apply_review(phases):
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
            digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if digest != manual["answer_sha256"]:
                continue
            row["manual_review"] = manual
            grade = {**row["grade"], **manual["grade"]}
            row["audited_quality"] = all(grade[k] for k in ("request_met", "correct", "grounded")) and row["source_valid"]
            row["audited_passed"] = row["audited_quality"] and row["mode_correct"] and row["coverage_correct"]


def write_blind(rows, cases):
    blind = ["# Слепое сравнение ответов на новых вопросах", "",
             "У каждого вопроса своя пара анонимных меток. Указаны фактический ответ и источники.", ""]
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
                blind.extend([
                    "Источники",
                    "",
                    *[f"- `{r['record_id']}#{r['section_id']}`" for r in result["sources"]],
                    "",
                ])
    (OUT / "blind-ab.md").write_text("\n".join(blind), encoding="utf-8")
    write_json(OUT / "blind-key.json", key)


def compact_row(row):
    keep = {"variant", "label", "grade", "source_valid", "mode_correct", "coverage_correct", "quality",
            "passed", "error", "plan", "plan_error", "latency_ms", "retrieval_ms", "answer_ms", "usage",
            "facts", "forbidden_hits", "manual_review", "audited_quality", "audited_passed"}
    report = {key: value for key, value in row.items() if key in keep}
    report["calls"] = len(row.get("calls", []))
    if result := row.get("result"):
        report.update(mode=result["mode"], coverage=result["coverage"], answer=result["answer"], sources=result["sources"])
    return report


def warm_legacy(row):
    calls = [{"usage": row["usage"]}]
    return {**row, "calls": calls, "usage": add_calls(calls), "variant": "A-6x8500"}


def build(include_repeats):
    new = questions(ROOT / "questions.json")
    llm = read("llm-scored.json")
    if not include_repeats:
        write_json(OUT / "repeat-cases.json", select_repeats(new, llm))
        return
    assert {case["id"] for case in new} == {r["id"] for r in llm if r["variant"] == "A"}
    write_blind(llm, new)
    sources = read("retrieval.json")
    traces = {row["id"]: row for row in sources["results"]}
    legacy = read("legacy-live.json")
    phases = {name: read(f"{name}-scored.json") for name in PHASES}
    apply_review(phases)
    phases["legacy"].extend(warm_legacy(row) for row in legacy["results"])
    summary = {phase: {variant: metrics(rows) for variant, rows in by_variant(data).items()}
               for phase, data in phases.items()}
    summary["llm_ambiguous"] = {
        variant: metrics(rows) for variant, rows in by_variant(
            [row for row in llm if row["id"] in {case["id"] for case in new if case.get("ambiguous")}]
        ).items()
    }
    summary["comparisons"] = {
        f"{a}/{b}": {key: pair(llm, a, b, key) for key in ("quality", "passed", "audited_quality", "audited_passed")}
        for a, b in combinations(("A", "B", "C", "C-low"), 2)
    }
    summary["repeated"] = {
        variant: metrics(rows) for variant, rows in by_variant(phases["repeat"] + phases["repeat2"]).items()
    }
    summary["retrieval"] = sources["metrics"]
    write_json(OUT / "summary.json", summary)

    case_reports = {}
    for suite, cases in (
        ("new", new),
        ("oae", questions(OLD / "oae_passage_questions.json")),
        ("legacy", questions(OLD / "full_questions.json")),
    ):
        case_reports[suite] = {}
        for case in cases:
            ident = case["id"]
            trace = traces.get(ident)
            report = {**case, "phases": {}}
            if trace:
                report["retrieval"] = {
                    "candidates": trace["candidates"],
                    "configs": {name: {k: v for k, v in c.items() if k != "evidence"}
                                for name, c in trace["configs"].items()},
                }
            for phase, data in phases.items():
                matched = [row for row in data if row["id"] == ident]
                if not matched:
                    continue
                report["phases"][phase] = []
                for row in matched:
                    compact = compact_row(row)
                    if trace and row.get("grade"):
                        config = row["variant"] if phase == "packing" else {
                            "A-base": "6x8500", "A-wide": "8x11000",
                        }.get(row["variant"], CONFIG)
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
