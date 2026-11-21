"""Full field-by-field EN→RU audit of every canonical glossary record.

Phases, each resumable from append-only JSONL checkpoints in data/eval/translation_audit/:

  collect  enumerate every non-empty translatable field of every record; identical EN/RU pairs
           form one review unit (deterministic, no network);
  review   reviewer model checks each unit in the context of its term;
  verify   an independent model re-checks every proposed correction;
  apply    writes verified corrections that pass deterministic guards into data/processed/terms,
           never touching reviewed records/media or earlier editorial edits, then synchronises
           related-term link texts with the final Russian titles;
  report   writes the per-record machine-checkable report (report.json + summary.md).

Automatic review is not a human confirmation: review.status of records is never changed here.
Usage: python -m scripts.translation_audit all --workers 8
"""

import argparse
import json
import os
from pathlib import Path

from app.infra.config import load_env
from scripts.translation_review.apply import run_apply, sync_editorial
from scripts.translation_review.collect import collect
from scripts.translation_review.common import OUT, UNITS, load_records
from scripts.translation_review.packages import (
    export_review,
    export_verify,
    import_review,
    import_verify,
)
from scripts.translation_review.report import run_report
from scripts.translation_review.review import run_review, run_verify


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "phase",
        choices=[
            "sync-editorial",
            "collect",
            "review",
            "verify",
            "export-review",
            "import-review",
            "export-verify",
            "import-verify",
            "apply",
            "report",
            "all",
        ],
    )
    parser.add_argument("--out", type=Path, default=OUT / "packages")
    parser.add_argument("--chunk", type=int, default=80)
    parser.add_argument("--files", type=Path, nargs="*", default=[])
    parser.add_argument("--model", default="gpt-6-astra (Codex CLI)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=9000)
    parser.add_argument(
        "--limit-batches", type=int, default=None, help="pilot: review only the first N pending packages"
    )
    args = parser.parse_args()
    load_env()
    records = load_records()
    if args.phase in ("sync-editorial", "all"):
        if sync_editorial(records):
            records = load_records()
        if args.phase == "sync-editorial":
            return
    if args.phase in ("collect", "all") or not UNITS.exists():
        payload = collect(records)
        print(
            f"collect: records={payload['records']} fields={payload['occurrences']} units={len(payload['units'])}",
            flush=True,
        )
    else:
        payload = json.loads(UNITS.read_text(encoding="utf-8"))
    units = payload["units"]
    if args.phase == "export-review":
        export_review(units, records, args.out, args.max_chars)
        return
    if args.phase == "import-review":
        import_review(units, args.files, args.model)
        return
    if args.phase == "export-verify":
        export_verify(units, records, args.out, args.chunk)
        return
    if args.phase == "import-verify":
        import_verify(units, args.files, args.model)
        return
    key = os.getenv("DEEPSEEK_API_KEY", "")
    for phase in ("review", "verify"):
        if args.phase not in (phase, "all"):
            continue
        backend = os.getenv(f"AUDIT_{phase.upper()}_BACKEND", "deepseek")
        if backend == "deepseek" and not key:
            raise SystemExit("DEEPSEEK_API_KEY missing")
        if backend == "runpod" and not (os.getenv("RUNPOD_API_KEY") and os.getenv("RUNPOD_ENDPOINT_ID")):
            raise SystemExit("RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID required")
        if backend not in ("deepseek", "runpod", "ollama"):
            raise SystemExit(f"Unknown audit backend: {backend}")
    if args.phase in ("review", "all"):
        run_review(units, records, key, args.workers, args.max_chars, args.limit_batches)
    if args.phase in ("verify", "all"):
        run_verify(units, records, key, args.workers)
    if args.phase in ("apply", "all"):
        run_apply(units, records)
        records = load_records()
    if args.phase in ("report", "all"):
        run_report(records, payload)
