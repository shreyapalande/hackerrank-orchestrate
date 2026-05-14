"""
Run all parsers and write their JSON outputs to code/.

Usage:
    python code/parser.py            # parse everything
    python code/parser.py --check    # dry-run: print counts only, no files written
"""

import json
import sys
import time
from pathlib import Path

CODE_DIR    = Path(__file__).parent
PARSED_DIR  = CODE_DIR / "parsed_files"

OUTPUTS = {
    "claude":       PARSED_DIR / "claude_docs.json",
    "hackerrank":   PARSED_DIR / "hackerrank_docs.json",
    "visa":         PARSED_DIR / "visa_docs.json",
    "visa_support": PARSED_DIR / "visa_support.json",
}


def _write(path: Path, data: list[dict], dry_run: bool) -> int:
    if not dry_run:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(data)


def run(dry_run: bool = False):
    from parser_claude      import parse_claude_docs
    from parser_hackerrank  import parse_hackerrank_docs
    from parser_visa        import parse_visa_docs
    from parse_visa_support import parse_visa_support

    steps = [
        ("Claude docs",      parse_claude_docs,     OUTPUTS["claude"]),
        ("HackerRank docs",  parse_hackerrank_docs,  OUTPUTS["hackerrank"]),
        ("Visa docs",        parse_visa_docs,        OUTPUTS["visa"]),
        ("Visa support Q&A", parse_visa_support,     OUTPUTS["visa_support"]),
    ]

    PARSED_DIR.mkdir(exist_ok=True)

    totals = {}
    grand_total = 0

    print(f"{'─'*50}")
    print(f"  PARSER {'(dry-run)' if dry_run else ''}")
    print(f"{'─'*50}")

    for label, parse_fn, out_path in steps:
        print(f"\n[{label}]")
        t0 = time.time()
        docs = parse_fn()
        elapsed = time.time() - t0
        count = _write(out_path, docs, dry_run)
        totals[label] = count
        grand_total += count
        action = "would write" if dry_run else "->"
        print(f"  {action} {out_path.name}  ({count} entries, {elapsed:.1f}s)")

    print(f"\n{'─'*50}")
    print(f"  TOTAL: {grand_total} documents")
    for label, count in totals.items():
        print(f"    {label:<20} {count:>5}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    dry_run = "--check" in sys.argv
    run(dry_run=dry_run)
