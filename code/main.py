"""
main.py — unified entry point

Usage:
  python main.py test                        # run all support_tickets.csv (skip done)
  python main.py ticket --issue "..." --subject "..." --company "..."
  python main.py build-db                    # (re)build parsers + ChromaDB index
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT_DIR    = Path(__file__).parent.parent
CODE_DIR    = Path(__file__).parent
TICKETS_CSV = ROOT_DIR / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV  = ROOT_DIR / "support_tickets" / "output.csv"
CHROMA_DIR  = ROOT_DIR / "chroma_db"

OUTPUT_FIELDS = ["Issue", "Subject", "Company", "Response", "Product Area", "Status", "Request Type"]

# JSON files that parsers produce — if any is missing we need to rebuild
_JSON_FILES = [
    CODE_DIR / "claude_docs.json",
    CODE_DIR / "hackerrank_docs.json",
    CODE_DIR / "visa_docs.json",
    CODE_DIR / "visa_support.json",
]

_PARSERS = [
    CODE_DIR / "parser_claude.py",
    CODE_DIR / "parser_hackerrank.py",
    CODE_DIR / "parser_visa.py",
    CODE_DIR / "parse_visa_support.py",
]


# ── vector-db helpers ──────────────────────────────────────────────────────────

def _db_is_ready() -> bool:
    """True if ChromaDB exists and the collection has at least one vector."""
    sqlite = CHROMA_DIR / "chroma.sqlite3"
    if not sqlite.exists():
        return False
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection("support_docs")
        return col.count() > 0
    except Exception:
        return False


def _run_parsers():
    print("Running parsers ...")
    for script in _PARSERS:
        print(f"  {script.name}", end=" ... ", flush=True)
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("FAILED")
            print(result.stderr[-500:])
            sys.exit(1)
        print("ok")


def _run_indexer():
    print("Running indexer ...")
    result = subprocess.run(
        [sys.executable, str(CODE_DIR / "indexer.py")],
        capture_output=False, text=True
    )
    if result.returncode != 0:
        print("Indexer failed.")
        sys.exit(1)


def ensure_db():
    """Build the DB if it doesn't exist yet."""
    if _db_is_ready():
        return
    print("Vector DB not found — building it now.")
    jsons_missing = [f for f in _JSON_FILES if not f.exists()]
    if jsons_missing:
        _run_parsers()
    _run_indexer()
    print("Vector DB ready.\n")


# ── output-csv helpers ─────────────────────────────────────────────────────────

def _load_done(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                row.get("Issue", "").strip(),
                row.get("Subject", "").strip(),
                row.get("Company", "").strip(),
            )
            if any(key):
                done.add(key)
    return done


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_test(args):
    ensure_db()
    from agent import resolve

    with open(TICKETS_CSV, newline="", encoding="utf-8") as f:
        tickets = list(csv.DictReader(f))

    done = _load_done(OUTPUT_CSV)
    needs_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    total   = len(tickets)
    skipped = 0

    out_f  = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
    if needs_header:
        writer.writeheader()
        out_f.flush()

    try:
        for i, ticket in enumerate(tickets, 1):
            issue   = ticket.get("Issue", "").strip()
            subject = ticket.get("Subject", "").strip()
            company = ticket.get("Company", "").strip()
            key = (issue, subject, company)

            if key in done:
                skipped += 1
                print(f"  [{i:02d}/{total}] SKIP: {company} | {(subject or issue)[:50]}")
                continue

            print(f"  [{i:02d}/{total}] {company} | {(subject or issue)[:50]}", end=" ... ", flush=True)
            result = resolve(issue, subject, company)

            writer.writerow({
                "Issue":        issue,
                "Subject":      subject,
                "Company":      company,
                "Response":     result["Response"],
                "Product Area": result["Product Area"],
                "Status":       result["Status"],
                "Request Type": result["Request Type"],
            })
            out_f.flush()
            done.add(key)
            print(f"{result['Status']} / {result['Request Type']}")
    finally:
        out_f.close()

    processed = total - skipped
    print(f"\nDone. {processed} processed, {skipped} skipped. Output: {OUTPUT_CSV}")


def cmd_ticket(args):
    ensure_db()
    from agent import resolve

    issue   = args.issue
    subject = args.subject or ""
    company = args.company or ""

    print(f"Company : {company}")
    print(f"Subject : {subject}")
    print(f"Issue   : {issue}\n")

    result = resolve(issue, subject, company)

    print(f"Status       : {result['Status']}")
    print(f"Product Area : {result['Product Area']}")
    print(f"Request Type : {result['Request Type']}")
    print(f"Response:\n{result['Response']}")


def cmd_build_db(args):
    print("Rebuilding vector DB from scratch ...")
    _run_parsers()
    _run_indexer()
    print("Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Support-ticket agent entry point",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("test", help="Run all tickets in support_tickets.csv (skips already-done rows)")

    p_ticket = sub.add_parser("ticket", help="Resolve a single ticket")
    p_ticket.add_argument("--issue",   required=True,  help="Issue text")
    p_ticket.add_argument("--subject", default="",     help="Subject line (optional)")
    p_ticket.add_argument("--company", default="",     help="Company name (optional)")

    sub.add_parser("build-db", help="(Re)build all parser JSONs and the ChromaDB index")

    args = parser.parse_args()

    if args.command == "test":
        cmd_test(args)
    elif args.command == "ticket":
        cmd_ticket(args)
    elif args.command == "build-db":
        cmd_build_db(args)


if __name__ == "__main__":
    main()
