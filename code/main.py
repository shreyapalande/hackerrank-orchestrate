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
import time
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


# ── Groq free-tier limits for llama-3.3-70b-versatile ─────────────────────────
_RPM_LIMIT = 30          # requests per minute  → 2 s minimum gap
_TPM_LIMIT = 12_000      # tokens per minute
_TPM_BUFFER = 0.85       # start sleeping at 85 % of limit
_MIN_GAP = 60.0 / _RPM_LIMIT          # 2 s between requests


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_test(args):
    ensure_db()
    from agent import resolve
    from openai import RateLimitError

    with open(TICKETS_CSV, newline="", encoding="utf-8") as f:
        tickets = list(csv.DictReader(f))

    total = len(tickets)
    print(f"Processing {total} tickets → {OUTPUT_CSV}")
    print(f"Rate limits: {_RPM_LIMIT} RPM · {_TPM_LIMIT} TPM\n")

    # rolling 60-second token window
    win_start  = time.monotonic()
    win_tokens = 0
    day_tokens = 0

    out_f  = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
    writer.writeheader()
    out_f.flush()

    try:
        for i, ticket in enumerate(tickets, 1):
            issue   = ticket.get("Issue", "").strip()
            subject = ticket.get("Subject", "").strip()
            company = ticket.get("Company", "").strip()

            # ── TPM guard: sleep until window resets if near limit ────────────
            if win_tokens >= _TPM_LIMIT * _TPM_BUFFER:
                elapsed  = time.monotonic() - win_start
                sleep_for = max(60.0 - elapsed + 1.0, 1.0)
                print(f"  [TPM {win_tokens}/{_TPM_LIMIT}] sleeping {sleep_for:.0f}s for window reset ...", flush=True)
                time.sleep(sleep_for)
                win_start  = time.monotonic()
                win_tokens = 0

            # Reset window counter every 60 s
            if time.monotonic() - win_start >= 60.0:
                win_start  = time.monotonic()
                win_tokens = 0

            print(f"  [{i:02d}/{total}] {company} | {(subject or issue)[:50]}", end=" ... ", flush=True)

            # ── API call with retry on 429 ────────────────────────────────────
            t0 = time.monotonic()
            for attempt in range(1, 4):
                try:
                    result = resolve(issue, subject, company)
                    break
                except RateLimitError:
                    if attempt == 3:
                        raise
                    wait = 30 * attempt
                    print(f"\n  [429] rate limited — waiting {wait}s (attempt {attempt}/3) ...", flush=True)
                    time.sleep(wait)
                    win_start  = time.monotonic()
                    win_tokens = 0

            # ── token accounting ──────────────────────────────────────────────
            used = result.pop("_tokens", 0)
            win_tokens += used
            day_tokens += used

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
            print(f"{result['Status']} / {result['Request Type']}  (tokens: {used}, win: {win_tokens}, day: {day_tokens})")

            # ── RPM guard: enforce minimum gap between requests ────────────────
            elapsed = time.monotonic() - t0
            gap = _MIN_GAP - elapsed
            if gap > 0:
                time.sleep(gap)

    finally:
        out_f.close()

    print(f"\nDone. {total} tickets written to {OUTPUT_CSV}")
    print(f"Total tokens used today: {day_tokens} / 100,000")


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


def cmd_interactive(_args):
    ensure_db()
    from agent import resolve

    print("Interactive mode — press Ctrl-C to quit.\n")

    while True:
        try:
            company = input("Company  (HackerRank / Claude / Visa / leave blank): ").strip()
            subject = input("Subject  (optional, press Enter to skip)         : ").strip()
            print("Issue    (press Enter twice when done)            :")
            lines = []
            while True:
                line = input()
                if line == "" and lines:
                    break
                lines.append(line)
            issue = "\n".join(lines).strip()

            if not issue:
                print("  [empty issue, skipping]\n")
                continue

            print()
            result = resolve(issue, subject, company)

            print(f"  Status       : {result['Status']}")
            print(f"  Product Area : {result['Product Area']}")
            print(f"  Request Type : {result['Request Type']}")
            print(f"  Response:\n")
            for ln in result["Response"].splitlines():
                print(f"    {ln}")
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break


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

    sub.add_parser("build-db",    help="(Re)build all parser JSONs and the ChromaDB index")
    sub.add_parser("interactive", help="Prompt for issue/subject/company and resolve interactively")

    args = parser.parse_args()

    if args.command == "test":
        cmd_test(args)
    elif args.command == "ticket":
        cmd_ticket(args)
    elif args.command == "build-db":
        cmd_build_db(args)
    elif args.command == "interactive":
        cmd_interactive(args)


if __name__ == "__main__":
    main()
