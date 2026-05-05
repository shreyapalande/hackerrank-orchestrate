import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
TICKETS_CSV = ROOT_DIR / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV  = ROOT_DIR / "support_tickets" / "output.csv"

OUTPUT_FIELDS = ["Issue", "Subject", "Company", "Response", "Product Area", "Status", "Request Type"]


def _load_done(output_path: Path) -> set:
    """Return set of (issue, subject, company) already written to output."""
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


def run(tickets_path: Path = TICKETS_CSV, output_path: Path = OUTPUT_CSV):
    from agent import resolve

    with open(tickets_path, newline="", encoding="utf-8") as f:
        tickets = list(csv.DictReader(f))

    done = _load_done(output_path)

    needs_header = not output_path.exists() or output_path.stat().st_size == 0
    out_f = open(output_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
    if needs_header:
        writer.writeheader()
        out_f.flush()

    total = len(tickets)
    skipped = 0

    try:
        for i, ticket in enumerate(tickets, 1):
            issue   = ticket.get("Issue", "").strip()
            subject = ticket.get("Subject", "").strip()
            company = ticket.get("Company", "").strip()

            key = (issue, subject, company)
            if key in done:
                skipped += 1
                print(f"  [{i:02d}/{total}] SKIP (already done): {company} | {subject[:40] or issue[:40]}")
                continue

            print(f"  [{i:02d}/{total}] {company} | {subject[:50] or issue[:50]}", end=" ... ", flush=True)

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
    print(f"\nDone. {processed} processed, {skipped} skipped. Output: {output_path}")


if __name__ == "__main__":
    tickets_path = Path(sys.argv[1]) if len(sys.argv) > 1 else TICKETS_CSV
    run(tickets_path=tickets_path)
