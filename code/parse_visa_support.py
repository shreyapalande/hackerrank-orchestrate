import re
import json
import yaml
from pathlib import Path

SOURCE_FILE = Path(__file__).parent.parent / "data" / "visa" / "support.md"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.index("---", 3)
    meta = yaml.safe_load(text[3:end].strip()) or {}
    body = text[end + 3:].strip()
    return meta, body


def collapse(text: str) -> str:
    """Flatten multi-line text into a single space-separated string."""
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def parse_phone_table(body: str) -> dict | None:
    """Extract the lost/stolen card phone number block as a single Q&A entry."""
    # Everything from start up to the ## FAQ heading
    faq_idx = body.find("## FAQ")
    if faq_idx == -1:
        return None
    table_block = body[:faq_idx].strip()
    if not table_block:
        return None
    return {
        "question": "What are the phone numbers to call for a lost or stolen Visa card?",
        "answer": collapse(table_block),
        "category": "Contact & Support",
        "url": "https://www.visa.co.in/support.html",
        "company": "visa",
    }


def parse_faq_section(body: str, source_url: str) -> list[dict]:
    """Extract Q&A pairs from the ## FAQ section."""
    faq_idx = body.find("## FAQ")
    if faq_idx == -1:
        return []
    faq_block = body[faq_idx:]

    entries = []
    current_category = "General"
    current_question = None
    current_answer_lines = []

    def flush():
        if current_question and current_answer_lines:
            entries.append({
                "question": current_question,
                "answer": collapse("\n".join(current_answer_lines)),
                "category": current_category,
                "url": source_url,
                "company": "visa",
            })

    for line in faq_block.splitlines():
        # Skip ## FAQ line itself
        if re.match(r"^## FAQ", line):
            continue

        # ### heading — decide if category or question
        h3 = re.match(r"^### (.+)$", line)
        if h3:
            heading = h3.group(1).strip()
            # A heading with no question mark and short = category label
            if "?" not in heading and len(heading.split()) <= 6:
                flush()
                current_category = heading
                current_question = None
                current_answer_lines = []
            else:
                flush()
                current_question = heading
                current_answer_lines = []
            continue

        # Skip section dividers and empty lines when no question is open
        if not current_question:
            continue

        current_answer_lines.append(line)

    flush()
    return entries


def parse_visa_support() -> list[dict]:
    raw = SOURCE_FILE.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)

    # Remove title heading and last modified line
    body = re.sub(r"^# .+\n?", "", body, count=1)
    body = re.sub(r"^_Last modified:.*_\n?", "", body, flags=re.MULTILINE)

    source_url = meta.get("final_url") or meta.get("source_url", "")

    results = []

    phone_entry = parse_phone_table(body)
    if phone_entry:
        results.append(phone_entry)

    results.extend(parse_faq_section(body, source_url))
    return results


if __name__ == "__main__":
    entries = parse_visa_support()
    out_path = Path(__file__).parent / "visa_support.json"
    out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Parsed {len(entries)} entries → {out_path}")
    for e in entries:
        print(f"\n[{e['category']}] Q: {e['question']}")
        print(f"  A: {e['answer'][:120]}...")
