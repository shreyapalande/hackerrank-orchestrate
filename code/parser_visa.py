import re
import json
import yaml
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "visa"
INDEX_FILE = DATA_DIR / "index.md"

SKIP_FILES = {"support.md"}


def parse_index(limit: int | None = None) -> list[dict]:
    entries = []
    current_category = None

    for line in INDEX_FILE.read_text(encoding="utf-8").splitlines():
        cat_match = re.match(r"^## (.+)$", line)
        if cat_match:
            current_category = cat_match.group(1).strip()
            continue

        art_match = re.match(r"^- \[(.+?)\]\((.+?)\)$", line)
        if art_match and current_category:
            rel_path = art_match.group(2).strip()
            if rel_path in SKIP_FILES:
                continue
            entries.append({
                "title": art_match.group(1).strip(),
                "rel_path": rel_path,
                "category": current_category,
            })
            if limit and len(entries) >= limit:
                break

    return entries


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.index("---", 3)
    meta = yaml.safe_load(text[3:end].strip()) or {}
    body = text[end + 3:].strip()
    return meta, body


def strip_cloudflare_emails(text: str) -> str:
    """Replace any markdown link to a Cloudflare email-protection URL with [email protected]."""
    # Handles both [[email protected]](url) and [\[email@...\]](url) variants
    return re.sub(
        r"\[+[^\n]*?\]\(/cdn-cgi/l/email-protection#[a-f0-9]+\)",
        "[email protected]",
        text,
    )


def clean_content(body: str) -> str:
    # Replace Cloudflare-obfuscated email links with their display label
    body = strip_cloudflare_emails(body)
    # Remove _Last modified: ..._ or _Last updated: ..._ line
    body = re.sub(r"^_Last (?:modified|updated):.*_\n?", "", body, flags=re.MULTILINE)
    # Remove title (first # heading)
    body = re.sub(r"^# .+\n?", "", body, count=1)
    # Collapse all newlines into a single space
    return " ".join(line.strip() for line in body.splitlines() if line.strip())


def parse_article(rel_path: str, category: str) -> dict | None:
    abs_path = DATA_DIR / rel_path
    print(f"  Reading: {abs_path.resolve()}")
    if not abs_path.exists():
        print(f"  [WARN] File not found: {abs_path}")
        return None

    raw = abs_path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)

    title = meta.get("title", "").strip()
    if not title:
        print(f"  [WARN] No title in: {rel_path}")
        return None

    # Derive article_id from filename stem
    article_id = Path(rel_path).stem

    return {
        "article_id": article_id,
        "title": title,
        "url": meta.get("final_url") or meta.get("source_url", ""),
        "company": "visa",
        "category": category,
        "last_updated": meta.get("last_modified", ""),
        "content": clean_content(body),
    }


def parse_visa_docs(limit: int | None = None) -> list[dict]:
    entries = parse_index(limit=limit)
    print(f"Index entries to process: {len(entries)}")

    results = []
    for entry in entries:
        print(f"  Parsing: {entry['rel_path']}")
        doc = parse_article(entry["rel_path"], entry["category"])
        if doc:
            results.append(doc)

    return results


if __name__ == "__main__":
    docs = parse_visa_docs()
    out_path = Path(__file__).parent / "parsed_files" / "visa_docs.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nParsed {len(docs)} articles → {out_path}")
    print("\n--- Sample (first doc) ---")
    if docs:
        print(json.dumps(docs[0], indent=2, ensure_ascii=False))
