import re
import json
import yaml
from pathlib import Path
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "hackerrank"
INDEX_FILE = DATA_DIR / "index.md"


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
            entries.append({
                "title": art_match.group(1).strip(),
                "rel_path": art_match.group(2).strip(),
                "category": current_category,
            })
            if limit and len(entries) >= limit:
                break

    return entries


def _regex_frontmatter(fm_raw: str) -> dict:
    """Fallback: extract key fields from raw frontmatter text via regex."""
    meta = {}
    for key in ("title", "source_url", "article_slug", "last_updated_exact"):
        m = re.search(rf'^{key}:\s*"?([^\n"]+)"?', fm_raw, re.MULTILINE)
        if m:
            meta[key] = m.group(1).strip()
    return meta


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.index("---", 3)
    fm_raw = text[3:end].strip()
    body = text[end + 3:].strip()
    try:
        meta = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError:
        meta = _regex_frontmatter(fm_raw)
    return meta, body


def strip_html(text: str) -> str:
    """Replace HTML blocks with their plain text equivalent."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def clean_content(body: str) -> str:
    # Remove _Last updated: ..._ line
    body = re.sub(r"^_Last updated:.*_\n?", "", body, flags=re.MULTILINE)
    # Remove title (first # heading)
    body = re.sub(r"^# .+\n?", "", body, count=1)
    # Convert any HTML to plain text
    body = strip_html(body)
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

    return {
        "article_id": meta.get("article_slug", ""),
        "title": title,
        "url": meta.get("source_url", ""),
        "company": "hackerrank",
        "category": category,
        "last_updated": meta.get("last_updated_exact", ""),
        "content": clean_content(body),
    }


def parse_hackerrank_docs(limit: int | None = None) -> list[dict]:
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
    docs = parse_hackerrank_docs()
    out_path = Path(__file__).parent / "hackerrank_docs.json"
    out_path.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nParsed {len(docs)} articles → {out_path}")
    print("\n--- Sample (first doc) ---")
    if docs:
        print(json.dumps(docs[0], indent=2, ensure_ascii=False))
