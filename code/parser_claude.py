import re
import json
import yaml
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "claude"
INDEX_FILE = DATA_DIR / "index.md"


def parse_index(limit: int | None = None) -> list[dict]:
    """
    Read index.md and return a list of {title, rel_path, category} dicts.
    limit: if set, return only the first N entries.
    """
    entries = []
    current_category = None

    for line in INDEX_FILE.read_text(encoding="utf-8").splitlines():
        # ## Category heading
        cat_match = re.match(r"^## (.+)$", line)
        if cat_match:
            current_category = cat_match.group(1).strip()
            continue

        # - [Title](relative/path.md)
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


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter and body. Returns (meta_dict, body_str)."""
    if not text.startswith("---"):
        return {}, text

    end = text.index("---", 3)
    fm_raw = text[3:end].strip()
    body = text[end + 3:].strip()
    meta = yaml.safe_load(fm_raw) or {}
    return meta, body


def clean_content(body: str) -> str:
    """Remove title, last updated line, related articles, then flatten to single paragraph."""
    # Remove _Last updated: ..._ line
    body = re.sub(r"^_Last updated:.*_\n?", "", body, flags=re.MULTILINE)
    # Remove title (first # heading)
    body = re.sub(r"^# .+\n?", "", body, count=1)
    # Remove ## Related Articles and everything after
    body = re.split(r"^## Related Articles", body, maxsplit=1, flags=re.MULTILINE)[0]
    # Collapse all newlines into a single space
    return " ".join(line.strip() for line in body.splitlines() if line.strip())


def parse_article(rel_path: str, category: str) -> dict | None:
    """Parse a single article file and return a structured dict."""
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
        "article_id": meta.get("article_id", ""),
        "title": title,
        "url": meta.get("source_url", ""),
        "company": "claude",
        "category": category,
        "last_updated": meta.get("last_updated_iso", ""),
        "content": clean_content(body),
    }


def parse_claude_docs(limit: int | None = None) -> list[dict]:
    """Parse Claude docs. limit applies to index entries (articles)."""
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
    docs = parse_claude_docs()
    out_path = Path(__file__).parent / "claude_docs.json"
    out_path.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nParsed {len(docs)} articles → {out_path}")
    print("\n--- Sample (first doc) ---")
    if docs:
        print(json.dumps(docs[0], indent=2, ensure_ascii=False))
