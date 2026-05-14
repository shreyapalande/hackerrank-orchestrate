# Multi-Domain Support Ticket Agent

An AI agent that automatically triages and resolves customer support tickets across three product domains — **HackerRank**, **Claude (Anthropic)**, and **Visa** — using RAG (Retrieval-Augmented Generation) over a local knowledge base.

Built for the [HackerRank Orchestrate Hackathon (May 2026)](https://github.com/interviewstreet/hackerrank-orchestrate-may26).

---

## What it does

Given a support ticket (issue text, subject, company), the agent:

1. **Detects** prompt injection and harmful requests — refuses without an API call
2. **Retrieves** the most relevant knowledge base articles using semantic search (ChromaDB + sentence-transformers)
3. **Escalates** automatically when no relevant KB content is found
4. **Generates** a grounded, structured response via LLM (Llama 3.3 70B on Groq)
5. **Classifies** the ticket into `Product Area`, `Status`, and `Request Type`

All responses are grounded in the provided support corpus — no hallucinated policies or invented URLs.

---

## Architecture

```
support_tickets.csv
        │
        ▼
┌───────────────────┐
│     main.py       │  CLI entry point — batch / interactive / single ticket
└────────┬──────────┘
         │
         ▼
┌───────────────────┐      ┌─────────────────────────┐
│     agent.py      │─────▶│      retriever.py        │
│                   │      │  ChromaDB vector search  │
│  1. Safety check  │      │  (company-filtered,      │
│  2. RAG retrieval │      │   cosine similarity)     │
│  3. Score check   │      └──────────┬──────────────┘
│  4. LLM call      │                 │
└────────┬──────────┘      ┌──────────▼──────────────┐
         │                 │      chroma_db/          │
         │                 │  1,529 chunks across     │
         │                 │  739 KB articles         │
         │                 └─────────────────────────┘
         ▼
  Structured output
  (Response, Product Area, Status, Request Type)
        │
        ▼
  output.csv
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Llama 3.3 70B via [Groq](https://console.groq.com) (free tier) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, free) |
| Vector store | ChromaDB (persistent, cosine similarity) |
| Structured output | OpenAI-compatible tool calling |
| Knowledge base | 739 articles parsed from markdown (HackerRank · Claude · Visa) |

---

## Project Structure

```
.
├── code/
│   ├── main.py               # CLI entry point (batch / interactive / single)
│   ├── agent.py              # Core pipeline: safety → RAG → LLM → structured output
│   ├── retriever.py          # ChromaDB query wrapper
│   ├── indexer.py            # Builds the ChromaDB vector store from parsed JSON
│   ├── parser.py             # Runs all parsers in one shot
│   ├── parser_claude.py      # Parses Claude help center markdown
│   ├── parser_hackerrank.py  # Parses HackerRank articles (handles HTML + YAML quirks)
│   ├── parser_visa.py        # Parses Visa articles (strips Cloudflare email obfuscation)
│   └── parse_visa_support.py # Parses Visa support Q&A pairs
├── data/
│   ├── claude/               # 318 Claude support articles (markdown)
│   ├── hackerrank/           # 394 HackerRank articles (markdown + HTML)
│   └── visa/                 # 12 Visa articles + 15 Q&A pairs (markdown)
├── support_tickets/
│   ├── support_tickets.csv         # 29 evaluation tickets (Issue, Subject, Company)
│   ├── sample_support_tickets.csv  # 10 labelled examples used as few-shot prompts
│   └── output.csv                  # Agent predictions (generated)
├── .env.example              # Copy to .env and add your API key
└── requirements.txt
```

---

## Setup

**1. Clone and create a virtual environment**
```bash
git clone <your-repo-url>
cd hackerrank-orchestrate-may26
python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # macOS / Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Get a free Groq API key at [console.groq.com](https://console.groq.com).
```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

**4. Parse the knowledge base**
```bash
python code/parser.py
```

**5. Build the vector index**
```bash
python code/indexer.py
```

---

## Usage

```bash
# Run all 29 evaluation tickets → support_tickets/output.csv
python code/main.py test

# Test a single ticket via flags
python code/main.py ticket --issue "My Visa card was stolen" --company "Visa"

# Interactive mode — prompts for each field
python code/main.py interactive

# Rebuild the vector DB from scratch
python code/main.py build-db
```

---

## Key Capabilities

### Zero Hallucination
Every response is grounded in the local knowledge base. The agent retrieves the top-3 most relevant KB chunks (cosine similarity, filtered by company) and passes them verbatim as context to the LLM. If no chunk scores above a minimum similarity threshold (`MIN_SCORE = 0.25`), the agent escalates rather than generating an answer from general knowledge. The LLM is instructed to use only the provided context — no invented policies, no fabricated URLs.

### Smart Escalation
Escalation is handled at two layers:

1. **Score-based**: If the best KB chunk similarity is below the threshold for a known company (HackerRank / Claude / Visa), the ticket is automatically escalated — the agent can't give a reliable answer without relevant KB content.
2. **LLM-based**: The system prompt classifies complete platform outages, critical security incidents, and requests beyond agent scope as `status: "Escalated"`, regardless of KB coverage.

Escalated tickets return `"Escalate to a human"` as the response and are flagged with `status = Escalated` in the output CSV for downstream routing.

### Edge-Case Handling
Malicious and irrelevant inputs are caught before any LLM call:

- **Prompt injection**: 14 patterns detected (including multilingual variants) — any attempt to extract system internals, override instructions, or jailbreak is immediately refused.
- **Harmful requests**: 11 patterns covering destructive commands (`rm -rf`, `drop table`, `malware`, `ddos`, etc.) are blocked at the input stage.
- **Out-of-scope requests**: Trivia, social niceties, and unrelated tasks are classified as `request_type: "invalid"` with a polite decline.
- **Foreign-language tickets**: Handled normally; the agent responds in English regardless of input language.

Safety checks short-circuit the pipeline — no API call is made for blocked inputs.

---

## How the RAG Pipeline Works

### Parsing
Each company's markdown corpus is parsed by a dedicated parser that strips frontmatter, title headings, and boilerplate. HackerRank articles with embedded HTML are converted to plain text via BeautifulSoup. Visa articles with Cloudflare-obfuscated emails are cleaned with a regex. Output is normalised JSON with `title`, `content`, `company`, `category`, and `url`.

### Indexing
Articles are split into sentence-aware chunks (max 400 words, 2-sentence overlap — never cuts mid-sentence). Each chunk is prefixed with its article title (if ≤ 15 words) before embedding, so the title boosts retrieval for all chunks of that article. Embeddings are generated locally with `all-MiniLM-L6-v2` and stored in ChromaDB with cosine similarity.

### Retrieval
Queries are embedded with the same model and searched against ChromaDB filtered by company metadata. The top-3 chunks (by cosine similarity) are passed as context to the LLM.

### Generation
A single LLM call uses tool calling to return a guaranteed-valid JSON object with all four output fields. The system prompt includes 10 few-shot examples loaded directly from `sample_support_tickets.csv`. Safety checks (prompt injection, harmful requests) short-circuit before the API call.

---

## Knowledge Base Stats

| Company | Articles | Chunks | Avg chunk |
|---|---|---|---|
| HackerRank | 394 | 828 | 315 words |
| Claude | 318 | 666 | 312 words |
| Visa | 27 | 35 | 205 words |
| **Total** | **739** | **1,529** | — |
