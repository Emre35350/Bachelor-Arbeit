"""Evaluation question set — the labelled ground truth for Chapter 4 metrics.

A question set is a JSON file (QUESTIONS_PATH, default questions.json):

  {
    "questions": [
      {
        "id": "q1",
        "question": "What is the minimum coating thickness?",
        "question_de": "...",          # optional bilingual variant (Step 9)
        "type": "factoid",             # free-form label: factoid|multi-hop|comparison|...
        "content_type": "table",       # optional: prose|table — where the answer lives
        "gold_pages": {"EN 14509.pdf": [42, 43]},   # doc -> the pages that truly answer it
        "notes": ""
      }
    ]
  }

gold_pages are compared against each pipeline's evidence set to compute
recall / precision / MRR / steps-to-evidence. Everything here is plain JSON and
hand-editable; the UI can also read and write it.
"""
import json
import re
from pathlib import Path

from . import config

QUESTION_TYPES = ["factoid", "multi-hop", "comparison", "numeric",
                  "definition", "cross-document", "unanswerable", "other"]


def _path() -> Path:
    return Path(config.QUESTIONS_PATH)


def _slugify_id(existing: set, text: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (text or "q").lower()).strip("-")[:24] or "q"
    cand, i = base, 2
    while cand in existing:
        cand, i = f"{base}-{i}", i + 1
    return cand


def _clean(q: dict, existing_ids: set) -> dict:
    """Validate/normalise one question entry; raises ValueError on bad shape."""
    question = str(q.get("question", "")).strip()
    if not question:
        raise ValueError("each question needs a non-empty 'question'")
    qid = str(q.get("id", "")).strip() or _slugify_id(existing_ids, question)

    gold = q.get("gold_pages", {}) or {}
    if not isinstance(gold, dict):
        raise ValueError(f"{qid}: gold_pages must be an object of doc -> [pages]")
    clean_gold = {}
    for doc, pages in gold.items():
        if not isinstance(pages, list):
            raise ValueError(f"{qid}: gold_pages['{doc}'] must be a list of page numbers")
        nums = sorted({int(p) for p in pages if int(p) > 0})
        if nums:
            clean_gold[str(doc)] = nums

    entry = {
        "id": qid,
        "question": question,
        "type": str(q.get("type", "other")).strip() or "other",
        "gold_pages": clean_gold,
        "notes": str(q.get("notes", "")).strip(),
    }
    # optional fields kept verbatim when present. content_type (prose|table) is
    # the retrieval-content dimension used to break metrics down by whether the
    # answer lives in running text or a table — do NOT drop it on import.
    for opt in ("question_de", "content_type"):
        if q.get(opt):
            entry[opt] = str(q[opt]).strip()
    return entry


def load() -> list[dict]:
    p = _path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    items = data.get("questions", data) if isinstance(data, dict) else data
    out, seen = [], set()
    for q in items if isinstance(items, list) else []:
        if not isinstance(q, dict):
            continue
        try:
            e = _clean(q, seen)
        except ValueError:
            continue  # skip malformed rows on read; save() is strict
        seen.add(e["id"])
        out.append(e)
    return out


def save(questions: list[dict]) -> list[dict]:
    """Validate the whole set and write it. Returns the normalised list."""
    out, seen = [], set()
    for q in questions or []:
        e = _clean(q, seen)
        if e["id"] in seen:
            raise ValueError(f"duplicate question id: {e['id']}")
        seen.add(e["id"])
        out.append(e)
    tmp = _path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"questions": out}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(_path())
    return out


def get(qid: str) -> dict | None:
    for q in load():
        if q["id"] == qid:
            return q
    return None
