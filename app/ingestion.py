"""Shared ingestion layer — PyMuPDF (fitz) parsing into a page-level document store.

Both pipelines consume the SAME parsed pages; the only experimental variable
downstream is the retrieval strategy.
"""
import fitz  # PyMuPDF

from . import config

# name -> {"name", "pages": [{"page", "text"}], "toc": [[level, title, page]...],
#          "tree": PageIndex-style node dict | None, "tree_nodes": int,
#          "tables": [table record, ...]}
DOCS: dict[str, dict] = {}


def add_pdf(name: str, data: bytes) -> None:
    pdf = fitz.open(stream=data, filetype="pdf")
    pages = [{"page": i + 1, "text": (page.get_text("text") or "").strip()}
             for i, page in enumerate(pdf)]
    if not any(p["text"] for p in pages):
        raise ValueError("no extractable text layer (scanned PDF? OCR is not supported)")
    doc = {"name": name, "pages": pages,
           "toc": pdf.get_toc(simple=True) or [], "tree": None, "tree_nodes": 0,
           "tables": []}
    if config.TABLE_SUMMARIES:
        # detection + PNG rendering only — local and cheap. The vision summaries
        # are made once at index-build time, where there is progress reporting.
        from . import tables
        doc["tables"] = tables.detect(doc, pdf)
    DOCS[name] = doc


def add_text(name: str, text: str) -> None:
    text = text.strip()
    if not text:
        raise ValueError("file is empty")
    DOCS[name] = {"name": name, "pages": [{"page": 1, "text": text}],
                  "toc": [], "tree": None, "tree_nodes": 0, "tables": []}


def clear() -> None:
    DOCS.clear()


def get(name: str) -> dict | None:
    """Fuzzy lookup: exact -> case-insensitive -> substring."""
    if name in DOCS:
        return DOCS[name]
    low = (name or "").lower()
    for d in DOCS.values():
        if d["name"].lower() == low:
            return d
    for d in DOCS.values():
        if low and low in d["name"].lower():
            return d
    return None


def list_documents() -> list[dict]:
    return [{"name": d["name"], "pages": len(d["pages"]),
             "has_toc": bool(d["toc"]), "tree_nodes": d.get("tree_nodes", 0),
             "tables": len(d.get("tables") or [])}
            for d in DOCS.values()]


def collect_page_range(doc: dict, start: int, end: int,
                       char_cap: int = 8000) -> tuple[str, list[int]]:
    """Concatenate a page range with per-page headers, capped for context safety.

    Returns (text, included_pages) — included_pages are the pages whose FULL text
    actually made it under the char cap, i.e. what truly entered the agent's
    context. This is the ground truth for "pages read" in the evidence set.
    """
    start = max(1, int(start))
    end = min(len(doc["pages"]), int(end))
    if end < start:
        start, end = end, start
    out, total, included = [], 0, []
    for p in doc["pages"][start - 1:end]:
        seg = f"--- {doc['name']} · p.{p['page']} ---\n{p['text']}\n"
        if total + len(seg) > char_cap:
            out.append("...(length cap reached — request a narrower range)")
            break
        out.append(seg)
        total += len(seg)
        included.append(p["page"])
    return ("\n".join(out) if out else "(empty range)"), included


def page_range_text(doc: dict, start: int, end: int, char_cap: int = 8000) -> str:
    return collect_page_range(doc, start, end, char_cap)[0]
