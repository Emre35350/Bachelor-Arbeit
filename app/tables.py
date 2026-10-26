"""Multi-vector table handling (domain customization).

PDF text extraction destroys tables: the column headers, the units row and the
data cells all arrive as separate lines in reading order, so the row/column
association a requirement depends on is simply gone. Embedding that text gives
a vector of scrambled prose, and no amount of retrieval tuning repairs it.

The multi-vector pattern used here separates what is INDEXED from what is READ:

  index   a clean natural-language summary of the table (embeds well, retrieves
          on the words a question actually uses)
  read    the rendered table IMAGE, attached to the final answering call, so the
          model reads exact values off the real table instead of a paraphrase

Detection runs at upload time (cheap, local, no API). Summarization runs once at
index-build time and is cached on the document, so repeated experiment runs
never re-pay for it.

Measured on this corpus (11 DIN/EN standards, 1220 pages): PyMuPDF's default
find_tables() detects 201 of 211 caption-bearing table pages (95%). The default
("lines") strategy is used deliberately — strategy="text" fires on virtually
every page and is useless as a detector.
"""
import base64
import re
from pathlib import Path

from . import config, llm

# --- detection filters, tuned against the real corpus ------------------------
MIN_ROWS = 3          # 2-row "tables" are almost always layout boxes
MIN_COLS = 2
COVER_PAGE = 1        # every DIN cover page has a ruled title block, never a table

SUMMARY_PROMPT = (
    "Describe this table in plain text. Preserve exact values, units, and "
    "row/column labels. Start with one sentence naming what the table specifies, "
    "then list each row as 'row label: column label = value unit'. Do not "
    "summarize away numbers and do not invent values that are not visible."
)

# A real DIN caption is "Tabelle 3 — Title". A prose sentence can also start with
# "Tabelle A.1 enthält …", so the dash form is preferred and the bare form is
# only a fallback.
_CAP_DASH = re.compile(r"^[ \t]*(Tabelle|Table)[\s\xa0]+([A-Z]\.)?\d+(\.\d+)*[\s\xa0]*[-–—].*$", re.M)
_CAP_ANY = re.compile(r"^[ \t]*(Tabelle|Table)[\s\xa0]+([A-Z]\.)?\d+(\.\d+)*\b.*$", re.M)
# the caption NUMBER only, for the offline backfill (matches "Tabelle E.2 — …")
_CAP_NUM = re.compile(r"^[ \t]*(?:Tabelle|Table)[\s\xa0]+((?:[A-Z]\.)?\d+(?:\.\d+)*)[\s\xa0]*[-–—].*$", re.M)
_LEADER = re.compile(r"\.{4,}")   # table-of-contents dot leaders — never a caption


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s)[:48] or "doc"


def image_dir() -> Path:
    p = Path(config.DATA_DIR) / "tables"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()[:120]


def _caption_positions(page) -> list[tuple[float, str]]:
    """Every caption on the page with its vertical position, top to bottom.

    A page can hold several tables (E.1 above E.2); a page-level "first caption"
    then mislabels every table but the first. Positions let each table take the
    caption that sits directly above it.
    """
    out = []
    try:
        data = page.get_text("dict")
    except Exception:
        return out
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            text = _clean("".join(s.get("text", "") for s in line.get("spans", [])))
            if _LEADER.search(text):
                continue
            if _CAP_DASH.match(text) or _CAP_ANY.match(text):
                out.append((float(line["bbox"][1]), text))
    return sorted(out, key=lambda yc: yc[0])


def _caption_for_table(caps: list[tuple[float, str]], bbox) -> str:
    """The caption sitting just above this table's top edge."""
    if not caps:
        return ""
    top = float(bbox[1])
    above = [c for y, c in caps if y <= top + 6]      # small tolerance
    if above:
        return above[-1]                              # nearest above (list is y-sorted)
    return min(caps, key=lambda yc: abs(yc[0] - top))[1]


def _caption_for(page_text: str) -> str:
    m = _CAP_DASH.search(page_text or "") or _CAP_ANY.search(page_text or "")
    return _clean(m.group(0)) if m else ""


def _is_real_table(table, page_no: int) -> bool:
    """Filter the systematic false positives found in the corpus audit."""
    if page_no == COVER_PAGE:
        return False                      # DIN cover-page title block
    try:
        data = table.extract()
    except Exception:
        return False
    if len(data) < MIN_ROWS or not data or len(data[0]) < MIN_COLS:
        return False
    filled = sum(1 for row in data for c in row if (c or "").strip())
    return filled >= MIN_ROWS * MIN_COLS // 2


def _grid_markdown(data: list[list]) -> str:
    """Deterministic fallback rendering of the extracted cells."""
    rows = []
    for row in data[:40]:
        cells = [re.sub(r"\s+", " ", (c or "").strip())[:60] for c in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


# ------------------------------------------------------------------ detection
def detect(doc: dict, pdf) -> list[dict]:
    """Find tables, render each to PNG, return their records (no LLM calls).

    `pdf` is an open fitz.Document. Records are stored on the doc dict and
    persisted with it, so this runs once per upload.
    """
    import fitz  # already a hard dependency via ingestion

    out = []
    for pno, page in enumerate(pdf, start=1):
        try:
            found = page.find_tables().tables
        except Exception:
            continue
        caps = _caption_positions(page)          # positional, so each table gets its own
        for ti, t in enumerate(found):
            if not _is_real_table(t, pno):
                continue
            rect = fitz.Rect(t.bbox) + (-8, -14, 8, 8)   # a little air + the caption
            rect = rect & page.rect
            name = f"{_slug(doc['name'])}-p{pno}-t{ti}.png"
            path = image_dir() / name
            try:
                # 2x zoom: small type in scanned-style DIN tables stays legible
                page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2)).save(path)
            except Exception:
                continue
            data = t.extract()
            out.append({
                "doc": doc["name"], "page": pno, "index": ti,
                "caption": _caption_for_table(caps, t.bbox) or _caption_for(page.get_text("text") or ""),
                "image": name,                     # relative to image_dir()
                "rows": len(data), "cols": len(data[0]) if data else 0,
                "grid": _grid_markdown(data),
                "summary": None,                   # filled at index-build time
            })
    return out


def _ordered_captions(page_text: str) -> list[str]:
    """Distinct captions on a page, in reading order (for the offline backfill)."""
    seen, out = set(), []
    for m in _CAP_NUM.finditer(page_text or ""):
        if _LEADER.search(m.group(0)):
            continue
        num = m.group(1)
        if num not in seen:
            seen.add(num)
            out.append(_clean(m.group(0)))
    return out


def backfill_captions(doc: dict) -> int:
    """Fix mislabeled captions on ALREADY-DETECTED records, using only the stored
    page text — no PDF, no vision call. Earlier detection copied the page's first
    caption onto every table on the page; here the i-th table takes the i-th
    caption in reading order. Only pages where the table count matches the
    caption count are touched, so a split-table page (one caption, several
    pieces) is correctly left alone. Returns the number of captions corrected.
    """
    from collections import defaultdict
    pages = {p["page"]: p["text"] for p in doc["pages"]}
    by_page: dict[int, list] = defaultdict(list)
    for t in index_tables(doc):
        by_page[int(t["page"])].append(t)

    fixed = 0
    for p, ts in by_page.items():
        caps = _ordered_captions(pages.get(p, ""))
        if len(caps) < 2 or len(caps) != len(ts):
            continue                              # unambiguous or a split table
        for t in sorted(ts, key=lambda x: int(x["index"])):
            want = caps[int(t["index"])]
            if t.get("caption") != want:
                t["caption"] = want
                # the old summary was written under the wrong caption -> redo it
                t["summary"] = None
                fixed += 1
    return fixed


def index_tables(doc: dict) -> list[dict]:
    return doc.get("tables") or []


def for_pages(doc: dict, pages) -> list[dict]:
    """Table records whose page is in `pages` (used to attach images at answer time)."""
    want = {int(p) for p in pages or []}
    return [t for t in index_tables(doc) if int(t["page"]) in want]


# -------------------------------------------------------------- summarization
def _data_uri(rec: dict) -> str | None:
    p = image_dir() / rec["image"]
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def image_part(rec: dict) -> dict | None:
    """OpenAI/LangChain-compatible image content part for a table record."""
    uri = _data_uri(rec)
    return {"type": "image_url", "image_url": {"url": uri}} if uri else None


def summarize(rec: dict) -> str:
    """One vision call -> clean prose describing the table.

    Falls back to the deterministic extracted grid if the call fails, so a
    build never dies on a single unreadable table.
    """
    part = image_part(rec)
    if part is None:
        return rec.get("grid") or ""
    head = (f'Table on page {rec["page"]} of "{rec["doc"]}"'
            + (f' — caption: {rec["caption"]}' if rec.get("caption") else "") + ".")
    try:
        resp = llm.chat(
            [{"role": "user", "content": [{"type": "text", "text": f"{head}\n\n{SUMMARY_PROMPT}"},
                                          part]}],
            max_completion_tokens=1200,
            model=config.VISION_MODEL or None,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        text = ""
    return text or rec.get("grid") or ""


def ensure_summaries(doc: dict, on_progress=None) -> int:
    """Summarize every table that has no summary yet. Returns how many were made."""
    made = 0
    recs = index_tables(doc)
    for i, rec in enumerate(recs):
        if rec.get("summary"):
            continue
        rec["summary"] = summarize(rec)
        made += 1
        if on_progress:
            on_progress(i + 1, len(recs))
    return made


def chunk_text(rec: dict) -> str:
    """What actually gets embedded: caption + prose summary (never the raw grid)."""
    head = rec.get("caption") or f"Tabelle auf Seite {rec['page']}"
    return f"{head}\n{rec.get('summary') or rec.get('grid') or ''}".strip()
