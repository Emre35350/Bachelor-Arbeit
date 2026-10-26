"""Clause-number index for standards documents (domain customization, Step 9).

DIN/EN standards are organized by numbered clauses (7.4.1, Annex B, …). This
lexically indexes those headings so pipeline B gets a structure-native tool set
(`list_clauses`, `read_clause`) for this document type — a domain-adapted
capability, not a generic reader. Purely lexical (regex over page text): no
embeddings, so paradigm isolation holds.

The index is computed lazily and cached on the doc dict (persists with it).
"""
import re

# PDF text extraction of these standards puts the clause number and its title
# either on ONE line ("5.1 Anforderungen an die Werkstoffe") or — very often —
# on TWO lines ("5 " / "Anforderungen, Eigenschaften und Prüfverfahren").
# Both forms must be recognised, or the real headings are missed while table
# rows like "5 mm < h ≤ 50 mm" get mistaken for clauses.
_NUM_ONLY = re.compile(r"^[ \t]*(\d+(?:\.\d+){0,4})[ \t]*$")
_NUM_TITLE = re.compile(r"^[ \t]*(\d+(?:\.\d+){0,4})[ \t]+(.{1,90})$")
_ANNEX_LINE = re.compile(r"^[ \t]*(Annex|Anhang)[ \t]+([A-Z])\b[ \t]*(.{0,80})$")

# A heading title is a capitalised phrase — not a measurement, formula or
# table cell. German section titles are capitalised nouns, so this is reliable.
_TITLE_START = re.compile(r"^[A-ZÄÖÜ(]")
_NOT_TITLE = re.compile(
    r"[<>=≤≥]|^\d|\bmm\b|\bcm\b|\bkg\b|\bN/mm|\bkN\b|\bMPa\b|%"
    r"|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}"      # print timestamps / dates
    r"|Uhrzeit|Ausdruck|Normen-Download|Alle Rechte")   # DIN print watermarks
# Table-of-contents lines ("Begriffe ........... 12") must not become clauses:
# the index must point at the clause BODY, not at the contents page.
_TOC_LEADER = re.compile(r"\.{4,}|·{4,}|(?:\. ){4,}")


def _looks_like_heading(title: str) -> bool:
    t = (title or "").strip()
    if len(t) < 3 or not _TITLE_START.match(t):
        return False
    if t.endswith("-"):                 # hyphenated line-break fragment
        return False
    if _TOC_LEADER.search(t):           # table-of-contents entry
        return False
    if _NOT_TITLE.search(t):            # measurement / formula / table row
        return False
    if t.endswith(".") and len(t) > 40:  # a sentence, not a heading
        return False
    return True


# --------------------------------------------------------------- TOC-derived
# When the PDF ships an outline, it is authoritative: it carries the clause
# number, its title and its true page. Lexical scanning of body text can never
# match that (numbered list items and table rows look identical to headings), so
# the outline is preferred and the scan is only a fallback.
_TOC_NUM = re.compile(
    r"^(\d+(?:\.\d+){0,4}|(?:Annex|Anhang)\s+[A-Z])\s+(.{1,90})$")


def _find_offset(text: str, clause: str, title: str) -> int:
    """Locate the heading inside its page so read() can slice precisely."""
    for probe in (f"{clause} {title[:30]}", title[:30], clause):
        if not probe.strip():
            continue
        i = text.find(probe)
        if i != -1:
            return i
    return 0


def _from_toc(doc: dict) -> list[dict]:
    out, n = [], len(doc["pages"])
    for entry in (doc.get("toc") or []):
        try:
            title_raw, page = entry[1], int(entry[2])
        except (IndexError, TypeError, ValueError):
            continue
        title = re.sub(r"[\s ]+", " ", str(title_raw)).strip()
        m = _TOC_NUM.match(title)
        if not m:
            continue
        clause, rest = m.group(1), m.group(2).strip()
        page = max(1, min(n, page))
        out.append({"clause": clause, "title": rest[:90], "page": page,
                    "offset": _find_offset(doc["pages"][page - 1]["text"], clause, rest)})
    out.sort(key=lambda c: (c["page"], c["offset"]))
    return out


def build(doc: dict) -> list[dict]:
    """Ordered clause index: [{clause, title, page, offset}] in reading order.

    Uses the PDF outline when it carries clause numbers; otherwise falls back to
    lexically scanning the page text.
    """
    from_toc = _from_toc(doc)
    if len(from_toc) >= 5:
        return from_toc
    return _build_lexical(doc["pages"])


def _build_lexical(pages: list[dict]) -> list[dict]:
    """Fallback for documents with no (usable) PDF outline."""
    out = []
    for p in pages:
        text = p["text"] or ""
        lines, offset = text.splitlines(keepends=True), 0
        stripped = [l.rstrip("\n") for l in lines]
        # A printed page number sits alone in the header/footer and looks exactly
        # like a bare clause number. Require BOTH signals before discarding one:
        # it must sit in the header/footer band AND its value must be close to
        # this page's number — so a real "5 Anforderungen" heading survives even
        # when it happens to fall on page 5.
        filled = [i for i, l in enumerate(stripped) if l.strip()]
        edge = set(filled[:2] + filled[-2:])
        for i, raw in enumerate(lines):
            line = stripped[i]

            m = _ANNEX_LINE.match(line)
            if m:
                word, letter, rest = m.group(1), m.group(2), m.group(3).strip()
                title = rest or _next_title(stripped, i) or f"{word} {letter}"
                out.append({"clause": f"{word} {letter}", "title": title[:90],
                            "page": p["page"], "offset": offset})
                offset += len(raw)
                continue

            m = _NUM_ONLY.match(line)
            if m and not _is_page_number(m.group(1), i in edge, p["page"]):
                title = _next_title(stripped, i)    # number alone -> title follows
                if title:
                    out.append({"clause": m.group(1), "title": title[:90],
                                "page": p["page"], "offset": offset})
                offset += len(raw)
                continue

            m = _NUM_TITLE.match(line)
            if m and _looks_like_heading(m.group(2)):
                out.append({"clause": m.group(1), "title": m.group(2).strip()[:90],
                            "page": p["page"], "offset": offset})
            offset += len(raw)
    return out


def _is_page_number(token: str, in_edge_band: bool, page: int) -> bool:
    """A bare number in the header/footer whose value tracks the page number."""
    if not in_edge_band or "." in token:
        return False
    try:
        return abs(int(token) - page) <= 3
    except ValueError:
        return False


def _next_title(lines: list[str], i: int, lookahead: int = 2) -> str | None:
    """Title on one of the next non-empty lines (number-on-its-own-line form)."""
    for j in range(i + 1, min(i + 1 + lookahead, len(lines))):
        cand = lines[j].strip()
        if not cand:
            continue
        return cand if _looks_like_heading(cand) else None
    return None


def index(doc: dict) -> list[dict]:
    """Return the cached clause index, building it on first use."""
    if doc.get("clauses") is None:
        doc["clauses"] = build(doc)
    return doc["clauses"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower().rstrip(".")


def find(doc: dict, clause: str) -> dict | None:
    """Locate a clause heading.

    Numbered list items look exactly like clause headings ("7 Vermeiden der
    Ansammlung von Wasser"), so when a number matches several places we pick the
    one consistent with its own sub-clauses: the real clause 7 is the occurrence
    sitting just before 7.1. Falls back to the first match.
    """
    idx = index(doc)
    q = _norm(clause)
    exact = [(i, c) for i, c in enumerate(idx) if _norm(c["clause"]) == q]

    if len(exact) > 1:
        # Numbered list items and stray lines look just like clause headings.
        # The distinguishing signal is what FOLLOWS: the real clause 7 is the one
        # directly followed by 7.1, and it owns the longest run of sub-clauses.
        def run(i: int) -> int:
            return sum(1 for w in idx[i + 1:i + 16]
                       if _norm(w["clause"]).startswith(q + "."))

        immediate = [(i, c) for i, c in exact
                     if i + 1 < len(idx)
                     and _norm(idx[i + 1]["clause"]).startswith(q + ".")]
        pool = immediate or [ic for ic in exact if run(ic[0]) > 0]
        if pool:
            return max(pool, key=lambda ic: (run(ic[0]), -ic[0]))[1]
    if exact:
        return exact[0][1]

    for c in idx:                       # then prefix (e.g. "7.4" matches "7.4.1")
        if _norm(c["clause"]).startswith(q):
            return c
    return None


def outline(doc: dict, limit: int = 60) -> str:
    idx = index(doc)
    if not idx:
        return "(no numbered clauses detected in this document)"
    rows = [f"{c['clause']} · {c['title']} (p.{c['page']})" for c in idx[:limit]]
    more = f"\n… and {len(idx) - limit} more" if len(idx) > limit else ""
    return "\n".join(rows) + more


def read(doc: dict, clause: str, char_cap: int = 8000) -> tuple[str, list[int]]:
    """Read one clause: from its heading to the next clause heading.

    Returns (text, pages_covered). pages_covered feeds the evidence 'read' set.
    """
    idx = index(doc)
    hit = find(doc, clause)
    if not hit:
        return (f'Clause "{clause}" not found. Call list_clauses to see available ones.', [])

    pos = idx.index(hit)
    nxt = idx[pos + 1] if pos + 1 < len(idx) else None
    pages = doc["pages"]
    start_pg, start_off = hit["page"], hit["offset"]
    end_pg = nxt["page"] if nxt else pages[-1]["page"]
    end_off = nxt["offset"] if nxt else None

    chunks, covered, total = [], [], 0
    for p in pages:
        if p["page"] < start_pg or p["page"] > end_pg:
            continue
        text = p["text"] or ""
        s = start_off if p["page"] == start_pg else 0
        e = end_off if (nxt and p["page"] == end_pg) else len(text)
        seg = text[s:e].strip()
        if not seg:
            continue
        header = f"--- {doc['name']} · p.{p['page']} ---\n"
        piece = header + seg + "\n"
        if total + len(piece) > char_cap:
            chunks.append("...(clause truncated — read the next pages directly)")
            break
        chunks.append(piece)
        covered.append(p["page"])
        total += len(piece)

    head = f"[Clause {hit['clause']} · {hit['title']} · from p.{start_pg}]\n"
    return head + ("\n".join(chunks) if chunks else "(empty clause)"), covered
