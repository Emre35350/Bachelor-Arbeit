"""Pipeline B index — PageIndex-style hierarchical document tree.

Inspired by VectifyAI/PageIndex (https://github.com/VectifyAI/PageIndex):
"vectorless, reasoning-based RAG" — each document becomes a tree of nodes
(title, one-line summary, page range, children). The agent retrieves by
REASONING over this map instead of by embedding similarity.

Tree construction strategy (cheapest sufficient source of structure first):
  1. If the PDF ships a table of contents (outline), use it as the skeleton
     and fill one-line node summaries with a single LLM call.
  2. Otherwise ask the LLM to segment the document into a two-level tree
     from per-page previews (PageIndex's LLM-structuring approach).
  3. If no LLM key / structuring fails, fall back to fixed page windows,
     so the system stays functional.
"""
import json
import re

from . import config, ingestion, llm, progress

# The structuring pass must see the WHOLE document — capping it at the first N
# pages left the tail of long standards unstructured (one vague trailing node).
# Instead of a hard page cap we spend a fixed character budget across all pages,
# shrinking the per-page preview as the document grows.
MAX_STRUCTURE_PAGES = 400      # beyond this, pages are sampled evenly (not truncated)
PREVIEW_BUDGET_CHARS = 60000   # ~15-20k tokens of page previews
PAGE_PREVIEW_MAX = 300         # per-page preview chars (short documents)
PAGE_PREVIEW_MIN = 110         # floor for very long documents
TARGET_PAGES_PER_LEAF = 12     # drives how finely the LLM is asked to segment
MAX_SUMMARY_NODES = 60         # nodes summarised in the single summary call
NODE_READ_CAP = 8000           # chars returned by read_node


# ------------------------------------------------------------------ building
def build_all_trees(only: list[str] | None = None, force: bool = False) -> dict:
    """Build trees for documents that do not have one yet.

    Structuring a document costs an LLM call, and a document's tree depends only
    on the document — not on chunk size, embedding model or top-k. Rebuilding all
    eleven trees because a Pipeline-A parameter changed was pure waste, so a
    document with a tree is skipped unless `force` is set. `only` restricts the
    build to named documents (adding an 11th document structures just that one).
    """
    out = {}
    docs = [d for d in ingestion.DOCS.values() if not only or d["name"] in set(only)]
    targets = [d for d in docs if force or not d.get("tree")]
    reused = [d["name"] for d in docs if d not in targets]
    total = len(targets)
    if not total:
        progress.set_phase("trees", 1, f"All {len(reused)} tree(s) already built — reused.")
        return {"built": {}, "reused": reused}

    for i, d in enumerate(targets, start=1):
        progress.set_phase("trees", (i - 1) / total, f"Building tree {i}/{total}: {d['name']}…")
        d["tree"] = build_tree(d)
        d["tree_nodes"] = count_nodes(d["tree"])
        out[d["name"]] = {"nodes": d["tree_nodes"],
                          "source": d["tree"].get("source", "?")}
        progress.set_phase("trees", i / total, f"Built tree {i}/{total}: {d['name']}")
    return {"built": out, "reused": reused}


def build_tree(doc: dict) -> dict:
    """Pick the finest structure the document itself provides.

    Neither source wins everywhere: the PDF outline is richer where it exists
    and is complete (it lists unnumbered front matter too), but it is unusable in
    documents like EN 14509, which ships three entries for 174 pages. The clause
    index is authoritative there, yet coarser where the outline is good, because
    it only keeps numbered headings. So both are built and the one that actually
    navigates better — finer leaves — is kept. Both are deterministic and free;
    the LLM is only asked when the document offers no structure at all.
    """
    from . import clauses
    n = len(doc["pages"])
    candidates = []
    if len(clauses.index(doc)) >= MIN_CLAUSES_FOR_TREE:
        candidates.append(_tree_from_clauses(doc))
    if doc["toc"]:
        candidates.append(_tree_from_toc(doc))

    if candidates:
        # finer leaves first; on a tie prefer the tree with more nodes
        def rank(t):
            m = _tree_metrics(t, n)
            return (m["avg_leaf"], -m["nodes"])
        tree = min(candidates, key=rank)
    elif config.OPENAI_API_KEY:
        try:
            tree = _tree_from_llm(doc)
        except Exception:
            tree = _fallback_tree(doc)
    else:
        tree = _fallback_tree(doc)

    _assign_ids(tree)
    if config.OPENAI_API_KEY:
        try:
            _fill_summaries(doc, tree)
        except Exception:
            pass  # summaries are best-effort; titles alone still navigate
    return tree


def _extract_json(raw: str) -> dict:
    """Parse a JSON object out of an LLM reply (tolerates fences / stray prose)."""
    s = re.sub(r"^```(json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start, depth = s.find("{"), 0          # fall back to the outermost {...} block
    if start == -1:
        raise ValueError("no JSON object in the model reply")
    for i, ch in enumerate(s[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            return json.loads(s[start:i + 1])
    raise ValueError("JSON object was truncated (raise the token budget)")


def _llm_json(prompt: str, retries: int = 1, max_tokens: int = 8000) -> dict:
    """LLM call that must return JSON, with a retry — the structuring pass is
    long and gpt-5.x reasoning tokens eat into the completion budget, so a
    truncated/malformed reply is a real (intermittent) failure mode."""
    last = None
    for attempt in range(retries + 1):
        try:
            resp = llm.chat([{"role": "user", "content": prompt}],
                            max_completion_tokens=max_tokens)
            return _extract_json(resp.choices[0].message.content or "")
        except (json.JSONDecodeError, ValueError) as e:
            last = e
            prompt += ("\n\nYour previous reply was not valid JSON. Reply with ONLY "
                       "the JSON object, nothing else, and keep it compact.")
    raise ValueError(f"LLM did not return valid JSON after {retries + 1} attempts: {last}")


def _node(title, start, end, children=None, summary=""):
    return {"title": title, "summary": summary, "start_page": start,
            "end_page": end, "children": children or []}


MIN_CLAUSES_FOR_TREE = 8   # below this the clause index is too thin to navigate


def _clause_depth(clause: str) -> int:
    """'5' -> 1, '5.1' -> 2, '7.4.1' -> 3; annexes sit at the top level."""
    c = (clause or "").strip()
    if re.match(r"^(Annex|Anhang)\b", c, re.I):
        return 1
    return len([p for p in c.split(".") if p])


def _tree_from_clauses(doc: dict) -> dict:
    """Clause index -> tree. The document's OWN numbering as the navigation map.

    Standards already carry an authoritative hierarchy (5 → 5.1 → 5.1.1). Using
    it beats both alternatives: a PDF outline is often unusable (EN 14509 ships
    three entries for 174 pages) and LLM structuring only guesses at boundaries,
    which produced 16-page leaves for that document. This is deterministic, free,
    and purely lexical — no embeddings, so the paradigm split still holds. It
    also gives pipelines A and B the same structural view of the document.

    `clauses.index()` already prefers the PDF outline when that outline is good,
    so this is strictly better-informed than building from the outline directly.
    """
    from . import clauses  # local import keeps the module import graph flat
    n = len(doc["pages"])
    idx = clauses.index(doc)

    root = _node(doc["name"], 1, n)
    root["source"] = "clauses"
    stack = [(0, root)]
    for c in idx:
        page = max(1, min(n, int(c["page"])))
        depth = _clause_depth(c["clause"])
        node = _node(f"{c['clause']} {c['title']}".strip()[:120], page, n)
        while stack and stack[-1][0] >= depth:
            stack.pop()
        (stack[-1][1] if stack else root)["children"].append(node)
        stack.append((depth, node))
    _fix_end_pages(root, n)
    return root


def _tree_from_toc(doc: dict) -> dict:
    """PDF outline -> tree. End pages are inferred from the next entry."""
    n = len(doc["pages"])
    entries = [(max(1, lvl), title.strip() or "(untitled)", max(1, min(n, page)))
               for lvl, title, page in doc["toc"]]
    root = _node(doc["name"], 1, n)
    root["source"] = "pdf-toc"
    stack = [(0, root)]
    for lvl, title, page in entries:
        node = _node(title, page, n)
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        (stack[-1][1] if stack else root)["children"].append(node)
        stack.append((lvl, node))
    _fix_end_pages(root, n)
    return root


def _fix_end_pages(node: dict, doc_end: int) -> None:
    kids = node["children"]
    for i, kid in enumerate(kids):
        nxt = kids[i + 1]["start_page"] - 1 if i + 1 < len(kids) else node["end_page"]
        kid["end_page"] = max(kid["start_page"], min(nxt, doc_end))
        _fix_end_pages(kid, doc_end)


def _structure_previews(doc: dict) -> str:
    """Per-page previews covering the WHOLE document within a char budget.

    Long standards used to be truncated to the first 100 pages, which left their
    tail as one vague trailing node. Here every page is represented; the per-page
    preview shrinks as the document grows, and only documents beyond
    MAX_STRUCTURE_PAGES are sampled (evenly, so the tail is still seen).
    """
    pages = [p for p in doc["pages"] if p["text"]]
    if len(pages) > MAX_STRUCTURE_PAGES:
        step = len(pages) / MAX_STRUCTURE_PAGES
        pages = [pages[int(i * step)] for i in range(MAX_STRUCTURE_PAGES)]
    per_page = PAGE_PREVIEW_MAX
    if pages:
        per_page = max(PAGE_PREVIEW_MIN,
                       min(PAGE_PREVIEW_MAX, PREVIEW_BUDGET_CHARS // len(pages)))
    return "\n".join(f"[page {p['page']}] {p['text'][:per_page]}" for p in pages)


def _tree_from_llm(doc: dict) -> dict:
    """PageIndex-style LLM structuring from per-page previews."""
    previews = _structure_previews(doc)
    n = len(doc["pages"])
    prompt = (
        "You segment documents into a hierarchical table of contents.\n"
        f"Document: \"{doc['name']}\" ({n} pages). Page previews:\n\n{previews}\n\n"
        "Return ONLY JSON (no markdown fences) with this exact shape:\n"
        '{"children":[{"title":"...","summary":"one line","start_page":1,'
        '"end_page":3,"children":[...]}]}\n'
        f"Rules: at most 2 levels below the root. Segment the WHOLE document "
        f"(pages 1-{n}), including annexes at the end — the last section must reach "
        f"page {n}. No leaf section may span more than {max(6, n // TARGET_PAGES_PER_LEAF)} "
        f"pages: split long parts (e.g. a big annex) into sub-sections so that a leaf "
        f"is a readable unit. Aim for roughly {max(4, n // TARGET_PAGES_PER_LEAF)} leaf "
        "sections. Page ranges must be within bounds, non-overlapping and in reading "
        "order. Keep titles short; use the document's own language."
    )
    data = _llm_json(prompt, retries=1)

    root = _node(doc["name"], 1, n, children=[
        _clamp(child, n) for child in data.get("children", []) if isinstance(child, dict)
    ])
    if not root["children"]:
        raise ValueError("LLM returned no sections")
    root["source"] = "llm-structured"
    return root


def _clamp(raw: dict, n: int) -> dict:
    start = max(1, min(n, int(raw.get("start_page", 1))))
    end = max(start, min(n, int(raw.get("end_page", start))))
    return _node(str(raw.get("title", "(untitled)"))[:120], start, end,
                 children=[_clamp(c, n) for c in raw.get("children", [])
                           if isinstance(c, dict)][:30],
                 summary=str(raw.get("summary", ""))[:200])


def _fallback_tree(doc: dict, window: int = 5) -> dict:
    n = len(doc["pages"])
    root = _node(doc["name"], 1, n)
    root["source"] = "page-windows"
    for s in range(1, n + 1, window):
        e = min(n, s + window - 1)
        root["children"].append(_node(f"Pages {s}-{e}", s, e))
    return root


def _assign_ids(root: dict) -> None:
    counter = [0]

    def walk(node):
        counter[0] += 1
        node["id"] = f"n{counter[0]}"
        for c in node["children"]:
            walk(c)
    walk(root)


def _fill_summaries(doc: dict, root: dict) -> None:
    """One batched LLM call writes one-line summaries for nodes lacking one."""
    todo = [nd for nd in _flatten(root) if not nd["summary"]][:MAX_SUMMARY_NODES]
    if not todo:
        return
    samples = "\n\n".join(
        f"### {nd['id']} · {nd['title']} (p.{nd['start_page']}-{nd['end_page']})\n"
        f"{doc['pages'][nd['start_page'] - 1]['text'][:PAGE_PREVIEW_MAX]}"
        for nd in todo)
    prompt = ("For each section below, write a one-line summary (max 18 words) of what "
              "it contains. Return ONLY JSON, no fences:\n"
              '{"summaries": {"<id>": "summary", ...}}\n\n' + samples)
    summaries = _llm_json(prompt, retries=0, max_tokens=4000).get("summaries", {})
    for nd in todo:
        if nd["id"] in summaries:
            nd["summary"] = str(summaries[nd["id"]])[:200]


# ---------------------------------------------------------------- navigation
def _flatten(root: dict) -> list[dict]:
    out = []

    def walk(node):
        out.append(node)
        for c in node["children"]:
            walk(c)
    walk(root)
    return out


def count_nodes(root: dict | None) -> int:
    return len(_flatten(root)) if root else 0


def render_tree(doc: dict) -> str:
    lines = []

    def walk(node, depth):
        pad = "  " * depth
        summ = f" — {node['summary']}" if node["summary"] else ""
        lines.append(f"{pad}{node['id']} · {node['title']} "
                     f"(p.{node['start_page']}-{node['end_page']}){summ}")
        for c in node["children"]:
            walk(c, depth + 1)
    walk(doc["tree"], 0)
    return "\n".join(lines)


def find_node(doc: dict, node_id: str) -> dict | None:
    for nd in _flatten(doc["tree"]):
        if nd["id"] == node_id.strip():
            return nd
    return None


def read_node(doc: dict, node_id: str) -> str:
    nd = find_node(doc, node_id)
    if not nd:
        return f"Node not found: {node_id}. Call get_tree first."
    header = f"[{nd['id']} · {nd['title']} · p.{nd['start_page']}-{nd['end_page']}]\n"
    return header + ingestion.page_range_text(doc, nd["start_page"], nd["end_page"],
                                              char_cap=NODE_READ_CAP)


# ---------------------------------------------------------------- tree audit
DEGENERATE_AVG_LEAF = 15    # avg pages per leaf above this = too coarse to navigate


def _depth(node: dict) -> int:
    return 1 + max((_depth(c) for c in node["children"]), default=0)


def _tree_metrics(tree: dict, pages: int) -> dict:
    """Navigability of a tree — shared by the audit and by source selection."""
    nodes = _flatten(tree)
    leaves = [n for n in nodes if not n["children"]]
    spans = [n["end_page"] - n["start_page"] + 1 for n in leaves] or [pages]
    return {"nodes": len(nodes), "leaves": len(leaves),
            "avg_leaf": sum(spans) / len(spans), "max_leaf": max(spans)}


def audit_tree(doc: dict) -> dict:
    """Quality report for one document's tree — flags degenerate structure.

    A tree with a handful of nodes spanning a large document (e.g. 174 pages /
    4 nodes) can't discriminate evidence: read_node then returns huge, mostly
    irrelevant ranges. This surfaces that so it can be re-structured.
    """
    tree = doc.get("tree")
    pages = len(doc["pages"])
    if not tree:
        return {"name": doc["name"], "built": False, "pages": pages}

    m = _tree_metrics(tree, pages)
    nodes, leaves = m["nodes"], m["leaves"]
    avg_leaf, max_leaf = m["avg_leaf"], m["max_leaf"]

    reasons = []
    if avg_leaf > DEGENERATE_AVG_LEAF:
        reasons.append(f"coarse leaves (avg {avg_leaf:.0f} pages/leaf)")
    if max_leaf > max(12, pages * 0.5):
        reasons.append(f"one node spans {max_leaf} pages")
    if nodes <= 3 and pages > 15:
        reasons.append(f"only {nodes} nodes for {pages} pages")

    return {
        "name": doc["name"], "built": True,
        "source": tree.get("source", "?"),
        "nodes": nodes, "leaves": leaves, "pages": pages,
        "avg_pages_per_leaf": round(avg_leaf, 1),
        "max_leaf_span": max_leaf, "depth": _depth(tree),
        "degenerate": bool(reasons), "reasons": reasons,
    }


def audit_all() -> list[dict]:
    return [audit_tree(d) for d in ingestion.DOCS.values()]


def rebuild_tree(doc: dict, method: str = "llm") -> dict:
    """Force-rebuild one document's tree with a chosen method (default LLM
    structuring, which usually fixes a degenerate TOC/window tree)."""
    if method == "llm":
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY required for LLM structuring")
        tree = _tree_from_llm(doc)          # may raise -> surfaced to the caller
    elif method == "clauses":
        from . import clauses
        found = len(clauses.index(doc))
        if found < MIN_CLAUSES_FOR_TREE:
            raise RuntimeError(f"only {found} clause headings detected — too few "
                               f"to build a tree from")
        tree = _tree_from_clauses(doc)
    elif method == "toc":
        if not doc["toc"]:
            raise RuntimeError("document has no PDF outline/TOC")
        tree = _tree_from_toc(doc)
    elif method == "windows":
        tree = _fallback_tree(doc)
    else:  # "auto" -> the default preference order
        tree = build_tree(doc)
        doc["tree"], doc["tree_nodes"] = tree, count_nodes(tree)
        return audit_tree(doc)

    _assign_ids(tree)
    if config.OPENAI_API_KEY:
        try:
            _fill_summaries(doc, tree)
        except Exception:
            pass
    doc["tree"], doc["tree_nodes"] = tree, count_nodes(tree)
    return audit_tree(doc)
