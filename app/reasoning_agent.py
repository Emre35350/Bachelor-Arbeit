"""Pipeline B — Reasoning-Based Retrieval agent.

Architecture: an EXPLICIT LangGraph StateGraph (not the prebuilt black box):

        START ──> [agent] ──(tool_calls?)──> [tools] ──> [agent] ──> ... ──> END

The agent navigates PageIndex-style trees. Its tools are deliberately
lexical/structural only (no embeddings), so this pipeline can never touch
Pipeline A's vector index — that isolation is what makes the two paradigms
comparable in the experiment.
"""
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from . import clauses, config, ingestion, llm, pageindex, tables


# ----------------------------------------------------------------- the tools
@tool
def list_documents() -> str:
    """List all loaded documents with page counts and how their tree was built. Call this first."""
    if not ingestion.DOCS:
        return "No documents loaded."
    rows = []
    for d in ingestion.DOCS.values():
        src = d["tree"].get("source", "?") if d["tree"] else "no tree"
        rows.append(f'- "{d["name"]}" — {len(d["pages"])} pages, '
                    f'{d.get("tree_nodes", 0)} tree nodes ({src}). '
                    f'Begins: "{d["pages"][0]["text"][:180]}..."')
    return "\n".join(rows)


@tool
def get_tree(doc: str) -> str:
    """Return the hierarchical PageIndex-style tree of one document: node ids, titles, page ranges and one-line summaries. Reason over this map to decide which node to open."""
    d = ingestion.get(doc)
    if not d:
        return f'Document not found: "{doc}". Call list_documents first.'
    if not d["tree"]:
        return "Tree not built yet — build the indexes first."
    return pageindex.render_tree(d)


@tool
def read_node(doc: str, node_id: str) -> str:
    """Read the full text of one tree node (its whole page range). Prefer the smallest node that plausibly contains the answer."""
    d = ingestion.get(doc)
    if not d:
        return f'Document not found: "{doc}".'
    return pageindex.read_node(d, node_id)


@tool
def read_pages(doc: str, start_page: int, end_page: int) -> str:
    """Read a raw page range (max 5 pages per call) when tree nodes are too coarse."""
    d = ingestion.get(doc)
    if not d:
        return f'Document not found: "{doc}".'
    start_page = int(start_page)
    end_page = min(int(end_page), start_page + 4)  # cap 5 pages/call
    return ingestion.page_range_text(d, start_page, end_page)


_DE_SUFFIXES = ("ungen", "ung", "en", "er", "es", "e", "n", "s")


def _de_stem(term: str) -> str:
    """Light German de-suffixing so inflected forms match (ablation flag)."""
    for suf in _DE_SUFFIXES:
        if term.endswith(suf) and len(term) - len(suf) >= 4:
            return term[: -len(suf)]
    return term


@tool
def keyword_search(query: str) -> str:
    """Case-insensitive lexical keyword search over all pages (plain string matching — no embeddings). Returns up to 8 pages with snippets. For Turkish, prefer word stems (e.g. 'yontem' not 'yontemleri')."""
    terms = [t for t in (query or "").lower().split() if len(t) >= 2]
    if config.KEYWORD_STEM_DE:
        terms = [_de_stem(t) for t in terms]
    if not terms:
        return "Provide at least one keyword (min 2 chars)."
    scored = []
    for d in ingestion.DOCS.values():
        for p in d["pages"]:
            low = p["text"].lower()
            score, first = 0, -1
            for t in terms:
                pos = low.find(t)
                if pos != -1 and first == -1:
                    first = pos
                while pos != -1:
                    score += 1
                    pos = low.find(t, pos + len(t))
            if score:
                scored.append((score, d["name"], p["page"], p["text"], first))
    if not scored:
        return f"No lexical match for: {', '.join(terms)}. Try synonyms or shorter stems."
    scored.sort(key=lambda x: -x[0])
    out = []
    for score, name, page, text, first in scored[:8]:
        a = max(0, first - 100)
        out.append(f'"{name}" p.{page} (hits:{score}): ...{text[a:first + 180]}...')
    return "\n".join(out)


@tool
def list_clauses(doc: str) -> str:
    """List the numbered clauses/sections of a standards document (e.g. 7.4.1, Annex B) with titles and pages. Use this for DIN/EN-style standards to navigate by clause number instead of by tree node."""
    d = ingestion.get(doc)
    if not d:
        return f'Document not found: "{doc}". Call list_documents first.'
    return clauses.outline(d)


@tool
def read_clause(doc: str, clause: str) -> str:
    """Read one numbered clause of a standards document by its number (e.g. "7.4.1", "Annex B"). Returns the clause text up to the next heading. Structure-native for standards — prefer this over read_pages when you know the clause number."""
    d = ingestion.get(doc)
    if not d:
        return f'Document not found: "{doc}".'
    return clauses.read(d, clause)[0]


TOOLS = [list_documents, get_tree, read_node, read_pages, keyword_search,
         list_clauses, read_clause]


# ----------------------------------------------------------------- the graph
SYS_TEMPLATE = """You are a document research agent inside a retrieval experiment.
You have NO prior knowledge of the loaded documents; the ONLY admissible evidence is what you obtain through the tools.

Method:
1) Call list_documents, then get_tree on promising documents.
2) Reason over the tree (titles, summaries, page ranges) and open the smallest relevant node with read_node. Use keyword_search to locate terms, read_pages for fine control.
   For DIN/EN-style standards organized by numbered clauses, prefer list_clauses then read_clause('7.4.1', doc) to navigate by clause number.
3) Decompose multi-part questions into sub-questions and pursue each.
4) Before each tool call, state in one short sentence what you are looking for and why.
5) Stop searching once evidence is sufficient, then write the final answer.

Final answer rules: base every claim strictly on text you actually read; cite as (DocumentName, p.X); if the documents do not answer the question, say so explicitly.
Write your final answer in {lang}."""


def _chat_model(with_tools: bool = False) -> ChatOpenAI:
    base = dict(model=config.OPENAI_MODEL,
                api_key=config.OPENAI_API_KEY or "missing",
                temperature=1)  # gpt-5.x only accepts the default temperature
    extra = {"max_completion_tokens": 3000}
    # gpt-5.x rejects reasoning_effort combined with tool calling on
    # /v1/chat/completions, so it's only set for tool-less calls.
    if config.is_gpt5_reasoning() and not with_tools:
        extra["reasoning_effort"] = config.REASONING_EFFORT
    if config.RUN_SEED is not None:
        extra["seed"] = config.RUN_SEED  # best-effort determinism (OpenAI)
    try:  # recent langchain-openai: first-class params
        return ChatOpenAI(**base, **extra)
    except Exception:  # older versions: pass through to the API
        return ChatOpenAI(**base, model_kwargs=extra)


def _build_graph(system_prompt: str):
    model = _chat_model(with_tools=True).bind_tools(TOOLS)

    def agent_node(state: MessagesState):
        reply = model.invoke([SystemMessage(system_prompt)] + state["messages"])
        return {"messages": [reply]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    g.add_edge("tools", "agent")
    return g.compile()


# ------------------------------------------------------------------ plumbing
def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some models return content blocks
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content)
    return str(content or "")


def _final_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not m.tool_calls and _text(m.content).strip():
            return _text(m.content).strip()
    return ""


def _sanitize_for_forcing(messages):
    """OpenAI rejects an assistant tool_call turn without tool results after it."""
    msgs = list(messages)
    while msgs and isinstance(msgs[-1], AIMessage) and msgs[-1].tool_calls:
        msgs.pop()
    return msgs


def _sorted_map(m: dict) -> dict:
    """{doc: set(pages)} -> {doc: sorted list} (drops empty entries)."""
    return {doc: sorted(pages) for doc, pages in m.items() if pages}


_KW_HIT = re.compile(r'"([^"]+)" p\.(\d+)')


def _extract_evidence(messages) -> dict:
    """Reconstruct pipeline B's evidence sets by replaying tool calls purely.

    - read (full text seen): pages that entered context via read_node / read_pages
      (respecting the same char cap the tool applied).
    - searched (snippet seen): pages surfaced by keyword_search — a weaker,
      peeked-at signal, kept separate from truly-read pages.
    """
    read: dict[str, set] = {}
    searched: dict[str, set] = {}
    read_order: list[list] = []          # [[doc, page], ...] first-read order
    seen_order: set = set()
    calls: dict[str, tuple] = {}         # tool_call_id -> (name, args)

    def _add_read(doc_name, pages):
        bucket = read.setdefault(doc_name, set())
        for p in pages:
            bucket.add(p)
            if (doc_name, p) not in seen_order:
                seen_order.add((doc_name, p))
                read_order.append([doc_name, p])

    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                calls[tc.get("id")] = (tc["name"], tc.get("args", {}))
        elif isinstance(m, ToolMessage):
            name, args = calls.get(m.tool_call_id, (None, {}))
            if name == "read_node":
                d = ingestion.get(str(args.get("doc", "")))
                nd = pageindex.find_node(d, str(args.get("node_id", ""))) if d else None
                if d and nd:
                    _, pages = ingestion.collect_page_range(
                        d, nd["start_page"], nd["end_page"], char_cap=pageindex.NODE_READ_CAP)
                    _add_read(d["name"], pages)
            elif name == "read_pages":
                d = ingestion.get(str(args.get("doc", "")))
                if d:
                    try:
                        s = int(args.get("start_page", 1))
                        e = min(int(args.get("end_page", s)), s + 4)  # tool caps 5 pages
                    except (TypeError, ValueError):
                        continue
                    _, pages = ingestion.collect_page_range(d, s, e)
                    _add_read(d["name"], pages)
            elif name == "read_clause":
                d = ingestion.get(str(args.get("doc", "")))
                if d:
                    _, pages = clauses.read(d, str(args.get("clause", "")))
                    _add_read(d["name"], pages)
            elif name == "keyword_search":
                for mo in _KW_HIT.finditer(_text(m.content)):
                    d = ingestion.get(mo.group(1))
                    if d:
                        searched.setdefault(d["name"], set()).add(int(mo.group(2)))

    read_map = _sorted_map(read)
    searched_map = _sorted_map(searched)
    return {
        "read": read_map,
        "searched": searched_map,
        "read_order": read_order,        # first-read order -> reads-to-gold metric
        "pages_read": sum(len(v) for v in read_map.values()),
        "pages_searched": sum(len(v) for v in searched_map.values()),
    }


def _trace_items(m) -> list[dict]:
    """Trace entries for a single message (shared by _parse and the live stream)."""
    items = []
    if isinstance(m, AIMessage):
        txt = _text(m.content).strip()
        if txt:
            items.append({"t": "thought", "text": txt[:400]})
        for tc in (m.tool_calls or []):
            items.append({"t": "tool", "name": tc["name"], "input": tc["args"]})
    elif isinstance(m, ToolMessage):
        items.append({"t": "obs", "preview": _text(m.content)[:200]})
    return items


def _parse(messages) -> tuple[list, dict, int, int]:
    trace, usage, calls, tool_calls = [], {"in": 0, "out": 0}, 0, 0
    for m in messages:
        if isinstance(m, AIMessage):
            calls += 1
            um = m.usage_metadata or {}
            usage["in"] += um.get("input_tokens", 0)
            usage["out"] += um.get("output_tokens", 0)
            tool_calls += len(m.tool_calls or [])
        trace.extend(_trace_items(m))
    return trace, usage, calls, tool_calls


# ------------------------------------------------------------------ pipeline
def _force_final(messages, lang):
    """Produce a forced final answer when the agent stopped mid-tool or hit the cap."""
    msgs = _sanitize_for_forcing(messages)
    forced = _chat_model().invoke(
        [SystemMessage(SYS_TEMPLATE.format(lang=lang)), *msgs,
         HumanMessage("Step limit reached. Do NOT call tools. Write your final "
                      "answer now based only on the evidence gathered, with citations.")])
    return msgs + [forced], forced


def _tables_in_evidence(evidence: dict) -> list[dict]:
    """Table records sitting on the pages the agent actually READ."""
    if not (config.TABLE_SUMMARIES and config.ATTACH_TABLE_IMAGES):
        return []
    found = []
    for doc_name, pages in (evidence.get("read") or {}).items():
        d = ingestion.get(doc_name)
        if d:
            found.extend(tables.for_pages(d, pages))
    return found[:config.MAX_TABLE_IMAGES]


def _verify_against_tables(messages, draft: str, evidence: dict, lang: str):
    """Final answering call with the REAL table images attached.

    Pipeline B reads pages as text, so a table it relied on reached it in the
    same scrambled form pipeline A avoids by indexing summaries. When the pages
    it read contain tables, the answer is re-grounded on the rendered images:
    exact values come off the table, not off the extracted text.
    """
    recs = _tables_in_evidence(evidence)
    if not recs or not draft:
        return messages, draft, []
    parts = [{"type": "text", "text":
              f"Your draft answer:\n\n{draft}\n\nThe pages you read contain the "
              f"table(s) shown below. Correct any value, unit or row/column label "
              f"that disagrees with them, keep every citation, and change nothing "
              f"else. Answer in {lang}."}]
    attached = []
    for rec in recs:
        part = tables.image_part(rec)
        if part:
            parts.append({"type": "text",
                          "text": f'Table from p.{rec["page"]} of "{rec["doc"]}":'})
            parts.append(part)
            attached.append(rec["image"])
    if not attached:
        return messages, draft, []
    try:
        final = _chat_model().invoke([*messages, HumanMessage(content=parts)])
        text = _text(final.content).strip()
    except Exception:
        return messages, draft, []
    return (messages + [final], text or draft, attached) if text else (messages, draft, [])


def _result(messages, hit_limit, max_steps, lang, t0, table_images=None) -> dict:
    trace, usage, calls, tool_calls = _parse(messages)
    return {
        "answer": _final_text(messages) or "(no final answer produced within the step limit)",
        "trace": trace, "usage": usage, "calls": calls, "tool_calls": tool_calls,
        "max_steps": max_steps, "hit_limit": hit_limit,
        "evidence": _extract_evidence(messages),
        "table_images": table_images or [],
        "latency_ms": int((time.time() - t0) * 1000), "language": lang,
    }


def answer(question: str, max_steps: int) -> dict:
    t0 = time.time()
    max_steps = max(2, min(12, int(max_steps)))
    lang = llm.question_language(question)
    graph = _build_graph(SYS_TEMPLATE.format(lang=lang))

    # one "step" = agent node + tools node -> 2 graph transitions
    run_config = {"recursion_limit": 2 * max_steps + 1}
    state, hit_limit = None, False
    try:
        for state in graph.stream({"messages": [HumanMessage(question)]},
                                  run_config, stream_mode="values"):
            pass
    except GraphRecursionError:
        hit_limit = True

    messages = (state or {}).get("messages", [HumanMessage(question)])
    if hit_limit or not _final_text(messages):
        messages, _ = _force_final(messages, lang)
    messages, _, imgs = _verify_against_tables(
        messages, _final_text(messages), _extract_evidence(messages), lang)
    return _result(messages, hit_limit, max_steps, lang, t0, imgs)


def answer_stream(question: str, max_steps: int):
    """Generator yielding live trace events, then a final result — for SSE.

    Yields {"type":"step","item":{...}} as each thought/tool-call/observation
    lands, then {"type":"done","result":{...}} with the same shape as answer().
    """
    t0 = time.time()
    max_steps = max(2, min(12, int(max_steps)))
    lang = llm.question_language(question)
    yield {"type": "start", "language": lang, "max_steps": max_steps}
    graph = _build_graph(SYS_TEMPLATE.format(lang=lang))
    run_config = {"recursion_limit": 2 * max_steps + 1}

    state, hit_limit, emitted = None, False, 0
    try:
        for state in graph.stream({"messages": [HumanMessage(question)]},
                                  run_config, stream_mode="values"):
            msgs = state.get("messages", [])
            while emitted < len(msgs):
                for item in _trace_items(msgs[emitted]):
                    yield {"type": "step", "item": item}
                emitted += 1
    except GraphRecursionError:
        hit_limit = True

    messages = (state or {}).get("messages", [HumanMessage(question)])
    if hit_limit or not _final_text(messages):
        messages, forced = _force_final(messages, lang)
        for item in _trace_items(forced):
            yield {"type": "step", "item": item}

    messages, _, imgs = _verify_against_tables(
        messages, _final_text(messages), _extract_evidence(messages), lang)
    if imgs:
        yield {"type": "step",
               "item": {"t": "obs", "preview": f"re-grounded on {len(imgs)} table image(s)"}}

    yield {"type": "done",
           "result": _result(messages, hit_limit, max_steps, lang, t0, imgs)}
