"""Pipeline A — Vector-Based RAG.

Components (per thesis spec):
  chunking   : clause-aware (cut at clause headings; character split only INSIDE
               an over-long clause), page-aware
  embeddings : intfloat/multilingual-E5 (sentence-transformers; "query:"/"passage:" prefixes)
  vector DB  : Pinecone (serverless, cosine metric)
  retrieval  : top-k nearest neighbours
  generation : single grounded LLM call with [n] citations

Heavy dependencies (sentence-transformers, pinecone) are imported lazily so the
app starts fast and other endpoints work even before the first index build.
"""
import hashlib
import re
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import clauses, config, ingestion, llm, progress, tables

CHUNKS: list[dict] = []          # local copy for stats/inspection; text also lives in Pinecone metadata
_embedder = None
_index = None
_index_name = None


# ---------------------------------------------------------------- embeddings
# Different embedding models expect the query/passage distinction to be signalled
# differently. E5 prepends "query: " / "passage: " to the text; Jina v3 selects a
# task-specific LoRA adapter via a `task` argument and takes no textual prefix.
# Encoding the wrong way silently degrades retrieval, so it is dispatched by model
# family here rather than hard-coded.
def _model_family(name: str | None = None) -> str:
    n = (name or config.EMBEDDING_MODEL or "").lower()
    if "jina-embeddings-v3" in n:
        return "jina-v3"
    if "e5" in n:
        return "e5"
    return "generic"


def embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # lazy: pulls torch
        # Jina v3 ships custom modelling code (LoRA task adapters) that must be
        # allowed to load; E5 and plain sentence-transformers models do not.
        kwargs = {"trust_remote_code": True} if _model_family() == "jina-v3" else {}
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL, **kwargs)
    return _embedder


def _encode(model, texts: list[str], *, is_query: bool):
    """Embed texts using the convention the CURRENT model expects."""
    fam = _model_family()
    if fam == "jina-v3":
        # task adapter carries the query/passage instruction — no textual prefix
        task = "retrieval.query" if is_query else "retrieval.passage"
        return model.encode(texts, task=task, normalize_embeddings=True,
                            show_progress_bar=False)
    if fam == "e5":
        prefix = "query: " if is_query else "passage: "
        return model.encode([prefix + t for t in texts], normalize_embeddings=True,
                            show_progress_bar=False)
    # generic sentence-transformers model: no prefix, just normalise
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def reset_state() -> None:
    """Drop the cached embedder + Pinecone handle after an embedding_model
    change, so the next build loads the new model (and checks dimensions)."""
    global _embedder, _index, _index_name
    _embedder = None
    _index = None
    _index_name = None


# ------------------------------------------------------------------ pinecone
def _pinecone_index(dimension: int):
    """Handle for the index of this embedding DIMENSION.

    One index per dimension (`retrievallab-768`, `retrievallab-384`, …): Pinecone
    dimensions are immutable, and the previous behaviour — delete and recreate
    the single fixed index whenever the dimension disagreed — destroyed every
    previously built experimental condition on every embedding-model switch.
    """
    global _index, _index_name
    name = config.index_name(dimension)
    if _index is not None and _index_name == name:
        return _index
    from pinecone import Pinecone, ServerlessSpec  # lazy
    if not config.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set in .env")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    try:
        exists = pc.has_index(name)
    except AttributeError:  # older SDKs
        listed = pc.list_indexes()
        names = getattr(listed, "names", lambda: [i["name"] for i in listed])()
        exists = name in names

    if not exists:
        pc.create_index(name=name, dimension=dimension, metric="cosine",
                        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD,
                                            region=config.PINECONE_REGION))
        for _ in range(60):  # wait until serverless index is ready
            if pc.describe_index(name).status.get("ready"):
                break
            time.sleep(2)

    _index, _index_name = pc.Index(name), name
    return _index


def _slug(s: str) -> str:
    """Pinecone vector ids must be ASCII."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s)[:48] or "doc"


def _chunk_id(doc_name: str, page: int, text: str) -> str:
    """Content-addressed vector id: `<slug>-p<page>-<sha1(text)[:10]>`.

    Deterministic (same text -> same id) so two builds of the same corpus can be
    diffed, and self-documenting: an id that changed means the TEXT changed, not
    just its position. The old `-c{i}` counter had neither property — after a
    chunking change the same id pointed at completely different text, which made
    configurations impossible to compare.

    The readable prefix is the 48-char truncated slug, but the hash is taken over
    the FULL document name, so two documents whose names collide after truncation
    still get different ids instead of silently overwriting each other in Pinecone.
    """
    key = f"{doc_name}\x00{page}\x00{text}".encode("utf-8")
    return f"{_slug(doc_name)}-p{page}-{hashlib.sha1(key).hexdigest()[:10]}"


def _assign_ids(chunks: list[dict]) -> int:
    """Give every chunk its content id, disambiguating exact duplicates.

    Identical text can legitimately occur twice on one page (a repeated header
    row, a boilerplate note). Content addressing would map both to one id and
    Pinecone would keep only the last one, so a `#2` suffix is appended instead:
    nothing is silently dropped, and the result is still deterministic.
    Returns how many duplicates were disambiguated.
    """
    seen: dict[str, int] = {}
    dupes = 0
    for c in chunks:
        base = _chunk_id(c["doc"], c["page"], c["text"])
        n = seen.get(base, 0) + 1
        seen[base] = n
        if n > 1:
            dupes += 1
        c["id"] = base if n == 1 else f"{base}#{n}"
    return dupes


# ------------------------------------------------------------------ chunking
# A fixed sliding window cuts straight through the structure these documents are
# written in: a 1000-char window ends mid-requirement, separates a requirement
# from the "7.4.1" heading that scopes it, and glues the tail of one clause onto
# the head of the next — so a single embedding mixes two unrelated obligations.
# Clause headings are the authoritative semantic boundary here, so they are the
# primary cut points and characters are only a fallback INSIDE a long clause.
#
# The heading index comes from app.clauses (purely lexical regex/PDF outline —
# no embeddings), so paradigm isolation between the pipelines still holds; both
# now share the same structural view of the document.

MIN_SEGMENT_CHARS = 200      # below this a segment is merged into the next one
MIN_CLAUSES_FOR_STRUCTURE = 3  # fewer headings than this = no usable structure


def _char_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # recursive fallback order
    )


def _clause_segments(doc: dict) -> list[dict]:
    """Cut every page at the clause headings that start on it.

    Returns [{page, clause, title, text}] in reading order. Segments stay
    page-scoped (one chunk never spans two pages) because page numbers are the
    unit of evidence in the metrics — but a clause that continues onto the next
    page keeps its clause label, so the continuation is still attributable.
    """
    by_page: dict[int, list[dict]] = {}
    for c in clauses.index(doc):
        by_page.setdefault(int(c["page"]), []).append(c)

    segments, current = [], None   # current = clause still open at page start
    for p in doc["pages"]:
        text = p["text"] or ""
        heads = sorted(by_page.get(p["page"], []), key=lambda c: int(c["offset"]))
        # cut points: the page start (tail of the previous clause) + each heading
        points = [(0, current)] + [(max(0, min(len(text), int(h["offset"]))), h)
                                   for h in heads]
        for i, (start, owner) in enumerate(points):
            end = points[i + 1][0] if i + 1 < len(points) else len(text)
            body = text[start:end].strip()
            if body:   # a heading at offset 0 yields an empty tail — skipped here
                segments.append({"page": p["page"], "text": body,
                                 "clause": (owner or {}).get("clause"),
                                 "title": (owner or {}).get("title")})
        if heads:
            current = heads[-1]
    return segments


def _merge_tiny(segments: list[dict]) -> list[dict]:
    """Fold undersized segments into the following one on the same page.

    Standards are full of one-line clauses ("5.1.1 Allgemeines — Es gilt 5.2.")
    and of parent headings whose body is just the sub-clause that follows. Alone
    they embed as noise; attached to the clause they introduce they carry it.
    """
    out: list[dict] = []
    for seg in segments:
        prev = out[-1] if out else None
        if prev and prev["page"] == seg["page"] and len(prev["text"]) < MIN_SEGMENT_CHARS:
            prev["text"] = f"{prev['text']}\n{seg['text']}"
            if prev["clause"] is None:   # tail-of-page piece adopts the real heading
                prev["clause"], prev["title"] = seg["clause"], seg["title"]
        else:
            out.append(dict(seg))
    return out


def _doc_chunks(doc: dict) -> tuple[list[dict], str]:
    """Chunks for one document, plus the strategy actually used.

    Falls back to pure character splitting for documents without a usable clause
    structure (plain .txt uploads, PDFs whose headings the indexer cannot see).
    """
    idx = clauses.index(doc) if config.CLAUSE_CHUNKING else []
    if len(idx) >= MIN_CLAUSES_FOR_STRUCTURE:
        strategy = "clause"
        segments = _merge_tiny(_clause_segments(doc))
    else:
        strategy = "chars"
        segments = [{"page": p["page"], "text": (p["text"] or "").strip(),
                     "clause": None, "title": None}
                    for p in doc["pages"] if (p["text"] or "").strip()]

    splitter = _char_splitter()
    out, i = [], 0
    for seg in segments:
        # only an unusually long clause is cut by characters — and the cut stays
        # inside that clause, so no chunk ever straddles a clause boundary
        pieces = ([seg["text"]] if len(seg["text"]) <= config.CHUNK_SIZE
                  else splitter.split_text(seg["text"]))
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            out.append({
                "id": None,                     # set by _assign_ids below
                "doc": doc["name"], "page": seg["page"], "chunk": i,
                "text": piece, "clause": seg["clause"],
                "clause_title": seg["title"], "strategy": strategy,
                "table_image": None,
            })
            i += 1

    # multi-vector: the SUMMARY of each table is indexed as its own chunk, and
    # keeps a pointer to the rendered image for the answering call
    if config.TABLE_SUMMARIES:
        clause_of = _clause_by_page(doc)
        for rec in tables.index_tables(doc):
            text = tables.chunk_text(rec)
            if not text:
                continue
            clause = clause_of(int(rec["page"]))
            out.append({
                "id": None,                     # set by _assign_ids below
                "doc": doc["name"], "page": int(rec["page"]), "chunk": i,
                "text": text, "clause": clause and clause["clause"],
                "clause_title": clause and clause["title"], "strategy": "table",
                "table_image": rec["image"],
            })
            i += 1

    _assign_ids(out)
    return out, strategy


def _clause_by_page(doc: dict):
    """Which clause is open on a given page — used to label table chunks."""
    idx = clauses.index(doc) if config.CLAUSE_CHUNKING else []

    def lookup(page: int):
        best = None
        for c in idx:
            if int(c["page"]) <= page:
                best = c
            else:
                break
        return best
    return lookup


# --------------------------------------------------------------- index build
def doc_signature(doc: dict) -> str:
    """Fingerprint of everything about a document that affects its vectors.

    Page text plus the table summaries — so re-uploading an unchanged file is a
    no-op, while a document whose parsing changed (e.g. tables were extracted for
    the first time) is re-indexed on its own, without touching the other ten.
    """
    h = hashlib.sha1()
    h.update(doc["name"].encode("utf-8"))
    for p in doc["pages"]:
        h.update(f"\x00{p['page']}\x00".encode("utf-8"))
        h.update((p["text"] or "").encode("utf-8"))
    for t in tables.index_tables(doc):
        # caption is part of the embedded text (chunk_text = caption + summary),
        # so a corrected caption must change the signature and trigger a re-embed
        h.update(f"\x01{t['page']}\x01{t.get('caption') or ''}"
                 f"\x01{t.get('summary') or ''}".encode("utf-8"))
    return h.hexdigest()[:16]


def _summarize_tables(docs: list[dict]) -> int:
    """Vision summaries for the given documents only (cached on the document)."""
    if not config.TABLE_SUMMARIES:
        return 0
    pending = [t for d in docs for t in tables.index_tables(d) if not t.get("summary")]
    if not pending:
        return 0
    progress.set_phase("tables", 0, f"Summarizing {len(pending)} tables (vision, once)…")
    done, made = 0, 0
    for d in docs:
        def bump(_i, _n, _d=d):
            nonlocal done
            done += 1
            progress.set_phase("tables", done / len(pending),
                               f"Summarizing table {done}/{len(pending)} ({_d['name'][:28]})…")
        made += tables.ensure_summaries(d, on_progress=bump)
    progress.set_phase("tables", 1, f"{made} table summaries created.")
    return made


def build_index(only: list[str] | None = None, force: bool = False) -> dict:
    """Embed only what this configuration does not already have.

    Three things make a build incremental:
      * the namespace is derived from the index-time settings, so changing the
        embedding model or chunk size selects another namespace instead of
        overwriting this one — flipping back is free;
      * each document is skipped when its content signature is already recorded
        for this namespace, so a `top_k` tweak or a second document upload does
        not re-embed the other ten;
      * a document that DID change is re-indexed alone, by deleting exactly the
        vector ids the manifest recorded for it.

    `only` restricts the build to named documents; `force` re-embeds regardless
    of the manifest.
    """
    global CHUNKS
    from . import store  # local import: store imports config+ingestion, not us

    ns = config.namespace()
    manifest = store.load_manifest()
    cond = manifest.setdefault(ns, {"settings": config.index_settings(),
                                    "index": None, "dimension": None, "docs": {}})
    built = cond.setdefault("docs", {})

    docs = [d for d in ingestion.DOCS.values()
            if not only or d["name"] in set(only)]
    if not docs:
        raise ValueError("no matching documents to index")

    # repair captions that earlier (per-page) detection got wrong on multi-table
    # pages; this is free (uses stored page text), idempotent, and clears the
    # summary of any table it corrects so it is re-described under the right one
    caption_fixes = sum(tables.backfill_captions(d) for d in docs) if config.TABLE_SUMMARIES else 0

    # a document is rebuilt only when its own content changed for THIS condition
    targets = [d for d in docs
               if force or built.get(d["name"], {}).get("signature") != doc_signature(d)]
    target_names = {d["name"] for d in targets}
    reused = [d["name"] for d in docs if d["name"] not in target_names]

    # summaries are part of the signature, so summarize before the final hash
    summarized = _summarize_tables(targets)
    signatures = {d["name"]: doc_signature(d) for d in targets}

    if not targets:
        CHUNKS = []
        progress.set_phase("done", 1,
                           f"Nothing to embed — all {len(reused)} document(s) already "
                           f"in namespace {ns}.")
        return {"chunks": sum(v.get("chunks", 0) for v in built.values()),
                "namespace": ns, "index": cond.get("index"),
                "dimension": cond.get("dimension"),
                "reused_docs": reused, "indexed_docs": [], "reused": True,
                "clause_chunks": 0, "clause_aware_docs": 0, "table_chunks": 0,
                "tables_summarized": 0, "duplicate_ids": 0}

    progress.set_phase("chunking", 0,
                       f"Splitting {len(targets)} document(s) into chunks…")
    CHUNKS, structured = [], 0
    for d in targets:
        chunks, strategy = _doc_chunks(d)
        CHUNKS.extend(chunks)
        structured += strategy == "clause"
    if not CHUNKS:
        raise ValueError("no chunks produced from the loaded documents")
    clause_chunks = sum(1 for c in CHUNKS if c.get("clause"))
    # ids are content hashes, so a collision here means genuinely identical text
    duplicate_ids = sum(1 for c in CHUNKS if "#" in c["id"])
    if len({c["id"] for c in CHUNKS}) != len(CHUNKS):
        raise ValueError("chunk id collision — refusing to upsert and lose vectors")
    progress.set_phase("chunking", 1,
                       f"{len(CHUNKS)} chunks created from {len(targets)} document(s) "
                       f"({structured} clause-aware); {len(reused)} document(s) reused.")

    already_loaded = _embedder is not None
    progress.set_phase("loading_model", 0 if already_loaded else 0.3,
                        "Embedding model ready." if already_loaded
                        else "Loading embedding model (first run downloads it)…")
    model = embedder()
    dim = model.get_sentence_embedding_dimension()
    progress.set_phase("loading_model", 1, "Embedding model ready.")
    index = _pinecone_index(dim)

    # replace only the documents being rebuilt — the namespace is never wiped,
    # so other documents (and every other condition) survive untouched
    for d in targets:
        old = built.get(d["name"], {}).get("ids") or []
        for s in range(0, len(old), 500):
            try:
                index.delete(ids=old[s:s + 500], namespace=ns)
            except Exception:
                pass  # vectors may not exist yet (first build of this namespace)

    B = 64
    total_batches = (len(CHUNKS) + B - 1) // B
    for s in range(0, len(CHUNKS), B):
        batch = CHUNKS[s:s + B]
        vecs = _encode(model, [c["text"] for c in batch], is_query=False)
        index.upsert(
            vectors=[{
                "id": c["id"],
                "values": v.tolist(),
                # Pinecone rejects null metadata values, so clause keys are only
                # written for chunks that actually sit under a heading
                "metadata": {"doc": c["doc"], "page": c["page"],
                             "chunk": c["chunk"], "text": c["text"][:2500],
                             **({"clause": c["clause"]} if c.get("clause") else {}),
                             **({"clause_title": c["clause_title"]}
                                if c.get("clause_title") else {}),
                             **({"table_image": c["table_image"]}
                                if c.get("table_image") else {})},
            } for c, v in zip(batch, vecs)],
            namespace=ns,
        )
        batch_no = s // B + 1
        progress.set_phase(
            "embedding", batch_no / total_batches,
            f"Embedding & upserting batch {batch_no}/{total_batches} "
            f"({min(s + B, len(CHUNKS))}/{len(CHUNKS)} chunks) → {ns}…")

    # record what this condition now holds, so the next build can skip it
    for d in targets:
        mine = [c for c in CHUNKS if c["doc"] == d["name"]]
        built[d["name"]] = {"signature": signatures[d["name"]],
                            "chunks": len(mine), "ids": [c["id"] for c in mine]}
    cond.update({"settings": config.index_settings(),
                 "index": config.index_name(dim), "dimension": dim})
    store.save_manifest(manifest)

    return {"chunks": sum(v["chunks"] for v in built.values()),
            "embedded_chunks": len(CHUNKS), "dimension": dim,
            "index": config.index_name(dim), "namespace": ns,
            "reused_docs": reused, "indexed_docs": [d["name"] for d in targets],
            "reused": False,
            "clause_chunks": clause_chunks, "clause_aware_docs": structured,
            "table_chunks": sum(1 for c in CHUNKS if c.get("table_image")),
            "tables_summarized": summarized, "duplicate_ids": duplicate_ids,
            "caption_fixes": caption_fixes}


def namespace_state(docs=None) -> dict:
    """Which documents this configuration's namespace already holds."""
    from . import store
    ns = config.namespace()
    built = (store.load_manifest().get(ns) or {}).get("docs") or {}
    names = [d["name"] for d in (docs if docs is not None else ingestion.DOCS.values())]
    missing = [n for n in names
               if built.get(n, {}).get("signature") != doc_signature(ingestion.DOCS[n])]
    return {"namespace": ns, "ready": bool(names) and not missing,
            "missing": missing, "built": [n for n in names if n not in missing],
            "chunks": sum(v.get("chunks", 0) for v in built.values())}


def known_conditions() -> list[dict]:
    """Every configuration namespace already built — nothing is ever destroyed."""
    from . import store
    out = []
    for ns, cond in (store.load_manifest() or {}).items():
        docs = cond.get("docs") or {}
        out.append({"namespace": ns, "settings": cond.get("settings") or {},
                    "index": cond.get("index"), "dimension": cond.get("dimension"),
                    "docs": len(docs),
                    "chunks": sum(v.get("chunks", 0) for v in docs.values()),
                    "current": ns == config.namespace()})
    return sorted(out, key=lambda c: (not c["current"], c["namespace"]))


# ------------------------------------------------------------------ pipeline
SYS_VECTOR = """You are a source-grounded assistant. You will receive numbered source passages and a question.
Rules:
- Answer ONLY from the passages. Never use outside knowledge.
- Cite after each claim using bracketed passage numbers like [1] or [2][3].
- If the passages do not contain the answer, say clearly that it was not found in the sources.
- Be concise but complete.
- Some passages are SUMMARIES of a table. When the matching table image is attached,
  read exact values, units and row/column labels off the image, not off the summary.
"""


def _attach_table_images(parts: list[dict], hits: list[dict]) -> list[str]:
    """Append the rendered image of every retrieved table chunk. Returns the
    image names attached (also logged, so a run records what the model saw)."""
    if not config.ATTACH_TABLE_IMAGES:
        return []
    attached = []
    for h in hits:
        if not h.get("table_image") or len(attached) >= config.MAX_TABLE_IMAGES:
            continue
        part = tables.image_part({"image": h["table_image"]})
        if part:
            parts.append({"type": "text",
                          "text": f'Table image for p.{h["page"]} of "{h["doc"]}":'})
            parts.append(part)
            attached.append(h["table_image"])
    return attached


def answer(question: str, top_k: int) -> dict:
    t0 = time.time()
    model = embedder()
    qv = _encode(model, [question], is_query=True)[0]
    index = _pinecone_index(model.get_sentence_embedding_dimension())

    res = index.query(vector=qv.tolist(), top_k=max(1, min(10, int(top_k))),
                      include_metadata=True, namespace=config.namespace())
    matches = res["matches"] if isinstance(res, dict) else res.matches

    hits = []
    for m in matches:
        md = m["metadata"] if isinstance(m, dict) else m.metadata
        score = m["score"] if isinstance(m, dict) else m.score
        hits.append({"doc": md.get("doc"), "page": int(md.get("page", 0)),
                     "chunk": int(md.get("chunk", 0)), "score": round(float(score), 4),
                     "clause": md.get("clause"), "table_image": md.get("table_image"),
                     "text": md.get("text", "")})

    # the clause a passage belongs to is part of its citation in this domain
    context = "\n\n".join(
        f"[{i+1}] (\"{h['doc']}\", p.{h['page']}, #{h['chunk']}"
        + (f", clause {h['clause']}" if h.get("clause") else "")
        + (" — TABLE SUMMARY, image attached" if h.get("table_image") else "")
        + f")\n{h['text']}"
        for i, h in enumerate(hits))

    # multi-vector: retrieval ran on the summaries, generation gets the real table
    parts = [{"type": "text",
              "text": f"SOURCE PASSAGES:\n\n{context}\n\nQUESTION: {question}"}]
    attached = _attach_table_images(parts, hits)

    lang = llm.question_language(question)
    resp = llm.chat(
        [{"role": "system", "content": SYS_VECTOR + f"\nWrite your final answer in {lang}."},
         {"role": "user", "content": parts if attached else parts[0]["text"]}],
        max_completion_tokens=2500,
    )
    msg = (resp.choices[0].message.content or "").strip()

    # structured evidence: which pages the top-k chunks came from (one chunk can
    # share a page with others, so collapse to a unique per-doc page set)
    retrieved_pages: dict[str, set] = {}
    for h in hits:
        retrieved_pages.setdefault(h["doc"], set()).add(int(h["page"]))
    retrieved_map = {doc: sorted(pages) for doc, pages in retrieved_pages.items()}

    return {
        "answer": msg,
        "retrieved": [{**{k: h[k] for k in ("doc", "page", "chunk", "score")},
                       **({"clause": h["clause"]} if h.get("clause") else {}),
                       **({"table_image": h["table_image"]} if h.get("table_image") else {}),
                       "snippet": h["text"][:400]} for h in hits],
        "table_images": attached,
        "evidence": {"retrieved": retrieved_map,
                     "pages_retrieved": sum(len(v) for v in retrieved_map.values())},
        "usage": {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens},
        "latency_ms": int((time.time() - t0) * 1000),
        "top_k": int(top_k), "calls": 1, "language": lang,
    }
