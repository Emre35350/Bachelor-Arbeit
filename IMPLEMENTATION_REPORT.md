# Implementation report — domain customization of Pipeline A

Date: 2026-07-22 · Scope: clause-aware chunking, multi-vector table handling,
content-addressed chunk IDs, and one deliberate non-implementation (reranking).

All figures below were measured on the working corpus in `../datasets`
(11 DIN/EN/ISO standards + the IFBS guideline, 1220 pages), not estimated.

---

## 0. Summary of what changed

| # | Change | Status | Files |
|---|---|---|---|
| 1 | Chunking cut at clause boundaries instead of a fixed character window | implemented | `app/vector_rag.py`, `app/config.py`, `static/index.html`, `tests/test_chunking.py` |
| 2 | Tables indexed as vision-written summaries, answered with the real table image | implemented | `app/tables.py` (new), `app/ingestion.py`, `app/vector_rag.py`, `app/reasoning_agent.py`, `app/main.py`, `static/index.html`, `tests/test_tables.py` |
| 3 | Content-addressed chunk IDs | implemented | `app/vector_rag.py`, `tests/test_chunking.py` |
| 4 | Incremental, non-destructive index builds (one namespace per condition) | implemented | `app/config.py`, `app/vector_rag.py`, `app/pageindex.py`, `app/store.py`, `app/main.py`, `static/index.html`, `tests/test_incremental.py` |
| 5 | Cross-encoder reranker | **deliberately not implemented** | — |
| 6 | Figures/diagrams | not implemented (scope limitation, feasibility measured) | — |

Documentation updated in `README.md` (§3 component table, §Configuration) and
`.env.example`.

---

## 1. Clause-aware chunking

### 1.1 Problem

Chunking was a pure sliding window: 1000 characters with 150 overlap, applied to
each page's text. The window has no knowledge of where a clause starts or ends,
so it routinely ended mid-requirement, separated a requirement from the numbered
heading that scopes it, and merged the tail of one clause with the head of the
next into a single embedding.

### 1.2 What was implemented

Three functions in `app/vector_rag.py` replace the old loop:

- **`_clause_segments(doc)`** — cuts every page at the clause-heading offsets
  reported by `app/clauses.py`. The heading detector was **reused, not
  reimplemented**: it already handles the PDF-outline-first strategy, two-line
  headings, table rows, TOC leaders, print watermarks and sub-clause
  disambiguation. Segments stay page-scoped because `app/metrics.py` scores
  evidence per `(doc, page)`; a clause continuing onto the next page keeps its
  clause label, so nothing loses attribution.
- **`_merge_tiny(segments)`** — folds segments under 200 characters into the
  following segment on the same page. Standards are full of one-line clauses
  and of parent headings whose body is the sub-clause beneath them; alone they
  embed as noise.
- **`_doc_chunks(doc)`** — splits by characters **only** when a segment exceeds
  `chunk_size`, and that split stays inside the clause. Every resulting piece
  inherits the clause number. Documents with fewer than 3 detected headings
  (plain TXT, PDFs whose structure is invisible) fall back to the old per-page
  character window.

Each chunk now carries `clause` and `clause_title` in addition to
`{doc, page, chunk}`. These travel into Pinecone metadata, into the `[n]`
citation headers the generator sees, into the logged `retrieved` records, and
into the UI.

### 1.3 Measured effect

| Metric | Value |
|---|---|
| Baseline chunks whose interior contained a clause heading | 841 of 3559 (**24 %**) |
| Clause-aware segment spans with a heading strictly inside | **0 of 2649** |
| Total chunks | 3559 → 3876 (text chunks only; table chunks add 350 more, see §7) |
| Chunks carrying a clause number | 89 % (rest is front matter / TOC before the first heading) |

The 24 % baseline figure is a string-search estimate (chunks were located by
matching their opening text, which is imprecise where page headers repeat). The
0-violation figure is exact: it was computed in offset space, and it is also
structural — spans are cut *at* heading offsets, merging only joins adjacent
spans on the same page, and the character splitter only subdivides a single
span, so no operation can produce a chunk crossing a heading.

### 1.4 Open tuning question

At `chunk_size = 1000` only **46 %** of clause segments survive as one chunk
(median segment 1104 characters); the rest are still split internally. Raising
the value now buys more than it used to, because the splits are semantic:

| chunk_size | clauses that stay whole |
|---|---|
| 1000 (current) | 46 % |
| 1500 | 62 % |
| 2000 | 75 % |
| 3000 | 94 % |

This is a candidate ablation sweep, not a change that was made.

### 1.5 Configuration

`CLAUSE_CHUNKING` (default on) is an **index-time** key, so it is part of the
namespace fingerprint: switching it selects a different condition rather than
invalidating the current one (§4). `CLAUSE_CHUNKING=0` restores the plain
sliding window as the ablation baseline, and flipping between the two costs
nothing after each has been built once.

---

## 2. Multi-vector table handling

### 2.1 Feasibility check (run before implementing, as instructed)

Ground truth: pages carrying a real DIN caption `Tabelle N — Title`, with
table-of-contents lines excluded — 211 pages corpus-wide.

| Finding | Result |
|---|---|
| Recall of default `page.find_tables()` | **201/211 = 95 %** |
| `find_tables(strategy="text")` as a detector | unusable — fires on 236 of 238 pages in EN 1090-2 |
| Systematic false positive | the ruled title block on page 1 of every document |
| Pages flagged in total | 366, of which 165 have no caption |
| Borderless-table risk | did not materialize; DIN tables are ruled |

Two intermediate numbers were wrong before correction and are recorded here for
honesty: an initial "100 % recall" came from counting either strategy as a hit
(meaningless, since the text strategy hits everything), and an initial caption
pattern missed clause-numbered captions such as `Tabelle 6.1 —`, understating
the ground-truth count.

Relevance to the evaluation: `questions.json` currently holds **one** question
with empty `gold_pages`, so `app/metrics.py` can score nothing. Using the 15
distinct questions in `experiments.jsonl` as the de facto test set, at least
five are table-answered (zinc coating thickness unit, corrosivity categories
for immersion, γM0, straightedge gap/flatness, EXC classes).

### 2.2 What was implemented

New module `app/tables.py`, wired into both pipelines:

1. **Detect** — `page.find_tables()` at upload time (`app/ingestion.py`), local
   and free. Filters drop the cover-page title block and anything under 3×2
   cells.
2. **Render** — `page.get_pixmap(clip=bbox, matrix=2×)` to
   `data/tables/<doc>-p<page>-t<i>.png`, with padding so the caption is inside
   the image.
3. **Summarize** — one vision call per table at index-build time, prompted to
   preserve exact values, units and row/column labels. Cached on the document,
   so rebuilds and repeated experiment runs never re-pay. Falls back to the
   deterministically extracted cell grid if a call fails.
4. **Index the summary** — table chunks are appended in `_doc_chunks()` and
   inherit the clause open on their page, so they integrate with the clause
   chunker rather than bypassing it. The raw grid is never embedded when a
   summary exists.
5. **Answer with the image** — Pipeline A attaches the images of retrieved
   table chunks to its single grounded call (`_attach_table_images()`).
   Pipeline B has no single final call — its answer emerges inside the agent
   loop — so `_verify_against_tables()` adds one image-grounded pass after the
   loop, re-grounding the draft on tables that sit on pages the agent actually
   read. That extra call is counted in `calls` and token usage like any other.

### 2.3 Measured effect

Real example, ISO 12944-2 p.15 (the corrosivity table a logged question needs).

Text that **was** being embedded:

```
gorie / Flächenbezogener Massenverlust/Dickenabnahme / unlegierter Stahl /
Zink / Freiluft / Innenraum / Massen- verlust / Dickenab- nahme / g/m2 / µm /
C1 / unbedeutend
```

Text that **is now** embedded (one vision call):

> C2 gering: unlegierter Stahl – Massenverlust = > 10 bis 200 g/m²; unlegierter
> Stahl – Dickenabnahme = > 1,3 bis 25 µm; Zink – Massenverlust = > 0,7 bis
> 5 g/m²; Freiluft = Atmosphäre mit geringem Verunreinigungsgrad; meistens
> ländliche Gebiete; Innenraum = unbeheizte Gebäude, in denen Kondensation
> auftreten kann …

Row/column association is restored and units survive. The chunk is labelled
`clause 5.1`, and its rendered image is attached to the answering call.

| Metric | Value |
|---|---|
| Tables detected corpus-wide after filtering | **350** (EN 1090-2 alone: 118) |
| One-time vision calls | 350, cached — **zero per experiment run** |
| IFBS 8.01 filtering effect | 132 raw → 38 kept (most were layout boxes) |

### 2.4 Configuration

| Key | Class | Default | Purpose |
|---|---|---|---|
| `TABLE_SUMMARIES` | index-time | on | off = raw page text only (ablation baseline) |
| `ATTACH_TABLE_IMAGES` | query-time | on | off = measure what the image adds over the summary alone |
| `MAX_TABLE_IMAGES` | query-time | 3 | cap per answering call |
| `VISION_MODEL` | — | empty | empty = use `OPENAI_MODEL`, already vision-capable |

New endpoint `GET /table_image/{name}` serves a rendered PNG; the UI shows a 📊
marker and an inline thumbnail on retrieved table chunks, so a summary can be
checked against the table it describes.

---

## 3. Content-addressed chunk IDs

### 3.1 Problem

The old scheme was `slug(doc_name)-p{page}-c{i}`, where the slug is truncated to
48 characters and `i` is a per-document counter. Two defects:

- **Collision risk** — two documents whose names agree in their first 48
  slugged characters produce the same prefix, so their chunks can overwrite each
  other in Pinecone without any error.
- **Not content-addressed** — the id does not depend on the text. After a
  chunking change the same id string points at completely different text, which
  makes builds impossible to diff across configurations.

### 3.2 What was implemented

`_chunk_id(doc_name, page, text)` in `app/vector_rag.py`:

```
<doc-slug>-p<page>-<sha1(full_doc_name \0 page \0 text)[:10]>
```

e.g. `DIN_EN_1090-2_2024-09-00_DE_3537520-pdf-p125-ff1e686ddb`.

The readable prefix stays the truncated slug, but the **hash covers the full
document name**, so truncation collisions cannot produce equal ids. Table chunks
use the same function (hashing their summary text), replacing the old
`-tab{index}` suffix.

`_assign_ids()` handles the one case pure content addressing gets wrong:
genuinely identical text occurring twice on the same page (a repeated header
row, a boilerplate note) would map to one id and Pinecone would keep only the
last copy. Such chunks get a `#2` suffix instead — nothing is silently dropped,
and the result is still deterministic. `build_index()` additionally refuses to
upsert if any duplicate id survives, so vector loss cannot happen quietly.

### 3.3 Verification

| Property | Result on the real corpus (4226 chunks) |
|---|---|
| All ids distinct | 4226/4226 |
| Duplicates needing a `#2` suffix | 0 |
| Distinct slug prefixes | 11 for 11 documents |

Unit-tested in `tests/test_chunking.py`: determinism (same text → same id),
sensitivity (different text, or a different page → different id), the truncation
collision (two 60-character names sharing a slug now get different ids), and
duplicate disambiguation.

### 3.4 Note

The id scheme is not a configuration key, so changing it does **not** mark the
index stale. The currently built index still carries old-style `-c{i}` ids;
they are opaque to retrieval, so nothing is broken, and the next rebuild
replaces them (a rebuild wipes the namespace before upserting, so no orphans
remain).

---

## 4. Incremental, non-destructive index builds

### 4.1 Problem

`/build_index` re-embedded every chunk of every document and rebuilt every
Pipeline B tree on every call, and it wiped the whole Pinecone namespace first —
regardless of whether the change was an architectural one or a `top_k` tweak.
A sweep of several embedding models × chunk settings × repeats therefore re-paid
the full cost of reprocessing the corpus for every condition, and each condition
destroyed the one before it.

### 4.2 Rule 1 — reprocess only for a real reason

Work is now keyed to a **per-document content signature** (`doc_signature()`:
page text + table summaries). A document is re-embedded only when that signature
changed for the condition being built — i.e. when parsing changed, when the
document is new, or when `force=true` is passed. Query-time parameters
(`top_k`, `max_steps`, model, seed) are not part of any index key and trigger
nothing.

### 4.3 Rule 2 — one namespace per configuration, nothing overwritten

`config.namespace()` returns `<base>-<sha1(index_settings())[:8]>`, and
`config.index_name(dim)` returns `retrievallab-<dimension>`. Consequences:

- switching an index-time setting **selects** another namespace rather than
  overwriting the current one;
- a condition built earlier is queried instantly when you switch back;
- the previous "delete and recreate the index when the embedding dimension
  disagrees" behaviour is gone — it silently destroyed every stored condition on
  every model switch. Dimensions now live in separate indexes.

`data/index_manifest.json` records, per namespace: the settings, index name,
dimension, and for each document its signature, chunk count and vector ids.
Because the ids are recorded, a changed document is replaced **by id** — the
namespace is never wiped (the test suite asserts `delete_all` is never called).
`GET /conditions` lists everything built so far.

### 4.4 Rule 3 — pipelines decoupled, per-document builds

`POST /build_index` takes `{"scope": "vector"|"trees"|"both", "docs": [...],
"force": bool}`. `pageindex.build_all_trees(only, force)` skips any document
that already has a tree, so testing a Pipeline-A parameter costs **zero** LLM
calls for Pipeline B. Adding an 11th document with `{"docs": ["new.pdf"]}`
processes that document only, in both pipelines.

Staleness was redefined accordingly: it no longer means "your index is invalid",
it means "this condition does not hold every document yet", and the UI lists the
missing documents instead of a settings diff.

### 4.5 Verification

`tests/test_incremental.py` (27 checks, Pinecone and the embedder stubbed)
asserts the full matrix:

| Scenario | Asserted outcome |
|---|---|
| Rebuild, nothing changed | 0 vectors written |
| `top_k` changed | 0 vectors written, same namespace |
| `chunk_size` changed | new namespace built, old namespace still populated |
| Switch back | 0 vectors written, original namespace reused |
| Document added | only that document embedded |
| Document content changed | only that document rebuilt, old ids deleted, others intact |
| Any build | `delete_all` never called |
| Trees | existing trees reused; only the treeless document structured |

The same flows were also driven through the real FastAPI endpoints, confirming
two conditions coexisting in one Pinecone account and `scope="vector"` leaving
Pipeline B untouched.

---

## 5. Reranking — deliberately not implemented

A cross-encoder reranker was proposed and then **ruled out**: general-purpose
rerankers are trained on broad web/QA relevance, which does not transfer to
numbered clauses in engineering standards, and this was confirmed in earlier
thesis work. Retrieval is therefore intentionally single-stage — Pinecone top-k
in `app/vector_rag.py`, term-frequency lexical search in
`app/reasoning_agent.py`. This is a defensible methodology position and should
be stated as such in the thesis, since a reviewer familiar with the RAG
literature will otherwise read it as an omission.

---

## 6. Scope limitations

- **Figures/diagrams are not handled.** Feasibility was measured: 247 of 265
  figure-caption pages (93 %) carry an extractable raster image (median ≈
  500×250 px, 801 distinct images, none reused across pages, so they are real
  figures rather than logos). Adding them would reuse the same
  summarize-and-attach path. Not verified: whether each raster *is* the
  captioned figure — EN 1993-1-3 has 218 images across 225 pages, which may
  include inline formula snippets.
- **Table detection precision is loose.** 165 flagged pages have no caption.
  The systematic false positive (cover pages) is filtered; the remainder are a
  mix of genuine uncaptioned tables and layout boxes and were not individually
  reviewed.
- **Pipeline isolation.** Pipeline A now uses the same lexical clause index as
  Pipeline B, and both may receive table images. No embeddings cross between
  paradigms, so the isolation the design depends on holds, but the thesis
  should state that both pipelines receive the same domain adaptation rather
  than A remaining generic.
- **Evaluation set.** `questions.json` has one question and no gold pages;
  nothing can be scored until it is filled in.

---

## 7. Current state of the working corpus

Verified against `data/` on 2026-07-22, after the corpus was re-uploaded and
rebuilt (table detection only runs at upload, and the original PDF bytes are not
stored, so the re-upload was required — it has since happened):

| Item | State |
|---|---|
| Documents in `data/docs.json` | 11, all carrying a `tables` key |
| Tables detected and rendered | 350 (350 PNGs in `data/tables/`) |
| Tables with a vision summary | 350 of 350 — none left on the grid fallback |
| Chunks produced | **4226** = 3876 text chunks + 350 table chunks |
| Chunk id uniqueness | 4226 distinct ids, 0 duplicates needing a `#2` suffix |
| `built_with` | `chunk_size 1000`, `chunk_overlap 150`, `clause_chunking true`, `table_summaries true`, `multilingual-e5-base` — i.e. the index is **not** stale |

The 350 vision calls have therefore already been paid once; later rebuilds reuse
the cached summaries.

**One rebuild is now required**, because §4 changed where vectors live:

- vectors used to sit in index `retrievallab`, namespace `main`; they now go to
  `retrievallab-768`, namespace `main-4446bd89` (the fingerprint of the current
  index-time settings);
- the old index/namespace is **not** touched or deleted — it simply stops being
  read. Delete it in the Pinecone console if you want the storage back.

The rebuild is cheap: embeddings are computed locally by sentence-transformers
(no API cost), the 350 table summaries are already cached on the documents, and
Pipeline B's trees are persisted in `docs.json`, so `build_all_trees()` will
skip all eleven — **zero LLM calls**. After it, every later condition switch is
incremental.

---

## 8. Tests

Runnable without pytest, in the style of the existing `tests/test_clauses.py`:

```
python -m tests.test_clauses      # unchanged — 11 checks
python -m tests.test_chunking     # new — 25 checks
python -m tests.test_tables       # new — 20 checks
python -m tests.test_incremental  # new — 27 checks
```

`test_chunking.py` covers boundary integrity, oversize splitting, cross-page
continuation, stub merging, both fallback paths, and the chunk-id properties
(determinism, sensitivity to text and page, the slug-truncation collision,
duplicate disambiguation). `test_tables.py` covers the detection filters,
caption picking, summary-vs-grid selection, table-chunk labelling, both ablation
switches, missing-image robustness, and real detection against `../datasets`
when present (the vision call is not made, so the test costs nothing).

`test_incremental.py` stubs Pinecone and the embedder to assert the build rules:
that an unchanged rebuild writes nothing, that a query-time parameter triggers
no work, that index-time parameters open a second namespace without harming the
first, that switching back is free, that a new or changed document is processed
alone, that `delete_all` is never called, and that Pipeline B skips documents
that already have a tree.

All four suites pass (83 checks total).
