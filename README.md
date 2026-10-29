# RetrievalLab v2 — Vector-Based RAG × Reasoning-Based Retrieval Comparison Prototype

A NotebookLM-style, source-grounded question-answering prototype that answers the
same question with two different retrieval paradigms, **sequentially and in
isolation**:

- **A — Vector-Based RAG:** question and document chunks are embedded; the k
  nearest chunks (cosine similarity, Pinecone) become the context of a single
  grounded LLM call.
- **B — Reasoning-Based Retrieval:** a LangGraph agent navigates a
  **PageIndex-style hierarchical tree** of each document — it reads the map,
  opens nodes, searches keywords — looping until the evidence suffices.
  Vectorless by design.

Architecture: **Python backend (FastAPI) + single-page web UI**.

```
retrievallab-py/
├── app/
│   ├── main.py             FastAPI routes (upload / build_index / ask / judge / metrics / log …)
│   ├── config.py           .env-driven settings + runtime /config overrides
│   ├── ingestion.py        PyMuPDF parsing → page-level document store (shared)
│   ├── tables.py           table detection, PNG rendering, vision summaries (multi-vector)
│   ├── clauses.py          clause-heading index (PDF outline or lexical scan)
│   ├── vector_rag.py       Pipeline A: clause-aware chunking → embeddings → Pinecone → top-k → generation
│   ├── pageindex.py        Pipeline B index: PageIndex-style tree builder + tree audit
│   ├── reasoning_agent.py  Pipeline B: explicit LangGraph StateGraph agent + tools
│   ├── llm.py              OpenAI helpers + question-language detection
│   ├── store.py            persistence of docs, build state and index manifest (./data)
│   ├── progress.py         build-progress state polled by the UI
│   ├── logstore.py         JSONL experiment log
│   ├── questions.py        evaluation question set (gold_pages)
│   ├── batch.py            batch runner (UI + `python -m app.batch`)
│   ├── metrics.py          metrics dashboard, CSV and Excel export
│   ├── judge.py            blind pairwise LLM judge (A vs B)
│   ├── judge_batch.py      post-hoc pairwise judging of a logged batch (CLI)
│   ├── judge_ref.py        reference-guided 1–5 rubric judge (CLI, Anthropic)
│   └── ragas_eval.py       RAGAS-style answer-quality metrics (CLI)
├── static/index.html       English UI (sequential A→B protocol, trace, metrics)
├── tests/                  tests (chunking, clauses, embedding, incremental builds, metrics, tables)
├── data/                   persisted docs, trees, table images, index manifest (created at runtime)
├── questions.json          evaluation question set (schema: questions.example.json)
├── *.py (root level)       ablation runners, batch comparison, Excel/markdown table exporters (§18)
├── requirements.txt
└── .env.example
```

---

## 1. Setup (VS Code)

**Requirements:** Python ≥ 3.10 (tested with 3.14).

**Windows (PowerShell):**

```powershell
# in the project folder (retrievallab-py — the one containing app/ and requirements.txt)
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env   # first setup only — overwrites an existing .env
# edit .env → put your OPENAI_API_KEY and PINECONE_API_KEY
python -m uvicorn app.main:app --reload
```

**macOS / Linux:**

```bash
# in the project folder (retrievallab-py — the one containing app/ and requirements.txt)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env     # first setup only — overwrites an existing .env
# edit .env → put your OPENAI_API_KEY and PINECONE_API_KEY
python -m uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** in the browser. Flow: upload sources → **Build
indexes** → ask.

Notes on activation:
- Every new terminal tab needs the activate step again (`.venv\Scripts\Activate.ps1` /
  `source .venv/bin/activate`) — once active, the prompt is prefixed with `(.venv)`.
- If PowerShell blocks the activation script with an execution-policy error, run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, then retry.
- If `python --version` fails with "Python not found" on Windows even
  though Python is installed, disable the `python.exe`/`python3.exe` **App execution
  aliases** (Settings → Apps → Advanced app settings → App execution aliases) — the
  Microsoft Store stub shadows a real install otherwise.
- **Wrong-venv trap:** if your prompt already shows `(.venv)` *before* you `cd` into
  `retrievallab-py`, you're inside a different virtual environment (e.g. one created
  in a parent folder) — it won't have this project's packages, so
  `python -m uvicorn app.main:app --reload` fails with `No module named uvicorn`. Always `cd`
  into `retrievallab-py` **first**, then run `.venv\Scripts\Activate.ps1` so the
  right `.venv` wins. Sanity check anytime with
  `python -c "import sys; print(sys.executable)"` — it must end in
  `retrievallab-py\.venv\Scripts\python.exe`.
- **Use `python -m pip` / `python -m uvicorn`, not bare `pip` / `uvicorn`.** The
  `pip.exe` / `uvicorn.exe` launchers inside `.venv\Scripts` hard-code the absolute
  path the venv was created at; if the project folder is moved or copied, they fail
  (exit code 1, often with no message). `python -m …` always runs the venv's own
  interpreter. To repair the launchers, delete `.venv` and recreate it in place.

Notes:
- **Pinecone key:** free Starter tier at https://app.pinecone.io → API Keys.
  The app creates one index per embedding dimension automatically on the first
  build (`retrievallab-768` for E5-base, `retrievallab-1024` for bge-m3; cosine,
  serverless; the prefix comes from `PINECONE_INDEX`).
- **First build is slow:** sentence-transformers downloads
  `multilingual-e5-base` (~1.1 GB). Lighter alternative in `.env`:
  `EMBEDDING_MODEL=intfloat/multilingual-e5-small` (~470 MB). Cached afterwards.
- **Persistence:** parsed pages, PageIndex trees and the build state are
  written to `./data` (configurable via `DATA_DIR`; disable with `PERSIST=0`),
  so a backend restart restores your sources and indexes automatically — no
  re-upload/rebuild. Pinecone vectors persist in the cloud and the experiment
  log (`experiments.jsonl`) persists on disk. If you change an index-time
  setting in `.env` while the app is down, the next start selects that setting's
  own condition (namespace): if it was built before, it is used directly;
  otherwise the UI reports it as *not built yet* and lists the missing
  documents (see §17).

---

## 2. Sequential protocol & isolation (experiment validity)

1. **Sequencing:** the frontend calls `POST /ask/vector`, renders answer A,
   then starts pipeline B with the same `run_id` — via `GET /ask/reasoning/stream`
   when **live trace** is on (the UI default) or `POST /ask/reasoning` when it is
   off. B starts only after A has finished.
2. **Information isolation:** pipeline B receives only the original question
   and the raw page store — never A's answer, chunks or scores.
3. **Mechanism isolation:** B's tools are lexical/structural only (tree
   navigation + plain string search). B cannot touch the Pinecone index, so
   its performance cannot be attributed to dense embeddings.
4. **Controlled variables:** both pipelines share the same PyMuPDF parse and
   the same generator model (`OPENAI_MODEL`); the *retrieval strategy* is the
   intended independent variable. The generation calls are not
   parameter-identical, though:
   - A's single answering call sends `reasoning_effort` (default `low`) and
     `max_completion_tokens=2500`.
   - B's tool-calling agent turns send **no** `reasoning_effort` (gpt-5.x rejects
     it together with tools on Chat Completions), plus `temperature=1` and
     `max_completion_tokens=3000`; B's forced-final and table re-grounding calls
     do send `reasoning_effort`.
   - When the pages B read contain tables, B makes one extra call that
     re-grounds its draft answer on the table images (§3, Tables).

   These differences are listed here so they can be reported with the results.

---

## 3. Pipeline A — Vector-Based RAG components

| Component | Choice | Detail |
|---|---|---|
| PDF parsing | **PyMuPDF (fitz)** | `page.get_text("text")` per page; page numbers kept as citation metadata; PDF outline (`get_toc`) also captured for pipeline B. TXT/MD read directly. Scanned PDFs without a text layer are rejected with a clear error (no OCR). |
| Tables | **Multi-vector** (`app/tables.py`) | PDF text extraction destroys the row/column association a requirement depends on, so tables are handled separately: `page.find_tables()` at upload → render `get_pixmap(clip=bbox)` to PNG → one vision call per table writes a plain-text summary preserving values, units and labels → **the summary is what gets embedded and retrieved**, while the **image is attached to the final answering call** of whichever pipeline pulls that evidence, so exact numbers are read off the real table. Summaries are cached on the document: 350 one-time calls for this corpus, zero per experiment run. Detection recall measured at 95% (201/211 caption-bearing table pages); filters drop the ruled DIN cover-page title block and anything under 3×2 cells. Falls back to the extracted cell grid if a vision call fails. `TABLE_SUMMARIES=0` / `ATTACH_TABLE_IMAGES=0` are the ablation baselines. |
| Chunking | **Clause-aware** (headings from `app/clauses.py`), `RecursiveCharacterTextSplitter` inside a clause | Primary cut points are the clause headings of the standard, so a chunk never straddles two clauses; segments shorter than 200 chars (bare headings, one-line sub-clauses) merge into the clause they introduce. Only a clause longer than `chunk_size` is split further by `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=150`, separator cascade `["\n\n", "\n", ". ", " ", ""]`), and that split stays inside the clause. Chunks remain page-scoped, so every chunk keeps `{doc, page, chunk}` metadata plus `clause`/`clause_title` when it sits under a heading. Documents with fewer than 3 detected headings (plain TXT, PDFs whose headings the indexer cannot see) fall back to the plain per-page character window. Set `CLAUSE_CHUNKING=0` for the sliding-window baseline (ablation). |
| Embedding model | **Selectable** (Configuration dropdown / `EMBEDDING_MODEL`): `intfloat/multilingual-e5-base` (768d) or `BAAI/bge-m3` (1024d) | The query/passage convention is dispatched per model in `app/vector_rag.py`: **E5** prepends `"query: "` / `"passage: "`; **bge-m3** uses no prefix (generic path). Vectors are L2-normalised. Switching models is **non-destructive** — each model gets its own Pinecone index (per dimension, `retrievallab-768` / `retrievallab-1024`) and namespace (per index-time config), so both stay built and queryable; flipping back is free (see §Incremental builds). bge-m3 is the stronger cross-lingual option for EN-query / DE-corpus retrieval. (An encode path for `jinaai/jina-embeddings-v3` also exists, but Jina's remote code targets transformers 4.x and breaks on the 5.x used here, so it is not offered in the dropdown.) |
| Vector database | **Pinecone** (serverless, cosine metric) | One index per embedder dimension, auto-created (`retrievallab-768`, `retrievallab-1024`). A rebuild **never wipes the namespace**: only the old vectors of the documents being re-indexed are deleted by id (ids recorded in `data/index_manifest.json`), then the new chunks are upserted in batches of 64; chunk text (first 2,500 chars) is stored in vector metadata. Vector ids are **content-addressed** — `<doc-slug>-p<page>-<sha1(text)[:10]>`, hashed over the full document name, page and chunk text — so the same text always yields the same id (two builds are diffable), different text always yields a different id (an id that changed means the text changed), and documents whose names collide after slug truncation can no longer overwrite each other. Exact duplicate text on one page gets a `#2` suffix rather than being silently overwritten. |
| Retrieval | **top-k** nearest neighbours | Default k=5 (UI-adjustable 1–10); similarity scores surfaced in the UI and the log. |
| Generation | Single grounded LLM call | Answer ONLY from passages; `[n]` citations; explicit "not found" when absent; answer language injected (see §5). |

## 4. Pipeline B — Reasoning-Based Retrieval components

| Component | Choice | Detail |
|---|---|---|
| Hierarchical index | **PageIndex-style tree** (inspired by [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)) | Each document becomes a tree of nodes `{id, title, one-line summary, page range, children}`. When the document offers structure, two candidate trees are built deterministically — one from the clause index (if ≥ 8 clause headings are detected, `app/clauses.py`) and one from the PDF outline/TOC — and the one with finer leaves (lower average pages per leaf; on a tie, more nodes) is kept. Only when neither exists does an LLM structuring pass over per-page previews build the tree; fixed 5-page windows are the last resort (no API key, or the structuring call fails). One batched LLM call then writes one-line summaries for up to 60 nodes that lack one; nodes beyond that keep only their title. |
| Agent framework | **LangGraph — explicit `StateGraph`** | Not the prebuilt agent: the graph is hand-wired (`START → agent ⇄ tools → END` via `tools_condition`), which makes the architecture fully inspectable and citable. Step cap enforced via `recursion_limit` (1 step = agent+tools pair); on hitting the limit a forced final answer is produced from the evidence gathered. |
| Tools (lexical only) | `list_documents`, `get_tree`, `read_node`, `read_pages`, `keyword_search`, `list_clauses`, `read_clause` | Tree navigation + raw page reads + plain term-frequency string search (`keyword_search`: raw substring hit counts per page, top 8 pages with snippets — no IDF or length normalisation, so not BM25) + **clause-number navigation** (`list_clauses`/`read_clause`, Step 9) for DIN/EN standards. Deliberately **no embeddings** → paradigm isolation. Per-call caps (5 pages / ~8 k chars) protect the context window. |
| Planning & stopping | Prompt-enforced | State the goal before each tool call; decompose multi-part questions; stop when evidence suffices; cite `(Document, p.X)`; admit when the documents don't answer. |
| Trace | Full reasoning trace | Every thought, tool call (name + args) and observation preview is captured, rendered in the UI and written to the log — the primary data for qualitative analysis. |

## 5. Answer-language handling

The question's language is detected server-side with `langdetect`
(deterministic seed; only trusted at ≥0.80 confidence and ≥12 chars, otherwise
a mirror-the-question fallback). An explicit instruction like *"Write your
final answer in Turkish."* is appended to every system prompt in both
pipelines — this removes the wrong-language answers observed in v1.

## 6. LLM

Default `gpt-5.4-mini` (OpenAI Chat Completions), configured in `.env`.
gpt-5.x quirks: `max_completion_tokens` instead of `max_tokens`, and no custom
temperature (B's LangChain client sends `temperature=1`, the default value).
`reasoning_effort` (default `low`, settable in `.env` / the UI) is sent on every
call made through `app/llm.py` (A's answer, tree structuring and node summaries,
table summaries, the OpenAI judge) and on B's tool-less calls (forced final
answer, table re-grounding), but **not** on B's tool-calling agent turns —
gpt-5.x rejects `reasoning_effort` together with tools on Chat Completions
(see §2). The rules are applied in two places: `app/llm.py` and B's
`ChatOpenAI` client in `app/reasoning_agent.py`.

## 7. Experiment log

Every pipeline run is appended to `experiments.jsonl` (one JSON per line):
`run_id` links the A and B rows of the same question. Fields: question, model,
answer, retrieved chunks + scores (A), full trace + step/tool counts (B),
latency, token usage, detected answer language, and a full `config` snapshot.
Download via the UI button or `GET /log/download`.

### Evidence instrumentation

Each run also records **structured evidence page sets** (the raw material for
the metrics dashboard and the page heatmap):

- **A — retrieved:** `evidence.retrieved = {doc: [pages]}` — the unique pages
  whose chunks landed in the top-k.
- **B — read vs searched:** `evidence.read = {doc: [pages]}` are the pages whose
  **full text actually entered the agent's context** via `read_node`, `read_pages` or `read_clause`
  (respecting the tool's char cap, so a degenerate huge node only counts the
  pages that truly fit). `evidence.searched = {doc: [pages]}` are pages seen only
  as `keyword_search` snippets — a weaker, peeked-at signal, kept separate.

B's evidence is reconstructed by **replaying the tool calls** from the message
trace (pure, no shared run-state), so it stays correct under concurrent runs.
This lets Chapter 4 ask, per question, *"did the two paradigms look at the same
evidence, and who missed the gold pages?"*

## 8. API surface

| Method & path | Purpose |
|---|---|
| `GET /` | The web UI (`static/index.html`) |
| `GET /status` | Docs, readiness flags, models, key presence, current settings, condition (index) state |
| `GET /config`, `POST /config` | Read / update runtime configuration (see §17) |
| `POST /upload` | Multipart PDF/TXT/MD upload (PyMuPDF parse + table detection) |
| `POST /clear` | Remove all loaded sources and the persisted docs / state / manifest files (Pinecone vectors and the log are kept) |
| `POST /build_index` | Builds Pinecone vectors and/or PageIndex trees, only what is missing (`scope`, `docs`, `force`; see §17) |
| `GET /build_progress` | Progress of a running build |
| `GET /conditions` | Every index condition (namespace) built so far |
| `GET /table_image/{name}` | Rendered PNG of a detected table |
| `POST /ask/vector` | Pipeline A `{question, top_k, run_id}` — 409 if the current condition's namespace does not yet hold every document |
| `POST /ask/reasoning` | Pipeline B `{question, max_steps, run_id}` — 409 if any document has no tree (independent of Pipeline A's condition) |
| `GET /ask/reasoning/stream` | SSE variant of B — streams the trace live, then logs; same 409 rule as `POST /ask/reasoning` |
| `GET /page` | Full text of one page (backs clickable citations) |
| `POST /judge` | Blind LLM judgement of a logged run's A vs B |
| `POST /batch`, `GET /batch_progress`, `GET /batch/profiles` | Batch runner — `POST /batch` returns 409 if the condition or a tree is missing, or a batch is already running |
| `GET /metrics`, `GET /metrics/detail` | Metrics dashboard aggregates + the per-run records behind the drill-down |
| `GET /metrics.csv`, `GET /metrics.xlsx` | CSV export; 3-sheet Excel export (Summary, Detail, Answers) |
| `GET /trees/audit`, `GET /trees/detail`, `POST /trees/rebuild` | Tree quality audit, full tree of one document, re-structuring |
| `GET /questions`, `POST /questions` | Evaluation question set |
| `GET /log`, `GET /log/download`, `POST /log/clear` | Experiment log; clearing archives it to a timestamped `.bak` file |

## 9. Evaluation question set (gold_pages)

A labelled question set lives in `questions.json` (schema in
`questions.example.json`): each entry has an `id`, the `question` (optional
`question_de` bilingual variant), a free-form `type`
(factoid / multi-hop / comparison / numeric / cross-document / unanswerable / …),
an optional `content_type` (`prose` | `table` — where the answer lives; carried
into every log row and shown in the drill-down, CSV and Excel export), and
`gold_pages = {doc: [pages]}` — the pages that truly answer it. The **Question
set** panel in the UI lists, adds and deletes entries (persisted via
`GET`/`POST /questions`), can load any question back into the ask box, can
quick-fill gold pages from the last run's A-retrieved or B-read evidence to speed
up labelling, and bulk-imports (**Import JSON**, replace or merge) or downloads
(**Download JSON**) the whole set. `gold_pages` are the ground truth the batch
runner and metrics compare each pipeline's evidence set against.

## 10. Batch runner & variance

The **Batch run** control (in the Question set panel) drives every labelled
question through **A** then **B**, `repeat`× each, in a background thread with a
progress bar; each run is appended to the log tagged with `batch_id`,
`question_id`, `repeat_index`, `profile`, `lang`, `question_type`,
`content_type` and `gold_pages`.

- **Repeats** (`repeat=n`) counter gpt-5.x non-determinism — n runs per question
  yield mean ± spread rather than a single lucky/unlucky shot.
- **Profiles** provide budget-equalized ablations: `budget-b-low` /
  `budget-b-min` cap B to 3 / 2 steps ("does the reasoning advantage survive
  when cost is matched?"), `deep-b` gives it 12.

Headless / reproducible equivalent (reuses the persisted doc store; no server
needed once indexes are built). Run it from `retrievallab-py` with the venv
active; `--ids` takes question ids from `questions.json`:

```
python -m app.batch --repeat 3 --profile budget-b-low
python -m app.batch --ids hole-clearance-m20 slip-factor-class-a
```

Endpoints: `POST /batch`, `GET /batch_progress`, `GET /batch/profiles`.

## 11. Metrics dashboard & exports

The **Metrics dashboard** computes Chapter-4 numbers straight from the log,
pairing each question's A and B rows by `run_id` and scoring evidence against
`gold_pages`, broken down by **system × question-type** (`GET /metrics`; the
UI's **Download Excel** button gets a 3-sheet workbook — Summary, Detail,
Answers — from `GET /metrics.xlsx`; a CSV of the Detail layout is available at
`GET /metrics.csv`):

| Metric | Definition |
|---|---|
| recall | \|evidence ∩ gold\| / \|gold\| — did it reach the gold pages? |
| precision | \|evidence ∩ gold\| / \|evidence\| |
| F1 | 2·P·R / (P + R), computed per run, then averaged (macro) |
| MRR (A) | 1 / rank of the first gold page, ranking the top-k chunks' pages in order and counting a repeated page once |
| reads→gold (B) | 1-based position of the first gold page in B's read order |
| overlap | Jaccard(A retrieved, B read) — same evidence or not? |
| latency / tokens / cost / effort | per system (A = 1 call; B = tool-calls) |

All evidence metrics are page-level: A's top-k chunks are collapsed to their
unique pages, so precision's denominator is the number of distinct retrieved
pages, not k.

Only evaluation runs (those carrying `gold_pages`, i.e. batch runs) are scored;
ad-hoc interactive queries are ignored. Unanswerable controls are excluded from
recall/precision means and counted separately. The dashboard shows
**mean ± std** per system × question-type; the std is taken over all runs in
that group (different questions and their repeats pooled), not per question
across repeats. The per-question drill-down (click a row's *n*), the CSV and
the Excel Detail sheet report a question's **mean over its repeats** (no std).
Cost columns appear in the dashboard only when
`COST_PER_1M_IN`/`COST_PER_1M_OUT` are set in `.env` (no fabricated prices);
the CSV and Excel exports have no cost columns.

## 12. Evidence heatmap

The **Evidence heatmap** panel draws each document as a two-lane page strip —
**lane A** = pages retrieved by vector RAG (teal), **lane B** = pages the agent
read (purple; lighter = seen only as a keyword snippet) — sharing the page axis,
with gold pages framed and marked by a small caret above the strip (labelled ◇
in the legend). A filled column in both lanes means the
paradigms looked at the *same* page; a framed column with an empty lane is a
gold page that system **missed**. Hover any cell for its page and roles.

It renders the last interactive query automatically (gold pulled from the
question set when the question matches) and can visualise any of the 60 most
recent logged evaluation runs via the selector. **Export SVG** produces a standalone, crisp vector figure for
the thesis. This is the single clearest answer to *"do the two paradigms look at
the same evidence, and who misses?"*.

## 13. LLM judge (blind A vs B)

A **blind, paired, randomized-order** LLM judge scores the two answers
(`POST /judge {run_id}`; **Blind-judge A vs B** button after an interactive run;
`judge` toggle / `--judge` flag for batches; `python -m app.judge_batch` for a
batch that is already logged, §18). The judge sees only "Answer 1" /
"Answer 2" in a randomized order — reproducible when `RUN_SEED` is set (seeded
per run), otherwise random — rates each on grounded/complete/correct,
and picks a winner or tie; the verdict is mapped back to A/B and logged
(`phase:"judge"`). The dashboard shows **A / B / tie win-rates** and the CSV
and Excel exports carry a `judge` column.

Cross-family judging (recommended for the thesis) mitigates self-preference
bias since the generator is GPT: set `JUDGE_PROVIDER=anthropic`, an
`ANTHROPIC_API_KEY`, **and `JUDGE_MODEL` to a Claude model id** (the default
`gpt-5.4` is passed to the Anthropic API unchanged), then `pip install anthropic`
(not in `requirements.txt`). Without an `ANTHROPIC_API_KEY` the OpenAI judge is
used. Documented caveat: A's `[n]`
vs B's `(Doc, p.X)` citation styles can leak pipeline identity — the randomized
order and blind labels are the primary controls.

## 14. Domain customization & bilingual runs (Step 9)

*Note: the tool set is an experiment variable — these were added and are meant to
be **frozen before final data collection**.*

- **Clause-aware tools** (`clauses.py`): DIN/EN standards are organized by
  numbered clauses (7.4.1, Annex B). A clause-heading index — taken from the PDF
  outline when it carries at least 5 numbered entries, otherwise from a lexical
  regex scan of the page text — gives pipeline B two structure-native tools — `list_clauses(doc)` and
  `read_clause(doc, "7.4.1")` (reads a clause up to the next heading) — a
  domain-adapted capability, still embedding-free. Built at index time and
  persisted; clause reads count toward B's evidence `read` set.
- **Bilingual question pairs:** a question may carry a `question_de` variant
  (UI field or `questions.json`). The batch `lang` selector / `--lang` flag runs
  **EN**, **DE**, or **EN+DE** (`--lang en|de|both`); every run is tagged with `lang` for the
  cross-lingual robustness analysis.
- **German morphology (ablation):** `KEYWORD_STEM_DE=1` lightly de-suffixes
  `keyword_search` terms so `Beschichtungen` also matches `Beschichtung(sdicke)`
  — a small on/off recall ablation, default off.

## 15. Tree quality auditor (Step 10)

The sidebar **Tree quality** panel reports each document's tree — source, node /
leaf counts, average pages per leaf, largest leaf span — and flags
**degenerate** structures (e.g. 174 pages / 4 nodes) that force pipeline B to
read huge, undiscriminating ranges. Per document, **view structure** shows the
full tree (`GET /trees/detail`), **rebuild with LLM** re-runs the PageIndex
structuring pass, and **rebuild from clauses** rebuilds it deterministically
from the clause index (`GET /trees/audit` — which also returns the tree depth —,
`POST /trees/rebuild {doc, method}` where method ∈ llm|clauses|toc|windows|auto);
the new tree is persisted. This both improves the system and makes the thesis's
"structure dependence" finding measurable.

## 16. Showcase / defense-day features (Step 11)

- **Live reasoning trace (SSE):** with **live trace** on, pipeline B streams each
  thought / tool-call / observation to the screen as it happens
  (`GET /ask/reasoning/stream`), then logs identically to the POST endpoint.
- **Clickable citations:** A's `[n]` and B's `(Doc, p.X)` are clickable — they
  open the actual page text in a side drawer (`GET /page`), the live proof that
  "the answer is verifiable".
- **Case report export:** one click builds a **self-contained HTML** file —
  question, both answers, evidence sets, full trace, judge verdict, config
  snapshot and the embedded heatmap SVG — that prints straight to PDF. The
  thesis appendix generates itself.

## 17. Configuration tab (runtime settings)

The sidebar **Configuration** panel edits a whitelisted subset of settings live,
without restarting the backend (`GET`/`POST /config`). Two classes:

- **Query-time** — `model`, `reasoning_effort`, `seed`, `top_k`, `max_steps`
  (in the panel) and `attach_table_images` (no UI control). Take effect on the
  next query.
- **Index-time** — `chunk_size`, `chunk_overlap`, `embedding_model` (in the
  panel) and `clause_chunking`, `table_summaries` (no UI control). These no
  longer invalidate anything:
  each combination gets **its own Pinecone namespace** (`main-<sha1(settings)[:8]>`)
  and each embedding dimension its own index (`retrievallab-768`), so switching a
  setting selects another condition instead of destroying the current one. If
  that condition was built before, it is queried immediately; if not, only the
  missing documents are embedded. Until this condition holds every document,
  the UI shows *"this condition is not built yet"* (listing what is missing) and
  `POST /ask/vector` and `POST /batch` return **409**. Pipeline B is not gated by
  the condition: it only needs every document to have a tree.

Settings without a UI control are set in `.env` (startup defaults) or with
`POST /config`, e.g. `{"clause_chunking": false}`.

### Incremental builds

`POST /build_index` accepts `{"scope": "vector"|"trees"|"both", "docs": [...],
"force": false}` and rebuilds **only what is missing**:

| Change | What is recomputed |
|---|---|
| `top_k`, `max_steps`, model, seed | nothing — query-time parameters never touch the index |
| `chunk_size`, `embedding_model`, … | the new condition's namespace only; the previous one stays queryable |
| switching back to a previous condition | nothing — it is already built |
| a new document | that document only (`{"docs": ["new.pdf"]}`) |
| a document whose content changed | that document only — its old vectors are deleted **by id**, the namespace is never wiped |
| a Pipeline-A parameter | no LLM calls for Pipeline B — trees are only built for documents that lack one (`scope`) |

What is already embedded is tracked in `data/index_manifest.json`
(`{namespace: {settings, index, dimension, docs: {name: {signature, chunks, ids}}}}`).
The per-document *signature* covers its page text and table summaries, so an
unchanged re-upload is a no-op. `GET /conditions` lists every condition built so
far; the UI shows them under the Configuration panel.

Every experiment-log row now carries a full `config` snapshot — model,
reasoning effort, **seed**, chunking, embedding model, Pinecone namespace,
top-k/max-steps. The snapshot's `top_k`/`max_steps` are the server's current
defaults; the values a run actually used are in the row's own `top_k` (A) and
`max_steps` (B) fields. They differ when a request or a batch profile overrides
them (e.g. a `budget-b-low` run logs `max_steps: 3` while its snapshot shows the
default). Together, the snapshot and these fields make each result reproducible
from its own record. Setting a `seed` (UI or `RUN_SEED` in `.env`) requests
best-effort determinism from OpenAI.

## 18. Offline evaluation tools (CLI)

Run from `retrievallab-py` with the venv active, on a batch that is already in
the log (the pipelines are not re-run); each tool appends its own rows to
`experiments.jsonl` and skips run_ids it has already scored, so an interrupted
run can be restarted.

| Command | What it does |
|---|---|
| `python -m app.judge_batch --batch <batch_id> [--round N] [--workers 6] [--limit n]` | Blind pairwise judge (§13) over a logged batch. Rounds (`--round 2`, `3`, …) re-judge with a re-randomized order to measure judge consistency. Rows: `phase: "judge"` with `judge_round`. |
| `python -m app.ragas_eval --batch <batch_id> --build-contexts`, then `--round N [--workers 8] [--limit n]` | RAGAS metrics (Es et al., 2024): faithfulness, answer relevance (n = 3 generated questions, `text-embedding-ada-002`) and context relevance; evaluator model = `JUDGE_MODEL`. `--build-contexts` saves exactly the text each pipeline had in front of it (A: top-k chunks from Pinecone; B: replayed tool outputs) to `data/ragas_contexts_<batch>.json`. Rows: `phase: "ragas"`. |
| `python -m app.judge_ref --batch <batch_id> --round 1 (--pilot N \| --submit \| --collect <msgbatch_id>)` | Reference-guided 1–5 rubric judge, each answer scored on its own: groundedness (against that pipeline's context) and answer correctness + completeness (against `data/reference_answers.json`). Needs the contexts file from `ragas_eval --build-contexts`. Uses the Anthropic API (`ANTHROPIC_API_KEY`, `pip install anthropic`; model set in the module, `claude-sonnet-5-5`); `--submit` / `--collect` go through the Message Batches API. Rows: `phase: "judge_ref"`. |

The root-level scripts run the single-pipeline ablations (`ablate_a.py`,
`ablate_b.py`), compare an ablation batch with the baseline (`compare_a.py`) and
export tables (`ablation_tables.py`, `ablation_xlsx.py`, `answer_quality_xlsx.py`,
`judge_ref_xlsx.py`, `ragas_xlsx.py`).

## 19. Related literature (starter set)

Lewis et al. 2020 (RAG) · Yao et al. 2022 (ReAct) · Wang et al. 2022 (E5) ·
Asai et al. 2023 (Self-RAG) · Trivedi et al. 2023 (IRCoT) ·
VectifyAI PageIndex (vectorless, reasoning-based RAG) · BRIGHT benchmark
(reasoning-intensive retrieval).
