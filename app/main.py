"""RetrievalLab backend — FastAPI app wiring the two retrieval pipelines.

Sequential protocol: the frontend calls /ask/vector first, renders the result,
then calls /ask/reasoning with the same run_id. Pipeline B never receives
pipeline A's answer or retrieved chunks.
"""
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, PlainTextResponse, Response,
                               StreamingResponse)
from pydantic import BaseModel

from . import (batch, clauses, config, ingestion, judge, logstore, metrics,
               pageindex, progress, questions, reasoning_agent, store, tables,
               vector_rag)

app = FastAPI(title="RetrievalLab", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
# built_with = index-time config snapshot at the last successful build;
# diverging from the current settings marks the indexes STALE.
STATE = {"vector_ready": False, "trees_ready": False, "built_with": None,
         "restored": False}


@app.on_event("startup")
def _restore_from_disk():
    """Reload persisted docs/trees + build state so a restart doesn't require
    a full re-upload/rebuild. Pinecone vectors already persist in the cloud."""
    if not config.PERSIST:
        return
    n = store.load_docs()
    saved = store.load_state()
    if saved:
        STATE.update({k: saved.get(k, STATE[k])
                      for k in ("vector_ready", "trees_ready", "built_with")})
    if n:
        STATE["restored"] = True


class AskVector(BaseModel):
    question: str
    top_k: int | None = None
    run_id: str | None = None


class AskReasoning(BaseModel):
    question: str
    max_steps: int | None = None
    run_id: str | None = None


def _index_state() -> dict:
    """Readiness of the CURRENT configuration's namespace.

    Index-time settings no longer invalidate anything: each configuration has its
    own namespace, so switching one either lands on a namespace that is already
    built (query it immediately) or on one that still needs building. Nothing is
    ever destroyed, so "stale" now means "this condition is not built yet",
    listing the documents that are missing rather than a diff to rebuild from.
    """
    if not ingestion.DOCS:
        return {"stale": False, "diffs": [], "namespace": config.namespace(),
                "missing": [], "ready": False}
    st = vector_rag.namespace_state()
    diffs = [f"{n}: not embedded in this condition" for n in st["missing"]]
    return {"stale": not st["ready"], "diffs": diffs, "namespace": st["namespace"],
            "missing": st["missing"], "ready": st["ready"],
            "chunks": st["chunks"]}


def _reject_if_stale():
    st = _index_state()
    if st["stale"]:
        raise HTTPException(
            409, f"Condition {st['namespace']} is not built for "
            f"{len(st['missing'])} document(s) ("
            + ", ".join(st["missing"][:3])
            + ("…" if len(st["missing"]) > 3 else "")
            + "). Build the index for this configuration first — other "
              "conditions are untouched.")


def _reject_if_trees_missing():
    """Pipeline B's gate — it reads trees and page text, never Pinecone.

    Kept separate from the vector check so an unbuilt Pipeline-A condition does
    not block Pipeline B (and vice versa); that is the point of decoupling them.
    """
    missing = [d["name"] for d in ingestion.DOCS.values() if not d.get("tree")]
    if missing:
        raise HTTPException(
            409, f"Pipeline B has no tree for {len(missing)} document(s) ("
            + ", ".join(missing[:3]) + ("…" if len(missing) > 3 else "")
            + "). Build with scope='trees' (only those documents are processed).")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/status")
def status():
    return {
        "docs": ingestion.list_documents(),
        "vector_ready": STATE["vector_ready"],
        "trees_ready": STATE["trees_ready"],
        "model": config.OPENAI_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "pinecone_index": config.PINECONE_INDEX,
        "keys": {"openai": bool(config.OPENAI_API_KEY),
                 "pinecone": bool(config.PINECONE_API_KEY)},
        "settings": config.get_settings(),
        "embedding_models": config.KNOWN_EMBEDDING_MODELS,
        "index": _index_state(),
        "restored": STATE["restored"],
    }


@app.get("/table_image/{name}")
def table_image(name: str):
    """Serve a rendered table PNG so the UI can show what the model was given."""
    path = (tables.image_dir() / name).resolve()
    if path.parent != tables.image_dir().resolve() or not path.exists():
        raise HTTPException(404, "no such table image")
    return FileResponse(path, media_type="image/png")


@app.get("/config")
def get_config():
    return {"settings": config.get_settings(),
            "built_with": STATE["built_with"], **_index_state()}


@app.post("/config")
def set_config(changes: dict):
    """Partial runtime-settings update from the Configuration tab."""
    try:
        applied = config.update_settings(changes or {})
    except ValueError as e:
        raise HTTPException(400, str(e))
    if "embedding_model" in applied:
        vector_rag.reset_state()
    return {"applied": applied, "settings": config.get_settings(),
            **_index_state()}


@app.post("/upload")
def upload(files: list[UploadFile] = File(...)):
    added = []
    for f in files:
        name = f.filename or "unnamed"
        data = f.file.read()
        try:
            if name.lower().endswith(".pdf"):
                ingestion.add_pdf(name, data)
            elif name.lower().endswith((".txt", ".md")):
                ingestion.add_text(name, data.decode("utf-8", "replace"))
            else:
                raise ValueError("unsupported file type (pdf/txt/md only)")
        except Exception as e:
            raise HTTPException(400, f"{name}: {e}")
        added.append(name)
    # only the new documents need work: everything already embedded stays valid,
    # so readiness is recomputed rather than reset
    st = _index_state()
    STATE["vector_ready"] = st["ready"]
    STATE["trees_ready"] = all(d.get("tree") for d in ingestion.DOCS.values())
    store.save_docs()
    store.save_state(STATE)
    return {"added": added, "docs": ingestion.list_documents(),
            "pending": st["missing"]}


@app.post("/clear")
def clear():
    ingestion.clear()
    STATE["vector_ready"] = False
    STATE["trees_ready"] = False
    STATE["built_with"] = None
    store.clear()
    return {"ok": True}


class BuildRequest(BaseModel):
    scope: str = "both"          # "vector" (A) | "trees" (B) | "both"
    docs: list[str] | None = None  # restrict to these documents
    force: bool = False          # ignore the manifest and rebuild anyway


@app.post("/build_index")
def build_index(body: BuildRequest | None = None):
    """Build what is missing — nothing more.

    Pipeline A and Pipeline B are decoupled (`scope`), the work is per document
    (`docs`), and both skip anything already built. Testing a Pipeline-A
    parameter therefore costs no LLM calls for B, and adding an 11th document
    processes only that document.
    """
    body = body or BuildRequest()
    if body.scope not in ("both", "vector", "trees"):
        raise HTTPException(400, "scope must be 'vector', 'trees' or 'both'")
    if not ingestion.DOCS:
        raise HTTPException(400, "Upload at least one document first.")
    unknown = [n for n in (body.docs or []) if n not in ingestion.DOCS]
    if unknown:
        raise HTTPException(400, f"unknown document(s): {', '.join(unknown)}")
    if not config.OPENAI_API_KEY:
        raise HTTPException(400, "OPENAI_API_KEY is missing in .env")
    if body.scope in ("both", "vector") and not config.PINECONE_API_KEY:
        raise HTTPException(400, "PINECONE_API_KEY is missing in .env")

    progress.reset()
    vec, trees, t_vec, t_tree = None, None, 0.0, 0.0
    try:
        if body.scope in ("both", "vector"):
            t0 = time.time()
            vec = vector_rag.build_index(only=body.docs, force=body.force)
            t_vec = round(time.time() - t0, 1)

        if body.scope in ("both", "trees"):
            t1 = time.time()
            trees = pageindex.build_all_trees(only=body.docs, force=body.force)
            t_tree = round(time.time() - t1, 1)

        for d in ingestion.DOCS.values():  # clause index (Step 9), persisted
            if d.get("clauses") is None or body.force:
                d["clauses"] = clauses.build(d)
    except Exception as e:
        progress.error(str(e))
        raise HTTPException(500, f"Index build failed: {e}")

    progress.set_phase("done", 1, "Build complete.")
    st = _index_state()
    STATE["vector_ready"] = st["ready"]
    STATE["trees_ready"] = all(d.get("tree") for d in ingestion.DOCS.values())
    STATE["built_with"] = config.index_settings()
    store.save_docs()   # trees are now attached to the doc store
    store.save_state(STATE)
    return {"scope": body.scope,
            "vector": {**vec, "seconds": t_vec} if vec else None,
            "trees": ({**trees, "docs": trees["built"], "seconds": t_tree}
                      if trees else None),
            "conditions": vector_rag.known_conditions()}


@app.get("/conditions")
def conditions():
    """Every configuration namespace built so far — none is ever overwritten."""
    return {"current": config.namespace(), "settings": config.index_settings(),
            "conditions": vector_rag.known_conditions()}


@app.get("/build_progress")
def build_progress():
    return progress.STATE


@app.post("/ask/vector")
def ask_vector(body: AskVector):
    _reject_if_stale()           # A depends only on its own condition
    run_id = body.run_id or str(uuid.uuid4())
    try:
        result = vector_rag.answer(body.question, body.top_k or config.DEFAULT_TOP_K)
    except Exception as e:
        raise HTTPException(500, f"Vector pipeline failed: {e}")
    logstore.append({"run_id": run_id, "phase": "vector",
                     "question": body.question, "model": config.OPENAI_MODEL,
                     **result})
    return {"run_id": run_id, **result}


@app.post("/ask/reasoning")
def ask_reasoning(body: AskReasoning):
    _reject_if_trees_missing()   # B does not depend on Pipeline A's namespace
    run_id = body.run_id or str(uuid.uuid4())
    try:
        result = reasoning_agent.answer(body.question, body.max_steps or config.DEFAULT_MAX_STEPS)
    except Exception as e:
        raise HTTPException(500, f"Reasoning pipeline failed: {e}")
    logstore.append({"run_id": run_id, "phase": "reasoning",
                     "question": body.question, "model": config.OPENAI_MODEL,
                     **result})
    return {"run_id": run_id, **result}


class QuestionSet(BaseModel):
    questions: list[dict]


@app.get("/questions")
def get_questions():
    return {"questions": questions.load(), "types": questions.QUESTION_TYPES}


@app.post("/questions")
def save_questions(body: QuestionSet):
    try:
        saved = questions.save(body.questions)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"questions": saved}


class BatchRequest(BaseModel):
    question_ids: list[str] | None = None
    repeat: int = 1
    profile: str = "default"
    top_k: int | None = None
    max_steps: int | None = None
    judge: bool = False
    lang: str = "en"


@app.post("/batch")
def start_batch(body: BatchRequest):
    _reject_if_stale()           # a batch runs BOTH pipelines
    _reject_if_trees_missing()
    if body.profile not in batch.PROFILES:
        raise HTTPException(400, f"unknown profile: {body.profile}")
    if not questions.load():
        raise HTTPException(400, "No questions in the set. Add some first.")
    try:
        batch_id = batch.start_background(
            question_ids=body.question_ids, repeat=body.repeat,
            profile=body.profile, top_k=body.top_k, max_steps=body.max_steps,
            judge_runs=body.judge, lang=body.lang)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    tk, ms = batch.resolve_params(body.profile, body.top_k, body.max_steps)
    return {"batch_id": batch_id, "profile": body.profile,
            "top_k": tk, "max_steps": ms, "repeat": max(1, body.repeat)}


@app.get("/batch_progress")
def batch_progress():
    return batch.STATE


@app.get("/batch/profiles")
def batch_profiles():
    return {"profiles": {k: v for k, v in batch.PROFILES.items()}}


class JudgeRequest(BaseModel):
    run_id: str


@app.post("/judge")
def judge_run(body: JudgeRequest):
    """Blind-judge the A vs B answers already logged under this run_id."""
    rows = logstore.read_all()
    a = next((r for r in reversed(rows)
              if r.get("run_id") == body.run_id and r.get("phase") == "vector"), None)
    b = next((r for r in reversed(rows)
              if r.get("run_id") == body.run_id and r.get("phase") == "reasoning"), None)
    if not a or not b:
        raise HTTPException(404, "Both A and B answers must be logged for this run_id.")
    try:
        verdict = judge.judge_pair(a.get("question", ""), a.get("answer", ""),
                                   b.get("answer", ""), seed_material=body.run_id)
    except Exception as e:
        raise HTTPException(500, f"Judge failed: {e}")
    logstore.append({"run_id": body.run_id, "phase": "judge",
                     "question": a.get("question", ""),
                     "question_id": a.get("question_id"),
                     "question_type": a.get("question_type"),
                     "batch_id": a.get("batch_id"), **verdict})
    return {"run_id": body.run_id, **verdict}


@app.get("/trees/audit")
def trees_audit():
    return {"docs": pageindex.audit_all()}


@app.get("/trees/detail")
def trees_detail(doc: str):
    """The full PageIndex tree of one document — what pipeline B navigates."""
    d = ingestion.get(doc)
    if not d:
        raise HTTPException(404, f"Document not found: {doc}")
    if not d.get("tree"):
        raise HTTPException(400, "Tree not built yet — build the indexes first.")
    return {"doc": d["name"], "pages": len(d["pages"]),
            "tree": d["tree"], "audit": pageindex.audit_tree(d),
            "clauses": len(clauses.index(d))}


class TreeRebuild(BaseModel):
    doc: str
    method: str = "llm"   # llm | toc | windows | auto


@app.post("/trees/rebuild")
def trees_rebuild(body: TreeRebuild):
    if not STATE["trees_ready"]:
        raise HTTPException(400, "Build the indexes first.")
    d = ingestion.get(body.doc)
    if not d:
        raise HTTPException(404, f"Document not found: {body.doc}")
    try:
        audit = pageindex.rebuild_tree(d, body.method)
    except Exception as e:
        raise HTTPException(500, f"Tree rebuild failed: {e}")
    store.save_docs()   # persist the new tree
    return {"audit": audit, "docs": ingestion.list_documents()}


@app.get("/metrics")
def get_metrics():
    return metrics.summarize(logstore.read_all())


@app.get("/metrics/detail")
def get_metrics_detail():
    """Per-run records behind the dashboard aggregates — drives the drill-down."""
    return {"runs": metrics.per_run(logstore.read_all())}


@app.get("/metrics.csv")
def get_metrics_csv():
    """Per-question metrics laid out like the drill-down popup (see metrics.to_csv)."""
    csv_text = metrics.to_csv(logstore.read_all())
    return PlainTextResponse(csv_text, media_type="text/csv", headers={
        "Content-Disposition": 'attachment; filename="retrievallab_metrics.csv"'})


@app.get("/metrics.xlsx")
def get_metrics_xlsx():
    """Three-sheet Excel: Summary (grouped), Detail (per question), Answers."""
    data = metrics.to_xlsx(logstore.read_all())
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="retrievallab_metrics.xlsx"'})


@app.get("/ask/reasoning/stream")
def ask_reasoning_stream(question: str, max_steps: int = config.DEFAULT_MAX_STEPS,
                         run_id: str | None = None):
    """SSE variant of /ask/reasoning: streams the reasoning trace live, then a
    'done' event. Logs the run on completion, exactly like the POST endpoint."""
    _reject_if_trees_missing()   # B does not depend on Pipeline A's namespace
    rid = run_id or str(uuid.uuid4())

    def gen():
        result = None
        try:
            for ev in reasoning_agent.answer_stream(question, max_steps):
                if ev.get("type") == "done":
                    result = ev["result"]
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if result is not None:
                logstore.append({"run_id": rid, "phase": "reasoning",
                                 "question": question, "model": config.OPENAI_MODEL,
                                 **result})
                yield f"data: {json.dumps({'type': 'logged', 'run_id': rid})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/page")
def get_page(doc: str, page: int):
    """Full text of one page — backs clickable citations (verifiability)."""
    d = ingestion.get(doc)
    if not d:
        raise HTTPException(404, f"Document not found: {doc}")
    match = next((p for p in d["pages"] if p["page"] == page), None)
    if not match:
        raise HTTPException(404, f"Page {page} not found in {d['name']}")
    return {"doc": d["name"], "page": page, "pages_total": len(d["pages"]),
            "text": match["text"]}


@app.post("/log/clear")
def clear_log():
    """Reset the metrics dashboard by archiving the current log (non-destructive)."""
    return {"cleared": True, **logstore.clear()}


@app.get("/log")
def get_log():
    return logstore.read_all()


@app.get("/log/download")
def download_log():
    p = Path(config.LOG_PATH)
    if not p.exists():
        raise HTTPException(404, "No experiments logged yet.")
    return FileResponse(p, filename="experiments.jsonl",
                        media_type="application/x-ndjson")
