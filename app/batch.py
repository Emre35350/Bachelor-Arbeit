"""Batch runner — drive the whole question set through both pipelines.

Produces the rows Chapter 4 is written from. Each run is appended to the
experiment log tagged with batch/question metadata AND the question's
gold_pages + type, so the metrics module can compute recall/precision/MRR and
per-type breakdowns straight from the log.

Two entry points over one core:
  * POST /batch  — runs in a background thread, GET /batch_progress polls it (UI).
  * python -m app.batch — headless/reproducible run; restores the persisted doc
    store first (Pinecone vectors persist in the cloud), so no server needed.

Repeats (`repeat=n`) address gpt-5.x non-determinism: n runs per question give
mean ± spread instead of a single lucky/unlucky shot. Profiles provide
budget-equalized ablations (e.g. cap B to a low step budget to test whether the
reasoning advantage survives when cost is matched).
"""
import threading
import time
import uuid

from . import config, ingestion, judge, logstore, questions, reasoning_agent, vector_rag

# name -> parameter overrides applied before per-call explicit params
PROFILES = {
    "default": {},
    "budget-b-low": {"max_steps": 3},   # B on a tight step budget (ablation)
    "budget-b-min": {"max_steps": 2},
    "deep-b": {"max_steps": 12},
}

STATE = {
    "running": False, "done": 0, "total": 0, "batch_id": None,
    "current": "", "profile": None, "repeat": 1,
    "started_at": None, "finished_at": None, "error": None, "results": [],
}
_thread: threading.Thread | None = None


def _reset(total: int, batch_id: str, profile: str, repeat: int) -> None:
    STATE.update(running=True, done=0, total=total, batch_id=batch_id,
                 current="", profile=profile, repeat=repeat,
                 started_at=time.time(), finished_at=None, error=None, results=[])


def resolve_params(profile: str, top_k=None, max_steps=None) -> tuple[int, int]:
    prof = PROFILES.get(profile, {})
    tk = int(top_k or prof.get("top_k") or config.DEFAULT_TOP_K)
    ms = int(max_steps or prof.get("max_steps") or config.DEFAULT_MAX_STEPS)
    return tk, ms


def _variants(q: dict, lang: str) -> list[tuple]:
    """(lang_code, question_text) pairs to run for one question."""
    out = []
    if lang in ("en", "both"):
        out.append(("en", q["question"]))
    if lang in ("de", "both") and q.get("question_de"):
        out.append(("de", q["question_de"]))
    return out or [("en", q["question"])]


def run(question_ids=None, repeat=1, profile="default", top_k=None, max_steps=None,
        judge_runs=False, lang="en", answer_vector=None, answer_reasoning=None,
        judge_fn=None, on_progress=None) -> dict:
    """Core loop. Pipeline fns are injectable for testing; default to the real ones."""
    answer_vector = answer_vector or vector_rag.answer
    answer_reasoning = answer_reasoning or reasoning_agent.answer
    judge_fn = judge_fn or judge.judge_pair

    qs = questions.load()
    if question_ids:
        wanted = set(question_ids)
        qs = [q for q in qs if q["id"] in wanted]
    if not qs:
        raise ValueError("no questions to run (empty set or no id match)")

    repeat = max(1, int(repeat))
    tk, ms = resolve_params(profile, top_k, max_steps)
    batch_id = "batch-" + uuid.uuid4().hex[:8]
    plan = [(q, lc, qt) for q in qs for (lc, qt) in _variants(q, lang)]
    _reset(len(plan) * repeat, batch_id, profile, repeat)

    for q, lang_code, qtext in plan:
        for r in range(1, repeat + 1):
            STATE["current"] = f"{q['id']} [{lang_code}] (run {r}/{repeat})"
            run_id = f"{batch_id}:{q['id']}:{lang_code}:{r}"
            meta = {"run_id": run_id, "batch_id": batch_id, "question_id": q["id"],
                    "question_type": q.get("type", "other"),
                    "content_type": q.get("content_type"),   # prose|table, for the breakdown
                    "repeat_index": r,
                    "profile": profile, "lang": lang_code,
                    "gold_pages": q.get("gold_pages", {}),
                    "question": qtext, "model": config.OPENAI_MODEL}

            row = {"question_id": q["id"], "repeat": r, "lang": lang_code}
            answers = {}
            for phase, fn, arg in (("vector", answer_vector, tk),
                                   ("reasoning", answer_reasoning, ms)):
                try:
                    res = fn(qtext, arg)
                    err = None
                except Exception as e:  # one bad run must not kill the batch
                    res, err = {"error": str(e)}, str(e)
                logstore.append({**meta, "phase": phase, **res})
                answers[phase] = res.get("answer", "")
                row[phase] = {"error": err,
                              "latency_ms": res.get("latency_ms"),
                              "evidence": res.get("evidence")}

            if judge_runs and not row["vector"]["error"] and not row["reasoning"]["error"]:
                try:
                    verdict = judge_fn(qtext, answers["vector"],
                                       answers["reasoning"], seed_material=run_id)
                    logstore.append({**meta, "phase": "judge", **verdict})
                    row["judge"] = verdict.get("winner")
                except Exception as e:
                    row["judge_error"] = str(e)

            STATE["results"].append(row)
            STATE["done"] += 1
            if on_progress:
                on_progress(STATE)

    STATE["running"] = False
    STATE["finished_at"] = time.time()
    STATE["current"] = "done"
    return {"batch_id": batch_id, "runs": STATE["done"], "profile": profile,
            "top_k": tk, "max_steps": ms, "repeat": repeat}


def start_background(**kwargs) -> str:
    """Launch run() in a daemon thread; returns the batch_id. One at a time."""
    global _thread
    if STATE["running"]:
        raise RuntimeError("a batch is already running")
    batch_id = "batch-" + uuid.uuid4().hex[:8]
    STATE.update(running=True, error=None, batch_id=batch_id, done=0, total=0)

    def _worker():
        try:
            run(**kwargs)
        except Exception as e:
            STATE.update(running=False, error=str(e), finished_at=time.time())

    _thread = threading.Thread(target=_worker, daemon=True)
    _thread.start()
    return batch_id


# ---------------------------------------------------------------------- CLI
def _main() -> None:
    import argparse

    from . import store
    parser = argparse.ArgumentParser(description="RetrievalLab batch runner")
    parser.add_argument("--repeat", type=int, default=1, help="runs per question (variance)")
    parser.add_argument("--profile", default="default", choices=list(PROFILES),
                        help="parameter profile / budget ablation")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ids", nargs="*", default=None, help="subset of question ids")
    parser.add_argument("--judge", action="store_true", help="blind-judge A vs B per run")
    parser.add_argument("--lang", default="en", choices=["en", "de", "both"],
                        help="question language variant(s) to run")
    args = parser.parse_args()

    if config.PERSIST:
        store.load_docs()  # restore persisted pages + trees
    if not ingestion.DOCS:
        raise SystemExit("No documents loaded. Build the indexes in the app first "
                         "(persisted state is reused here).")

    def _tick(st):
        print(f"  [{st['done']}/{st['total']}] {st['current']}", flush=True)

    print(f"Running batch: profile={args.profile} repeat={args.repeat}")
    summary = run(question_ids=args.ids, repeat=args.repeat, profile=args.profile,
                  top_k=args.top_k, max_steps=args.max_steps, judge_runs=args.judge,
                  lang=args.lang, on_progress=_tick)
    print(f"Done: {summary['runs']} runs -> {config.LOG_PATH} "
          f"(batch_id={summary['batch_id']}, top_k={summary['top_k']}, "
          f"max_steps={summary['max_steps']})")


if __name__ == "__main__":
    _main()
