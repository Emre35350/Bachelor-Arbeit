"""Pipeline-A-only ablation runner.

Runs ONLY Vector-Based RAG over a question set and appends one log row per run,
in exactly the shape batch.py writes for phase "vector", so the existing metrics
code reads these runs unchanged.

Why this exists: top_k, chunking and the embedding model do not touch
Reasoning-Based Retrieval at all, so running phase B for these ablations costs
about twelve times the tokens and answers nothing. batch.py always runs both
phases; this script runs one. It does not modify the prototype.

Usage (from the retrievallab-py folder):

    python ablate_a.py --dry-run          # show the condition, run nothing
    python ablate_a.py --top-k 3
    python ablate_a.py --top-k 10
    python ablate_a.py                    # uses DEFAULT_TOP_K from .env

The condition (embedding model, chunking, namespace) comes from .env, exactly
as it does for the app. Always check the header before letting a run start.
"""
import argparse
import time
import uuid

from app import config, ingestion, logstore, questions, store, vector_rag


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Vector-Based RAG only (ablations)")
    ap.add_argument("--questions", default="questions.numeric-multihop.json",
                    help="question set file (default: the 41-question subset)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="retrieved neighbours (default: DEFAULT_TOP_K from .env)")
    ap.add_argument("--repeat", type=int, default=3, help="runs per question")
    ap.add_argument("--tag", default="ablation-a",
                    help="label written into the profile field of each row")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the condition and exit without calling anything")
    args = ap.parse_args()

    config.QUESTIONS_PATH = args.questions
    qs = questions.load()
    if not qs:
        raise SystemExit(f"No questions found in {args.questions}")

    if config.PERSIST:
        store.load_docs()
    if not ingestion.DOCS:
        raise SystemExit("No documents loaded. Build the indexes in the app first.")

    top_k = args.top_k or config.DEFAULT_TOP_K
    repeat = max(1, args.repeat)
    total = len(qs) * repeat
    batch_id = "batch-" + uuid.uuid4().hex[:8]

    print("-" * 72)
    print("PIPELINE A ONLY — ablation run")
    print("-" * 72)
    print(f"  question set     : {args.questions}  ({len(qs)} questions)")
    print(f"  repeats          : {repeat}   ->  {total} runs")
    print(f"  top_k            : {top_k}")
    print(f"  embedding model  : {config.EMBEDDING_MODEL}")
    print(f"  chunking         : clause_chunking={config.CLAUSE_CHUNKING}  "
          f"size={config.CHUNK_SIZE} overlap={config.CHUNK_OVERLAP}")
    print(f"  NAMESPACE        : {config.namespace()}")
    print(f"  generator        : {config.OPENAI_MODEL} (effort {config.REASONING_EFFORT})")
    print(f"  log              : {config.LOG_PATH}")
    print(f"  batch_id         : {batch_id}")
    print("-" * 72)

    if args.dry_run:
        print("dry run — nothing was called, nothing was written.")
        return

    t0 = time.time()
    done = failed = 0
    tok_in = tok_out = 0

    for q in qs:
        for r in range(1, repeat + 1):
            run_id = f"{batch_id}:{q['id']}:en:{r}"
            meta = {"run_id": run_id, "batch_id": batch_id, "question_id": q["id"],
                    "question_type": q.get("type", "other"),
                    "content_type": q.get("content_type"),
                    "repeat_index": r,
                    "profile": args.tag, "lang": "en",
                    "gold_pages": q.get("gold_pages", {}),
                    "question": q["question"], "model": config.OPENAI_MODEL}
            try:
                res = vector_rag.answer(q["question"], top_k)
                u = res.get("usage") or {}
                tok_in += u.get("in", 0)
                tok_out += u.get("out", 0)
            except Exception as e:            # one bad run must not kill the run
                res = {"error": str(e)}
                failed += 1
            logstore.append({**meta, "phase": "vector", **res})
            done += 1
            print(f"  [{done}/{total}] {q['id']} (run {r}/{repeat})"
                  f"{'  ERROR: ' + res['error'][:60] if res.get('error') else ''}",
                  flush=True)

    mins = (time.time() - t0) / 60
    print("-" * 72)
    print(f"done: {done} runs ({failed} failed) in {mins:.1f} min")
    print(f"tokens: {tok_in:,} in + {tok_out:,} out = {tok_in + tok_out:,}")
    print(f"batch_id: {batch_id}   namespace: {config.namespace()}   top_k: {top_k}")
    print("-" * 72)


if __name__ == "__main__":
    main()
