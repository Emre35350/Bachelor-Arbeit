"""Pipeline-B-only ablation runner (step budget).

Runs ONLY Reasoning-Based Retrieval and appends one log row per run, in exactly
the shape batch.py writes for phase "reasoning", so the existing metrics code
reads these runs unchanged.

The step budget does not touch Vector-Based RAG, so phase A is not re-run.

Usage (from the retrievallab-py folder):

    python ablate_b.py --dry-run
    python ablate_b.py --profile budget-b-min     # 2 steps
    python ablate_b.py --profile budget-b-low     # 3 steps
    python ablate_b.py --profile deep-b           # 12 steps
    python ablate_b.py --max-steps 6              # any budget directly
"""
import argparse
import time
import uuid

print("loading libraries (10-30 s on first start, do not interrupt)...", flush=True)

from app import batch, config, ingestion, logstore, questions, reasoning_agent, store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Reasoning-Based Retrieval only (ablations)")
    ap.add_argument("--questions", default="questions.numeric-multihop.json",
                    help="question set file (default: the 41-question subset)")
    ap.add_argument("--profile", default="default", choices=list(batch.PROFILES),
                    help="budget profile (default / budget-b-low / budget-b-min / deep-b)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override the profile's step budget")
    ap.add_argument("--repeat", type=int, default=3, help="runs per question")
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

    _, max_steps = batch.resolve_params(args.profile, None, args.max_steps)
    repeat = max(1, args.repeat)
    total = len(qs) * repeat
    batch_id = "batch-" + uuid.uuid4().hex[:8]
    trees = {d["tree"].get("source", "?") for d in ingestion.DOCS.values() if d.get("tree")}

    print("-" * 72)
    print("PIPELINE B ONLY — ablation run")
    print("-" * 72)
    print(f"  question set     : {args.questions}  ({len(qs)} questions)")
    print(f"  repeats          : {repeat}   ->  {total} runs")
    print(f"  profile          : {args.profile}")
    print(f"  STEP BUDGET      : {max_steps}")
    print(f"  tree sources     : {', '.join(sorted(trees))}")
    print(f"  generator        : {config.OPENAI_MODEL} (effort {config.REASONING_EFFORT})")
    print(f"  log              : {config.LOG_PATH}")
    print(f"  batch_id         : {batch_id}")
    print("-" * 72)

    if args.dry_run:
        print("dry run — nothing was called, nothing was written.")
        return

    t0 = time.time()
    done = failed = capped = 0
    tok_in = tok_out = 0

    for q in qs:
        for r in range(1, repeat + 1):
            run_id = f"{batch_id}:{q['id']}:en:{r}"
            meta = {"run_id": run_id, "batch_id": batch_id, "question_id": q["id"],
                    "question_type": q.get("type", "other"),
                    "content_type": q.get("content_type"),
                    "repeat_index": r,
                    "profile": args.profile, "lang": "en",
                    "gold_pages": q.get("gold_pages", {}),
                    "question": q["question"], "model": config.OPENAI_MODEL}
            try:
                res = reasoning_agent.answer(q["question"], max_steps)
                u = res.get("usage") or {}
                tok_in += u.get("in", 0)
                tok_out += u.get("out", 0)
                capped += 1 if res.get("hit_limit") else 0
            except Exception as e:            # one bad run must not kill the run
                res = {"error": str(e)}
                failed += 1
            logstore.append({**meta, "phase": "reasoning", **res})
            done += 1
            flag = ""
            if res.get("error"):
                flag = "  ERROR: " + res["error"][:60]
            elif res.get("hit_limit"):
                flag = "  [budget exhausted]"
            print(f"  [{done}/{total}] {q['id']} (run {r}/{repeat}){flag}", flush=True)

    mins = (time.time() - t0) / 60
    print("-" * 72)
    print(f"done: {done} runs ({failed} failed, {capped} hit the step budget) in {mins:.1f} min")
    print(f"tokens: {tok_in:,} in + {tok_out:,} out = {tok_in + tok_out:,}")
    print(f"batch_id: {batch_id}   profile: {args.profile}   step budget: {max_steps}")
    print("-" * 72)


if __name__ == "__main__":
    main()
