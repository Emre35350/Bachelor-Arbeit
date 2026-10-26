"""Post-hoc judging — run the blind LLM judge over a batch that is already logged.

The pipelines are NOT re-run: the A and B answers stored in the experiment log
under each run_id are passed to judge.judge_pair, and the verdict is appended as
a `phase: "judge"` row with the same run_id/batch metadata, exactly as a batch
started with --judge would have written it. The answer-quality figures are
therefore computed on the very answers the retrieval metrics were computed on.

Rounds: the judge itself is non-deterministic, so the same batch can be judged
several times (--round 2, --round 3, ...) to measure how consistent its
verdicts are. Each verdict row carries `judge_round`; rows written before rounds
existed count as round 1. The answer order is re-randomized in every round.

Idempotent: run_ids that already carry a verdict for the requested round are
skipped, so an interrupted round can simply be started again.

    python -m app.judge_batch --batch batch-46ded84d [--round 2] [--workers 6] [--limit 3]
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, judge, logstore

META_KEYS = ("run_id", "batch_id", "question_id", "question_type", "content_type",
             "repeat_index", "profile", "lang", "question", "gold_pages")


def judge_round(row: dict) -> int:
    return int(row.get("judge_round") or 1)


def pending_pairs(rows: list, batch_id: str, rnd: int = 1) -> list[tuple[dict, dict]]:
    """(A row, B row) per run_id of the batch that has both answers and no verdict
    for round `rnd` yet."""
    runs: dict[str, dict] = {}
    judged = set()
    for r in rows:
        if r.get("batch_id") == batch_id and r.get("run_id"):
            if r.get("phase") == "judge":
                if judge_round(r) == rnd:
                    judged.add(r["run_id"])
            else:
                runs.setdefault(r["run_id"], {})[r.get("phase")] = r
    out = []
    for rid in sorted(runs):
        ph = runs[rid]
        a, b = ph.get("vector"), ph.get("reasoning")
        if rid in judged or not a or not b or a.get("error") or b.get("error"):
            continue
        out.append((a, b))
    return out


def _judge_with_retry(a: dict, b: dict, rnd: int, attempts: int = 3) -> dict:
    # the round is part of the seed material so a seeded setup still draws a
    # fresh answer order per round (unseeded, the order is random anyway)
    seed = a["run_id"] if rnd == 1 else f"{a['run_id']}#round{rnd}"
    for i in range(attempts):
        try:
            return judge.judge_pair(a.get("question", ""), a.get("answer", ""),
                                    b.get("answer", ""), seed_material=seed)
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc blind judge over a logged batch")
    parser.add_argument("--batch", required=True, help="batch_id to judge")
    parser.add_argument("--round", type=int, default=1, help="judge round (1, 2, 3, ...)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="judge only the first n pairs")
    args = parser.parse_args()

    pairs = pending_pairs(logstore.read_all(), args.batch, args.round)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"{args.batch} round {args.round}: {len(pairs)} pair(s) to judge with "
          f"{config.JUDGE_PROVIDER}/{config.JUDGE_MODEL}", flush=True)

    done = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(_judge_with_retry, a, b, args.round): a for a, b in pairs}
        for fut in as_completed(futs):
            a = futs[fut]
            try:
                verdict = fut.result()
            except Exception as e:
                failed += 1
                print(f"  FAILED {a['run_id']}: {e}", flush=True)
                continue
            # appended from the main thread only, so log lines never interleave
            logstore.append({**{k: a.get(k) for k in META_KEYS}, "phase": "judge",
                             "posthoc": True, "judge_round": args.round, **verdict})
            done += 1
            print(f"  [{done + failed}/{len(pairs)}] {a['run_id']} -> {verdict['winner']}",
                  flush=True)
    print(f"Done: {done} judged, {failed} failed -> {config.LOG_PATH}")


if __name__ == "__main__":
    main()
