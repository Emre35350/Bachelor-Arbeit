"""Compare one Pipeline-A ablation batch against the baseline, on the same questions.

The baseline is automatically restricted to the questions the ablation actually ran,
so both columns always cover the identical question set.

Usage (from the retrievallab-py folder):

    python compare_a.py --list                      # every run in the log
    python compare_a.py --ablation batch-6d5f93d3   # compare against the baseline
"""
import argparse

from app import logstore, metrics


def condition(row):
    """The condition a run executed under, read off its own log row."""
    c = row.get("config") or {}
    if row.get("phase") == "reasoning":
        return (f"profile={row.get('profile')} · max_steps={row.get('max_steps')} · "
                f"{c.get('model')} (effort {c.get('reasoning_effort')})")
    return (f"top_k={row.get('top_k')} · {c.get('embedding_model')} · "
            f"clause_chunking={c.get('clause_chunking')} · {c.get('pinecone_namespace')}")


def head_row(rows, bid):
    """A representative row of that batch for the phase being compared."""
    want = "reasoning" if KEY == "B" else "vector"
    return (next((r for r in rows if r.get("batch_id") == bid and r.get("phase") == want), None)
            or next((r for r in rows if r.get("batch_id") == bid), {}))


def list_batches(rows):
    seen = {}
    for r in rows:
        b = r.get("batch_id")
        if not b:
            continue
        e = seen.setdefault(b, {"rows": 0, "phases": set(), "when": r.get("logged_at"),
                                "cond": None, "qs": set()})
        e["rows"] += 1
        e["phases"].add(r.get("phase"))
        e["qs"].add(r.get("question_id"))
        if e["cond"] is None and r.get("phase") in ("vector", "reasoning"):
            e["cond"] = condition(r)
    print(f"{'batch_id':<18}{'when':<21}{'rows':>6}{'questions':>11}  phases        condition")
    print("-" * 118)
    for b, e in sorted(seen.items(), key=lambda kv: kv[1]["when"] or ""):
        ph = "+".join(sorted(p for p in e["phases"] if p))
        print(f"{b:<18}{(e['when'] or '')[:19]:<21}{e['rows']:>6}{len(e['qs']):>11}  {ph:<14}{e['cond'] or ''}")


def fmt(st, pct=False, dp=2):
    if not st or st.get("mean") is None:
        return "—"
    m, s = st["mean"], st.get("std")
    if pct:
        return f"{m * 100:.0f}%" + (f" ± {s * 100:.0f}" if s else "")
    return f"{m:.{dp}f}" + (f" ± {s:.{dp}f}" if s else "")


def delta(a, b, pct=False, dp=2):
    if not a or not b or a.get("mean") is None or b.get("mean") is None:
        return ""
    d = b["mean"] - a["mean"]
    return f"{d * 100:+.0f} pp" if pct else f"{d:+.{dp}f}"


ROWS_A = [("Recall@k", "recall", True, 2), ("Precision@k", "precision", True, 2),
          ("F1", "f1", True, 2), ("MRR", "mrr", False, 2),
          ("pages retrieved", "pages", False, 1),
          ("latency (ms)", "latency_ms", False, 0), ("tokens", "tokens", False, 0)]

ROWS_B = [("Recall", "recall", True, 2), ("Precision", "precision", True, 2),
          ("F1", "f1", True, 2), ("reads->gold", "reads_to_gold", False, 1),
          ("pages read", "pages", False, 1),
          ("latency (ms)", "latency_ms", False, 0), ("tokens", "tokens", False, 0),
          ("tool calls", "effort", False, 1)]

ROWS = ROWS_A          # rebound in main() when --phase b is given
KEY = "A"


def table(title, base_A, abl_A):
    print(f"\n{title}")
    print(f"  {'metric':<18}{'baseline':>18}{'ablation':>18}{'Δ':>12}")
    print("  " + "-" * 66)
    for label, key, pct, dp in ROWS:
        print(f"  {label:<18}{fmt(base_A.get(key), pct, dp):>18}"
              f"{fmt(abl_A.get(key), pct, dp):>18}"
              f"{delta(base_A.get(key), abl_A.get(key), pct, dp):>12}")
    if KEY == "B":
        hb, ha = base_A.get("hit_limit_rate"), abl_A.get("hit_limit_rate")
        print(f"  {'budget exhausted':<18}"
              f"{(f'{hb*100:.0f}%' if hb is not None else '—'):>18}"
              f"{(f'{ha*100:.0f}%' if ha is not None else '—'):>18}")
    print(f"  {'runs':<18}{base_A.get('n_runs', 0):>18}{abl_A.get('n_runs', 0):>18}")


def multi(rows, bids):
    """One column per batch, over the questions all of them share."""
    sets = []
    for b in bids:
        br = [r for r in rows if r.get("batch_id") == b]
        if not br:
            raise SystemExit(f"no rows found for {b}")
        sets.append({r.get("question_id") for r in br if r.get("question_id")})
    common = set.intersection(*sets)

    print("=" * 78)
    for i, b in enumerate(bids, 1):
        print(f"  [{i}] {b}   {condition(head_row(rows, b))}")
    print(f"  questions common to all columns: {len(common)}")
    print("=" * 78)

    cols = [metrics.summarize([r for r in rows if r.get("batch_id") == b
                               and r.get("question_id") in common])["overall"][KEY]
            for b in bids]

    head = "".join(f"{'[' + str(i) + ']':>14}" for i in range(1, len(bids) + 1))
    print(f"\n  {'metric':<18}{head}")
    print("  " + "-" * (18 + 14 * len(bids)))
    for label, key, pct, dp in ROWS:
        line = "".join(f"{fmt(c.get(key), pct, dp):>14}" for c in cols)
        print(f"  {label:<18}{line}")
    print(f"  {'runs':<18}" + "".join(f"{c.get('n_runs', 0):>14}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", help="batch_id of the ablation run")
    ap.add_argument("--baseline", default="batch-bb633707")
    ap.add_argument("--batches", nargs="+", help="two or more batch_ids, side by side")
    ap.add_argument("--list", action="store_true", help="list every run in the log")
    ap.add_argument("--phase", choices=["a", "b"], default="a",
                    help="a = Vector-Based RAG (default), b = Reasoning-Based Retrieval")
    args = ap.parse_args()

    global ROWS, KEY
    ROWS, KEY = (ROWS_B, "B") if args.phase == "b" else (ROWS_A, "A")

    rows = logstore.read_all()
    if args.batches:
        multi(rows, args.batches)
        return
    if args.list or not args.ablation:
        list_batches(rows)
        if not args.ablation:
            return

    abl_rows = [r for r in rows if r.get("batch_id") == args.ablation]
    if not abl_rows:
        raise SystemExit(f"no rows found for {args.ablation}")
    qids = {r.get("question_id") for r in abl_rows if r.get("question_id")}

    m_abl = metrics.summarize(abl_rows)
    m_base = metrics.summarize([r for r in rows if r.get("batch_id") == args.baseline
                                and r.get("question_id") in qids])

    a0 = head_row(rows, args.ablation)
    b0 = head_row(rows, args.baseline)
    errs = sum(1 for r in abl_rows if r.get("error"))

    print("=" * 70)
    print(f"BASELINE  {args.baseline}   {condition(b0)}")
    print(f"ABLATION  {args.ablation}   {condition(a0)}")
    print(f"questions {len(qids)}   ablation rows {len(abl_rows)} ({errs} failed)")
    print("=" * 70)

    table("OVERALL", m_base["overall"][KEY], m_abl["overall"][KEY])
    bt = {t["type"]: t for t in m_base["per_type"]}
    for t in m_abl["per_type"]:
        if t["type"] in bt:
            table(f"TYPE: {t['type']}", bt[t["type"]][KEY], t[KEY])


if __name__ == "__main__":
    main()
