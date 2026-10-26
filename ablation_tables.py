"""Render the three Pipeline-A ablation tables as markdown.

Every column is computed over the same 41 questions (numeric + multi-hop), so the
baseline column is the baseline batch restricted to exactly those questions.

    python ablation_tables.py            # print
    python ablation_tables.py > ..\\ablation-tables.md
"""
from app import logstore, metrics

BASE = "batch-bb633707"

GROUPS = [
    ("Table 1 — Retrieval depth (top-k)", [
        ("k = 3", "batch-6d5f93d3"), ("k = 5 (baseline)", BASE), ("k = 10", "batch-59c4d149")]),
    ("Table 2 — Embedding model", [
        ("BGE-M3 (baseline)", BASE), ("multilingual-E5-base", "batch-e9f57386")]),
    ("Table 3 — Chunking strategy", [
        ("clause-aware (baseline)", BASE), ("character windows", "batch-c9000fd0")]),
]


def pct(st):
    if not st or st.get("mean") is None:
        return "—"
    s = f"{st['mean'] * 100:.0f}%"
    return s + (f" ± {st['std'] * 100:.0f}" if st.get("std") else "")


def num(st, dp=2):
    if not st or st.get("mean") is None:
        return "—"
    s = f"{st['mean']:.{dp}f}"
    return s + (f" ± {st['std']:.{dp}f}" if st.get("std") else "")


def main():
    rows = logstore.read_all()
    print("# Pipeline A ablations\n")
    print("All columns cover the same 41 questions (25 numeric + 16 multi-hop), "
          "three repeats each, 123 runs per column. Pipeline B is unaffected by these "
          "settings and was not run.\n")

    for title, cols in GROUPS:
        qids = None
        for _, bid in cols:
            s = {r.get("question_id") for r in rows if r.get("batch_id") == bid}
            qids = s if qids is None else (qids & s)
        blocks = []
        for _, bid in cols:
            sel = [r for r in rows if r.get("batch_id") == bid and r.get("question_id") in qids]
            m = metrics.summarize(sel)
            per = {t["type"]: t["A"] for t in m["per_type"]}
            blocks.append((m["overall"]["A"], per))

        head = " | ".join(label for label, _ in cols)
        print(f"\n## {title}\n")
        print(f"| Metric | {head} |")
        print("| :--- | " + " | ".join([":---:"] * len(cols)) + " |")

        def line(label, fn):
            print(f"| {label} | " + " | ".join(fn(b) for b in blocks) + " |")

        line("Recall@k", lambda b: pct(b[0].get("recall")))
        line("Precision@k", lambda b: pct(b[0].get("precision")))
        line("F1", lambda b: pct(b[0].get("f1")))
        line("MRR", lambda b: num(b[0].get("mrr")))
        line("Pages retrieved", lambda b: num(b[0].get("pages"), 1))
        line("Latency (s)", lambda b: f"{b[0]['latency_ms']['mean'] / 1000:.1f}")
        line("Tokens", lambda b: f"{b[0]['tokens']['mean']:,.0f}")
        print("| | | | |" if len(cols) == 3 else "| | | |")
        for t in ("numeric", "multi-hop"):
            line(f"Recall@k — {t}", lambda b, t=t: pct(b[1].get(t, {}).get("recall")))
            line(f"F1 — {t}", lambda b, t=t: pct(b[1].get(t, {}).get("f1")))

    print("\n*Values are means over runs with the standard deviation across repeats. "
          "Recall, precision, F1 and MRR are page-level against the gold pages; "
          "precision's denominator is the number of distinct retrieved pages.*")


if __name__ == "__main__":
    main()
