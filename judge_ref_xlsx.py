"""Answer-quality tables of the reference-guided LLM judge (app/judge_ref.py).

Reads the `phase: "judge_ref"` rows of the final batch (rounds 1-3; the pilot round 99
is ignored) and writes one workbook:

  Rubric           the exact instructions the judge received (both calls)
  Run 1 … Run N    the score tables of that judge round, an Excel-formula check, and
                   every scored test case (scores, rationales, question, reference, answers)
  Mean of N runs   per-run headline numbers with mean ± sd, the tables recomputed on the
                   per-answer mean over the rounds, and the per-question means
  Consistency      how stable the judge's scores are between the rounds

The tables:
  1) mean score per metric (Groundedness, Answer Correctness, Completeness) per system,
     with Δ = B − A and a Wilcoxon signed-rank test on the 89 per-question differences;
  2) score distribution (how often each level 1-5 was given);
  3) / 4) the same means broken down by question type and by content type.

    python judge_ref_xlsx.py      -> "..\\LLM as a Judge.xlsx" (+ tables on stdout)
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Border, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.stats import wilcoxon

from answer_quality_xlsx import (HEAD_FILL, NOTE, THIN, WIN_FILL, better, md_table, mean_sd,
                                 num, pct, write_table)
from app import config, logstore
from app.judge_ref import MODEL, PHASE, RUBRIC_CC, RUBRIC_G

FINAL = "batch-46ded84d"
ROUNDS = (1, 2, 3)
OUT = Path(__file__).resolve().parent.parent / "LLM as a Judge.xlsx"
METRICS = ("groundedness", "correctness", "completeness")
LABEL = {"groundedness": "Groundedness", "correctness": "Answer Correctness",
         "completeness": "Completeness"}
SHORT = {"groundedness": "G", "correctness": "C", "completeness": "Comp"}
SYSTEMS = ("A", "B")
SYSTEM = {"A": "Vector-Based RAG (A)", "B": "Reasoning-Based Retrieval (B)", "tie": "Equal"}
# Message Batches API price of the judge model, $ per million tokens (50 % of the list price)
PRICE = {"in": 1.0, "out": 5.0, "cache_read": 0.10, "cache_write": 1.25}


# ------------------------------------------------------------------ records
def load(rows: list) -> dict[int, list[dict]]:
    """{round: [test case]} — one record per run_id with both systems' three scores,
    rationales and the run's question, reference and answers."""
    meta = {r["run_id"]: r for r in rows if r.get("batch_id") == FINAL and r.get("phase") == "vector"}
    b_ans = {r["run_id"]: r.get("answer", "") for r in rows
             if r.get("batch_id") == FINAL and r.get("phase") == "reasoning"}
    refs = json.loads((Path(config.DATA_DIR) / "reference_answers.json").read_text("utf-8"))["references"]
    cell = defaultdict(dict)       # (round, run_id) -> {system: {...}}
    usage = defaultdict(list)
    qno = {}
    for r in rows:
        if (r.get("phase") != PHASE or r.get("batch_id") != FINAL or r.get("judge_round") not in ROUNDS
                or r.get("error") or r.get("run_id") not in meta):
            continue
        qno[r["run_id"]] = r.get("question_no")
        s = cell[(r["judge_round"], r["run_id"])].setdefault(r["system"], {})
        if r["call"] == "G":
            s["groundedness"], s["rat_groundedness"] = r["groundedness"], r.get("rationale", "")
        else:
            s["correctness"], s["rat_correctness"] = r["correctness"], r.get("rationale_correctness", "")
            s["completeness"], s["rat_completeness"] = r["completeness"], r.get("rationale_completeness", "")
        usage[r["judge_round"]].append(r)
    out = defaultdict(list)
    for (rnd, rid), by_sys in cell.items():
        m = meta[rid]
        out[rnd].append({
            "run_id": rid, "question_no": qno[rid], "question_id": m.get("question_id"),
            "question_type": m.get("question_type"),
            "content_type": m.get("content_type") if m.get("gold_pages") else "unanswerable",
            "repeat_index": m.get("repeat_index"), "question": m.get("question", ""),
            "reference": refs.get(m.get("question_id"), ""),
            "A_answer": m.get("answer", ""), "B_answer": b_ans.get(rid, ""),
            "scores": {w: {k: by_sys.get(w, {}).get(k) for k in METRICS} for w in SYSTEMS},
            "rationales": {w: {k: by_sys.get(w, {}).get("rat_" + k, "") for k in METRICS} for w in SYSTEMS},
        })
    for recs in out.values():
        recs.sort(key=lambda x: (x["question_no"] or 0, x["repeat_index"] or 0))
    return {k: out[k] for k in sorted(out)}, usage


def score(x, who, metric=None):
    s = x["scores"][who]
    if metric:
        v = s.get(metric)
        return float(v) if isinstance(v, (int, float)) else None
    vals = [s.get(k) for k in METRICS]
    return sum(vals) / 3 if all(isinstance(v, (int, float)) for v in vals) else None


def complete(x):
    return all(score(x, w, k) is not None for w in SYSTEMS for k in METRICS)


def combine(rounds):
    """One record per test case scored in every round; each score is the mean over the rounds."""
    by_run = [{x["run_id"]: x for x in recs if complete(x)} for recs in rounds.values()]
    common = sorted(set.intersection(*(set(d) for d in by_run)),
                    key=lambda rid: (by_run[0][rid]["question_no"] or 0, by_run[0][rid]["repeat_index"] or 0))
    out = []
    for rid in common:
        g = [d[rid] for d in by_run]
        out.append({**g[0], "scores": {w: {k: sum(score(y, w, k) for y in g) / len(g) for k in METRICS}
                                       for w in SYSTEMS},
                    "per_round": {w: {k: [score(y, w, k) for y in g] for k in METRICS} for w in SYSTEMS}})
    return out


def per_question(recs, metric=None):
    """{question_no: (mean A, mean B)} over the repeats."""
    by_q = defaultdict(list)
    for x in recs:
        by_q[x["question_no"]].append(x)
    res = {}
    for q, sub in sorted(by_q.items()):
        a = [score(x, "A", metric) for x in sub if score(x, "A", metric) is not None]
        b = [score(x, "B", metric) for x in sub if score(x, "B", metric) is not None]
        if a and b:
            res[q] = (sum(a) / len(a), sum(b) / len(b))
    return res


def wilcoxon_p(recs, metric=None):
    d = [b - a for a, b in per_question(recs, metric).values()]
    return wilcoxon(d).pvalue if any(abs(v) > 1e-12 for v in d) else None


# ------------------------------------------------------------------ tables
def build(recs):
    rows1 = []
    for k in METRICS + (None,):
        ma, sa = mean_sd([score(x, "A", k) for x in recs])
        mb, sb = mean_sd([score(x, "B", k) for x in recs])
        p = wilcoxon_p(recs, k)
        rows1.append([LABEL[k] if k else "Mean of the three", num(ma, sa), num(mb, sb),
                      num(None if ma is None else mb - ma), "—" if p is None else f"{p:.2g}"])
    nq = len(per_question(recs))
    t1 = (f"Table 1 — Scores (1–5, mean ± sd over {len(recs)} test cases)",
          ["Metric", "A · Vector-Based RAG", "B · Reasoning-Based Retrieval", "Δ (B − A)",
           f"Wilcoxon p ({nq} questions)"], rows1)

    rows2 = []
    for k in METRICS:
        for w in SYSTEMS:
            vals = [score(x, w, k) for x in recs if score(x, w, k) is not None]
            n = len(vals)
            if all(float(v).is_integer() for v in vals):
                cnt = [sum(1 for v in vals if v == lvl) for lvl in (5, 4, 3, 2, 1)]
                rows2.append([f"{w} · {LABEL[k]}", n] + [f"{c} ({pct(c / n)})" for c in cnt])
            else:   # means over rounds: bin to the nearest level
                cnt = [sum(1 for v in vals if lvl - 0.5 <= v < lvl + 0.5 or (lvl == 5 and v == 5))
                       for lvl in (5, 4, 3, 2, 1)]
                rows2.append([f"{w} · {LABEL[k]}", n] + [f"{c} ({pct(c / n)})" for c in cnt])
    t2 = ("Table 2 — Score distribution (share of test cases per level)",
          ["System · metric", "n", "5", "4", "3", "2", "1"], rows2)

    def breakdown(key, order=None):
        groups = defaultdict(list)
        for x in recs:
            groups[x.get(key) or "—"].append(x)
        keys = order or sorted(groups, key=lambda g: -len(groups[g]))
        body = []
        for g in keys:
            sub = groups.get(g)
            if not sub:
                continue
            row = [g, len(sub)]
            for k in METRICS + (None,):
                row += [num(mean_sd([score(x, "A", k) for x in sub])[0]),
                        num(mean_sd([score(x, "B", k) for x in sub])[0])]
            body.append(row)
        return body

    bhead = ["n"] + [f"{SHORT[k] if k else 'Mean'} {w}" for k in METRICS + (None,) for w in SYSTEMS]
    t3 = ("Table 3 — By question type (mean score)", ["Question type"] + bhead, breakdown("question_type"))
    t4 = ("Table 4 — By content type (mean score)", ["Content type"] + bhead,
          breakdown("content_type", ["prose", "table", "unanswerable"]))
    return [t1, t2, t3, t4]


BREAK_PAIRS = [(3, 4), (5, 6), (7, 8), (9, 10)]


def audit_table(rnd, recs, usage_rows):
    tok = {k: sum((r.get("usage") or {}).get(k, 0) for r in usage_rows) for k in PRICE}
    cost = sum(tok[k] / 1e6 * PRICE[k] for k in PRICE)
    modes = sorted({r.get("mode") for r in usage_rows})
    return ("Judge audit", ["Check", "Value"],
            [["Judge model", f"{MODEL} (effort: {usage_rows[0].get('effort')}, mode: {', '.join(modes)})"],
             ["Scored test cases", f"{sum(1 for x in recs if complete(x))} of 267 (× 2 systems)"],
             ["Judge calls", f"{len(usage_rows)} (G = groundedness, CC = correctness + completeness)"],
             ["Tokens in / out", f"{tok['in'] + tok['cache_read'] + tok['cache_write']:,} / {tok['out']:,}"],
             ["Cost of this round (batch price)", f"${cost:.2f}"]])


def per_run_table(rounds):
    keys = [(k, w) for k in METRICS + (None,) for w in SYSTEMS]
    body = []
    for k, w in keys:
        vals = [mean_sd([score(x, w, k) for x in recs if complete(x)])[0] for recs in rounds.values()]
        m, sd = mean_sd(vals)
        body.append([f"{LABEL[k] if k else 'Mean of the three'} — {w}"] + [num(v) for v in vals]
                    + [num(m), f"± {sd:.2f}"])
    head = ["Measure"] + [f"Run {r}" for r in rounds] + ["Mean", "SD between runs"]
    return ("Headline numbers per judge run (mean score over test cases)", head, body)


def krippendorff_ordinal(units):
    """Krippendorff's alpha, ordinal metric. units: list of lists of integer ratings
    (one list per rated item; here the item = one answer, the raters = the judge rounds)."""
    units = [[int(v) for v in u if v is not None] for u in units]
    units = [u for u in units if len(u) >= 2]
    vals = sorted({v for u in units for v in u})
    if len(vals) < 2:
        return None
    idx = {v: i for i, v in enumerate(vals)}
    o = [[0.0] * len(vals) for _ in vals]
    for u in units:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    o[idx[a]][idx[b]] += 1 / (m - 1)
    n_c = [sum(row) for row in o]
    n = sum(n_c)

    def d2(c, k):
        lo, hi = min(c, k), max(c, k)
        return (sum(n_c[lo:hi + 1]) - (n_c[c] + n_c[k]) / 2) ** 2
    rng = range(len(vals))
    do = sum(o[c][k] * d2(c, k) for c in rng for k in rng)
    de = sum(n_c[c] * n_c[k] * d2(c, k) for c in rng for k in rng)
    return 1 - (n - 1) * do / de if de else None


def consistency_tables(rounds):
    by_run = [{x["run_id"]: x for x in recs if complete(x)} for recs in rounds.values()]
    common = sorted(set.intersection(*(set(d) for d in by_run)))
    labels = list(rounds)
    rows = []
    for k in METRICS:
        for w in SYSTEMS + ("A+B",):
            ws_ = SYSTEMS if w == "A+B" else (w,)
            units = [[score(d[rid], s, k) for d in by_run] for rid in common for s in ws_]
            n = len(units)
            ident = sum(1 for u in units if len(set(u)) == 1)
            within1 = sum(1 for u in units if max(u) - min(u) <= 1)
            sd = mean_sd([mean_sd(u)[1] for u in units])[0]
            alpha = krippendorff_ordinal(units)
            rows.append([f"{w} · {LABEL[k]}", n, f"{ident} ({pct(ident / n)})",
                         f"{within1} ({pct(within1 / n)})", num(sd),
                         "—" if alpha is None else f"{alpha:.2f}"])
    t1 = ("Score stability between the judge runs (per answer)",
          ["System · metric", "Answers", "Identical score in all runs", "Max − min ≤ 1",
           "Mean SD between runs", "Krippendorff's α (ordinal)"], rows)

    pair_rows = []
    for k in METRICS:
        for i in range(len(by_run)):
            for j in range(i + 1, len(by_run)):
                pairs = [(score(by_run[i][rid], s, k), score(by_run[j][rid], s, k))
                         for rid in common for s in SYSTEMS]
                same = sum(1 for a, b in pairs if a == b)
                pair_rows.append([f"{LABEL[k]}: Run {labels[i]} vs Run {labels[j]}", len(pairs),
                                  f"{same} ({pct(same / len(pairs))})"])
    t2 = ("Exact agreement between pairs of runs (A and B answers pooled)",
          ["Metric · runs", "Answers", "Same score"], pair_rows)

    # does the B − A comparison per question point the same way in every run?
    rows3 = []
    for k in METRICS + (None,):
        signs = []
        for d in by_run:
            pq = per_question(list(d.values()), k)
            signs.append({q: (b > a) - (b < a) for q, (a, b) in pq.items()})
        qs = sorted(set.intersection(*(set(s) for s in signs)))
        same = sum(1 for q in qs if len({s[q] for s in signs}) == 1)
        rows3.append([LABEL[k] if k else "Mean of the three", len(qs), f"{same} ({pct(same / len(qs))})"])
    t3 = ("Direction of the per-question difference (B better / equal / A better)",
          ["Metric", "Questions", "Same direction in all runs"], rows3)
    return [t1, t2, t3]


# ------------------------------------------------------------------ workbook
def clean(v):
    return ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v


def sheet_header(ws, title, subtitle):
    ws["A1"] = title
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = subtitle
    ws["A2"].font = Font(size=9, color="5B646E")
    ws["A3"] = ("Groundedness: answer vs. its own retrieved context (reference not shown) · "
                "Answer Correctness and Completeness: answer vs. reference answer · rubric 1–5, see sheet 'Rubric'")
    ws["A3"].font = Font(size=9, italic=True, color="5B646E")
    ws.column_dimensions["A"].width = 40
    for c in range(2, 26):
        ws.column_dimensions[get_column_letter(c)].width = 15


def head_row(ws, r, head):
    for c, h in enumerate(head, start=1):
        cell = ws.cell(row=r, column=c, value=h)
        cell.font, cell.fill = Font(bold=True, color="FFFFFF", size=10), HEAD_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def write_tables(ws, r, tables):
    t1, t2, t3, t4 = tables
    r = write_table(ws, r, *t1, bold_last=True, pairs=[(2, 3)], winner_head="Higher score")
    r = write_table(ws, r, *t2)
    for t in (t3, t4):
        r = write_table(ws, r, *t, pairs=BREAK_PAIRS)
    return r


RAT_COLS = [f"{w} rationale — {LABEL[k].lower()}" for w in SYSTEMS for k in METRICS]
TEXT_COLS = ["Question", "Reference answer", "A answer (Vector-Based RAG)",
             "B answer (Reasoning-Based Retrieval)", "run_id"]
SCORE_COLS = "FGHIJK"     # A G, A C, A Comp, B G, B C, B Comp


def write_raw(ws, title_row, recs, title, with_rationales=True):
    cols = (["Q no.", "Question ID", "Question type", "Content type", "Repeat"]
            + [f"{w} {SHORT[k]}" for w in SYSTEMS for k in METRICS]
            + (RAT_COLS if with_rationales else []) + TEXT_COLS)
    first, last = title_row + 2, title_row + 1 + len(recs)
    ws.cell(row=title_row, column=1, value=title).font = Font(bold=True, size=11)
    head_row(ws, title_row + 1, cols)
    for i, x in enumerate(recs):
        sc = [x["scores"][w][k] for w in SYSTEMS for k in METRICS]
        rat = [x["rationales"][w][k] for w in SYSTEMS for k in METRICS] if with_rationales else []
        vals = ([x["question_no"], x["question_id"], x["question_type"], x["content_type"], x["repeat_index"]]
                + sc + rat + [x["question"], x["reference"], x["A_answer"], x["B_answer"], x["run_id"]])
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=first + i, column=c, value=clean(v))
            if 6 <= c <= 11 and not with_rationales:
                cell.number_format = "0.00"
    ws.auto_filter.ref = f"A{title_row + 1}:{get_column_letter(len(cols))}{last}"
    widths = ([45] * 6 if with_rationales else []) + [50, 60, 60, 60, 22]
    for c, w in zip(range(12, 12 + len(widths)), widths):
        ws.column_dimensions[get_column_letter(c)].width = w
    return first, last


def write_check(ws, r, recs, first, last):
    ws.cell(row=r, column=1, value="Excel check — the means recomputed by Excel formulas over "
            "the test cases listed below").font = Font(bold=True, size=11)
    head_row(ws, r + 1, ["Check", "Python (tables above)", "Excel formula"])
    rr = r + 1
    checks = []
    for col, (w, k) in zip(SCORE_COLS, [(w, k) for w in SYSTEMS for k in METRICS]):
        checks.append((f"Mean {LABEL[k].lower()} — {w}", f"=AVERAGE({col}{first}:{col}{last})",
                       mean_sd([score(x, w, k) for x in recs])[0]))
    checks += [("Mean of the three — A", f"=AVERAGE(F{first}:H{last})", mean_sd([score(x, "A") for x in recs])[0]),
               ("Mean of the three — B", f"=AVERAGE(I{first}:K{last})", mean_sd([score(x, "B") for x in recs])[0])]
    for label, formula, py in checks:
        rr += 1
        ws.cell(row=rr, column=1, value=label)
        ws.cell(row=rr, column=2, value=round(py, 4)).number_format = "0.00"
        ws.cell(row=rr, column=3, value=formula).number_format = "0.00"
    for row in ws.iter_rows(min_row=r + 1, max_row=rr, max_col=3):
        for cell in row:
            cell.border = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
            if cell.column > 1:
                cell.alignment = Alignment(horizontal="center")
    return rr


CHECK_ROWS = 10


def write_run_sheet(wb, rnd, recs, usage_rows):
    ws = wb.create_sheet(f"Run {rnd}")
    sheet_header(ws, f"LLM as a Judge — run {rnd}",
                 f"batch id: {FINAL} · judge {MODEL} · each answer scored on its own (single-answer grading), "
                 "system labels hidden from the judge")
    recs = [x for x in recs if complete(x)]
    r = write_tables(ws, 4, build(recs))
    r = write_table(ws, r, *audit_table(rnd, recs, usage_rows))
    raw_title = r + CHECK_ROWS + 4
    first, last = raw_title + 2, raw_title + 1 + len(recs)
    assert write_check(ws, r, recs, first, last) < raw_title, "check block overlaps the list"
    write_raw(ws, raw_title, recs, f"All {len(recs)} test cases of run {rnd} (scores and judge rationales)")


def verdict_banner(ws, r, tables, ncols=10):
    rows = tables[0][2]
    lines = []
    for row in rows:
        w = better(row[1], row[2])
        lines.append(f"• {row[0]}: A {row[1].split(' ±')[0]} vs B {row[2].split(' ±')[0]} "
                     f"(Δ = {row[3]}, Wilcoxon p = {row[4]}) → higher: {SYSTEM.get(w, '—')}")
    overall = better(rows[-1][1], rows[-1][2])
    text = [(f"Result: {SYSTEM.get(overall, '—')} scores higher on the mean of the three metrics",
             Font(bold=True, size=14, color="006100"))] + [(l, Font(size=11)) for l in lines] + [
            ("Scores are the mean over the 3 judge runs per answer, then the mean over the 267 test cases. "
             "Green cell = the higher of the two systems.", NOTE)]
    top = r
    for t, font in text:
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=ncols)
        cell = ws.cell(row=r, column=1, value=t)
        cell.font, cell.alignment = font, Alignment(vertical="center")
        ws.row_dimensions[r].height = 24 if r == top else 17
        for c in range(1, ncols + 1):
            ws.cell(row=r, column=c).fill = PatternFill("solid", fgColor="EAF6EC")
        r += 1
    return r + 2


def write_mean_sheet(wb, rounds, combined):
    k = len(rounds)
    ws = wb.create_sheet(f"Mean of {k} runs")
    sheet_header(ws, f"LLM as a Judge — mean of {k} judge runs",
                 f"batch id: {FINAL} · {len(combined)} test cases scored in every run · "
                 "A = Vector-Based RAG, B = Reasoning-Based Retrieval")
    tables = build(combined)
    r = verdict_banner(ws, 4, tables)
    title, head, body = per_run_table(rounds)
    start = r
    r = write_table(ws, r, title, head, body)
    for i in range(0, len(body), 2):               # rows come in A / B pairs
        ra, rb = start + 2 + i, start + 3 + i
        for c in range(2, len(head)):              # run columns and the mean, not the SD
            w = better(ws.cell(row=ra, column=c).value, ws.cell(row=rb, column=c).value)
            if w in ("A", "B"):
                ws.cell(row=ra if w == "A" else rb, column=c).fill = WIN_FILL
    ws.cell(row=r - 1, column=1, value=(
        f"The tables below use the per-answer mean over the {k} runs (e.g. scores 4, 5, 5 → 4.67). "
        "Wilcoxon signed-rank test: on the per-question mean difference B − A (zero differences "
        "dropped).")).font = NOTE
    r = write_tables(ws, r + 1, tables)

    # per-question means
    ws.cell(row=r, column=1, value="Per question — mean over the 3 repeats and the 3 runs").font = Font(bold=True, size=11)
    head = ["Q no.", "Question ID", "Question type", "Content type"] + \
           [f"{SHORT[m]} {w}" for m in METRICS for w in SYSTEMS] + ["Mean A", "Mean B", "Δ mean (B − A)"]
    head_row(ws, r + 1, head)
    by_q = defaultdict(list)
    for x in combined:
        by_q[x["question_no"]].append(x)
    rr = r + 1
    for q, sub in sorted(by_q.items()):
        rr += 1
        vals = [q, sub[0]["question_id"], sub[0]["question_type"], sub[0]["content_type"]]
        for m in METRICS:
            for w in SYSTEMS:
                vals.append(mean_sd([score(x, w, m) for x in sub])[0])
        ma, mb = mean_sd([score(x, "A") for x in sub])[0], mean_sd([score(x, "B") for x in sub])[0]
        vals += [ma, mb, mb - ma]
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=rr, column=c, value=v)
            if isinstance(v, float):
                cell.number_format = "0.00"
                cell.alignment = Alignment(horizontal="center")
        for ca in (5, 7, 9, 11):
            w = better(f"{vals[ca - 1]:.4f}", f"{vals[ca]:.4f}")
            if w in ("A", "B"):
                ws.cell(row=rr, column=ca if w == "A" else ca + 1).fill = WIN_FILL
    for row in ws.iter_rows(min_row=r + 1, max_row=rr, max_col=len(head)):
        for cell in row:
            cell.border = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    write_raw(ws, rr + 3, combined, f"All {len(combined)} test cases — mean score over the {k} runs "
              "(rationales: see the Run sheets)", with_rationales=False)


def write_consistency_sheet(wb, rounds):
    ws = wb.create_sheet("Consistency")
    sheet_header(ws, "LLM as a Judge — consistency between the judge runs",
                 f"batch id: {FINAL} · the same {MODEL} prompts sent {len(rounds)} times (independent samples)")
    r = 4
    for t in consistency_tables(rounds):
        r = write_table(ws, r, *t)
    ws.cell(row=r - 1, column=1, value=(
        "Krippendorff's α (ordinal) treats each judge run as a rater and each answer as a unit; "
        "it corrects agreement for chance and weights a 4-vs-5 disagreement less than a 1-vs-5 one. "
        "α ≥ 0.80 is conventionally read as reliable, 0.667–0.80 as acceptable for tentative "
        "conclusions (Krippendorff, 2004). When almost all scores are 5, α can be low although raw "
        "agreement is high.")).font = NOTE


def write_rubric_sheet(wb):
    ws = wb.create_sheet("Rubric")
    ws["A1"] = "Judge instructions (system prompts sent verbatim)"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = (f"Model {MODEL} · two calls per answer: G (groundedness; user message = question, retrieved "
                "context, answer) and CC (correctness + completeness; user message = question, reference "
                "answer, answer). The judge never sees which system wrote the answer.")
    ws["A2"].font = Font(size=9, color="5B646E")
    ws.column_dimensions["A"].width = 150
    r = 4
    for title, text in (("Call G — Groundedness", RUBRIC_G), ("Call CC — Answer Correctness and Completeness", RUBRIC_CC)):
        ws.cell(row=r, column=1, value=title).font = Font(bold=True, size=11)
        r += 1
        for line in text.splitlines():
            c = ws.cell(row=r, column=1, value=line)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            r += 1
        r += 2


def to_numbers(wb):
    """Table cells are built as formatted text ('3.71'); store plain numbers as numbers so
    Excel can calculate with them, keeping the shown decimals. Text like '4.42 ± 0.92' stays."""
    for ws in wb.worksheets:
        if ws.title == "Rubric":
            continue
        for row in ws.iter_rows():
            for c in row:
                m = isinstance(c.value, str) and re.fullmatch(r"-?\d+\.(\d+)", c.value)
                if m:
                    c.value = float(c.value)
                    c.number_format = "0." + "0" * len(m.group(1))


def main():
    rounds, usage = load(logstore.read_all())
    if not rounds:
        raise SystemExit(f"No {PHASE} rows for rounds {ROUNDS} of {FINAL}.")
    wb = Workbook()
    wb.remove(wb.active)
    write_rubric_sheet(wb)
    for rnd, recs in rounds.items():
        write_run_sheet(wb, rnd, recs, usage[rnd])
        print(f"run {rnd}: {sum(1 for x in recs if complete(x))} complete test cases of {len(recs)}")
    if len(rounds) > 1:
        combined = combine(rounds)
        write_mean_sheet(wb, rounds, combined)
        write_consistency_sheet(wb, rounds)
        for title, head, body in [per_run_table(rounds)] + build(combined) + consistency_tables(rounds):
            print(f"\n### {title}\n")
            print(md_table(head, body))
    to_numbers(wb)
    wb.calculation.fullCalcOnLoad = True
    wb.save(OUT)
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
