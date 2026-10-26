"""Chapter-4 metrics — computed straight from the experiment log.

Pairs each question's A (vector) and B (reasoning) rows by run_id and scores the
evidence sets against gold_pages. The retrieval-quality trio (page-level, the
unit gold_pages is defined in):

  recall     |evidence ∩ gold| / |gold|          Recall@k (A) / Context Recall (B)
  precision  |evidence ∩ gold| / |evidence|      Precision@k (A) / Context Precision (B)
  f1         2·P·R / (P + R)                      retrieval F1 — balances the two

  mrr (A)    1 / rank-of-first-gold in the ranked top-k
  reads_to_gold (B)  1-based position of the first gold page in B's read order
  overlap    Jaccard(A.retrieved, B.read)         do the paradigms look at the same pages?
  plus latency, tokens, cost, and effort (A calls / B tool-calls).

IMPORTANT — these are PAGE-LEVEL (context) metrics, not chunk-level:
  * the retrieved top-k chunks are collapsed to their UNIQUE pages before scoring,
    because gold is defined in pages. So precision's denominator is the number of
    distinct retrieved pages, NOT k: a page retrieved by three chunks counts once.
  * they are therefore Recall@k / Precision@k in the sense "recall/precision of
    gold PAGES using top-k retrieval" (= RAGAS-style context recall/precision),
    not "fraction of the k chunks that are relevant".
  * Pipeline B has no fixed k (it reads pages by reasoning), so the identical
    formulas are its context recall / precision over the pages it read.
  * aggregation is MACRO: the per-run value is computed first, then averaged over
    runs (mean of per-query F1, not F1 of the mean precision/recall).

Rows whose gold_pages are empty (unanswerable controls) are excluded from
recall/precision means and counted separately — abstention correctness is the
LLM judge's job (Step 8), not a page-overlap metric.

Aggregation is by system × question-type, reporting mean and, when a question was
repeated, the standard deviation across repeats.
"""
import csv
import io
import math

from . import config


# ------------------------------------------------------------------ set utils
def pageset(evmap) -> set:
    """{doc: [pages]} -> {(doc, page)}."""
    out = set()
    if isinstance(evmap, dict):
        for doc, pages in evmap.items():
            for p in (pages or []):
                out.add((str(doc), int(p)))
    return out


def _recall(hit: set, gold: set):
    return len(hit & gold) / len(gold) if gold else None


def _precision(hit: set, gold: set):
    return len(hit & gold) / len(hit) if hit else (None if not gold else 0.0)


def _f1(hit: set, gold: set):
    """Retrieval F1 = harmonic mean of context precision and recall.

    None when there is no gold (unanswerable control — excluded from means).
    0 when nothing relevant was retrieved, so precision and recall are both 0.
    """
    if not gold:
        return None
    inter = len(hit & gold)
    if inter == 0:
        return 0.0
    precision = inter / len(hit)      # hit is non-empty here (inter > 0)
    recall = inter / len(gold)
    return 2 * precision * recall / (precision + recall)


def _jaccard(a: set, b: set):
    u = a | b
    return len(a & b) / len(u) if u else None


def _mrr_from_ranked(retrieved: list, gold: set):
    """rank of the first retrieved (doc,page) that is gold, over the ranked list."""
    if not gold:
        return None
    seen = set()
    rank = 0
    for h in retrieved or []:
        key = (str(h.get("doc")), int(h.get("page", 0)))
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        if key in gold:
            return 1.0 / rank
    return 0.0


def _reads_to_gold(read_order: list, gold: set):
    """1-based position of the first gold page in B's first-read order."""
    if not gold:
        return None
    for i, dp in enumerate(read_order or [], start=1):
        try:
            key = (str(dp[0]), int(dp[1]))
        except (TypeError, ValueError, IndexError):
            continue
        if key in gold:
            return i
    return None  # gold never read


# ------------------------------------------------------------- pairing + rows
def _pair_by_run(rows: list) -> dict:
    runs: dict[str, dict] = {}
    for r in rows:
        rid = r.get("run_id")
        if not rid:
            continue
        runs.setdefault(rid, {})[r.get("phase")] = r
    return runs


def per_run(rows: list) -> list[dict]:
    """One record per run_id, with A and B metrics where present.

    Only evaluation runs are scored: a run must carry gold_pages metadata (which
    the batch runner attaches, even as {} for unanswerable controls). Ad-hoc
    interactive /ask runs have no gold label and are skipped so they don't
    pollute the dashboard.
    """
    out = []
    for rid, phases in _pair_by_run(rows).items():
        a, b = phases.get("vector"), phases.get("reasoning")
        base = a or b or {}
        if "gold_pages" not in (a or {}) and "gold_pages" not in (b or {}):
            continue  # not an evaluation run
        gold = pageset(base.get("gold_pages"))
        rec = {
            "run_id": rid,
            "batch_id": base.get("batch_id"),
            "question_id": base.get("question_id"),
            "question": base.get("question"),
            "question_type": base.get("question_type", "other"),
            "content_type": base.get("content_type"),
            "lang": base.get("lang"),
            "repeat_index": base.get("repeat_index"),
            "profile": base.get("profile"),
            "has_gold": bool(gold),
            "gold_pages": sum(len(v) for v in (base.get("gold_pages") or {}).values()),
        }
        a_set = b_set = set()
        if a and not a.get("error"):
            a_set = pageset((a.get("evidence") or {}).get("retrieved"))
            au = a.get("usage") or {}
            rec["A"] = {
                "recall": _recall(a_set, gold),
                "precision": _precision(a_set, gold),
                "f1": _f1(a_set, gold),
                "mrr": _mrr_from_ranked(a.get("retrieved"), gold),
                "pages": len(a_set),
                "latency_ms": a.get("latency_ms"),
                "tokens": (au.get("in", 0) + au.get("out", 0)),
                "cost": config.token_cost(au.get("in", 0), au.get("out", 0)),
                "effort": a.get("calls", 1),
            }
        if b and not b.get("error"):
            b_ev = b.get("evidence") or {}
            b_set = pageset(b_ev.get("read"))
            bu = b.get("usage") or {}
            rec["B"] = {
                "recall": _recall(b_set, gold),
                "precision": _precision(b_set, gold),
                "f1": _f1(b_set, gold),
                "reads_to_gold": _reads_to_gold(b_ev.get("read_order"), gold),
                "pages": len(b_set),
                "latency_ms": b.get("latency_ms"),
                "tokens": (bu.get("in", 0) + bu.get("out", 0)),
                "cost": config.token_cost(bu.get("in", 0), bu.get("out", 0)),
                "effort": b.get("tool_calls", b.get("calls", 0)),
                "hit_limit": bool(b.get("hit_limit")),
            }
        rec["overlap"] = _jaccard(a_set, b_set) if (a_set or b_set) else None
        j = phases.get("judge")
        rec["judge"] = j.get("winner") if j else None
        out.append(rec)
    return out


# ------------------------------------------------------------- aggregation
def _stats(vals: list):
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(xs) / len(xs)
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(xs)}


_A_FIELDS = ("recall", "precision", "f1", "mrr", "pages", "latency_ms", "tokens", "cost", "effort")
_B_FIELDS = ("recall", "precision", "f1", "reads_to_gold", "pages", "latency_ms", "tokens", "cost", "effort")
_GOLD_FIELDS = ("recall", "precision", "f1", "mrr", "reads_to_gold")


def _agg_system(records: list, syskey: str, fields, gold_only: bool) -> dict:
    """Aggregate one system's per-run values into mean/std per field."""
    sub = [r for r in records if syskey in r and (r["has_gold"] or not gold_only)]
    agg = {}
    for f in fields:
        # retrieval-quality + rank fields are only meaningful when gold exists
        pool = [r for r in sub if (r["has_gold"] or f not in _GOLD_FIELDS)]
        agg[f] = _stats([r[syskey].get(f) for r in pool])
    agg["n_runs"] = len(sub)
    if syskey == "B":
        agg["hit_limit_rate"] = _stats([1.0 if r["B"].get("hit_limit") else 0.0 for r in sub])["mean"]
    return agg


def summarize(rows: list) -> dict:
    records = per_run(rows)
    types = sorted({r["question_type"] for r in records})

    def block(recs):
        judged = [r for r in recs if r.get("judge")]
        n_j = len(judged)
        wins = {"A": 0, "B": 0, "tie": 0}
        for r in judged:
            wins[r["judge"]] = wins.get(r["judge"], 0) + 1
        return {
            "A": _agg_system(recs, "A", _A_FIELDS, gold_only=False),
            "B": _agg_system(recs, "B", _B_FIELDS, gold_only=False),
            "overlap": _stats([r["overlap"] for r in recs if r["overlap"] is not None]),
            "judge": {"n": n_j,
                      "A_win": wins["A"] / n_j if n_j else None,
                      "B_win": wins["B"] / n_j if n_j else None,
                      "tie": wins["tie"] / n_j if n_j else None},
            "n": len(recs),
        }

    per_type = [{"type": t, **block([r for r in records if r["question_type"] == t])}
                for t in types]

    return {
        "n_runs": len(records),
        "n_questions": len({r["question_id"] for r in records if r["question_id"]}),
        "n_unanswerable": sum(1 for r in records if not r["has_gold"]),
        "cost_model": {"in": config.COST_PER_1M_IN, "out": config.COST_PER_1M_OUT},
        "overall": block(records),
        "per_type": per_type,
    }


# -------------------------------------------------------------------- CSV
# Column order mirrors the dashboard drill-down popup: identity, content_type +
# gold, then Recall / Precision / F1 as A|B pairs, MRR (A) and reads→gold (B),
# and the judge verdict.
_CSV_COLS = [
    "question_id", "question", "question_type", "content_type", "gold_pages", "lang",
    "A_recall", "B_recall", "A_precision", "B_precision", "A_f1", "B_f1",
    "A_mrr", "B_reads_to_gold", "judge",
]


def _mean(vals) -> float | None:
    xs = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(xs) / len(xs), 4) if xs else None


def _av(subrecs: list, syskey: str, field: str):
    """Mean of one metric over runs (skipping runs where it is absent/None)."""
    return _mean([r[syskey][field] for r in subrecs
                  if r.get(syskey) and r[syskey].get(field) is not None])


def _judge_str(subrecs: list, as_pct: bool) -> str:
    """A/B/tie as counts (per-question) or percentages (totals) — like the popup."""
    js = [r["judge"] for r in subrecs if r.get("judge")]
    if not js:
        return ""
    a = sum(1 for x in js if x == "A")   # judge_pair already maps 1/2 back to A/B
    b = sum(1 for x in js if x == "B")
    t = sum(1 for x in js if x == "tie")
    if as_pct:
        n = len(js)
        return f"A{round(100 * a / n)}% B{round(100 * b / n)}% tie{round(100 * t / n)}%"
    return f"A{a}·B{b}·t{t}"


def _csv_metrics(subrecs: list) -> dict:
    return {
        "A_recall": _av(subrecs, "A", "recall"), "B_recall": _av(subrecs, "B", "recall"),
        "A_precision": _av(subrecs, "A", "precision"), "B_precision": _av(subrecs, "B", "precision"),
        "A_f1": _av(subrecs, "A", "f1"), "B_f1": _av(subrecs, "B", "f1"),
        "A_mrr": _av(subrecs, "A", "mrr"), "B_reads_to_gold": _av(subrecs, "B", "reads_to_gold"),
    }


# retrieval metrics are only defined where gold exists — restrict their means to
# has_gold runs so the numbers match the dashboard/popup exactly (judge stays over
# all runs, being meaningful for unanswerable questions too)
def _gold(recs: list) -> list:
    return [r for r in recs if r.get("has_gold")]


def detail_rows(recs: list) -> list[dict]:
    """One dict per question (repeats averaged) — the drill-down popup structure.
    Shared by the CSV and the Excel export so they never diverge."""
    from collections import OrderedDict
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for r in sorted(recs, key=lambda r: (r.get("question_type") or "", r.get("question_id") or "")):
        groups.setdefault((r.get("question_id"), r.get("lang")), []).append(r)
    out = []
    for (qid, lang), sub in groups.items():
        base = sub[0]
        out.append({"question_id": qid, "question": base.get("question"),
                    "question_type": base.get("question_type"),
                    "content_type": base.get("content_type"),
                    "gold_pages": base.get("gold_pages"), "lang": lang,
                    "judge": _judge_str(sub, as_pct=False), **_csv_metrics(_gold(sub))})
    return out


def totals_rows(recs: list) -> list[dict]:
    """Per-type and overall totals (mean over runs) — the popup's totals line."""
    out = []
    for t in sorted({r.get("question_type") or "other" for r in recs}):
        sub = [r for r in recs if (r.get("question_type") or "other") == t]
        out.append({"question_id": "__TOTAL__", "question": f"mean over {len(sub)} run(s)",
                    "question_type": t, "content_type": "", "gold_pages": "", "lang": "",
                    "judge": _judge_str(sub, as_pct=True), **_csv_metrics(_gold(sub))})
    if recs:
        out.append({"question_id": "__TOTAL__", "question": f"mean over {len(recs)} run(s)",
                    "question_type": "ALL", "content_type": "", "gold_pages": "", "lang": "",
                    "judge": _judge_str(recs, as_pct=True), **_csv_metrics(_gold(recs))})
    return out


def to_csv(rows: list) -> str:
    """One row per question, laid out exactly like the drill-down popup, then
    per-type and overall totals rows. Metric values are 0-1 floats for analysis;
    judge carries A/B/tie counts per question and percentages in the totals."""
    recs = per_run(rows)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_CSV_COLS, extrasaction="ignore")
    w.writeheader()
    for row in detail_rows(recs) + totals_rows(recs):
        w.writerow(row)
    return buf.getvalue()


# -------------------------------------------------------------------- Excel
def _answer_records(rows: list) -> list[dict]:
    """Per evaluation run: the question, gold, and BOTH pipelines' answers."""
    runs: dict[str, dict] = {}
    for r in rows:
        rid = r.get("run_id")
        if rid:
            runs.setdefault(rid, {})[r.get("phase")] = r
    out = []
    for rid, ph in runs.items():
        base = ph.get("vector") or ph.get("reasoning") or {}
        if "question_id" not in base:
            continue  # ad-hoc /ask run, not part of the evaluation set
        a, b, j = ph.get("vector", {}), ph.get("reasoning", {}), ph.get("judge", {})
        gold = base.get("gold_pages") or {}
        gold_str = "; ".join(f"{d} p.{','.join(map(str, sorted(p)))}" for d, p in gold.items())
        out.append({
            "question_id": base.get("question_id"), "lang": base.get("lang"),
            "question_type": base.get("question_type"), "content_type": base.get("content_type"),
            "repeat": base.get("repeat_index"), "question": base.get("question"),
            "gold": gold_str, "A_answer": a.get("answer", ""), "B_answer": b.get("answer", ""),
            "judge": j.get("winner", ""),
            "judge_rationale": j.get("rationale", ""),
        })
    out.sort(key=lambda r: (r.get("question_type") or "", r.get("question_id") or "", r.get("repeat") or 0))
    return out


def _pct_sd(stat: dict) -> str:
    """'11% ± 31' — mirrors how the dashboard shows a mean ± std cell."""
    if not stat or stat.get("mean") is None:
        return "—"
    s = f"{round(stat['mean'] * 100)}%"
    if stat.get("std") and stat.get("n", 0) > 1:
        s += f" ± {round(stat['std'] * 100)}"
    return s


def to_xlsx(rows: list) -> bytes:
    """Three-sheet workbook:
      1) Summary   — grouped metrics per question-type (the dashboard, mean ± sd)
      2) Detail    — one numeric row per question (the drill-down popup)
      3) Answers   — every question with both pipelines' answers
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    m = summarize(rows)
    recs = per_run(rows)
    HEAD = Font(bold=True, color="FFFFFF")
    HFILL = PatternFill("solid", fgColor="3B5B33")
    TOTFILL = PatternFill("solid", fgColor="F1F5F2")
    wrap_top = Alignment(vertical="top", wrap_text=True)

    def style_header(ws, ncols):
        for c in range(1, ncols + 1):
            cell = ws.cell(row=1, column=c)
            cell.font, cell.fill = HEAD, HFILL
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.freeze_panes = "A2"

    wb = Workbook()

    # ---- Sheet 1: Summary (dashboard, grouped by question-type) --------------
    ws = wb.active
    ws.title = "Summary"
    cols1 = ["Type", "n", "Recall@k A", "Recall@k B", "Precision@k A", "Precision@k B",
             "F1 A", "F1 B", "MRR (A)", "reads→gold (B)", "Overlap",
             "Latency A (s)", "Latency B (s)", "Tokens A", "Tokens B",
             "Judge A%", "Judge B%", "Tie%"]
    ws.append(cols1)

    def summary_row(label, blk):
        A, B, jb = blk["A"], blk["B"], blk.get("judge") or {}
        mn = lambda st, dp=2: None if not st or st.get("mean") is None else round(st["mean"], dp)
        sec = lambda st: None if not st or st.get("mean") is None else round(st["mean"] / 1000, 1)
        jp = lambda k: None if not jb.get("n") or jb.get(k) is None else round(jb[k] * 100)
        return [label, blk["n"],
                _pct_sd(A["recall"]), _pct_sd(B["recall"]),
                _pct_sd(A["precision"]), _pct_sd(B["precision"]),
                _pct_sd(A["f1"]), _pct_sd(B["f1"]),
                mn(A["mrr"]), mn(B["reads_to_gold"], 1), mn(blk["overlap"]),
                sec(A["latency_ms"]), sec(B["latency_ms"]),
                mn(A["tokens"], 0), mn(B["tokens"], 0),
                jp("A_win"), jp("B_win"), jp("tie")]

    for t in m["per_type"]:
        ws.append(summary_row(t["type"], t))
    ws.append(summary_row("OVERALL", m["overall"]))
    for c in range(1, len(cols1) + 1):          # bold the overall row
        ws.cell(row=ws.max_row, column=c).font = Font(bold=True)
        ws.cell(row=ws.max_row, column=c).fill = TOTFILL
    style_header(ws, len(cols1))

    # ---- Sheet 2: Detail (per question, numeric) -----------------------------
    ws2 = wb.create_sheet("Detail")
    ws2.append(_CSV_COLS)
    pct_cols = {_CSV_COLS.index(c) + 1 for c in
                ("A_recall", "B_recall", "A_precision", "B_precision", "A_f1", "B_f1")}
    for row in detail_rows(recs) + totals_rows(recs):
        ws2.append([row.get(c) for c in _CSV_COLS])
        is_total = row["question_id"] == "__TOTAL__"
        for ci in pct_cols:                     # show 0-1 floats as percentages
            cell = ws2.cell(row=ws2.max_row, column=ci)
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0%"
        if is_total:
            for c in range(1, len(_CSV_COLS) + 1):
                ws2.cell(row=ws2.max_row, column=c).fill = TOTFILL
                ws2.cell(row=ws2.max_row, column=c).font = Font(bold=True)
    style_header(ws2, len(_CSV_COLS))

    # ---- Sheet 3: Answers ----------------------------------------------------
    ws3 = wb.create_sheet("Answers")
    cols3 = ["question_id", "lang", "question_type", "content_type", "repeat",
             "question", "gold", "A_answer", "B_answer", "judge", "judge_rationale"]
    ws3.append(cols3)
    for rec in _answer_records(rows):
        ws3.append([rec.get(c, "") for c in cols3])
        for ci in (6, 7, 8, 9, 11):             # wrap question, gold, answers, rationale
            ws3.cell(row=ws3.max_row, column=ci).alignment = wrap_top
    style_header(ws3, len(cols3))

    # column widths (readable defaults; answer columns wide)
    widths = {"Summary": [14, 5] + [12] * 16,
              "Detail": [22, 44, 14, 12, 9, 6] + [11] * 9,
              "Answers": [20, 6, 14, 12, 7, 46, 22, 60, 60, 7, 40]}
    for name, ws_ in (("Summary", ws), ("Detail", ws2), ("Answers", ws3)):
        for i, wdt in enumerate(widths[name], start=1):
            ws_.column_dimensions[get_column_letter(i)].width = wdt

    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()
