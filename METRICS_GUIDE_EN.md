# RetrievalLab — Metrics Guide (how every number is computed)

This document defines every metric in the Metrics dashboard and the exported
Excel/CSV, gives the exact formula, points at the code that computes it
(`app/metrics.py`), and works a concrete example for each. All numbers are
computed the same way in the dashboard, the drill-down popup, the CSV and the
Excel — they share one code path, so they never disagree.

---

## 0. What is being measured

Two retrieval pipelines answer the same question and are scored independently:

- **A — Vector RAG**: embeds the question, retrieves the top-*k* chunks from
  Pinecone, and answers from them.
- **B — Reasoning agent**: navigates a document tree and *reads* pages by
  reasoning (no embeddings).

The **unit of evaluation is the (document, page) pair.** Each question carries a
gold label `gold_pages` — the pages that actually contain the answer, e.g.

```json
"gold_pages": { "DIN_EN_ISO_1461_2022-12-00_DE_3388189.pdf": [23] }
```

Both pipelines produce an **evidence set** of pages, which is compared to the
gold set:

- for **A**, the evidence is the pages of the top-*k* retrieved chunks;
- for **B**, the evidence is the pages the agent actually read.

**Page-level, deduplicated.** Retrieved chunks are collapsed to their unique
pages before scoring, because gold is defined in pages (`pageset()` in
`app/metrics.py`). If three chunks all come from page 23, page 23 counts once.

Throughout, let:
- **G** = set of gold pages,
- **H** = set of evidence pages (the "hit" set),
- **H ∩ G** = the gold pages that were found.

---

## 1. Recall@k / Context Recall

**Question it answers:** *of the gold pages, how many were found?*

**Formula**
```
Recall = |H ∩ G| / |G|
```
Code: `_recall(hit, gold)`. Returns `None` when `G` is empty (unanswerable
control — excluded from averages, see §8).

For **A** this is *Recall@k* (recall using the top-*k* retrieval). For **B**,
which has no fixed *k*, the identical formula is *Context Recall* over the pages
it read.

**Worked example**
Gold `G = {(ISO1461, 23)}`. Pipeline A's top-5 chunks land on pages
`23, 23, 24, 5, 9` → unique evidence `H = {23, 24, 5, 9}`.
```
H ∩ G = {23}  →  Recall = 1 / 1 = 1.00 = 100%
```
If the gold were two pages `G = {23, 24}` and only 23 was retrieved,
Recall = 1/2 = 50%.

---

## 2. Precision@k / Context Precision

**Question it answers:** *of the pages that were retrieved, how many were gold?*

**Formula**
```
Precision = |H ∩ G| / |H|
```
Code: `_precision(hit, gold)`. If `H` is empty: `None` when `G` is also empty,
otherwise `0.0` (nothing retrieved on an answerable question scores 0).

**Important — the denominator is unique retrieved PAGES, not k.** A page
retrieved by three chunks counts once. So this is *context precision* at the
page level, not "fraction of the k chunks that are relevant".

**Worked example** (same retrieval as §1)
`H = {23, 24, 5, 9}` (4 unique pages), `G = {23}`.
```
H ∩ G = {23}  →  Precision = 1 / 4 = 0.25 = 25%
```
Note: this is **25%**, not 20% (= 1/5 chunk-level), because the five chunks
covered only four distinct pages.

---

## 3. F1 (retrieval)

**Question it answers:** *a single balanced score combining precision and recall.*

**Formula**
```
F1 = 2 · P · R / (P + R)      (equivalently: 2·|H ∩ G| / (|H| + |G|))
```
Code: `_f1(hit, gold)`. Returns `None` when `G` is empty; `0.0` when nothing
relevant was retrieved (`H ∩ G = ∅`).

**Worked example** (same as §1–2)
`P = 0.25`, `R = 1.00`.
```
F1 = 2 · 0.25 · 1.00 / (0.25 + 1.00) = 0.5 / 1.25 = 0.40 = 40%
```
Check with the direct form: `2·|H∩G| / (|H|+|G|) = 2·1 / (4+1) = 2/5 = 0.40`. ✓

**Second example** — gold two pages, one extra retrieved:
`G = {p1, p2}`, `H = {p1, p2, p3}` → `P = 2/3`, `R = 1` →
`F1 = 2·(2/3)·1 / (2/3 + 1) = 0.8 = 80%`.

---

## 4. MRR — Mean Reciprocal Rank (Pipeline A only)

**Question it answers:** *how high up the ranked list is the first gold page?*

**Formula (per question)**
```
RR = 1 / (rank of the first retrieved gold page)      (0 if no gold retrieved)
```
Code: `_mrr_from_ranked(retrieved, gold)`. It walks A's ranked chunk list (in
Pinecone similarity order), **deduplicating by page** so each distinct page gets
the next rank, and returns `1/rank` at the first gold page. `None` when `G` is
empty. "MRR" is the mean of these RR values over questions (§8).

Only A has a ranking; B uses *reads→gold* instead (§5).

**Worked example**
Ranked pages `[5, 5, 9, 23, 24]`, gold `G = {23}`. After page-dedup the ranks
are 5→1, 9→2, 23→3.
```
first gold (23) at rank 3  →  RR = 1/3 = 0.33
```
If page 23 were the very first retrieved, RR = 1/1 = 1.00. If no gold page were
retrieved at all, RR = 0.

---

## 5. reads→gold (Pipeline B only)

**Question it answers:** *how many pages did the agent read before reaching the
first gold page?*

**Formula (per question)**
```
reads→gold = 1-based position of the first gold page in B's read order
```
Code: `_reads_to_gold(read_order, gold)`. `None` when `G` is empty **or when the
agent never read a gold page** — so this metric only reflects *successful*
navigations and must be read alongside B's recall (which captures the misses).

**Worked example**
The agent reads pages in order `[10, 23, 24]`, gold `G = {23}`.
```
first gold (23) is the 2nd page read  →  reads→gold = 2
```
Lower is better (found it sooner). If the agent read `[10, 24, 30]` and never hit
23, the value is `None` and the run is excluded from the reads→gold average
(but its Recall is 0).

---

## 6. Overlap — Jaccard(A retrieved, B read)

**Question it answers:** *do the two paradigms look at the same pages?* (descriptive,
not a quality score).

**Formula**
```
Overlap = |A_pages ∩ B_pages| / |A_pages ∪ B_pages|
```
Code: `_jaccard(a_set, b_set)` with `a_set` = A's retrieved pages, `b_set` = B's
read pages. `None` when both are empty.

**Worked example**
A retrieved `{5, 9, 23, 24}`, B read `{23, 24, 30}`.
```
∩ = {23, 24} (2)   ∪ = {5, 9, 23, 24, 30} (5)   →  Overlap = 2/5 = 0.40
```
`0` = the pipelines looked at disjoint pages; `1` = identical page sets.

---

## 7. Judge — LLM-as-judge (answer quality)

**Question it answers:** *which pipeline gave the better answer?* This is separate
from retrieval: §1–5 measure whether the right pages were found; the judge
measures the quality of the written answer.

**Method** (`app/judge.py`):
- **Blind** — the judge sees "Answer 1" / "Answer 2", never "vector"/"reasoning".
- **Randomized order** — a seeded coin flip decides which pipeline is Answer 1,
  so position bias cancels out; reproducible when a seed is set.
- **Cross-family recommended** — the generator is GPT, so an Anthropic judge best
  mitigates self-preference bias (configurable; falls back to an OpenAI judge).
- Each answer is scored 1–5 on **grounded**, **complete**, **correct**; the judge
  returns a winner `1` / `2` / `tie`.

**Aggregation:** the dashboard maps `1→A`, `2→B` and reports the share of wins:
`A_win%`, `B_win%`, `tie%` over the judged runs.

**Worked example**
9 judged questions: A wins 5, B wins 3, tie 1 →
`A 56% · B 33% · tie 11%` (5/9, 3/9, 1/9).

**Caveat (state it in the thesis):** A cites as `[n]` and B as `(Doc, p.X)`;
these stylistic tells can leak pipeline identity. Blind labels + randomized order
are the primary controls. A same-family judge (e.g. GPT judging GPT) carries
residual self-preference risk.

---

## 8. Aggregation — how per-question values become a table cell

Every metric above is computed **per run** first, then averaged. For a group of
runs (a question-type, a content-type, or overall):

**Mean (macro-average over runs)**
```
mean = (1/N) · Σ xᵢ
```
This is the average of the per-question values — *not* `F1(mean P, mean R)`. Code:
`_stats()`; the field is aggregated across runs in `_agg_system()`.

**Standard deviation (population, ÷N)**
```
std = sqrt( (1/N) · Σ (xᵢ − mean)² )
```
Note it divides by **N**, not N−1 (population std, not sample std).

**Display: `mean% ± std`** — the dashboard and the Excel "Summary" sheet show, for
Recall/Precision/F1, the mean and std as percentages (`mMean()`), e.g. `11% ± 31`.
The **± is the spread of the values, not a confidence interval** — do not label it
as a 95% CI.

**Only gold-bearing runs count for the gold metrics.** Recall, Precision, F1, MRR
and reads→gold are averaged **only over runs that have gold** (`has_gold`), because
these are undefined without gold. Unanswerable controls are excluded from those
means (and counted separately as "Unanswerable"); the **judge still applies** to
them, since "the better answer is the one that clearly says it is not in the
sources".

**Worked example — where `11% ± 31` comes from**
A question-type has 9 questions. On one, Pipeline A found the gold page
(metric = 100%); on the other eight it did not (metric = 0%).
```
values = [100%, 0, 0, 0, 0, 0, 0, 0, 0]
mean   = 1.0 / 9                          = 0.111  → 11%
var    = [ (1−0.111)² + 8·(0−0.111)² ] / 9 = 0.0988
std    = √0.0988                          = 0.314  → 31
```
→ displayed as **`11% ± 31`**. A large std relative to the mean is normal for
recall/F1 on single-gold questions: the value is near-bimodal (0 or 1), so the
system "nails some, misses others", and that scatter is itself a finding.

**Repeats.** With `repeat = 1` the group's runs are its distinct questions, so the
spread is *between questions*. With `repeat ≥ 2` each question runs several times
(non-determinism), and the spread mixes between-question and run-to-run variance.

---

## 9. Efficiency metrics

Descriptive cost/speed columns, averaged like everything else (§8):

| Metric | Meaning | Source |
|---|---|---|
| **Latency** | wall-clock seconds per answer | `latency_ms` |
| **Tokens** | prompt + completion tokens | `usage.in + usage.out` |
| **Cost** | `tokens × price` (0 unless `COST_PER_1M_*` set in `.env`) | `config.token_cost()` |
| **Effort** | A: number of LLM calls · B: number of tool calls | `calls` / `tool_calls` |

---

## 10. Content-type breakdown (prose vs table)

Every question carries `content_type` (`prose` or `table`). It travels into the
log and appears in the drill-down popup and the Excel "Detail" sheet. It lets you
compare retrieval quality on **prose** answers versus **table** answers — the
direct test of whether the multi-vector table handling helps. (A dedicated
per-content-type summary row can be added on request.)

---

## 11. Summary of definitions

| Metric | Formula | Code | None when |
|---|---|---|---|
| Recall@k / Context Recall | `\|H∩G\| / \|G\|` | `_recall` | no gold |
| Precision@k / Context Precision | `\|H∩G\| / \|H\|` | `_precision` | no gold **and** nothing retrieved |
| F1 | `2·\|H∩G\| / (\|H\|+\|G\|)` | `_f1` | no gold |
| MRR (A) | `1 / rank of first gold page` | `_mrr_from_ranked` | no gold |
| reads→gold (B) | position of first gold page read | `_reads_to_gold` | no gold or never read |
| Overlap | `\|A∩B\| / \|A∪B\|` (pages) | `_jaccard` | both empty |
| Judge | share of A/B/tie wins | `app/judge.py` | not judged |
| mean / std | macro-mean / population std | `_stats` | no numeric values |

**Key methodological points to state in the thesis:** metrics are page-level
(chunks deduped to pages, so precision's denominator is pages, not *k*);
macro-averaged over questions; population standard deviation; `±` is spread, not a
confidence interval; unanswerable questions are excluded from the retrieval means
and handled by the judge.
