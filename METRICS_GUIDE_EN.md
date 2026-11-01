# RetrievalLab — Metrics Guide (how every number is computed)

This document defines every metric in the Metrics dashboard, the exported
Excel/CSV and the LLM-as-a-Judge workbooks, gives the exact formula, points at the
code that computes it, and works a concrete example for each. The dashboard and
the Excel/CSV export read the same records from `app/metrics.py`. The
reference-guided judge and the multi-round judge results (§7.2–7.4) come from
offline scripts and appear only in their own workbooks.

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

## 7. LLM-as-a-Judge (answer quality)

§1–5 measure whether the right pages were found; the judges measure the quality
of the written answer. Two LLM judges score the same logged answers:

| | Pairwise judge (§7.1) | Reference-guided judge (§7.2) |
|---|---|---|
| Question | which of the two answers is better? | how good is this answer on its own? |
| Judge sees | question + both answers | one answer + its context, or + the reference answer |
| Output | winner `A`/`B`/`tie`, plus three 1–5 scores per answer | Groundedness, Answer Correctness, Completeness (1–5 per answer) |
| Model | `JUDGE_PROVIDER`/`JUDGE_MODEL` (default `openai`/`gpt-5.4`) | `claude-sonnet-5-5`, effort `low` |
| Code | `app/judge.py`, `app/judge_batch.py` | `app/judge_ref.py` |
| Log rows | `phase: "judge"` | `phase: "judge_ref"` |
| Report | `answer_quality_xlsx.py` → `../answer-quality.xlsx` | `judge_ref_xlsx.py` → `../LLM as a Judge.xlsx` |
| Dashboard / CSV | win shares | not shown |

**Same words, different measures.** The pairwise scores `grounded`, `complete` and
`correct` are not the §7.2 metrics. The pairwise judge sees no sources and no
reference, so its `grounded` rests on the citations visible in the answer and its
`correct` on the judge's own knowledge. `answer-quality.xlsx` also labels them
Faithfulness, Comprehensiveness and Correctness; that Faithfulness is not the RAGAS
metric of the same name.

### 7.1 Pairwise judge — which answer is better?

**Method** (`app/judge.py`):
- **Blind** — the judge sees "Answer 1" / "Answer 2", never "vector"/"reasoning".
- **Randomized order** — a coin flip per verdict decides which pipeline is Answer 1,
  so position bias cancels out; reproducible when `RUN_SEED` is set.
- **Scoring** — each answer gets 1–5 on `grounded`, `complete`, `correct`, and the
  judge returns a winner `1` / `2` / `tie`.
- **Mapping** — `judge_pair()` converts `1`/`2` back to `A`/`B` using the coin flip
  (stored as `a_was_answer`) before logging, so the log holds `A`, `B` or `tie`.

**Formula**
```
A_win = wins(A) / judged runs    B_win = wins(B) / judged runs    tie = ties / judged runs
```
Code: `summarize()` and `_judge_str()` in `app/metrics.py`. A run is judged only if
both pipelines answered without error. The dashboard and export keep one verdict
per run, the last one logged; the results over several rounds are in
`answer-quality.xlsx` (§7.3).

**Worked example**
9 judged questions: A wins 5, B wins 3, tie 1 →
`A 56% · B 33% · tie 11%` (5/9, 3/9, 1/9).

### 7.2 Reference-guided judge — Groundedness, Answer Correctness, Completeness

**Question it answers:** *how good is each answer, measured against its evidence and
against a reference answer?*

**Method** (`app/judge_ref.py`): each answer of each pipeline is scored alone, never
beside the other pipeline's answer, so there is no winner and no order to randomize.
System labels are hidden. Each answer gets two calls:

| Call | Metric | Judge sees | Judge does not see |
|---|---|---|---|
| `G` | Groundedness | question, the context this pipeline had, answer | reference answer |
| `CC` | Answer Correctness, Completeness | question, reference answer, answer | context |

`CC` is one call because the inputs are the same, but the two metrics have separate
rubrics and are scored independently.

- **Context** comes from `data/ragas_contexts_<batch>.json`
  (`python -m app.ragas_eval --batch <id> --build-contexts`). For A it is the full
  text of the top-*k* chunks; for B, the output of every content tool the agent
  called (`read_node`, `read_pages`, `read_clause`, `keyword_search`), replayed on
  the stored documents. Table images are not part of it.
- **Reference** comes from `data/reference_answers.json`, one text per question id.

**What each metric measures**

| Metric | Compared against | Ignores |
|---|---|---|
| **Groundedness** | the retrieved context | whether the claim is true in reality |
| **Answer Correctness** | the reference answer | missing information; only what is stated counts |
| **Completeness** | the question and the key points of the reference | wrong information; only coverage counts |

**Rubric** (summary; the verbatim prompts are `RUBRIC_G` and `RUBRIC_CC`, reproduced
in the workbook's `Rubric` sheet)

| Score | Groundedness | Answer Correctness | Completeness |
|---|---|---|---|
| 5 | all claims explicitly supported by the context | all stated facts and numbers agree with the reference | all parts of the question and all key points covered |
| 4 | one very small detail missing from the context, no contradiction | main facts right, a minor detail slightly vague, nothing wrong | main parts covered, only a secondary detail missing |
| 3 | main claim supported, an important part is not | partly right, with an important error or wrong number | part of the main points answered, another important part missing |
| 2 | most claims not in the context | most stated information wrong | only a very small part covered |
| 1 | main claim absent from or contradicted by the context | wrong, contradicts the reference, or off-topic | question not answered, or no key point present |

**Edge-case rules**
- An answer that only says the information is not in the sources makes no factual
  claim: Groundedness = 5.
- Answerable question, but the answer declines or is empty: Correctness = 1.
- Unanswerable control (the reference says the sources lack the information): an
  answer that says so gets Correctness = 5 and Completeness = 5; an answer that gives
  a specific value or rule gets Correctness = 1.
- Information beyond the reference is not wrong unless it contradicts the reference.
- Key points are what the question asks for. Details it does not ask for (clause or
  table numbers, background, exceptions for other cases) cost no Completeness.

**Formula**
The judge returns the scores directly as integers from 1 to 5. The only derived
per-answer value is an unweighted mean:
```
Mean of the three = (Groundedness + Correctness + Completeness) / 3
```
Code: `score()` in `judge_ref_xlsx.py`. A test case counts only if both systems have
all three scores. Rows with a parse or API error are ignored by the workbook.

**Worked example** (illustrative scores following the rubric, not logged verdicts)
Question: *What is the minimum value, and for which steel thickness does it apply?*
Reference: *70 µm, for steel thicker than 6 mm.*

| Answer | Context contains | G | C | Comp | Why |
|---|---|---|---|---|---|
| "70 µm." | 70 µm | 5 | 5 | 3 | what it states is right; the thickness was asked and is missing |
| "85 µm, for steel thicker than 6 mm." | 85 µm, > 6 mm (wrong passage) | 5 | 3 | 5 | faithful to the context, but the number is wrong |
| "70 µm, for steel thicker than 6 mm." | neither value | 1 | 5 | 5 | right, but not from the retrieved evidence |
| "The sources do not contain this." | anything | 5 | 1 | 1 | no claim to ground, and the question stays unanswered |

### 7.3 From scores to table cells

**Repeat vs round.** A *repeat* runs the pipeline again and produces a new answer. A
*round* judges the same answer again, and it exists because the judge itself is
non-deterministic. The final run has 89 questions × 3 repeats = 267 test cases, each
judged in rounds 1–3 (pilot round 99 is ignored).

**Reference-guided judge**
```
score per answer  = mean of its 3 rounds                  combine()
table cell        = mean ± sd over the 267 test cases     build(), mean_sd()
```
`mean_sd()` uses the population standard deviation, as in §8. Unanswerable controls
are included in these means and appear as their own group in the content-type table.

**Worked example**
One answer of A scores G `5, 5, 4`, C `3, 3, 3`, Comp `4, 5, 4` in the three rounds.
```
G = 4.67    C = 3.00    Comp = 4.33    Mean of the three = (4.67 + 3.00 + 4.33) / 3 = 4.00
```

**Pairwise judge** (`combine()` in `answer_quality_xlsx.py`): rubric scores are
averaged over the rounds, and a verdict becomes fractions of a win. If B won 2 of 3
rounds and A won 1, the test case counts as B 0.67, A 0.33, so the win shares equal
the mean of the rounds' shares. For a question, the system with the larger summed
weight over its 3 repeats wins; equal weights are a tie.

### 7.4 Significance tests and judge consistency

Tests run per **question**, not per test case, because the repeats of a question are
not independent.

**Wilcoxon signed-rank test** (`scipy.stats.wilcoxon`, two-sided). Per question,
`d = mean(B) − mean(A)` over its repeats; zero differences are dropped. For example,
a question whose answers have mean-of-three values A `4.00, 4.00, 4.00` and
B `5.00, 4.67, 4.67` gives `d = 4.78 − 4.00 = 0.78`. Five questions with
`d = 0.50, −0.20, 0.30, 0.00, 0.80` lose the zero, and the remaining ranks give
`W⁺ = 9`, `W⁻ = 1`, so `W = 1`, `p = 0.25`. `LLM as a Judge.xlsx` reports it per metric
and for the mean of the three; `answer-quality.xlsx` only for the mean of the three.

**Sign test** (pairwise only). A two-sided binomial test of the questions won by B
out of those won by A or B, with `H₀ = 0.5`; tied questions are dropped. B 7, A 2,
tie 1 gives `p = 0.18`. The same test on the share of decisive verdicts won by the
answer shown first checks for position bias.

**Consistency between rounds** (*would the judge give the same result again?*)

| Measure | Judge | Formula | Code |
|---|---|---|---|
| Identical score / same winner | both | share of items equal in all rounds | `consistency_tables`, `agreement_tables` |
| Max − min ≤ 1 | reference-guided | share of answers whose rounds differ by at most 1 | `consistency_tables` |
| Mean SD between runs | both | mean over answers of the SD of their round scores | both of the above |
| Fleiss' κ | pairwise | `(P̄ − Pₑ) / (1 − Pₑ)` | `agreement_tables` |
| Krippendorff's α (ordinal) | reference-guided | `1 − (n−1) · Σ o_ck δ²_ck / Σ n_c n_k δ²_ck` | `krippendorff_ordinal` |
| Same direction | reference-guided | share of questions where B better / equal / A better holds in all rounds | `consistency_tables` |

Both coefficients treat each round as a rater and correct agreement for chance.
For κ, five test cases with verdicts `BBB, BBB, AAA, BBA, tBt` give `P̄ = 0.733`,
`Pₑ = 0.449`, `κ = 0.52`. For α, four answers scored `555, 455, 333, 232` give
`α = 0.87`. Read α ≥ 0.80 as reliable and 0.667–0.80 as tentative (Krippendorff,
2004); read κ 0.61–0.80 as substantial and above 0.80 as almost perfect (Landis &
Koch, 1977). Both can be low when one value dominates: 20 answers, 18 of them scored
`555` and two with a single 4, give `α = −0.02` despite 90% identical scores. Report
raw agreement next to them.

### 7.5 Caveats (state them in the thesis)

- **Identity leaks.** A cites as `[n]` and B as `(Doc, p.X)`, and in call `G` the
  context layout also differs (A: numbered chunks; B: `[Excerpt n]` blocks). Hidden
  labels are not full blinding. The pairwise judge adds randomized order; the
  reference-guided rubric tells the judge to ignore citation style.
- **Same-family risk.** The generator is `gpt-5.4-mini`, so the default pairwise judge
  (`gpt-5.4`) is the same family unless `JUDGE_PROVIDER=anthropic` is set. The
  reference-guided judge is cross-family (Claude judging GPT answers).
- **Read Groundedness with Correctness.** An abstention scores 5 on Groundedness, and
  a grounded answer can still be wrong when the retrieved passage is the wrong one.
- **Table images are not in the context**, so a value read only from an attached image
  can lose Groundedness.
- **Reference answers.** Correctness and Completeness are only as good as
  `data/reference_answers.json`.
- **Ordinal scale.** Means of 1–5 scores treat the levels as equally spaced; report
  the score distribution (Table 2 of `LLM as a Judge.xlsx`) with them.

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
| Pairwise judge | share of A/B/tie wins | `app/judge.py` | not judged |
| Groundedness | judge score 1–5 vs. retrieved context | `app/judge_ref.py`, call `G` | no valid verdict |
| Answer Correctness | judge score 1–5 vs. reference (stated facts) | `app/judge_ref.py`, call `CC` | no valid verdict |
| Completeness | judge score 1–5 vs. reference (coverage) | `app/judge_ref.py`, call `CC` | no valid verdict |
| Mean of the three | `(G + C + Comp) / 3` | `score` in `judge_ref_xlsx.py` | any of the three missing |
| mean / std | macro-mean / population std | `_stats` | no numeric values |

**Key methodological points to state in the thesis:** metrics are page-level
(chunks deduped to pages, so precision's denominator is pages, not *k*);
macro-averaged over questions; population standard deviation; `±` is spread, not a
confidence interval; unanswerable questions are excluded from the retrieval means
and handled by the judges. The reference-guided scores are means over 3 judge rounds
and 267 test cases, tested per question with a Wilcoxon signed-rank test.
