"""Reference-guided LLM judge — single-answer 1-5 rubric scoring of the logged answers.

Three metrics per answer (each answer of each pipeline is scored on its own, never
side by side with the other pipeline):

  Groundedness        answer vs. the context THIS pipeline retrieved / read;
                      the reference answer is NOT shown           -> call "G"
  Answer Correctness  answer vs. the reference answer: only what is stated counts
  Completeness        answer vs. the question and the key points of the reference;
                      only what should be said counts             -> both in call "CC"

Correctness and Completeness share one call (same inputs) but are scored
independently, each against its own rubric. The rubric texts are the author's
1-5 tables (Chapter 3) plus three edge-case rules (declined answers, unanswerable
questions, correct information beyond the reference).

Inputs:
  answers     experiment log, phases "vector" (A) and "reasoning" (B) of the batch
  contexts    data/ragas_contexts_<batch>.json (A: top-k chunks, B: replayed tool output)
  references  data/reference_answers.json

Every verdict is appended to the log as  phase "judge_ref"  with  judge_round  and
call ("G" | "CC"); run_ids already scored for a round/call are skipped, so a run
can be resumed. Two modes:

    python -m app.judge_ref --batch batch-46ded84d --round 1 --pilot 5      # sync, first 5 questions
    python -m app.judge_ref --batch batch-46ded84d --round 1 --submit       # Message Batches API (-50 %)
    python -m app.judge_ref --batch batch-46ded84d --collect <msgbatch_id>  # fetch results into the log
"""
import argparse
import json
import re
import time
from pathlib import Path

from . import config, logstore

MODEL = "claude-sonnet-5-5"
EFFORT = "low"
MAX_TOKENS = 4000          # thinking counts toward max_tokens
PHASE = "judge_ref"

RULES_COMMON = """You are an impartial expert evaluator of answers to questions about technical standards (steel construction, corrosion protection, tolerances).
The answer was produced by a retrieval-augmented system. Citation markers in the answer, such as [1] or (document, p.X), only point to sources; judge the content, not the citation style.
Be strict and consistent. Do not prefer longer answers. Return ONLY a JSON object, no other text."""

RUBRIC_G = RULES_COMMON + """

TASK: Score GROUNDEDNESS — whether every claim in the answer is supported by the RETRIEVED CONTEXT given below. Judge only against this context; do not use your own knowledge and do not judge whether the answer is correct in reality.

5 Fully grounded: all claims and information in the answer are explicitly supported by the given context. There are no additions from outside the context and no hallucinations.
4 Mostly grounded: almost all claims are based on the context. One very small detail that does not affect the core of the answer is not in the context, but it does not contradict the context.
3 Partly grounded: the main claim is supported by the context, but an important part or claim of the answer is not found in the given context.
2 Mostly ungrounded: most claims are not in the given context; only a very small part matches the context.
1 Clear hallucination: the main claim of the answer is not in the given context at all, or it directly contradicts the information in the context.

Rule: an answer that only states that the information is not contained in the sources makes no factual claim and scores 5.

Output format: {"groundedness": <1-5>, "rationale": "<one or two sentences naming any unsupported or contradicting claim>"}"""

RUBRIC_CC = RULES_COMMON + """

TASK: Score two things INDEPENDENTLY against the REFERENCE ANSWER given below: ANSWER CORRECTNESS and COMPLETENESS.

ANSWER CORRECTNESS — how far the stated facts agree with the reference answer. Missing information does NOT lower this score; only the correctness of what is stated counts.
5 Fully correct: all information, numbers and details stated in the answer agree 100 % with the reference answer.
4 Correct (minor uncertainty): the main facts are fully correct; a minor detail is slightly vague, but the answer contains no wrong information.
3 Partly correct: part of the main information is correct, but the answer contains an important factual error or a wrong number or statement.
2 Mostly wrong: most of the stated information is wrong or contradicts the reference answer; only a very small part is correct.
1 Completely wrong: the answer is factually wrong, directly contradicts the reference answer, or is unrelated to the question.
Rules for correctness:
 (a) Information that is not contained in the reference answer does not count as wrong unless it contradicts the reference answer.
 (b) If the question is answerable (the reference gives an answer) and the answer declines to answer, says the information was not found, or is empty -> correctness = 1.
 (c) If the reference answer states that the sources do not contain the information: an answer that says so -> 5; an answer that gives a specific value or rule -> 1.

COMPLETENESS — how far the answer covers all parts of the question and all key points of the reference answer. Wrong information does NOT lower this score; only whether what should be said is covered.
5 Complete and comprehensive: all parts of the question are answered and all key points of the reference answer are fully covered.
4 Adequate (small gap): the main parts of the question and the essential points of the reference answer are covered; only a minor detail of secondary importance is missing.
3 Partly incomplete: part of the main points of the question / reference answer is answered, but another important part is missing.
2 Mostly incomplete: only a very small part of the question and of the reference answer is covered; most of the question remains unanswered.
1 No coverage: the question is not answered, or none of the key points of the reference answer appears in the answer.
Rules for completeness:
 (a) The key points are what the QUESTION asks for. Measure coverage against those key points as stated in the reference answer.
 (b) Supporting details in the reference that the question does not ask for (clause, table or footnote numbers, lists of related items, background, exceptions for other cases) are not key points. "Key points of the reference answer" in the levels above means only the key points defined in (a). If every point the question asks for is covered, the score is 5 even when such supporting details are missing.
 (c) If the reference answer states that the sources do not contain the information, an answer that says so -> 5.

Output format: {"correctness": <1-5>, "completeness": <1-5>, "rationale_correctness": "<one or two sentences naming any wrong statement>", "rationale_completeness": "<one or two sentences naming any missing key point>"}"""


# ------------------------------------------------------------------ data
def load_items(batch_id: str) -> list[dict]:
    """One item per (run_id, system) of the batch, with everything both calls need."""
    qfile = json.load(open(Path(config.DATA_DIR).parent / "questions.json", encoding="utf-8"))
    qs = qfile["questions"] if isinstance(qfile, dict) else qfile
    qno = {q["id"]: i for i, q in enumerate(qs, start=1)}
    refs = json.load(open(Path(config.DATA_DIR) / "reference_answers.json", encoding="utf-8"))["references"]
    ctx = json.load(open(Path(config.DATA_DIR) / f"ragas_contexts_{batch_id}.json", encoding="utf-8"))
    items = []
    for r in logstore.read_all():
        if r.get("batch_id") != batch_id or r.get("phase") not in ("vector", "reasoning") or r.get("error"):
            continue
        system = "A" if r["phase"] == "vector" else "B"
        texts = ctx[system][r["run_id"]]
        if system == "A":   # number the chunks exactly as pipeline A saw them ([1]..[k])
            blocks = [f"[{i}] ({h['doc']}, p.{h['page']})\n{t}" for i, (h, t) in
                      enumerate(zip(r["retrieved"], texts), start=1)]
        else:               # B: output of every content tool the agent called, in order
            blocks = [f"[Excerpt {i}]\n{t}" for i, t in enumerate(texts, start=1)]
        items.append({
            "run_id": r["run_id"], "system": system, "question_no": qno[r["question_id"]],
            "question_id": r["question_id"], "question_type": r.get("question_type"),
            "content_type": r.get("content_type"), "repeat_index": r.get("repeat_index"),
            "answerable": bool(r.get("gold_pages")), "question": r["question"],
            "answer": r.get("answer", ""), "context": "\n\n".join(blocks) or "(no context retrieved)",
            "reference": refs[r["question_id"]],
        })
    items.sort(key=lambda x: (x["question_no"], x["repeat_index"], x["system"]))
    return items


def build_request(item: dict, call: str) -> dict:
    """Messages API params for one call. The rubric is the (cacheable) system prompt."""
    if call == "G":
        system, user = RUBRIC_G, (f"QUESTION:\n{item['question']}\n\nRETRIEVED CONTEXT:\n{item['context']}"
                                  f"\n\nANSWER TO EVALUATE:\n{item['answer'] or '(empty answer)'}")
    else:
        system, user = RUBRIC_CC, (f"QUESTION:\n{item['question']}\n\nREFERENCE ANSWER:\n{item['reference']}"
                                   f"\n\nANSWER TO EVALUATE:\n{item['answer'] or '(empty answer)'}")
    return {"model": MODEL, "max_tokens": MAX_TOKENS, "output_config": {"effort": EFFORT},
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}]}


def parse(call: str, text: str) -> dict:
    raw = re.sub(r"^```(json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    keys = ("groundedness",) if call == "G" else ("correctness", "completeness")
    try:
        data = json.loads(m.group(0) if m else raw)
        for k in keys:
            data[k] = int(data[k])
    except (ValueError, TypeError, KeyError):
        data = parse_lenient(call, m.group(0) if m else raw)
    for k in keys:
        if not 1 <= data[k] <= 5:
            raise ValueError(f"{k}={data[k]} outside 1-5")
    return data


def parse_lenient(call: str, raw: str) -> dict:
    """Fallback for almost-JSON replies: unescaped quotes inside a rationale, or a rationale
    written under the score's key (e.g. a second "completeness": "<text>"). Scores are the
    first integer given for each key; each string value runs up to the next `", "key":`."""
    keys = ("groundedness",) if call == "G" else ("correctness", "completeness")
    data = {}
    for k in keys:
        m = re.search(rf'"{k}"\s*:\s*"?([1-5])\b', raw)
        if not m:
            raise ValueError(f"no score for {k}")
        data[k] = int(m.group(1))
    for k, v in re.findall(r'"(\w+)"\s*:\s*"(.*?)"\s*(?=,\s*"\w+"\s*:|\}\s*$)', raw, flags=re.S):
        name = k if k.startswith("rationale") else (f"rationale_{k}" if call == "CC" else "rationale")
        data.setdefault(name, v)
    data["parse"] = "lenient"
    return data


def reparse_errors(batch_id: str) -> None:
    """Re-parse the stored raw replies of rows that failed to parse; append fixed rows."""
    rows = logstore.read_all()
    ok = {(r["run_id"], r["system"], r["call"], r["judge_round"]) for r in rows
          if r.get("phase") == PHASE and r.get("batch_id") == batch_id and not r.get("error")}
    n = 0
    for r in rows:
        key = (r.get("run_id"), r.get("system"), r.get("call"), r.get("judge_round"))
        if (r.get("phase") != PHASE or r.get("batch_id") != batch_id or not r.get("raw")
                or key in ok):
            continue
        try:
            fixed = {k: v for k, v in r.items() if k not in ("error", "raw")}
            fixed.update(parse(r["call"], r["raw"]))
        except Exception as e:
            print(f"  still failing: Q{r['question_no']} {r['system']} {r['call']} r{r['judge_round']}: {e}")
            continue
        logstore.append(fixed)   # keeps the call's usage; the failed row is ignored by the reports
        ok.add(key)
        n += 1
    print(f"re-parsed {n} row(s)")


def log_row(item: dict, call: str, rnd: int, mode: str, msg=None, error: str | None = None) -> dict:
    row = {k: item[k] for k in ("run_id", "system", "question_no", "question_id", "question_type",
                                "content_type", "repeat_index", "answerable")}
    row.update({"batch_id": item.get("batch_id"), "phase": PHASE, "call": call, "judge_round": rnd,
                "judge_model": MODEL, "effort": EFFORT, "mode": mode})
    if msg is not None:
        text = next((b.text for b in msg.content if b.type == "text"), "")
        u = msg.usage
        row["usage"] = {"in": u.input_tokens, "out": u.output_tokens,
                        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}
        row["stop_reason"] = msg.stop_reason
        try:
            row.update(parse(call, text))
        except Exception as e:  # keep the raw text so nothing is lost
            row["error"] = f"parse: {e}"
            row["raw"] = text[:2000]
    if error:
        row["error"] = error
    return row


def done_keys(batch_id: str, rnd: int) -> set:
    return {(r["run_id"], r["system"], r["call"]) for r in logstore.read_all()
            if r.get("phase") == PHASE and r.get("batch_id") == batch_id
            and r.get("judge_round") == rnd and not r.get("error")}


# ------------------------------------------------------------------ modes
def client():
    import anthropic
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def run_sync(batch_id: str, rnd: int, items: list[dict]) -> None:
    c, done = client(), done_keys(batch_id, rnd)
    todo = [(it, call) for it in items for call in ("G", "CC") if (it["run_id"], it["system"], call) not in done]
    print(f"sync round {rnd}: {len(todo)} call(s)", flush=True)
    for n, (it, call) in enumerate(todo, start=1):
        it["batch_id"] = batch_id
        try:
            msg = c.messages.create(**build_request(it, call))
            row = log_row(it, call, rnd, "sync", msg=msg)
        except Exception as e:
            row = log_row(it, call, rnd, "sync", error=f"{type(e).__name__}: {e}"[:500])
        logstore.append(row)
        scores = {k: row.get(k) for k in ("groundedness", "correctness", "completeness") if k in row}
        print(f"  [{n}/{len(todo)}] Q{it['question_no']} r{it['repeat_index']} {it['system']} {call} "
              f"{scores or row.get('error')}", flush=True)


def manifest_path(batch_id: str) -> Path:
    return Path(config.DATA_DIR) / f"judge_ref_manifest_{batch_id}.json"


def submit(batch_id: str, rounds: list[int], items: list[dict]) -> str:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    mp = manifest_path(batch_id)
    manifest = json.loads(mp.read_text("utf-8")) if mp.exists() else {}
    reqs = []
    for rnd in rounds:
        done = done_keys(batch_id, rnd)
        for idx, it in enumerate(items):
            for call in ("G", "CC"):
                if (it["run_id"], it["system"], call) in done:
                    continue
                cid = f"r{rnd}-{call}-{it['system']}-{idx}"     # custom_id: [A-Za-z0-9_-]{1,64}
                manifest[cid] = {"run_id": it["run_id"], "system": it["system"], "call": call, "round": rnd}
                reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**build_request(it, call))))
    if not reqs:
        raise SystemExit("nothing to submit")
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    b = client().messages.batches.create(requests=reqs)
    print(f"submitted {len(reqs)} request(s) as {b.id} ({b.processing_status})")
    return b.id


def collect(batch_id: str, msg_batch_id: str, items: list[dict], wait: bool = True) -> None:
    c = client()
    while True:
        b = c.messages.batches.retrieve(msg_batch_id)
        if b.processing_status == "ended" or not wait:
            break
        print(f"  {b.processing_status}: {b.request_counts}", flush=True)
        time.sleep(60)
    if b.processing_status != "ended":
        print("not finished yet:", b.request_counts)
        return
    manifest = json.loads(manifest_path(batch_id).read_text("utf-8"))
    by_key = {(it["run_id"], it["system"]): it for it in items}
    n_ok = n_err = 0
    for res in c.messages.batches.results(msg_batch_id):
        m = manifest[res.custom_id]
        if (m["run_id"], m["system"], m["call"]) in done_keys(batch_id, m["round"]):
            continue
        it = dict(by_key[(m["run_id"], m["system"])], batch_id=batch_id)
        if res.result.type == "succeeded":
            row = log_row(it, m["call"], m["round"], "batch", msg=res.result.message)
        else:
            row = log_row(it, m["call"], m["round"], "batch", error=f"batch result: {res.result.type}")
        logstore.append(row)
        n_err += bool(row.get("error"))
        n_ok += not row.get("error")
    print(f"collected: {n_ok} ok, {n_err} error(s) -> {config.LOG_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reference-guided 1-5 judge over a logged batch")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--round", type=int, nargs="+", default=[1])
    ap.add_argument("--pilot", type=int, default=None, help="sync mode on the first N questions")
    ap.add_argument("--submit", action="store_true", help="submit as a Message Batch (-50 %%)")
    ap.add_argument("--collect", default=None, help="collect a finished Message Batch into the log")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--reparse", action="store_true", help="re-parse stored raw replies that failed")
    args = ap.parse_args()
    if args.reparse:
        return reparse_errors(args.batch)
    items = load_items(args.batch)
    if args.collect:
        collect(args.batch, args.collect, items, wait=not args.no_wait)
    elif args.submit:
        submit(args.batch, args.round, items)
    else:
        if args.pilot:
            keep = sorted({it["question_no"] for it in items})[:args.pilot]
            items = [it for it in items if it["question_no"] in keep]
        for rnd in args.round:
            run_sync(args.batch, rnd, items)


if __name__ == "__main__":
    main()
