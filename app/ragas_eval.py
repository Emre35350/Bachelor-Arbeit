"""RAGAS-style answer-quality metrics, implemented as defined in the paper.

Es, S., James, J., Espinosa-Anke, L. & Schockaert, S. (2024). Ragas: Automated
Evaluation of Retrieval Augmented Generation. EACL 2024 (System Demonstrations).

The three reference-free metrics, each computed per answer of each pipeline:

  Faithfulness       F  = |V| / |S|   statements of the answer (extracted by the LLM)
                                      that the retrieved context supports, over all
                                      statements
  Answer relevance   AR = 1/n Σ sim(q, qᵢ)   cosine similarity between the question
                                      and n questions the LLM generates from the answer
  Context relevance  CR = extracted sentences / sentences in the context
                                      sentences the LLM judges necessary to answer
                                      the question, over all context sentences

The prompt texts are the paper's, verbatim; only a JSON output format is appended
so the replies can be parsed reliably. Two operational choices are documented
here because the paper leaves them open:
  * n = 3 generated questions for answer relevance (the RAGAS library default);
  * for context relevance the context is split into numbered sentences up front
    and the LLM returns the numbers of the sentences it extracts — sentences are
    then counted identically for both pipelines, and they cannot be altered,
    which is what the paper's prompt demands ("not allowed to make any changes").

Context = exactly the text each pipeline had in front of it when answering:
  A (Vector-Based RAG)          the top-k chunks, full text from the Pinecone
                                namespace of the run (the log keeps only a
                                400-char snippet; the prefix is checked against it)
  B (Reasoning-Based Retrieval) the output of every content tool the agent called
                                (read_node, read_pages, read_clause,
                                keyword_search), replayed deterministically on the
                                persisted document store from the logged trace
Table images attached to either pipeline's answering call are not text and are
not part of the context.

    python -m app.ragas_eval --batch batch-46ded84d --build-contexts
    python -m app.ragas_eval --batch batch-46ded84d --round 1 [--workers 8] [--limit 4]
"""
import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config, llm, logstore

EVAL_MODEL = config.JUDGE_MODEL                     # gpt-5.4: not the generator (gpt-5.4-mini)
EMBED_MODEL = "text-embedding-ada-002"              # the paper's embedding model
N_QUESTIONS = 3
CONTENT_TOOLS = ("read_node", "read_pages", "read_clause", "keyword_search")


def _contexts_path(batch_id: str) -> Path:
    return Path(config.DATA_DIR) / f"ragas_contexts_{batch_id}.json"


# ------------------------------------------------------------------ contexts
def _context_a(rows_a: list[dict], namespace: str) -> dict[str, list[str]]:
    """Full text of the retrieved chunks, fetched from Pinecone by content id."""
    from pinecone import Pinecone
    manifest = json.loads((Path(config.DATA_DIR) / "index_manifest.json").read_text("utf-8"))[namespace]
    index = Pinecone(api_key=config.PINECONE_API_KEY).Index(manifest["index"])
    cache: dict[tuple, str] = {}
    out = {}
    for r in rows_a:
        texts = []
        for h in r["retrieved"]:
            key = (h["doc"], h["page"], h["chunk"])
            if key not in cache:
                ids = [i for i in manifest["docs"][h["doc"]]["ids"] if f"-p{h['page']}-" in i]
                for v in index.fetch(ids=ids, namespace=namespace).vectors.values():
                    md = v.metadata
                    cache[(md["doc"], int(md["page"]), int(md["chunk"]))] = md["text"]
            text = cache[key]
            if not text.startswith(h["snippet"]):
                raise ValueError(f"chunk text does not match the logged snippet: {r['run_id']} {key}")
            texts.append(text)
        out[r["run_id"]] = texts
    return out


def _context_b(rows_b: list[dict]) -> dict[str, list[str]]:
    """Replay the agent's content tools on the persisted store, checking every
    output against the logged observation preview."""
    from . import reasoning_agent, store
    store.load_docs()
    tools = {t.name: t for t in reasoning_agent.TOOLS}
    out = {}
    for r in rows_b:
        texts, pending = [], []
        for it in r.get("trace") or []:
            if it["t"] == "tool":
                pending.append(it)
            elif it["t"] == "obs" and pending:
                call = pending.pop(0)           # observations return in call order
                if call["name"] not in CONTENT_TOOLS:
                    continue
                text = tools[call["name"]].invoke(call["input"])
                if text[:200] != it["preview"][:200]:
                    raise ValueError(f"replayed tool output differs from the log: {r['run_id']}")
                texts.append(text)
        out[r["run_id"]] = texts
    return out


def build_contexts(batch_id: str) -> Path:
    rows = [r for r in logstore.read_all() if r.get("batch_id") == batch_id]
    a = [r for r in rows if r.get("phase") == "vector"]
    b = [r for r in rows if r.get("phase") == "reasoning"]
    namespace = a[0]["config"]["pinecone_namespace"]
    data = {"batch_id": batch_id, "namespace": namespace,
            "A": _context_a(a, namespace), "B": _context_b(b)}
    p = _contexts_path(batch_id)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# ------------------------------------------------------------------ sentences
_HEADER = re.compile(r"^\s*(\[(n\d+|Clause)\b[^\]]*\]|---.*---)\s*$")
_SEARCH_PREFIX = re.compile(r'^"[^"]+" p\.\d+ \(hits:\d+\): \.\.\.')
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9(])")
_NOISE = ("Document not found", "No lexical match", "Provide at least one keyword",
          "Tree not built", "Clause not found")
# German abbreviations whose full stop must not end a sentence
_ABBREV = re.compile(r"\b(bzw|z\. ?B|d\. ?h|u\. ?a|ggf|vgl|Nr|ca|max|min|evtl|inkl|usw|etc|"
                     r"sog|Abs|Abschn|Tab|Bild|Anm|einschl|ausschl|gem|lt|o\. ?ä|s)\.")
_DOT = "․"                                   # placeholder, restored after splitting


def _split(text: str) -> list[str]:
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", _DOT), text)
    return [s.replace(_DOT, ".") for s in _SPLIT.split(protected)]


def sentences(texts: list[str]) -> list[str]:
    """Context -> unique sentences. Tool headers are dropped, keyword-search
    prefixes stripped, wrapped PDF lines joined, then split at sentence ends;
    exact duplicates (text read twice) count once."""
    out, seen = [], set()
    for text in texts:
        if text.startswith(_NOISE):
            continue
        lines = []
        for line in text.splitlines():
            if _HEADER.match(line):
                lines.append("")                # a header ends the paragraph
                continue
            lines.append(_SEARCH_PREFIX.sub("", line).rstrip(". ") if _SEARCH_PREFIX.match(line) else line)
        for para in re.split(r"\n\s*\n", "\n".join(lines)):
            joined = " ".join(l.strip() for l in para.splitlines() if l.strip())
            for s in _split(joined):
                s = s.strip()
                if len(re.findall(r"\w", s)) < 3:
                    continue
                norm = re.sub(r"\s+", " ", s)
                if norm not in seen:
                    seen.add(norm)
                    out.append(norm)
    return out


# ------------------------------------------------------------------ LLM calls
FAITH_EXTRACT = ("Given a question and answer, create one or more statements from each sentence "
                 "in the given answer.\nquestion: {question}\nanswer: {answer}")
FAITH_VERIFY = ("Consider the given context and following statements, then determine whether they "
                "are supported by the information present in the context. Provide a brief "
                "explanation for each statement before arriving at the verdict (Yes/No). Provide a "
                "final verdict for each statement in order at the end in the given format. Do not "
                "deviate from the specified format.\ncontext: {context}\n{statements}")
AR_GENERATE = "Generate a question for the given answer.\nanswer: {answer}"
CR_EXTRACT = ("Please extract relevant sentences from the provided context that can potentially "
              "help answer the following question. If no relevant sentences are found, or if you "
              "believe the question cannot be answered from the given context, return the phrase "
              "'Insufficient Information'. While extracting candidate sentences you're not allowed "
              "to make any changes to sentences from given context.\nquestion: {question}\n"
              "context:\n{context}")

JSON_STATEMENTS = '\n\nReturn ONLY JSON: {"statements": ["...", "..."]}'
JSON_VERDICTS = ('\n\nReturn ONLY JSON, one entry per statement in the given order: '
                 '{"verdicts": [{"explanation": "...", "verdict": "Yes" | "No"}]}')
JSON_QUESTIONS = (f"\n\nGenerate {N_QUESTIONS} such questions, each one a question this answer "
                  f'would answer. Return ONLY JSON: {{"questions": ["...", "...", "..."]}}')
JSON_SENTENCES = ('\n\nThe context sentences are numbered. Return ONLY JSON with the numbers of the '
                  'extracted sentences: {"sentences": [3, 7]} — or {"sentences": [], '
                  '"insufficient_information": true}')


def _ask(prompt: str, usage: dict, max_tokens: int = 4000) -> dict:
    last = None
    for attempt in range(3):
        try:
            resp = llm.chat([{"role": "user", "content": prompt}],
                            max_completion_tokens=max_tokens, model=EVAL_MODEL)
            usage["in"] += resp.usage.prompt_tokens
            usage["out"] += resp.usage.completion_tokens
            raw = re.sub(r"^```(json)?|```$", "", (resp.choices[0].message.content or "").strip(),
                         flags=re.M).strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # LaTeX in answers (\(f_y\), \gamma) comes back as invalid JSON escapes
                return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw))
        except Exception as e:                       # parse error or transient API error
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after 3 attempts: {last}")


def faithfulness(question, answer, ctx_text, usage):
    stmts = [s for s in _ask(FAITH_EXTRACT.format(question=question, answer=answer)
                             + JSON_STATEMENTS, usage).get("statements", []) if str(s).strip()]
    if not stmts:
        return None, {"statements": [], "verdicts": []}
    listed = "\n".join(f"statement: {s}" for s in stmts)
    verdicts = _ask(FAITH_VERIFY.format(context=ctx_text, statements=listed) + JSON_VERDICTS,
                    usage, max_tokens=6000).get("verdicts", [])
    yes = [str(v.get("verdict", "")).strip().lower().startswith("yes") for v in verdicts]
    if len(yes) != len(stmts):
        raise RuntimeError(f"{len(stmts)} statements but {len(yes)} verdicts")
    return sum(yes) / len(stmts), {"statements": stmts, "verdicts": verdicts}


def _cos(u, v):
    return sum(a * b for a, b in zip(u, v)) / (math.sqrt(sum(a * a for a in u)) *
                                               math.sqrt(sum(b * b for b in v)))


def answer_relevance(question, answer, usage):
    qs = [q for q in _ask(AR_GENERATE.format(answer=answer) + JSON_QUESTIONS, usage)
          .get("questions", []) if str(q).strip()][:N_QUESTIONS]
    if not qs:
        return None, {"questions": []}
    emb = llm.client().embeddings.create(model=EMBED_MODEL, input=[question] + qs)
    vecs = [d.embedding for d in emb.data]
    sims = [_cos(vecs[0], v) for v in vecs[1:]]
    return sum(sims) / len(sims), {"questions": qs, "similarities": sims}


def context_relevance(question, sents, usage):
    if not sents:
        return None, {"n_sentences": 0, "selected": []}
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sents, start=1))
    reply = _ask(CR_EXTRACT.format(question=question, context=numbered) + JSON_SENTENCES, usage)
    sel = sorted({int(i) for i in reply.get("sentences", []) if str(i).isdigit()
                  and 1 <= int(i) <= len(sents)})
    return len(sel) / len(sents), {"n_sentences": len(sents), "selected": sel,
                                   "insufficient_information": bool(reply.get("insufficient_information"))}


# Citations are metadata about where a claim came from, not claims about the
# domain, and the two pipelines write them differently (A: "[1][4]", B:
# "(Doc.pdf, p.29)"). Left in, B's would be extracted as statements that the
# context text cannot support, and both would leak into the generated questions.
# They are removed from both answers identically before scoring.
_CITE_NUM = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\](?:\s*\[\d+(?:\s*[,–-]\s*\d+)*\])*")
_CITE_DOC = re.compile(r"\s*\((?=[^()]*(?:\.pdf|\bp\.\s?\d))[^()]*\)")


def strip_citations(answer: str) -> str:
    return re.sub(r"[ \t]+([.,;:])", r"\1", _CITE_DOC.sub("", _CITE_NUM.sub("", answer or ""))).strip()


def evaluate(question: str, answer: str, ctx_texts: list[str]) -> dict:
    usage = {"in": 0, "out": 0}
    sents = sentences(ctx_texts)
    ctx_text = "\n".join(sents)
    clean = strip_citations(answer)
    f, fd = faithfulness(question, clean, ctx_text, usage)
    ar, ad = answer_relevance(question, clean, usage)
    cr, cd = context_relevance(question, sents, usage)
    return {"faithfulness": f, "answer_relevance": ar, "context_relevance": cr,
            "detail": {"faithfulness": fd, "answer_relevance": ad, "context_relevance": cd,
                       "scored_answer": clean},
            "usage": usage}


# ------------------------------------------------------------------ batch
def ragas_round(row: dict) -> int:
    return int(row.get("ragas_round") or 1)


def pending(rows: list, batch_id: str, rnd: int) -> list[tuple[dict, str]]:
    done = {(r["run_id"], r["system"]) for r in rows
            if r.get("phase") == "ragas" and r.get("batch_id") == batch_id and ragas_round(r) == rnd}
    out = []
    for r in rows:
        if r.get("batch_id") != batch_id or r.get("error"):
            continue
        system = {"vector": "A", "reasoning": "B"}.get(r.get("phase"))
        if system and (r["run_id"], system) not in done:
            out.append((r, system))
    return sorted(out, key=lambda x: (x[0]["run_id"], x[1]))


META = ("run_id", "batch_id", "question_id", "question_type", "content_type",
        "repeat_index", "profile", "lang", "question", "gold_pages")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGAS metrics over a logged batch")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--build-contexts", action="store_true",
                    help="reconstruct and save both pipelines' contexts, then exit")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.build_contexts:
        print("written:", build_contexts(args.batch))
        return
    ctx = json.loads(_contexts_path(args.batch).read_text("utf-8"))
    todo = pending(logstore.read_all(), args.batch, args.round)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.batch} round {args.round}: {len(todo)} answer(s) to score with "
          f"{EVAL_MODEL} + {EMBED_MODEL}", flush=True)

    def job(row, system):
        return evaluate(row["question"], row.get("answer", ""), ctx[system][row["run_id"]])

    done = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(job, r, s): (r, s) for r, s in todo}
        for fut in as_completed(futs):
            r, s = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                failed += 1
                print(f"  FAILED {r['run_id']} {s}: {e}", flush=True)
                continue
            # appended from the main thread only, so log lines never interleave
            logstore.append({**{k: r.get(k) for k in META}, "phase": "ragas", "system": s,
                             "ragas_round": args.round, "eval_model": EVAL_MODEL,
                             "embedding_model": EMBED_MODEL, **res})
            done += 1
            print(f"  [{done + failed}/{len(todo)}] {r['run_id']} {s}  "
                  f"F={res['faithfulness']} AR={res['answer_relevance']} CR={res['context_relevance']}",
                  flush=True)
    print(f"Done: {done} scored, {failed} failed -> {config.LOG_PATH}")


if __name__ == "__main__":
    main()
