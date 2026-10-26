"""LLM-as-judge — blind, paired, randomized-order preference evaluation.

Methodology (stated in Chapter 4):
  * Blind: the judge sees "Answer 1" / "Answer 2", never "vector" / "reasoning".
  * Randomized order: which pipeline is Answer 1 is decided by a seeded coin flip
    per run, so position bias cancels out and results are reproducible.
  * Cross-family: the generator is GPT, so a different-family judge (Anthropic)
    mitigates self-preference bias. Configurable; falls back to an OpenAI judge.

Known caveat (documented, not silently hidden): A cites as [n] and B as
(Doc, p.X); these stylistic tells can leak pipeline identity to the judge. The
randomized order + blind labels are the primary controls.
"""
import json
import random
import re

from . import config, llm

RUBRIC = """You are an impartial expert evaluator of answers about technical standards documents.
A user asked a question; two assistants (Answer 1 and Answer 2) responded. You are NOT told how either was produced — judge only their quality.

Score each answer 1-5 on:
- grounded: claims are supported by cited evidence, no invention
- complete: covers what the question asks
- correct: factually precise, right numbers/clauses

Rules: be objective; do NOT prefer the longer answer; a confident but unsupported answer scores low on grounded. If a question is unanswerable from the sources, the better answer is the one that clearly says so. Pick the overall better answer, or "tie" if genuinely equal.

Return ONLY JSON, no prose, no code fences:
{"winner":"1"|"2"|"tie","scores":{"1":{"grounded":n,"complete":n,"correct":n},"2":{"grounded":n,"complete":n,"correct":n}},"rationale":"one or two sentences"}"""


def _coin(seed_material: str) -> bool:
    """Deterministic per-run order when a seed is set; else truly random."""
    if config.RUN_SEED is not None:
        rng = random.Random(f"{config.RUN_SEED}:{seed_material}")
    else:
        rng = random.Random()
    return rng.random() < 0.5


def _parse_verdict(raw: str) -> dict:
    raw = re.sub(r"^```(json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    data = json.loads(raw)
    winner = str(data.get("winner", "tie")).strip().lower()
    if winner not in ("1", "2", "tie"):
        winner = "tie"
    return {"winner": winner, "scores": data.get("scores", {}),
            "rationale": str(data.get("rationale", ""))[:600]}


def _call_judge(prompt: str) -> tuple[str, dict]:
    """Return (raw_text, usage). Anthropic if configured, else OpenAI."""
    if config.JUDGE_PROVIDER == "anthropic" and config.ANTHROPIC_API_KEY:
        import anthropic  # lazy; only needed for the Anthropic judge
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=config.JUDGE_MODEL, max_tokens=1000,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        usage = {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens}
        return text, usage
    resp = llm.chat([{"role": "user", "content": prompt}],
                    max_completion_tokens=1000, model=config.JUDGE_MODEL)
    return (resp.choices[0].message.content or ""), \
           {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}


def judge_pair(question: str, answer_a: str, answer_b: str,
               seed_material: str = "") -> dict:
    """Blind paired judgement of A vs B. Returns the verdict mapped back to A/B."""
    a_is_first = _coin(seed_material or question)
    first, second = (answer_a, answer_b) if a_is_first else (answer_b, answer_a)

    prompt = (f"{RUBRIC}\n\nQUESTION:\n{question}\n\n"
              f"ANSWER 1:\n{first or '(no answer)'}\n\n"
              f"ANSWER 2:\n{second or '(no answer)'}")
    raw, usage = _call_judge(prompt)
    v = _parse_verdict(raw)

    # map winner/scores from 1/2 back to A/B honouring the randomized order
    def label_to_ab(lbl):
        if lbl == "tie":
            return "tie"
        first_is_a = a_is_first
        return "A" if (lbl == "1") == first_is_a else "B"

    scores_ab = {}
    for lbl, who in (("1", "A" if a_is_first else "B"), ("2", "B" if a_is_first else "A")):
        if lbl in v["scores"]:
            scores_ab[who] = v["scores"][lbl]

    return {
        "winner": label_to_ab(v["winner"]),
        "scores": scores_ab,
        "rationale": v["rationale"],
        "judge_model": config.JUDGE_MODEL,
        "judge_provider": config.JUDGE_PROVIDER,
        "a_was_answer": 1 if a_is_first else 2,   # audit trail of the blind order
        "usage": usage,
    }
