"""OpenAI client helpers + question-language detection.

Language handling: instead of hoping the model mirrors the question's language,
we DETECT it server-side (langdetect) and inject an explicit instruction like
"Write your final answer in Turkish." into every system prompt.
"""
from openai import OpenAI

from . import config

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def chat(messages: list[dict], tools: list[dict] | None = None,
         max_completion_tokens: int = 2500, model: str | None = None):
    """Single Chat Completions call with gpt-5.x quirks handled.

    `model` overrides config.OPENAI_MODEL (used by the LLM judge to run a
    different model from the generator).
    """
    model = model or config.OPENAI_MODEL
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if config.is_gpt5_reasoning(model):
        kwargs["reasoning_effort"] = config.REASONING_EFFORT
        # note: gpt-5.x rejects custom temperature -> we never send one
    if config.RUN_SEED is not None:
        kwargs["seed"] = config.RUN_SEED  # best-effort determinism (OpenAI)
    if tools:
        kwargs["tools"] = tools
    return client().chat.completions.create(**kwargs)


LANG_NAMES = {
    "tr": "Turkish", "en": "English", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "nl": "Dutch", "pt": "Portuguese",
    "ru": "Russian", "ar": "Arabic", "fa": "Persian", "az": "Azerbaijani",
    "el": "Greek", "pl": "Polish", "sv": "Swedish", "ja": "Japanese",
    "ko": "Korean", "zh-cn": "Chinese", "zh-tw": "Chinese",
}


def question_language(text: str) -> str:
    """Return an explicit language name for the answer, with a safe fallback.

    langdetect is unreliable on very short strings, so we only trust
    high-confidence detections; otherwise the model is told to mirror
    the question's language.
    """
    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 0  # deterministic
        cands = detect_langs(text)
        if cands and cands[0].prob >= 0.80 and len(text.strip()) >= 12:
            return LANG_NAMES.get(cands[0].lang, cands[0].lang)
    except Exception:
        pass
    return "the same language as the question"
