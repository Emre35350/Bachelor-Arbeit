"""Central configuration — .env provides the defaults, the UI Configuration
tab can override a whitelisted subset at runtime via /config.

Two classes of runtime settings:
  query-time : model, reasoning_effort, seed, top_k, max_steps
  index-time : chunk_size, chunk_overlap, clause_chunking, embedding_model
Index-time changes make the built indexes STALE (compared against the
snapshot taken at the last successful build) until a rebuild.

snapshot() is attached to every experiment-log row so each run records the
exact configuration (and seed) it was produced under.
"""
import hashlib
import json
import os

from dotenv import load_dotenv

load_dotenv()


def _int_or_none(v) -> int | None:
    if v is None or str(v).strip() == "":
        return None
    return int(v)


# --- LLM (OpenAI) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")  # gpt-5.x family only
RUN_SEED: int | None = _int_or_none(os.getenv("RUN_SEED"))  # best-effort determinism; logged per run

# --- Embeddings (Pipeline A) ---
# multilingual-e5-base = 768 dims (~1.1 GB download on first run).
# For a lighter setup: intfloat/multilingual-e5-small (384 dims, ~470 MB).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# Models offered in the Configuration dropdown. Switching between them is
# non-destructive: each gets its own Pinecone index (per dimension) and namespace
# (per index-time config), so both stay queryable once built. The encode
# convention is dispatched per model in vector_rag.
# NOTE: jinaai/jina-embeddings-v3 is intentionally NOT offered — its remote
# modelling code targets transformers 4.x and breaks on the 5.x installed here
# ('all_tied_weights_keys'). bge-m3 gives comparable/stronger cross-lingual
# retrieval with no custom code. (The Jina encode path is kept in vector_rag for
# anyone who pins transformers 4.x and sets it via .env.)
KNOWN_EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-base",     # 768d, E5 "query:/passage:" prefixes
    "BAAI/bge-m3",                       # 1024d, no prefix; strong cross-lingual EN↔DE
]

# --- Pinecone (Pipeline A vector database) ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "retrievallab")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "main")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Chunking (Pipeline A) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))        # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))   # characters

# Cut chunks at clause headings instead of at a fixed character window; the
# character splitter above then only applies INSIDE a clause longer than
# chunk_size. Off = the plain sliding window over page text (the pre-Step-9
# behaviour), which is the ablation baseline for this domain customization.
CLAUSE_CHUNKING = os.getenv("CLAUSE_CHUNKING", "1").strip().lower() not in ("0", "false", "no", "")

# --- Tables: multi-vector handling (Pipeline A + B) ---
# Index a vision-written SUMMARY of each detected table (clean prose embeds and
# retrieves far better than the scrambled column text PDF extraction yields),
# and attach the rendered table IMAGE to the final answering call so exact
# values are read off the real table. Detection runs at upload, summarization
# once per table at index-build time (cached on the document, so repeated
# experiment runs never re-pay for it).
TABLE_SUMMARIES = os.getenv("TABLE_SUMMARIES", "1").strip().lower() not in ("0", "false", "no", "")
# Attaching images is query-time: it can be switched off to measure how much the
# image itself contributes over the indexed summary alone.
ATTACH_TABLE_IMAGES = os.getenv("ATTACH_TABLE_IMAGES", "1").strip().lower() not in ("0", "false", "no", "")
MAX_TABLE_IMAGES = int(os.getenv("MAX_TABLE_IMAGES", "3"))  # per answering call
# Empty = use OPENAI_MODEL (it is already vision-capable).
VISION_MODEL = os.getenv("VISION_MODEL", "").strip()

# --- Defaults / limits ---
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_MAX_STEPS = int(os.getenv("DEFAULT_MAX_STEPS", "8"))

# --- Experiment log ---
LOG_PATH = os.getenv("LOG_PATH", "experiments.jsonl")

# --- Persistence ---
# Parsed pages + PageIndex trees + build state are written here so a restart
# doesn't require re-upload/rebuild (Pinecone vectors persist in the cloud).
DATA_DIR = os.getenv("DATA_DIR", "data")
PERSIST = os.getenv("PERSIST", "1").strip().lower() not in ("0", "false", "no", "")

# --- Evaluation question set (gold_pages) ---
QUESTIONS_PATH = os.getenv("QUESTIONS_PATH", "questions.json")

# --- German morphology for keyword_search (Step 9 ablation, default off) ---
# When on, query terms are lightly de-suffixed so "Beschichtungen" also matches
# "Beschichtung(sdicke)". A tiny on/off ablation for German recall.
KEYWORD_STEM_DE = os.getenv("KEYWORD_STEM_DE", "0").strip().lower() in ("1", "true", "yes", "on")

# --- LLM judge (Step 8) ---
# Cross-family judging mitigates self-preference bias: the generator is GPT, so
# set an Anthropic judge (JUDGE_PROVIDER=anthropic + ANTHROPIC_API_KEY) for the
# strongest methodology. Falls back to an OpenAI judge model otherwise.
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "openai").strip().lower()  # openai | anthropic
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.4")  # a different model from the generator
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Cost model (USD per 1M tokens) ---
# Left at 0 by default so the dashboard never shows fabricated prices — put your
# model's real rates here (or in .env) to get cost columns.
COST_PER_1M_IN = float(os.getenv("COST_PER_1M_IN", "0"))
COST_PER_1M_OUT = float(os.getenv("COST_PER_1M_OUT", "0"))


def token_cost(tin: int, tout: int) -> float:
    return (tin / 1_000_000) * COST_PER_1M_IN + (tout / 1_000_000) * COST_PER_1M_OUT


def is_gpt5_reasoning(model: str | None = None) -> bool:
    """gpt-5.x models accept reasoning_effort but reject custom temperature."""
    m = model or OPENAI_MODEL
    return m.startswith("gpt-5") and "chat" not in m


# ====================================================================
# Runtime-mutable settings (the /config endpoint + UI Configuration tab)
# ====================================================================
REASONING_EFFORTS = ("minimal", "low", "medium", "high")
INDEX_KEYS = ("chunk_size", "chunk_overlap", "embedding_model", "clause_chunking",
              "table_summaries")
BOOL_KEYS = ("clause_chunking", "table_summaries", "attach_table_images")

LIMITS = {"top_k": (1, 10), "max_steps": (2, 12),
          "chunk_size": (200, 4000), "chunk_overlap": (0, 1000)}


def get_settings() -> dict:
    return {
        "model": OPENAI_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "seed": RUN_SEED,
        "top_k": DEFAULT_TOP_K,
        "max_steps": DEFAULT_MAX_STEPS,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "clause_chunking": CLAUSE_CHUNKING,
        "table_summaries": TABLE_SUMMARIES,
        "attach_table_images": ATTACH_TABLE_IMAGES,
        "embedding_model": EMBEDDING_MODEL,
    }


def index_settings() -> dict:
    """The index-time subset — the settings that actually change the vectors."""
    s = get_settings()
    return {k: s[k] for k in INDEX_KEYS}


def index_signature(settings: dict | None = None) -> str:
    """Short deterministic fingerprint of the index-time settings."""
    s = settings or index_settings()
    blob = json.dumps({k: s[k] for k in sorted(INDEX_KEYS)}, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def namespace(settings: dict | None = None) -> str:
    """Pinecone namespace for a configuration — one namespace per condition.

    Experiment conditions are not destructive of one another: switching the
    embedding model or the chunk size selects a DIFFERENT namespace instead of
    overwriting the current one, so flipping back and forth costs nothing after
    the first build of each condition.
    """
    return f"{PINECONE_NAMESPACE}-{index_signature(settings)}"


def index_name(dimension: int) -> str:
    """Pinecone index for an embedding dimension.

    Pinecone dimensions are immutable, so a 384-dim model cannot share an index
    with a 768-dim one. Keying the index NAME by dimension means switching
    embedding models no longer deletes and recreates the index (which used to
    destroy every previously built condition).
    """
    return f"{PINECONE_INDEX}-{int(dimension)}"


def _validate(key: str, raw):
    if key in ("model", "embedding_model"):
        v = str(raw or "").strip()
        if not v:
            raise ValueError(f"{key} cannot be empty")
        return v
    if key == "reasoning_effort":
        v = str(raw or "").strip().lower()
        if v not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}")
        return v
    if key in BOOL_KEYS:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if key == "seed":
        try:
            return _int_or_none(raw)
        except (TypeError, ValueError):
            raise ValueError("seed must be an integer or empty")
    if key in LIMITS:
        try:
            v = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer")
        lo, hi = LIMITS[key]
        if not lo <= v <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
        return v
    raise ValueError(f"unknown setting: {key}")


def update_settings(changes: dict) -> dict:
    """Validate and apply a partial settings update; returns only the keys
    whose value actually changed. Raises ValueError on any invalid value
    (nothing is applied in that case)."""
    global OPENAI_MODEL, REASONING_EFFORT, RUN_SEED, DEFAULT_TOP_K, DEFAULT_MAX_STEPS
    global CHUNK_SIZE, CHUNK_OVERLAP, CLAUSE_CHUNKING, EMBEDDING_MODEL
    global TABLE_SUMMARIES, ATTACH_TABLE_IMAGES

    current = get_settings()
    staged = dict(current)
    for key, raw in (changes or {}).items():
        staged[key] = _validate(key, raw)
    if staged["chunk_overlap"] >= staged["chunk_size"]:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    applied = {k: v for k, v in staged.items() if current[k] != v}
    OPENAI_MODEL = staged["model"]
    REASONING_EFFORT = staged["reasoning_effort"]
    RUN_SEED = staged["seed"]
    DEFAULT_TOP_K = staged["top_k"]
    DEFAULT_MAX_STEPS = staged["max_steps"]
    CHUNK_SIZE = staged["chunk_size"]
    CHUNK_OVERLAP = staged["chunk_overlap"]
    CLAUSE_CHUNKING = staged["clause_chunking"]
    TABLE_SUMMARIES = staged["table_summaries"]
    ATTACH_TABLE_IMAGES = staged["attach_table_images"]
    EMBEDDING_MODEL = staged["embedding_model"]
    return applied


def snapshot() -> dict:
    """Full experiment-relevant configuration — attached to every log row."""
    return {**get_settings(),
            "pinecone_index": PINECONE_INDEX,
            "pinecone_namespace": namespace(),      # the condition actually queried
            "index_signature": index_signature()}
