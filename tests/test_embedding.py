"""Regression tests for embedding-model dispatch (app/vector_rag.py).

Each model family signals the query/passage distinction differently — E5 uses
"query:/passage:" text prefixes, Jina v3 uses a `task` argument and no prefix.
Encoding the wrong way silently degrades retrieval, so this locks the dispatch
in without downloading any model.

Run:  python -m tests.test_embedding      (no pytest needed)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, vector_rag  # noqa: E402


class FakeModel:
    """Records exactly how encode() was called."""
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        return [[0.0] for _ in texts]


def check(label, got, want):
    ok = got == want
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}: got={got!r} want={want!r}")
    return ok


def main():
    ok = True
    original = config.EMBEDDING_MODEL

    # ---- family detection
    ok &= check("e5 family", vector_rag._model_family("intfloat/multilingual-e5-base"), "e5")
    ok &= check("jina v3 family", vector_rag._model_family("jinaai/jina-embeddings-v3"), "jina-v3")
    ok &= check("generic family", vector_rag._model_family("BAAI/bge-m3"), "generic")

    try:
        # ---- E5: textual prefixes, no task kwarg
        config.EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
        m = FakeModel()
        vector_rag._encode(m, ["was ist X?"], is_query=True)
        vector_rag._encode(m, ["Absatz A", "Absatz B"], is_query=False)
        ok &= check("E5 query gets 'query:' prefix", m.calls[0]["texts"], ["query: was ist X?"])
        ok &= check("E5 passage gets 'passage:' prefix",
                    m.calls[1]["texts"], ["passage: Absatz A", "passage: Absatz B"])
        ok &= check("E5 sends no task kwarg", "task" in m.calls[0], False)

        # ---- Jina v3: task adapter, NO textual prefix
        config.EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
        m = FakeModel()
        vector_rag._encode(m, ["was ist X?"], is_query=True)
        vector_rag._encode(m, ["Absatz A"], is_query=False)
        ok &= check("Jina query: no prefix", m.calls[0]["texts"], ["was ist X?"])
        ok &= check("Jina query task", m.calls[0]["task"], "retrieval.query")
        ok &= check("Jina passage task", m.calls[1]["task"], "retrieval.passage")

        # ---- generic: no prefix, no task
        config.EMBEDDING_MODEL = "BAAI/bge-m3"
        m = FakeModel()
        vector_rag._encode(m, ["x"], is_query=True)
        ok &= check("generic: no prefix", m.calls[0]["texts"], ["x"])
        ok &= check("generic: no task kwarg", "task" in m.calls[0], False)

        # ---- normalisation is always requested (cosine index)
        ok &= check("normalize_embeddings passed through", m.calls[0]["normalize_embeddings"], True)
    finally:
        config.EMBEDDING_MODEL = original

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
