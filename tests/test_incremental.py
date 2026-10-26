"""Regression tests for incremental, non-destructive index building.

The rules being locked in:
  1. only reprocess for a real reason (content changed, new document, or an
     architectural switch) — never for a parameter that is merely being tested;
  2. each configuration gets its OWN namespace, and switching back to a
     previously built one costs nothing and destroys nothing;
  3. Pipeline A and Pipeline B rebuild independently, one document at a time.

Pinecone and the embedding model are stubbed, so this runs offline.

Run:  python -m tests.test_incremental      (no pytest needed)
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, ingestion, pageindex, store, vector_rag  # noqa: E402


class FakeIndex:
    """Records upserts/deletes per namespace instead of talking to Pinecone."""

    def __init__(self):
        self.vectors = {}          # namespace -> {id: metadata}
        self.upserts = 0
        self.deleted = []

    def upsert(self, vectors, namespace):
        self.upserts += len(vectors)
        ns = self.vectors.setdefault(namespace, {})
        for v in vectors:
            ns[v["id"]] = v["metadata"]

    def delete(self, ids=None, namespace=None, delete_all=False):
        if delete_all:
            raise AssertionError("the namespace must never be wiped wholesale")
        self.deleted.extend(ids or [])
        for i in (ids or []):
            self.vectors.get(namespace, {}).pop(i, None)


class FakeModel:
    def __init__(self, dim=8):
        self.dim = dim

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, **kw):
        import numpy as np
        return np.array([[float(len(t) % 7)] * self.dim for t in texts])


def check(label, got, want):
    status = "ok  " if got == want else "FAIL"
    print(f"  [{status}] {label}: got={got!r} want={want!r}")
    return got == want


def _doc(name, body):
    return {"name": name, "toc": [], "tree": None, "tree_nodes": 0, "tables": [],
            "clauses": None,
            "pages": [{"page": 1, "text": f"5 \nWerkstoffe \n{body} " * 6},
                      {"page": 2, "text": f"6 \nPruefung \n{body} pruefen. " * 6}]}


def main():
    ok = True
    idx = FakeIndex()
    model = FakeModel()

    # stub out Pinecone + the embedder, and give the manifest a temp home
    vector_rag._pinecone_index = lambda dim: idx
    vector_rag.embedder = lambda: model
    tmp = tempfile.mkdtemp()
    config.DATA_DIR = tmp
    config.PINECONE_API_KEY = "test"

    ingestion.DOCS.clear()
    ingestion.DOCS["A.pdf"] = _doc("A.pdf", "Der Werkstoff ist geeignet.")
    ingestion.DOCS["B.pdf"] = _doc("B.pdf", "Die Schrauben sind vorgespannt.")

    # ---- 1. first build embeds everything
    r1 = vector_rag.build_index()
    ns1 = r1["namespace"]
    ok &= check("first build embeds both docs", sorted(r1["indexed_docs"]), ["A.pdf", "B.pdf"])
    ok &= check("nothing reused yet", r1["reused_docs"], [])
    first_upserts = idx.upserts
    ok &= check("vectors written", first_upserts > 0, True)

    # ---- 2. rebuilding an unchanged corpus embeds nothing
    idx.upserts = 0
    r2 = vector_rag.build_index()
    ok &= check("second build re-embeds nothing", idx.upserts, 0)
    ok &= check("both docs reported reused", sorted(r2["reused_docs"]), ["A.pdf", "B.pdf"])
    ok &= check("namespace unchanged", r2["namespace"], ns1)

    # ---- 3. a query-time parameter is NOT a reason to reprocess
    idx.upserts = 0
    config.DEFAULT_TOP_K = 9
    r3 = vector_rag.build_index()
    ok &= check("top_k change re-embeds nothing", idx.upserts, 0)
    ok &= check("top_k change keeps the same namespace", r3["namespace"], ns1)

    # ---- 4. an index-time parameter selects a DIFFERENT namespace, and the old
    #         one is neither destroyed nor rebuilt when we switch back
    idx.upserts = 0
    original_size = config.CHUNK_SIZE
    config.CHUNK_SIZE = 400
    r4 = vector_rag.build_index()
    ok &= check("chunk_size change selects a new namespace", r4["namespace"] != ns1, True)
    ok &= check("new condition is built", sorted(r4["indexed_docs"]), ["A.pdf", "B.pdf"])
    ok &= check("old namespace still holds its vectors", len(idx.vectors[ns1]) > 0, True)

    idx.upserts = 0
    config.CHUNK_SIZE = original_size
    r5 = vector_rag.build_index()
    ok &= check("switching back re-embeds nothing", idx.upserts, 0)
    ok &= check("switching back reuses the original namespace", r5["namespace"], ns1)
    ok &= check("both conditions coexist", len(idx.vectors), 2)
    ok &= check("no namespace was ever wiped", idx.deleted, [])

    # ---- 5. adding a document processes ONLY that document
    idx.upserts = 0
    ingestion.DOCS["C.pdf"] = _doc("C.pdf", "Die Beschichtung ist zweilagig.")
    r6 = vector_rag.build_index()
    ok &= check("only the new document is embedded", r6["indexed_docs"], ["C.pdf"])
    ok &= check("the other two are reused", sorted(r6["reused_docs"]), ["A.pdf", "B.pdf"])

    # ---- 6. a document whose CONTENT changed is re-indexed alone, and its old
    #         vectors are deleted by id (not by wiping the namespace)
    idx.upserts, idx.deleted = 0, []
    before = set(idx.vectors[ns1])
    ingestion.DOCS["A.pdf"]["pages"][0]["text"] = "5 \nWerkstoffe \nGeaenderter Inhalt. " * 6
    r7 = vector_rag.build_index()
    ok &= check("changed doc rebuilt alone", r7["indexed_docs"], ["A.pdf"])
    ok &= check("its old vectors were deleted by id", len(idx.deleted) > 0, True)
    ok &= check("other docs' vectors untouched",
                all(i in idx.vectors[ns1] for i in before
                    if not i.startswith("A-pdf")), True)

    # ---- 7. restricting a build to one document
    idx.upserts = 0
    r8 = vector_rag.build_index(only=["B.pdf"], force=True)
    ok &= check("only=[B] rebuilds just B", r8["indexed_docs"], ["B.pdf"])

    # ---- 8. namespace readiness reporting
    st = vector_rag.namespace_state()
    ok &= check("namespace reports ready", (st["ready"], st["missing"]), (True, []))
    conds = vector_rag.known_conditions()
    ok &= check("both conditions listed", len(conds), 2)
    ok &= check("exactly one is current", sum(1 for c in conds if c["current"]), 1)

    # ---- 9. Pipeline B is independent and per-document
    for d in ingestion.DOCS.values():
        d["tree"] = {"title": "t", "source": "toc", "children": []}
        d["tree_nodes"] = 1
    res = pageindex.build_all_trees()
    ok &= check("existing trees are not rebuilt", res["built"], {})
    ok &= check("all trees reported reused", len(res["reused"]), 3)

    ingestion.DOCS["C.pdf"]["tree"] = None
    called = []
    original_build = pageindex.build_tree
    pageindex.build_tree = lambda d: (called.append(d["name"])
                                      or {"title": "t", "source": "toc", "children": []})
    try:
        res = pageindex.build_all_trees()
    finally:
        pageindex.build_tree = original_build
    ok &= check("only the treeless document is structured", called, ["C.pdf"])

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
