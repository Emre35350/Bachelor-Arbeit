"""Regression tests for the clause-aware chunker (app/vector_rag.py).

The property that matters for this domain: a chunk must never straddle a clause
boundary, because a requirement that lands in the same embedding as the tail of
the previous clause is retrievable for the wrong question. Character splitting
is allowed only INSIDE a clause that is longer than chunk_size.

Run:  python -m tests.test_chunking      (no pytest needed)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, vector_rag  # noqa: E402


def _doc(pages_text):
    return {"name": "T.pdf",
            "pages": [{"page": i, "text": t} for i, t in enumerate(pages_text, 1)],
            "toc": [], "clauses": None}


def check(label, got, want):
    status = "ok  " if got == want else "FAIL"
    print(f"  [{status}] {label}: got={got!r} want={want!r}")
    return got == want


def main():
    ok = True

    # 1. several clauses on one page -> one chunk each, cut exactly at the heading
    d = _doc(["5 \nWerkstoffe \n" + "Der Werkstoff muss geeignet sein. " * 6
              + "\n6 \nPruefverfahren \n" + "Die Pruefung erfolgt nach Anhang A. " * 6
              + "\n7 \nKennzeichnung \n" + "Die Kennzeichnung ist dauerhaft. " * 6])
    chunks, strategy = vector_rag._doc_chunks(d)
    ok &= check("clause strategy used", strategy, "clause")
    ok &= check("one chunk per clause", [c["clause"] for c in chunks], ["5", "6", "7"])
    ok &= check("clause 6 text stays out of clause 5's chunk",
                "Pruefverfahren" in chunks[0]["text"], False)
    ok &= check("clause 5 heading opens its chunk",
                chunks[0]["text"].startswith("5"), True)

    # 2. a clause longer than chunk_size is split by characters, but the split
    #    stays inside the clause — no piece may reach into the next clause
    body = "Die Anforderungen an die Beschichtung sind einzuhalten. " * 40  # ~2200 chars
    d = _doc(["7 \nBeschichtung \n" + body + "\n8 \nMontage \n" + "Montagetext. " * 20
              + "\n9 \nAbnahme \n" + "Die Abnahme erfolgt schriftlich. " * 6])
    chunks, _ = vector_rag._doc_chunks(d)
    seven = [c for c in chunks if c["clause"] == "7"]
    ok &= check("long clause is split", len(seven) > 1, True)
    ok &= check("no split piece leaks into clause 8",
                any("Montagetext" in c["text"] for c in seven), False)
    ok &= check("clause 8 keeps its own chunk",
                [c["clause"] for c in chunks if "Montagetext" in c["text"]], ["8"])

    # 3. a clause continuing onto the next page keeps its label, and the chunk
    #    stays page-scoped (page numbers are the unit of evidence in metrics.py)
    d = _doc(["7 \nSchweissen \n" + "Schweissen muss geplant werden. " * 8,
              "Fortsetzung des Schweisstextes. " * 8 + "\n8 \nMontage \nMontagetext. "
              + "\n9 \nAbnahme \n" + "Die Abnahme erfolgt schriftlich. " * 6])
    chunks, _ = vector_rag._doc_chunks(d)
    cont = [c for c in chunks if "Fortsetzung" in c["text"]]
    ok &= check("continuation labelled with the open clause",
                [(c["clause"], c["page"]) for c in cont], [("7", 2)])
    ok &= check("no chunk spans two pages",
                len({c["page"] for c in chunks}) == 2 and len(chunks) >= 3, True)

    # 4. a heading-only stub merges forward instead of embedding as noise
    d = _doc(["5 \nAllgemeines \n5.1 \nWerkstoffe \n"
              + "Der Werkstoff muss geeignet sein. " * 8
              + "\n6 \nPruefung \n" + "Die Pruefung erfolgt taeglich. " * 8])
    chunks, _ = vector_rag._doc_chunks(d)
    ok &= check("stub merged into the clause it introduces",
                [c["clause"] for c in chunks], ["5", "6"])
    ok &= check("merged chunk carries the sub-clause body",
                "Werkstoffe" in chunks[0]["text"] and "Der Werkstoff" in chunks[0]["text"],
                True)

    # 5. no usable clause structure -> character fallback (old behaviour)
    d = _doc(["Ein Fliesstext ohne jede nummerierte Gliederung. " * 40])
    chunks, strategy = vector_rag._doc_chunks(d)
    ok &= check("character fallback for unstructured text", strategy, "chars")
    ok &= check("fallback still produces chunks", len(chunks) > 1, True)
    ok &= check("fallback chunks carry no clause label",
                {c["clause"] for c in chunks}, {None})

    # 6. the ablation switch restores the plain sliding window
    d = _doc(["5 \nWerkstoffe \n" + "Der Werkstoff muss geeignet sein. " * 6
              + "\n6 \nPruefverfahren \n" + "Die Pruefung erfolgt nach Anhang A. " * 6])
    original = config.CLAUSE_CHUNKING
    try:
        config.CLAUSE_CHUNKING = False
        chunks, strategy = vector_rag._doc_chunks(d)
    finally:
        config.CLAUSE_CHUNKING = original
    ok &= check("CLAUSE_CHUNKING=0 falls back to characters", strategy, "chars")
    ok &= check("baseline merges both clauses into one window",
                len(chunks) == 1 and "Pruefverfahren" in chunks[0]["text"], True)

    # 7. chunk ids are content-addressed: deterministic, and collision-safe even
    #    when two document names truncate to the same 48-char slug
    d = _doc(["5 \nWerkstoffe \n" + "Der Werkstoff muss geeignet sein. " * 6
              + "\n6 \nPruefverfahren \n" + "Die Pruefung erfolgt. " * 6
              + "\n7 \nKennzeichnung \n" + "Die Kennzeichnung ist dauerhaft. " * 6])
    ids_a = [c["id"] for c in vector_rag._doc_chunks(d)[0]]
    d2 = _doc(d["pages"][0]["text"] and [d["pages"][0]["text"]])
    ids_b = [c["id"] for c in vector_rag._doc_chunks(d2)[0]]
    ok &= check("same text -> same ids (deterministic)", ids_a, ids_b)
    ok &= check("ids are unique", len(set(ids_a)), len(ids_a))
    ok &= check("id embeds page and a hash",
                bool(re.fullmatch(r"T-pdf-p1-[0-9a-f]{10}", ids_a[0])), True)

    long_a = "A" * 60 + "-one.pdf"      # both truncate to the same 48-char slug
    long_b = "A" * 60 + "-two.pdf"
    ok &= check("truncated slugs collide (the old risk)",
                vector_rag._slug(long_a) == vector_rag._slug(long_b), True)
    ok &= check("but ids do not, because the hash covers the full name",
                vector_rag._chunk_id(long_a, 3, "same text")
                != vector_rag._chunk_id(long_b, 3, "same text"), True)

    ok &= check("different text -> different id",
                vector_rag._chunk_id("T.pdf", 1, "a") != vector_rag._chunk_id("T.pdf", 1, "b"),
                True)
    ok &= check("same text on a different page -> different id",
                vector_rag._chunk_id("T.pdf", 1, "a") != vector_rag._chunk_id("T.pdf", 2, "a"),
                True)

    # exact duplicate text within one page must not overwrite itself in Pinecone
    dupes = [{"doc": "T.pdf", "page": 4, "text": "Wiederholte Zeile."},
             {"doc": "T.pdf", "page": 4, "text": "Wiederholte Zeile."},
             {"doc": "T.pdf", "page": 4, "text": "Andere Zeile."}]
    n = vector_rag._assign_ids(dupes)
    ok &= check("duplicate text disambiguated, not dropped",
                (n, len({c["id"] for c in dupes})), (1, 3))
    ok &= check("the duplicate is marked", dupes[1]["id"].endswith("#2"), True)

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
