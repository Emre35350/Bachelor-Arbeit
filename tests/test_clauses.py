"""Regression tests for the clause indexer (app/clauses.py).

The heading heuristics are subtle and were tuned against real DIN/EN PDF text
extraction, where PyMuPDF puts a clause number and its title on SEPARATE lines
and pages are littered with table rows, TOC leaders and print watermarks that
all superficially look like "7 Something". These cases lock that behaviour in.

Run:  python -m tests.test_clauses      (no pytest needed)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import clauses  # noqa: E402


def _doc(pages_text):
    return {"name": "T.pdf",
            "pages": [{"page": i, "text": t} for i, t in enumerate(pages_text, 1)],
            "clauses": None}


def check(label, got, want):
    status = "ok  " if got == want else "FAIL"
    print(f"  [{status}] {label}: got={got!r} want={want!r}")
    return got == want


def main():
    ok = True

    # 1. number and title on separate lines (the dominant real-world form)
    d = _doc(["EN 1:2013 (D) \n16 \n5 \nAnforderungen und Prüfverfahren \n"
              "Das Produkt ist nach 5.1 herzustellen. \n"])
    idx = clauses.index(d)
    ok &= check("two-line heading found", [(c["clause"], c["title"]) for c in idx],
                [("5", "Anforderungen und Prüfverfahren")])

    # 2. table rows / measurements must NOT become clauses
    d = _doc(["5 mm < h ≤ 50 mm \n40 mm ≤ dc < 60 mm \n"
              "50 kg/m3 bis 150 kg/m3 geprüften Dichten \n2 h). \n"])
    ok &= check("table rows rejected", clauses.index(d), [])

    # 3. table-of-contents lines must NOT become clauses (index must point at
    #    the clause BODY, not the contents page)
    d = _doc(["Inhalt \n1 Anwendungsbereich .......................... 7 \n"
              "5 Anforderungen ............................. 16 \n"])
    ok &= check("TOC leaders rejected", clauses.index(d), [])

    # 4. DIN print watermarks / timestamps must NOT become clauses
    d = _doc(["7 \nDatum / Uhrzeit des Ausdrucks: 2025-12-19, 15:04 \n"])
    ok &= check("print timestamp rejected", clauses.index(d), [])

    # 5. a bare printed page number in the header must not become a clause
    d = _doc(["2 \nAnwendungsbeginn \nText text text. \nText. \nText. \nText. \n"])
    got = [c["clause"] for c in clauses.index(d)]
    ok &= check("header page number ignored", got, [])

    # 6. sub-clause disambiguation: a numbered LIST item "7 ..." must lose to the
    #    real clause 7, which is the one immediately followed by 7.1
    d = _doc([
        "6 \nVorbereitung \n",                       # p1 real clause 6
        "7 \nVermeiden der Ansammlung von Wasser \n",  # p2 list item, also "7"
        "7 \nSchweißen \n7.1 \nAllgemeines \n",      # p3 real clause 7 + 7.1
    ])
    hit = clauses.find(d, "7")
    ok &= check("clause 7 resolves to the real heading",
                (hit["title"], hit["page"]), ("Schweißen", 3))

    # 7. annex headings
    d = _doc(["Anhang A \n(normativ) \nPrüfverfahren \n"])
    hit = clauses.find(d, "Anhang A")
    ok &= check("annex found", hit["clause"] if hit else None, "Anhang A")

    # 8. read() spans from the heading to the next heading, reporting its pages
    d = _doc(["7 \nSchweißen \nSchweißen muss geplant werden. \n",
              "Fortsetzung des Textes. \n8 \nMontage \nMontagetext. \n"])
    text, pages = clauses.read(d, "7")
    ok &= check("read() covers heading..next-heading", pages, [1, 2])
    ok &= check("read() stops before clause 8", "Montagetext" not in text, True)

    # 9. when the PDF ships an outline it is authoritative and wins over the
    #    lexical scan (body text has list items that mimic headings)
    d = _doc([
        "Inhalt \n",
        "7 Vermeiden der Ansammlung von Wasser \n",   # decoy list item on p2
        "7 \nSchweißen \nSchweißtext. \n",             # real clause 7 on p3
        "8 \nMontage \n",
    ])
    d["toc"] = [[1, "Titel", 1], [2, "1 Anwendungsbereich", 1],
                [2, "6 Vorbereitung", 2], [2, "7 Schweißen", 3],
                [3, "7.1 Allgemeines", 3], [2, "8 Montage", 4]]
    hit = clauses.find(d, "7")
    ok &= check("TOC outline is authoritative",
                (hit["title"], hit["page"]), ("Schweißen", 3))
    ok &= check("TOC entries without a clause number skipped",
                all(c["clause"] != "Titel" for c in clauses.index(d)), True)

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
