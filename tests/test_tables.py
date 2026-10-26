"""Regression tests for multi-vector table handling (app/tables.py).

Uses a real standard from ../datasets when available (detection quality is the
whole point and cannot be faked with a synthetic PDF), and stubs the vision call
so the test costs nothing and stays deterministic.

Run:  python -m tests.test_tables      (no pytest needed)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, tables, vector_rag  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent.parent.parent / "datasets"
SAMPLE = DATASET / "DIN-EN_ISO_12944-2_2018-04-00_DE_2718187.pdf"


def check(label, got, want):
    status = "ok  " if got == want else "FAIL"
    print(f"  [{status}] {label}: got={got!r} want={want!r}")
    return got == want


def main():
    ok = True

    # ---- filters: the corpus audit found ruled title blocks on every cover page
    class FakeTable:
        def __init__(self, data):
            self._d = data
        def extract(self):
            return self._d

    grid = [["a", "b"], ["1", "2"], ["3", "4"]]
    ok &= check("cover-page block rejected", tables._is_real_table(FakeTable(grid), 1), False)
    ok &= check("3x2 table accepted", tables._is_real_table(FakeTable(grid), 7), True)
    ok &= check("2-row table rejected",
                tables._is_real_table(FakeTable(grid[:2]), 7), False)
    ok &= check("1-column table rejected",
                tables._is_real_table(FakeTable([["a"], ["b"], ["c"]]), 7), False)

    # ---- caption picking: a prose mention must lose to the real dash caption
    ok &= check("dash caption preferred",
                tables._caption_for("Tabelle A.1 enthält Angaben zur Berechnung\n"
                                    "Tabelle 2 — Kategorien für Wasser und Erdreich\n"),
                "Tabelle 2 — Kategorien für Wasser und Erdreich")
    ok &= check("bare caption as fallback",
                tables._caption_for("Tabelle 5 Werkstoffe\n"), "Tabelle 5 Werkstoffe")

    # ---- per-table caption by position: two tables on a page, one caption each.
    #      Regression for the bug where every table took the page's FIRST caption.
    caps = [(84.0, "Tabelle E.1 — Materialeigenschaften"),
            (354.0, "Tabelle E.2 — Maßeigenschaften")]
    ok &= check("top table gets E.1",
                tables._caption_for_table(caps, (70, 102, 500, 300)),
                "Tabelle E.1 — Materialeigenschaften")
    ok &= check("bottom table gets E.2 (not the page's first caption)",
                tables._caption_for_table(caps, (70, 372, 500, 560)),
                "Tabelle E.2 — Maßeigenschaften")

    # ---- offline backfill of already-detected records
    doc_bf = {"name": "T.pdf",
              "pages": [{"page": 5,
                         "text": "Tabelle 3 — Produktnormen für Blech\n... rows ...\n"
                                 "Tabelle 4 — Produktnormen für nichtrostende Stähle\n... rows ..."}],
              "tables": [
                  {"doc": "T.pdf", "page": 5, "index": 0,
                   "caption": "Tabelle 3 — Produktnormen für Blech", "summary": "S0"},
                  {"doc": "T.pdf", "page": 5, "index": 1,
                   "caption": "Tabelle 3 — Produktnormen für Blech", "summary": "S1"}]}
    n_fixed = tables.backfill_captions(doc_bf)
    ok &= check("backfill corrects one caption", n_fixed, 1)
    ok &= check("second table now labelled Tabelle 4",
                doc_bf["tables"][1]["caption"], "Tabelle 4 — Produktnormen für nichtrostende Stähle")
    ok &= check("corrected table's summary invalidated", doc_bf["tables"][1]["summary"], None)
    ok &= check("correct table left untouched", doc_bf["tables"][0]["summary"], "S0")
    ok &= check("backfill is idempotent", tables.backfill_captions(doc_bf), 0)

    # a split table (several pieces, ONE caption) must be left alone
    doc_split = {"name": "T.pdf",
                 "pages": [{"page": 8, "text": "Tabelle 9 — Lange Tabelle\n... many rows ..."}],
                 "tables": [
                     {"doc": "T.pdf", "page": 8, "index": 0,
                      "caption": "Tabelle 9 — Lange Tabelle", "summary": "A"},
                     {"doc": "T.pdf", "page": 8, "index": 1,
                      "caption": "Tabelle 9 — Lange Tabelle", "summary": "B"}]}
    ok &= check("split table not disturbed (one caption)",
                tables.backfill_captions(doc_split), 0)

    # ---- chunk text is the SUMMARY, never the raw grid, when a summary exists
    rec = {"page": 9, "caption": "Tabelle 2 — Kategorien", "grid": "a | b\n1 | 2",
           "summary": "This table specifies categories. Im1: fresh water = ...",
           "image": "x.png", "doc": "T.pdf", "index": 0}
    txt = tables.chunk_text(rec)
    ok &= check("summary is what gets embedded", "This table specifies" in txt, True)
    ok &= check("raw grid stays out of the embedded text", "a | b" in txt, False)
    rec_nosum = {**rec, "summary": None}
    ok &= check("grid is the fallback when summarization failed",
                "a | b" in tables.chunk_text(rec_nosum), True)

    # ---- table chunks join the document's chunks, labelled with their clause
    doc = {"name": "T.pdf", "toc": [], "clauses": None,
           "pages": [{"page": 1, "text": "5 \nWerkstoffe \n" + "Der Werkstoff ist geeignet. " * 8},
                     {"page": 2, "text": "6 \nPruefung \n" + "Die Pruefung erfolgt. " * 8},
                     {"page": 3, "text": "7 \nAbnahme \n" + "Die Abnahme erfolgt. " * 8}],
           "tables": [{**rec, "page": 2}]}
    chunks, _ = vector_rag._doc_chunks(doc)
    tabs = [c for c in chunks if c.get("table_image")]
    ok &= check("one table chunk added", len(tabs), 1)
    ok &= check("table chunk carries page + image",
                (tabs[0]["page"], tabs[0]["table_image"]), (2, "x.png"))
    ok &= check("table chunk inherits the clause open on its page",
                tabs[0]["clause"], "6")
    ok &= check("table chunk marked as such", tabs[0]["strategy"], "table")

    # ---- the switch removes table chunks entirely (ablation baseline)
    original = config.TABLE_SUMMARIES
    try:
        config.TABLE_SUMMARIES = False
        chunks_off, _ = vector_rag._doc_chunks(doc)
    finally:
        config.TABLE_SUMMARIES = original
    ok &= check("TABLE_SUMMARIES=0 drops table chunks",
                [c for c in chunks_off if c.get("table_image")], [])

    # ---- attaching images to the answering call
    original_attach = config.ATTACH_TABLE_IMAGES
    try:
        config.ATTACH_TABLE_IMAGES = False
        parts = [{"type": "text", "text": "ctx"}]
        off = vector_rag._attach_table_images(parts, [{"doc": "T.pdf", "page": 2,
                                                       "table_image": "x.png"}])
        ok &= check("ATTACH_TABLE_IMAGES=0 attaches nothing", (off, len(parts)), ([], 1))
    finally:
        config.ATTACH_TABLE_IMAGES = original_attach

    # a missing PNG must not crash the answering call
    parts = [{"type": "text", "text": "ctx"}]
    missing = vector_rag._attach_table_images(
        parts, [{"doc": "T.pdf", "page": 2, "table_image": "does-not-exist.png"}])
    ok &= check("missing image is skipped, not fatal", (missing, len(parts)), ([], 1))

    # ---- real detection, if the dataset is present
    if SAMPLE.exists():
        import fitz
        pdf = fitz.open(SAMPLE)
        d = {"name": SAMPLE.name,
             "pages": [{"page": i + 1, "text": p.get_text("text") or ""}
                       for i, p in enumerate(pdf)],
             "toc": [], "clauses": None, "tables": []}
        recs = tables.detect(d, pdf)
        pdf.close()
        ok &= check("real tables detected in ISO 12944-2", len(recs) >= 3, True)
        ok &= check("every record rendered a PNG",
                    all((tables.image_dir() / r["image"]).exists() for r in recs), True)
        ok &= check("no cover-page false positive",
                    [r for r in recs if r["page"] == 1], [])
        ok &= check("corrosivity table found on p.15",
                    any(r["page"] == 15 for r in recs), True)
    else:
        print(f"  [skip] dataset not found at {DATASET} — detection test skipped")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
