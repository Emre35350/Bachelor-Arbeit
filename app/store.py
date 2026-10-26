"""On-disk persistence for the parsed document store + build state.

Why: uploaded documents and their PageIndex trees live in memory, so a backend
restart used to force a full re-upload + rebuild. The Pinecone vectors already
persist in the cloud; here we persist the rest — the page store and the trees —
plus a small state file, so an experiment day survives a restart.

Layout (under DATA_DIR, default ./data):
  docs.json   the full ingestion.DOCS map (pages, toc, trees, tree_nodes)
  state.json  {vector_ready, trees_ready, built_with}

Everything stored is plain JSON (pages are {page,text}; trees are nested dicts),
so no custom (de)serialization is needed. Writes are atomic (temp file + os.replace).
"""
import json
import os
from pathlib import Path

from . import config, ingestion

_DOCS_FILE = "docs.json"
_STATE_FILE = "state.json"
_MANIFEST_FILE = "index_manifest.json"


def _dir() -> Path:
    p = Path(config.DATA_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(name: str, data) -> None:
    d = _dir()
    tmp = d / (name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, d / name)  # atomic on the same filesystem


def _read_json(name: str):
    p = _dir() / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ------------------------------------------------------------------ documents
def save_docs() -> None:
    if not config.PERSIST:
        return
    _write_json(_DOCS_FILE, ingestion.DOCS)


def load_docs() -> int:
    """Repopulate ingestion.DOCS from disk. Returns the document count."""
    data = _read_json(_DOCS_FILE)
    if not isinstance(data, dict):
        return 0
    ingestion.DOCS.clear()
    ingestion.DOCS.update(data)
    return len(ingestion.DOCS)


# ---------------------------------------------------------------------- state
def save_state(state: dict) -> None:
    if not config.PERSIST:
        return
    _write_json(_STATE_FILE, {k: state.get(k) for k in
                              ("vector_ready", "trees_ready", "built_with")})


def load_state() -> dict | None:
    data = _read_json(_STATE_FILE)
    return data if isinstance(data, dict) else None


# ------------------------------------------------------------------- manifest
def load_manifest() -> dict:
    """What is already embedded, per configuration namespace and per document.

    Shape: {namespace: {doc_name: {"signature": str, "ids": [...], "chunks": n}}}

    This is what makes builds incremental: a document whose content signature is
    unchanged in this namespace is skipped entirely, and switching an index-time
    setting selects another namespace rather than destroying this one. The vector
    ids are recorded so a single document can be re-indexed exactly (delete its
    old vectors, upsert the new ones) without wiping the whole namespace.
    """
    data = _read_json(_MANIFEST_FILE)
    return data if isinstance(data, dict) else {}


def save_manifest(manifest: dict) -> None:
    if not config.PERSIST:
        return
    _write_json(_MANIFEST_FILE, manifest)


def clear() -> None:
    """Remove persisted files (called when the user clears all sources)."""
    for name in (_DOCS_FILE, _STATE_FILE, _MANIFEST_FILE):
        try:
            (_dir() / name).unlink()
        except OSError:
            pass
