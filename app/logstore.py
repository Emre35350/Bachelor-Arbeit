"""Append-only JSONL experiment log (one line per pipeline run)."""
import json
import time
from pathlib import Path

from . import config


def append(entry: dict) -> None:
    entry = {"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    # every row records the exact configuration (incl. seed) it ran under
    entry.setdefault("config", config.snapshot())
    with Path(config.LOG_PATH).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_all() -> list[dict]:
    p = Path(config.LOG_PATH)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def clear() -> dict:
    """Reset the experiment log — non-destructively.

    The current log is renamed to a timestamped `.bak` sibling rather than
    deleted, so a reset before a real batch can never lose earlier runs. Returns
    the archive filename and how many rows it held.
    """
    p = Path(config.LOG_PATH)
    if not p.exists():
        return {"archived": None, "rows": 0}
    rows = len(read_all())
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = p.with_name(f"{p.stem}.{stamp}.bak{p.suffix}")
    p.replace(backup)          # atomic move; a fresh log is created on next append
    return {"archived": backup.name, "rows": rows}
