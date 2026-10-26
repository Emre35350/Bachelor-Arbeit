"""Shared build-index progress state, polled by the frontend while /build_index runs.

Runs in-process: FastAPI serves sync endpoints from a thread pool, so GET
/build_progress can be read concurrently while POST /build_index is in flight.
"""

PHASE_RANGES = {
    "idle": (0, 0),
    "loading_model": (0, 5),
    "chunking": (5, 10),
    "embedding": (10, 70),
    "trees": (70, 100),
    "done": (100, 100),
}

STATE = {"stage": "idle", "detail": "", "pct": 0}


def set_phase(stage: str, frac: float = 0.0, detail: str = "") -> None:
    lo, hi = PHASE_RANGES.get(stage, (0, 100))
    STATE["stage"] = stage
    STATE["detail"] = detail
    STATE["pct"] = round(lo + (hi - lo) * max(0.0, min(1.0, frac)))


def error(detail: str) -> None:
    STATE["stage"] = "error"
    STATE["detail"] = detail


def reset() -> None:
    STATE.update(stage="idle", detail="", pct=0)
