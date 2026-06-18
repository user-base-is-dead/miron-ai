"""Core support modules for the Miron LLM (imported by scripts/)."""

# ── Python version guard ─────────────────────────────────────────────────────
# Project ke pinned deps (torch==2.5.1, numpy==2.4.6, ...) sirf Python 3.11.x pe
# chalte hain. Galat version (jaise system ka 3.14) pe saaf error do — chahe koi
# bhi entry point ho: `python -m core.config` ya scripts/* jo `core` import karte
# hain. (Detail: docs/python-version.md)
import sys as _sys

if _sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"[Miron] Python 3.11.x chahiye (abhi {_sys.version.split()[0]} chal raha hai).\n"
        "        venv activate karo -> Windows: Miron311\\Scripts\\activate"
        "  |  Linux/Mac: source Miron311/bin/activate"
    )
